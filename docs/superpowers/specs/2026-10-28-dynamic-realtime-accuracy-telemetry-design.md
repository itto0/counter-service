# Technical Design Spec: Dynamic Real-Time Accuracy Telemetry

**Date:** 2026-10-28  
**Status:** Approved  
**Topic:** Dynamic Tracking Completion Accuracy in Health Telemetry  

---

## 1. Goal & Problem Statement

Currently, `CounterService` records a static accuracy percentage (`accuracy_pct`) in its periodic `HealthSampleDTO` telemetry by reading offline validation metrics (`mAP@50`) from `<model_stem>_metrics.json`. 

When running on low-resource hardware (e.g. single-core CPU or edge devices):
- Frame rate can degrade significantly (e.g., to 6 FPS with >100ms inference latency).
- Video frames are dropped by ingestion queues (`ThreadedVideoSource`), causing carcasses to jump past the counting line or causing ByteTrack association to fail.
- Carcasses are skipped in large quantities, yet telemetry falsely reports `99%` accuracy because it only reads the static offline model score.

This specification defines a **dynamic, real-time tracking completion accuracy metric** calculated from live stream behavior and reported in `HealthSampleDTO.accuracy_pct`.

---

## 2. Requirements & Constraints

### Functional Requirements
- **FR-1: Real-Time Tracking Completion Tracking:** `ChickenCounter` in `src/tracking/tracker.py` must track the lifecycle completion of carcass tracks:
  - **Completed Track:** A track that successfully crosses the virtual line (`track.counted == True`).
  - **Lost/Skipped Track:** A valid track that expires (`missed_frames > max_allowed_missed`) without crossing the line, provided it was observed for at least a minimum threshold of detections (`age >= min_hits`, default 2) to exclude single-frame spurious noise.
- **FR-2: Dynamic Accuracy Calculation:** Calculate dynamic tracking accuracy for each evaluation window:
  $$\text{accuracy\_pct} = \left( \frac{\text{completed\_tracks}}{\text{completed\_tracks} + \text{lost\_tracks}} \right) \times 100.0$$
  Clamped to `[0.0, 100.0]` rounded to 2 decimal places.
- **FR-3: Idle State Fallback:**
  - If no tracks completed or expired in the health window (e.g. idle conveyor), `accuracy_pct` maintains the `last_active_accuracy_pct`.
  - If no tracks have ever been observed yet (fresh startup), fall back to `model_accuracy_pct` (loaded from companion metrics) or `None`.
- **FR-4: Telemetry Heartbeat Integration:** In `main.py`, `_record_health()` queries `ChickenCounter.get_tracking_accuracy_pct()` every 10 seconds and populates `HealthSampleDTO.accuracy_pct`.
- **FR-5: Degraded Stream Logging:** If `accuracy_pct` drops below 80.0% while the conveyor is active and FPS is low, log a warning indicating edge tracking degradation.

### Non-Functional & Compatibility Constraints
- **Zero Schema Breaking Changes:** `HealthSampleDTO.accuracy_pct` remains an optional `float | None` in `src/contracts.py`. SQLite storage and central cloud sync require no schema changes.
- **Performance:** Tracking lifecycle counters are updated via $O(1)$ operations on track termination, adding negligible CPU overhead (< 0.05ms).
- **Both Trackers Supported:** Works identically for both `ByteTrack` (`--tracker-type bytetrack`) and `CentroidTracker` (`--tracker-type centroid`).

---

## 3. Architecture & Data Flow

```
[ Camera / RTSP Stream ]
         │
         ▼
[ Ingestion Queue (ThreadedVideoSource) ]
         │ (drops frames if inference latency > 33ms)
         ▼
[ Detector (ONNXDetector / ShackleDetector) ]
         │ (emits Bounding Boxes per processed frame)
         ▼
[ ChickenCounter (tracker.py) ]
    ├── ByteTrack / Centroid update
    ├── Track crosses line ──> completed_tracks += 1
    └── Track pruned uncounted (age >= min_hits) ──> lost_tracks += 1
         │
         │ Every 10 seconds: get_tracking_accuracy_pct(reset=True)
         ▼
[ CounterService._record_health() (main.py) ]
    ├── Computes dynamic accuracy (or idle fallback)
    ├── HealthSampleDTO(..., accuracy_pct=dynamic_accuracy)
    │
    ▼
[ LocalStorage & OutboxSyncDispatcher ]
    └── Persists & syncs real operational health
```

---

## 4. Component Changes

### 4.1 Tracking Lifecycle (`src/tracking/tracker.py`)

1. Add lifecycle counters to `ChickenCounter`:
   ```python
   self._window_completed_tracks: int = 0
   self._window_lost_tracks: int = 0
   self._total_completed_tracks: int = 0
   self._total_lost_tracks: int = 0
   self._min_track_hits: int = 2  # Minimum frames to qualify as a valid carcass track
   ```

2. When a track is counted (crossing event):
   ```python
   if track_id not in self.counted_track_ids and not track.counted:
       if self.line_zone.crosses(track.last_anchor, track.current_anchor):
           track.counted = True
           self.counted_track_ids.add(track_id)
           self.total_count += 1
           self._window_completed_tracks += 1
           self._total_completed_tracks += 1
   ```

3. When stale tracks are pruned in `_update_bytetrack` and `_update_centroid`:
   ```python
   for track_id in stale_ids:
       track = self.tracks[track_id]
       if not track.counted and track.age >= self._min_track_hits:
           self._window_lost_tracks += 1
           self._total_lost_tracks += 1
       del self.tracks[track_id]
   ```

4. Add inquiry method:
   ```python
   def get_tracking_accuracy_pct(self, reset_window: bool = True) -> float | None:
       """Calculate completion accuracy percentage for the current window.
       
       Returns None if no tracks finished during the window (idle).
       """
       total = self._window_completed_tracks + self._window_lost_tracks
       if total == 0:
           return None
       acc = (self._window_completed_tracks / total) * 100.0
       if reset_window:
           self._window_completed_tracks = 0
           self._window_lost_tracks = 0
       return round(min(100.0, max(0.0, acc)), 2)
   ```

### 4.2 CounterService Telemetry (`main.py`)

1. In `CounterService.__init__()`:
   ```python
   self._last_active_accuracy_pct: float | None = None
   ```

2. In `CounterService._record_health()`:
   ```python
   # Query dynamic tracking accuracy from the counter
   dynamic_acc = self.counter.get_tracking_accuracy_pct(reset_window=True)
   if dynamic_acc is not None:
       self._last_active_accuracy_pct = dynamic_acc
       current_accuracy = dynamic_acc
   else:
       # Conveyor idle: fall back to last active reading, or static model accuracy
       current_accuracy = (
           self._last_active_accuracy_pct
           if self._last_active_accuracy_pct is not None
           else self.model_accuracy_pct
       )

   # Warning on significant degradation
   if current_accuracy is not None and current_accuracy < 80.0:
       logger.warning(
           "Operational tracking accuracy degraded: %.1f%% (FPS=%.1f, latency_p95=%.1fms)",
           current_accuracy,
           self.health_fps,
           lat_p95,
       )
   ```

3. Pass `current_accuracy` to `HealthSampleDTO(accuracy_pct=current_accuracy, ...)`.

---

## 5. Verification & Test Plan

1. **Unit Tests for Tracking Accuracy (`tests/test_tracker.py`):**
   - Test healthy run: all tracks cross the line $\implies$ accuracy is `100.0%`.
   - Test lost tracks: 5 tracks cross the line, 5 tracks die before crossing $\implies$ accuracy is `50.0%`.
   - Test noise suppression: tracks with `age < 2` (spurious 1-frame detections) are pruned without counting as lost tracks.
   - Test idle window: 0 tracks terminated $\implies$ returns `None`.
   - Test window reset: accuracy resets after query.
2. **Integration Tests for Telemetry (`tests/test_counter_service.py`):**
   - Assert `_record_health` records dynamic accuracy into `HealthSampleDTO`.
   - Assert idle fallback preserves `_last_active_accuracy_pct`.
   - Assert warning log triggered when accuracy drops below 80%.
3. **End-to-End Low-Resource Simulation:**
   - Run benchmark with throttled frame rate and verify telemetry reflects real accuracy drop rather than static 99%.
