# Task 1 Report: Track Lifecycle Tracking & Completion Accuracy in ChickenCounter

## Overview
Implemented track lifecycle tracking (completed tracks vs lost tracks) and dynamic tracking completion accuracy calculation in `ChickenCounter` (`src/tracking/tracker.py`), filtering single-frame spurious noise and supporting idle window fallback.

## Commit
- **Commit SHA:** `65cf9c9d1c2955137d48b4eaa7bd4562019380a3`
- **Commit Message:** `feat(tracker): implement tracking completion accuracy calculation`
- **Files Modified:**
  - `src/tracking/tracker.py`
  - `tests/test_tracking.py`
  - `tests/test_tracker_bytetrack.py`

## Implementation Details

### 1. `ChickenCounter.__init__` Lifecycle Counters
Added initialization for tracking completion and loss metrics:
- `self._window_completed_tracks: int = 0`
- `self._window_lost_tracks: int = 0`
- `self._total_completed_tracks: int = 0`
- `self._total_lost_tracks: int = 0`
- `self._min_track_hits: int = 2` (minimum frames detected to qualify as a valid carcass track)

### 2. Crossing Detection (ByteTrack & Centroid)
When a track crosses the virtual line zone:
- `track.counted = True`
- `self.counted_track_ids.add(track_id)`
- `self.total_count += 1`
- `self._window_completed_tracks += 1`
- `self._total_completed_tracks += 1`

### 3. Track Detection Age & Stale Pruning
- Newly registered tracks start with `age = 1` representing the detection hit.
- Matched tracks increment `track.age += 1`.
- Missed frames only increment `track.missed_frames += 1` without incrementing `track.age`. This ensures `track.age` accurately reflects detected hit count so that single-frame transient noise (`age == 1 < min_track_hits`) is cleanly ignored upon expiration, while carcass tracks that were tracked for at least 2 frames (`age >= min_track_hits`) are properly flagged as lost.
- When tracks exceed `max_age`, uncounted tracks with `track.age >= self._min_track_hits` increment lost counters:
  - `self._window_lost_tracks += 1`
  - `self._total_lost_tracks += 1`

### 4. `get_tracking_accuracy_pct(reset_window: bool = True) -> float | None`
Calculates completion accuracy for the current window:
- Returns `None` when `total == 0` (idle conveyor state).
- Calculates `acc = (completed / (completed + lost)) * 100.0`, clamped to `[0.0, 100.0]` and rounded to 2 decimal places.
- Clears `self._window_completed_tracks` and `self._window_lost_tracks` when `reset_window=True`.

## Test Results

### TDD Cycle
1. **RED:** Added tests in `tests/test_tracking.py`. Verified tests failed with:
   ```
   AttributeError: 'ChickenCounter' object has no attribute 'get_tracking_accuracy_pct'
   ```
2. **GREEN:** Implemented lifecycle tracking and accuracy inquiry method in `src/tracking/tracker.py`. All tests passed.
3. **Full Suite:** Ran all 30 tests in `tests/test_tracking.py` and `tests/test_tracker_bytetrack.py`:
   ```
   tests/test_tracking.py ......................                            [ 73%]
   tests/test_tracker_bytetrack.py ........                                 [100%]
   ============================= 30 passed in 2.09s ==============================
   ```

### Specific Test Coverage
- `test_tracker_completion_accuracy_healthy`: Verifies 100.0% completion accuracy when all tracks cross the line and window reset behavior.
- `test_tracker_completion_accuracy_with_lost_tracks`: Verifies accuracy drops to 50.0% when 1 track crosses and 1 valid track (`age >= 2`) is lost.
- `test_tracker_completion_accuracy_ignores_single_frame_noise`: Verifies 1-frame spurious detection is filtered and not recorded as lost, yielding 100.0% accuracy.
- `test_tracker_completion_accuracy_idle_returns_none`: Verifies `None` is returned when no tracks finish in the window.
- `test_bytetrack_completion_accuracy`: Verifies ByteTrack tracker correctly updates completed track counters and computes accuracy.

## Concerns / Notes for Downstream Tasks
- `TrackedObject.age` default in dataclass remains `0` preserving backward compatibility with direct dataclass instantiations.
- `ChickenCounter` sets initial `age = 1` for newly created detections and increments only on successful associations, which aligns with `_min_track_hits = 2`.
- Ready for Task 2 (`CounterService` telemetry integration in `main.py`).
