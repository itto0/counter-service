# Task 1: Track Lifecycle Tracking & Completion Accuracy in `ChickenCounter`

**Files:**
- Modify: `src/tracking/tracker.py`
- Test: `tests/test_tracking.py`

**Interfaces:**
- Consumes: `ChickenCounter` track creation, line crossing, and stale pruning.
- Produces: `ChickenCounter.get_tracking_accuracy_pct(reset_window: bool = True) -> float | None`

## Requirements:
1. In `ChickenCounter.__init__`, initialize lifecycle counters:
   - `self._window_completed_tracks: int = 0`
   - `self._window_lost_tracks: int = 0`
   - `self._total_completed_tracks: int = 0`
   - `self._total_lost_tracks: int = 0`
   - `self._min_track_hits: int = 2`

2. In line crossing detection (both `_update_bytetrack` and `_update_centroid`):
   When a track crosses the line (`self.line_zone.crosses(track.last_anchor, track.current_anchor)`):
   - `track.counted = True`
   - `self.counted_track_ids.add(track_id)`
   - `self.total_count += 1`
   - `self._window_completed_tracks += 1`
   - `self._total_completed_tracks += 1`

3. In stale track pruning (both `_update_bytetrack` and `_update_centroid`):
   When pruning `stale_ids`:
   ```python
   for track_id in stale_ids:
       track = self.tracks[track_id]
       if not track.counted and track.age >= self._min_track_hits:
           self._window_lost_tracks += 1
           self._total_lost_tracks += 1
       del self.tracks[track_id]
   ```

4. Implement `get_tracking_accuracy_pct(self, reset_window: bool = True) -> float | None`:
   ```python
   def get_tracking_accuracy_pct(self, reset_window: bool = True) -> float | None:
       """Calculate tracking completion accuracy percentage for the current window.

       Returns:
           Accuracy percentage in range [0.0, 100.0] or None if no tracks finished in window.
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

5. Tests to add to `tests/test_tracking.py`:
   - `test_tracker_completion_accuracy_healthy`: verify 100.0% when all tracks cross.
   - `test_tracker_completion_accuracy_with_lost_tracks`: verify 50.0% when 1 crosses and 1 is lost (age >= 2).
   - `test_tracker_completion_accuracy_ignores_single_frame_noise`: verify 100.0% when single-frame noise is pruned.
   - `test_tracker_completion_accuracy_idle_returns_none`: verify returns None when 0 tracks finished.

Follow TDD: Write failing tests first, run to verify failure, implement minimal code, verify pass, commit with message:
`feat(tracker): implement tracking completion accuracy calculation`
