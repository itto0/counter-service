# Task 2 Report: Dynamic Telemetry Integration & Degraded Stream Alerting in CounterService

## Overview
Integrated dynamic tracking completion accuracy telemetry into `CounterService` (`main.py`), supporting idle window caching fallback (`_last_active_accuracy_pct` -> `model_accuracy_pct`) and degraded tracking alerting when dynamic operational accuracy drops below 80.0%.

## Commit
- **Commit SHA:** `62848a80b80639cc713d231501b8e5fb19f5d2ab`
- **Commit Message:** `feat(telemetry): update health heartbeat with dynamic tracking accuracy`
- **Files Modified:**
  - `main.py`
  - `tests/test_counter_telemetry_accuracy.py` (created)

## Implementation Details

### 1. `CounterService.__init__` Telemetry Cache State
Added `self._last_active_accuracy_pct: Optional[float] = None` to track the most recent non-idle operational accuracy score.

### 2. `CounterService._record_health` Dynamic Query & Idle Fallback
In `_record_health`:
- Safely queries dynamic tracking completion accuracy from the counter:
  `dynamic_acc = self.counter.get_tracking_accuracy_pct(reset_window=True) if hasattr(self, "counter") and hasattr(self.counter, "get_tracking_accuracy_pct") else None`
- If `dynamic_acc is not None`: updates `self._last_active_accuracy_pct = dynamic_acc` and sets `current_acc = dynamic_acc`.
- If `dynamic_acc is None` (conveyor idle / no tracks completed in window): falls back to `self._last_active_accuracy_pct` if available, or static `self.model_accuracy_pct`.
- Emits a warning log when `current_acc is not None and current_acc < 80.0`:
  `logger.warning("Operational tracking accuracy degraded: %.1f%% (FPS=%.1f, latency_p95=%.1fms)", current_acc, fps, latency_p95)`
- Populates `HealthSampleDTO(..., accuracy_pct=current_acc)` and persists sample to `LocalStorage`.

## Test Results

### TDD Cycle
1. **RED:** Created `tests/test_counter_telemetry_accuracy.py` with 3 test cases. Verified all 3 failed with missing attribute `_last_active_accuracy_pct` and missing alert log.
2. **GREEN:** Implemented caching and fallback logic in `main.py`. Ran pytest on `tests/test_counter_telemetry_accuracy.py`:
   ```
   tests/test_counter_telemetry_accuracy.py ...                             [100%]
   ============================== 3 passed in 2.08s ==============================
   ```
3. **Regression & Full Suite:** Ran existing test suites across `main.py`, idle mode, storage, and tracker:
   ```
   tests/test_counter_telemetry_accuracy.py ...                             [  3%]
   tests/test_main.py .........................................             [ 44%]
   tests/test_idle_mode.py ..............                                   [ 58%]
   tests/test_storage.py ............                                       [ 70%]
   tests/test_tracking.py ......................                            [ 92%]
   tests/test_tracker_bytetrack.py ........                                 [100%]
   ======================= 100 passed in 104.90s (0:01:44) =======================
   ```

### Specific Test Coverage
- `test_record_health_uses_dynamic_tracking_accuracy`: Verifies dynamic tracking accuracy (85.5%) is queried with `reset_window=True`, stored in SQLite `HealthSampleDTO`, and cached in `_last_active_accuracy_pct`.
- `test_record_health_idle_fallback`: Verifies 3-stage fallback:
  1. Initial idle with no active history falls back to `model_accuracy_pct` (92.0%).
  2. Active window sets dynamic accuracy (88.0%) and caches it.
  3. Subsequent idle window falls back to cached `_last_active_accuracy_pct` (88.0%) rather than static model accuracy.
- `test_record_health_degraded_warning_log`: Verifies `logger.warning` is emitted when accuracy is < 80.0% (75.0%), and is suppressed when accuracy is healthy (>= 80.0%).

## Concerns / Notes for Downstream Tasks
- Dynamic accuracy resets window metrics each time `_record_health()` runs (`reset_window=True`), keeping telemetry bounded to the heartbeat interval.
- When `self.counter` is not yet processing tracks or conveyor is paused, fallback to `_last_active_accuracy_pct` prevents false drops or `None` gaps in time-series telemetry charts on central dashboard.
