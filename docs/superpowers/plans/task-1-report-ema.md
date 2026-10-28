# Task 1 Report: Exponential Moving Average (EMA) Smoothing in CounterService

## Summary
Implemented Exponential Moving Average (EMA) smoothing for operational tracking accuracy within `CounterService._record_health()` in `main.py`, along with comprehensive unit tests in `tests/test_counter_telemetry_accuracy.py`.

Prior to this change, `_record_health()` recorded raw dynamic window accuracy directly from `counter.get_tracking_accuracy_pct(reset_window=True)`. When few carcasses traversed the inspection line during sparse conveyor periods, single ID switches or missed detections could cause dramatic, jittery drops in reported telemetry accuracy. With EMA smoothing, accuracy evolves smoothly according to a configurable smoothing factor (`alpha=0.15` default), mitigating noise while accurately reflecting sustained performance trends.

---

## Changes Implemented

### 1. `main.py` (`CounterService`)
- **Constructor Parameters & Attributes:**
  - Added optional parameter `accuracy_ema_alpha: Optional[float] = None` to `CounterService.__init__`.
  - Added `self._smoothed_accuracy_pct: Optional[float] = None` to track the smoothed operational score across health intervals.
  - Added `self.accuracy_ema_alpha: float` resolved with precedence:
    `accuracy_ema_alpha (constructor)` > `pipeline_config.accuracy_ema_alpha` > `os.getenv("ACCURACY_EMA_ALPHA", "0.15")`.
- **Health Calculation & Storage in `_record_health()`:**
  - On the first active window (`dynamic_acc is not None` and `_smoothed_accuracy_pct is None`):
    Directly seeds `self._smoothed_accuracy_pct = dynamic_acc`.
  - On subsequent active windows:
    Applies EMA blending:
    $$\text{blended} = (\alpha \times \text{dynamic\_acc}) + ((1 - \alpha) \times \text{\_smoothed\_accuracy\_pct})$$
    Clamps the result between `0.0%` and `100.0%` and rounds to 2 decimal places:
    `self._smoothed_accuracy_pct = round(min(100.0, max(0.0, blended)), 2)`
  - Updates `_last_active_accuracy_pct = self._smoothed_accuracy_pct` and sets `current_acc = self._smoothed_accuracy_pct`.
  - Conveyor idle handling:
    When dynamic accuracy is `None` (no tracks in window), falls back smoothly to `_smoothed_accuracy_pct`, then `_last_active_accuracy_pct`, and finally static `model_accuracy_pct`.
  - Degraded alerting:
    Preserved warning log `logger.warning("Operational tracking accuracy degraded: %.1f%% ...")` when `stream_state != "IDLE"` and `current_acc < 80.0`.
  - Passed `accuracy_pct=current_acc` into `HealthSampleDTO` for persistent SQLite storage.

### 2. `tests/test_counter_telemetry_accuracy.py`
Implemented thorough test coverage verifying:
1. `test_smoothed_accuracy_seeded_on_first_active_window`:
   Verifies that the first active window directly seeds `_smoothed_accuracy_pct` without distortion.
2. `test_smoothed_accuracy_ema_gradual_decay_and_recovery`:
   Verifies sequential raw readings cause smooth, gradual decay and recovery with default $\alpha = 0.15$ (raw drop $90\% \rightarrow 70\%$ yields smoothed $87.0\%$, sustained $70\%$ yields $84.45\%$, and recovery $95\%$ yields $86.03\%$).
3. `test_smoothed_accuracy_idle_preserves_score`:
   Verifies that idle windows (`dynamic_acc is None`) preserve `_smoothed_accuracy_pct` across multiple cycles and subsequent active windows resume smoothing from the preserved score.
4. `test_smoothed_accuracy_alpha_configuration`:
   Verifies configuration precedence: constructor argument > `ACCURACY_EMA_ALPHA` environment variable > default `0.15`.
5. `test_smoothed_accuracy_clamping`:
   Verifies that anomalous values (>100% or <0%) are strictly clamped within `[0.0, 100.0]`.
6. Isolated degraded warning test in `test_record_health_degraded_warning_log` so healthy state verification uses a clean instance.

---

## TDD Execution Summary

1. **RED Phase:**
   - Authored new tests in `tests/test_counter_telemetry_accuracy.py`.
   - Executed:
     ```bash
     .venv\Scripts\python.exe -m pytest tests/test_counter_telemetry_accuracy.py -k test_smoothed_accuracy
     ```
   - Observed expected failure:
     - `AssertionError: assert False where False = hasattr(service, '_smoothed_accuracy_pct')`
     - `TypeError: CounterService.__init__() got an unexpected keyword argument 'accuracy_ema_alpha'`
2. **GREEN Phase:**
   - Implemented EMA fields and computation in `main.py`.
   - Re-ran tests:
     - 3 of 3 smoothed accuracy tests passed.
     - Full file `tests/test_counter_telemetry_accuracy.py` passed (8 of 8 tests).
3. **Regression & Full Suite Verification:**
   - Executed required tracking and telemetry suite:
     ```bash
     .venv\Scripts\python.exe -m pytest tests/test_counter_telemetry_accuracy.py tests/test_tracking.py tests/test_tracker_bytetrack.py
     ```
     Result: **39 passed in 8.50s**.
   - Executed health/idle regression suite (`tests/test_idle_mode.py`, `tests/test_main.py`):
     Result: **55 passed in 102.08s**.

---

## Git Commit Information
- **Commit SHA:** `1a8d88390634469476cfb871f664500b2ac28dc3`
- **Branch:** `feat/ema-accuracy-smoothing`
- **Commit Message:**
  `feat(telemetry): smooth operational accuracy using exponential moving average`
- **Files Committed:**
  - `main.py`
  - `tests/test_counter_telemetry_accuracy.py`

---

## Concerns & Recommendations
- **Telemetry Discontinuity on Restarts:**
  `self._smoothed_accuracy_pct` currently initializes to `None` in memory per process lifecycle, so on edge daemon restarts, it re-seeds on the first active window. This is standard behavior; if cold-start continuity across process restarts is desired in the future, `_smoothed_accuracy_pct` could optionally be primed from the most recent SQLite health sample on startup.
