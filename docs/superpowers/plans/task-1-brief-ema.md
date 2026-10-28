# Task 1: Implement EMA Smoothing in `CounterService` and Comprehensive Tests

**Files:**
- Modify: `main.py`
- Modify: `tests/test_counter_telemetry_accuracy.py`

**Interfaces:**
- Consumes: `ChickenCounter.get_tracking_accuracy_pct(reset_window: bool = True) -> float | None`
- Produces: Smoothed `HealthSampleDTO.accuracy_pct` recorded into SQLite storage.

## Requirements:
1. In `CounterService.__init__`:
   Add:
   ```python
   self._smoothed_accuracy_pct: Optional[float] = None
   self.accuracy_ema_alpha: float = (
       float(accuracy_ema_alpha)
       if accuracy_ema_alpha is not None
       else float(os.getenv("ACCURACY_EMA_ALPHA", "0.15"))
   )
   ```
   (Also accept `accuracy_ema_alpha: Optional[float] = None` in `CounterService.__init__`).

2. In `CounterService._record_health()`:
   Update accuracy computation:
   ```python
   dynamic_acc = (
       self.counter.get_tracking_accuracy_pct(reset_window=True)
       if hasattr(self, "counter") and hasattr(self.counter, "get_tracking_accuracy_pct")
       else None
   )
   if dynamic_acc is not None:
       if self._smoothed_accuracy_pct is None:
           self._smoothed_accuracy_pct = dynamic_acc
       else:
           alpha = self.accuracy_ema_alpha
           blended = (alpha * dynamic_acc) + ((1.0 - alpha) * self._smoothed_accuracy_pct)
           self._smoothed_accuracy_pct = round(min(100.0, max(0.0, blended)), 2)
       self._last_active_accuracy_pct = self._smoothed_accuracy_pct
       current_acc = self._smoothed_accuracy_pct
   else:
       # Conveyor idle: fall back to smoothed score, last active reading, or static model accuracy
       current_acc = (
           self._smoothed_accuracy_pct
           if self._smoothed_accuracy_pct is not None
           else (
               self._last_active_accuracy_pct
               if self._last_active_accuracy_pct is not None
               else self.model_accuracy_pct
           )
       )

   if stream_state != "IDLE" and current_acc is not None and current_acc < 80.0:
       logger.warning(
           "Operational tracking accuracy degraded: %.1f%% (FPS=%.1f, latency_p95=%.1fms)",
           current_acc,
           fps,
           latency_p95,
       )
   ```
   Pass `accuracy_pct=current_acc` to `HealthSampleDTO(...)`.

3. Tests to add in `tests/test_counter_telemetry_accuracy.py`:
   - `test_smoothed_accuracy_seeded_on_first_active_window(tmp_path)`: verify first active window directly seeds `_smoothed_accuracy_pct`.
   - `test_smoothed_accuracy_ema_gradual_decay_and_recovery(tmp_path)`: verify sequential raw readings cause smooth, gradual decay and recovery with `alpha=0.15`.
   - `test_smoothed_accuracy_idle_preserves_score(tmp_path)`: verify idle windows do not change the smoothed accuracy score.

Follow TDD: Write failing tests first, run to verify failure, implement minimal code, verify pass, commit with message:
`feat(telemetry): smooth operational accuracy using exponential moving average`
