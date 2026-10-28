# Task 2: Dynamic Telemetry Integration & Degraded Stream Alerting in `CounterService`

**Files:**
- Modify: `main.py`
- Create: `tests/test_counter_telemetry_accuracy.py`

**Interfaces:**
- Consumes: `ChickenCounter.get_tracking_accuracy_pct(reset_window: bool = True) -> float | None`
- Produces: Dynamic `HealthSampleDTO.accuracy_pct` recorded into SQLite storage.

## Requirements:
1. In `CounterService.__init__`:
   Add `self._last_active_accuracy_pct: float | None = None`.

2. In `CounterService._record_health(self, now_perf: Optional[float] = None, stream_state: str = "ONLINE") -> None`:
   Query dynamic tracking accuracy from the counter:
   ```python
   dynamic_acc = self.counter.get_tracking_accuracy_pct(reset_window=True)
   if dynamic_acc is not None:
       self._last_active_accuracy_pct = dynamic_acc
       current_acc = dynamic_acc
   else:
       # Conveyor idle: fall back to last active reading, or static model accuracy
       current_acc = (
           self._last_active_accuracy_pct
           if self._last_active_accuracy_pct is not None
           else self.model_accuracy_pct
       )

   if current_acc is not None and current_acc < 80.0:
       logger.warning(
           "Operational tracking accuracy degraded: %.1f%% (FPS=%.1f, latency_p95=%.1fms)",
           current_acc,
           fps,
           latency_p95,
       )
   ```
   Pass `accuracy_pct=current_acc` to `HealthSampleDTO(...)`.

3. Tests to create in `tests/test_counter_telemetry_accuracy.py`:
   - `test_record_health_uses_dynamic_tracking_accuracy(tmp_path)`: verify dynamic accuracy (e.g. 85.5) is stored in `HealthSampleDTO` and cached in `_last_active_accuracy_pct`.
   - `test_record_health_idle_fallback(tmp_path)`: verify idle window (`dynamic_acc is None`) falls back to `_last_active_accuracy_pct` (or `model_accuracy_pct` if `_last_active_accuracy_pct` is None).
   - `test_record_health_degraded_warning_log(tmp_path, caplog)`: verify warning log emitted when dynamic accuracy < 80%.

Follow TDD: Write failing tests in `tests/test_counter_telemetry_accuracy.py`, verify failure, implement changes in `main.py`, verify pass, commit with message:
`feat(telemetry): update health heartbeat with dynamic tracking accuracy`
