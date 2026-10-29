# Task 1: Add `use_dynamic_accuracy` Configuration, CLI Flag, and Telemetry Branching

**Files:**
- Modify: `main.py`
- Modify: `tests/test_counter_telemetry_accuracy.py`

**Interfaces:**
- Consumes: `self.model_accuracy_pct` (from `detector.load_model_accuracy_pct`), `counter.get_tracking_accuracy_pct(reset_window=True)`
- Produces: `HealthSampleDTO.accuracy_pct` set to `model_accuracy_pct` (by default) or smoothed operational accuracy (when enabled).

## Requirements:
1. In `CounterService.__init__` (`main.py`):
   - Accept `use_dynamic_accuracy: Optional[bool] = None`.
   - Precedence:
     ```python
     if use_dynamic_accuracy is None and pipeline_config is not None:
         use_dynamic_accuracy = getattr(pipeline_config, "use_dynamic_accuracy", None)
     if use_dynamic_accuracy is None:
         env_dynamic = os.getenv("USE_DYNAMIC_ACCURACY", "").strip().lower()
         self.use_dynamic_accuracy: bool = env_dynamic in ("true", "1", "yes")
     else:
         self.use_dynamic_accuracy = bool(use_dynamic_accuracy)
     ```
2. In `CounterService._record_health()` (`main.py`):
   ```python
   if not self.use_dynamic_accuracy:
       current_acc = self.model_accuracy_pct
   else:
       dynamic_acc = (
           self.counter.get_tracking_accuracy_pct(reset_window=True)
           if hasattr(self, "counter") and hasattr(self.counter, "get_tracking_accuracy_pct")
           else None
       )
       if dynamic_acc is not None:
           if self._smoothed_accuracy_pct is None:
               self._smoothed_accuracy_pct = round(min(100.0, max(0.0, dynamic_acc)), 2)
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
3. In `build_cli_parser()` (`main.py`):
   ```python
   parser.add_argument(
       "--use-dynamic-accuracy",
       action="store_true",
       default=os.getenv("USE_DYNAMIC_ACCURACY", "").lower() in ("1", "true", "yes"),
       help="Enable dynamic operational tracking accuracy calculation in health telemetry (default: False)",
   )
   ```
4. In `main()` (`main.py`):
   Pass `use_dynamic_accuracy=args.use_dynamic_accuracy` when instantiating `CounterService(..., use_dynamic_accuracy=args.use_dynamic_accuracy)`.
5. Update `tests/test_counter_telemetry_accuracy.py`:
   - Add new tests:
     - `test_telemetry_defaults_to_model_accuracy(tmp_path)`: default records `model_accuracy_pct`, ignores tracking accuracy.
     - `test_telemetry_opt_in_dynamic_accuracy(tmp_path)`: with `use_dynamic_accuracy=True`, records dynamic tracking accuracy.
     - `test_telemetry_dynamic_accuracy_precedence(tmp_path, monkeypatch)`: verifies constructor > env var > default False.
     - `test_degraded_warning_suppressed_when_dynamic_accuracy_disabled(tmp_path, caplog)`: no warning when dynamic accuracy is disabled.
     - `test_cli_parser_use_dynamic_accuracy()`: verifies `--use-dynamic-accuracy` in `build_cli_parser()`.
   - Update existing dynamic tests in `tests/test_counter_telemetry_accuracy.py` that specifically test dynamic EMA behavior to pass `use_dynamic_accuracy=True` so they continue testing dynamic EMA logic.

Follow TDD: Write/update failing tests first, run to verify failure, implement minimal code, verify pass, commit with message:
`feat(telemetry): default health accuracy to model accuracy with dynamic accuracy toggle`
