# Task 1 Report: Add `use_dynamic_accuracy` Configuration, CLI Flag, and Telemetry Branching

## Summary
Reverted default health telemetry accuracy in `CounterService._record_health()` (`main.py`) to static model validation accuracy (`self.model_accuracy_pct`), and provided an opt-in toggle `use_dynamic_accuracy` (default: `False`). When dynamic accuracy is not opted into, telemetry reports `self.model_accuracy_pct` and bypasses dynamic tracking accuracy calls (`counter.get_tracking_accuracy_pct`) and degraded tracking alerts. When `use_dynamic_accuracy=True`, dynamic smoothed tracking accuracy calculations and alerts run cleanly.

Configuration precedence:
1. Constructor argument `use_dynamic_accuracy` in `CounterService`
2. `pipeline_config.use_dynamic_accuracy`
3. `USE_DYNAMIC_ACCURACY` environment variable (`"true"`, `"1"`, `"yes"`)
4. Default: `False`

CLI flag `--use-dynamic-accuracy` added to `build_cli_parser()` and passed to `CounterService` in `main()`.

---

## Changes Implemented

### 1. `main.py`
- **`CounterService.__init__`**:
  - Added parameter `use_dynamic_accuracy: Optional[bool] = None`.
  - Resolved `self.use_dynamic_accuracy` according to precedence:
    ```python
    if use_dynamic_accuracy is None and pipeline_config is not None:
        use_dynamic_accuracy = getattr(pipeline_config, "use_dynamic_accuracy", None)
    if use_dynamic_accuracy is None:
        env_dynamic = os.getenv("USE_DYNAMIC_ACCURACY", "").strip().lower()
        self.use_dynamic_accuracy: bool = env_dynamic in ("true", "1", "yes")
    else:
        self.use_dynamic_accuracy = bool(use_dynamic_accuracy)
    ```
- **`CounterService._record_health()`**:
  - Branched telemetry accuracy evaluation:
    - If `not self.use_dynamic_accuracy`: set `current_acc = self.model_accuracy_pct` directly.
    - Else: compute smoothed dynamic tracking accuracy from `self.counter.get_tracking_accuracy_pct(reset_window=True)` and emit degraded tracking warning (`current_acc < 80.0` when not idle).
- **`build_cli_parser()`**:
  - Added `--use-dynamic-accuracy` CLI argument:
    ```python
    parser.add_argument(
        "--use-dynamic-accuracy",
        action="store_true",
        default=os.getenv("USE_DYNAMIC_ACCURACY", "").lower() in ("1", "true", "yes"),
        help="Enable dynamic operational tracking accuracy calculation in health telemetry (default: False)",
    )
    ```
- **`main()`**:
  - Passed `use_dynamic_accuracy=args.use_dynamic_accuracy` to `CounterService` instantiation.

### 2. `tests/test_counter_telemetry_accuracy.py`
- Added 5 new unit tests:
  1. `test_telemetry_defaults_to_model_accuracy`: verifies default behavior records `model_accuracy_pct` in health samples and does not invoke or record dynamic tracking accuracy.
  2. `test_telemetry_opt_in_dynamic_accuracy`: verifies setting `use_dynamic_accuracy=True` records dynamic tracking accuracy.
  3. `test_telemetry_dynamic_accuracy_precedence`: verifies configuration precedence across constructor argument, `pipeline_config`, environment variable, and default `False`.
  4. `test_degraded_warning_suppressed_when_dynamic_accuracy_disabled`: verifies no degraded warning log is emitted when `use_dynamic_accuracy=False`.
  5. `test_cli_parser_use_dynamic_accuracy`: verifies `--use-dynamic-accuracy` argument parsing, flag behavior, and `USE_DYNAMIC_ACCURACY` environment fallback.
- Updated 9 existing dynamic tracking tests to explicitly pass `use_dynamic_accuracy=True` to test the dynamic tracking branch cleanly.

---

## TDD Execution Summary

1. **RED Phase**:
   - Added new tests in `tests/test_counter_telemetry_accuracy.py`.
   - Executed:
     ```bash
     .venv\Scripts\python.exe -m pytest tests/test_counter_telemetry_accuracy.py -k "test_telemetry_defaults_to_model_accuracy or test_cli_parser_use_dynamic_accuracy"
     ```
   - Confirmed expected RED failure:
     - `AttributeError: 'CounterService' object has no attribute 'use_dynamic_accuracy'`
     - `AttributeError: 'Namespace' object has no attribute 'use_dynamic_accuracy'`

2. **GREEN Phase**:
   - Implemented constructor parameter, precedence resolution, telemetry branching, and CLI flag in `main.py`.
   - Updated existing dynamic tracking tests in `tests/test_counter_telemetry_accuracy.py` with `use_dynamic_accuracy=True`.
   - Re-ran tests:
     ```bash
     .venv\Scripts\python.exe -m pytest tests/test_counter_telemetry_accuracy.py
     ```
     Result: **16 passed in 12.07s**.

3. **Regression Suite**:
   - Executed:
     ```bash
     .venv\Scripts\python.exe -m pytest tests/test_counter_telemetry_accuracy.py tests/test_main.py
     ```
     Result: **57 passed in 57.17s**.

---

## Commits
- `8485352`: `feat(telemetry): default health accuracy to model accuracy with dynamic accuracy toggle`
