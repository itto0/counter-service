# Task 2: Polish & Full Test Suite Verification

**Files:**
- Modify: `main.py`
- Modify: `tests/test_counter_telemetry_accuracy.py`
- Test: All test suites in `tests/`

## Requirements:
1. In `main.py`:
   - In `build_cli_parser()` line 1440, add `.strip()`: `default=os.getenv("USE_DYNAMIC_ACCURACY", "").strip().lower() in ("1", "true", "yes")`
2. In `tests/test_counter_telemetry_accuracy.py`:
   - In `test_telemetry_defaults_to_model_accuracy`, assert `service.counter.get_tracking_accuracy_pct.assert_not_called()`.
   - Add test checking that `main()` forwards `args.use_dynamic_accuracy` into `CounterService(..., use_dynamic_accuracy=...)` when invoked.
3. Run the full pytest suite:
   `.venv\Scripts\python.exe -m pytest tests/`
   Verify all tests pass (2 skipped standalone tests expected).
4. Commit:
   `fix(telemetry): strip env var in cli parser and verify main argument forwarding`
5. Write report to `docs/superpowers/plans/task-2-report-config-acc.md`.
