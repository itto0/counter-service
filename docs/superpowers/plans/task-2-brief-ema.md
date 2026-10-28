# Task 2: Polish, Robust Parameter Parsing & Full Test Suite Verification

**Files:**
- Modify: `main.py`
- Modify: `tests/test_counter_telemetry_accuracy.py`
- Run: Full test suite across `tests/`

## Requirements:
1. In `main.py`:
   - Safeguard `accuracy_ema_alpha` parsing against empty string or invalid input from env var:
     ```python
     raw_alpha = accuracy_ema_alpha if accuracy_ema_alpha is not None else (os.getenv("ACCURACY_EMA_ALPHA") or "0.15")
     try:
         parsed_alpha = float(raw_alpha)
     except (ValueError, TypeError):
         parsed_alpha = 0.15
     self.accuracy_ema_alpha: float = min(1.0, max(0.01, parsed_alpha))
     ```
   - On first active window seeding in `_record_health()`, clamp and round defensively:
     ```python
     if self._smoothed_accuracy_pct is None:
         self._smoothed_accuracy_pct = round(min(100.0, max(0.0, dynamic_acc)), 2)
     ```
2. In `tests/test_counter_telemetry_accuracy.py`:
   - Add test case verifying that degraded accuracy (`< 80.0%`) does NOT emit warning log when `stream_state == "IDLE"`.
   - Verify fallback handling with invalid `ACCURACY_EMA_ALPHA=""` or `"invalid"`.
3. Run full test suite:
   `.venv\Scripts\python.exe -m pytest tests/`
   Verify 244/244 active tests pass with zero failures.
4. Commit with message:
   `fix(telemetry): harden alpha parsing, seed clamping, and idle warning suppression`
5. Write report to `docs/superpowers/plans/task-2-report-ema.md`.
