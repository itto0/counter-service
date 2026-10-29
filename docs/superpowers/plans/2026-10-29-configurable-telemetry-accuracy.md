# Configurable Telemetry Accuracy Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Revert default health telemetry accuracy to static model validation accuracy (`self.model_accuracy_pct`) and provide an opt-in configuration flag (`use_dynamic_accuracy`, default: `False`) to enable dynamic operational accuracy.

**Architecture:** Add `use_dynamic_accuracy: Optional[bool] = None` to `CounterService` and CLI `--use-dynamic-accuracy`. In `_record_health()`, if `not self.use_dynamic_accuracy`, set `current_acc = self.model_accuracy_pct` and bypass dynamic calculations and degradation alerts. If `self.use_dynamic_accuracy` is `True`, evaluate smoothed dynamic tracking accuracy.

**Tech Stack:** Python 3.10+, argparse, SQLite, pytest.

## Global Constraints
- Default to model accuracy: By default, `HealthSampleDTO.accuracy_pct` must equal `self.model_accuracy_pct`.
- Precedence: constructor argument `use_dynamic_accuracy` > `pipeline_config.use_dynamic_accuracy` > `USE_DYNAMIC_ACCURACY` env var (`"true"`, `"1"`, `"yes"`) > default `False`.
- CLI flag `--use-dynamic-accuracy` exposed on `main.py`.
- Suppress tracking degradation warnings when `use_dynamic_accuracy` is `False`.
- Backward compatibility: Existing `ChickenCounter`, database schemas, and contracts remain unchanged.

---

### Task 1: Add `use_dynamic_accuracy` Configuration, CLI Flag, and Telemetry Branching

**Files:**
- Modify: `main.py`
- Modify: `tests/test_counter_telemetry_accuracy.py`

**Interfaces:**
- Consumes: `self.model_accuracy_pct` (from `detector.load_model_accuracy_pct`), `counter.get_tracking_accuracy_pct(reset_window=True)`
- Produces: `HealthSampleDTO.accuracy_pct` set to `model_accuracy_pct` (by default) or smoothed operational accuracy (when enabled).

- [ ] **Step 1: Write failing unit tests in `tests/test_counter_telemetry_accuracy.py`**

Add tests asserting:
1. Default behavior (`use_dynamic_accuracy=False`): `accuracy_pct` records `self.model_accuracy_pct` and does not call or use `counter.get_tracking_accuracy_pct`.
2. Opt-in behavior (`use_dynamic_accuracy=True`): `accuracy_pct` records smoothed dynamic accuracy.
3. Precedence hierarchy: constructor arg > env var > default `False`.
4. Degraded alert suppression: low tracking accuracy is not logged when `use_dynamic_accuracy=False`.
5. CLI argument `--use-dynamic-accuracy` parses correctly.

```python
def test_telemetry_defaults_to_model_accuracy(tmp_path):
    """Verify CounterService defaults to recording model_accuracy_pct in telemetry."""
    db_file = str(tmp_path / "test_default_model.db")
    storage = LocalStorage(db_path=db_file)

    service = CounterService(
        line_id="lineDefaultModel",
        storage=storage,
        enable_sync=False,
    )
    assert service.use_dynamic_accuracy is False
    service.model_accuracy_pct = 95.5
    service.counter.get_tracking_accuracy_pct = MagicMock(return_value=60.0)

    service._record_health()

    samples = storage.get_unsynced_health(line_id="lineDefaultModel")
    assert len(samples) == 1
    # By default, must record model_accuracy_pct (95.5) NOT dynamic (60.0)
    assert samples[0]["accuracy_pct"] == 95.5


def test_telemetry_opt_in_dynamic_accuracy(tmp_path):
    """Verify CounterService records dynamic tracking accuracy when use_dynamic_accuracy=True."""
    db_file = str(tmp_path / "test_opt_in.db")
    storage = LocalStorage(db_path=db_file)

    service = CounterService(
        line_id="lineOptIn",
        storage=storage,
        enable_sync=False,
        use_dynamic_accuracy=True,
    )
    assert service.use_dynamic_accuracy is True
    service.model_accuracy_pct = 95.5
    service.counter.get_tracking_accuracy_pct = MagicMock(return_value=75.0)

    service._record_health()

    samples = storage.get_unsynced_health(line_id="lineOptIn")
    assert len(samples) == 1
    assert samples[0]["accuracy_pct"] == 75.0


def test_telemetry_dynamic_accuracy_precedence(tmp_path, monkeypatch):
    """Verify precedence: constructor arg > env var > default False."""
    db_file = str(tmp_path / "test_precedence.db")
    storage = LocalStorage(db_path=db_file)

    # 1. Default is False
    monkeypatch.delenv("USE_DYNAMIC_ACCURACY", raising=False)
    s1 = CounterService(line_id="l1", storage=storage, enable_sync=False)
    assert s1.use_dynamic_accuracy is False

    # 2. Env var true
    monkeypatch.setenv("USE_DYNAMIC_ACCURACY", "true")
    s2 = CounterService(line_id="l2", storage=storage, enable_sync=False)
    assert s2.use_dynamic_accuracy is True

    # 3. Constructor arg overrides env var
    s3 = CounterService(line_id="l3", storage=storage, enable_sync=False, use_dynamic_accuracy=False)
    assert s3.use_dynamic_accuracy is False


def test_degraded_warning_suppressed_when_dynamic_accuracy_disabled(tmp_path, caplog):
    """Verify degraded tracking accuracy warning is suppressed when use_dynamic_accuracy=False."""
    db_file = str(tmp_path / "test_suppress.db")
    storage = LocalStorage(db_path=db_file)

    service = CounterService(
        line_id="lineSuppress",
        storage=storage,
        enable_sync=False,
        use_dynamic_accuracy=False,
    )
    service.model_accuracy_pct = 90.0
    service.counter.get_tracking_accuracy_pct = MagicMock(return_value=50.0)

    with caplog.at_level(logging.WARNING, logger="counter.service"):
        service._record_health()

    assert "Operational tracking accuracy degraded" not in caplog.text


def test_cli_parser_use_dynamic_accuracy():
    """Verify --use-dynamic-accuracy flag in CLI parser."""
    from main import build_cli_parser
    parser = build_cli_parser()

    # Default is False
    args_default = parser.parse_args([])
    assert args_default.use_dynamic_accuracy is False

    # Flag passed
    args_flag = parser.parse_args(["--use-dynamic-accuracy"])
    assert args_flag.use_dynamic_accuracy is True
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv\Scripts\python.exe -m pytest tests/test_counter_telemetry_accuracy.py -k "test_telemetry_defaults_to_model_accuracy or test_cli_parser_use_dynamic_accuracy"`
Expected: FAIL (missing attribute `use_dynamic_accuracy` or unrecognized CLI argument).

- [ ] **Step 3: Implement `use_dynamic_accuracy` in `main.py`**

1. In `CounterService.__init__` (`main.py`):
   Accept `use_dynamic_accuracy: Optional[bool] = None`.
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
   Pass `use_dynamic_accuracy=args.use_dynamic_accuracy` when instantiating `CounterService`.

5. Update existing tests in `tests/test_counter_telemetry_accuracy.py` that specifically test dynamic EMA behavior to pass `use_dynamic_accuracy=True`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv\Scripts\python.exe -m pytest tests/test_counter_telemetry_accuracy.py`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add main.py tests/test_counter_telemetry_accuracy.py
git commit -m "feat(telemetry): default health accuracy to model accuracy with dynamic accuracy toggle"
```

---

### Task 2: Full Test Suite Verification & Integration Sanity

**Files:**
- Test: All test suites in `tests/`

- [ ] **Step 1: Run the full pytest suite**

Run: `.venv\Scripts\python.exe -m pytest tests/`
Expected: All tests pass with zero regressions.

- [ ] **Step 2: Commit any final polish if needed**
