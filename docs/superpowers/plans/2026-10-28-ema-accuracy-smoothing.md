# EMA Accuracy Telemetry Smoothing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Dampen and smooth runtime operational tracking accuracy in `CounterService` health telemetry using Exponential Moving Average (EMA) with $\alpha=0.15$ and unbiased startup seeding.

**Architecture:** Maintain `_smoothed_accuracy_pct` and `accuracy_ema_alpha` in `CounterService`. On each 10s health tick with active tracking results, update the smoothed score via $Smoothed_t = \alpha \cdot Raw_t + (1 - \alpha) \cdot Smoothed_{t-1}$. When idle, preserve the smoothed score. Pass smoothed accuracy to `HealthSampleDTO`.

**Tech Stack:** Python 3.10+, SQLite (via LocalStorage), pytest.

## Global Constraints
- Smoothing formula: $Smoothed_t = \alpha \cdot Raw_t + (1 - \alpha) \cdot Smoothed_{t-1}$ rounded to 2 decimal places and clamped in $[0.0, 100.0]$.
- Default $\alpha = 0.15$, configurable via `accuracy_ema_alpha` parameter or `ACCURACY_EMA_ALPHA` environment variable.
- Unbiased seeding: On the first active window with `_smoothed_accuracy_pct is None`, seed directly with $Raw_0$.
- Idle conveyor fallback: When $Raw_t$ is `None`, preserve existing `_smoothed_accuracy_pct`; if `_smoothed_accuracy_pct` is `None`, fall back to `model_accuracy_pct`.
- Degraded alert logging: Emit warning when `stream_state != "IDLE"` and smoothed accuracy < 80.0%.

---

### Task 1: Implement EMA Smoothing in `CounterService` and Comprehensive Tests

**Files:**
- Modify: `main.py`
- Modify: `tests/test_counter_telemetry_accuracy.py`

**Interfaces:**
- Consumes: `ChickenCounter.get_tracking_accuracy_pct(reset_window: bool = True) -> float | None`
- Produces: Smoothed `HealthSampleDTO.accuracy_pct` recorded into SQLite storage.

- [ ] **Step 1: Write the failing tests in `tests/test_counter_telemetry_accuracy.py`**

Add tests for seeding, gradual degradation, idle preservation, and custom alpha:
```python
def test_smoothed_accuracy_seeded_on_first_active_window(tmp_path):
    """Verify first active window directly seeds _smoothed_accuracy_pct without EMA distortion."""
    db_file = str(tmp_path / "test_seed.db")
    storage = LocalStorage(db_path=db_file)

    service = CounterService(
        line_id="lineSeed",
        storage=storage,
        enable_sync=False,
    )
    assert service._smoothed_accuracy_pct is None

    # First active window: raw is 85.0%
    service.counter.get_tracking_accuracy_pct = MagicMock(return_value=85.0)
    service._record_health()

    assert service._smoothed_accuracy_pct == 85.0
    samples = storage.get_unsynced_health(line_id="lineSeed")
    assert samples[0]["accuracy_pct"] == 85.0


def test_smoothed_accuracy_ema_gradual_decay_and_recovery(tmp_path):
    """Verify sequential raw readings cause smooth, gradual decay and recovery with alpha=0.15."""
    db_file = str(tmp_path / "test_ema.db")
    storage = LocalStorage(db_path=db_file)

    service = CounterService(
        line_id="lineEMA",
        storage=storage,
        enable_sync=False,
        accuracy_ema_alpha=0.15,
    )

    # Window 1: Seed at 100.0%
    service.counter.get_tracking_accuracy_pct = MagicMock(return_value=100.0)
    service._record_health()
    assert service._smoothed_accuracy_pct == 100.0

    # Window 2: Degraded raw reading 60.0%
    # Expected: 0.15 * 60.0 + 0.85 * 100.0 = 9.0 + 85.0 = 94.0
    service.counter.get_tracking_accuracy_pct = MagicMock(return_value=60.0)
    service._record_health()
    assert service._smoothed_accuracy_pct == 94.0

    # Window 3: Degraded raw reading 60.0%
    # Expected: 0.15 * 60.0 + 0.85 * 94.0 = 9.0 + 79.9 = 88.9
    service.counter.get_tracking_accuracy_pct = MagicMock(return_value=60.0)
    service._record_health()
    assert service._smoothed_accuracy_pct == 88.9

    # Window 4: Recovered raw reading 100.0%
    # Expected: 0.15 * 100.0 + 0.85 * 88.9 = 15.0 + 75.565 = 90.57
    service.counter.get_tracking_accuracy_pct = MagicMock(return_value=100.0)
    service._record_health()
    assert service._smoothed_accuracy_pct == 90.57


def test_smoothed_accuracy_idle_preserves_score(tmp_path):
    """Verify idle windows do not change the smoothed accuracy score."""
    db_file = str(tmp_path / "test_idle_preserve.db")
    storage = LocalStorage(db_path=db_file)

    service = CounterService(
        line_id="lineIdlePreserve",
        storage=storage,
        enable_sync=False,
    )

    # Active window
    service.counter.get_tracking_accuracy_pct = MagicMock(return_value=92.0)
    service._record_health()
    assert service._smoothed_accuracy_pct == 92.0

    # Idle window (None)
    service.counter.get_tracking_accuracy_pct = MagicMock(return_value=None)
    service._record_health()
    assert service._smoothed_accuracy_pct == 92.0

    samples = storage.get_unsynced_health(line_id="lineIdlePreserve")
    assert len(samples) == 2
    assert samples[1]["accuracy_pct"] == 92.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv\Scripts\python.exe -m pytest tests/test_counter_telemetry_accuracy.py -k test_smoothed_accuracy`
Expected: FAIL with `AttributeError: 'CounterService' object has no attribute '_smoothed_accuracy_pct'`

- [ ] **Step 3: Implement EMA smoothing in `main.py`**

In `CounterService.__init__`:
```python
self._smoothed_accuracy_pct: Optional[float] = None
self.accuracy_ema_alpha: float = (
    float(accuracy_ema_alpha)
    if accuracy_ema_alpha is not None
    else float(os.getenv("ACCURACY_EMA_ALPHA", "0.15"))
)
```

In `CounterService._record_health()`:
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

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv\Scripts\python.exe -m pytest tests/test_counter_telemetry_accuracy.py`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add main.py tests/test_counter_telemetry_accuracy.py
git commit -m "feat(telemetry): smooth operational accuracy using exponential moving average"
```

---

### Task 2: Full Test Suite Verification & Integration Sanity

**Files:**
- Test: All test suites in `tests/`

- [ ] **Step 1: Run the full pytest suite**

Run: `.venv\Scripts\python.exe -m pytest tests/`
Expected: All tests pass with zero regressions.

- [ ] **Step 2: Commit any final cleanup if needed**
