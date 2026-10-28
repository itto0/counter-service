# Dynamic Real-Time Accuracy Telemetry Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement dynamic tracking completion accuracy calculation in `ChickenCounter` and integrate it into `CounterService` health telemetry to replace static training metrics with real operational accuracy.

**Architecture:** Extend `ChickenCounter` in `src/tracking/tracker.py` to record completed vs. lost tracks during execution and expose `get_tracking_accuracy_pct()`. In `main.py`, query this accuracy every 10-second health interval, falling back to last active reading or static metrics when idle, and logging warnings if accuracy degrades below 80%.

**Tech Stack:** Python 3.10+, Pytest, OpenCV, NumPy, Pydantic

## Global Constraints

- **Real-Time Tracking Completion Formula:** $\text{accuracy\_pct} = \frac{\text{completed}}{\text{completed} + \text{lost}} \times 100.0$, rounded to 2 decimal places in range `[0.0, 100.0]`.
- **Lost Track Filtering:** Only tracks with `age >= min_hits` (default 2) that expire uncounted qualify as lost tracks (filters 1-frame spurious detections).
- **Idle State Fallback:** When no tracks terminate in the window, retain `_last_active_accuracy_pct` (or static `model_accuracy_pct` if never active).
- **Degraded Logging:** Emit warning log when `accuracy_pct < 80.0%` on an active stream.
- **Zero Schema Breaking Changes:** `HealthSampleDTO.accuracy_pct` remains `float | None` in range `[0.0, 100.0]`.

---

### Task 1: Track Lifecycle Tracking & Completion Accuracy in `ChickenCounter`

**Files:**
- Modify: `src/tracking/tracker.py:210-238`, `360-375`, `450-475`
- Test: `tests/test_tracking.py`

**Interfaces:**
- Consumes: `ChickenCounter` track creation, line crossing, and stale pruning.
- Produces: `ChickenCounter.get_tracking_accuracy_pct(reset_window: bool = True) -> float | None`

- [ ] **Step 1: Write failing tests in `tests/test_tracking.py`**

Add tests covering completion accuracy, lost tracks, noise filtering, idle window, and window resetting:

```python
def test_tracker_completion_accuracy_healthy():
    """Verify 100% accuracy when all valid tracks cross the line."""
    line = VirtualLineZone(start=(0, 200), end=(640, 200), direction="down")
    counter = ChickenCounter(line_zone=line)

    # Track 1 crosses
    det1_a = Detection(x1=100, y1=150, x2=140, y2=190, confidence=0.9, class_id=0, class_name="chicken")
    counter.update([det1_a])
    det1_b = Detection(x1=100, y1=170, x2=140, y2=215, confidence=0.9, class_id=0, class_name="chicken")
    counter.update([det1_b])

    assert counter.total_count == 1
    acc = counter.get_tracking_accuracy_pct(reset_window=False)
    assert acc == 100.0


def test_tracker_completion_accuracy_with_lost_tracks():
    """Verify accuracy drops when tracks die before crossing."""
    line = VirtualLineZone(start=(0, 200), end=(640, 200), direction="down")
    counter = ChickenCounter(line_zone=line, max_age=2)

    # Track 1 crosses
    det1_a = Detection(x1=100, y1=150, x2=140, y2=190, confidence=0.9, class_id=0, class_name="chicken")
    counter.update([det1_a])
    det1_b = Detection(x1=100, y1=170, x2=140, y2=215, confidence=0.9, class_id=0, class_name="chicken")
    counter.update([det1_b])

    # Track 2 appears for 2 frames then disappears without crossing
    det2_a = Detection(x1=300, y1=50, x2=340, y2=90, confidence=0.9, class_id=0, class_name="chicken")
    counter.update([det2_a])
    det2_b = Detection(x1=300, y1=60, x2=340, y2=100, confidence=0.9, class_id=0, class_name="chicken")
    counter.update([det2_b])

    # Advance 4 empty frames to prune Track 2 (max_age=2)
    for _ in range(4):
        counter.update([])

    # 1 completed, 1 lost -> 50.0%
    acc = counter.get_tracking_accuracy_pct(reset_window=True)
    assert acc == 50.0


def test_tracker_completion_accuracy_ignores_single_frame_noise():
    """Verify single-frame transient detections do not count as lost carcass tracks."""
    line = VirtualLineZone(start=(0, 200), end=(640, 200), direction="down")
    counter = ChickenCounter(line_zone=line, max_age=2)

    # Track 1 crosses
    det1_a = Detection(x1=100, y1=150, x2=140, y2=190, confidence=0.9, class_id=0, class_name="chicken")
    counter.update([det1_a])
    det1_b = Detection(x1=100, y1=170, x2=140, y2=215, confidence=0.9, class_id=0, class_name="chicken")
    counter.update([det1_b])

    # Spurious 1-frame noise detection
    noise = Detection(x1=500, y1=50, x2=510, y2=60, confidence=0.5, class_id=0, class_name="chicken")
    counter.update([noise])

    # Advance 4 empty frames to prune noise
    for _ in range(4):
        counter.update([])

    # 1 completed, noise ignored -> 100.0%
    acc = counter.get_tracking_accuracy_pct(reset_window=True)
    assert acc == 100.0


def test_tracker_completion_accuracy_idle_returns_none():
    """Verify get_tracking_accuracy_pct returns None when no tracks terminate in the window."""
    line = VirtualLineZone(start=(0, 200), end=(640, 200), direction="down")
    counter = ChickenCounter(line_zone=line)

    assert counter.get_tracking_accuracy_pct() is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_tracking.py -k test_tracker_completion_accuracy`
Expected: FAIL with `AttributeError: 'ChickenCounter' object has no attribute 'get_tracking_accuracy_pct'`

- [ ] **Step 3: Implement tracking lifecycle in `src/tracking/tracker.py`**

1. In `ChickenCounter.__init__`:
```python
self._window_completed_tracks: int = 0
self._window_lost_tracks: int = 0
self._total_completed_tracks: int = 0
self._total_lost_tracks: int = 0
self._min_track_hits: int = 2
```

2. In line crossing detection (both `_update_bytetrack` and `_update_centroid`):
```python
if track_id not in self.counted_track_ids and not track.counted:
    if self.line_zone.crosses(track.last_anchor, track.current_anchor):
        track.counted = True
        self.counted_track_ids.add(track_id)
        self.total_count += 1
        self._window_completed_tracks += 1
        self._total_completed_tracks += 1
```

3. In stale track pruning (both `_update_bytetrack` and `_update_centroid`):
```python
for track_id in stale_ids:
    track = self.tracks[track_id]
    if not track.counted and track.age >= self._min_track_hits:
        self._window_lost_tracks += 1
        self._total_lost_tracks += 1
    del self.tracks[track_id]
```

4. Add `get_tracking_accuracy_pct()`:
```python
def get_tracking_accuracy_pct(self, reset_window: bool = True) -> float | None:
    """Calculate tracking completion accuracy percentage for the current window.

    Returns:
        Accuracy percentage in range [0.0, 100.0] or None if no tracks finished in window.
    """
    total = self._window_completed_tracks + self._window_lost_tracks
    if total == 0:
        return None
    acc = (self._window_completed_tracks / total) * 100.0
    if reset_window:
        self._window_completed_tracks = 0
        self._window_lost_tracks = 0
    return round(min(100.0, max(0.0, acc)), 2)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_tracking.py -k test_tracker_completion_accuracy`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/tracking/tracker.py tests/test_tracking.py
git commit -m "feat(tracker): implement tracking completion accuracy calculation"
```

---

### Task 2: Dynamic Telemetry Integration & Degraded Stream Alerting in `CounterService`

**Files:**
- Modify: `main.py:320-340`, `495-530`
- Create: `tests/test_counter_telemetry_accuracy.py`

**Interfaces:**
- Consumes: `ChickenCounter.get_tracking_accuracy_pct()`
- Produces: Dynamic `HealthSampleDTO.accuracy_pct` recorded into SQLite storage.

- [ ] **Step 1: Write failing tests in `tests/test_counter_telemetry_accuracy.py`**

```python
"""Tests for dynamic tracking accuracy integration in CounterService health telemetry."""

from unittest.mock import MagicMock
import pytest
from main import CounterService
from src.storage.db import LocalStorage
from src.detection.detector import Detection
from src.tracking.tracker import VirtualLineZone, ChickenCounter


def test_record_health_uses_dynamic_tracking_accuracy(tmp_path):
    """Verify _record_health populates accuracy_pct with dynamic tracking accuracy."""
    db_file = str(tmp_path / "test_health_acc.db")
    storage = LocalStorage(db_path=db_file)

    service = CounterService(
        line_id="lineAccTest",
        video_source_type="synthetic",
        storage=storage,
    )
    # Simulate a dynamic accuracy reading of 85.5%
    service.counter.get_tracking_accuracy_pct = MagicMock(return_value=85.5)

    service._record_health(now_perf=100.0)

    samples = storage.get_unsynced_batch(batch_size=10).health
    assert len(samples) == 1
    assert samples[0].accuracy_pct == 85.5
    assert service._last_active_accuracy_pct == 85.5


def test_record_health_idle_fallback(tmp_path):
    """Verify _record_health falls back to last active accuracy or model accuracy when idle."""
    db_file = str(tmp_path / "test_health_idle.db")
    storage = LocalStorage(db_path=db_file)

    service = CounterService(
        line_id="lineAccTest",
        video_source_type="synthetic",
        storage=storage,
    )
    service.model_accuracy_pct = 94.2
    service._last_active_accuracy_pct = 88.0

    # Idle: returns None
    service.counter.get_tracking_accuracy_pct = MagicMock(return_value=None)

    service._record_health(now_perf=100.0)

    samples = storage.get_unsynced_batch(batch_size=10).health
    assert len(samples) == 1
    assert samples[0].accuracy_pct == 88.0


def test_record_health_degraded_warning_log(tmp_path, caplog):
    """Verify a warning is logged when operational accuracy drops below 80%."""
    import logging

    db_file = str(tmp_path / "test_health_warn.db")
    storage = LocalStorage(db_path=db_file)

    service = CounterService(
        line_id="lineAccTest",
        video_source_type="synthetic",
        storage=storage,
    )
    service.counter.get_tracking_accuracy_pct = MagicMock(return_value=65.0)

    with caplog.at_level(logging.WARNING):
        service._record_health(now_perf=100.0)

    samples = storage.get_unsynced_batch(batch_size=10).health
    assert samples[0].accuracy_pct == 65.0
    assert any("Operational tracking accuracy degraded: 65.0%" in record.message for record in caplog.records)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_counter_telemetry_accuracy.py`
Expected: FAIL with `AssertionError: assert None == 85.5` or `AttributeError: '_last_active_accuracy_pct'`

- [ ] **Step 3: Implement dynamic health accuracy querying and idle fallback in `main.py`**

1. In `CounterService.__init__`:
```python
self._last_active_accuracy_pct: float | None = None
```

2. In `CounterService._record_health`:
Replace the static `accuracy_pct=self.model_accuracy_pct` with dynamic query and fallback:
```python
# Query dynamic tracking completion accuracy
dynamic_acc = self.counter.get_tracking_accuracy_pct(reset_window=True)
if dynamic_acc is not None:
    self._last_active_accuracy_pct = dynamic_acc
    current_acc = dynamic_acc
else:
    # Idle conveyor fallback: use last known active accuracy, or static model accuracy
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
Pass `accuracy_pct=current_acc` to `HealthSampleDTO`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_counter_telemetry_accuracy.py`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add main.py tests/test_counter_telemetry_accuracy.py
git commit -m "feat(telemetry): update health heartbeat with dynamic tracking accuracy"
```

---

### Task 3: Full Test Suite Verification & Integration Sanity

**Files:**
- Test: All test suites in `tests/`

- [ ] **Step 1: Run the full pytest suite**

Run: `pytest`
Expected: All tests pass with zero regressions.

- [ ] **Step 2: Commit any final cleanup if needed**

```bash
git status
```
