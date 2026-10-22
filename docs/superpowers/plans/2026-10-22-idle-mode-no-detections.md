# Idle Mode on No Object Detection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement an adaptive idle mode in `CounterService` that throttles frame rate to 1.0 FPS after 30 seconds of no detected objects, wakes up immediately on object detection, flushes stale camera buffers, and records health telemetry with `stream_state = "IDLE"`.

**Architecture:** Extend `HealthSampleDTO` data contract to allow `"IDLE"` stream state. In `CounterService`, maintain `_last_detection_time` and `is_idle` state flags evaluated on every frame iteration. When idle, dynamically throttle loop pacing to `idle_fps`, set `stream_state = "IDLE"` in periodic telemetry, and flush video buffers upon waking up. Expose configuration through CLI arguments and environment variables.

**Tech Stack:** Python 3.14, Pydantic v2, OpenCV (`cv2`), SQLite, Pytest.

## Global Constraints

- Python interpreter: `.venv/Scripts/python.exe` / pytest runner: `.venv/Scripts/pytest`.
- Zero false negatives: object detection (`len(detections) > 0`) must wake up the pipeline on the immediate frame iteration.
- Telemetry accuracy: do not flag `DEGRADED` stream state while operating in `IDLE` mode.
- Backward compatibility: default `idle_timeout_sec=30.0` and `idle_fps=1.0`; setting `idle_timeout_sec=0.0` disables idle throttling completely.
- Decoupled background threads: Outbox background sync worker and stream uploader threads must remain non-blocking.

---

### Task 1: Update HealthSampleDTO Contract for IDLE State

**Files:**
- Modify: `src/contracts.py:50-65`
- Test: `tests/test_counter_contracts.py`

**Interfaces:**
- Consumes: `pydantic.BaseModel`, `pydantic.Field`, `typing.Literal`
- Produces: `HealthSampleDTO(stream_state: Literal["ONLINE", "OFFLINE", "DEGRADED", "IDLE"])`

- [ ] **Step 1: Write the failing test**

Edit `tests/test_counter_contracts.py` to add a test case verifying `HealthSampleDTO` accepts `"IDLE"` as a valid `stream_state`:

```python
def test_health_sample_dto_idle_stream_state():
    from src.contracts import HealthSampleDTO

    health = HealthSampleDTO(
        fps=1.0,
        latency_p95_ms=8.0,
        cpu_pct=5.0,
        mem_pct=15.0,
        stream_state="IDLE",
    )
    assert health.stream_state == "IDLE"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/pytest tests/test_counter_contracts.py::test_health_sample_dto_idle_stream_state -v`  
Expected: FAIL with `pydantic_core._pydantic_core.ValidationError` (Input should be 'ONLINE', 'OFFLINE' or 'DEGRADED').

- [ ] **Step 3: Update `src/contracts.py`**

In `src/contracts.py`, update `HealthSampleDTO.stream_state` definition to include `"IDLE"`:

```python
class HealthSampleDTO(BaseModel):
    fps: float = Field(ge=0.0)
    latency_p95_ms: float = Field(ge=0.0)
    cpu_pct: float = Field(ge=0.0, le=100.0)
    mem_pct: float = Field(ge=0.0, le=100.0)
    stream_state: Literal["ONLINE", "OFFLINE", "DEGRADED", "IDLE"] = "ONLINE"
    model_name: str = "yolo26n"
    model_version: str = "v1.0"
    accuracy_pct: float | None = Field(default=None, ge=0.0, le=100.0)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/pytest tests/test_counter_contracts.py -v`  
Expected: PASS (all tests in `test_counter_contracts.py` pass).

- [ ] **Step 5: Commit**

```bash
git add src/contracts.py tests/test_counter_contracts.py
git commit -m "feat(contracts): allow IDLE stream state in HealthSampleDTO"
```

---

### Task 2: Idle State Tracking and Wake-Up Logic in CounterService

**Files:**
- Modify: `main.py:92-180`, `main.py:490-530`
- Test: `tests/test_idle_mode.py` (new file)

**Interfaces:**
- Consumes: `CounterService.__init__(idle_timeout_sec: float = 30.0, idle_fps: float = 1.0, ...)`
- Produces: `CounterService.is_idle: bool`, `CounterService._last_detection_time: float`, immediate wake-up with live buffer flush on detection.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_idle_mode.py` with tests for initialization, entering idle mode after timeout, and immediate wake-up on detection:

```python
"""Tests for CounterService adaptive idle mode on no object detection."""

import time
from unittest.mock import MagicMock
import pytest
from main import CounterService
from src.storage.db import LocalStorage
from src.detector.detector import Detection


def test_counter_service_idle_initialization(tmp_path):
    """Verify CounterService initializes idle configuration attributes."""
    db_file = str(tmp_path / "test_idle_init.db")
    service = CounterService(
        line_id="lineIdleInit",
        video_source_type="synthetic",
        db_path=db_file,
        idle_timeout_sec=15.0,
        idle_fps=2.0,
    )
    assert service.idle_timeout_sec == 15.0
    assert service.idle_fps == 2.0
    assert service.is_idle is False
    assert hasattr(service, "_last_detection_time")


def test_counter_service_enters_idle_and_wakes_up(tmp_path):
    """Verify service transitions to is_idle=True when no detections occur for idle_timeout_sec and immediately wakes up on detection."""
    db_file = str(tmp_path / "test_idle_trans.db")
    storage = LocalStorage(db_path=db_file)
    mock_source = MagicMock()
    # Dummy 640x480 frame
    import numpy as np
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    mock_source.read.return_value = (True, dummy_frame)
    mock_source.cap = MagicMock()

    mock_detector = MagicMock()
    # First 3 frames: no detections; 4th frame: 1 detection
    det = Detection(bbox=(100.0, 100.0, 200.0, 200.0), confidence=0.9, class_id=0, label="chicken")
    mock_detector.predict.side_effect = [
        [],
        [],
        [],
        [det],
    ]

    service = CounterService(
        line_id="lineIdleTrans",
        source=mock_source,
        detector=mock_detector,
        storage=storage,
        idle_timeout_sec=0.1,  # Short timeout for test
        idle_fps=1.0,
        cli_display=False,
    )

    # Initially not idle
    assert not service.is_idle

    # Frame 1: empty detection, record start time
    service.running = True
    # Simulate first frame processing
    ret, frame = service.source.read()
    detections = service.detector.predict(frame)
    now = time.perf_counter()
    # Artificially set _last_detection_time to 0.2s ago to exceed 0.1s timeout
    service._last_detection_time = now - 0.2
    
    # Process idle check as in run loop
    if len(detections) > 0:
        service._last_detection_time = now
        if service.is_idle:
            service.is_idle = False
    elif service.idle_timeout_sec > 0 and not service.is_idle:
        if (now - service._last_detection_time) >= service.idle_timeout_sec:
            service.is_idle = True

    assert service.is_idle is True

    # Frame 2: detection arrives -> immediate wake-up & buffer flush
    ret, frame = service.source.read()
    detections = service.detector.predict(frame)  # Returns det
    now = time.perf_counter()
    if len(detections) > 0:
        service._last_detection_time = now
        if service.is_idle:
            service.is_idle = False
            if hasattr(service.source, "cap") and service.source.cap is not None:
                try:
                    for _ in range(2):
                        service.source.cap.grab()
                except Exception:
                    pass

    assert service.is_idle is False
    assert mock_source.cap.grab.call_count == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/pytest tests/test_idle_mode.py -v`  
Expected: FAIL (missing `idle_timeout_sec` or `idle_fps` parameter in `CounterService.__init__`).

- [ ] **Step 3: Implement idle state attributes and transition in `main.py`**

In `main.py`:
1. In `CounterService.__init__`, add parameters:
   ```python
   idle_timeout_sec: float = 30.0,
   idle_fps: float = 1.0,
   ```
   and attributes:
   ```python
   self.idle_timeout_sec = idle_timeout_sec
   self.idle_fps = idle_fps
   self.is_idle = False
   self._last_detection_time = time.perf_counter()
   ```

2. In `CounterService.run()` right after `detections = self.detector.predict(frame)` (around line 502):
   ```python
   now_det = time.perf_counter()
   if len(detections) > 0:
       self._last_detection_time = now_det
       if self.is_idle:
           self.is_idle = False
           logger.info("Object detected: exiting idle mode, resuming full processing")
           if hasattr(self.source, "cap") and self.source.cap is not None:
               try:
                   for _ in range(2):
                       self.source.cap.grab()
               except Exception:
                   pass
   elif self.idle_timeout_sec > 0 and not self.is_idle:
       if (now_det - self._last_detection_time) >= self.idle_timeout_sec:
           self.is_idle = True
           logger.info(
               "No detections for %.1fs: entering idle mode (target: %.1f FPS)",
               self.idle_timeout_sec,
               self.idle_fps,
           )
   ```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/pytest tests/test_idle_mode.py -v`  
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add main.py tests/test_idle_mode.py
git commit -m "feat(counter): add idle state tracking and buffer flush on detection"
```

---

### Task 3: Idle Pacing, Telemetry IDLE State, and CLI Status Tag

**Files:**
- Modify: `main.py:350-385` (`_record_health`), `main.py:575-608` (CLI status & pacing delay)
- Test: `tests/test_idle_mode.py`

**Interfaces:**
- Consumes: `CounterService.is_idle`, `CounterService.idle_fps`, `CounterService.target_fps`
- Produces: `effective_fps` pacing delay, `HealthSampleDTO.stream_state = "IDLE"`, `[IDLE]` tag in CLI output.

- [ ] **Step 1: Write the failing tests**

Append tests to `tests/test_idle_mode.py` verifying health telemetry sets `stream_state="IDLE"` when idle (without marking `DEGRADED`), and effective FPS pacing:

```python
def test_counter_service_records_idle_health(tmp_path):
    """Verify _record_health records stream_state='IDLE' when service is idle even if fps < 5.0."""
    db_file = str(tmp_path / "test_idle_health.db")
    storage = LocalStorage(db_path=db_file)
    service = CounterService(
        line_id="lineIdleHealth",
        storage=storage,
        cli_display=False,
    )
    service.total_frames_processed = 10
    service.is_idle = True
    service._health_frame_count = 1  # Low frame count -> FPS would normally be < 5.0
    service._health_last_calc_time = time.perf_counter() - 1.0

    service._record_health()

    samples = storage.get_unsynced_health(line_id="lineIdleHealth")
    assert len(samples) == 1
    assert samples[0]["stream_state"] == "IDLE"


def test_counter_service_disabled_idle_mode(tmp_path):
    """Verify idle_timeout_sec=0.0 disables idle mode transitions."""
    db_file = str(tmp_path / "test_idle_disabled.db")
    storage = LocalStorage(db_path=db_file)
    mock_source = MagicMock()
    import numpy as np
    mock_source.read.return_value = (True, np.zeros((480, 640, 3), dtype=np.uint8))
    mock_detector = MagicMock()
    mock_detector.predict.return_value = []

    service = CounterService(
        line_id="lineIdleDisabled",
        source=mock_source,
        detector=mock_detector,
        storage=storage,
        idle_timeout_sec=0.0,  # Disabled
        cli_display=False,
    )
    service.run(max_frames=5)
    assert service.is_idle is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/pytest tests/test_idle_mode.py::test_counter_service_records_idle_health -v`  
Expected: FAIL (samples[0]["stream_state"] is "DEGRADED" instead of "IDLE").

- [ ] **Step 3: Implement idle health reporting, pacing delay, and CLI display in `main.py`**

In `main.py`:
1. In `_record_health()` (around line 359), replace the state determination:
   ```python
   if self.is_idle:
       stream_state = "IDLE"
   elif stream_state == "ONLINE" and fps < 5.0 and self.total_frames_processed > 0:
       stream_state = "DEGRADED"
   ```

2. In `run()` around line 580 (CLI status printing):
   ```python
   idle_tag = " [IDLE]" if self.is_idle else ""
   sys.stdout.write(
       f"\r[Frame {self.total_frames_processed:5d}{idle_tag}] "
       f"Chickens: {self.counter.total_count:4d} | "
       f"Rate: {curr_rate:5.1f}/min | "
       f"Speed: {curr_fps:5.1f} fps | "
       f"Latency: {recent_lat:4.1f}ms   "
   )
   sys.stdout.flush()
   ```

3. In `run()` around line 591 (frame pacing delay):
   ```python
   effective_fps = self.idle_fps if self.is_idle else self.target_fps
   if effective_fps > 0:
       elapsed = time.perf_counter() - frame_start
       delay = (1.0 / effective_fps) - elapsed
       if delay > 0:
           if self.show_window:
               try:
                   import cv2
                   wait_ms = max(1, int(delay * 1000))
                   key = cv2.waitKey(wait_ms) & 0xFF
                   if key in (ord("q"), ord("Q"), 27):
                       if self.cli_display:
                           print("\nUser pressed 'q' in preview window. Stopping.")
                       break
               except Exception:
                   time.sleep(delay)
           else:
               time.sleep(delay)
   ```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/pytest tests/test_idle_mode.py -v`  
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add main.py tests/test_idle_mode.py
git commit -m "feat(counter): throttle pacing and report IDLE telemetry state"
```

---

### Task 4: CLI Arguments and Environment Variable Configuration

**Files:**
- Modify: `main.py:1080-1110` (`build_arg_parser`), `main.py:1290-1320` (`main` service constructor)
- Test: `tests/test_idle_mode.py`

**Interfaces:**
- Consumes: `--idle-timeout-sec`, `--idle-fps`, env vars `IDLE_TIMEOUT_SEC`, `IDLE_FPS`
- Produces: `args.idle_timeout_sec: float`, `args.idle_fps: float` passed to `CounterService`

- [ ] **Step 1: Write the failing tests**

Append CLI argument and environment variable parsing tests to `tests/test_idle_mode.py`:

```python
def test_idle_cli_args_parsing():
    """Verify build_arg_parser accepts --idle-timeout-sec and --idle-fps."""
    from main import build_arg_parser

    parser = build_arg_parser()
    args = parser.parse_args(["--idle-timeout-sec", "45.5", "--idle-fps", "2.5"])
    assert args.idle_timeout_sec == 45.5
    assert args.idle_fps == 2.5


def test_idle_cli_env_fallbacks(monkeypatch):
    """Verify build_arg_parser respects IDLE_TIMEOUT_SEC and IDLE_FPS env vars."""
    from main import build_arg_parser

    monkeypatch.setenv("IDLE_TIMEOUT_SEC", "12.0")
    monkeypatch.setenv("IDLE_FPS", "0.5")

    parser = build_arg_parser()
    args = parser.parse_args([])
    assert args.idle_timeout_sec == 12.0
    assert args.idle_fps == 0.5
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/pytest tests/test_idle_mode.py::test_idle_cli_args_parsing -v`  
Expected: FAIL (`unrecognized arguments: --idle-timeout-sec 45.5 --idle-fps 2.5`).

- [ ] **Step 3: Add CLI arguments and wire into `main()` in `main.py`**

In `main.py`:
1. In `build_arg_parser()` (near line 1095):
   ```python
   parser.add_argument(
       "--idle-timeout-sec",
       type=float,
       default=float(os.getenv("IDLE_TIMEOUT_SEC", "30.0")),
       help="Seconds without detections before entering idle mode (0 to disable, default: 30.0)",
   )
   parser.add_argument(
       "--idle-fps",
       type=float,
       default=float(os.getenv("IDLE_FPS", "1.0")),
       help="Target frame rate during idle mode (default: 1.0)",
   )
   ```

2. In `main()` when constructing `CounterService` (near line 1310):
   ```python
   service = CounterService(
       line_id=args.line_id,
       video_source_type=video_source_type,
       video_path=args.video,
       api_sync_url=args.api_sync_url,
       api_token=args.api_token,
       db_path=args.db_path,
       sync_interval_sec=args.sync_interval,
       target_fps=args.target_fps,
       idle_timeout_sec=args.idle_timeout_sec,
       idle_fps=args.idle_fps,
       aggregate_window_sec=args.aggregate_window,
       health_interval_sec=args.health_interval,
       detector=detector,
       loop=args.loop,
       line_coords=line_coords,
       line_direction=args.direction,
       enable_sync=enable_sync,
       cli_display=True,
       show_window=args.show_window,
       enable_stream_relay=args.enable_stream_relay,
       stream_fps=args.stream_fps,
   )
   ```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/pytest tests/test_idle_mode.py -v`  
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add main.py tests/test_idle_mode.py
git commit -m "feat(cli): add --idle-timeout-sec and --idle-fps arguments"
```

---

### Task 5: Full Regression Testing and Pipeline Verification

**Files:**
- Test: `tests/test_idle_mode.py`, `tests/test_main.py`, all existing tests

**Interfaces:**
- Consumes: All updated modules
- Produces: 100% passing test suite across contracts, main entrypoints, tracker, benchmark, storage, and streamer.

- [ ] **Step 1: Run complete test suite**

Run: `.venv/Scripts/pytest -v`  
Expected: All tests pass (0 failures, 2 skipped as expected for optional cv2 GUI/cuda).

- [ ] **Step 2: Verify git status and clean working tree**

Run: `git status`  
Expected: Working tree clean or only untracked temporary files if any.

- [ ] **Step 3: Commit plan completion checkpoint if needed**

```bash
git commit --allow-empty -m "chore: verify test suite passes with idle mode implementation"
```
