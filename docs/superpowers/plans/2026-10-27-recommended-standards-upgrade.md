# Recommended Standards Upgrade Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Upgrade `counter-service` to production edge CV standards with threaded zero-lag capture (size-1 frame queue and auto-reconnect), ByteTrack tracking (Kalman filter + 2-stage association), and dynamic ONNX hardware acceleration (CUDA/OpenVINO/DirectML cascade with CPU fallback).

**Architecture:** Layered in-place upgrade maintaining full contract compatibility with `main.py`, local SQLite WAL outbox, and central sync API. `ThreadedVideoSource` decouples RTSP frame consumption from inference; `ChickenCounter` integrates ByteTrack via a `supervision`/`trackers` adapter with centroid fallback; `ONNXDetector` auto-negotiates execution providers with resilient CPU recovery.

**Tech Stack:** Python 3.11+, OpenCV, ONNX Runtime (`onnxruntime`), ByteTrack (`trackers`, `supervision`, `scipy`), SQLite (WAL), Pydantic v2, HTTPX, Pytest, `uv`.

## Global Constraints

- Preserve existing `CountEventDTO`, `CountAggregateDTO`, and SQLite outbox schemas without breaking changes.
- All existing tests in `tests/` must continue to pass (`uv run pytest`).
- `ChickenCounter` must fall back gracefully to centroid tracking if `supervision` or `trackers` is unavailable.
- `ONNXDetector` must fall back gracefully to `CPUExecutionProvider` if hardware acceleration initialization fails.
- All CLI interfaces in `main.py` must retain current defaults while supporting new configuration flags.

---

### Task 1: Dependencies and Configuration Contracts

**Files:**
- Modify: `requirements.txt`
- Modify: `src/contracts.py`
- Test: `tests/test_contracts.py`

**Interfaces:**
- Consumes: None
- Produces: `supervision`, `trackers`, `scipy`, `av` in requirements; config constants/models in `src/contracts.py`.

- [ ] **Step 1: Write the failing test for configuration contracts**

Add configuration model validation test to `tests/test_contracts.py`:

```python
def test_capture_and_tracker_config():
    from src.contracts import PipelineConfig

    config = PipelineConfig()
    assert config.capture_mode == "threaded"
    assert config.capture_pace_files is True
    assert config.tracker_type == "bytetrack"
    assert config.onnx_providers == "auto"
    assert config.tracker_track_thresh == 0.4
    assert config.tracker_match_thresh == 0.7
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_contracts.py -k test_capture_and_tracker_config`  
Expected: FAIL with `ImportError: cannot import name 'PipelineConfig' from 'src.contracts'`

- [ ] **Step 3: Update `requirements.txt` and `src/contracts.py`**

In `requirements.txt`, append:
```text
supervision>=0.20.0
trackers>=2.6.0
scipy>=1.11.0
av>=12.0.0
```

In `src/contracts.py`, add `PipelineConfig`:
```python
from pydantic import BaseModel, Field

class PipelineConfig(BaseModel):
    """Runtime configuration for capture, tracking, and inference."""
    capture_mode: str = Field(default="threaded", description="threaded or direct")
    capture_pace_files: bool = Field(default=True, description="Pace video file reads to native FPS")
    tracker_type: str = Field(default="bytetrack", description="bytetrack or centroid")
    onnx_providers: str = Field(default="auto", description="auto or comma-separated list of providers")
    tracker_track_thresh: float = Field(default=0.4, description="High-confidence detection threshold for tracker")
    tracker_match_thresh: float = Field(default=0.7, description="IoU matching threshold for tracker")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_contracts.py -k test_capture_and_tracker_config`  
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add requirements.txt src/contracts.py tests/test_contracts.py
git commit -m "feat: add pipeline configuration contract and tracking dependencies"
```

---

### Task 2: Threaded Video Capture Ingestion

**Files:**
- Modify: `src/capture/source.py`
- Test: `tests/test_capture.py`

**Interfaces:**
- Consumes: `VideoSource` ABC from `src/capture/source.py`
- Produces: `ThreadedVideoSource` in `src/capture/source.py` implementing `start()`, `read()`, `stop()`.

- [ ] **Step 1: Write the failing tests for `ThreadedVideoSource`**

Create `tests/test_capture.py`:

```python
import time
import numpy as np
import pytest
from src.capture.source import SyntheticVideoSource, ThreadedVideoSource


def test_threaded_video_source_basic_lifecycle():
    inner = SyntheticVideoSource(width=320, height=240, fps=30)
    source = ThreadedVideoSource(inner, maxsize=1, pace_fps=0.0)

    with source:
        success, frame = source.read()
        assert success is True
        assert frame is not None
        assert frame.shape == (240, 320, 3)

    # After stop, reading returns False
    success, frame = source.read()
    assert success is False
    assert frame is None


def test_threaded_video_source_drops_stale_frames():
    # Synthetic source generates frames with incrementing frame_idx
    inner = SyntheticVideoSource(width=100, height=100, fps=100)
    source = ThreadedVideoSource(inner, maxsize=1, pace_fps=0.0)

    with source:
        # Let worker push several frames while consumer sleeps
        time.sleep(0.08)
        success1, frame1 = source.read()
        assert success1 is True
        # Check that queue only kept the latest frame rather than buffering all
        assert source._frame_queue.qsize() <= 1


def test_threaded_video_source_pacing():
    inner = SyntheticVideoSource(width=100, height=100, fps=30)
    source = ThreadedVideoSource(inner, maxsize=1, pace_fps=30.0)

    start = time.perf_counter()
    with source:
        for _ in range(3):
            source.read()
    elapsed = time.perf_counter() - start
    # 3 frames at 30 FPS should take at least ~0.05 seconds with pacing
    assert elapsed >= 0.04
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_capture.py`  
Expected: FAIL with `ImportError: cannot import name 'ThreadedVideoSource'`

- [ ] **Step 3: Implement `ThreadedVideoSource` in `src/capture/source.py`**

Add `ThreadedVideoSource` to `src/capture/source.py`:

```python
import queue
import threading
import time

class ThreadedVideoSource(VideoSource):
    """Wraps any VideoSource with a dedicated background ingestion thread.
    
    Uses a bounded queue (default size 1) to drop stale frames when consumption
    lags behind production, eliminating RTSP latency accumulation.
    """

    def __init__(
        self,
        source: VideoSource,
        maxsize: int = 1,
        pace_fps: float = 0.0,
        reconnect_timeout: float = 5.0,
    ) -> None:
        self.source = source
        self.maxsize = maxsize
        self.pace_fps = pace_fps
        self.reconnect_timeout = reconnect_timeout
        self._frame_queue: queue.Queue[tuple[bool, np.ndarray | None]] = queue.Queue(maxsize=maxsize)
        self._stop_event = threading.Event()
        self._worker_thread: threading.Thread | None = None

    def start(self) -> None:
        self.source.start()
        self._stop_event.clear()
        self._worker_thread = threading.Thread(
            target=self._reader_loop,
            name="ThreadedVideoSource-Reader",
            daemon=True,
        )
        self._worker_thread.start()

    def _reader_loop(self) -> None:
        last_frame_time = time.perf_counter()
        frame_interval = (1.0 / self.pace_fps) if self.pace_fps > 0 else 0.0

        while not self._stop_event.is_set():
            if frame_interval > 0:
                now = time.perf_counter()
                elapsed = now - last_frame_time
                if elapsed < frame_interval:
                    time.sleep(frame_interval - elapsed)
                last_frame_time = time.perf_counter()

            success, frame = self.source.read()
            if not success or frame is None:
                if self._stop_event.is_set():
                    break
                # Put EOF / failure and small backoff
                self._put_latest((False, None))
                time.sleep(0.01)
                continue

            self._put_latest((True, frame))

    def _put_latest(self, item: tuple[bool, np.ndarray | None]) -> None:
        while not self._stop_event.is_set():
            try:
                self._frame_queue.put_nowait(item)
                break
            except queue.Full:
                try:
                    self._frame_queue.get_nowait()
                except queue.Empty:
                    pass

    def read(self) -> tuple[bool, np.ndarray | None]:
        if self._stop_event.is_set() and self._frame_queue.empty():
            return False, None
        try:
            return self._frame_queue.get(timeout=0.5)
        except queue.Empty:
            return False, None

    def stop(self) -> None:
        self._stop_event.set()
        if self._worker_thread is not None and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=2.0)
        self.source.stop()
        # Drain queue
        while not self._frame_queue.empty():
            try:
                self._frame_queue.get_nowait()
            except queue.Empty:
                break
```
Export `ThreadedVideoSource` in `__all__` in `src/capture/source.py`.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_capture.py`  
Expected: PASS with 3 passed

- [ ] **Step 5: Commit**

```bash
git add src/capture/source.py tests/test_capture.py
git commit -m "feat: implement ThreadedVideoSource with size-1 queue and playback pacing"
```

---

### Task 3: ByteTrack Tracker Integration with Fallback

**Files:**
- Modify: `src/tracking/tracker.py`
- Create: `tests/test_tracker_bytetrack.py`

**Interfaces:**
- Consumes: `Detection` from `src/detection/detector.py`, `CountEventDTO` from `src/contracts.py`, `supervision` / `trackers`.
- Produces: `ChickenCounter(tracker_type="bytetrack")` with seamless fallback to `tracker_type="centroid"`.

- [ ] **Step 1: Write the failing tests for ByteTrack tracking**

Create `tests/test_tracker_bytetrack.py`:

```python
import pytest
from src.detection.detector import Detection
from src.tracking.tracker import ChickenCounter, VirtualLineZone


def test_bytetrack_initialization():
    line = VirtualLineZone(start=(0, 200), end=(400, 200), direction="down")
    counter = ChickenCounter(line_zone=line, tracker_type="bytetrack")
    assert counter.tracker_type == "bytetrack"
    assert counter.total_count == 0


def test_bytetrack_centroid_fallback():
    line = VirtualLineZone(start=(0, 200), end=(400, 200), direction="down")
    counter = ChickenCounter(line_zone=line, tracker_type="centroid")
    assert counter.tracker_type == "centroid"


def test_bytetrack_crossing_increments_count():
    line = VirtualLineZone(start=(0, 200), end=(400, 200), direction="down")
    counter = ChickenCounter(
        line_zone=line,
        tracker_type="bytetrack",
        track_thresh=0.25,
        match_thresh=0.7,
    )

    # Frame 1: Detection above line (y: 100 to 180)
    det1 = [Detection(bbox=(180, 100, 220, 180), confidence=0.9, class_name="chicken")]
    events1 = counter.update(det1, timestamp=1000.0)

    # Frame 2: Same object moving across line (y: 170 to 220)
    det2 = [Detection(bbox=(180, 170, 220, 220), confidence=0.9, class_name="chicken")]
    events2 = counter.update(det2, timestamp=1000.033)

    # Frame 3: Object continues downward (y: 210 to 260)
    det3 = [Detection(bbox=(180, 210, 220, 260), confidence=0.9, class_name="chicken")]
    events3 = counter.update(det3, timestamp=1000.066)

    total_events = events1 + events2 + events3
    assert len(total_events) == 1
    assert total_events[0].line_id == "line-1"
    assert total_events[0].class_name == "chicken"
    assert counter.total_count == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_tracker_bytetrack.py`  
Expected: FAIL with `TypeError: ChickenCounter.__init__() got an unexpected keyword argument 'tracker_type'`

- [ ] **Step 3: Implement ByteTrack adapter in `src/tracking/tracker.py`**

In `src/tracking/tracker.py`:
1. Safely import `trackers.ByteTrackTracker` and `supervision.Detections`:
```python
try:
    import numpy as np
    import supervision as sv
    try:
        from trackers import ByteTrackTracker
        HAS_BYTETRACK = True
    except ImportError:
        if hasattr(sv, "ByteTrack"):
            ByteTrackTracker = sv.ByteTrack
            HAS_BYTETRACK = True
        else:
            ByteTrackTracker = None
            HAS_BYTETRACK = False
except ImportError:
    HAS_BYTETRACK = False
    ByteTrackTracker = None
    sv = None
```
2. In `ChickenCounter.__init__`, accept:
   `tracker_type: str = "bytetrack"`, `track_thresh: float = 0.4`, `match_thresh: float = 0.7`.
   If `tracker_type == "bytetrack"` and `HAS_BYTETRACK`:
   initialize `ByteTrackTracker(lost_track_buffer=max_age, frame_rate=30.0, track_activation_threshold=track_thresh, minimum_matching_threshold=match_thresh, minimum_consecutive_frames=1)`.
   Otherwise fallback to `tracker_type = "centroid"`.
3. In `ChickenCounter.update()`:
   If `self.tracker_type == "bytetrack"` and `self._bytetrack is not None`:
   - Construct `sv.Detections` from `detections`.
   - Call `self._bytetrack.update_with_detections(sv_dets)`.
   - Iterate tracked items: extract `track_id`, update `self.tracks[track_id]`, bottom-center anchor `((x1+x2)/2, y2)`.
   - Evaluate `self.line_zone.crosses(track.last_anchor, track.current_anchor)`.
   - If crosses and `track_id not in self.counted_track_ids`: emit `CountEventDTO`, increment `self.total_count`, add `track_id` to `self.counted_track_ids`.
   - Return emitted `events`.
   Else:
   - Run existing centroid Euclidean matching logic.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_tracker_bytetrack.py`  
Expected: PASS with 3 passed

- [ ] **Step 5: Run existing tracker tests to ensure zero regressions**

Run: `uv run pytest tests/test_tracker.py`  
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/tracking/tracker.py tests/test_tracker_bytetrack.py
git commit -m "feat: integrate ByteTrack tracker with graceful centroid fallback"
```

---

### Task 4: ONNX Hardware Execution Provider Auto-Negotiation

**Files:**
- Modify: `src/detection/detector.py`
- Create: `tests/test_detector_providers.py`

**Interfaces:**
- Consumes: `ort.InferenceSession`, `ort.get_available_providers()`
- Produces: `ONNXDetector._resolve_providers()` and resilient multi-provider session creation.

- [ ] **Step 1: Write the failing tests for ONNX provider negotiation**

Create `tests/test_detector_providers.py`:

```python
from unittest.mock import patch
import pytest
from src.detection.detector import ONNXDetector


def test_resolve_providers_auto():
    with patch("onnxruntime.get_available_providers", return_value=[
        "CPUExecutionProvider",
        "OpenVINOExecutionProvider",
        "CUDAExecutionProvider",
    ]):
        providers = ONNXDetector._resolve_providers("auto")
        # CUDA takes priority over OpenVINO, followed by CPU
        assert providers == [
            "CUDAExecutionProvider",
            "OpenVINOExecutionProvider",
            "CPUExecutionProvider",
        ]


def test_resolve_providers_explicit():
    providers = ONNXDetector._resolve_providers("DirectMLExecutionProvider,CPUExecutionProvider")
    assert providers == ["DirectMLExecutionProvider", "CPUExecutionProvider"]


def test_resolve_providers_fallback_on_empty():
    providers = ONNXDetector._resolve_providers("")
    assert providers == ["CPUExecutionProvider"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_detector_providers.py`  
Expected: FAIL with `AttributeError: type object 'ONNXDetector' has no attribute '_resolve_providers'`

- [ ] **Step 3: Implement dynamic provider resolution in `src/detection/detector.py`**

In `src/detection/detector.py`:
1. Add `_resolve_providers(providers_config: str | None = None) -> list[str]` static method to `ONNXDetector`:
   - Checks `providers_config or os.getenv("ONNX_PROVIDERS", "auto")`.
   - If explicit list, parses comma-separated string, ensures `CPUExecutionProvider` is at the end.
   - If `"auto"`:
     - Queries `ort.get_available_providers()`.
     - Orders according to preference priority:
       1. `CUDAExecutionProvider`
       2. `TensorrtExecutionProvider`
       3. `OpenVINOExecutionProvider`
       4. `DirectMLExecutionProvider`
       5. `CoreMLExecutionProvider`
       6. `CPUExecutionProvider`
2. Update session initialization in `ONNXDetector.__init__`:
   - Try creating `ort.InferenceSession(self.model_path, sess_options, providers=resolved_providers)`.
   - If exception occurs, log warning and retry with `providers=['CPUExecutionProvider']`.
   - Store `self.active_providers = self.session.get_providers()`.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_detector_providers.py`  
Expected: PASS with 3 passed

- [ ] **Step 5: Run existing detector tests**

Run: `uv run pytest tests/test_detector.py`  
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/detection/detector.py tests/test_detector_providers.py
git commit -m "feat: implement dynamic ONNX execution provider negotiation and fallback"
```

---

### Task 5: Main Integration and End-to-End Pipeline

**Files:**
- Modify: `main.py`
- Create: `tests/test_e2e_pipeline.py`

**Interfaces:**
- Consumes: `ThreadedVideoSource`, `ChickenCounter(tracker_type=...)`, `ONNXDetector(providers=...)`.
- Produces: Updated `CounterService` CLI arguments `--capture-mode`, `--pace-files`, `--tracker-type`, `--onnx-providers`.

- [x] **Step 1: Write the failing end-to-end integration test**

Create `tests/test_e2e_pipeline.py`:

```python
import os
import tempfile
import pytest
from src.capture.source import SyntheticVideoSource, ThreadedVideoSource
from src.detection.detector import MockDetector, Detection
from src.tracking.tracker import ChickenCounter, VirtualLineZone
from src.storage.db import LocalStorage
from src.sync.dispatcher import OutboxSyncDispatcher


def test_e2e_pipeline_with_threaded_capture_and_bytetrack():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test_outbox.db")
        storage = LocalStorage(db_path=db_path)

        synthetic = SyntheticVideoSource(width=320, height=240, fps=30)
        source = ThreadedVideoSource(synthetic, maxsize=1, pace_fps=0.0)

        # Mock detector that simulates crossing line at y=120
        class StepCrossingDetector(MockDetector):
            def __init__(self):
                super().__init__(detections=[])
                self.step = 0

            def detect(self, frame):
                self.step += 1
                if self.step == 1:
                    return [Detection(bbox=(140, 70, 180, 110), confidence=0.9, class_name="chicken")]
                elif self.step == 2:
                    return [Detection(bbox=(140, 115, 180, 140), confidence=0.9, class_name="chicken")]
                else:
                    return [Detection(bbox=(140, 150, 180, 180), confidence=0.9, class_name="chicken")]

        detector = StepCrossingDetector()
        line = VirtualLineZone(start=(0, 120), end=(320, 120), direction="down")
        counter = ChickenCounter(line_zone=line, tracker_type="bytetrack", track_thresh=0.25)

        with source:
            for _ in range(3):
                success, frame = source.read()
                assert success is True
                dets = detector.detect(frame)
                events = counter.update(dets)
                for ev in events:
                    storage.insert_count_event(ev)

        assert counter.total_count == 1
        unsynced = storage.get_unsynced_events(limit=10)
        assert len(unsynced) == 1
        assert unsynced[0]["line_id"] == "line-1"
        storage.close()
```

- [x] **Step 2: Run test to verify it passes with components from Tasks 2-4**

Run: `uv run pytest tests/test_e2e_pipeline.py`  
Expected: PASS

- [x] **Step 3: Update `main.py` CLI parser and `CounterService`**

In `main.py`:
1. Add CLI flags to `build_cli_parser()`:
   - `--capture-mode`: default `"threaded"`, choices `["threaded", "direct"]`.
   - `--pace-files`: boolean flag (default True).
   - `--tracker-type`: default `"bytetrack"`, choices `["bytetrack", "centroid"]`.
   - `--onnx-providers`: default `"auto"`.
2. In `create_source()`:
   If `mode == "threaded"`, wrap raw source in `ThreadedVideoSource(raw_source, maxsize=1, pace_fps=(fps if pace_files else 0.0))`.
3. In `CounterService.__init__()`:
   Pass `tracker_type=args.tracker_type` to `ChickenCounter`.
   Pass `providers=args.onnx_providers` to `ONNXDetector`.

- [x] **Step 4: Verify `main.py --help` runs cleanly**

Run: `uv run python main.py --help`  
Expected: Output showing `--capture-mode`, `--pace-files`, `--tracker-type`, `--onnx-providers`.

- [x] **Step 5: Run synthetic dry run via `main.py`**

Run: `uv run python main.py --source synthetic --detector mock --tracker-type bytetrack --capture-mode threaded --duration 2`  
Expected: Service starts, processes frames with ThreadedVideoSource + ByteTrack, and exits cleanly.

- [x] **Step 6: Commit**

```bash
git add main.py tests/test_e2e_pipeline.py
git commit -m "feat: wire threaded capture, bytetrack, and provider options into main.py"
```

---

### Task 6: Full Verification, Documentation & Cleanup

**Files:**
- Modify: `README.md`
- Verify: Full test suite (`tests/`)

- [ ] **Step 1: Run complete test suite**

Run: `uv run pytest -q`  
Expected: All tests pass (215+ passed, 0 failed).

- [ ] **Step 2: Update `README.md`**

Update `README.md` documentation:
- Add section explaining `ThreadedVideoSource` (zero RTSP latency drift).
- Add section explaining `ByteTrack` tracking and centroid fallback.
- Add hardware acceleration guide (`ONNX_PROVIDERS=auto`, OpenVINO, CUDA).

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs: document threaded capture, bytetrack tracker, and hardware acceleration"
```
