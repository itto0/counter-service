# Technical Design Spec: Idle Mode on No Object Detection

**Date:** 2026-10-22  
**Status:** Approved  
**Topic:** Adaptive FPS Throttling & IDLE Telemetry State When No Objects Are Detected  

---

## 1. Goal & Overview

In poultry processing environments, conveyors frequently experience empty stretches or line pauses where no birds pass the inspection camera for extended periods. In the current implementation, `CounterService` processes frames and executes full ONNX/YOLO neural network inference at full camera speed continuously, consuming high CPU/GPU resources and power needlessly. Furthermore, if pacing slows down, health telemetry might incorrectly flag the stream as `DEGRADED` because FPS drops below 5.0.

This design introduces an **Adaptive Idle Throttling Mode**:
1. When no objects are detected for a configurable timeout (default: 30 seconds), the service enters an `IDLE` state.
2. In `IDLE` state, the processing loop throttles down to an idle frame rate (default: 1.0 FPS), drastically reducing CPU/GPU load while keeping the pipeline responsive.
3. Upon detecting any candidate object, the service immediately wakes up and resumes full processing speed with zero delay.
4. Health telemetry reports `stream_state = "IDLE"` while throttled, preventing false `DEGRADED` alerts.
5. On wake-up, live camera capture buffers (e.g., RTSP streams) are flushed to ensure real-time synchronization.

---

## 2. Requirements & Constraints

### Functional Requirements
- **FR-1:** Add configuration options to `CounterService`:
  - `idle_timeout_sec: float = 30.0`: Continuous time without detections required to enter idle state. A value of `0.0` disables idle throttling.
  - `idle_fps: float = 1.0`: Target frame rate during idle mode.
- **FR-2:** Add corresponding CLI arguments:
  - `--idle-timeout-sec` (float, default `30.0`)
  - `--idle-fps` (float, default `1.0`)
- **FR-3:** Immediate Wake-Up: When `len(detections) > 0`, the service must exit idle mode immediately on that same frame iteration (`is_idle = False`), restoring normal `target_fps` pacing.
- **FR-4:** Telemetry Integration: Update `HealthSampleDTO` in `src/contracts.py` to allow `"IDLE"` in `stream_state` (`Literal["ONLINE", "OFFLINE", "DEGRADED", "IDLE"]`). During idle mode, `_record_health()` sets `stream_state = "IDLE"` and bypasses the `fps < 5.0` check.
- **FR-5:** Live Buffer Flush: On transition from `is_idle == True` to `False`, if the video source has an underlying OpenCV `VideoCapture` (`cap`), discard stale decoded frames (via `grab()`) so detection aligns with real-time conveyor motion.
- **FR-6:** CLI Display: When CLI status output is active, display `[IDLE (1.0 FPS)]` status indicator during idle mode.

### Non-Functional & Architecture Constraints
- **Zero False Negatives:** Wake-up must be immediate on the first frame containing a detected object.
- **Thread Safety & Non-blocking:** Background sync dispatcher and stream uploader threads remain decoupled and unaffected.
- **Backward Compatibility:** All existing CLI arguments, environment configurations, and database schemas remain fully compatible.

---

## 3. Architecture & State Flow

```
                      ┌───────────────────────────────────────┐
                      │             ACTIVE MODE               │
                      │  - Runs at full speed / target_fps    │
                      │  - stream_state: "ONLINE"             │
                      └──────────────────┬────────────────────┘
                                         │
             No detections for >= 30.0s  │  Object detected (len(detections) > 0)
             (now - _last_detection_time)│  (immediate wake-up & buffer flush)
                                         │
                                         ▼
                      ┌───────────────────────────────────────┐
                      │              IDLE MODE                │
                      │  - Throttled to 1.0 FPS (idle_fps)    │
                      │  - stream_state: "IDLE"               │
                      │  - Low-power background monitoring    │
                      └───────────────────────────────────────┘
```

---

## 4. Detailed Component Design

### 4.1 Data Contract Update (`src/contracts.py`)
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

### 4.2 CounterService Initialization (`main.py`)
Add parameters to `CounterService.__init__`:
```python
idle_timeout_sec: float = 30.0,
idle_fps: float = 1.0,
```
Initialize internal state attributes:
```python
self.idle_timeout_sec = idle_timeout_sec
self.idle_fps = idle_fps
self.is_idle = False
self._last_detection_time = time.perf_counter()
```

### 4.3 Pipeline Main Loop (`CounterService.run`)
Inside the processing loop:
```python
# 1. Detection
t0 = time.perf_counter()
detections = self.detector.predict(frame)

# Check detection presence for idle state transitions
now = time.perf_counter()
if len(detections) > 0:
    self._last_detection_time = now
    if self.is_idle:
        self.is_idle = False
        logger.info("Object detected: exiting idle mode, resuming full processing")
        # Flush stale live stream frames if live capture source
        if hasattr(self.source, "cap") and self.source.cap is not None:
            try:
                # Grab latest frame to clear any buffered backlog
                for _ in range(2):
                    self.source.cap.grab()
            except Exception:
                pass
elif self.idle_timeout_sec > 0 and not self.is_idle:
    if (now - self._last_detection_time) >= self.idle_timeout_sec:
        self.is_idle = True
        logger.info(
            "No detections for %.1fs: entering idle mode (target: %.1f FPS)",
            self.idle_timeout_sec,
            self.idle_fps,
        )
```

### 4.4 Frame Rate Pacing
Modify the pacing delay block at the end of the frame iteration:
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
                    break
            except Exception:
                time.sleep(delay)
        else:
            time.sleep(delay)
```

### 4.5 Health Telemetry Recording (`_record_health`)
In `_record_health`:
```python
if self.is_idle:
    stream_state = "IDLE"
elif stream_state == "ONLINE" and fps < 5.0 and self.total_frames_processed > 0:
    stream_state = "DEGRADED"
```

### 4.6 CLI Display Status Line
In the live CLI status output:
```python
idle_tag = " [IDLE]" if self.is_idle else ""
sys.stdout.write(
    f"\r[Frame {self.total_frames_processed:5d}{idle_tag}] "
    f"Chickens: {self.counter.total_count:4d} | "
    f"Rate: {curr_rate:5.1f}/min | "
    f"Speed: {curr_fps:5.1f} fps | "
    f"Latency: {recent_lat:4.1f}ms   "
)
```

### 4.7 CLI Parser & Environment Variables
In `build_arg_parser()`:
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

---

## 5. Testing Plan

1. **Unit Tests for Contracts:**
   - Verify `HealthSampleDTO` accepts `"IDLE"` as `stream_state`.
2. **State Transition Tests:**
   - Simulate frames with no detections: verify `is_idle` transitions to `True` after `idle_timeout_sec`.
   - Provide a frame with detections: verify `is_idle` immediately returns to `False`.
   - Test disabling via `idle_timeout_sec = 0.0`.
3. **Pacing & Timing Tests:**
   - Measure frame intervals during idle mode to ensure effective rate matches `idle_fps`.
4. **Telemetry Health Tests:**
   - Verify `_record_health` records `stream_state == "IDLE"` when `is_idle` is `True` and does not mark it `DEGRADED`.
5. **Full Regression Pass:**
   - Run the full pytest test suite to ensure existing tracking, storage, sync, and benchmark tests continue passing.
