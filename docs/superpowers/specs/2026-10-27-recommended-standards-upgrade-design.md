# Design Specification: Recommended Standards Upgrade for Counter Service

**Date**: 2026-10-27  
**Status**: Approved  
**Target Service**: `counter-service`  

---

## 1. Executive Summary

The `counter-service` edge CV service counts chicken carcasses / shackles on factory processing lines and synchronizes counts to a central API using an offline-first SQLite WAL outbox.

This upgrade brings the service to the recommended production standard across three critical edge pillars:
1. **Threaded Video Ingestion**: Decoupled size-1 frame buffer with automatic RTSP reconnection and file playback pacing to eliminate latency drift.
2. **ByteTrack Object Tracking**: Integration of ByteTrack (via `supervision`) with Kalman filtering and two-stage association, replacing greedy Euclidean centroid matching to eliminate ID switches and double counting.
3. **Hardware Execution Provider Auto-Selection**: Dynamic provider resolution in ONNX Runtime (`CUDA` -> `TensorRT` -> `OpenVINO` -> `DirectML` -> `CPU`) with configurable overrides and resilient CPU fallback.

---

## 2. Ingestion Architecture: Threaded Video Capture

### 2.1 Problem Statement
When streaming video from an RTSP camera, network jitter and temporary inference slowdowns cause OpenCV's internal FFmpeg buffer to accumulate frames. The video pipeline drifts out of sync with real-time events.

### 2.2 Solution Design: `ThreadedVideoSource`
Introduce `ThreadedVideoSource` implementing the existing `VideoSource` abstract base class in `src/capture/source.py`.

* **Buffer Strategy**: `queue.Queue(maxsize=1)`.
  * Dedicated daemon reader thread executes `cap.read()`.
  * When a new frame arrives and the queue is full, the thread discards the stale frame (`queue.get_nowait()`) and enqueues the fresh frame (`queue.put_nowait()`).
  * The main inference loop calling `.read()` always receives the most recent real-time frame with zero backlog.
* **Resilient RTSP Reconnect**:
  * Tracks consecutive failed reads and stall durations (default 5.0 seconds).
  * Automatically tears down `cv2.VideoCapture` and retries connection using exponential backoff (1s, 2s, 4s, up to 30s) until the RTSP endpoint recovers.
* **File Playback Pacing**:
  * For file-based sources (MP4/MKV), a precision timer paces frame reads according to the container's native FPS ($1.0 / \text{fps}$ delay), preventing 500+ FPS CPU spikes in evaluation/test runs.
* **Lifecycle & Clean Shutdown**:
  * Controlled via `threading.Event` (`_stop_event`). `stop()` signals the worker thread and waits with a 2-second timeout before releasing hardware capture resources.

---

## 3. Tracking Architecture: ByteTrack Integration

### 3.1 Problem Statement
The current `ChickenCounter` matches tracks using Euclidean distance between detection bottom-center anchors. Swaying carcasses and overlapping shackles frequently exceed the distance threshold or create ambiguity, causing ID swaps and duplicate count events.

### 3.2 Solution Design: ByteTrack Adapter with Fallback
Integrate ByteTrack from `supervision` while maintaining 100% backward compatibility with `ChickenCounter`, `VirtualLineZone`, and `CountEventDTO`.

* **Adapter Layer**:
  * Converts internal `Detection` objects (dataclasses with `bbox: (x1, y1, x2, y2)`, `confidence`, `class_name`) to `sv.Detections`.
* **Two-Stage Association**:
  * **Stage 1 (High Confidence)**: Associates high-confidence detections ($conf \ge \text{track\_thresh}$) with Kalman filter predicted motion trajectories using bounding box IoU.
  * **Stage 2 (Low Confidence)**: Associates remaining active tracks with lower confidence detections ($0.1 \le conf < \text{track\_thresh}$) to maintain continuity through temporary occlusions or motion blur.
* **Anchor Extraction & Line Crossing**:
  * Computes bottom-center anchor: $(x_1 + x_2) / 2.0, y_2$.
  * Updates `(last_anchor, current_anchor)` for each active track ID.
  * Evaluates intersection against `VirtualLineZone` using vector cross-product orientation (`ccw`).
  * Enforces `counted_track_ids: set[int]` to ensure each physical track is counted at most once per line crossing.
* **Fallback Guarantee**:
  * If `supervision` is not installed or `tracker_type == "centroid"`, `ChickenCounter` transparently uses the Euclidean centroid matching implementation.

---

## 4. Inference Architecture: ONNX Hardware Providers

### 4.1 Problem Statement
The current `ONNXDetector` initializes `ort.InferenceSession` with default provider arguments, missing hardware accelerators present on edge hosts (Intel iGPU/NPU, NVIDIA Jetson/GPU, DirectX).

### 4.2 Solution Design: Dynamic Provider Negotiation
Enhance `ONNXDetector` in `src/detection/detector.py`:

* **Provider Cascade**:
  1. `CUDAExecutionProvider` (NVIDIA GPUs / Jetson)
  2. `TensorrtExecutionProvider` (NVIDIA TensorRT)
  3. `OpenVINOExecutionProvider` (Intel CPUs / iGPUs / NPUs)
  4. `DirectMLExecutionProvider` (Windows DirectX / AMD)
  5. `CPUExecutionProvider` (Universal baseline)
* **Configuration & Override**:
  * `ONNX_PROVIDERS`: If set to `auto` (default), filters available providers via `ort.get_available_providers()` and selects the highest-priority supported engine.
  * If set to an explicit comma-separated string (e.g. `OpenVINOExecutionProvider,CPUExecutionProvider`), uses that exact preference order.
* **Fault-Tolerant Initialization**:
  * Wraps session creation in a try-except block. If initializing an accelerated provider fails (e.g. missing CUDA runtime libraries or incompatible OpenVINO DLLs), logs a warning and falls back immediately to `CPUExecutionProvider` without failing startup.
* **Graph Optimizations**:
  * Sets `ort.GraphOptimizationLevel.ORT_ENABLE_ALL`.

---

## 5. Configuration & Contracts

### 5.1 Environment Variables

| Variable | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `CAPTURE_MODE` | `str` | `threaded` | `threaded` (size-1 drop buffer) or `direct` (synchronous). |
| `CAPTURE_PACE_FILES`| `bool` | `true` | Pace file playback according to container FPS. |
| `TRACKER_TYPE` | `str` | `bytetrack` | `bytetrack` (Kalman + IoU) or `centroid` (Euclidean). |
| `TRACKER_TRACK_THRESH` | `float`| `0.4` | High-confidence threshold for Stage 1 ByteTrack. |
| `TRACKER_MATCH_THRESH` | `float`| `0.7` | IoU association threshold for ByteTrack. |
| `ONNX_PROVIDERS` | `str` | `auto` | `auto` or comma-separated list of execution providers. |

### 5.2 Dependencies (`requirements.txt`)
```text
opencv-python-headless>=4.8.0
numpy>=1.24.0
pydantic>=2.5.0
httpx>=0.27.0
python-dotenv>=1.0.0
onnxruntime>=1.18.0
supervision>=0.20.0
scipy>=1.11.0
```

---

## 6. Verification & Test Plan

1. **Capture Pipeline Tests (`tests/test_capture.py`)**:
   * Test `ThreadedVideoSource` start, frame acquisition, and clean stop.
   * Verify size-1 queue drops stale frames under slow consumption.
   * Verify file pacing maintains target FPS timings.
   * Verify RTSP reconnect handling logic under simulated failures.
2. **ByteTrack Tracker Tests (`tests/test_tracker_bytetrack.py`)**:
   * Verify detection conversion to `sv.Detections`.
   * Verify synthetic crossing sequence correctly increments count and emits `CountEventDTO`.
   * Verify duplicate crossing prevention (`counted_track_ids`).
   * Verify centroid fallback when `tracker_type="centroid"`.
3. **ONNX Provider Tests (`tests/test_detector_providers.py`)**:
   * Verify provider resolution logic with `auto` and explicit lists.
   * Verify fallback to `CPUExecutionProvider` when primary provider fails.
4. **End-to-End Pipeline Test (`tests/test_e2e_pipeline.py`)**:
   * Execute synthetic video source -> detector -> ByteTrack counter -> SQLite outbox -> dispatcher flush.
   * Ensure 100% test pass across pytest suite with zero regressions.
