# Edge Poultry Counter Service

A standalone, lightweight edge computer vision service designed for industrial poultry processing lines. It processes high-speed conveyor video feeds, detects and tracks shackles/chickens across virtual counting lines, records real-time counting and health metrics into an offline-first SQLite outbox, and syncs batches to the central API in the background.

---

## Key Features

- **Decoupled Architecture**: Fully self-contained service that can be built and deployed without the monorepo root or central API code.
- **Offline-First Store & Forward**: Uses a local SQLite database (`counter_outbox.db`) to queue count aggregates and health telemetry. If network connectivity drops, edge processing continues uninterrupted and backlogs flush automatically upon reconnection.
- **Zero Inbound Attack Surface**: Operates purely as an outbound HTTP client — exposes no listening network ports, making it secure for deployment in factory OT networks.
- **Zero-Drift Threaded Capture**: Decoupled ingestion via `ThreadedVideoSource` using a drop-on-full size-1 frame queue to eliminate RTSP buffer accumulation and latency drift.
- **ByteTrack Multi-Object Tracking**: Industrial-grade tracking using Kalman filter motion prediction and two-stage IoU association, with seamless automatic fallback to `CentroidTracker`.
- **Hardware Acceleration Auto-Negotiation**: Automatic execution provider discovery and priority cascade (`CUDA` -> `TensorRT` -> `OpenVINO` -> `DirectML` -> `CoreML` -> `CPU`) for ONNX Runtime YOLO models with resilient CPU fallback.
- **Flexible Detection Engines**:
  - `auto`: Automatic detector selection.
  - `shackle`: Industrial poultry conveyor detector with carcass occupancy gradient analysis to reject empty shackles.
  - `onnx`: Hardware-accelerated YOLO inference via ONNX Runtime.
  - `mock`: Light intensity contour segmentation for benchmarking and CPU-only testing.
- **Remote Configuration**: Automatically fetches line calibration parameters (virtual line coordinates, direction, RTSP stream) from the central API on startup using Bearer API Key authentication (`API_TOKEN`), with offline fallback to local defaults.
- **Real-Time Visual Preview & Headless Modes**: Optional OpenCV display window for on-site line tuning, or headless execution for production Docker containers.

---

## Directory Structure

```text
counter-service/
├── Dockerfile             # Standalone Docker build definition
├── docker-compose.yml     # Standalone Docker Compose deployment
├── README.md              # Documentation (this file)
├── requirements.txt       # Edge Python runtime dependencies
├── main.py                # Service entry point and CLI runner
├── benchmark.py           # Performance and latency benchmark suite
├── src/
│   ├── capture/           # Video capture (synthetic, file, RTSP, ThreadedVideoSource)
│   ├── detection/         # Shackle, ONNX, and mock detectors with hardware cascade
│   ├── tracking/          # ByteTrack & centroid trackers, virtual line crossing zone
│   ├── storage/           # Local SQLite outbox database manager
│   ├── sync/              # Background HTTP batch dispatcher
│   ├── contracts.py       # Data transfer schemas (Pydantic models)
│   └── env.py             # Environment configuration loader
└── tests/                 # Unit and integration test suite
```

---

## Requirements

- Python 3.11+
- System libraries for OpenCV (when running on host Linux: `libgl1`, `libglib2.0-0`)

---

## Local Installation

```bash
pip install -r requirements.txt
```

Dependencies defined in `requirements.txt`:
```text
opencv-python-headless>=4.8.0
numpy>=1.24.0
pydantic>=2.5.0
httpx>=0.27.0
python-dotenv>=1.0.0
onnxruntime>=1.18.0
supervision>=0.20.0
trackers>=2.6.0
scipy>=1.11.0
av>=12.0.0
```

> **Note on Visual Preview Window (`--show-window`)**:
> `requirements.txt` specifies `opencv-python-headless` for lightweight container and server execution without display dependencies. If you intend to use the live OpenCV GUI preview window (`--show-window`), install standard `opencv-python`:
> ```bash
> pip install opencv-python
> ```

---

## Video Ingestion & Latency Management (ThreadedVideoSource)

Industrial edge deployments processing live RTSP camera feeds face a fundamental latency challenge: standard OpenCV `cv2.VideoCapture` sequentially blocks on frame retrieval and buffers packets internally. If downstream YOLO inference, tracking, or network I/O takes even 40–80 ms per frame, the ingest queue falls behind real-time. Over minutes or hours, the processed frames become seconds or minutes behind real-time—accumulating severe latency drift.

`ThreadedVideoSource` decouples capture ingestion from downstream inference:
- **Dedicated Background Ingestion**: A daemon worker thread continuously pulls frames from the camera or video source at full speed.
- **Drop-on-Full Size-1 Queue**: Received frames are placed into a bounded `queue.Queue(maxsize=1)`. When a new frame arrives and the queue already contains an unprocessed frame (because inference is busy), the stale frame is dropped instantly via `get_nowait()`, and the latest frame is inserted (`_put_latest`).
- **Zero RTSP Latency Accumulation**: Guarantees that the counter pipeline always processes the freshest frame, maintaining sub-second latency from optical conveyor passage to outbox event generation.
- **Configurable Modes**:
  - `threaded` (default): Uses the background thread and drop-on-full size-1 queue for live RTSP streams and simulated real-time playback.
  - `direct`: Synchronous, blocking sequential reads without dropping frames; ideal for offline batch benchmarking or processing pre-recorded test datasets where every frame must be evaluated.
- **Playback Pacing (`--pace-files` / `PACE_FILES=1`)**: When running against recorded video files in `threaded` mode, playback is paced to match the file's native FPS (e.g. 30 FPS). This prevents file playback from flooding CPU/GPU at 200+ FPS and realistically replicates real-time RTSP production conditions.

```bash
# Production live RTSP feed with threaded zero-drift capture (default)
python main.py --source rtsp -v "rtsp://camera.local:554/stream1" --capture-mode threaded

# Offline batch file processing: analyze every frame as fast as possible
python main.py -v /path/to/batch.mp4 --capture-mode direct --no-pace-files
```

---

## Multi-Object Tracking (ByteTrack & Centroid Fallback)

Conveyor lines in poultry processing facilities present severe tracking challenges: rapid carcass movement, dense shackle spacing, temporary occlusion, and fluctuating lighting conditions.

### Modern ByteTrack Tracking
The service integrates **ByteTrack** as its primary multi-object tracking algorithm:
- **Two-Stage Association**: Instead of discarding low-confidence bounding box detections (e.g., partially occluded shackles or distant silhouettes), ByteTrack associates high-confidence detections first, then uses low-confidence detections in a second matching stage to preserve trajectories across momentary occlusions.
- **Kalman Filter Motion Prediction**: Predicts bounding box locations frame-over-frame using velocity and position state estimation, handling non-linear jerky motion and conveyor vibration.
- **Supervision & Trackers Integration**: Employs an internal adapter (`_ByteTrackAdapter`) supporting both `trackers` (v2.6.0+) and `supervision` libraries.

### Exactly-Once Counting Guarantees
- **VirtualLineZone Vector Geometry**: Conveyor counting uses continuous crossing vectors evaluated against the virtual counting line. The intersection test evaluates orientation cross-products (`(p1, p2)` movement across `(start, end)` line) and directional alignment (`down`, `up`, `left`, `right`).
- **Deterministic Deduplication**: Each tracked entity maintains a persistent integer `tracker_id`. Once a trajectory crosses the virtual zone, its ID is committed to the counter's `counted_track_ids` set. Even if a carcass lingers or oscillates near the line boundary, it is counted **exactly once**.

### Graceful Centroid Fallback
If ByteTrack dependencies (`trackers` or `supervision`) are missing or cannot be imported in the edge environment:
- The service logs an informational notice and automatically falls back to `CentroidTracker`.
- Centroid tracking computes Euclidean center coordinates and matches nearest points up to `--max-distance` (default 60 pixels), providing zero-dependency tracking on minimal edge appliances.

```bash
# Run with modern ByteTrack (default)
python main.py -v conveyor.mp4 --tracker-type bytetrack

# Explicitly force lightweight Centroid tracking
python main.py -v conveyor.mp4 --tracker-type centroid
```

---

## Hardware Acceleration & ONNX Execution Providers

The ONNX YOLO detection engine (`src.detection.detector.ONNXDetector`) features dynamic hardware provider auto-negotiation and resilient fallback.

### Auto-Priority Cascade
When `--onnx-providers auto` (or `ONNX_PROVIDERS=auto`) is active, the detector queries ONNX Runtime for all available hardware acceleration providers on the host system via `ort.get_available_providers()` and selects the highest-priority accelerator available:

```text
Priority Cascade:
  1. CUDAExecutionProvider        (Nvidia GPUs)
  2. TensorrtExecutionProvider    (Nvidia TensorRT optimized engine)
  3. OpenVINOExecutionProvider    (Intel CPUs, iGPUs, VPUs, Arc discrete GPUs)
  4. DirectMLExecutionProvider    (Windows DirectX 12 / AMD / Intel / Qualcomm)
  5. CoreMLExecutionProvider      (Apple Silicon M-series & macOS)
  6. CPUExecutionProvider         (Universal fallback, always active)
```

### Resilient Runtime Fallback
Edge environments often encounter subtle driver or library mismatches (for example, missing CUDA runtime libraries, outdated Intel OpenVINO DLLs, or unsupported GPU architectures).
- If session initialization fails on the selected hardware provider, the error is logged as a warning.
- The detector **automatically catches the initialization failure** and immediately retries session creation with `['CPUExecutionProvider']`.
- This ensures the edge counter service never crashes on startup due to GPU/NPU driver faults.

### Specifying Explicit Providers
You can configure providers via the CLI or environment variables:
- **`--onnx-providers auto`**: (Default) Discovers and chooses the best available hardware accelerator.
- **Explicit Comma-Separated List**: Pass specific providers in order of preference, e.g.:
  - `--onnx-providers OpenVINOExecutionProvider,CPUExecutionProvider`
  - `--onnx-providers CUDAExecutionProvider,CPUExecutionProvider`
  - `CPUExecutionProvider` is automatically appended to explicit lists as a safety guarantee.

```bash
# Auto-detect best hardware accelerator (CUDA, OpenVINO, DirectML, etc.)
python main.py -m models/yolo26m.onnx --detector onnx --onnx-providers auto

# Intel edge appliance: force OpenVINO with CPU fallback
python main.py -m models/yolo26m.onnx --detector onnx --onnx-providers OpenVINOExecutionProvider,CPUExecutionProvider
```

---

## CLI Usage

The service entrypoint is `main.py`. You can configure execution using command-line arguments or environment variables.

### Quick Start Examples

1. **Synthetic Video Test (Zero configuration):**
   ```bash
   python main.py --source synthetic --show-window
   ```

2. **Live RTSP Camera Stream with ByteTrack and Hardware Acceleration:**
   ```bash
   python main.py \
     --source rtsp \
     --video "rtsp://camera.local:554/live" \
     --capture-mode threaded \
     --tracker-type bytetrack \
     --model models/yolo26m.onnx \
     --detector onnx \
     --onnx-providers auto \
     --line-id line-1
   ```

3. **Run on Video File with Central API Sync & Authentication:**
   ```bash
   python main.py \
     --video /path/to/conveyor_sample.mp4 \
     --line-id line-1 \
     --direction right \
     --tracker-type bytetrack \
     --api-sync-url http://localhost:8000/api/v1/sync/batch \
     --api-token cdc_live_xxxxxxxxxxxxxxxxxxxxxxxx \
     --loop
   ```

4. **Standalone Mode (No network sync, local storage only):**
   ```bash
   python main.py \
     --video /path/to/conveyor_sample.mp4 \
     --line-id line-1 \
     --no-sync \
     --show-window
   ```

5. **Timed Test / Benchmark Run (30 seconds duration):**
   ```bash
   python main.py \
     --video /path/to/conveyor_sample.mp4 \
     --duration 30.0 \
     --no-sync
   ```

### Command Line Options

| Option | Flag | Default | Description |
|---|---|---|---|
| `--video` | `-v` | `""` | Path to video file (`.mp4`, `.mov`, `.avi`, `.mkv`). Sets source type to `file`. |
| `--source` | | `synthetic` | Video source type: `synthetic`, `file`, or `rtsp`. |
| `--capture-mode` | | `threaded` | Video capture ingestion mode: `threaded` (background thread with bounded size-1 queue for zero RTSP latency accumulation) or `direct` (blocking synchronous read). |
| `--pace-files` / `--no-pace-files` | | `True` | Pace video file playback to native FPS in threaded mode to emulate real-time stream ingestion. |
| `--tracker-type` | | `bytetrack` | Multi-object tracking algorithm: `bytetrack` (Kalman filter + dual-stage IoU association) or `centroid` (Euclidean distance matching). |
| `--onnx-providers` | | `auto` | Execution providers for ONNX Runtime inference: `auto` (priority cascade: `CUDA -> TensorRT -> OpenVINO -> DirectML -> CoreML -> CPU`) or comma-separated provider names. |
| `--duration` | `-d` | `None` | Maximum service runtime duration in seconds before stopping cleanly (e.g. `30.0`). |
| `--line-id` | | `lineA` | Conveyor line identifier (e.g. `line-1`, `lineA`). |
| `--model` | `-m` | `""` | Path to ONNX YOLO model file (optional). |
| `--detector` | | `auto` | Detector engine: `auto`, `shackle`, `mock`, or `onnx`. |
| `--conf` | | `0.7` | Detection confidence threshold (0.0 to 1.0). |
| `--occupancy-threshold` | | `4.5` | Carcass occupancy gradient threshold for `ShackleDetector` empty shackle rejection. |
| `--no-occupancy-check` | | `False` | Disable carcass occupancy verification (counts all shackles). |
| `--line` | | `""` | Virtual crossing line as `x1,y1,x2,y2` (e.g. `0,240,640,240`). |
| `--direction` | | `down` | Line crossing direction: `down`, `up`, `left`, `right`, `in`, `out`. |
| `--no-sync` | | `False` | Disable HTTP sync to central API (runs completely offline). |
| `--api-sync-url` | | `http://localhost:8000/api/v1/sync/batch` | Central API endpoint for batch outbox sync. |
| `--api-token` | `--api-key` | `""` | Bearer API key/token for authenticating with central cloud API. Eliminates HTTP 401 warnings and enables remote line calibration sync. |
| `--db-path` | | `counter_outbox.db` | Local SQLite database file path. |
| `--loop` | | `False` | Loop video continuously instead of exiting at EOF. |
| `--max-frames` | | `None` | Stop processing after reaching frame count limit. |
| `--target-fps` | | `0.0` | Target playback FPS pacing (e.g. `30.0`). Set `0.0` for maximum speed. |
| `--show-window` | `--gui` | `False` | Display OpenCV GUI window with tracking overlays and line counters. |
| `--sync-interval` | | `5.0` | Seconds between background batch sync attempts. |
| `--aggregate-window` | | `60.0` | Rolling aggregate time window in seconds. |
| `--health-interval` | | `10.0` | Telemetry health recording interval in seconds. |
| `--verbose` | | `False` | Enable verbose DEBUG logging. |

---

## Environment Variables

All command-line arguments can also be set via environment variables or a `.env` file:

| Variable | Default | Description |
|---|---|---|
| `LINE_ID` | `line-1` | Line identifier matching registered lines in central API |
| `VIDEO_PATH` | `/samples/Rekaman_2.MOV` | Video file path inside container or host |
| `VIDEO_SOURCE_TYPE` | `synthetic` | `synthetic`, `file`, or `rtsp` |
| `CAPTURE_MODE` | `threaded` | Video capture mode: `threaded` (background thread with drop-on-full size-1 queue) or `direct` (blocking synchronous) |
| `PACE_FILES` | `1` | `1` to pace file video playback to native FPS in threaded mode, `0` for unthrottled processing speed |
| `TRACKER_TYPE` | `bytetrack` | Multi-object tracking algorithm: `bytetrack` or `centroid` |
| `ONNX_PROVIDERS` | `auto` | ONNX execution providers: `auto` (hardware priority cascade) or comma-separated list (e.g. `OpenVINOExecutionProvider,CPUExecutionProvider`) |
| `DURATION_SEC` | `0.0` | Maximum service runtime in seconds before stopping (`0.0` or unset runs indefinitely) |
| `SAMPLES_DIR` | `./samples` | Host directory containing video files for volume mount |
| `LOOP_VIDEO` | `1` | `1` to loop video endlessly, `0` to stop at EOF |
| `LINE_DIRECTION` | `right` | Crossing trigger direction (`right`, `left`, `down`, `up`) |
| `LINE_COORDS` | `""` | Manual line coordinates `x1,y1,x2,y2` |
| `TARGET_FPS` | `30.0` | Frame rate throttle (`0.0` for unthrottled maximum speed) |
| `API_SYNC_URL` | `http://localhost:8000/api/v1/sync/batch` | Destination URL for outbox batch sync |
| `API_TOKEN` | `""` | Bearer API key/token for central cloud API authentication (alias: `API_KEY`). Resolves HTTP 401 warnings and enables remote line sync |
| `DB_PATH` | `/data/counter_outbox.db` | Path to persistent SQLite outbox file |
| `SYNC_INTERVAL_SEC` | `5.0` | Interval between background sync pushes |
| `HEALTH_INTERVAL_SEC` | `10.0` | Interval between system health telemetry samples |
| `CONF_THRESHOLD` | `0.7` | Shackle/chicken detection confidence threshold |
| `OCCUPANCY_THRESHOLD` | `4.5` | Carcass occupancy threshold for empty shackle rejection |
| `NO_SYNC` | `0` | Set `1` to disable central sync completely |
| `SHOW_WINDOW` | `0` | Set `1` to show visual preview window |
| `VERBOSE` | `0` | Set `1` to enable debug log output |

---

## Authentication & API Keys

The edge counter interacts with the central API for two distinct operations:
1. **Startup Remote Line Sync (`GET /api/v1/lines/{line_id}`)**: Pulls authoritative line settings (video RTSP stream URL, line crossing coordinates, line direction) configured in the central cloud dashboard.
2. **Background Batch Sync (`POST /api/v1/sync/batch`)**: Transmits buffered counting events, aggregates, and system health telemetry.

### Resolving HTTP 401 Unauthorized Warning
The line details endpoint requires authenticated access. If `API_TOKEN` (or `API_KEY`) is missing or invalid, the service logs:
```text
[WARNING] counter.service: Remote line config returned HTTP 401 for line line-1. Check that API_TOKEN or API_KEY is set in your environment. Falling back to local configuration.
```
When this occurs, the counter falls back to local command-line or `.env` defaults, and will not adopt settings managed from the cloud dashboard. Setting `API_TOKEN` eliminates this warning and allows the edge counter to automatically adopt line settings (RTSP stream, coordinates, direction) from the central cloud dashboard.

### How to Generate an API Key

#### Option 1: Via Central API CLI
From the backend server:
```bash
python -m services.api.src.cli create-api-key --name "Line 1 Edge"
```
Or specify an explicit role (`OPERATOR` or `ADMIN`, defaults to `OPERATOR`):
```bash
python -m services.api.src.cli create-api-key --name "Line 1 Edge" --role OPERATOR
```
Output:
```text
API Key created successfully for 'Line 1 Edge' [OPERATOR]:
  cdc_live_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
Store this key securely in the edge device's .env file as API_TOKEN.
```

#### Option 2: Via REST API (`POST /api/v1/auth/api-keys`)
Admin users can generate keys programmatically:
```bash
curl -X POST https://<cdc-api-subdomain>.onrender.com/api/v1/auth/api-keys \
  -H "Authorization: Bearer <ADMIN_JWT_OR_KEY>" \
  -H "Content-Type: application/json" \
  -d '{"name": "Line 1 Edge", "role": "OPERATOR"}'
```
Response:
```json
{
  "id": 1,
  "name": "Line 1 Edge",
  "api_key": "cdc_live_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
  "prefix": "cdc_live_xxxx",
  "role": "OPERATOR",
  "created_at": "2026-10-19T10:00:00Z"
}
```

Assign the returned `api_key` to `API_TOKEN` in `.env.counter` or your container's environment variables.

---

## Standalone Docker Build

The counter service is completely self-contained. You can build it directly with Docker using only the directory as the build context.

### 1. Build Docker Image

From inside:
```bash
docker build -t cdc-counter .
```

Or from repository root:
```bash
docker build -t cdc-counter .
```

### 2. Run Container Standalone

```bash
docker run --rm \
  -v $(pwd)/data:/data \
  -v /path/to/samples:/samples:ro \
  -e LINE_ID=line-1 \
  -e VIDEO_PATH=/samples/Rekaman_2.MOV \
  -e LOOP_VIDEO=1 \
  -e LINE_DIRECTION=right \
  -e API_SYNC_URL=http://your-central-api:8000/api/v1/sync/batch \
  -e API_TOKEN=cdc_live_xxxxxxxxxxxxxxxxxxxxxxxx \
  cdc-counter
```

---

## Docker Compose Execution

For edge server deployments, a dedicated Docker Compose configuration `docker-compose.yml` is provided at the repository root.

### 1. Setup Environment File

Copy the template configuration:
```bash
cp .env.counter.example .env.counter
```

Edit `.env.counter` with your central API endpoint, authentication key, and conveyor parameters:
```dotenv
API_SYNC_URL=https://cdc-api.onrender.com/api/v1/sync/batch
API_TOKEN=cdc_live_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
LINE_ID=line-1
VIDEO_PATH=/samples/Rekaman_2.MOV
SAMPLES_DIR=./samples
LOOP_VIDEO=1
LINE_DIRECTION=right
TARGET_FPS=30.0
SYNC_INTERVAL_SEC=5.0
HEALTH_INTERVAL_SEC=10.0
CONF_THRESHOLD=0.7
OCCUPANCY_THRESHOLD=4.5
```

### 2. Start Service

```bash
docker compose -f docker-compose.yml --env-file .env.counter up -d --build
```

### 3. Check Logs & Status

```bash
docker compose -f docker-compose.yml logs -f counter
```

### 4. Stop Service

```bash
docker compose -f docker-compose.yml down
```

---

## Testing

Run unit and integration tests for the counter service:

```bash
# Run compose verification test
py -m pytest tests/test_compose_counter.py -v

# Run all counter service tests
py -m pytest tests/ -v
```
