# YOLO ONNX High-Accuracy Detection Pipeline Design

**Date:** 2026-10-20  
**Status:** Approved  
**Target:** Edge Poultry Counter Service (`counter-service`)

---

## 1. Overview & Objectives

In automated poultry processing facilities, counting chickens reliably requires robust detection under challenging visual conditions: hanging upside-down carcasses, metal shackles, varying lighting, water spray, and continuous conveyor motion. 

While `ShackleDetector` provides edge-based heuristic detection, deep-learning-based detection via `ONNXDetector` offers superior generalization across varying camera positions and lighting. However, the current `ONNXDetector` exhibits two key limitations:
1. **Aspect-Ratio Distortion**: Preprocessing squashes arbitrary camera resolutions (such as 1920×1080 16:9) directly into 1:1 square inputs (640×640) with `cv2.resize`, vertically distorting carcasses and degrading feature map activations.
2. **Domain Gap**: Standard off-the-shelf YOLO models are trained on COCO (class 14 `bird`), which covers live feathered birds on the ground, resulting in low confidence and missed detections on defeathered, inverted carcasses.

This design introduces:
- **Aspect-Preserving Letterbox Preprocessing & Inverse Coordinate Mapping** in `ONNXDetector`.
- **YOLO to ONNX Export Tool** (`tools/export_yolo.py`) for producing production-ready, simplified ONNX models.
- **Semi-Supervised Frame Extraction & Fine-Tuning Pipeline** (`tools/train_carcass_detector.py`) to bootstrap factory carcass datasets from existing video recordings (`samples/Rekaman_2.MOV`) and fine-tune lightweight YOLO models (YOLOv8n/11n) for edge CPU deployment.

---

## 2. Architecture & Components

```
+-----------------------------------------------------------------------------------+
| Factory Video Stream / File (e.g. 1920x1080)                                      |
+-----------------------------------------------------------------------------------+
                                    |
                                    v
+-----------------------------------------------------------------------------------+
| ONNXDetector Preprocessing                                                        |
|  1. Letterbox resize: scale r = min(640/W, 640/H), symmetric pad with (114,114,114)|
|  2. BGR -> RGB conversion                                                         |
|  3. CHW float32 normalization in [0.0, 1.0]                                       |
|  4. Store transformation metadata: (r, pad_w, pad_h, orig_w, orig_h)              |
+-----------------------------------------------------------------------------------+
                                    |
                                    v
+-----------------------------------------------------------------------------------+
| ONNX Runtime Inference (CPUExecutionProvider / OpenVINO / CUDA)                  |
|  - Graph simplified, Opset 17, optimized thread pooling                           |
|  - Evaluates models/carcass_yolov8n.onnx (or stock YOLO)                          |
+-----------------------------------------------------------------------------------+
                                    |
                                    v
+-----------------------------------------------------------------------------------+
| ONNXDetector Postprocessing                                                       |
|  1. Filter candidates by confidence threshold (conf >= 0.25)                      |
|  2. Class filtering (default class 0 for custom carcass, or COCO class 14)        |
|  3. Vectorized NumPy Non-Maximum Suppression (IoU >= 0.45)                        |
|  4. Inverse coordinate mapping:                                                   |
|       x = clip((x_letterbox - pad_w) / r, 0, orig_w)                              |
|       y = clip((y_letterbox - pad_h) / r, 0, orig_h)                              |
|  5. Emit List[Detection] to Tracking Pipeline (ChickenCounter / VirtualLineZone)  |
+-----------------------------------------------------------------------------------+
```

---

## 3. Component Specifications

### 3.1 `ONNXDetector` Pre- and Post-Processing Refactoring (`src/detection/detector.py`)

- **Letterbox Transform Function**:
  - `letterbox_image(image: np.ndarray, target_shape=(640, 640), pad_color=(114, 114, 114)) -> (tensor, ratio, (pad_w, pad_h))`
  - Preserves exact aspect ratio.
  - Symmetrically splits padding across top/bottom and left/right.
- **Inverse Coordinate Projection**:
  - Translates letterbox bounding coordinates back to native unpadded frame space $(W_{\text{orig}}, H_{\text{orig}})$.
  - Bounds boxes strictly inside $[0, W_{\text{orig}}]$ and $[0, H_{\text{orig}}]$.
- **Execution Provider Configuration**:
  - Default: `CPUExecutionProvider` with configurable thread count (`intra_op_num_threads=os.cpu_count()`).
  - Fallback: Gracefully utilizes `OpenVINOExecutionProvider` or `CUDAExecutionProvider` when available in the environment.

### 3.2 YOLO to ONNX Export Tool (`tools/export_yolo.py`)

- **Command Interface**:
  ```bash
  python tools/export_yolo.py --model yolov8n.pt --output models/yolov8n.onnx --imgsz 640 --opset 17 --simplify
  ```
- **Functionality**:
  - Verifies presence of base model weights; downloads official checkpoint if missing.
  - Exports to ONNX using `ultralytics.YOLO.export(format='onnx', imgsz=..., opset=17, simplify=True)`.
  - Performs an ONNX runtime verification run on a dummy tensor to validate shapes and outputs.

### 3.3 Semi-Supervised Auto-Labeler & Trainer (`tools/train_carcass_detector.py`)

- **Command Interface**:
  ```bash
  python tools/train_carcass_detector.py --video samples/Rekaman_2.MOV --frames 80 --epochs 25 --output models/carcass_yolov8n.onnx
  ```
- **Pipeline Stages**:
  1. **Frame Extraction**: Evenly samples $N$ frames across the video duration using OpenCV.
  2. **Bootstrap Annotation**: Evaluates each frame using `ShackleDetector` (and carcass gradient heuristics) to predict initial bounding boxes in standard YOLO format: `class x_center y_center width height` (normalized 0.0–1.0).
  3. **Dataset Packaging**: Generates YOLO directory structure (`dataset/carcass/images/{train,val}` and `dataset/carcass/labels/{train,val}`) and writes `carcass.yaml`.
  4. **Fine-Tuning**: Trains `yolov8n.pt` for specified epochs on the target dataset.
  5. **Automated Export**: Converts the trained `runs/detect/train/weights/best.pt` directly to `models/carcass_yolov8n.onnx`.

### 3.4 Runtime Integration & Dependencies

- **`requirements.txt`**:
  - Add `onnxruntime>=1.18.0` for production inference.
- **`main.py` CLI**:
  - `--model <path>`: When provided, automatically switches detector mode to `onnx` and loads the specified model.
  - `--conf <float>`: Configures confidence threshold (default: 0.25).
  - `--nms-iou <float>`: Configures NMS IoU threshold (default: 0.45).

---

## 4. Error Handling & Edge Cases

| Scenario | Handling Strategy |
|---|---|
| Non-standard aspect ratio input | Symmetrically padded to 640×640 with `(114, 114, 114)`; coordinates mapped back via exact affine scaling. |
| Missing `onnxruntime` package | `ONNXDetector` raises clear `ImportError` directing the user to install `requirements.txt`. |
| Zero detections in frame | Returns empty `list[Detection]` immediately without entering NMS. |
| Model output format variations | Supports both standard YOLO `[1, 4+classes, N]` and end-to-end NMS-free `[1, N, 6]` formats. |
| Bounding box boundary overflow | Coordinates clipped to `[0, orig_w]` and `[0, orig_h]` after padding subtraction. |

---

## 5. Verification & Testing Plan

1. **Unit Tests (`tests/test_detector.py`)**:
   - `test_letterbox_image`: Validates tensor output shape, gray padding values, and aspect preservation across 16:9, 4:3, and vertical 9:16 inputs.
   - `test_inverse_letterbox_coordinates`: Validates that padded coordinates accurately project back to true image bounds without offset errors.
   - `test_onnx_detector_postprocess_nms`: Validates multi-box suppression, confidence thresholding, and class filtering.
2. **End-to-End Pipeline Verification**:
   - Run `py -3 -m pytest` across the entire test suite ensuring 100% pass rate.
   - Benchmark throughput using `benchmark.py` to confirm latency targets (<35ms on edge CPU).
