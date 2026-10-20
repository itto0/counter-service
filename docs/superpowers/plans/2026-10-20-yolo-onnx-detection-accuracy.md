# YOLO ONNX Detection Accuracy Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Upgrade `ONNXDetector` with aspect-ratio-preserving letterbox pre/post-processing and deliver production-ready YOLO ONNX export and semi-supervised carcass fine-tuning tools to maximize poultry detection accuracy on edge CPUs.

**Architecture:** Refactor `ONNXDetector` in `src/detection/detector.py` to use an aspect-preserving letterbox transform with exact inverse coordinate projection. Provide `tools/export_yolo.py` for exporting YOLO models to simplified ONNX and `tools/train_carcass_detector.py` for bootstrapping factory carcass annotations using `ShackleDetector` and fine-tuning lightweight YOLO models.

**Tech Stack:** Python 3.10+, OpenCV, NumPy, ONNX Runtime (`onnxruntime`), Ultralytics YOLO (`ultralytics`), Pytest.

## Global Constraints

- Preserve backwards compatibility with existing detector interfaces (`Detection` frozen dataclass with `(x1, y1, x2, y2, confidence, class_id, class_name)` and `anchor_bottom_center`).
- All coordinates returned by `ONNXDetector.detect()` must be in original unpadded frame pixel space `[0, orig_w]` and `[0, orig_h]`.
- Runtime inference must only require `onnxruntime` and OpenCV (no PyTorch or Ultralytics required for runtime deployment).
- All existing 137 unit and integration tests must continue to pass.

---

### Task 1: Add Letterbox Preprocessing & Inverse Coordinate Projection Helper Functions

**Files:**
- Modify: `src/detection/detector.py:440-470`
- Test: `tests/test_detector.py`

**Interfaces:**
- Produces:
  - `letterbox_image(image: np.ndarray, target_shape: tuple[int, int] = (640, 640), pad_color: tuple[int, int, int] = (114, 114, 114)) -> tuple[np.ndarray, float, tuple[int, int]]`
  - Returns `(padded_image, scale_ratio, (pad_left, pad_top))`

- [ ] **Step 1: Write the failing tests for `letterbox_image`**

Add to `tests/test_detector.py`:

```python
def test_letterbox_image_aspect_ratio_and_padding():
    """Verify letterbox resizes with constant aspect ratio and symmetric gray padding."""
    from src.detection.detector import letterbox_image
    # 1920x1080 frame (16:9)
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    target_shape = (640, 640)
    padded, ratio, (pad_w, pad_h) = letterbox_image(frame, target_shape=target_shape)

    assert padded.shape == (640, 640, 3)
    # Ratio is 640 / 1920 = 0.333333...
    assert pytest.approx(ratio, rel=1e-3) == 640 / 1920
    # Width should have no padding: 1920 * ratio = 640
    assert pad_w == 0
    # Height should have padding: (640 - 1080 * ratio) / 2 = (640 - 360) / 2 = 140
    assert pad_h == 140
    # Top padding area should be constant 114
    assert np.all(padded[0:140, :, :] == 114)
    assert np.all(padded[500:640, :, :] == 114)


def test_letterbox_image_vertical_frame():
    """Verify letterbox handles vertical 9:16 aspect ratio correctly."""
    from src.detection.detector import letterbox_image
    frame = np.zeros((1920, 1080, 3), dtype=np.uint8)
    target_shape = (640, 640)
    padded, ratio, (pad_w, pad_h) = letterbox_image(frame, target_shape=target_shape)

    assert padded.shape == (640, 640, 3)
    assert pytest.approx(ratio, rel=1e-3) == 640 / 1920
    assert pad_h == 0
    assert pad_w == 140
    assert np.all(padded[:, 0:140, :] == 114)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `py -3 -m pytest tests/test_detector.py::test_letterbox_image_aspect_ratio_and_padding -v`
Expected: FAIL with `ImportError: cannot import name 'letterbox_image' from 'src.detection.detector'`

- [ ] **Step 3: Implement `letterbox_image` in `src/detection/detector.py`**

Add function `letterbox_image` before `ONNXDetector`:

```python
def letterbox_image(
    image: np.ndarray,
    target_shape: tuple[int, int] = (640, 640),
    pad_color: tuple[int, int, int] = (114, 114, 114),
) -> tuple[np.ndarray, float, tuple[int, int]]:
    """Resize image preserving aspect ratio with symmetric border padding.

    Args:
        image: Input BGR image (H, W, C).
        target_shape: (target_width, target_height).
        pad_color: BGR tuple for padding fill.

    Returns:
        (padded_image, scale_ratio, (pad_w, pad_h))
    """
    orig_h, orig_w = image.shape[:2]
    target_w, target_h = target_shape

    ratio = min(target_w / orig_w, target_h / orig_h)
    new_w = int(round(orig_w * ratio))
    new_h = int(round(orig_h * ratio))

    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    pad_w = (target_w - new_w) / 2
    pad_h = (target_h - new_h) / 2

    top = int(round(pad_h - 0.1))
    bottom = int(round(pad_h + 0.1))
    left = int(round(pad_w - 0.1))
    right = int(round(pad_w + 0.1))

    padded = cv2.copyMakeBorder(
        resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=pad_color
    )
    return padded, ratio, (left, top)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `py -3 -m pytest tests/test_detector.py::test_letterbox_image_aspect_ratio_and_padding tests/test_detector.py::test_letterbox_image_vertical_frame -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/detection/detector.py tests/test_detector.py
git commit -m "feat(detection): add aspect-preserving letterbox_image function"
```

---

### Task 2: Integrate Letterbox and Inverse Mapping into `ONNXDetector`

**Files:**
- Modify: `src/detection/detector.py:465-590`
- Test: `tests/test_detector.py`

**Interfaces:**
- Consumes: `letterbox_image`
- Produces: `ONNXDetector.preprocess(frame) -> np.ndarray`, `ONNXDetector.postprocess(outputs) -> list[Detection]`

- [ ] **Step 1: Write failing tests for `ONNXDetector` letterbox coordinate inverse mapping**

Add to `tests/test_detector.py`:

```python
def test_onnx_detector_letterbox_inverse_mapping():
    """Verify ONNXDetector maps letterboxed coordinates back to original frame accurately."""
    from unittest.mock import MagicMock
    from src.detection.detector import ONNXDetector

    # Create dummy detector with mock session
    detector = ONNXDetector(model_path="dummy.onnx", input_size=(640, 640), conf_threshold=0.5)
    detector._session = MagicMock()
    detector._input_name = "images"
    detector._output_name = "output0"

    # Original 1920x1080 frame
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    _ = detector.preprocess(frame)

    assert detector._orig_w == 1920
    assert detector._orig_h == 1080
    assert detector._pad_w == 0
    assert detector._pad_h == 140

    # Simulate a raw YOLO detection on the 640x640 letterboxed frame
    # Carcass located at center x=320, y=320, w=100, h=100 on letterboxed canvas
    # [1, 5, 1] format: [cx, cy, w, h, conf]
    raw_output = np.array([[[320.0], [320.0], [100.0], [100.0], [0.9]]], dtype=np.float32)
    detections = detector.postprocess([raw_output])

    assert len(detections) == 1
    det = detections[0]
    # On letterbox: x1 = 320 - 50 = 270, x2 = 320 + 50 = 370
    # y1 = 320 - 50 = 270, y2 = 320 + 50 = 370
    # In original frame: ratio = 640 / 1920 = 1/3
    # orig_x1 = (270 - 0) / (1/3) = 810
    # orig_x2 = (370 - 0) / (1/3) = 1110
    # orig_y1 = (270 - 140) / (1/3) = 130 * 3 = 390
    # orig_y2 = (370 - 140) / (1/3) = 230 * 3 = 690
    assert pytest.approx(det.x1, abs=1.0) == 810.0
    assert pytest.approx(det.x2, abs=1.0) == 1110.0
    assert pytest.approx(det.y1, abs=1.0) == 390.0
    assert pytest.approx(det.y2, abs=1.0) == 690.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `py -3 -m pytest tests/test_detector.py::test_onnx_detector_letterbox_inverse_mapping -v`
Expected: FAIL because `ONNXDetector` currently uses naive resizing and doesn't store/apply letterbox padding offsets.

- [ ] **Step 3: Update `ONNXDetector` in `src/detection/detector.py`**

Update `ONNXDetector.__init__`, `preprocess`, and `postprocess`:

```python
    def __init__(
        self,
        model_path: str,
        input_size: tuple[int, int] = (640, 640),
        conf_threshold: float = 0.25,
        nms_iou_threshold: float = 0.45,
        classes: list[int] | None = None,
        class_names: dict[int, str] | None = None,
        providers: list[str] | None = None,
    ) -> None:
        self.model_path = model_path
        self.input_width, self.input_height = input_size
        self.conf_threshold = conf_threshold
        self.nms_iou_threshold = nms_iou_threshold
        self.classes = set(classes) if classes is not None else None
        self.class_names = class_names or {}
        self.providers = providers

        self._session = None
        self._input_name: str = ""
        self._output_name: str = ""
        self._orig_w: int = 0
        self._orig_h: int = 0
        self._ratio: float = 1.0
        self._pad_w: int = 0
        self._pad_h: int = 0

    def preprocess(self, frame: np.ndarray) -> np.ndarray:
        self._orig_h, self._orig_w = frame.shape[:2]
        padded, self._ratio, (self._pad_w, self._pad_h) = letterbox_image(
            frame, target_shape=(self.input_width, self.input_height)
        )
        rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
        tensor = rgb.transpose(2, 0, 1).astype(np.float32) / 255.0
        return np.expand_dims(tensor, axis=0)

    def _inverse_coords(self, x1: float, y1: float, x2: float, y2: float) -> tuple[float, float, float, float]:
        """Project letterbox coordinates back to original unpadded frame space."""
        x1 = float(np.clip((x1 - self._pad_w) / self._ratio, 0, self._orig_w))
        y1 = float(np.clip((y1 - self._pad_h) / self._ratio, 0, self._orig_h))
        x2 = float(np.clip((x2 - self._pad_w) / self._ratio, 0, self._orig_w))
        y2 = float(np.clip((y2 - self._pad_h) / self._ratio, 0, self._orig_h))
        return x1, y1, x2, y2
```
And adjust `postprocess` to use `self._inverse_coords` when converting boxes.

- [ ] **Step 4: Run test to verify it passes and check regression**

Run: `py -3 -m pytest tests/test_detector.py -v`
Expected: ALL pass.

- [ ] **Step 5: Commit**

```bash
git add src/detection/detector.py tests/test_detector.py
git commit -m "feat(detection): integrate letterbox preprocessing and inverse projection in ONNXDetector"
```

---

### Task 3: Add `onnxruntime` to Dependencies and Configure CPU Session Options

**Files:**
- Modify: `requirements.txt`
- Modify: `src/detection/detector.py`
- Test: `tests/test_detector.py`

- [ ] **Step 1: Write test for CPU session option optimization**

Add to `tests/test_detector.py`:

```python
def test_onnx_detector_cpu_thread_options(monkeypatch):
    """Verify ONNXDetector configures SessionOptions for edge CPU efficiency."""
    from unittest.mock import MagicMock
    from src.detection.detector import ONNXDetector

    mock_ort = MagicMock()
    mock_session_options = MagicMock()
    mock_ort.SessionOptions.return_value = mock_session_options
    monkeypatch.setattr("src.detection.detector.ort", mock_ort, raising=False)

    detector = ONNXDetector(model_path="dummy.onnx")
    detector._load_model()

    assert mock_ort.SessionOptions.called
    assert mock_session_options.intra_op_num_threads > 0
```

- [ ] **Step 2: Run test to verify failure**

Run: `py -3 -m pytest tests/test_detector.py::test_onnx_detector_cpu_thread_options -v`
Expected: FAIL

- [ ] **Step 3: Update `requirements.txt` and `src/detection/detector.py`**

In `requirements.txt`, append:
```
onnxruntime>=1.18.0
```

In `src/detection/detector.py`, update `_load_model`:
```python
        session_options = ort.SessionOptions()
        session_options.intra_op_num_threads = max(1, os.cpu_count() or 1)
        session_options.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        )

        self._session = ort.InferenceSession(
            self.model_path,
            sess_options=session_options,
            providers=providers,
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `py -3 -m pytest tests/test_detector.py::test_onnx_detector_cpu_thread_options -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add requirements.txt src/detection/detector.py tests/test_detector.py
git commit -m "feat(detection): add onnxruntime to requirements and configure CPU thread optimization"
```

---

### Task 4: Implement YOLO to ONNX Exporter (`tools/export_yolo.py`)

**Files:**
- Create: `tools/__init__.py`
- Create: `tools/export_yolo.py`
- Test: `tests/test_tools.py`

**Interfaces:**
- Produces: CLI script `tools/export_yolo.py --model yolov8n.pt --output models/yolov8n.onnx --imgsz 640 --opset 17`

- [ ] **Step 1: Write test for `tools/export_yolo.py` argument parsing and logic**

Create `tests/test_tools.py`:

```python
import pytest
from unittest.mock import MagicMock, patch
from tools.export_yolo import parse_args, export_yolo_to_onnx


def test_export_yolo_arg_parser():
    args = parse_args(["--model", "yolov8n.pt", "--output", "models/custom.onnx", "--imgsz", "640", "--opset", "17"])
    assert args.model == "yolov8n.pt"
    assert args.output == "models/custom.onnx"
    assert args.imgsz == 640
    assert args.opset == 17


def test_export_yolo_to_onnx_mocked(tmp_path):
    mock_yolo = MagicMock()
    mock_instance = MagicMock()
    mock_yolo.return_value = mock_instance
    exported_dummy = tmp_path / "yolov8n.onnx"
    exported_dummy.write_text("dummy")
    mock_instance.export.return_value = str(exported_dummy)

    with patch.dict("sys.modules", {"ultralytics": MagicMock(YOLO=mock_yolo)}):
        out_path = export_yolo_to_onnx(
            model_name="yolov8n.pt",
            output_path=str(tmp_path / "final.onnx"),
            imgsz=640,
            opset=17,
            simplify=True,
        )
        assert out_path == str(tmp_path / "final.onnx")
        mock_instance.export.assert_called_once_with(
            format="onnx", imgsz=640, opset=17, simplify=True
        )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `py -3 -m pytest tests/test_tools.py::test_export_yolo_arg_parser -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'tools'`

- [ ] **Step 3: Implement `tools/export_yolo.py`**

Create `tools/__init__.py` and `tools/export_yolo.py`:

```python
"""YOLO to ONNX model export utility."""
from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export YOLO model (.pt) to optimized ONNX.")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="Path to YOLO checkpoint or name (e.g. yolov8n.pt)")
    parser.add_argument("--output", type=str, default="models/yolov8n.onnx", help="Destination path for .onnx file")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image dimension (default: 640)")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version (default: 17)")
    parser.add_argument("--no-simplify", action="store_true", help="Disable onnx graph simplification")
    return parser.parse_args(args)


def export_yolo_to_onnx(
    model_name: str,
    output_path: str,
    imgsz: int = 640,
    opset: int = 17,
    simplify: bool = True,
) -> str:
    try:
        from ultralytics import YOLO
    except ImportError:
        raise ImportError(
            "ultralytics is required for exporting models. Install with: pip install ultralytics"
        )

    print(f"Loading YOLO model: {model_name}")
    model = YOLO(model_name)

    print(f"Exporting to ONNX (imgsz={imgsz}, opset={opset}, simplify={simplify})...")
    exported_file = model.export(
        format="onnx",
        imgsz=imgsz,
        opset=opset,
        simplify=simplify,
    )

    out_dest = Path(output_path)
    out_dest.parent.mkdir(parents=True, exist_ok=True)
    if Path(exported_file).resolve() != out_dest.resolve():
        shutil.move(exported_file, out_dest)

    print(f"Model exported successfully to: {out_dest}")
    return str(out_dest)


def main() -> None:
    args = parse_args()
    export_yolo_to_onnx(
        model_name=args.model,
        output_path=args.output,
        imgsz=args.imgsz,
        opset=args.opset,
        simplify=not args.no_simplify,
    )


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `py -3 -m pytest tests/test_tools.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tools/__init__.py tools/export_yolo.py tests/test_tools.py
git commit -m "feat(tools): add export_yolo CLI utility"
```

---

### Task 5: Implement Semi-Supervised Frame Extraction & Auto-Labeling Pipeline (`tools/train_carcass_detector.py`)

**Files:**
- Create: `tools/train_carcass_detector.py`
- Test: `tests/test_tools.py`

**Interfaces:**
- Produces:
  - `extract_keyframes(video_path: str, output_dir: str, num_frames: int = 60) -> list[str]`
  - `auto_label_frame(frame: np.ndarray, shackle_detector: ShackleDetector) -> list[tuple[int, float, float, float, float]]`
  - `create_dataset_structure(image_paths: list[str], labels: list[list[tuple]], dataset_dir: str, train_ratio: float = 0.8) -> str`
  - CLI: `python tools/train_carcass_detector.py --video samples/Rekaman_2.MOV --frames 60 --epochs 15 --output models/carcass_yolov8n.onnx`

- [ ] **Step 1: Write unit tests for frame extraction and auto-labeling**

Add to `tests/test_tools.py`:

```python
def test_auto_label_frame():
    """Verify auto_label_frame generates normalized [class, x_center, y_center, w, h] boxes from ShackleDetector."""
    from tools.train_carcass_detector import auto_label_frame
    from unittest.mock import MagicMock
    from src.detection.detector import Detection

    mock_detector = MagicMock()
    # Mock a detection on 1920x1080 frame
    mock_detector.detect.return_value = [
        Detection(x1=910.0, y1=200.0, x2=1010.0, y2=800.0, confidence=0.85, class_id=0, class_name="chicken")
    ]

    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    labels = auto_label_frame(frame, mock_detector)

    assert len(labels) == 1
    cls_id, xc, yc, w, h = labels[0]
    assert cls_id == 0
    # Center x = 960 / 1920 = 0.5
    assert pytest.approx(xc, abs=1e-3) == 0.5
    # Center y = 500 / 1080 = 0.46296
    assert pytest.approx(yc, abs=1e-3) == 500 / 1080
    # Width = 100 / 1920 = 0.05208
    assert pytest.approx(w, abs=1e-3) == 100 / 1920
    # Height = 600 / 1080 = 0.5555
    assert pytest.approx(h, abs=1e-3) == 600 / 1080


def test_create_dataset_structure(tmp_path):
    """Verify dataset files and dataset.yaml are correctly written."""
    from tools.train_carcass_detector import create_dataset_structure

    img_file = tmp_path / "frame_001.jpg"
    img_file.write_text("fake image data")

    dataset_yaml = create_dataset_structure(
        image_paths=[str(img_file)],
        labels=[[(0, 0.5, 0.5, 0.1, 0.2)]],
        dataset_dir=str(tmp_path / "dataset"),
        train_ratio=1.0,
    )

    assert Path(dataset_yaml).exists()
    content = Path(dataset_yaml).read_text()
    assert "names:" in content
    assert "0: chicken_carcass" in content
```

- [ ] **Step 2: Run test to verify failure**

Run: `py -3 -m pytest tests/test_tools.py::test_auto_label_frame -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'tools.train_carcass_detector'`

- [ ] **Step 3: Implement `tools/train_carcass_detector.py`**

Create `tools/train_carcass_detector.py` with frame sampling, auto-labeling via `ShackleDetector`, dataset packaging, and Ultralytics training/export execution.

- [ ] **Step 4: Run test to verify it passes**

Run: `py -3 -m pytest tests/test_tools.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tools/train_carcass_detector.py tests/test_tools.py
git commit -m "feat(tools): add semi-supervised carcass frame extraction and training utility"
```

---

### Task 6: End-to-End Pipeline Verification and Regression Test

**Files:**
- Modify: `main.py` (ensure `--model` argument properly instantiates `ONNXDetector` with letterbox)
- Test: Full pytest suite

- [ ] **Step 1: Verify `main.py` CLI handles `--model` with letterboxed `ONNXDetector`**

Run: `py -3 -m pytest tests/test_main.py -v`
Ensure detector instantiation with `--model` works seamlessly.

- [ ] **Step 2: Run complete test suite**

Run: `py -3 -m pytest`
Expected: All 137+ tests PASS.

- [ ] **Step 3: Benchmark latency**

Run: `py -3 benchmark.py --detector shackle`
Confirm baseline performance and ensure no slowdown.

- [ ] **Step 4: Final commit**

```bash
git add .
git commit -m "chore: complete YOLO ONNX detection accuracy enhancements"
```
