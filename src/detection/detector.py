"""Detector abstractions and concrete detector implementations."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
import os
import sys
from typing import Any, Sequence
import cv2
import numpy as np

try:
    import onnxruntime as ort
except Exception:
    ort = None

__all__ = [
    "Detection",
    "Detector",
    "MockDetector",
    "ONNXDetector",
    "ShackleDetector",
    "letterbox_image",
]


@dataclass(frozen=True)
class Detection:
    """Bounding box detection with class information and anchor point."""

    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    class_id: int
    class_name: str

    @property
    def anchor_bottom_center(self) -> tuple[float, float]:
        """Anchor point used for virtual line crossing (x_mid, y2)."""
        return ((self.x1 + self.x2) / 2.0, self.y2)

    @property
    def width(self) -> float:
        """Width of bounding box."""
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        """Height of bounding box."""
        return max(0.0, self.y2 - self.y1)


class Detector(ABC):
    """Abstract base class for all object detectors."""

    @abstractmethod
    def predict(self, frame: np.ndarray) -> list[Detection]:
        """Run inference on the given BGR frame and return a list of detections.

        Args:
            frame: Numpy array containing BGR image frame (H, W, C).

        Returns:
            List of Detection objects detected in the frame.
        """
        pass


class MockDetector(Detector):
    """Threshold-based procedural detector for testing without deep learning models."""

    def __init__(
        self,
        min_confidence: float = 0.85,
        min_width: float = 10.0,
        min_height: float = 10.0,
        min_area: float = 0.0,
        max_area: float | None = None,
        intensity_threshold: int = 150,
        class_name: str = "chicken",
        class_id: int = 0,
        apply_blur: bool = False,
        morph_open: bool = False,
        roi: tuple[int, int, int, int] | None = None,
    ) -> None:
        self.min_confidence = min_confidence
        self.min_width = min_width
        self.min_height = min_height
        self.min_area = min_area
        self.max_area = max_area
        self.intensity_threshold = intensity_threshold
        self.class_name = class_name
        self.class_id = class_id
        self.apply_blur = apply_blur
        self.morph_open = morph_open
        self.roi = roi

    def predict(self, frame: np.ndarray) -> list[Detection]:
        """Detect bright contour regions above the intensity threshold."""
        detections: list[Detection] = []
        if frame is None or frame.size == 0:
            return detections

        x_offset, y_offset = 0, 0
        working_frame = frame
        if self.roi is not None:
            ymin, ymax, xmin, xmax = self.roi
            working_frame = working_frame[ymin:ymax, xmin:xmax]
            x_offset, y_offset = xmin, ymin
            if working_frame.size == 0:
                return detections

        if working_frame.ndim == 3:
            gray = cv2.cvtColor(working_frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = working_frame

        if self.apply_blur:
            gray = cv2.GaussianBlur(gray, (7, 7), 0)

        _, thresh = cv2.threshold(
            gray, self.intensity_threshold, 255, cv2.THRESH_BINARY
        )

        if self.morph_open:
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        for cnt in contours:
            area = float(cv2.contourArea(cnt))
            if self.min_area > 0 and area < self.min_area:
                continue
            if self.max_area is not None and area > self.max_area:
                continue

            x, y, w, h = cv2.boundingRect(cnt)
            if float(w) > self.min_width and float(h) > self.min_height:
                detections.append(
                    Detection(
                        x1=float(x + x_offset),
                        y1=float(y + y_offset),
                        x2=float(x + w + x_offset),
                        y2=float(y + h + y_offset),
                        confidence=self.min_confidence,
                        class_id=self.class_id,
                        class_name=self.class_name,
                    )
                )
        return detections


class ShackleDetector(Detector):
    """Specialized detector for industrial poultry on overhead shackle conveyor lines.

    Analyzes vertical steel shackle brackets to locate regularly spaced shackle centers,
    then frames each suspended chicken carcass with high spatial stability.
    """

    def __init__(
        self,
        min_confidence: float = 0.90,
        shackle_y_ratio: tuple[float, float] = (0.24, 0.48),
        min_spacing: float = 55.0,
        chicken_width: float = 115.0,
        chicken_height: float = 580.0,
        chicken_y_top: float | None = 260.0,
        chicken_y_top_ratio: float = 0.25,
        class_name: str = "chicken",
        class_id: int = 0,
        occupancy_check: bool = True,
        carcass_y_ratio: tuple[float, float] = (0.62, 0.76),
        occupancy_threshold: float = 4.5,
        min_peak_energy: float = 12.0,
        max_bar_spacing: float = 25.0,
    ) -> None:
        self.min_confidence = min_confidence
        self.shackle_y_ratio = shackle_y_ratio
        self.min_spacing = min_spacing
        self.chicken_width = chicken_width
        self.chicken_height = chicken_height
        self.chicken_y_top = chicken_y_top
        self.chicken_y_top_ratio = chicken_y_top_ratio
        self.class_name = class_name
        self.class_id = class_id
        self.occupancy_check = occupancy_check
        self.carcass_y_ratio = carcass_y_ratio
        self.occupancy_threshold = occupancy_threshold
        self.min_peak_energy = min_peak_energy
        self.max_bar_spacing = max_bar_spacing

    def predict(self, frame: np.ndarray) -> list[Detection]:
        """Detect individual chickens on the shackle line with regular spacing."""
        detections: list[Detection] = []
        if frame is None or frame.size == 0:
            return detections

        h, w = frame.shape[:2]
        y1_shackle = int(max(0.0, min(1.0, self.shackle_y_ratio[0])) * h)
        y2_shackle = int(max(0.0, min(1.0, self.shackle_y_ratio[1])) * h)
        if y2_shackle <= y1_shackle:
            return detections

        shackle_roi = frame[y1_shackle:y2_shackle, :]
        if shackle_roi.ndim == 3:
            gray = cv2.cvtColor(shackle_roi, cv2.COLOR_BGR2GRAY)
        else:
            gray = shackle_roi

        sobelx = np.abs(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)).astype(np.uint8)
        col_energy = cv2.GaussianBlur(
            sobelx.mean(axis=0).reshape(1, -1), (25, 1), 0
        ).flatten()

        max_energy = float(col_energy.max()) if len(col_energy) > 0 else 0.0
        if max_energy < 10.0:
            return detections

        threshold = max(self.min_peak_energy, float(max_energy * 0.40))
        peaks: list[tuple[int, float]] = []
        i = 1
        n = len(col_energy)
        while i < n - 1:
            if col_energy[i] >= threshold and col_energy[i] > col_energy[i - 1]:
                start = i
                while i < n - 1 and col_energy[i] == col_energy[i + 1]:
                    i += 1
                if i < n - 1 and col_energy[i] > col_energy[i + 1]:
                    peak_pos = (start + i) // 2
                    peaks.append((peak_pos, float(col_energy[start])))
            i += 1

        # Cluster paired bracket bars (separated by <= max_bar_spacing)
        clusters: list[list[tuple[int, float]]] = []
        for p, val in peaks:
            if not clusters:
                clusters.append([(p, val)])
            elif (p - clusters[-1][-1][0]) <= self.max_bar_spacing:
                clusters[-1].append((p, val))
            else:
                clusters.append([(p, val)])

        cluster_info: list[tuple[int, float]] = []
        for c in clusters:
            cx = int(np.mean([pt[0] for pt in c]))
            max_val = max(pt[1] for pt in c)
            cluster_info.append((cx, max_val))

        # Filter clusters closer than min_spacing (retain stronger bracket candidate)
        filtered_centers: list[tuple[int, float]] = []
        for cx, val in cluster_info:
            if not filtered_centers:
                filtered_centers.append((cx, val))
            elif (cx - filtered_centers[-1][0]) < self.min_spacing:
                if val > filtered_centers[-1][1]:
                    filtered_centers[-1] = (cx, val)
            else:
                filtered_centers.append((cx, val))

        centers = [c[0] for c in filtered_centers]

        y_top = (
            self.chicken_y_top
            if self.chicken_y_top is not None
            else float(h * self.chicken_y_top_ratio)
        )
        box_w = float(self.chicken_width)
        box_h = float(self.chicken_height)

        full_gray = None
        y1_carcass = 0
        y2_carcass = 0
        if self.occupancy_check:
            full_gray = (
                cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                if frame.ndim == 3
                else frame
            )
            y1_carcass = int(max(0.0, min(1.0, self.carcass_y_ratio[0])) * h)
            y2_carcass = int(max(0.0, min(1.0, self.carcass_y_ratio[1])) * h)

        for cx in centers:
            if self.occupancy_check and full_gray is not None and y2_carcass > y1_carcass:
                pw = max(6, int(0.006 * w))
                px1 = max(0, cx - pw)
                px2 = min(w, cx + pw)
                patch = full_gray[y1_carcass:y2_carcass, px1:px2]
                if patch.size == 0:
                    continue
                sx = float(np.abs(cv2.Sobel(patch, cv2.CV_16S, 1, 0)).mean())
                if sx < self.occupancy_threshold:
                    continue

            x1 = max(0.0, float(cx - box_w / 2.0))
            x2 = min(float(w), float(cx + box_w / 2.0))
            y1 = max(0.0, float(y_top))
            y2 = min(float(h), float(y_top + box_h))
            detections.append(
                Detection(
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                    confidence=self.min_confidence,
                    class_id=self.class_id,
                    class_name=self.class_name,
                )
            )
        return detections


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
        (padded_image, scale_ratio, (pad_left, pad_top))
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


class ONNXDetector(Detector):
    """ONNX Runtime detector supporting YOLO models with aspect-preserving letterboxing."""

    def __init__(
        self,
        model_path: str,
        input_size: tuple[int, int] = (640, 640),
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        nms_iou_threshold: float | None = None,
        classes: list[int] | None = None,
        class_names: Sequence[str] | dict[int, str] | None = None,
        providers: list[str] | None = None,
        input_width: int | None = None,
        input_height: int | None = None,
    ) -> None:
        self.model_path = str(model_path)

        # Handle positional conf_threshold if passed in place of input_size
        if isinstance(input_size, (int, float)) and not isinstance(input_size, (tuple, list)):
            conf_threshold = float(input_size)
            input_size = (640, 640)

        if input_width is not None or input_height is not None:
            self.input_width = int(input_width) if input_width is not None else 640
            self.input_height = int(input_height) if input_height is not None else 640
        elif input_size is not None and isinstance(input_size, (tuple, list)):
            self.input_width = int(input_size[0])
            self.input_height = int(input_size[1])
        else:
            self.input_width = 640
            self.input_height = 640

        self.conf_threshold = float(conf_threshold)
        if nms_iou_threshold is not None:
            self.nms_iou_threshold = float(nms_iou_threshold)
            self.iou_threshold = self.nms_iou_threshold
        else:
            self.iou_threshold = float(iou_threshold)
            self.nms_iou_threshold = self.iou_threshold

        self.classes = set(classes) if classes is not None else None

        if class_names is None:
            self.class_names: dict[int, str] = {0: "chicken"}
        elif isinstance(class_names, dict):
            self.class_names = class_names
        else:
            self.class_names = {idx: name for idx, name in enumerate(class_names)}

        self.providers = providers

        self._session: Any = None
        self._input_name: str = ""
        self._output_name: str = ""
        self._orig_w: int = 0
        self._orig_h: int = 0
        self._ratio: float = 1.0
        self._pad_w: int = 0
        self._pad_h: int = 0

        # Eagerly load model unless dummy/mock
        if "dummy" not in self.model_path.lower():
            self._load_model()

    @property
    def session(self) -> Any:
        return self._session

    @session.setter
    def session(self, value: Any) -> None:
        self._session = value

    @property
    def input_name(self) -> str:
        return self._input_name

    @input_name.setter
    def input_name(self, value: str) -> None:
        self._input_name = value

    @property
    def output_name(self) -> str:
        return self._output_name

    @output_name.setter
    def output_name(self, value: str) -> None:
        self._output_name = value

    @property
    def input_size(self) -> tuple[int, int]:
        return (self.input_width, self.input_height)

    @input_size.setter
    def input_size(self, size: tuple[int, int]) -> None:
        self.input_width = int(size[0])
        self.input_height = int(size[1])

    def _load_model(self) -> None:
        """Initialize ONNX inference session."""
        global ort
        if "onnxruntime" in sys.modules:
            ort = sys.modules["onnxruntime"]

        if ort is None:
            try:
                import onnxruntime as _ort

                ort = _ort
                if ort is None:
                    raise ImportError("onnxruntime is None")
            except Exception as exc:
                raise RuntimeError(
                    f"onnxruntime is not installed or unavailable: {exc}. "
                    "Please install onnxruntime or onnxruntime-gpu to use ONNXDetector."
                ) from exc

        providers = self.providers
        if providers is None:
            available = (
                ort.get_available_providers()
                if hasattr(ort, "get_available_providers")
                else []
            )
            preferred = [
                "CUDAExecutionProvider",
                "TensorrtExecutionProvider",
                "CoreMLExecutionProvider",
                "OpenVINOExecutionProvider",
                "CPUExecutionProvider",
            ]
            providers = [p for p in preferred if p in available]
            if not providers:
                providers = ["CPUExecutionProvider"]

        self._session = ort.InferenceSession(self.model_path, providers=providers)
        inputs = self._session.get_inputs() if hasattr(self._session, "get_inputs") else []
        if inputs:
            self._input_name = inputs[0].name
            input_shape = getattr(inputs[0], "shape", None)
            if (
                input_shape
                and len(input_shape) >= 4
                and isinstance(input_shape[2], int)
                and isinstance(input_shape[3], int)
            ):
                self.input_height = input_shape[2]
                self.input_width = input_shape[3]

        outputs = self._session.get_outputs() if hasattr(self._session, "get_outputs") else []
        if outputs:
            self._output_name = getattr(outputs[0], "name", "")

    def preprocess(self, frame: np.ndarray) -> np.ndarray:
        """Preprocess frame using aspect-preserving letterboxing."""
        self._orig_h, self._orig_w = frame.shape[:2]
        if frame.ndim == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        padded, self._ratio, (self._pad_w, self._pad_h) = letterbox_image(
            frame, target_shape=(self.input_width, self.input_height)
        )
        rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
        tensor = np.ascontiguousarray(rgb.transpose(2, 0, 1).astype(np.float32) / 255.0)
        return np.expand_dims(tensor, axis=0)

    def _inverse_coords(
        self, x1: float, y1: float, x2: float, y2: float
    ) -> tuple[float, float, float, float]:
        """Project letterbox coordinates back to original unpadded frame space."""
        ratio = self._ratio if self._ratio > 0 else 1.0
        max_w = float(self._orig_w) if self._orig_w > 0 else float(self.input_width)
        max_h = float(self._orig_h) if self._orig_h > 0 else float(self.input_height)
        inv_x1 = float(np.clip((x1 - self._pad_w) / ratio, 0, max_w))
        inv_y1 = float(np.clip((y1 - self._pad_h) / ratio, 0, max_h))
        inv_x2 = float(np.clip((x2 - self._pad_w) / ratio, 0, max_w))
        inv_y2 = float(np.clip((y2 - self._pad_h) / ratio, 0, max_h))
        return inv_x1, inv_y1, inv_x2, inv_y2

    def postprocess(
        self, outputs: Sequence[np.ndarray] | list[np.ndarray]
    ) -> list[Detection]:
        """Convert ONNX model outputs to Detection objects in original frame coordinates."""
        if not outputs:
            return []

        raw = outputs[0]
        if raw is None or getattr(raw, "size", 0) == 0:
            return []

        if raw.ndim == 3 and raw.shape[0] == 1:
            raw = raw[0]

        # Case 1: YOLO26 / end-to-end NMS-free format: [N, 6] -> (x1, y1, x2, y2, conf, cls_id)
        if raw.ndim == 2 and (
            raw.shape[1] == 6 or (raw.shape[0] == 6 and raw.shape[1] > 6)
        ):
            if raw.shape[0] == 6 and raw.shape[1] > 6:
                raw = raw.T

            detections: list[Detection] = []
            for row in raw:
                x1, y1, x2, y2, conf, cls_id = row
                if conf < self.conf_threshold:
                    continue

                cid = int(cls_id)
                if self.classes is not None and cid not in self.classes:
                    continue
                cname = self.class_names.get(cid, str(cid))
                inv_x1, inv_y1, inv_x2, inv_y2 = self._inverse_coords(
                    float(x1), float(y1), float(x2), float(y2)
                )
                real_x1, real_x2 = min(inv_x1, inv_x2), max(inv_x1, inv_x2)
                real_y1, real_y2 = min(inv_y1, inv_y2), max(inv_y1, inv_y2)
                detections.append(
                    Detection(
                        x1=real_x1,
                        y1=real_y1,
                        x2=real_x2,
                        y2=real_y2,
                        confidence=float(conf),
                        class_id=cid,
                        class_name=cname,
                    )
                )
            return detections

        # Case 2: Raw YOLO format: [4 + num_classes, N] or [N, 4 + num_classes]
        if raw.ndim == 2:
            expected_feat = 4 + len(self.class_names)
            if raw.shape[0] == expected_feat and raw.shape[1] != expected_feat:
                raw = raw.T
            elif raw.shape[1] < 5 and raw.shape[0] >= 5:
                raw = raw.T
            elif raw.shape[0] < raw.shape[1] and raw.shape[0] <= 100 < raw.shape[1]:
                raw = raw.T

            candidate_detections: list[Detection] = []
            nms_boxes: list[list[int]] = []
            nms_scores: list[float] = []

            for row in raw:
                if len(row) < 5:
                    continue
                cx, cy, w, h = row[:4]
                class_scores = row[4:]
                if len(class_scores) == 0:
                    continue

                best_cls = int(np.argmax(class_scores))
                best_conf = float(class_scores[best_cls])
                if best_conf < self.conf_threshold:
                    continue

                if self.classes is not None and best_cls not in self.classes:
                    continue

                box_x1 = float(cx - w / 2.0)
                box_y1 = float(cy - h / 2.0)
                box_x2 = float(cx + w / 2.0)
                box_y2 = float(cy + h / 2.0)

                inv_x1, inv_y1, inv_x2, inv_y2 = self._inverse_coords(
                    box_x1, box_y1, box_x2, box_y2
                )
                real_x1, real_x2 = min(inv_x1, inv_x2), max(inv_x1, inv_x2)
                real_y1, real_y2 = min(inv_y1, inv_y2), max(inv_y1, inv_y2)

                candidate_detections.append(
                    Detection(
                        x1=real_x1,
                        y1=real_y1,
                        x2=real_x2,
                        y2=real_y2,
                        confidence=best_conf,
                        class_id=best_cls,
                        class_name=self.class_names.get(best_cls, str(best_cls)),
                    )
                )
                nms_boxes.append(
                    [
                        int(real_x1),
                        int(real_y1),
                        int(max(0.0, real_x2 - real_x1)),
                        int(max(0.0, real_y2 - real_y1)),
                    ]
                )
                nms_scores.append(best_conf)

            if not nms_boxes:
                return []

            indices = cv2.dnn.NMSBoxes(
                nms_boxes,
                nms_scores,
                self.conf_threshold,
                self.nms_iou_threshold,
            )

            results: list[Detection] = []
            if len(indices) > 0:
                for idx in np.array(indices).flatten():
                    results.append(candidate_detections[idx])
            return results

        return []

    def predict(self, frame: np.ndarray) -> list[Detection]:
        """Preprocess frame, run ONNX inference, and parse detection boxes."""
        if frame is None or getattr(frame, "size", 0) == 0:
            return []

        if self._session is None:
            self._load_model()

        tensor = self.preprocess(frame)
        outputs = self._session.run(None, {self._input_name: tensor})
        if not outputs:
            return []

        return self.postprocess(outputs)

    # Alias for API uniformity
    detect = predict
