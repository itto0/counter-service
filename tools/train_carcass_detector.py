"""Semi-supervised frame extraction, auto-labeling, and YOLO carcass detector training utility."""
from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import sys
from typing import Any

# Ensure counter root and monorepo root are in sys.path when script is executed directly
counter_root = Path(__file__).resolve().parent.parent
monorepo_root = counter_root.parent.parent

if str(counter_root) not in sys.path:
    sys.path.insert(0, str(counter_root))

if monorepo_root.is_dir() and str(monorepo_root) not in sys.path:
    sys.path.append(str(monorepo_root))

import cv2
import numpy as np

try:
    from tools.export_yolo import export_yolo_to_onnx
except ImportError:
    from export_yolo import export_yolo_to_onnx


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    """Parse command line arguments for the carcass detector training pipeline."""
    parser = argparse.ArgumentParser(
        description="Extract keyframes from poultry conveyor video, auto-label using shackle analysis, and train YOLO carcass detector."
    )
    parser.add_argument(
        "--video",
        type=str,
        default="samples/Rekaman_2.MOV",
        help="Path to conveyor video file (default: samples/Rekaman_2.MOV)",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=60,
        help="Number of keyframes to extract for training/validation (default: 60)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=15,
        help="Number of training epochs (default: 15)",
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default="yolov8n.pt",
        help="Pretrained YOLO base checkpoint (default: yolov8n.pt)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="models/carcass_yolov8n.onnx",
        help="Destination path for exported ONNX model (default: models/carcass_yolov8n.onnx)",
    )
    parser.add_argument(
        "--work-dir",
        type=str,
        default="dataset/carcass",
        help="Working directory for dataset and training artifacts (default: dataset/carcass)",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Input image resolution for training and export (default: 640)",
    )
    return parser.parse_args(args)


def extract_keyframes(
    video_path: str,
    output_dir: str,
    num_frames: int = 60,
) -> list[str]:
    """Extract evenly spaced keyframes from a video file and save them as JPEG images.

    Args:
        video_path: Path to video file.
        output_dir: Directory where extracted images will be saved.
        num_frames: Number of evenly spaced frames to extract.

    Returns:
        Sorted list of written image file paths.

    Raises:
        FileNotFoundError: If the video cannot be opened.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video file: {video_path}")

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    try:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if num_frames <= 0:
            return []

        if total_frames > 0:
            if total_frames <= num_frames:
                frame_indices = list(range(total_frames))
            else:
                step = total_frames / num_frames
                frame_indices = [int(i * step) for i in range(num_frames)]
        else:
            frame_indices = list(range(num_frames))

        saved_paths: list[str] = []
        for i, frame_idx in enumerate(frame_indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret or frame is None:
                continue
            file_path = out_path / f"frame_{i:04d}.jpg"
            if cv2.imwrite(str(file_path), frame):
                saved_paths.append(str(file_path))

        return sorted(saved_paths)
    finally:
        cap.release()


def auto_label_frame(
    frame: np.ndarray,
    shackle_detector: Any,
) -> list[tuple[int, float, float, float, float]]:
    """Detect chicken carcasses in a frame and convert detections to normalized YOLO format.

    Args:
        frame: BGR image frame (H, W, C).
        shackle_detector: Detector instance or callable supporting detect() or predict().

    Returns:
        List of tuples formatted as (class_id, x_center, y_center, w, h) with values clamped to [0.0, 1.0].
    """
    if frame is None or frame.size == 0:
        return []

    height, width = frame.shape[:2]
    if height <= 0 or width <= 0:
        return []

    if hasattr(shackle_detector, "detect"):
        detections = shackle_detector.detect(frame)
    elif hasattr(shackle_detector, "predict"):
        detections = shackle_detector.predict(frame)
    elif callable(shackle_detector):
        detections = shackle_detector(frame)
    else:
        detections = []

    labels: list[tuple[int, float, float, float, float]] = []
    for det in detections:
        class_id = 0
        if hasattr(det, "x1") and hasattr(det, "y1") and hasattr(det, "x2") and hasattr(det, "y2"):
            x1 = float(det.x1)
            y1 = float(det.y1)
            x2 = float(det.x2)
            y2 = float(det.y2)
            class_id = int(getattr(det, "class_id", 0))
        elif isinstance(det, (list, tuple)) and len(det) >= 4:
            x1 = float(det[0])
            y1 = float(det[1])
            x2 = float(det[2])
            y2 = float(det[3])
            if len(det) >= 6:
                class_id = int(det[5])
            elif len(det) == 5 and isinstance(det[4], int):
                class_id = int(det[4])
        elif isinstance(det, dict) and "x1" in det and "y1" in det and "x2" in det and "y2" in det:
            x1 = float(det["x1"])
            y1 = float(det["y1"])
            x2 = float(det["x2"])
            y2 = float(det["y2"])
            class_id = int(det.get("class_id", 0))
        else:
            continue

        # Pre-clamp bounding box to image boundaries before calculating normalized centers
        x1 = max(0.0, min(float(width), x1))
        y1 = max(0.0, min(float(height), y1))
        x2 = max(0.0, min(float(width), x2))
        y2 = max(0.0, min(float(height), y2))

        if x2 <= x1 or y2 <= y1:
            continue

        x_center = ((x1 + x2) / 2.0) / width
        y_center = ((y1 + y2) / 2.0) / height
        w = (x2 - x1) / width
        h = (y2 - y1) / height

        x_center = max(0.0, min(1.0, float(x_center)))
        y_center = max(0.0, min(1.0, float(y_center)))
        w = max(0.0, min(1.0, float(w)))
        h = max(0.0, min(1.0, float(h)))

        labels.append((class_id, x_center, y_center, w, h))

    return labels


def create_dataset_structure(
    image_paths: list[str],
    labels: list[list[tuple[int, float, float, float, float]]],
    dataset_dir: str,
    train_ratio: float = 0.8,
) -> str:
    """Organize image and label files into YOLO standard format and generate dataset.yaml.

    Args:
        image_paths: List of file paths to images.
        labels: List of label sets corresponding to each image.
        dataset_dir: Root directory where dataset will be created.
        train_ratio: Ratio of data reserved for training (remainder for validation).

    Returns:
        Path to generated dataset.yaml file.
    """
    if len(image_paths) != len(labels):
        raise ValueError(
            f"Mismatch between images count ({len(image_paths)}) and labels count ({len(labels)})"
        )

    base_dir = Path(dataset_dir)
    img_train_dir = base_dir / "images" / "train"
    img_val_dir = base_dir / "images" / "val"
    lbl_train_dir = base_dir / "labels" / "train"
    lbl_val_dir = base_dir / "labels" / "val"

    for d in (img_train_dir, img_val_dir, lbl_train_dir, lbl_val_dir):
        d.mkdir(parents=True, exist_ok=True)

    n = len(image_paths)
    if n <= 1:
        # For single-sample datasets, mirror into both train and val so YOLO validation never raises
        train_indices = set(range(n))
        val_indices = set(range(n))
    else:
        train_count = int(round(n * train_ratio))
        if train_count >= n:
            train_count = n - 1
        elif train_count < 1:
            train_count = 1
        train_indices = set(range(train_count))
        val_indices = set(range(train_count, n))

    for idx, (img_path_str, item_labels) in enumerate(zip(image_paths, labels)):
        target_dirs = []
        if idx in train_indices:
            target_dirs.append((img_train_dir, lbl_train_dir))
        if idx in val_indices:
            target_dirs.append((img_val_dir, lbl_val_dir))

        for target_img_dir, target_lbl_dir in target_dirs:
            src_img = Path(img_path_str)
            dst_img = target_img_dir / src_img.name

            if src_img.exists():
                if src_img.resolve() != dst_img.resolve():
                    shutil.copy2(src_img, dst_img)
            else:
                dst_img.touch()

            lbl_file = target_lbl_dir / f"{src_img.stem}.txt"
            lines = [
                f"{cid} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n"
                for cid, xc, yc, w, h in item_labels
            ]
            lbl_file.write_text("".join(lines), encoding="utf-8")

    yaml_path = base_dir / "dataset.yaml"
    yaml_content = (
        f"path: {base_dir.resolve().as_posix()}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"names:\n"
        f"  0: chicken_carcass\n"
    )
    yaml_path.write_text(yaml_content, encoding="utf-8")
    return str(yaml_path)


def evaluate_trained_model(
    model_path: str | Path,
    yaml_path: str | Path,
    imgsz: int = 640,
) -> dict[str, float]:
    """Evaluate trained YOLO checkpoint on validation split and return accuracy metrics.

    Args:
        model_path: Path to YOLO weights checkpoint (.pt).
        yaml_path: Path to dataset.yaml.
        imgsz: Evaluation image resolution.

    Returns:
        Dictionary containing map50, map50_95, precision, and recall metrics.
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        raise ImportError(
            "ultralytics is required for evaluating models. Install with: pip install ultralytics"
        )

    model = YOLO(str(model_path))
    val_results = model.val(data=str(yaml_path), imgsz=imgsz, split="val")

    box = getattr(val_results, "box", None)
    map50 = float(getattr(box, "map50", 0.0) or 0.0) if box is not None else 0.0
    map50_95 = float(getattr(box, "map", 0.0) or 0.0) if box is not None else 0.0
    precision = float(getattr(box, "mp", 0.0) or 0.0) if box is not None else 0.0
    recall = float(getattr(box, "mr", 0.0) or 0.0) if box is not None else 0.0

    return {
        "map50": round(map50, 4),
        "map50_95": round(map50_95, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
    }


def train_carcass_detector(
    video_path: str = "samples/Rekaman_2.MOV",
    num_frames: int = 60,
    epochs: int = 15,
    base_model: str = "yolov8n.pt",
    output_model: str = "models/carcass_yolov8n.onnx",
    work_dir: str = "dataset/carcass",
    imgsz: int = 640,
    detector: Any = None,
) -> str:
    """Run end-to-end semi-supervised pipeline: extract frames, auto-label, train YOLO, and export ONNX."""
    work_path = Path(work_dir)
    raw_frames_dir = work_path / "raw_frames"

    if detector is None:
        from src.detection.detector import ShackleDetector
        detector = ShackleDetector()

    print(f"Extracting {num_frames} keyframes from {video_path}...")
    image_paths = extract_keyframes(video_path, str(raw_frames_dir), num_frames=num_frames)

    print(f"Auto-labeling {len(image_paths)} frames using shackle detector...")
    labels = []
    for p in image_paths:
        frame = cv2.imread(p)
        if frame is None:
            print(f"Warning: could not read extracted frame {p}")
            labels.append([])
        else:
            labels.append(auto_label_frame(frame, detector))

    print(f"Creating YOLO dataset structure in {work_dir}...")
    yaml_path = create_dataset_structure(image_paths, labels, str(work_path))

    try:
        from ultralytics import YOLO
    except ImportError:
        raise ImportError(
            "ultralytics is required for training models. Install with: pip install ultralytics"
        )

    print(f"Loading base YOLO model: {base_model}...")
    model = YOLO(base_model)

    runs_dir = work_path / "runs"
    print(f"Starting training on {yaml_path} for {epochs} epochs (imgsz={imgsz})...")
    results = model.train(
        data=str(yaml_path),
        epochs=epochs,
        imgsz=imgsz,
        project=str(runs_dir),
        name="train",
    )

    best_pt: Path | None = None
    if hasattr(model, "trainer") and model.trainer is not None:
        if getattr(model.trainer, "best", None):
            cand = Path(model.trainer.best)
            if cand.exists():
                best_pt = cand
        if best_pt is None and getattr(model.trainer, "save_dir", None):
            cand = Path(model.trainer.save_dir) / "weights" / "best.pt"
            if cand.exists():
                best_pt = cand

    if best_pt is None and hasattr(results, "save_dir") and results.save_dir is not None:
        cand = Path(results.save_dir) / "weights" / "best.pt"
        if cand.exists():
            best_pt = cand

    if best_pt is None:
        candidates = list(work_path.glob("**/weights/best.pt"))
        if candidates:
            best_pt = candidates[-1]

    if best_pt is None:
        last_candidates = list(work_path.glob("**/weights/last.pt"))
        if last_candidates:
            best_pt = last_candidates[-1]
        else:
            best_pt = Path(base_model)

    print(f"Exporting best model ({best_pt}) to ONNX ({output_model})...")
    exported_path = export_yolo_to_onnx(
        model_name=str(best_pt),
        output_path=output_model,
        imgsz=imgsz,
    )
    return exported_path


def main(args: list[str] | None = None) -> None:
    """CLI entrypoint for carcass detector training."""
    parsed = parse_args(args)
    train_carcass_detector(
        video_path=parsed.video,
        num_frames=parsed.frames,
        epochs=parsed.epochs,
        base_model=parsed.base_model,
        output_model=parsed.output,
        work_dir=parsed.work_dir,
        imgsz=parsed.imgsz,
    )


if __name__ == "__main__":
    main()
