"""YOLO to ONNX model export utility."""
from __future__ import annotations

import argparse
import shutil
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
