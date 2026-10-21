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
    parser.add_argument(
        "--simplify",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable/disable onnx graph simplification (default: enabled)",
    )
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

    # Optional sanity check on exported model using onnxruntime if available
    try:
        import numpy as np
        import onnxruntime as ort

        sess = ort.InferenceSession(str(out_dest), providers=["CPUExecutionProvider"])
        input_name = sess.get_inputs()[0].name
        dummy_tensor = np.zeros((1, 3, imgsz, imgsz), dtype=np.float32)
        sess.run(None, {input_name: dummy_tensor})
        print(f"ONNX model verified with dummy input shape (1, 3, {imgsz}, {imgsz}).")
    except Exception as exc:
        print(f"Note: ONNX runtime verification skipped: {exc}")

    return str(out_dest)


def main(args: list[str] | None = None) -> None:
    parsed = parse_args(args)
    export_yolo_to_onnx(
        model_name=parsed.model,
        output_path=parsed.output,
        imgsz=parsed.imgsz,
        opset=parsed.opset,
        simplify=bool(parsed.simplify),
    )


if __name__ == "__main__":
    main()
