import pytest
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

from tools.export_yolo import parse_args, export_yolo_to_onnx, main


def test_export_yolo_arg_parser():
    args = parse_args(["--model", "yolov8n.pt", "--output", "models/custom.onnx", "--imgsz", "640", "--opset", "17"])
    assert args.model == "yolov8n.pt"
    assert args.output == "models/custom.onnx"
    assert args.imgsz == 640
    assert args.opset == 17
    assert args.simplify is True

    args_no_simp = parse_args(["--no-simplify"])
    assert args_no_simp.simplify is False

    args_simp = parse_args(["--simplify"])
    assert args_simp.simplify is True

    args_default = parse_args([])
    assert args_default.model == "yolov8n.pt"
    assert args_default.output == "models/yolov8n.onnx"
    assert args_default.imgsz == 640
    assert args_default.opset == 17
    assert args_default.simplify is True

    args_no_simplify = parse_args(["--no-simplify"])
    assert args_no_simplify.simplify is False


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
        mock_yolo.assert_called_once_with("yolov8n.pt")
        mock_instance.export.assert_called_once_with(
            format="onnx", imgsz=640, opset=17, simplify=True
        )


def test_export_yolo_missing_ultralytics(tmp_path):
    with patch.dict("sys.modules", {"ultralytics": None}):
        with pytest.raises(ImportError) as exc_info:
            export_yolo_to_onnx("yolov8n.pt", str(tmp_path / "out.onnx"))
        assert "ultralytics is required" in str(exc_info.value)


def test_export_yolo_same_dest(tmp_path):
    mock_yolo = MagicMock()
    mock_instance = MagicMock()
    mock_yolo.return_value = mock_instance
    target = tmp_path / "exported.onnx"
    target.write_text("dummy")
    mock_instance.export.return_value = str(target)

    with patch.dict("sys.modules", {"ultralytics": MagicMock(YOLO=mock_yolo)}):
        out_path = export_yolo_to_onnx(
            model_name="yolov8n.pt",
            output_path=str(target),
            imgsz=320,
            opset=12,
            simplify=False,
        )
        assert out_path == str(target)
        mock_instance.export.assert_called_once_with(
            format="onnx", imgsz=320, opset=12, simplify=False
        )


def test_export_yolo_main(tmp_path):
    target = tmp_path / "cli_out.onnx"
    test_args = ["tools/export_yolo.py", "--model", "yolov8s.pt", "--output", str(target), "--no-simplify"]
    with patch.object(sys, "argv", test_args):
        with patch("tools.export_yolo.export_yolo_to_onnx") as mock_export:
            main()
            mock_export.assert_called_once_with(
                model_name="yolov8s.pt",
                output_path=str(target),
                imgsz=640,
                opset=17,
                simplify=False,
            )
