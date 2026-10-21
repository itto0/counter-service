import json
import pytest
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

from tools.export_yolo import parse_args, export_yolo_to_onnx, main
from tools.train_carcass_detector import (
    parse_args as parse_train_args,
    extract_keyframes,
    auto_label_frame,
    create_dataset_structure,
    train_carcass_detector,
    evaluate_trained_model,
    format_metrics_summary,
    main as train_main,
)


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


def test_train_carcass_detector_cli_parse_args():
    args = parse_train_args([])
    assert args.video == "samples/Rekaman_2.MOV"
    assert args.frames == 60
    assert args.epochs == 15
    assert args.base_model == "yolov8n.pt"
    assert args.output == "models/carcass_yolov8n.onnx"
    assert args.work_dir == "dataset/carcass"
    assert args.imgsz == 640

    custom_args = parse_train_args([
        "--video", "custom/video.mp4",
        "--frames", "100",
        "--epochs", "25",
        "--base-model", "yolov8s.pt",
        "--output", "models/custom.onnx",
        "--work-dir", "custom/dataset",
        "--imgsz", "320",
    ])
    assert custom_args.video == "custom/video.mp4"
    assert custom_args.frames == 100
    assert custom_args.epochs == 25
    assert custom_args.base_model == "yolov8s.pt"
    assert custom_args.output == "models/custom.onnx"
    assert custom_args.work_dir == "custom/dataset"
    assert custom_args.imgsz == 320


def test_auto_label_frame():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    det1 = MagicMock(x1=100.0, y1=120.0, x2=200.0, y2=360.0, class_id=0)
    mock_detector = MagicMock()
    mock_detector.detect.return_value = [det1]

    labels = auto_label_frame(frame, mock_detector)
    assert len(labels) == 1
    cls_id, xc, yc, w, h = labels[0]
    assert cls_id == 0
    assert pytest.approx(xc, rel=1e-4) == 0.234375
    assert pytest.approx(yc, rel=1e-4) == 0.5
    assert pytest.approx(w, rel=1e-4) == 0.15625
    assert pytest.approx(h, rel=1e-4) == 0.5

    det_oob = MagicMock(x1=-100.0, y1=-50.0, x2=1000.0, y2=800.0, class_id=1)
    mock_detector.detect.return_value = [det_oob]
    labels_oob = auto_label_frame(frame, mock_detector)
    assert len(labels_oob) == 1
    cls_id, xc, yc, w, h = labels_oob[0]
    assert cls_id == 1
    assert 0.0 <= xc <= 1.0
    assert 0.0 <= yc <= 1.0
    assert 0.0 <= w <= 1.0
    assert 0.0 <= h <= 1.0

    mock_predict_detector = MagicMock(spec=["predict"])
    mock_predict_detector.predict.return_value = [(100.0, 100.0, 200.0, 200.0)]
    labels_fallback = auto_label_frame(frame, mock_predict_detector)
    assert len(labels_fallback) == 1
    assert labels_fallback[0][0] == 0


def test_create_dataset_structure(tmp_path):
    dataset_dir = tmp_path / "test_carcass_dataset"

    src_dir = tmp_path / "src"
    src_dir.mkdir()
    image_paths = []
    labels = []
    for i in range(5):
        img_file = src_dir / f"img_{i}.jpg"
        img_file.write_bytes(b"dummy")
        image_paths.append(str(img_file))
        labels.append([(0, 0.5, 0.5, 0.2, 0.4)])

    yaml_path = create_dataset_structure(image_paths, labels, str(dataset_dir), train_ratio=0.8)

    assert Path(yaml_path).exists()
    yaml_content = Path(yaml_path).read_text(encoding="utf-8")
    assert "train: images/train" in yaml_content
    assert "val: images/val" in yaml_content
    assert "0: chicken_carcass" in yaml_content

    train_imgs = list((dataset_dir / "images" / "train").glob("*.jpg"))
    val_imgs = list((dataset_dir / "images" / "val").glob("*.jpg"))
    assert len(train_imgs) == 4
    assert len(val_imgs) == 1

    train_lbls = list((dataset_dir / "labels" / "train").glob("*.txt"))
    val_lbls = list((dataset_dir / "labels" / "val").glob("*.txt"))
    assert len(train_lbls) == 4
    assert len(val_lbls) == 1

    sample_lbl = train_lbls[0].read_text(encoding="utf-8").strip()
    assert sample_lbl == "0 0.500000 0.500000 0.200000 0.400000"

    edge_dir = tmp_path / "edge_dataset"
    create_dataset_structure(image_paths[:2], labels[:2], str(edge_dir), train_ratio=0.9)
    assert len(list((edge_dir / "images" / "train").glob("*.jpg"))) == 1
    assert len(list((edge_dir / "images" / "val").glob("*.jpg"))) == 1


def test_extract_keyframes_mocked(tmp_path):
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = False
    with patch("cv2.VideoCapture", return_value=mock_cap):
        with pytest.raises(FileNotFoundError):
            extract_keyframes("nonexistent.mp4", str(tmp_path / "out"), num_frames=5)

    mock_cap_ok = MagicMock()
    mock_cap_ok.isOpened.return_value = True
    mock_cap_ok.get.return_value = 100.0
    dummy_frame = np.ones((50, 50, 3), dtype=np.uint8) * 255
    mock_cap_ok.read.return_value = (True, dummy_frame)

    out_frames_dir = tmp_path / "extracted_frames"
    with patch("cv2.VideoCapture", return_value=mock_cap_ok):
        paths = extract_keyframes("fake_video.mp4", str(out_frames_dir), num_frames=5)

        assert len(paths) == 5
        assert all(Path(p).exists() for p in paths)
        assert mock_cap_ok.set.call_count == 5
        mock_cap_ok.release.assert_called_once()


def test_format_metrics_summary():
    metrics = {"map50": 0.9421, "map50_95": 0.7615, "precision": 0.915, "recall": 0.8872}
    summary = format_metrics_summary(metrics)
    assert "YOLO Carcass Detection Accuracy Metrics" in summary
    assert "mAP@50       : 0.9421 (94.21%)" in summary
    assert "mAP@50-95    : 0.7615 (76.15%)" in summary
    assert "Precision    : 0.9150 (91.50%)" in summary
    assert "Recall       : 0.8872 (88.72%)" in summary


def test_format_metrics_summary_empty():
    summary = format_metrics_summary({})
    assert "mAP@50       : 0.0000 (0.00%)" in summary
    assert "mAP@50-95    : 0.0000 (0.00%)" in summary
    assert "Precision    : 0.0000 (0.00%)" in summary
    assert "Recall       : 0.0000 (0.00%)" in summary


def test_format_metrics_summary_none_values():
    summary = format_metrics_summary(
        {"map50": None, "map50_95": None, "precision": None, "recall": None}
    )
    assert "mAP@50       : 0.0000 (0.00%)" in summary
    assert "mAP@50-95    : 0.0000 (0.00%)" in summary
    assert "Precision    : 0.0000 (0.00%)" in summary
    assert "Recall       : 0.0000 (0.00%)" in summary


def test_train_carcass_detector_pipeline_mocked(tmp_path):
    work_dir = tmp_path / "train_work"
    output_model = str(tmp_path / "models" / "carcass.onnx")

    dummy_images = [str(tmp_path / "img1.jpg"), str(tmp_path / "img2.jpg")]
    for p in dummy_images:
        Path(p).write_bytes(b"dummy")

    mock_yolo_cls = MagicMock()
    mock_yolo_inst = MagicMock()
    mock_yolo_cls.return_value = mock_yolo_inst

    mock_best_pt = tmp_path / "best.pt"
    mock_best_pt.write_text("weights")
    mock_yolo_inst.trainer.best = str(mock_best_pt)

    expected_metrics = {
        "map50": 0.9421,
        "map50_95": 0.7615,
        "precision": 0.9150,
        "recall": 0.8872,
    }

    with patch("tools.train_carcass_detector.extract_keyframes", return_value=dummy_images) as mock_extract, \
         patch("cv2.imread", return_value=np.zeros((100, 100, 3), dtype=np.uint8)), \
         patch("tools.train_carcass_detector.auto_label_frame", return_value=[(0, 0.5, 0.5, 0.2, 0.2)]) as mock_label, \
         patch("tools.train_carcass_detector.create_dataset_structure", return_value=str(work_dir / "dataset.yaml")) as mock_create_ds, \
         patch.dict("sys.modules", {"ultralytics": MagicMock(YOLO=mock_yolo_cls)}), \
         patch("tools.train_carcass_detector.evaluate_trained_model", return_value=expected_metrics) as mock_evaluate, \
         patch("tools.train_carcass_detector.export_yolo_to_onnx", return_value=output_model) as mock_export:

        res = train_carcass_detector(
            video_path="sample.mov",
            num_frames=2,
            epochs=3,
            base_model="yolov8n.pt",
            output_model=output_model,
            work_dir=str(work_dir),
            imgsz=320,
        )

        assert res == {
            "model_path": output_model,
            "metrics": expected_metrics,
        }
        mock_extract.assert_called_once_with("sample.mov", str(work_dir / "raw_frames"), num_frames=2)
        assert mock_label.call_count == 2
        mock_create_ds.assert_called_once()
        mock_yolo_cls.assert_called_once_with("yolov8n.pt")
        mock_yolo_inst.train.assert_called_once_with(
            data=str(work_dir / "dataset.yaml"),
            epochs=3,
            imgsz=320,
            project=str(work_dir / "runs"),
            name="train",
        )
        mock_evaluate.assert_called_once_with(
            mock_best_pt,
            str(work_dir / "dataset.yaml"),
            imgsz=320,
        )
        mock_export.assert_called_once_with(
            model_name=str(mock_best_pt),
            output_path=output_model,
            imgsz=320,
        )

        # Check val_metrics.json persistence
        val_json_path = work_dir / "val_metrics.json"
        assert val_json_path.exists()
        val_data = json.loads(val_json_path.read_text(encoding="utf-8"))
        assert val_data["best_checkpoint"] == str(mock_best_pt)
        assert val_data["exported_onnx"] == output_model
        assert val_data["metrics"] == expected_metrics

        # Check <output_stem>_metrics.json persistence
        out_json_path = Path(output_model).parent / "carcass_metrics.json"
        assert out_json_path.exists()
        out_data = json.loads(out_json_path.read_text(encoding="utf-8"))
        assert out_data["best_checkpoint"] == str(mock_best_pt)
        assert out_data["exported_onnx"] == output_model
        assert out_data["metrics"] == expected_metrics


def test_train_carcass_detector_missing_ultralytics(tmp_path):
    dummy_img = tmp_path / "frame.jpg"
    dummy_img.write_bytes(b"dummy")

    with patch("tools.train_carcass_detector.extract_keyframes", return_value=[str(dummy_img)]), \
         patch("cv2.imread", return_value=np.zeros((50, 50, 3), dtype=np.uint8)), \
         patch("tools.train_carcass_detector.auto_label_frame", return_value=[]), \
         patch("tools.train_carcass_detector.create_dataset_structure", return_value=str(tmp_path / "dataset.yaml")), \
         patch.dict("sys.modules", {"ultralytics": None}):
        with pytest.raises(ImportError) as exc_info:
            train_carcass_detector(
                video_path="sample.mov",
                num_frames=1,
                epochs=1,
                work_dir=str(tmp_path / "work"),
            )
        assert "ultralytics is required" in str(exc_info.value)


def test_train_carcass_detector_lazy_import_isolation():
    with patch.dict("sys.modules"):
        sys.modules.pop("ultralytics", None)
        sys.modules.pop("tools.train_carcass_detector", None)
        import tools.train_carcass_detector  # noqa: F401
        assert "ultralytics" not in sys.modules


def test_train_carcass_detector_main_cli():
    test_args = [
        "tools/train_carcass_detector.py",
        "--video", "input.mp4",
        "--frames", "10",
        "--epochs", "2",
        "--base-model", "base.pt",
        "--output", "out.onnx",
        "--work-dir", "work",
        "--imgsz", "320",
    ]
    with patch.object(sys, "argv", test_args):
        with patch("tools.train_carcass_detector.train_carcass_detector") as mock_train:
            mock_train.return_value = {
                "model_path": "out.onnx",
                "metrics": {"map50": 0.9, "map50_95": 0.7, "precision": 0.85, "recall": 0.8},
            }
            train_main()
            mock_train.assert_called_once_with(
                video_path="input.mp4",
                num_frames=10,
                epochs=2,
                base_model="base.pt",
                output_model="out.onnx",
                work_dir="work",
                imgsz=320,
            )


def test_evaluate_trained_model_success():
    from tools.train_carcass_detector import evaluate_trained_model

    mock_box = MagicMock()
    mock_box.map50 = 0.9421
    mock_box.map = 0.7615
    mock_box.mp = 0.9150
    mock_box.mr = 0.8872

    mock_val_res = MagicMock()
    mock_val_res.box = mock_box

    mock_yolo_instance = MagicMock()
    mock_yolo_instance.val.return_value = mock_val_res
    mock_yolo_cls = MagicMock(return_value=mock_yolo_instance)

    with patch.dict("sys.modules", {"ultralytics": MagicMock(YOLO=mock_yolo_cls)}):
        metrics = evaluate_trained_model("best.pt", "dataset.yaml", imgsz=640)

    assert metrics == {
        "map50": 0.9421,
        "map50_95": 0.7615,
        "precision": 0.9150,
        "recall": 0.8872,
    }
    mock_yolo_cls.assert_called_once_with("best.pt")
    mock_yolo_instance.val.assert_called_once_with(data="dataset.yaml", imgsz=640, split="val")


def test_evaluate_trained_model_missing_attrs():
    from tools.train_carcass_detector import evaluate_trained_model

    mock_val_res = MagicMock(spec=[])
    mock_yolo_instance = MagicMock()
    mock_yolo_instance.val.return_value = mock_val_res
    mock_yolo_cls = MagicMock(return_value=mock_yolo_instance)

    with patch.dict("sys.modules", {"ultralytics": MagicMock(YOLO=mock_yolo_cls)}):
        metrics = evaluate_trained_model("best.pt", "dataset.yaml", imgsz=640)

    assert metrics == {
        "map50": 0.0,
        "map50_95": 0.0,
        "precision": 0.0,
        "recall": 0.0,
    }


def test_evaluate_trained_model_missing_ultralytics():
    from tools.train_carcass_detector import evaluate_trained_model

    with patch.dict("sys.modules", {"ultralytics": None}):
        with pytest.raises(ImportError) as exc_info:
            evaluate_trained_model("best.pt", "dataset.yaml")
        assert "ultralytics is required" in str(exc_info.value)


