"""Tests for Detector abstractions and Mock/ONNX implementations."""

from dataclasses import FrozenInstanceError
import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.detection.detector import (
    Detection,
    Detector,
    MockDetector,
    ONNXDetector,
    ShackleDetector,
)


def test_cannot_instantiate_abstract_detector():
    """Verify Detector ABC cannot be instantiated directly."""
    with pytest.raises(TypeError):
        Detector()  # type: ignore[abstract]


def test_detection_dataclass_properties():
    """Verify Detection dataclass fields, frozen immutability, and anchor_bottom_center."""
    det = Detection(
        x1=100.0,
        y1=50.0,
        x2=200.0,
        y2=150.0,
        confidence=0.92,
        class_id=0,
        class_name="chicken",
    )

    assert det.x1 == 100.0
    assert det.y1 == 50.0
    assert det.x2 == 200.0
    assert det.y2 == 150.0
    assert det.confidence == 0.92
    assert det.class_id == 0
    assert det.class_name == "chicken"

    # Bottom-center anchor point: ((x1 + x2) / 2, y2)
    assert det.anchor_bottom_center == (150.0, 150.0)

    # Immutability check
    with pytest.raises(FrozenInstanceError):
        det.confidence = 0.5  # type: ignore[misc]

    # Width and height properties
    assert det.width == 100.0
    assert det.height == 100.0


def test_mock_detector_detects_moving_targets():
    """Verify MockDetector identifies simulated chicken blob correctly."""
    detector = MockDetector(min_confidence=0.8)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    # Paint white blob simulating chicken
    frame[100:160, 200:250] = 220

    detections = detector.predict(frame)
    assert len(detections) >= 1
    det = detections[0]
    assert isinstance(det, Detection)
    assert det.confidence >= 0.8
    assert det.class_id == 0
    assert det.class_name == "chicken"
    assert det.x2 > det.x1
    assert det.y2 > det.y1
    assert det.x1 >= 195.0 and det.x2 <= 255.0
    assert det.y1 >= 95.0 and det.y2 <= 165.0
    # Anchor point should be at bottom-center
    anchor_x, anchor_y = det.anchor_bottom_center
    assert det.x1 <= anchor_x <= det.x2
    assert anchor_y == det.y2


def test_mock_detector_empty_or_dark_frame_returns_empty():
    """Verify MockDetector returns empty list on dark or low-intensity frames."""
    detector = MockDetector(min_confidence=0.85)

    # All black frame
    black_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    assert detector.predict(black_frame) == []

    # Low-contrast background frame below threshold 150
    dim_frame = np.full((480, 640, 3), 100, dtype=np.uint8)
    assert detector.predict(dim_frame) == []


def test_mock_detector_filters_small_blobs():
    """Verify MockDetector filters out noise blobs smaller than minimum dimensions."""
    detector = MockDetector(min_confidence=0.85, min_width=10.0, min_height=10.0)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    # 5x5 tiny bright spot (noise)
    frame[50:55, 50:55] = 255

    detections = detector.predict(frame)
    assert detections == []


def test_mock_detector_detects_multiple_blobs():
    """Verify MockDetector identifies multiple separate chicken blobs in one frame."""
    detector = MockDetector(min_confidence=0.85)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    # Blob 1 (left)
    frame[50:110, 100:160] = 230
    # Blob 2 (right)
    frame[250:310, 400:460] = 230

    detections = detector.predict(frame)
    assert len(detections) == 2
    # Check that both distinct regions are represented
    x_centers = [(d.x1 + d.x2) / 2.0 for d in detections]
    assert any(x < 250 for x in x_centers)
    assert any(x > 300 for x in x_centers)


def test_onnx_detector_raises_when_onnxruntime_missing():
    """Verify ONNXDetector raises descriptive RuntimeError when onnxruntime is unavailable."""
    with patch.dict(sys.modules, {"onnxruntime": None}):
        with pytest.raises(RuntimeError, match="onnxruntime"):
            ONNXDetector("models/yolo26n.onnx")


def test_onnx_detector_mocked_end_to_end_yolo26():
    """Verify ONNXDetector with mocked YOLO26 (NMS-free [N, 6]) model output."""
    mock_ort = MagicMock()
    mock_session = MagicMock()

    # Mock model input metadata
    mock_input = MagicMock()
    mock_input.name = "images"
    mock_input.shape = [1, 3, 640, 640]
    mock_session.get_inputs.return_value = [mock_input]

    # Mock model output: 2 detections in 640x640 space:
    # 1. Chicken with conf 0.9 (box [100, 100, 200, 200], class 0)
    # 2. Low-confidence detection with conf 0.2 (box [300, 300, 400, 400], class 0)
    mock_output_data = np.array(
        [
            [
                [100.0, 100.0, 200.0, 200.0, 0.90, 0.0],
                [300.0, 300.0, 400.0, 400.0, 0.20, 0.0],
            ]
        ],
        dtype=np.float32,
    )
    mock_session.run.return_value = [mock_output_data]
    mock_ort.InferenceSession.return_value = mock_session

    with patch.dict(sys.modules, {"onnxruntime": mock_ort}):
        detector = ONNXDetector(
            model_path="models/yolo26n.onnx",
            conf_threshold=0.5,
            iou_threshold=0.45,
            class_names=["chicken"],
        )

        test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        detections = detector.predict(test_frame)

        # Only high confidence (0.90) detection should survive threshold (0.50)
        assert len(detections) == 1
        det = detections[0]
        assert det.class_id == 0
        assert det.class_name == "chicken"
        assert pytest.approx(det.confidence, 0.01) == 0.90
        # Check coordinates mapped back from 640x640 letterbox to 640x480 original
        # In 640x640 input with 640x480 frame: ratio = 1.0, pad_w = 0, pad_h = 80
        # x: (100 - 0) / 1.0 = 100.0
        # y: (100 - 80) / 1.0 = 20.0
        assert pytest.approx(det.x1, 1.0) == 100.0
        assert pytest.approx(det.y1, 1.0) == 20.0
        assert pytest.approx(det.x2, 1.0) == 200.0
        assert pytest.approx(det.y2, 1.0) == 120.0
        assert det.anchor_bottom_center == pytest.approx((150.0, 120.0), abs=1.0)


def test_onnx_detector_mocked_raw_yolo_with_nms():
    """Verify ONNXDetector handles raw YOLO [1, 4+classes, N] outputs with NMS."""
    mock_ort = MagicMock()
    mock_session = MagicMock()

    mock_input = MagicMock()
    mock_input.name = "images"
    mock_input.shape = [1, 3, 640, 640]
    mock_session.get_inputs.return_value = [mock_input]

    # YOLO raw format: [batch, 4 + num_classes, num_anchors]
    # Here 1 class, 3 candidate boxes:
    # Box 1: [cx=150, cy=150, w=100, h=100, score=0.85] -> x1=100, y1=100, x2=200, y2=200
    # Box 2: [cx=152, cy=152, w=98, h=98, score=0.75] -> overlapping Box 1 (should be suppressed by NMS)
    # Box 3: [cx=450, cy=350, w=80, h=80, score=0.88] -> distant box
    raw_preds = np.zeros((1, 5, 3), dtype=np.float32)
    # Box 1
    raw_preds[0, :, 0] = [150.0, 150.0, 100.0, 100.0, 0.85]
    # Box 2 (overlap)
    raw_preds[0, :, 1] = [152.0, 152.0, 98.0, 98.0, 0.75]
    # Box 3
    raw_preds[0, :, 2] = [450.0, 350.0, 80.0, 80.0, 0.88]

    mock_session.run.return_value = [raw_preds]
    mock_ort.InferenceSession.return_value = mock_session

    with patch.dict(sys.modules, {"onnxruntime": mock_ort}):
        detector = ONNXDetector(
            model_path="models/yolo11n.onnx",
            conf_threshold=0.5,
            iou_threshold=0.45,
            class_names=["chicken"],
        )

        test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        detections = detector.predict(test_frame)

        # Box 2 should have been suppressed by NMS against Box 1
        assert len(detections) == 2
        confidences = sorted([d.confidence for d in detections], reverse=True)
        assert pytest.approx(confidences[0], 0.01) == 0.88
        assert pytest.approx(confidences[1], 0.01) == 0.85


def test_mock_detector_edge_cases():
    """Verify MockDetector handles None, empty, and 2D grayscale frames cleanly."""
    detector = MockDetector()

    assert detector.predict(None) == []  # type: ignore[arg-type]
    assert detector.predict(np.array([])) == []

    # 2D grayscale frame with bright blob
    gray_frame = np.zeros((200, 200), dtype=np.uint8)
    gray_frame[50:80, 50:80] = 200
    detections = detector.predict(gray_frame)
    assert len(detections) == 1
    assert detections[0].class_name == "chicken"


def test_onnx_detector_edge_cases_and_options():
    """Verify ONNXDetector handles empty frames, custom providers, and dictionary class_names."""
    mock_ort = MagicMock()
    mock_session = MagicMock()
    mock_input = MagicMock()
    mock_input.name = "images"
    mock_input.shape = [1, 3, 320, 320]
    mock_session.get_inputs.return_value = [mock_input]
    mock_ort.InferenceSession.return_value = mock_session
    mock_ort.get_available_providers.return_value = ["CPUExecutionProvider"]

    with patch.dict(sys.modules, {"onnxruntime": mock_ort}):
        detector = ONNXDetector(
            model_path="models/custom.onnx",
            conf_threshold=0.6,
            providers=["CPUExecutionProvider"],
            class_names={0: "broiler", 1: "layer"},
        )

        # Empty and None frame handling
        assert detector.predict(None) == []  # type: ignore[arg-type]
        assert detector.predict(np.array([])) == []

        # Empty session output
        mock_session.run.return_value = []
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        assert detector.predict(frame) == []


def test_shackle_detector_empty_and_blank():
    """Verify ShackleDetector handles None, empty, and flat zero-gradient frames."""
    detector = ShackleDetector()
    assert detector.predict(None) == []  # type: ignore[arg-type]
    assert detector.predict(np.array([])) == []

    flat_frame = np.full((1080, 1920, 3), 128, dtype=np.uint8)
    assert detector.predict(flat_frame) == []


def test_shackle_detector_detects_spaced_shackles():
    """Verify ShackleDetector locates vertical bars and frames suspended objects."""
    detector = ShackleDetector(
        shackle_y_ratio=(0.20, 0.50),
        min_spacing=50.0,
        chicken_width=80.0,
        chicken_height=300.0,
        chicken_y_top=250.0,
        occupancy_check=False,
    )
    frame = np.zeros((600, 800, 3), dtype=np.uint8)
    # Draw three distinct vertical shackle bars in the shackle band y in [120, 300]
    frame[150:280, 195:205] = 255
    frame[150:280, 395:405] = 255
    frame[150:280, 595:605] = 255

    detections = detector.predict(frame)
    assert len(detections) == 3
    for det in detections:
        assert isinstance(det, Detection)
        assert det.class_name == "chicken"
        assert det.confidence == 0.90
        assert det.y1 == 250.0
        assert det.y2 == 550.0

    centers = [(d.x1 + d.x2) / 2.0 for d in detections]
    assert pytest.approx(centers[0], abs=15) == 200.0
    assert pytest.approx(centers[1], abs=15) == 400.0
    assert pytest.approx(centers[2], abs=15) == 600.0


def test_mock_detector_area_and_morph_filters():
    """Verify MockDetector respects min_area, max_area, and morph_open."""
    detector = MockDetector(
        min_confidence=0.85,
        min_width=5.0,
        min_height=5.0,
        min_area=100.0,
        max_area=5000.0,
        morph_open=True,
    )
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    # Small speckle (area 16) - below min_area
    frame[50:54, 50:54] = 255
    # Medium blob (area 900) - valid
    frame[100:130, 100:130] = 255
    # Huge background blob (area 10,000) - exceeds max_area
    frame[200:300, 200:300] = 255

    detections = detector.predict(frame)
    assert len(detections) == 1
    det = detections[0]
    assert 95.0 <= det.x1 <= 105.0
    assert 125.0 <= det.x2 <= 135.0


def test_shackle_detector_rejects_empty_shackle():
    """Verify ShackleDetector rejects empty shackles where no chicken carcass is present."""
    detector = ShackleDetector(
        shackle_y_ratio=(0.20, 0.50),
        min_spacing=50.0,
        chicken_width=80.0,
        chicken_height=300.0,
        chicken_y_top=250.0,
        occupancy_check=True,
    )
    frame = np.zeros((600, 800, 3), dtype=np.uint8)
    # Shackle 1 at cx=200 has a chicken carcass below it
    frame[150:280, 195:205] = 255
    frame[370:440, 197:203] = 220  # carcass presence in [0.62*h, 0.74*h]

    # Shackle 2 at cx=400 is EMPTY (only the metal bar, zero carcass below)
    frame[150:280, 395:405] = 255

    # Shackle 3 at cx=600 has a chicken carcass below it
    frame[150:280, 595:605] = 255
    frame[370:440, 597:603] = 220

    detections = detector.predict(frame)
    # Only the 2 occupied shackles should be detected; empty shackle at cx=400 must be rejected
    assert len(detections) == 2
    centers = [(d.x1 + d.x2) / 2.0 for d in detections]
    assert pytest.approx(centers[0], abs=15) == 200.0
    assert pytest.approx(centers[1], abs=15) == 600.0


def test_shackle_detector_occupancy_check_disabled_detects_all():
    """Verify ShackleDetector with occupancy_check=False detects all shackle brackets."""
    detector = ShackleDetector(
        shackle_y_ratio=(0.20, 0.50),
        min_spacing=50.0,
        chicken_width=80.0,
        chicken_height=300.0,
        chicken_y_top=250.0,
        occupancy_check=False,
    )
    frame = np.zeros((600, 800, 3), dtype=np.uint8)
    frame[150:280, 195:205] = 255
    frame[150:280, 395:405] = 255  # Empty shackle

    detections = detector.predict(frame)
    assert len(detections) == 2


def test_shackle_detector_video_frame_rejects_empty_shackle():
    """Verify ShackleDetector on Rekaman_2.MOV frame 3160 suppresses empty shackle at cx ~ 1322."""
    import os
    import cv2

    video_path = "samples/Rekaman_2.MOV"
    if not os.path.exists(video_path):
        pytest.skip(f"Video file {video_path} not found")

    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 3160)
    ret, frame = cap.read()
    cap.release()
    assert ret and frame is not None

    detector = ShackleDetector(occupancy_check=True)
    detections = detector.predict(frame)

    # In frame 3160, the empty shackle bracket is at cx ~ 1322
    # Verify no detection has center in range [1300, 1345]
    empty_shackle_dets = [
        d for d in detections if 1300.0 <= (d.x1 + d.x2) / 2.0 <= 1345.0
    ]
    assert len(empty_shackle_dets) == 0

    # Ensure occupied chickens on both sides are still detected
    occupied_left = [
        d for d in detections if 1200.0 <= (d.x1 + d.x2) / 2.0 <= 1245.0
    ]
    occupied_right = [
        d for d in detections if 1420.0 <= (d.x1 + d.x2) / 2.0 <= 1460.0
    ]
    assert len(occupied_left) == 1
    assert len(occupied_right) == 1


def test_shackle_detector_noise_rejection_and_bracket_clustering():
    """Verify ShackleDetector groups paired bracket bars accurately without cascading into noise."""
    detector = ShackleDetector(
        shackle_y_ratio=(0.20, 0.50),
        min_spacing=55.0,
        chicken_width=80.0,
        chicken_height=300.0,
        chicken_y_top=250.0,
        occupancy_check=False,
    )
    frame = np.zeros((600, 800, 3), dtype=np.uint8)

    # Shackle bracket 1 at cx=300: two vertical bars at 292 and 308 (intensity 255)
    frame[150:280, 290:294] = 255
    frame[150:280, 306:310] = 255

    # Faint noise reflection nearby at x=345 (intensity 50, below 40% of max energy)
    frame[150:280, 344:346] = 50

    # Shackle bracket 2 at cx=500: two vertical bars at 492 and 508
    frame[150:280, 490:494] = 255
    frame[150:280, 506:510] = 255

    detections = detector.predict(frame)
    assert len(detections) == 2
    centers = [(d.x1 + d.x2) / 2.0 for d in detections]
    assert pytest.approx(centers[0], abs=5) == 300.0
    assert pytest.approx(centers[1], abs=5) == 500.0


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


def test_letterbox_image_square_frame_and_custom_color():
    """Verify letterbox on already-square frame with custom padding color."""
    from src.detection.detector import letterbox_image

    frame = np.ones((500, 500, 3), dtype=np.uint8) * 50
    target_shape = (640, 640)
    padded, ratio, (pad_w, pad_h) = letterbox_image(
        frame, target_shape=target_shape, pad_color=(0, 255, 0)
    )

    assert padded.shape == (640, 640, 3)
    assert pytest.approx(ratio, rel=1e-3) == 640 / 500
    assert pad_w == 0
    assert pad_h == 0


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


def test_onnx_detector_input_size_and_backward_compatibility():
    """Verify input_size and backward-compatible input_width/input_height access and mutation."""
    from src.detection.detector import ONNXDetector

    # Default initialization
    detector1 = ONNXDetector(model_path="dummy.onnx")
    assert detector1.input_size == (640, 640)
    assert detector1.input_width == 640
    assert detector1.input_height == 640
    assert detector1.detect == detector1.predict

    # Custom input_size tuple
    detector2 = ONNXDetector(model_path="dummy.onnx", input_size=(1280, 720), classes=[0])
    assert detector2.input_size == (1280, 720)
    assert detector2.input_width == 1280
    assert detector2.input_height == 720
    assert detector2.classes == {0}

    # Backward compatibility with input_width and input_height
    detector3 = ONNXDetector(model_path="dummy.onnx", input_width=320, input_height=240)
    assert detector3.input_size == (320, 240)
    assert detector3.input_width == 320
    assert detector3.input_height == 240

    # Mutability via input_size property
    detector3.input_size = (640, 480)
    assert detector3.input_width == 640
    assert detector3.input_height == 480
    assert detector3.input_size == (640, 480)

    # Session property aliases
    assert detector1.session is None
    assert detector1._session is None
    detector1.session = "mock_session"
    assert detector1._session == "mock_session"
    assert detector1.session == "mock_session"

    assert detector1.input_name == ""
    detector1.input_name = "test_input"
    assert detector1._input_name == "test_input"
    assert detector1.input_name == "test_input"


def test_onnx_detector_classes_filter_and_portrait_letterbox():
    """Verify classes filter drops unselected classes and portrait frame inverse letterbox is accurate."""
    from unittest.mock import MagicMock
    from src.detection.detector import ONNXDetector

    # Only accept class 0
    detector = ONNXDetector(
        model_path="dummy.onnx",
        input_size=(640, 640),
        classes=[0],
        class_names={0: "chicken", 1: "other"},
        conf_threshold=0.3,
    )
    detector._session = MagicMock()

    # Portrait 1080x1920 frame
    frame = np.zeros((1920, 1080, 3), dtype=np.uint8)
    _ = detector.preprocess(frame)
    # Ratio = 640 / 1920 = 1/3; pad_w = (640 - 360)/2 = 140; pad_h = 0
    assert detector._pad_w == 140
    assert detector._pad_h == 0

    # Raw YOLO output with 2 classes: [1, 6, 2] -> [cx, cy, w, h, cls0_score, cls1_score]
    # Box 1: class 0 with 0.8 score (should pass)
    # Box 2: class 1 with 0.9 score (should be filtered out by classes=[0])
    raw_output = np.array(
        [
            [
                [320.0, 320.0],  # cx
                [320.0, 320.0],  # cy
                [60.0, 60.0],    # w
                [60.0, 60.0],    # h
                [0.85, 0.05],    # cls 0
                [0.10, 0.95],    # cls 1
            ]
        ],
        dtype=np.float32,
    )

    detections = detector.postprocess([raw_output])
    assert len(detections) == 1
    assert detections[0].class_id == 0
    assert detections[0].class_name == "chicken"
    # Inverse mapping for Box 1:
    # letterbox: x1 = 320 - 30 = 290, x2 = 320 + 30 = 350
    # orig_x1 = (290 - 140) / (1/3) = 150 * 3 = 450
    # orig_x2 = (350 - 140) / (1/3) = 210 * 3 = 630
    assert pytest.approx(detections[0].x1, abs=1.0) == 450.0
    assert pytest.approx(detections[0].x2, abs=1.0) == 630.0


def test_onnx_detector_cpu_thread_options(monkeypatch):
    """Verify ONNXDetector configures SessionOptions for edge CPU efficiency."""
    from unittest.mock import MagicMock
    from src.detection.detector import ONNXDetector

    mock_ort = MagicMock()
    mock_session_options = MagicMock()
    mock_ort.SessionOptions.return_value = mock_session_options
    monkeypatch.setattr("src.detection.detector.ort", mock_ort, raising=False)
    monkeypatch.setitem(sys.modules, "onnxruntime", mock_ort)

    detector = ONNXDetector(model_path="dummy.onnx")
    detector._load_model()

    assert mock_ort.SessionOptions.called
    assert mock_session_options.intra_op_num_threads > 0
    assert (
        mock_session_options.graph_optimization_level
        == mock_ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    )
    # Check sess_options was passed into InferenceSession
    call_args = mock_ort.InferenceSession.call_args
    assert call_args is not None
    assert call_args.kwargs.get("sess_options") == mock_session_options


def test_load_model_accuracy_pct(tmp_path):
    """Verify load_model_accuracy_pct discovers companion metrics or falls back to None."""
    from src.detection.detector import load_model_accuracy_pct

    # Non-existent path returns None
    assert load_model_accuracy_pct(None) is None
    assert load_model_accuracy_pct("") is None
    assert load_model_accuracy_pct("non_existent_model.onnx") is None

    # Model with companion carcass_yolov8n_metrics.json
    model_file = tmp_path / "carcass_yolov8n.onnx"
    model_file.touch()

    metrics_file = tmp_path / "carcass_yolov8n_metrics.json"
    metrics_file.write_text('{"metrics": {"map50": 0.9421}}', encoding="utf-8")

    # Supports str path
    assert load_model_accuracy_pct(str(model_file)) == 94.21
    # Supports Path object directly
    assert load_model_accuracy_pct(model_file) == 94.21

    # Model with companion val_metrics.json fallback
    model_file_2 = tmp_path / "model2.onnx"
    model_file_2.touch()
    metrics_file_2 = tmp_path / "val_metrics.json"
    metrics_file_2.write_text('{"metrics": {"map50": 0.885}}', encoding="utf-8")

    assert load_model_accuracy_pct(str(model_file_2)) == 88.50

    # Flat JSON without top-level 'metrics' key and fallback to 'map'
    model_file_3 = tmp_path / "model3.onnx"
    model_file_3.touch()
    (tmp_path / "model3_metrics.json").write_text('{"map": 0.9123}', encoding="utf-8")
    assert load_model_accuracy_pct(model_file_3) == 91.23

    # Already percentage value in [0, 100]
    model_file_4 = tmp_path / "model4.onnx"
    model_file_4.touch()
    (tmp_path / "model4_metrics.json").write_text('{"metrics": {"map50": 92.4}}', encoding="utf-8")
    assert load_model_accuracy_pct(model_file_4) == 92.40

    # Malformed / empty json returns None gracefully when no valid fallback exists
    corrupt_dir = tmp_path / "corrupt_sub"
    corrupt_dir.mkdir()
    corrupt_file = corrupt_dir / "corrupt.onnx"
    corrupt_file.touch()
    (corrupt_dir / "corrupt_metrics.json").write_text("invalid json", encoding="utf-8")
    assert load_model_accuracy_pct(str(corrupt_file)) is None

    # Out of bounds clamping: high values clamped to 100.0
    model_high_1 = tmp_path / "model_high_1.onnx"
    model_high_1.touch()
    (tmp_path / "model_high_1_metrics.json").write_text('{"metrics": {"map50": 150.0}}', encoding="utf-8")
    assert load_model_accuracy_pct(model_high_1) == 100.0

    model_high_2 = tmp_path / "model_high_2.onnx"
    model_high_2.touch()
    (tmp_path / "model_high_2_metrics.json").write_text('{"metrics": {"map50": 120.0}}', encoding="utf-8")
    assert load_model_accuracy_pct(model_high_2) == 100.0

    # Out of bounds clamping: low values clamped to 0.0
    model_low_1 = tmp_path / "model_low_1.onnx"
    model_low_1.touch()
    (tmp_path / "model_low_1_metrics.json").write_text('{"metrics": {"map50": -10.0}}', encoding="utf-8")
    assert load_model_accuracy_pct(model_low_1) == 0.0

    model_low_2 = tmp_path / "model_low_2.onnx"
    model_low_2.touch()
    (tmp_path / "model_low_2_metrics.json").write_text('{"metrics": {"map50": -5.0}}', encoding="utf-8")
    assert load_model_accuracy_pct(model_low_2) == 0.0

    model_low_3 = tmp_path / "model_low_3.onnx"
    model_low_3.touch()
    (tmp_path / "model_low_3_metrics.json").write_text('{"metrics": {"map50": -0.2}}', encoding="utf-8")
    assert load_model_accuracy_pct(model_low_3) == 0.0

    # Boolean values are safely ignored (bool is subclass of int in Python)
    bool_dir = tmp_path / "bool_tests"
    bool_dir.mkdir()
    model_bool_1 = bool_dir / "model_bool_1.onnx"
    model_bool_1.touch()
    (bool_dir / "model_bool_1_metrics.json").write_text('{"metrics": {"map50": true}}', encoding="utf-8")
    assert load_model_accuracy_pct(model_bool_1) is None

    model_bool_2 = bool_dir / "model_bool_2.onnx"
    model_bool_2.touch()
    (bool_dir / "model_bool_2_metrics.json").write_text('{"metrics": {"map50": false}}', encoding="utf-8")
    assert load_model_accuracy_pct(model_bool_2) is None

    # Boolean map50 ignored with fallback to valid map
    model_bool_fallback = bool_dir / "model_bool_fallback.onnx"
    model_bool_fallback.touch()
    (bool_dir / "model_bool_fallback_metrics.json").write_text(
        '{"metrics": {"map50": true, "map": 0.85}}', encoding="utf-8"
    )
    assert load_model_accuracy_pct(model_bool_fallback) == 85.0

    # NaN / Inf ignored
    nan_dir = tmp_path / "nan_tests"
    nan_dir.mkdir()
    model_nan = nan_dir / "model_nan.onnx"
    model_nan.touch()
    (nan_dir / "model_nan_metrics.json").write_text('{"metrics": {"map50": NaN}}', encoding="utf-8")
    assert load_model_accuracy_pct(model_nan) is None

    # Corrupt specific file falls back to val_metrics.json
    model_corrupt_fb = tmp_path / "model_cfb.onnx"
    model_corrupt_fb.touch()
    (tmp_path / "model_cfb_metrics.json").write_text("corrupted json", encoding="utf-8")
    (tmp_path / "val_metrics.json").write_text('{"metrics": {"map50": 0.892}}', encoding="utf-8")
    assert load_model_accuracy_pct(model_corrupt_fb) == 89.20


def test_production_model_companion_metrics_file():
    """Verify production ONNX model has a companion metrics file that loads valid accuracy."""
    from pathlib import Path
    from src.detection.detector import load_model_accuracy_pct

    model_path = Path("models/carcass_yolov8n.onnx")
    if model_path.exists():
        acc = load_model_accuracy_pct(model_path)
        assert acc is not None
        assert 90.0 <= acc <= 100.0


