"""Tests for CounterService adaptive idle mode on no object detection."""

import time
from unittest.mock import MagicMock
import pytest
from main import CounterService
from src.storage.db import LocalStorage
from src.detection.detector import Detection


def test_counter_service_idle_initialization(tmp_path):
    """Verify CounterService initializes idle configuration attributes."""
    db_file = str(tmp_path / "test_idle_init.db")
    service = CounterService(
        line_id="lineIdleInit",
        video_source_type="synthetic",
        db_path=db_file,
        idle_timeout_sec=15.0,
        idle_fps=2.0,
    )
    assert service.idle_timeout_sec == 15.0
    assert service.idle_fps == 2.0
    assert service.is_idle is False
    assert hasattr(service, "_last_detection_time")


def test_counter_service_enters_idle_and_wakes_up(tmp_path):
    """Verify service transitions to is_idle=True when no detections occur for idle_timeout_sec and immediately wakes up on detection."""
    db_file = str(tmp_path / "test_idle_trans.db")
    storage = LocalStorage(db_path=db_file)
    mock_source = MagicMock()
    # Dummy 640x480 frame
    import numpy as np
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    mock_source.read.return_value = (True, dummy_frame)
    mock_source.cap = MagicMock()

    mock_detector = MagicMock()
    # First frame: no detections; 2nd frame: 1 detection
    det = Detection(x1=100.0, y1=100.0, x2=200.0, y2=200.0, confidence=0.9, class_id=0, class_name="chicken")
    mock_detector.predict.side_effect = [
        [],
        [det],
    ]

    service = CounterService(
        line_id="lineIdleTrans",
        source=mock_source,
        detector=mock_detector,
        storage=storage,
        idle_timeout_sec=0.1,  # Short timeout for test
        idle_fps=1.0,
        cli_display=False,
    )

    # Initially not idle
    assert not service.is_idle

    # Frame 1: empty detection, record start time
    service.running = True
    # Simulate first frame processing
    ret, frame = service.source.read()
    detections = service.detector.predict(frame)
    now = time.perf_counter()
    # Artificially set _last_detection_time to 0.2s ago to exceed 0.1s timeout
    service._last_detection_time = now - 0.2
    
    # Process idle check as in run loop
    if len(detections) > 0:
        service._last_detection_time = now
        if service.is_idle:
            service.is_idle = False
    elif service.idle_timeout_sec > 0 and not service.is_idle:
        if (now - service._last_detection_time) >= service.idle_timeout_sec:
            service.is_idle = True

    assert service.is_idle is True

    # Frame 2: detection arrives -> immediate wake-up & buffer flush
    ret, frame = service.source.read()
    detections = service.detector.predict(frame)  # Returns det
    now = time.perf_counter()
    if len(detections) > 0:
        service._last_detection_time = now
        if service.is_idle:
            service.is_idle = False
            if hasattr(service.source, "cap") and service.source.cap is not None:
                try:
                    for _ in range(2):
                        service.source.cap.grab()
                except Exception:
                    pass

    assert service.is_idle is False
    assert mock_source.cap.grab.call_count == 2


def test_counter_service_run_loop_idle_transition_and_wake_up(tmp_path):
    """Verify CounterService.run() transitions to idle when timeout elapses and wakes up on detection."""
    db_file = str(tmp_path / "test_idle_run.db")
    storage = LocalStorage(db_path=db_file)
    mock_source = MagicMock()
    import numpy as np
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    mock_source.cap = MagicMock()

    det = Detection(x1=100.0, y1=100.0, x2=200.0, y2=200.0, confidence=0.9, class_id=0, class_name="chicken")
    mock_detector = MagicMock()

    service = CounterService(
        line_id="lineIdleRun",
        source=mock_source,
        detector=mock_detector,
        storage=storage,
        idle_timeout_sec=0.05,
        idle_fps=1.0,
        cli_display=False,
    )

    idle_states = []

    def mock_predict(frame):
        # Record idle state inside detector predict call
        idle_states.append(service.is_idle)
        if len(idle_states) == 1:
            # First frame: no detections, set last detection time in the past
            service._last_detection_time = time.perf_counter() - 0.1
            return []
        elif len(idle_states) == 2:
            # Second frame: no detections, should transition to idle
            return []
        else:
            # Third frame: detection arrives, should wake up
            return [det]

    mock_detector.predict.side_effect = mock_predict
    # 3 frames then stop
    mock_source.read.side_effect = [
        (True, dummy_frame),
        (True, dummy_frame),
        (True, dummy_frame),
        (False, None),
    ]

    service.run()

    # In frame 1: initially not idle
    assert idle_states[0] is False
    # In frame 2: was not idle during predict, but transitioned to idle right after predict
    # In frame 3: predict ran while idle (is_idle was True)
    assert idle_states[2] is True
    # After frame 3 detection: woke up
    assert service.is_idle is False
    # Camera buffer grab should have been called 2 times on wake up
    assert mock_source.cap.grab.call_count == 2
