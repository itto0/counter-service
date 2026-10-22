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


@pytest.mark.parametrize(
    "video_source_type,expect_grab",
    [
        ("rtsp", True),
        ("file", False),
    ],
)
def test_counter_service_enters_idle_and_wakes_up(tmp_path, video_source_type, expect_grab):
    """Verify CounterService.run() transitions to idle when timeout elapses and wakes up on detection, grabbing buffer only for live sources."""
    db_file = str(tmp_path / f"test_idle_trans_{video_source_type}.db")
    storage = LocalStorage(db_path=db_file)
    mock_source = MagicMock()
    import numpy as np
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    mock_source.read.side_effect = [
        (True, dummy_frame),
        (True, dummy_frame),
        (False, None),
    ]
    mock_source.cap = MagicMock()

    det = Detection(x1=100.0, y1=100.0, x2=200.0, y2=200.0, confidence=0.9, class_id=0, class_name="chicken")
    mock_detector = MagicMock()

    service = CounterService(
        line_id=f"lineIdleTrans_{video_source_type}",
        source=mock_source,
        detector=mock_detector,
        storage=storage,
        video_source_type=video_source_type,
        idle_timeout_sec=0.05,
        idle_fps=1.0,
        cli_display=False,
    )

    idle_states = []

    def mock_predict(frame):
        idle_states.append(service.is_idle)
        if len(idle_states) == 1:
            # First frame: no detections, set last detection time in past to trigger idle transition
            service._last_detection_time = time.perf_counter() - 0.1
            return []
        else:
            # Second frame: detection arrives, should wake up
            return [det]

    mock_detector.predict.side_effect = mock_predict

    service.run()

    # Frame 1: initially not idle; Frame 2: predict ran while idle
    assert idle_states[0] is False
    assert idle_states[1] is True
    # Woke up after frame 2 detection
    assert service.is_idle is False
    # Live source calls grab twice; file source does not grab
    if expect_grab:
        assert mock_source.cap.grab.call_count == 2
    else:
        assert mock_source.cap.grab.call_count == 0


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
        video_source_type="rtsp",
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


def test_counter_service_records_idle_health(tmp_path):
    """Verify _record_health records stream_state='IDLE' when service is idle even if fps < 5.0, but preserves OFFLINE."""
    db_file = str(tmp_path / "test_idle_health.db")
    storage = LocalStorage(db_path=db_file)
    service = CounterService(
        line_id="lineIdleHealth",
        storage=storage,
        cli_display=False,
    )
    service.total_frames_processed = 10
    service.is_idle = True
    service._health_frame_count = 1  # Low frame count -> FPS would normally be < 5.0
    service._health_last_calc_time = time.perf_counter() - 1.0

    service._record_health(stream_state="ONLINE")
    service._record_health(stream_state="OFFLINE")

    samples = storage.get_unsynced_health(line_id="lineIdleHealth")
    assert len(samples) == 2
    assert samples[0]["stream_state"] == "IDLE"
    assert samples[1]["stream_state"] == "OFFLINE"


def test_counter_service_wake_up_flushes_idle_health_and_resets_window(tmp_path):
    """Verify waking up from idle flushes accumulated idle frames as IDLE and resets health window to prevent false DEGRADED reports."""
    db_file = str(tmp_path / "test_idle_health_wakeup.db")
    storage = LocalStorage(db_path=db_file)
    mock_source = MagicMock()
    import numpy as np
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    mock_source.cap = MagicMock()

    det = Detection(x1=100.0, y1=100.0, x2=200.0, y2=200.0, confidence=0.9, class_id=0, class_name="chicken")
    mock_detector = MagicMock()

    service = CounterService(
        line_id="lineWakeupHealth",
        source=mock_source,
        detector=mock_detector,
        storage=storage,
        video_source_type="rtsp",
        idle_timeout_sec=0.05,
        idle_fps=1.0,
        cli_display=False,
    )

    frame_calls = []

    def mock_predict(frame):
        frame_calls.append(len(frame_calls) + 1)
        if len(frame_calls) == 1:
            # First frame: no detection, backdate detection time to enter idle
            service._last_detection_time = time.perf_counter() - 0.1
            return []
        elif len(frame_calls) == 2:
            # Second frame: in idle mode, simulate elapsed time with low FPS (e.g., 2 frames in 2s)
            assert service.is_idle is True
            service._health_last_calc_time = time.perf_counter() - 2.0
            return []
        else:
            # Third frame: chicken detected -> triggers wake-up!
            assert service.is_idle is True
            return [det]

    mock_detector.predict.side_effect = mock_predict
    mock_source.read.side_effect = [
        (True, dummy_frame),
        (True, dummy_frame),
        (True, dummy_frame),
        (False, None),
    ]

    service.run()

    # The wake-up should have flushed accumulated idle frames as stream_state="IDLE"
    samples = storage.get_unsynced_health(line_id="lineWakeupHealth")
    assert len(samples) >= 1
    flushed_idle = samples[0]
    assert flushed_idle["stream_state"] == "IDLE"
    assert flushed_idle["fps"] < 5.0  # Even with FPS < 5.0, not marked DEGRADED

    # Verify health window was reset at the wake-up instant
    assert service._health_frame_count == 0

    # Simulate subsequent active processing: high FPS should be reported as ONLINE without false DEGRADED
    service.total_frames_processed += 10
    service._health_frame_count = 10
    now_perf = service._health_last_calc_time + 0.2  # 10 frames in 0.2s = 50 FPS
    service._record_health(now_perf=now_perf, stream_state="ONLINE")

    updated_samples = storage.get_unsynced_health(line_id="lineWakeupHealth")
    assert len(updated_samples) == 2
    active_sample = updated_samples[1]
    assert active_sample["stream_state"] == "ONLINE"
    assert active_sample["fps"] == 50.0


def test_counter_service_run_resets_is_idle(tmp_path):
    """Verify CounterService.run() resets self.is_idle to False upon invocation."""
    db_file = str(tmp_path / "test_idle_run_reset.db")
    storage = LocalStorage(db_path=db_file)
    mock_source = MagicMock()
    mock_source.read.return_value = (False, None)
    service = CounterService(
        line_id="lineResetIdle",
        source=mock_source,
        storage=storage,
        cli_display=False,
    )
    service.is_idle = True
    service.run()
    assert service.is_idle is False


def test_counter_service_disabled_idle_mode(tmp_path):
    """Verify idle_timeout_sec=0.0 disables idle mode transitions."""
    db_file = str(tmp_path / "test_idle_disabled.db")
    storage = LocalStorage(db_path=db_file)
    mock_source = MagicMock()
    import numpy as np
    mock_source.read.return_value = (True, np.zeros((480, 640, 3), dtype=np.uint8))
    mock_detector = MagicMock()
    mock_detector.predict.return_value = []

    service = CounterService(
        line_id="lineIdleDisabled",
        source=mock_source,
        detector=mock_detector,
        storage=storage,
        idle_timeout_sec=0.0,  # Disabled
        cli_display=False,
    )
    service.run(max_frames=5)
    assert service.is_idle is False


def test_counter_service_idle_pacing_delay(tmp_path, monkeypatch):
    """Verify frame pacing uses idle_fps when is_idle is True."""
    db_file = str(tmp_path / "test_idle_pacing.db")
    storage = LocalStorage(db_path=db_file)
    mock_source = MagicMock()
    import numpy as np
    mock_source.read.return_value = (True, np.zeros((480, 640, 3), dtype=np.uint8))
    mock_detector = MagicMock()
    mock_detector.predict.return_value = []

    sleep_calls = []
    monkeypatch.setattr(time, "sleep", lambda s: sleep_calls.append(s))

    service = CounterService(
        line_id="lineIdlePacing",
        source=mock_source,
        detector=mock_detector,
        storage=storage,
        target_fps=30.0,
        idle_fps=2.0,
        idle_timeout_sec=10.0,
        cli_display=False,
    )

    def mock_predict(frame):
        # Backdate detection time to trigger idle mode transition
        service._last_detection_time = time.perf_counter() - 20.0
        return []

    mock_detector.predict.side_effect = mock_predict
    service.run(max_frames=1)

    assert service.is_idle is True
    assert len(sleep_calls) >= 1
    assert sleep_calls[0] > 0.4


def test_counter_service_cli_display_idle_tag(tmp_path, capsys):
    """Verify CLI display output includes [IDLE] tag when service is idle."""
    db_file = str(tmp_path / "test_idle_cli.db")
    storage = LocalStorage(db_path=db_file)
    mock_source = MagicMock()
    import numpy as np
    mock_source.read.return_value = (True, np.zeros((480, 640, 3), dtype=np.uint8))
    mock_detector = MagicMock()
    mock_detector.predict.return_value = []

    service = CounterService(
        line_id="lineIdleCLI",
        source=mock_source,
        detector=mock_detector,
        storage=storage,
        cli_display=True,
    )

    def mock_predict(frame):
        # Backdate detection time to trigger idle mode transition
        service._last_detection_time = time.perf_counter() - 40.0
        return []

    mock_detector.predict.side_effect = mock_predict
    service.run(max_frames=1)

    captured = capsys.readouterr().out
    assert "[IDLE]" in captured


def test_idle_cli_args_parsing():
    """Verify build_arg_parser accepts --idle-timeout-sec and --idle-fps."""
    from main import build_arg_parser

    parser = build_arg_parser()
    args = parser.parse_args(["--idle-timeout-sec", "45.5", "--idle-fps", "2.5"])
    assert args.idle_timeout_sec == 45.5
    assert args.idle_fps == 2.5


def test_idle_cli_env_fallbacks(monkeypatch):
    """Verify build_arg_parser respects IDLE_TIMEOUT_SEC and IDLE_FPS env vars."""
    from main import build_arg_parser

    monkeypatch.setenv("IDLE_TIMEOUT_SEC", "12.0")
    monkeypatch.setenv("IDLE_FPS", "0.5")

    parser = build_arg_parser()
    args = parser.parse_args([])
    assert args.idle_timeout_sec == 12.0
    assert args.idle_fps == 0.5


def test_idle_cli_defaults_when_no_flags_or_env(monkeypatch):
    """Verify default values (idle_timeout_sec == 30.0, idle_fps == 1.0) when no CLI flags and no environment variables are set."""
    from main import build_arg_parser

    monkeypatch.delenv("IDLE_TIMEOUT_SEC", raising=False)
    monkeypatch.delenv("IDLE_FPS", raising=False)

    parser = build_arg_parser()
    args = parser.parse_args([])
    assert args.idle_timeout_sec == 30.0
    assert args.idle_fps == 1.0


def test_idle_cli_env_empty_string_fallbacks(monkeypatch):
    """Verify build_arg_parser falls back to defaults when env vars are empty strings."""
    from main import build_arg_parser

    monkeypatch.setenv("IDLE_TIMEOUT_SEC", "")
    monkeypatch.setenv("IDLE_FPS", "")

    parser = build_arg_parser()
    args = parser.parse_args([])
    assert args.idle_timeout_sec == 30.0
    assert args.idle_fps == 1.0


