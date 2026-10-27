import os
import tempfile
import pytest
from src.capture.source import SyntheticVideoSource, ThreadedVideoSource
from src.detection.detector import MockDetector, Detection
from src.tracking.tracker import ChickenCounter, VirtualLineZone
from src.storage.db import LocalStorage
from src.sync.dispatcher import OutboxSyncDispatcher


def test_e2e_pipeline_with_threaded_capture_and_bytetrack():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test_outbox.db")
        storage = LocalStorage(db_path=db_path)

        synthetic = SyntheticVideoSource(width=320, height=240, fps=30)
        source = ThreadedVideoSource(synthetic, maxsize=1, pace_fps=0.0)

        # Mock detector that simulates crossing line at y=120
        class StepCrossingDetector(MockDetector):
            def __init__(self):
                super().__init__(detections=[])
                self.step = 0

            def detect(self, frame):
                self.step += 1
                if self.step == 1:
                    return [Detection(bbox=(140, 77, 180, 117), confidence=0.9, class_name="chicken")]
                elif self.step == 2:
                    return [Detection(bbox=(140, 83, 180, 123), confidence=0.9, class_name="chicken")]
                else:
                    return [Detection(bbox=(140, 89, 180, 129), confidence=0.9, class_name="chicken")]

        detector = StepCrossingDetector()
        line = VirtualLineZone(start=(0, 120), end=(320, 120), direction="down")
        counter = ChickenCounter(line_zone=line, tracker_type="bytetrack", track_thresh=0.25)

        with source:
            for _ in range(3):
                success, frame = source.read()
                assert success is True
                dets = detector.detect(frame)
                events = counter.update(dets)
                for ev in events:
                    storage.insert_count_event(ev)

        assert counter.total_count == 1
        unsynced = storage.get_unsynced_events(limit=10)
        assert len(unsynced) == 1
        assert unsynced[0]["line_id"] == "line-1"
        storage.close()


def test_e2e_pipeline_via_counter_service_run(tmp_path):
    """Test full CounterService execution using threaded capture and bytetrack tracker."""
    from main import CounterService
    db_file = str(tmp_path / "test_cs_e2e.db")

    service = CounterService(
        line_id="line-e2e",
        video_source_type="synthetic",
        db_path=db_file,
        capture_mode="threaded",
        tracker_type="bytetrack",
        enable_sync=False,
        cli_display=False,
    )
    assert isinstance(service.source, ThreadedVideoSource)
    assert service.counter.tracker_type == "bytetrack"

    total = service.run(max_frames=15)
    assert service.total_frames_processed == 15
    assert total >= 0
    assert not service.source.is_running


def test_e2e_pipeline_via_main_cli(tmp_path):
    """Test main() with all new flags."""
    from main import main, parse_args
    db_file = str(tmp_path / "test_main_e2e.db")

    args = parse_args([
        "--source", "synthetic",
        "--detector", "mock",
        "--tracker-type", "bytetrack",
        "--capture-mode", "threaded",
        "--onnx-providers", "CPUExecutionProvider",
        "--no-sync",
        "--db-path", db_file,
        "--max-frames", "5",
    ])
    assert args.capture_mode == "threaded"
    assert args.tracker_type == "bytetrack"
    assert args.onnx_providers == "CPUExecutionProvider"

    main([
        "--source", "synthetic",
        "--detector", "mock",
        "--tracker-type", "bytetrack",
        "--capture-mode", "threaded",
        "--onnx-providers", "CPUExecutionProvider",
        "--no-sync",
        "--db-path", db_file,
        "--max-frames", "5",
    ])


def test_create_source_modes():
    """Verify create_source correctly handles threaded and direct modes."""
    from main import create_source
    from src.capture.source import FileVideoSource, SyntheticVideoSource

    threaded_source = create_source(source_type="synthetic", capture_mode="threaded")
    assert isinstance(threaded_source, ThreadedVideoSource)

    direct_source = create_source(source_type="synthetic", capture_mode="direct")
    assert isinstance(direct_source, SyntheticVideoSource)
    assert not isinstance(direct_source, ThreadedVideoSource)

    # File pacing behavior
    file_threaded_paced = create_source(source_type="file", video_path="dummy.mp4", capture_mode="threaded", pace_files=True)
    assert isinstance(file_threaded_paced, ThreadedVideoSource)
    assert file_threaded_paced.pace_fps > 0

    file_threaded_unpaced = create_source(source_type="file", video_path="dummy.mp4", capture_mode="threaded", pace_files=False)
    assert isinstance(file_threaded_unpaced, ThreadedVideoSource)
    assert file_threaded_unpaced.pace_fps == 0.0


def test_cli_flags_parsing(monkeypatch):
    """Verify CLI parser handles --capture-mode, --pace-files, --tracker-type, --onnx-providers, and --duration."""
    from main import parse_args

    args_default = parse_args([])
    assert args_default.capture_mode == "threaded"
    assert args_default.pace_files is True
    assert args_default.tracker_type == "bytetrack"
    assert args_default.onnx_providers == "auto"
    assert args_default.duration is None

    args_explicit = parse_args([
        "--capture-mode", "direct",
        "--no-pace-files",
        "--tracker-type", "centroid",
        "--onnx-providers", "CUDAExecutionProvider,CPUExecutionProvider",
        "--duration", "5.5",
    ])
    assert args_explicit.capture_mode == "direct"
    assert args_explicit.pace_files is False
    assert args_explicit.tracker_type == "centroid"
    assert args_explicit.onnx_providers == "CUDAExecutionProvider,CPUExecutionProvider"
    assert args_explicit.duration == 5.5

    # Environment variable fallbacks
    monkeypatch.setenv("CAPTURE_MODE", "direct")
    monkeypatch.setenv("TRACKER_TYPE", "centroid")
    monkeypatch.setenv("ONNX_PROVIDERS", "OpenVINOExecutionProvider")
    monkeypatch.setenv("DURATION_SEC", "12.0")
    args_env = parse_args([])
    assert args_env.capture_mode == "direct"
    assert args_env.tracker_type == "centroid"
    assert args_env.onnx_providers == "OpenVINOExecutionProvider"
    assert args_env.duration == 12.0
