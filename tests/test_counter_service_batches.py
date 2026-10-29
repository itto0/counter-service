"""Integration tests for CounterService batch management, UI HUD display, and CLI arguments."""

import io
from unittest.mock import MagicMock, patch
import numpy as np
import pytest

from main import CounterService, build_cli_parser, parse_args
from src.contracts import PipelineConfig
from src.detection.detector import Detection
from src.storage.db import LocalStorage
from src.tracking.batch import BatchManager
from src.tracking.tracker import ChickenCounter, VirtualLineZone


def test_counter_service_batch_initialization_defaults(tmp_path):
    """Verify CounterService creates a BatchManager with default configuration."""
    db_file = str(tmp_path / "test_init_defaults.db")
    storage = LocalStorage(db_path=db_file)

    service = CounterService(
        line_id="lineDefault",
        storage=storage,
        enable_sync=False,
    )

    assert hasattr(service, "batch_manager")
    assert isinstance(service.batch_manager, BatchManager)
    assert service.batch_manager.line_id == "lineDefault"
    assert service.batch_manager.idle_timeout_sec == 30.0
    assert service.batch_prefix is None
    assert service.batch_manager.batch_prefix is None
    assert service.batch_manager.on_batch_start == service._on_batch_start


def test_counter_service_batch_initialization_custom_and_precedence(tmp_path, monkeypatch):
    """Verify batch configuration precedence: explicit arg > pipeline_config > env / idle_timeout."""
    db_file = str(tmp_path / "test_init_custom.db")
    storage = LocalStorage(db_path=db_file)

    # 1. Environment variable fallback for prefix
    monkeypatch.setenv("BATCH_PREFIX", "env_prefix")
    s1 = CounterService(line_id="l1", storage=storage, enable_sync=False)
    assert s1.batch_prefix == "env_prefix"
    assert s1.batch_manager.batch_prefix == "env_prefix"

    # 2. PipelineConfig overrides env
    cfg = PipelineConfig(batch_idle_timeout_sec=42.0, batch_prefix="cfg_prefix")
    s2 = CounterService(
        line_id="l2",
        storage=storage,
        enable_sync=False,
        pipeline_config=cfg,
    )
    assert s2.batch_idle_timeout_sec == 42.0
    assert s2.batch_prefix == "cfg_prefix"
    assert s2.batch_manager.idle_timeout_sec == 42.0
    assert s2.batch_manager.batch_prefix == "cfg_prefix"

    # 3. Explicit args override pipeline_config and env
    s3 = CounterService(
        line_id="l3",
        storage=storage,
        enable_sync=False,
        pipeline_config=cfg,
        batch_idle_timeout_sec=15.0,
        batch_prefix="explicit_prefix",
    )
    assert s3.batch_idle_timeout_sec == 15.0
    assert s3.batch_prefix == "explicit_prefix"
    assert s3.batch_manager.idle_timeout_sec == 15.0
    assert s3.batch_manager.batch_prefix == "explicit_prefix"

    # 4. Custom BatchManager instance passed in
    custom_mgr = BatchManager(line_id="custom_line", storage=storage, idle_timeout_sec=99.0)
    s4 = CounterService(
        line_id="custom_line",
        storage=storage,
        enable_sync=False,
        batch_manager=custom_mgr,
    )
    assert s4.batch_manager is custom_mgr
    assert s4.batch_manager.on_batch_start == s4._on_batch_start


def test_counter_service_on_batch_start_resets_counter(tmp_path):
    """Verify _on_batch_start resets counter.total_count and clears counted_track_ids."""
    db_file = str(tmp_path / "test_reset.db")
    storage = LocalStorage(db_path=db_file)

    service = CounterService(line_id="lineReset", storage=storage, enable_sync=False)

    # Artificially populate counter state
    service.counter.total_count = 27
    service.counter.counted_track_ids.add(101)
    service.counter.counted_track_ids.add(102)
    assert service.counter.total_count == 27
    assert len(service.counter.counted_track_ids) == 2

    # Trigger batch start callback
    service._on_batch_start("batch_lineReset_20261029_000000")

    assert service.counter.total_count == 0
    assert len(service.counter.counted_track_ids) == 0


def test_counter_service_tags_events_with_batch_id_in_outbox(tmp_path):
    """Verify that during pipeline execution, events are assigned batch_id and saved to outbox."""
    db_file = str(tmp_path / "test_tags_events.db")
    storage = LocalStorage(db_path=db_file)

    service = CounterService(
        line_id="lineTag",
        storage=storage,
        enable_sync=False,
        cli_display=False,
        batch_prefix="shift1",
        line_coords=(0.0, 200.0, 640.0, 200.0),
        line_direction="down",
        tracker_type="centroid",
    )

    # Create dummy frame and detections that cross the line (y=200)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    # Frame 1: Detection above line (anchor bottom center y=180) -> Starts batch
    det1 = Detection(x1=280, y1=140, x2=320, y2=180, confidence=0.95, class_id=0, class_name="chicken")
    # Frame 2: Detection crossing line (anchor bottom center y=220) -> Triggers count event
    det2 = Detection(x1=280, y1=180, x2=320, y2=220, confidence=0.96, class_id=0, class_name="chicken")

    class MockSource:
        def __init__(self):
            self.frames = [frame.copy(), frame.copy()]
            self.idx = 0

        def start(self):
            pass

        def stop(self):
            pass

        def read(self):
            if self.idx < len(self.frames):
                f = self.frames[self.idx]
                self.idx += 1
                return True, f
            return False, None

    class SequencedDetector:
        def __init__(self):
            self.seq = [[det1], [det2]]
            self.idx = 0

        def predict(self, frame):
            if self.idx < len(self.seq):
                res = self.seq[self.idx]
                self.idx += 1
                return res
            return []

    service.source = MockSource()
    service.detector = SequencedDetector()

    total_counted = service.run(max_frames=2)
    assert total_counted == 1

    # Since run() completes and invokes _cleanup(), the batch is finalized as COMPLETED
    assert service.batch_manager.is_active is False
    unsynced_batches = storage.get_unsynced_batches(line_id="lineTag")
    assert len(unsynced_batches) == 1
    batch = unsynced_batches[0]
    assert batch.batch_id.startswith("shift1_lineTag_")
    assert batch.status == "COMPLETED"
    assert batch.total_count == 1

    # Verify event stored in outbox_events was tagged with this batch_id
    unsynced_events = storage.get_unsynced_events(line_id="lineTag")
    assert len(unsynced_events) == 1
    assert unsynced_events[0]["batch_id"] == batch.batch_id


def test_counter_service_stop_and_cleanup_force_finishes_batch(tmp_path):
    """Verify stop() and _cleanup() finalize any active batch and mark it COMPLETED in storage."""
    db_file = str(tmp_path / "test_stop_finish.db")
    storage = LocalStorage(db_path=db_file)

    service = CounterService(
        line_id="lineStop",
        storage=storage,
        enable_sync=False,
        cli_display=False,
    )

    # Start an active batch
    service.batch_manager.on_activity(count_increment=5)
    assert service.batch_manager.is_active is True
    batch_id = service.batch_manager.current_batch_id

    # Call stop()
    service.stop()

    assert service.batch_manager.is_active is False
    assert service.batch_manager.current_batch_id is None

    # Verify batch record in storage is COMPLETED
    unsynced_batches = storage.get_unsynced_batches(line_id="lineStop")
    assert len(unsynced_batches) == 1
    assert unsynced_batches[0].batch_id == batch_id
    assert unsynced_batches[0].status == "COMPLETED"
    assert unsynced_batches[0].total_count == 5


def test_counter_service_idle_timeout_check_in_loop(tmp_path):
    """Verify check_idle_timeout completes an active batch when idle threshold is exceeded."""
    db_file = str(tmp_path / "test_idle_timeout.db")
    storage = LocalStorage(db_path=db_file)

    service = CounterService(
        line_id="lineIdle",
        storage=storage,
        enable_sync=False,
        batch_idle_timeout_sec=5.0,
    )

    # Start batch at perf time 100.0
    service.batch_manager.on_activity(count_increment=3, now_perf=100.0)
    assert service.batch_manager.is_active is True
    batch_id = service.batch_manager.current_batch_id

    # Periodic check before timeout (at 102.0s) -> remains active
    service.batch_manager.check_idle_timeout(now_perf=102.0)
    assert service.batch_manager.is_active is True

    # Periodic check after timeout (at 106.0s) -> completes batch
    service.batch_manager.check_idle_timeout(now_perf=106.0)
    assert service.batch_manager.is_active is False

    unsynced = storage.get_unsynced_batches(line_id="lineIdle")
    assert len(unsynced) == 1
    assert unsynced[0].batch_id == batch_id
    assert unsynced[0].status == "COMPLETED"


def test_render_annotated_frame_hud_batch_indicator(tmp_path):
    """Verify HUD text includes Batch indicator (IDLE or active batch ID)."""
    db_file = str(tmp_path / "test_hud.db")
    storage = LocalStorage(db_path=db_file)

    service = CounterService(line_id="lineHUD", storage=storage, enable_sync=False)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    # 1. When batch is IDLE
    with patch("cv2.putText") as mock_put_text:
        service.render_annotated_frame(frame, detections=[], events=[])
        # Find HUD putText call
        hud_calls = [c for c in mock_put_text.call_args_list if "Batch: IDLE" in str(c)]
        assert len(hud_calls) >= 1, "Expected 'Batch: IDLE' in cv2.putText HUD call"

    # 2. When batch is active
    service.batch_manager.on_activity(count_increment=1)
    b_id = service.batch_manager.current_batch_id
    assert b_id is not None

    with patch("cv2.putText") as mock_put_text:
        service.render_annotated_frame(frame, detections=[], events=[])
        hud_calls = [c for c in mock_put_text.call_args_list if f"Batch: {b_id}" in str(c)]
        assert len(hud_calls) >= 1, f"Expected 'Batch: {b_id}' in cv2.putText HUD call"


def test_cli_parser_batch_arguments():
    """Verify CLI parser supports --batch-idle-timeout and --batch-prefix."""
    parser = build_cli_parser()

    # Test explicit batch args
    args = parser.parse_args(["--batch-idle-timeout", "45.0", "--batch-prefix", "shiftA"])
    assert args.batch_idle_timeout_sec == 45.0
    assert args.batch_prefix == "shiftA"

    # Test alias --batch-idle-timeout-sec
    args2 = parser.parse_args(["--batch-idle-timeout-sec", "50.0"])
    assert args2.batch_idle_timeout_sec == 50.0

    # Test environment variable fallback
    with patch.dict("os.environ", {"BATCH_IDLE_TIMEOUT_SEC": "35.5", "BATCH_PREFIX": "envPref"}):
        parser_env = build_cli_parser()
        args_env = parser_env.parse_args([])
        assert args_env.batch_idle_timeout_sec == 35.5
        assert args_env.batch_prefix == "envPref"


def test_cli_header_and_summary_batch_display(tmp_path, capsys):
    """Verify CLI header and summary include batch mode and active batch info."""
    db_file = str(tmp_path / "test_cli_display.db")
    storage = LocalStorage(db_path=db_file)

    service = CounterService(
        line_id="lineDisp",
        storage=storage,
        enable_sync=False,
        batch_idle_timeout_sec=25.0,
        batch_prefix="shift_noon",
    )

    # 1. Check header output
    service._print_cli_header()
    captured_header = capsys.readouterr().out
    assert "Batch Mode:" in captured_header
    assert "25.0s idle timeout" in captured_header
    assert "shift_noon" in captured_header

    # 2. Check summary output when idle
    service._print_cli_summary()
    captured_summary = capsys.readouterr().out
    assert "Active Batch ID:" in captured_summary
    assert "None (idle/closed)" in captured_summary

    # 3. Check summary output when active
    service.batch_manager.on_activity(count_increment=2)
    b_id = service.batch_manager.current_batch_id
    service._print_cli_summary()
    captured_active_summary = capsys.readouterr().out
    assert f"Active Batch ID:            {b_id}" in captured_active_summary

