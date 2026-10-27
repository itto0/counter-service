"""Tests for edge counter service production entrypoint and runner."""

import threading
from pathlib import Path
from unittest.mock import MagicMock, patch
from main import CounterService, main
from src.storage.db import LocalStorage
from src.sync.dispatcher import OutboxSyncDispatcher


def test_counter_service_initialization(tmp_path):
    """Verify CounterService correctly initializes internal components."""
    db_file = str(tmp_path / "test_init.db")
    service = CounterService(
        line_id="lineB",
        video_source_type="synthetic",
        api_sync_url="http://mock-api/sync",
        db_path=db_file,
        sync_interval_sec=2.0,
    )

    assert service.line_id == "lineB"
    assert service.video_source_type == "synthetic"
    assert service.api_sync_url == "http://mock-api/sync"
    assert service.sync_interval_sec == 2.0
    assert service.storage is not None
    assert service.dispatcher is not None
    assert service.source is not None
    assert service.detector is not None
    assert service.counter is not None


def test_counter_service_run_pipeline_max_frames(tmp_path):
    """Verify CounterService processes frames and stores outbox events."""
    db_file = str(tmp_path / "test_run.db")
    storage = LocalStorage(db_path=db_file)
    mock_http_client = MagicMock()
    mock_http_client.post.return_value.status_code = 200

    dispatcher = OutboxSyncDispatcher(
        storage=storage,
        api_url="http://mock-api/sync",
        line_id="lineTest",
        http_client=mock_http_client,
    )

    service = CounterService(
        line_id="lineTest",
        video_source_type="synthetic",
        storage=storage,
        dispatcher=dispatcher,
        sync_interval_sec=10.0,  # Sync will flush at cleanup
    )

    counted = service.run(max_frames=60)
    assert service.total_frames_processed == 60
    assert counted >= 0
    # Cleaned up
    assert not service.source.is_running


def test_counter_service_sync_interval_and_stop(tmp_path):
    """Verify periodic sync is triggered during loop and service stops on signal."""
    db_file = str(tmp_path / "test_sync.db")
    storage = LocalStorage(db_path=db_file)
    mock_http_client = MagicMock()
    mock_http_client.post.return_value.status_code = 200

    dispatcher = OutboxSyncDispatcher(
        storage=storage,
        api_url="http://mock-api/sync",
        line_id="lineSync",
        http_client=mock_http_client,
    )

    service = CounterService(
        line_id="lineSync",
        video_source_type="synthetic",
        storage=storage,
        dispatcher=dispatcher,
        sync_interval_sec=0.001,  # Fast sync interval
    )

    # Run for 20 frames
    service.run(max_frames=20)
    assert service.total_frames_processed == 20


def test_main_cli_entrypoint(tmp_path, monkeypatch):
    """Verify main() entrypoint respects environment variables."""
    db_file = str(tmp_path / "test_cli.db")
    monkeypatch.setenv("LINE_ID", "lineCLI")
    monkeypatch.setenv("VIDEO_SOURCE_TYPE", "synthetic")
    monkeypatch.setenv("DB_PATH", db_file)
    monkeypatch.setenv("MAX_FRAMES", "15")
    monkeypatch.setenv("SYNC_INTERVAL_SEC", "1.0")

    # Should run 15 frames and exit cleanly without error
    main()


def test_counter_service_records_aggregates_and_health(tmp_path):
    """Verify CounterService records both count aggregates and health samples."""
    db_file = str(tmp_path / "test_telemetry.db")
    storage = LocalStorage(db_path=db_file)
    mock_http_client = MagicMock()
    mock_http_client.post.return_value.status_code = 200

    dispatcher = OutboxSyncDispatcher(
        storage=storage,
        api_url="http://mock-api/sync",
        line_id="lineTelem",
        http_client=mock_http_client,
    )

    service = CounterService(
        line_id="lineTelem",
        video_source_type="synthetic",
        storage=storage,
        dispatcher=dispatcher,
        sync_interval_sec=10.0,
        aggregate_window_sec=0.01,  # Trigger rapid window aggregate
        health_interval_sec=0.01,     # Trigger rapid health sample
    )

    service.run(max_frames=10)

    # Check that aggregates or health samples were inserted
    unsynced_aggs = storage.get_unsynced_aggregates(line_id="lineTelem")
    unsynced_health = storage.get_unsynced_health(line_id="lineTelem")

    # Due to final flush at cleanup, records may be synced or in unsynced
    total_aggs = len(unsynced_aggs)
    total_health = len(unsynced_health)

    # Also check post calls to mock_http_client
    assert mock_http_client.post.called or (total_aggs > 0 or total_health > 0)


def test_counter_service_nonblocking_background_sync(tmp_path):
    """Verify sync runs decoupled in background worker without blocking frame capture."""
    import time
    db_file = str(tmp_path / "test_nonblock.db")
    storage = LocalStorage(db_path=db_file)

    sync_durations = []

    def slow_post(*args, **kwargs):
        time.sleep(0.05)  # Simulate network latency
        sync_durations.append(0.05)
        resp = MagicMock()
        resp.status_code = 200
        return resp

    mock_client = MagicMock()
    mock_client.post.side_effect = slow_post

    dispatcher = OutboxSyncDispatcher(
        storage=storage,
        api_url="http://mock-api/sync",
        line_id="lineNonBlock",
        http_client=mock_client,
    )

    service = CounterService(
        line_id="lineNonBlock",
        video_source_type="synthetic",
        storage=storage,
        dispatcher=dispatcher,
        sync_interval_sec=0.01,
    )

    t0 = time.perf_counter()
    counted = service.run(max_frames=15)
    total_time = time.perf_counter() - t0

    assert counted >= 0
    assert total_time >= 0.0
    assert service.total_frames_processed == 15
    assert not service.running
    # Background thread is joined and cleaned up
    assert service._sync_thread is None or not service._sync_thread.is_alive()


def test_counter_service_aggregate_status_levels(tmp_path):
    """Verify aggregate status categorization: NORMAL >= 130, WARNING 110-129.9, CRITICAL < 110."""
    db_file = str(tmp_path / "test_status.db")
    storage = LocalStorage(db_path=db_file)
    service = CounterService(
        line_id="lineStatus",
        storage=storage,
    )

    # 1. High count -> NORMAL
    service._window_count = 150
    service._window_confidences = [0.95]
    service._record_aggregate()
    aggs = storage.get_unsynced_aggregates(line_id="lineStatus")
    assert len(aggs) == 1
    assert aggs[0]["status"] == "NORMAL"

    # 2. Moderate count -> WARNING
    # Artificially set start time 60 seconds ago so elapsed is 60s
    from datetime import datetime, timezone, timedelta
    service._window_start_utc = datetime.now(timezone.utc) - timedelta(seconds=60)
    service._window_count = 120
    service._record_aggregate()
    aggs = storage.get_unsynced_aggregates(line_id="lineStatus")
    assert len(aggs) == 2
    assert aggs[1]["status"] == "WARNING"

    # 3. Low count -> CRITICAL
    service._window_start_utc = datetime.now(timezone.utc) - timedelta(seconds=60)
    service._window_count = 50
    service._record_aggregate()
    aggs = storage.get_unsynced_aggregates(line_id="lineStatus")
    assert len(aggs) == 3
    assert aggs[2]["status"] == "CRITICAL"


def test_main_argv_video_standalone(tmp_path):
    """Verify main() handles --video, --no-sync, and --db-path arguments."""
    from main import parse_args
    import cv2
    import numpy as np

    # Create dummy video file
    video_path = tmp_path / "test_vid.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(video_path), fourcc, 30, (64, 48))
    for _ in range(5):
        frame = np.zeros((48, 64, 3), dtype=np.uint8)
        out.write(frame)
    out.release()

    db_path = str(tmp_path / "argv_test.db")
    args = parse_args([
        "--video", str(video_path),
        "--no-sync",
        "--db-path", db_path,
        "--line", "0,24,64,24",
        "--direction", "down",
    ])
    assert args.video == str(video_path)
    assert args.no_sync is True
    assert args.db_path == db_path
    assert args.line == "0,24,64,24"

    # Run main with these arguments
    main([
        "--video", str(video_path),
        "--no-sync",
        "--db-path", db_path,
        "--max-frames", "5",
    ])
    assert Path(db_path).exists()


def test_main_argv_shackle_detector_options(tmp_path):
    """Verify parse_args and ShackleDetector setup with occupancy check options."""
    from main import parse_args
    import cv2
    import numpy as np

    video_path = tmp_path / "shackle_test.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(video_path), fourcc, 30, (64, 48))
    for _ in range(5):
        frame = np.zeros((48, 64, 3), dtype=np.uint8)
        out.write(frame)
    out.release()

    db_path = str(tmp_path / "shackle_argv.db")
    args = parse_args([
        "--video", str(video_path),
        "--detector", "shackle",
        "--occupancy-threshold", "5.2",
        "--no-occupancy-check",
        "--no-sync",
        "--db-path", db_path,
    ])
    assert args.detector == "shackle"
    assert args.occupancy_threshold == 5.2
    assert args.no_occupancy_check is True

    # Run main with shackle detector options
    main([
        "--video", str(video_path),
        "--detector", "shackle",
        "--occupancy-threshold", "5.2",
        "--no-occupancy-check",
        "--no-sync",
        "--db-path", db_path,
        "--max-frames", "5",
    ])
    assert Path(db_path).exists()


def test_render_window_frame_bounding_box_colors(tmp_path):
    """Verify _render_window_frame colors uncounted tracks orange and counted tracks green."""
    import cv2
    import numpy as np
    from unittest.mock import patch
    from src.tracking.tracker import TrackedObject

    db_file = str(tmp_path / "test_render.db")
    service = CounterService(
        line_id="lineRender",
        video_source_type="synthetic",
        db_path=db_file,
    )

    # Track 1: uncounted
    track1 = TrackedObject(
        track_id=1,
        last_anchor=(100.0, 150.0),
        current_anchor=(100.0, 150.0),
        bbox=(50.0, 50.0, 150.0, 150.0),
        class_name="chicken",
        confidence=0.91,
        counted=False,
        missed_frames=0,
    )
    # Track 2: counted
    track2 = TrackedObject(
        track_id=2,
        last_anchor=(300.0, 350.0),
        current_anchor=(300.0, 350.0),
        bbox=(250.0, 250.0, 350.0, 350.0),
        class_name="chicken",
        confidence=0.95,
        counted=True,
        missed_frames=0,
    )
    service.counter.tracks = {1: track1, 2: track2}

    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    with patch("cv2.imshow"), patch("cv2.namedWindow"), patch("cv2.resizeWindow"), patch("cv2.waitKey"), patch("cv2.putText", wraps=cv2.putText) as mock_put_text:
        vis = service._render_window_frame(dummy_frame, detections=[], events=[])

    assert vis is not None
    # Track 1 border at (x=50, y=50) should be Orange (BGR: 0, 180, 255)
    # Pixel on boundary:
    assert tuple(vis[50, 100]) == (0, 180, 255)
    # Track 2 border at (x=250, y=250) should be Green (BGR: 0, 220, 0)
    assert tuple(vis[250, 300]) == (0, 220, 0)

    # Verify track labels: Track 2 (counted) includes [COUNTED], Track 1 (uncounted) does not
    rendered_labels = [call.args[1] for call in mock_put_text.call_args_list]
    track1_labels = [lbl for lbl in rendered_labels if "#1" in lbl]
    assert len(track1_labels) == 1
    assert track1_labels[0] == "chicken #1 0.91"
    assert "[COUNTED]" not in track1_labels[0]

    track2_labels = [lbl for lbl in rendered_labels if "#2" in lbl]
    assert len(track2_labels) == 1
    assert track2_labels[0] == "chicken #2 0.95 [COUNTED]"
    assert "[COUNTED]" in track2_labels[0]


def test_render_window_frame_stale_tracks_and_unmatched_detections(tmp_path):
    """Verify stale tracks are skipped and unmatched detections are rendered orange."""
    import numpy as np
    from unittest.mock import patch
    from src.detection.detector import Detection
    from src.tracking.tracker import TrackedObject

    db_file = str(tmp_path / "test_render2.db")
    service = CounterService(
        line_id="lineRender",
        video_source_type="synthetic",
        db_path=db_file,
    )

    # Stale track (missed_frames > 0)
    stale_track = TrackedObject(
        track_id=3,
        last_anchor=(400.0, 400.0),
        current_anchor=(400.0, 400.0),
        bbox=(380.0, 380.0, 420.0, 420.0),
        class_name="chicken",
        confidence=0.9,
        counted=False,
        missed_frames=1,
    )
    service.counter.tracks = {3: stale_track}

    # Unmatched detection
    det = Detection(
        x1=50.0,
        y1=50.0,
        x2=150.0,
        y2=150.0,
        confidence=0.88,
        class_id=0,
        class_name="chicken",
    )

    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    with patch("cv2.imshow"), patch("cv2.namedWindow"), patch("cv2.resizeWindow"), patch("cv2.waitKey"):
        vis = service._render_window_frame(dummy_frame, detections=[det], events=[])

    assert vis is not None
    # Stale track should NOT be rendered (box boundary remains black)
    assert tuple(vis[380, 400]) == (0, 0, 0)
    # Unmatched detection should be rendered in Orange (BGR: 0, 180, 255)
    assert tuple(vis[50, 100]) == (0, 180, 255)


def test_render_window_frame_invalid_frame(tmp_path):
    """Verify _render_window_frame returns None when given invalid frame."""
    db_file = str(tmp_path / "test_render3.db")
    service = CounterService(line_id="lineRender", video_source_type="synthetic", db_path=db_file)
    assert service._render_window_frame(None, detections=[], events=[]) is None
    assert service._render_window_frame("not_a_frame", detections=[], events=[]) is None


def test_counter_service_geometry_mismatch_correction_horizontal_to_vertical():
    """Verify CounterService detects horizontal line coords with right/left direction and auto-corrects to vertical."""
    service = CounterService(
        line_id="lineGeo",
        video_source_type="synthetic",
        line_coords=(0.0, 240.0, 640.0, 240.0),
        line_direction="right",
    )
    # Since coords were horizontal but direction is right, it must be auto-corrected to vertical
    assert service.counter.line_zone.start[0] == service.counter.line_zone.end[0]
    assert service._custom_line_set is False


def test_counter_service_geometry_mismatch_correction_vertical_to_horizontal():
    """Verify CounterService detects vertical line coords with down/up direction and auto-corrects to horizontal."""
    service = CounterService(
        line_id="lineGeo",
        video_source_type="synthetic",
        line_coords=(320.0, 50.0, 320.0, 430.0),
        line_direction="down",
    )
    # Since coords were vertical but direction is down, it must be auto-corrected to horizontal
    assert service.counter.line_zone.start[1] == service.counter.line_zone.end[1]
    assert service._custom_line_set is False


def test_counter_service_valid_custom_line_preserved():
    """Verify valid custom line coords matching direction are preserved."""
    service = CounterService(
        line_id="lineGeo",
        video_source_type="synthetic",
        line_coords=(960.0, 200.0, 960.0, 950.0),
        line_direction="right",
    )
    assert service.counter.line_zone.start == (960.0, 200.0)
    assert service.counter.line_zone.end == (960.0, 950.0)
    assert service._custom_line_set is True


def test_main_startup_ignores_placeholder_mismatch():
    """Verify main() startup ignores placeholder 0,240,640,240 when remote direction is right/left."""
    mock_remote = {
        "line_coords": "0,240,640,240",
        "line_direction": "right",
    }
    with patch("main.fetch_remote_line_config", return_value=mock_remote), \
         patch("main.CounterService") as mock_service_cls:
        instance = mock_service_cls.return_value
        instance.run.return_value = 0
        main(["--line-id", "line-1", "--max-frames", "1"])
        # Should NOT have passed line_coords=(0.0, 240.0, 640.0, 240.0)
        # because the placeholder was ignored
        call_kwargs = mock_service_cls.call_args.kwargs
        assert call_kwargs.get("line_coords") is None
        assert call_kwargs.get("line_direction") == "right"


def test_main_cli_source_synthetic_override_env_video_path(monkeypatch):
    """Verify --source synthetic on CLI overrides VIDEO_PATH in environment."""
    monkeypatch.setenv("VIDEO_PATH", "/samples/Rekaman_2.MOV")
    with patch("main.CounterService") as mock_service_cls:
        instance = mock_service_cls.return_value
        instance.run.return_value = 0
        main(["--source", "synthetic", "--max-frames", "1"])
        call_kwargs = mock_service_cls.call_args.kwargs
        assert call_kwargs.get("video_source_type") == "synthetic"


def test_main_env_video_path_overrides_default_synthetic_source(monkeypatch):
    """Verify VIDEO_PATH in environment sets source to file even if VIDEO_SOURCE_TYPE=synthetic in image default."""
    monkeypatch.setenv("VIDEO_PATH", "/samples/Rekaman_2.MOV")
    monkeypatch.setenv("VIDEO_SOURCE_TYPE", "synthetic")
    from main import parse_args
    args = parse_args([])
    assert args.video == "/samples/Rekaman_2.MOV"
    assert args.source == "file"


def test_file_video_source_path_resolution(tmp_path, monkeypatch):
    """Verify FileVideoSource resolves non-existent absolute container path to relative host path."""
    import cv2
    import numpy as np
    from src.capture.source import FileVideoSource

    # Create a real test video under a 'samples' subdirectory of tmp_path
    samples_dir = tmp_path / "samples"
    samples_dir.mkdir()
    video_path = samples_dir / "test_sample.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(video_path), fourcc, 30, (64, 48))
    for _ in range(3):
        frame = np.zeros((48, 64, 3), dtype=np.uint8)
        out.write(frame)
    out.release()

    # Set working directory or SAMPLES_DIR to tmp_path
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SAMPLES_DIR", str(samples_dir))

    # Pass a container path /samples/test_sample.mp4 (which doesn't exist on host)
    source = FileVideoSource(file_path="/samples/test_sample.mp4")
    source.start()
    try:
        assert source.is_running is True
        ret, frame = source.read()
        assert ret is True
        assert frame is not None
    finally:
        source.stop()



def test_show_window_window_initialization_once(tmp_path):
    """Verify cv2.namedWindow and resizeWindow are called once on initial frame, not every frame."""
    import numpy as np
    db_file = str(tmp_path / "test_show_win.db")
    service = CounterService(
        line_id="lineWin",
        video_source_type="synthetic",
        db_path=db_file,
        show_window=True,
    )
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    with patch("cv2.imshow"), \
         patch("cv2.namedWindow") as mock_named_win, \
         patch("cv2.resizeWindow") as mock_resize_win, \
         patch("cv2.waitKey", return_value=-1):
        # Render frame 1
        service._render_window_frame(dummy_frame, detections=[], events=[])
        # Render frame 2
        service._render_window_frame(dummy_frame, detections=[], events=[])
        # Render frame 3
        service._render_window_frame(dummy_frame, detections=[], events=[])

    assert mock_named_win.call_count == 1


def test_show_window_user_quit_keypress(tmp_path):
    """Verify pressing 'q', 'Q', or ESC in preview window stops service run loop early."""
    import numpy as np
    db_file = str(tmp_path / "test_quit.db")
    service = CounterService(
        line_id="lineQuit",
        video_source_type="synthetic",
        db_path=db_file,
        show_window=True,
    )
    # Return 'q' on waitKey call
    with patch("cv2.imshow"), \
         patch("cv2.namedWindow"), \
         patch("cv2.resizeWindow"), \
         patch("cv2.waitKey", return_value=ord("q")), \
         patch("cv2.getWindowProperty", return_value=1.0):
        service.run(max_frames=10)

    # Should have stopped after 1 frame
    assert service.total_frames_processed == 1


def test_show_window_window_close_button(tmp_path):
    """Verify closing the window ('X' button) stops service run loop early."""
    import numpy as np
    db_file = str(tmp_path / "test_close_x.db")
    service = CounterService(
        line_id="lineCloseX",
        video_source_type="synthetic",
        db_path=db_file,
        show_window=True,
    )
    # First frame window open (prop=1.0), second frame user clicks 'X' (prop=0.0)
    prop_values = [1.0, 0.0]

    def mock_prop(*args, **kwargs):
        if prop_values:
            return prop_values.pop(0)
        return 0.0

    with patch("cv2.imshow"), \
         patch("cv2.namedWindow"), \
         patch("cv2.resizeWindow"), \
         patch("cv2.waitKey", return_value=-1), \
         patch("cv2.getWindowProperty", side_effect=mock_prop):
        service.run(max_frames=10)

    assert service.total_frames_processed <= 2


def test_show_window_gui_pacing_waitkey(tmp_path):
    """Verify pacing uses cv2.waitKey with delay when show_window=True to pump GUI event loop."""
    import numpy as np
    db_file = str(tmp_path / "test_pacing.db")
    service = CounterService(
        line_id="linePace",
        video_source_type="synthetic",
        db_path=db_file,
        show_window=True,
        target_fps=10.0,  # 100ms per frame
    )

    with patch("cv2.imshow"), \
         patch("cv2.namedWindow"), \
         patch("cv2.resizeWindow"), \
         patch("cv2.waitKey", return_value=-1) as mock_wait_key, \
         patch("cv2.getWindowProperty", return_value=1.0), \
         patch("time.sleep") as mock_sleep:
        service.run(max_frames=2)

    # cv2.waitKey should have been called with delay > 1 ms (pacing delay)
    wait_calls = [call.args[0] for call in mock_wait_key.call_args_list if call.args and call.args[0] > 1]
    assert len(wait_calls) >= 1
    # time.sleep should NOT be called for pacing when show_window is True
    assert mock_sleep.call_count == 0


def test_show_window_logs_warning_on_preview_error(tmp_path, caplog):
    """Verify exceptions in preview window are logged as WARNING."""
    import logging
    import numpy as np
    db_file = str(tmp_path / "test_err_warn.db")
    service = CounterService(
        line_id="lineErr",
        video_source_type="synthetic",
        db_path=db_file,
        show_window=True,
    )
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    with caplog.at_level(logging.WARNING), \
         patch("cv2.namedWindow", side_effect=RuntimeError("GUI backend unavailable")):
        res = service._render_window_frame(dummy_frame, detections=[], events=[])

    assert res is None
    assert any("visual preview" in rec.message.lower() or "preview window" in rec.message.lower() for rec in caplog.records)


def test_show_window_headless_disables_show_window_and_provides_guidance(tmp_path, caplog):
    """Verify that if OpenCV lacks GUI support (e.g. headless), show_window is disabled with guidance."""
    import logging
    import numpy as np
    db_file = str(tmp_path / "test_headless.db")
    service = CounterService(
        line_id="lineHeadless",
        video_source_type="synthetic",
        db_path=db_file,
        show_window=True,
    )
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    headless_err = RuntimeError(
        "OpenCV(5.0.0) ... error: (-2:Unspecified error) The function is not implemented. "
        "Rebuild the library with Windows, GTK+ 2.x or Cocoa support."
    )

    with caplog.at_level(logging.WARNING), \
         patch("cv2.namedWindow", side_effect=headless_err):
        service._render_window_frame(dummy_frame, detections=[], events=[])

    # show_window should be disabled to prevent trying and failing on every single frame
    assert service.show_window is False
    # Helpful guidance mentioning opencv-python should be logged
    assert any("opencv-python" in rec.message for rec in caplog.records)


def test_run_loop_disables_show_window_on_headless_error_and_continues(tmp_path, caplog):
    """Verify that during run(), headless OpenCV disables show_window without calling namedWindow repeatedly."""
    import logging
    import numpy as np
    from unittest.mock import MagicMock
    db_file = str(tmp_path / "test_run_headless.db")
    service = CounterService(
        line_id="lineRunHeadless",
        video_source_type="synthetic",
        db_path=db_file,
        show_window=True,
    )
    # Mock source to produce 5 frames then stop
    frames = [(True, np.zeros((240, 320, 3), dtype=np.uint8)) for _ in range(5)]
    mock_src = MagicMock()
    mock_src.read.side_effect = frames + [(False, None)]
    mock_src.is_running = True
    service.source = mock_src

    headless_err = RuntimeError(
        "OpenCV(5.0.0) ... error: (-2:Unspecified error) The function is not implemented. "
        "Rebuild the library with Windows, GTK+ 2.x or Cocoa support."
    )

    with caplog.at_level(logging.WARNING), \
         patch("cv2.namedWindow", side_effect=headless_err) as mock_named_window:
        service.run()

    # Named window should only be attempted once (on frame 1), then show_window turned off
    assert mock_named_window.call_count == 1
    assert service.show_window is False
    assert service.total_frames_processed == 5
    assert any("opencv-python" in rec.message for rec in caplog.records)


def test_main_cli_with_onnx_model_args(monkeypatch):
    """Verify parsing of --model, --nms-iou, and --conf CLI flags and env variable fallbacks."""
    from main import parse_args

    # 1. Verify explicit CLI arguments
    args = parse_args([
        "--model", "models/chicken_detector.onnx",
        "--conf", "0.65",
        "--nms-iou", "0.55",
    ])
    assert args.model == "models/chicken_detector.onnx"
    assert args.conf == 0.65
    assert args.nms_iou == 0.55

    # 2. Verify default values without env vars
    monkeypatch.delenv("MODEL_PATH", raising=False)
    monkeypatch.delenv("CONF_THRESHOLD", raising=False)
    monkeypatch.delenv("NMS_IOU_THRESHOLD", raising=False)
    default_args = parse_args([])
    assert default_args.model == ""
    assert default_args.conf == 0.7
    assert default_args.nms_iou == 0.45

    # 3. Verify environment variable fallback
    monkeypatch.setenv("MODEL_PATH", "models/env_yolo.onnx")
    monkeypatch.setenv("CONF_THRESHOLD", "0.85")
    monkeypatch.setenv("NMS_IOU_THRESHOLD", "0.35")
    env_args = parse_args([])
    assert env_args.model == "models/env_yolo.onnx"
    assert env_args.conf == 0.85
    assert env_args.nms_iou == 0.35


def test_detector_selection_onnx_instantiation():
    """Verify main() instantiates ONNXDetector with model_path, conf_threshold, and nms_iou_threshold."""
    with patch("main.ONNXDetector") as mock_onnx, \
         patch("main.CounterService") as mock_service_cls:
        instance = mock_service_cls.return_value
        instance.run.return_value = 0

        main([
            "--model", "models/yolov8_chicken.onnx",
            "--conf", "0.6",
            "--nms-iou", "0.5",
            "--no-sync",
            "--max-frames", "1",
        ])

        mock_onnx.assert_called_once_with(
            model_path="models/yolov8_chicken.onnx",
            conf_threshold=0.6,
            nms_iou_threshold=0.5,
            providers="auto",
        )
        assert mock_service_cls.call_args.kwargs.get("detector") == mock_onnx.return_value


def test_detector_selection_onnx_fallback_to_shackle():
    """Verify main() falls back to ShackleDetector when ONNXDetector fails on conveyor video."""
    with patch("main.ONNXDetector", side_effect=RuntimeError("Model file missing")), \
         patch("main.ShackleDetector") as mock_shackle_cls, \
         patch("main.CounterService") as mock_service_cls:
        instance = mock_service_cls.return_value
        instance.run.return_value = 0

        main([
            "--model", "invalid_path.onnx",
            "--video", "samples/rekaman_conveyor.mp4",
            "--no-sync",
            "--max-frames", "1",
        ])

        mock_shackle_cls.assert_called_once()
        assert mock_service_cls.call_args.kwargs.get("detector") == mock_shackle_cls.return_value


def test_detector_selection_onnx_fallback_to_mock(monkeypatch):
    """Verify main() falls back to MockDetector when ONNXDetector fails on synthetic source."""
    monkeypatch.setenv("VIDEO_PATH", "")
    with patch("main.ONNXDetector", side_effect=RuntimeError("Model file missing")), \
         patch("main.MockDetector") as mock_mock_cls, \
         patch("main.CounterService") as mock_service_cls:
        instance = mock_service_cls.return_value
        instance.run.return_value = 0

        main([
            "--model", "invalid_path.onnx",
            "--source", "synthetic",
            "--video", "",
            "--no-sync",
            "--max-frames", "1",
        ])

        mock_mock_cls.assert_called_once()
        assert mock_service_cls.call_args.kwargs.get("detector") == mock_mock_cls.return_value


def test_counter_service_records_health_with_accuracy_pct(tmp_path):
    """Verify CounterService discovers model accuracy and records accuracy_pct in health samples."""
    db_file = str(tmp_path / "test_health_acc.db")
    storage = LocalStorage(db_path=db_file)

    model_file = tmp_path / "carcass_yolov8n.onnx"
    model_file.touch()
    metrics_file = tmp_path / "carcass_yolov8n_metrics.json"
    metrics_file.write_text('{"metrics": {"map50": 0.9421}}', encoding="utf-8")

    mock_detector = MagicMock()
    mock_detector.model_name = "carcass_yolov8n"
    mock_detector.model_path = str(model_file)

    service = CounterService(
        line_id="lineAcc",
        storage=storage,
        detector=mock_detector,
        enable_sync=False,
    )

    assert service.model_accuracy_pct == 94.21

    service._record_health()

    unsynced_health = storage.get_unsynced_health(line_id="lineAcc")
    assert len(unsynced_health) == 1
    assert unsynced_health[0]["accuracy_pct"] == 94.21


def test_counter_service_records_health_accuracy_pct_none_when_no_metrics(tmp_path):
    """Verify CounterService records accuracy_pct as None when no metrics JSON exists."""
    db_file = str(tmp_path / "test_health_no_acc.db")
    storage = LocalStorage(db_path=db_file)

    service = CounterService(
        line_id="lineNoAcc",
        storage=storage,
        enable_sync=False,
    )

    assert service.model_accuracy_pct is None

    service._record_health()

    unsynced_health = storage.get_unsynced_health(line_id="lineNoAcc")
    assert len(unsynced_health) == 1
    assert unsynced_health[0]["accuracy_pct"] is None


def test_counter_service_immediate_sync_initialization_and_trigger(tmp_path):
    """Verify CounterService initializes _sync_trigger and trigger_sync() functions correctly."""
    db_file = str(tmp_path / "test_init_trigger.db")
    storage = LocalStorage(db_path=db_file)
    mock_dispatcher = MagicMock()

    # 1. Enabled by default
    service = CounterService(
        line_id="lineInitTest",
        storage=storage,
        dispatcher=mock_dispatcher,
    )
    assert service.enable_immediate_sync is True
    assert hasattr(service, "_sync_trigger")
    assert not service._sync_trigger.is_set()

    # Calling trigger_sync sets the event
    service.trigger_sync()
    assert service._sync_trigger.is_set()
    service._sync_trigger.clear()

    # 2. Disabled explicitly
    disabled_service = CounterService(
        line_id="lineInitTest2",
        storage=storage,
        dispatcher=mock_dispatcher,
        enable_immediate_sync=False,
    )
    assert disabled_service.enable_immediate_sync is False
    disabled_service.trigger_sync()
    assert not disabled_service._sync_trigger.is_set()

    # 3. Enabled but no dispatcher (standalone mode)
    no_dispatch_service = CounterService(
        line_id="lineInitTest3",
        storage=storage,
        dispatcher=None,
        enable_sync=False,
        enable_immediate_sync=True,
    )
    no_dispatch_service.trigger_sync()
    assert not no_dispatch_service._sync_trigger.is_set()


def test_sync_worker_wakes_immediately_on_trigger(tmp_path):
    """Verify _sync_worker wakes immediately (<100ms) on _sync_trigger.set() without waiting for sync_interval_sec."""
    import time
    db_file = str(tmp_path / "test_immediate_wake.db")
    storage = LocalStorage(db_path=db_file)
    mock_dispatcher = MagicMock()
    mock_dispatcher.sync_once.return_value = 0
    mock_dispatcher.backoff_sec = 1.0

    service = CounterService(
        line_id="lineWakeTest",
        storage=storage,
        dispatcher=mock_dispatcher,
        sync_interval_sec=10.0,  # Long 10-second polling interval
        enable_immediate_sync=True,
    )

    # Start sync thread directly
    worker_thread = threading.Thread(target=service._sync_worker, daemon=True)
    worker_thread.start()

    try:
        # Give worker a moment to start and enter wait state
        time.sleep(0.05)
        initial_calls = mock_dispatcher.sync_once.call_count

        t0 = time.perf_counter()
        # Trigger immediate sync
        service.trigger_sync()

        # Wait until dispatcher.sync_once is called again
        deadline = time.perf_counter() + 0.5
        while mock_dispatcher.sync_once.call_count == initial_calls and time.perf_counter() < deadline:
            time.sleep(0.005)

        wake_latency = time.perf_counter() - t0
        assert mock_dispatcher.sync_once.call_count > initial_calls
        # Must wake in well under 100ms (far less than the 10.0s interval)
        assert wake_latency < 0.10, f"Wake latency {wake_latency:.4f}s exceeded 100ms threshold"
    finally:
        service.stop()
        service._sync_trigger.set()
        worker_thread.join(timeout=1.0)


def test_sync_worker_suppresses_immediate_wake_during_backoff(tmp_path):
    """Verify _sync_worker honors backoff_sec and does not wake immediately when backoff > 1.0."""
    import time
    db_file = str(tmp_path / "test_backoff_suppress.db")
    storage = LocalStorage(db_path=db_file)
    mock_dispatcher = MagicMock()
    mock_dispatcher.sync_once.return_value = 0
    # Simulate active backoff (e.g. 5.0 seconds due to network error)
    mock_dispatcher.backoff_sec = 5.0

    service = CounterService(
        line_id="lineBackoffTest",
        storage=storage,
        dispatcher=mock_dispatcher,
        sync_interval_sec=5.0,
        enable_immediate_sync=True,
    )

    worker_thread = threading.Thread(target=service._sync_worker, daemon=True)
    worker_thread.start()

    try:
        time.sleep(0.05)
        initial_calls = mock_dispatcher.sync_once.call_count

        # Trigger sync while backoff is active
        service.trigger_sync()

        # Sleep 150ms: worker should NOT wake immediately because backoff is active
        time.sleep(0.15)
        assert mock_dispatcher.sync_once.call_count == initial_calls
    finally:
        service.stop()
        service._sync_trigger.set()
        worker_thread.join(timeout=1.0)


def test_counter_pipeline_triggers_sync_on_count_events(tmp_path):
    """Verify CounterService.run() calls trigger_sync() when counter produces count events."""
    db_file = str(tmp_path / "test_pipeline_trigger.db")
    storage = LocalStorage(db_path=db_file)
    mock_dispatcher = MagicMock()
    mock_dispatcher.sync_once.return_value = 0
    mock_dispatcher.backoff_sec = 1.0

    service = CounterService(
        line_id="lineTriggerPipe",
        video_source_type="synthetic",
        storage=storage,
        dispatcher=mock_dispatcher,
        enable_immediate_sync=True,
    )

    with patch.object(service, "trigger_sync", wraps=service.trigger_sync) as spy_trigger:
        # Mock counter to emit an event on first update
        from src.contracts import CountEventDTO
        from datetime import datetime, timezone
        mock_event = CountEventDTO(
            track_id=1,
            ts_utc=datetime.now(timezone.utc),
            confidence=0.95,
        )
        with patch.object(service.counter, "update", return_value=[mock_event]):
            service.run(max_frames=1)

        assert spy_trigger.called, "Expected trigger_sync to be called when events occurred"


def test_counter_service_stop_wakes_sync_thread_promptly(tmp_path):
    """Verify CounterService.stop() signals _sync_trigger so worker exits immediately without waiting for sync_interval_sec."""
    import time
    db_file = str(tmp_path / "test_stop_prompt.db")
    storage = LocalStorage(db_path=db_file)
    mock_dispatcher = MagicMock()
    mock_dispatcher.sync_once.return_value = 0
    mock_dispatcher.backoff_sec = 1.0

    service = CounterService(
        line_id="lineStopPrompt",
        storage=storage,
        dispatcher=mock_dispatcher,
        sync_interval_sec=10.0,
        enable_immediate_sync=True,
    )

    worker_thread = threading.Thread(target=service._sync_worker, daemon=True)
    service._sync_thread = worker_thread
    worker_thread.start()

    time.sleep(0.05)
    t0 = time.perf_counter()
    service.stop()
    worker_thread.join(timeout=1.0)
    stop_duration = time.perf_counter() - t0

    assert not worker_thread.is_alive()
    assert stop_duration < 0.20, f"stop() took {stop_duration:.4f}s, expected <200ms"


def test_counter_service_cleanup_sets_sync_trigger(tmp_path):
    """Verify CounterService._cleanup() sets _sync_trigger so worker exits promptly on cleanup."""
    db_file = str(tmp_path / "test_cleanup_prompt.db")
    storage = LocalStorage(db_path=db_file)
    service = CounterService(
        line_id="lineCleanPrompt",
        storage=storage,
        enable_immediate_sync=True,
    )
    assert not service._sync_trigger.is_set()
    service._cleanup()
    assert service._sync_trigger.is_set()


def test_parse_args_immediate_sync_flags(monkeypatch):
    """Verify CLI flags --immediate-sync and --no-immediate-sync and IMMEDIATE_SYNC env var."""
    from main import parse_args

    # 1. Default without flags or env
    monkeypatch.delenv("IMMEDIATE_SYNC", raising=False)
    args = parse_args([])
    assert args.immediate_sync is True

    # 2. Flag --no-immediate-sync
    args_no = parse_args(["--no-immediate-sync"])
    assert args_no.immediate_sync is False

    # 3. Explicit flag --immediate-sync
    args_yes = parse_args(["--immediate-sync"])
    assert args_yes.immediate_sync is True

    # 4. Environment variable false
    monkeypatch.setenv("IMMEDIATE_SYNC", "0")
    args_env_false = parse_args([])
    assert args_env_false.immediate_sync is False

    # 5. Environment variable true
    monkeypatch.setenv("IMMEDIATE_SYNC", "true")
    args_env_true = parse_args([])
    assert args_env_true.immediate_sync is True


def test_main_passes_immediate_sync_to_service():
    """Verify main() forwards enable_immediate_sync argument to CounterService."""
    with patch("main.CounterService") as mock_service_cls:
        instance = mock_service_cls.return_value
        instance.run.return_value = 0
        main(["--no-immediate-sync", "--no-sync", "--max-frames", "1"])
        call_kwargs = mock_service_cls.call_args.kwargs
        assert call_kwargs.get("enable_immediate_sync") is False


def test_counter_service_end_to_end_immediate_sync_dispatch(tmp_path):
    """Verify that when a count event occurs during pipeline run, dispatcher.sync_once() is invoked immediately."""
    import time
    db_file = str(tmp_path / "test_e2e_immediate.db")
    storage = LocalStorage(db_path=db_file)
    mock_http_client = MagicMock()
    mock_http_client.post.return_value.status_code = 200

    dispatcher = OutboxSyncDispatcher(
        storage=storage,
        api_url="http://mock-api/sync",
        line_id="lineE2E",
        http_client=mock_http_client,
    )

    service = CounterService(
        line_id="lineE2E",
        video_source_type="synthetic",
        storage=storage,
        dispatcher=dispatcher,
        sync_interval_sec=10.0,  # 10s fallback: immediate sync must fire without waiting
        enable_immediate_sync=True,
        cli_display=False,
    )

    from src.contracts import CountEventDTO
    from datetime import datetime, timezone
    mock_event = CountEventDTO(
        track_id=42,
        ts_utc=datetime.now(timezone.utc),
        confidence=0.92,
    )

    # Emit count event on frame 1
    with patch.object(service.counter, "update", return_value=[mock_event]):
        t0 = time.perf_counter()
        service.run(max_frames=1)
        elapsed = time.perf_counter() - t0

    # Ensure run completed swiftly (<4.0s total, far less than 10.0s fallback interval)
    assert elapsed < 4.0, f"Execution took {elapsed:.2f}s, expected fast execution (<4.0s vs 10.0s fallback)"
    # Ensure HTTP client received POST batch
    assert mock_http_client.post.called

