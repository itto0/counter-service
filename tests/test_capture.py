"""Tests for VideoSource capture abstractions."""

from pathlib import Path
import threading
import time
import cv2
import numpy as np
import pytest

from src.capture.source import (
    FileVideoSource,
    SyntheticVideoSource,
    ThreadedVideoSource,
    VideoSource,
)


def test_cannot_instantiate_abstract_video_source():
    """Verify VideoSource cannot be instantiated directly."""
    with pytest.raises(TypeError):
        VideoSource()  # type: ignore[abstract]


def test_synthetic_video_source_generates_frames():
    """Verify SyntheticVideoSource starts, generates valid frames, and stops."""
    source = SyntheticVideoSource(width=640, height=480, fps=30)
    source.start()
    success, frame = source.read()
    source.stop()

    assert success is True
    assert isinstance(frame, np.ndarray)
    assert frame.shape == (480, 640, 3)
    assert frame.dtype == np.uint8


def test_synthetic_video_source_stopped_returns_false():
    """Verify SyntheticVideoSource returns (False, None) when stopped or not started."""
    source = SyntheticVideoSource(width=320, height=240)
    # Before start
    success, frame = source.read()
    assert success is False
    assert frame is None

    # After start and stop
    source.start()
    source.stop()
    success, frame = source.read()
    assert success is False
    assert frame is None


def test_synthetic_video_source_moving_object():
    """Verify that successive frames simulate movement (pixels change)."""
    source = SyntheticVideoSource(width=320, height=240, fps=30)
    source.start()
    ret1, frame1 = source.read()
    ret2, frame2 = source.read()
    source.stop()

    assert ret1 is True and ret2 is True
    assert frame1 is not None and frame2 is not None
    # Verify the frames are not identical because of moving target
    assert not np.array_equal(frame1, frame2)


def test_synthetic_video_source_context_manager():
    """Verify SyntheticVideoSource works as a context manager."""
    with SyntheticVideoSource(width=320, height=240, fps=30) as source:
        assert source.is_running is True
        ret, frame = source.read()
        assert ret is True
        assert frame is not None

    assert source.is_running is False
    ret, frame = source.read()
    assert ret is False
    assert frame is None


def test_file_video_source_handles_non_existent_file():
    """Verify FileVideoSource handles non-existent file cleanly without crashing."""
    source = FileVideoSource("non_existent_dummy_video.mp4")
    # Before start
    ret, frame = source.read()
    assert ret is False
    assert frame is None

    # After start
    source.start()
    ret, frame = source.read()
    assert ret is False
    assert frame is None

    # After stop
    source.stop()
    ret, frame = source.read()
    assert ret is False
    assert frame is None


@pytest.fixture
def sample_video_path(tmp_path: Path) -> Path:
    """Create a temporary valid MP4 video with 5 frames."""
    video_path = tmp_path / "test_feed.mp4"
    width, height = 160, 120
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(video_path), fourcc, 30.0, (width, height))

    for i in range(5):
        frame = np.full((height, width, 3), i * 40, dtype=np.uint8)
        writer.write(frame)
    writer.release()
    return video_path


def test_file_video_source_read_video(sample_video_path: Path):
    """Verify FileVideoSource reads frames from a valid video file."""
    source = FileVideoSource(str(sample_video_path), loop=False)
    source.start()

    frames_read = 0
    while True:
        ret, frame = source.read()
        if not ret:
            break
        assert isinstance(frame, np.ndarray)
        assert frame.shape == (120, 160, 3)
        assert frame.dtype == np.uint8
        frames_read += 1

    source.stop()
    assert frames_read == 5

    # Reading after stop returns False
    ret, frame = source.read()
    assert ret is False
    assert frame is None


def test_file_video_source_looping(sample_video_path: Path):
    """Verify FileVideoSource loops back to the start when loop=True."""
    source = FileVideoSource(str(sample_video_path), loop=True)
    source.start()

    # Read more frames than the 5 frames in the file (e.g. 12 frames)
    frames_read = 0
    for _ in range(12):
        ret, frame = source.read()
        assert ret is True
        assert isinstance(frame, np.ndarray)
        frames_read += 1

    source.stop()
    assert frames_read == 12


def test_file_video_source_context_manager(sample_video_path: Path):
    """Verify FileVideoSource works as a context manager."""
    with FileVideoSource(str(sample_video_path)) as source:
        assert source.is_running is True
        ret, frame = source.read()
        assert ret is True
        assert frame is not None

    assert source.is_running is False
    ret, frame = source.read()
    assert ret is False
    assert frame is None


def test_threaded_video_source_basic_lifecycle():
    inner = SyntheticVideoSource(width=320, height=240, fps=30)
    source = ThreadedVideoSource(inner, maxsize=1, pace_fps=0.0)

    # Before start, reading returns False
    success, frame = source.read()
    assert success is False
    assert frame is None

    with source:
        success, frame = source.read()
        assert success is True
        assert frame is not None
        assert frame.shape == (240, 320, 3)

    # After stop, reading returns False
    success, frame = source.read()
    assert success is False
    assert frame is None


def test_threaded_video_source_drops_stale_frames():
    # Synthetic source generates frames with incrementing frame_idx
    inner = SyntheticVideoSource(width=100, height=100, fps=100)
    source = ThreadedVideoSource(inner, maxsize=1, pace_fps=0.0)

    with source:
        # Let worker push several frames while consumer sleeps
        time.sleep(0.08)
        success1, frame1 = source.read()
        assert success1 is True
        # Check that queue only kept the latest frame rather than buffering all
        assert source._frame_queue.qsize() <= 1


def test_threaded_video_source_pacing():
    inner = SyntheticVideoSource(width=100, height=100, fps=30)
    source = ThreadedVideoSource(inner, maxsize=1, pace_fps=30.0)

    start = time.perf_counter()
    with source:
        for _ in range(3):
            source.read()
    elapsed = time.perf_counter() - start
    # 3 frames at 30 FPS should take at least ~0.05 seconds with pacing
    assert elapsed >= 0.04


def test_threaded_video_source_queue_timeout_transient_delay():
    """Verify read() does not produce premature EOF on transient queue timeout while worker is alive."""
    class DelayedSource(VideoSource):
        def __init__(self):
            self.is_running = False

        def start(self):
            self.is_running = True

        def read(self):
            if not self.is_running:
                return False, None
            # Simulate a 120ms frame delay
            time.sleep(0.12)
            if not self.is_running:
                return False, None
            return True, np.zeros((10, 10, 3), dtype=np.uint8)

        def stop(self):
            self.is_running = False

    delayed = DelayedSource()
    threaded = ThreadedVideoSource(delayed, maxsize=1, pace_fps=0.0)
    with threaded:
        # Call read with a small timeout of 0.03s; worker is still alive and will produce frame at 0.12s
        ret, frame = threaded.read(timeout=0.03)
        assert ret is True
        assert frame is not None
        assert frame.shape == (10, 10, 3)


def test_threaded_video_source_live_reconnect_success():
    """Verify ThreadedVideoSource attempts reconnect with stop/start on live stream read failure."""
    class FlakyLiveSource(VideoSource):
        def __init__(self):
            self.is_running = False
            self.read_count = 0
            self.stop_count = 0
            self.start_count = 0
            self.is_live = True
            self.frame1_consumed = threading.Event()

        def start(self):
            self.is_running = True
            self.start_count += 1

        def read(self):
            self.read_count += 1
            if self.read_count == 1:
                return True, np.full((10, 10, 3), 1, dtype=np.uint8)
            elif self.read_count == 2:
                # Wait until frame 1 is consumed by caller before simulating disconnect
                self.frame1_consumed.wait(timeout=1.0)
                return False, None
            else:
                # Reconnection successful
                return True, np.full((10, 10, 3), 2, dtype=np.uint8)

        def stop(self):
            self.is_running = False
            self.stop_count += 1

    flaky = FlakyLiveSource()
    source = ThreadedVideoSource(flaky, maxsize=1, is_live=True, reconnect_timeout=2.0)
    with source:
        ret1, frame1 = source.read()
        assert ret1 is True
        assert frame1 is not None
        assert frame1[0, 0, 0] == 1
        flaky.frame1_consumed.set()

        # Second read triggers live reconnection
        ret2, frame2 = source.read()
        assert ret2 is True
        assert frame2 is not None
        assert frame2[0, 0, 0] == 2

    assert flaky.stop_count >= 1
    assert flaky.start_count >= 2


def test_threaded_video_source_live_reconnect_timeout():
    """Verify ThreadedVideoSource produces EOF when live reconnect exceeds reconnect_timeout."""
    class DeadLiveSource(VideoSource):
        def __init__(self):
            self.is_running = False
            self.is_live = True

        def start(self):
            self.is_running = True

        def read(self):
            return False, None

        def stop(self):
            self.is_running = False

    dead = DeadLiveSource()
    source = ThreadedVideoSource(dead, maxsize=1, is_live=True, reconnect_timeout=0.2)
    with source:
        ret, frame = source.read(timeout=0.05)
        assert ret is False
        assert frame is None

