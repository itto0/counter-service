"""Capture pipeline abstractions and concrete video sources."""

from abc import ABC, abstractmethod
import logging
import os
from pathlib import Path
import queue
import threading
import time
import cv2
import numpy as np

logger = logging.getLogger("counter.capture")

__all__ = ["VideoSource", "SyntheticVideoSource", "FileVideoSource", "ThreadedVideoSource"]


class VideoSource(ABC):
    """Abstract base class for all video sources."""

    @abstractmethod
    def start(self) -> None:
        """Initialize and open the video stream or file."""
        pass

    @abstractmethod
    def read(self) -> tuple[bool, np.ndarray | None]:
        """Read the next frame.

        Returns:
            (success, frame_bgr) where frame_bgr is a numpy array in BGR format
            if success is True, or (False, None) if frame read fails or EOF.
        """
        pass

    @abstractmethod
    def stop(self) -> None:
        """Release video resources and stop acquisition."""
        pass

    def __enter__(self) -> "VideoSource":
        """Context manager entry - starts source."""
        self.start()
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        """Context manager exit - stops source."""
        self.stop()


class SyntheticVideoSource(VideoSource):
    """Generates procedural frames with simulated chicken movement for testing/dev."""

    def __init__(self, width: int = 640, height: int = 480, fps: int = 30) -> None:
        self.width = width
        self.height = height
        self.fps = fps
        self.frame_idx = 0
        self.is_running = False

    def start(self) -> None:
        """Start generating synthetic video frames."""
        self.is_running = True
        self.frame_idx = 0

    def read(self) -> tuple[bool, np.ndarray | None]:
        """Read the next synthetic frame."""
        if not self.is_running:
            return False, None

        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        # Background conveyor tone
        frame[:] = (40, 40, 40)

        # Draw a moving white block representing a chicken shackle
        y_pos = (self.frame_idx * 5) % (self.height + 60) - 30
        x_pos = self.width // 2
        y1 = max(0, y_pos - 30)
        y2 = min(self.height, y_pos + 30)
        x1 = max(0, x_pos - 25)
        x2 = min(self.width, x_pos + 25)
        if y2 > y1 and x2 > x1:
            frame[y1:y2, x1:x2] = (220, 220, 220)

        self.frame_idx += 1
        return True, frame

    def stop(self) -> None:
        """Stop generating synthetic video frames."""
        self.is_running = False


class FileVideoSource(VideoSource):
    """OpenCV VideoCapture based wrapper for MP4/AVI replay with optional looping."""

    def __init__(self, file_path: str, loop: bool = False) -> None:
        self.file_path = str(file_path)
        self.loop = loop
        self.cap: cv2.VideoCapture | None = None
        self.is_running = False
        self.resolved_path = self._resolve_path(self.file_path)

    @staticmethod
    def _resolve_path(raw_path: str) -> str:
        """Resolve path, handling Docker vs host filesystem differences."""
        if not raw_path:
            return raw_path

        if os.path.exists(raw_path):
            return raw_path

        candidates: list[str] = []

        # 1. Strip leading slashes to convert container-style /samples/... to samples/...
        clean_rel = raw_path.lstrip("/\\")
        candidates.append(clean_rel)

        # 2. Check relative to SAMPLES_DIR if defined
        samples_dir = os.getenv("SAMPLES_DIR", "").strip()
        filename = os.path.basename(raw_path)
        if samples_dir:
            candidates.append(os.path.join(samples_dir, filename))
            candidates.append(os.path.join(samples_dir, clean_rel))

        # 3. Check relative to common local directory names: "samples", "sample", "data"
        candidates.append(os.path.join("samples", filename))
        candidates.append(os.path.join("sample", filename))
        candidates.append(os.path.join("data", filename))

        for cand in candidates:
            if os.path.exists(cand):
                logger.info("Resolved video path '%s' -> '%s'", raw_path, cand)
                return cand

        return raw_path

    def start(self) -> None:
        """Open the video file for capture."""
        self.resolved_path = self._resolve_path(self.file_path)
        path_to_open = self.resolved_path or self.file_path
        self.cap = cv2.VideoCapture(path_to_open)
        if not self.cap.isOpened():
            self.is_running = False
            logger.error(
                "Failed to open video file at '%s' (resolved as: '%s'). "
                "File does not exist or OpenCV VideoCapture failed to decode it.",
                self.file_path,
                path_to_open,
            )
            return
        self.is_running = True

    def read(self) -> tuple[bool, np.ndarray | None]:
        """Read the next frame from the video file."""
        if not self.is_running or self.cap is None or not self.cap.isOpened():
            return False, None

        ret, frame = self.cap.read()
        if not ret:
            if self.loop:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = self.cap.read()
                if not ret:
                    return False, None
                return True, frame
            return False, None

        return True, frame

    def stop(self) -> None:
        """Release the video file resource."""
        self.is_running = False
        if self.cap is not None:
            self.cap.release()
            self.cap = None


class ThreadedVideoSource(VideoSource):
    """Wraps any VideoSource with a dedicated background ingestion thread.

    Uses a bounded queue (default size 1) to drop stale frames when consumption
    lags behind production, eliminating RTSP latency accumulation.
    """

    def __init__(
        self,
        source: VideoSource,
        maxsize: int = 1,
        pace_fps: float = 0.0,
        reconnect_timeout: float = 5.0,
    ) -> None:
        self.source = source
        self.maxsize = maxsize
        self.pace_fps = pace_fps
        self.reconnect_timeout = reconnect_timeout
        self._frame_queue: queue.Queue[tuple[bool, np.ndarray | None]] = queue.Queue(maxsize=maxsize)
        self._stop_event = threading.Event()
        self._worker_thread: threading.Thread | None = None

    def start(self) -> None:
        """Start the underlying video source and dedicated ingestion thread."""
        if self._worker_thread is not None and self._worker_thread.is_alive():
            return
        self.source.start()
        self._stop_event.clear()
        self._worker_thread = threading.Thread(
            target=self._reader_loop,
            name="ThreadedVideoSource-Reader",
            daemon=True,
        )
        self._worker_thread.start()

    def _reader_loop(self) -> None:
        last_frame_time = time.perf_counter()
        frame_interval = (1.0 / self.pace_fps) if self.pace_fps > 0 else 0.0

        while not self._stop_event.is_set():
            if frame_interval > 0:
                now = time.perf_counter()
                elapsed = now - last_frame_time
                if elapsed < frame_interval:
                    time.sleep(frame_interval - elapsed)
                last_frame_time = time.perf_counter()

            try:
                success, frame = self.source.read()
            except Exception:
                logger.exception("Error reading from underlying video source")
                if self._stop_event.is_set():
                    break
                self._put_latest((False, None))
                time.sleep(0.01)
                continue

            if not success or frame is None:
                if self._stop_event.is_set():
                    break
                # Put EOF / failure and small backoff
                self._put_latest((False, None))
                time.sleep(0.01)
                continue

            self._put_latest((True, frame))

    def _put_latest(self, item: tuple[bool, np.ndarray | None]) -> None:
        while not self._stop_event.is_set():
            try:
                self._frame_queue.put_nowait(item)
                break
            except queue.Full:
                try:
                    self._frame_queue.get_nowait()
                except queue.Empty:
                    pass

    def read(self) -> tuple[bool, np.ndarray | None]:
        """Read the latest available frame from the bounded queue."""
        if self._worker_thread is None or (self._stop_event.is_set() and self._frame_queue.empty()):
            return False, None
        try:
            return self._frame_queue.get(timeout=0.5)
        except queue.Empty:
            return False, None

    def stop(self) -> None:
        """Stop ingestion thread and underlying source, and flush queue."""
        self._stop_event.set()
        if self._worker_thread is not None and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=2.0)
        self._worker_thread = None
        self.source.stop()
        # Drain queue
        while not self._frame_queue.empty():
            try:
                self._frame_queue.get_nowait()
            except queue.Empty:
                break

    @property
    def is_running(self) -> bool:
        """Return True if background ingestion thread is active."""
        return (
            self._worker_thread is not None
            and self._worker_thread.is_alive()
            and not self._stop_event.is_set()
        )

    @property
    def cap(self) -> Any:
        """Forward underlying OpenCV VideoCapture instance if available."""
        return getattr(self.source, "cap", None)

    @property
    def fps(self) -> float:
        """Forward underlying video source frame rate if available."""
        return float(getattr(self.source, "fps", 30.0))

