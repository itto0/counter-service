"""Capture pipeline abstractions and concrete video sources."""

from abc import ABC, abstractmethod
import logging
import os
from pathlib import Path
import cv2
import numpy as np

logger = logging.getLogger("counter.capture")

__all__ = ["VideoSource", "SyntheticVideoSource", "FileVideoSource"]


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
