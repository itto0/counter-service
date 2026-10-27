"""Capture pipeline abstractions."""

try:
    from .source import (
        FileVideoSource,
        SyntheticVideoSource,
        ThreadedVideoSource,
        VideoSource,
    )
except ImportError:
    try:
        from src.capture.source import (
            FileVideoSource,
            SyntheticVideoSource,
            ThreadedVideoSource,
            VideoSource,
        )
    except ImportError:
        from src.capture.source import (
            FileVideoSource,
            SyntheticVideoSource,
            ThreadedVideoSource,
            VideoSource,
        )

__all__ = ["VideoSource", "SyntheticVideoSource", "FileVideoSource", "ThreadedVideoSource"]
