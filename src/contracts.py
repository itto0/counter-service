"""Self-contained data contracts for the edge counter service."""

from datetime import datetime
import sys
from typing import Literal
from pydantic import BaseModel, Field


class CountEventDTO(BaseModel):
    track_id: int
    ts_utc: datetime
    confidence: float = Field(ge=0.0, le=1.0)


class CountAggregateDTO(BaseModel):
    window_start: datetime
    window_end: datetime
    count: int = Field(ge=0)
    rate_per_min: float = Field(ge=0.0)
    confidence_avg: float = Field(ge=0.0, le=1.0)
    status: Literal["NORMAL", "WARNING", "CRITICAL"] = "NORMAL"


class HealthSampleDTO(BaseModel):
    fps: float = Field(ge=0.0)
    latency_p95_ms: float = Field(ge=0.0)
    cpu_pct: float = Field(ge=0.0, le=100.0)
    mem_pct: float = Field(ge=0.0, le=100.0)
    stream_state: Literal["ONLINE", "OFFLINE", "DEGRADED", "IDLE"] = "ONLINE"
    model_name: str = "yolo26n"
    model_version: str = "v1.0"
    accuracy_pct: float | None = Field(default=None, ge=0.0, le=100.0)


class SyncBatchPayload(BaseModel):
    line_id: str
    events: list[CountEventDTO] = Field(default_factory=list)
    aggregates: list[CountAggregateDTO] = Field(default_factory=list)
    health: HealthSampleDTO | None = None


class PipelineConfig(BaseModel):
    """Runtime configuration for capture, tracking, and inference."""

    capture_mode: str = Field(default="threaded", description="threaded or direct")
    capture_pace_files: bool = Field(default=True, description="Pace video file reads to native FPS")
    tracker_type: str = Field(default="bytetrack", description="bytetrack or centroid")
    onnx_providers: str = Field(default="auto", description="auto or comma-separated list of providers")
    tracker_track_thresh: float = Field(default=0.4, description="High-confidence detection threshold for tracker")
    tracker_match_thresh: float = Field(default=0.7, description="IoU matching threshold for tracker")


__all__ = [
    "CountEventDTO",
    "CountAggregateDTO",
    "HealthSampleDTO",
    "SyncBatchPayload",
    "PipelineConfig",
]

# Ensure unified module identity across standalone and monorepo import paths
_self_module = sys.modules.get(__name__)
if _self_module is not None and __name__ != "__main__":
    for _alias in ("src.contracts", "src.contracts"):
        sys.modules.setdefault(_alias, _self_module)

