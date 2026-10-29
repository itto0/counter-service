from datetime import datetime, timezone
import pytest


def test_capture_and_tracker_config():
    from src.contracts import PipelineConfig

    config = PipelineConfig()
    assert config.capture_mode == "threaded"
    assert config.capture_pace_files is True
    assert config.tracker_type == "bytetrack"
    assert config.onnx_providers == "auto"
    assert config.tracker_track_thresh == 0.4
    assert config.tracker_match_thresh == 0.7
    assert config.batch_idle_timeout_sec == 30.0
    assert config.batch_prefix is None


def test_counting_batch_dto():
    from src.contracts import CountingBatchDTO

    now = datetime.now(timezone.utc)
    batch = CountingBatchDTO(
        batch_id="line-1_20261029_120000",
        line_id="line-1",
        start_time=now,
    )
    assert batch.batch_id == "line-1_20261029_120000"
    assert batch.line_id == "line-1"
    assert batch.start_time == now
    assert batch.finish_time is None
    assert batch.total_count == 0
    assert batch.status == "ACTIVE"


def test_count_event_and_sync_payload_with_batches():
    from src.contracts import CountEventDTO, CountingBatchDTO, SyncBatchPayload

    now = datetime.now(timezone.utc)
    event = CountEventDTO(
        track_id=1,
        ts_utc=now,
        confidence=0.95,
        batch_id="batch_123",
    )
    assert event.batch_id == "batch_123"

    batch_dto = CountingBatchDTO(
        batch_id="batch_123",
        line_id="line-1",
        start_time=now,
        finish_time=now,
        total_count=42,
        status="COMPLETED",
    )
    payload = SyncBatchPayload(
        line_id="line-1",
        events=[event],
        batches=[batch_dto],
    )
    assert len(payload.batches) == 1
    assert payload.batches[0].batch_id == "batch_123"

