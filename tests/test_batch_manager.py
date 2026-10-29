from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock
import pytest

from src.contracts import CountingBatchDTO
from src.tracking.batch import BatchManager
from src.storage.db import LocalStorage
from src.tracking.tracker import ChickenCounter, VirtualLineZone, TrackedObject


@pytest.fixture
def storage(tmp_path) -> LocalStorage:
    store = LocalStorage(db_path=str(tmp_path / "batch_mgr.db"))
    store.init_db()
    return store


def test_batch_manager_starts_on_activity(storage):
    started_ids = []
    manager = BatchManager(
        line_id="line-1",
        storage=storage,
        idle_timeout_sec=5.0,
        on_batch_start=lambda b_id: started_ids.append(b_id),
    )
    assert not manager.is_active
    assert manager.current_batch_id is None
    assert manager.current_batch is None

    # Trigger activity with 0 count increment
    now_utc = datetime(2026, 10, 29, 14, 0, 0, tzinfo=timezone.utc)
    batch = manager.on_activity(count_increment=0, now_utc=now_utc)

    assert manager.is_active
    assert manager.current_batch_id is not None
    assert manager.current_batch_id.startswith("line-1_20261029_140000")
    assert len(started_ids) == 1
    assert started_ids[0] == manager.current_batch_id
    assert batch.total_count == 0
    assert batch.status == "ACTIVE"

    # Verify active batch persisted in storage
    active_in_store = storage.get_active_batch("line-1")
    assert active_in_store is not None
    assert active_in_store.batch_id == manager.current_batch_id
    assert active_in_store.total_count == 0


def test_batch_manager_custom_prefix(storage):
    manager = BatchManager(
        line_id="line-2",
        storage=storage,
        batch_prefix="shift_a",
    )
    now_utc = datetime(2026, 10, 29, 14, 0, 0, tzinfo=timezone.utc)
    batch = manager.on_activity(count_increment=1, now_utc=now_utc)

    assert batch.batch_id.startswith("shift_a_line-2_20261029_140000")
    assert manager.current_batch_id == batch.batch_id


def test_batch_manager_increments_count(storage):
    manager = BatchManager(line_id="line-1", storage=storage, idle_timeout_sec=5.0)
    now_utc = datetime(2026, 10, 29, 14, 0, 0, tzinfo=timezone.utc)
    manager.on_activity(count_increment=1, now_utc=now_utc)
    manager.on_activity(count_increment=4, now_utc=now_utc)

    assert manager.current_batch is not None
    assert manager.current_batch.total_count == 5

    # Verify persisted count in storage
    active_in_store = storage.get_active_batch("line-1")
    assert active_in_store is not None
    assert active_in_store.total_count == 5


def test_batch_manager_idle_timeout_completes_batch(storage):
    finished_batches = []
    manager = BatchManager(
        line_id="line-1",
        storage=storage,
        idle_timeout_sec=2.0,
        on_batch_finish=lambda b: finished_batches.append(b),
    )
    t0 = 100.0
    manager.on_activity(count_increment=3, now_perf=t0)
    assert manager.is_active

    # Check before timeout
    res = manager.check_idle_timeout(now_perf=t0 + 1.0)
    assert res is None
    assert manager.is_active

    # Check after timeout
    finish_utc = datetime(2026, 10, 29, 14, 5, 0, tzinfo=timezone.utc)
    completed = manager.check_idle_timeout(now_perf=t0 + 2.5, now_utc=finish_utc)
    assert completed is not None
    assert completed.status == "COMPLETED"
    assert completed.total_count == 3
    assert completed.finish_time == finish_utc
    assert not manager.is_active
    assert manager.current_batch is None
    assert manager.current_batch_id is None
    assert len(finished_batches) == 1
    assert finished_batches[0].batch_id == completed.batch_id

    # Verify storage has it completed and no active batch remains
    assert storage.get_active_batch("line-1") is None
    unsynced = storage.get_unsynced_batches(line_id="line-1")
    assert len(unsynced) == 1
    assert unsynced[0].batch_id == completed.batch_id
    assert unsynced[0].total_count == 3
    assert unsynced[0].status == "COMPLETED"


def test_batch_manager_check_idle_timeout_when_not_active():
    manager = BatchManager(line_id="line-1")
    assert manager.check_idle_timeout(now_perf=100.0) is None


def test_batch_manager_force_finish_on_shutdown(storage):
    finished_batches = []
    manager = BatchManager(
        line_id="line-1",
        storage=storage,
        idle_timeout_sec=10.0,
        on_batch_finish=lambda b: finished_batches.append(b),
    )
    manager.on_activity(count_increment=7)
    assert manager.is_active

    shutdown_time = datetime(2026, 10, 29, 15, 0, 0, tzinfo=timezone.utc)
    completed = manager.force_finish(now_utc=shutdown_time)
    assert completed is not None
    assert completed.total_count == 7
    assert completed.status == "COMPLETED"
    assert completed.finish_time == shutdown_time
    assert not manager.is_active
    assert len(finished_batches) == 1

    # Force finish when not active returns None
    assert manager.force_finish() is None


def test_batch_manager_lifecycle_multi_batch(storage):
    manager = BatchManager(line_id="line-1", storage=storage, idle_timeout_sec=1.0)
    t0 = 10.0
    now_utc_1 = datetime(2026, 10, 29, 14, 0, 0, tzinfo=timezone.utc)
    manager.on_activity(count_increment=2, now_perf=t0, now_utc=now_utc_1)
    batch1_id = manager.current_batch_id

    # Complete batch 1 via timeout
    manager.check_idle_timeout(now_perf=t0 + 1.5, now_utc=now_utc_1 + timedelta(seconds=1))
    assert not manager.is_active

    # Subsequent activity starts batch 2
    t1 = 20.0
    now_utc_2 = datetime(2026, 10, 29, 14, 10, 0, tzinfo=timezone.utc)
    manager.on_activity(count_increment=5, now_perf=t1, now_utc=now_utc_2)
    assert manager.is_active
    batch2_id = manager.current_batch_id
    assert batch2_id != batch1_id
    assert manager.current_batch.total_count == 5

    # Force finish batch 2
    manager.force_finish(now_utc=now_utc_2 + timedelta(minutes=5))
    assert not manager.is_active

    # Both batches should be in unsynced completed batches
    unsynced = storage.get_unsynced_batches(line_id="line-1")
    assert len(unsynced) == 2
    batch_ids = {b.batch_id for b in unsynced}
    assert batch1_id in batch_ids
    assert batch2_id in batch_ids


def test_batch_manager_callback_and_storage_error_resilience():
    # Flaky storage raising exceptions
    mock_storage = MagicMock()
    mock_storage.start_batch.side_effect = RuntimeError("disk error")
    mock_storage.update_batch_count.side_effect = RuntimeError("disk error")
    mock_storage.finish_batch.side_effect = RuntimeError("disk error")

    # Flaky callbacks raising exceptions
    def error_start_cb(b_id):
        raise ValueError("callback crashed")

    def error_finish_cb(batch):
        raise ValueError("callback crashed")

    manager = BatchManager(
        line_id="line-1",
        storage=mock_storage,
        idle_timeout_sec=1.0,
        on_batch_start=error_start_cb,
        on_batch_finish=error_finish_cb,
    )

    # Activity should not raise
    batch = manager.on_activity(count_increment=1, now_perf=100.0)
    assert manager.is_active
    assert batch.total_count == 1

    # Count update should not raise
    manager.on_activity(count_increment=2, now_perf=100.5)
    assert batch.total_count == 3

    # Finish should not raise
    completed = manager.check_idle_timeout(now_perf=102.0)
    assert completed is not None
    assert completed.status == "COMPLETED"
    assert not manager.is_active


def test_batch_manager_without_storage():
    manager = BatchManager(line_id="line-1", storage=None, idle_timeout_sec=2.0)
    manager.on_activity(count_increment=10, now_perf=50.0)
    assert manager.is_active
    assert manager.current_batch.total_count == 10

    completed = manager.check_idle_timeout(now_perf=53.0)
    assert completed is not None
    assert completed.total_count == 10
    assert not manager.is_active


def test_chicken_counter_reset_count():
    line_zone = VirtualLineZone(start=(0, 200), end=(640, 200), direction="down")
    counter = ChickenCounter(line_zone=line_zone, line_id="line-1")

    # Simulate some count and tracked IDs
    counter.total_count = 15
    counter.counted_track_ids.add(1)
    counter.counted_track_ids.add(2)
    counter.counted_track_ids.add(3)

    assert counter.total_count == 15
    assert len(counter.counted_track_ids) == 3

    # Reset count
    counter.reset_count()

    assert counter.total_count == 0
    assert len(counter.counted_track_ids) == 0
