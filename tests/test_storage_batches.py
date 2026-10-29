from datetime import datetime, timezone, timedelta
import sqlite3
import pytest
from src.contracts import CountingBatchDTO, CountEventDTO, SyncBatchPayload
from src.storage.db import LocalStorage


@pytest.fixture
def storage(tmp_path) -> LocalStorage:
    db_file = tmp_path / "test_batches.db"
    store = LocalStorage(db_path=str(db_file))
    store.init_db()
    return store


def test_batch_table_created(storage, tmp_path):
    conn = sqlite3.connect(storage.db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = {row[0] for row in cursor.fetchall()}
    assert "outbox_batches" in tables

    # Verify batch_id column exists on outbox_events
    cursor.execute("PRAGMA table_info(outbox_events);")
    cols = {row[1] for row in cursor.fetchall()}
    assert "batch_id" in cols
    conn.close()


def test_start_and_get_active_batch(storage):
    now = datetime.now(timezone.utc)
    batch = CountingBatchDTO(
        batch_id="line-1_20261029_100000",
        line_id="line-1",
        start_time=now,
    )
    row_id = storage.start_batch(batch)
    assert row_id > 0

    active = storage.get_active_batch("line-1")
    assert active is not None
    assert active.batch_id == "line-1_20261029_100000"
    assert active.total_count == 0
    assert active.status == "ACTIVE"


def test_update_and_finish_batch(storage):
    now = datetime.now(timezone.utc)
    batch = CountingBatchDTO(
        batch_id="line-1_20261029_110000",
        line_id="line-1",
        start_time=now,
    )
    storage.start_batch(batch)
    storage.update_batch_count("line-1_20261029_110000", count=25)

    finish_time = now + timedelta(minutes=15)
    finished = storage.finish_batch("line-1_20261029_110000", finish_time=finish_time, total_count=30)
    assert finished is True

    # Active batch should now be None
    assert storage.get_active_batch("line-1") is None

    # Should appear in unsynced completed batches
    unsynced = storage.get_unsynced_batches(line_id="line-1")
    assert len(unsynced) == 1
    assert unsynced[0].batch_id == "line-1_20261029_110000"
    assert unsynced[0].total_count == 30
    assert unsynced[0].status == "COMPLETED"


def test_insert_event_with_batch_id(storage):
    now = datetime.now(timezone.utc)
    event = CountEventDTO(
        track_id=1,
        ts_utc=now,
        confidence=0.9,
        batch_id="batch_xyz",
    )
    event_id = storage.insert_event("line-1", event)
    assert event_id > 0

    events = storage.get_unsynced_events(limit=10, line_id="line-1")
    assert len(events) == 1
    assert events[0]["batch_id"] == "batch_xyz"


def test_mark_batches_synced(storage):
    now = datetime.now(timezone.utc)
    batch = CountingBatchDTO(
        batch_id="line-1_20261029_120000",
        line_id="line-1",
        start_time=now,
    )
    storage.start_batch(batch)
    storage.finish_batch("line-1_20261029_120000", finish_time=now, total_count=10)

    unsynced = storage.get_unsynced_batches(line_id="line-1")
    assert len(unsynced) == 1

    marked = storage.mark_batches_synced(["line-1_20261029_120000"])
    assert marked == 1

    remaining = storage.get_unsynced_batches(line_id="line-1")
    assert len(remaining) == 0

    # Idempotent mark on empty list
    assert storage.mark_batches_synced([]) == 0


def test_get_unsynced_batch_includes_batches_and_batch_ids(storage):
    now = datetime.now(timezone.utc)
    batch = CountingBatchDTO(
        batch_id="line-1_20261029_130000",
        line_id="line-1",
        start_time=now,
    )
    storage.start_batch(batch)
    storage.finish_batch("line-1_20261029_130000", finish_time=now, total_count=5)

    payload, ids = storage.get_unsynced_batch(line_id="line-1")
    assert payload is not None
    assert isinstance(payload, SyncBatchPayload)
    assert len(payload.batches) == 1
    assert payload.batches[0].batch_id == "line-1_20261029_130000"
    assert payload.batches[0].total_count == 5
    assert ids.get("batch_ids") == ["line-1_20261029_130000"]

    # Mark batch synced
    storage.mark_batches_synced(ids["batch_ids"])
    payload_after, ids_after = storage.get_unsynced_batch(line_id="line-1")
    assert payload_after is None
    assert ids_after == {"event_ids": [], "aggregate_ids": [], "health_ids": []}


def test_outbox_events_migration_existing_db(tmp_path):
    db_file = tmp_path / "old_outbox.db"
    conn = sqlite3.connect(str(db_file))
    conn.execute("""
    CREATE TABLE outbox_events (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        line_id TEXT NOT NULL,
        track_id INTEGER NOT NULL,
        ts_utc TEXT NOT NULL,
        confidence REAL NOT NULL,
        synced_at TEXT NULL
    );
    """)
    conn.execute("""
    INSERT INTO outbox_events (line_id, track_id, ts_utc, confidence)
    VALUES ('line-1', 42, '2026-10-29T10:00:00+00:00', 0.95);
    """)
    conn.commit()
    conn.close()

    migrated_storage = LocalStorage(db_path=str(db_file))
    migrated_storage.init_db()

    # Column should exist now
    conn = sqlite3.connect(str(db_file))
    cols = [col[1] for col in conn.execute("PRAGMA table_info(outbox_events);").fetchall()]
    conn.close()
    assert "batch_id" in cols

    events = migrated_storage.get_unsynced_events(limit=10, line_id="line-1")
    assert len(events) == 1
    assert events[0]["batch_id"] is None

    # Inserting new event with batch_id works
    ev = CountEventDTO(
        track_id=43,
        ts_utc=datetime.now(timezone.utc),
        confidence=0.88,
        batch_id="migrated_batch_1",
    )
    migrated_storage.insert_event("line-1", ev)

    events2 = migrated_storage.get_unsynced_events(limit=10, line_id="line-1")
    assert len(events2) == 2
    assert events2[1]["batch_id"] == "migrated_batch_1"
