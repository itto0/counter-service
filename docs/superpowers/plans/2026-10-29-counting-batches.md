# Counting Batches Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement automated counting batches with start/finish lifecycle, per-batch counter resets, SQLite outbox persistence, upstream API synchronization, and real-time dashboard/preview display.

**Architecture:** A modular `BatchManager` acts as the state engine driving batch start, count accumulation, and idle timeout completion. Data contracts and SQLite storage (`outbox_batches` and `outbox_events.batch_id`) track batch lineage, while `OutboxSyncDispatcher` transmits completed batches upstream and `main.py` orchestrates resets and telemetry display.

**Tech Stack:** Python 3.14 / 3.11+, Pydantic v2, SQLite3 (WAL mode), httpx, pytest, OpenCV, rich.

## Global Constraints

- Python version: Python 3.11+ / 3.14 compatible.
- All timestamps in UTC (`datetime.now(timezone.utc)`).
- Batch ID format: `<line_id>_YYYYMMDD_HHMMSS` (or `<batch_prefix>_<line_id>_YYYYMMDD_HHMMSS` if prefix configured).
- Strict per-batch counting: counter resets to `0` at every batch start.
- Backward compatibility: existing contracts and sync payload schemas must remain compatible with optional fields.

---

### Task 1: Data Contracts for Counting Batches

**Files:**
- Modify: `src/contracts.py`
- Modify: `tests/test_contracts.py`

**Interfaces:**
- Consumes: None (base models)
- Produces: `CountingBatchDTO`, updated `CountEventDTO`, `SyncBatchPayload`, and `PipelineConfig`

- [ ] **Step 1: Write the failing test for batch contracts**

Add the following test functions to `tests/test_contracts.py`:

```python
from datetime import datetime, timezone
import pytest


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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv\Scripts\python.exe -m pytest tests/test_contracts.py::test_counting_batch_dto -v`
Expected: FAIL with `ImportError: cannot import name 'CountingBatchDTO' from 'src.contracts'`

- [ ] **Step 3: Update `src/contracts.py` with batch contracts**

Edit `src/contracts.py`:
1. Add `CountingBatchDTO`:
```python
class CountingBatchDTO(BaseModel):
    batch_id: str
    line_id: str = "line-1"
    start_time: datetime
    finish_time: datetime | None = None
    total_count: int = Field(default=0, ge=0)
    status: Literal["ACTIVE", "COMPLETED"] = "ACTIVE"
```
2. Add `batch_id: str | None = None` to `CountEventDTO`.
3. Add `batches: list[CountingBatchDTO] = Field(default_factory=list)` to `SyncBatchPayload`.
4. Add `batch_idle_timeout_sec: float = Field(default=30.0, description="Idle timeout in seconds to finish counting batch")` and `batch_prefix: str | None = Field(default=None, description="Optional custom prefix for batch IDs")` to `PipelineConfig`.
5. Add `"CountingBatchDTO"` to `__all__`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv\Scripts\python.exe -m pytest tests/test_contracts.py -v`
Expected: PASS (all 3 tests pass)

- [ ] **Step 5: Commit**

```bash
git add src/contracts.py tests/test_contracts.py
git commit -m "feat(contracts): add CountingBatchDTO and batch fields to pipeline contracts"
```

---

### Task 2: LocalStorage Batch Persistence & Schema Migration

**Files:**
- Modify: `src/storage/db.py`
- Create: `tests/test_storage_batches.py`

**Interfaces:**
- Consumes: `CountingBatchDTO`, `CountEventDTO`, `SyncBatchPayload` from `src.contracts`
- Produces: `LocalStorage.start_batch`, `LocalStorage.update_batch_count`, `LocalStorage.finish_batch`, `LocalStorage.get_active_batch`, `LocalStorage.get_unsynced_batches`, `LocalStorage.mark_batches_synced`

- [ ] **Step 1: Write the failing tests for batch database operations**

Create `tests/test_storage_batches.py`:

```python
from datetime import datetime, timezone, timedelta
import sqlite3
import pytest
from src.contracts import CountingBatchDTO, CountEventDTO
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv\Scripts\python.exe -m pytest tests/test_storage_batches.py -v`
Expected: FAIL with missing methods on `LocalStorage` and missing table/column.

- [ ] **Step 3: Implement batch storage logic in `src/storage/db.py`**

In `src/storage/db.py`:
1. In `init_db()`:
   - Add SQL table creation for `outbox_batches`:
     ```sql
     CREATE TABLE IF NOT EXISTS outbox_batches (
         id INTEGER PRIMARY KEY AUTOINCREMENT,
         batch_id TEXT NOT NULL UNIQUE,
         line_id TEXT NOT NULL,
         start_time TEXT NOT NULL,
         finish_time TEXT NULL,
         total_count INTEGER NOT NULL DEFAULT 0,
         status TEXT NOT NULL DEFAULT 'ACTIVE',
         synced_at TEXT NULL
     );
     CREATE INDEX IF NOT EXISTS idx_outbox_batches_synced ON outbox_batches(synced_at);
     CREATE INDEX IF NOT EXISTS idx_outbox_batches_line ON outbox_batches(line_id);
     ```
   - Migration for `outbox_events`:
     ```python
     try:
         conn.execute("ALTER TABLE outbox_events ADD COLUMN batch_id TEXT NULL")
     except sqlite3.OperationalError:
         pass
     ```
2. Update `insert_event`:
   - Include `batch_id` column in `INSERT INTO outbox_events (line_id, track_id, ts_utc, confidence, batch_id) VALUES (?, ?, ?, ?, ?)`.
3. Implement `start_batch`:
   ```python
   def start_batch(self, batch: CountingBatchDTO) -> int:
       with self._lock:
           conn = self._get_connection()
           try:
               with conn:
                   cursor = conn.execute(
                       """
                       INSERT INTO outbox_batches (batch_id, line_id, start_time, finish_time, total_count, status)
                       VALUES (?, ?, ?, ?, ?, ?)
                       """,
                       (
                           batch.batch_id,
                           batch.line_id,
                           batch.start_time.isoformat(),
                           batch.finish_time.isoformat() if batch.finish_time else None,
                           batch.total_count,
                           batch.status,
                       ),
                   )
                   return cursor.lastrowid or 0
           finally:
               conn.close()
   ```
4. Implement `update_batch_count`:
   ```python
   def update_batch_count(self, batch_id: str, count: int) -> None:
       with self._lock:
           conn = self._get_connection()
           try:
               with conn:
                   conn.execute(
                       "UPDATE outbox_batches SET total_count = ? WHERE batch_id = ?",
                       (count, batch_id),
                   )
           finally:
               conn.close()
   ```
5. Implement `finish_batch`:
   ```python
   def finish_batch(self, batch_id: str, finish_time: datetime, total_count: int) -> bool:
       with self._lock:
           conn = self._get_connection()
           try:
               with conn:
                   cursor = conn.execute(
                       """
                       UPDATE outbox_batches
                       SET finish_time = ?, total_count = ?, status = 'COMPLETED'
                       WHERE batch_id = ?
                       """,
                       (finish_time.isoformat(), total_count, batch_id),
                   )
                   return cursor.rowcount > 0
           finally:
               conn.close()
   ```
6. Implement `get_active_batch`:
   ```python
   def get_active_batch(self, line_id: str) -> Optional[CountingBatchDTO]:
       with self._lock:
           conn = self._get_connection()
           try:
               cursor = conn.execute(
                   """
                   SELECT batch_id, line_id, start_time, finish_time, total_count, status
                   FROM outbox_batches
                   WHERE line_id = ? AND status = 'ACTIVE'
                   ORDER BY id DESC LIMIT 1
                   """,
                   (line_id,),
               )
               row = cursor.fetchone()
               if not row:
                   return None
               return CountingBatchDTO(
                   batch_id=row["batch_id"],
                   line_id=row["line_id"],
                   start_time=datetime.fromisoformat(row["start_time"]),
                   finish_time=datetime.fromisoformat(row["finish_time"]) if row["finish_time"] else None,
                   total_count=row["total_count"],
                   status=row["status"],
               )
           finally:
               conn.close()
   ```
7. Implement `get_unsynced_batches`:
   ```python
   def get_unsynced_batches(self, limit: int = 50, line_id: str | None = None) -> list[CountingBatchDTO]:
       with self._lock:
           conn = self._get_connection()
           try:
               query = """
                   SELECT batch_id, line_id, start_time, finish_time, total_count, status
                   FROM outbox_batches
                   WHERE synced_at IS NULL AND status = 'COMPLETED'
               """
               params: list[Any] = []
               if line_id:
                   query += " AND line_id = ?"
                   params.append(line_id)
               query += " ORDER BY id ASC LIMIT ?"
               params.append(limit)
               cursor = conn.execute(query, tuple(params))
               rows = cursor.fetchall()
               return [
                   CountingBatchDTO(
                       batch_id=r["batch_id"],
                       line_id=r["line_id"],
                       start_time=datetime.fromisoformat(r["start_time"]),
                       finish_time=datetime.fromisoformat(r["finish_time"]) if r["finish_time"] else None,
                       total_count=r["total_count"],
                       status=r["status"],
                   )
                   for r in rows
               ]
           finally:
               conn.close()
   ```
8. Implement `mark_batches_synced`:
   ```python
   def mark_batches_synced(self, batch_ids: list[str], synced_at: datetime | None = None) -> int:
       if not batch_ids:
           return 0
       ts = (synced_at or datetime.now(timezone.utc)).isoformat()
       with self._lock:
           conn = self._get_connection()
           try:
               with conn:
                   placeholders = ",".join("?" for _ in batch_ids)
                   cursor = conn.execute(
                       f"UPDATE outbox_batches SET synced_at = ? WHERE batch_id IN ({placeholders})",
                       [ts] + list(batch_ids),
                   )
                   return cursor.rowcount
           finally:
               conn.close()
   ```
9. Update `get_unsynced_batch`:
   - Include completed batches in payload and include `"batch_ids"` in `item_ids`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv\Scripts\python.exe -m pytest tests/test_storage_batches.py tests/test_storage.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/storage/db.py tests/test_storage_batches.py
git commit -m "feat(storage): implement SQLite outbox persistence and schema migrations for batches"
```

---

### Task 3: Outbox Sync Dispatcher Integration

**Files:**
- Modify: `src/sync/dispatcher.py`
- Modify: `tests/test_sync.py`

**Interfaces:**
- Consumes: `LocalStorage.get_unsynced_batch`, `LocalStorage.mark_batches_synced`
- Produces: `OutboxSyncDispatcher.sync_once` with batch syncing support

- [ ] **Step 1: Write failing test in `tests/test_sync.py`**

Add test to `tests/test_sync.py`:

```python
def test_sync_dispatcher_pushes_completed_batches_and_marks_synced(tmp_path):
    storage = LocalStorage(db_path=str(tmp_path / "sync_batch.db"))
    storage.init_db()

    now = datetime.now(timezone.utc)
    batch = CountingBatchDTO(
        batch_id="line-1_20261029_090000",
        line_id="line-1",
        start_time=now,
        finish_time=now,
        total_count=50,
        status="COMPLETED",
    )
    storage.start_batch(batch)
    storage.finish_batch("line-1_20261029_090000", finish_time=now, total_count=50)

    mock_http_client = MagicMock()
    mock_http_client.post.return_value.status_code = 200

    dispatcher = OutboxSyncDispatcher(
        storage=storage,
        api_url="http://mock-api/api/v1/sync/batch",
        line_id="line-1",
        http_client=mock_http_client,
    )

    synced_count = dispatcher.sync_once()
    assert synced_count >= 1
    assert len(storage.get_unsynced_batches(line_id="line-1")) == 0
    mock_http_client.post.assert_called_once()
    sent_payload = mock_http_client.post.call_args[1]["json"]
    assert len(sent_payload.get("batches", [])) == 1
    assert sent_payload["batches"][0]["batch_id"] == "line-1_20261029_090000"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv\Scripts\python.exe -m pytest tests/test_sync.py::test_sync_dispatcher_pushes_completed_batches_and_marks_synced -v`
Expected: FAIL because `OutboxSyncDispatcher` doesn't handle `batch_ids` yet.

- [ ] **Step 3: Update `src/sync/dispatcher.py`**

In `src/sync/dispatcher.py`:
1. In `sync_once()`:
   ```python
   batch_ids = item_ids.get("batch_ids", [])
   total_items = len(event_ids) + len(aggregate_ids) + len(health_ids) + len(batch_ids)
   ```
2. When HTTP status is 200/201:
   ```python
   if batch_ids:
       self.storage.mark_batches_synced(batch_ids)
   ```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv\Scripts\python.exe -m pytest tests/test_sync.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/sync/dispatcher.py tests/test_sync.py
git commit -m "feat(sync): support dispatching and marking synced counting batches"
```

---

### Task 4: `BatchManager` State Machine Component

**Files:**
- Create: `src/tracking/batch.py`
- Modify: `src/tracking/tracker.py:214` (add `reset_count` method)
- Create: `tests/test_batch_manager.py`

**Interfaces:**
- Consumes: `CountingBatchDTO` from `src.contracts`, `LocalStorage` from `src.storage.db`
- Produces: `BatchManager` class with `on_activity`, `check_idle_timeout`, `force_finish`, `current_batch_id`, `current_batch`, `is_active`

- [ ] **Step 1: Write failing tests in `tests/test_batch_manager.py`**

Create `tests/test_batch_manager.py`:

```python
from datetime import datetime, timezone, timedelta
import time
import pytest
from src.contracts import CountingBatchDTO
from src.tracking.batch import BatchManager
from src.storage.db import LocalStorage


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

    # Trigger activity with 0 count increment
    now_utc = datetime(2026, 10, 29, 14, 0, 0, tzinfo=timezone.utc)
    batch = manager.on_activity(count_increment=0, now_utc=now_utc)

    assert manager.is_active
    assert manager.current_batch_id is not None
    assert manager.current_batch_id.startswith("line-1_20261029_140000")
    assert len(started_ids) == 1
    assert started_ids[0] == manager.current_batch_id
    assert batch.total_count == 0


def test_batch_manager_increments_count(storage):
    manager = BatchManager(line_id="line-1", storage=storage, idle_timeout_sec=5.0)
    now_utc = datetime(2026, 10, 29, 14, 0, 0, tzinfo=timezone.utc)
    manager.on_activity(count_increment=1, now_utc=now_utc)
    manager.on_activity(count_increment=4, now_utc=now_utc)

    assert manager.current_batch.total_count == 5


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
    assert not manager.is_active
    assert len(finished_batches) == 1

    # Verify storage has it completed
    unsynced = storage.get_unsynced_batches(line_id="line-1")
    assert len(unsynced) == 1
    assert unsynced[0].batch_id == completed.batch_id
    assert unsynced[0].total_count == 3


def test_batch_manager_force_finish_on_shutdown(storage):
    manager = BatchManager(line_id="line-1", storage=storage, idle_timeout_sec=10.0)
    manager.on_activity(count_increment=7)
    assert manager.is_active

    completed = manager.force_finish()
    assert completed is not None
    assert completed.total_count == 7
    assert not manager.is_active
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv\Scripts\python.exe -m pytest tests/test_batch_manager.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.tracking.batch'`

- [ ] **Step 3: Implement `reset_count` in `src/tracking/tracker.py` and `BatchManager` in `src/tracking/batch.py`**

1. In `src/tracking/tracker.py`:
   Add method:
   ```python
   def reset_count(self) -> None:
       """Reset running total count to zero (e.g. on new batch start)."""
       self.total_count = 0
       self.counted_track_ids.clear()
   ```

2. Create `src/tracking/batch.py`:
   ```python
   """Batch lifecycle manager handling automated start/finish, persistence, and count reset."""

   from datetime import datetime, timezone
   import logging
   import time
   from typing import Callable, Optional

   from src.contracts import CountingBatchDTO
   from src.storage.db import LocalStorage

   logger = logging.getLogger(__name__)


   class BatchManager:
       """State machine orchestrating operational counting batches."""

       def __init__(
           self,
           line_id: str,
           storage: Optional[LocalStorage] = None,
           idle_timeout_sec: float = 30.0,
           batch_prefix: Optional[str] = None,
           on_batch_start: Optional[Callable[[str], None]] = None,
           on_batch_finish: Optional[Callable[[CountingBatchDTO], None]] = None,
       ) -> None:
           self.line_id = line_id
           self.storage = storage
           self.idle_timeout_sec = idle_timeout_sec
           self.batch_prefix = batch_prefix
           self.on_batch_start = on_batch_start
           self.on_batch_finish = on_batch_finish

           self._current_batch: Optional[CountingBatchDTO] = None
           self._last_activity_perf: float = 0.0

       @property
       def current_batch(self) -> Optional[CountingBatchDTO]:
           return self._current_batch

       @property
       def current_batch_id(self) -> Optional[str]:
           return self._current_batch.batch_id if self._current_batch else None

       @property
       def is_active(self) -> bool:
           return self._current_batch is not None and self._current_batch.status == "ACTIVE"

       def _generate_batch_id(self, now: datetime) -> str:
           timestamp_str = now.strftime("%Y%m%d_%H%M%S")
           prefix = f"{self.batch_prefix}_" if self.batch_prefix else ""
           return f"{prefix}{self.line_id}_{timestamp_str}"

       def on_activity(
           self,
           count_increment: int = 0,
           now_perf: Optional[float] = None,
           now_utc: Optional[datetime] = None,
       ) -> CountingBatchDTO:
           now_p = now_perf if now_perf is not None else time.perf_counter()
           now_u = now_utc if now_utc is not None else datetime.now(timezone.utc)

           self._last_activity_perf = now_p

           if not self.is_active:
               batch_id = self._generate_batch_id(now_u)
               self._current_batch = CountingBatchDTO(
                   batch_id=batch_id,
                   line_id=self.line_id,
                   start_time=now_u,
                   total_count=count_increment,
                   status="ACTIVE",
               )
               if self.storage:
                   try:
                       self.storage.start_batch(self._current_batch)
                   except Exception as e:
                       logger.error("Failed to persist batch start: %s", e)

               if self.on_batch_start:
                   try:
                       self.on_batch_start(batch_id)
                   except Exception as e:
                       logger.error("Error in on_batch_start callback: %s", e)

               logger.info("Batch started: %s (line: %s)", batch_id, self.line_id)
           else:
               if count_increment > 0 and self._current_batch:
                   self._current_batch.total_count += count_increment
                   if self.storage:
                       try:
                           self.storage.update_batch_count(
                               self._current_batch.batch_id, self._current_batch.total_count
                           )
                       except Exception as e:
                           logger.error("Failed to update batch count: %s", e)

           return self._current_batch

       def check_idle_timeout(
           self,
           now_perf: Optional[float] = None,
           now_utc: Optional[datetime] = None,
       ) -> Optional[CountingBatchDTO]:
           if not self.is_active or self._current_batch is None:
               return None

           now_p = now_perf if now_perf is not None else time.perf_counter()
           if (now_p - self._last_activity_perf) >= self.idle_timeout_sec:
               now_u = now_utc if now_utc is not None else datetime.now(timezone.utc)
               return self._complete_batch(now_u)
           return None

       def force_finish(self, now_utc: Optional[datetime] = None) -> Optional[CountingBatchDTO]:
           if not self.is_active or self._current_batch is None:
               return None
           now_u = now_utc if now_utc is not None else datetime.now(timezone.utc)
           return self._complete_batch(now_u)

       def _complete_batch(self, finish_utc: datetime) -> CountingBatchDTO:
           batch = self._current_batch
           batch.finish_time = finish_utc
           batch.status = "COMPLETED"

           if self.storage:
               try:
                   self.storage.finish_batch(batch.batch_id, finish_utc, batch.total_count)
               except Exception as e:
                   logger.error("Failed to persist batch finish: %s", e)

           if self.on_batch_finish:
               try:
                   self.on_batch_finish(batch)
               except Exception as e:
                   logger.error("Error in on_batch_finish callback: %s", e)

           logger.info(
               "Batch completed: %s with total count: %d (runtime: %s)",
               batch.batch_id,
               batch.total_count,
               (batch.finish_time - batch.start_time),
           )
           self._current_batch = None
           return batch
   ```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv\Scripts\python.exe -m pytest tests/test_batch_manager.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/tracking/batch.py src/tracking/tracker.py tests/test_batch_manager.py
git commit -m "feat(tracking): implement BatchManager state machine and ChickenCounter.reset_count"
```

---

### Task 5: Pipeline Integration & UI Display

**Files:**
- Modify: `main.py`
- Create: `tests/test_counter_service_batches.py`

**Interfaces:**
- Consumes: `BatchManager` from `src.tracking.batch`
- Produces: Integrated batch tracking, CLI batch indicators, and OpenCV HUD batch information

- [ ] **Step 1: Write integration tests in `tests/test_counter_service_batches.py`**

Create `tests/test_counter_service_batches.py`:

```python
from datetime import datetime, timezone
import pytest
from unittest.mock import MagicMock
from main import CounterService
from src.contracts import PipelineConfig, CountEventDTO
from src.storage.db import LocalStorage


@pytest.fixture
def test_service(tmp_path):
    db_file = tmp_path / "service_test.db"
    storage = LocalStorage(db_path=str(db_file))
    storage.init_db()

    cfg = PipelineConfig(
        capture_mode="direct",
        tracker_type="centroid",
        batch_idle_timeout_sec=2.0,
    )
    svc = CounterService(
        video_source="test.mp4",
        config=cfg,
        line_id="line-test",
        storage=storage,
        cli_display=False,
    )
    return svc


def test_service_initializes_batch_manager(test_service):
    assert hasattr(test_service, "batch_manager")
    assert test_service.batch_manager.line_id == "line-test"
    assert not test_service.batch_manager.is_active


def test_service_resets_count_on_new_batch(test_service):
    # Simulate existing count
    test_service.counter.total_count = 15

    # Simulate activity starting a batch
    batch = test_service.batch_manager.on_activity(count_increment=0)
    assert test_service.counter.total_count == 0
    assert test_service.batch_manager.is_active


def test_service_force_finish_on_stop(test_service):
    test_service.batch_manager.on_activity(count_increment=5)
    assert test_service.batch_manager.is_active

    test_service.stop()
    assert not test_service.batch_manager.is_active
    unsynced = test_service.storage.get_unsynced_batches(line_id="line-test")
    assert len(unsynced) == 1
    assert unsynced[0].total_count == 5
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv\Scripts\python.exe -m pytest tests/test_counter_service_batches.py -v`
Expected: FAIL because `CounterService` does not yet instantiate `BatchManager`.

- [ ] **Step 3: Update `main.py` with batch integration**

In `main.py`:
1. Import `BatchManager` from `src.tracking.batch`.
2. In `CounterService.__init__`:
   - Initialize `self.batch_manager = BatchManager(...)`:
     ```python
     self.batch_manager = BatchManager(
         line_id=self.line_id,
         storage=self.storage,
         idle_timeout_sec=self.idle_timeout_sec,
         batch_prefix=getattr(self.config, "batch_prefix", None),
         on_batch_start=self._on_batch_start,
     )
     ```
   - Define `_on_batch_start(self, batch_id: str)`:
     ```python
     def _on_batch_start(self, batch_id: str) -> None:
         logger.info("New batch started: %s. Resetting counter to 0.", batch_id)
         self.counter.reset_count()
     ```
3. In `run()` loop:
   - When detections or crossing events occur:
     ```python
     if len(detections) > 0 or len(events) > 0:
         self.batch_manager.on_activity(count_increment=len(events))
     ```
   - In event tagging loop:
     ```python
     for ev in events:
         ev.batch_id = self.batch_manager.current_batch_id
         self.storage.insert_event(self.line_id, ev)
     ```
   - In loop idle / periodic check:
     ```python
     self.batch_manager.check_idle_timeout(now_perf=time.perf_counter())
     ```
4. In `stop()`:
   - Call `self.batch_manager.force_finish()`.
5. In CLI status update:
   - Add batch tag showing `Batch: <batch_id>` (or `[IDLE]`):
     ```python
     b_id = self.batch_manager.current_batch_id or "IDLE"
     ```
6. In `render_annotated_frame()`:
   - In HUD box overlay, display:
     ```python
     b_id = self.batch_manager.current_batch_id or "IDLE"
     hud_text = (
         f"Batch: {b_id} | "
         f"Count: {self.counter.total_count} | "
         f"FPS: {fps:.1f}"
     )
     ```

- [ ] **Step 4: Run all tests to verify they pass**

Run: `.venv\Scripts\python.exe -m pytest -v`
Expected: All tests pass across the entire suite.

- [ ] **Step 5: Commit**

```bash
git add main.py tests/test_counter_service_batches.py
git commit -m "feat(pipeline): integrate BatchManager with CounterService, CLI, and visual preview"
```

---

### Task 6: Full Verification & End-to-End Test

**Files:**
- Run: Entire test suite
- Verify: Database tables and sample video processing

- [ ] **Step 1: Run complete test suite**

Run: `.venv\Scripts\python.exe -m pytest tests/ -v`
Expected: 100% test pass rate with no regressions.

- [ ] **Step 2: Commit any final cleanup or docs**

```bash
git status
```
Confirm working tree is clean.
