# Counting Batches Design Specification

## 1. Overview
In poultry carcass counting operations, processing runs in operational batches (e.g., specific truck loads, client orders, or conveyor runs). This specification defines the counting batch lifecycle for `counter-service`. Every batch has an explicit start time, finish time, total count, and unique identifier.

Counts are tracked strictly per-batch: when a new batch starts, the running counter resets to zero. Completed batches are persisted locally in SQLite (`outbox_batches`) and synced to the central API alongside count events and aggregates.

## 2. Requirements & Operational Flow

### 2.1 Batch Lifecycle
1. **Triggering Start**: Automatic by activity. When the conveyor starts moving birds and either YOLO carcass detections (`len(detections) > 0`) or line crossing events occur while in an `IDLE` state:
   - A new batch is created with an ID formatted as `<line_id>_YYYYMMDD_HHMMSS`.
   - The counter is reset to `0`.
   - The batch is persisted to the local SQLite database with status `ACTIVE`.
2. **Accumulating Count**:
   - Each detected carcass crossing the virtual line increments the active batch count.
   - Every `CountEventDTO` emitted during this time is tagged with the current `batch_id`.
3. **Triggering Finish**:
   - When no carcass detections or crossing events occur for a configurable duration (`idle_timeout_sec`, default: 30.0s), the batch automatically finishes.
   - The batch is closed with `finish_time = utcnow()` and status `COMPLETED`.
   - If the edge service receives a shutdown signal (SIGINT/SIGTERM), any active batch is cleanly force-finished.
4. **Subsequent Batches**:
   - The system returns to `IDLE` state.
   - When birds appear again, a new batch starts, generating a new batch ID and resetting the count to `0`.

## 3. Data Contracts (`src/contracts.py`)

### 3.1 `CountingBatchDTO`
```python
from datetime import datetime
from typing import Literal, Optional
from pydantic import BaseModel, Field

class CountingBatchDTO(BaseModel):
    batch_id: str
    line_id: str = "line-1"
    start_time: datetime
    finish_time: Optional[datetime] = None
    total_count: int = Field(default=0, ge=0)
    status: Literal["ACTIVE", "COMPLETED"] = "ACTIVE"
```

### 3.2 Updates to Existing Contracts
- **`CountEventDTO`**:
  - Add optional `batch_id: Optional[str] = None`.
- **`SyncBatchPayload`**:
  - Add `batches: list[CountingBatchDTO] = Field(default_factory=list)` to support dispatching completed batches to upstream backend.
- **`PipelineConfig`**:
  - Add `batch_idle_timeout_sec: float = 30.0`.
  - Add `batch_prefix: Optional[str] = None`.

## 4. SQLite Outbox Storage (`src/storage/db.py`)

### 4.1 Schema Migration
A new table `outbox_batches` is created:
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

Add `batch_id` column to `outbox_events` if not present:
```sql
ALTER TABLE outbox_events ADD COLUMN batch_id TEXT NULL;
```

### 4.2 LocalStorage Methods
- `start_batch(batch: CountingBatchDTO) -> int`: Persists a new active batch row.
- `update_batch_count(batch_id: str, count: int) -> None`: Updates the running count for the active batch.
- `finish_batch(batch_id: str, finish_time: datetime, total_count: int) -> bool`: Updates `finish_time`, `total_count`, and sets `status = 'COMPLETED'`.
- `get_active_batch(line_id: str) -> Optional[CountingBatchDTO]`: Recovers any currently open batch across process restarts.
- `get_unsynced_batches(limit: int = 50) -> list[CountingBatchDTO]`: Fetches completed, unsynced batches for upstream dispatch.
- `mark_batches_synced(batch_ids: list[str], synced_at: datetime) -> int`: Marks dispatched batches as synced.

## 5. `BatchManager` Component (`src/tracking/batch.py`)

A dedicated state-machine class responsible for batch transitions:
- Manages states: `IDLE`, `ACTIVE`.
- Formats IDs: `<line_id>_YYYYMMDD_HHMMSS`.
- Handles `on_activity(count_increment: int, now_utc: Optional[datetime])`:
  - When IDLE: generates batch, triggers `on_batch_start` callback (resets counter to 0), writes to storage.
  - When ACTIVE: increments batch count and updates `last_activity_time`.
- Handles `check_idle_timeout(now_perf: float, now_utc: Optional[datetime])`:
  - If `ACTIVE` and `(now_perf - last_activity_time) >= idle_timeout_sec`, finishes batch and updates storage.
- Handles `force_finish(now_utc: Optional[datetime])`:
  - Closes any active batch on service shutdown.

## 6. Pipeline & Service Integration (`main.py`)

### 6.1 Integration Flow
- `CounterService.__init__`:
  - Initializes `BatchManager` with `line_id`, `storage`, and `idle_timeout_sec`.
  - Attaches `on_batch_start` callback to invoke `self.counter.reset_count()` and reset live counters.
- Processing loop:
  - On detections or line-crossings: invokes `batch_manager.on_activity(count_increment=len(events))`.
  - Tags outgoing `CountEventDTO` with `batch_manager.current_batch_id`.
  - Regularly calls `batch_manager.check_idle_timeout(time.perf_counter())`.
- Shutdown:
  - `CounterService.stop()` calls `batch_manager.force_finish()`.

### 6.2 UI & Dashboard Updates
- **CLI Dashboard (`rich` table)**:
  - Displays current **Batch ID** (or `IDLE (Waiting for birds)`).
  - Displays active **Batch Duration** (`MM:SS`).
  - Total Count accurately represents the current batch count.
- **OpenCV Preview Window**:
  - Displays `Batch: <batch_id>` and `Count: <batch_count>`.

## 7. Sync Dispatcher Integration (`src/storage/sync.py`)

- `OutboxSyncDispatcher.sync_cycle()`:
  - Retrieves unsynced batches via `storage.get_unsynced_batches()`.
  - Bundles them into `SyncBatchPayload(..., batches=batches)`.
  - On successful HTTP sync confirmation, marks them as synced with `storage.mark_batches_synced()`.

## 8. Testing Strategy

1. **`tests/test_batch_manager.py`**:
   - Unit tests for batch start on detection/crossing.
   - Unit tests for ID formatting `<line_id>_YYYYMMDD_HHMMSS`.
   - Unit tests for idle timeout expiration and completion.
   - Unit tests verifying counter reset callback trigger on batch start.
   - Unit tests for graceful `force_finish()`.
2. **`tests/test_storage_batches.py`**:
   - Schema creation and column addition verification.
   - Batch CRUD operations and status transitions (`ACTIVE` -> `COMPLETED`).
   - Sync retrieval and marking as synced.
   - Event insertion with associated `batch_id`.
3. **End-to-End Pipeline Smoke Test**:
   - Verify pipeline starts, processes mock frames or video clip, resets count on new batch, and writes batch record to SQLite.
