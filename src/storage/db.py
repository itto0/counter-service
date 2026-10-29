"""SQLite outbox storage with WAL mode for local edge resilience."""

from datetime import datetime, timezone
from pathlib import Path
import sqlite3
import threading
from typing import Any
try:
    from src.contracts import (
        CountAggregateDTO,
        CountEventDTO,
        CountingBatchDTO,
        HealthSampleDTO,
        SyncBatchPayload,
    )
except ImportError:
    try:
        from src.contracts import (
            CountAggregateDTO,
            CountEventDTO,
            CountingBatchDTO,
            HealthSampleDTO,
            SyncBatchPayload,
        )
    except ImportError:
        from shared.contracts import (
            CountAggregateDTO,
            CountEventDTO,
            CountingBatchDTO,
            HealthSampleDTO,
            SyncBatchPayload,
        )


class LocalStorage:
    """Manages local SQLite WAL outbox queue for events, aggregates, and health metrics."""

    def __init__(self, db_path: str = "counter_outbox.db") -> None:
        self.db_path = db_path
        self._lock = threading.RLock()

    def _get_connection(self) -> sqlite3.Connection:
        if self.db_path != ":memory:":
            parent = Path(self.db_path).parent
            if str(parent) not in ("", "."):
                parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        return conn

    def init_db(self) -> None:
        """Initialize database tables and indices with WAL journal mode."""
        with self._lock:
            conn = self._get_connection()
            try:
                with conn:
                    conn.executescript("""
                    CREATE TABLE IF NOT EXISTS outbox_events (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        line_id TEXT NOT NULL,
                        track_id INTEGER NOT NULL,
                        ts_utc TEXT NOT NULL,
                        confidence REAL NOT NULL,
                        batch_id TEXT NULL,
                        synced_at TEXT NULL
                    );

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

                    CREATE TABLE IF NOT EXISTS outbox_aggregates (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        line_id TEXT NOT NULL,
                        window_start TEXT NOT NULL,
                        window_end TEXT NOT NULL,
                        count INTEGER NOT NULL,
                        rate_per_min REAL NOT NULL,
                        confidence_avg REAL NOT NULL,
                        status TEXT NOT NULL DEFAULT 'NORMAL',
                        synced_at TEXT NULL
                    );

                    CREATE TABLE IF NOT EXISTS health_samples (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        line_id TEXT NOT NULL,
                        ts_utc TEXT NOT NULL,
                        fps REAL NOT NULL,
                        latency_p95_ms REAL NOT NULL,
                        cpu_pct REAL NOT NULL,
                        mem_pct REAL NOT NULL,
                        stream_state TEXT NOT NULL,
                        model_name TEXT NOT NULL DEFAULT 'yolo26n',
                        model_version TEXT NOT NULL DEFAULT 'v1.0',
                        accuracy_pct REAL NULL,
                        synced_at TEXT NULL
                    );

                    CREATE INDEX IF NOT EXISTS idx_outbox_events_synced ON outbox_events(synced_at);
                    CREATE INDEX IF NOT EXISTS idx_outbox_events_line ON outbox_events(line_id);

                    CREATE INDEX IF NOT EXISTS idx_outbox_batches_synced ON outbox_batches(synced_at);
                    CREATE INDEX IF NOT EXISTS idx_outbox_batches_line ON outbox_batches(line_id);

                    CREATE INDEX IF NOT EXISTS idx_outbox_aggregates_synced ON outbox_aggregates(synced_at);
                    CREATE INDEX IF NOT EXISTS idx_outbox_aggregates_line ON outbox_aggregates(line_id);

                    CREATE INDEX IF NOT EXISTS idx_health_samples_synced ON health_samples(synced_at);
                    CREATE INDEX IF NOT EXISTS idx_health_samples_line ON health_samples(line_id);
                    """)
                    try:
                        conn.execute("ALTER TABLE outbox_events ADD COLUMN batch_id TEXT NULL")
                    except sqlite3.OperationalError:
                        pass
                    try:
                        conn.execute("ALTER TABLE health_samples ADD COLUMN accuracy_pct REAL NULL")
                    except sqlite3.OperationalError:
                        pass
            finally:
                conn.close()

    def insert_event(self, line_id: str, event: CountEventDTO) -> int:
        """Insert a count event into outbox_events table."""
        with self._lock:
            conn = self._get_connection()
            try:
                with conn:
                    cursor = conn.execute(
                        """
                        INSERT INTO outbox_events (line_id, track_id, ts_utc, confidence, batch_id)
                        VALUES (?, ?, ?, ?, ?)
                        """,
                        (
                            line_id,
                            event.track_id,
                            event.ts_utc.isoformat(),
                            event.confidence,
                            getattr(event, "batch_id", None),
                        ),
                    )
                    return cursor.lastrowid or 0
            finally:
                conn.close()

    def insert_count_event(self, event: CountEventDTO, line_id: str | None = None) -> int:
        """Insert a count event into outbox_events, initializing db if needed."""
        self.init_db()
        target_line = line_id or getattr(event, "line_id", None) or "line-1"
        return self.insert_event(target_line, event)

    def close(self) -> None:
        """Close storage resources (no-op for connection-per-call SQLite)."""
        pass

    def insert_aggregate(self, line_id: str, aggregate: CountAggregateDTO) -> int:
        """Insert a window aggregate into outbox_aggregates table."""
        with self._lock:
            conn = self._get_connection()
            try:
                with conn:
                    cursor = conn.execute(
                        """
                        INSERT INTO outbox_aggregates (
                            line_id, window_start, window_end, count, rate_per_min,
                            confidence_avg, status
                        ) VALUES (?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            line_id,
                            aggregate.window_start.isoformat(),
                            aggregate.window_end.isoformat(),
                            aggregate.count,
                            aggregate.rate_per_min,
                            aggregate.confidence_avg,
                            aggregate.status,
                        ),
                    )
                    return cursor.lastrowid or 0
            finally:
                conn.close()

    def insert_health(
        self,
        line_id: str,
        health: HealthSampleDTO,
        ts_utc: datetime | None = None,
    ) -> int:
        """Insert a health sample into health_samples table."""
        ts = ts_utc or datetime.now(timezone.utc)
        with self._lock:
            conn = self._get_connection()
            try:
                with conn:
                    cursor = conn.execute(
                        """
                        INSERT INTO health_samples (
                            line_id, ts_utc, fps, latency_p95_ms, cpu_pct, mem_pct,
                            stream_state, model_name, model_version, accuracy_pct
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            line_id,
                            ts.isoformat(),
                            health.fps,
                            health.latency_p95_ms,
                            health.cpu_pct,
                            health.mem_pct,
                            health.stream_state,
                            health.model_name,
                            health.model_version,
                            health.accuracy_pct,
                        ),
                    )
                    return cursor.lastrowid or 0
            finally:
                conn.close()

    def get_unsynced_events(
        self, limit: int = 100, line_id: str | None = None
    ) -> list[dict[str, Any]]:
        """Retrieve unsynced count events up to limit, optionally filtered by line_id."""
        query = "SELECT id, line_id, track_id, ts_utc, confidence, batch_id FROM outbox_events WHERE synced_at IS NULL"
        params: list[Any] = []
        if line_id is not None:
            query += " AND line_id = ?"
            params.append(line_id)
        query += " ORDER BY id ASC LIMIT ?"
        params.append(limit)

        with self._lock:
            conn = self._get_connection()
            try:
                cursor = conn.execute(query, params)
                return [dict(row) for row in cursor.fetchall()]
            finally:
                conn.close()

    def get_unsynced_aggregates(
        self, limit: int = 100, line_id: str | None = None
    ) -> list[dict[str, Any]]:
        """Retrieve unsynced window aggregates up to limit, optionally filtered by line_id."""
        query = (
            "SELECT id, line_id, window_start, window_end, count, rate_per_min, confidence_avg, status "
            "FROM outbox_aggregates WHERE synced_at IS NULL"
        )
        params: list[Any] = []
        if line_id is not None:
            query += " AND line_id = ?"
            params.append(line_id)
        query += " ORDER BY id ASC LIMIT ?"
        params.append(limit)

        with self._lock:
            conn = self._get_connection()
            try:
                cursor = conn.execute(query, params)
                return [dict(row) for row in cursor.fetchall()]
            finally:
                conn.close()

    def get_unsynced_health(
        self, limit: int = 100, line_id: str | None = None
    ) -> list[dict[str, Any]]:
        """Retrieve unsynced health samples up to limit, optionally filtered by line_id."""
        query = (
            "SELECT * "
            "FROM health_samples WHERE synced_at IS NULL"
        )
        params: list[Any] = []
        if line_id is not None:
            query += " AND line_id = ?"
            params.append(line_id)
        query += " ORDER BY id ASC LIMIT ?"
        params.append(limit)

        with self._lock:
            conn = self._get_connection()
            try:
                cursor = conn.execute(query, params)
                return [dict(row) for row in cursor.fetchall()]
            finally:
                conn.close()

    def mark_events_synced(self, event_ids: list[int]) -> None:
        """Mark the specified event IDs as synced."""
        if not event_ids:
            return
        placeholders = ",".join("?" for _ in event_ids)
        now_iso = datetime.now(timezone.utc).isoformat()
        with self._lock:
            conn = self._get_connection()
            try:
                with conn:
                    conn.execute(
                        f"UPDATE outbox_events SET synced_at = ? WHERE id IN ({placeholders})",
                        [now_iso, *event_ids],
                    )
            finally:
                conn.close()

    def mark_aggregates_synced(self, aggregate_ids: list[int]) -> None:
        """Mark the specified aggregate IDs as synced."""
        if not aggregate_ids:
            return
        placeholders = ",".join("?" for _ in aggregate_ids)
        now_iso = datetime.now(timezone.utc).isoformat()
        with self._lock:
            conn = self._get_connection()
            try:
                with conn:
                    conn.execute(
                        f"UPDATE outbox_aggregates SET synced_at = ? WHERE id IN ({placeholders})",
                        [now_iso, *aggregate_ids],
                    )
            finally:
                conn.close()

    def mark_health_synced(self, health_ids: list[int]) -> None:
        """Mark the specified health sample IDs as synced."""
        if not health_ids:
            return
        placeholders = ",".join("?" for _ in health_ids)
        now_iso = datetime.now(timezone.utc).isoformat()
        with self._lock:
            conn = self._get_connection()
            try:
                with conn:
                    conn.execute(
                        f"UPDATE health_samples SET synced_at = ? WHERE id IN ({placeholders})",
                        [now_iso, *health_ids],
                    )
            finally:
                conn.close()

    def start_batch(self, batch: CountingBatchDTO) -> int:
        """Insert a new active counting batch into outbox_batches table."""
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

    def update_batch_count(self, batch_id: str, count: int) -> None:
        """Update running total_count for an active counting batch."""
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

    def finish_batch(self, batch_id: str, finish_time: datetime, total_count: int) -> bool:
        """Mark a batch as COMPLETED with finish_time and final total_count."""
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

    def get_active_batch(self, line_id: str) -> CountingBatchDTO | None:
        """Retrieve the latest ACTIVE batch for line_id if present."""
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

    def get_unsynced_batches(
        self, limit: int = 50, line_id: str | None = None
    ) -> list[CountingBatchDTO]:
        """Retrieve unsynced completed batches up to limit, optionally filtered by line_id."""
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

    def mark_batches_synced(
        self, batch_ids: list[str], synced_at: datetime | None = None
    ) -> int:
        """Mark the specified batch IDs as synced."""
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

    def get_unsynced_batch(
        self, line_id: str, limit: int = 100
    ) -> tuple[SyncBatchPayload | None, dict[str, list[Any]]]:
        """
        Package pending items for line_id into a SyncBatchPayload and return their IDs.

        Returns (payload, {"event_ids": [...], "aggregate_ids": [...], "health_ids": [...], "batch_ids": [...]})
        If nothing is pending, returns (None, empty_ids_dict).
        """
        with self._lock:
            unsynced_events = self.get_unsynced_events(limit=limit, line_id=line_id)
            unsynced_aggs = self.get_unsynced_aggregates(limit=limit, line_id=line_id)
            unsynced_health = self.get_unsynced_health(limit=1, line_id=line_id)
            unsynced_batches = self.get_unsynced_batches(limit=limit, line_id=line_id)

            event_ids = [row["id"] for row in unsynced_events]
            agg_ids = [row["id"] for row in unsynced_aggs]
            health_ids = [row["id"] for row in unsynced_health]
            batch_ids = [b.batch_id for b in unsynced_batches]

            empty_ids: dict[str, list[int]] = {
                "event_ids": [],
                "aggregate_ids": [],
                "health_ids": [],
            }

            if not event_ids and not agg_ids and not health_ids and not batch_ids:
                return None, empty_ids

            events = [
                CountEventDTO(
                    track_id=row["track_id"],
                    ts_utc=datetime.fromisoformat(row["ts_utc"]),
                    confidence=row["confidence"],
                    line_id=row["line_id"] if "line_id" in row.keys() else line_id,
                    batch_id=row["batch_id"] if "batch_id" in row.keys() else None,
                )
                for row in unsynced_events
            ]

            aggregates = [
                CountAggregateDTO(
                    window_start=datetime.fromisoformat(row["window_start"]),
                    window_end=datetime.fromisoformat(row["window_end"]),
                    count=row["count"],
                    rate_per_min=row["rate_per_min"],
                    confidence_avg=row["confidence_avg"],
                    status=row["status"],
                )
                for row in unsynced_aggs
            ]

            health_dto: HealthSampleDTO | None = None
            if unsynced_health:
                h = unsynced_health[0]
                accuracy_val = (
                    h["accuracy_pct"]
                    if "accuracy_pct" in h.keys() and h["accuracy_pct"] is not None
                    else None
                )
                health_dto = HealthSampleDTO(
                    fps=h["fps"],
                    latency_p95_ms=h["latency_p95_ms"],
                    cpu_pct=h["cpu_pct"],
                    mem_pct=h["mem_pct"],
                    stream_state=h["stream_state"],
                    model_name=h["model_name"],
                    model_version=h["model_version"],
                    accuracy_pct=accuracy_val,
                )

            payload = SyncBatchPayload(
                line_id=line_id,
                events=events,
                aggregates=aggregates,
                health=health_dto,
                batches=unsynced_batches,
            )

            return payload, {
                "event_ids": event_ids,
                "aggregate_ids": agg_ids,
                "health_ids": health_ids,
                "batch_ids": batch_ids,
            }

