"""Batch lifecycle manager handling automated start/finish, persistence, and count reset."""

from collections.abc import Callable
from datetime import datetime, timezone
import logging
import threading
import time
from typing import Optional

try:
    from src.contracts import CountingBatchDTO
except ImportError:
    from shared.contracts import CountingBatchDTO

try:
    from src.storage.db import LocalStorage
except ImportError:
    from storage.db import LocalStorage

logger = logging.getLogger("counter.batch")

__all__ = ["BatchManager"]


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
        """Initialize BatchManager state machine.

        Args:
            line_id: Identifier of the monitored production line.
            storage: Optional LocalStorage instance for batch persistence.
            idle_timeout_sec: Duration in seconds of inactivity before auto-completing a batch.
            batch_prefix: Optional custom prefix prepended to generated batch IDs.
            on_batch_start: Optional callback invoked with batch_id when a batch starts.
            on_batch_finish: Optional callback invoked with CountingBatchDTO when a batch completes.
        """
        self.line_id = line_id
        self.storage = storage
        self.idle_timeout_sec = float(idle_timeout_sec)
        self.batch_prefix = batch_prefix
        self.on_batch_start = on_batch_start
        self.on_batch_finish = on_batch_finish

        self._current_batch: Optional[CountingBatchDTO] = None
        self._last_activity_perf: float = 0.0
        self._lock = threading.RLock()

    @property
    def current_batch(self) -> Optional[CountingBatchDTO]:
        """Return the currently active CountingBatchDTO or None."""
        with self._lock:
            return self._current_batch

    @property
    def current_batch_id(self) -> Optional[str]:
        """Return the batch ID of the active batch or None."""
        with self._lock:
            return self._current_batch.batch_id if self._current_batch else None

    @property
    def is_active(self) -> bool:
        """Return True if a batch is currently active, False otherwise."""
        with self._lock:
            return self._current_batch is not None and self._current_batch.status == "ACTIVE"

    def _generate_batch_id(self, now: datetime) -> str:
        """Generate a formatted batch ID: [<prefix>_]<line_id>_YYYYMMDD_HHMMSS."""
        timestamp_str = now.strftime("%Y%m%d_%H%M%S")
        prefix = f"{self.batch_prefix.rstrip('_')}_" if self.batch_prefix else ""
        return f"{prefix}{self.line_id}_{timestamp_str}"

    def on_activity(
        self,
        count_increment: int = 0,
        now_perf: Optional[float] = None,
        now_utc: Optional[datetime] = None,
    ) -> CountingBatchDTO:
        """Record activity, starting a new batch if idle, and incrementing count.

        Args:
            count_increment: Number to add to the batch total_count.
            now_perf: Optional performance counter timestamp (seconds).
            now_utc: Optional UTC datetime for batch timestamping.

        Returns:
            The current active CountingBatchDTO.
        """
        count_increment = max(0, count_increment)
        now_p = float(now_perf) if now_perf is not None else time.perf_counter()
        now_u = now_utc if now_utc is not None else datetime.now(timezone.utc)

        callback = None
        b_id = None

        with self._lock:
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

                callback = self.on_batch_start
                b_id = batch_id
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

            batch = self._current_batch

        if callback and b_id:
            try:
                callback(b_id)
            except Exception as e:
                logger.error("Error in on_batch_start callback: %s", e)

        return batch

    def check_idle_timeout(
        self,
        now_perf: Optional[float] = None,
        now_utc: Optional[datetime] = None,
    ) -> Optional[CountingBatchDTO]:
        """Check if inactive duration exceeds idle_timeout_sec; if so, complete batch.

        Args:
            now_perf: Optional performance counter timestamp (seconds).
            now_utc: Optional UTC datetime for batch completion timestamp.

        Returns:
            The completed CountingBatchDTO if timed out, None otherwise.
        """
        now_p = float(now_perf) if now_perf is not None else time.perf_counter()
        now_u = now_utc if now_utc is not None else datetime.now(timezone.utc)

        with self._lock:
            if not self.is_active or self._current_batch is None:
                return None
            if (now_p - self._last_activity_perf) < self.idle_timeout_sec:
                return None

        return self._complete_batch(now_u)

    def force_finish(
        self,
        now_utc: Optional[datetime] = None,
    ) -> Optional[CountingBatchDTO]:
        """Force completion of the current active batch (e.g. on service shutdown).

        Args:
            now_utc: Optional UTC datetime for batch completion timestamp.

        Returns:
            The completed CountingBatchDTO if a batch was active, None otherwise.
        """
        now_u = now_utc if now_utc is not None else datetime.now(timezone.utc)
        return self._complete_batch(now_u)

    def _complete_batch(self, finish_utc: datetime) -> Optional[CountingBatchDTO]:
        """Mark the current batch as COMPLETED, persist to storage, and notify callbacks."""
        callback = None
        with self._lock:
            batch = self._current_batch
            if batch is None:
                return None

            self._current_batch = None
            batch.finish_time = finish_utc
            batch.status = "COMPLETED"

            if self.storage:
                try:
                    self.storage.finish_batch(batch.batch_id, finish_utc, batch.total_count)
                except Exception as e:
                    logger.error("Failed to persist batch finish: %s", e)

            callback = self.on_batch_finish

        if callback:
            try:
                callback(batch)
            except Exception as e:
                logger.error("Error in on_batch_finish callback: %s", e)

        try:
            start_t = batch.start_time
            if finish_utc.tzinfo is None and start_t.tzinfo is not None:
                finish_t = finish_utc.replace(tzinfo=timezone.utc)
            else:
                finish_t = finish_utc
            runtime_str = str(finish_t - start_t)
        except Exception:
            runtime_str = "unknown"

        logger.info(
            "Batch completed: %s with total count: %d (runtime: %s)",
            batch.batch_id,
            batch.total_count,
            runtime_str,
        )
        return batch
