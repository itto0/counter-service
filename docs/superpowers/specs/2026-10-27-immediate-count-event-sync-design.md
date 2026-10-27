# Technical Design Spec: Immediate Count Event Outbox Synchronization

**Date:** 2026-10-27  
**Status:** Approved  
**Topic:** Event-Triggered Outbox Dispatch for Real-Time Dashboard Updates at High Speed (150 BPM)  

---

## 1. Goal & Overview

In the edge counter pipeline, when an object (e.g. poultry carcass) crosses the detection line, a count event is recorded locally in the SQLite WAL outbox (`storage.insert_event`). Currently, the background worker polls and dispatches records to the central API on a fixed periodic interval (`sync_interval_sec = 5.0`).

While periodic batching protects the camera pipeline from network jitter, a 5-second polling delay is too sluggish for live monitoring dashboards where operators expect real-time (<500ms, typically <100ms) tally increments. At the same time, naive synchronous HTTP calls inside the camera frame loop would introduce severe frame drops whenever network latency spikes.

This design introduces an **Event-Triggered Immediate Outbox Dispatch** mechanism:
1. **Decoupled Event Triggering:** When a count event is detected, `CounterService` records it to SQLite and immediately signals a dedicated thread event (`_sync_trigger.set()`).
2. **Instant Worker Wake-up:** The background sync thread immediately wakes up and calls `dispatcher.sync_once()`, dispatching the count event to `/api/v1/sync/batch` within milliseconds.
3. **High-Speed 150 BPM Throughput:** Supports conveyor throughputs exceeding 150 BPM (~2.5 birds/sec) with natural coalescing when birds cross in rapid succession.
4. **Network Backoff Protection:** When network connectivity is lost, exponential backoff suppresses immediate wake-ups, preventing high-frequency retry loops while safely buffering events in SQLite.
5. **Configurable Toggle:** An optional flag (`enable_immediate_sync: bool = True` / `--no-immediate-sync`) allows reverting to pure periodic batching if desired.

---

## 2. Requirements & Constraints

### Functional Requirements
- **FR-1:** Add configuration option `enable_immediate_sync: bool = True` to `CounterService.__init__`.
- **FR-2:** Add CLI arguments:
  - `--immediate-sync` (default: True, action="store_true" or store_boolean)
  - `--no-immediate-sync` (action="store_false", dest="immediate_sync")
- **FR-3:** Immediate Dispatch Trigger:
  - Add `trigger_sync()` method to `CounterService` that signals `_sync_trigger.set()` if `enable_immediate_sync` is True and `dispatcher` is active.
  - When `events` list is non-empty after `counter.update()`, invoke `self.trigger_sync()`.
- **FR-4:** Worker Wake Logic:
  - Update `_sync_worker()` to wait on `_sync_trigger` rather than a static `_stop_event.wait(timeout=sleep_time)`.
  - When `dispatcher.backoff_sec > 1.0`, the worker must honor the backoff duration via `_stop_event.wait(timeout=backoff)` and suppress immediate triggers.
  - When network is healthy, the worker waits up to `sync_interval_sec` for fallback triggers (handling health and aggregate telemetry).
- **FR-5:** Prompt Shutdown:
  - `stop()` must set both `_stop_event` and `_sync_trigger` to ensure the sync thread terminates immediately without waiting out the interval timeout.

### Non-Functional & Architecture Constraints
- **Zero Frame Drops:** Frame capture and ONNX/YOLO inference run strictly decoupled from network operations. Setting a `threading.Event` takes <1 microsecond.
- **Data Resilience:** SQLite WAL remains the single source of truth for outbox persistence. No counts are held only in volatile memory.
- **Backward Compatibility:** Default API schemas, database schemas, and contracts remain 100% identical.

---

## 3. Architecture & Data Flow

```
                     CAMERA FRAME PIPELINE
                               │
                       Detect & Track
                               │
                       Count Event Occurred
                               │
               ┌───────────────┴───────────────┐
               ▼                               ▼
    storage.insert_event()            trigger_sync()
    (Write to SQLite WAL)         (Sets _sync_trigger Event)
               │                               │
               │                               ▼
               │                    ┌──────────────────────┐
               │                    │ Background Sync Loop │
               │                    └──────────┬───────────┘
               │                               │ (Wakes immediately)
               ▼                               ▼
      [ counter_outbox.db ] ◄────── dispatcher.sync_once()
                                               │
                                               ▼
                                      POST /api/v1/sync/batch
                                    (via httpx keep-alive pool)
```

---

## 4. Component Changes

### 4.1 `CounterService` in `main.py`

1. **State Initialization:**
   ```python
   self.enable_immediate_sync = enable_immediate_sync
   self._sync_trigger = threading.Event()
   ```

2. **Trigger Method:**
   ```python
   def trigger_sync(self) -> None:
       """Signal the background sync thread to wake and flush outbox immediately."""
       if self.enable_immediate_sync and self.dispatcher is not None:
           self._sync_trigger.set()
   ```

3. **Count Loop Hook:**
   ```python
   events = self.counter.update(detections, timestamp=time.time())
   for ev in events:
       self.storage.insert_event(self.line_id, ev)
       # ...
   if events:
       self.trigger_sync()
   ```

4. **Sync Worker Wake Loop (`_sync_worker`):**
   ```python
   def _sync_worker(self) -> None:
       if self.dispatcher is None:
           return
       logger.info("Background sync worker started for line '%s'", self.line_id)
       while not self._stop_event.is_set():
           self._sync_trigger.clear()
           try:
               synced = self.dispatcher.sync_once()
               if synced > 0:
                   logger.debug("Background synced %d outbox records to central API", synced)
           except Exception as e:
               logger.warning("Error in background sync worker: %s", e)

           backoff = getattr(self.dispatcher, "backoff_sec", 1.0)
           if backoff > 1.0:
               # Network failure / backoff: wait strictly on stop event
               self._stop_event.wait(timeout=backoff)
           else:
               # Healthy: wait for next count event or fallback heartbeat interval
               self._sync_trigger.wait(timeout=self.sync_interval_sec)

       logger.info("Background sync worker stopped for line '%s'", self.line_id)
   ```

5. **Shutdown Cleanup (`stop`):**
   ```python
   self._stop_event.set()
   self._sync_trigger.set()
   ```

6. **CLI Parser Flags:**
   ```python
   parser.add_argument(
       "--immediate-sync",
       dest="immediate_sync",
       action="store_true",
       default=True,
       help="Dispatch count events immediately via background worker (default: True).",
   )
   parser.add_argument(
       "--no-immediate-sync",
       dest="immediate_sync",
       action="store_false",
       help="Disable immediate event dispatch and use periodic sync interval only.",
   )
   ```

---

## 5. Verification & Testing

1. **Unit Test: Immediate Wake (`tests/test_sync.py`):**
   - Configure a large `sync_interval_sec = 60.0`.
   - Call `trigger_sync()` and verify `sync_once()` executes within <50ms.
2. **Unit Test: Backoff Suppression (`tests/test_sync.py`):**
   - With `backoff_sec > 1.0`, verify calls to `trigger_sync()` do not trigger immediate network requests.
3. **Unit Test: Toggle Behavior (`tests/test_main.py`):**
   - Verify `enable_immediate_sync=False` respects only `sync_interval_sec`.
4. **Full Regression Suite:**
   - Run `pytest` across all existing tests to confirm no regressions.
