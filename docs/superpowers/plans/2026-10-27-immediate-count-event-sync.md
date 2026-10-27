# Immediate Count Event Outbox Synchronization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Enable event-triggered immediate outbox dispatch when poultry count events occur so live dashboards update with sub-second (<100ms) latency at conveyor speeds up to 150 BPM, while maintaining non-blocking camera frame processing and network backoff protection.

**Architecture:** Decouple SQLite WAL event insertion from network delivery using a dedicated `threading.Event` (`_sync_trigger`). When detection/tracking identifies a crossing event, the frame loop immediately signals `_sync_trigger.set()`. The background sync worker thread wakes instantly to flush the outbox via `dispatcher.sync_once()`. When network backoff is active (`backoff_sec > 1.0`), immediate wake-ups are suppressed and backoff intervals are honored.

**Tech Stack:** Python 3.14, `threading` (`Thread`, `Event`), SQLite3 WAL, `pytest`, `argparse`.

## Global Constraints

- **Python Interpreter for Tests:** Always execute tests using `.venv\Scripts\python.exe -m pytest <path>::<test_name> -v`.
- **Zero Frame Drops:** Frame capture and ONNX/YOLO inference run strictly decoupled from network operations; signaling `_sync_trigger.set()` must take <0.005ms.
- **Data Resilience:** SQLite WAL remains the single source of truth for outbox persistence.
- **Backward Compatibility:** All existing `CounterService` arguments, defaults, and API schemas remain compatible.
- **Configurable Control:** `enable_immediate_sync: bool = True` by default; toggleable via `--immediate-sync` and `--no-immediate-sync` CLI flags or `IMMEDIATE_SYNC` env var.

---

### Task 1: Add `enable_immediate_sync`, `_sync_trigger`, and `trigger_sync()` to `CounterService`

**Files:**
- Modify: `main.py:120-175`, `main.py:270-300`
- Test: `tests/test_main.py`

**Interfaces:**
- Consumes: `threading.Event`, `LocalStorage`, `OutboxSyncDispatcher`
- Produces:
  - `CounterService.enable_immediate_sync: bool` (default `True`)
  - `CounterService._sync_trigger: threading.Event`
  - `CounterService.trigger_sync() -> None`: sets `self._sync_trigger` if `enable_immediate_sync` is True and `dispatcher` is not None.

- [ ] **Step 1: Write the failing unit tests for `trigger_sync()` and initialization**

Add to `tests/test_main.py`:

```python
def test_counter_service_immediate_sync_initialization_and_trigger(tmp_path):
    """Verify CounterService initializes _sync_trigger and trigger_sync() functions correctly."""
    db_file = str(tmp_path / "test_init_trigger.db")
    storage = LocalStorage(db_path=db_file)
    mock_dispatcher = MagicMock()

    # 1. Enabled by default
    service = CounterService(
        line_id="lineInitTest",
        storage=storage,
        dispatcher=mock_dispatcher,
    )
    assert service.enable_immediate_sync is True
    assert hasattr(service, "_sync_trigger")
    assert not service._sync_trigger.is_set()

    # Calling trigger_sync sets the event
    service.trigger_sync()
    assert service._sync_trigger.is_set()
    service._sync_trigger.clear()

    # 2. Disabled explicitly
    disabled_service = CounterService(
        line_id="lineInitTest2",
        storage=storage,
        dispatcher=mock_dispatcher,
        enable_immediate_sync=False,
    )
    assert disabled_service.enable_immediate_sync is False
    disabled_service.trigger_sync()
    assert not disabled_service._sync_trigger.is_set()

    # 3. Enabled but no dispatcher (standalone mode)
    no_dispatch_service = CounterService(
        line_id="lineInitTest3",
        storage=storage,
        dispatcher=None,
        enable_sync=False,
        enable_immediate_sync=True,
    )
    no_dispatch_service.trigger_sync()
    assert not no_dispatch_service._sync_trigger.is_set()
```

- [ ] **Step 2: Run test to verify it fails**

Run:
```bash
.venv\Scripts\python.exe -m pytest tests/test_main.py::test_counter_service_immediate_sync_initialization_and_trigger -v
```
Expected: FAIL with `AttributeError: 'CounterService' object has no attribute 'enable_immediate_sync'` or `no attribute 'trigger_sync'`.

- [ ] **Step 3: Implement `enable_immediate_sync`, `_sync_trigger`, and `trigger_sync` in `CounterService`**

In `main.py`:
1. Add `enable_immediate_sync: bool = True` to `CounterService.__init__` signature (around line 123):
```python
        idle_timeout_sec: float = 30.0,
        idle_fps: float = 1.0,
        enable_immediate_sync: bool = True,
    ) -> None:
```

2. Store `self.enable_immediate_sync = enable_immediate_sync` and `self._sync_trigger = threading.Event()` in `CounterService.__init__` (around line 171):
```python
        # Background sync worker control
        self._stop_event = threading.Event()
        self._sync_trigger = threading.Event()
        self.enable_immediate_sync = enable_immediate_sync
        self._sync_thread: Optional[threading.Thread] = None
```

3. Add `trigger_sync()` method on `CounterService`:
```python
    def trigger_sync(self) -> None:
        """Signal the background sync thread to wake and flush outbox immediately."""
        if self.enable_immediate_sync and self.dispatcher is not None:
            self._sync_trigger.set()
```

- [ ] **Step 4: Run test to verify it passes**

Run:
```bash
.venv\Scripts\python.exe -m pytest tests/test_main.py::test_counter_service_immediate_sync_initialization_and_trigger -v
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add main.py tests/test_main.py
git commit -m "feat(sync): add enable_immediate_sync and trigger_sync to CounterService"
```

---

### Task 2: Implement Event-Triggered Wake and Backoff Suppression in `_sync_worker()`

**Files:**
- Modify: `main.py:270-305`
- Test: `tests/test_main.py`

**Interfaces:**
- Consumes: `self._sync_trigger`, `self._stop_event`, `self.enable_immediate_sync`, `dispatcher.backoff_sec`, `dispatcher.sync_once()`
- Produces: Responsive `_sync_worker` loop that wakes immediately on count trigger when network is healthy, but sleeps for backoff period when `backoff_sec > 1.0`.

- [ ] **Step 1: Write the failing tests for immediate wake and backoff suppression**

Add to `tests/test_main.py`:

```python
def test_sync_worker_wakes_immediately_on_trigger(tmp_path):
    """Verify _sync_worker wakes immediately (<100ms) on _sync_trigger.set() without waiting for sync_interval_sec."""
    import time
    db_file = str(tmp_path / "test_immediate_wake.db")
    storage = LocalStorage(db_path=db_file)
    mock_dispatcher = MagicMock()
    mock_dispatcher.sync_once.return_value = 0
    mock_dispatcher.backoff_sec = 1.0

    service = CounterService(
        line_id="lineWakeTest",
        storage=storage,
        dispatcher=mock_dispatcher,
        sync_interval_sec=10.0,  # Long 10-second polling interval
        enable_immediate_sync=True,
    )

    # Start sync thread directly
    worker_thread = threading.Thread(target=service._sync_worker, daemon=True)
    worker_thread.start()

    try:
        # Give worker a moment to start and enter wait state
        time.sleep(0.05)
        initial_calls = mock_dispatcher.sync_once.call_count

        t0 = time.perf_counter()
        # Trigger immediate sync
        service.trigger_sync()

        # Wait until dispatcher.sync_once is called again
        deadline = time.perf_counter() + 0.5
        while mock_dispatcher.sync_once.call_count == initial_calls and time.perf_counter() < deadline:
            time.sleep(0.005)

        wake_latency = time.perf_counter() - t0
        assert mock_dispatcher.sync_once.call_count > initial_calls
        # Must wake in well under 100ms (far less than the 10.0s interval)
        assert wake_latency < 0.10, f"Wake latency {wake_latency:.4f}s exceeded 100ms threshold"
    finally:
        service.stop()
        worker_thread.join(timeout=1.0)


def test_sync_worker_suppresses_immediate_wake_during_backoff(tmp_path):
    """Verify _sync_worker honors backoff_sec and does not wake immediately when backoff > 1.0."""
    import time
    db_file = str(tmp_path / "test_backoff_suppress.db")
    storage = LocalStorage(db_path=db_file)
    mock_dispatcher = MagicMock()
    mock_dispatcher.sync_once.return_value = 0
    # Simulate active backoff (e.g. 5.0 seconds due to network error)
    mock_dispatcher.backoff_sec = 5.0

    service = CounterService(
        line_id="lineBackoffTest",
        storage=storage,
        dispatcher=mock_dispatcher,
        sync_interval_sec=5.0,
        enable_immediate_sync=True,
    )

    worker_thread = threading.Thread(target=service._sync_worker, daemon=True)
    worker_thread.start()

    try:
        time.sleep(0.05)
        initial_calls = mock_dispatcher.sync_once.call_count

        # Trigger sync while backoff is active
        service.trigger_sync()

        # Sleep 150ms: worker should NOT wake immediately because backoff is active
        time.sleep(0.15)
        assert mock_dispatcher.sync_once.call_count == initial_calls
    finally:
        service.stop()
        worker_thread.join(timeout=1.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run:
```bash
.venv\Scripts\python.exe -m pytest tests/test_main.py::test_sync_worker_wakes_immediately_on_trigger tests/test_main.py::test_sync_worker_suppresses_immediate_wake_during_backoff -v
```
Expected: FAIL (the worker currently waits on `_stop_event.wait(timeout=sleep_time)` and does not react to `_sync_trigger`).

- [ ] **Step 3: Update `_sync_worker()` in `main.py`**

Replace `_sync_worker` in `main.py` (around lines 269-291):

```python
    def _sync_worker(self) -> None:
        """Background worker periodically or immediately pushing unsynced outbox batches to central API."""
        if self.dispatcher is None:
            return
        logger.info("Background sync worker started for line '%s'", self.line_id)
        while not self._stop_event.is_set():
            # If backoff is active (> 1.0s), sleep for backoff duration and suppress immediate wakeups
            backoff = getattr(self.dispatcher, "backoff_sec", 1.0)
            if backoff > 1.0:
                self._stop_event.wait(timeout=backoff)
                if self._stop_event.is_set():
                    break

            try:
                synced = self.dispatcher.sync_once()
                if synced > 0:
                    logger.debug("Background synced %d outbox records to central API", synced)
            except Exception as e:
                logger.warning("Error in background sync worker: %s", e)

            # Check backoff after sync attempt
            backoff = getattr(self.dispatcher, "backoff_sec", 1.0)
            if backoff > 1.0:
                self._stop_event.wait(timeout=backoff)
            elif self.enable_immediate_sync:
                # Wait for immediate trigger, periodic fallback timeout, or stop signal
                self._sync_trigger.wait(timeout=self.sync_interval_sec)
                self._sync_trigger.clear()
            else:
                self._stop_event.wait(timeout=self.sync_interval_sec)

        logger.info("Background sync worker stopped for line '%s'", self.line_id)
```

- [ ] **Step 4: Run tests to verify they pass**

Run:
```bash
.venv\Scripts\python.exe -m pytest tests/test_main.py::test_sync_worker_wakes_immediately_on_trigger tests/test_main.py::test_sync_worker_suppresses_immediate_wake_during_backoff -v
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add main.py tests/test_main.py
git commit -m "feat(sync): implement event-triggered immediate worker wake with backoff suppression"
```

---

### Task 3: Hook Frame Loop Event Detection and Clean Shutdown

**Files:**
- Modify: `main.py:398-407`, `main.py:550-570`, `main.py:895-905`
- Test: `tests/test_main.py`

**Interfaces:**
- Consumes: `CounterService.trigger_sync()`, `CounterService._stop_event`, `CounterService._sync_trigger`
- Produces:
  - `trigger_sync()` called whenever `events` returned by `counter.update()` is non-empty.
  - Prompt shutdown via setting both `_stop_event` and `_sync_trigger` in `stop()` and `_cleanup()`.

- [ ] **Step 1: Write the failing tests for frame loop event triggering and prompt shutdown**

Add to `tests/test_main.py`:

```python
def test_counter_pipeline_triggers_sync_on_count_events(tmp_path):
    """Verify CounterService.run() calls trigger_sync() when counter produces count events."""
    db_file = str(tmp_path / "test_pipeline_trigger.db")
    storage = LocalStorage(db_path=db_file)
    mock_dispatcher = MagicMock()
    mock_dispatcher.sync_once.return_value = 0

    service = CounterService(
        line_id="lineTriggerPipe",
        video_source_type="synthetic",
        storage=storage,
        dispatcher=mock_dispatcher,
        enable_immediate_sync=True,
    )

    with patch.object(service, "trigger_sync", wraps=service.trigger_sync) as spy_trigger:
        # Mock counter to emit an event on first update
        from src.contracts import CountEventDTO
        from datetime import datetime, timezone
        mock_event = CountEventDTO(
            track_id=1,
            line_id="lineTriggerPipe",
            timestamp=datetime.now(timezone.utc),
            confidence=0.95,
        )
        with patch.object(service.counter, "update", return_value=[mock_event]):
            service.run(max_frames=1)

        assert spy_trigger.called, "Expected trigger_sync to be called when events occurred"


def test_counter_service_stop_wakes_sync_thread_promptly(tmp_path):
    """Verify CounterService.stop() signals _sync_trigger so worker exits immediately without waiting for sync_interval_sec."""
    import time
    db_file = str(tmp_path / "test_stop_prompt.db")
    storage = LocalStorage(db_path=db_file)
    mock_dispatcher = MagicMock()
    mock_dispatcher.sync_once.return_value = 0

    service = CounterService(
        line_id="lineStopPrompt",
        storage=storage,
        dispatcher=mock_dispatcher,
        sync_interval_sec=10.0,
        enable_immediate_sync=True,
    )

    worker_thread = threading.Thread(target=service._sync_worker, daemon=True)
    service._sync_thread = worker_thread
    worker_thread.start()

    time.sleep(0.05)
    t0 = time.perf_counter()
    service.stop()
    worker_thread.join(timeout=1.0)
    stop_duration = time.perf_counter() - t0

    assert not worker_thread.is_alive()
    assert stop_duration < 0.20, f"stop() took {stop_duration:.4f}s, expected <200ms"
```

- [ ] **Step 2: Run test to verify it fails**

Run:
```bash
.venv\Scripts\python.exe -m pytest tests/test_main.py::test_counter_pipeline_triggers_sync_on_count_events tests/test_main.py::test_counter_service_stop_wakes_sync_thread_promptly -v
```
Expected: FAIL (`spy_trigger.called` is False because frame loop doesn't call `trigger_sync()` yet).

- [ ] **Step 3: Update frame loop, `stop()`, and `_cleanup()` in `main.py`**

1. In `main.py` inside `run()`, right after the `for ev in events:` loop (around line 568):
```python
                # 3. LocalStorage Outbox (non-blocking)
                for ev in events:
                    self.storage.insert_event(self.line_id, ev)
                    self._window_count += 1
                    self._window_confidences.append(ev.confidence)
                    if self.cli_display:
                        # Clear in-place line and print count event
                        sys.stdout.write("\r" + " " * 100 + "\r")
                        print(
                            f" [COUNT +1] Frame #{self.total_frames_processed:<5d} | "
                            f"Track #{ev.track_id:<3d} | "
                            f"Conf: {ev.confidence:>5.1%} | "
                            f"Line Total: {self.counter.total_count}"
                        )
                    logger.debug(
                        "Saved count event: track_id=%d, line=%s",
                        ev.track_id,
                        self.line_id,
                    )

                if events:
                    self.trigger_sync()
```

2. In `stop()` (around line 400):
```python
    def stop(self) -> None:
        """Signal the pipeline loop and background sync worker to stop."""
        self.running = False
        self._stop_event.set()
        self._sync_trigger.set()
        if self.stream_uploader is not None:
            try:
                self.stream_uploader.stop()
            except Exception as e:
                logger.warning("Error stopping stream uploader: %s", e)
```

3. In `_cleanup()` (around line 900):
```python
    def _cleanup(self) -> None:
        """Stop background worker, release video source, record telemetry, and flush unsynced records."""
        if not self.cli_display:
            logger.info("Cleaning up counter pipeline resources...")
        self.running = False
        self._stop_event.set()
        self._sync_trigger.set()
```

- [ ] **Step 4: Run tests to verify they pass**

Run:
```bash
.venv\Scripts\python.exe -m pytest tests/test_main.py::test_counter_pipeline_triggers_sync_on_count_events tests/test_main.py::test_counter_service_stop_wakes_sync_thread_promptly -v
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add main.py tests/test_main.py
git commit -m "feat(sync): trigger immediate sync on count events and ensure prompt thread termination"
```

---

### Task 4: CLI Arguments and Environment Variable Configuration

**Files:**
- Modify: `main.py:1135-1145`, `main.py:1360-1370`
- Test: `tests/test_main.py`

**Interfaces:**
- Consumes: `build_arg_parser()`, `parse_args()`, `os.environ`
- Produces:
  - `--immediate-sync` flag (default True)
  - `--no-immediate-sync` flag
  - `IMMEDIATE_SYNC` env var fallback
  - Wiring to `CounterService(..., enable_immediate_sync=args.immediate_sync)`

- [ ] **Step 1: Write the failing tests for CLI flags and env var**

Add to `tests/test_main.py`:

```python
def test_parse_args_immediate_sync_flags(monkeypatch):
    """Verify CLI flags --immediate-sync and --no-immediate-sync and IMMEDIATE_SYNC env var."""
    from main import parse_args

    # 1. Default without flags or env
    monkeypatch.delenv("IMMEDIATE_SYNC", raising=False)
    args = parse_args([])
    assert args.immediate_sync is True

    # 2. Flag --no-immediate-sync
    args_no = parse_args(["--no-immediate-sync"])
    assert args_no.immediate_sync is False

    # 3. Explicit flag --immediate-sync
    args_yes = parse_args(["--immediate-sync"])
    assert args_yes.immediate_sync is True

    # 4. Environment variable false
    monkeypatch.setenv("IMMEDIATE_SYNC", "0")
    args_env_false = parse_args([])
    assert args_env_false.immediate_sync is False

    # 5. Environment variable true
    monkeypatch.setenv("IMMEDIATE_SYNC", "true")
    args_env_true = parse_args([])
    assert args_env_true.immediate_sync is True
```

- [ ] **Step 2: Run test to verify it fails**

Run:
```bash
.venv\Scripts\python.exe -m pytest tests/test_main.py::test_parse_args_immediate_sync_flags -v
```
Expected: FAIL (`AttributeError: 'Namespace' object has no attribute 'immediate_sync'`).

- [ ] **Step 3: Implement CLI flags in `build_arg_parser()` and wire in `main()`**

1. In `main.py` inside `build_arg_parser()` (near line 1138, after `--sync-interval`):
```python
    parser.add_argument(
        "--immediate-sync",
        dest="immediate_sync",
        action="store_true",
        default=os.getenv("IMMEDIATE_SYNC", "1").lower() in ("1", "true", "yes"),
        help="Enable immediate outbox sync dispatch upon each count event (default: True)",
    )
    parser.add_argument(
        "--no-immediate-sync",
        dest="immediate_sync",
        action="store_false",
        help="Disable immediate count event outbox sync; use periodic interval only",
    )
```

2. In `main.py` inside `main()`, pass `enable_immediate_sync` to `CounterService` (around line 1365):
```python
    service = CounterService(
        line_id=args.line_id,
        video_source_type=video_source_type,
        video_path=args.video,
        api_sync_url=args.api_sync_url,
        api_token=args.api_token,
        db_path=args.db_path,
        sync_interval_sec=args.sync_interval,
        target_fps=args.target_fps,
        aggregate_window_sec=args.aggregate_window,
        health_interval_sec=args.health_interval,
        loop=args.loop,
        detector=detector,
        line_coords=line_coords,
        line_direction=args.direction,
        enable_sync=enable_sync,
        enable_immediate_sync=args.immediate_sync,
        cli_display=True,
        show_window=args.show_window,
        enable_stream_relay=args.enable_stream_relay,
        stream_fps=args.stream_fps,
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run:
```bash
.venv\Scripts\python.exe -m pytest tests/test_main.py::test_parse_args_immediate_sync_flags -v
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add main.py tests/test_main.py
git commit -m "feat(cli): add --immediate-sync and --no-immediate-sync CLI options"
```

---

### Task 5: End-to-End Latency Verification & Full Regression Testing

**Files:**
- Test: `tests/test_main.py`

**Interfaces:**
- Consumes: All updated components in `main.py`
- Produces: Verified end-to-end count-to-sync immediate dispatch with sub-50ms latency and 100% passing test suite.

- [ ] **Step 1: Write an end-to-end integration test verifying count-to-sync delivery**

Add to `tests/test_main.py`:

```python
def test_counter_service_end_to_end_immediate_sync_dispatch(tmp_path):
    """Verify that when a count event occurs during pipeline run, dispatcher.sync_once() is invoked immediately."""
    import time
    db_file = str(tmp_path / "test_e2e_immediate.db")
    storage = LocalStorage(db_path=db_file)
    mock_http_client = MagicMock()
    mock_http_client.post.return_value.status_code = 200

    dispatcher = OutboxSyncDispatcher(
        storage=storage,
        api_url="http://mock-api/sync",
        line_id="lineE2E",
        http_client=mock_http_client,
    )

    service = CounterService(
        line_id="lineE2E",
        video_source_type="synthetic",
        storage=storage,
        dispatcher=dispatcher,
        sync_interval_sec=10.0,  # 10s fallback: immediate sync must fire without waiting
        enable_immediate_sync=True,
    )

    from src.contracts import CountEventDTO
    from datetime import datetime, timezone
    mock_event = CountEventDTO(
        track_id=42,
        line_id="lineE2E",
        timestamp=datetime.now(timezone.utc),
        confidence=0.92,
    )

    # Emit count event on frame 1
    with patch.object(service.counter, "update", return_value=[mock_event]):
        t0 = time.perf_counter()
        service.run(max_frames=1)
        elapsed = time.perf_counter() - t0

    # Ensure run completed swiftly (<1.5s total including cleanup flush)
    assert elapsed < 1.5, f"Execution took {elapsed:.2f}s, expected fast execution"
    # Ensure HTTP client received POST batch
    assert mock_http_client.post.called
```

- [ ] **Step 2: Run the new integration test**

Run:
```bash
.venv\Scripts\python.exe -m pytest tests/test_main.py::test_counter_service_end_to_end_immediate_sync_dispatch -v
```
Expected: PASS

- [ ] **Step 3: Run the complete `tests/test_main.py` test suite**

Run:
```bash
.venv\Scripts\python.exe -m pytest tests/test_main.py -v
```
Expected: All tests PASS (37+ tests passed, 0 failures).

- [ ] **Step 4: Run the fast unit test suite across the project**

Run:
```bash
.venv\Scripts\python.exe -m pytest tests/test_tracking.py tests/test_storage.py tests/test_detector.py tests/test_counter_contracts.py -v
```
Expected: All PASS.

- [ ] **Step 5: Commit**

```bash
git add tests/test_main.py
git commit -m "test(sync): add end-to-end immediate sync latency verification test"
```
