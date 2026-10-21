# Health Heartbeat Accuracy Percentage Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Surface model accuracy percentage (`accuracy_pct`) in the edge counter's periodic health heartbeat (`HealthSampleDTO`), local SQLite outbox database, and central API sync payload via automatic companion metrics discovery.

**Architecture:** 
- `HealthSampleDTO` in `src/contracts.py` gains an optional `accuracy_pct: float | None = Field(default=None, ge=0.0, le=100.0)`.
- SQLite `health_samples` in `src/storage/db.py` persists `accuracy_pct` with safe migration for existing databases.
- `src/detection/detector.py` adds helper `load_model_accuracy_pct` discovering `<model_stem>_metrics.json` adjacent to the ONNX model and extracting `map50` as a percentage (`0.0 - 100.0`).
- `main.py` discovers the model accuracy on startup and populates `HealthSampleDTO` in `_record_health`.

**Tech Stack:** Python 3.10+, Pydantic v2, SQLite3, Pytest

## Global Constraints
- **Zero Breaking Changes:** `accuracy_pct` must default to `None` for full backward compatibility with existing central API sync consumers.
- **Resilient Discovery:** Missing or malformed companion metrics files must never crash or delay pipeline initialization.
- **Clamping:** Extracted accuracy percentages must strictly reside in `[0.0, 100.0]`.

---

### Task 1: Update Data Contract and Tests

**Files:**
- Modify: `src/contracts.py:24-32`
- Test: `tests/test_counter_contracts.py`

**Interfaces:**
- Produces:
  ```python
  class HealthSampleDTO(BaseModel):
      fps: float = Field(ge=0.0)
      latency_p95_ms: float = Field(ge=0.0)
      cpu_pct: float = Field(ge=0.0, le=100.0)
      mem_pct: float = Field(ge=0.0, le=100.0)
      stream_state: Literal["ONLINE", "OFFLINE", "DEGRADED"] = "ONLINE"
      model_name: str = "yolo26n"
      model_version: str = "v1.0"
      accuracy_pct: float | None = Field(default=None, ge=0.0, le=100.0)
  ```

- [ ] **Step 1: Write failing tests for `accuracy_pct` on `HealthSampleDTO`**

In `tests/test_counter_contracts.py`:
```python
def test_health_sample_dto_accuracy_pct():
    h_default = HealthSampleDTO(
        fps=30.0,
        latency_p95_ms=12.5,
        cpu_pct=45.0,
        mem_pct=50.0,
    )
    assert h_default.accuracy_pct is None

    h_with_acc = HealthSampleDTO(
        fps=30.0,
        latency_p95_ms=12.5,
        cpu_pct=45.0,
        mem_pct=50.0,
        accuracy_pct=94.21,
    )
    assert h_with_acc.accuracy_pct == 94.21

    # Clamping / bounds validation
    with pytest.raises(Exception):
        HealthSampleDTO(
            fps=30.0,
            latency_p95_ms=12.5,
            cpu_pct=45.0,
            mem_pct=50.0,
            accuracy_pct=-1.0,
        )

    with pytest.raises(Exception):
        HealthSampleDTO(
            fps=30.0,
            latency_p95_ms=12.5,
            cpu_pct=45.0,
            mem_pct=50.0,
            accuracy_pct=101.0,
        )
```

- [ ] **Step 2: Run test to verify failure**

Run: `.venv\Scripts\python.exe -m pytest tests/test_counter_contracts.py -k test_health_sample_dto_accuracy_pct`
Expected: FAIL with `Extra inputs are not permitted` or attribute missing.

- [ ] **Step 3: Update `HealthSampleDTO` in `src/contracts.py`**

Add `accuracy_pct: float | None = Field(default=None, ge=0.0, le=100.0)` to `HealthSampleDTO`.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv\Scripts\python.exe -m pytest tests/test_counter_contracts.py`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/contracts.py tests/test_counter_contracts.py
git commit -m "feat(contracts): add accuracy_pct field to HealthSampleDTO"
```

---

### Task 2: SQLite Storage Persistence and Migration

**Files:**
- Modify: `src/storage/db.py`
- Test: `tests/test_storage.py`

**Interfaces:**
- Consumes: `HealthSampleDTO.accuracy_pct`
- Produces: SQLite `health_samples.accuracy_pct` persistence and batch retrieval.

- [ ] **Step 1: Write failing storage tests for `accuracy_pct` and database migration**

In `tests/test_storage.py`:
```python
def test_storage_health_accuracy_pct(storage):
    h = HealthSampleDTO(
        fps=25.0,
        latency_p95_ms=15.0,
        cpu_pct=30.0,
        mem_pct=40.0,
        accuracy_pct=91.50,
    )
    row_id = storage.insert_health("line-1", h)
    assert row_id > 0

    payload, ids = storage.get_unsynced_batch("line-1")
    assert payload.health is not None
    assert payload.health.accuracy_pct == 91.50


def test_storage_health_migration_existing_db(tmp_path):
    db_file = tmp_path / "old.db"
    import sqlite3
    conn = sqlite3.connect(str(db_file))
    conn.execute("""
    CREATE TABLE health_samples (
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
        synced_at TEXT NULL
    );
    """)
    conn.execute("""
    INSERT INTO health_samples (line_id, ts_utc, fps, latency_p95_ms, cpu_pct, mem_pct, stream_state)
    VALUES ('line-1', '2026-10-21T00:00:00', 30.0, 10.0, 20.0, 30.0, 'ONLINE');
    """)
    conn.commit()
    conn.close()

    from src.storage.db import LocalStorage
    migrated_storage = LocalStorage(db_path=str(db_file))
    payload, _ = migrated_storage.get_unsynced_batch("line-1")
    assert payload.health is not None
    assert payload.health.accuracy_pct is None
```

- [ ] **Step 2: Run test to verify failure**

Run: `.venv\Scripts\python.exe -m pytest tests/test_storage.py -k test_storage_health_accuracy_pct`
Expected: FAIL

- [ ] **Step 3: Update `src/storage/db.py`**

1. In `_init_db()`:
   - Add `accuracy_pct REAL NULL` to `CREATE TABLE IF NOT EXISTS health_samples`.
   - Add migration block:
     ```python
     try:
         conn.execute("ALTER TABLE health_samples ADD COLUMN accuracy_pct REAL NULL")
     except sqlite3.OperationalError:
         pass
     ```
2. In `insert_health()`:
   - Include `accuracy_pct` in columns and values tuple: `health.accuracy_pct`.
3. In `get_unsynced_batch()`:
   - Read `accuracy_pct=h["accuracy_pct"] if "accuracy_pct" in h.keys() and h["accuracy_pct"] is not None else None`.

- [ ] **Step 4: Run tests to verify all storage tests pass**

Run: `.venv\Scripts\python.exe -m pytest tests/test_storage.py`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/storage/db.py tests/test_storage.py
git commit -m "feat(storage): support accuracy_pct in health_samples table and migration"
```

---

### Task 3: Metrics Auto-Discovery Helper and Service Integration

**Files:**
- Modify: `src/detection/detector.py`
- Modify: `main.py`
- Test: `tests/test_detector.py`
- Test: `tests/test_main.py`

**Interfaces:**
- Produces:
  ```python
  def load_model_accuracy_pct(model_path: str | Path | None) -> float | None:
      """Discover and load validation accuracy percentage from companion metrics JSON."""
  ```

- [ ] **Step 1: Write failing tests for `load_model_accuracy_pct` and `CounterService` health recording**

In `tests/test_detector.py`:
```python
def test_load_model_accuracy_pct(tmp_path):
    from src.detection.detector import load_model_accuracy_pct

    # Non-existent path returns None
    assert load_model_accuracy_pct(None) is None
    assert load_model_accuracy_pct("non_existent_model.onnx") is None

    # Model with companion carcass_metrics.json
    model_file = tmp_path / "carcass_yolov8n.onnx"
    model_file.touch()

    metrics_file = tmp_path / "carcass_yolov8n_metrics.json"
    metrics_file.write_text('{"metrics": {"map50": 0.9421}}', encoding="utf-8")

    assert load_model_accuracy_pct(str(model_file)) == 94.21

    # Model with companion val_metrics.json fallback
    model_file_2 = tmp_path / "model2.onnx"
    model_file_2.touch()
    metrics_file_2 = tmp_path / "val_metrics.json"
    metrics_file_2.write_text('{"metrics": {"map50": 0.885}}', encoding="utf-8")

    assert load_model_accuracy_pct(str(model_file_2)) == 88.50
```

In `tests/test_main.py`:
Add a test asserting that when `CounterService` runs with a model that has companion metrics, `_record_health()` stores `accuracy_pct` in the database.

- [ ] **Step 2: Run tests to verify failure**

Run: `.venv\Scripts\python.exe -m pytest tests/test_detector.py -k test_load_model_accuracy_pct`
Expected: FAIL with `ImportError: cannot import name 'load_model_accuracy_pct'`

- [ ] **Step 3: Implement `load_model_accuracy_pct` in `src/detection/detector.py`**

```python
def load_model_accuracy_pct(model_path: str | Path | None) -> float | None:
    """Discover and parse model accuracy percentage from companion metrics JSON file.

    Args:
        model_path: File path to the detection model (e.g. models/carcass_yolov8n.onnx).

    Returns:
        Accuracy percentage in range [0.0, 100.0] or None if metrics cannot be discovered.
    """
    if not model_path:
        return None

    path = Path(model_path)
    candidates = [
        path.parent / f"{path.stem}_metrics.json",
        path.parent / "val_metrics.json",
    ]

    for cand in candidates:
        if cand.is_file():
            try:
                import json
                data = json.loads(cand.read_text(encoding="utf-8"))
                metrics = data.get("metrics", data) if isinstance(data, dict) else {}
                raw_acc = metrics.get("map50", metrics.get("map"))
                if raw_acc is not None:
                    val = float(raw_acc)
                    # Convert [0.0, 1.0] to percentage if needed
                    pct = val * 100.0 if 0.0 <= val <= 1.0 else val
                    return round(max(0.0, min(100.0, pct)), 2)
            except Exception:
                continue

    return None
```

- [ ] **Step 4: Integrate in `main.py`**

In `CounterService.__init__`:
```python
from src.detection.detector import load_model_accuracy_pct
model_path = getattr(self.detector, "model_path", None)
self.model_accuracy_pct = load_model_accuracy_pct(model_path)
if self.model_accuracy_pct is not None:
    logger.info("Loaded model accuracy: %.2f%% from companion metrics", self.model_accuracy_pct)
```

In `CounterService._record_health(...)`:
Pass `accuracy_pct=self.model_accuracy_pct` to `HealthSampleDTO(...)`.

- [ ] **Step 5: Run tests to verify all pass**

Run: `.venv\Scripts\python.exe -m pytest tests/test_detector.py tests/test_main.py`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/detection/detector.py main.py tests/test_detector.py tests/test_main.py
git commit -m "feat(service): auto-discover companion accuracy metrics and record in health samples"
```

---

### Task 4: Full Suite Regression Verification

**Files:**
- Test: `tests/`

- [ ] **Step 1: Run complete test suite**

Run: `.venv\Scripts\python.exe -m pytest`
Expected: All 170+ tests pass (with 2 expected skipped).

- [ ] **Step 2: Verify CLI entry points**

Run:
`.venv\Scripts\python.exe main.py --help`
`.venv\Scripts\python.exe benchmark.py --help`
`.venv\Scripts\python.exe tools/train_carcass_detector.py --help`
Expected: All commands exit 0 with clean help displays.

- [ ] **Step 3: Commit ledger / progress updates**

```bash
git add .superpowers/
git commit -m "chore: complete health heartbeat accuracy percentage implementation"
```
