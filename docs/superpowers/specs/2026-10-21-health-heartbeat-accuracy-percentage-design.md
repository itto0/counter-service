# Technical Design Spec: Health Heartbeat Accuracy Percentage Integration

**Date:** 2026-10-21  
**Status:** Approved  
**Topic:** Model Accuracy Percentage in Health Heartbeat Sync Payload  

---

## 1. Goal & Overview

The edge counter service periodically records and synchronizes operational health heartbeats (`HealthSampleDTO`) to the central cloud API. While training generates validation accuracy metrics (`mAP@50`, `mAP@50-95`, `Precision`, `Recall`) in `<model_stem>_metrics.json`, this validation accuracy is currently not surfaced to the central API.

This specification defines the integration of model accuracy percentage (`accuracy_pct`) into:
1. The `HealthSampleDTO` data contract.
2. The local SQLite outbox database (`health_samples` table).
3. The runtime `CounterService` through automatic discovery of companion metrics JSON files.
4. The central sync payload sent via `OutboxSyncDispatcher`.

---

## 2. Requirements & Constraints

### Functional Requirements
- **FR-1:** Add an optional `accuracy_pct: float | None = Field(default=None, ge=0.0, le=100.0)` attribute to `HealthSampleDTO`.
- **FR-2:** Maintain automatic companion JSON discovery: Given a model path `path/to/<model>.onnx`, discover `path/to/<model>_metrics.json` (or `path/to/val_metrics.json`) and extract `map50` converted to a percentage `0.0 - 100.0` rounded to 2 decimal places (e.g. `0.9421` -> `94.21`).
- **FR-3:** Update SQLite storage in `src/storage/db.py` to persist and retrieve `accuracy_pct` in `health_samples`.
- **FR-4:** Ensure existing SQLite databases automatically migrate without data loss via `ALTER TABLE health_samples ADD COLUMN accuracy_pct REAL NULL`.
- **FR-5:** If no companion metrics file exists, the file is corrupted, or model has no path, `accuracy_pct` must cleanly evaluate to `None` without raising errors or affecting service startup.

### Non-Functional & Architecture Constraints
- **Zero Breaking Changes:** Central API sync payloads must remain backward-compatible; receivers expecting the previous schema can ignore the new optional field.
- **Resilience:** Companion JSON file reading must never block or crash pipeline startup or health sampling loops.
- **Clamping:** Accuracy percentage must strictly reside within `[0.0, 100.0]`.

---

## 3. Architecture & Data Flow

```
[ tools/train_carcass_detector.py ]
               │
               ▼
Writes: models/carcass_yolov8n.onnx
        models/carcass_yolov8n_metrics.json  {"metrics": {"map50": 0.9421, ...}}
               │
               ▼
[ main.py: CounterService ]
   ├── load_model_accuracy_pct("models/carcass_yolov8n.onnx") -> 94.21
   │
   └── Periodic Health Sample (every 10s)
         ├── HealthSampleDTO(..., accuracy_pct=94.21)
         │
         ▼
[ src/storage/db.py: LocalStorage ]
   └── INSERT INTO health_samples (..., accuracy_pct) VALUES (..., 94.21)
         │
         ▼
[ src/sync/dispatcher.py: OutboxSyncDispatcher ]
   └── POST /api/v1/sync { "health": { "accuracy_pct": 94.21, ... } }
```

---

## 4. Detailed Component Design

### 4.1 Data Contract (`src/contracts.py`)
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

### 4.2 Metrics Discovery Helper (`src/detection/detector.py` or `src/contracts.py`)
A helper function `load_model_accuracy_pct(model_path: str | Path | None) -> float | None`:
1. If `model_path` is falsy or `None`, returns `None`.
2. Resolves possible candidate paths:
   - `<model_parent>/<model_stem>_metrics.json`
   - `<model_parent>/val_metrics.json`
3. If candidate exists:
   - Parses JSON content.
   - Looks for `metrics.map50`, then `metrics.map`, then top-level `map50` or `map`.
   - If found and numeric, computes `val = round(float(raw_val) * 100.0, 2)` (or leaves as-is if already in 0-100 range), clamped to `[0.0, 100.0]`.
   - Returns float percentage.
4. On `FileNotFoundError`, `json.JSONDecodeError`, `KeyError`, `ValueError`, or `PermissionError`, safely returns `None`.

### 4.3 SQLite Storage (`src/storage/db.py`)
1. Table creation statement:
   ```sql
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
   ```
2. Dynamic migration in `_init_db()`:
   ```python
   try:
       conn.execute("ALTER TABLE health_samples ADD COLUMN accuracy_pct REAL NULL")
   except sqlite3.OperationalError:
       pass
   ```
3. Update `insert_health()` to include `accuracy_pct`:
   ```sql
   INSERT INTO health_samples (
       line_id, ts_utc, fps, latency_p95_ms, cpu_pct, mem_pct,
       stream_state, model_name, model_version, accuracy_pct
   ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
   ```
4. Update `get_unsynced_batch()` row mapping:
   ```python
   health_dto = HealthSampleDTO(
       fps=h["fps"],
       latency_p95_ms=h["latency_p95_ms"],
       cpu_pct=h["cpu_pct"],
       mem_pct=h["mem_pct"],
       stream_state=h["stream_state"],
       model_name=h["model_name"],
       model_version=h["model_version"],
       accuracy_pct=h["accuracy_pct"] if "accuracy_pct" in h.keys() and h["accuracy_pct"] is not None else None,
   )
   ```

### 4.4 Counter Service Runtime (`main.py`)
In `CounterService.__init__`:
```python
model_path = getattr(self.detector, "model_path", None)
self.model_accuracy_pct = load_model_accuracy_pct(model_path)
if self.model_accuracy_pct is not None:
    logger.info("Discovered model accuracy: %.2f%% from companion metrics", self.model_accuracy_pct)
```
In `CounterService._record_health(...)`:
```python
health = HealthSampleDTO(
    fps=round(fps, 1),
    latency_p95_ms=round(latency_p95, 2),
    cpu_pct=round(min(100.0, max(0.0, cpu_pct)), 1),
    mem_pct=round(min(100.0, max(0.0, mem_pct)), 1),
    stream_state=stream_state,
    model_name=getattr(self.detector, "model_name", "yolo26n") if hasattr(self.detector, "model_name") else "yolo26n",
    model_version="v1.0",
    accuracy_pct=self.model_accuracy_pct,
)
```

---

## 5. Testing & Verification

1. **Contract Tests (`tests/test_counter_contracts.py`):**
   - Assert `HealthSampleDTO` serializes with `accuracy_pct=94.21` and `accuracy_pct=None`.
   - Validate validation limits (`0.0 <= accuracy_pct <= 100.0`).
2. **Discovery Helper Tests (`tests/test_detector.py` or `tests/test_tools.py`):**
   - Test companion discovery when `carcass_metrics.json` exists.
   - Test fallback to `val_metrics.json`.
   - Test graceful `None` return on missing file, corrupted JSON, or empty payload.
3. **Storage Tests (`tests/test_storage.py`):**
   - Test inserting health record with `accuracy_pct=95.5`.
   - Test database migration when opening an older database missing `accuracy_pct`.
   - Test `get_unsynced_batch()` populates `accuracy_pct`.
4. **Full Suite Regression:**
   - Execute `.venv\Scripts\python.exe -m pytest` ensuring 100% passes.
