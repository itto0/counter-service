# Carcass Detector Accuracy Metrics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add post-training YOLO validation accuracy evaluation (mAP@50, mAP@50-95, Precision, Recall) to `tools/train_carcass_detector.py` with console summary, JSON metadata persistence, and structured return.

**Architecture:** After `model.train()` in `tools/train_carcass_detector.py`, an explicit evaluation helper evaluates `best.pt` on the validation split via Ultralytics YOLO `val()`. Extracted metrics are printed in a formatted summary banner, saved to JSON alongside the model and work directory, and returned in a structured dictionary.

**Tech Stack:** Python 3.10+, PyTorch/Ultralytics YOLO, OpenCV, pytest.

## Global Constraints

- Standalone executable script via Python CLI (`python tools/train_carcass_detector.py ...`).
- Ultralytics YOLO remains an optional runtime dependency imported lazily inside functions.
- All metric values are floats clamped or defaulted to `0.0` if missing or unavailable.
- Maintain test coverage without requiring live GPU or video files by mocking YOLO and OpenCV in unit/pipeline tests.

---

### Task 1: Implement `evaluate_trained_model` Helper and Unit Tests

**Files:**
- Modify: `tools/train_carcass_detector.py`
- Test: `tests/test_tools.py`

**Interfaces:**
- Produces:
  ```python
  def evaluate_trained_model(
      model_path: str | Path,
      yaml_path: str | Path,
      imgsz: int = 640,
  ) -> dict[str, float]:
      """Evaluates YOLO checkpoint on dataset.yaml validation split and returns accuracy metrics."""
  ```

- [ ] **Step 1: Write the failing unit tests for `evaluate_trained_model`**

Add tests to `tests/test_tools.py` testing metric extraction from mocked YOLO validation results and handling missing attributes.

```python
def test_evaluate_trained_model_success():
    from tools.train_carcass_detector import evaluate_trained_model

    mock_box = MagicMock()
    mock_box.map50 = 0.9421
    mock_box.map = 0.7615
    mock_box.mp = 0.9150
    mock_box.mr = 0.8872

    mock_val_res = MagicMock()
    mock_val_res.box = mock_box

    mock_yolo_instance = MagicMock()
    mock_yolo_instance.val.return_value = mock_val_res
    mock_yolo_cls = MagicMock(return_value=mock_yolo_instance)

    with patch.dict("sys.modules", {"ultralytics": MagicMock(YOLO=mock_yolo_cls)}):
        metrics = evaluate_trained_model("best.pt", "dataset.yaml", imgsz=640)

    assert metrics == {
        "map50": 0.9421,
        "map50_95": 0.7615,
        "precision": 0.9150,
        "recall": 0.8872,
    }
    mock_yolo_cls.assert_called_once_with("best.pt")
    mock_yolo_instance.val.assert_called_once_with(data="dataset.yaml", imgsz=640, split="val")


def test_evaluate_trained_model_missing_attrs():
    from tools.train_carcass_detector import evaluate_trained_model

    mock_val_res = MagicMock(spec=[])
    mock_yolo_instance = MagicMock()
    mock_yolo_instance.val.return_value = mock_val_res
    mock_yolo_cls = MagicMock(return_value=mock_yolo_instance)

    with patch.dict("sys.modules", {"ultralytics": MagicMock(YOLO=mock_yolo_cls)}):
        metrics = evaluate_trained_model("best.pt", "dataset.yaml", imgsz=640)

    assert metrics == {
        "map50": 0.0,
        "map50_95": 0.0,
        "precision": 0.0,
        "recall": 0.0,
    }
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv\Scripts\python.exe -m pytest tests/test_tools.py -k test_evaluate_trained_model`
Expected: FAIL with `ImportError: cannot import name 'evaluate_trained_model' from 'tools.train_carcass_detector'`

- [ ] **Step 3: Implement `evaluate_trained_model`**

In `tools/train_carcass_detector.py`:

```python
def evaluate_trained_model(
    model_path: str | Path,
    yaml_path: str | Path,
    imgsz: int = 640,
) -> dict[str, float]:
    """Evaluate trained YOLO checkpoint on validation split and return accuracy metrics.

    Args:
        model_path: Path to YOLO weights checkpoint (.pt).
        yaml_path: Path to dataset.yaml.
        imgsz: Evaluation image resolution.

    Returns:
        Dictionary containing map50, map50_95, precision, and recall metrics.
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        raise ImportError(
            "ultralytics is required for evaluating models. Install with: pip install ultralytics"
        )

    model = YOLO(str(model_path))
    val_results = model.val(data=str(yaml_path), imgsz=imgsz, split="val")

    box = getattr(val_results, "box", None)
    map50 = float(getattr(box, "map50", 0.0) or 0.0) if box is not None else 0.0
    map50_95 = float(getattr(box, "map", 0.0) or 0.0) if box is not None else 0.0
    precision = float(getattr(box, "mp", 0.0) or 0.0) if box is not None else 0.0
    recall = float(getattr(box, "mr", 0.0) or 0.0) if box is not None else 0.0

    return {
        "map50": round(map50, 4),
        "map50_95": round(map50_95, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv\Scripts\python.exe -m pytest tests/test_tools.py -k test_evaluate_trained_model`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tools/train_carcass_detector.py tests/test_tools.py
git commit -m "feat(tools): add evaluate_trained_model helper and tests"
```

---

### Task 2: Integrate Validation Evaluation, Summary Formatting, and JSON Persistence into `train_carcass_detector`

**Files:**
- Modify: `tools/train_carcass_detector.py`
- Test: `tests/test_tools.py`

**Interfaces:**
- Consumes: `evaluate_trained_model` from Task 1
- Produces:
  ```python
  def format_metrics_summary(metrics: dict[str, float]) -> str:
      """Formats accuracy metrics into a clean console banner."""

  def train_carcass_detector(...) -> dict[str, Any]:
      """Returns {'model_path': str, 'metrics': dict[str, float]}."""
  ```

- [ ] **Step 1: Write failing tests for summary formatting, JSON persistence, and pipeline output**

Add tests to `tests/test_tools.py`:

```python
def test_format_metrics_summary():
    from tools.train_carcass_detector import format_metrics_summary

    metrics = {"map50": 0.9421, "map50_95": 0.7615, "precision": 0.915, "recall": 0.8872}
    summary = format_metrics_summary(metrics)
    assert "YOLO Carcass Detection Accuracy Metrics" in summary
    assert "mAP@50       : 0.9421 (94.21%)" in summary
    assert "mAP@50-95    : 0.7615 (76.15%)" in summary
    assert "Precision    : 0.9150 (91.50%)" in summary
    assert "Recall       : 0.8872 (88.72%)" in summary
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv\Scripts\python.exe -m pytest tests/test_tools.py -k test_format_metrics_summary`
Expected: FAIL with `ImportError: cannot import name 'format_metrics_summary' from 'tools.train_carcass_detector'`

- [ ] **Step 3: Implement `format_metrics_summary` and integrate into `train_carcass_detector`**

In `tools/train_carcass_detector.py`:
1. Implement `format_metrics_summary(metrics: dict[str, float]) -> str`.
2. In `train_carcass_detector(...)`:
   - Call `metrics = evaluate_trained_model(best_pt, yaml_path, imgsz=imgsz)`
   - Print `format_metrics_summary(metrics)`
   - Save `<work_dir>/val_metrics.json`
   - Save `<output_parent>/<output_stem>_metrics.json`
   - Return `{"model_path": exported_path, "metrics": metrics}`

- [ ] **Step 4: Update pipeline integration tests in `tests/test_tools.py`**

Update `test_train_carcass_detector_pipeline_mocked` to verify:
- `evaluate_trained_model` is invoked with `(mock_best_pt, ...)`
- Result returns `{"model_path": output_model, "metrics": ...}`
- JSON files `val_metrics.json` and `carcass_metrics.json` exist with valid content.

- [ ] **Step 5: Run tests to verify all pass**

Run: `.venv\Scripts\python.exe -m pytest tests/test_tools.py`
Expected: All tests pass.

- [ ] **Step 6: Commit**

```bash
git add tools/train_carcass_detector.py tests/test_tools.py
git commit -m "feat(tools): integrate validation metrics and persistence into train_carcass_detector"
```

---

### Task 3: Full Test Suite Verification

**Files:**
- Test: `tests/`

- [ ] **Step 1: Run full test suite**

Run: `.venv\Scripts\python.exe -m pytest`
Expected: 165+ tests pass, 2 skipped.

- [ ] **Step 2: Run CLI help and dry runs**

Run:
`.venv\Scripts\python.exe tools/train_carcass_detector.py --help`
Expected: Returncode 0 with help text.

- [ ] **Step 3: Commit all changes**

```bash
git add .
git commit -m "test: verify all tests pass with carcass detector accuracy metrics"
```
