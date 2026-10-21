# Design Spec: YOLO Carcass Detector Accuracy Detection & Validation Metrics

- **Date:** 2026-10-21
- **Status:** Approved
- **Topic:** Training Validation mAP and Detection Accuracy Reporting

---

## 1. Overview & Purpose
During training of the custom carcass detector (`tools/train_carcass_detector.py`), users need visibility into model detection accuracy (Precision, Recall, mAP@50, mAP@50-95) evaluated on the validation dataset split.

This feature adds:
1. An explicit post-training evaluation pass on the `best.pt` checkpoint using the validation split.
2. Extraction of standard object detection accuracy metrics (`mAP50`, `mAP50-95`, `precision`, `recall`).
3. Formatted console output summarizing validation accuracy percentages.
4. Persistent JSON metadata (`val_metrics.json` and `<output_stem>_metrics.json`) recording the accuracy metrics alongside the model artifacts.
5. Structured dictionary return value in `train_carcass_detector()` containing `model_path` and `metrics`.

---

## 2. Architecture & Data Flow

```
[Keyframe Extraction & Auto-labeling]
                 │
                 ▼
     [YOLO Dataset Generation]
      (dataset.yaml, train, val)
                 │
                 ▼
        [YOLO model.train()]
                 │
                 ▼
       [Resolve best.pt weights]
                 │
                 ▼
   [Validation Pass: YOLO.val(best.pt)]
  Extract mAP@50, mAP@50-95, P, R
                 │
        ┌────────┴────────┐
        ▼                 ▼
[Console Summary]   [JSON Metric Files]
 (stdout banner)     (val_metrics.json)
        │
        ▼
[Export best.pt -> ONNX]
```

---

## 3. Detailed Specifications

### 3.1 Metric Extraction Logic
A dedicated helper function `evaluate_trained_model(best_pt: Path, yaml_path: Path, imgsz: int) -> dict[str, float]` will:
- Load the best model weights via `YOLO(str(best_pt))`.
- Execute validation: `val_results = model.val(data=str(yaml_path), imgsz=imgsz, split="val")`.
- Safely extract metrics from `val_results.box`:
  - `map50`: `float(getattr(val_results.box, "map50", 0.0))`
  - `map50_95`: `float(getattr(val_results.box, "map", 0.0))`
  - `precision`: `float(getattr(val_results.box, "mp", 0.0))`
  - `recall`: `float(getattr(val_results.box, "mr", 0.0))`
- Return a standardized dictionary:
  ```python
  {
      "map50": map50,
      "map50_95": map50_95,
      "precision": precision,
      "recall": recall,
  }
  ```

### 3.2 Console Summary
Output a clean, readable text block to `stdout`:
```text
============================================================
           YOLO Carcass Detection Accuracy Metrics
============================================================
  mAP@50       : 0.9421 (94.21%)
  mAP@50-95    : 0.7615 (76.15%)
  Precision    : 0.9150 (91.50%)
  Recall       : 0.8872 (88.72%)
============================================================
```

### 3.3 Artifact Persistence
Save metric artifacts in JSON format:
1. `<work_dir>/val_metrics.json`:
   ```json
   {
     "best_checkpoint": ".../best.pt",
     "exported_onnx": ".../carcass_yolov8n.onnx",
     "metrics": {
       "map50": 0.9421,
       "map50_95": 0.7615,
       "precision": 0.9150,
       "recall": 0.8872
     }
   }
   ```
2. Alongside the target ONNX file: `<output_dir>/<model_stem>_metrics.json` (e.g. `models/carcass_yolov8n_metrics.json`).

### 3.4 Function Return Signature
`train_carcass_detector()` returns a `dict[str, Any]` with:
```python
{
    "model_path": str,  # Path to exported ONNX model
    "metrics": dict[str, float],  # Evaluated validation metrics
}
```
CLI `main()` executes the training and prints the final model path and accuracy summary.

---

## 4. Testing Strategy
- **Unit test**: Test `evaluate_trained_model` with a mocked Ultralytics validation result object.
- **Integration pipeline test**: Test `train_carcass_detector` verifying that validation is called on `best.pt`, JSON artifacts are written, and the returned dictionary contains the expected metric keys.
- **CLI test**: Test `train_carcass_detector.main()` via CLI mock ensuring clean execution.
