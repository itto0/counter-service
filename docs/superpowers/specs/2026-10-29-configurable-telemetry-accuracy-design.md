# Design Specification: Configurable Telemetry Accuracy (Defaulting to Model Accuracy)

**Date:** 2026-10-29  
**Topic:** Revert default health telemetry accuracy to static model validation accuracy with an opt-in toggle for dynamic operational accuracy.  
**Target Component:** `CounterService` & CLI in `main.py`

---

## 1. Context and Problem Statement

Previously, `CounterService._record_health()` in `main.py` was updated to calculate real-time operational tracking accuracy smoothed via Exponential Moving Average (EMA). However, in production and field operations, users often expect health telemetry (`HealthSampleDTO.accuracy_pct`) to report the verified benchmark accuracy of the trained AI model (e.g. `mAP@50` from companion `<model>_metrics.json`), while dynamic operational accuracy should only be enabled when explicitly configured.

---

## 2. Requirements

- **FR-1: Default to Static Model Accuracy:** By default (`use_dynamic_accuracy=False`), `HealthSampleDTO.accuracy_pct` recorded into SQLite storage during periodic health heartbeats must be `self.model_accuracy_pct`.
- **FR-2: Configurable Dynamic Accuracy Toggle:** Provide a boolean flag `use_dynamic_accuracy: bool` that defaults to `False`. The configuration precedence is:
  1. Constructor argument `use_dynamic_accuracy: Optional[bool]`
  2. `pipeline_config.use_dynamic_accuracy` (if provided)
  3. Environment variable `USE_DYNAMIC_ACCURACY` (parsed truthy: `"true"`, `"1"`, `"yes"`)
  4. Default: `False`
- **FR-3: CLI Flag:** Expose `--use-dynamic-accuracy` on the command-line interface in `main.py` (with default `False` or reading from env).
- **FR-4: Dynamic Accuracy Branch:** When `use_dynamic_accuracy` is explicitly enabled (`True`), `CounterService` performs the existing EMA-smoothed dynamic tracking accuracy calculation, idle score preservation, and degraded stream alert logging (`< 80.0%`).
- **FR-5: Degraded Warning Behavior:** When `use_dynamic_accuracy` is disabled (`False`), suppress dynamic tracking degradation warnings. Telemetry cleanly reports `self.model_accuracy_pct`.
- **FR-6: Backward Compatibility:** Maintain existing contracts for `HealthSampleDTO`, `ChickenCounter`, and database schemas with zero regressions.

---

## 3. Architecture & Algorithm Details

### 3.1 Parameter Parsing in `CounterService.__init__`
```python
if use_dynamic_accuracy is None and pipeline_config is not None:
    use_dynamic_accuracy = getattr(pipeline_config, "use_dynamic_accuracy", None)
if use_dynamic_accuracy is None:
    env_dynamic = os.getenv("USE_DYNAMIC_ACCURACY", "false").strip().lower()
    self.use_dynamic_accuracy: bool = env_dynamic in ("true", "1", "yes")
else:
    self.use_dynamic_accuracy = bool(use_dynamic_accuracy)
```

### 3.2 Health Recording Flow in `CounterService._record_health()`
```python
if not self.use_dynamic_accuracy:
    current_acc = self.model_accuracy_pct
else:
    dynamic_acc = (
        self.counter.get_tracking_accuracy_pct(reset_window=True)
        if hasattr(self, "counter") and hasattr(self.counter, "get_tracking_accuracy_pct")
        else None
    )
    if dynamic_acc is not None:
        if self._smoothed_accuracy_pct is None:
            self._smoothed_accuracy_pct = round(min(100.0, max(0.0, dynamic_acc)), 2)
        else:
            alpha = self.accuracy_ema_alpha
            blended = (alpha * dynamic_acc) + ((1.0 - alpha) * self._smoothed_accuracy_pct)
            self._smoothed_accuracy_pct = round(min(100.0, max(0.0, blended)), 2)
        self._last_active_accuracy_pct = self._smoothed_accuracy_pct
        current_acc = self._smoothed_accuracy_pct
    else:
        # Conveyor idle: fall back to smoothed score, last active reading, or static model accuracy
        current_acc = (
            self._smoothed_accuracy_pct
            if self._smoothed_accuracy_pct is not None
            else (
                self._last_active_accuracy_pct
                if self._last_active_accuracy_pct is not None
                else self.model_accuracy_pct
            )
        )

    if stream_state != "IDLE" and current_acc is not None and current_acc < 80.0:
        logger.warning(
            "Operational tracking accuracy degraded: %.1f%% (FPS=%.1f, latency_p95=%.1fms)",
            current_acc,
            fps,
            latency_p95,
        )

health = HealthSampleDTO(
    ...
    accuracy_pct=current_acc,
)
```

---

## 4. Verification and Testing

1. **Default State Test:** Verify that `CounterService` without arguments records `accuracy_pct = self.model_accuracy_pct` even when `counter.get_tracking_accuracy_pct()` returns a different dynamic value.
2. **Opt-in Toggle Test:** Verify that `use_dynamic_accuracy=True` or `USE_DYNAMIC_ACCURACY=true` activates EMA dynamic accuracy smoothing and records the dynamic score.
3. **Alert Suppression Test:** Verify that when `use_dynamic_accuracy=False`, low tracking accuracy does not emit degraded warning logs.
4. **CLI Flag Test:** Verify that `--use-dynamic-accuracy` passes correctly to `CounterService`.
5. **Full Test Suite:** All existing tests in `tests/` pass with zero regressions.
