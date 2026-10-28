# Design Specification: Exponential Moving Average (EMA) Accuracy Telemetry Smoothing

**Date:** 2026-10-28  
**Topic:** Dampening and smoothing operational tracking accuracy scores in runtime health telemetry.  
**Target Component:** `CounterService` (`main.py`)

---

## 1. Context and Problem Statement

In `main.py`, `CounterService._record_health()` records a `HealthSampleDTO` telemetry heartbeat every 10 seconds. In the previous implementation, the service queried `counter.get_tracking_accuracy_pct(reset_window=True)` each heartbeat. Because the evaluation window is short (10 seconds) and the number of carcasses crossing in that window may be small (e.g. 2–5 carcasses), a single lost or uncounted carcass could cause the raw window accuracy score to plunge violently (e.g. from 100% to 50% or 66.7%), and then immediately jump back to 100% in the following window.

This instability creates noisy metrics on remote dashboards and false or fluttering degraded alerts. The goal is to dampen score movements so that accuracy degrades and recovers gracefully via small, smooth increments.

---

## 2. Requirements

- **FR-1: Exponential Moving Average (EMA) Smoothing:** When an active window completes with dynamic tracking accuracy $Raw_t$, update the smoothed accuracy using:
  $$Smoothed_t = \alpha \cdot Raw_t + (1 - \alpha) \cdot Smoothed_{t-1}$$
- **FR-2: Configurable Alpha Factor:** Default $\alpha = 0.15$ (very smooth, ~15% adjustment towards the latest reading per 10s tick). Allow override via constructor parameter `accuracy_ema_alpha: float = 0.15` and environment variable `ACCURACY_EMA_ALPHA`.
- **FR-3: Unbiased Seeding on Startup:** On the very first active tracking window (when $Smoothed$ is `None`), seed $Smoothed_0 = Raw_0$ directly rather than blending against an arbitrary starting constant.
- **FR-4: Idle Conveyor State Preservation:** When no tracks terminate during a 10s window (conveyor stopped / idle, `dynamic_acc is None`), do not degrade or alter the smoothed score. Maintain the existing $Smoothed$ score; if no active reading has ever been observed since boot, fall back to `model_accuracy_pct`.
- **FR-5: Degraded Stream Alerting:** Retain degraded stream logging (`logger.warning("Operational tracking accuracy degraded: ...")`) when `stream_state != "IDLE"` and the smoothed score drops below 80.0%.
- **FR-6: Backward Compatibility:** Maintain existing `HealthSampleDTO.accuracy_pct` contract (`float | None`, range `[0.0, 100.0]`) and storage schemas without database migration.

---

## 3. Architecture & Algorithm Details

### 3.1 Attributes in `CounterService`
```python
self._smoothed_accuracy_pct: Optional[float] = None
self.accuracy_ema_alpha: float = float(os.getenv("ACCURACY_EMA_ALPHA", "0.15"))
```

### 3.2 Update Algorithm in `CounterService._record_health()`
```python
dynamic_acc = (
    self.counter.get_tracking_accuracy_pct(reset_window=True)
    if hasattr(self, "counter") and hasattr(self.counter, "get_tracking_accuracy_pct")
    else None
)

if dynamic_acc is not None:
    if self._smoothed_accuracy_pct is None:
        self._smoothed_accuracy_pct = dynamic_acc
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
```

---

## 4. Verification and Testing

1. **Seeding Test:** Ensure initial active window directly sets `_smoothed_accuracy_pct` to the observed raw value without EMA weighting.
2. **Gradual Degradation & Recovery Test:** Simulate sequence of raw readings:
   - Initial: 100.0% -> Smoothed: 100.0%
   - Raw: 60.0% -> Smoothed: $0.15 \times 60.0 + 0.85 \times 100.0 = 94.0\%$
   - Raw: 60.0% -> Smoothed: $0.15 \times 60.0 + 0.85 \times 94.0 = 88.9\%$
   - Raw: 100.0% -> Smoothed: $0.15 \times 100.0 + 0.85 \times 88.9 = 90.57\%$
3. **Idle Persistence Test:** Ensure idle intervals (`dynamic_acc is None`) do not mutate `_smoothed_accuracy_pct`.
4. **Degraded Alert Threshold Test:** Ensure alert is logged only when `stream_state != "IDLE"` and smoothed accuracy < 80.0%.
