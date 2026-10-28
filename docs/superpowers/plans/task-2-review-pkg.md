# Review Package: Task 2

**Base:** 65cf9c9d1c2955137d48b4eaa7bd4562019380a3
**Head:** 62848a80b80639cc713d231501b8e5fb19f5d2ab

## Commits
62848a8 feat(telemetry): update health heartbeat with dynamic tracking accuracy

## Stat Summary
 main.py                                  |  27 ++++++-
 tests/test_counter_telemetry_accuracy.py | 101 +++++++++++++++++++++++++++++++++
 2 files changed, 127 insertions(+), 1 deletion(-)

## Full Diff
```diff
diff --git a/main.py b/main.py
index 1b6f3ed..37afb42 100644
--- a/main.py
+++ b/main.py
@@ -293,6 +293,7 @@ class CounterService:
         self._health_frame_count = 0
         self._health_latencies_ms: list[float] = []
         self._all_latencies_ms: list[float] = []
+        self._last_active_accuracy_pct: Optional[float] = None
 
         # Storage
         self.storage = storage or LocalStorage(db_path=self.db_path)
@@ -499,6 +500,30 @@ class CounterService:
         elif stream_state == "ONLINE" and fps < 5.0 and self.total_frames_processed > 0:
             stream_state = "DEGRADED"
 
+        dynamic_acc = (
+            self.counter.get_tracking_accuracy_pct(reset_window=True)
+            if hasattr(self, "counter") and hasattr(self.counter, "get_tracking_accuracy_pct")
+            else None
+        )
+        if dynamic_acc is not None:
+            self._last_active_accuracy_pct = dynamic_acc
+            current_acc = dynamic_acc
+        else:
+            # Conveyor idle: fall back to last active reading, or static model accuracy
+            current_acc = (
+                self._last_active_accuracy_pct
+                if self._last_active_accuracy_pct is not None
+                else self.model_accuracy_pct
+            )
+
+        if current_acc is not None and current_acc < 80.0:
+            logger.warning(
+                "Operational tracking accuracy degraded: %.1f%% (FPS=%.1f, latency_p95=%.1fms)",
+                current_acc,
+                fps,
+                latency_p95,
+            )
+
         health = HealthSampleDTO(
             fps=round(fps, 1),
             latency_p95_ms=round(latency_p95, 2),
@@ -509,7 +534,7 @@ class CounterService:
             if hasattr(self.detector, "model_name") and isinstance(self.detector.model_name, str)
             else "yolo26n",
             model_version="v1.0",
-            accuracy_pct=self.model_accuracy_pct,
+            accuracy_pct=current_acc,
         )
         try:
             self.storage.insert_health(self.line_id, health)
diff --git a/tests/test_counter_telemetry_accuracy.py b/tests/test_counter_telemetry_accuracy.py
new file mode 100644
index 0000000..2cd6c4d
--- /dev/null
+++ b/tests/test_counter_telemetry_accuracy.py
@@ -0,0 +1,101 @@
+"""Tests for dynamic telemetry tracking accuracy integration and degraded stream alerting."""
+
+import logging
+from unittest.mock import MagicMock
+import pytest
+from main import CounterService
+from src.storage.db import LocalStorage
+
+
+def test_record_health_uses_dynamic_tracking_accuracy(tmp_path):
+    """Verify dynamic accuracy is stored in HealthSampleDTO and cached in _last_active_accuracy_pct."""
+    db_file = str(tmp_path / "test_dynamic_acc.db")
+    storage = LocalStorage(db_path=db_file)
+
+    service = CounterService(
+        line_id="lineDynamicAcc",
+        storage=storage,
+        enable_sync=False,
+    )
+
+    assert hasattr(service, "_last_active_accuracy_pct")
+    assert service._last_active_accuracy_pct is None
+
+    service.counter.get_tracking_accuracy_pct = MagicMock(return_value=85.5)
+
+    service._record_health()
+
+    service.counter.get_tracking_accuracy_pct.assert_called_once_with(reset_window=True)
+    assert service._last_active_accuracy_pct == 85.5
+
+    samples = storage.get_unsynced_health(line_id="lineDynamicAcc")
+    assert len(samples) == 1
+    assert samples[0]["accuracy_pct"] == 85.5
+
+
+def test_record_health_idle_fallback(tmp_path):
+    """Verify idle window falls back to _last_active_accuracy_pct or model_accuracy_pct."""
+    db_file = str(tmp_path / "test_idle_fallback.db")
+    storage = LocalStorage(db_path=db_file)
+
+    service = CounterService(
+        line_id="lineFallback",
+        storage=storage,
+        enable_sync=False,
+    )
+    service.model_accuracy_pct = 92.0
+
+    # Case 1: Initial idle state (_last_active_accuracy_pct is None, dynamic is None)
+    # Falls back to model_accuracy_pct
+    service.counter.get_tracking_accuracy_pct = MagicMock(return_value=None)
+    service._record_health()
+
+    samples = storage.get_unsynced_health(line_id="lineFallback")
+    assert len(samples) == 1
+    assert samples[0]["accuracy_pct"] == 92.0
+    assert service._last_active_accuracy_pct is None
+
+    # Case 2: Active conveyor produces dynamic accuracy reading
+    service.counter.get_tracking_accuracy_pct = MagicMock(return_value=88.0)
+    service._record_health()
+
+    samples = storage.get_unsynced_health(line_id="lineFallback")
+    assert len(samples) == 2
+    assert samples[1]["accuracy_pct"] == 88.0
+    assert service._last_active_accuracy_pct == 88.0
+
+    # Case 3: Subsequent idle window (dynamic is None)
+    # Falls back to _last_active_accuracy_pct (88.0), NOT model_accuracy_pct (92.0)
+    service.counter.get_tracking_accuracy_pct = MagicMock(return_value=None)
+    service._record_health()
+
+    samples = storage.get_unsynced_health(line_id="lineFallback")
+    assert len(samples) == 3
+    assert samples[2]["accuracy_pct"] == 88.0
+
+
+def test_record_health_degraded_warning_log(tmp_path, caplog):
+    """Verify warning log emitted when dynamic accuracy < 80%."""
+    db_file = str(tmp_path / "test_degraded.db")
+    storage = LocalStorage(db_path=db_file)
+
+    service = CounterService(
+        line_id="lineDegraded",
+        storage=storage,
+        enable_sync=False,
+    )
+
+    # Sub-80% accuracy triggers warning
+    service.counter.get_tracking_accuracy_pct = MagicMock(return_value=75.0)
+    with caplog.at_level(logging.WARNING, logger="counter.service"):
+        service._record_health()
+
+    assert "Operational tracking accuracy degraded: 75.0%" in caplog.text
+
+    # Accuracy >= 80% does not trigger warning
+    caplog.clear()
+    service.counter.get_tracking_accuracy_pct = MagicMock(return_value=85.0)
+    with caplog.at_level(logging.WARNING, logger="counter.service"):
+        service._record_health()
+
+    assert "Operational tracking accuracy degraded" not in caplog.text
```
