# Review Package: Task 1 (EMA Accuracy Smoothing)

**Base:** 7e74ed4
**Head:** 1a8d883

## Commits
1a8d883 feat(telemetry): smooth operational accuracy using exponential moving average

## Stat Summary
 main.py                                  |  29 ++++++-
 tests/test_counter_telemetry_accuracy.py | 171 +++++++++++++++++++++++++++++++++
 2 files changed, 195 insertions(+), 5 deletions(-)

## Full Diff
```diff
diff --git a/main.py b/main.py
index 3012843..ff22ef2 100644
--- a/main.py
+++ b/main.py
@@ -203,6 +203,7 @@ class CounterService:
         pipeline_config: Optional[Any] = None,
         track_thresh: float = 0.4,
         match_thresh: float = 0.7,
+        accuracy_ema_alpha: Optional[float] = None,
     ) -> None:
         self.line_id = line_id
         self.video_source_type = video_source_type.lower()
@@ -294,6 +295,14 @@ class CounterService:
         self._health_latencies_ms: list[float] = []
         self._all_latencies_ms: list[float] = []
         self._last_active_accuracy_pct: Optional[float] = None
+        self._smoothed_accuracy_pct: Optional[float] = None
+        if accuracy_ema_alpha is None and pipeline_config is not None:
+            accuracy_ema_alpha = getattr(pipeline_config, "accuracy_ema_alpha", None)
+        self.accuracy_ema_alpha: float = (
+            float(accuracy_ema_alpha)
+            if accuracy_ema_alpha is not None
+            else float(os.getenv("ACCURACY_EMA_ALPHA", "0.15"))
+        )
 
         # Storage
         self.storage = storage or LocalStorage(db_path=self.db_path)
@@ -506,14 +515,24 @@ class CounterService:
             else None
         )
         if dynamic_acc is not None:
-            self._last_active_accuracy_pct = dynamic_acc
-            current_acc = dynamic_acc
+            if self._smoothed_accuracy_pct is None:
+                self._smoothed_accuracy_pct = dynamic_acc
+            else:
+                alpha = self.accuracy_ema_alpha
+                blended = (alpha * dynamic_acc) + ((1.0 - alpha) * self._smoothed_accuracy_pct)
+                self._smoothed_accuracy_pct = round(min(100.0, max(0.0, blended)), 2)
+            self._last_active_accuracy_pct = self._smoothed_accuracy_pct
+            current_acc = self._smoothed_accuracy_pct
         else:
-            # Conveyor idle: fall back to last active reading, or static model accuracy
+            # Conveyor idle: fall back to smoothed score, last active reading, or static model accuracy
             current_acc = (
-                self._last_active_accuracy_pct
-                if self._last_active_accuracy_pct is not None
-                else self.model_accuracy_pct
+                self._smoothed_accuracy_pct
+                if self._smoothed_accuracy_pct is not None
+                else (
+                    self._last_active_accuracy_pct
+                    if self._last_active_accuracy_pct is not None
+                    else self.model_accuracy_pct
+                )
             )
 
         if stream_state != "IDLE" and current_acc is not None and current_acc < 80.0:
diff --git a/tests/test_counter_telemetry_accuracy.py b/tests/test_counter_telemetry_accuracy.py
index aa58945..9433504 100644
--- a/tests/test_counter_telemetry_accuracy.py
+++ b/tests/test_counter_telemetry_accuracy.py
@@ -94,8 +94,175 @@ def test_record_health_degraded_warning_log(tmp_path, caplog):
 
     # Accuracy >= 80% does not trigger warning
     caplog.clear()
-    service.counter.get_tracking_accuracy_pct = MagicMock(return_value=85.0)
+    service_healthy = CounterService(
+        line_id="lineHealthy",
+        storage=storage,
+        enable_sync=False,
+    )
+    service_healthy.counter.get_tracking_accuracy_pct = MagicMock(return_value=85.0)
     with caplog.at_level(logging.WARNING, logger="counter.service"):
-        service._record_health()
+        service_healthy._record_health()
 
     assert "Operational tracking accuracy degraded" not in caplog.text
+
+
+def test_smoothed_accuracy_seeded_on_first_active_window(tmp_path):
+    """Verify first active window directly seeds _smoothed_accuracy_pct."""
+    db_file = str(tmp_path / "test_seed.db")
+    storage = LocalStorage(db_path=db_file)
+    service = CounterService(
+        line_id="lineSeed",
+        storage=storage,
+        enable_sync=False,
+    )
+
+    assert hasattr(service, "_smoothed_accuracy_pct")
+    assert service._smoothed_accuracy_pct is None
+    assert service.accuracy_ema_alpha == 0.15
+
+    service.counter.get_tracking_accuracy_pct = MagicMock(return_value=85.0)
+    service._record_health()
+
+    assert service._smoothed_accuracy_pct == 85.0
+    assert service._last_active_accuracy_pct == 85.0
+
+    samples = storage.get_unsynced_health(line_id="lineSeed")
+    assert len(samples) == 1
+    assert samples[0]["accuracy_pct"] == 85.0
+
+
+def test_smoothed_accuracy_ema_gradual_decay_and_recovery(tmp_path):
+    """Verify sequential raw readings cause smooth, gradual decay and recovery with alpha=0.15."""
+    db_file = str(tmp_path / "test_ema_decay.db")
+    storage = LocalStorage(db_path=db_file)
+    service = CounterService(
+        line_id="lineEma",
+        storage=storage,
+        enable_sync=False,
+        accuracy_ema_alpha=0.15,
+    )
+
+    # Window 1: Seed with 90.0%
+    service.counter.get_tracking_accuracy_pct = MagicMock(return_value=90.0)
+    service._record_health()
+    assert service._smoothed_accuracy_pct == 90.0
+
+    # Window 2: Sudden drop to 70.0%
+    # Expected: 0.15 * 70.0 + 0.85 * 90.0 = 10.5 + 76.5 = 87.0
+    service.counter.get_tracking_accuracy_pct = MagicMock(return_value=70.0)
+    service._record_health()
+    assert service._smoothed_accuracy_pct == 87.0
+
+    # Window 3: Sustained low 70.0%
+    # Expected: 0.15 * 70.0 + 0.85 * 87.0 = 10.5 + 73.95 = 84.45
+    service.counter.get_tracking_accuracy_pct = MagicMock(return_value=70.0)
+    service._record_health()
+    assert service._smoothed_accuracy_pct == 84.45
+
+    # Window 4: Recovery jump to 95.0%
+    # Expected: 0.15 * 95.0 + 0.85 * 84.45 = 14.25 + 71.7825 = 86.0325 -> 86.03
+    service.counter.get_tracking_accuracy_pct = MagicMock(return_value=95.0)
+    service._record_health()
+    assert service._smoothed_accuracy_pct == 86.03
+
+    samples = storage.get_unsynced_health(line_id="lineEma")
+    assert len(samples) == 4
+    assert [s["accuracy_pct"] for s in samples] == [90.0, 87.0, 84.45, 86.03]
+
+
+def test_smoothed_accuracy_idle_preserves_score(tmp_path):
+    """Verify idle windows do not change the smoothed accuracy score."""
+    db_file = str(tmp_path / "test_idle_preserves.db")
+    storage = LocalStorage(db_path=db_file)
+    service = CounterService(
+        line_id="lineIdlePreserve",
+        storage=storage,
+        enable_sync=False,
+        accuracy_ema_alpha=0.15,
+    )
+
+    # Window 1: Active window seeds smoothed accuracy
+    service.counter.get_tracking_accuracy_pct = MagicMock(return_value=88.0)
+    service._record_health()
+    assert service._smoothed_accuracy_pct == 88.0
+
+    # Window 2: Idle window (dynamic_acc is None)
+    service.counter.get_tracking_accuracy_pct = MagicMock(return_value=None)
+    service._record_health()
+    assert service._smoothed_accuracy_pct == 88.0
+
+    # Window 3: Another idle window
+    service.counter.get_tracking_accuracy_pct = MagicMock(return_value=None)
+    service._record_health()
+    assert service._smoothed_accuracy_pct == 88.0
+
+    # Window 4: Resumed active window
+    # Expected: 0.15 * 80.0 + 0.85 * 88.0 = 12.0 + 74.8 = 86.8
+    service.counter.get_tracking_accuracy_pct = MagicMock(return_value=80.0)
+    service._record_health()
+    assert service._smoothed_accuracy_pct == 86.8
+
+    samples = storage.get_unsynced_health(line_id="lineIdlePreserve")
+    assert len(samples) == 4
+    assert [s["accuracy_pct"] for s in samples] == [88.0, 88.0, 88.0, 86.8]
+
+
+def test_smoothed_accuracy_alpha_configuration(tmp_path, monkeypatch):
+    """Verify accuracy_ema_alpha precedence: constructor arg > env var > default 0.15."""
+    db_file = str(tmp_path / "test_alpha_cfg.db")
+    storage = LocalStorage(db_path=db_file)
+
+    # Case 1: default 0.15
+    monkeypatch.delenv("ACCURACY_EMA_ALPHA", raising=False)
+    service_default = CounterService(line_id="lineDefaultAlpha", storage=storage, enable_sync=False)
+    assert service_default.accuracy_ema_alpha == 0.15
+
+    # Case 2: env var
+    monkeypatch.setenv("ACCURACY_EMA_ALPHA", "0.25")
+    service_env = CounterService(line_id="lineEnvAlpha", storage=storage, enable_sync=False)
+    assert service_env.accuracy_ema_alpha == 0.25
+
+    # Case 3: constructor argument overrides env var
+    service_arg = CounterService(
+        line_id="lineArgAlpha",
+        storage=storage,
+        enable_sync=False,
+        accuracy_ema_alpha=0.4,
+    )
+    assert service_arg.accuracy_ema_alpha == 0.4
+
+
+def test_smoothed_accuracy_clamping(tmp_path):
+    """Verify blended smoothed accuracy is clamped between 0.0 and 100.0."""
+    db_file = str(tmp_path / "test_clamping.db")
+    storage = LocalStorage(db_path=db_file)
+    service = CounterService(
+        line_id="lineClamp",
+        storage=storage,
+        enable_sync=False,
+        accuracy_ema_alpha=0.5,
+    )
+
+    # Seed at 90.0
+    service.counter.get_tracking_accuracy_pct = MagicMock(return_value=90.0)
+    service._record_health()
+    assert service._smoothed_accuracy_pct == 90.0
+
+    # Over 100% anomaly
+    service.counter.get_tracking_accuracy_pct = MagicMock(return_value=120.0)
+    service._record_health()
+    # Blended = 0.5 * 120 + 0.5 * 90 = 105.0 -> clamped to 100.0
+    assert service._smoothed_accuracy_pct == 100.0
+
+    # Negative anomaly
+    service.counter.get_tracking_accuracy_pct = MagicMock(return_value=-50.0)
+    service._record_health()
+    # Blended = 0.5 * -50 + 0.5 * 100 = 25.0 -> 25.0
+    assert service._smoothed_accuracy_pct == 25.0
+
+    service.counter.get_tracking_accuracy_pct = MagicMock(return_value=-100.0)
+    service._record_health()
+    # Blended = 0.5 * -100 + 0.5 * 25 = -37.5 -> clamped to 0.0
+    assert service._smoothed_accuracy_pct == 0.0
+
```
