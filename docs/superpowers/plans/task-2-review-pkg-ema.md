# Review Package: Task 2 (Hardening, Polish & Full Test Suite Verification)

**Base:** 1a8d883
**Head:** 5d3fc72

## Commits
5d3fc72 fix(telemetry): harden alpha parsing, seed clamping, and idle warning suppression

## Stat Summary
 main.py                                  |  13 ++-
 tests/test_counter_telemetry_accuracy.py | 109 +++++++++++++++++++++++++++++++++
 2 files changed, 115 insertions(+), 7 deletions(-)

## Full Diff
```diff
diff --git a/main.py b/main.py
index ff22ef2..c03af78 100644
--- a/main.py
+++ b/main.py
@@ -298,11 +298,12 @@ class CounterService:
         self._smoothed_accuracy_pct: Optional[float] = None
         if accuracy_ema_alpha is None and pipeline_config is not None:
             accuracy_ema_alpha = getattr(pipeline_config, "accuracy_ema_alpha", None)
-        self.accuracy_ema_alpha: float = (
-            float(accuracy_ema_alpha)
-            if accuracy_ema_alpha is not None
-            else float(os.getenv("ACCURACY_EMA_ALPHA", "0.15"))
-        )
+        raw_alpha = accuracy_ema_alpha if accuracy_ema_alpha is not None else (os.getenv("ACCURACY_EMA_ALPHA") or "0.15")
+        try:
+            parsed_alpha = float(raw_alpha)
+        except (ValueError, TypeError):
+            parsed_alpha = 0.15
+        self.accuracy_ema_alpha: float = min(1.0, max(0.01, parsed_alpha))
 
         # Storage
         self.storage = storage or LocalStorage(db_path=self.db_path)
@@ -516,7 +517,7 @@ class CounterService:
         )
         if dynamic_acc is not None:
             if self._smoothed_accuracy_pct is None:
-                self._smoothed_accuracy_pct = dynamic_acc
+                self._smoothed_accuracy_pct = round(min(100.0, max(0.0, dynamic_acc)), 2)
             else:
                 alpha = self.accuracy_ema_alpha
                 blended = (alpha * dynamic_acc) + ((1.0 - alpha) * self._smoothed_accuracy_pct)
diff --git a/tests/test_counter_telemetry_accuracy.py b/tests/test_counter_telemetry_accuracy.py
index 9433504..0455278 100644
--- a/tests/test_counter_telemetry_accuracy.py
+++ b/tests/test_counter_telemetry_accuracy.py
@@ -266,3 +266,112 @@ def test_smoothed_accuracy_clamping(tmp_path):
     assert service._smoothed_accuracy_pct == 0.0
 
 
+def test_record_health_degraded_warning_suppressed_when_idle(tmp_path, caplog):
+    """Verify degraded accuracy (< 80.0%) does NOT emit warning log when stream_state is IDLE."""
+    db_file = str(tmp_path / "test_idle_suppressed.db")
+    storage = LocalStorage(db_path=db_file)
+
+    service = CounterService(
+        line_id="lineIdleDegraded",
+        storage=storage,
+        enable_sync=False,
+    )
+
+    # Low accuracy (< 80%) on active counter window
+    service.counter.get_tracking_accuracy_pct = MagicMock(return_value=72.0)
+    # Service stream is idle
+    service.is_idle = True
+
+    with caplog.at_level(logging.WARNING, logger="counter.service"):
+        service._record_health()
+
+    # Even though accuracy is 72.0% (< 80%), stream_state is IDLE so warning must not be logged
+    assert "Operational tracking accuracy degraded" not in caplog.text
+
+    # Check database health sample records stream_state as IDLE
+    samples = storage.get_unsynced_health(line_id="lineIdleDegraded")
+    assert len(samples) == 1
+    assert samples[0]["stream_state"] == "IDLE"
+    assert samples[0]["accuracy_pct"] == 72.0
+
+    # Also test subsequent idle cycle where dynamic_acc is None and falls back to degraded smoothed score
+    caplog.clear()
+    service.counter.get_tracking_accuracy_pct = MagicMock(return_value=None)
+    with caplog.at_level(logging.WARNING, logger="counter.service"):
+        service._record_health()
+
+    assert "Operational tracking accuracy degraded" not in caplog.text
+
+    # Also test when stream_state="IDLE" is passed directly as argument
+    service_arg_idle = CounterService(
+        line_id="lineArgIdle",
+        storage=storage,
+        enable_sync=False,
+    )
+    service_arg_idle.counter.get_tracking_accuracy_pct = MagicMock(return_value=65.0)
+    caplog.clear()
+    with caplog.at_level(logging.WARNING, logger="counter.service"):
+        service_arg_idle._record_health(stream_state="IDLE")
+    assert "Operational tracking accuracy degraded" not in caplog.text
+
+
+def test_smoothed_accuracy_alpha_invalid_and_empty_fallback(tmp_path, monkeypatch):
+    """Verify ACCURACY_EMA_ALPHA empty string, invalid value, or out-of-range clamps/fallbacks safely."""
+    db_file = str(tmp_path / "test_alpha_fallback.db")
+    storage = LocalStorage(db_path=db_file)
+
+    # Empty string env var falls back to default 0.15
+    monkeypatch.setenv("ACCURACY_EMA_ALPHA", "")
+    service_empty = CounterService(line_id="lineEmptyAlpha", storage=storage, enable_sync=False)
+    assert service_empty.accuracy_ema_alpha == 0.15
+
+    # Non-numeric string env var falls back to default 0.15
+    monkeypatch.setenv("ACCURACY_EMA_ALPHA", "invalid")
+    service_invalid = CounterService(line_id="lineInvalidAlpha", storage=storage, enable_sync=False)
+    assert service_invalid.accuracy_ema_alpha == 0.15
+
+    # Empty string or invalid passed as constructor parameter also falls back safely
+    service_arg_invalid = CounterService(
+        line_id="lineArgInvalid",
+        storage=storage,
+        enable_sync=False,
+        accuracy_ema_alpha="not_a_number",  # type: ignore[arg-type]
+    )
+    assert service_arg_invalid.accuracy_ema_alpha == 0.15
+
+    # Out of bounds clamping: above 1.0 clamped to 1.0
+    monkeypatch.setenv("ACCURACY_EMA_ALPHA", "2.5")
+    service_high = CounterService(line_id="lineHighAlpha", storage=storage, enable_sync=False)
+    assert service_high.accuracy_ema_alpha == 1.0
+
+    # Out of bounds clamping: below 0.01 clamped to 0.01
+    monkeypatch.setenv("ACCURACY_EMA_ALPHA", "-0.5")
+    service_low = CounterService(line_id="lineLowAlpha", storage=storage, enable_sync=False)
+    assert service_low.accuracy_ema_alpha == 0.01
+
+
+def test_smoothed_accuracy_seed_clamping(tmp_path):
+    """Verify initial seeding of _smoothed_accuracy_pct clamps and rounds out-of-range values."""
+    db_file = str(tmp_path / "test_seed_clamp.db")
+    storage = LocalStorage(db_path=db_file)
+
+    # Initial seed > 100.0 is clamped to 100.0
+    service_high = CounterService(line_id="lineHighSeed", storage=storage, enable_sync=False)
+    service_high.counter.get_tracking_accuracy_pct = MagicMock(return_value=125.456)
+    service_high._record_health()
+    assert service_high._smoothed_accuracy_pct == 100.0
+
+    # Initial seed < 0.0 is clamped to 0.0
+    service_low = CounterService(line_id="lineLowSeed", storage=storage, enable_sync=False)
+    service_low.counter.get_tracking_accuracy_pct = MagicMock(return_value=-15.8)
+    service_low._record_health()
+    assert service_low._smoothed_accuracy_pct == 0.0
+
+    # Initial seed rounding to 2 decimal places
+    service_round = CounterService(line_id="lineRoundSeed", storage=storage, enable_sync=False)
+    service_round.counter.get_tracking_accuracy_pct = MagicMock(return_value=85.6789)
+    service_round._record_health()
+    assert service_round._smoothed_accuracy_pct == 85.68
+
+
+
```
