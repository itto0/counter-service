"""Tests for dynamic telemetry tracking accuracy integration and degraded stream alerting."""

import logging
from types import SimpleNamespace
from unittest.mock import MagicMock

from main import CounterService, build_cli_parser
from src.storage.db import LocalStorage


def test_telemetry_defaults_to_model_accuracy(tmp_path):
    """Verify CounterService defaults to recording model_accuracy_pct in telemetry."""
    db_file = str(tmp_path / "test_default_model.db")
    storage = LocalStorage(db_path=db_file)

    service = CounterService(
        line_id="lineDefaultModel",
        storage=storage,
        enable_sync=False,
    )
    assert service.use_dynamic_accuracy is False
    service.model_accuracy_pct = 95.5
    service.counter.get_tracking_accuracy_pct = MagicMock(return_value=60.0)

    service._record_health()

    samples = storage.get_unsynced_health(line_id="lineDefaultModel")
    assert len(samples) == 1
    # By default, must record model_accuracy_pct (95.5) NOT dynamic (60.0)
    assert samples[0]["accuracy_pct"] == 95.5


def test_telemetry_opt_in_dynamic_accuracy(tmp_path):
    """Verify CounterService records dynamic tracking accuracy when use_dynamic_accuracy=True."""
    db_file = str(tmp_path / "test_opt_in.db")
    storage = LocalStorage(db_path=db_file)

    service = CounterService(
        line_id="lineOptIn",
        storage=storage,
        enable_sync=False,
        use_dynamic_accuracy=True,
    )
    assert service.use_dynamic_accuracy is True
    service.model_accuracy_pct = 95.5
    service.counter.get_tracking_accuracy_pct = MagicMock(return_value=75.0)

    service._record_health()

    samples = storage.get_unsynced_health(line_id="lineOptIn")
    assert len(samples) == 1
    assert samples[0]["accuracy_pct"] == 75.0


def test_telemetry_dynamic_accuracy_precedence(tmp_path, monkeypatch):
    """Verify precedence: constructor arg > pipeline_config > env var > default False."""
    db_file = str(tmp_path / "test_precedence.db")
    storage = LocalStorage(db_path=db_file)

    # 1. Default is False
    monkeypatch.delenv("USE_DYNAMIC_ACCURACY", raising=False)
    s1 = CounterService(line_id="l1", storage=storage, enable_sync=False)
    assert s1.use_dynamic_accuracy is False

    # 2. Env var true
    monkeypatch.setenv("USE_DYNAMIC_ACCURACY", "true")
    s2 = CounterService(line_id="l2", storage=storage, enable_sync=False)
    assert s2.use_dynamic_accuracy is True

    # 3. pipeline_config overrides env var
    cfg_false = SimpleNamespace(use_dynamic_accuracy=False)
    s3 = CounterService(line_id="l3", storage=storage, enable_sync=False, pipeline_config=cfg_false)
    assert s3.use_dynamic_accuracy is False

    cfg_true = SimpleNamespace(use_dynamic_accuracy=True)
    monkeypatch.delenv("USE_DYNAMIC_ACCURACY", raising=False)
    s4 = CounterService(line_id="l4", storage=storage, enable_sync=False, pipeline_config=cfg_true)
    assert s4.use_dynamic_accuracy is True

    # 4. Constructor arg overrides pipeline_config and env var
    monkeypatch.setenv("USE_DYNAMIC_ACCURACY", "true")
    s5 = CounterService(
        line_id="l5",
        storage=storage,
        enable_sync=False,
        pipeline_config=cfg_true,
        use_dynamic_accuracy=False,
    )
    assert s5.use_dynamic_accuracy is False


def test_degraded_warning_suppressed_when_dynamic_accuracy_disabled(tmp_path, caplog):
    """Verify degraded tracking accuracy warning is suppressed when use_dynamic_accuracy=False."""
    db_file = str(tmp_path / "test_suppress.db")
    storage = LocalStorage(db_path=db_file)

    service = CounterService(
        line_id="lineSuppress",
        storage=storage,
        enable_sync=False,
        use_dynamic_accuracy=False,
    )
    service.model_accuracy_pct = 90.0
    service.counter.get_tracking_accuracy_pct = MagicMock(return_value=50.0)

    with caplog.at_level(logging.WARNING, logger="counter.service"):
        service._record_health()

    assert "Operational tracking accuracy degraded" not in caplog.text


def test_cli_parser_use_dynamic_accuracy(monkeypatch):
    """Verify --use-dynamic-accuracy flag in CLI parser."""
    monkeypatch.delenv("USE_DYNAMIC_ACCURACY", raising=False)
    parser = build_cli_parser()

    # Default is False
    args_default = parser.parse_args([])
    assert args_default.use_dynamic_accuracy is False

    # Flag passed
    args_flag = parser.parse_args(["--use-dynamic-accuracy"])
    assert args_flag.use_dynamic_accuracy is True

    # Env var default when not specified
    monkeypatch.setenv("USE_DYNAMIC_ACCURACY", "1")
    parser_env = build_cli_parser()
    args_env = parser_env.parse_args([])
    assert args_env.use_dynamic_accuracy is True



def test_record_health_uses_dynamic_tracking_accuracy(tmp_path):
    """Verify dynamic accuracy is stored in HealthSampleDTO and cached in _last_active_accuracy_pct."""
    db_file = str(tmp_path / "test_dynamic_acc.db")
    storage = LocalStorage(db_path=db_file)

    service = CounterService(
        line_id="lineDynamicAcc",
        storage=storage,
        enable_sync=False,
        use_dynamic_accuracy=True,
    )

    assert hasattr(service, "_last_active_accuracy_pct")
    assert service._last_active_accuracy_pct is None

    service.counter.get_tracking_accuracy_pct = MagicMock(return_value=85.5)

    service._record_health()

    service.counter.get_tracking_accuracy_pct.assert_called_once_with(reset_window=True)
    assert service._last_active_accuracy_pct == 85.5

    samples = storage.get_unsynced_health(line_id="lineDynamicAcc")
    assert len(samples) == 1
    assert samples[0]["accuracy_pct"] == 85.5


def test_record_health_idle_fallback(tmp_path):
    """Verify idle window falls back to _last_active_accuracy_pct or model_accuracy_pct."""
    db_file = str(tmp_path / "test_idle_fallback.db")
    storage = LocalStorage(db_path=db_file)

    service = CounterService(
        line_id="lineFallback",
        storage=storage,
        enable_sync=False,
        use_dynamic_accuracy=True,
    )
    service.model_accuracy_pct = 92.0

    # Case 1: Initial idle state (_last_active_accuracy_pct is None, dynamic is None)
    # Falls back to model_accuracy_pct
    service.counter.get_tracking_accuracy_pct = MagicMock(return_value=None)
    service._record_health()

    samples = storage.get_unsynced_health(line_id="lineFallback")
    assert len(samples) == 1
    assert samples[0]["accuracy_pct"] == 92.0
    assert service._last_active_accuracy_pct is None

    # Case 2: Active conveyor produces dynamic accuracy reading
    service.counter.get_tracking_accuracy_pct = MagicMock(return_value=88.0)
    service._record_health()

    samples = storage.get_unsynced_health(line_id="lineFallback")
    assert len(samples) == 2
    assert samples[1]["accuracy_pct"] == 88.0
    assert service._last_active_accuracy_pct == 88.0

    # Case 3: Subsequent idle window (dynamic is None)
    # Falls back to _last_active_accuracy_pct (88.0), NOT model_accuracy_pct (92.0)
    service.counter.get_tracking_accuracy_pct = MagicMock(return_value=None)
    service._record_health()

    samples = storage.get_unsynced_health(line_id="lineFallback")
    assert len(samples) == 3
    assert samples[2]["accuracy_pct"] == 88.0


def test_record_health_degraded_warning_log(tmp_path, caplog):
    """Verify warning log emitted when dynamic accuracy < 80%."""
    db_file = str(tmp_path / "test_degraded.db")
    storage = LocalStorage(db_path=db_file)

    service = CounterService(
        line_id="lineDegraded",
        storage=storage,
        enable_sync=False,
        use_dynamic_accuracy=True,
    )

    # Sub-80% accuracy triggers warning
    service.counter.get_tracking_accuracy_pct = MagicMock(return_value=75.0)
    with caplog.at_level(logging.WARNING, logger="counter.service"):
        service._record_health()

    assert "Operational tracking accuracy degraded: 75.0%" in caplog.text

    # Accuracy >= 80% does not trigger warning
    caplog.clear()
    service_healthy = CounterService(
        line_id="lineHealthy",
        storage=storage,
        enable_sync=False,
        use_dynamic_accuracy=True,
    )
    service_healthy.counter.get_tracking_accuracy_pct = MagicMock(return_value=85.0)
    with caplog.at_level(logging.WARNING, logger="counter.service"):
        service_healthy._record_health()

    assert "Operational tracking accuracy degraded" not in caplog.text


def test_smoothed_accuracy_seeded_on_first_active_window(tmp_path):
    """Verify first active window directly seeds _smoothed_accuracy_pct."""
    db_file = str(tmp_path / "test_seed.db")
    storage = LocalStorage(db_path=db_file)
    service = CounterService(
        line_id="lineSeed",
        storage=storage,
        enable_sync=False,
        use_dynamic_accuracy=True,
    )

    assert hasattr(service, "_smoothed_accuracy_pct")
    assert service._smoothed_accuracy_pct is None
    assert service.accuracy_ema_alpha == 0.15

    service.counter.get_tracking_accuracy_pct = MagicMock(return_value=85.0)
    service._record_health()

    assert service._smoothed_accuracy_pct == 85.0
    assert service._last_active_accuracy_pct == 85.0

    samples = storage.get_unsynced_health(line_id="lineSeed")
    assert len(samples) == 1
    assert samples[0]["accuracy_pct"] == 85.0


def test_smoothed_accuracy_ema_gradual_decay_and_recovery(tmp_path):
    """Verify sequential raw readings cause smooth, gradual decay and recovery with alpha=0.15."""
    db_file = str(tmp_path / "test_ema_decay.db")
    storage = LocalStorage(db_path=db_file)
    service = CounterService(
        line_id="lineEma",
        storage=storage,
        enable_sync=False,
        accuracy_ema_alpha=0.15,
        use_dynamic_accuracy=True,
    )

    # Window 1: Seed with 90.0%
    service.counter.get_tracking_accuracy_pct = MagicMock(return_value=90.0)
    service._record_health()
    assert service._smoothed_accuracy_pct == 90.0

    # Window 2: Sudden drop to 70.0%
    # Expected: 0.15 * 70.0 + 0.85 * 90.0 = 10.5 + 76.5 = 87.0
    service.counter.get_tracking_accuracy_pct = MagicMock(return_value=70.0)
    service._record_health()
    assert service._smoothed_accuracy_pct == 87.0

    # Window 3: Sustained low 70.0%
    # Expected: 0.15 * 70.0 + 0.85 * 87.0 = 10.5 + 73.95 = 84.45
    service.counter.get_tracking_accuracy_pct = MagicMock(return_value=70.0)
    service._record_health()
    assert service._smoothed_accuracy_pct == 84.45

    # Window 4: Recovery jump to 95.0%
    # Expected: 0.15 * 95.0 + 0.85 * 84.45 = 14.25 + 71.7825 = 86.0325 -> 86.03
    service.counter.get_tracking_accuracy_pct = MagicMock(return_value=95.0)
    service._record_health()
    assert service._smoothed_accuracy_pct == 86.03

    samples = storage.get_unsynced_health(line_id="lineEma")
    assert len(samples) == 4
    assert [s["accuracy_pct"] for s in samples] == [90.0, 87.0, 84.45, 86.03]


def test_smoothed_accuracy_idle_preserves_score(tmp_path):
    """Verify idle windows do not change the smoothed accuracy score."""
    db_file = str(tmp_path / "test_idle_preserves.db")
    storage = LocalStorage(db_path=db_file)
    service = CounterService(
        line_id="lineIdlePreserve",
        storage=storage,
        enable_sync=False,
        accuracy_ema_alpha=0.15,
        use_dynamic_accuracy=True,
    )

    # Window 1: Active window seeds smoothed accuracy
    service.counter.get_tracking_accuracy_pct = MagicMock(return_value=88.0)
    service._record_health()
    assert service._smoothed_accuracy_pct == 88.0

    # Window 2: Idle window (dynamic_acc is None)
    service.counter.get_tracking_accuracy_pct = MagicMock(return_value=None)
    service._record_health()
    assert service._smoothed_accuracy_pct == 88.0

    # Window 3: Another idle window
    service.counter.get_tracking_accuracy_pct = MagicMock(return_value=None)
    service._record_health()
    assert service._smoothed_accuracy_pct == 88.0

    # Window 4: Resumed active window
    # Expected: 0.15 * 80.0 + 0.85 * 88.0 = 12.0 + 74.8 = 86.8
    service.counter.get_tracking_accuracy_pct = MagicMock(return_value=80.0)
    service._record_health()
    assert service._smoothed_accuracy_pct == 86.8

    samples = storage.get_unsynced_health(line_id="lineIdlePreserve")
    assert len(samples) == 4
    assert [s["accuracy_pct"] for s in samples] == [88.0, 88.0, 88.0, 86.8]


def test_smoothed_accuracy_alpha_configuration(tmp_path, monkeypatch):
    """Verify accuracy_ema_alpha precedence: constructor arg > env var > default 0.15."""
    db_file = str(tmp_path / "test_alpha_cfg.db")
    storage = LocalStorage(db_path=db_file)

    # Case 1: default 0.15
    monkeypatch.delenv("ACCURACY_EMA_ALPHA", raising=False)
    service_default = CounterService(line_id="lineDefaultAlpha", storage=storage, enable_sync=False)
    assert service_default.accuracy_ema_alpha == 0.15

    # Case 2: env var
    monkeypatch.setenv("ACCURACY_EMA_ALPHA", "0.25")
    service_env = CounterService(line_id="lineEnvAlpha", storage=storage, enable_sync=False)
    assert service_env.accuracy_ema_alpha == 0.25

    # Case 3: constructor argument overrides env var
    service_arg = CounterService(
        line_id="lineArgAlpha",
        storage=storage,
        enable_sync=False,
        accuracy_ema_alpha=0.4,
    )
    assert service_arg.accuracy_ema_alpha == 0.4


def test_smoothed_accuracy_clamping(tmp_path):
    """Verify blended smoothed accuracy is clamped between 0.0 and 100.0."""
    db_file = str(tmp_path / "test_clamping.db")
    storage = LocalStorage(db_path=db_file)
    service = CounterService(
        line_id="lineClamp",
        storage=storage,
        enable_sync=False,
        accuracy_ema_alpha=0.5,
        use_dynamic_accuracy=True,
    )

    # Seed at 90.0
    service.counter.get_tracking_accuracy_pct = MagicMock(return_value=90.0)
    service._record_health()
    assert service._smoothed_accuracy_pct == 90.0

    # Over 100% anomaly
    service.counter.get_tracking_accuracy_pct = MagicMock(return_value=120.0)
    service._record_health()
    # Blended = 0.5 * 120 + 0.5 * 90 = 105.0 -> clamped to 100.0
    assert service._smoothed_accuracy_pct == 100.0

    # Negative anomaly
    service.counter.get_tracking_accuracy_pct = MagicMock(return_value=-50.0)
    service._record_health()
    # Blended = 0.5 * -50 + 0.5 * 100 = 25.0 -> 25.0
    assert service._smoothed_accuracy_pct == 25.0

    service.counter.get_tracking_accuracy_pct = MagicMock(return_value=-100.0)
    service._record_health()
    # Blended = 0.5 * -100 + 0.5 * 25 = -37.5 -> clamped to 0.0
    assert service._smoothed_accuracy_pct == 0.0


def test_record_health_degraded_warning_suppressed_when_idle(tmp_path, caplog):
    """Verify degraded accuracy (< 80.0%) does NOT emit warning log when stream_state is IDLE."""
    db_file = str(tmp_path / "test_idle_suppressed.db")
    storage = LocalStorage(db_path=db_file)

    service = CounterService(
        line_id="lineIdleDegraded",
        storage=storage,
        enable_sync=False,
        use_dynamic_accuracy=True,
    )

    # Low accuracy (< 80%) on active counter window
    service.counter.get_tracking_accuracy_pct = MagicMock(return_value=72.0)
    # Service stream is idle
    service.is_idle = True

    with caplog.at_level(logging.WARNING, logger="counter.service"):
        service._record_health()

    # Even though accuracy is 72.0% (< 80%), stream_state is IDLE so warning must not be logged
    assert "Operational tracking accuracy degraded" not in caplog.text

    # Check database health sample records stream_state as IDLE
    samples = storage.get_unsynced_health(line_id="lineIdleDegraded")
    assert len(samples) == 1
    assert samples[0]["stream_state"] == "IDLE"
    assert samples[0]["accuracy_pct"] == 72.0

    # Also test subsequent idle cycle where dynamic_acc is None and falls back to degraded smoothed score
    caplog.clear()
    service.counter.get_tracking_accuracy_pct = MagicMock(return_value=None)
    with caplog.at_level(logging.WARNING, logger="counter.service"):
        service._record_health()

    assert "Operational tracking accuracy degraded" not in caplog.text

    # Also test when stream_state="IDLE" is passed directly as argument
    service_arg_idle = CounterService(
        line_id="lineArgIdle",
        storage=storage,
        enable_sync=False,
        use_dynamic_accuracy=True,
    )
    service_arg_idle.counter.get_tracking_accuracy_pct = MagicMock(return_value=65.0)
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="counter.service"):
        service_arg_idle._record_health(stream_state="IDLE")
    assert "Operational tracking accuracy degraded" not in caplog.text


def test_smoothed_accuracy_alpha_invalid_and_empty_fallback(tmp_path, monkeypatch):
    """Verify ACCURACY_EMA_ALPHA empty string, invalid value, or out-of-range clamps/fallbacks safely."""
    db_file = str(tmp_path / "test_alpha_fallback.db")
    storage = LocalStorage(db_path=db_file)

    # Empty string env var falls back to default 0.15
    monkeypatch.setenv("ACCURACY_EMA_ALPHA", "")
    service_empty = CounterService(line_id="lineEmptyAlpha", storage=storage, enable_sync=False)
    assert service_empty.accuracy_ema_alpha == 0.15

    # Non-numeric string env var falls back to default 0.15
    monkeypatch.setenv("ACCURACY_EMA_ALPHA", "invalid")
    service_invalid = CounterService(line_id="lineInvalidAlpha", storage=storage, enable_sync=False)
    assert service_invalid.accuracy_ema_alpha == 0.15

    # Empty string or invalid passed as constructor parameter also falls back safely
    service_arg_invalid = CounterService(
        line_id="lineArgInvalid",
        storage=storage,
        enable_sync=False,
        accuracy_ema_alpha="not_a_number",  # type: ignore[arg-type]
    )
    assert service_arg_invalid.accuracy_ema_alpha == 0.15

    # Out of bounds clamping: above 1.0 clamped to 1.0
    monkeypatch.setenv("ACCURACY_EMA_ALPHA", "2.5")
    service_high = CounterService(line_id="lineHighAlpha", storage=storage, enable_sync=False)
    assert service_high.accuracy_ema_alpha == 1.0

    # Out of bounds clamping: below 0.01 clamped to 0.01
    monkeypatch.setenv("ACCURACY_EMA_ALPHA", "-0.5")
    service_low = CounterService(line_id="lineLowAlpha", storage=storage, enable_sync=False)
    assert service_low.accuracy_ema_alpha == 0.01


def test_smoothed_accuracy_seed_clamping(tmp_path):
    """Verify initial seeding of _smoothed_accuracy_pct clamps and rounds out-of-range values."""
    db_file = str(tmp_path / "test_seed_clamp.db")
    storage = LocalStorage(db_path=db_file)

    # Initial seed > 100.0 is clamped to 100.0
    service_high = CounterService(line_id="lineHighSeed", storage=storage, enable_sync=False, use_dynamic_accuracy=True)
    service_high.counter.get_tracking_accuracy_pct = MagicMock(return_value=125.456)
    service_high._record_health()
    assert service_high._smoothed_accuracy_pct == 100.0

    # Initial seed < 0.0 is clamped to 0.0
    service_low = CounterService(line_id="lineLowSeed", storage=storage, enable_sync=False, use_dynamic_accuracy=True)
    service_low.counter.get_tracking_accuracy_pct = MagicMock(return_value=-15.8)
    service_low._record_health()
    assert service_low._smoothed_accuracy_pct == 0.0

    # Initial seed rounding to 2 decimal places
    service_round = CounterService(line_id="lineRoundSeed", storage=storage, enable_sync=False, use_dynamic_accuracy=True)
    service_round.counter.get_tracking_accuracy_pct = MagicMock(return_value=85.6789)
    service_round._record_health()
    assert service_round._smoothed_accuracy_pct == 85.68



