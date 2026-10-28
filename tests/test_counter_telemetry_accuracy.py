"""Tests for dynamic telemetry tracking accuracy integration and degraded stream alerting."""

import logging
from unittest.mock import MagicMock

from main import CounterService
from src.storage.db import LocalStorage


def test_record_health_uses_dynamic_tracking_accuracy(tmp_path):
    """Verify dynamic accuracy is stored in HealthSampleDTO and cached in _last_active_accuracy_pct."""
    db_file = str(tmp_path / "test_dynamic_acc.db")
    storage = LocalStorage(db_path=db_file)

    service = CounterService(
        line_id="lineDynamicAcc",
        storage=storage,
        enable_sync=False,
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
    )

    # Sub-80% accuracy triggers warning
    service.counter.get_tracking_accuracy_pct = MagicMock(return_value=75.0)
    with caplog.at_level(logging.WARNING, logger="counter.service"):
        service._record_health()

    assert "Operational tracking accuracy degraded: 75.0%" in caplog.text

    # Accuracy >= 80% does not trigger warning
    caplog.clear()
    service.counter.get_tracking_accuracy_pct = MagicMock(return_value=85.0)
    with caplog.at_level(logging.WARNING, logger="counter.service"):
        service._record_health()

    assert "Operational tracking accuracy degraded" not in caplog.text
