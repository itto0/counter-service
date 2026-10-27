"""Tests for ONNX execution provider auto-negotiation and fallback."""

import sys
import types
from unittest.mock import MagicMock
import pytest

from src.detection.detector import ONNXDetector


@pytest.fixture(autouse=True)
def isolate_onnxruntime(monkeypatch):
    """Ensure onnxruntime in sys.modules is isolated and does not leak across test files."""
    if "onnxruntime" not in sys.modules:
        dummy_ort = types.ModuleType("onnxruntime")
        dummy_ort.get_available_providers = lambda: []
        monkeypatch.setitem(sys.modules, "onnxruntime", dummy_ort)


def test_resolve_providers_auto(monkeypatch):
    dummy_ort = types.ModuleType("onnxruntime")
    dummy_ort.get_available_providers = lambda: [
        "CPUExecutionProvider",
        "OpenVINOExecutionProvider",
        "CUDAExecutionProvider",
    ]
    monkeypatch.setitem(sys.modules, "onnxruntime", dummy_ort)
    monkeypatch.setattr("src.detection.detector.ort", dummy_ort, raising=False)
    providers = ONNXDetector._resolve_providers("auto")
    # CUDA takes priority over OpenVINO, followed by CPU
    assert providers == [
        "CUDAExecutionProvider",
        "OpenVINOExecutionProvider",
        "CPUExecutionProvider",
    ]


def test_resolve_providers_explicit():
    providers = ONNXDetector._resolve_providers("DirectMLExecutionProvider,CPUExecutionProvider")
    assert providers == ["DirectMLExecutionProvider", "CPUExecutionProvider"]


def test_resolve_providers_fallback_on_empty():
    providers = ONNXDetector._resolve_providers("")
    assert providers == ["CPUExecutionProvider"]


def test_resolve_providers_env_var(monkeypatch):
    monkeypatch.setenv("ONNX_PROVIDERS", "DirectMLExecutionProvider,CPUExecutionProvider")
    providers = ONNXDetector._resolve_providers(None)
    assert providers == ["DirectMLExecutionProvider", "CPUExecutionProvider"]


def test_resolve_providers_list_input():
    providers = ONNXDetector._resolve_providers(["CUDAExecutionProvider"])
    assert providers == ["CUDAExecutionProvider", "CPUExecutionProvider"]


def test_onnx_detector_fallback_on_session_error(monkeypatch):
    """Verify ONNXDetector falls back to CPUExecutionProvider when initial provider fails."""
    mock_ort = MagicMock()
    mock_session = MagicMock()
    mock_session.get_providers.return_value = ["CPUExecutionProvider"]

    # First call with GPU providers fails, second call with CPU succeeds
    mock_ort.InferenceSession.side_effect = [
        RuntimeError("CUDA failure: CUDA driver/toolkit not available"),
        mock_session,
    ]
    mock_ort.get_available_providers.return_value = [
        "CUDAExecutionProvider",
        "CPUExecutionProvider",
    ]

    monkeypatch.setattr("src.detection.detector.ort", mock_ort, raising=False)
    monkeypatch.setitem(sys.modules, "onnxruntime", mock_ort)

    detector = ONNXDetector(
        model_path="models/test_model.onnx",
        providers=["CUDAExecutionProvider"],
    )

    assert mock_ort.InferenceSession.call_count == 2
    # First call attempted CUDA
    first_call_providers = mock_ort.InferenceSession.call_args_list[0].kwargs.get("providers")
    assert "CUDAExecutionProvider" in first_call_providers
    # Fallback call used CPU
    second_call_providers = mock_ort.InferenceSession.call_args_list[1].kwargs.get("providers")
    assert second_call_providers == ["CPUExecutionProvider"]
    assert detector.active_providers == ["CPUExecutionProvider"]


def test_onnx_detector_default_initialization_queries_auto(monkeypatch):
    """Verify ONNXDetector with default providers auto-negotiates and sets active_providers."""
    mock_ort = MagicMock()
    mock_session = MagicMock()
    mock_session.get_providers.return_value = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    mock_ort.InferenceSession.return_value = mock_session
    mock_ort.get_available_providers.return_value = [
        "CPUExecutionProvider",
        "CUDAExecutionProvider",
    ]

    monkeypatch.setattr("src.detection.detector.ort", mock_ort, raising=False)
    monkeypatch.setitem(sys.modules, "onnxruntime", mock_ort)

    detector = ONNXDetector(model_path="models/test_model.onnx")

    mock_ort.InferenceSession.assert_called_once()
    call_providers = mock_ort.InferenceSession.call_args.kwargs.get("providers")
    assert call_providers == ["CUDAExecutionProvider", "CPUExecutionProvider"]
    assert detector.active_providers == ["CUDAExecutionProvider", "CPUExecutionProvider"]
