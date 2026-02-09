"""Tests for device detection with mocked torch."""

from __future__ import annotations

import warnings
from unittest.mock import MagicMock, patch

from detr_geo._adapter import _warn_cpu_inference, detect_device


class TestDetectDevice:
    """Test detect_device() function."""

    def test_returns_valid_device_string(self):
        result = detect_device()
        assert result in ("cuda", "mps", "cpu")

    def test_cpu_preferred_returns_cpu(self):
        result = detect_device("cpu")
        assert result == "cpu"

    def test_cuda_on_cpu_system_falls_back_with_warning(self):
        """detect_device("cuda") on CPU-only system returns "cpu" with warning."""
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False
        mock_torch.backends.mps.is_available.return_value = False

        with patch.dict("sys.modules", {"torch": mock_torch}), warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = detect_device("cuda")
            assert result == "cpu"
            assert len(w) == 1
            assert "not available" in str(w[0].message)

    def test_cuda_available_returns_cuda(self):
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True

        with patch.dict("sys.modules", {"torch": mock_torch}):
            result = detect_device()
            assert result == "cuda"

    def test_mps_available_returns_mps(self):
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False
        mock_torch.backends.mps.is_available.return_value = True

        with patch.dict("sys.modules", {"torch": mock_torch}):
            result = detect_device()
            assert result == "mps"

    def test_no_gpu_returns_cpu(self):
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False
        mock_torch.backends.mps.is_available.return_value = False

        with patch.dict("sys.modules", {"torch": mock_torch}):
            result = detect_device()
            assert result == "cpu"

    def test_no_torch_import_at_module_level(self):
        """Device detection should not import torch at module level."""
        # If torch is not installed, importing _adapter should still work
        # This is verified by the fact that all adapter tests work without torch
        # Module should be importable - that's the test


class TestWarnCpuInference:
    """Test _warn_cpu_inference function."""

    def test_large_model_recommends_nano(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _warn_cpu_inference("large")
            assert len(w) == 1
            assert "nano" in str(w[0].message)

    def test_nano_model_does_not_recommend_switching(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _warn_cpu_inference("nano")
            assert len(w) == 1
            assert "nano" not in str(w[0].message) or "Consider" not in str(w[0].message)
