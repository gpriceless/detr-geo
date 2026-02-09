"""Tests for batch inference and optimization with mocked model."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image

from detr_geo._adapter import RFDETRAdapter


class TestPredictTiles:
    """Test predict_tiles method."""

    def _make_mocked_adapter(self):
        """Create an adapter with a mocked rfdetr model."""
        adapter = RFDETRAdapter("nano", device="cpu")

        mock_model_instance = MagicMock()

        def mock_predict(image, threshold=0.5):
            det = MagicMock()
            det.xyxy = np.array([[10.0, 20.0, 30.0, 40.0]])
            det.confidence = np.array([0.9])
            det.class_id = np.array([0])
            det.__len__ = lambda self: 1
            return det

        mock_model_instance.predict = mock_predict

        mock_rfdetr = MagicMock()
        mock_rfdetr.RFDETRNano.return_value = mock_model_instance

        return adapter, mock_rfdetr

    def test_single_image_returns_list_of_one(self):
        adapter, mock_rfdetr = self._make_mocked_adapter()
        img = Image.new("RGB", (384, 384))

        with patch.dict("sys.modules", {"rfdetr": mock_rfdetr}):
            results = adapter.predict_tiles([img])

        assert len(results) == 1
        assert "bbox" in results[0]
        assert "confidence" in results[0]
        assert "class_id" in results[0]

    def test_multiple_images_returns_correct_length(self):
        adapter, mock_rfdetr = self._make_mocked_adapter()
        imgs = [Image.new("RGB", (384, 384)) for _ in range(3)]

        with patch.dict("sys.modules", {"rfdetr": mock_rfdetr}):
            results = adapter.predict_tiles(imgs)

        assert len(results) == 3

    def test_each_result_has_correct_keys(self):
        adapter, mock_rfdetr = self._make_mocked_adapter()
        imgs = [Image.new("RGB", (384, 384)) for _ in range(2)]

        with patch.dict("sys.modules", {"rfdetr": mock_rfdetr}):
            results = adapter.predict_tiles(imgs)

        for result in results:
            assert "bbox" in result
            assert "confidence" in result
            assert "class_id" in result

    def test_predict_batch_backward_compat(self):
        """Test that predict_batch still works for backward compatibility."""
        adapter, mock_rfdetr = self._make_mocked_adapter()
        imgs = [Image.new("RGB", (384, 384)) for _ in range(2)]

        with patch.dict("sys.modules", {"rfdetr": mock_rfdetr}), pytest.warns(DeprecationWarning):
            results = adapter.predict_batch(imgs)

        assert len(results) == 2


class TestAutoBatchSize:
    """Test auto_batch_size method."""

    def test_cpu_returns_one(self):
        adapter = RFDETRAdapter("nano", device="cpu")
        assert adapter.auto_batch_size() == 1

    def test_gpu_with_mocked_memory_returns_greater_than_one(self):
        adapter = RFDETRAdapter("nano", device="cpu")
        adapter._device = "cuda"  # Override for test

        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        # 8 GB free memory
        mock_torch.cuda.mem_get_info.return_value = (8 * 1024**3, 16 * 1024**3)

        with patch.dict("sys.modules", {"torch": mock_torch}):
            batch_size = adapter.auto_batch_size()

        assert batch_size > 1


class TestOptimize:
    """Test optimization methods."""

    def test_optimize_calls_model_method(self):
        adapter = RFDETRAdapter("nano", device="cpu")

        mock_model = MagicMock()
        mock_model.optimize_for_inference = MagicMock()

        mock_rfdetr = MagicMock()
        mock_rfdetr.RFDETRNano.return_value = mock_model

        with patch.dict("sys.modules", {"rfdetr": mock_rfdetr}):
            adapter.optimize(batch_size=4)

        mock_model.optimize_for_inference.assert_called_once_with(batch_size=4)
        assert adapter._is_optimized is True

    def test_remove_optimization_clears_state(self):
        adapter = RFDETRAdapter("nano", device="cpu")
        adapter._is_optimized = True
        adapter.remove_optimization()
        assert adapter._is_optimized is False
