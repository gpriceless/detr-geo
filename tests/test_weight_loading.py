"""Tests for weight loading and lazy initialization with mocked rfdetr."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from detr_geo._adapter import RFDETRAdapter
from detr_geo.exceptions import ModelError


class TestLazyLoading:
    """Test lazy model loading."""

    def test_not_loaded_at_init(self):
        adapter = RFDETRAdapter("nano", device="cpu")
        assert adapter.is_loaded is False

    def test_ensure_model_called_before_predict(self):
        adapter = RFDETRAdapter("nano", device="cpu")

        mock_model = MagicMock()
        mock_det = MagicMock()
        mock_det.xyxy = __import__("numpy").array([[1, 2, 3, 4]])
        mock_det.confidence = __import__("numpy").array([0.9])
        mock_det.class_id = __import__("numpy").array([0])
        mock_det.__len__ = lambda self: 1
        mock_model.predict.return_value = mock_det

        mock_rfdetr = MagicMock()
        mock_rfdetr.RFDETRNano.return_value = mock_model

        from PIL import Image

        img = Image.new("RGB", (384, 384))

        with patch.dict("sys.modules", {"rfdetr": mock_rfdetr}):
            adapter.predict_tile(img)

        assert adapter.is_loaded is True

    def test_second_ensure_model_is_noop(self):
        adapter = RFDETRAdapter("nano", device="cpu")

        mock_model = MagicMock()
        mock_rfdetr = MagicMock()
        mock_rfdetr.RFDETRNano.return_value = mock_model

        with patch.dict("sys.modules", {"rfdetr": mock_rfdetr}):
            adapter._ensure_model()
            adapter._ensure_model()

        mock_rfdetr.RFDETRNano.assert_called_once()


class TestCustomWeights:
    """Test custom weight loading."""

    def test_nonexistent_path_raises_model_error(self):
        with pytest.raises(ModelError, match="does not exist"):
            RFDETRAdapter("nano", device="cpu", pretrain_weights="/nonexistent/weights.pt")

    def test_valid_path_accepted(self, tmp_path):
        weights_file = tmp_path / "weights.pt"
        weights_file.touch()
        adapter = RFDETRAdapter("nano", device="cpu", pretrain_weights=str(weights_file))
        assert adapter._pretrain_weights == str(weights_file)


class TestMissingRfdetr:
    """Test behavior when rfdetr is not installed."""

    @pytest.mark.skipif(
        __import__("tests.conftest", fromlist=["rfdetr_installed"]).rfdetr_installed(),
        reason="Test assumes rfdetr is NOT installed",
    )
    def test_rfdetr_not_installed_raises_model_error(self):
        adapter = RFDETRAdapter("nano", device="cpu")

        from PIL import Image

        img = Image.new("RGB", (384, 384))

        with pytest.raises(ModelError, match="pip install"):
            adapter.predict_tile(img)


class TestModelReconfiguration:
    """Test model can be reconfigured before first predict."""

    def test_change_threshold_before_predict(self):
        adapter = RFDETRAdapter("nano", device="cpu", confidence_threshold=0.5)
        adapter._confidence_threshold = 0.8
        assert adapter._confidence_threshold == 0.8
        assert not adapter.is_loaded
