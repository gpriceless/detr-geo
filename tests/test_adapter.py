"""Tests for the RFDETRAdapter with mocked rfdetr."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from detr_geo._adapter import RFDETRAdapter
from detr_geo.exceptions import BandError, ModelError


class TestVariants:
    """Test model variant mapping."""

    def test_all_five_sizes_present(self):
        expected = {"nano", "small", "medium", "base", "large"}
        assert set(RFDETRAdapter.VARIANTS.keys()) == expected

    def test_variant_details_has_all_five(self):
        expected = {"nano", "small", "medium", "base", "large"}
        assert set(RFDETRAdapter.VARIANT_DETAILS.keys()) == expected

    @pytest.mark.parametrize(
        "size, expected_resolution, expected_block_size",
        [
            ("nano", 384, 32),
            ("small", 512, 32),
            ("medium", 576, 32),
            ("base", 560, 56),
            ("large", 704, 32),
        ],
    )
    def test_variant_properties(self, size, expected_resolution, expected_block_size):
        adapter = RFDETRAdapter(size, device="cpu")
        assert adapter.resolution == expected_resolution
        assert adapter.block_size == expected_block_size

    def test_each_variant_has_all_fields(self):
        required_fields = {
            "class_name",
            "resolution",
            "block_size",
            "patch_size",
            "num_windows",
            "license",
        }
        for size, details in RFDETRAdapter.VARIANT_DETAILS.items():
            for field in required_fields:
                assert field in details, f"Missing '{field}' in variant '{size}'"

    def test_block_size_equals_patch_times_windows(self):
        for size, details in RFDETRAdapter.VARIANT_DETAILS.items():
            assert details["block_size"] == details["patch_size"] * details["num_windows"], (
                f"block_size mismatch for {size}"
            )


class TestUnsupportedVariants:
    """Test unsupported variant handling."""

    def test_unsupported_provides_license_explanation(self):
        for _name, msg in RFDETRAdapter.UNSUPPORTED_VARIANTS.items():
            assert "Roboflow Platform License" in msg

    def test_xlarge_raises_model_error_with_license(self):
        with pytest.raises(ModelError, match="Roboflow Platform License"):
            RFDETRAdapter("xlarge")

    def test_2xlarge_raises_model_error_with_license(self):
        with pytest.raises(ModelError, match="Roboflow Platform License"):
            RFDETRAdapter("2xlarge")

    def test_giant_raises_model_error_listing_valid_sizes(self):
        with pytest.raises(ModelError, match="Valid sizes"):
            RFDETRAdapter("giant")


class TestInvalidModelSize:
    """Test error handling for unsupported model sizes."""

    def test_invalid_raises_model_error_with_valid_sizes(self):
        with pytest.raises(ModelError, match="Valid sizes"):
            RFDETRAdapter("invalid")

    def test_invalid_error_lists_all_sizes(self):
        with pytest.raises(ModelError) as exc_info:
            RFDETRAdapter("invalid")
        msg = str(exc_info.value)
        for size in ("base", "large", "medium", "nano", "small"):
            assert size in msg


class TestPatchSizeProperty:
    """Test patch_size property."""

    def test_base_patch_size_is_14(self):
        adapter = RFDETRAdapter("base", device="cpu")
        assert adapter.patch_size == 14

    @pytest.mark.parametrize("size", ["nano", "small", "medium", "large"])
    def test_non_base_patch_size_is_16(self, size):
        adapter = RFDETRAdapter(size, device="cpu")
        assert adapter.patch_size == 16


class TestNumWindowsProperty:
    """Test num_windows property."""

    def test_base_num_windows_is_4(self):
        adapter = RFDETRAdapter("base", device="cpu")
        assert adapter.num_windows == 4

    @pytest.mark.parametrize("size", ["nano", "small", "medium", "large"])
    def test_non_base_num_windows_is_2(self, size):
        adapter = RFDETRAdapter(size, device="cpu")
        assert adapter.num_windows == 2


class TestModelInfo:
    """Test model_info() method."""

    def test_model_info_returns_complete_dict(self):
        adapter = RFDETRAdapter("nano", device="cpu")
        info = adapter.model_info()
        assert info["model_size"] == "nano"
        assert info["resolution"] == 384
        assert info["block_size"] == 32
        assert info["patch_size"] == 16
        assert info["num_windows"] == 2
        assert info["license"] == "Apache 2.0"
        assert "device" in info
        assert "is_loaded" in info
        assert "confidence_threshold" in info

    def test_model_info_all_variants(self):
        for size in RFDETRAdapter.VARIANT_DETAILS:
            adapter = RFDETRAdapter(size, device="cpu")
            info = adapter.model_info()
            assert info["model_size"] == size
            assert isinstance(info["resolution"], int)
            assert isinstance(info["block_size"], int)


class TestBlockSize:
    """Test block_size property."""

    def test_base_block_size_is_56(self):
        adapter = RFDETRAdapter("base", device="cpu")
        assert adapter.block_size == 56

    @pytest.mark.parametrize("size", ["nano", "small", "medium", "large"])
    def test_non_base_block_size_is_32(self, size):
        adapter = RFDETRAdapter(size, device="cpu")
        assert adapter.block_size == 32


class TestValidTileSizes:
    """Test valid tile size calculation."""

    def test_base_tile_sizes_divisible_by_56(self):
        adapter = RFDETRAdapter("base", device="cpu")
        sizes = adapter.valid_tile_sizes(min_size=128, max_size=2048)
        assert all(s % 56 == 0 for s in sizes)
        assert all(128 <= s <= 2048 for s in sizes)
        assert len(sizes) > 0

    def test_medium_tile_sizes_divisible_by_32(self):
        adapter = RFDETRAdapter("medium", device="cpu")
        sizes = adapter.valid_tile_sizes(min_size=128, max_size=2048)
        assert all(s % 32 == 0 for s in sizes)
        assert all(128 <= s <= 2048 for s in sizes)
        assert len(sizes) > 0

    def test_tile_sizes_sorted(self):
        adapter = RFDETRAdapter("nano", device="cpu")
        sizes = adapter.valid_tile_sizes()
        assert sizes == sorted(sizes)


class TestDetectDevice:
    """Test device detection."""

    def test_returns_string(self):
        RFDETRAdapter("nano", device="cpu")
        from detr_geo._adapter import detect_device

        result = detect_device()
        assert result in ("cuda", "mps", "cpu")

    def test_explicit_device_skips_detection(self):
        adapter = RFDETRAdapter("nano", device="cpu")
        assert adapter._device == "cpu"


class TestIsLoaded:
    """Test lazy loading state."""

    def test_not_loaded_at_init(self):
        adapter = RFDETRAdapter("nano", device="cpu")
        assert adapter.is_loaded is False


class TestNormalizeDetections:
    """Test detection normalization from supervision format."""

    def test_normalize_mock_detections(self):
        adapter = RFDETRAdapter("nano", device="cpu")

        mock_detections = MagicMock()
        mock_detections.xyxy = np.array([[10.0, 20.0, 30.0, 40.0], [50.0, 60.0, 70.0, 80.0]])
        mock_detections.confidence = np.array([0.9, 0.8])
        mock_detections.class_id = np.array([0, 1])
        mock_detections.__len__ = lambda self: 2

        result = adapter._normalize_detections(mock_detections)

        assert result["bbox"] == [
            [10.0, 20.0, 30.0, 40.0],
            [50.0, 60.0, 70.0, 80.0],
        ]
        assert result["confidence"] == [0.9, 0.8]
        assert result["class_id"] == [0, 1]

    def test_normalize_empty_detections(self):
        adapter = RFDETRAdapter("nano", device="cpu")

        mock_detections = MagicMock()
        mock_detections.__len__ = lambda self: 0

        result = adapter._normalize_detections(mock_detections)
        assert result["bbox"] == []
        assert result["confidence"] == []
        assert result["class_id"] == []

    def test_normalize_none_detections(self):
        adapter = RFDETRAdapter("nano", device="cpu")
        result = adapter._normalize_detections(None)
        assert result["bbox"] == []


class TestPredictWithoutRfdetr:
    """Test that predict_tile without rfdetr raises ModelError."""

    @pytest.mark.skipif(
        __import__("tests.conftest", fromlist=["rfdetr_installed"]).rfdetr_installed(),
        reason="Test assumes rfdetr is NOT installed",
    )
    def test_predict_tile_without_rfdetr(self):
        adapter = RFDETRAdapter("nano", device="cpu")

        from PIL import Image

        img = Image.new("RGB", (384, 384))

        with pytest.raises(ModelError, match="pip install detr-geo\\[rfdetr\\]"):
            adapter.predict_tile(img)


class TestInputValidation:
    """Test image input validation."""

    def test_four_channel_image_raises_band_error(self):
        adapter = RFDETRAdapter("nano", device="cpu")

        from PIL import Image

        img = Image.new("RGBA", (384, 384))

        with pytest.raises(BandError, match="4 channels"):
            adapter.predict_tile(img)

    def test_float32_values_above_one_raises_error(self):
        adapter = RFDETRAdapter("nano", device="cpu")

        arr = np.ones((384, 384, 3), dtype=np.float32) * 255.0

        with pytest.raises(ModelError, match="\\[0, 1\\]"):
            adapter.predict_tile(arr)


class TestPredictWithMockedModel:
    """Test prediction with a fully mocked rfdetr model."""

    def test_predict_tile_loads_model(self):
        adapter = RFDETRAdapter("nano", device="cpu")

        mock_model_instance = MagicMock()
        mock_sv_detections = MagicMock()
        mock_sv_detections.xyxy = np.array([[10.0, 20.0, 30.0, 40.0]])
        mock_sv_detections.confidence = np.array([0.95])
        mock_sv_detections.class_id = np.array([0])
        mock_sv_detections.__len__ = lambda self: 1
        mock_model_instance.predict.return_value = mock_sv_detections

        mock_rfdetr = MagicMock()
        mock_rfdetr.RFDETRNano.return_value = mock_model_instance

        from PIL import Image

        img = Image.new("RGB", (384, 384))

        with patch.dict("sys.modules", {"rfdetr": mock_rfdetr}):
            result = adapter.predict_tile(img)

        assert adapter.is_loaded is True
        assert result["bbox"] == [[10.0, 20.0, 30.0, 40.0]]
        assert result["confidence"] == [0.95]
        assert result["class_id"] == [0]

    def test_predict_batch_deprecation_warning(self):
        """Test that predict_batch issues a deprecation warning."""
        adapter = RFDETRAdapter("nano", device="cpu")

        mock_model_instance = MagicMock()
        mock_sv_detections = MagicMock()
        mock_sv_detections.xyxy = np.array([[10.0, 20.0, 30.0, 40.0]])
        mock_sv_detections.confidence = np.array([0.95])
        mock_sv_detections.class_id = np.array([0])
        mock_sv_detections.__len__ = lambda self: 1
        mock_model_instance.predict.return_value = mock_sv_detections

        mock_rfdetr = MagicMock()
        mock_rfdetr.RFDETRNano.return_value = mock_model_instance

        from PIL import Image

        img = Image.new("RGB", (384, 384))

        with patch.dict("sys.modules", {"rfdetr": mock_rfdetr}):
            with pytest.warns(DeprecationWarning, match="predict_batch.*deprecated"):
                result = adapter.predict_batch([img])

        assert len(result) == 1


class TestWeightLoading:
    """Test weight loading and lazy initialization."""

    def test_nonexistent_weights_raises_model_error(self):
        with pytest.raises(ModelError, match="does not exist"):
            RFDETRAdapter("nano", device="cpu", pretrain_weights="/nonexistent/path.pt")

    def test_second_ensure_model_is_noop(self):
        adapter = RFDETRAdapter("nano", device="cpu")

        mock_model_instance = MagicMock()
        mock_rfdetr = MagicMock()
        mock_rfdetr.RFDETRNano.return_value = mock_model_instance

        with patch.dict("sys.modules", {"rfdetr": mock_rfdetr}):
            adapter._ensure_model()
            adapter._ensure_model()  # Second call should be no-op

        # RFDETRNano should only be called once
        mock_rfdetr.RFDETRNano.assert_called_once()
