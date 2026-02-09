"""Tests for input validation functions."""

from __future__ import annotations

import warnings

import numpy as np
import pytest
from PIL import Image

from detr_geo._adapter import prepare_tile_image, validate_tile_input
from detr_geo.exceptions import BandError, ModelError


class TestValidateTileInput:
    """Test validate_tile_input function."""

    def test_valid_3channel_uint8_passes(self):
        img = Image.new("RGB", (256, 256))
        validate_tile_input(img, block_size=32)  # Should not raise

    def test_valid_3channel_float32_passes(self):
        arr = np.random.rand(256, 256, 3).astype(np.float32)
        validate_tile_input(arr, block_size=32)  # Should not raise

    def test_4channel_rgba_raises_band_error_pil(self):
        img = Image.new("RGBA", (256, 256))
        with pytest.raises(BandError, match="4 channels"):
            validate_tile_input(img, block_size=32)

    def test_4channel_rgba_raises_band_error_numpy(self):
        arr = np.zeros((256, 256, 4), dtype=np.uint8)
        with pytest.raises(BandError, match="4 channels"):
            validate_tile_input(arr, block_size=32)

    def test_1channel_grayscale_raises_band_error_pil(self):
        img = Image.new("L", (256, 256))
        with pytest.raises(BandError, match="1 channel"):
            validate_tile_input(img, block_size=32)

    def test_float32_above_one_raises_model_error(self):
        arr = np.ones((256, 256, 3), dtype=np.float32) * 255.0
        with pytest.raises(ModelError, match="\\[0, 1\\]"):
            validate_tile_input(arr, block_size=32)

    def test_float64_issues_warning(self):
        arr = np.random.rand(256, 256, 3).astype(np.float64)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            validate_tile_input(arr, block_size=32)
            assert any("float64" in str(warning.message) for warning in w)

    def test_non_divisible_dimensions_issue_warning(self):
        img = Image.new("RGB", (257, 257))
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            validate_tile_input(img, block_size=32)
            assert any("not divisible" in str(warning.message) for warning in w)

    def test_float32_exactly_one_passes(self):
        arr = np.ones((256, 256, 3), dtype=np.float32)
        validate_tile_input(arr, block_size=32)  # Should not raise

    def test_float32_value_1_0001_fails(self):
        arr = np.full((256, 256, 3), 1.0001, dtype=np.float32)
        with pytest.raises(ModelError, match="\\[0, 1\\]"):
            validate_tile_input(arr, block_size=32)

    def test_none_input_passes(self):
        validate_tile_input(None, block_size=32)  # Should not raise


class TestPrepareTileImage:
    """Test prepare_tile_image function."""

    def test_converts_3hw_float32_to_pil_rgb(self):
        arr = np.random.rand(3, 64, 64).astype(np.float32)
        result = prepare_tile_image(arr)
        assert result.mode == "RGB"
        assert result.size == (64, 64)

    def test_wrong_ndim_raises_band_error(self):
        arr = np.random.rand(64, 64).astype(np.float32)
        with pytest.raises(BandError, match="3D array"):
            prepare_tile_image(arr)

    def test_wrong_channels_raises_band_error(self):
        arr = np.random.rand(4, 64, 64).astype(np.float32)
        with pytest.raises(BandError, match="3 channels"):
            prepare_tile_image(arr)

    def test_output_is_uint8(self):
        arr = np.random.rand(3, 64, 64).astype(np.float32)
        result = prepare_tile_image(arr)
        result_arr = np.array(result)
        assert result_arr.dtype == np.uint8

    def test_values_clipped_to_valid_range(self):
        arr = np.full((3, 64, 64), 1.5, dtype=np.float32)
        result = prepare_tile_image(arr)
        result_arr = np.array(result)
        assert result_arr.max() == 255
