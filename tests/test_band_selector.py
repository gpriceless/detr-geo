"""Tests for BandSelector class."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from detr_geo.exceptions import BandError
from detr_geo.io import BandSelector


class TestPresets:
    """Test band preset selection."""

    def test_rgb_selects_bands_123_from_4band(self):
        sel = BandSelector("rgb")
        data = np.random.randint(0, 255, (4, 64, 64), dtype=np.uint8)
        rgb, alpha = sel.select(data, num_bands=4)
        assert rgb.shape == (3, 64, 64)
        np.testing.assert_array_equal(rgb[0], data[0])
        np.testing.assert_array_equal(rgb[1], data[1])
        np.testing.assert_array_equal(rgb[2], data[2])

    def test_sentinel2_rgb_selects_bands_432(self):
        sel = BandSelector("sentinel2_rgb")
        assert sel.band_indices == [4, 3, 2]
        data = np.random.randint(0, 255, (13, 64, 64), dtype=np.uint8)
        rgb, alpha = sel.select(data, num_bands=13)
        assert rgb.shape == (3, 64, 64)
        # Band 4 (0-indexed: 3), Band 3 (0-indexed: 2), Band 2 (0-indexed: 1)
        np.testing.assert_array_equal(rgb[0], data[3])
        np.testing.assert_array_equal(rgb[1], data[2])
        np.testing.assert_array_equal(rgb[2], data[1])

    def test_worldview_rgb_selects_bands_532_from_8band(self):
        sel = BandSelector("worldview_rgb")
        assert sel.band_indices == [5, 3, 2]
        data = np.random.randint(0, 255, (8, 64, 64), dtype=np.uint8)
        rgb, alpha = sel.select(data, num_bands=8)
        assert rgb.shape == (3, 64, 64)
        np.testing.assert_array_equal(rgb[0], data[4])
        np.testing.assert_array_equal(rgb[1], data[2])
        np.testing.assert_array_equal(rgb[2], data[1])

    def test_naip_rgb_same_as_rgb(self):
        sel = BandSelector("naip_rgb")
        assert sel.band_indices == [1, 2, 3]


class TestCustomBands:
    """Test custom band tuple selection."""

    def test_custom_tuple_432_works(self):
        sel = BandSelector((4, 3, 2))
        assert sel.band_indices == [4, 3, 2]
        data = np.random.randint(0, 255, (13, 64, 64), dtype=np.uint8)
        rgb, _ = sel.select(data, num_bands=13)
        np.testing.assert_array_equal(rgb[0], data[3])
        np.testing.assert_array_equal(rgb[1], data[2])
        np.testing.assert_array_equal(rgb[2], data[1])


class TestSingleBandTriplication:
    """Test single-band raster handling."""

    def test_single_band_triplicated_with_warning(self):
        sel = BandSelector((1,))
        data = np.random.randint(0, 255, (1, 64, 64), dtype=np.uint8)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            rgb, alpha = sel.select(data, num_bands=1)
            assert len(w) == 1
            assert "Triplicating" in str(w[0].message)
        assert rgb.shape == (3, 64, 64)
        np.testing.assert_array_equal(rgb[0], data[0])
        np.testing.assert_array_equal(rgb[1], data[0])
        np.testing.assert_array_equal(rgb[2], data[0])


class TestAlphaDetection:
    """Test alpha band extraction."""

    def test_4band_rgba_extracts_alpha(self):
        sel = BandSelector("rgb")
        data = np.random.randint(0, 255, (4, 64, 64), dtype=np.uint8)
        # Set some pixels to transparent (alpha = 0)
        data[3, :10, :10] = 0
        data[3, 10:, :] = 255

        rgb, alpha = sel.select(data, num_bands=4)
        assert alpha is not None
        assert alpha.shape == (64, 64)
        # Alpha == 0 means nodata (True)
        assert alpha[:10, :10].all()
        assert not alpha[10:, :].any()


class TestValidation:
    """Test error handling."""

    def test_band_index_exceeds_count_raises_band_error(self):
        sel = BandSelector((5, 3, 2))
        data = np.random.randint(0, 255, (3, 64, 64), dtype=np.uint8)
        with pytest.raises(BandError, match="exceeds"):
            sel.select(data, num_bands=3)

    def test_band_index_zero_raises_band_error(self):
        with pytest.raises(BandError, match="1-indexed"):
            BandSelector((0, 1, 2))

    def test_invalid_preset_raises_band_error(self):
        with pytest.raises(BandError, match="Unknown"):
            BandSelector("invalid_preset")

    def test_empty_tuple_raises_band_error(self):
        with pytest.raises(BandError, match="empty"):
            BandSelector(())
