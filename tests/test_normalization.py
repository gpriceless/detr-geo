"""Tests for normalization pipeline."""

from __future__ import annotations

import numpy as np

from detr_geo.io import normalize_to_float32


class TestNormalizeUint8:
    """Test uint8 normalization."""

    def test_uint8_normalized_to_0_1(self):
        data = np.random.randint(0, 255, (3, 64, 64), dtype=np.uint8)
        result, params = normalize_to_float32(data, stretch="minmax")
        assert result.dtype == np.float32
        assert result.min() >= 0.0
        assert result.max() <= 1.0


class TestNormalizeUint16:
    """Test 16-bit normalization."""

    def test_uint16_percentile_stretch_to_0_1(self):
        data = np.random.randint(0, 10000, (3, 64, 64), dtype=np.uint16)
        result, params = normalize_to_float32(data, stretch="percentile")
        assert result.dtype == np.float32
        assert result.min() >= 0.0
        assert result.max() <= 1.0


class TestNormalizeFloat32:
    """Test float32 passthrough."""

    def test_float32_passthrough_with_none_stretch(self):
        data = np.random.rand(3, 64, 64).astype(np.float32)
        result, params = normalize_to_float32(data, stretch="none")
        assert result.dtype == np.float32
        assert result.min() >= 0.0
        assert result.max() <= 1.0


class TestNodataMask:
    """Test nodata exclusion from percentile computation."""

    def test_nodata_excluded_from_percentiles(self):
        data = np.full((3, 64, 64), 100, dtype=np.uint16)
        # Set some pixels to nodata value
        data[:, :10, :10] = 0

        nodata_mask = np.zeros((64, 64), dtype=bool)
        nodata_mask[:10, :10] = True

        result, params = normalize_to_float32(data, stretch="percentile", nodata_mask=nodata_mask)
        assert result.dtype == np.float32
        assert result.max() <= 1.0


class TestStretchParams:
    """Test pre-computed stretch params."""

    def test_precomputed_params_produce_identical_output(self):
        data = np.random.randint(0, 10000, (3, 64, 64), dtype=np.uint16)

        # First pass: compute params
        result1, params = normalize_to_float32(data, stretch="percentile")

        # Second pass: use pre-computed params
        result2, _ = normalize_to_float32(data, stretch_params=params)

        np.testing.assert_array_almost_equal(result1, result2)


class TestPerBandStretch:
    """Test per-band normalization."""

    def test_bands_with_different_ranges(self):
        data = np.zeros((3, 64, 64), dtype=np.uint16)
        data[0] = np.random.randint(0, 100, (64, 64))
        data[1] = np.random.randint(500, 1000, (64, 64))
        data[2] = np.random.randint(5000, 10000, (64, 64))

        result, params = normalize_to_float32(data, stretch="percentile")
        assert result.dtype == np.float32
        assert result.max() <= 1.0

        # Each band's params should reflect different ranges
        assert params.vmin[0] < params.vmin[1] < params.vmin[2]


class TestOutputConstraints:
    """Test output always meets RF-DETR constraints."""

    def test_output_dtype_always_float32(self):
        for dtype in [np.uint8, np.uint16, np.int32, np.float64]:
            data = np.random.randint(0, 255, (3, 64, 64)).astype(dtype)
            result, _ = normalize_to_float32(data, stretch="minmax")
            assert result.dtype == np.float32

    def test_output_never_exceeds_1(self):
        data = np.random.randint(0, 65535, (3, 64, 64), dtype=np.uint16)
        result, _ = normalize_to_float32(data, stretch="percentile")
        assert result.max() <= 1.0

    def test_output_never_below_0(self):
        data = np.random.randint(0, 65535, (3, 64, 64), dtype=np.uint16)
        result, _ = normalize_to_float32(data, stretch="percentile")
        assert result.min() >= 0.0
