"""Integration tests for edge cases with real-like data.

Tests adapted from scripts/test_edge_cases.py but integrated into pytest.
No model inference required - tests pipeline plumbing with edge case imagery.
"""

import numpy as np
import pytest
import rasterio
from pyproj import CRS
from rasterio.transform import from_bounds

from detr_geo.io import BandSelector, compute_nodata_fraction, normalize_to_float32, read_tile


@pytest.mark.integration
class TestEdgeCaseNormalization:
    """Test normalization with edge case pixel values."""

    def test_all_black_normalization(self, tmp_path):
        """All-zero data normalizes without division-by-zero."""
        # Create an all-black GeoTIFF
        tiff_path = tmp_path / "all_black.tif"
        width, height = 128, 128
        transform = from_bounds(-121.9, 37.33, -121.89, 37.34, width, height)

        data = np.zeros((3, height, width), dtype=np.uint8)

        with rasterio.open(
            tiff_path,
            "w",
            driver="GTiff",
            height=height,
            width=width,
            count=3,
            dtype="uint8",
            crs=CRS.from_epsg(4326),
            transform=transform,
        ) as dst:
            dst.write(data)

        # Read and normalize
        window = (0, 0, 128, 128)
        tile, nodata_mask = read_tile(str(tiff_path), window, bands=[1, 2, 3])

        # Normalize with percentile stretch
        normalized, params = normalize_to_float32(tile, stretch="percentile")

        # Should not raise, should produce valid output
        assert normalized.dtype == np.float32
        assert normalized.shape == (3, 128, 128)
        assert not np.isnan(normalized).any()
        assert not np.isinf(normalized).any()
        # All-black should normalize to all zeros (or very close)
        assert np.allclose(normalized, 0.0, atol=1e-6)

    def test_all_white_normalization(self, tmp_path):
        """All-255 data normalizes without error."""
        # Create an all-white GeoTIFF
        tiff_path = tmp_path / "all_white.tif"
        width, height = 128, 128
        transform = from_bounds(-121.9, 37.33, -121.89, 37.34, width, height)

        data = np.full((3, height, width), 255, dtype=np.uint8)

        with rasterio.open(
            tiff_path,
            "w",
            driver="GTiff",
            height=height,
            width=width,
            count=3,
            dtype="uint8",
            crs=CRS.from_epsg(4326),
            transform=transform,
        ) as dst:
            dst.write(data)

        # Read and normalize
        window = (0, 0, 128, 128)
        tile, nodata_mask = read_tile(str(tiff_path), window, bands=[1, 2, 3])

        # Normalize with percentile stretch
        normalized, params = normalize_to_float32(tile, stretch="percentile")

        # Should not raise, should produce valid output
        assert normalized.dtype == np.float32
        assert normalized.shape == (3, 128, 128)
        assert not np.isnan(normalized).any()
        assert not np.isinf(normalized).any()
        # All-white with percentile stretch: no dynamic range, may normalize to 0 or 1
        # Just verify it's constant and in valid range
        assert normalized.min() >= 0.0
        assert normalized.max() <= 1.0

    def test_constant_value_normalization(self, tmp_path):
        """Data with constant value normalizes safely."""
        # Create a GeoTIFF with constant value (128 everywhere)
        tiff_path = tmp_path / "constant.tif"
        width, height = 128, 128
        transform = from_bounds(-121.9, 37.33, -121.89, 37.34, width, height)

        data = np.full((3, height, width), 128, dtype=np.uint8)

        with rasterio.open(
            tiff_path,
            "w",
            driver="GTiff",
            height=height,
            width=width,
            count=3,
            dtype="uint8",
            crs=CRS.from_epsg(4326),
            transform=transform,
        ) as dst:
            dst.write(data)

        # Read and normalize
        window = (0, 0, 128, 128)
        tile, nodata_mask = read_tile(str(tiff_path), window, bands=[1, 2, 3])

        # Normalize with percentile stretch
        normalized, params = normalize_to_float32(tile, stretch="percentile")

        # Should not raise
        assert normalized.dtype == np.float32
        assert not np.isnan(normalized).any()
        assert not np.isinf(normalized).any()


@pytest.mark.integration
class TestEdgeCaseBandSelector:
    """Test BandSelector with edge case band configurations."""

    def test_single_band_selector_triplicate(self):
        """BandSelector triplicates single-band data."""
        import warnings

        selector = BandSelector((1,))

        # Create single-band data
        data = np.random.RandomState(42).randint(0, 255, (1, 64, 64), dtype=np.uint8)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            rgb, alpha = selector.select(data, num_bands=1)
            # Should get warning about triplication
            assert len(w) == 1
            assert "Triplicating" in str(w[0].message)

        # Should produce 3-band output
        assert rgb.shape == (3, 64, 64)
        # All bands should be identical
        assert np.array_equal(rgb[0], data[0])
        assert np.array_equal(rgb[1], data[0])
        assert np.array_equal(rgb[2], data[0])

    def test_single_band_explicit_list(self):
        """BandSelector with explicit band list [1] works."""
        selector = BandSelector((1,))

        data = np.random.RandomState(43).randint(0, 255, (1, 64, 64), dtype=np.uint8)

        # Should produce 1-band output when triplicating
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rgb, alpha = selector.select(data, num_bands=1)
            # Will be triplicated to 3 bands
            assert rgb.shape == (3, 64, 64)


@pytest.mark.integration
class TestEdgCase16BitNormalization:
    """Test normalization with 16-bit data."""

    def test_16bit_normalization(self, real_geotiff_rgb_uint16):
        """uint16 data with percentile stretch to [0,1]."""
        window = (0, 0, 128, 128)
        tile, nodata_mask = read_tile(real_geotiff_rgb_uint16, window, bands=[1, 2, 3])

        assert tile.dtype == np.uint16

        # Normalize with percentile stretch
        normalized, params = normalize_to_float32(tile, stretch="percentile")

        # Should produce valid float32 in [0,1]
        assert normalized.dtype == np.float32
        assert normalized.shape == (3, 128, 128)
        assert normalized.min() >= 0.0
        assert normalized.max() <= 1.0
        assert not np.isnan(normalized).any()
        assert not np.isinf(normalized).any()

    def test_16bit_minmax_stretch(self, real_geotiff_rgb_uint16):
        """uint16 data with minmax stretch."""
        window = (0, 0, 128, 128)
        tile, nodata_mask = read_tile(real_geotiff_rgb_uint16, window, bands=[1, 2, 3])

        # Normalize with minmax stretch
        normalized, params = normalize_to_float32(tile, stretch="minmax")

        assert normalized.dtype == np.float32
        assert normalized.shape == (3, 128, 128)
        # minmax should produce [0,1] range
        assert normalized.min() >= 0.0
        assert normalized.max() <= 1.0
        assert not np.isnan(normalized).any()
        assert not np.isinf(normalized).any()


@pytest.mark.integration
class TestEdgeCaseNodataFraction:
    """Test nodata fraction computation with high nodata content."""

    def test_nodata_80pct_skip(self, real_geotiff_with_nodata):
        """Tile with >80% nodata should be flagged."""
        # Read the top-right corner where nodata exists
        window = (96, 0, 32, 32)
        tile, nodata_mask = read_tile(real_geotiff_with_nodata, window, bands=[1, 2, 3])

        # This tile is 100% nodata
        assert nodata_mask is not None
        nodata_fraction = compute_nodata_fraction(tile, nodata_value=0, alpha_mask=nodata_mask)
        assert nodata_fraction == 1.0
        # Should be skipped with threshold=0.8
        assert nodata_fraction > 0.8

    def test_partial_nodata(self, real_geotiff_with_nodata):
        """Tile with <50% nodata should not be skipped."""
        # Read the center-left region with some nodata patches
        window = (0, 40, 64, 64)
        tile, nodata_mask = read_tile(real_geotiff_with_nodata, window, bands=[1, 2, 3])

        # This region has some nodata but not overwhelming
        if nodata_mask is not None:
            nodata_fraction = compute_nodata_fraction(tile, nodata_value=0, alpha_mask=nodata_mask)
            assert nodata_fraction < 0.5


@pytest.mark.integration
class TestEdgeCaseSmallImages:
    """Test with very small images."""

    def test_tiny_image_64px(self, tmp_path):
        """Handle 64x64 image (smaller than typical tile size)."""
        tiff_path = tmp_path / "tiny.tif"
        width, height = 64, 64
        transform = from_bounds(-121.9, 37.33, -121.89, 37.34, width, height)

        rng = np.random.RandomState(50)
        data = rng.randint(50, 200, (3, height, width), dtype=np.uint8)

        with rasterio.open(
            tiff_path,
            "w",
            driver="GTiff",
            height=height,
            width=width,
            count=3,
            dtype="uint8",
            crs=CRS.from_epsg(4326),
            transform=transform,
        ) as dst:
            dst.write(data)

        # Should be able to read and normalize
        from detr_geo.io import load_raster_metadata

        meta = load_raster_metadata(str(tiff_path))

        assert meta.width == 64
        assert meta.height == 64

        # Read full image
        window = (0, 0, 64, 64)
        tile, nodata_mask = read_tile(str(tiff_path), window, bands=[1, 2, 3])

        normalized, params = normalize_to_float32(tile, stretch="percentile")
        assert normalized.shape == (3, 64, 64)
