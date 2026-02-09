"""Integration tests for detr_geo.io with real GeoTIFF fixtures.

Tests I/O functions using small real GeoTIFF files instead of synthetic data.
Exercises the full pipeline including:
- Metadata loading from real rasters
- Windowed reads with rasterio
- Normalization with real pixel value distributions
- Band selection with various band counts and dtypes
"""

import numpy as np
import pytest
import rasterio

from detr_geo.io import (
    BandSelector,
    RasterMetadata,
    compute_scene_stretch_params,
    load_raster_metadata,
    normalize_to_float32,
    read_tile,
)


@pytest.mark.integration
class TestRealDataIO:
    """Test io.py functions with real GeoTIFF fixtures."""

    def test_load_real_metadata_geographic(self, real_geotiff_rgb_uint8):
        """Load metadata from a real 3-band uint8 GeoTIFF in EPSG:4326."""
        meta = load_raster_metadata(real_geotiff_rgb_uint8)

        assert isinstance(meta, RasterMetadata)
        assert meta.crs is not None
        assert meta.crs.to_epsg() == 4326
        assert meta.width == 128
        assert meta.height == 128
        assert meta.count == 3
        assert meta.dtype == "uint8"
        assert meta.transform is not None
        assert meta.bounds is not None
        # Bounds should be in degrees (lon/lat)
        assert -180 <= meta.bounds[0] <= 180  # west
        assert -90 <= meta.bounds[1] <= 90  # south

    def test_load_real_metadata_utm(self, real_geotiff_rgb_uint8_utm):
        """Load metadata from a real 3-band uint8 GeoTIFF in EPSG:32610."""
        meta = load_raster_metadata(real_geotiff_rgb_uint8_utm)

        assert isinstance(meta, RasterMetadata)
        assert meta.crs is not None
        assert meta.crs.to_epsg() == 32610
        assert meta.width == 128
        assert meta.height == 128
        assert meta.count == 3
        assert meta.dtype == "uint8"
        # UTM coordinates should be in meters, large values
        assert meta.bounds[0] > 100000  # easting (west)

    def test_load_single_band_metadata(self, real_geotiff_single_band):
        """Load metadata from a single-band GeoTIFF."""
        meta = load_raster_metadata(real_geotiff_single_band)

        assert isinstance(meta, RasterMetadata)
        assert meta.count == 1
        assert meta.dtype == "uint8"
        assert meta.width == 128
        assert meta.height == 128

    def test_load_uint16_metadata(self, real_geotiff_rgb_uint16):
        """Load metadata from a uint16 GeoTIFF."""
        meta = load_raster_metadata(real_geotiff_rgb_uint16)

        assert isinstance(meta, RasterMetadata)
        assert meta.count == 3
        assert meta.dtype == "uint16"
        assert meta.width == 128
        assert meta.height == 128

    def test_read_real_tile_shape(self, real_geotiff_rgb_uint8):
        """Windowed read from real GeoTIFF returns correct shape."""
        # Read a 64x64 tile from the top-left corner
        # window = (col_off, row_off, width, height)
        window = (0, 0, 64, 64)
        tile, nodata_mask = read_tile(real_geotiff_rgb_uint8, window, bands=[1, 2, 3])

        assert tile.shape == (3, 64, 64)
        assert tile.dtype == np.uint8

    def test_read_real_tile_nonzero(self, real_geotiff_rgb_uint8):
        """Data values from real imagery are non-zero."""
        window = (0, 0, 64, 64)
        tile, nodata_mask = read_tile(real_geotiff_rgb_uint8, window, bands=[1, 2, 3])

        # Real imagery should have some non-zero pixels
        assert np.any(tile > 0)
        # Should not be all the same value (synthetic pavement has texture)
        assert tile.std() > 0

    def test_read_single_band_tile(self, real_geotiff_single_band):
        """Read tile from single-band GeoTIFF."""
        window = (0, 0, 64, 64)
        tile, nodata_mask = read_tile(real_geotiff_single_band, window, bands=[1])

        assert tile.shape == (1, 64, 64)
        assert tile.dtype == np.uint8
        assert np.any(tile > 0)

    def test_read_uint16_tile(self, real_geotiff_rgb_uint16):
        """Read tile from uint16 GeoTIFF."""
        window = (0, 0, 64, 64)
        tile, nodata_mask = read_tile(real_geotiff_rgb_uint16, window, bands=[1, 2, 3])

        assert tile.shape == (3, 64, 64)
        assert tile.dtype == np.uint16
        assert np.any(tile > 0)
        # uint16 values should be in a reasonable range (0-10000 for reflectance)
        assert tile.max() < 65535  # Not at saturation

    def test_normalization_real_uint8(self, real_geotiff_rgb_uint8):
        """Percentile stretch on real uint8 data produces valid [0,1] output."""
        window = (0, 0, 128, 128)
        tile, nodata_mask = read_tile(real_geotiff_rgb_uint8, window, bands=[1, 2, 3])

        # Normalize with percentile stretch
        normalized, params = normalize_to_float32(tile, stretch="percentile")

        assert normalized.dtype == np.float32
        assert normalized.shape == (3, 128, 128)
        assert normalized.min() >= 0.0
        assert normalized.max() <= 1.0
        # Should use full dynamic range after stretch
        assert normalized.max() > 0.9  # Some bright pixels
        assert normalized.min() < 0.1  # Some dark pixels

    def test_normalization_real_uint16(self, real_geotiff_rgb_uint16):
        """Percentile stretch on real uint16 data produces valid [0,1] output."""
        window = (0, 0, 128, 128)
        tile, nodata_mask = read_tile(real_geotiff_rgb_uint16, window, bands=[1, 2, 3])

        # Normalize with percentile stretch
        normalized, params = normalize_to_float32(tile, stretch="percentile")

        assert normalized.dtype == np.float32
        assert normalized.shape == (3, 128, 128)
        assert normalized.min() >= 0.0
        assert normalized.max() <= 1.0

    def test_scene_stretch_params_real(self, real_geotiff_rgb_uint8):
        """Scene stretch parameter sampling works on real fixture."""
        # Compute stretch params by sampling the entire 128x128 image
        params = compute_scene_stretch_params(
            str(real_geotiff_rgb_uint8),
            bands=[1, 2, 3],
            percentiles=(2.0, 98.0),
            sample_tiles=1,  # Only one tile needed for 128x128 fixture
            tile_size=128,
        )

        # params should have vmin/vmax arrays
        assert hasattr(params, "vmin")
        assert hasattr(params, "vmax")
        assert len(params.vmin) == 3
        assert len(params.vmax) == 3
        # vmin/vmax should be valid percentiles
        for vmin, vmax in zip(params.vmin, params.vmax, strict=False):
            assert isinstance(vmin, (int, float))
            assert isinstance(vmax, (int, float))
            assert 0 <= vmin < vmax <= 255  # uint8 range
            # Should have reasonable contrast
            assert vmax - vmin > 10  # At least some dynamic range

    def test_band_selector_rgb_preset(self):
        """BandSelector RGB preset works."""
        selector = BandSelector("rgb")
        assert selector.band_indices == [1, 2, 3]

    def test_band_selector_custom_bands(self):
        """BandSelector with custom band tuple."""
        selector = BandSelector((4, 3, 2))
        assert selector.band_indices == [4, 3, 2]

    def test_band_selector_single_band_triplicate(self):
        """BandSelector triplicates single-band data."""
        import warnings

        selector = BandSelector((1,))

        # Create single-band data
        data = np.random.randint(0, 255, (1, 64, 64), dtype=np.uint8)

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

    def test_nodata_detection_real(self, real_geotiff_with_nodata):
        """Nodata regions are correctly identified in real fixture."""
        with rasterio.open(real_geotiff_with_nodata) as src:
            assert src.nodata == 0

            # Read the top-right corner where nodata exists
            # window = (col_off, row_off, width, height)
            window = (96, 0, 32, 32)
            data, nodata_mask = read_tile(real_geotiff_with_nodata, window, bands=[1, 2, 3])

            # This region should be all nodata (value 0)
            assert np.all(data == 0)
            # nodata_mask should indicate all pixels are nodata
            if nodata_mask is not None:
                assert np.all(nodata_mask)

            # Read the center where valid data exists
            window_center = (32, 32, 32, 32)
            data_center, nodata_mask_center = read_tile(real_geotiff_with_nodata, window_center, bands=[1, 2, 3])

            # This region should have some non-zero data
            assert np.any(data_center > 0)
