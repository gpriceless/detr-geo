"""Tests for raster loading and windowed reads."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_bounds

from detr_geo.io import (
    compute_nodata_fraction,
    fill_nodata,
    load_raster_metadata,
    read_tile,
    resolve_raster_source,
    stac_item_to_uri,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def create_test_geotiff(
    path: str,
    width: int = 256,
    height: int = 256,
    bands: int = 3,
    dtype: str = "uint8",
    crs: str | None = "EPSG:4326",
    nodata: float | None = None,
) -> str:
    """Create a synthetic GeoTIFF for testing."""
    transform = from_bounds(0, 0, 1, 1, width, height)

    if dtype == "uint8":
        data = np.random.randint(0, 255, (bands, height, width), dtype=np.uint8)
    elif dtype == "uint16":
        data = np.random.randint(0, 10000, (bands, height, width), dtype=np.uint16)
    else:
        data = np.random.rand(bands, height, width).astype(dtype)

    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=width,
        height=height,
        count=bands,
        dtype=dtype,
        crs=crs,
        transform=transform,
        nodata=nodata,
    ) as dst:
        dst.write(data)

    return path


@pytest.fixture
def tmp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as d:
        yield d


# ---------------------------------------------------------------------------
# Tests: load_raster_metadata
# ---------------------------------------------------------------------------


class TestLoadRasterMetadata:
    """Test raster metadata loading."""

    def test_reads_crs_transform_dimensions(self, tmp_dir):
        path = create_test_geotiff(
            str(Path(tmp_dir) / "test.tif"),
            width=512,
            height=512,
            bands=3,
            crs="EPSG:4326",
        )
        meta = load_raster_metadata(path)
        assert meta.crs is not None
        assert meta.width == 512
        assert meta.height == 512
        assert meta.count == 3
        assert meta.dtype == "uint8"

    def test_crs_less_raster_returns_none_crs(self, tmp_dir):
        path = create_test_geotiff(
            str(Path(tmp_dir) / "no_crs.tif"),
            crs=None,
        )
        meta = load_raster_metadata(path)
        assert meta.crs is None

    def test_file_not_found_raises(self):
        with pytest.raises(FileNotFoundError):
            load_raster_metadata("/nonexistent/raster.tif")

    def test_nodata_value_preserved(self, tmp_dir):
        path = create_test_geotiff(
            str(Path(tmp_dir) / "nodata.tif"),
            nodata=0.0,
        )
        meta = load_raster_metadata(path)
        assert meta.nodata == 0.0


# ---------------------------------------------------------------------------
# Tests: read_tile
# ---------------------------------------------------------------------------


class TestReadTile:
    """Test windowed tile reading."""

    def test_window_returns_correct_subset(self, tmp_dir):
        path = create_test_geotiff(
            str(Path(tmp_dir) / "test.tif"),
            width=256,
            height=256,
            bands=3,
        )
        data, mask = read_tile(path, window=(0, 0, 64, 64))
        assert data.shape == (3, 64, 64)

    def test_specific_bands(self, tmp_dir):
        path = create_test_geotiff(
            str(Path(tmp_dir) / "test.tif"),
            width=256,
            height=256,
            bands=4,
        )
        data, mask = read_tile(path, window=(0, 0, 64, 64), bands=[1, 2, 3])
        assert data.shape == (3, 64, 64)

    def test_boundless_window_pads(self, tmp_dir):
        path = create_test_geotiff(
            str(Path(tmp_dir) / "test.tif"),
            width=64,
            height=64,
            bands=3,
        )
        # Read beyond raster boundary
        data, mask = read_tile(path, window=(32, 32, 64, 64))
        assert data.shape == (3, 64, 64)


# ---------------------------------------------------------------------------
# Tests: compute_nodata_fraction
# ---------------------------------------------------------------------------


class TestComputeNodataFraction:
    """Test nodata fraction computation."""

    def test_fully_valid_tile_returns_0(self):
        data = np.random.randint(1, 255, (3, 64, 64), dtype=np.uint8)
        fraction = compute_nodata_fraction(data, nodata_value=0)
        assert fraction == 0.0

    def test_fully_nodata_tile_returns_1(self):
        data = np.zeros((3, 64, 64), dtype=np.uint8)
        fraction = compute_nodata_fraction(data, nodata_value=0)
        assert fraction == 1.0

    def test_partial_nodata_correct_fraction(self):
        data = np.ones((3, 64, 64), dtype=np.uint8)
        # Set first 32 rows to nodata
        data[:, :32, :] = 0
        fraction = compute_nodata_fraction(data, nodata_value=0)
        assert fraction == pytest.approx(0.5)

    def test_alpha_mask_priority(self):
        data = np.ones((3, 64, 64), dtype=np.uint8)
        alpha_mask = np.zeros((64, 64), dtype=bool)
        alpha_mask[:16, :] = True  # 25% nodata
        fraction = compute_nodata_fraction(data, nodata_value=None, alpha_mask=alpha_mask)
        assert fraction == pytest.approx(0.25)

    def test_no_nodata_info_returns_0(self):
        data = np.ones((3, 64, 64), dtype=np.uint8)
        fraction = compute_nodata_fraction(data, nodata_value=None)
        assert fraction == 0.0


# ---------------------------------------------------------------------------
# Tests: fill_nodata
# ---------------------------------------------------------------------------


class TestFillNodata:
    """Test nodata filling."""

    def test_fills_with_per_band_mean(self):
        data = np.ones((3, 4, 4), dtype=np.float32)
        data[0] *= 10
        data[1] *= 20
        data[2] *= 30
        nodata_mask = np.zeros((4, 4), dtype=bool)
        nodata_mask[0, 0] = True  # One nodata pixel
        data[:, 0, 0] = 0  # Set nodata pixel to 0

        result = fill_nodata(data, nodata_mask)

        # Band 0: mean of valid pixels (10) - only one pixel is 0, rest are 10
        # Valid mean should be close to 10 (15 pixels of 10, 0 pixels excluded)
        assert result[0, 0, 0] == pytest.approx(10.0, abs=1.0)
        assert result[1, 0, 0] == pytest.approx(20.0, abs=1.0)
        assert result[2, 0, 0] == pytest.approx(30.0, abs=1.0)

    def test_all_nodata_fills_with_zeros(self):
        data = np.ones((3, 4, 4), dtype=np.float32) * 42
        nodata_mask = np.ones((4, 4), dtype=bool)  # All nodata

        result = fill_nodata(data, nodata_mask)
        assert result[0, 0, 0] == 0.0

    def test_no_nodata_returns_copy(self):
        data = np.ones((3, 4, 4), dtype=np.float32)
        nodata_mask = np.zeros((4, 4), dtype=bool)

        result = fill_nodata(data, nodata_mask)
        np.testing.assert_array_equal(result, data)
        # Should be a copy, not the same object
        assert result is not data


# ---------------------------------------------------------------------------
# Tests: compute_gsd
# ---------------------------------------------------------------------------


class TestComputeGSD:
    """Test GSD computation from transform and CRS."""

    def test_projected_crs_meters(self):
        """UTM CRS with 1m pixel size returns 1.0."""
        from pyproj import CRS
        from rasterio.transform import Affine

        from detr_geo.io import compute_gsd

        # UTM zone 10N (meters)
        crs = CRS.from_epsg(32610)
        transform = Affine(1.0, 0.0, 500000.0, 0.0, -1.0, 4000000.0)

        gsd = compute_gsd(transform, crs)
        assert gsd == pytest.approx(1.0)

    def test_projected_crs_non_square(self):
        """Non-square pixels use average of x and y."""
        from pyproj import CRS
        from rasterio.transform import Affine

        from detr_geo.io import compute_gsd

        crs = CRS.from_epsg(32610)
        transform = Affine(2.0, 0.0, 500000.0, 0.0, -3.0, 4000000.0)

        gsd = compute_gsd(transform, crs)
        assert gsd == pytest.approx(2.5)  # (2.0 + 3.0) / 2

    def test_geographic_crs_degrees(self):
        """WGS84 CRS with degrees converts to approximate meters."""
        from pyproj import CRS
        from rasterio.transform import Affine

        from detr_geo.io import compute_gsd

        crs = CRS.from_epsg(4326)
        # ~0.0000027 degrees at equator is roughly 0.3m
        transform = Affine(0.0000027, 0.0, -122.0, 0.0, -0.0000027, 37.0)

        gsd = compute_gsd(transform, crs)
        # At mid-latitudes, ~0.0000027 degrees ≈ 0.3m
        assert gsd is not None
        assert gsd == pytest.approx(0.3, abs=0.15)  # rough approximation

    def test_no_crs_returns_none(self):
        """Missing CRS returns None."""
        from rasterio.transform import Affine

        from detr_geo.io import compute_gsd

        transform = Affine(1.0, 0.0, 0.0, 0.0, -1.0, 0.0)
        gsd = compute_gsd(transform, None)
        assert gsd is None


# ---------------------------------------------------------------------------
# Tests: check_gsd
# ---------------------------------------------------------------------------


class TestCheckGSD:
    """Test GSD warning logic."""

    def test_optimal_range_no_warning(self):
        """GSD=0.3m is optimal, no warning."""
        import warnings

        from detr_geo.io import check_gsd

        with warnings.catch_warnings():
            warnings.simplefilter("error")  # Any warning becomes an error
            check_gsd(0.3)  # Should not raise

    def test_high_resolution_warning(self):
        """GSD=0.03m issues high resolution warning."""

        from detr_geo.exceptions import GSDWarning
        from detr_geo.io import check_gsd

        with pytest.warns(GSDWarning, match="extremely high resolution"):
            check_gsd(0.03)

    def test_low_resolution_warning(self):
        """GSD=2.0m issues low resolution warning."""

        from detr_geo.exceptions import GSDWarning
        from detr_geo.io import check_gsd

        with pytest.warns(GSDWarning, match="well below training GSD"):
            check_gsd(2.0)

    def test_extreme_gsd_strict_mode(self):
        """GSD=10.0m with strict=True raises error."""
        from detr_geo.exceptions import DetrGeoError
        from detr_geo.io import check_gsd

        with pytest.raises(DetrGeoError, match="Objects are likely undetectable"):
            check_gsd(10.0, strict=True)

    def test_none_gsd_no_warning(self):
        """GSD=None skips check."""
        import warnings

        from detr_geo.io import check_gsd

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            check_gsd(None)  # Should not raise

    def test_slightly_degraded_warning(self):
        """GSD=0.7m issues degraded resolution warning."""
        from detr_geo.exceptions import GSDWarning
        from detr_geo.io import check_gsd

        with pytest.warns(GSDWarning, match="degraded compared to training"):
            check_gsd(0.7)


# ---------------------------------------------------------------------------
# Cloud-native raster source resolution tests
# ---------------------------------------------------------------------------


class TestResolveRasterSource:
    """Test resolve_raster_source function."""

    def test_local_path_string(self, tmp_path):
        """Local path string resolves correctly."""
        test_file = tmp_path / "test.tif"
        test_file.touch()

        result = resolve_raster_source(str(test_file))
        assert result == str(test_file)

    def test_local_path_object(self, tmp_path):
        """Local Path object resolves correctly."""
        test_file = tmp_path / "test.tif"
        test_file.touch()

        result = resolve_raster_source(test_file)
        assert result == str(test_file)

    def test_http_url(self):
        """HTTP URL passes through unchanged."""
        url = "https://example.com/data/cog.tif"
        result = resolve_raster_source(url)
        assert result == url

    def test_https_url(self):
        """HTTPS URL passes through unchanged."""
        url = "https://example.com/secure/cog.tif"
        result = resolve_raster_source(url)
        assert result == url

    def test_s3_uri(self):
        """S3 URI passes through unchanged."""
        uri = "s3://my-bucket/path/to/data.tif"
        result = resolve_raster_source(uri)
        assert uri == result

    def test_nonexistent_local_path(self, tmp_path):
        """Nonexistent local path raises FileNotFoundError."""
        nonexistent = tmp_path / "missing.tif"

        with pytest.raises(FileNotFoundError, match="not found"):
            resolve_raster_source(nonexistent)

    def test_stac_item_mock(self):
        """Mock STAC Item with GeoTIFF asset extracts URL."""
        pystac = pytest.importorskip("pystac")  # noqa: F841

        # Create mock STAC item
        class MockAsset:
            def __init__(self, href, media_type=None):
                self.href = href
                self.media_type = media_type

        class MockItem:
            def __init__(self):
                self.id = "test-item"
                self.assets = {
                    "visual": MockAsset("https://example.com/visual.tif", "image/tiff"),
                    "metadata": MockAsset("https://example.com/meta.json", "application/json"),
                }

        item = MockItem()
        result = resolve_raster_source(item)
        assert result == "https://example.com/visual.tif"

    def test_stac_item_no_suitable_asset(self):
        """STAC Item with no GeoTIFF asset raises error."""
        pystac = pytest.importorskip("pystac")  # noqa: F841

        class MockAsset:
            def __init__(self, href, media_type=None):
                self.href = href
                self.media_type = media_type

        class MockItem:
            def __init__(self):
                self.id = "test-item"
                self.assets = {
                    "metadata": MockAsset("https://example.com/meta.json", "application/json"),
                }

        item = MockItem()
        from detr_geo.exceptions import DetrGeoError

        with pytest.raises(DetrGeoError, match="no GeoTIFF asset"):
            resolve_raster_source(item)


class TestStacItemToUri:
    """Test STAC item to URI extraction."""

    def test_preferred_visual_asset(self):
        """Prefer 'visual' asset if present."""

        class MockAsset:
            def __init__(self, href, media_type=None):
                self.href = href
                self.media_type = media_type

        class MockItem:
            def __init__(self):
                self.id = "test"
                self.assets = {
                    "red": MockAsset("https://example.com/red.tif", "image/tiff"),
                    "visual": MockAsset("https://example.com/visual.tif", "image/tiff"),
                }

        result = stac_item_to_uri(MockItem())
        assert result == "https://example.com/visual.tif"

    def test_fallback_to_tiff_extension(self):
        """Fallback to .tif extension if no media_type."""

        class MockAsset:
            def __init__(self, href, media_type=None):
                self.href = href
                self.media_type = media_type

        class MockItem:
            def __init__(self):
                self.id = "test"
                self.assets = {
                    "data": MockAsset("https://example.com/data.tiff", None),
                }

        result = stac_item_to_uri(MockItem())
        assert result == "https://example.com/data.tiff"


class TestLoadRasterMetadataWithRemote:
    """Test load_raster_metadata with remote sources."""

    def test_load_metadata_with_url_requires_network(self):
        """Loading metadata from URL requires network (skip in unit tests)."""
        # This would need a real COG URL or mocking rasterio.open
        # For now, just verify the resolve_raster_source part works
        url = "https://example.com/cog.tif"

        # The URL resolves correctly
        from detr_geo.io import resolve_raster_source

        assert resolve_raster_source(url) == url

        # Actually opening would require network, skip in unit tests
        pytest.skip("Network tests require integration test setup")

    def test_load_metadata_local_path_still_works(self, tmp_path):
        """Ensure local path loading still works (regression test)."""
        test_file = create_test_geotiff(
            str(tmp_path / "test.tif"),
            width=64,
            height=64,
            crs="EPSG:4326",
        )

        meta = load_raster_metadata(test_file)
        assert meta.width == 64
        assert meta.height == 64
        assert meta.crs.to_epsg() == 4326
