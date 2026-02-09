"""Integration tests for detr_geo.tiling with real GeoTIFF fixtures.

Tests tiling functions using small real GeoTIFF files.
Exercises:
- Tile grid generation with real rasters
- Nodata fraction computation with real nodata regions
- Tile boundary handling
"""

import numpy as np
import pytest

from detr_geo.io import compute_nodata_fraction, load_raster_metadata, read_tile
from detr_geo.tiling import generate_tile_grid


@pytest.mark.integration
class TestRealDataTiling:
    """Test tiling.py functions with real GeoTIFF fixtures."""

    def test_tile_grid_on_128px(self, real_geotiff_rgb_uint8):
        """Generate tile grid from a 128x128 real GeoTIFF."""
        meta = load_raster_metadata(real_geotiff_rgb_uint8)

        # Generate tiles with tile_size=64, no overlap
        tiles = generate_tile_grid(
            raster_width=meta.width,
            raster_height=meta.height,
            tile_size=64,
            overlap_ratio=0.0,
        )

        # 128x128 with 64x64 tiles = 2x2 grid = 4 tiles
        assert len(tiles) == 4

        # Check first tile
        assert tiles[0]["window"] == (0, 0, 64, 64)
        assert tiles[0]["global_offset_x"] == 0
        assert tiles[0]["global_offset_y"] == 0

        # Check last tile
        assert tiles[3]["window"] == (64, 64, 64, 64)
        assert tiles[3]["global_offset_x"] == 64
        assert tiles[3]["global_offset_y"] == 64

    def test_tile_grid_with_overlap(self, real_geotiff_rgb_uint8):
        """Generate overlapping tile grid from real GeoTIFF."""
        meta = load_raster_metadata(real_geotiff_rgb_uint8)

        # Generate tiles with tile_size=64, overlap=16 (25%)
        tiles = generate_tile_grid(
            raster_width=meta.width,
            raster_height=meta.height,
            tile_size=64,
            overlap_ratio=0.25,  # 16/64 = 0.25
        )

        # With overlap, we get more tiles covering the same area
        # 128px / stride(48) = ~2.67, so ceil(2.67) = 3 tiles per dimension = 9 tiles
        assert len(tiles) == 9

        # First tile should start at (0, 0)
        assert tiles[0]["window"] == (0, 0, 64, 64)

        # Second tile should be offset by stride (tile_size - overlap)
        # stride = 64 - 16 = 48
        assert tiles[1]["window"] == (48, 0, 64, 64)

    def test_tile_grid_exact_division(self, real_geotiff_rgb_uint8_utm):
        """Tile grid with exact division of image dimensions."""
        meta = load_raster_metadata(real_geotiff_rgb_uint8_utm)

        # tile_size=32 divides 128 evenly
        tiles = generate_tile_grid(
            raster_width=meta.width,
            raster_height=meta.height,
            tile_size=32,
            overlap_ratio=0.0,
        )

        # 128/32 = 4, so 4x4 = 16 tiles
        assert len(tiles) == 16

        # All tiles should be 32x32
        for tile in tiles:
            _, _, w, h = tile["window"]
            assert w == 32
            assert h == 32

    def test_tile_grid_single_tile(self, real_geotiff_rgb_uint8):
        """Tile size larger than image produces single tile."""
        meta = load_raster_metadata(real_geotiff_rgb_uint8)

        # tile_size=256 > 128, should produce single tile
        tiles = generate_tile_grid(
            raster_width=meta.width,
            raster_height=meta.height,
            tile_size=256,
            overlap_ratio=0.0,
        )

        # Should only produce one tile covering the entire image
        assert len(tiles) == 1
        assert tiles[0]["window"] == (0, 0, 128, 128)

    def test_nodata_fraction_full_data(self, real_geotiff_rgb_uint8):
        """Full data fixture has nodata_fraction=0.0."""
        # Read a tile from the fixture (no nodata in this one)
        window = (0, 0, 64, 64)
        tile, nodata_mask = read_tile(real_geotiff_rgb_uint8, window, bands=[1, 2, 3])

        # This fixture has no nodata value defined
        assert tile.shape == (3, 64, 64)
        # nodata_mask should be None
        assert nodata_mask is None

    def test_nodata_fraction_with_nodata(self, real_geotiff_with_nodata):
        """Nodata fixture has regions with nodata."""
        # Read the top-right corner where nodata exists
        window = (96, 0, 32, 32)
        tile, nodata_mask = read_tile(real_geotiff_with_nodata, window, bands=[1, 2, 3])

        # This region is all nodata (value 0)
        assert nodata_mask is not None
        nodata_fraction = compute_nodata_fraction(tile, nodata_value=0, alpha_mask=nodata_mask)
        # Should be 100% nodata in this tile
        assert nodata_fraction == 1.0
        # Should be skipped with threshold=0.8
        assert nodata_fraction > 0.8

        # Read the center where valid data exists
        window_center = (32, 32, 32, 32)
        tile_center, nodata_mask_center = read_tile(real_geotiff_with_nodata, window_center, bands=[1, 2, 3])

        # Should have much less nodata
        if nodata_mask_center is not None:
            nodata_fraction_center = compute_nodata_fraction(tile_center, nodata_value=0, alpha_mask=nodata_mask_center)
            assert nodata_fraction_center < 0.5

    def test_tile_coordinates_are_consistent(self, real_geotiff_rgb_uint8):
        """Tile window coordinates match global offsets."""
        meta = load_raster_metadata(real_geotiff_rgb_uint8)

        tiles = generate_tile_grid(
            raster_width=meta.width,
            raster_height=meta.height,
            tile_size=64,
            overlap_ratio=0.0,
        )

        for tile in tiles:
            col_off, row_off, w, h = tile["window"]
            assert tile["global_offset_x"] == col_off
            assert tile["global_offset_y"] == row_off

    def test_tile_coverage_complete(self, real_geotiff_rgb_uint8):
        """All tiles together cover the entire image without gaps."""
        meta = load_raster_metadata(real_geotiff_rgb_uint8)

        tiles = generate_tile_grid(
            raster_width=meta.width,
            raster_height=meta.height,
            tile_size=64,
            overlap_ratio=0.0,
        )

        # Track which pixels are covered by at least one tile
        coverage = np.zeros((meta.height, meta.width), dtype=bool)

        for tile in tiles:
            col_off, row_off, w, h = tile["window"]
            coverage[row_off : row_off + h, col_off : col_off + w] = True

        # All pixels should be covered
        assert coverage.all()
