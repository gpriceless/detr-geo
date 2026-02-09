"""Tests for tile grid generation and helper functions."""

from __future__ import annotations

import numpy as np
import pytest

from detr_geo.exceptions import TilingError
from detr_geo.tiling import (
    detection_range,
    generate_tile_grid,
    recommended_overlap,
)

# ---------------------------------------------------------------------------
# Tests: generate_tile_grid
# ---------------------------------------------------------------------------


class TestGenerateTileGrid:
    """Test tile grid generation."""

    def test_non_overlapping_evenly_divisible(self):
        """Non-overlapping grid on evenly divisible raster -> exact 2x2."""
        tiles = generate_tile_grid(100, 100, 50, overlap_ratio=0.0)
        assert len(tiles) == 4  # 2x2 grid

    def test_non_overlapping_coverage(self):
        """Non-overlapping grid covers every pixel."""
        width, height, tile_size = 100, 100, 50
        tiles = generate_tile_grid(width, height, tile_size, overlap_ratio=0.0)
        coverage = np.zeros((height, width), dtype=bool)
        for tile in tiles:
            col, row, w, h = tile["window"]
            coverage[row : row + h, col : col + w] = True
        assert coverage.all(), "Not all pixels covered"

    def test_overlapping_produces_more_tiles(self):
        """Overlapping grid produces denser coverage than non-overlapping."""
        tiles_no_overlap = generate_tile_grid(100, 100, 50, overlap_ratio=0.0)
        tiles_overlap = generate_tile_grid(100, 100, 50, overlap_ratio=0.2)
        assert len(tiles_overlap) > len(tiles_no_overlap)

    def test_overlapping_coverage(self):
        """Overlapping grid still covers every pixel."""
        width, height, tile_size = 100, 100, 50
        tiles = generate_tile_grid(width, height, tile_size, overlap_ratio=0.2)
        coverage = np.zeros((height, width), dtype=bool)
        for tile in tiles:
            col, row, w, h = tile["window"]
            coverage[row : row + h, col : col + w] = True
        assert coverage.all(), "Not all pixels covered"

    def test_full_coverage_arbitrary_dimensions(self):
        """Full coverage verification on an odd-sized raster."""
        width, height, tile_size = 237, 189, 64
        tiles = generate_tile_grid(width, height, tile_size, overlap_ratio=0.15)
        coverage = np.zeros((height, width), dtype=bool)
        for tile in tiles:
            col, row, w, h = tile["window"]
            coverage[row : row + h, col : col + w] = True
        assert coverage.all(), "Not all pixels covered on arbitrary dimensions"

    def test_tile_larger_than_raster(self):
        """Tile larger than raster -> single tile."""
        tiles = generate_tile_grid(200, 200, 576, overlap_ratio=0.2)
        assert len(tiles) == 1
        col, row, w, h = tiles[0]["window"]
        assert col == 0
        assert row == 0
        assert w == 200
        assert h == 200

    def test_large_raster_full_coverage(self):
        """Full coverage on a 1000x1000 raster with standard settings."""
        width, height, tile_size = 1000, 1000, 576
        tiles = generate_tile_grid(width, height, tile_size, overlap_ratio=0.2)
        coverage = np.zeros((height, width), dtype=bool)
        for tile in tiles:
            col, row, w, h = tile["window"]
            coverage[row : row + h, col : col + w] = True
        assert coverage.all(), "Not all pixels covered on 1000x1000 raster"
        assert len(tiles) > 1


class TestTileMetadata:
    """Test tile metadata completeness."""

    def test_tile_has_required_fields(self):
        """Each TileInfo has all required metadata."""
        tiles = generate_tile_grid(256, 256, 128, overlap_ratio=0.2)
        for tile in tiles:
            assert "window" in tile
            assert "global_offset_x" in tile
            assert "global_offset_y" in tile
            assert "nodata_fraction" in tile
            col, row, w, h = tile["window"]
            assert tile["global_offset_x"] == col
            assert tile["global_offset_y"] == row

    def test_edge_tiles_clipped(self):
        """Edge tiles have dimensions <= tile_size."""
        tiles = generate_tile_grid(100, 100, 60, overlap_ratio=0.0)
        for tile in tiles:
            _, _, w, h = tile["window"]
            assert w <= 60
            assert h <= 60
            assert w > 0
            assert h > 0


# ---------------------------------------------------------------------------
# Tests: Overlap ratio constraints
# ---------------------------------------------------------------------------


class TestOverlapConstraints:
    """Test overlap ratio validation."""

    def test_overlap_at_0_5_raises(self):
        """overlap_ratio=0.5 raises TilingError."""
        with pytest.raises(TilingError):
            generate_tile_grid(100, 100, 50, overlap_ratio=0.5)

    def test_overlap_above_0_5_raises(self):
        """overlap_ratio > 0.5 raises TilingError."""
        with pytest.raises(TilingError):
            generate_tile_grid(100, 100, 50, overlap_ratio=0.7)

    def test_negative_overlap_raises(self):
        """Negative overlap_ratio raises TilingError."""
        with pytest.raises(TilingError):
            generate_tile_grid(100, 100, 50, overlap_ratio=-0.1)

    def test_overlap_0_49_succeeds(self):
        """overlap_ratio=0.49 is valid (just under limit)."""
        tiles = generate_tile_grid(100, 100, 50, overlap_ratio=0.49)
        assert len(tiles) > 0

    def test_tile_size_zero_raises(self):
        """tile_size=0 raises TilingError."""
        with pytest.raises(TilingError):
            generate_tile_grid(100, 100, 0)

    def test_tile_size_negative_raises(self):
        """Negative tile_size raises TilingError."""
        with pytest.raises(TilingError):
            generate_tile_grid(100, 100, -10)

    def test_stride_computation(self):
        """Stride is computed correctly from overlap_ratio."""
        # overlap_ratio=0.2, tile_size=100 -> stride = 100 * 0.8 = 80
        tiles = generate_tile_grid(200, 200, 100, overlap_ratio=0.2)
        # First two tiles in same row should have stride 80
        assert tiles[0]["global_offset_x"] == 0
        assert tiles[1]["global_offset_x"] == 80


# ---------------------------------------------------------------------------
# Tests: recommended_overlap
# ---------------------------------------------------------------------------


class TestRecommendedOverlap:
    """Test overlap recommendation."""

    def test_returns_reasonable_range(self):
        """Recommended overlap is in valid range."""
        result = recommended_overlap(30, tile_size=576)
        assert 0.05 <= result <= 0.49

    def test_larger_objects_need_more_overlap(self):
        """Larger objects -> larger overlap."""
        small = recommended_overlap(10, tile_size=576)
        large = recommended_overlap(100, tile_size=576)
        assert large > small

    def test_capped_at_0_49(self):
        """Very large objects cap overlap at 0.49."""
        result = recommended_overlap(1000, tile_size=100)
        assert result == 0.49

    def test_minimum_overlap_0_05(self):
        """Very small objects still get at least 0.05 overlap."""
        result = recommended_overlap(1, tile_size=1000)
        assert result >= 0.05

    def test_zero_tile_size_returns_default(self):
        """tile_size=0 returns default 0.2."""
        result = recommended_overlap(30, tile_size=0)
        assert result == 0.2


# ---------------------------------------------------------------------------
# Tests: detection_range
# ---------------------------------------------------------------------------


class TestDetectionRange:
    """Test detection envelope calculator."""

    def test_returns_tuple_of_two_floats(self):
        result = detection_range(576, 0.3)
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], float)
        assert isinstance(result[1], float)

    def test_min_detectable_is_10_pixels(self):
        """Minimum detectable object = 10 pixels * GSD."""
        min_m, _ = detection_range(576, 0.3)
        assert min_m == pytest.approx(3.0)  # 10 * 0.3

    def test_max_detectable_is_80_percent_tile(self):
        """Maximum detectable object = 80% of tile * GSD."""
        _, max_m = detection_range(576, 0.3)
        assert max_m == pytest.approx(576 * 0.8 * 0.3)

    def test_min_less_than_max(self):
        """min_meters < max_meters always."""
        min_m, max_m = detection_range(576, 0.3)
        assert min_m < max_m
