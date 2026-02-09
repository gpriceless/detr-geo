"""Tests for geo_to_pixel() inverse affine transform (Proposal 006, Task 006.0).

Tests the round-trip pixel->geo->pixel and handles rotated affine transforms.
"""

from __future__ import annotations

import math

from rasterio.transform import Affine, from_bounds

from detr_geo.crs import geo_to_pixel, pixel_to_geo


class TestGeoToPixel:
    """Tests for geo_to_pixel inverse affine transform."""

    def test_round_trip_standard_transform(self):
        """GIVEN a standard north-up affine transform,
        WHEN pixel_to_geo then geo_to_pixel,
        THEN result within 0.01px of original."""
        transform = from_bounds(500000, 4000000, 500256, 4000256, 256, 256)

        # pixel_to_geo expects (x_min, y_min, x_max, y_max) bbox
        geo_poly = pixel_to_geo((128, 128, 129, 129), transform)
        # Get the first vertex (top-left in geo space)
        geo_point = geo_poly.exterior.coords[0]

        px = geo_to_pixel(geo_point, transform)
        assert abs(px[0] - 128) < 0.01, f"Column mismatch: {px[0]} vs 128"
        assert abs(px[1] - 128) < 0.01, f"Row mismatch: {px[1]} vs 128"

    def test_round_trip_origin(self):
        """GIVEN transform origin, WHEN round-trip, THEN (0, 0)."""
        transform = from_bounds(500000, 4000000, 500256, 4000256, 256, 256)
        # Get geo coords at pixel (0, 0)
        geo_point = transform * (0, 0)
        px = geo_to_pixel(geo_point, transform)
        assert abs(px[0]) < 0.01
        assert abs(px[1]) < 0.01

    def test_round_trip_bottom_right(self):
        """GIVEN bottom-right pixel, WHEN round-trip, THEN correct."""
        transform = from_bounds(500000, 4000000, 500256, 4000256, 256, 256)
        geo_point = transform * (255, 255)
        px = geo_to_pixel(geo_point, transform)
        assert abs(px[0] - 255) < 0.01
        assert abs(px[1] - 255) < 0.01

    def test_round_trip_fractional(self):
        """GIVEN fractional pixel coords, WHEN round-trip, THEN within tolerance."""
        transform = from_bounds(0, 0, 100, 100, 100, 100)
        original = (33.7, 67.2)
        geo_point = transform * original
        px = geo_to_pixel(geo_point, transform)
        assert abs(px[0] - 33.7) < 0.01
        assert abs(px[1] - 67.2) < 0.01

    def test_rotated_affine_transform(self):
        """GIVEN a rotated affine (non-zero b/d terms),
        WHEN round-trip, THEN within 0.01px tolerance."""
        # Rotated affine: 45 degree rotation
        angle = math.radians(30)
        scale = 1.0
        a = scale * math.cos(angle)
        b = -scale * math.sin(angle)
        d = scale * math.sin(angle)
        e = -scale * math.cos(angle)
        transform = Affine(a, b, 500000, d, e, 4000256)

        original = (100, 100)
        geo_point = transform * original
        px = geo_to_pixel(geo_point, transform)
        assert abs(px[0] - 100) < 0.01, f"Col mismatch: {px[0]} vs 100"
        assert abs(px[1] - 100) < 0.01, f"Row mismatch: {px[1]} vs 100"

    def test_different_pixel_sizes(self):
        """GIVEN non-square pixel sizes, WHEN round-trip, THEN correct."""
        # 2m x 1m pixels
        transform = Affine(2.0, 0, 500000, 0, -1.0, 4000256)
        original = (50, 75)
        geo_point = transform * original
        px = geo_to_pixel(geo_point, transform)
        assert abs(px[0] - 50) < 0.01
        assert abs(px[1] - 75) < 0.01

    def test_returns_float_tuple(self):
        """THEN return value is a tuple of two floats."""
        transform = from_bounds(0, 0, 100, 100, 100, 100)
        result = geo_to_pixel((50, 50), transform)
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], float)
        assert isinstance(result[1], float)
