"""Tests for CRS management module."""

from __future__ import annotations

import pytest
from pyproj import CRS
from rasterio.transform import Affine
from shapely.geometry import box

from detr_geo.crs import (
    auto_utm_crs,
    get_transformer,
    has_rotation,
    pixel_to_geo,
    validate_crs,
)
from detr_geo.exceptions import CRSError, MissingCRSError


class TestPixelToGeo:
    """Test pixel-to-geographic coordinate conversion."""

    def test_identity_transform(self):
        """Identity transform should return pixel coords as-is."""
        transform = Affine.identity()
        bbox = (10.0, 20.0, 30.0, 40.0)
        poly = pixel_to_geo(bbox, transform)

        # With identity transform, geo coords == pixel coords
        coords = list(poly.exterior.coords)
        assert len(coords) == 5  # 4 corners + closing point
        assert coords[0] == pytest.approx((10.0, 20.0))
        assert coords[1] == pytest.approx((30.0, 20.0))
        assert coords[2] == pytest.approx((30.0, 40.0))
        assert coords[3] == pytest.approx((10.0, 40.0))

    def test_utm_north_up_transform(self):
        """Standard north-up UTM transform (0.3m GSD)."""
        # a=0.3, b=0, c=500000, d=0, e=-0.3, f=4000000
        transform = Affine(0.3, 0, 500000, 0, -0.3, 4000000)
        bbox = (0.0, 0.0, 100.0, 100.0)
        poly = pixel_to_geo(bbox, transform)

        coords = list(poly.exterior.coords)
        # Top-left: (0, 0) -> (500000, 4000000)
        assert coords[0] == pytest.approx((500000.0, 4000000.0))
        # Top-right: (100, 0) -> (500030, 4000000)
        assert coords[1] == pytest.approx((500030.0, 4000000.0))
        # Bottom-right: (100, 100) -> (500030, 3999970)
        assert coords[2] == pytest.approx((500030.0, 3999970.0))
        # Bottom-left: (0, 100) -> (500000, 3999970)
        assert coords[3] == pytest.approx((500000.0, 3999970.0))

    def test_rotated_affine_produces_non_axis_aligned_polygon(self):
        """Rotated affine should produce a non-rectangular result."""
        # Add rotation: b=0.1, d=-0.1
        transform = Affine(0.3, 0.1, 500000, -0.1, -0.3, 4000000)
        bbox = (0.0, 0.0, 100.0, 100.0)
        poly = pixel_to_geo(bbox, transform)

        # With rotation, the result should differ from a simple box
        simple_box = box(*pixel_to_geo(bbox, Affine(0.3, 0, 500000, 0, -0.3, 4000000)).bounds)
        # The rotated polygon should not be identical to the axis-aligned version
        assert not poly.equals(simple_box)

    def test_rotated_differs_from_naive_2corner(self):
        """Rotated result should differ from naive 2-corner box approach."""
        transform = Affine(0.3, 0.1, 500000, -0.1, -0.3, 4000000)
        bbox = (0.0, 0.0, 100.0, 100.0)
        poly = pixel_to_geo(bbox, transform)

        # Naive 2-corner approach: only transform TL and BR
        tl = transform * (bbox[0], bbox[1])
        br = transform * (bbox[2], bbox[3])
        naive_box = box(min(tl[0], br[0]), min(tl[1], br[1]), max(tl[0], br[0]), max(tl[1], br[1]))

        # The 4-corner polygon should not equal the naive box
        assert not poly.equals(naive_box)


class TestHasRotation:
    """Test rotation detection."""

    def test_standard_north_up_no_rotation(self):
        transform = Affine(0.3, 0, 500000, 0, -0.3, 4000000)
        assert has_rotation(transform) is False

    def test_identity_no_rotation(self):
        assert has_rotation(Affine.identity()) is False

    def test_rotated_aerial_has_rotation(self):
        transform = Affine(0.3, 0.1, 500000, -0.1, -0.3, 4000000)
        assert has_rotation(transform) is True


class TestGetTransformer:
    """Test cached transformer creation."""

    def test_returns_transformer(self):
        t = get_transformer("EPSG:4326", "EPSG:32610")
        assert t is not None

    def test_cached_returns_same_object(self):
        t1 = get_transformer("EPSG:4326", "EPSG:32610")
        t2 = get_transformer("EPSG:4326", "EPSG:32610")
        assert t1 is t2

    def test_invalid_crs_raises_crs_error(self):
        with pytest.raises(CRSError):
            get_transformer("INVALID", "EPSG:4326")


class TestValidateCrs:
    """Test CRS validation."""

    def test_epsg_string_returns_crs_object(self):
        result = validate_crs("EPSG:4326")
        assert isinstance(result, CRS)
        assert result == CRS.from_epsg(4326)

    def test_none_georeferenced_raises_missing_crs_error(self):
        with pytest.raises(MissingCRSError):
            validate_crs(None, georeferenced=True)

    def test_none_not_georeferenced_returns_none(self):
        result = validate_crs(None, georeferenced=False)
        assert result is None

    def test_invalid_string_raises_crs_error(self):
        with pytest.raises(CRSError):
            validate_crs("INVALID_CRS_STRING")

    def test_pyproj_crs_object_passes_through(self):
        crs = CRS.from_epsg(32617)
        result = validate_crs(crs)
        assert result is crs


class TestAutoUtmCrs:
    """Test automatic UTM zone detection."""

    def test_san_francisco_utm_zone_10n(self):
        crs = auto_utm_crs(-122.0, 37.0)
        assert crs == CRS.from_epsg(32610)

    def test_cape_town_southern_hemisphere(self):
        crs = auto_utm_crs(15.0, -34.0)
        # UTM zone: floor((15 + 180) / 6) + 1 = 33
        # Southern hemisphere: EPSG:32733
        assert crs == CRS.from_epsg(32733)

    def test_london_utm_zone_30n(self):
        crs = auto_utm_crs(-0.1, 51.5)
        assert crs == CRS.from_epsg(32630)
