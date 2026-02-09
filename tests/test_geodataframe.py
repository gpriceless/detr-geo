"""Tests for GeoDataFrame construction and area computation."""

from __future__ import annotations

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from pyproj import CRS
from rasterio.transform import Affine
from shapely.geometry import Polygon

from detr_geo.export import build_dataframe_pixel, build_geodataframe, compute_areas

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def make_detections(n: int = 3):
    """Create synthetic detection arrays for testing."""
    rng = np.random.RandomState(42)
    x1 = rng.uniform(0, 400, n).astype(np.float32)
    y1 = rng.uniform(0, 400, n).astype(np.float32)
    x2 = x1 + rng.uniform(20, 80, n).astype(np.float32)
    y2 = y1 + rng.uniform(20, 80, n).astype(np.float32)
    boxes = np.stack([x1, y1, x2, y2], axis=1)
    scores = rng.uniform(0.5, 1.0, n).astype(np.float32)
    class_ids = rng.randint(0, 3, n).astype(np.int32)
    return boxes, scores, class_ids


# ---------------------------------------------------------------------------
# Tests: build_geodataframe
# ---------------------------------------------------------------------------


class TestBuildGeoDataFrame:
    """Test GeoDataFrame construction from detection arrays."""

    def test_correct_row_count(self):
        """N detections produce N-row GeoDataFrame."""
        boxes, scores, class_ids = make_detections(5)
        transform = Affine(0.3, 0, 500000, 0, -0.3, 4000000)
        crs = CRS.from_epsg(32610)
        gdf = build_geodataframe(boxes, scores, class_ids, None, transform, crs)
        assert len(gdf) == 5

    def test_required_columns_present(self):
        """GeoDataFrame has all required columns."""
        boxes, scores, class_ids = make_detections(3)
        transform = Affine(0.3, 0, 500000, 0, -0.3, 4000000)
        crs = CRS.from_epsg(32610)
        gdf = build_geodataframe(boxes, scores, class_ids, None, transform, crs)
        required = {"geometry", "class_id", "class_name", "confidence", "centroid_x", "centroid_y"}
        assert required.issubset(set(gdf.columns))

    def test_crs_matches_input(self):
        """GeoDataFrame CRS matches the input CRS."""
        boxes, scores, class_ids = make_detections(2)
        transform = Affine(0.3, 0, 500000, 0, -0.3, 4000000)
        crs = CRS.from_epsg(32610)
        gdf = build_geodataframe(boxes, scores, class_ids, None, transform, crs)
        assert gdf.crs == crs

    def test_geometries_are_polygons(self):
        """All geometries are shapely Polygons."""
        boxes, scores, class_ids = make_detections(3)
        transform = Affine(0.3, 0, 500000, 0, -0.3, 4000000)
        crs = CRS.from_epsg(32610)
        gdf = build_geodataframe(boxes, scores, class_ids, None, transform, crs)
        for geom in gdf.geometry:
            assert isinstance(geom, Polygon)

    def test_class_names_mapping(self):
        """Class names are resolved from mapping."""
        boxes = np.array([[0, 0, 10, 10], [20, 20, 30, 30]], dtype=np.float32)
        scores = np.array([0.95, 0.85], dtype=np.float32)
        class_ids = np.array([1, 2], dtype=np.int32)
        class_names = {1: "building", 2: "vehicle"}
        transform = Affine(0.3, 0, 500000, 0, -0.3, 4000000)
        crs = CRS.from_epsg(32610)

        gdf = build_geodataframe(boxes, scores, class_ids, class_names, transform, crs)
        assert gdf["class_name"].iloc[0] == "building"
        assert gdf["class_name"].iloc[1] == "vehicle"

    def test_unknown_class_id_fallback(self):
        """Unknown class IDs get fallback name."""
        boxes = np.array([[0, 0, 10, 10]], dtype=np.float32)
        scores = np.array([0.9], dtype=np.float32)
        class_ids = np.array([99], dtype=np.int32)
        class_names = {1: "building"}
        transform = Affine.identity()
        crs = CRS.from_epsg(4326)

        gdf = build_geodataframe(boxes, scores, class_ids, class_names, transform, crs)
        assert "class_99" in gdf["class_name"].iloc[0]

    def test_none_class_names(self):
        """None class_names uses default naming."""
        boxes = np.array([[0, 0, 10, 10]], dtype=np.float32)
        scores = np.array([0.9], dtype=np.float32)
        class_ids = np.array([5], dtype=np.int32)
        transform = Affine.identity()
        crs = CRS.from_epsg(4326)

        gdf = build_geodataframe(boxes, scores, class_ids, None, transform, crs)
        assert "class_5" in gdf["class_name"].iloc[0]

    def test_centroids_within_polygon(self):
        """Centroid coordinates are within the polygon bounds."""
        boxes, scores, class_ids = make_detections(3)
        transform = Affine(0.3, 0, 500000, 0, -0.3, 4000000)
        crs = CRS.from_epsg(32610)
        gdf = build_geodataframe(boxes, scores, class_ids, None, transform, crs)

        for _, row in gdf.iterrows():
            bounds = row.geometry.bounds
            assert bounds[0] <= row["centroid_x"] <= bounds[2]
            assert bounds[1] <= row["centroid_y"] <= bounds[3]

    def test_empty_detections(self):
        """Zero detections produce empty GeoDataFrame with correct schema."""
        boxes = np.array([], dtype=np.float32).reshape(0, 4)
        scores = np.array([], dtype=np.float32)
        class_ids = np.array([], dtype=np.int32)
        transform = Affine.identity()
        crs = CRS.from_epsg(4326)

        gdf = build_geodataframe(boxes, scores, class_ids, None, transform, crs)
        assert len(gdf) == 0
        assert isinstance(gdf, gpd.GeoDataFrame)
        assert gdf.crs == crs
        # Check schema
        required = {"geometry", "class_id", "class_name", "confidence", "centroid_x", "centroid_y"}
        assert required.issubset(set(gdf.columns))

    def test_confidence_is_float(self):
        """Confidence values are float type."""
        boxes, scores, class_ids = make_detections(2)
        transform = Affine(0.3, 0, 500000, 0, -0.3, 4000000)
        crs = CRS.from_epsg(32610)
        gdf = build_geodataframe(boxes, scores, class_ids, None, transform, crs)
        assert gdf["confidence"].dtype == np.float64  # float cast


# ---------------------------------------------------------------------------
# Tests: build_dataframe_pixel
# ---------------------------------------------------------------------------


class TestBuildDataFramePixel:
    """Test pixel-only DataFrame construction."""

    def test_correct_columns(self):
        """Pixel DataFrame has all required columns."""
        boxes, scores, class_ids = make_detections(3)
        df = build_dataframe_pixel(boxes, scores, class_ids)
        expected_cols = {"x1", "y1", "x2", "y2", "class_id", "class_name", "confidence"}
        assert expected_cols == set(df.columns)

    def test_returns_dataframe_not_geodataframe(self):
        """Returns plain DataFrame, not GeoDataFrame."""
        boxes, scores, class_ids = make_detections(3)
        df = build_dataframe_pixel(boxes, scores, class_ids)
        assert isinstance(df, pd.DataFrame)
        assert not isinstance(df, gpd.GeoDataFrame)

    def test_correct_row_count(self):
        """N detections produce N rows."""
        boxes, scores, class_ids = make_detections(5)
        df = build_dataframe_pixel(boxes, scores, class_ids)
        assert len(df) == 5

    def test_coordinates_preserved(self):
        """Pixel coordinates are preserved exactly."""
        boxes = np.array([[10.5, 20.3, 30.7, 40.1]], dtype=np.float32)
        scores = np.array([0.9], dtype=np.float32)
        class_ids = np.array([0], dtype=np.int32)
        df = build_dataframe_pixel(boxes, scores, class_ids)
        assert df["x1"].iloc[0] == pytest.approx(10.5, abs=0.01)
        assert df["y1"].iloc[0] == pytest.approx(20.3, abs=0.01)
        assert df["x2"].iloc[0] == pytest.approx(30.7, abs=0.01)
        assert df["y2"].iloc[0] == pytest.approx(40.1, abs=0.01)

    def test_empty_detections(self):
        """Zero detections produce empty DataFrame with correct schema."""
        boxes = np.array([], dtype=np.float32).reshape(0, 4)
        scores = np.array([], dtype=np.float32)
        class_ids = np.array([], dtype=np.int32)
        df = build_dataframe_pixel(boxes, scores, class_ids)
        assert len(df) == 0
        expected_cols = {"x1", "y1", "x2", "y2", "class_id", "class_name", "confidence"}
        assert expected_cols == set(df.columns)

    def test_class_names_mapping(self):
        """Class names from mapping appear in pixel DataFrame."""
        boxes = np.array([[0, 0, 10, 10]], dtype=np.float32)
        scores = np.array([0.9], dtype=np.float32)
        class_ids = np.array([1], dtype=np.int32)
        df = build_dataframe_pixel(boxes, scores, class_ids, class_names={1: "tree"})
        assert df["class_name"].iloc[0] == "tree"


# ---------------------------------------------------------------------------
# Tests: compute_areas
# ---------------------------------------------------------------------------


class TestComputeAreas:
    """Test area computation."""

    def test_projected_crs_direct_area(self):
        """Projected CRS (meters) computes area directly."""
        # Create a 100m x 100m square polygon in UTM
        poly = Polygon(
            [
                (500000, 4000000),
                (500100, 4000000),
                (500100, 4000100),
                (500000, 4000100),
            ]
        )
        gdf = gpd.GeoDataFrame(
            {"geometry": [poly]},
            crs=CRS.from_epsg(32610),
        )
        areas = compute_areas(gdf)
        assert areas.iloc[0] == pytest.approx(10000.0, rel=0.01)

    def test_geographic_crs_auto_utm(self):
        """Geographic CRS triggers auto-UTM reprojection."""
        # Small polygon near San Francisco
        poly = Polygon(
            [
                (-122.0, 37.0),
                (-121.999, 37.0),
                (-121.999, 37.001),
                (-122.0, 37.001),
            ]
        )
        gdf = gpd.GeoDataFrame(
            {"geometry": [poly]},
            crs=CRS.from_epsg(4326),
        )
        areas = compute_areas(gdf)
        # Should return a positive area in square meters
        assert areas.iloc[0] > 0
        # Roughly 100m x 100m area
        assert 1000 < areas.iloc[0] < 100000

    def test_explicit_equal_area_crs(self):
        """Explicit equal_area_crs overrides auto-detection."""
        poly = Polygon(
            [
                (-122.0, 37.0),
                (-121.999, 37.0),
                (-121.999, 37.001),
                (-122.0, 37.001),
            ]
        )
        gdf = gpd.GeoDataFrame(
            {"geometry": [poly]},
            crs=CRS.from_epsg(4326),
        )
        areas = compute_areas(gdf, equal_area_crs=CRS.from_epsg(32610))
        assert areas.iloc[0] > 0

    def test_empty_geodataframe(self):
        """Empty GeoDataFrame returns empty Series."""
        gdf = gpd.GeoDataFrame({"geometry": []}, crs=CRS.from_epsg(32610))
        areas = compute_areas(gdf)
        assert len(areas) == 0

    def test_no_crs_returns_raw_area(self):
        """No CRS returns raw geometric area."""
        poly = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
        gdf = gpd.GeoDataFrame({"geometry": [poly]})
        gdf.crs = None
        areas = compute_areas(gdf)
        assert areas.iloc[0] == pytest.approx(100.0)

    def test_areas_are_non_negative(self):
        """All areas are non-negative."""
        boxes, scores, class_ids = make_detections(5)
        transform = Affine(0.3, 0, 500000, 0, -0.3, 4000000)
        crs = CRS.from_epsg(32610)
        gdf = build_geodataframe(boxes, scores, class_ids, None, transform, crs)
        areas = compute_areas(gdf)
        assert (areas >= 0).all()
