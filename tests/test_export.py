"""Tests for export to GeoPackage, GeoJSON, and Shapefile formats."""

from __future__ import annotations

import json
import tempfile
import warnings
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from pyproj import CRS
from rasterio.transform import Affine

from detr_geo.exceptions import ExportError
from detr_geo.export import (
    build_geodataframe,
    export_geojson,
    export_gpkg,
    export_shp,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def make_test_gdf(n: int = 5, crs_epsg: int = 32610) -> gpd.GeoDataFrame:
    """Create a synthetic GeoDataFrame for export testing."""
    rng = np.random.RandomState(42)
    x1 = rng.uniform(0, 400, n).astype(np.float32)
    y1 = rng.uniform(0, 400, n).astype(np.float32)
    x2 = x1 + rng.uniform(20, 80, n).astype(np.float32)
    y2 = y1 + rng.uniform(20, 80, n).astype(np.float32)
    boxes = np.stack([x1, y1, x2, y2], axis=1)
    scores = rng.uniform(0.5, 1.0, n).astype(np.float32)
    class_ids = rng.randint(0, 3, n).astype(np.int32)

    transform = Affine(0.3, 0, 500000, 0, -0.3, 4000000)
    crs = CRS.from_epsg(crs_epsg)

    return build_geodataframe(boxes, scores, class_ids, None, transform, crs)


@pytest.fixture
def tmp_dir():
    """Create a temporary directory for export files."""
    with tempfile.TemporaryDirectory() as d:
        yield d


# ---------------------------------------------------------------------------
# Tests: GeoPackage export
# ---------------------------------------------------------------------------


class TestExportGpkg:
    """Test GeoPackage export."""

    def test_creates_gpkg_file(self, tmp_dir):
        """Export creates a .gpkg file."""
        gdf = make_test_gdf(5)
        path = str(Path(tmp_dir) / "test.gpkg")
        export_gpkg(gdf, path)
        assert Path(path).exists()

    def test_default_layer_name(self, tmp_dir):
        """Default layer name is 'detections'."""
        gdf = make_test_gdf(3)
        path = str(Path(tmp_dir) / "test.gpkg")
        export_gpkg(gdf, path)
        # Read back with the default layer (which should be "detections")
        reimported = gpd.read_file(path, layer="detections")
        assert len(reimported) == 3

    def test_round_trip_fidelity(self, tmp_dir):
        """Exported data matches when re-imported."""
        gdf = make_test_gdf(3)
        path = str(Path(tmp_dir) / "test.gpkg")
        export_gpkg(gdf, path)

        reimported = gpd.read_file(path)
        assert len(reimported) == len(gdf)
        assert "class_id" in reimported.columns
        assert "confidence" in reimported.columns

    def test_crs_preserved(self, tmp_dir):
        """CRS is preserved in exported file."""
        gdf = make_test_gdf(3, crs_epsg=32610)
        path = str(Path(tmp_dir) / "test.gpkg")
        export_gpkg(gdf, path)

        reimported = gpd.read_file(path)
        assert reimported.crs is not None
        assert reimported.crs.to_epsg() == 32610

    def test_custom_layer_name(self, tmp_dir):
        """Custom layer name is respected."""
        gdf = make_test_gdf(3)
        path = str(Path(tmp_dir) / "test.gpkg")
        export_gpkg(gdf, path, layer="buildings")

        # Read back using the custom layer name
        reimported = gpd.read_file(path, layer="buildings")
        assert len(reimported) == 3

    def test_invalid_path_raises_export_error(self):
        """Invalid path raises ExportError."""
        gdf = make_test_gdf(3)
        with pytest.raises(ExportError):
            export_gpkg(gdf, "/nonexistent/directory/test.gpkg")

    def test_empty_geodataframe(self, tmp_dir):
        """Empty GeoDataFrame exports without error."""
        gdf = gpd.GeoDataFrame(
            {
                "geometry": [],
                "class_id": pd.Series([], dtype="int64"),
                "class_name": pd.Series([], dtype="str"),
                "confidence": pd.Series([], dtype="float64"),
            },
            crs=CRS.from_epsg(32610),
        )
        path = str(Path(tmp_dir) / "empty.gpkg")
        export_gpkg(gdf, path)
        assert Path(path).exists()


# ---------------------------------------------------------------------------
# Tests: GeoJSON export
# ---------------------------------------------------------------------------


class TestExportGeoJSON:
    """Test GeoJSON export."""

    def test_creates_geojson_file(self, tmp_dir):
        """Export creates a .geojson file."""
        gdf = make_test_gdf(3)
        path = str(Path(tmp_dir) / "test.geojson")
        export_geojson(gdf, path)
        assert Path(path).exists()

    def test_utm_reprojected_to_4326(self, tmp_dir):
        """UTM input is reprojected to EPSG:4326."""
        gdf = make_test_gdf(3, crs_epsg=32610)
        path = str(Path(tmp_dir) / "test.geojson")

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            export_geojson(gdf, path)
            # Check reprojection warning was issued
            reprojection_warnings = [x for x in w if "EPSG:4326" in str(x.message)]
            assert len(reprojection_warnings) > 0

        # Verify output is in WGS84
        reimported = gpd.read_file(path)
        assert reimported.crs is not None
        assert reimported.crs.to_epsg() == 4326

    def test_wgs84_no_reprojection_warning(self, tmp_dir):
        """WGS84 input exports without reprojection warning."""
        gdf = make_test_gdf(3, crs_epsg=4326)
        path = str(Path(tmp_dir) / "test.geojson")

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            export_geojson(gdf, path)
            reprojection_warnings = [
                x for x in w if issubclass(x.category, RuntimeWarning) and "Reprojecting" in str(x.message)
            ]
            assert len(reprojection_warnings) == 0

    def test_valid_geojson_structure(self, tmp_dir):
        """Output is valid GeoJSON with FeatureCollection type."""
        gdf = make_test_gdf(3)
        path = str(Path(tmp_dir) / "test.geojson")
        export_geojson(gdf, path)

        with open(path) as f:
            data = json.load(f)
        assert data["type"] == "FeatureCollection"
        assert len(data["features"]) == 3

    def test_coordinate_precision(self, tmp_dir):
        """Coordinates respect precision setting."""
        gdf = make_test_gdf(3)
        path = str(Path(tmp_dir) / "test.geojson")
        export_geojson(gdf, path, coordinate_precision=4)

        with open(path) as f:
            data = json.load(f)

        # Check that coordinates don't have excessive precision
        coords = data["features"][0]["geometry"]["coordinates"][0]
        for coord in coords:
            for val in coord:
                # String representation shouldn't have more than 4 decimal places
                parts = str(val).split(".")
                if len(parts) == 2:
                    assert len(parts[1]) <= 4

    def test_invalid_path_raises_export_error(self):
        """Invalid path raises ExportError."""
        gdf = make_test_gdf(3)
        with pytest.raises(ExportError):
            export_geojson(gdf, "/nonexistent/directory/test.geojson")


# ---------------------------------------------------------------------------
# Tests: Shapefile export
# ---------------------------------------------------------------------------


class TestExportShapefile:
    """Test Shapefile export."""

    def test_creates_shapefile_components(self, tmp_dir):
        """Export creates .shp, .dbf, .prj, .shx files."""
        gdf = make_test_gdf(3)
        path = str(Path(tmp_dir) / "test.shp")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            export_shp(gdf, path)

        base = Path(tmp_dir) / "test"
        assert (base.with_suffix(".shp")).exists()
        assert (base.with_suffix(".dbf")).exists()
        assert (base.with_suffix(".prj")).exists()
        assert (base.with_suffix(".shx")).exists()

    def test_shapefile_legacy_warning(self, tmp_dir):
        """Shapefile export issues legacy format warning."""
        gdf = make_test_gdf(3)
        path = str(Path(tmp_dir) / "test.shp")

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            export_shp(gdf, path)
            legacy_warnings = [x for x in w if "10-char" in str(x.message) or "legacy" in str(x.message).lower()]
            assert len(legacy_warnings) > 0

    def test_round_trip(self, tmp_dir):
        """Exported Shapefile can be re-imported."""
        gdf = make_test_gdf(3)
        path = str(Path(tmp_dir) / "test.shp")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            export_shp(gdf, path)

        reimported = gpd.read_file(path)
        assert len(reimported) == 3

    def test_invalid_path_raises_export_error(self):
        """Invalid path raises ExportError."""
        gdf = make_test_gdf(3)
        with pytest.raises(ExportError), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            export_shp(gdf, "/nonexistent/directory/test.shp")

    def test_empty_geodataframe(self, tmp_dir):
        """Empty GeoDataFrame exports without error."""
        gdf = gpd.GeoDataFrame(
            {
                "geometry": [],
                "class_id": pd.Series([], dtype="int64"),
                "class_name": pd.Series([], dtype="str"),
                "confidence": pd.Series([], dtype="float64"),
            },
            crs=CRS.from_epsg(32610),
        )
        path = str(Path(tmp_dir) / "empty.shp")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            export_shp(gdf, path)
        assert Path(path).exists()
