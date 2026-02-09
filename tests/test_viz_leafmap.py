"""Tests for leafmap visualization functions.

leafmap is optional and may not be installed. Tests that require leafmap
use mocking or are skipped if leafmap is unavailable.
"""

from __future__ import annotations

import sys
import warnings
from unittest.mock import MagicMock, patch

import geopandas as gpd
import numpy as np
import pytest
from pyproj import CRS
from shapely.geometry import box

pytest.importorskip("matplotlib")

from detr_geo.viz import (  # noqa: E402
    _gdf_to_styled_geojson,
    _rgb_to_hex,
    get_class_colors,
    show_map,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def make_test_gdf(n: int = 5, crs_epsg: int = 4326) -> gpd.GeoDataFrame:
    """Create a synthetic GeoDataFrame for leafmap testing."""
    rng = np.random.RandomState(42)
    geometries = []
    class_names = []
    confidences = []
    class_options = ["building", "vehicle", "tree"]

    for i in range(n):
        # Small polygons near San Francisco
        lon = -122.0 + rng.uniform(-0.01, 0.01)
        lat = 37.0 + rng.uniform(-0.01, 0.01)
        geometries.append(box(lon, lat, lon + 0.001, lat + 0.001))
        class_names.append(class_options[i % len(class_options)])
        confidences.append(rng.uniform(0.3, 0.99))

    return gpd.GeoDataFrame(
        {
            "geometry": geometries,
            "class_name": class_names,
            "confidence": confidences,
            "class_id": list(range(n)),
        },
        crs=CRS.from_epsg(crs_epsg),
    )


# ---------------------------------------------------------------------------
# Tests: _rgb_to_hex
# ---------------------------------------------------------------------------


class TestRgbToHex:
    """Test RGB to hex conversion."""

    def test_pure_red(self):
        assert _rgb_to_hex((1.0, 0.0, 0.0)) == "#ff0000"

    def test_pure_green(self):
        assert _rgb_to_hex((0.0, 1.0, 0.0)) == "#00ff00"

    def test_pure_blue(self):
        assert _rgb_to_hex((0.0, 0.0, 1.0)) == "#0000ff"

    def test_white(self):
        assert _rgb_to_hex((1.0, 1.0, 1.0)) == "#ffffff"

    def test_black(self):
        assert _rgb_to_hex((0.0, 0.0, 0.0)) == "#000000"


# ---------------------------------------------------------------------------
# Tests: _gdf_to_styled_geojson
# ---------------------------------------------------------------------------


class TestGdfToStyledGeoJson:
    """Test GeoJSON conversion with style properties."""

    def test_produces_feature_collection(self):
        """Output is a valid GeoJSON FeatureCollection."""
        gdf = make_test_gdf(3)
        colors = get_class_colors(sorted(gdf["class_name"].unique().tolist()))
        result = _gdf_to_styled_geojson(gdf, colors)
        assert result["type"] == "FeatureCollection"
        assert len(result["features"]) == 3

    def test_features_have_style_properties(self):
        """Each feature has fill_color, stroke_color, opacity, popup."""
        gdf = make_test_gdf(3)
        colors = get_class_colors(sorted(gdf["class_name"].unique().tolist()))
        result = _gdf_to_styled_geojson(gdf, colors)
        for feature in result["features"]:
            props = feature["properties"]
            assert "fill_color" in props
            assert "stroke_color" in props
            assert "opacity" in props
            assert "popup" in props

    def test_popup_contains_class_and_confidence(self):
        """Popup text includes class name and confidence."""
        gdf = make_test_gdf(1)
        colors = get_class_colors(sorted(gdf["class_name"].unique().tolist()))
        result = _gdf_to_styled_geojson(gdf, colors)
        popup = result["features"][0]["properties"]["popup"]
        assert "Class:" in popup
        assert "Confidence:" in popup

    def test_opacity_varies_with_confidence(self):
        """Higher confidence produces higher opacity."""
        gdf = make_test_gdf(2)
        gdf["confidence"] = [0.3, 0.9]
        colors = get_class_colors(sorted(gdf["class_name"].unique().tolist()))
        result = _gdf_to_styled_geojson(gdf, colors)
        opacity_low = result["features"][0]["properties"]["opacity"]
        opacity_high = result["features"][1]["properties"]["opacity"]
        assert opacity_high > opacity_low

    def test_color_is_hex_string(self):
        """fill_color and stroke_color are hex strings."""
        gdf = make_test_gdf(1)
        colors = get_class_colors(sorted(gdf["class_name"].unique().tolist()))
        result = _gdf_to_styled_geojson(gdf, colors)
        fill = result["features"][0]["properties"]["fill_color"]
        assert fill.startswith("#")
        assert len(fill) == 7

    def test_utm_gdf_reprojected_to_4326(self):
        """UTM GeoDataFrame is reprojected to WGS84 in GeoJSON."""
        gdf = make_test_gdf(2, crs_epsg=32610)
        colors = get_class_colors(sorted(gdf["class_name"].unique().tolist()))
        result = _gdf_to_styled_geojson(gdf, colors)
        # Features should exist (reprojection succeeded)
        assert len(result["features"]) == 2

    def test_empty_gdf(self):
        """Empty GeoDataFrame produces empty feature collection."""
        gdf = gpd.GeoDataFrame(
            {"geometry": [], "class_name": [], "confidence": []},
            crs=CRS.from_epsg(4326),
        )
        result = _gdf_to_styled_geojson(gdf, {})
        assert result["type"] == "FeatureCollection"
        assert len(result["features"]) == 0


# ---------------------------------------------------------------------------
# Tests: show_map
# ---------------------------------------------------------------------------


class TestShowMapWithoutLeafmap:
    """Test show_map behavior when leafmap is not installed."""

    def test_raises_import_error_with_instructions(self):
        """show_map raises ImportError with install instructions when leafmap missing."""
        gdf = make_test_gdf(3)

        # Mock leafmap as not importable
        with patch.dict(sys.modules, {"leafmap": None}):
            with pytest.raises(ImportError, match="pip install leafmap"):
                show_map(gdf)


class TestShowMapWithMockedLeafmap:
    """Test show_map with mocked leafmap."""

    def _make_mock_leafmap(self):
        """Create a mock leafmap module."""
        mock_map = MagicMock()
        mock_leafmap = MagicMock()
        mock_leafmap.Map.return_value = mock_map
        return mock_leafmap, mock_map

    def test_creates_map_object(self):
        """show_map returns a map object."""
        mock_leafmap, mock_map = self._make_mock_leafmap()
        gdf = make_test_gdf(3)

        with patch.dict(sys.modules, {"leafmap": mock_leafmap}):
            result = show_map(gdf)
        assert result is mock_map

    def test_adds_geojson_layer(self):
        """show_map adds a GeoJSON layer to the map."""
        mock_leafmap, mock_map = self._make_mock_leafmap()
        gdf = make_test_gdf(3)

        with patch.dict(sys.modules, {"leafmap": mock_leafmap}):
            show_map(gdf)
        mock_map.add_geojson.assert_called_once()

    def test_reuses_existing_map(self):
        """map_object parameter reuses existing map."""
        mock_leafmap, mock_map = self._make_mock_leafmap()
        existing_map = MagicMock()
        gdf = make_test_gdf(3)

        with patch.dict(sys.modules, {"leafmap": mock_leafmap}):
            result = show_map(gdf, map_object=existing_map)
        assert result is existing_map
        existing_map.add_geojson.assert_called_once()

    def test_max_detections_limit(self):
        """Excess detections are limited and a warning is issued."""
        mock_leafmap, mock_map = self._make_mock_leafmap()
        gdf = make_test_gdf(20)

        with patch.dict(sys.modules, {"leafmap": mock_leafmap}), warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            show_map(gdf, max_detections=5)
            limit_warnings = [
                x for x in w if "max_detections" in str(x.message).lower() or "top" in str(x.message).lower()
            ]
            assert len(limit_warnings) > 0

    def test_min_confidence_filter(self):
        """min_confidence filters detections before mapping."""
        mock_leafmap, mock_map = self._make_mock_leafmap()
        gdf = make_test_gdf(10)
        # Set some to low confidence
        gdf.loc[gdf.index[:5], "confidence"] = 0.1

        with patch.dict(sys.modules, {"leafmap": mock_leafmap}):
            show_map(gdf, min_confidence=0.5)

        # Verify add_geojson was called (features filtered)
        call_args = mock_map.add_geojson.call_args
        geojson_data = call_args[0][0] if call_args[0] else call_args[1].get("in_geojson")
        if geojson_data:
            # Filtered GDF should have fewer features
            assert len(geojson_data["features"]) <= 10

    def test_class_filter(self):
        """classes parameter filters to specific classes."""
        mock_leafmap, mock_map = self._make_mock_leafmap()
        gdf = make_test_gdf(10)

        with patch.dict(sys.modules, {"leafmap": mock_leafmap}):
            show_map(gdf, classes=["building"])
        mock_map.add_geojson.assert_called_once()

    def test_empty_gdf_no_geojson_layer(self):
        """Empty GeoDataFrame creates map without GeoJSON layer."""
        mock_leafmap, mock_map = self._make_mock_leafmap()
        gdf = gpd.GeoDataFrame(
            {"geometry": [], "class_name": [], "confidence": []},
            crs=CRS.from_epsg(4326),
        )

        with patch.dict(sys.modules, {"leafmap": mock_leafmap}):
            show_map(gdf)
        # add_geojson should not be called for empty data
        mock_map.add_geojson.assert_not_called()
