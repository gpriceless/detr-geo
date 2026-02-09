"""Tests for the DetrGeo main class."""

from __future__ import annotations

import inspect

import pytest
from pyproj import CRS

from detr_geo.core import DetrGeo
from detr_geo.exceptions import CRSError, DetrGeoError


class TestInstantiation:
    """Test DetrGeo instantiation."""

    def test_default_instantiation(self):
        """DetrGeo() should create an instance with default parameters."""
        dg = DetrGeo()
        assert dg is not None
        # Adapter created but model not loaded
        assert dg._adapter.is_loaded is False

    def test_nano_model_size(self):
        """DetrGeo(model_size="nano") should use nano configuration."""
        dg = DetrGeo(model_size="nano")
        assert dg.resolution == 384

    def test_explicit_device(self):
        """DetrGeo(device="cpu") should set device without detection."""
        dg = DetrGeo(device="cpu")
        assert dg._adapter._device == "cpu"

    def test_confidence_threshold(self):
        """DetrGeo(confidence_threshold=0.7) should store threshold."""
        dg = DetrGeo(confidence_threshold=0.7)
        assert dg._adapter._confidence_threshold == 0.7


class TestCRSProperty:
    """Test CRS property read/write."""

    def test_crs_none_before_set_image(self):
        """CRS should be None before any image is loaded."""
        dg = DetrGeo()
        assert dg.crs is None

    def test_crs_setter_epsg_string(self):
        """CRS setter should accept EPSG string."""
        dg = DetrGeo()
        dg.crs = "EPSG:4326"
        assert dg.crs is not None
        assert dg.crs == CRS.from_epsg(4326)

    def test_crs_setter_pyproj_crs(self):
        """CRS setter should accept pyproj.CRS object."""
        dg = DetrGeo()
        crs_obj = CRS.from_epsg(32617)
        dg.crs = crs_obj
        assert dg.crs == crs_obj

    def test_crs_setter_invalid_raises_crs_error(self):
        """CRS setter should raise CRSError for invalid value."""
        dg = DetrGeo()
        with pytest.raises(CRSError):
            dg.crs = "not_a_crs"


class TestResolutionProperty:
    """Test resolution delegation to adapter."""

    def test_resolution_delegates_to_adapter(self):
        """resolution should return the adapter's resolution."""
        dg = DetrGeo(model_size="base")
        assert dg.resolution == 560

    def test_resolution_nano(self):
        dg = DetrGeo(model_size="nano")
        assert dg.resolution == 384


class TestDetectionsProperty:
    """Test detections property."""

    def test_detections_none_before_detection(self):
        """Detections should be None before any detection is run."""
        dg = DetrGeo()
        assert dg.detections is None


class TestDetectGuard:
    """Test that detect raises error without image."""

    def test_detect_without_image_raises_error(self):
        """detect() without set_image() should raise DetrGeoError."""
        dg = DetrGeo()
        with pytest.raises(DetrGeoError, match="No image has been set"):
            dg.detect()

    def test_detect_tiled_without_image_raises_error(self):
        """detect_tiled() without set_image() should raise DetrGeoError."""
        dg = DetrGeo()
        with pytest.raises(DetrGeoError, match="No image has been set"):
            dg.detect_tiled()


class TestMethodSignatures:
    """Test that all public methods exist with correct signatures."""

    def test_set_image_signature(self):
        sig = inspect.signature(DetrGeo.set_image)
        params = sig.parameters
        assert "source" in params
        assert "bands" in params
        assert "georeferenced" in params
        assert params["bands"].default == "rgb"
        assert params["georeferenced"].default is True

    def test_detect_signature(self):
        sig = inspect.signature(DetrGeo.detect)
        params = sig.parameters
        assert "threshold" in params
        assert "classes" in params
        assert params["threshold"].default is None
        assert params["classes"].default is None

    def test_detect_tiled_signature(self):
        sig = inspect.signature(DetrGeo.detect_tiled)
        params = sig.parameters
        assert "tile_size" in params
        assert "overlap" in params
        assert "nms_threshold" in params
        assert "nodata_threshold" in params
        assert "threshold" in params
        assert "classes" in params
        assert "batch_size" in params
        assert params["tile_size"].default is None
        assert params["overlap"].default == 0.2
        assert params["nms_threshold"].default == 0.5
        assert params["nodata_threshold"].default == 0.5
        assert params["batch_size"].default is None

    def test_to_geojson_signature(self):
        sig = inspect.signature(DetrGeo.to_geojson)
        params = sig.parameters
        assert "path" in params
        assert "simplify_tolerance" in params
        assert params["simplify_tolerance"].default is None

    def test_to_gpkg_signature(self):
        sig = inspect.signature(DetrGeo.to_gpkg)
        params = sig.parameters
        assert "path" in params
        assert "layer" in params
        assert params["layer"].default == "detections"

    def test_to_shp_signature(self):
        sig = inspect.signature(DetrGeo.to_shp)
        params = sig.parameters
        assert "path" in params

    def test_show_map_signature(self):
        sig = inspect.signature(DetrGeo.show_map)
        params = sig.parameters
        assert "basemap" in params
        assert params["basemap"].default == "SATELLITE"

    def test_show_detections_signature(self):
        sig = inspect.signature(DetrGeo.show_detections)
        params = sig.parameters
        assert "figsize" in params
        assert params["figsize"].default == (12, 10)
