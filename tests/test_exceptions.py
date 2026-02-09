"""Tests for the exception hierarchy."""

from __future__ import annotations

import pytest

from detr_geo.exceptions import (
    BandError,
    CRSError,
    DetrGeoError,
    ExportError,
    MissingCRSError,
    ModelError,
    TilingError,
)


class TestExceptionHierarchy:
    """Test that exceptions form the correct inheritance tree."""

    def test_detr_geo_error_inherits_from_exception(self):
        assert issubclass(DetrGeoError, Exception)

    def test_crs_error_inherits_from_detr_geo_error(self):
        assert issubclass(CRSError, DetrGeoError)

    def test_missing_crs_error_inherits_from_crs_error(self):
        assert issubclass(MissingCRSError, CRSError)

    def test_missing_crs_error_inherits_from_detr_geo_error(self):
        assert issubclass(MissingCRSError, DetrGeoError)

    def test_tiling_error_inherits_from_detr_geo_error(self):
        assert issubclass(TilingError, DetrGeoError)

    def test_model_error_inherits_from_detr_geo_error(self):
        assert issubclass(ModelError, DetrGeoError)

    def test_band_error_inherits_from_detr_geo_error(self):
        assert issubclass(BandError, DetrGeoError)

    def test_export_error_inherits_from_detr_geo_error(self):
        assert issubclass(ExportError, DetrGeoError)


class TestCatchAll:
    """Test that all exceptions can be caught by catching DetrGeoError."""

    @pytest.mark.parametrize(
        "exc_class",
        [CRSError, MissingCRSError, TilingError, ModelError, BandError, ExportError],
    )
    def test_catch_all_with_detr_geo_error(self, exc_class):
        with pytest.raises(DetrGeoError):
            raise exc_class("test message")

    def test_missing_crs_caught_by_crs_error(self):
        with pytest.raises(CRSError):
            raise MissingCRSError("no CRS found")


class TestMessageStorage:
    """Test that exceptions store informative messages."""

    @pytest.mark.parametrize(
        "exc_class",
        [DetrGeoError, CRSError, MissingCRSError, TilingError, ModelError, BandError, ExportError],
    )
    def test_message_preserved(self, exc_class):
        msg = "This is a test error message"
        exc = exc_class(msg)
        assert str(exc) == msg

    def test_keyword_context_stored(self):
        exc = DetrGeoError("test", source="file.tif", band=3)
        assert exc.context == {"source": "file.tif", "band": 3}

    def test_keyword_context_on_subclass(self):
        exc = ModelError("bad model", model_size="xlarge")
        assert exc.context == {"model_size": "xlarge"}
        assert str(exc) == "bad model"
