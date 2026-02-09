"""Integration tests for DetrGeo core orchestration layer.

These tests verify that the DetrGeo class correctly wires together
all the underlying modules (io, tiling, export, viz, adapter) and
validates the end-to-end workflows.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import geopandas as gpd
import numpy as np
import pytest
from pyproj import CRS
from rasterio.transform import Affine

from detr_geo.core import DetrGeo
from detr_geo.exceptions import DetrGeoError, ExportError, MissingCRSError


@pytest.fixture
def mock_raster_metadata():
    """Mock raster metadata for a georeferenced raster."""
    from detr_geo.io import RasterMetadata

    return RasterMetadata(
        crs=CRS.from_epsg(32618),
        transform=Affine(1.0, 0.0, 0.0, 0.0, -1.0, 1000.0),
        width=1000,
        height=1000,
        count=3,
        dtype="uint8",
        nodata=None,
        has_alpha=False,
        bounds=(0.0, 0.0, 1000.0, 1000.0),
        gsd=1.0,
    )


@pytest.fixture
def mock_detection_result():
    """Mock detection result from adapter."""
    return {
        "bbox": [[10, 20, 50, 60], [100, 200, 150, 250]],
        "confidence": [0.95, 0.87],
        "class_id": [1, 2],
    }


@pytest.fixture
def mock_adapter():
    """Mock RFDETRAdapter for testing without rfdetr installed."""
    adapter = MagicMock()
    adapter.resolution = 576
    adapter.class_names = {0: "person", 1: "car", 2: "building"}
    adapter.predict_tile = MagicMock()
    adapter.predict_batch = MagicMock()
    return adapter


class TestSetImage:
    """Tests for set_image() method."""

    def test_set_image_georeferenced(self, tmp_path, mock_raster_metadata):
        """Test loading a georeferenced raster."""
        raster_path = tmp_path / "test.tif"
        raster_path.touch()

        with patch("detr_geo.io.load_raster_metadata", return_value=mock_raster_metadata):
            dg = DetrGeo()
            dg.set_image(str(raster_path))

            assert dg._source_path == str(raster_path)
            assert dg._crs is not None
            assert dg._transform is not None
            assert dg._meta is not None
            # Image should NOT be loaded yet (deferred loading)
            assert dg._image is None

    def test_set_image_missing_crs_georeferenced_true(self, tmp_path):
        """Test error when georeferenced=True but raster has no CRS."""
        from detr_geo.io import RasterMetadata

        raster_path = tmp_path / "test.tif"
        raster_path.touch()

        meta_no_crs = RasterMetadata(
            crs=None,
            transform=Affine(1.0, 0.0, 0.0, 0.0, -1.0, 1000.0),
            width=100,
            height=100,
            count=3,
            dtype="uint8",
            nodata=None,
            has_alpha=False,
            bounds=(0.0, 0.0, 100.0, 100.0),
            gsd=None,
        )

        with patch("detr_geo.io.load_raster_metadata", return_value=meta_no_crs):
            dg = DetrGeo()
            with pytest.raises(MissingCRSError, match="Raster has no CRS"):
                dg.set_image(str(raster_path), georeferenced=True)

    def test_set_image_pixel_mode(self, tmp_path):
        """Test loading in pixel-only mode (georeferenced=False)."""
        from detr_geo.io import RasterMetadata

        raster_path = tmp_path / "test.tif"
        raster_path.touch()

        meta_no_crs = RasterMetadata(
            crs=None,
            transform=Affine(1.0, 0.0, 0.0, 0.0, -1.0, 1000.0),
            width=100,
            height=100,
            count=3,
            dtype="uint8",
            nodata=None,
            has_alpha=False,
            bounds=(0.0, 0.0, 100.0, 100.0),
            gsd=None,
        )

        with patch("detr_geo.io.load_raster_metadata", return_value=meta_no_crs):
            dg = DetrGeo()
            dg.set_image(str(raster_path), georeferenced=False)

            assert dg._source_path == str(raster_path)
            assert dg._crs is None
            # Image should NOT be loaded yet
            assert dg._image is None

    def test_set_image_defers_pixel_loading(self, tmp_path, mock_raster_metadata):
        """Test that set_image() does not load pixels immediately."""
        raster_path = tmp_path / "test.tif"
        raster_path.touch()

        with patch("detr_geo.io.load_raster_metadata", return_value=mock_raster_metadata):
            dg = DetrGeo()
            dg.set_image(str(raster_path))

            # Image pixels should NOT be loaded
            assert dg._image is None
            # Metadata should be stored
            assert dg._meta is not None
            assert dg._bands == "rgb"

    def test_large_raster_warning(self, tmp_path):
        """Test that loading a large raster issues a ResourceWarning."""
        from detr_geo.io import RasterMetadata

        raster_path = tmp_path / "test.tif"
        raster_path.touch()

        # Create metadata for a large raster (10000 x 10000 pixels = ~1.2 GB)
        large_meta = RasterMetadata(
            crs=CRS.from_epsg(32618),
            transform=Affine(1.0, 0.0, 0.0, 0.0, -1.0, 1000.0),
            width=10000,
            height=10000,
            count=3,
            dtype="uint8",
            nodata=None,
            has_alpha=False,
            bounds=(0.0, 0.0, 10000.0, 10000.0),
            gsd=1.0,
        )

        with patch("detr_geo.io.load_raster_metadata", return_value=large_meta):
            with patch("detr_geo.io.read_tile") as mock_read:
                with patch("detr_geo.io.BandSelector") as mock_band_selector:
                    mock_read.return_value = (np.ones((3, 100, 100), dtype=np.uint8), None)
                    mock_selector_instance = MagicMock()
                    mock_selector_instance.band_indices = [1, 2, 3]
                    mock_selector_instance.select.return_value = (np.ones((3, 100, 100), dtype=np.uint8), None)
                    mock_band_selector.return_value = mock_selector_instance

                    dg = DetrGeo()
                    dg.set_image(str(raster_path))

                    # Now trigger the load and check for warning
                    with pytest.warns(ResourceWarning, match="Consider detect_tiled"):
                        dg._load_full_image()


class TestDetect:
    """Tests for detect() method."""

    def test_detect_no_image_error(self):
        """Test error when detect() called before set_image()."""
        dg = DetrGeo()
        with pytest.raises(DetrGeoError, match="No image has been set"):
            dg.detect()

    def test_detect_triggers_lazy_load(self, mock_adapter, mock_detection_result, tmp_path):
        """Test that detect() triggers lazy loading of the image."""
        from detr_geo.io import RasterMetadata

        raster_path = tmp_path / "test.tif"
        raster_path.touch()

        meta = RasterMetadata(
            crs=CRS.from_epsg(32618),
            transform=Affine(1.0, 0.0, 0.0, 0.0, -1.0, 1000.0),
            width=100,
            height=100,
            count=3,
            dtype="uint8",
            nodata=None,
            has_alpha=False,
            bounds=(0.0, 0.0, 100.0, 100.0),
            gsd=1.0,
        )

        dg = DetrGeo()
        dg._adapter = mock_adapter
        dg._source_path = str(raster_path)
        dg._crs = CRS.from_epsg(32618)
        dg._transform = Affine(1.0, 0.0, 0.0, 0.0, -1.0, 1000.0)
        dg._meta = meta
        dg._bands = "rgb"
        dg._image = None  # Not loaded yet

        mock_adapter.predict_tile.return_value = mock_detection_result

        with patch("detr_geo.io.read_tile") as mock_read:
            with patch("detr_geo.io.BandSelector") as mock_band_selector:
                with patch("detr_geo.io.normalize_to_float32") as mock_norm:
                    with patch("detr_geo._adapter.prepare_tile_image") as mock_prep:
                        with patch("detr_geo.export.build_geodataframe") as mock_build:
                            mock_read.return_value = (np.ones((3, 100, 100), dtype=np.uint8), None)
                            mock_selector_instance = MagicMock()
                            mock_selector_instance.band_indices = [1, 2, 3]
                            mock_selector_instance.select.return_value = (np.ones((3, 100, 100), dtype=np.uint8), None)
                            mock_band_selector.return_value = mock_selector_instance
                            mock_norm.return_value = (np.ones((3, 100, 100), dtype=np.float32), None)
                            mock_prep.return_value = MagicMock()
                            mock_gdf = gpd.GeoDataFrame({"geometry": [], "class_id": [], "confidence": []}, crs=dg._crs)
                            mock_build.return_value = mock_gdf

                            # Image should not be loaded yet
                            assert dg._image is None

                            result = dg.detect()

                            # Image should now be loaded
                            assert dg._image is not None
                            assert result is not None
                            assert dg._detections is not None

    def test_detect_georeferenced(self, mock_adapter, mock_detection_result):
        """Test detect() with georeferenced mode."""
        dg = DetrGeo()
        dg._adapter = mock_adapter
        dg._image = np.ones((3, 100, 100), dtype=np.float32)
        dg._crs = CRS.from_epsg(32618)
        dg._transform = Affine(1.0, 0.0, 0.0, 0.0, -1.0, 1000.0)
        dg._source_path = "/tmp/test.tif"  # Add source_path to avoid error

        mock_adapter.predict_tile.return_value = mock_detection_result

        with patch("detr_geo.io.normalize_to_float32") as mock_norm:
            with patch("detr_geo._adapter.prepare_tile_image") as mock_prep:
                with patch("detr_geo.export.build_geodataframe") as mock_build:
                    mock_norm.return_value = (dg._image, None)
                    mock_prep.return_value = MagicMock()
                    mock_gdf = gpd.GeoDataFrame({"geometry": [], "class_id": [], "confidence": []}, crs=dg._crs)
                    mock_build.return_value = mock_gdf

                    result = dg.detect()

                    assert result is not None
                    assert dg._detections is not None
                    mock_build.assert_called_once()

    def test_detect_pixel_mode(self, mock_adapter, mock_detection_result):
        """Test detect() with pixel-only mode."""
        dg = DetrGeo()
        dg._adapter = mock_adapter
        dg._image = np.ones((3, 100, 100), dtype=np.float32)
        dg._crs = None
        dg._transform = None
        dg._source_path = "/tmp/test.tif"  # Add source_path to avoid error

        mock_adapter.predict_tile.return_value = mock_detection_result

        with patch("detr_geo.io.normalize_to_float32") as mock_norm:
            with patch("detr_geo._adapter.prepare_tile_image") as mock_prep:
                with patch("detr_geo.export.build_dataframe_pixel") as mock_build:
                    import pandas as pd

                    mock_norm.return_value = (dg._image, None)
                    mock_prep.return_value = MagicMock()
                    mock_df = pd.DataFrame({"x1": [], "y1": [], "x2": [], "y2": [], "class_id": [], "confidence": []})
                    mock_build.return_value = mock_df

                    result = dg.detect()

                    assert result is not None
                    mock_build.assert_called_once()

    def test_detect_with_class_filter(self, mock_adapter):
        """Test detect() with class name filtering."""
        dg = DetrGeo()
        dg._adapter = mock_adapter
        dg._image = np.ones((3, 100, 100), dtype=np.float32)
        dg._crs = CRS.from_epsg(32618)
        dg._transform = Affine(1.0, 0.0, 0.0, 0.0, -1.0, 1000.0)
        dg._source_path = "/tmp/test.tif"  # Add source_path to avoid error

        detection_result = {
            "bbox": [[10, 20, 50, 60], [100, 200, 150, 250], [300, 400, 350, 450]],
            "confidence": [0.95, 0.87, 0.92],
            "class_id": [1, 2, 1],
        }
        mock_adapter.predict_tile.return_value = detection_result

        with patch("detr_geo.io.normalize_to_float32") as mock_norm:
            with patch("detr_geo._adapter.prepare_tile_image") as mock_prep:
                with patch("detr_geo.export.build_geodataframe") as mock_build:
                    mock_norm.return_value = (dg._image, None)
                    mock_prep.return_value = MagicMock()
                    mock_gdf = gpd.GeoDataFrame({"geometry": [], "class_id": [], "confidence": []}, crs=dg._crs)
                    mock_build.return_value = mock_gdf

                    dg.detect(classes=["car"])

                    call_args = mock_build.call_args
                    assert call_args is not None


class TestDetectTiled:
    """Tests for detect_tiled() method."""

    def test_detect_tiled_no_image_error(self):
        """Test error when detect_tiled() called before set_image()."""
        dg = DetrGeo()
        with pytest.raises(DetrGeoError, match="No image has been set"):
            dg.detect_tiled()

    def test_detect_tiled_does_not_load_image(self, mock_adapter, tmp_path):
        """Test that detect_tiled() does NOT trigger full image loading."""
        raster_path = tmp_path / "test.tif"
        raster_path.touch()

        dg = DetrGeo()
        dg._adapter = mock_adapter
        dg._source_path = str(raster_path)
        dg._crs = CRS.from_epsg(32618)
        dg._transform = Affine(1.0, 0.0, 0.0, 0.0, -1.0, 1000.0)
        dg._image = None  # Should remain None

        mock_boxes = np.array([], dtype=np.float32).reshape(0, 4)
        mock_scores = np.array([], dtype=np.float32)
        mock_class_ids = np.array([], dtype=np.int32)

        with patch("detr_geo.tiling.process_tiles") as mock_process:
            with patch("detr_geo.export.build_geodataframe") as mock_build:
                mock_process.return_value = (mock_boxes, mock_scores, mock_class_ids)
                mock_gdf = gpd.GeoDataFrame({"geometry": [], "class_id": [], "confidence": []}, crs=dg._crs)
                mock_build.return_value = mock_gdf

                dg.detect_tiled()

                # Image should still be None (not loaded)
                assert dg._image is None

    def test_detect_tiled_georeferenced(self, mock_adapter, tmp_path):
        """Test detect_tiled() with georeferenced mode."""
        raster_path = tmp_path / "test.tif"
        raster_path.touch()

        dg = DetrGeo()
        dg._adapter = mock_adapter
        dg._source_path = str(raster_path)
        dg._crs = CRS.from_epsg(32618)
        dg._transform = Affine(1.0, 0.0, 0.0, 0.0, -1.0, 1000.0)

        mock_boxes = np.array([[10, 20, 50, 60], [100, 200, 150, 250]], dtype=np.float32)
        mock_scores = np.array([0.95, 0.87], dtype=np.float32)
        mock_class_ids = np.array([1, 2], dtype=np.int32)

        with patch("detr_geo.tiling.process_tiles") as mock_process:
            with patch("detr_geo.export.build_geodataframe") as mock_build:
                mock_process.return_value = (mock_boxes, mock_scores, mock_class_ids)
                mock_gdf = gpd.GeoDataFrame({"geometry": [], "class_id": [], "confidence": []}, crs=dg._crs)
                mock_build.return_value = mock_gdf

                result = dg.detect_tiled()

                assert result is not None
                assert dg._detections is not None
                mock_process.assert_called_once()

    def test_detect_tiled_auto_tile_size(self, mock_adapter, tmp_path):
        """Test detect_tiled() auto-selects tile size from model resolution."""
        raster_path = tmp_path / "test.tif"
        raster_path.touch()

        dg = DetrGeo()
        dg._adapter = mock_adapter
        dg._source_path = str(raster_path)
        dg._crs = CRS.from_epsg(32618)
        dg._transform = Affine(1.0, 0.0, 0.0, 0.0, -1.0, 1000.0)

        mock_boxes = np.array([], dtype=np.float32).reshape(0, 4)
        mock_scores = np.array([], dtype=np.float32)
        mock_class_ids = np.array([], dtype=np.int32)

        with patch("detr_geo.tiling.process_tiles") as mock_process:
            with patch("detr_geo.export.build_geodataframe") as mock_build:
                mock_process.return_value = (mock_boxes, mock_scores, mock_class_ids)
                mock_gdf = gpd.GeoDataFrame({"geometry": [], "class_id": [], "confidence": []}, crs=dg._crs)
                mock_build.return_value = mock_gdf

                dg.detect_tiled()

                call_args = mock_process.call_args
                assert call_args[1]["tile_size"] == 576


class TestExportMethods:
    """Tests for export methods."""

    def test_to_geojson_no_detections(self, tmp_path):
        """Test to_geojson() raises error when no detections."""
        dg = DetrGeo()
        with pytest.raises(ExportError, match="No detections to export"):
            dg.to_geojson(str(tmp_path / "out.geojson"))

    def test_to_geojson(self, tmp_path):
        """Test to_geojson() calls export function."""
        dg = DetrGeo()
        dg._detections = gpd.GeoDataFrame({"geometry": [None], "class_id": [1]})

        with patch("detr_geo.export.export_geojson") as mock_export:
            dg.to_geojson(str(tmp_path / "out.geojson"))
            mock_export.assert_called_once()

    def test_to_gpkg_no_detections(self, tmp_path):
        """Test to_gpkg() raises error when no detections."""
        dg = DetrGeo()
        with pytest.raises(ExportError, match="No detections to export"):
            dg.to_gpkg(str(tmp_path / "out.gpkg"))

    def test_to_gpkg(self, tmp_path):
        """Test to_gpkg() calls export function."""
        dg = DetrGeo()
        dg._detections = gpd.GeoDataFrame({"geometry": [None], "class_id": [1]})

        with patch("detr_geo.export.export_gpkg") as mock_export:
            dg.to_gpkg(str(tmp_path / "out.gpkg"))
            mock_export.assert_called_once()

    def test_to_shp_no_detections(self, tmp_path):
        """Test to_shp() raises error when no detections."""
        dg = DetrGeo()
        with pytest.raises(ExportError, match="No detections to export"):
            dg.to_shp(str(tmp_path / "out.shp"))

    def test_to_shp(self, tmp_path):
        """Test to_shp() calls export function."""
        dg = DetrGeo()
        dg._detections = gpd.GeoDataFrame({"geometry": [None], "class_id": [1]})

        with patch("detr_geo.export.export_shp") as mock_export:
            dg.to_shp(str(tmp_path / "out.shp"))
            mock_export.assert_called_once()


class TestVisualizationMethods:
    """Tests for visualization methods."""

    def test_show_map_no_detections(self):
        """Test show_map() raises error when no detections."""
        dg = DetrGeo()
        with pytest.raises(DetrGeoError, match="No detections to show"):
            dg.show_map()

    def test_show_map(self):
        """Test show_map() calls viz function."""
        dg = DetrGeo()
        dg._detections = gpd.GeoDataFrame({"geometry": [None], "class_id": [1]})

        with patch("detr_geo.viz.show_map") as mock_viz:
            mock_viz.return_value = MagicMock()
            result = dg.show_map(basemap="SATELLITE")
            mock_viz.assert_called_once()
            assert result is not None

    def test_show_detections_no_detections(self):
        """Test show_detections() raises error when no detections."""
        dg = DetrGeo()
        with pytest.raises(DetrGeoError, match="No detections to show"):
            dg.show_detections()

    def test_show_detections_no_image(self):
        """Test show_detections() raises error when no image available."""
        dg = DetrGeo()
        dg._detections = gpd.GeoDataFrame({"geometry": [None], "class_id": [1]})
        dg._image = None
        with pytest.raises(DetrGeoError, match="No image available"):
            dg.show_detections()

    def test_show_detections(self):
        """Test show_detections() calls viz function."""
        dg = DetrGeo()
        dg._detections = gpd.GeoDataFrame({"geometry": [None], "class_id": [1]})
        dg._image = np.ones((3, 100, 100), dtype=np.float32)
        dg._source_path = "/tmp/test.tif"  # Add source_path to avoid error

        with patch("detr_geo.io.normalize_to_float32") as mock_norm:
            with patch("detr_geo.viz.show_detections") as mock_viz:
                mock_norm.return_value = (dg._image, None)
                dg.show_detections()
                mock_viz.assert_called_once()


class TestEndToEndWorkflows:
    """Integration tests for complete workflows."""

    def test_workflow_detect_export(self, tmp_path, mock_adapter, mock_raster_metadata):
        """Test complete workflow: set_image -> detect -> export."""
        raster_path = tmp_path / "test.tif"
        raster_path.touch()

        dg = DetrGeo()
        dg._adapter = mock_adapter

        mock_detection_result = {
            "bbox": [[10, 20, 50, 60]],
            "confidence": [0.95],
            "class_id": [1],
        }
        mock_adapter.predict_tile.return_value = mock_detection_result

        with patch("detr_geo.io.load_raster_metadata", return_value=mock_raster_metadata):
            with patch("detr_geo.io.read_tile") as mock_read:
                with patch("detr_geo.io.BandSelector") as mock_band_selector:
                    with patch("detr_geo.io.normalize_to_float32") as mock_norm:
                        with patch("detr_geo._adapter.prepare_tile_image") as mock_prep:
                            with patch("detr_geo.export.build_geodataframe") as mock_build:
                                with patch("detr_geo.export.export_geojson") as mock_export:
                                    mock_read.return_value = (np.ones((3, 100, 100), dtype=np.uint8), None)
                                    mock_selector_instance = MagicMock()
                                    mock_selector_instance.band_indices = [1, 2, 3]
                                    mock_selector_instance.select.return_value = (
                                        np.ones((3, 100, 100), dtype=np.uint8),
                                        None,
                                    )
                                    mock_band_selector.return_value = mock_selector_instance
                                    mock_norm.return_value = (np.ones((3, 100, 100), dtype=np.float32), None)
                                    mock_prep.return_value = MagicMock()
                                    mock_gdf = gpd.GeoDataFrame(
                                        {"geometry": [None], "class_id": [1], "confidence": [0.95]},
                                        crs=mock_raster_metadata.crs,
                                    )
                                    mock_build.return_value = mock_gdf

                                    dg.set_image(str(raster_path))
                                    detections = dg.detect()
                                    dg.to_geojson(str(tmp_path / "out.geojson"))

                                    assert detections is not None
                                    assert len(detections) > 0
                                    mock_export.assert_called_once()


# ---------------------------------------------------------------------------
# Tests: GSD Warning Integration
# ---------------------------------------------------------------------------


class TestGSDWarningIntegration:
    """Test GSD warning integration in set_image()."""

    def test_set_image_warns_on_high_gsd(self):
        """set_image() with high GSD imagery issues warning."""
        from unittest.mock import patch

        from pyproj import CRS
        from rasterio.transform import Affine

        from detr_geo import DetrGeo, GSDWarning
        from detr_geo.io import RasterMetadata

        dg = DetrGeo()

        # Mock metadata with 2.0m GSD (outside optimal range)
        mock_meta = RasterMetadata(
            crs=CRS.from_epsg(32610),
            transform=Affine(2.0, 0.0, 500000.0, 0.0, -2.0, 4000000.0),
            width=100,
            height=100,
            count=3,
            dtype="uint8",
            nodata=None,
            has_alpha=False,
            bounds=(0, 0, 1, 1),
            gsd=2.0,  # High GSD
        )

        with patch("detr_geo.io.load_raster_metadata", return_value=mock_meta):
            with pytest.warns(GSDWarning, match="well below training GSD"):
                dg.set_image("fake_path.tif")

    def test_set_image_no_warning_optimal_gsd(self):
        """set_image() with optimal GSD (0.3m) does not warn."""
        import warnings
        from unittest.mock import patch

        from pyproj import CRS
        from rasterio.transform import Affine

        from detr_geo import DetrGeo
        from detr_geo.io import RasterMetadata

        dg = DetrGeo()

        # Mock metadata with 0.3m GSD (optimal)
        mock_meta = RasterMetadata(
            crs=CRS.from_epsg(32610),
            transform=Affine(0.3, 0.0, 500000.0, 0.0, -0.3, 4000000.0),
            width=100,
            height=100,
            count=3,
            dtype="uint8",
            nodata=None,
            has_alpha=False,
            bounds=(0, 0, 1, 1),
            gsd=0.3,  # Optimal GSD
        )

        with patch("detr_geo.io.load_raster_metadata", return_value=mock_meta):
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                dg.set_image("fake_path.tif")  # Should not raise

    def test_set_image_suppress_gsd_warning(self):
        """suppress_gsd_warning=True suppresses warning."""
        import warnings
        from unittest.mock import patch

        from pyproj import CRS
        from rasterio.transform import Affine

        from detr_geo import DetrGeo
        from detr_geo.io import RasterMetadata

        dg = DetrGeo()

        # Mock metadata with 2.0m GSD (would normally warn)
        mock_meta = RasterMetadata(
            crs=CRS.from_epsg(32610),
            transform=Affine(2.0, 0.0, 500000.0, 0.0, -2.0, 4000000.0),
            width=100,
            height=100,
            count=3,
            dtype="uint8",
            nodata=None,
            has_alpha=False,
            bounds=(0, 0, 1, 1),
            gsd=2.0,
        )

        with patch("detr_geo.io.load_raster_metadata", return_value=mock_meta):
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                dg.set_image("fake_path.tif", suppress_gsd_warning=True)  # Should not raise
