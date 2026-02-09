"""DetrGeo main class - public API for geospatial object detection.

This module provides the primary user-facing interface. The DetrGeo class
delegates to specialized internal modules for I/O, CRS handling, tiling,
export, and visualization.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import geopandas as gpd
from pyproj import CRS

from detr_geo._adapter import RFDETRAdapter
from detr_geo._typing import ModelSize
from detr_geo.exceptions import CRSError, DetrGeoError


class DetrGeo:
    """Geospatial object detection using RF-DETR.

    This is the main entry point for the detr_geo library. It provides
    a high-level API for loading geospatial images, running object detection,
    and exporting georeferenced results.

    Example::

        dg = DetrGeo(model_size="medium")
        dg.set_image("path/to/geotiff.tif")
        detections = dg.detect()
        dg.to_geojson("output.geojson")

    Args:
        model_size: One of "nano", "small", "medium", "base", "large".
        device: Compute device ("cuda", "mps", "cpu"). Auto-detected if None.
        lazy_load: If True (default), defer model weight loading until first prediction.
        confidence_threshold: Default confidence threshold for predictions.
        pretrain_weights: Path to custom pretrained weights checkpoint. Uses default
            COCO-pretrained weights if None.
        custom_class_names: Override class name mapping for fine-tuned models.
            Maps class_id (int) to label (str), e.g. ``{0: "Car", 1: "Bus"}``.
            When provided, bypasses the default COCO label lookup.
    """

    def __init__(
        self,
        model_size: ModelSize = "medium",
        device: str | None = None,
        lazy_load: bool = True,
        confidence_threshold: float = 0.5,
        pretrain_weights: str | None = None,
        custom_class_names: dict[int, str] | None = None,
    ) -> None:
        self._adapter = RFDETRAdapter(
            model_size=model_size,
            device=device,
            pretrain_weights=pretrain_weights,
            confidence_threshold=confidence_threshold,
            custom_class_names=custom_class_names,
        )
        self._crs: CRS | None = None
        self._transform: Any = None  # rasterio Affine
        self._detections: gpd.GeoDataFrame | None = None
        self._source_path: str | None = None
        self._image: Any = None  # numpy array or None
        self._lazy_load = lazy_load
        self._meta: Any = None  # RasterMetadata for deferred loading
        self._bands: tuple[int, ...] | str = "rgb"  # Band selection for deferred loading

    @property
    def crs(self) -> CRS | None:
        """The coordinate reference system of the loaded image.

        Returns None if no image has been loaded.
        """
        return self._crs

    @crs.setter
    def crs(self, value: str | CRS) -> None:
        """Set the coordinate reference system.

        Args:
            value: EPSG string (e.g., "EPSG:4326") or pyproj.CRS object.

        Raises:
            CRSError: If the value is not a valid CRS.
        """
        if isinstance(value, CRS):
            self._crs = value
        elif isinstance(value, str):
            try:
                self._crs = CRS.from_user_input(value)
            except Exception as exc:
                raise CRSError(
                    f"Invalid CRS value: '{value}'. "
                    f"Provide an EPSG string (e.g., 'EPSG:4326') or a pyproj.CRS object. "
                    f"Original error: {exc}"
                ) from exc
        else:
            raise CRSError(f"CRS must be a string or pyproj.CRS object, got {type(value).__name__}")

    @property
    def resolution(self) -> int:
        """The model's native square input resolution in pixels.

        Delegated to the underlying RFDETRAdapter.
        """
        return self._adapter.resolution

    @property
    def detections(self) -> gpd.GeoDataFrame | None:
        """The most recent detection results as a GeoDataFrame.

        Returns None if no detection has been run.
        """
        return self._detections

    def set_image(
        self,
        source: str | Path,
        bands: tuple[int, ...] | str = "rgb",
        georeferenced: bool = True,
        suppress_gsd_warning: bool = False,
    ) -> None:
        """Load a geospatial image for detection.

        Args:
            source: Path to a raster file (GeoTIFF, etc.), HTTP/HTTPS URL to a COG,
                S3 URI (s3://bucket/key), or pystac.Item object.
            bands: Band selection as tuple of 1-based indices or preset string.
            georeferenced: If True, read and store CRS/transform from the raster.
            suppress_gsd_warning: If True, suppress warnings about imagery GSD being
                outside the optimal range (0.1-0.5m).

        Raises:
            DetrGeoError: If the image cannot be loaded.
            MissingCRSError: If georeferenced=True but the raster has no CRS.
            FileNotFoundError: If local path doesn't exist.
        """
        from detr_geo.exceptions import MissingCRSError
        from detr_geo.io import check_gsd, load_raster_metadata

        meta = load_raster_metadata(source)

        self._source_path = str(source)
        self._crs = meta.crs
        self._transform = meta.transform
        self._meta = meta
        self._bands = bands
        self._image = None  # Defer loading until needed

        if georeferenced and self._crs is None:
            raise MissingCRSError(
                "Raster has no CRS but georeferenced=True. "
                "Either set georeferenced=False for pixel-only mode or set .crs property explicitly."
            )

        # Check GSD and warn if outside optimal range
        if not suppress_gsd_warning and meta.gsd is not None:
            check_gsd(meta.gsd)

    def _load_full_image(self) -> None:
        """Lazily load the full raster image into memory.

        Called by detect() and show_detections() on first use.
        Issues a ResourceWarning if the raster would use > 1 GB of RAM.
        """
        if self._image is not None:
            return

        if self._meta is None or self._source_path is None:
            raise DetrGeoError("Cannot load image: metadata not available. Call set_image() first.")

        from detr_geo.io import BandSelector, read_tile

        # Check if raster is large enough to warrant a warning
        pixel_bytes = self._meta.width * self._meta.height * 3 * 4  # float32 RGB
        if pixel_bytes > 1_000_000_000:
            warnings.warn(
                f"Loading full raster ({self._meta.width}x{self._meta.height}) will use "
                f"~{pixel_bytes // (1024**2)} MB. Consider detect_tiled() for large imagery.",
                ResourceWarning,
                stacklevel=2,
            )

        band_selector = BandSelector(self._bands)
        band_selector.clamp_to_band_count(self._meta.count)
        data, alpha = read_tile(
            self._source_path,
            window=(0, 0, self._meta.width, self._meta.height),
            bands=band_selector.read_indices,
        )
        rgb_data, _ = band_selector.select(data, self._meta.count)

        self._image = rgb_data

    def detect(
        self,
        threshold: float | None = None,
        classes: list[str] | None = None,
    ) -> gpd.GeoDataFrame:
        """Run object detection on the loaded image.

        Args:
            threshold: Confidence threshold override. Uses default if None.
            classes: Filter to only these class names. All classes if None.

        Returns:
            GeoDataFrame with detection results including geometry column.

        Raises:
            DetrGeoError: If no image has been set via set_image().
        """
        if self._source_path is None:
            raise DetrGeoError("No image has been set. Call set_image() before detect().")

        # Lazy load the full image into memory
        self._load_full_image()

        import numpy as np

        from detr_geo._adapter import prepare_tile_image
        from detr_geo.export import build_dataframe_pixel, build_geodataframe
        from detr_geo.io import normalize_to_float32

        normalized, _ = normalize_to_float32(self._image, stretch="percentile")
        pil_image = prepare_tile_image(normalized)

        result = self._adapter.predict_tile(pil_image, threshold=threshold)

        if classes is not None:
            class_mask = np.isin(
                [self._adapter.class_names.get(cid, f"class_{cid}") for cid in result["class_id"]], classes
            )
            result = {
                "bbox": [result["bbox"][i] for i in range(len(result["bbox"])) if class_mask[i]],
                "confidence": [result["confidence"][i] for i in range(len(result["confidence"])) if class_mask[i]],
                "class_id": [result["class_id"][i] for i in range(len(result["class_id"])) if class_mask[i]],
            }

        if self._crs is not None and self._transform is not None:
            boxes = np.array(result["bbox"], dtype=np.float32)
            scores = np.array(result["confidence"], dtype=np.float32)
            class_ids = np.array(result["class_id"], dtype=np.int32)

            gdf = build_geodataframe(
                boxes,
                scores,
                class_ids,
                class_names=self._adapter.class_names,
                transform=self._transform,
                crs=self._crs,
            )
        else:
            gdf = build_dataframe_pixel(
                np.array(result["bbox"], dtype=np.float32),
                np.array(result["confidence"], dtype=np.float32),
                np.array(result["class_id"], dtype=np.int32),
                class_names=self._adapter.class_names,
            )

        # Store GSD in detection metadata
        if self._meta is not None:
            gdf.attrs["gsd_meters"] = self._meta.gsd

        self._detections = gdf
        return gdf

    def detect_tiled(
        self,
        tile_size: int | None = None,
        overlap: float = 0.2,
        nms_threshold: float = 0.5,
        nodata_threshold: float = 0.5,
        threshold: float | None = None,
        classes: list[str] | None = None,
        batch_size: int | None = None,
    ) -> gpd.GeoDataFrame:
        """Run tiled object detection on a large image.

        Args:
            tile_size: Tile size in pixels. Auto-selected if None.
            overlap: Fractional overlap between adjacent tiles (0.0-1.0).
            nms_threshold: IoU threshold for non-maximum suppression across tiles.
            nodata_threshold: Maximum fraction of nodata pixels before skipping a tile.
            threshold: Confidence threshold override. Uses default if None.
            classes: Filter to only these class names. All classes if None.
            batch_size: Number of tiles to process in parallel. Auto if None.

        Returns:
            GeoDataFrame with detection results including geometry column.

        Raises:
            DetrGeoError: If no image has been set via set_image().
        """
        if self._source_path is None:
            raise DetrGeoError("No image has been set. Call set_image() before detect_tiled().")

        import numpy as np

        from detr_geo.export import build_dataframe_pixel, build_geodataframe
        from detr_geo.tiling import process_tiles

        if tile_size is None:
            tile_size = self._adapter.resolution

        boxes, scores, class_ids = process_tiles(
            raster_path=self._source_path,
            adapter=self._adapter,
            tile_size=tile_size,
            overlap=overlap,
            nms_threshold=nms_threshold,
            nodata_threshold=nodata_threshold,
            threshold=threshold,
            batch_size=batch_size,
            bands="rgb",
            show_progress=True,
        )

        if classes is not None:
            class_names_dict = self._adapter.class_names
            valid_class_ids = [cid for cid, name in class_names_dict.items() if name in classes]
            mask = np.isin(class_ids, valid_class_ids)
            boxes = boxes[mask]
            scores = scores[mask]
            class_ids = class_ids[mask]

        if self._crs is not None and self._transform is not None:
            gdf = build_geodataframe(
                boxes,
                scores,
                class_ids,
                class_names=self._adapter.class_names,
                transform=self._transform,
                crs=self._crs,
            )
        else:
            gdf = build_dataframe_pixel(boxes, scores, class_ids, class_names=self._adapter.class_names)

        # Store GSD in detection metadata
        if self._meta is not None:
            gdf.attrs["gsd_meters"] = self._meta.gsd

        self._detections = gdf
        return gdf

    def to_geojson(self, path: str, simplify_tolerance: float | None = None) -> None:
        """Export detections to GeoJSON format.

        Args:
            path: Output file path.
            simplify_tolerance: Optional geometry simplification tolerance.

        Raises:
            ExportError: If export fails.
        """
        from detr_geo.exceptions import ExportError
        from detr_geo.export import export_geojson

        if self._detections is None or len(self._detections) == 0:
            raise ExportError("No detections to export. Run detect() or detect_tiled() first.")

        export_geojson(self._detections, path)

    def to_gpkg(self, path: str, layer: str = "detections") -> None:
        """Export detections to GeoPackage format.

        Args:
            path: Output file path.
            layer: Layer name within the GeoPackage.

        Raises:
            ExportError: If export fails.
        """
        from detr_geo.exceptions import ExportError
        from detr_geo.export import export_gpkg

        if self._detections is None or len(self._detections) == 0:
            raise ExportError("No detections to export. Run detect() or detect_tiled() first.")

        export_gpkg(self._detections, path, layer=layer)

    def to_shp(self, path: str) -> None:
        """Export detections to Shapefile format.

        Args:
            path: Output file path.

        Raises:
            ExportError: If export fails.
        """
        from detr_geo.exceptions import ExportError
        from detr_geo.export import export_shp

        if self._detections is None or len(self._detections) == 0:
            raise ExportError("No detections to export. Run detect() or detect_tiled() first.")

        export_shp(self._detections, path)

    def show_map(self, basemap: str = "SATELLITE", **kwargs: Any) -> Any:
        """Display detections on an interactive leafmap map.

        Args:
            basemap: Basemap style string.
            **kwargs: Additional arguments passed to the map widget.

        Returns:
            The map widget object.
        """
        if self._detections is None or len(self._detections) == 0:
            raise DetrGeoError("No detections to show. Run detect() or detect_tiled() first.")

        from detr_geo.viz import show_map

        return show_map(self._detections, basemap=basemap, **kwargs)

    def show_detections(self, figsize: tuple[int, int] = (12, 10), **kwargs: Any) -> Any:
        """Display detections using matplotlib.

        Args:
            figsize: Figure size as (width, height) tuple.
            **kwargs: Additional arguments passed to matplotlib.

        Returns:
            The matplotlib figure and axes.
        """
        if self._detections is None or len(self._detections) == 0:
            raise DetrGeoError("No detections to show. Run detect() or detect_tiled() first.")
        if self._source_path is None:
            raise DetrGeoError("No image available. show_detections() requires detect(), not detect_tiled().")

        # Lazy load the full image if not already loaded
        self._load_full_image()

        import numpy as np

        from detr_geo.io import normalize_to_float32
        from detr_geo.viz import show_detections

        normalized, _ = normalize_to_float32(self._image, stretch="percentile")
        image_hwc = np.transpose(normalized, (1, 2, 0))

        return show_detections(image_hwc, self._detections, figsize=figsize, **kwargs)
