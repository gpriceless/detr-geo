"""GeoDataFrame export to GeoJSON, GeoPackage, and Shapefile formats.

This module handles constructing GeoDataFrames from detection results
and serializing them to standard geospatial vector formats.
"""

from __future__ import annotations

import warnings

import geopandas as gpd
import pandas as pd
from numpy.typing import NDArray
from pyproj import CRS
from rasterio.transform import Affine

from detr_geo.crs import auto_utm_crs, pixel_to_geo
from detr_geo.exceptions import ExportError

# ---------------------------------------------------------------------------
# GeoDataFrame Construction
# ---------------------------------------------------------------------------


def build_geodataframe(
    boxes: NDArray,
    scores: NDArray,
    class_ids: NDArray,
    class_names: dict[int, str] | None,
    transform: Affine,
    crs: CRS,
) -> gpd.GeoDataFrame:
    """Convert detection arrays into a georeferenced GeoDataFrame.

    Args:
        boxes: Array of shape (N, 4) in pixel coordinates [x1, y1, x2, y2].
        scores: Array of shape (N,) with confidence scores.
        class_ids: Array of shape (N,) with class IDs.
        class_names: Optional mapping of class_id to class_name.
        transform: Rasterio affine transform for pixel-to-geo conversion.
        crs: pyproj CRS for the output GeoDataFrame.

    Returns:
        GeoDataFrame with geometry, class_id, class_name, confidence,
        centroid_x, centroid_y columns.
    """
    if len(boxes) == 0:
        return gpd.GeoDataFrame(
            {
                "geometry": [],
                "class_id": pd.Series([], dtype="int64"),
                "class_name": pd.Series([], dtype="str"),
                "confidence": pd.Series([], dtype="float64"),
                "centroid_x": pd.Series([], dtype="float64"),
                "centroid_y": pd.Series([], dtype="float64"),
            },
            crs=crs,
        )

    geometries = []
    centroid_xs = []
    centroid_ys = []

    for box in boxes:
        bbox = (float(box[0]), float(box[1]), float(box[2]), float(box[3]))
        poly = pixel_to_geo(bbox, transform)
        geometries.append(poly)
        centroid = poly.centroid
        centroid_xs.append(centroid.x)
        centroid_ys.append(centroid.y)

    names = []
    if class_names is not None:
        for cid in class_ids:
            names.append(class_names.get(int(cid), f"class_{cid}"))
    else:
        names = [f"class_{cid}" for cid in class_ids]

    gdf = gpd.GeoDataFrame(
        {
            "geometry": geometries,
            "class_id": class_ids.astype(int),
            "class_name": names,
            "confidence": scores.astype(float),
            "centroid_x": centroid_xs,
            "centroid_y": centroid_ys,
        },
        crs=crs,
    )

    return gdf


def build_dataframe_pixel(
    boxes: NDArray,
    scores: NDArray,
    class_ids: NDArray,
    class_names: dict[int, str] | None = None,
) -> pd.DataFrame:
    """Build a plain DataFrame with pixel-space coordinates.

    Args:
        boxes: Array of shape (N, 4) in pixel coordinates [x1, y1, x2, y2].
        scores: Array of shape (N,) with confidence scores.
        class_ids: Array of shape (N,) with class IDs.
        class_names: Optional mapping of class_id to class_name.

    Returns:
        DataFrame with x1, y1, x2, y2, class_id, class_name, confidence columns.
    """
    if len(boxes) == 0:
        return pd.DataFrame(columns=["x1", "y1", "x2", "y2", "class_id", "class_name", "confidence"])

    names = []
    if class_names is not None:
        for cid in class_ids:
            names.append(class_names.get(int(cid), f"class_{cid}"))
    else:
        names = [f"class_{cid}" for cid in class_ids]

    return pd.DataFrame(
        {
            "x1": boxes[:, 0],
            "y1": boxes[:, 1],
            "x2": boxes[:, 2],
            "y2": boxes[:, 3],
            "class_id": class_ids.astype(int),
            "class_name": names,
            "confidence": scores.astype(float),
        }
    )


def compute_areas(
    gdf: gpd.GeoDataFrame,
    equal_area_crs: CRS | None = None,
) -> pd.Series:
    """Compute areas of detection polygons in square meters.

    If the GeoDataFrame CRS is geographic (degrees), auto-detects the
    appropriate UTM zone for accurate area computation.

    Args:
        gdf: GeoDataFrame with detection polygons.
        equal_area_crs: Optional explicit CRS for area computation.

    Returns:
        Series of float area values in square meters.
    """
    if len(gdf) == 0:
        return pd.Series([], dtype=float)

    if equal_area_crs is not None:
        projected = gdf.to_crs(equal_area_crs)
        return projected.geometry.area

    if gdf.crs is None:
        return gdf.geometry.area

    # Check if CRS is geographic (degrees)
    if gdf.crs.is_geographic:
        # Auto-detect UTM from centroid of first geometry
        centroid = gdf.geometry.iloc[0].centroid
        utm_crs = auto_utm_crs(centroid.x, centroid.y)
        projected = gdf.to_crs(utm_crs)
        return projected.geometry.area

    # Already projected (meters)
    return gdf.geometry.area


# ---------------------------------------------------------------------------
# Export Formats
# ---------------------------------------------------------------------------


def export_gpkg(
    gdf: gpd.GeoDataFrame,
    path: str,
    layer: str = "detections",
) -> None:
    """Export GeoDataFrame to GeoPackage format.

    Args:
        path: Output file path.
        layer: Layer name within the GeoPackage.

    Raises:
        ExportError: If export fails.
    """
    try:
        gdf.to_file(str(path), driver="GPKG", layer=layer)
    except Exception as exc:
        raise ExportError(f"Failed to export GeoPackage: {exc}") from exc


def export_geojson(
    gdf: gpd.GeoDataFrame,
    path: str,
    coordinate_precision: int = 6,
) -> None:
    """Export GeoDataFrame to GeoJSON format.

    GeoJSON requires WGS84 (EPSG:4326) coordinates. If the input CRS
    is different, it will be reprojected automatically.

    Args:
        path: Output file path.
        coordinate_precision: Decimal places for coordinates.

    Raises:
        ExportError: If export fails.
    """
    try:
        output_gdf = gdf

        # GeoJSON should be in EPSG:4326
        if gdf.crs is not None:
            target = CRS.from_epsg(4326)
            if not gdf.crs.equals(target):
                warnings.warn(
                    f"Reprojecting from {gdf.crs} to EPSG:4326 for GeoJSON export.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                output_gdf = gdf.to_crs(target)

        output_gdf.to_file(
            str(path),
            driver="GeoJSON",
            coordinate_precision=coordinate_precision,
        )
    except ExportError:
        raise
    except Exception as exc:
        raise ExportError(f"Failed to export GeoJSON: {exc}") from exc


def export_shp(
    gdf: gpd.GeoDataFrame,
    path: str,
) -> None:
    """Export GeoDataFrame to Shapefile format.

    Note: Shapefiles have legacy limitations:
    - Field names truncated to 10 characters
    - 2GB file size limit
    - No null value support

    Args:
        path: Output file path.

    Raises:
        ExportError: If export fails.
    """
    warnings.warn(
        "Shapefile format has legacy limitations: 10-char field names, 2GB size limit. Consider GeoPackage instead.",
        RuntimeWarning,
        stacklevel=2,
    )

    try:
        gdf.to_file(str(path), driver="ESRI Shapefile")
    except Exception as exc:
        raise ExportError(f"Failed to export Shapefile: {exc}") from exc
