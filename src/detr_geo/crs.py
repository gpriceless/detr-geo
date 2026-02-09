"""CRS handling and pixel-to-geographic coordinate transforms.

This module provides coordinate reference system utilities for converting
pixel-space bounding boxes from the detector into georeferenced geometries
using rasterio affine transforms and pyproj CRS objects.
"""

from __future__ import annotations

import math
from functools import lru_cache

from pyproj import CRS, Transformer
from rasterio.transform import Affine
from shapely.geometry import Polygon

from detr_geo._typing import PixelBBox
from detr_geo.exceptions import CRSError, MissingCRSError


@lru_cache(maxsize=32)
def get_transformer(src_crs: str, dst_crs: str) -> Transformer:
    """Get a cached pyproj Transformer for CRS conversion.

    Args:
        src_crs: Source CRS as EPSG string or WKT.
        dst_crs: Destination CRS as EPSG string or WKT.

    Returns:
        Cached Transformer instance with always_xy=True.

    Raises:
        CRSError: If either CRS string is invalid.
    """
    try:
        src = CRS.from_user_input(src_crs)
    except Exception as exc:
        raise CRSError(f"Invalid source CRS: '{src_crs}'. Error: {exc}") from exc

    try:
        dst = CRS.from_user_input(dst_crs)
    except Exception as exc:
        raise CRSError(f"Invalid destination CRS: '{dst_crs}'. Error: {exc}") from exc

    return Transformer.from_crs(src, dst, always_xy=True)


def pixel_to_geo(
    bbox: PixelBBox,
    transform: Affine,
) -> Polygon:
    """Convert a pixel-space bounding box to a CRS-space polygon.

    Always uses 4-corner conversion for correctness with rotated affine transforms.

    Args:
        bbox: (x_min, y_min, x_max, y_max) in pixel coordinates.
        transform: Rasterio affine transform of the raster.

    Returns:
        Shapely Polygon with 4 corners in CRS coordinate space.
    """
    x_min, y_min, x_max, y_max = bbox

    # Transform all 4 corners using the affine transform
    # rasterio transform: (col, row) -> (x, y) in CRS space
    tl = transform * (x_min, y_min)  # top-left
    tr = transform * (x_max, y_min)  # top-right
    br = transform * (x_max, y_max)  # bottom-right
    bl = transform * (x_min, y_max)  # bottom-left

    return Polygon([tl, tr, br, bl, tl])


def geo_to_pixel(
    point: tuple[float, float],
    transform: Affine,
) -> tuple[float, float]:
    """Convert a CRS-space coordinate to pixel-space (col, row).

    This is the inverse of the affine transform applied in pixel_to_geo().
    Uses the inverse affine (~transform) for correctness with rotated
    transforms.

    Args:
        point: (x, y) in CRS coordinate space.
        transform: Rasterio affine transform of the raster.

    Returns:
        (col, row) in pixel coordinates as floats.
    """
    inv = ~transform
    col, row = inv * point
    return (col, row)


def has_rotation(transform: Affine) -> bool:
    """Check if an affine transform includes rotation/shear terms.

    A standard north-up transform has b=0 and d=0 in the affine matrix:
    | a  b  c |
    | d  e  f |
    | 0  0  1 |

    Args:
        transform: Rasterio affine transform.

    Returns:
        True if the transform has non-zero rotation/shear terms.
    """
    return transform.b != 0.0 or transform.d != 0.0


def validate_crs(crs: CRS | str | None, georeferenced: bool = True) -> CRS | None:
    """Validate a CRS value.

    Args:
        crs: CRS to validate. Can be pyproj.CRS, EPSG string, or None.
        georeferenced: If True, None CRS raises MissingCRSError.

    Returns:
        Validated pyproj.CRS instance, or None in pixel-only mode.

    Raises:
        MissingCRSError: If crs is None and georeferenced is True.
        CRSError: If the CRS string is invalid.
    """
    if crs is None:
        if georeferenced:
            raise MissingCRSError(
                "Raster has no CRS but georeferenced=True was requested. "
                "Set georeferenced=False for pixel-only mode, or provide a CRS."
            )
        return None

    if isinstance(crs, CRS):
        return crs

    if isinstance(crs, str):
        try:
            return CRS.from_user_input(crs)
        except Exception as exc:
            raise CRSError(
                f"Invalid CRS string: '{crs}'. "
                f"Provide a valid EPSG code (e.g., 'EPSG:4326') or WKT string. "
                f"Error: {exc}"
            ) from exc

    raise CRSError(f"CRS must be a pyproj.CRS, EPSG string, or None, got {type(crs).__name__}")


def auto_utm_crs(longitude: float, latitude: float) -> CRS:
    """Auto-detect the UTM zone CRS for a given lon/lat point.

    Used for area calculations when source CRS is geographic.

    Args:
        longitude: Longitude in degrees.
        latitude: Latitude in degrees.

    Returns:
        pyproj.CRS for the appropriate UTM zone.
    """
    # UTM zone number
    zone_number = int(math.floor((longitude + 180) / 6)) + 1

    # EPSG code: 326xx for north, 327xx for south
    if latitude >= 0:
        epsg_code = 32600 + zone_number
    else:
        epsg_code = 32700 + zone_number

    return CRS.from_epsg(epsg_code)
