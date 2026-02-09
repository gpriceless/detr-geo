"""Type aliases shared across the detr_geo library.

This module defines all shared type aliases used by multiple modules.
It uses ``from __future__ import annotations`` so that type aliases
are forward-compatible and don't require runtime evaluation.
"""

from __future__ import annotations

from typing import TypedDict

import numpy as np
from numpy.typing import NDArray

# Bounding box in pixel coordinates: (x_min, y_min, x_max, y_max)
PixelBBox = tuple[float, float, float, float]

# Bounding box in geographic CRS coordinates: (west, south, east, north)
GeoBBox = tuple[float, float, float, float]

# A rasterio-style window for tile reading: (col_off, row_off, width, height)
TileWindow = tuple[int, int, int, int]


class DetectionResult(TypedDict):
    """Normalized detection output from the adapter.

    This dict is the internal representation of detection results,
    decoupled from supervision.Detections to prevent external types
    from leaking into the geospatial processing pipeline.
    """

    bbox: list[list[float]]  # List of [x1, y1, x2, y2] in tile pixel coords
    confidence: list[float]  # List of confidence scores
    class_id: list[int]  # List of class IDs


class TileInfo(TypedDict):
    """Metadata for a single tile in the grid.

    Contains window coordinates for rasterio reads and metadata
    about the tile's position and data quality.
    """

    window: TileWindow
    global_offset_x: int
    global_offset_y: int
    nodata_fraction: float


# Image array type: (bands, height, width) float32 [0, 1]
ImageArray = NDArray[np.float32]

# Model size literal
ModelSize = str  # One of: "nano", "small", "medium", "base", "large"
