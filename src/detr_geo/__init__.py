"""detr_geo - Geospatial object detection using RF-DETR."""

from __future__ import annotations

__version__ = "0.1.0"

# DetrGeo import is deferred to avoid requiring rfdetr at import time.
# Users can still do `from detr_geo import DetrGeo`.
from detr_geo.core import DetrGeo
from detr_geo.exceptions import (
    BandError,
    CRSError,
    DetrGeoError,
    ExportError,
    GSDWarning,
    MissingCRSError,
    ModelError,
    TilingError,
)
from detr_geo.io import resolve_raster_source, stac_item_to_uri
from detr_geo.training import (
    AUGMENTATION_PRESETS,
    AugmentationPreset,
    SpatialSplitter,
    detections_to_coco,
    detections_to_yolo,
    prepare_training_dataset,
    train,
)

__all__ = [
    "__version__",
    "DetrGeo",
    "DetrGeoError",
    "CRSError",
    "MissingCRSError",
    "TilingError",
    "ModelError",
    "BandError",
    "ExportError",
    "GSDWarning",
    "resolve_raster_source",
    "stac_item_to_uri",
    "AUGMENTATION_PRESETS",
    "AugmentationPreset",
    "SpatialSplitter",
    "detections_to_coco",
    "detections_to_yolo",
    "prepare_training_dataset",
    "train",
]
