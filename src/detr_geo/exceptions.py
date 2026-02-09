"""Exception hierarchy for detr_geo.

All custom exceptions inherit from DetrGeoError to enable catch-all
error handling. Each exception type maps to a specific failure domain
and carries actionable error messages.
"""

from __future__ import annotations


class DetrGeoError(Exception):
    """Base exception for all detr_geo errors.

    Catching this exception will catch any error raised by the detr_geo
    library, providing a convenient catch-all for library consumers.
    """

    def __init__(self, message: str = "", **kwargs: object) -> None:
        self.context = kwargs
        super().__init__(message)


class CRSError(DetrGeoError):
    """Raised for coordinate reference system issues.

    This covers invalid CRS strings, incompatible CRS transformations,
    and other projection-related failures.
    """

    pass


class MissingCRSError(CRSError):
    """Raised when a raster has no CRS in georeferenced mode.

    This is a specific sub-case of CRSError for when a raster file
    lacks embedded CRS metadata but georeferenced output is requested.
    """

    pass


class TilingError(DetrGeoError):
    """Raised for tile generation or processing failures.

    This covers invalid tile sizes, overlap parameters, and failures
    during the tiled detection pipeline.
    """

    pass


class ModelError(DetrGeoError):
    """Raised for inference failures or model configuration issues.

    This covers unsupported model sizes, missing rfdetr installation,
    model loading failures, and prediction errors.
    """

    pass


class BandError(DetrGeoError):
    """Raised for band selection or normalization issues.

    This covers invalid band indices, unsupported band counts,
    and image normalization failures.
    """

    pass


class ExportError(DetrGeoError):
    """Raised for output format or file writing issues.

    This covers unsupported export formats, file permission errors,
    and serialization failures.
    """

    pass


class GSDWarning(UserWarning):
    """Warning issued when imagery GSD is outside the optimal detection range.

    The model is trained on imagery with ~0.3m ground sample distance (GSD).
    This warning is issued when loaded imagery has a GSD significantly
    different from the training data, which may affect detection quality.
    """

    pass
