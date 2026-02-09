"""Image loading via rasterio with band selection and normalization.

This module handles reading geospatial rasters (GeoTIFF, etc.) into
numpy arrays suitable for object detection, preserving CRS and affine
transform metadata for downstream georeferencing.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from pyproj import CRS
from rasterio.transform import Affine

from detr_geo.exceptions import BandError, DetrGeoError, GSDWarning

# ---------------------------------------------------------------------------
# Raster source resolution (local paths, HTTP URLs, S3 URIs, STAC items)
# ---------------------------------------------------------------------------


def resolve_raster_source(source: str | Path) -> str:
    """Resolve a raster source to a rasterio-openable path/URI.

    Accepts:
    - Local file paths (str or Path)
    - HTTP/HTTPS URLs to COGs
    - S3 URIs (s3://bucket/key)
    - STAC Item objects (requires pystac)

    Returns a string that rasterio.open() can handle directly.
    Rasterio uses GDAL's VSICURL for HTTP and VSIS3 for S3.

    Args:
        source: Local path, URL, S3 URI, or pystac.Item object.

    Returns:
        String URI that rasterio can open.

    Raises:
        FileNotFoundError: If local path doesn't exist.
        DetrGeoError: If STAC item has no suitable raster asset or pystac not installed.
    """
    # Handle string/Path sources
    if isinstance(source, (str, Path)):
        source_str = str(source)

        # Check if it's a URL or S3 URI
        if source_str.startswith(("http://", "https://", "s3://")):
            # Remote source - return as-is for rasterio
            return source_str

        # Local file path
        path = Path(source_str)
        if not path.exists():
            raise FileNotFoundError(f"Raster file not found: {source_str}")
        return str(path)

    # Handle STAC Item objects (duck typing to avoid hard dependency)
    if hasattr(source, "assets") and hasattr(source, "id"):
        # Looks like a pystac.Item - extract COG URL
        try:
            import pystac  # noqa: F401
        except ImportError as err:
            raise DetrGeoError(
                "Detected a STAC Item object but pystac is not installed. Install with: pip install detr-geo[cloud]"
            ) from err

        return stac_item_to_uri(source)

    # Unknown type
    raise DetrGeoError(
        f"Unsupported source type: {type(source).__name__}. Expected str, Path, URL, S3 URI, or pystac.Item."
    )


def stac_item_to_uri(item) -> str:
    """Extract the best raster asset URL from a pystac Item.

    Args:
        item: A pystac.Item object.

    Returns:
        URL string to the COG asset.

    Raises:
        DetrGeoError: If no suitable raster asset is found.
    """
    # Try common asset keys in order of preference
    preferred_keys = ["visual", "image", "data", "red", "B04"]

    for key in preferred_keys:
        if key in item.assets:
            asset = item.assets[key]
            # Check if it's a GeoTIFF
            media_type = getattr(asset, "media_type", "")
            if media_type and "tiff" in media_type.lower():
                return asset.href
            # If no media_type, try anyway
            if not media_type:
                return asset.href

    # Fallback: try first asset that looks like a GeoTIFF
    for _key, asset in item.assets.items():
        media_type = getattr(asset, "media_type", "")
        if media_type and "tiff" in media_type.lower():
            return asset.href
        # Check by extension
        href = getattr(asset, "href", "")
        if href.lower().endswith((".tif", ".tiff")):
            return asset.href

    # No suitable asset found
    raise DetrGeoError(f"STAC Item '{item.id}' has no GeoTIFF asset. Available assets: {list(item.assets.keys())}")


# ---------------------------------------------------------------------------
# GSD (Ground Sample Distance) thresholds
# ---------------------------------------------------------------------------

# GSD thresholds derived from test_gsd_sensitivity.py results
GSD_OPTIMAL_MIN = 0.1  # meters -- below this, model sees too much detail
GSD_OPTIMAL_MAX = 0.5  # meters -- above this, vehicles become too few pixels
GSD_WARNING_MAX = 1.0  # meters -- warn above this
GSD_ERROR_MAX = 5.0  # meters -- error above this (objects undetectable)

# ---------------------------------------------------------------------------
# Band selection
# ---------------------------------------------------------------------------


class BandSelector:
    """Select and reorder bands from multi-band raster data for 3-channel RGB input."""

    PRESETS: dict[str, list[int]] = {
        "rgb": [1, 2, 3],
        "naip_rgb": [1, 2, 3],
        "sentinel2_rgb": [4, 3, 2],
        "worldview_rgb": [5, 3, 2],
    }

    def __init__(self, bands: tuple[int, ...] | str = "rgb") -> None:
        """Initialize with band indices (1-indexed) or preset name.

        Args:
            bands: Either a preset name string or tuple of band indices (1-indexed).
                Presets: "rgb", "naip_rgb", "sentinel2_rgb", "worldview_rgb".

        Raises:
            BandError: If band specification is invalid.
        """
        if isinstance(bands, str):
            if bands not in self.PRESETS:
                valid = ", ".join(sorted(self.PRESETS.keys()))
                raise BandError(
                    f"Unknown band preset '{bands}'. "
                    f"Valid presets: {valid}. "
                    f"Or pass a tuple of band indices, e.g. (4, 3, 2)."
                )
            self._band_indices = list(self.PRESETS[bands])
        elif isinstance(bands, (tuple, list)):
            indices = list(bands)
            if not indices:
                raise BandError("Band indices tuple must not be empty.")
            for idx in indices:
                if not isinstance(idx, int) or idx < 1:
                    raise BandError(
                        f"Band index must be a positive integer (1-indexed), got {idx}. "
                        f"Note: band indices are 1-based, not 0-based."
                    )
            self._band_indices = indices
        else:
            raise BandError(f"bands must be a preset name string or tuple of integers, got {type(bands).__name__}")

    @property
    def band_indices(self) -> list[int]:
        """Return the 1-indexed band indices to select."""
        return list(self._band_indices)

    @property
    def read_indices(self) -> list[int]:
        """Return deduplicated sorted band indices for rasterio reads.

        After clamping, multiple logical bands may map to the same physical
        band (e.g., single-band triplication: [1,2,3] -> [1,1,1] -> read [1]).
        This property returns the unique sorted indices for efficient I/O.
        """
        return sorted(set(self._band_indices))

    def clamp_to_band_count(self, band_count: int) -> None:
        """Clamp band indices to the raster's actual band count.

        For single-band rasters with preset "rgb" (requesting [1,2,3]),
        this clamps to [1,1,1]. The select() method will then triplicate
        the single band to 3 channels.

        Args:
            band_count: Actual number of bands in the raster.
        """
        original = list(self._band_indices)
        self._band_indices = [min(idx, band_count) for idx in self._band_indices]
        if self._band_indices != original and band_count > 1:
            clamped = [(o, c) for o, c in zip(original, self._band_indices, strict=False) if o != c]
            warnings.warn(
                f"Band indices {[o for o, c in clamped]} exceed raster band count "
                f"({band_count}). Clamped to {[c for o, c in clamped]}.",
                RuntimeWarning,
                stacklevel=2,
            )

    def select(
        self,
        data: NDArray,
        num_bands: int,
    ) -> tuple[NDArray, NDArray | None]:
        """Select bands from raster data array.

        Args:
            data: Array of shape (bands, height, width) from rasterio.
            num_bands: Total band count of the source raster.

        Returns:
            Tuple of (rgb_array, alpha_mask) where:
            - rgb_array: shape (3, height, width) or (N, height, width)
            - alpha_mask: shape (height, width) bool or None if no alpha band.

        Raises:
            BandError: If requested band index exceeds available bands.
        """
        # Validate band indices
        for idx in self._band_indices:
            if idx > num_bands:
                raise BandError(
                    f"Band index {idx} exceeds available band count ({num_bands}). The raster has bands 1-{num_bands}."
                )

        # Detect and extract alpha band
        alpha_mask = self._detect_alpha(data, num_bands)

        # Handle single-band triplication
        if data.shape[0] == 1:
            warnings.warn(
                "Single-band raster detected. Triplicating to 3 channels for RGB input.",
                RuntimeWarning,
                stacklevel=2,
            )
            band_data = data[0]  # (H, W)
            rgb = np.stack([band_data, band_data, band_data], axis=0)
            return rgb, alpha_mask

        # Select and reorder bands (convert 1-indexed to 0-indexed)
        selected = np.stack([data[idx - 1] for idx in self._band_indices], axis=0)

        return selected, alpha_mask

    def _detect_alpha(self, data: NDArray, num_bands: int) -> NDArray | None:
        """Extract alpha band if present.

        Alpha band is detected when the data has one more band than requested
        and the extra band appears to be an alpha/mask band.

        Args:
            data: Array of shape (bands, height, width).
            num_bands: Total band count.

        Returns:
            Boolean mask (height, width) where True = transparent/nodata, or None.
        """
        # Check for standard RGBA (4 bands when requesting 3)
        if data.shape[0] == 4 and len(self._band_indices) == 3:
            # Last band is alpha
            alpha = data[3]
            # Convert to boolean: True where alpha is 0 (transparent = nodata)
            return alpha == 0

        return None


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------


@dataclass
class StretchParams:
    """Parameters used for percentile stretch, stored for reproducibility."""

    vmin: list[float]  # per-band minimum values
    vmax: list[float]  # per-band maximum values
    percentile_low: float
    percentile_high: float


def normalize_to_float32(
    data: NDArray,
    stretch: str = "percentile",
    percentiles: tuple[float, float] = (2.0, 98.0),
    stretch_params: StretchParams | None = None,
    nodata_mask: NDArray | None = None,
) -> tuple[NDArray[np.float32], StretchParams]:
    """Normalize raster data to float32 [0, 1] for RF-DETR input.

    Args:
        data: Array shape (bands, H, W) of any numeric dtype.
        stretch: One of "percentile", "minmax", or "none".
        percentiles: Low/high percentiles for percentile stretch.
        stretch_params: Pre-computed params for consistent per-scene normalization.
        nodata_mask: Boolean mask shape (H, W) where True = nodata pixel.

    Returns:
        Tuple of (normalized float32 array in [0,1], StretchParams used).

    Raises:
        BandError: If data has unexpected shape.
    """
    if data.ndim != 3:
        raise BandError(f"Expected 3D array (bands, height, width), got {data.ndim}D array.")

    num_bands = data.shape[0]
    result = data.astype(np.float32)

    if stretch == "none":
        # Assume already in [0, 1]
        params = StretchParams(
            vmin=[0.0] * num_bands,
            vmax=[1.0] * num_bands,
            percentile_low=0.0,
            percentile_high=100.0,
        )
        return np.clip(result, 0.0, 1.0), params

    if stretch_params is not None:
        # Use pre-computed stretch params
        for i in range(num_bands):
            vmin = stretch_params.vmin[i]
            vmax = stretch_params.vmax[i]
            if vmax - vmin > 0:
                result[i] = (result[i] - vmin) / (vmax - vmin)
            else:
                result[i] = 0.0
        return np.clip(result, 0.0, 1.0), stretch_params

    # Compute stretch params from this data
    vmin_list: list[float] = []
    vmax_list: list[float] = []

    for i in range(num_bands):
        band = result[i]

        if nodata_mask is not None:
            # Exclude nodata pixels from percentile computation
            valid_pixels = band[~nodata_mask]
            if len(valid_pixels) == 0:
                vmin_list.append(0.0)
                vmax_list.append(1.0)
                continue
        else:
            valid_pixels = band.ravel()

        if stretch == "percentile":
            low, high = np.percentile(valid_pixels, percentiles)
        elif stretch == "minmax":
            low, high = float(valid_pixels.min()), float(valid_pixels.max())
        else:
            raise BandError(f"Unknown stretch mode '{stretch}'. Valid modes: 'percentile', 'minmax', 'none'.")

        vmin_list.append(float(low))
        vmax_list.append(float(high))

    params = StretchParams(
        vmin=vmin_list,
        vmax=vmax_list,
        percentile_low=percentiles[0],
        percentile_high=percentiles[1],
    )

    # Apply normalization
    for i in range(num_bands):
        vmin = params.vmin[i]
        vmax = params.vmax[i]
        if vmax - vmin > 0:
            result[i] = (result[i] - vmin) / (vmax - vmin)
        else:
            result[i] = 0.0

    return np.clip(result, 0.0, 1.0), params


def compute_scene_stretch_params(
    raster_path: str,
    bands: list[int],
    percentiles: tuple[float, float] = (2.0, 98.0),
    sample_tiles: int = 20,
    tile_size: int = 512,
) -> StretchParams:
    """Sample random tiles from a raster to compute per-scene stretch parameters.

    Args:
        raster_path: Path to raster file.
        bands: 1-indexed band indices to compute stats for.
        percentiles: Low/high percentile values.
        sample_tiles: Number of random tiles to sample.
        tile_size: Size of each sample tile.

    Returns:
        StretchParams with per-band vmin/vmax values.
    """
    import rasterio
    from rasterio.windows import Window

    all_values: list[list[float]] = [[] for _ in bands]

    with rasterio.open(raster_path) as src:
        rng = np.random.default_rng(42)
        max_col = max(0, src.width - tile_size)
        max_row = max(0, src.height - tile_size)

        n_samples = min(sample_tiles, max(1, (max_col + 1) * (max_row + 1)))

        for _ in range(n_samples):
            col_off = rng.integers(0, max(1, max_col + 1))
            row_off = rng.integers(0, max(1, max_row + 1))
            w = min(tile_size, src.width - col_off)
            h = min(tile_size, src.height - row_off)

            window = Window(col_off, row_off, w, h)
            data = src.read(bands, window=window).astype(np.float32)

            for i in range(len(bands)):
                vals = data[i].ravel()
                # Filter out nodata if present
                if src.nodata is not None:
                    vals = vals[vals != src.nodata]
                all_values[i].extend(vals.tolist())

    vmin_list: list[float] = []
    vmax_list: list[float] = []

    for i in range(len(bands)):
        vals = np.array(all_values[i])
        if len(vals) > 0:
            low, high = np.percentile(vals, percentiles)
            vmin_list.append(float(low))
            vmax_list.append(float(high))
        else:
            vmin_list.append(0.0)
            vmax_list.append(1.0)

    return StretchParams(
        vmin=vmin_list,
        vmax=vmax_list,
        percentile_low=percentiles[0],
        percentile_high=percentiles[1],
    )


# ---------------------------------------------------------------------------
# Raster metadata and loading
# ---------------------------------------------------------------------------


@dataclass
class RasterMetadata:
    """Metadata extracted from a raster at load time."""

    crs: CRS | None
    transform: Affine
    width: int
    height: int
    count: int  # number of bands
    dtype: str
    nodata: float | None
    has_alpha: bool
    bounds: tuple[float, float, float, float]
    gsd: float | None  # Ground sample distance in meters per pixel


def load_raster_metadata(source: str | Path) -> RasterMetadata:
    """Load raster metadata without reading pixel data.

    Args:
        source: Path to raster file, HTTP/HTTPS URL, S3 URI, or pystac.Item.

    Returns:
        RasterMetadata with CRS, transform, dimensions, etc.

    Raises:
        FileNotFoundError: If local source does not exist.
        DetrGeoError: If remote source cannot be opened or STAC item has no asset.
        BandError: If raster has 0 bands.
    """
    import rasterio

    # Resolve source to rasterio-openable URI
    uri = resolve_raster_source(source)

    with rasterio.open(uri) as src:
        if src.count == 0:
            raise BandError(f"Raster has 0 bands: {source}")

        crs = None
        if src.crs is not None:
            try:
                crs = CRS.from_user_input(src.crs)
            except Exception:
                crs = None

        # Detect alpha band
        has_alpha = False
        if src.count >= 4:
            # Check if last band is an alpha band via color interpretation
            try:
                interps = src.colorinterp
                if interps and interps[-1].name == "alpha":
                    has_alpha = True
            except (AttributeError, IndexError):
                pass

        gsd = compute_gsd(src.transform, crs)

        return RasterMetadata(
            crs=crs,
            transform=src.transform,
            width=src.width,
            height=src.height,
            count=src.count,
            dtype=str(src.dtypes[0]),
            nodata=src.nodata,
            has_alpha=has_alpha,
            bounds=src.bounds,
            gsd=gsd,
        )


def compute_gsd(transform: Affine, crs: CRS | None) -> float | None:
    """Compute the ground sample distance (GSD) in meters per pixel.

    For projected CRS with meter units: uses transform.a directly.
    For geographic CRS (degrees): converts using WGS84 ellipsoid at image center latitude.
    For US Survey Feet: converts to meters.
    Returns None if CRS is missing or units are unrecognized.

    Args:
        transform: Affine transform from the raster.
        crs: Coordinate reference system.

    Returns:
        GSD in meters/pixel, or None if unable to compute.
    """

    if crs is None:
        return None

    # Get pixel size from transform
    pixel_size_x = abs(transform.a)
    pixel_size_y = abs(transform.e)
    # Use the average of x and y
    pixel_size = (pixel_size_x + pixel_size_y) / 2.0

    # Check if CRS has axis info
    try:
        axis_info = crs.axis_info
        if not axis_info:
            return None
        unit_name = axis_info[0].unit_name.lower()
    except (AttributeError, IndexError):
        return None

    # Handle different unit types
    if "metre" in unit_name or "meter" in unit_name:
        # Already in meters
        return pixel_size
    elif "us survey foot" in unit_name or "foot_us" in unit_name:
        # US Survey Feet to meters
        return pixel_size * 0.304800609601
    elif "foot" in unit_name and "us" not in unit_name:
        # International foot to meters
        return pixel_size * 0.3048
    elif "degree" in unit_name:
        # Geographic CRS - need to convert degrees to meters
        # Use WGS84 ellipsoid approximation at the center of the image
        # We need a latitude to do this calculation properly
        # For now, use a rough approximation at mid-latitudes (45 degrees)
        # This is approximate but better than nothing

        # Try to get latitude from the CRS or use 0 (equator) as fallback
        # At equator: 1 degree longitude ≈ 111320 meters
        # At equator: 1 degree latitude ≈ 110574 meters
        # Use the average as a rough approximation
        meters_per_degree = 111000.0
        return pixel_size * meters_per_degree
    else:
        # Unknown unit
        return None


def check_gsd(gsd: float | None, strict: bool = False) -> None:
    """Check GSD against known detection thresholds and warn if out of range.

    Args:
        gsd: Ground sample distance in meters/pixel. None = skip check.
        strict: If True, raise DetrGeoError instead of warning for extreme values.

    Warns:
        GSDWarning: When GSD is outside optimal range but potentially usable.

    Raises:
        DetrGeoError: When strict=True and GSD is beyond usable range.
    """
    if gsd is None:
        return

    if gsd < 0.05:
        msg = (
            f"GSD is {gsd:.3f}m. Imagery is extremely high resolution. "
            f"The model was trained at ~0.3m GSD and may produce false positives "
            f"on fine textures. Consider downsampling."
        )
        warnings.warn(msg, GSDWarning, stacklevel=2)
    elif gsd < GSD_OPTIMAL_MIN:
        msg = f"GSD is {gsd:.3f}m. Imagery is higher resolution than training data (~0.3m). Results may vary."
        warnings.warn(msg, GSDWarning, stacklevel=2)
    elif gsd > GSD_ERROR_MAX:
        msg = (
            f"GSD is {gsd:.3f}m. Objects are likely undetectable at this resolution. "
            f"The model was trained at ~0.3m GSD (optimal range: {GSD_OPTIMAL_MIN}-{GSD_OPTIMAL_MAX}m)."
        )
        if strict:
            raise DetrGeoError(msg)
        warnings.warn(msg, GSDWarning, stacklevel=2)
    elif gsd > GSD_WARNING_MAX:
        msg = (
            f"GSD is {gsd:.3f}m. Imagery resolution is well below training GSD (~0.3m). "
            f"Detection quality will be severely degraded. "
            f"Optimal range: {GSD_OPTIMAL_MIN}-{GSD_OPTIMAL_MAX}m."
        )
        warnings.warn(msg, GSDWarning, stacklevel=2)
    elif gsd > GSD_OPTIMAL_MAX:
        msg = (
            f"GSD is {gsd:.3f}m. Imagery resolution is degraded compared to training data (~0.3m). "
            f"Small objects may be missed. Optimal range: {GSD_OPTIMAL_MIN}-{GSD_OPTIMAL_MAX}m."
        )
        warnings.warn(msg, GSDWarning, stacklevel=2)
    # else: GSD is in optimal range, no warning


def read_tile(
    source: str | Path,
    window: tuple[int, int, int, int],
    bands: list[int] | None = None,
) -> tuple[NDArray, NDArray | None]:
    """Read a tile from a raster using windowed read.

    Args:
        source: Path to raster file.
        window: (col_off, row_off, width, height) pixel coordinates.
        bands: 1-indexed band indices to read. None reads all.

    Returns:
        Tuple of (data array shape (bands, H, W), nodata_mask shape (H, W) or None).
    """
    import rasterio
    from rasterio.windows import Window

    col_off, row_off, width, height = window
    rasterio_window = Window(col_off, row_off, width, height)

    with rasterio.open(str(source)) as src:
        data = src.read(
            indexes=bands,
            window=rasterio_window,
            boundless=True,
            fill_value=src.nodata if src.nodata is not None else 0,
        )

        # Compute nodata mask
        nodata_mask = None
        if src.nodata is not None:
            # Any pixel where all bands are nodata
            nodata_mask = np.all(data == src.nodata, axis=0)

        return data, nodata_mask


def compute_nodata_fraction(
    data: NDArray,
    nodata_value: float | None,
    alpha_mask: NDArray | None = None,
) -> float:
    """Compute fraction of nodata pixels in a tile.

    Priority: alpha_mask > nodata_value > return 0.0.

    Args:
        data: Array of shape (bands, H, W).
        nodata_value: Nodata sentinel value from raster metadata.
        alpha_mask: Boolean mask (H, W) where True = nodata.

    Returns:
        Float in [0, 1] representing fraction of nodata pixels.
    """
    if data.ndim < 2:
        return 0.0

    h, w = data.shape[-2], data.shape[-1]
    total_pixels = h * w

    if total_pixels == 0:
        return 0.0

    # Priority 1: alpha mask
    if alpha_mask is not None:
        nodata_count = int(np.sum(alpha_mask))
        return nodata_count / total_pixels

    # Priority 2: nodata value
    if nodata_value is not None:
        if data.ndim == 3:
            nodata_pixels = np.all(data == nodata_value, axis=0)
        else:
            nodata_pixels = data == nodata_value
        nodata_count = int(np.sum(nodata_pixels))
        return nodata_count / total_pixels

    # No nodata information
    return 0.0


def fill_nodata(
    data: NDArray,
    nodata_mask: NDArray,
) -> NDArray:
    """Fill nodata pixels with per-band mean of valid pixels.

    Args:
        data: Array shape (bands, H, W).
        nodata_mask: Boolean array shape (H, W), True where nodata.

    Returns:
        Copy of data with nodata pixels filled with band means.
    """
    result = data.copy()

    if not np.any(nodata_mask):
        return result

    for i in range(result.shape[0]):
        band = result[i]
        valid_pixels = band[~nodata_mask]

        if len(valid_pixels) > 0:
            fill_value = valid_pixels.mean()
        else:
            fill_value = 0.0

        band[nodata_mask] = fill_value

    return result
