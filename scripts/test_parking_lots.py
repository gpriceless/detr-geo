#!/usr/bin/env python3
"""Real-world integration test: detect vehicles in California parking lots.

Supports two detection modes:
  - "coco": COCO-pretrained weights (ground-level classes, broad vehicle search)
  - "vme":  VME fine-tuned weights (overhead satellite, Car/Bus/Truck only)

Downloads high-resolution aerial imagery from Esri World Imagery for two
large parking lots in California, runs RF-DETR object detection through the
full detr_geo pipeline, and exports georeferenced results as GeoJSON.

Sites:
  1. Costco Wholesale, Sunnyvale, CA  (37.3723, -121.9960)
     Large big-box store lot visible in aerial imagery, typically full.
  2. Great Mall, Milpitas, CA  (37.4163, -121.9000)
     Large open-air parking lot with dense car coverage.

Both locations are in the greater California area and have clear overhead
views at ~0.3 m GSD from Esri World Imagery tiles (zoom 19).

Usage:
    # VME fine-tuned model (auto-selects best EMA checkpoint)
    python scripts/test_parking_lots.py --mode vme --visualize --tiled --skip-download

    # COCO pretrained (original behaviour)
    python scripts/test_parking_lots.py --mode coco --visualize --tiled --skip-download --model medium

    # Custom weights path
    python scripts/test_parking_lots.py --mode vme --weights /path/to/checkpoint.pth

Requirements:
    pip install detr-geo[all]
    (includes rfdetr, torch, rasterio, matplotlib)
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
import traceback
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Site definitions
# ---------------------------------------------------------------------------

@dataclass
class ParkingLotSite:
    """A parking lot test site with geographic bounds."""

    name: str
    description: str
    city: str
    # Bounding box in WGS84: (west, south, east, north)
    bbox: tuple[float, float, float, float]
    # Expected zoom level for Esri tile download (higher = more detail)
    zoom: int

    @property
    def center_lon(self) -> float:
        return (self.bbox[0] + self.bbox[2]) / 2

    @property
    def center_lat(self) -> float:
        return (self.bbox[1] + self.bbox[3]) / 2


# Two California parking lots chosen for:
# - Large area with many visible cars
# - Clear overhead view (no tree canopy)
# - High-resolution Esri imagery available
SITES = [
    ParkingLotSite(
        name="costco_sunnyvale",
        description="Costco Wholesale parking lot, Sunnyvale CA",
        city="Sunnyvale",
        # ~300m x 200m area covering the main parking lot
        bbox=(-121.9985, 37.3705, -121.9935, 37.3740),
        zoom=19,
    ),
    ParkingLotSite(
        name="great_mall_milpitas",
        description="Great Mall parking lot, Milpitas CA",
        city="Milpitas",
        # ~400m x 300m area covering the large open-air parking lot
        bbox=(-121.9025, 37.4145, -121.8975, 37.4180),
        zoom=19,
    ),
]


# ---------------------------------------------------------------------------
# Mode-specific configuration
# ---------------------------------------------------------------------------

# COCO vehicle classes -- from overhead/aerial perspective, the COCO-pretrained
# model often classifies cars as "motorcycle", "bicycle", "truck", or "boat"
# because it was trained on ground-level images.
COCO_VEHICLE_CLASSES = ["car", "motorcycle", "bicycle", "truck", "bus", "boat"]

# VME classes -- the fine-tuned model was trained on exactly these 3 classes
VME_CLASSES = ["Car", "Bus", "Truck"]

# VME class ID mapping -- RF-DETR fine-tuned checkpoint outputs integer class
# IDs that need explicit remapping (otherwise they get looked up in the COCO
# label table and produce wrong names like "person", "bicycle").
VME_CLASS_NAMES: dict[int, str] = {0: "Car", 1: "Bus", 2: "Truck"}

# Default VME checkpoint (best EMA model)
VME_DEFAULT_CHECKPOINT = "checkpoint_best_ema.pth"

# xView classes -- the fine-tuned model was trained on 5 vehicle classes
XVIEW_CLASSES = ["Car", "Pickup", "Truck", "Bus", "Other"]

# xView class ID mapping
XVIEW_CLASS_NAMES: dict[int, str] = {0: "Car", 1: "Pickup", 2: "Truck", 3: "Bus", 4: "Other"}

# Default xView checkpoint (best EMA model)
XVIEW_DEFAULT_CHECKPOINT = "checkpoint_best_ema.pth"


# ---------------------------------------------------------------------------
# Metrics data structures
# ---------------------------------------------------------------------------

@dataclass
class TileMetrics:
    """Metrics collected per tile during inference."""
    tile_index: int
    window: tuple[int, int, int, int]
    detection_count: int
    inference_time_s: float
    skipped: bool = False
    error: str | None = None


@dataclass
class DetectionMetrics:
    """Comprehensive metrics for a detection run."""
    site_name: str
    mode: str
    model_size: str
    confidence_threshold: float
    weights_path: str | None

    # Timing
    download_time_s: float = 0.0
    detection_time_s: float = 0.0
    export_time_s: float = 0.0

    # Image metadata
    image_width_px: int = 0
    image_height_px: int = 0
    image_crs: str = ""
    gsd_x_m: float = 0.0
    gsd_y_m: float = 0.0
    area_hectares: float = 0.0

    # Detection counts
    total_detections: int = 0
    class_counts: dict[str, int] = field(default_factory=dict)

    # Confidence stats
    conf_min: float = 0.0
    conf_max: float = 0.0
    conf_mean: float = 0.0
    conf_median: float = 0.0
    conf_std: float = 0.0

    # Bbox size stats (in pixels)
    bbox_area_min_px: float = 0.0
    bbox_area_max_px: float = 0.0
    bbox_area_mean_px: float = 0.0
    bbox_area_min_m2: float = 0.0
    bbox_area_max_m2: float = 0.0
    bbox_area_mean_m2: float = 0.0

    # Tile-level stats
    total_tiles: int = 0
    processed_tiles: int = 0
    empty_tiles: int = 0
    failed_tiles: int = 0
    detections_per_tile_min: int = 0
    detections_per_tile_max: int = 0
    detections_per_tile_mean: float = 0.0

    # Per-tile timing
    tile_inference_min_s: float = 0.0
    tile_inference_max_s: float = 0.0
    tile_inference_mean_s: float = 0.0
    tiles_per_second: float = 0.0

    # Spatial stats
    detection_density_per_hectare: float = 0.0
    spatial_dispersion_index: float = 0.0  # 0 = perfectly clustered, 1 = uniform

    # Anomalies
    anomalies: list[str] = field(default_factory=list)

    # Memory
    gpu_memory_peak_mb: float = 0.0
    ram_usage_mb: float = 0.0

    # Geospatial checks
    geo_checks: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Imagery download
# ---------------------------------------------------------------------------

def _lat_lon_to_tile(lat: float, lon: float, zoom: int) -> tuple[int, int]:
    """Convert lat/lon (EPSG:4326) to slippy map tile indices."""
    n = 2 ** zoom
    x = int((lon + 180.0) / 360.0 * n)
    lat_rad = math.radians(lat)
    y = int((1.0 - math.log(math.tan(lat_rad) + 1.0 / math.cos(lat_rad)) / math.pi) / 2.0 * n)
    x = max(0, min(n - 1, x))
    y = max(0, min(n - 1, y))
    return x, y


def _tile_bounds(tx: int, ty: int, zoom: int) -> tuple[float, float, float, float]:
    """Return WGS84 bounding box (west, south, east, north) for a tile."""
    n = 2 ** zoom
    west = tx / n * 360.0 - 180.0
    east = (tx + 1) / n * 360.0 - 180.0
    north_rad = math.atan(math.sinh(math.pi * (1 - 2 * ty / n)))
    south_rad = math.atan(math.sinh(math.pi * (1 - 2 * (ty + 1) / n)))
    north = math.degrees(north_rad)
    south = math.degrees(south_rad)
    return west, south, east, north


def download_imagery(site: ParkingLotSite, output_path: str) -> float:
    """Download aerial imagery for a site as a GeoTIFF.

    Returns:
        Download time in seconds.
    """
    import rasterio
    from rasterio.transform import from_bounds
    import requests
    from PIL import Image

    t0 = time.time()
    print(f"  Downloading Esri World Imagery for {site.description}...")
    print(f"  Bounding box: {site.bbox}")
    print(f"  Zoom level: {site.zoom}")

    west, south, east, north = site.bbox
    zoom = site.zoom

    TILE_SIZE = 256
    ESRI_URL = (
        "https://server.arcgisonline.com/ArcGIS/rest/services/"
        "World_Imagery/MapServer/tile/{z}/{y}/{x}"
    )

    tx_min, ty_min = _lat_lon_to_tile(north, west, zoom)
    tx_max, ty_max = _lat_lon_to_tile(south, east, zoom)

    if tx_min > tx_max:
        tx_min, tx_max = tx_max, tx_min
    if ty_min > ty_max:
        ty_min, ty_max = ty_max, ty_min

    num_tiles_x = tx_max - tx_min + 1
    num_tiles_y = ty_max - ty_min + 1
    total_tiles = num_tiles_x * num_tiles_y

    print(f"  Tiles: {num_tiles_x} x {num_tiles_y} = {total_tiles} tiles")

    mosaic_width = num_tiles_x * TILE_SIZE
    mosaic_height = num_tiles_y * TILE_SIZE
    mosaic = Image.new("RGB", (mosaic_width, mosaic_height))

    session = requests.Session()
    session.headers.update({
        "User-Agent": "detr_geo/0.1 (geospatial object detection)",
        "Referer": "https://github.com/detr-geo",
    })

    downloaded = 0
    for ty in range(ty_min, ty_max + 1):
        for tx in range(tx_min, tx_max + 1):
            url = ESRI_URL.format(z=zoom, y=ty, x=tx)
            try:
                resp = session.get(url, timeout=30)
                resp.raise_for_status()
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to download tile z={zoom}/y={ty}/x={tx}.\n"
                    f"URL: {url}\nError: {exc}"
                ) from exc

            tile_img = Image.open(__import__("io").BytesIO(resp.content)).convert("RGB")
            px_x = (tx - tx_min) * TILE_SIZE
            px_y = (ty - ty_min) * TILE_SIZE
            mosaic.paste(tile_img, (px_x, px_y))

            downloaded += 1
            if downloaded % 10 == 0 or downloaded == total_tiles:
                print(f"    Downloaded {downloaded}/{total_tiles} tiles")

    mosaic_west, _, _, mosaic_north = _tile_bounds(tx_min, ty_min, zoom)
    _, mosaic_south, mosaic_east, _ = _tile_bounds(tx_max, ty_max, zoom)

    def lon_to_px(lon: float) -> int:
        frac = (lon - mosaic_west) / (mosaic_east - mosaic_west)
        return int(round(frac * mosaic_width))

    def lat_to_px(lat: float) -> int:
        frac = (mosaic_north - lat) / (mosaic_north - mosaic_south)
        return int(round(frac * mosaic_height))

    crop_left = max(0, lon_to_px(west))
    crop_right = min(mosaic_width, lon_to_px(east))
    crop_top = max(0, lat_to_px(north))
    crop_bottom = min(mosaic_height, lat_to_px(south))

    cropped = mosaic.crop((crop_left, crop_top, crop_right, crop_bottom))
    crop_w, crop_h = cropped.size

    print(f"  Mosaic size: {mosaic_width}x{mosaic_height} -> cropped to {crop_w}x{crop_h}")

    img_array = np.array(cropped)  # (H, W, 3)
    img_bands = np.transpose(img_array, (2, 0, 1))  # (3, H, W)

    transform = from_bounds(west, south, east, north, crop_w, crop_h)

    try:
        with rasterio.open(
            output_path, "w", driver="GTiff",
            height=crop_h, width=crop_w, count=3,
            dtype="uint8", crs="EPSG:4326", transform=transform,
            compress="deflate",
        ) as dst:
            dst.write(img_bands)
    except Exception as exc:
        raise RuntimeError(f"Failed to write GeoTIFF: {output_path}\nError: {exc}") from exc

    if not Path(output_path).exists():
        raise RuntimeError(f"Imagery download completed but output file not found: {output_path}")

    file_size_mb = Path(output_path).stat().st_size / (1024 * 1024)
    elapsed = time.time() - t0
    print(f"  Saved: {output_path} ({file_size_mb:.1f} MB) in {elapsed:.1f}s")

    return elapsed


# ---------------------------------------------------------------------------
# GSD and area computation
# ---------------------------------------------------------------------------

def compute_gsd(site: ParkingLotSite, width_px: int, height_px: int) -> tuple[float, float]:
    """Compute ground sample distance (GSD) in meters for a site.

    Uses the WGS84 ellipsoid to account for latitude-dependent scaling.

    Returns:
        (gsd_x_meters, gsd_y_meters) per pixel.
    """
    west, south, east, north = site.bbox
    center_lat = (south + north) / 2.0

    # WGS84 ellipsoid
    a = 6378137.0  # semi-major axis (m)
    e2 = 0.00669437999014  # eccentricity squared

    lat_rad = math.radians(center_lat)
    sin_lat = math.sin(lat_rad)

    # Radius of curvature in the prime vertical (N) and meridian (M)
    N = a / math.sqrt(1 - e2 * sin_lat ** 2)
    M = a * (1 - e2) / (1 - e2 * sin_lat ** 2) ** 1.5

    # Meters per degree at this latitude
    meters_per_deg_lon = math.pi / 180.0 * N * math.cos(lat_rad)
    meters_per_deg_lat = math.pi / 180.0 * M

    extent_lon_deg = east - west
    extent_lat_deg = north - south

    gsd_x = (extent_lon_deg * meters_per_deg_lon) / width_px
    gsd_y = (extent_lat_deg * meters_per_deg_lat) / height_px

    return gsd_x, gsd_y


def compute_area_hectares(site: ParkingLotSite) -> float:
    """Compute the area of the site bounding box in hectares."""
    west, south, east, north = site.bbox
    center_lat = (south + north) / 2.0
    lat_rad = math.radians(center_lat)

    a = 6378137.0
    e2 = 0.00669437999014
    sin_lat = math.sin(lat_rad)
    N = a / math.sqrt(1 - e2 * sin_lat ** 2)
    M = a * (1 - e2) / (1 - e2 * sin_lat ** 2) ** 1.5

    meters_per_deg_lon = math.pi / 180.0 * N * math.cos(lat_rad)
    meters_per_deg_lat = math.pi / 180.0 * M

    width_m = (east - west) * meters_per_deg_lon
    height_m = (north - south) * meters_per_deg_lat

    return (width_m * height_m) / 10000.0  # m^2 -> hectares


# ---------------------------------------------------------------------------
# Memory tracking
# ---------------------------------------------------------------------------

def get_gpu_memory_mb() -> float:
    """Return current GPU memory usage in MB, or 0 if unavailable."""
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / (1024 ** 2)
    except ImportError:
        pass
    return 0.0


def get_ram_usage_mb() -> float:
    """Return current process RSS in MB."""
    try:
        import resource
        # getrusage returns KB on Linux
        usage = resource.getrusage(resource.RUSAGE_SELF)
        return usage.ru_maxrss / 1024.0  # KB -> MB
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# Tiled detection with per-tile metrics
# ---------------------------------------------------------------------------

def run_tiled_detection_with_metrics(
    tiff_path: str,
    model_size: str,
    confidence: float,
    classes: list[str],
    weights_path: str | None = None,
    custom_class_names: dict[int, str] | None = None,
) -> tuple[Any, list[TileMetrics]]:
    """Run tiled detection and collect per-tile metrics.

    Returns:
        (GeoDataFrame, list of TileMetrics)
    """
    from detr_geo._adapter import RFDETRAdapter, prepare_tile_image
    from detr_geo.io import (
        BandSelector,
        compute_nodata_fraction,
        compute_scene_stretch_params,
        fill_nodata,
        load_raster_metadata,
        normalize_to_float32,
        read_tile,
    )
    from detr_geo.tiling import generate_tile_grid, cross_tile_nms, offset_detections
    from detr_geo.export import build_geodataframe
    import geopandas as gpd
    from pyproj import CRS

    # Create adapter with custom class names for fine-tuned models
    adapter = RFDETRAdapter(
        model_size=model_size,
        pretrain_weights=weights_path,
        confidence_threshold=confidence,
        custom_class_names=custom_class_names,
    )

    # Load metadata
    meta = load_raster_metadata(tiff_path)
    tile_size = adapter.resolution

    print(f"  Image size: {meta.width}x{meta.height} px")
    print(f"  Image CRS: {meta.crs}")
    print(f"  Model resolution: {tile_size}px")
    print(f"  Searching for classes: {classes}")

    # Generate tile grid
    tiles = generate_tile_grid(meta.width, meta.height, tile_size, overlap_ratio=0.25)
    print(f"  Tile grid: {len(tiles)} tiles (size={tile_size}, overlap=0.25)")

    band_selector = BandSelector("rgb")

    # Compute scene-level stretch parameters
    scene_stretch_params = compute_scene_stretch_params(
        tiff_path,
        bands=band_selector.band_indices,
        percentiles=(2.0, 98.0),
    )

    # Process tiles one by one to capture per-tile metrics
    all_boxes: list[list[float]] = []
    all_scores: list[float] = []
    all_class_ids: list[int] = []
    tile_metrics_list: list[TileMetrics] = []

    for tile_idx, tile in enumerate(tiles):
        window = tile["window"]
        col_off, row_off, w, h = window

        t_tile = time.time()
        try:
            data, nodata_mask = read_tile(tiff_path, window, bands=band_selector.band_indices)

            nodata_frac = compute_nodata_fraction(data, meta.nodata, alpha_mask=nodata_mask)
            if nodata_frac > 0.5:
                tile_metrics_list.append(TileMetrics(
                    tile_index=tile_idx, window=window,
                    detection_count=0, inference_time_s=0.0, skipped=True,
                ))
                continue

            if nodata_mask is not None and np.any(nodata_mask):
                data = fill_nodata(data, nodata_mask)

            normalized, _ = normalize_to_float32(
                data, stretch="percentile", stretch_params=scene_stretch_params
            )
            pil_image = prepare_tile_image(normalized)

            result = adapter.predict_tile(pil_image, threshold=confidence)
            inference_time = time.time() - t_tile

            # Offset to global coords
            shifted = offset_detections(result, col_off, row_off)
            det_count = len(shifted["bbox"])

            for box in shifted["bbox"]:
                all_boxes.append(box)
            all_scores.extend(shifted["confidence"])
            all_class_ids.extend(shifted["class_id"])

            tile_metrics_list.append(TileMetrics(
                tile_index=tile_idx, window=window,
                detection_count=det_count, inference_time_s=inference_time,
            ))

        except Exception as exc:
            tile_metrics_list.append(TileMetrics(
                tile_index=tile_idx, window=window,
                detection_count=0, inference_time_s=time.time() - t_tile,
                error=str(exc),
            ))

        # Progress
        if (tile_idx + 1) % 5 == 0 or tile_idx == len(tiles) - 1:
            print(f"    Processed {tile_idx + 1}/{len(tiles)} tiles")

    if not all_boxes:
        # Return empty GeoDataFrame
        empty_gdf = gpd.GeoDataFrame(
            {"class_name": [], "confidence": [], "class_id": []},
            geometry=[], crs=meta.crs,
        )
        return empty_gdf, tile_metrics_list

    boxes_arr = np.array(all_boxes, dtype=np.float32)
    scores_arr = np.array(all_scores, dtype=np.float32)
    class_ids_arr = np.array(all_class_ids, dtype=np.int32)

    # Cross-tile NMS
    from detr_geo.tiling import cross_tile_nms
    keep = cross_tile_nms(boxes_arr, scores_arr, class_ids_arr, iou_threshold=0.4)

    boxes_nms = boxes_arr[keep]
    scores_nms = scores_arr[keep]
    class_ids_nms = class_ids_arr[keep]

    print(f"  Pre-NMS: {len(boxes_arr)} detections -> Post-NMS: {len(boxes_nms)}")

    # Filter to requested classes
    class_names_dict = adapter.class_names
    valid_class_ids = [cid for cid, name in class_names_dict.items() if name in classes]
    if valid_class_ids:
        mask = np.isin(class_ids_nms, valid_class_ids)
        boxes_nms = boxes_nms[mask]
        scores_nms = scores_nms[mask]
        class_ids_nms = class_ids_nms[mask]
        print(f"  After class filter ({classes}): {len(boxes_nms)} detections")

    # Build GeoDataFrame
    if meta.crs is not None and meta.transform is not None:
        gdf = build_geodataframe(
            boxes_nms, scores_nms, class_ids_nms,
            class_names=adapter.class_names,
            transform=meta.transform,
            crs=meta.crs,
        )
    else:
        from detr_geo.export import build_dataframe_pixel
        gdf = build_dataframe_pixel(boxes_nms, scores_nms, class_ids_nms,
                                     class_names=adapter.class_names)

    return gdf, tile_metrics_list


# ---------------------------------------------------------------------------
# Metrics computation
# ---------------------------------------------------------------------------

def compute_spatial_dispersion(gdf: Any) -> float:
    """Compute a spatial dispersion index (0=clustered, 1=uniform).

    Uses normalized nearest-neighbor distance compared to expected
    distance under a uniform random distribution.
    """
    if len(gdf) < 2:
        return 0.0

    try:
        from scipy.spatial import cKDTree

        # Get centroids in the GeoDataFrame's CRS
        centroids = np.array([(g.centroid.x, g.centroid.y) for g in gdf.geometry])
        tree = cKDTree(centroids)

        # Nearest neighbor distances (k=2 because first neighbor is self)
        dists, _ = tree.query(centroids, k=2)
        nn_dists = dists[:, 1]  # skip self

        mean_nn = nn_dists.mean()

        # Expected mean nearest neighbor under complete spatial randomness (CSR)
        # E(d) = 0.5 / sqrt(density)
        bounds = gdf.total_bounds
        area = (bounds[2] - bounds[0]) * (bounds[3] - bounds[1])
        if area <= 0:
            return 0.0

        density = len(gdf) / area
        expected_nn = 0.5 / math.sqrt(density)

        # R = observed / expected. R < 1 = clustered, R > 1 = dispersed
        if expected_nn > 0:
            R = min(mean_nn / expected_nn, 2.0)  # cap at 2
            return R / 2.0  # normalize to [0, 1]
        return 0.0
    except ImportError:
        return -1.0  # scipy not available


def compute_bbox_areas_px(gdf: Any, transform: Any) -> np.ndarray:
    """Compute bounding box areas in pixels for each detection."""
    if len(gdf) == 0:
        return np.array([])

    inv_transform = ~transform
    areas = []
    for geom in gdf.geometry:
        bounds = geom.bounds  # (minx, miny, maxx, maxy) in CRS coords
        px_tl = inv_transform * (bounds[0], bounds[3])  # top-left
        px_br = inv_transform * (bounds[2], bounds[1])  # bottom-right
        w_px = abs(px_br[0] - px_tl[0])
        h_px = abs(px_br[1] - px_tl[1])
        areas.append(w_px * h_px)
    return np.array(areas)


def build_metrics(
    gdf: Any,
    tile_metrics: list[TileMetrics],
    site: ParkingLotSite,
    mode: str,
    model_size: str,
    confidence: float,
    weights_path: str | None,
    tiff_path: str,
    download_time: float,
    detection_time: float,
    export_time: float,
) -> DetectionMetrics:
    """Build comprehensive metrics from detection results."""
    import rasterio

    metrics = DetectionMetrics(
        site_name=site.name,
        mode=mode,
        model_size=model_size,
        confidence_threshold=confidence,
        weights_path=weights_path,
    )

    # Timing
    metrics.download_time_s = download_time
    metrics.detection_time_s = detection_time
    metrics.export_time_s = export_time

    # Image metadata
    with rasterio.open(tiff_path) as src:
        metrics.image_width_px = src.width
        metrics.image_height_px = src.height
        metrics.image_crs = str(src.crs)
        raster_transform = src.transform

    gsd_x, gsd_y = compute_gsd(site, metrics.image_width_px, metrics.image_height_px)
    metrics.gsd_x_m = gsd_x
    metrics.gsd_y_m = gsd_y
    metrics.area_hectares = compute_area_hectares(site)

    # Detection counts
    n = len(gdf)
    metrics.total_detections = n

    if n > 0 and "class_name" in gdf.columns:
        metrics.class_counts = dict(gdf["class_name"].value_counts())

    # Confidence stats
    if n > 0:
        scores = gdf["confidence"].values.astype(float)
        metrics.conf_min = float(scores.min())
        metrics.conf_max = float(scores.max())
        metrics.conf_mean = float(scores.mean())
        metrics.conf_median = float(np.median(scores))
        metrics.conf_std = float(scores.std())

    # Bbox size stats
    if n > 0:
        areas_px = compute_bbox_areas_px(gdf, raster_transform)
        if len(areas_px) > 0:
            metrics.bbox_area_min_px = float(areas_px.min())
            metrics.bbox_area_max_px = float(areas_px.max())
            metrics.bbox_area_mean_px = float(areas_px.mean())
            # Convert pixel areas to m^2 using GSD
            px_to_m2 = gsd_x * gsd_y
            metrics.bbox_area_min_m2 = float(areas_px.min() * px_to_m2)
            metrics.bbox_area_max_m2 = float(areas_px.max() * px_to_m2)
            metrics.bbox_area_mean_m2 = float(areas_px.mean() * px_to_m2)

    # Tile-level stats
    if tile_metrics:
        metrics.total_tiles = len(tile_metrics)
        processed = [tm for tm in tile_metrics if not tm.skipped and tm.error is None]
        metrics.processed_tiles = len(processed)
        metrics.empty_tiles = sum(1 for tm in processed if tm.detection_count == 0)
        metrics.failed_tiles = sum(1 for tm in tile_metrics if tm.error is not None)

        if processed:
            det_counts = [tm.detection_count for tm in processed]
            metrics.detections_per_tile_min = min(det_counts)
            metrics.detections_per_tile_max = max(det_counts)
            metrics.detections_per_tile_mean = float(np.mean(det_counts))

            inf_times = [tm.inference_time_s for tm in processed if tm.inference_time_s > 0]
            if inf_times:
                metrics.tile_inference_min_s = min(inf_times)
                metrics.tile_inference_max_s = max(inf_times)
                metrics.tile_inference_mean_s = float(np.mean(inf_times))
                metrics.tiles_per_second = 1.0 / metrics.tile_inference_mean_s if metrics.tile_inference_mean_s > 0 else 0

    # Spatial stats
    if n > 0 and metrics.area_hectares > 0:
        metrics.detection_density_per_hectare = n / metrics.area_hectares
    if n > 1:
        metrics.spatial_dispersion_index = compute_spatial_dispersion(gdf)

    # Memory
    metrics.gpu_memory_peak_mb = get_gpu_memory_mb()
    metrics.ram_usage_mb = get_ram_usage_mb()

    # Anomaly detection
    metrics.anomalies = detect_anomalies(metrics, gdf, tile_metrics, raster_transform, site)

    # Geospatial checks
    metrics.geo_checks = run_geospatial_checks(gdf, site, raster_transform, gsd_x, gsd_y)

    return metrics


# ---------------------------------------------------------------------------
# Anomaly detection
# ---------------------------------------------------------------------------

def detect_anomalies(
    metrics: DetectionMetrics,
    gdf: Any,
    tile_metrics: list[TileMetrics],
    transform: Any,
    site: ParkingLotSite,
) -> list[str]:
    """Flag unexpected patterns in detection results."""
    anomalies = []
    n = metrics.total_detections

    if n == 0:
        anomalies.append("CRITICAL: Zero detections. Model may not be detecting overhead vehicles.")
        return anomalies

    # Confidence distribution anomalies
    if metrics.conf_mean < 0.3:
        anomalies.append(
            f"LOW_CONFIDENCE: Mean confidence {metrics.conf_mean:.3f} is very low. "
            f"Model may be poorly calibrated for this imagery."
        )
    low_conf_count = (gdf["confidence"].values < 0.3).sum() if n > 0 else 0
    if n > 0 and low_conf_count / n > 0.5:
        anomalies.append(
            f"NOISY_MODEL: {low_conf_count}/{n} ({100*low_conf_count/n:.0f}%) detections "
            f"below 0.3 confidence. Many may be false positives."
        )

    # Extremely large/small bounding boxes
    if metrics.bbox_area_mean_m2 > 0:
        # A typical car from above is ~4.5m x 2m = 9 m^2
        # A bus is ~12m x 2.5m = 30 m^2
        # A truck is ~8m x 2.5m = 20 m^2
        if metrics.bbox_area_max_m2 > 200:
            anomalies.append(
                f"GIANT_BBOX: Largest bbox is {metrics.bbox_area_max_m2:.1f} m^2 "
                f"(expected <50 m^2 for vehicles). Likely false positive."
            )
        if metrics.bbox_area_min_m2 < 1.0:
            anomalies.append(
                f"TINY_BBOX: Smallest bbox is {metrics.bbox_area_min_m2:.1f} m^2 "
                f"(expected >2 m^2 for vehicles). Likely noise."
            )

    # Class distribution warnings
    if metrics.class_counts:
        total_cls = sum(metrics.class_counts.values())
        for cls_name, count in metrics.class_counts.items():
            if total_cls > 10 and count / total_cls > 0.95:
                anomalies.append(
                    f"CLASS_IMBALANCE: {cls_name} is {100*count/total_cls:.0f}% of all detections. "
                    f"Could indicate labeling bias or model overfit to one class."
                )

    # Empty tile percentage
    if metrics.processed_tiles > 0:
        empty_pct = metrics.empty_tiles / metrics.processed_tiles
        if empty_pct > 0.7:
            anomalies.append(
                f"SPARSE_DETECTION: {metrics.empty_tiles}/{metrics.processed_tiles} "
                f"({100*empty_pct:.0f}%) tiles have zero detections. "
                f"Model may be under-detecting."
            )

    # Failed tiles
    if metrics.failed_tiles > 0:
        anomalies.append(
            f"TILE_FAILURES: {metrics.failed_tiles} tiles failed inference."
        )

    # Detection density sanity check
    # A busy parking lot at 0.3m GSD should have 50-500 vehicles per hectare
    if metrics.detection_density_per_hectare > 1000:
        anomalies.append(
            f"DENSITY_HIGH: {metrics.detection_density_per_hectare:.0f} detections/hectare "
            f"is unusually high. Possible duplicate detections or NMS failure."
        )
    elif metrics.detection_density_per_hectare < 5 and metrics.area_hectares > 0.5:
        anomalies.append(
            f"DENSITY_LOW: {metrics.detection_density_per_hectare:.0f} detections/hectare "
            f"is unusually low for a parking lot. Model may be under-detecting."
        )

    return anomalies


# ---------------------------------------------------------------------------
# Geospatial checks
# ---------------------------------------------------------------------------

def run_geospatial_checks(
    gdf: Any,
    site: ParkingLotSite,
    transform: Any,
    gsd_x: float,
    gsd_y: float,
) -> list[str]:
    """Run geospatial integrity checks on detection results."""
    checks = []
    n = len(gdf)

    if n == 0:
        checks.append("SKIP: No detections to validate.")
        return checks

    # CRS check
    if gdf.crs is None:
        checks.append("FAIL: GeoDataFrame has no CRS assigned.")
    else:
        epsg = gdf.crs.to_epsg()
        if epsg == 4326:
            checks.append("PASS: CRS is EPSG:4326 (WGS84) as expected for GeoJSON output.")
        else:
            checks.append(f"WARN: CRS is EPSG:{epsg}, expected 4326 for web-friendly output.")

    # Bounds check: do detections fall within the site bbox?
    west, south, east, north = site.bbox
    bounds = gdf.total_bounds  # (minx, miny, maxx, maxy)
    margin = 0.001  # ~100m margin for edge detections

    if (bounds[0] < west - margin or bounds[1] < south - margin or
            bounds[2] > east + margin or bounds[3] > north + margin):
        checks.append(
            f"WARN: Detection bounds [{bounds[0]:.6f}, {bounds[1]:.6f}, "
            f"{bounds[2]:.6f}, {bounds[3]:.6f}] extend beyond site bbox "
            f"[{west:.6f}, {south:.6f}, {east:.6f}, {north:.6f}]. "
            f"Possible georeferencing issue."
        )
    else:
        checks.append("PASS: All detections fall within the expected site bounding box.")

    # Null island check
    if abs(bounds[0]) < 0.01 and abs(bounds[1]) < 0.01:
        checks.append("FAIL: Detections near (0,0) -- 'Null Island' syndrome. CRS likely missing.")

    # GSD check
    checks.append(
        f"INFO: GSD = {gsd_x:.4f}m x {gsd_y:.4f}m per pixel. "
        f"Esri zoom 19 is typically ~0.3m."
    )
    if gsd_x > 0.5 or gsd_y > 0.5:
        checks.append(
            f"WARN: GSD > 0.5m. Vehicle detection may degrade at this resolution."
        )

    # GSD mismatch warning for VME
    # VME was trained on satellite imagery; we should note if GSD differs significantly
    checks.append(
        f"INFO: VME training GSD is unknown. Current imagery GSD is ~{gsd_x:.3f}m. "
        f"If training GSD was significantly different, detection quality may vary."
    )

    # Tile boundary artifact check -- look for detection clustering at tile edges
    # We check if detection centroids cluster at multiples of the stride
    try:
        centroids_x = np.array([g.centroid.x for g in gdf.geometry])
        centroids_y = np.array([g.centroid.y for g in gdf.geometry])

        # Convert to pixel coordinates to check tile alignment
        inv_transform = ~transform
        px_x = []
        px_y = []
        for cx, cy in zip(centroids_x, centroids_y):
            px = inv_transform * (cx, cy)
            px_x.append(px[0])
            px_y.append(px[1])
        px_x = np.array(px_x)
        px_y = np.array(px_y)

        # Check if many detections have centroids near tile boundaries
        # (This is a rough heuristic)
        tile_size = 576  # medium model
        stride = int(tile_size * 0.75)  # overlap=0.25
        if stride > 0 and len(px_x) > 10:
            x_mod = px_x % stride
            y_mod = px_y % stride
            # Near boundary = within 5% of stride from edge
            boundary_zone = stride * 0.05
            x_near_boundary = ((x_mod < boundary_zone) | (x_mod > stride - boundary_zone)).sum()
            y_near_boundary = ((y_mod < boundary_zone) | (y_mod > stride - boundary_zone)).sum()
            total_near = x_near_boundary + y_near_boundary
            expected_near = len(px_x) * 2 * 0.10  # 10% expected in 5% zones on each side
            if total_near > expected_near * 2:
                checks.append(
                    f"WARN: {total_near} detection centroids cluster near tile boundaries "
                    f"(expected ~{expected_near:.0f}). Possible NMS failure at tile edges."
                )
            else:
                checks.append("PASS: No significant tile-boundary clustering detected.")
    except Exception:
        checks.append("SKIP: Could not run tile-boundary artifact check.")

    # Band order check
    checks.append(
        "INFO: Esri tiles are RGB uint8. Model expects RGB input. "
        "detr_geo normalizes via percentile stretch to float32 [0,1]."
    )

    return checks


# ---------------------------------------------------------------------------
# Metrics display
# ---------------------------------------------------------------------------

def print_metrics_dashboard(metrics: DetectionMetrics) -> None:
    """Print a comprehensive metrics dashboard."""
    W = 70
    print(f"\n{'=' * W}")
    print(f"  METRICS DASHBOARD: {metrics.site_name} [{metrics.mode.upper()} mode]")
    print(f"{'=' * W}")

    # Configuration
    print(f"\n  Configuration:")
    print(f"    Mode:              {metrics.mode}")
    print(f"    Model:             RF-DETR {metrics.model_size}")
    print(f"    Confidence:        {metrics.confidence_threshold:.2f}")
    if metrics.weights_path:
        print(f"    Weights:           {Path(metrics.weights_path).name}")
    else:
        print(f"    Weights:           COCO pretrained (default)")

    # Image info
    print(f"\n  Image:")
    print(f"    Size:              {metrics.image_width_px} x {metrics.image_height_px} px")
    print(f"    CRS:               {metrics.image_crs}")
    print(f"    GSD:               {metrics.gsd_x_m:.4f}m x {metrics.gsd_y_m:.4f}m")
    print(f"    Area:              {metrics.area_hectares:.2f} hectares")

    # Detection summary
    print(f"\n  Detections:")
    print(f"    Total:             {metrics.total_detections}")
    if metrics.area_hectares > 0:
        print(f"    Density:           {metrics.detection_density_per_hectare:.1f} per hectare")

    # Class breakdown
    if metrics.class_counts:
        print(f"\n  Class Breakdown:")
        for cls_name, count in sorted(metrics.class_counts.items(), key=lambda x: -x[1]):
            pct = 100 * count / metrics.total_detections if metrics.total_detections > 0 else 0
            bar = "#" * min(int(pct / 2), 30)
            print(f"    {cls_name:15s}  {count:4d}  ({pct:5.1f}%)  {bar}")

    # Confidence stats
    if metrics.total_detections > 0:
        print(f"\n  Confidence Distribution:")
        print(f"    Min:               {metrics.conf_min:.3f}")
        print(f"    Max:               {metrics.conf_max:.3f}")
        print(f"    Mean:              {metrics.conf_mean:.3f}")
        print(f"    Median:            {metrics.conf_median:.3f}")
        print(f"    Std Dev:           {metrics.conf_std:.3f}")

    # Bounding box analysis
    if metrics.bbox_area_mean_px > 0:
        print(f"\n  Bounding Box Size:")
        print(f"    Pixel area:        min={metrics.bbox_area_min_px:.0f}  "
              f"max={metrics.bbox_area_max_px:.0f}  mean={metrics.bbox_area_mean_px:.0f}")
        print(f"    Ground area (m2):  min={metrics.bbox_area_min_m2:.1f}  "
              f"max={metrics.bbox_area_max_m2:.1f}  mean={metrics.bbox_area_mean_m2:.1f}")

    # Tile-level stats
    if metrics.total_tiles > 0:
        print(f"\n  Tile Processing:")
        print(f"    Total tiles:       {metrics.total_tiles}")
        print(f"    Processed:         {metrics.processed_tiles}")
        print(f"    Empty (0 dets):    {metrics.empty_tiles}")
        print(f"    Failed:            {metrics.failed_tiles}")
        if metrics.processed_tiles > 0:
            print(f"    Dets/tile:         min={metrics.detections_per_tile_min}  "
                  f"max={metrics.detections_per_tile_max}  "
                  f"mean={metrics.detections_per_tile_mean:.1f}")

    # Timing
    print(f"\n  Timing:")
    if metrics.download_time_s > 0:
        print(f"    Download:          {metrics.download_time_s:.1f}s")
    print(f"    Detection:         {metrics.detection_time_s:.1f}s")
    if metrics.export_time_s > 0:
        print(f"    Export:            {metrics.export_time_s:.1f}s")
    total = metrics.download_time_s + metrics.detection_time_s + metrics.export_time_s
    print(f"    Total:             {total:.1f}s")

    if metrics.tile_inference_mean_s > 0:
        print(f"\n  Tile Inference Timing:")
        print(f"    Per-tile:          min={metrics.tile_inference_min_s:.3f}s  "
              f"max={metrics.tile_inference_max_s:.3f}s  "
              f"mean={metrics.tile_inference_mean_s:.3f}s")
        print(f"    Throughput:        {metrics.tiles_per_second:.1f} tiles/sec")
        # Estimate area throughput
        if metrics.area_hectares > 0 and metrics.detection_time_s > 0:
            km2_per_hour = (metrics.area_hectares / 100) / (metrics.detection_time_s / 3600)
            print(f"    Area throughput:   {km2_per_hour:.2f} km2/hour")

    # Spatial dispersion
    if metrics.spatial_dispersion_index >= 0:
        print(f"\n  Spatial Analysis:")
        print(f"    Dispersion index:  {metrics.spatial_dispersion_index:.3f} "
              f"(0=clustered, 0.5=random, 1=uniform)")
    elif metrics.spatial_dispersion_index == -1:
        print(f"\n  Spatial Analysis:")
        print(f"    Dispersion index:  N/A (scipy not installed)")

    # Memory
    if metrics.gpu_memory_peak_mb > 0 or metrics.ram_usage_mb > 0:
        print(f"\n  Memory:")
        if metrics.gpu_memory_peak_mb > 0:
            print(f"    GPU peak:          {metrics.gpu_memory_peak_mb:.0f} MB")
        if metrics.ram_usage_mb > 0:
            print(f"    RAM (RSS):         {metrics.ram_usage_mb:.0f} MB")

    # Anomalies
    if metrics.anomalies:
        print(f"\n  {'!' * W}")
        print(f"  ANOMALIES DETECTED ({len(metrics.anomalies)}):")
        for a in metrics.anomalies:
            print(f"    - {a}")
        print(f"  {'!' * W}")
    else:
        print(f"\n  No anomalies detected.")

    # Geospatial checks
    if metrics.geo_checks:
        print(f"\n  Geospatial Checks:")
        for check in metrics.geo_checks:
            print(f"    {check}")

    print(f"\n{'=' * W}")


# ---------------------------------------------------------------------------
# Comparison mode
# ---------------------------------------------------------------------------

def print_comparison(metrics_a: DetectionMetrics, metrics_b: DetectionMetrics) -> None:
    """Print side-by-side comparison of two detection runs."""
    W = 70
    print(f"\n{'*' * W}")
    print(f"  COMPARISON: {metrics_a.mode.upper()} vs {metrics_b.mode.upper()}")
    print(f"{'*' * W}")

    def cmp_line(label: str, val_a: Any, val_b: Any, fmt: str = "{}") -> None:
        sa = fmt.format(val_a)
        sb = fmt.format(val_b)
        print(f"    {label:25s}  {sa:>15s}  {sb:>15s}")

    print(f"    {'':25s}  {'[' + metrics_a.mode.upper() + ']':>15s}  {'[' + metrics_b.mode.upper() + ']':>15s}")
    print(f"    {'-' * 25}  {'-' * 15}  {'-' * 15}")

    cmp_line("Total detections", metrics_a.total_detections, metrics_b.total_detections)
    cmp_line("Density (per ha)", f"{metrics_a.detection_density_per_hectare:.1f}",
             f"{metrics_b.detection_density_per_hectare:.1f}", "{}")
    cmp_line("Confidence mean", f"{metrics_a.conf_mean:.3f}", f"{metrics_b.conf_mean:.3f}", "{}")
    cmp_line("Confidence median", f"{metrics_a.conf_median:.3f}", f"{metrics_b.conf_median:.3f}", "{}")
    cmp_line("Bbox mean area (m2)", f"{metrics_a.bbox_area_mean_m2:.1f}", f"{metrics_b.bbox_area_mean_m2:.1f}", "{}")
    cmp_line("Detection time (s)", f"{metrics_a.detection_time_s:.1f}", f"{metrics_b.detection_time_s:.1f}", "{}")
    cmp_line("Tiles/second", f"{metrics_a.tiles_per_second:.1f}", f"{metrics_b.tiles_per_second:.1f}", "{}")
    cmp_line("Empty tiles", metrics_a.empty_tiles, metrics_b.empty_tiles)
    cmp_line("Anomalies", len(metrics_a.anomalies), len(metrics_b.anomalies))

    # Class comparison
    all_classes = sorted(set(list(metrics_a.class_counts.keys()) + list(metrics_b.class_counts.keys())))
    if all_classes:
        print(f"\n    Class counts:")
        for cls in all_classes:
            ca = metrics_a.class_counts.get(cls, 0)
            cb = metrics_b.class_counts.get(cls, 0)
            print(f"      {cls:15s}  {ca:>8d}  {cb:>8d}")

    print(f"\n{'*' * W}")


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def save_visualization(
    tiff_path: str,
    gdf: Any,
    output_path: str,
    site: ParkingLotSite,
    mode: str,
) -> None:
    """Save matplotlib visualization of detections overlaid on imagery."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    import rasterio

    print(f"  Generating visualization...")

    with rasterio.open(tiff_path) as src:
        data = src.read([1, 2, 3])
        img = np.transpose(data, (1, 2, 0)).astype(np.float32)
        if img.max() > 1.0:
            img = img / 255.0
        img = np.clip(img, 0, 1)
        transform = src.transform

    fig, ax = plt.subplots(1, 1, figsize=(16, 12))
    ax.imshow(img)
    ax.set_title(
        f"{site.description}\n"
        f"{len(gdf)} detections [{mode.upper()} mode]",
        fontsize=14, fontweight="bold",
    )
    ax.set_axis_off()

    # Color map per class
    class_colors = {
        "Car": (0.0, 1.0, 0.0),      # green
        "Bus": (1.0, 0.5, 0.0),      # orange
        "Truck": (0.0, 0.5, 1.0),    # blue
        "Pickup": (1.0, 1.0, 0.0),   # yellow
        "Other": (1.0, 0.0, 1.0),    # magenta
        "car": (0.0, 1.0, 0.0),
        "motorcycle": (1.0, 1.0, 0.0),
        "bicycle": (0.0, 1.0, 1.0),
        "truck": (0.0, 0.5, 1.0),
        "bus": (1.0, 0.5, 0.0),
        "boat": (1.0, 0.0, 1.0),
    }

    if len(gdf) > 0:
        inv_transform = ~transform
        for _, row in gdf.iterrows():
            bounds = row.geometry.bounds
            px_tl = inv_transform * (bounds[0], bounds[1])
            px_br = inv_transform * (bounds[2], bounds[3])
            x1 = min(px_tl[0], px_br[0])
            y1 = min(px_tl[1], px_br[1])
            x2 = max(px_tl[0], px_br[0])
            y2 = max(px_tl[1], px_br[1])

            conf = row.get("confidence", 0.5)
            cls_name = row.get("class_name", "unknown")
            color = class_colors.get(cls_name, (1.0, 1.0, 1.0))
            linewidth = 0.8 + conf * 1.5

            rect = patches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                linewidth=linewidth, edgecolor=color, facecolor="none",
            )
            ax.add_patch(rect)

    # Legend
    from matplotlib.lines import Line2D
    if len(gdf) > 0 and "class_name" in gdf.columns:
        legend_classes = gdf["class_name"].unique()
        legend_handles = []
        for cls_name in sorted(legend_classes):
            color = class_colors.get(cls_name, (1.0, 1.0, 1.0))
            handle = Line2D([0], [0], color=color, linewidth=2, label=cls_name)
            legend_handles.append(handle)
        ax.legend(handles=legend_handles, loc="upper right", fontsize=10)

    plt.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    file_size_mb = Path(output_path).stat().st_size / (1024 * 1024)
    print(f"  Saved visualization: {output_path} ({file_size_mb:.1f} MB)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Detect vehicles in California parking lots using detr_geo.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="coco",
        choices=["coco", "vme", "xview"],
        help="Detection mode: 'coco' uses COCO pretrained weights with broad "
             "vehicle classes; 'vme' uses VME fine-tuned weights with "
             "Car/Bus/Truck classes; 'xview' uses xView fine-tuned weights "
             "with Car/Pickup/Truck/Bus/Other classes (default: coco).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="medium",
        choices=["nano", "small", "medium", "base", "large"],
        help="RF-DETR model size (default: medium).",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=None,
        help="Path to custom checkpoint weights. For --mode vme, defaults to "
             "output/checkpoint_best_ema.pth if not specified.",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=None,
        help="Confidence threshold. Default: 0.3 for COCO, 0.3 for VME.",
    )
    parser.add_argument(
        "--site",
        type=int,
        default=None,
        choices=[1, 2],
        help="Run on a single site (1=Costco Sunnyvale, 2=Great Mall Milpitas). "
             "Default: run both.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory for output files. Default: scripts/output/",
    )
    parser.add_argument(
        "--tiled",
        action="store_true",
        default=False,
        help="Use tiled detection (recommended for large images).",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        default=False,
        help="Save matplotlib visualization PNGs alongside GeoJSON output.",
    )
    parser.add_argument(
        "--zoom",
        type=int,
        default=None,
        help="Override the tile zoom level for imagery download (17-20).",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        default=False,
        help="Skip imagery download if GeoTIFF already exists in output dir.",
    )
    return parser.parse_args()


def resolve_config(args: argparse.Namespace) -> tuple[str, list[str], float, str | None, dict[int, str] | None]:
    """Resolve mode-specific configuration.

    Returns:
        (model_size, classes, confidence, weights_path, custom_class_names)
    """
    mode = args.mode
    model_size = args.model

    if mode == "vme":
        classes = VME_CLASSES
        custom_class_names = VME_CLASS_NAMES
        confidence = args.confidence if args.confidence is not None else 0.3

        # Resolve weights path
        if args.weights is not None:
            weights_path = args.weights
        else:
            # Auto-select best EMA checkpoint from output/
            project_root = Path(__file__).resolve().parent.parent
            default_path = project_root / "output" / VME_DEFAULT_CHECKPOINT
            if default_path.exists():
                weights_path = str(default_path)
                print(f"  Auto-selected VME weights: {default_path.name}")
            else:
                print(f"  WARNING: Default VME checkpoint not found at {default_path}")
                print(f"           Running with COCO pretrained weights instead.")
                weights_path = None
    elif mode == "xview":
        classes = XVIEW_CLASSES
        custom_class_names = XVIEW_CLASS_NAMES
        confidence = args.confidence if args.confidence is not None else 0.3

        # Resolve weights path
        if args.weights is not None:
            weights_path = args.weights
        else:
            project_root = Path(__file__).resolve().parent.parent
            default_path = project_root / "checkpoints" / XVIEW_DEFAULT_CHECKPOINT
            if default_path.exists():
                weights_path = str(default_path)
                print(f"  Auto-selected xView weights: {default_path.name}")
            else:
                print(f"  WARNING: Default xView checkpoint not found at {default_path}")
                print(f"           Running with COCO pretrained weights instead.")
                weights_path = None
    else:
        # COCO mode
        classes = COCO_VEHICLE_CLASSES
        custom_class_names = None
        confidence = args.confidence if args.confidence is not None else 0.3
        weights_path = args.weights  # None = COCO pretrained

    return model_size, classes, confidence, weights_path, custom_class_names


def check_dependencies() -> list[str]:
    """Check that required packages are installed."""
    missing = []
    for pkg_name, pip_hint in [
        ("detr_geo", "detr-geo"),
        ("rfdetr", "rfdetr (pip install detr-geo[rfdetr])"),
        ("matplotlib", "matplotlib (pip install detr-geo[viz])"),
        ("rasterio", "rasterio"),
        ("geopandas", "geopandas"),
    ]:
        try:
            __import__(pkg_name)
        except ImportError:
            missing.append(pip_hint)
    return missing


def main() -> int:
    """Run the parking lot vehicle detection pipeline."""
    args = parse_args()

    print("=" * 70)
    print(f"  PARKING LOT VEHICLE DETECTION")
    print(f"  Mode: {args.mode.upper()}")
    print("=" * 70)

    # Dependency check
    print("\nChecking dependencies...")
    missing = check_dependencies()
    if missing:
        print(f"\nMissing packages:")
        for pkg in missing:
            print(f"  - {pkg}")
        print(f"\nInstall all with: pip install detr-geo[all]")
        return 1
    print("  All dependencies present.")

    # GPU check
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"  GPU: {gpu_name} ({vram:.1f} GB VRAM)")
            # Reset peak memory counter for accurate tracking
            torch.cuda.reset_peak_memory_stats()
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            print(f"  Device: Apple Silicon (MPS)")
        else:
            print(f"  Device: CPU (inference will be slow)")
    except ImportError:
        print(f"  WARNING: torch not found.")

    # Resolve mode configuration
    model_size, classes, confidence, weights_path, custom_class_names = resolve_config(args)

    print(f"\n  Model:       RF-DETR {model_size}")
    print(f"  Classes:     {classes}")
    print(f"  Confidence:  {confidence}")
    if weights_path:
        print(f"  Weights:     {Path(weights_path).name}")
    else:
        print(f"  Weights:     COCO pretrained (default)")

    # Output directory -- separate by mode for comparison
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(__file__).resolve().parent / "output"

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"  Output:      {output_dir}")

    # Select sites
    if args.site is not None:
        sites_to_run = [SITES[args.site - 1]]
    else:
        sites_to_run = SITES

    print(f"\n  Sites ({len(sites_to_run)}):")
    for site in sites_to_run:
        print(f"    - {site.description}")

    # Process each site
    all_metrics: list[DetectionMetrics] = []
    overall_start = time.time()

    for site in sites_to_run:
        print(f"\n{'=' * 70}")
        print(f"  SITE: {site.description}")
        print(f"  Center: ({site.center_lat:.4f}, {site.center_lon:.4f})")
        print(f"  Mode: {args.mode.upper()}")
        print(f"{'=' * 70}")

        # File naming includes mode for separate outputs
        suffix = f"_{args.mode}"
        tiff_path = str(output_dir / f"{site.name}.tif")
        geojson_path = str(output_dir / f"{site.name}{suffix}_vehicles.geojson")
        gpkg_path = str(output_dir / f"{site.name}{suffix}_vehicles.gpkg")

        # Step 1: Download imagery
        download_time = 0.0
        if args.skip_download and Path(tiff_path).exists():
            print(f"\n[1/3] Skipping download (file exists): {Path(tiff_path).name}")
        else:
            print(f"\n[1/3] Downloading imagery...")
            try:
                if args.zoom is not None:
                    site_copy = ParkingLotSite(
                        name=site.name, description=site.description,
                        city=site.city, bbox=site.bbox, zoom=args.zoom,
                    )
                    download_time = download_imagery(site_copy, tiff_path)
                else:
                    download_time = download_imagery(site, tiff_path)
            except RuntimeError as exc:
                print(f"\n  ERROR downloading imagery: {exc}")
                print(f"  Skipping this site.")
                continue

        # Step 2: Run detection (always tiled for both modes)
        print(f"\n[2/3] Running vehicle detection (tiled)...")
        t_det = time.time()
        try:
            gdf, tile_metrics = run_tiled_detection_with_metrics(
                tiff_path=tiff_path,
                model_size=model_size,
                confidence=confidence,
                classes=classes,
                weights_path=weights_path,
                custom_class_names=custom_class_names,
            )
            detection_time = time.time() - t_det
            print(f"  Detection complete in {detection_time:.1f}s")
        except Exception as exc:
            print(f"\n  ERROR during detection: {exc}")
            traceback.print_exc()
            print(f"  Skipping this site.")
            continue

        # Step 3: Export results
        print(f"\n[3/3] Exporting results...")
        t_export = time.time()

        if len(gdf) > 0:
            try:
                gdf.to_file(geojson_path, driver="GeoJSON")
                print(f"  GeoJSON: {Path(geojson_path).name}")
            except Exception as exc:
                print(f"  WARNING: GeoJSON export failed: {exc}")

            try:
                gdf.to_file(gpkg_path, driver="GPKG", layer="detections")
                print(f"  GeoPackage: {Path(gpkg_path).name}")
            except Exception as exc:
                print(f"  WARNING: GeoPackage export failed: {exc}")
        else:
            print(f"  No detections to export.")

        export_time = time.time() - t_export

        # Optional visualization
        if args.visualize:
            viz_path = str(output_dir / f"{site.name}{suffix}_detections.png")
            try:
                save_visualization(tiff_path, gdf, viz_path, site, args.mode)
            except Exception as exc:
                print(f"  WARNING: Visualization failed: {exc}")
                traceback.print_exc()

        # Build and display metrics
        metrics = build_metrics(
            gdf=gdf,
            tile_metrics=tile_metrics,
            site=site,
            mode=args.mode,
            model_size=model_size,
            confidence=confidence,
            weights_path=weights_path,
            tiff_path=tiff_path,
            download_time=download_time,
            detection_time=detection_time,
            export_time=export_time,
        )
        print_metrics_dashboard(metrics)
        all_metrics.append(metrics)

    # Overall summary
    total_elapsed = time.time() - overall_start
    total_detections = sum(m.total_detections for m in all_metrics)

    print(f"\n{'=' * 70}")
    print(f"  OVERALL SUMMARY [{args.mode.upper()} mode]")
    print(f"{'=' * 70}")
    print(f"  Sites processed:     {len(all_metrics)}")
    print(f"  Total detections:    {total_detections}")
    print(f"  Total time:          {total_elapsed:.1f}s")
    print(f"  Output directory:    {output_dir}")

    print(f"\nOutput files:")
    for f in sorted(output_dir.iterdir()):
        size = f.stat().st_size
        if size > 1024 * 1024:
            size_str = f"{size / (1024 * 1024):.1f} MB"
        elif size > 1024:
            size_str = f"{size / 1024:.1f} KB"
        else:
            size_str = f"{size} B"
        print(f"  {f.name:50s} {size_str}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
