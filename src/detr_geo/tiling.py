"""Tile grid generation, NMS, and tiled detection pipeline.

This module generates tile grids for processing large geospatial images
in manageable chunks, handles overlap regions, performs cross-tile NMS
deduplication, and orchestrates the full tiled detection pipeline.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from detr_geo._typing import DetectionResult, TileInfo
from detr_geo.exceptions import TilingError

# ---------------------------------------------------------------------------
# Tile Grid Generation
# ---------------------------------------------------------------------------


def generate_tile_grid(
    raster_width: int,
    raster_height: int,
    tile_size: int,
    overlap_ratio: float = 0.2,
) -> list[TileInfo]:
    """Generate a tile grid covering the entire raster.

    Args:
        raster_width: Width of the raster in pixels.
        raster_height: Height of the raster in pixels.
        tile_size: Size of each square tile in pixels.
        overlap_ratio: Fractional overlap between adjacent tiles (0.0-0.5 exclusive).

    Returns:
        List of TileInfo dicts with window coordinates and metadata.

    Raises:
        TilingError: If overlap_ratio >= 0.5 or tile_size <= 0.
    """
    if tile_size <= 0:
        raise TilingError(f"tile_size must be positive, got {tile_size}")

    if overlap_ratio >= 0.5:
        raise TilingError(f"overlap_ratio must be less than 0.5, got {overlap_ratio}. Use 0.2-0.3 for best results.")

    if overlap_ratio < 0:
        raise TilingError(f"overlap_ratio must be non-negative, got {overlap_ratio}")

    stride = int(tile_size * (1 - overlap_ratio))
    if stride <= 0:
        stride = 1

    tiles: list[TileInfo] = []

    # Generate column (x) offsets
    col_offsets = list(range(0, raster_width, stride))
    if not col_offsets:
        col_offsets = [0]

    # Generate row (y) offsets
    row_offsets = list(range(0, raster_height, stride))
    if not row_offsets:
        row_offsets = [0]

    for row_off in row_offsets:
        for col_off in col_offsets:
            # Compute tile dimensions (may be smaller at edges)
            w = min(tile_size, raster_width - col_off)
            h = min(tile_size, raster_height - row_off)

            # Ensure minimum tile size
            if w <= 0 or h <= 0:
                continue

            tile = TileInfo(
                window=(col_off, row_off, w, h),
                global_offset_x=col_off,
                global_offset_y=row_off,
                nodata_fraction=0.0,
            )
            tiles.append(tile)

    return tiles


def recommended_overlap(
    object_size_pixels: float,
    object_size_fraction: float = 0.3,
    tile_size: int = 576,
) -> float:
    """Recommend overlap ratio based on expected object size.

    The overlap should be at least as large as the expected object size
    relative to the tile, to avoid splitting objects at tile boundaries.

    Args:
        object_size_pixels: Expected object size in pixels (e.g., 30 for buildings).
        object_size_fraction: Fraction of tile that should overlap. Default 0.3.
        tile_size: Tile size in pixels.

    Returns:
        Recommended overlap ratio (0.0 to 0.49).
    """
    if tile_size <= 0:
        return 0.2

    ratio = (object_size_pixels / tile_size) * (1.0 / object_size_fraction)
    return min(0.49, max(0.05, ratio))


def detection_range(
    tile_size: int,
    gsd: float,
    overlap: float = 0.2,
) -> tuple[float, float]:
    """Compute the effective detection size range for a given configuration.

    Args:
        tile_size: Tile size in pixels.
        gsd: Ground sample distance in meters per pixel.
        overlap: Overlap ratio.

    Returns:
        Tuple of (min_object_meters, max_object_meters) for effective detection.
    """
    min_pixels = 10  # Minimum detectable object in pixels
    max_pixels = tile_size * 0.8  # Object shouldn't fill more than 80% of tile

    min_meters = min_pixels * gsd
    max_meters = max_pixels * gsd

    return (min_meters, max_meters)


# ---------------------------------------------------------------------------
# Cross-Tile NMS
# ---------------------------------------------------------------------------


def compute_iou(box1: NDArray, box2: NDArray) -> float:
    """Compute IoU between two boxes in [x1, y1, x2, y2] format.

    Args:
        box1: Array of shape (4,).
        box2: Array of shape (4,).

    Returns:
        IoU value between 0.0 and 1.0.
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)

    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection

    if union <= 0:
        return 0.0

    return intersection / union


def cross_tile_nms(
    boxes: NDArray,
    scores: NDArray,
    class_ids: NDArray,
    iou_threshold: float = 0.5,
) -> NDArray:
    """Perform class-aware non-maximum suppression across tiles.

    Args:
        boxes: Array of shape (N, 4) in [x1, y1, x2, y2] format.
        scores: Array of shape (N,) with confidence scores.
        class_ids: Array of shape (N,) with class IDs.
        iou_threshold: IoU threshold above which to suppress.

    Returns:
        Boolean mask of shape (N,) where True means the detection survives.
    """
    n = len(boxes)
    if n == 0:
        return np.array([], dtype=bool)

    keep = np.ones(n, dtype=bool)

    # Sort by confidence descending
    order = np.argsort(-scores)

    for i in range(n):
        idx_i = order[i]
        if not keep[idx_i]:
            continue

        # Find remaining candidates after current position
        remaining = order[i + 1 :]
        if len(remaining) == 0:
            break

        # Filter to same class only (class-aware NMS)
        same_class_mask = class_ids[remaining] == class_ids[idx_i]
        candidates = remaining[same_class_mask & keep[remaining]]

        if len(candidates) == 0:
            continue

        # Vectorized IoU computation for box i against all candidates
        box_i = boxes[idx_i]
        candidate_boxes = boxes[candidates]

        # Compute intersection
        xx1 = np.maximum(box_i[0], candidate_boxes[:, 0])
        yy1 = np.maximum(box_i[1], candidate_boxes[:, 1])
        xx2 = np.minimum(box_i[2], candidate_boxes[:, 2])
        yy2 = np.minimum(box_i[3], candidate_boxes[:, 3])

        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)

        # Compute union
        area_i = (box_i[2] - box_i[0]) * (box_i[3] - box_i[1])
        area_c = (candidate_boxes[:, 2] - candidate_boxes[:, 0]) * (candidate_boxes[:, 3] - candidate_boxes[:, 1])
        union = area_i + area_c - inter

        # Compute IoU
        iou = np.where(union > 0, inter / union, 0.0)

        # Suppress candidates with IoU above threshold
        suppress_mask = iou > iou_threshold
        keep[candidates[suppress_mask]] = False

    return keep


def edge_zone_filter(
    boxes: NDArray,
    scores: NDArray,
    tile_windows: list[tuple[int, int, int, int]],
    tile_indices: NDArray,
    edge_buffer_ratio: float = 0.1,
    iou_threshold: float = 0.5,
) -> NDArray:
    """Filter detections in tile edge zones.

    Detections whose center is within edge_buffer of the tile boundary
    are suppressed when an overlapping center-tile detection exists.

    Args:
        boxes: Array of shape (N, 4) in global pixel coordinates.
        scores: Array of shape (N,).
        tile_windows: List of (col_off, row_off, width, height) for each tile.
        tile_indices: Array of shape (N,) mapping each detection to its tile.
        edge_buffer_ratio: Fraction of tile size to consider as edge zone.
        iou_threshold: IoU threshold for suppression.

    Returns:
        Boolean mask of shape (N,) where True means detection survives.
    """
    n = len(boxes)
    if n == 0:
        return np.array([], dtype=bool)

    # Determine if each detection center is in the edge zone
    centers_x = (boxes[:, 0] + boxes[:, 2]) / 2
    centers_y = (boxes[:, 1] + boxes[:, 3]) / 2

    is_edge = np.zeros(n, dtype=bool)

    for i in range(n):
        tile_idx = tile_indices[i]
        col_off, row_off, w, h = tile_windows[tile_idx]
        buffer_x = w * edge_buffer_ratio
        buffer_y = h * edge_buffer_ratio

        cx = centers_x[i] - col_off  # Tile-local center
        cy = centers_y[i] - row_off

        if cx < buffer_x or cx > (w - buffer_x) or cy < buffer_y or cy > (h - buffer_y):
            is_edge[i] = True

    keep = np.ones(n, dtype=bool)

    # Identify center detections (not edge)
    center_mask = ~is_edge

    # For edge detections, suppress if a center-tile detection overlaps
    edge_indices = np.where(is_edge)[0]
    center_indices = np.where(center_mask)[0]

    for edge_idx in edge_indices:
        if not keep[edge_idx]:
            continue

        if len(center_indices) == 0:
            continue

        # Vectorized IoU computation for edge detection against all center detections
        edge_box = boxes[edge_idx]
        center_boxes = boxes[center_indices]

        # Compute intersection
        xx1 = np.maximum(edge_box[0], center_boxes[:, 0])
        yy1 = np.maximum(edge_box[1], center_boxes[:, 1])
        xx2 = np.minimum(edge_box[2], center_boxes[:, 2])
        yy2 = np.minimum(edge_box[3], center_boxes[:, 3])

        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)

        # Compute union
        area_edge = (edge_box[2] - edge_box[0]) * (edge_box[3] - edge_box[1])
        area_center = (center_boxes[:, 2] - center_boxes[:, 0]) * (center_boxes[:, 3] - center_boxes[:, 1])
        union = area_edge + area_center - inter

        # Compute IoU
        iou = np.where(union > 0, inter / union, 0.0)

        # If any center detection overlaps above threshold, suppress this edge detection
        if np.any(iou > iou_threshold):
            keep[edge_idx] = False

    return keep


# ---------------------------------------------------------------------------
# Tile Processing Pipeline
# ---------------------------------------------------------------------------


def offset_detections(
    detection: DetectionResult,
    offset_x: int,
    offset_y: int,
) -> DetectionResult:
    """Shift detection bounding boxes from tile-local to global pixel coordinates.

    Args:
        detection: DetectionResult with tile-local coordinates.
        offset_x: X offset of the tile origin in global coordinates.
        offset_y: Y offset of the tile origin in global coordinates.

    Returns:
        New DetectionResult with globally-offset coordinates.
    """
    shifted_boxes = []
    for box in detection["bbox"]:
        shifted_boxes.append(
            [
                box[0] + offset_x,
                box[1] + offset_y,
                box[2] + offset_x,
                box[3] + offset_y,
            ]
        )

    return DetectionResult(
        bbox=shifted_boxes,
        confidence=list(detection["confidence"]),
        class_id=list(detection["class_id"]),
    )


def process_tiles(
    raster_path: str,
    adapter: Any,
    tile_size: int,
    overlap: float = 0.2,
    nms_threshold: float = 0.5,
    nodata_threshold: float = 0.5,
    threshold: float | None = None,
    batch_size: int | None = None,
    bands: tuple[int, ...] | str = "rgb",
    show_progress: bool = True,
) -> tuple[NDArray, NDArray, NDArray]:
    """Run tiled detection over a raster.

    Args:
        raster_path: Path to raster file.
        adapter: RFDETRAdapter instance.
        tile_size: Tile size in pixels.
        overlap: Overlap ratio.
        nms_threshold: IoU threshold for cross-tile NMS.
        nodata_threshold: Skip tiles with nodata fraction above this.
        threshold: Confidence threshold override.
        batch_size: Batch size for inference. Auto if None.
        bands: Band selection preset or tuple.
        show_progress: Whether to display progress bar.

    Returns:
        Tuple of (boxes, scores, class_ids) in global pixel coordinates,
        after NMS deduplication.
    """
    from detr_geo._adapter import prepare_tile_image
    from detr_geo.io import (
        BandSelector,
        check_gsd,
        compute_nodata_fraction,
        compute_scene_stretch_params,
        fill_nodata,
        load_raster_metadata,
        normalize_to_float32,
        read_tile,
    )

    # Load metadata
    meta = load_raster_metadata(raster_path)

    # Check GSD and warn if outside optimal range
    if meta.gsd is not None:
        check_gsd(meta.gsd)

    # Generate tile grid
    tiles = generate_tile_grid(meta.width, meta.height, tile_size, overlap)

    if not tiles:
        return np.array([]).reshape(0, 4), np.array([]), np.array([], dtype=np.int32)

    # Set up band selector
    band_selector = BandSelector(bands)
    band_selector.clamp_to_band_count(meta.count)

    # Compute scene-level stretch parameters once for consistent normalization
    scene_stretch_params = compute_scene_stretch_params(
        raster_path,
        bands=band_selector.read_indices,
        percentiles=(2.0, 98.0),
    )

    # Determine batch size
    if batch_size is None:
        batch_size = adapter.auto_batch_size()

    # Collect all detections
    all_boxes: list[list[float]] = []
    all_scores: list[float] = []
    all_class_ids: list[int] = []
    all_tile_indices: list[int] = []
    tile_windows: list[tuple[int, int, int, int]] = []

    # Optional progress bar
    tile_iter: Any = tiles
    if show_progress:
        try:
            from tqdm import tqdm

            tile_iter = tqdm(tiles, desc="Processing tiles", unit="tile")
        except ImportError:
            pass

    batch_images = []
    batch_offsets = []
    batch_tile_idx = []

    for tile_idx, tile in enumerate(tile_iter):
        window = tile["window"]
        tile_windows.append(window)
        col_off, row_off, w, h = window

        # Read tile data
        data, nodata_mask = read_tile(raster_path, window, bands=band_selector.read_indices)

        # Select/reorder/triplicate bands for model input
        data, alpha = band_selector.select(data, meta.count)
        if alpha is not None and nodata_mask is None:
            nodata_mask = alpha

        # Check nodata
        nodata_frac = compute_nodata_fraction(data, meta.nodata, alpha_mask=nodata_mask)
        if nodata_frac > nodata_threshold:
            continue

        # Fill nodata if partial
        if nodata_mask is not None and np.any(nodata_mask):
            data = fill_nodata(data, nodata_mask)

        # Normalize using scene-level stretch parameters for consistent brightness
        normalized, _ = normalize_to_float32(data, stretch="percentile", stretch_params=scene_stretch_params)

        # Convert to PIL image
        pil_image = prepare_tile_image(normalized)

        batch_images.append(pil_image)
        batch_offsets.append((col_off, row_off))
        batch_tile_idx.append(tile_idx)

        # Process batch when full
        if len(batch_images) >= batch_size:
            results = adapter.predict_tiles(batch_images, threshold=threshold)
            for k, result in enumerate(results):
                ox, oy = batch_offsets[k]
                shifted = offset_detections(result, ox, oy)
                for box in shifted["bbox"]:
                    all_boxes.append(box)
                all_scores.extend(shifted["confidence"])
                all_class_ids.extend(shifted["class_id"])
                all_tile_indices.extend([batch_tile_idx[k]] * len(shifted["bbox"]))

            batch_images.clear()
            batch_offsets.clear()
            batch_tile_idx.clear()

    # Process remaining batch
    if batch_images:
        results = adapter.predict_tiles(batch_images, threshold=threshold)
        for k, result in enumerate(results):
            ox, oy = batch_offsets[k]
            shifted = offset_detections(result, ox, oy)
            for box in shifted["bbox"]:
                all_boxes.append(box)
            all_scores.extend(shifted["confidence"])
            all_class_ids.extend(shifted["class_id"])
            all_tile_indices.extend([batch_tile_idx[k]] * len(shifted["bbox"]))

    if not all_boxes:
        return np.array([]).reshape(0, 4), np.array([]), np.array([], dtype=np.int32)

    boxes_arr = np.array(all_boxes, dtype=np.float32)
    scores_arr = np.array(all_scores, dtype=np.float32)
    class_ids_arr = np.array(all_class_ids, dtype=np.int32)

    # Apply NMS
    keep = cross_tile_nms(boxes_arr, scores_arr, class_ids_arr, nms_threshold)

    return boxes_arr[keep], scores_arr[keep], class_ids_arr[keep]
