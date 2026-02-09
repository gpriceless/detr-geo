#!/usr/bin/env python3
"""Process xView dataset: tile GeoTIFFs into COCO-format training dataset.

Takes remapped xView GeoJSON (from preprocess_xview.py) and raw GeoTIFF
images, tiles them into 576x576 chips with 0.2 overlap, clips annotations
to tile boundaries, applies image-level splitting (80/15/5), and writes
COCO-format output ready for RF-DETR fine-tuning via train_xview.py.

Split strategy: Each source GeoTIFF image is assigned entirely to one
split (train/valid/test). All tiles from that image inherit the same
split. This prevents spatial leakage while ensuring proper split ratios
with zero discarded tiles.

Pipeline position:
    preprocess_xview.py (60->5 classes) -> THIS SCRIPT -> train_xview.py

Processing is parallelized across GeoTIFF images using multiprocessing.
Each worker processes one image independently (tile, clip annotations, save
JPEG chips). The main process merges all worker outputs, applies global
image-level splitting, and writes COCO JSON files.

Dataset: xView Detection Challenge (Lam et al., 2018)
License: CC BY-NC-SA 4.0 (Attribution-NonCommercial-ShareAlike)
Citation:
    Lam, D. et al. "xView: Objects in Context in Overhead Imagery."
    arXiv:1802.07856, 2018.

Usage:
    python scripts/process_xview.py \\
      --input_dir xview_raw/train_images/ \\
      --annotations xview_dataset/xview_train_remapped.geojson \\
      --output_dir xview_coco/ \\
      --tile_size 576 \\
      --overlap 0.2 \\
      --workers 8

    python scripts/process_xview.py --dry_run \\
      --input_dir xview_raw/train_images/ \\
      --annotations xview_dataset/xview_train_remapped.geojson \\
      --output_dir xview_coco/

    python scripts/process_xview.py --resume \\
      --input_dir xview_raw/train_images/ \\
      --annotations xview_dataset/xview_train_remapped.geojson \\
      --output_dir xview_coco/
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing
import os
import sys
import time
import traceback
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_TILE_SIZE = 576
DEFAULT_OVERLAP = 0.2
DEFAULT_MIN_ANNOTATION_AREA = 50
DEFAULT_MAX_BACKGROUND_RATIO = 2.0
DEFAULT_SPLIT_RATIOS = (0.80, 0.15, 0.05)
DEFAULT_NUM_CLASSES = 5
JPEG_QUALITY = 95
CHECKPOINT_FILE = ".processing_state.json"

# xView 5-class mapping (must match preprocess_xview.py output)
XVIEW_CLASSES: dict[int, str] = {
    0: "Car",
    1: "Pickup",
    2: "Truck",
    3: "Bus",
    4: "Other",
}


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class TileRecord:
    """Record for a single tile produced by a worker."""

    image_filename: str       # Tile JPEG filename (relative to split/images/)
    tile_width: int
    tile_height: int
    source_image: str         # Source GeoTIFF filename
    global_col: int           # Column position in global virtual grid
    global_row: int           # Row position in global virtual grid
    annotations: list[dict]   # List of COCO annotation dicts (no id/image_id yet)
    has_annotations: bool


@dataclass
class ImageResult:
    """Result from processing a single GeoTIFF image."""

    image_name: str
    tiles: list[TileRecord]
    error: str | None = None
    skipped_annotations: int = 0
    processing_time: float = 0.0


# ---------------------------------------------------------------------------
# Annotation handling (pixel coordinate space)
# ---------------------------------------------------------------------------


def parse_annotations_for_image(
    features: list[dict],
    image_id_value: str,
) -> list[dict]:
    """Extract annotations belonging to a specific image.

    xView annotations use the ``image_id`` property to link features to
    specific image files. The ``bounds_imcoords`` field contains pixel-space
    bounding boxes as "xmin,ymin,xmax,ymax" strings.

    Args:
        features: List of GeoJSON feature dicts (remapped).
        image_id_value: The image_id value to filter on.

    Returns:
        List of annotation dicts with keys: class_id, class_name, bbox
        (as [xmin, ymin, xmax, ymax] in pixel coordinates).
    """
    annotations = []

    for feat in features:
        props = feat.get("properties", {})
        feat_image_id = props.get("image_id", "")

        if str(feat_image_id) != str(image_id_value):
            continue

        # Parse pixel bounding box from bounds_imcoords
        bounds_str = props.get("bounds_imcoords", "")
        if not bounds_str:
            continue

        try:
            parts = [float(x) for x in bounds_str.split(",")]
            if len(parts) != 4:
                continue
            xmin, ymin, xmax, ymax = parts
        except (ValueError, TypeError):
            continue

        # Ensure valid bbox (xmin < xmax, ymin < ymax)
        if xmax <= xmin or ymax <= ymin:
            continue

        class_id = props.get("class_id", 0)
        class_name = props.get("class_name", XVIEW_CLASSES.get(class_id, "Unknown"))

        annotations.append({
            "class_id": int(class_id),
            "class_name": str(class_name),
            "bbox_pixel": [xmin, ymin, xmax, ymax],
        })

    return annotations


def clip_annotations_to_tile(
    annotations: list[dict],
    tile_col_off: int,
    tile_row_off: int,
    tile_width: int,
    tile_height: int,
    min_area: float,
) -> list[dict]:
    """Clip pixel-space annotations to a tile's pixel bounds.

    Computes the intersection of each annotation bbox with the tile
    boundary. Annotations that fall below ``min_area`` after clipping
    are discarded.

    Args:
        annotations: List of annotation dicts with ``bbox_pixel``
            [xmin, ymin, xmax, ymax] in full-image pixel coordinates.
        tile_col_off: Tile column offset in pixels (from left edge of image).
        tile_row_off: Tile row offset in pixels (from top edge of image).
        tile_width: Tile width in pixels.
        tile_height: Tile height in pixels.
        min_area: Minimum annotation area in square pixels after clipping.

    Returns:
        List of COCO-format annotation dicts with ``bbox`` as
        [x, y, width, height] in tile-local pixel coordinates,
        ``area``, and ``category_id``.
    """
    tile_xmin = tile_col_off
    tile_ymin = tile_row_off
    tile_xmax = tile_col_off + tile_width
    tile_ymax = tile_row_off + tile_height

    results = []

    for ann in annotations:
        ax1, ay1, ax2, ay2 = ann["bbox_pixel"]

        # Compute intersection with tile bounds
        ix1 = max(ax1, tile_xmin)
        iy1 = max(ay1, tile_ymin)
        ix2 = min(ax2, tile_xmax)
        iy2 = min(ay2, tile_ymax)

        # Check for valid intersection
        if ix2 <= ix1 or iy2 <= iy1:
            continue

        # Convert to tile-local coordinates
        local_x = ix1 - tile_col_off
        local_y = iy1 - tile_row_off
        local_w = ix2 - ix1
        local_h = iy2 - iy1

        area = local_w * local_h

        if area < min_area:
            continue

        results.append({
            "bbox": [float(local_x), float(local_y), float(local_w), float(local_h)],
            "area": float(area),
            "category_id": ann["class_id"],
            "iscrowd": 0,
        })

    return results


# ---------------------------------------------------------------------------
# Per-image worker function
# ---------------------------------------------------------------------------


def process_single_image(
    tif_path: str,
    image_annotations: list[dict],
    output_base_dir: str,
    tile_size: int,
    overlap_ratio: float,
    min_annotation_area: float,
    global_grid_offset_col: int,
    global_grid_offset_row: int,
) -> ImageResult:
    """Process a single GeoTIFF: tile, clip annotations, save JPEG chips.

    This function runs in a worker process. It reads the GeoTIFF with
    windowed reads (never loads the full image), tiles it into square
    chips, clips annotations to each tile, and saves each tile as JPEG
    to a staging directory. Tile images are written to a flat staging
    area and later moved to the correct split directory.

    Args:
        tif_path: Absolute path to the GeoTIFF file.
        image_annotations: Pre-filtered annotations for this image.
        output_base_dir: Base output directory.
        tile_size: Size of each square tile in pixels.
        overlap_ratio: Fractional overlap between tiles.
        min_annotation_area: Minimum annotation area in square pixels.
        global_grid_offset_col: Column offset in the global virtual grid
            (for spatial splitting across images).
        global_grid_offset_row: Row offset in the global virtual grid.

    Returns:
        ImageResult with tile records and processing metadata.
    """
    import rasterio
    from rasterio.windows import Window
    from PIL import Image

    image_name = Path(tif_path).stem
    t0 = time.time()
    tiles: list[TileRecord] = []
    skipped_annotations = 0

    staging_dir = Path(output_base_dir) / "_staging" / "images"
    staging_dir.mkdir(parents=True, exist_ok=True)

    try:
        with rasterio.open(tif_path) as src:
            img_width = src.width
            img_height = src.height
            band_count = src.count

            # Determine bands to read (first 3 for RGB, handle various band counts)
            if band_count >= 3:
                read_bands = [1, 2, 3]
            elif band_count == 1:
                read_bands = [1]
            else:
                read_bands = list(range(1, band_count + 1))

            # Compute tile grid for this image
            stride = int(tile_size * (1.0 - overlap_ratio))
            if stride <= 0:
                stride = 1

            col_offsets = list(range(0, img_width, stride))
            row_offsets = list(range(0, img_height, stride))

            # Track grid position relative to this image for spatial splitting
            grid_col_idx = 0

            for col_off in col_offsets:
                grid_row_idx = 0
                for row_off in row_offsets:
                    # Compute tile dimensions (may be smaller at edges)
                    tw = min(tile_size, img_width - col_off)
                    th = min(tile_size, img_height - row_off)

                    # Skip very small edge tiles (less than 25% of full tile)
                    if tw < tile_size * 0.25 or th < tile_size * 0.25:
                        grid_row_idx += 1
                        continue

                    # --- Windowed read (memory-safe) ---
                    window = Window(col_off, row_off, tw, th)
                    try:
                        data = src.read(
                            indexes=read_bands,
                            window=window,
                            boundless=True,
                            fill_value=src.nodata if src.nodata is not None else 0,
                        )
                    except Exception:
                        # Skip unreadable windows
                        grid_row_idx += 1
                        continue

                    # Handle nodata: skip tiles that are mostly nodata
                    if src.nodata is not None:
                        nodata_mask = np.all(data == src.nodata, axis=0)
                        nodata_frac = float(np.sum(nodata_mask)) / (tw * th)
                        if nodata_frac > 0.5:
                            grid_row_idx += 1
                            continue

                    # --- Normalize to uint8 ---
                    if data.dtype != np.uint8:
                        # Percentile stretch per-tile
                        out = np.zeros_like(data, dtype=np.float32)
                        for b in range(data.shape[0]):
                            band = data[b].astype(np.float32)
                            valid = band.ravel()
                            if len(valid) == 0:
                                continue
                            p2 = np.percentile(valid, 2)
                            p98 = np.percentile(valid, 98)
                            if p98 - p2 > 0:
                                band = (band - p2) / (p98 - p2)
                            else:
                                band = np.zeros_like(band)
                            out[b] = band
                        uint8_data = (np.clip(out, 0, 1) * 255).astype(np.uint8)
                    else:
                        uint8_data = data

                    # Convert (C, H, W) -> (H, W, C) for PIL
                    if uint8_data.shape[0] == 1:
                        # Single band: triplicate to RGB
                        img_array = np.stack(
                            [uint8_data[0], uint8_data[0], uint8_data[0]], axis=-1
                        )
                    elif uint8_data.shape[0] >= 3:
                        img_array = np.transpose(uint8_data[:3], (1, 2, 0))
                    else:
                        # 2 bands: pad with zeros
                        padded = np.zeros(
                            (3, uint8_data.shape[1], uint8_data.shape[2]),
                            dtype=np.uint8,
                        )
                        padded[:uint8_data.shape[0]] = uint8_data
                        img_array = np.transpose(padded, (1, 2, 0))

                    # --- Clip annotations to this tile ---
                    tile_anns = clip_annotations_to_tile(
                        image_annotations,
                        col_off,
                        row_off,
                        tw,
                        th,
                        min_annotation_area,
                    )

                    has_annotations = len(tile_anns) > 0

                    # --- Save tile JPEG ---
                    tile_filename = f"{image_name}_c{col_off}_r{row_off}.jpg"
                    tile_path = staging_dir / tile_filename

                    pil_img = Image.fromarray(img_array, mode="RGB")

                    # Pad undersized edge tiles to full tile_size for consistent
                    # input dimensions during training
                    if tw < tile_size or th < tile_size:
                        padded_img = Image.new("RGB", (tile_size, tile_size), (0, 0, 0))
                        padded_img.paste(pil_img, (0, 0))
                        pil_img = padded_img
                        # Update dimensions to reflect the padded tile
                        tw = tile_size
                        th = tile_size

                    pil_img.save(str(tile_path), "JPEG", quality=JPEG_QUALITY)

                    # Global grid position for spatial splitting
                    global_col = global_grid_offset_col + grid_col_idx
                    global_row = global_grid_offset_row + grid_row_idx

                    tiles.append(TileRecord(
                        image_filename=tile_filename,
                        tile_width=tw,
                        tile_height=th,
                        source_image=image_name,
                        global_col=global_col,
                        global_row=global_row,
                        annotations=tile_anns,
                        has_annotations=has_annotations,
                    ))

                    grid_row_idx += 1
                grid_col_idx += 1

    except Exception as exc:
        return ImageResult(
            image_name=image_name,
            tiles=[],
            error=f"{type(exc).__name__}: {exc}",
            processing_time=time.time() - t0,
        )

    return ImageResult(
        image_name=image_name,
        tiles=tiles,
        skipped_annotations=skipped_annotations,
        processing_time=time.time() - t0,
    )


# ---------------------------------------------------------------------------
# Spatial splitting (image-level assignment)
# ---------------------------------------------------------------------------


def image_level_split(
    tiles: list[TileRecord],
    ratios: tuple[float, float, float],
    seed: int = 42,
) -> dict[int, str]:
    """Assign tiles to train/valid/test splits at the source-image level.

    Each source GeoTIFF image is assigned entirely to one split (train,
    valid, or test). All tiles from that image inherit the same split.
    This prevents spatial leakage between splits (tiles from the same
    image never appear in different splits) while ensuring proper split
    ratios with zero discarded tiles.

    This replaces the previous block-based spatial split which discarded
    ~47% of tiles as buffer-zone tiles at block boundaries.

    Args:
        tiles: List of TileRecord objects with source_image field.
        ratios: (train, valid, test) split ratios summing to 1.0.
        seed: Random seed for reproducibility.

    Returns:
        Dict mapping tile index to split name ("train", "valid", "test").
    """
    if not tiles:
        return {}

    rng = np.random.RandomState(seed)

    # Group tiles by source image
    image_tile_indices: dict[str, list[int]] = {}
    for idx, tile in enumerate(tiles):
        src = tile.source_image
        if src not in image_tile_indices:
            image_tile_indices[src] = []
        image_tile_indices[src].append(idx)

    # Sort image names for deterministic ordering before shuffle
    image_names = sorted(image_tile_indices.keys())
    n_images = len(image_names)

    # Shuffle images
    rng.shuffle(image_names)

    # Assign images to splits based on ratios
    train_ratio, val_ratio, _test_ratio = ratios
    n_train = max(1, round(n_images * train_ratio))
    n_val = max(1, round(n_images * val_ratio))
    # Ensure we don't exceed total
    if n_train + n_val > n_images:
        n_val = n_images - n_train
    n_test = n_images - n_train - n_val

    image_split: dict[str, str] = {}
    for i, img_name in enumerate(image_names):
        if i < n_train:
            image_split[img_name] = "train"
        elif i < n_train + n_val:
            image_split[img_name] = "valid"
        else:
            image_split[img_name] = "test"

    # Assign each tile to its source image's split
    assignment: dict[int, str] = {}
    for idx, tile in enumerate(tiles):
        assignment[idx] = image_split.get(tile.source_image, "train")

    return assignment


# ---------------------------------------------------------------------------
# Background tile sampling
# ---------------------------------------------------------------------------


def sample_background_tiles(
    tiles: list[TileRecord],
    split_assignment: dict[int, str],
    max_ratio: float,
    seed: int = 42,
) -> set[int]:
    """Determine which background tiles to keep based on max ratio.

    Limits the number of background (unannotated) tiles to at most
    ``max_ratio`` times the number of annotated tiles, per split.

    Args:
        tiles: All tile records.
        split_assignment: Dict mapping tile index to split name.
        max_ratio: Maximum background-to-annotated ratio.
        seed: Random seed.

    Returns:
        Set of tile indices to skip (excess background tiles).
    """
    rng = np.random.RandomState(seed)
    skip_set: set[int] = set()

    for split_name in ("train", "valid", "test"):
        annotated_indices = []
        background_indices = []

        for idx, tile in enumerate(tiles):
            if split_assignment.get(idx) != split_name:
                continue
            if tile.has_annotations:
                annotated_indices.append(idx)
            else:
                background_indices.append(idx)

        max_bg = int(len(annotated_indices) * max_ratio)

        if len(background_indices) > max_bg:
            rng.shuffle(background_indices)
            skip_set.update(background_indices[max_bg:])

    return skip_set


# ---------------------------------------------------------------------------
# COCO JSON assembly
# ---------------------------------------------------------------------------


def build_coco_json(
    tiles: list[TileRecord],
    tile_indices: list[int],
    categories: list[dict],
) -> dict:
    """Build a COCO-format annotation dict from tile records.

    Args:
        tiles: All tile records.
        tile_indices: Indices of tiles to include in this COCO JSON.
        categories: COCO categories list.

    Returns:
        COCO dict with images, annotations, and categories.
    """
    images = []
    annotations = []
    ann_id = 0

    for image_id, tile_idx in enumerate(tile_indices):
        tile = tiles[tile_idx]

        images.append({
            "id": image_id,
            "file_name": f"images/{tile.image_filename}",
            "width": tile.tile_width,
            "height": tile.tile_height,
        })

        for ann in tile.annotations:
            annotations.append({
                "id": ann_id,
                "image_id": image_id,
                "category_id": ann["category_id"],
                "bbox": ann["bbox"],
                "area": ann["area"],
                "iscrowd": ann.get("iscrowd", 0),
            })
            ann_id += 1

    return {
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }


# ---------------------------------------------------------------------------
# Checkpoint / resume
# ---------------------------------------------------------------------------


def load_checkpoint(output_dir: str) -> dict[str, Any]:
    """Load processing checkpoint from the output directory.

    Args:
        output_dir: Output directory containing the checkpoint file.

    Returns:
        Checkpoint dict with completed images and metadata, or empty dict.
    """
    ckpt_path = Path(output_dir) / CHECKPOINT_FILE
    if ckpt_path.exists():
        try:
            with open(ckpt_path) as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            pass
    return {}


def save_checkpoint(
    output_dir: str,
    completed_images: list[str],
    tile_records: list[dict],
    stats: dict[str, Any],
) -> None:
    """Save processing checkpoint.

    Args:
        output_dir: Output directory.
        completed_images: List of completed image filenames.
        tile_records: Serialized tile records from completed images.
        stats: Processing statistics.
    """
    ckpt_path = Path(output_dir) / CHECKPOINT_FILE
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "completed_images": completed_images,
        "tile_count": len(tile_records),
        "tile_records": tile_records,
        "stats": stats,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    with open(ckpt_path, "w") as f:
        json.dump(checkpoint, f)


def serialize_tile_record(tile: TileRecord) -> dict:
    """Convert a TileRecord to a JSON-serializable dict."""
    return {
        "image_filename": tile.image_filename,
        "tile_width": tile.tile_width,
        "tile_height": tile.tile_height,
        "source_image": tile.source_image,
        "global_col": tile.global_col,
        "global_row": tile.global_row,
        "annotations": tile.annotations,
        "has_annotations": tile.has_annotations,
    }


def deserialize_tile_record(d: dict) -> TileRecord:
    """Reconstruct a TileRecord from a serialized dict."""
    return TileRecord(
        image_filename=d["image_filename"],
        tile_width=d["tile_width"],
        tile_height=d["tile_height"],
        source_image=d["source_image"],
        global_col=d["global_col"],
        global_row=d["global_row"],
        annotations=d["annotations"],
        has_annotations=d["has_annotations"],
    )


# ---------------------------------------------------------------------------
# Image discovery and annotation indexing
# ---------------------------------------------------------------------------


def discover_images(input_dir: str) -> list[Path]:
    """Find all GeoTIFF images in the input directory tree.

    Searches recursively for .tif and .tiff files.

    Args:
        input_dir: Root directory to search.

    Returns:
        Sorted list of Path objects for discovered images.
    """
    input_path = Path(input_dir)
    images = []

    for ext in ("*.tif", "*.tiff", "*.TIF", "*.TIFF"):
        images.extend(input_path.rglob(ext))

    # Deduplicate and sort for deterministic ordering
    seen = set()
    unique = []
    for img in sorted(images):
        resolved = img.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(img)

    return unique


def build_annotation_index(
    features: list[dict],
) -> dict[str, list[dict]]:
    """Index annotations by image_id for fast per-image lookup.

    Builds a dict mapping image_id values to lists of annotation dicts
    with parsed pixel bounding boxes.

    Args:
        features: List of GeoJSON feature dicts from remapped GeoJSON.

    Returns:
        Dict mapping image_id string to list of annotation dicts.
    """
    index: dict[str, list[dict]] = {}

    for feat in features:
        props = feat.get("properties", {})
        image_id = str(props.get("image_id", ""))

        if not image_id:
            continue

        bounds_str = props.get("bounds_imcoords", "")
        if not bounds_str:
            continue

        try:
            parts = [float(x) for x in bounds_str.split(",")]
            if len(parts) != 4:
                continue
            xmin, ymin, xmax, ymax = parts
        except (ValueError, TypeError):
            continue

        if xmax <= xmin or ymax <= ymin:
            continue

        class_id = props.get("class_id", 0)
        class_name = props.get("class_name", XVIEW_CLASSES.get(int(class_id), "Unknown"))

        ann = {
            "class_id": int(class_id),
            "class_name": str(class_name),
            "bbox_pixel": [xmin, ymin, xmax, ymax],
        }

        if image_id not in index:
            index[image_id] = []
        index[image_id].append(ann)

    return index


def match_image_to_annotations(
    image_path: Path,
    annotation_index: dict[str, list[dict]],
) -> list[dict]:
    """Match a GeoTIFF file to its annotations via image_id lookup.

    xView image_id values typically match the image filename (with
    extension). This function tries several matching strategies:
    filename with extension, stem only, and full path basename.

    Args:
        image_path: Path to the GeoTIFF image.
        annotation_index: Pre-built annotation index from
            ``build_annotation_index()``.

    Returns:
        List of annotation dicts for this image (may be empty).
    """
    # Try exact filename match first (e.g., "123.tif")
    candidates = [
        image_path.name,
        str(image_path.name),
        image_path.stem,
    ]

    for candidate in candidates:
        if candidate in annotation_index:
            return annotation_index[candidate]

    return []


# ---------------------------------------------------------------------------
# Global grid offset computation
# ---------------------------------------------------------------------------


def compute_global_grid_offsets(
    images: list[Path],
    tile_size: int,
    overlap_ratio: float,
) -> list[tuple[int, int]]:
    """Compute global grid offsets for each image.

    Each image occupies a region of the global virtual grid. Images are
    laid out sequentially in a long horizontal strip so that spatial
    block splitting treats tiles from different images as spatially
    separated (preventing cross-image leakage within a block).

    Args:
        images: List of image paths.
        tile_size: Tile size in pixels.
        overlap_ratio: Overlap ratio between tiles.

    Returns:
        List of (col_offset, row_offset) tuples, one per image.
    """
    import rasterio

    stride = int(tile_size * (1.0 - overlap_ratio))
    if stride <= 0:
        stride = 1

    offsets = []
    current_col = 0

    for img_path in images:
        offsets.append((current_col, 0))

        # Estimate grid width for this image
        try:
            with rasterio.open(str(img_path)) as src:
                img_width = src.width
                n_cols = len(range(0, img_width, stride))
        except Exception:
            n_cols = 10  # Fallback estimate

        # Add gap between images to prevent cross-image spatial leakage
        current_col += n_cols + 3  # 3-tile gap between images

    return offsets


# ---------------------------------------------------------------------------
# Dry run estimation
# ---------------------------------------------------------------------------


def estimate_output(
    images: list[Path],
    annotation_index: dict[str, list[dict]],
    tile_size: int,
    overlap_ratio: float,
    workers: int,
) -> dict[str, Any]:
    """Estimate output size without processing.

    Samples a few images to estimate per-image tile count, then
    extrapolates to the full dataset.

    Args:
        images: All discovered image paths.
        annotation_index: Annotation index.
        tile_size: Tile size in pixels.
        overlap_ratio: Overlap ratio.
        workers: Number of workers.

    Returns:
        Estimation dict with tile counts, disk usage, and time.
    """
    import rasterio

    stride = int(tile_size * (1.0 - overlap_ratio))
    total_tiles = 0
    total_annotated_tiles = 0
    sample_count = min(20, len(images))

    # Sample images for estimation
    rng = np.random.RandomState(42)
    sample_indices = rng.choice(len(images), size=sample_count, replace=False)

    for idx in sample_indices:
        img_path = images[idx]
        try:
            with rasterio.open(str(img_path)) as src:
                n_cols = len(range(0, src.width, stride))
                n_rows = len(range(0, src.height, stride))
                tile_count = n_cols * n_rows
                total_tiles += tile_count
        except Exception:
            continue

        anns = match_image_to_annotations(img_path, annotation_index)
        if anns:
            total_annotated_tiles += max(1, tile_count // 4)

    # Extrapolate
    scale = len(images) / max(1, sample_count)
    est_tiles = int(total_tiles * scale)
    est_annotated = int(total_annotated_tiles * scale)

    # Estimate disk usage: ~50 KB per JPEG tile (576x576 at quality 95)
    est_disk_gb = (est_tiles * 50 * 1024) / (1024 ** 3)

    # Estimate time: ~0.5 sec per tile on a single core, divided by workers
    est_seconds = (est_tiles * 0.5) / max(1, workers)
    est_hours = est_seconds / 3600

    # Count total annotations
    total_annotations = sum(len(v) for v in annotation_index.values())
    matched_images = sum(
        1 for img in images
        if match_image_to_annotations(img, annotation_index)
    )

    return {
        "num_images": len(images),
        "matched_images": matched_images,
        "total_annotations": total_annotations,
        "estimated_tiles": est_tiles,
        "estimated_annotated_tiles": est_annotated,
        "estimated_disk_gb": round(est_disk_gb, 1),
        "estimated_hours": round(est_hours, 1),
        "workers": workers,
    }


# ---------------------------------------------------------------------------
# COCO validation
# ---------------------------------------------------------------------------


def validate_coco_json(json_path: str) -> tuple[bool, str]:
    """Validate a COCO JSON file using pycocotools.

    Args:
        json_path: Path to the COCO JSON file.

    Returns:
        Tuple of (is_valid, message).
    """
    try:
        from pycocotools.coco import COCO
        import io
        import contextlib

        # Suppress pycocotools print output
        with contextlib.redirect_stdout(io.StringIO()):
            coco = COCO(json_path)

        n_images = len(coco.getImgIds())
        n_annotations = len(coco.getAnnIds())
        n_categories = len(coco.getCatIds())

        return True, (
            f"Valid COCO JSON: {n_images} images, "
            f"{n_annotations} annotations, {n_categories} categories"
        )

    except ImportError:
        # pycocotools not installed; do basic validation
        try:
            with open(json_path) as f:
                data = json.load(f)

            required_keys = {"images", "annotations", "categories"}
            missing = required_keys - set(data.keys())
            if missing:
                return False, f"Missing keys: {missing}"

            return True, (
                f"Basic validation passed: {len(data['images'])} images, "
                f"{len(data['annotations'])} annotations, "
                f"{len(data['categories'])} categories "
                f"(install pycocotools for full validation)"
            )
        except Exception as exc:
            return False, f"Invalid JSON: {exc}"

    except Exception as exc:
        return False, f"Validation failed: {exc}"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Argument list (defaults to sys.argv[1:]).

    Returns:
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Process xView dataset: tile GeoTIFFs into COCO-format "
            "training dataset with spatial splitting."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Directory containing xView GeoTIFF images",
    )
    parser.add_argument(
        "--annotations",
        type=str,
        required=True,
        help="Path to remapped GeoJSON from preprocess_xview.py",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for COCO-format dataset",
    )
    parser.add_argument(
        "--tile_size",
        type=int,
        default=DEFAULT_TILE_SIZE,
        help=f"Tile size in pixels (default: {DEFAULT_TILE_SIZE})",
    )
    parser.add_argument(
        "--overlap",
        type=float,
        default=DEFAULT_OVERLAP,
        help=f"Overlap ratio between tiles (default: {DEFAULT_OVERLAP})",
    )
    parser.add_argument(
        "--min_area",
        type=float,
        default=DEFAULT_MIN_ANNOTATION_AREA,
        help=f"Minimum annotation area in square pixels (default: {DEFAULT_MIN_ANNOTATION_AREA})",
    )
    parser.add_argument(
        "--max_background_ratio",
        type=float,
        default=DEFAULT_MAX_BACKGROUND_RATIO,
        help=f"Max background-to-annotated tile ratio (default: {DEFAULT_MAX_BACKGROUND_RATIO})",
    )
    parser.add_argument(
        "--num_classes",
        type=int,
        default=DEFAULT_NUM_CLASSES,
        help=(
            f"Number of valid classes (default: {DEFAULT_NUM_CLASSES}). "
            "Annotations with class_id >= num_classes are filtered out "
            "(e.g., background class from preprocessing)."
        ),
    )
    parser.add_argument(
        "--split_ratios",
        type=str,
        default="0.80,0.15,0.05",
        help="Train/valid/test split ratios (default: 0.80,0.15,0.05)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, multiprocessing.cpu_count() - 1),
        help=f"Number of parallel workers (default: CPU count - 1 = {max(1, multiprocessing.cpu_count() - 1)})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Estimate output without writing files",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from checkpoint, skipping already-processed images",
    )
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    """Run the xView dataset processing pipeline.

    Args:
        argv: Command-line arguments (defaults to sys.argv[1:]).

    Returns:
        Exit code: 0 on success, 1 on failure.
    """
    args = parse_args(argv)

    # Parse split ratios
    try:
        ratios = tuple(float(x) for x in args.split_ratios.split(","))
        if len(ratios) != 3:
            raise ValueError("Need exactly 3 ratios")
        if not math.isclose(sum(ratios), 1.0, rel_tol=1e-3):
            raise ValueError(f"Ratios must sum to 1.0, got {sum(ratios):.3f}")
    except Exception as exc:
        print(f"  ERROR: Invalid split ratios: {exc}")
        return 1

    print("=" * 64)
    print("  xView Dataset Processing: GeoTIFF Tiling + COCO Conversion")
    print("=" * 64)

    # -------------------------------------------------------------------
    # Validate inputs
    # -------------------------------------------------------------------
    input_dir = Path(args.input_dir)
    annotations_path = Path(args.annotations)
    output_dir = Path(args.output_dir)

    if not input_dir.exists():
        print(f"\n  ERROR: Input directory not found: {args.input_dir}")
        return 1

    if not annotations_path.exists():
        print(f"\n  ERROR: Annotations file not found: {args.annotations}")
        return 1

    print(f"\n  Input directory: {args.input_dir}")
    print(f"  Annotations: {args.annotations}")
    print(f"  Output directory: {args.output_dir}")
    print(f"  Tile size: {args.tile_size}px")
    print(f"  Overlap: {args.overlap}")
    print(f"  Min annotation area: {args.min_area} sq px")
    print(f"  Max background ratio: {args.max_background_ratio}")
    print(f"  Num classes: {args.num_classes} (class_id >= {args.num_classes} filtered out)")
    print(f"  Split ratios: {ratios}")
    print(f"  Split strategy: image-level (all tiles from an image go to same split)")
    print(f"  Workers: {args.workers}")
    print(f"  Seed: {args.seed}")

    # -------------------------------------------------------------------
    # [1/7] Discover images
    # -------------------------------------------------------------------
    print(f"\n[1/7] Discovering GeoTIFF images...")
    t0 = time.time()

    images = discover_images(args.input_dir)
    discover_time = time.time() - t0

    if not images:
        print(f"  ERROR: No GeoTIFF images found in {args.input_dir}")
        return 1

    print(f"  Found {len(images)} GeoTIFF images in {discover_time:.1f}s")

    # -------------------------------------------------------------------
    # [2/7] Load and index annotations
    # -------------------------------------------------------------------
    print(f"\n[2/7] Loading annotations...")
    t0 = time.time()

    ann_size_mb = annotations_path.stat().st_size / (1024 ** 2)
    print(f"  File size: {ann_size_mb:.1f} MB")

    try:
        with open(annotations_path) as f:
            geojson_data = json.load(f)
    except json.JSONDecodeError as exc:
        print(f"  ERROR: Invalid JSON: {exc}")
        return 1
    except MemoryError:
        print("  ERROR: File too large to load in memory.")
        return 1

    features = geojson_data.get("features", [])
    load_time = time.time() - t0
    print(f"  Loaded {len(features):,} features in {load_time:.1f}s")

    if not features:
        print("  ERROR: No features found in GeoJSON")
        return 1

    # Build annotation index for fast per-image lookup
    print(f"  Building annotation index...")
    t0 = time.time()
    annotation_index = build_annotation_index(features)
    index_time = time.time() - t0
    total_indexed = sum(len(v) for v in annotation_index.values())
    print(f"  Indexed {total_indexed:,} annotations across {len(annotation_index)} images in {index_time:.1f}s")

    # Filter out annotations with class_id >= num_classes (e.g., background)
    num_classes = args.num_classes
    filtered_count = 0
    for image_id in annotation_index:
        original = annotation_index[image_id]
        filtered = [a for a in original if a["class_id"] < num_classes]
        filtered_count += len(original) - len(filtered)
        annotation_index[image_id] = filtered

    if filtered_count > 0:
        remaining = sum(len(v) for v in annotation_index.values())
        print(f"  Filtered {filtered_count:,} annotations with class_id >= {num_classes} "
              f"({remaining:,} remaining)")

    # Free raw features to save memory
    del features
    del geojson_data

    # -------------------------------------------------------------------
    # Dry run: estimate and exit
    # -------------------------------------------------------------------
    if args.dry_run:
        print(f"\n[DRY RUN] Estimating output...")
        est = estimate_output(
            images, annotation_index, args.tile_size, args.overlap, args.workers
        )
        print(f"\n  {'Metric':<35} {'Value':>15}")
        print(f"  {'-'*35} {'-'*15}")
        print(f"  {'GeoTIFF images':<35} {est['num_images']:>15,}")
        print(f"  {'Images with annotations':<35} {est['matched_images']:>15,}")
        print(f"  {'Total annotations (indexed)':<35} {est['total_annotations']:>15,}")
        print(f"  {'Estimated tiles':<35} {est['estimated_tiles']:>15,}")
        print(f"  {'Estimated annotated tiles':<35} {est['estimated_annotated_tiles']:>15,}")
        print(f"  {'Estimated disk usage':<35} {est['estimated_disk_gb']:>14.1f} GB")
        print(f"  {'Estimated processing time':<35} {est['estimated_hours']:>13.1f} hrs")
        print(f"  {'Workers':<35} {est['workers']:>15}")

        print(f"\n  Split estimate (before background sampling):")
        train_pct, val_pct, test_pct = ratios
        print(f"    Train: ~{int(est['estimated_tiles'] * train_pct):,} tiles ({train_pct*100:.0f}%)")
        print(f"    Valid: ~{int(est['estimated_tiles'] * val_pct):,} tiles ({val_pct*100:.0f}%)")
        print(f"    Test:  ~{int(est['estimated_tiles'] * test_pct):,} tiles ({test_pct*100:.0f}%)")

        print(f"\n  [DRY RUN] No files written.")
        return 0

    # -------------------------------------------------------------------
    # [3/7] Load checkpoint (if resuming)
    # -------------------------------------------------------------------
    completed_images: set[str] = set()
    resumed_tiles: list[TileRecord] = []

    if args.resume:
        print(f"\n[3/7] Loading checkpoint...")
        checkpoint = load_checkpoint(args.output_dir)

        if checkpoint:
            completed_images = set(checkpoint.get("completed_images", []))
            resumed_tile_dicts = checkpoint.get("tile_records", [])
            resumed_tiles = [deserialize_tile_record(d) for d in resumed_tile_dicts]
            print(f"  Resuming: {len(completed_images)} images already processed")
            print(f"  Resumed tiles: {len(resumed_tiles):,}")
        else:
            print(f"  No checkpoint found, starting fresh")
    else:
        print(f"\n[3/7] Starting fresh (no --resume flag)")

    # Filter out already-completed images
    images_to_process = [
        img for img in images
        if img.stem not in completed_images
    ]
    print(f"  Images to process: {len(images_to_process)} (of {len(images)} total)")

    # -------------------------------------------------------------------
    # [4/7] Compute global grid offsets
    # -------------------------------------------------------------------
    print(f"\n[4/7] Computing global grid layout...")
    t0 = time.time()

    # Compute offsets for ALL images (including resumed ones) so the
    # global grid is consistent across runs
    all_grid_offsets = compute_global_grid_offsets(
        images, args.tile_size, args.overlap
    )
    grid_time = time.time() - t0

    # Build lookup from image stem to offset
    image_offset_map: dict[str, tuple[int, int]] = {}
    for img_path, offset in zip(images, all_grid_offsets):
        image_offset_map[img_path.stem] = offset

    print(f"  Grid layout computed in {grid_time:.1f}s")

    # -------------------------------------------------------------------
    # [5/7] Process images in parallel
    # -------------------------------------------------------------------
    print(f"\n[5/7] Processing {len(images_to_process)} images with {args.workers} workers...")

    # Create output structure
    output_dir.mkdir(parents=True, exist_ok=True)
    staging_dir = output_dir / "_staging" / "images"
    staging_dir.mkdir(parents=True, exist_ok=True)

    all_tiles = list(resumed_tiles)  # Start with resumed tiles
    completed_list = list(completed_images)
    errors: list[tuple[str, str]] = []
    total_processed = len(completed_images)
    total_to_process = len(images_to_process)

    pipeline_start = time.time()

    if total_to_process > 0:
        try:
            from tqdm import tqdm
            progress_bar = tqdm(
                total=total_to_process,
                desc="  Processing images",
                unit="img",
                ncols=80,
            )
        except ImportError:
            progress_bar = None
            print(f"  (Install tqdm for progress bars: pip install tqdm)")

        # Process in batches to enable periodic checkpointing
        batch_size = max(1, min(50, total_to_process))
        checkpoint_interval = max(10, batch_size)
        images_since_checkpoint = 0

        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            # Submit all jobs
            future_to_image = {}
            for img_path in images_to_process:
                offset = image_offset_map.get(img_path.stem, (0, 0))
                anns = match_image_to_annotations(img_path, annotation_index)

                future = executor.submit(
                    process_single_image,
                    str(img_path),
                    anns,
                    str(output_dir),
                    args.tile_size,
                    args.overlap,
                    args.min_area,
                    offset[0],
                    offset[1],
                )
                future_to_image[future] = img_path

            # Collect results as they complete
            for future in as_completed(future_to_image):
                img_path = future_to_image[future]

                try:
                    result = future.result()
                except Exception as exc:
                    error_msg = f"{type(exc).__name__}: {exc}"
                    errors.append((img_path.name, error_msg))
                    print(f"\n  WARNING: Failed to process {img_path.name}: {error_msg}")
                    if progress_bar:
                        progress_bar.update(1)
                    continue

                if result.error:
                    errors.append((result.image_name, result.error))
                    print(f"\n  WARNING: Error processing {result.image_name}: {result.error}")
                else:
                    all_tiles.extend(result.tiles)
                    completed_list.append(result.image_name)

                if progress_bar:
                    progress_bar.update(1)

                images_since_checkpoint += 1

                # Periodic checkpoint
                if images_since_checkpoint >= checkpoint_interval:
                    save_checkpoint(
                        str(output_dir),
                        completed_list,
                        [serialize_tile_record(t) for t in all_tiles],
                        {"processed_images": len(completed_list), "total_tiles": len(all_tiles)},
                    )
                    images_since_checkpoint = 0

        if progress_bar:
            progress_bar.close()

        # Final checkpoint
        save_checkpoint(
            str(output_dir),
            completed_list,
            [serialize_tile_record(t) for t in all_tiles],
            {"processed_images": len(completed_list), "total_tiles": len(all_tiles)},
        )

    process_time = time.time() - pipeline_start
    print(f"\n  Processed {len(completed_list)} images in {process_time:.1f}s ({process_time/60:.1f} min)")
    print(f"  Total tiles generated: {len(all_tiles):,}")

    if errors:
        print(f"  Errors: {len(errors)} images failed (see warnings above)")

    if not all_tiles:
        print("  ERROR: No tiles were generated. Check input data.")
        return 1

    # -------------------------------------------------------------------
    # [6/7] Spatial split + background sampling + move tiles
    # -------------------------------------------------------------------
    print(f"\n[6/7] Applying image-level split...")
    t0 = time.time()

    # Image-level splitting: assign entire source images to splits,
    # preventing spatial leakage with zero discarded tiles
    split_assignment = image_level_split(
        all_tiles,
        ratios=ratios,
        seed=args.seed,
    )

    # Count split assignments
    split_counts: dict[str, int] = {"train": 0, "valid": 0, "test": 0}
    for split_name in split_assignment.values():
        split_counts[split_name] = split_counts.get(split_name, 0) + 1

    # Count unique source images per split
    split_images: dict[str, set[str]] = {"train": set(), "valid": set(), "test": set()}
    for idx, tile in enumerate(all_tiles):
        sn = split_assignment.get(idx, "train")
        if sn in split_images:
            split_images[sn].add(tile.source_image)

    print(f"  Split assignment (before background sampling):")
    print(f"    Train:  {split_counts.get('train', 0):,} tiles ({len(split_images['train'])} images)")
    print(f"    Valid:  {split_counts.get('valid', 0):,} tiles ({len(split_images['valid'])} images)")
    print(f"    Test:   {split_counts.get('test', 0):,} tiles ({len(split_images['test'])} images)")

    # Background tile sampling
    skip_set = sample_background_tiles(
        all_tiles,
        split_assignment,
        max_ratio=args.max_background_ratio,
        seed=args.seed,
    )
    print(f"  Background tiles skipped: {len(skip_set):,}")

    # Create output directories
    for split_name in ("train", "valid", "test"):
        (output_dir / split_name / "images").mkdir(parents=True, exist_ok=True)

    # Move tile images from staging to split directories
    print(f"  Moving tiles to split directories...")
    t_move = time.time()

    split_tile_indices: dict[str, list[int]] = {"train": [], "valid": [], "test": []}
    skipped_count = 0
    moved_count = 0
    missing_count = 0

    for idx, tile in enumerate(all_tiles):
        split_name = split_assignment.get(idx, "train")

        if idx in skip_set:
            skipped_count += 1
            continue

        split_tile_indices[split_name].append(idx)

        # Move file from staging to split directory
        src_path = staging_dir / tile.image_filename
        dst_path = output_dir / split_name / "images" / tile.image_filename

        if src_path.exists():
            src_path.rename(dst_path)
            moved_count += 1
        elif dst_path.exists():
            # Already in place (from a previous run)
            moved_count += 1
        else:
            missing_count += 1

    move_time = time.time() - t_move
    print(f"  Moved {moved_count:,} tiles in {move_time:.1f}s")
    if missing_count > 0:
        print(f"  WARNING: {missing_count} tile images not found (may need re-processing)")

    # Clean up staging directory
    try:
        staging_parent = output_dir / "_staging"
        if staging_parent.exists():
            import shutil
            shutil.rmtree(str(staging_parent))
    except Exception:
        pass  # Non-critical cleanup

    split_time = time.time() - t0

    # -------------------------------------------------------------------
    # [7/7] Write COCO JSON files and validate
    # -------------------------------------------------------------------
    print(f"\n[7/7] Writing COCO JSON files...")
    t0 = time.time()

    # Build categories list -- ALWAYS include all 5 classes
    # even if some have zero annotations in this split.
    # This prevents class_id mismatches during training when a rare
    # class (e.g., Bus) has no annotations in a small validation set.
    categories = []
    for cid, name in sorted(XVIEW_CLASSES.items()):
        categories.append({
            "id": cid,
            "name": name,
            "supercategory": "vehicle",
        })

    # Write per-split COCO JSONs
    total_annotations = 0
    for split_name in ("train", "valid", "test"):
        indices = split_tile_indices[split_name]
        coco = build_coco_json(all_tiles, indices, categories)

        coco_path = output_dir / split_name / "_annotations.coco.json"
        with open(coco_path, "w") as f:
            json.dump(coco, f, indent=2)

        n_imgs = len(coco["images"])
        n_anns = len(coco["annotations"])
        total_annotations += n_anns
        print(f"  {split_name}: {n_imgs:,} images, {n_anns:,} annotations -> {coco_path.name}")

    write_time = time.time() - t0
    print(f"  COCO JSONs written in {write_time:.1f}s")

    # Validate COCO JSONs
    print(f"\n  Validating COCO JSONs...")
    all_valid = True
    for split_name in ("train", "valid", "test"):
        coco_path = output_dir / split_name / "_annotations.coco.json"
        valid, msg = validate_coco_json(str(coco_path))
        status = "OK" if valid else "FAIL"
        print(f"    {split_name}: [{status}] {msg}")
        if not valid:
            all_valid = False

    # -------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------
    total_time = time.time() - pipeline_start
    total_time_min = total_time / 60

    # Compute per-class annotation counts
    class_counts: dict[int, int] = {}
    for split_name in ("train", "valid", "test"):
        for idx in split_tile_indices[split_name]:
            for ann in all_tiles[idx].annotations:
                cid = ann["category_id"]
                class_counts[cid] = class_counts.get(cid, 0) + 1

    print(f"\n{'=' * 64}")
    print(f"  PROCESSING COMPLETE")
    print(f"{'=' * 64}")
    print(f"\n  Input:")
    print(f"    Images: {len(images)}")
    print(f"    Images processed: {len(completed_list)}")
    if errors:
        print(f"    Errors: {len(errors)}")
    print(f"\n  Output:")
    print(f"    Total tiles: {sum(len(v) for v in split_tile_indices.values()):,}")
    print(f"    Total annotations: {total_annotations:,}")
    print(f"    Train: {len(split_tile_indices['train']):,} tiles ({len(split_images['train'])} images)")
    print(f"    Valid: {len(split_tile_indices['valid']):,} tiles ({len(split_images['valid'])} images)")
    print(f"    Test:  {len(split_tile_indices['test']):,} tiles ({len(split_images['test'])} images)")
    print(f"    Background skipped: {len(skip_set):,} tiles")

    if class_counts:
        print(f"\n  Annotation distribution:")
        for cid in sorted(class_counts.keys()):
            name = XVIEW_CLASSES.get(cid, f"class_{cid}")
            count = class_counts[cid]
            pct = (count / total_annotations * 100) if total_annotations > 0 else 0
            print(f"    {name} (id={cid}): {count:,} ({pct:.1f}%)")

    print(f"\n  Timing:")
    print(f"    Total: {total_time_min:.1f} minutes ({total_time/3600:.2f} hours)")
    print(f"    Processing: {process_time/60:.1f} min")
    print(f"    Split + move: {split_time:.1f}s")
    print(f"    COCO write: {write_time:.1f}s")

    print(f"\n  Output directory: {args.output_dir}")
    print(f"    {args.output_dir}/train/_annotations.coco.json")
    print(f"    {args.output_dir}/valid/_annotations.coco.json")
    print(f"    {args.output_dir}/test/_annotations.coco.json")

    if not all_valid:
        print(f"\n  WARNING: Some COCO JSONs failed validation.")

    # Remove checkpoint file on successful completion
    ckpt_path = output_dir / CHECKPOINT_FILE
    if ckpt_path.exists():
        try:
            ckpt_path.unlink()
        except Exception:
            pass

    print(f"\n  Next step:")
    print(f"    python scripts/train_xview.py \\")
    print(f"      --dataset_dir {args.output_dir}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
