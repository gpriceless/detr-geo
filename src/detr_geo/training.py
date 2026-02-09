"""Training pipeline: spatial splitting, dataset preparation, export, augmentation.

This module provides the full geospatial training pipeline for RF-DETR:
- SpatialSplitter: block-based or random train/val/test splitting with buffer zones
- prepare_training_dataset(): GeoTIFF + GeoJSON to COCO-format image chips
- detections_to_coco() / detections_to_yolo(): active learning feedback loop
- AugmentationPreset + train(): thin wrapper delegating to RF-DETR's training API
"""

from __future__ import annotations

import json
import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
from shapely.geometry import MultiPolygon, Polygon

from detr_geo.exceptions import CRSError, ExportError, ModelError

# ---------------------------------------------------------------------------
# Spatial Splitting
# ---------------------------------------------------------------------------


@dataclass
class SplitResult:
    """Result of spatial splitting."""

    train_indices: list[int]
    val_indices: list[int]
    test_indices: list[int]
    buffer_indices: list[int]  # discarded tiles in buffer zones


class SpatialSplitter:
    """Split tiles into train/val/test with spatial awareness.

    Supports two methods:
    - "block": Assigns contiguous spatial blocks to splits, with buffer
      zones between blocks to prevent spatial data leakage. Default.
    - "random": Random assignment with a UserWarning about leakage risk.

    Note: "scene" method is deferred until multi-raster support lands.
    """

    VALID_METHODS = ("block", "random")

    def __init__(
        self,
        method: str = "block",
        ratios: tuple[float, float, float] = (0.8, 0.15, 0.05),
        buffer_tiles: int = 1,
        seed: int = 42,
    ) -> None:
        """Initialize the splitter.

        Args:
            method: Splitting method - "block" or "random".
            ratios: (train, val, test) ratios. Must sum to 1.0.
            buffer_tiles: Number of buffer tiles between blocks (block method only).
            seed: Random seed for reproducibility.

        Raises:
            ValueError: If method is invalid or ratios don't sum to 1.0.
        """
        if method not in self.VALID_METHODS:
            raise ValueError(f"Invalid splitting method '{method}'. Valid methods: {', '.join(self.VALID_METHODS)}")

        ratio_sum = sum(ratios)
        if not math.isclose(ratio_sum, 1.0, rel_tol=1e-6):
            raise ValueError(f"Split ratios must sum to 1.0, got {ratios} (sum={ratio_sum:.6f})")

        if len(ratios) != 3:
            raise ValueError(f"Expected 3 ratios (train, val, test), got {len(ratios)}")

        for r in ratios:
            if r < 0:
                raise ValueError(f"Split ratios must be non-negative, got {ratios}")

        self._method = method
        self._ratios = ratios
        self._buffer_tiles = buffer_tiles
        self._seed = seed

    def split(
        self,
        tiles: list[dict],
        raster_width: int,
        raster_height: int,
        tile_size: int,
    ) -> SplitResult:
        """Split tile indices into train/val/test groups.

        Args:
            tiles: List of tile dicts, each with a 'window' key
                containing (col_off, row_off, width, height).
            raster_width: Full raster width in pixels.
            raster_height: Full raster height in pixels.
            tile_size: Size of each tile in pixels.

        Returns:
            SplitResult with index lists for each split.
        """
        if self._method == "block":
            return self._block_split(tiles, raster_width, raster_height, tile_size)
        else:
            return self._random_split(tiles)

    def _block_split(
        self,
        tiles: list[dict],
        raster_width: int,
        raster_height: int,
        tile_size: int,
    ) -> SplitResult:
        """Block-based spatial splitting.

        Divides the raster into a grid of blocks, assigns blocks to splits,
        then identifies buffer tiles at block boundaries.
        """
        rng = np.random.RandomState(self._seed)
        n_tiles = len(tiles)

        if n_tiles == 0:
            return SplitResult(train_indices=[], val_indices=[], test_indices=[], buffer_indices=[])

        # Compute grid dimensions in tiles
        cols_in_grid = max(1, raster_width // tile_size)
        rows_in_grid = max(1, raster_height // tile_size)

        # Build a mapping from (grid_col, grid_row) to tile index
        tile_grid_map: dict[tuple[int, int], int] = {}
        for idx, tile in enumerate(tiles):
            window = tile["window"]
            col_off, row_off = window[0], window[1]
            gc = col_off // tile_size
            gr = row_off // tile_size
            tile_grid_map[(gc, gr)] = idx

        # Determine block size: aim for ~3x3 blocks minimum
        # We want enough blocks to distribute among 3 splits
        n_blocks_target = max(3, int(math.sqrt(n_tiles)))
        block_cols = max(1, int(math.ceil(cols_in_grid / max(1, int(math.sqrt(n_blocks_target))))))
        block_rows = max(1, int(math.ceil(rows_in_grid / max(1, int(math.sqrt(n_blocks_target))))))

        # Number of blocks along each axis
        n_blocks_x = max(1, int(math.ceil(cols_in_grid / block_cols)))
        n_blocks_y = max(1, int(math.ceil(rows_in_grid / block_rows)))

        # Assign each tile to a block
        tile_block: dict[int, tuple[int, int]] = {}
        for (gc, gr), tile_idx in tile_grid_map.items():
            bx = min(gc // block_cols, n_blocks_x - 1)
            by = min(gr // block_rows, n_blocks_y - 1)
            tile_block[tile_idx] = (bx, by)

        # Build list of unique blocks
        unique_blocks = sorted(set(tile_block.values()))
        n_blocks = len(unique_blocks)

        # Shuffle and assign blocks to splits
        block_order = list(range(n_blocks))
        rng.shuffle(block_order)

        train_ratio, val_ratio, _test_ratio = self._ratios
        n_train_blocks = max(1, round(n_blocks * train_ratio))
        n_val_blocks = max(0, round(n_blocks * val_ratio))
        # Ensure we don't exceed total
        if n_train_blocks + n_val_blocks > n_blocks:
            n_val_blocks = n_blocks - n_train_blocks
        # Handle small rasters
        if n_blocks <= 2:
            n_train_blocks = n_blocks
            n_val_blocks = 0
            if n_blocks <= 2 and (val_ratio > 0 or _test_ratio > 0):
                warnings.warn(
                    f"Only {n_blocks} spatial block(s) available. "
                    f"All tiles assigned to train. Val/test splits are empty.",
                    UserWarning,
                    stacklevel=3,
                )

        block_split_map: dict[tuple[int, int], str] = {}
        for i, bi in enumerate(block_order):
            block_id = unique_blocks[bi]
            if i < n_train_blocks:
                block_split_map[block_id] = "train"
            elif i < n_train_blocks + n_val_blocks:
                block_split_map[block_id] = "val"
            else:
                block_split_map[block_id] = "test"

        # Identify buffer tiles: tiles whose block is different from a neighbor
        buffer_set: set[int] = set()
        if self._buffer_tiles > 0:
            for tile_idx, (bx, by) in tile_block.items():
                split_label = block_split_map[(bx, by)]
                # Check if any neighboring tile (in grid) belongs to a different split
                window = tiles[tile_idx]["window"]
                gc = window[0] // tile_size
                gr = window[1] // tile_size

                is_boundary = False
                for dx in range(-self._buffer_tiles, self._buffer_tiles + 1):
                    for dy in range(-self._buffer_tiles, self._buffer_tiles + 1):
                        if dx == 0 and dy == 0:
                            continue
                        ngc, ngr = gc + dx, gr + dy
                        if (ngc, ngr) in tile_grid_map:
                            neighbor_idx = tile_grid_map[(ngc, ngr)]
                            if neighbor_idx in tile_block:
                                nbx, nby = tile_block[neighbor_idx]
                                if block_split_map.get((nbx, nby)) != split_label:
                                    is_boundary = True
                                    break
                    if is_boundary:
                        break
                if is_boundary:
                    buffer_set.add(tile_idx)

        # Assign tiles to splits
        train_indices: list[int] = []
        val_indices: list[int] = []
        test_indices: list[int] = []
        buffer_indices: list[int] = sorted(buffer_set)

        for tile_idx in range(n_tiles):
            if tile_idx in buffer_set:
                continue
            if tile_idx not in tile_block:
                # Tile not in grid map (e.g., edge tile) - assign to train
                train_indices.append(tile_idx)
                continue
            block_id = tile_block[tile_idx]
            split_label = block_split_map[block_id]
            if split_label == "train":
                train_indices.append(tile_idx)
            elif split_label == "val":
                val_indices.append(tile_idx)
            else:
                test_indices.append(tile_idx)

        return SplitResult(
            train_indices=train_indices,
            val_indices=val_indices,
            test_indices=test_indices,
            buffer_indices=buffer_indices,
        )

    def _random_split(self, tiles: list[dict]) -> SplitResult:
        """Random splitting with leakage warning."""
        warnings.warn(
            "Random splitting does not prevent spatial data leakage. "
            "Nearby tiles may appear in both train and validation sets, "
            "inflating validation metrics. Use method='block' for "
            "geospatial datasets.",
            UserWarning,
            stacklevel=3,
        )
        rng = np.random.RandomState(self._seed)
        n = len(tiles)
        indices = np.arange(n)
        rng.shuffle(indices)

        train_ratio, val_ratio, _test_ratio = self._ratios
        n_train = round(n * train_ratio)
        n_val = round(n * val_ratio)
        # Clamp
        if n_train + n_val > n:
            n_val = n - n_train

        train_indices = sorted(indices[:n_train].tolist())
        val_indices = sorted(indices[n_train : n_train + n_val].tolist())
        test_indices = sorted(indices[n_train + n_val :].tolist())

        return SplitResult(
            train_indices=train_indices,
            val_indices=val_indices,
            test_indices=test_indices,
            buffer_indices=[],  # no buffer in random mode
        )


# ---------------------------------------------------------------------------
# Detection Export (Active Learning Feedback Loop)
# ---------------------------------------------------------------------------


def detections_to_coco(
    gdf: gpd.GeoDataFrame,
    output_path: str,
    image_path: str | None = None,
    image_width: int | None = None,
    image_height: int | None = None,
) -> None:
    """Export a detection GeoDataFrame to COCO annotation JSON format.

    Converts detection polygons to pixel-space bounding boxes in COCO
    [x, y, width, height] format. This closes the active learning loop:
    detect -> human review -> correct -> retrain.

    Args:
        gdf: GeoDataFrame with geometry, class_id, and class_name columns.
        output_path: Path to write the COCO JSON file.
        image_path: Optional path to the source image (stored in COCO images array).
        image_width: Image width in pixels. Required for coordinate conversion.
        image_height: Image height in pixels. Required for coordinate conversion.

    Raises:
        ExportError: If writing the JSON file fails.
    """
    try:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)

        # Build images array
        images = []
        if image_width is not None and image_height is not None:
            img_entry: dict[str, int | str] = {
                "id": 0,
                "width": image_width,
                "height": image_height,
            }
            if image_path is not None:
                img_entry["file_name"] = str(Path(image_path).name)
            else:
                img_entry["file_name"] = "image.jpg"
            images.append(img_entry)

        # Build categories array from unique class_ids/class_names
        categories = []
        if len(gdf) > 0 and "class_id" in gdf.columns:
            if "class_name" in gdf.columns:
                class_pairs = gdf[["class_id", "class_name"]].drop_duplicates().sort_values("class_id")
                for _, row in class_pairs.iterrows():
                    categories.append({"id": int(row["class_id"]), "name": str(row["class_name"])})
            else:
                for cid in sorted(gdf["class_id"].unique()):
                    categories.append({"id": int(cid), "name": f"class_{cid}"})

        # Build annotations array
        annotations = []
        for ann_id, (_idx, row) in enumerate(gdf.iterrows()):
            geom = row.geometry
            if geom is None or geom.is_empty:
                continue

            bounds = geom.bounds  # (minx, miny, maxx, maxy)
            x = float(bounds[0])
            y = float(bounds[1])
            w = float(bounds[2] - bounds[0])
            h = float(bounds[3] - bounds[1])

            ann = {
                "id": ann_id,
                "image_id": 0,
                "category_id": int(row.get("class_id", 0)),
                "bbox": [x, y, w, h],
                "area": w * h,
                "iscrowd": 0,
            }
            annotations.append(ann)

        coco = {
            "images": images,
            "categories": categories,
            "annotations": annotations,
        }

        with open(output, "w") as f:
            json.dump(coco, f, indent=2)

    except (OSError, PermissionError) as exc:
        raise ExportError(f"Failed to write COCO JSON to '{output_path}': {exc}") from exc


def detections_to_yolo(
    gdf: gpd.GeoDataFrame,
    output_dir: str,
    image_name: str = "image",
    image_width: int | None = None,
    image_height: int | None = None,
) -> None:
    """Export a detection GeoDataFrame to YOLO annotation format.

    Writes one .txt file per image with normalized bounding boxes:
    class_id center_x center_y width height (all normalized to [0, 1]).

    Args:
        gdf: GeoDataFrame with geometry and class_id columns.
        output_dir: Directory to write the .txt file(s).
        image_name: Base name for the output file (without extension).
        image_width: Image width in pixels for normalization.
        image_height: Image height in pixels for normalization.

    Raises:
        ExportError: If writing fails or dimensions are missing.
    """
    try:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        output_file = out_dir / f"{image_name}.txt"

        if len(gdf) == 0 or image_width is None or image_height is None:
            # Write empty file
            output_file.write_text("")
            return

        lines = []
        for _, row in gdf.iterrows():
            geom = row.geometry
            if geom is None or geom.is_empty:
                continue

            bounds = geom.bounds  # (minx, miny, maxx, maxy)
            x_min, y_min, x_max, y_max = bounds

            # Convert to YOLO format: center_x, center_y, width, height (normalized)
            cx = ((x_min + x_max) / 2.0) / image_width
            cy = ((y_min + y_max) / 2.0) / image_height
            w = (x_max - x_min) / image_width
            h = (y_max - y_min) / image_height

            # Clamp to [0, 1]
            cx = max(0.0, min(1.0, cx))
            cy = max(0.0, min(1.0, cy))
            w = max(0.0, min(1.0, w))
            h = max(0.0, min(1.0, h))

            class_id = int(row.get("class_id", 0))
            lines.append(f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

        output_file.write_text("\n".join(lines) + ("\n" if lines else ""))

    except (OSError, PermissionError) as exc:
        raise ExportError(f"Failed to write YOLO annotations to '{output_dir}': {exc}") from exc


# ---------------------------------------------------------------------------
# Dataset Preparation Pipeline
# ---------------------------------------------------------------------------


def _clip_annotations_to_tile(
    annotations: gpd.GeoDataFrame,
    tile_polygon: Polygon,
    min_area: int,
    tile_transform: Any,
) -> list[dict]:
    """Find and clip annotations that intersect a tile, convert to COCO format.

    Uses spatial index (R-tree) for efficient intersection via
    gdf.sindex.query() -- O(log n) per Tobler's recommendation.

    Args:
        annotations: GeoDataFrame with polygon geometries and class attributes.
        tile_polygon: Shapely Polygon representing the tile boundary in CRS space.
        min_area: Minimum annotation area in square pixels to keep.
        tile_transform: Affine transform from CRS space to tile pixel space
            (the inverse of the tile's rasterio transform).

    Returns:
        List of COCO annotation dicts with bbox, category_id, and area.
    """

    # Use spatial index for O(log n) lookup
    candidate_indices = annotations.sindex.query(tile_polygon, predicate="intersects")

    if len(candidate_indices) == 0:
        return []

    results = []
    inv_transform = tile_transform

    for idx in candidate_indices:
        row = annotations.iloc[idx]
        geom = row.geometry

        if geom is None or geom.is_empty:
            continue

        # Clip annotation to tile boundary
        try:
            clipped = geom.intersection(tile_polygon)
        except Exception:
            continue

        if clipped.is_empty:
            continue

        # Explode MultiPolygon into individual polygons
        polygons = []
        if isinstance(clipped, MultiPolygon):
            polygons = list(clipped.geoms)
        elif isinstance(clipped, Polygon):
            polygons = [clipped]
        else:
            # Could be a GeometryCollection with mixed types
            if hasattr(clipped, "geoms"):
                polygons = [g for g in clipped.geoms if isinstance(g, Polygon)]

        for poly in polygons:
            if poly.is_empty:
                continue

            # Convert to tile pixel coordinates using inverse affine
            bbox = _polygon_to_coco_bbox(poly, inv_transform)
            if bbox is None:
                continue

            x, y, w, h = bbox
            pixel_area = w * h

            # Filter by minimum area
            if pixel_area < min_area:
                continue

            # Get class info
            class_id = int(row.get("class_id", 0)) if "class_id" in row.index else 0
            results.append(
                {
                    "bbox": [x, y, w, h],
                    "area": pixel_area,
                    "category_id": class_id,
                }
            )

    return results


def _polygon_to_coco_bbox(
    polygon: Polygon,
    inv_transform: Any,
) -> list[float] | None:
    """Convert a CRS-space polygon to COCO [x, y, w, h] in tile pixel coordinates.

    Uses the inverse affine transform to convert from CRS to pixel space.

    Args:
        polygon: Shapely Polygon in CRS coordinates.
        inv_transform: Inverse affine transform (~tile_transform).

    Returns:
        [x, y, width, height] in tile pixel coordinates, or None if invalid.
    """
    bounds = polygon.bounds  # (minx, miny, maxx, maxy)
    if bounds[2] <= bounds[0] or bounds[3] <= bounds[1]:
        return None

    # Transform all 4 corners to pixel space
    corners_crs = [
        (bounds[0], bounds[1]),  # bottom-left
        (bounds[2], bounds[1]),  # bottom-right
        (bounds[0], bounds[3]),  # top-left
        (bounds[2], bounds[3]),  # top-right
    ]

    px_cols = []
    px_rows = []
    for cx, cy in corners_crs:
        col, row = inv_transform * (cx, cy)
        px_cols.append(col)
        px_rows.append(row)

    x_min = min(px_cols)
    x_max = max(px_cols)
    y_min = min(px_rows)
    y_max = max(px_rows)

    w = x_max - x_min
    h = y_max - y_min

    if w <= 0 or h <= 0:
        return None

    return [float(x_min), float(y_min), float(w), float(h)]


def prepare_training_dataset(
    raster_path: str,
    annotations_path: str,
    output_dir: str,
    tile_size: int = 576,
    overlap_ratio: float = 0.0,
    class_mapping: dict[str, int] | None = None,
    min_annotation_area: int = 100,
    max_background_per_annotated: float = 3.0,
    bands: tuple[int, ...] | str = "rgb",
    split_method: str = "block",
    split_ratios: tuple[float, float, float] = (0.8, 0.15, 0.05),
    seed: int = 42,
) -> dict[str, int]:
    """Convert GeoTIFF + vector annotations to a COCO-format training dataset.

    Full pipeline: tile raster, align CRS, clip annotations to tiles,
    convert to COCO format, split with spatial awareness, write output.

    Args:
        raster_path: Path to GeoTIFF raster.
        annotations_path: Path to GeoJSON, GeoPackage, or Shapefile.
        output_dir: Directory for COCO-format output.
        tile_size: Size of each square tile in pixels.
        overlap_ratio: Fractional overlap between tiles.
        class_mapping: Dict mapping class attribute values to COCO category IDs.
            If None, auto-generated from unique values.
        min_annotation_area: Minimum annotation area in square pixels.
        max_background_per_annotated: Max ratio of background to annotated tiles.
        bands: Band selection (preset name or tuple of 1-indexed band indices).
        split_method: "block" or "random" splitting method.
        split_ratios: (train, val, test) ratios.
        seed: Random seed for reproducibility.

    Returns:
        Statistics dict with counts of tiles and annotations per split.

    Raises:
        CRSError: If CRS alignment fails.
        FileNotFoundError: If input files don't exist.
    """
    from PIL import Image
    from rasterio.transform import Affine

    from detr_geo.io import BandSelector, load_raster_metadata, normalize_to_float32, read_tile
    from detr_geo.tiling import generate_tile_grid

    # Validate inputs
    raster_path_obj = Path(raster_path)
    annotations_path_obj = Path(annotations_path)
    output_path = Path(output_dir)

    if not raster_path_obj.exists():
        raise FileNotFoundError(f"Raster file not found: {raster_path}")
    if not annotations_path_obj.exists():
        raise FileNotFoundError(f"Annotations file not found: {annotations_path}")

    # Load raster metadata
    meta = load_raster_metadata(raster_path)

    # Load annotations
    annotations = gpd.read_file(annotations_path)
    if len(annotations) == 0:
        warnings.warn(
            f"Annotations file has no features: {annotations_path}",
            UserWarning,
            stacklevel=2,
        )

    # CRS alignment: reproject annotations to raster CRS if needed
    if meta.crs is not None and annotations.crs is not None:
        from pyproj import CRS as ProjCRS

        raster_crs = ProjCRS.from_user_input(meta.crs)
        ann_crs = ProjCRS.from_user_input(annotations.crs)

        if not raster_crs.equals(ann_crs):
            try:
                warnings.warn(
                    f"Reprojecting annotations from {ann_crs.to_epsg() or ann_crs.name} "
                    f"to {raster_crs.to_epsg() or raster_crs.name} to match raster CRS.",
                    UserWarning,
                    stacklevel=2,
                )
                annotations = annotations.to_crs(raster_crs)
            except Exception as exc:
                raise CRSError(
                    f"Failed to reproject annotations from "
                    f"{ann_crs.to_epsg() or ann_crs.name} to "
                    f"{raster_crs.to_epsg() or raster_crs.name}: {exc}"
                ) from exc
    elif meta.crs is None:
        raise CRSError("Raster has no CRS. Cannot align annotations with raster.")

    # Apply class mapping
    if class_mapping is not None:
        # User-provided mapping: attribute value -> category_id
        if "class_name" in annotations.columns:
            annotations = annotations.copy()
            annotations["class_id"] = annotations["class_name"].map(class_mapping)
            # Drop rows with unmapped classes
            unmapped = annotations["class_id"].isna()
            if unmapped.any():
                warnings.warn(
                    f"Dropping {unmapped.sum()} annotations with unmapped class values.",
                    UserWarning,
                    stacklevel=2,
                )
                annotations = annotations[~unmapped].copy()
                annotations["class_id"] = annotations["class_id"].astype(int)
    else:
        # Auto-generate class mapping from sorted unique values
        if "class_name" in annotations.columns and len(annotations) > 0:
            unique_classes = sorted(annotations["class_name"].unique())
            auto_mapping = {name: idx for idx, name in enumerate(unique_classes)}
            annotations = annotations.copy()
            annotations["class_id"] = annotations["class_name"].map(auto_mapping)
        elif "class_id" not in annotations.columns:
            annotations = annotations.copy()
            annotations["class_id"] = 0
            annotations["class_name"] = "object"

    # Ensure class_id is int
    if "class_id" in annotations.columns:
        annotations["class_id"] = annotations["class_id"].astype(int)

    # Build spatial index on annotations (happens lazily via geopandas)
    # Just ensure sindex is initialized by accessing it
    _ = annotations.sindex

    # Generate tile grid
    tiles = generate_tile_grid(
        raster_width=meta.width,
        raster_height=meta.height,
        tile_size=tile_size,
        overlap_ratio=overlap_ratio,
    )

    # Convert tiles to the format expected by SpatialSplitter
    tile_dicts = [{"window": t["window"]} for t in tiles]

    # Split tiles into train/val/test
    splitter = SpatialSplitter(
        method=split_method,
        ratios=split_ratios,
        seed=seed,
    )
    with warnings.catch_warnings():
        # Suppress warnings during split (they'll be caught and re-raised if needed)
        warnings.simplefilter("always")
        split_result = splitter.split(tile_dicts, meta.width, meta.height, tile_size)

    # Build split assignment map: tile_index -> split_name
    split_map: dict[int, str] = {}
    for idx in split_result.train_indices:
        split_map[idx] = "train"
    for idx in split_result.val_indices:
        split_map[idx] = "valid"  # RF-DETR expects "valid" not "val"
    for idx in split_result.test_indices:
        split_map[idx] = "test"
    # buffer indices are dropped entirely

    # Create output directory structure
    split_names = ["train", "valid", "test"]
    for split_name in split_names:
        (output_path / split_name / "images").mkdir(parents=True, exist_ok=True)

    # Process each tile: read pixels, clip annotations, write image + accumulate COCO
    band_selector = BandSelector(bands)
    stats = {
        "train_images": 0,
        "val_images": 0,
        "test_images": 0,
        "train_annotations": 0,
        "val_annotations": 0,
        "test_annotations": 0,
        "skipped_tiles": 0,
        "skipped_annotations": 0,
    }

    # Per-split COCO accumulators
    coco_data: dict[str, dict] = {}
    for split_name in split_names:
        coco_data[split_name] = {
            "images": [],
            "annotations": [],
            "categories": [],
        }

    # Build categories list
    categories_list = []
    if "class_id" in annotations.columns and "class_name" in annotations.columns:
        pairs = annotations[["class_id", "class_name"]].drop_duplicates().sort_values("class_id")
        for _, row in pairs.iterrows():
            categories_list.append({"id": int(row["class_id"]), "name": str(row["class_name"])})
    elif "class_id" in annotations.columns:
        for cid in sorted(annotations["class_id"].unique()):
            categories_list.append({"id": int(cid), "name": f"class_{cid}"})

    for split_name in split_names:
        coco_data[split_name]["categories"] = list(categories_list)

    global_ann_id = 0

    # Process tiles
    rng = np.random.RandomState(seed)

    # First pass: determine which tiles have annotations (for background ratio)
    tile_has_annotations: dict[int, bool] = {}
    for tile_idx, tile in enumerate(tiles):
        if tile_idx not in split_map:
            continue

        window = tile["window"]
        col_off, row_off, tw, th = window

        # Build tile polygon in CRS space
        tile_bounds_crs = _tile_to_crs_polygon(col_off, row_off, tw, th, meta.transform)

        # Quick spatial index check
        candidates = annotations.sindex.query(tile_bounds_crs, predicate="intersects")
        tile_has_annotations[tile_idx] = len(candidates) > 0

    # Count annotated tiles
    annotated_count = sum(1 for v in tile_has_annotations.values() if v)
    background_count = sum(1 for v in tile_has_annotations.values() if not v)

    # Determine which background tiles to keep
    max_background = int(annotated_count * max_background_per_annotated)
    background_to_keep: set[int] = set()

    if background_count > max_background and max_background >= 0:
        bg_indices = [idx for idx, has in tile_has_annotations.items() if not has]
        rng.shuffle(bg_indices)
        background_to_keep = set(bg_indices[:max_background])
        stats["skipped_tiles"] += len(bg_indices) - max_background
    else:
        background_to_keep = {idx for idx, has in tile_has_annotations.items() if not has}

    # Second pass: process tiles and write images
    for tile_idx, tile in enumerate(tiles):
        if tile_idx not in split_map:
            stats["skipped_tiles"] += 1
            continue

        split_name = split_map[tile_idx]

        # Skip excess background tiles
        if not tile_has_annotations.get(tile_idx, False):
            if tile_idx not in background_to_keep:
                continue

        window = tile["window"]
        col_off, row_off, tw, th = window

        # Read tile pixels
        data, nodata_mask = read_tile(raster_path, window, bands=band_selector.band_indices)

        # Skip tiles that are mostly nodata
        if nodata_mask is not None:
            nodata_frac = float(np.sum(nodata_mask)) / (tw * th)
            if nodata_frac > 0.5:
                stats["skipped_tiles"] += 1
                continue

        # Normalize to uint8 for training tiles (per Tobler: RF-DETR handles normalization)
        if data.dtype != np.uint8:
            # Percentile stretch to uint8
            norm_data, _ = normalize_to_float32(data)
            uint8_data = (np.clip(norm_data, 0, 1) * 255).astype(np.uint8)
        else:
            uint8_data = data

        # Convert (C, H, W) -> (H, W, C) for PIL
        img_array = np.transpose(uint8_data, (1, 2, 0))
        if img_array.shape[2] == 1:
            img_array = np.repeat(img_array, 3, axis=2)

        # Save tile image as JPEG
        image_id = len(coco_data[split_name]["images"])
        image_filename = f"tile_{tile_idx:05d}.jpg"
        image_save_path = output_path / split_name / "images" / image_filename

        pil_img = Image.fromarray(img_array, mode="RGB")
        pil_img.save(str(image_save_path), quality=95)

        # Add image entry to COCO
        coco_data[split_name]["images"].append(
            {
                "id": image_id,
                "file_name": image_filename,
                "width": tw,
                "height": th,
            }
        )

        # Update stats
        stats_key = (
            "train_images" if split_name == "train" else ("val_images" if split_name == "valid" else "test_images")
        )
        stats[stats_key] += 1

        # Clip annotations to this tile
        tile_polygon = _tile_to_crs_polygon(col_off, row_off, tw, th, meta.transform)

        # Compute tile-local inverse affine transform
        # tile_transform maps pixel (relative to tile) -> CRS
        tile_transform = meta.transform * Affine.translation(col_off, row_off)
        inv_tile_transform = ~tile_transform

        tile_anns = _clip_annotations_to_tile(annotations, tile_polygon, min_annotation_area, inv_tile_transform)

        ann_stats_key = (
            "train_annotations"
            if split_name == "train"
            else ("val_annotations" if split_name == "valid" else "test_annotations")
        )

        for ann in tile_anns:
            coco_data[split_name]["annotations"].append(
                {
                    "id": global_ann_id,
                    "image_id": image_id,
                    "category_id": ann["category_id"],
                    "bbox": ann["bbox"],
                    "area": ann["area"],
                    "iscrowd": 0,
                }
            )
            global_ann_id += 1
            stats[ann_stats_key] += 1

    # Write COCO JSON files
    for split_name in split_names:
        coco_json_path = output_path / split_name / "_annotations.coco.json"
        with open(coco_json_path, "w") as f:
            json.dump(coco_data[split_name], f, indent=2)

    return stats


def _tile_to_crs_polygon(
    col_off: int,
    row_off: int,
    width: int,
    height: int,
    transform: Any,
) -> Polygon:
    """Convert a tile window to a CRS-space polygon.

    Args:
        col_off: Column offset in pixels.
        row_off: Row offset in pixels.
        width: Tile width in pixels.
        height: Tile height in pixels.
        transform: Rasterio affine transform.

    Returns:
        Shapely Polygon in CRS coordinate space.
    """
    # Transform 4 corners
    tl = transform * (col_off, row_off)
    tr = transform * (col_off + width, row_off)
    br = transform * (col_off + width, row_off + height)
    bl = transform * (col_off, row_off + height)

    return Polygon([tl, tr, br, bl, tl])


# ---------------------------------------------------------------------------
# Augmentation Presets and Training Wrapper
# ---------------------------------------------------------------------------


@dataclass
class AugmentationPreset:
    """Geospatial augmentation preset applied before RF-DETR training.

    These augmentations add geospatial-specific operations (vertical flip,
    90-degree rotation) that RF-DETR's built-in training augmentations omit.
    Applied as a pre-augmentation step on disk before RF-DETR's transforms.
    """

    name: str
    random_rotation_90: bool = True
    horizontal_flip: bool = True
    vertical_flip: bool = True
    brightness_jitter: float = 0.2
    contrast_jitter: float = 0.2
    saturation_jitter: float = 0.1


AUGMENTATION_PRESETS: dict[str, AugmentationPreset] = {
    "satellite_default": AugmentationPreset(
        name="satellite_default",
        random_rotation_90=True,
        horizontal_flip=True,
        vertical_flip=True,
        brightness_jitter=0.2,
        contrast_jitter=0.2,
        saturation_jitter=0.1,
    ),
    "aerial_default": AugmentationPreset(
        name="aerial_default",
        random_rotation_90=True,
        horizontal_flip=True,
        vertical_flip=True,
        brightness_jitter=0.3,
        contrast_jitter=0.3,
        saturation_jitter=0.15,
    ),
    "drone_default": AugmentationPreset(
        name="drone_default",
        random_rotation_90=True,
        horizontal_flip=True,
        vertical_flip=True,
        brightness_jitter=0.4,
        contrast_jitter=0.4,
        saturation_jitter=0.2,
    ),
}


def _apply_augmentation(
    dataset_dir: str,
    preset: AugmentationPreset,
    augmentation_factor: int = 2,
    seed: int = 42,
) -> None:
    """Apply pre-augmentation to training image chips on disk.

    Generates augmented copies of training images with geospatial-specific
    transforms (vertical flip, 90-degree rotation, color jitter) that
    RF-DETR's built-in augmentation pipeline does not include.

    Only augments the "train" split (not valid/test).

    Args:
        dataset_dir: Path to COCO-format dataset directory.
        preset: AugmentationPreset with operation flags and intensities.
        augmentation_factor: Number of total copies (including original).
            Default 2 means 1 original + 1 augmented = 2x storage.
        seed: Random seed for reproducibility.
    """
    from PIL import Image, ImageEnhance

    rng = np.random.RandomState(seed)
    train_images_dir = Path(dataset_dir) / "train" / "images"
    coco_json_path = Path(dataset_dir) / "train" / "_annotations.coco.json"

    if not train_images_dir.exists() or not coco_json_path.exists():
        return

    with open(coco_json_path) as f:
        coco = json.load(f)

    original_images = list(train_images_dir.glob("*.jpg"))
    if not original_images:
        return

    # Number of augmented copies per image (factor - 1 for the original)
    n_augmented = max(0, augmentation_factor - 1)
    if n_augmented == 0:
        return

    # Build image_id -> annotations lookup
    img_anns: dict[int, list[dict]] = {}
    for ann in coco["annotations"]:
        img_id = ann["image_id"]
        if img_id not in img_anns:
            img_anns[img_id] = []
        img_anns[img_id].append(ann)

    # Find max annotation id for new unique IDs
    max_ann_id = max((a["id"] for a in coco["annotations"]), default=-1) + 1

    new_images = []
    new_annotations = []
    next_image_id = max((img["id"] for img in coco["images"]), default=-1) + 1

    for orig_img_entry in list(coco["images"]):
        orig_filename = orig_img_entry["file_name"]
        orig_path = train_images_dir / orig_filename
        if not orig_path.exists():
            continue

        try:
            img = Image.open(orig_path)
        except Exception:
            continue

        w, h = img.size
        orig_id = orig_img_entry["id"]
        orig_annotations = img_anns.get(orig_id, [])

        for aug_idx in range(n_augmented):
            aug_img = img.copy()
            bbox_transforms = []  # Track transforms for bbox adjustment

            # Apply random subset of augmentations
            if preset.random_rotation_90 and rng.random() > 0.5:
                k = rng.choice([1, 2, 3])  # 90, 180, or 270 degrees
                aug_img = aug_img.rotate(-90 * k, expand=False)
                bbox_transforms.append(("rotate", k))

            if preset.horizontal_flip and rng.random() > 0.5:
                aug_img = aug_img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
                bbox_transforms.append(("hflip", None))

            if preset.vertical_flip and rng.random() > 0.5:
                aug_img = aug_img.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
                bbox_transforms.append(("vflip", None))

            # Color jitter
            if preset.brightness_jitter > 0:
                factor = 1.0 + rng.uniform(-preset.brightness_jitter, preset.brightness_jitter)
                aug_img = ImageEnhance.Brightness(aug_img).enhance(factor)

            if preset.contrast_jitter > 0:
                factor = 1.0 + rng.uniform(-preset.contrast_jitter, preset.contrast_jitter)
                aug_img = ImageEnhance.Contrast(aug_img).enhance(factor)

            if preset.saturation_jitter > 0:
                factor = 1.0 + rng.uniform(-preset.saturation_jitter, preset.saturation_jitter)
                aug_img = ImageEnhance.Color(aug_img).enhance(factor)

            # Save augmented image
            stem = orig_path.stem
            aug_filename = f"{stem}_aug{aug_idx}.jpg"
            aug_path = train_images_dir / aug_filename
            aug_img.save(str(aug_path), quality=95)

            # Add image entry
            new_images.append(
                {
                    "id": next_image_id,
                    "file_name": aug_filename,
                    "width": w,
                    "height": h,
                }
            )

            # Transform annotations for this augmented image
            for orig_ann in orig_annotations:
                bbox = list(orig_ann["bbox"])  # [x, y, w, h]
                new_bbox = _transform_coco_bbox(bbox, w, h, bbox_transforms)

                new_annotations.append(
                    {
                        "id": max_ann_id,
                        "image_id": next_image_id,
                        "category_id": orig_ann["category_id"],
                        "bbox": new_bbox,
                        "area": new_bbox[2] * new_bbox[3],
                        "iscrowd": 0,
                    }
                )
                max_ann_id += 1

            next_image_id += 1

    # Update COCO JSON with augmented entries
    coco["images"].extend(new_images)
    coco["annotations"].extend(new_annotations)

    with open(coco_json_path, "w") as f:
        json.dump(coco, f, indent=2)


def _transform_coco_bbox(
    bbox: list[float],
    img_w: int,
    img_h: int,
    transforms: list[tuple],
) -> list[float]:
    """Apply geometric transforms to a COCO bbox [x, y, w, h].

    Args:
        bbox: [x, y, width, height] in pixel coords.
        img_w: Image width.
        img_h: Image height.
        transforms: List of transform tuples to apply in order.

    Returns:
        Transformed [x, y, width, height].
    """
    x, y, w, h = bbox

    for t in transforms:
        if t[0] == "hflip":
            x = img_w - x - w
        elif t[0] == "vflip":
            y = img_h - y - h
        elif t[0] == "rotate":
            k = t[1]
            for _ in range(k):
                # 90 degree clockwise rotation
                new_x = img_h - y - h
                new_y = x
                new_w = h
                new_h = w
                x, y, w, h = new_x, new_y, new_w, new_h
                # After rotation, image dimensions swap
                img_w, img_h = img_h, img_w

    return [float(x), float(y), float(w), float(h)]


def train(
    adapter: Any,
    dataset_dir: str,
    epochs: int = 50,
    augmentation_preset: str | AugmentationPreset | None = "satellite_default",
    augmentation_factor: int = 2,
    batch_size: int = 8,
    learning_rate: float | None = None,
    resume: str | None = None,
    seed: int = 42,
    **kwargs: Any,
) -> dict:
    """Train an RF-DETR model on a COCO-format dataset with geospatial augmentation.

    This is a thin wrapper that:
    1. Resolves the augmentation preset
    2. Applies pre-augmentation to training images on disk
    3. Delegates to adapter.train() for the actual training

    Args:
        adapter: RFDETRAdapter instance (or any object with a train() method).
        dataset_dir: Path to COCO-format dataset (with train/valid/ subdirs).
        epochs: Number of training epochs.
        augmentation_preset: Preset name string, AugmentationPreset instance,
            or None to skip pre-augmentation.
        augmentation_factor: Number of copies including original (2 = 1 orig + 1 aug).
        batch_size: Training batch size.
        learning_rate: Learning rate override. None uses model default.
        resume: Path to checkpoint for resume training.
        seed: Random seed for augmentation.
        **kwargs: Additional keyword arguments passed to adapter.train().

    Returns:
        Training metrics dict from the adapter.

    Raises:
        ModelError: If rfdetr is not installed or training fails.
        ValueError: If preset name is invalid.
    """
    # Resolve augmentation preset
    preset: AugmentationPreset | None = None
    if augmentation_preset is not None:
        if isinstance(augmentation_preset, str):
            if augmentation_preset not in AUGMENTATION_PRESETS:
                raise ValueError(
                    f"Unknown augmentation preset '{augmentation_preset}'. "
                    f"Valid presets: {', '.join(sorted(AUGMENTATION_PRESETS.keys()))}. "
                    f"Or pass an AugmentationPreset instance."
                )
            preset = AUGMENTATION_PRESETS[augmentation_preset]
        elif isinstance(augmentation_preset, AugmentationPreset):
            preset = augmentation_preset
        else:
            raise ValueError(
                f"augmentation_preset must be a string, AugmentationPreset, or None, "
                f"got {type(augmentation_preset).__name__}"
            )

    # Apply pre-augmentation to training images
    if preset is not None and augmentation_factor > 1:
        _apply_augmentation(
            dataset_dir=dataset_dir,
            preset=preset,
            augmentation_factor=augmentation_factor,
            seed=seed,
        )

    # Build training kwargs
    train_kwargs: dict[str, Any] = {
        "epochs": epochs,
        "batch_size": batch_size,
    }

    if learning_rate is not None:
        train_kwargs["lr"] = learning_rate

    if resume is not None:
        train_kwargs["resume"] = resume

    train_kwargs.update(kwargs)

    # Delegate to adapter
    try:
        result = adapter.train(dataset_dir, **train_kwargs)
    except ModelError:
        raise
    except Exception as exc:
        raise ModelError(f"Training failed: {exc}. Ensure rfdetr is installed: pip install detr-geo[rfdetr]") from exc

    return result if isinstance(result, dict) else {}
