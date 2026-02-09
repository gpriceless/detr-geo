#!/usr/bin/env python3
"""Preprocess xView GeoJSON: remap 60 classes to 5 vehicle classes.

Loads the full xView GeoJSON annotation file (1M+ features), applies
class remapping from xView's 60 fine-grained classes to our 5 coarse
vehicle classes (Car, Pickup, Truck, Bus, Other), filters non-vehicle
annotations, reports class distribution and imbalance, and writes a
remapped GeoJSON ready for COCO dataset preparation.

Dataset: xView Detection Challenge (Lam et al., 2018)
License: CC BY-NC-SA 4.0 (Attribution-NonCommercial-ShareAlike)
Citation:
    Lam, D. et al. "xView: Objects in Context in Overhead Imagery."
    arXiv:1802.07856, 2018.

Usage:
    python scripts/preprocess_xview.py --input xview_raw/xView_train.geojson --output xview_dataset/xview_remapped.geojson
    python scripts/preprocess_xview.py --input xview_raw/xView_train.geojson --output xview_dataset/xview_remapped.geojson --include_background
    python scripts/preprocess_xview.py --input xview_raw/xView_train.geojson --output xview_dataset/xview_remapped.geojson --min_area 25
    python scripts/preprocess_xview.py --help
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Class Mapping: xView type_id -> our 5 vehicle classes
# ---------------------------------------------------------------------------

# Our 5 coarse vehicle classes (0-indexed for COCO compatibility)
XVIEW_CLASS_NAMES: dict[int, str] = {
    0: "Car",
    1: "Pickup",
    2: "Truck",
    3: "Bus",
    4: "Other",
}

# xView type_id -> our class_id
# Source: DIU xView Challenge specification mapped to 5-class scheme
XVIEW_TO_VEHICLE: dict[int, int] = {
    # Car (class 0): Passenger vehicles and small cars
    17: 0,  # Passenger Vehicle
    18: 0,  # Small Car
    # Pickup (class 1): Pickup trucks
    20: 1,  # Pickup Truck
    # Truck (class 2): All truck variants
    25: 2,  # Truck Tractor
    26: 2,  # Truck Tractor (alt id -- xView has both 25 and 26)
    21: 2,  # Utility Truck
    24: 2,  # Cargo Truck
    23: 2,  # Truck (generic)
    27: 2,  # Truck w/Flatbed
    28: 2,  # Truck w/Liquid
    29: 2,  # Crane Truck
    # Bus (class 3)
    19: 3,  # Bus
    # Other (class 4): Railway, trailers, engineering vehicles
    60: 4,  # Dump Truck (construction, not road vehicle)
    53: 4,  # Engineering Vehicle
    32: 4,  # Railway Vehicle
    33: 4,  # Passenger Car (rail)
    34: 4,  # Cargo Car (rail)
    35: 4,  # Flat Car
    36: 4,  # Tank Car
    37: 4,  # Locomotive
}

# xView class names for display
XVIEW_TYPE_NAMES: dict[int, str] = {
    11: "Fixed-wing Aircraft",
    12: "Small Aircraft",
    13: "Cargo Plane",
    15: "Helicopter",
    17: "Passenger Vehicle",
    18: "Small Car",
    19: "Bus",
    20: "Pickup Truck",
    21: "Utility Truck",
    23: "Truck",
    24: "Cargo Truck",
    25: "Truck w/Box",
    26: "Truck Tractor",
    27: "Truck w/Flatbed",
    28: "Truck w/Liquid",
    29: "Crane Truck",
    32: "Railway Vehicle",
    33: "Passenger Car (rail)",
    34: "Cargo Car (rail)",
    35: "Flat Car",
    36: "Tank Car",
    37: "Locomotive",
    40: "Maritime Vessel",
    41: "Motorboat",
    42: "Sailboat",
    44: "Tugboat",
    45: "Barge",
    47: "Fishing Vessel",
    49: "Ferry",
    50: "Yacht",
    51: "Container Ship",
    52: "Oil Tanker",
    53: "Engineering Vehicle",
    54: "Tower Crane",
    55: "Container Crane",
    56: "Reach Stacker",
    57: "Straddle Carrier",
    59: "Mobile Crane",
    60: "Dump Truck",
    61: "Haul Truck",
    62: "Scraper/Tractor",
    63: "Front loader/Bulldozer",
    64: "Excavator",
    65: "Cement Mixer",
    66: "Ground Grader",
    71: "Hut/Tent",
    72: "Shed",
    73: "Building",
    74: "Aircraft Hangar",
    76: "Damaged Building",
    77: "Facility",
    79: "Construction Site",
    83: "Vehicle Lot",
    84: "Helipad",
    86: "Storage Tank",
    89: "Shipping Container Lot",
    91: "Shipping Container",
    93: "Pylon",
    94: "Tower",
}


# ---------------------------------------------------------------------------
# Preprocessing functions
# ---------------------------------------------------------------------------


def compute_bbox_area(feature: dict) -> float:
    """Compute bounding box area from a GeoJSON feature.

    Uses bounds_imcoords (pixel coordinates) if available, otherwise
    computes from geometry bounds.

    Args:
        feature: GeoJSON feature dict.

    Returns:
        Area in square pixels (or square coordinate units).
    """
    props = feature.get("properties", {})

    # Prefer bounds_imcoords (pixel coordinates: "x1,y1,x2,y2")
    bounds_str = props.get("bounds_imcoords")
    if bounds_str:
        try:
            parts = [float(x) for x in bounds_str.split(",")]
            if len(parts) == 4:
                x1, y1, x2, y2 = parts
                return abs(x2 - x1) * abs(y2 - y1)
        except (ValueError, TypeError):
            pass

    # Fall back to geometry bounds
    geom = feature.get("geometry", {})
    coords = geom.get("coordinates", [])
    if coords:
        # Flatten to get all coordinate pairs
        flat_coords = []
        _flatten_coords(coords, flat_coords)
        if len(flat_coords) >= 2:
            xs = [c[0] for c in flat_coords]
            ys = [c[1] for c in flat_coords]
            return (max(xs) - min(xs)) * (max(ys) - min(ys))

    return 0.0


def _flatten_coords(coords: Any, result: list) -> None:
    """Recursively flatten nested coordinate arrays.

    Args:
        coords: Nested list of coordinates.
        result: Accumulator list for [x, y] pairs.
    """
    if not coords:
        return
    if isinstance(coords[0], (int, float)):
        # This is a coordinate pair
        result.append(coords)
        return
    for item in coords:
        _flatten_coords(item, result)


def remap_features(
    features: list[dict],
    include_background: bool = False,
    min_area: float = 0.0,
) -> tuple[list[dict], dict[str, Any]]:
    """Apply 60-to-5 class remapping on xView features.

    Processes each feature: looks up type_id in the mapping table,
    assigns our class_id and class_name, filters non-vehicle features
    (unless include_background is True), and skips features below
    min_area threshold.

    Args:
        features: List of GeoJSON feature dicts.
        include_background: If True, keep non-vehicle features as
            a 6th "Background" class (id=5).
        min_area: Minimum bounding box area in square pixels. Features
            below this threshold are skipped.

    Returns:
        Tuple of (remapped_features, statistics_dict).
    """
    remapped: list[dict] = []
    stats: dict[str, Any] = {
        "total_input": len(features),
        "remapped": 0,
        "skipped_unmapped": 0,
        "skipped_small": 0,
        "skipped_no_type_id": 0,
        "skipped_no_geometry": 0,
        "class_counts": {name: 0 for name in XVIEW_CLASS_NAMES.values()},
        "source_type_counts": {},
    }

    if include_background:
        stats["class_counts"]["Background"] = 0

    for feature in features:
        props = feature.get("properties", {})
        geom = feature.get("geometry")

        # Skip features without geometry
        if geom is None or not geom.get("coordinates"):
            stats["skipped_no_geometry"] += 1
            continue

        # Get type_id
        type_id = props.get("type_id")
        if type_id is None:
            stats["skipped_no_type_id"] += 1
            continue
        type_id = int(type_id)

        # Check area threshold
        if min_area > 0:
            area = compute_bbox_area(feature)
            if area < min_area:
                stats["skipped_small"] += 1
                continue

        # Look up class mapping
        our_class_id = XVIEW_TO_VEHICLE.get(type_id)

        if our_class_id is not None:
            # Vehicle class -- remap
            class_name = XVIEW_CLASS_NAMES[our_class_id]
        elif include_background:
            # Non-vehicle, keep as background
            our_class_id = 5
            class_name = "Background"
        else:
            # Non-vehicle, skip
            stats["skipped_unmapped"] += 1
            continue

        # Build remapped feature with simplified properties
        remapped_feature = {
            "type": "Feature",
            "properties": {
                "class_id": our_class_id,
                "class_name": class_name,
                "type_id": type_id,
                "type_name": XVIEW_TYPE_NAMES.get(type_id, f"Unknown ({type_id})"),
                "image_id": props.get("image_id", ""),
                "bounds_imcoords": props.get("bounds_imcoords", ""),
            },
            "geometry": geom,
        }

        remapped.append(remapped_feature)
        stats["remapped"] += 1
        stats["class_counts"][class_name] = stats["class_counts"].get(class_name, 0) + 1

        # Track source type_id distribution
        type_name = XVIEW_TYPE_NAMES.get(type_id, f"type_{type_id}")
        stats["source_type_counts"][type_name] = (
            stats["source_type_counts"].get(type_name, 0) + 1
        )

    return remapped, stats


def check_class_imbalance(class_counts: dict[str, int]) -> list[str]:
    """Check for class imbalance in the remapped dataset.

    Args:
        class_counts: Dict mapping class name to count.

    Returns:
        List of warning strings about imbalance.
    """
    warnings: list[str] = []

    # Filter to non-zero classes
    nonzero = {k: v for k, v in class_counts.items() if v > 0}
    if len(nonzero) < 2:
        if len(nonzero) == 0:
            warnings.append("No annotations found after remapping")
        return warnings

    max_class = max(nonzero, key=nonzero.get)
    min_class = min(nonzero, key=nonzero.get)
    max_count = nonzero[max_class]
    min_count = nonzero[min_class]
    ratio = max_count / min_count

    if ratio > 100:
        warnings.append(
            f"Extreme class imbalance: {max_class}:{min_class} ratio is "
            f"{ratio:.0f}:1 ({max_count:,} vs {min_count:,}). "
            f"Consider class weighting or oversampling minority classes."
        )
    elif ratio > 50:
        warnings.append(
            f"Severe class imbalance: {max_class}:{min_class} ratio is "
            f"{ratio:.0f}:1 ({max_count:,} vs {min_count:,})"
        )
    elif ratio > 10:
        warnings.append(
            f"Class imbalance: {max_class}:{min_class} ratio is "
            f"{ratio:.0f}:1 ({max_count:,} vs {min_count:,})"
        )

    return warnings


def write_class_mapping(output_path: str, include_background: bool = False) -> None:
    """Write class mapping metadata to a JSON file.

    This file documents the xView type_id to our class_id mapping
    for reproducibility and reference.

    Args:
        output_path: Path to write the JSON file.
        include_background: Whether background class is included.
    """
    mapping = {
        "description": "xView 60-class to 5-class vehicle mapping",
        "license": "CC BY-NC-SA 4.0",
        "citation": "Lam et al., xView: Objects in Context in Overhead Imagery, 2018",
        "classes": {},
        "xview_type_id_to_class_id": {},
    }

    for class_id, class_name in XVIEW_CLASS_NAMES.items():
        source_ids = [tid for tid, cid in XVIEW_TO_VEHICLE.items() if cid == class_id]
        source_names = [
            XVIEW_TYPE_NAMES.get(tid, f"type_{tid}") for tid in source_ids
        ]
        mapping["classes"][str(class_id)] = {
            "name": class_name,
            "xview_type_ids": source_ids,
            "xview_type_names": source_names,
        }

    if include_background:
        mapping["classes"]["5"] = {
            "name": "Background",
            "xview_type_ids": "all non-vehicle type_ids",
            "xview_type_names": "all non-vehicle classes",
        }

    for tid, cid in XVIEW_TO_VEHICLE.items():
        mapping["xview_type_id_to_class_id"][str(tid)] = cid

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        json.dump(mapping, f, indent=2)


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
            "Preprocess xView GeoJSON: remap 60 classes to 5 vehicle "
            "classes (Car, Pickup, Truck, Bus, Other)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to xView GeoJSON annotation file (e.g., xView_train.geojson)",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path for remapped GeoJSON output",
    )
    parser.add_argument(
        "--include_background",
        action="store_true",
        help="Keep non-vehicle annotations as 'Background' class (id=5) for hard negative mining",
    )
    parser.add_argument(
        "--min_area",
        type=float,
        default=0.0,
        help="Minimum bounding box area in square pixels (default: 0 = no filter)",
    )
    parser.add_argument(
        "--class_mapping_output",
        type=str,
        default=None,
        help="Path to write class mapping JSON (default: <output_dir>/class_mapping.json)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run xView preprocessing pipeline.

    Args:
        argv: Command-line arguments (defaults to sys.argv[1:]).

    Returns:
        Exit code: 0 on success, 1 on failure.
    """
    args = parse_args(argv)

    print("=" * 60)
    print("  xView Preprocessing: 60-class to 5-class Vehicle Mapping")
    print("=" * 60)

    # -------------------------------------------------------------------
    # Validate input
    # -------------------------------------------------------------------
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"\n  ERROR: Input file not found: {args.input}")
        return 1

    file_size_mb = input_path.stat().st_size / (1024**2)
    print(f"\n  Input: {args.input} ({file_size_mb:.0f} MB)")
    print(f"  Output: {args.output}")
    print(f"  Include background: {args.include_background}")
    print(f"  Min area filter: {args.min_area}")

    # -------------------------------------------------------------------
    # Load GeoJSON
    # -------------------------------------------------------------------
    print(f"\n[1/4] Loading xView annotations...")
    t0 = time.time()

    try:
        with open(input_path) as f:
            data = json.load(f)
    except json.JSONDecodeError as exc:
        print(f"  ERROR: Invalid JSON: {exc}")
        return 1
    except MemoryError:
        print(
            "  ERROR: GeoJSON too large to load in memory. "
            "Need 32+ GB RAM for xView annotations."
        )
        return 1

    features = data.get("features", [])
    load_time = time.time() - t0
    print(f"  Loaded {len(features):,} features in {load_time:.1f}s")

    if len(features) == 0:
        print("  ERROR: No features found in GeoJSON")
        return 1

    # -------------------------------------------------------------------
    # Apply class remapping
    # -------------------------------------------------------------------
    print(f"\n[2/4] Applying 60-to-5 class remapping...")
    t0 = time.time()

    remapped_features, stats = remap_features(
        features,
        include_background=args.include_background,
        min_area=args.min_area,
    )

    remap_time = time.time() - t0
    print(f"  Processed {stats['total_input']:,} features in {remap_time:.1f}s")
    print(f"  Remapped: {stats['remapped']:,}")
    print(f"  Skipped (unmapped/non-vehicle): {stats['skipped_unmapped']:,}")
    print(f"  Skipped (below min area): {stats['skipped_small']:,}")
    print(f"  Skipped (no type_id): {stats['skipped_no_type_id']:,}")
    print(f"  Skipped (no geometry): {stats['skipped_no_geometry']:,}")

    if stats["remapped"] == 0:
        print("\n  ERROR: No features survived remapping. Check input file and class mapping.")
        return 1

    # -------------------------------------------------------------------
    # Report class distribution
    # -------------------------------------------------------------------
    print(f"\n[3/4] Class distribution after remapping:")
    total_remapped = stats["remapped"]
    print(f"\n  {'Class':<15} {'ID':>4} {'Count':>10} {'Percentage':>10}")
    print(f"  {'-'*15} {'-'*4} {'-'*10} {'-'*10}")

    for class_id, class_name in XVIEW_CLASS_NAMES.items():
        count = stats["class_counts"].get(class_name, 0)
        pct = (count / total_remapped * 100) if total_remapped > 0 else 0
        print(f"  {class_name:<15} {class_id:>4} {count:>10,} {pct:>9.1f}%")

    if args.include_background:
        bg_count = stats["class_counts"].get("Background", 0)
        pct = (bg_count / total_remapped * 100) if total_remapped > 0 else 0
        print(f"  {'Background':<15} {'5':>4} {bg_count:>10,} {pct:>9.1f}%")

    print(f"\n  Total: {total_remapped:,}")

    # Check imbalance
    imbalance_warnings = check_class_imbalance(stats["class_counts"])
    for warning in imbalance_warnings:
        print(f"\n  WARNING: {warning}")

    # Show source type_id breakdown
    print(f"\n  Source xView type breakdown:")
    for type_name, count in sorted(
        stats["source_type_counts"].items(),
        key=lambda x: x[1],
        reverse=True,
    ):
        print(f"    {type_name}: {count:,}")

    # -------------------------------------------------------------------
    # Write output
    # -------------------------------------------------------------------
    print(f"\n[4/4] Writing remapped GeoJSON...")
    t0 = time.time()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    remapped_geojson = {
        "type": "FeatureCollection",
        "features": remapped_features,
    }

    with open(output_path, "w") as f:
        json.dump(remapped_geojson, f)

    write_time = time.time() - t0
    output_size_mb = output_path.stat().st_size / (1024**2)
    print(f"  Wrote {len(remapped_features):,} features to {args.output}")
    print(f"  Output size: {output_size_mb:.1f} MB (wrote in {write_time:.1f}s)")

    # Write class mapping JSON
    mapping_path = args.class_mapping_output
    if mapping_path is None:
        mapping_path = str(output_path.parent / "class_mapping.json")

    write_class_mapping(mapping_path, include_background=args.include_background)
    print(f"  Class mapping saved to: {mapping_path}")

    # -------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print(f"  PREPROCESSING COMPLETE")
    print(f"{'=' * 60}")
    print(f"  Input: {len(features):,} features ({file_size_mb:.0f} MB)")
    print(f"  Output: {len(remapped_features):,} features ({output_size_mb:.1f} MB)")
    print(f"  Classes: {len([c for c in stats['class_counts'].values() if c > 0])}")

    if imbalance_warnings:
        print(f"\n  Imbalance warnings ({len(imbalance_warnings)}):")
        for w in imbalance_warnings:
            print(f"    - {w}")

    print(f"\n  Next step:")
    print(f"    python scripts/train_xview.py \\")
    print(f"      --dataset_dir <coco_dataset_dir> \\")
    print(f"      --output_dir training_output/xview")

    return 0


if __name__ == "__main__":
    sys.exit(main())
