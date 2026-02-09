#!/usr/bin/env python3
"""Verify and analyze the xView dataset for overhead vehicle detection.

xView requires MANUAL registration and download from xviewdataset.org.
This script does NOT automate download -- it verifies downloaded files,
validates GeoTIFF imagery and GeoJSON annotations, analyzes class
distribution across all 60 xView classes, and identifies vehicle-specific
classes for fine-tuning.

Dataset: xView Detection Challenge (Lam et al., 2018)
License: CC BY-NC-SA 4.0 (Attribution-NonCommercial-ShareAlike)
Citation:
    Lam, D., Kuzma, R., McGee, K., Dooley, S., Laielli, M., Klaric, M.,
    Bulatov, Y., & McCord, B. (2018). xView: Objects in Context in
    Overhead Imagery. arXiv preprint arXiv:1802.07856.

Usage:
    python scripts/download_xview.py                                      # Show registration instructions
    python scripts/download_xview.py --verify --data_dir xview_raw/       # Verify downloaded files
    python scripts/download_xview.py --init --output_dir xview_dataset/   # Initialize output directory
    python scripts/download_xview.py --help
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Complete xView class definitions: type_id -> class name
# Source: DIU xView Challenge specification (Lam et al. 2018)
XVIEW_CLASSES: dict[int, str] = {
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

# Vehicle-related type_ids grouped by our 5-class mapping
VEHICLE_TYPE_IDS: dict[str, list[int]] = {
    "Car": [17, 18],                              # Passenger Vehicle, Small Car
    "Pickup": [20],                                # Pickup Truck
    "Truck": [25, 26, 21, 24, 23, 27, 28, 29],   # Truck Tractor, Utility Truck, etc.
    "Bus": [19],                                   # Bus
    "Other": [60, 53, 32, 33, 34, 35, 36, 37],   # Trailer, Engineering Vehicle, Railway
}

ALL_VEHICLE_TYPE_IDS: set[int] = set()
for _ids in VEHICLE_TYPE_IDS.values():
    ALL_VEHICLE_TYPE_IDS.update(_ids)

# Minimum disk space for processing (35 GB)
MIN_DISK_SPACE_GB = 35

LICENSE_TEXT = """
================================================================================
  xView Dataset License: CC BY-NC-SA 4.0
================================================================================

  This dataset is licensed under Creative Commons
  Attribution-NonCommercial-ShareAlike 4.0 International.

  By using this dataset, you agree to:
  - Use for NON-COMMERCIAL research purposes only
  - Provide attribution to the original authors
  - Share derivative works (including fine-tuned model weights)
    under the same license

  Citation:
    Lam, D., Kuzma, R., McGee, K., Dooley, S., Laielli, M., Klaric, M.,
    Bulatov, Y., & McCord, B. (2018). xView: Objects in Context in Overhead
    Imagery. arXiv preprint arXiv:1802.07856.

  Full license: https://creativecommons.org/licenses/by-nc-sa/4.0/
================================================================================
"""

REGISTRATION_INSTRUCTIONS = """
================================================================================
  xView Dataset Registration & Download
================================================================================

  xView requires manual registration. Follow these steps:

  1. Register at: https://challenge.xviewdataset.org/data-download
     - Create an account (any email works)
     - Accept the dataset terms of use

  2. Download the training dataset:
     - train_images.tgz  (~20 GB) -- 846 satellite images as GeoTIFFs
     - train_labels.tgz   (~180 MB) -- GeoJSON annotations (1M+ objects)

     Note: Validation images have NO labels and cannot be used for training.

  3. Extract to a directory:
     mkdir -p xview_raw/
     tar -xzf train_images.tgz -C xview_raw/
     tar -xzf train_labels.tgz -C xview_raw/

  4. Verify the download:
     python scripts/download_xview.py --verify --data_dir xview_raw/

  Alternative mirrors (may require separate registration):
  - Dataset Ninja: https://datasetninja.com/xview
  - Supervisely: Search for xView dataset listing
    Note: Alternative mirrors may use different directory layouts.
          Use --verify to check structure regardless of source.

  Expected directory structure after extraction:
    xview_raw/
      train_images/          # 846 large GeoTIFF satellite images
        *.tif
      xView_train.geojson    # 1M+ object annotations

================================================================================
"""


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------


def find_geojson(data_dir: str) -> str | None:
    """Find the xView GeoJSON annotation file.

    Searches for common filenames used by the xView dataset.

    Args:
        data_dir: Directory to search.

    Returns:
        Path to the GeoJSON file, or None if not found.
    """
    ds = Path(data_dir)
    candidates = [
        ds / "xView_train.geojson",
        ds / "xview_train.geojson",
        ds / "train_labels" / "xView_train.geojson",
        ds / "xView_train.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)

    # Search for any .geojson file
    geojson_files = list(ds.rglob("*.geojson"))
    if len(geojson_files) == 1:
        return str(geojson_files[0])
    elif len(geojson_files) > 1:
        train_files = [f for f in geojson_files if "train" in f.name.lower()]
        if train_files:
            return str(train_files[0])
        return str(geojson_files[0])

    return None


def find_image_dir(data_dir: str) -> str | None:
    """Find the xView training images directory.

    Args:
        data_dir: Directory to search.

    Returns:
        Path to the images directory, or None if not found.
    """
    ds = Path(data_dir)
    candidates = [
        ds / "train_images",
        ds / "images",
        ds / "train",
    ]
    for candidate in candidates:
        if candidate.is_dir():
            tif_files = list(candidate.glob("*.tif"))
            if tif_files:
                return str(candidate)

    # Check if TIFFs are directly in the data_dir
    tif_files = list(ds.glob("*.tif"))
    if tif_files:
        return str(ds)

    return None


# ---------------------------------------------------------------------------
# Disk space check
# ---------------------------------------------------------------------------


def check_disk_space(path: str, required_gb: float = MIN_DISK_SPACE_GB) -> dict[str, Any]:
    """Check that sufficient disk space is available.

    Args:
        path: Directory to check (or parent if it doesn't exist yet).
        required_gb: Minimum free space required in GB.

    Returns:
        Dict with disk space info and warning if insufficient.
    """
    check_path = path
    while not os.path.exists(check_path):
        check_path = os.path.dirname(check_path)
        if not check_path:
            check_path = "/"
            break

    stat = shutil.disk_usage(check_path)
    free_gb = stat.free / (1024**3)

    result = {
        "free_gb": round(free_gb, 1),
        "required_gb": required_gb,
        "sufficient": free_gb >= required_gb,
    }

    if not result["sufficient"]:
        result["warning"] = (
            f"Insufficient disk space: need {required_gb:.1f} GB, "
            f"have {free_gb:.1f} GB free"
        )

    return result


# ---------------------------------------------------------------------------
# GeoTIFF validation
# ---------------------------------------------------------------------------


def verify_geotiff_images(image_dir: str, sample_count: int = 5) -> dict[str, Any]:
    """Verify GeoTIFF images are valid and readable via rasterio.

    Opens a sample of TIF files to check headers, CRS, bands, and
    basic readability.

    Args:
        image_dir: Directory containing .tif files.
        sample_count: Number of files to sample for deep validation.

    Returns:
        Dict with validation results.
    """
    result: dict[str, Any] = {
        "valid": True,
        "total_files": 0,
        "sampled": 0,
        "valid_files": 0,
        "invalid_files": [],
        "crs": None,
        "sample_dimensions": [],
        "errors": [],
        "warnings": [],
    }

    tif_files = sorted(Path(image_dir).glob("*.tif"))
    result["total_files"] = len(tif_files)

    if len(tif_files) == 0:
        result["valid"] = False
        result["errors"].append(f"No .tif files found in {image_dir}")
        return result

    # Compute total size
    total_bytes = sum(f.stat().st_size for f in tif_files)
    result["total_size_gb"] = round(total_bytes / (1024**3), 1)

    # Sample files for deep validation
    import random

    rng = random.Random(42)
    if len(tif_files) <= sample_count:
        sample_files = tif_files
    else:
        # Always include first and last, random middle
        indices = {0, len(tif_files) - 1}
        while len(indices) < min(sample_count, len(tif_files)):
            indices.add(rng.randint(0, len(tif_files) - 1))
        sample_files = [tif_files[i] for i in sorted(indices)]

    result["sampled"] = len(sample_files)

    try:
        import rasterio
    except ImportError:
        result["warnings"].append(
            "rasterio not installed -- cannot validate GeoTIFF headers. "
            "Install with: pip install rasterio"
        )
        return result

    for tif_path in sample_files:
        try:
            with rasterio.open(str(tif_path)) as src:
                result["valid_files"] += 1
                if result["crs"] is None and src.crs is not None:
                    result["crs"] = str(src.crs)
                result["sample_dimensions"].append({
                    "file": tif_path.name,
                    "width": src.width,
                    "height": src.height,
                    "bands": src.count,
                    "dtype": str(src.dtypes[0]),
                })
        except Exception as exc:
            result["invalid_files"].append({
                "file": tif_path.name,
                "error": str(exc),
            })

    if result["invalid_files"]:
        result["warnings"].append(
            f"{len(result['invalid_files'])} sampled file(s) could not be opened"
        )

    return result


# ---------------------------------------------------------------------------
# GeoJSON validation
# ---------------------------------------------------------------------------


def verify_geojson(geojson_path: str) -> dict[str, Any]:
    """Verify xView GeoJSON annotation file structure.

    Parses the GeoJSON, checks for required properties (type_id,
    bounds_imcoords, image_id), and counts features.

    Args:
        geojson_path: Path to the xView GeoJSON file.

    Returns:
        Dict with validation results.
    """
    result: dict[str, Any] = {
        "valid": True,
        "total_features": 0,
        "has_type_id": False,
        "has_bounds_imcoords": False,
        "has_image_id": False,
        "coordinate_system": "unknown",
        "errors": [],
        "warnings": [],
    }

    geojson_file = Path(geojson_path)
    if not geojson_file.exists():
        result["valid"] = False
        result["errors"].append(f"GeoJSON file not found: {geojson_path}")
        return result

    file_size_mb = geojson_file.stat().st_size / (1024**2)
    result["file_size_mb"] = round(file_size_mb, 1)

    print(f"  Loading GeoJSON ({file_size_mb:.0f} MB)... this may take a moment.")
    try:
        with open(geojson_path) as f:
            data = json.load(f)
    except json.JSONDecodeError as exc:
        result["valid"] = False
        result["errors"].append(f"Invalid JSON: {exc}")
        return result
    except MemoryError:
        result["valid"] = False
        result["errors"].append(
            "GeoJSON too large to load in memory. "
            "Need 32+ GB RAM for xView annotations."
        )
        return result

    if "features" not in data:
        result["valid"] = False
        result["errors"].append("GeoJSON missing 'features' key")
        return result

    features = data["features"]
    result["total_features"] = len(features)

    if len(features) == 0:
        result["valid"] = False
        result["errors"].append("GeoJSON has 0 features")
        return result

    # Check first feature for expected properties
    first_props = features[0].get("properties", {})
    result["has_type_id"] = "type_id" in first_props
    result["has_bounds_imcoords"] = "bounds_imcoords" in first_props
    result["has_image_id"] = "image_id" in first_props

    if not result["has_type_id"]:
        result["warnings"].append(
            "First feature missing 'type_id'. Class analysis may not work."
        )

    # Detect coordinate system from geometry
    first_geom = features[0].get("geometry", {})
    if first_geom.get("coordinates"):
        coords = first_geom["coordinates"]
        # Flatten nested coordinates
        while isinstance(coords, list) and isinstance(coords[0], list):
            coords = coords[0]
        if len(coords) >= 2:
            x, y = float(coords[0]), float(coords[1])
            if -180 <= x <= 180 and -90 <= y <= 90:
                result["coordinate_system"] = "geographic (WGS84)"
            else:
                result["coordinate_system"] = "pixel"

    return result


def analyze_class_distribution(geojson_path: str) -> dict[str, Any]:
    """Analyze full class distribution across all 60 xView classes.

    Loads the GeoJSON, counts annotations per type_id, identifies
    vehicle-specific classes, and reports distribution statistics.

    Args:
        geojson_path: Path to xView GeoJSON file.

    Returns:
        Dict with class counts, vehicle analysis, and imbalance warnings.
    """
    result: dict[str, Any] = {
        "all_classes": {},
        "vehicle_classes": {},
        "total_objects": 0,
        "total_vehicles": 0,
        "unique_images": 0,
        "errors": [],
        "warnings": [],
    }

    print("  Analyzing class distribution...")
    try:
        with open(geojson_path) as f:
            data = json.load(f)
    except Exception as exc:
        result["errors"].append(f"Failed to load GeoJSON: {exc}")
        return result

    features = data.get("features", [])
    result["total_objects"] = len(features)

    # Count by type_id and collect image_ids
    class_counts: dict[int, int] = {}
    image_ids: set[str] = set()

    for feature in features:
        props = feature.get("properties", {})

        type_id = props.get("type_id")
        if type_id is not None:
            type_id = int(type_id)
            class_counts[type_id] = class_counts.get(type_id, 0) + 1

        img_id = props.get("image_id")
        if img_id is not None:
            image_ids.add(str(img_id))

    result["all_classes"] = dict(sorted(class_counts.items()))
    result["unique_images"] = len(image_ids)

    # Map to our 5-class vehicle grouping
    coarse_counts: dict[str, dict[str, Any]] = {}
    total_vehicles = 0

    for class_name, type_ids in VEHICLE_TYPE_IDS.items():
        class_total = 0
        source_breakdown: dict[str, int] = {}
        for tid in type_ids:
            count = class_counts.get(tid, 0)
            class_total += count
            if count > 0:
                xview_name = XVIEW_CLASSES.get(tid, f"type_{tid}")
                source_breakdown[xview_name] = count

        total_vehicles += class_total
        coarse_counts[class_name] = {
            "count": class_total,
            "source_breakdown": source_breakdown,
        }

    result["total_vehicles"] = total_vehicles
    result["vehicle_classes"] = coarse_counts

    # Compute percentages
    for class_name, info in coarse_counts.items():
        if total_vehicles > 0:
            info["percentage"] = round(info["count"] / total_vehicles * 100, 1)
        else:
            info["percentage"] = 0.0

    # Check for class imbalance
    if total_vehicles > 0:
        counts = {k: v["count"] for k, v in coarse_counts.items()}
        nonzero_counts = {k: v for k, v in counts.items() if v > 0}
        if len(nonzero_counts) >= 2:
            max_class = max(nonzero_counts, key=nonzero_counts.get)
            min_class = min(nonzero_counts, key=nonzero_counts.get)
            ratio = nonzero_counts[max_class] / nonzero_counts[min_class]
            if ratio > 50:
                result["warnings"].append(
                    f"Severe class imbalance: {max_class}:{min_class} "
                    f"ratio is {ratio:.0f}:1"
                )
            elif ratio > 10:
                result["warnings"].append(
                    f"Class imbalance: {max_class}:{min_class} "
                    f"ratio is {ratio:.0f}:1"
                )

    # Count non-vehicle objects
    non_vehicle_count = sum(
        count for tid, count in class_counts.items()
        if tid not in ALL_VEHICLE_TYPE_IDS
    )
    result["non_vehicle_count"] = non_vehicle_count

    return result


# ---------------------------------------------------------------------------
# Full verification pipeline
# ---------------------------------------------------------------------------


def verify_full(data_dir: str) -> dict[str, Any]:
    """Run full verification on the xView dataset.

    Checks for images, annotations, validates file structures,
    and reports class distribution.

    Args:
        data_dir: Path to the xView data directory.

    Returns:
        Combined verification results.
    """
    results: dict[str, Any] = {
        "valid": True,
        "disk_space": {},
        "images": {},
        "annotations": {},
        "class_analysis": {},
        "errors": [],
        "warnings": [],
    }

    ds = Path(data_dir)
    if not ds.exists():
        results["valid"] = False
        results["errors"].append(f"Data directory not found: {data_dir}")
        return results

    # Check disk space
    disk = check_disk_space(data_dir)
    results["disk_space"] = disk
    if not disk["sufficient"]:
        results["warnings"].append(disk.get("warning", "Insufficient disk space"))

    # ---------------------------------------------------------------
    # Step 1: Find and verify images
    # ---------------------------------------------------------------
    print("\n  [1/3] Checking images...")
    image_dir = find_image_dir(data_dir)
    if image_dir is None:
        results["valid"] = False
        results["errors"].append(
            f"No image directory found in {data_dir}. "
            "Expected train_images/ with .tif files."
        )
        results["images"] = {"found": False}
    else:
        print(f"  Found images at: {image_dir}")
        img_result = verify_geotiff_images(image_dir)
        results["images"] = img_result
        results["images"]["directory"] = image_dir

        if img_result["total_files"] == 0:
            results["valid"] = False
            results["errors"].append("No .tif image files found")
        else:
            print(f"  Total images: {img_result['total_files']}")
            if img_result.get("total_size_gb"):
                print(f"  Total size: {img_result['total_size_gb']} GB")
            if img_result.get("crs"):
                print(f"  CRS: {img_result['crs']}")
            if img_result.get("sample_dimensions"):
                sample = img_result["sample_dimensions"][0]
                print(
                    f"  Sample: {sample['file']} "
                    f"({sample['width']}x{sample['height']}, "
                    f"{sample['bands']} bands, {sample['dtype']})"
                )
            # Aggregate sub-errors
            results["errors"].extend(img_result.get("errors", []))
            results["warnings"].extend(img_result.get("warnings", []))

    # ---------------------------------------------------------------
    # Step 2: Find and verify GeoJSON
    # ---------------------------------------------------------------
    print("\n  [2/3] Checking annotations...")
    geojson_path = find_geojson(data_dir)
    if geojson_path is None:
        results["valid"] = False
        results["errors"].append(
            f"No GeoJSON annotation file found in {data_dir}. "
            "Expected xView_train.geojson."
        )
        results["annotations"] = {"found": False}
    else:
        print(f"  Found annotations at: {geojson_path}")
        ann_result = verify_geojson(geojson_path)
        results["annotations"] = ann_result
        results["annotations"]["path"] = geojson_path

        if not ann_result["valid"]:
            results["valid"] = False
        else:
            print(f"  Total features: {ann_result['total_features']:,}")
            print(f"  File size: {ann_result.get('file_size_mb', 0):.0f} MB")
            print(f"  Coordinate system: {ann_result.get('coordinate_system', 'unknown')}")
            print(f"  Has type_id: {ann_result.get('has_type_id', False)}")
            print(f"  Has bounds_imcoords: {ann_result.get('has_bounds_imcoords', False)}")
            print(f"  Has image_id: {ann_result.get('has_image_id', False)}")

        results["errors"].extend(ann_result.get("errors", []))
        results["warnings"].extend(ann_result.get("warnings", []))

    # ---------------------------------------------------------------
    # Step 3: Analyze class distribution
    # ---------------------------------------------------------------
    print("\n  [3/3] Analyzing class distribution...")
    if geojson_path and results["annotations"].get("valid"):
        class_analysis = analyze_class_distribution(geojson_path)
        results["class_analysis"] = class_analysis

        print(f"  Total objects: {class_analysis['total_objects']:,}")
        print(f"  Unique images referenced: {class_analysis['unique_images']}")
        print(f"  Total vehicle annotations: {class_analysis['total_vehicles']:,}")
        print(f"  Non-vehicle annotations: {class_analysis.get('non_vehicle_count', 0):,}")

        # Print vehicle class breakdown
        vc = class_analysis.get("vehicle_classes", {})
        if vc:
            print()
            print(f"  {'Class':<12} {'Count':>10} {'Pct':>7}  Source xView classes")
            print(f"  {'-'*12} {'-'*10} {'-'*7}  {'-'*40}")
            for class_name, info in vc.items():
                source_str = ", ".join(
                    f"{name} ({count:,})"
                    for name, count in sorted(
                        info["source_breakdown"].items(),
                        key=lambda x: x[1],
                        reverse=True,
                    )
                )
                if not source_str:
                    source_str = "(none found)"
                print(
                    f"  {class_name:<12} {info['count']:>10,} "
                    f"{info['percentage']:>6.1f}%  {source_str}"
                )

        for warning in class_analysis.get("warnings", []):
            print(f"\n  WARNING: {warning}")

        results["warnings"].extend(class_analysis.get("warnings", []))
        results["errors"].extend(class_analysis.get("errors", []))

    return results


def print_verification_report(results: dict[str, Any]) -> None:
    """Print a formatted verification report.

    Args:
        results: Results from verify_full().
    """
    print("\n" + "=" * 60)
    print("  xVIEW DATASET VERIFICATION REPORT")
    print("=" * 60)

    # Disk space
    disk = results.get("disk_space", {})
    if disk:
        status = "OK" if disk.get("sufficient") else "LOW"
        print(f"  Disk space: {disk.get('free_gb', '?')} GB available [{status}]")

    # Images summary
    images = results.get("images", {})
    if images.get("directory"):
        size = images.get("total_size_gb", "?")
        print(f"  Images: {images.get('total_files', 0)} files ({size} GB)")
    else:
        print("  Images: NOT FOUND")

    # Annotations summary
    annotations = results.get("annotations", {})
    if annotations.get("path"):
        print(f"  Annotations: {annotations.get('total_features', 0):,} features")
    else:
        print("  Annotations: NOT FOUND")

    # Vehicle summary
    ca = results.get("class_analysis", {})
    if ca.get("total_vehicles"):
        print(f"  Vehicle annotations: {ca['total_vehicles']:,}")

    # Errors
    if results.get("errors"):
        print("\n  Errors:")
        for e in results["errors"]:
            print(f"    - {e}")

    # Warnings
    if results.get("warnings"):
        print("\n  Warnings:")
        for w in results["warnings"]:
            print(f"    - {w}")

    overall = "PASS" if results["valid"] else "FAIL"
    print(f"\n  Overall: {overall}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Directory initialization
# ---------------------------------------------------------------------------


def init_directory_structure(output_dir: str) -> None:
    """Initialize directory structure for xView processing.

    Args:
        output_dir: Root directory for xView dataset outputs.
    """
    output = Path(output_dir)

    dirs = [
        output / "raw",
        output / "remapped",
        output / "coco" / "train" / "images",
        output / "coco" / "valid" / "images",
        output / "coco" / "test" / "images",
        output / "checkpoints",
        output / "logs",
    ]

    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)

    print(f"  Initialized directory structure at {output_dir}")
    print(f"\n  Next steps:")
    print(f"    1. Download xView files from xviewdataset.org")
    print(f"    2. Extract to xview_raw/ (or your data_dir)")
    print(f"    3. Verify: python scripts/download_xview.py --verify --data_dir xview_raw/")
    print(f"    4. Preprocess: python scripts/preprocess_xview.py --input xview_raw/xView_train.geojson --output {output}/remapped/xview_remapped.geojson")


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
            "Verify and analyze the xView dataset for vehicle detection "
            "fine-tuning. xView requires manual registration at "
            "xviewdataset.org."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify downloaded dataset files (images + annotations)",
    )
    parser.add_argument(
        "--init",
        action="store_true",
        help="Initialize output directory structure",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="xview_raw",
        help="Directory containing downloaded xView files (default: xview_raw)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="xview_dataset",
        help="Output directory for processed files (default: xview_dataset)",
    )
    parser.add_argument(
        "--show_all_classes",
        action="store_true",
        help="Show distribution for all 60 xView classes (not just vehicles)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run xView download verification and analysis.

    Args:
        argv: Command-line arguments (defaults to sys.argv[1:]).

    Returns:
        Exit code: 0 on success, 1 on failure.
    """
    args = parse_args(argv)

    print("=" * 60)
    print("  xView Dataset: Verification & Analysis")
    print("=" * 60)

    # Always show license
    print(LICENSE_TEXT)

    # If no action specified, show registration instructions
    if not args.verify and not args.init:
        print(REGISTRATION_INSTRUCTIONS)
        return 0

    # Initialize directory structure
    if args.init:
        print("\n  Initializing output directory...")
        init_directory_structure(args.output_dir)
        if not args.verify:
            return 0

    # Verify downloaded dataset
    if args.verify:
        print(f"\n  Verifying xView dataset at: {args.data_dir}")

        if not Path(args.data_dir).exists():
            print(f"\n  ERROR: Directory not found: {args.data_dir}")
            print(REGISTRATION_INSTRUCTIONS)
            return 1

        results = verify_full(args.data_dir)

        # Show all 60 classes if requested
        if args.show_all_classes:
            all_classes = results.get("class_analysis", {}).get("all_classes", {})
            if all_classes:
                total = sum(all_classes.values())
                print(f"\n  Full class distribution ({len(all_classes)} classes, {total:,} objects):")
                print(f"\n  {'ID':>4}  {'Class':<30} {'Count':>10} {'Pct':>7}")
                print(f"  {'-'*4}  {'-'*30} {'-'*10} {'-'*7}")
                for type_id in sorted(all_classes.keys()):
                    count = all_classes[type_id]
                    pct = count / total * 100 if total > 0 else 0
                    name = XVIEW_CLASSES.get(type_id, f"Unknown (id={type_id})")
                    marker = " *" if type_id in ALL_VEHICLE_TYPE_IDS else ""
                    print(
                        f"  {type_id:>4}  {name:<30} "
                        f"{count:>10,} {pct:>6.1f}%{marker}"
                    )
                print(f"\n  * = Vehicle class (included in fine-tuning)")

        print_verification_report(results)

        if results["valid"]:
            geojson = results["annotations"].get("path", "xview_raw/xView_train.geojson")
            print("\n  xView dataset verified. Ready for preprocessing.")
            print(f"\n  Next step:")
            print(f"    python scripts/preprocess_xview.py \\")
            print(f"      --input {geojson} \\")
            print(f"      --output {args.output_dir}/xview_remapped.geojson")
        else:
            print("\n  Dataset verification failed. Check errors above.")
            print("  If files are missing, follow the download instructions:")
            print("    python scripts/download_xview.py")

        return 0 if results["valid"] else 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
