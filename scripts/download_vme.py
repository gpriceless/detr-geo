#!/usr/bin/env python3
"""Download and validate the VME (Vehicles in the Middle East) dataset from Zenodo.

Downloads VME_CDSI_datasets.zip, extracts it, validates COCO JSON structure,
and reports dataset statistics. The VME dataset contains 4,303 tiles at 512x512
with 100K+ vehicle annotations (Car, Bus, Truck) at 30-50cm GSD from Maxar
satellite imagery.

Source: https://zenodo.org/records/14185684
License: CC BY-NC-ND 4.0 (Non-commercial, No Derivatives)

Usage:
    python scripts/download_vme.py --output_dir vme_dataset --accept_license
    python scripts/download_vme.py --output_dir vme_dataset --verify_only
    python scripts/download_vme.py --help
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import zipfile
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ZENODO_RECORD_ID = "14185684"
ZENODO_DOWNLOAD_URL = (
    f"https://zenodo.org/records/{ZENODO_RECORD_ID}/files/VME_CDSI_datasets.zip"
)
ZIP_FILENAME = "VME_CDSI_datasets.zip"

# Expected dataset structure after extraction
# VME uses research-style layout: satellite_images/ + annotations_HBB/{train,val,test}.json
# NOT Roboflow-style: train/images/ + train/_annotations.coco.json
EXPECTED_IMAGE_DIR = "satellite_images"
EXPECTED_ANNOTATION_DIR = "annotations_HBB"
EXPECTED_ANNOTATION_FILES = ("train.json", "val.json", "test.json")
# Expected image counts from VME documentation
EXPECTED_TOTAL_IMAGES = 4282
EXPECTED_TRAIN_ANNOTATIONS = 70270  # Approximate
EXPECTED_CATEGORIES = ["Car", "Bus", "Truck"]

# Minimum disk space in bytes (2 GB for extraction + buffer)
MIN_DISK_SPACE_BYTES = 2 * 1024 * 1024 * 1024

LICENSE_TEXT = """
================================================================================
  VME Dataset License: CC BY-NC-ND 4.0
================================================================================

  This dataset is licensed under Creative Commons
  Attribution-NonCommercial-NoDerivatives 4.0 International.

  By downloading, you agree to:
  - Use for NON-COMMERCIAL research purposes only
  - Provide attribution to the original authors
  - NOT distribute derivative works (including fine-tuned model weights)

  Citation:
    Al-Emadi et al., "VME: Vehicles in the Middle East Dataset for
    Object Detection in Satellite Imagery" (2025)

  Full license: https://creativecommons.org/licenses/by-nc-nd/4.0/
================================================================================
"""


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------


def check_disk_space(path: str, required_bytes: int = MIN_DISK_SPACE_BYTES) -> None:
    """Check that sufficient disk space is available.

    Args:
        path: Directory to check (or parent if it doesn't exist yet).
        required_bytes: Minimum free space required in bytes.

    Raises:
        OSError: If insufficient disk space.
    """
    check_path = path
    while not os.path.exists(check_path):
        check_path = os.path.dirname(check_path)
        if not check_path:
            check_path = "/"
            break

    stat = shutil.disk_usage(check_path)
    if stat.free < required_bytes:
        required_gb = required_bytes / (1024**3)
        free_gb = stat.free / (1024**3)
        raise OSError(
            f"Insufficient disk space: need {required_gb:.1f} GB, "
            f"have {free_gb:.1f} GB free at {check_path}"
        )


def download_file(
    url: str,
    output_path: str,
    chunk_size: int = 8192,
) -> str:
    """Download a file with progress display and resume support.

    Uses HTTP Range requests to resume partial downloads.

    Args:
        url: URL to download from.
        output_path: Path to save the downloaded file.
        chunk_size: Size of download chunks in bytes.

    Returns:
        Path to the downloaded file.

    Raises:
        RuntimeError: If download fails.
    """
    try:
        import requests
    except ImportError:
        raise RuntimeError(
            "requests is required for download. Install with: pip install requests"
        )

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    # Check for partial download (resume support)
    existing_size = 0
    if output.exists():
        existing_size = output.stat().st_size

    headers = {}
    if existing_size > 0:
        headers["Range"] = f"bytes={existing_size}-"
        print(f"  Resuming download from byte {existing_size:,}")

    try:
        response = requests.get(url, headers=headers, stream=True, timeout=60)

        # Handle resume response
        if existing_size > 0 and response.status_code == 416:
            # Range not satisfiable -- file is already complete
            print("  Download already complete.")
            return str(output)

        if response.status_code not in (200, 206):
            response.raise_for_status()

        total_size = int(response.headers.get("content-length", 0))
        if response.status_code == 206:
            total_size += existing_size
        elif existing_size > 0:
            # Server doesn't support range requests, start over
            existing_size = 0

        mode = "ab" if existing_size > 0 else "wb"
        downloaded = existing_size

        # Try to use tqdm for progress bar
        try:
            from tqdm import tqdm

            progress = tqdm(
                total=total_size if total_size > 0 else None,
                initial=existing_size,
                unit="B",
                unit_scale=True,
                desc="Downloading VME",
            )
        except ImportError:
            progress = None

        with open(output, mode) as f:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if progress is not None:
                        progress.update(len(chunk))
                    elif total_size > 0 and downloaded % (10 * 1024 * 1024) < chunk_size:
                        pct = (downloaded / total_size) * 100
                        print(
                            f"  Downloaded {downloaded / (1024**2):.0f} MB / "
                            f"{total_size / (1024**2):.0f} MB ({pct:.1f}%)"
                        )

        if progress is not None:
            progress.close()

    except requests.exceptions.ConnectionError as exc:
        print(
            f"\n  Download interrupted. Partial file preserved at {output}."
        )
        print("  Re-run to resume download.")
        raise RuntimeError(f"Download interrupted: {exc}") from exc

    except Exception as exc:
        raise RuntimeError(f"Download failed: {exc}") from exc

    return str(output)


def compute_checksum(file_path: str, algorithm: str = "md5") -> str:
    """Compute file checksum.

    Args:
        file_path: Path to file.
        algorithm: Hash algorithm ("md5" or "sha256").

    Returns:
        Hex digest string.
    """
    hasher = hashlib.new(algorithm)
    with open(file_path, "rb") as f:
        while True:
            chunk = f.read(65536)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------


def extract_zip(
    zip_path: str,
    output_dir: str,
    skip_existing: bool = False,
) -> str:
    """Extract ZIP archive preserving directory structure.

    Handles nested directory structures by flattening if needed
    (VME ZIP may contain a top-level directory).

    Args:
        zip_path: Path to ZIP file.
        output_dir: Directory to extract into.
        skip_existing: If True, skip extraction if output dir exists.

    Returns:
        Path to the extracted dataset directory.

    Raises:
        RuntimeError: If extraction fails or ZIP is corrupted.
    """
    output = Path(output_dir)

    if skip_existing and output.exists():
        # Check if it looks like a valid extraction
        train_dir = output / "train"
        valid_dir = output / "valid"
        if train_dir.exists() and valid_dir.exists():
            print("  Dataset already extracted, skipping.")
            return str(output)

    check_disk_space(str(output))

    print(f"  Extracting {zip_path}...")
    output.mkdir(parents=True, exist_ok=True)

    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            # Test ZIP integrity
            bad_file = zf.testzip()
            if bad_file is not None:
                raise RuntimeError(
                    f"Corrupted ZIP: bad file '{bad_file}'. "
                    "Delete the ZIP and re-download."
                )

            # Check for nested directory (common in Zenodo downloads)
            names = zf.namelist()
            # Find the common prefix
            top_dirs = set()
            for name in names:
                parts = name.split("/")
                if len(parts) > 1:
                    top_dirs.add(parts[0])

            zf.extractall(str(output))

            # If everything is under a single top-level dir, flatten it
            if len(top_dirs) == 1:
                nested_dir = output / list(top_dirs)[0]
                if nested_dir.is_dir():
                    # Move contents up one level
                    for item in nested_dir.iterdir():
                        target = output / item.name
                        if target.exists():
                            if target.is_dir():
                                shutil.rmtree(target)
                            else:
                                target.unlink()
                        shutil.move(str(item), str(target))
                    # Remove empty nested dir
                    if nested_dir.exists() and not any(nested_dir.iterdir()):
                        nested_dir.rmdir()

    except zipfile.BadZipFile as exc:
        raise RuntimeError(
            f"Corrupted ZIP file: {exc}. Delete {zip_path} and re-download."
        ) from exc

    print(f"  Extracted to {output}")
    return str(output)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_dataset_structure(dataset_dir: str) -> dict[str, Any]:
    """Validate that extracted dataset has expected VME structure.

    VME uses research-style layout:
    - satellite_images/ containing all .png tiles
    - annotations_HBB/ containing train.json, val.json, test.json in COCO format

    Args:
        dataset_dir: Path to extracted dataset root.

    Returns:
        Statistics dict with image counts and annotation counts.

    Raises:
        FileNotFoundError: If required files are missing.
    """
    ds = Path(dataset_dir)
    stats: dict[str, Any] = {"valid": True, "errors": [], "warnings": []}

    # Check image directory
    image_dir = ds / EXPECTED_IMAGE_DIR
    if not image_dir.exists():
        stats["errors"].append(f"Missing {EXPECTED_IMAGE_DIR}/ directory")
        stats["valid"] = False
    else:
        # Count images
        image_files = list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png"))
        stats["total_images"] = len(image_files)
        if stats["total_images"] == 0:
            stats["errors"].append(f"{EXPECTED_IMAGE_DIR}/ contains 0 images")
            stats["valid"] = False
        elif abs(stats["total_images"] - EXPECTED_TOTAL_IMAGES) > 200:
            stats["warnings"].append(
                f"{EXPECTED_IMAGE_DIR}/ has {stats['total_images']} images "
                f"(expected ~{EXPECTED_TOTAL_IMAGES})"
            )

    # Check annotation directory and files
    ann_dir = ds / EXPECTED_ANNOTATION_DIR
    if not ann_dir.exists():
        stats["errors"].append(f"Missing {EXPECTED_ANNOTATION_DIR}/ directory")
        stats["valid"] = False
    else:
        for ann_file in EXPECTED_ANNOTATION_FILES:
            ann_path = ann_dir / ann_file
            if not ann_path.exists():
                stats["errors"].append(f"Missing {EXPECTED_ANNOTATION_DIR}/{ann_file}")
                stats["valid"] = False
            else:
                # Validate COCO JSON structure
                try:
                    with open(ann_path) as f:
                        data = json.load(f)
                    split_name = ann_file.replace(".json", "")
                    stats[f"{split_name}_images"] = len(data.get("images", []))
                    stats[f"{split_name}_annotations"] = len(data.get("annotations", []))

                    # Validate categories
                    categories = [c["name"] for c in data.get("categories", [])]
                    for expected_cat in EXPECTED_CATEGORIES:
                        if expected_cat not in categories:
                            stats["warnings"].append(
                                f"{ann_file} missing expected category: {expected_cat}"
                            )
                except json.JSONDecodeError as e:
                    stats["errors"].append(f"{ann_file} is not valid JSON: {e}")
                    stats["valid"] = False

    # Summary statistics
    if stats["valid"]:
        total_annotations = sum(
            stats.get(f"{split}_annotations", 0)
            for split in ["train", "val", "test"]
        )
        stats["total_annotations"] = total_annotations

    return stats


def validate_coco_json(json_path: str) -> dict[str, Any]:
    """Validate COCO JSON annotation file structure.

    Checks that the JSON contains required keys (images, annotations,
    categories) and that entries have expected fields.

    Args:
        json_path: Path to COCO annotation JSON file.

    Returns:
        Validation result dict with categories, counts, and any issues.

    Raises:
        ValueError: If JSON is corrupted or missing required fields.
    """
    try:
        with open(json_path) as f:
            data = json.load(f)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Corrupted annotation file: invalid JSON in {json_path}: {exc}"
        )

    result: dict[str, Any] = {"valid": True, "errors": [], "warnings": []}

    # Check required top-level keys
    required_keys = {"images", "annotations", "categories"}
    missing = required_keys - set(data.keys())
    if missing:
        result["errors"].append(f"Missing required keys: {missing}")
        result["valid"] = False
        return result

    # Validate categories
    categories = data["categories"]
    if not categories:
        result["errors"].append("Invalid COCO format: missing categories")
        result["valid"] = False
        return result

    cat_ids = []
    cat_names = []
    for cat in categories:
        if "id" not in cat or "name" not in cat:
            result["errors"].append(
                f"Category missing id or name: {cat}"
            )
            result["valid"] = False
        else:
            cat_ids.append(cat["id"])
            cat_names.append(cat["name"])

    result["categories"] = dict(zip(cat_ids, cat_names))
    result["num_categories"] = len(categories)

    # Check category indexing
    min_cat_id = min(cat_ids) if cat_ids else 0
    if min_cat_id == 0:
        result["category_indexing"] = "0-indexed"
    elif min_cat_id == 1:
        result["category_indexing"] = "1-indexed"
        result["warnings"].append(
            "Category IDs are 1-indexed. Remapping may be needed for "
            "some frameworks. Use --remap_categories flag if needed."
        )
    else:
        result["category_indexing"] = f"starts-at-{min_cat_id}"
        result["warnings"].append(
            f"Category IDs start at {min_cat_id} (unusual). "
            "Verify compatibility with your training framework."
        )

    # Validate images
    images = data["images"]
    result["num_images"] = len(images)
    if images:
        sample = images[0]
        img_required = {"id", "file_name", "width", "height"}
        img_missing = img_required - set(sample.keys())
        if img_missing:
            result["warnings"].append(
                f"Image entries missing fields: {img_missing}"
            )

    # Validate annotations
    annotations = data["annotations"]
    result["num_annotations"] = len(annotations)
    if annotations:
        sample = annotations[0]
        ann_required = {"id", "image_id", "category_id", "bbox"}
        ann_missing = ann_required - set(sample.keys())
        if ann_missing:
            result["warnings"].append(
                f"Annotation entries missing fields: {ann_missing}"
            )

        # Check bbox format
        if "bbox" in sample:
            bbox = sample["bbox"]
            if not isinstance(bbox, list) or len(bbox) != 4:
                result["errors"].append(
                    f"Invalid bbox format: expected [x, y, w, h], got {bbox}"
                )
                result["valid"] = False

    # Annotation statistics by category
    if cat_ids and annotations:
        cat_counts: dict[int, int] = {cid: 0 for cid in cat_ids}
        for ann in annotations:
            cid = ann.get("category_id")
            if cid in cat_counts:
                cat_counts[cid] += 1
            elif cid is not None:
                cat_counts.setdefault(cid, 0)
                cat_counts[cid] += 1
        result["annotations_per_category"] = cat_counts

    return result


def validate_full(dataset_dir: str) -> dict[str, Any]:
    """Run all validations on a dataset directory.

    Handles both VME's native structure (satellite_images/ + annotations_HBB/)
    and standard COCO layout (train/valid/ subdirs).

    Args:
        dataset_dir: Path to dataset root.

    Returns:
        Combined validation results.
    """
    results: dict[str, Any] = {"valid": True, "structure": {}, "coco": {}}

    # Structure validation
    print("  Validating dataset structure...")
    struct_result = validate_dataset_structure(dataset_dir)
    results["structure"] = struct_result
    if not struct_result["valid"]:
        results["valid"] = False

    # COCO JSON validation
    ds = Path(dataset_dir)
    ann_dir = ds / EXPECTED_ANNOTATION_DIR

    if ann_dir.exists():
        # VME native structure
        for ann_file in EXPECTED_ANNOTATION_FILES:
            ann_path = ann_dir / ann_file
            if ann_path.exists():
                split = ann_file.replace(".json", "")
                print(f"  Validating {EXPECTED_ANNOTATION_DIR}/{ann_file}...")
                try:
                    coco_result = validate_coco_json(str(ann_path))
                    results["coco"][split] = coco_result
                    if not coco_result["valid"]:
                        results["valid"] = False
                except ValueError as exc:
                    results["coco"][split] = {"valid": False, "errors": [str(exc)]}
                    results["valid"] = False
    else:
        # Standard COCO layout (train/valid/)
        for split in ("train", "valid"):
            ann_path = ds / split / "_annotations.coco.json"
            if ann_path.exists():
                print(f"  Validating {split}/_annotations.coco.json...")
                try:
                    coco_result = validate_coco_json(str(ann_path))
                    results["coco"][split] = coco_result
                    if not coco_result["valid"]:
                        results["valid"] = False
                except ValueError as exc:
                    results["coco"][split] = {"valid": False, "errors": [str(exc)]}
                    results["valid"] = False

    return results


def print_validation_report(results: dict[str, Any]) -> None:
    """Print a human-readable validation report.

    Args:
        results: Combined validation results from validate_full().
    """
    print("\n" + "=" * 60)
    print("  VME DATASET VALIDATION REPORT")
    print("=" * 60)

    struct = results["structure"]

    # Structure summary
    if "total_images" in struct:
        print(f"  Total images: {struct['total_images']}")
    if "total_annotations" in struct:
        print(f"  Total annotations: {struct['total_annotations']}")

    # Per-split counts
    for split in ["train", "val", "test"]:
        img_key = f"{split}_images"
        ann_key = f"{split}_annotations"
        if img_key in struct or ann_key in struct:
            imgs = struct.get(img_key, 0)
            anns = struct.get(ann_key, 0)
            print(f"  {split}: {imgs} images, {anns} annotations")

    # COCO validation details
    for split, coco in results.get("coco", {}).items():
        print(f"\n  {split} annotations:")
        if "num_images" in coco:
            print(f"    Images: {coco['num_images']}")
        if "num_annotations" in coco:
            print(f"    Annotations: {coco['num_annotations']}")
        if "categories" in coco:
            print(f"    Categories: {list(coco['categories'].values())}")
        if "category_indexing" in coco:
            print(f"    Category indexing: {coco['category_indexing']}")
        if "annotations_per_category" in coco:
            print("    Annotations per category:")
            cats = coco.get("categories", {})
            for cid, count in coco["annotations_per_category"].items():
                name = cats.get(cid, f"id={cid}")
                print(f"      {name}: {count:,}")

    # Errors and warnings
    all_errors = struct.get("errors", [])
    all_warnings = struct.get("warnings", [])
    for split_coco in results["coco"].values():
        all_errors.extend(split_coco.get("errors", []))
        all_warnings.extend(split_coco.get("warnings", []))

    if all_warnings:
        print("\n  Warnings:")
        for w in all_warnings:
            print(f"    - {w}")

    if all_errors:
        print("\n  Errors:")
        for e in all_errors:
            print(f"    - {e}")

    print(f"\n  Overall: {'PASS' if results['valid'] else 'FAIL'}")
    print("=" * 60)


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
        description="Download and validate the VME dataset from Zenodo.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="vme_dataset",
        help="Directory to save/extract the dataset (default: vme_dataset)",
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="Skip extraction if dataset directory already exists",
    )
    parser.add_argument(
        "--verify_only",
        action="store_true",
        help="Only validate existing dataset, skip download/extraction",
    )
    parser.add_argument(
        "--accept_license",
        action="store_true",
        help="Accept the CC BY-NC-ND 4.0 license terms without prompting",
    )
    parser.add_argument(
        "--remap_categories",
        action="store_true",
        help="Remap 1-indexed category IDs to 0-indexed",
    )
    parser.add_argument(
        "--url",
        type=str,
        default=ZENODO_DOWNLOAD_URL,
        help="Override download URL (for testing or mirrors)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the VME download and validation pipeline.

    Args:
        argv: Command-line arguments (defaults to sys.argv[1:]).

    Returns:
        Exit code: 0 on success, 1 on failure.
    """
    args = parse_args(argv)
    output_dir = args.output_dir

    # -------------------------------------------------------------------
    # License acceptance
    # -------------------------------------------------------------------
    if not args.verify_only:
        print(LICENSE_TEXT)
        if not args.accept_license:
            try:
                answer = input("  Accept license terms? (yes/no): ").strip().lower()
                if answer not in ("yes", "y"):
                    print("  Download cancelled.")
                    return 1
            except EOFError:
                print("  Non-interactive mode: use --accept_license flag.")
                return 1

    # -------------------------------------------------------------------
    # Verify-only mode
    # -------------------------------------------------------------------
    if args.verify_only:
        if not Path(output_dir).exists():
            print(f"  Dataset directory not found: {output_dir}")
            return 1

        results = validate_full(output_dir)
        print_validation_report(results)
        return 0 if results["valid"] else 1

    # -------------------------------------------------------------------
    # Download
    # -------------------------------------------------------------------
    zip_path = str(Path(output_dir) / ZIP_FILENAME)

    # Check if ZIP already exists and is complete
    if Path(zip_path).exists():
        print(f"  ZIP file exists: {zip_path}")
        size_mb = Path(zip_path).stat().st_size / (1024**2)
        print(f"  Size: {size_mb:.1f} MB")
    else:
        print(f"  Downloading from: {args.url}")
        check_disk_space(output_dir, MIN_DISK_SPACE_BYTES)

        try:
            download_file(args.url, zip_path)
        except RuntimeError as exc:
            print(f"\n  ERROR: {exc}")
            return 1

    # Compute checksum
    print("  Computing checksum...")
    checksum = compute_checksum(zip_path)
    print(f"  MD5: {checksum}")

    # -------------------------------------------------------------------
    # Extract
    # -------------------------------------------------------------------
    try:
        extract_zip(zip_path, output_dir, skip_existing=args.skip_existing)
    except RuntimeError as exc:
        print(f"\n  ERROR: {exc}")
        return 1

    # -------------------------------------------------------------------
    # Validate
    # -------------------------------------------------------------------
    results = validate_full(output_dir)
    print_validation_report(results)

    # -------------------------------------------------------------------
    # Optional category remapping
    # -------------------------------------------------------------------
    if args.remap_categories:
        ds = Path(output_dir)
        ann_dir = ds / EXPECTED_ANNOTATION_DIR

        if ann_dir.exists():
            # VME native structure
            for ann_file in EXPECTED_ANNOTATION_FILES:
                ann_path = ann_dir / ann_file
                split = ann_file.replace(".json", "")
                if ann_path.exists():
                    coco_result = results["coco"].get(split, {})
                    if coco_result.get("category_indexing") == "1-indexed":
                        print(f"  Remapping {split} categories from 1-indexed to 0-indexed...")
                        remap_categories_to_zero(str(ann_path))
                        print(f"  Done.")
        else:
            # Standard COCO layout
            for split in ("train", "valid"):
                ann_path = ds / split / "_annotations.coco.json"
                if ann_path.exists():
                    coco_result = results["coco"].get(split, {})
                    if coco_result.get("category_indexing") == "1-indexed":
                        print(f"  Remapping {split} categories from 1-indexed to 0-indexed...")
                        remap_categories_to_zero(str(ann_path))
                        print(f"  Done.")

    if results["valid"]:
        print("\n  VME dataset ready for training.")
    else:
        print("\n  Dataset validation failed. Check errors above.")

    return 0 if results["valid"] else 1


def remap_categories_to_zero(json_path: str) -> None:
    """Remap 1-indexed category IDs to 0-indexed in a COCO JSON file.

    Modifies the file in-place, updating both categories and annotations.

    Args:
        json_path: Path to COCO annotation JSON file.
    """
    with open(json_path) as f:
        data = json.load(f)

    # Find the offset
    cat_ids = [c["id"] for c in data["categories"]]
    min_id = min(cat_ids)
    if min_id == 0:
        return  # Already 0-indexed

    offset = min_id

    # Remap categories
    for cat in data["categories"]:
        cat["id"] -= offset

    # Remap annotations
    for ann in data["annotations"]:
        ann["category_id"] -= offset

    with open(json_path, "w") as f:
        json.dump(data, f, indent=2)


if __name__ == "__main__":
    sys.exit(main())
