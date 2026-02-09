"""Tests for scripts/download_vme.py -- VME dataset download and validation.

All tests use synthetic data and mocks -- no actual network calls or
real VME dataset downloads.
"""

from __future__ import annotations

import json
import os

# Import the download module
import sys
import zipfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from download_vme import (
    EXPECTED_ANNOTATION_FILES,
    check_disk_space,
    compute_checksum,
    extract_zip,
    main,
    parse_args,
    remap_categories_to_zero,
    validate_coco_json,
    validate_dataset_structure,
    validate_full,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def valid_coco_json(tmp_path: Path) -> Path:
    """Create a valid COCO JSON annotation file with VME structure."""
    coco = {
        "images": [
            {"id": 0, "file_name": "tile_0.jpg", "width": 512, "height": 512},
            {"id": 1, "file_name": "tile_1.jpg", "width": 512, "height": 512},
            {"id": 2, "file_name": "tile_2.jpg", "width": 512, "height": 512},
        ],
        "annotations": [
            {"id": 0, "image_id": 0, "category_id": 0, "bbox": [10, 10, 30, 20], "area": 600},
            {"id": 1, "image_id": 0, "category_id": 1, "bbox": [50, 50, 40, 30], "area": 1200},
            {"id": 2, "image_id": 1, "category_id": 0, "bbox": [100, 100, 25, 25], "area": 625},
            {"id": 3, "image_id": 1, "category_id": 2, "bbox": [200, 200, 60, 40], "area": 2400},
            {"id": 4, "image_id": 2, "category_id": 0, "bbox": [150, 50, 20, 15], "area": 300},
        ],
        "categories": [
            {"id": 0, "name": "Car"},
            {"id": 1, "name": "Bus"},
            {"id": 2, "name": "Truck"},
        ],
    }
    json_path = tmp_path / "_annotations.coco.json"
    json_path.write_text(json.dumps(coco, indent=2))
    return json_path


@pytest.fixture
def one_indexed_coco_json(tmp_path: Path) -> Path:
    """Create a COCO JSON with 1-indexed category IDs."""
    coco = {
        "images": [
            {"id": 0, "file_name": "tile_0.jpg", "width": 512, "height": 512},
        ],
        "annotations": [
            {"id": 0, "image_id": 0, "category_id": 1, "bbox": [10, 10, 30, 20], "area": 600},
            {"id": 1, "image_id": 0, "category_id": 2, "bbox": [50, 50, 40, 30], "area": 1200},
            {"id": 2, "image_id": 0, "category_id": 3, "bbox": [100, 100, 25, 25], "area": 625},
        ],
        "categories": [
            {"id": 1, "name": "Car"},
            {"id": 2, "name": "Bus"},
            {"id": 3, "name": "Truck"},
        ],
    }
    json_path = tmp_path / "one_indexed.json"
    json_path.write_text(json.dumps(coco, indent=2))
    return json_path


@pytest.fixture
def valid_vme_dataset(tmp_path: Path) -> Path:
    """Create a valid VME dataset directory structure with minimal files."""
    ds = tmp_path / "vme_dataset"

    # Create VME native structure: satellite_images/ + annotations_HBB/
    image_dir = ds / "satellite_images"
    image_dir.mkdir(parents=True)

    ann_dir = ds / "annotations_HBB"
    ann_dir.mkdir(parents=True)

    # Create images (shared across all splits)
    for i in range(13):
        (image_dir / f"tile_{i}.png").write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)

    # Create annotation files for each split
    splits_data = {
        "train": (10, 20),  # 10 images, 20 annotations
        "val": (2, 4),  # 2 images, 4 annotations
        "test": (1, 2),  # 1 image, 2 annotations
    }

    for split, (n_images, n_annotations) in splits_data.items():
        images = [{"id": i, "file_name": f"tile_{i}.png", "width": 512, "height": 512} for i in range(n_images)]
        annotations = [
            {
                "id": i,
                "image_id": i % n_images,
                "category_id": i % 3,
                "bbox": [10 + i, 10 + i, 30, 20],
                "area": 600,
            }
            for i in range(n_annotations)
        ]
        categories = [
            {"id": 0, "name": "Car"},
            {"id": 1, "name": "Bus"},
            {"id": 2, "name": "Truck"},
        ]
        coco = {"images": images, "annotations": annotations, "categories": categories}

        ann_path = ann_dir / f"{split}.json"
        ann_path.write_text(json.dumps(coco, indent=2))

    return ds


@pytest.fixture
def valid_vme_zip(valid_vme_dataset: Path, tmp_path: Path) -> Path:
    """Create a ZIP file containing a valid VME dataset."""
    zip_path = tmp_path / "VME_CDSI_datasets.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        for root, _dirs, files in os.walk(valid_vme_dataset):
            for f in files:
                full_path = Path(root) / f
                arcname = full_path.relative_to(valid_vme_dataset)
                zf.write(full_path, arcname)
    return zip_path


# ---------------------------------------------------------------------------
# COCO JSON Validation Tests
# ---------------------------------------------------------------------------


class TestValidateCocoJson:
    """Tests for validate_coco_json()."""

    def test_valid_coco_json(self, valid_coco_json: Path) -> None:
        """GIVEN a valid COCO JSON file WHEN validated THEN all checks pass."""
        result = validate_coco_json(str(valid_coco_json))
        assert result["valid"] is True
        assert result["num_images"] == 3
        assert result["num_annotations"] == 5
        assert result["num_categories"] == 3
        assert result["category_indexing"] == "0-indexed"
        assert 0 in result["categories"]
        assert result["categories"][0] == "Car"
        assert result["categories"][1] == "Bus"
        assert result["categories"][2] == "Truck"

    def test_one_indexed_categories(self, one_indexed_coco_json: Path) -> None:
        """GIVEN 1-indexed categories WHEN validated THEN warning is issued."""
        result = validate_coco_json(str(one_indexed_coco_json))
        assert result["valid"] is True
        assert result["category_indexing"] == "1-indexed"
        assert any("1-indexed" in w for w in result["warnings"])

    def test_corrupted_json(self, tmp_path: Path) -> None:
        """GIVEN invalid JSON WHEN parsed THEN ValueError is raised."""
        bad_path = tmp_path / "bad.json"
        bad_path.write_text("{not valid json!")
        with pytest.raises(ValueError, match="Corrupted annotation file"):
            validate_coco_json(str(bad_path))

    def test_missing_categories(self, tmp_path: Path) -> None:
        """GIVEN COCO JSON with empty categories WHEN validated THEN error."""
        coco = {"images": [], "annotations": [], "categories": []}
        json_path = tmp_path / "empty_cats.json"
        json_path.write_text(json.dumps(coco))
        result = validate_coco_json(str(json_path))
        assert result["valid"] is False
        assert any("missing categories" in e for e in result["errors"])

    def test_missing_required_keys(self, tmp_path: Path) -> None:
        """GIVEN COCO JSON missing 'images' key WHEN validated THEN error."""
        coco = {"annotations": [], "categories": [{"id": 0, "name": "Car"}]}
        json_path = tmp_path / "missing_key.json"
        json_path.write_text(json.dumps(coco))
        result = validate_coco_json(str(json_path))
        assert result["valid"] is False
        assert any("Missing required keys" in e for e in result["errors"])

    def test_annotation_counts_per_category(self, valid_coco_json: Path) -> None:
        """GIVEN valid COCO JSON WHEN validated THEN per-category counts computed."""
        result = validate_coco_json(str(valid_coco_json))
        counts = result["annotations_per_category"]
        assert counts[0] == 3  # Car: annotations 0, 2, 4
        assert counts[1] == 1  # Bus: annotation 1
        assert counts[2] == 1  # Truck: annotation 3

    def test_invalid_bbox_format(self, tmp_path: Path) -> None:
        """GIVEN annotation with invalid bbox WHEN validated THEN error."""
        coco = {
            "images": [{"id": 0, "file_name": "t.jpg", "width": 512, "height": 512}],
            "annotations": [{"id": 0, "image_id": 0, "category_id": 0, "bbox": [1, 2]}],
            "categories": [{"id": 0, "name": "Car"}],
        }
        json_path = tmp_path / "bad_bbox.json"
        json_path.write_text(json.dumps(coco))
        result = validate_coco_json(str(json_path))
        assert result["valid"] is False
        assert any("Invalid bbox" in e for e in result["errors"])


# ---------------------------------------------------------------------------
# Dataset Structure Validation Tests
# ---------------------------------------------------------------------------


class TestValidateDatasetStructure:
    """Tests for validate_dataset_structure()."""

    def test_valid_dataset(self, valid_vme_dataset: Path) -> None:
        """GIVEN valid VME dataset WHEN structure validated THEN passes."""
        result = validate_dataset_structure(str(valid_vme_dataset))
        assert result["valid"] is True
        assert result["train_images"] == 10
        assert result["val_images"] == 2
        assert result["test_images"] == 1
        assert result["total_images"] == 13

    def test_missing_image_dir(self, tmp_path: Path) -> None:
        """GIVEN dataset missing satellite_images/ WHEN validated THEN error."""
        ds = tmp_path / "bad_ds"
        ann_dir = ds / "annotations_HBB"
        ann_dir.mkdir(parents=True)
        (ann_dir / "train.json").write_text("{}")
        result = validate_dataset_structure(str(ds))
        assert result["valid"] is False
        assert any("Missing satellite_images/" in e for e in result["errors"])

    def test_missing_annotation_file(self, tmp_path: Path) -> None:
        """GIVEN dataset missing annotation JSON WHEN validated THEN error."""
        ds = tmp_path / "no_ann"
        (ds / "satellite_images").mkdir(parents=True)
        ann_dir = ds / "annotations_HBB"
        ann_dir.mkdir(parents=True)
        (ann_dir / "train.json").write_text("{}")
        # Missing val.json and test.json
        result = validate_dataset_structure(str(ds))
        assert result["valid"] is False
        assert any("Missing annotations_HBB/val.json" in e for e in result["errors"])
        assert any("Missing annotations_HBB/test.json" in e for e in result["errors"])

    def test_empty_image_dir(self, tmp_path: Path) -> None:
        """GIVEN dataset with no images WHEN validated THEN error."""
        ds = tmp_path / "empty_images"
        (ds / "satellite_images").mkdir(parents=True)
        ann_dir = ds / "annotations_HBB"
        ann_dir.mkdir(parents=True)
        for ann_file in EXPECTED_ANNOTATION_FILES:
            (ann_dir / ann_file).write_text("{}")
        result = validate_dataset_structure(str(ds))
        assert result["valid"] is False
        assert result.get("total_images", 0) == 0


# ---------------------------------------------------------------------------
# ZIP Extraction Tests
# ---------------------------------------------------------------------------


class TestExtractZip:
    """Tests for extract_zip()."""

    def test_successful_extraction(self, valid_vme_zip: Path, tmp_path: Path) -> None:
        """GIVEN valid ZIP WHEN extracted THEN contents are available."""
        output_dir = tmp_path / "extracted"
        extract_zip(str(valid_vme_zip), str(output_dir))
        assert (output_dir / "satellite_images").exists()
        assert (output_dir / "annotations_HBB").exists()
        assert (output_dir / "annotations_HBB" / "train.json").exists()
        assert (output_dir / "annotations_HBB" / "val.json").exists()
        assert (output_dir / "annotations_HBB" / "test.json").exists()

    def test_skip_existing(self, valid_vme_dataset: Path) -> None:
        """GIVEN dataset already extracted WHEN skip_existing=True THEN skips.

        Note: skip_existing checks for train/ and valid/ directories for backward compatibility.
        Since our fixture uses VME native structure, we create those directories to test the skip logic.
        """
        # Create train/ and valid/ so skip_existing detects an existing extraction
        (valid_vme_dataset / "train").mkdir(exist_ok=True)
        (valid_vme_dataset / "valid").mkdir(exist_ok=True)
        result = extract_zip("nonexistent.zip", str(valid_vme_dataset), skip_existing=True)
        assert result == str(valid_vme_dataset)

    def test_corrupted_zip(self, tmp_path: Path) -> None:
        """GIVEN corrupted ZIP WHEN extracted THEN RuntimeError."""
        bad_zip = tmp_path / "bad.zip"
        bad_zip.write_bytes(b"this is not a zip file")
        with pytest.raises(RuntimeError, match="Corrupted ZIP"):
            extract_zip(str(bad_zip), str(tmp_path / "output"))


# ---------------------------------------------------------------------------
# Checksum Tests
# ---------------------------------------------------------------------------


class TestComputeChecksum:
    """Tests for compute_checksum()."""

    def test_md5_checksum(self, tmp_path: Path) -> None:
        """GIVEN a file WHEN MD5 is computed THEN returns correct hash."""
        test_file = tmp_path / "test.bin"
        test_file.write_bytes(b"hello world")
        result = compute_checksum(str(test_file), "md5")
        # Known MD5 of "hello world"
        assert result == "5eb63bbbe01eeed093cb22bb8f5acdc3"

    def test_sha256_checksum(self, tmp_path: Path) -> None:
        """GIVEN a file WHEN SHA256 is computed THEN returns correct hash."""
        test_file = tmp_path / "test.bin"
        test_file.write_bytes(b"hello world")
        result = compute_checksum(str(test_file), "sha256")
        assert len(result) == 64  # SHA256 hex digest length


# ---------------------------------------------------------------------------
# Category Remapping Tests
# ---------------------------------------------------------------------------


class TestRemapCategories:
    """Tests for remap_categories_to_zero()."""

    def test_remap_one_indexed(self, one_indexed_coco_json: Path) -> None:
        """GIVEN 1-indexed categories WHEN remapped THEN becomes 0-indexed."""
        remap_categories_to_zero(str(one_indexed_coco_json))

        with open(one_indexed_coco_json) as f:
            data = json.load(f)

        cat_ids = [c["id"] for c in data["categories"]]
        assert cat_ids == [0, 1, 2]

        ann_cat_ids = [a["category_id"] for a in data["annotations"]]
        assert all(cid >= 0 for cid in ann_cat_ids)
        assert 0 in ann_cat_ids

    def test_already_zero_indexed(self, valid_coco_json: Path) -> None:
        """GIVEN 0-indexed categories WHEN remapped THEN no change."""
        original = valid_coco_json.read_text()
        remap_categories_to_zero(str(valid_coco_json))
        # File should be unchanged (function returns early)
        assert valid_coco_json.read_text() == original


# ---------------------------------------------------------------------------
# Full Validation Tests
# ---------------------------------------------------------------------------


class TestValidateFull:
    """Tests for validate_full()."""

    def test_valid_dataset_passes(self, valid_vme_dataset: Path) -> None:
        """GIVEN complete dataset WHEN fully validated THEN passes."""
        result = validate_full(str(valid_vme_dataset))
        assert result["valid"] is True

    def test_reports_structure_and_coco(self, valid_vme_dataset: Path) -> None:
        """GIVEN valid dataset WHEN validated THEN both structure and COCO checked."""
        result = validate_full(str(valid_vme_dataset))
        assert "structure" in result
        assert "coco" in result
        assert "train" in result["coco"]
        assert "val" in result["coco"]
        assert "test" in result["coco"]


# ---------------------------------------------------------------------------
# CLI Tests
# ---------------------------------------------------------------------------


class TestParseArgs:
    """Tests for CLI argument parsing."""

    def test_default_args(self) -> None:
        """GIVEN no arguments WHEN parsed THEN defaults are set."""
        args = parse_args([])
        assert args.output_dir == "vme_dataset"
        assert args.skip_existing is False
        assert args.verify_only is False
        assert args.accept_license is False

    def test_verify_only_flag(self) -> None:
        """GIVEN --verify_only WHEN parsed THEN flag is True."""
        args = parse_args(["--verify_only"])
        assert args.verify_only is True

    def test_custom_output_dir(self) -> None:
        """GIVEN --output_dir custom WHEN parsed THEN set correctly."""
        args = parse_args(["--output_dir", "/tmp/vme"])
        assert args.output_dir == "/tmp/vme"

    def test_accept_license(self) -> None:
        """GIVEN --accept_license WHEN parsed THEN flag is True."""
        args = parse_args(["--accept_license"])
        assert args.accept_license is True


class TestMainFunction:
    """Tests for the main() entry point."""

    def test_verify_only_valid(self, valid_vme_dataset: Path) -> None:
        """GIVEN valid dataset WHEN main runs with --verify_only THEN returns 0."""
        exit_code = main(["--output_dir", str(valid_vme_dataset), "--verify_only"])
        assert exit_code == 0

    def test_verify_only_missing_dir(self, tmp_path: Path) -> None:
        """GIVEN missing dataset WHEN main runs with --verify_only THEN returns 1."""
        exit_code = main(["--output_dir", str(tmp_path / "nonexistent"), "--verify_only"])
        assert exit_code == 1


# ---------------------------------------------------------------------------
# Disk Space Tests
# ---------------------------------------------------------------------------


class TestCheckDiskSpace:
    """Tests for check_disk_space()."""

    def test_sufficient_space(self, tmp_path: Path) -> None:
        """GIVEN sufficient disk space WHEN checked THEN no error."""
        # 1 KB should always be available
        check_disk_space(str(tmp_path), required_bytes=1024)

    def test_insufficient_space(self, tmp_path: Path) -> None:
        """GIVEN impossibly large requirement WHEN checked THEN OSError."""
        with pytest.raises(OSError, match="Insufficient disk space"):
            # 1 exabyte should never be available
            check_disk_space(str(tmp_path), required_bytes=10**18)
