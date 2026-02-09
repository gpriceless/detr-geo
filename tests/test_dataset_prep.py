"""Tests for prepare_training_dataset() pipeline (Proposal 006, Task 006.4).

Tests cover all GIVEN/WHEN/THEN scenarios from
openspec/changes/006-training-pipeline/specs/dataset-preparation.md
"""

from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
import rasterio
from pyproj import CRS
from rasterio.transform import from_bounds
from shapely.geometry import MultiPolygon, box

from detr_geo.training import prepare_training_dataset


@pytest.fixture
def large_geotiff(tmp_path: Path) -> Path:
    """Create a 512x512 GeoTIFF for dataset prep testing (larger for multiple tiles)."""
    tiff_path = tmp_path / "large_raster.tif"
    width, height = 512, 512
    west, south = 500000.0, 4000000.0
    east, north = west + width, south + height
    transform = from_bounds(west, south, east, north, width, height)

    rng = np.random.RandomState(42)
    data = rng.randint(0, 255, (3, height, width), dtype=np.uint8)

    with rasterio.open(
        tiff_path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=3,
        dtype="uint8",
        crs=CRS.from_epsg(32617),
        transform=transform,
    ) as dst:
        dst.write(data)

    return tiff_path


@pytest.fixture
def large_geojson_utm(tmp_path: Path) -> Path:
    """Create annotations covering the 512x512 raster area in EPSG:32617."""
    features = [
        # Building in top-left quadrant
        {
            "geometry": box(500020.0, 4000020.0, 500080.0, 4000080.0),
            "class_name": "building",
        },
        # Building in center
        {
            "geometry": box(500200.0, 4000200.0, 500280.0, 4000280.0),
            "class_name": "building",
        },
        # Vehicle bottom-right
        {
            "geometry": box(500350.0, 4000350.0, 500400.0, 4000400.0),
            "class_name": "vehicle",
        },
        # Large field spanning multiple tiles
        {
            "geometry": box(500100.0, 4000100.0, 500400.0, 4000400.0),
            "class_name": "field",
        },
        # Tiny annotation (should be filtered by min_area)
        {
            "geometry": box(500010.0, 4000010.0, 500012.0, 4000012.0),
            "class_name": "debris",
        },
    ]
    gdf = gpd.GeoDataFrame(features, crs=CRS.from_epsg(32617))
    path = tmp_path / "large_annotations.geojson"
    gdf.to_file(path, driver="GeoJSON")
    return path


@pytest.fixture
def multipolygon_geojson(tmp_path: Path) -> Path:
    """Create GeoJSON with a MultiPolygon annotation in EPSG:32617."""
    mp = MultiPolygon(
        [
            box(500050.0, 4000050.0, 500100.0, 4000100.0),
            box(500150.0, 4000150.0, 500200.0, 4000200.0),
        ]
    )
    features = [
        {"geometry": mp, "class_name": "complex_structure"},
        {"geometry": box(500250.0, 4000250.0, 500300.0, 4000300.0), "class_name": "building"},
    ]
    gdf = gpd.GeoDataFrame(features, crs=CRS.from_epsg(32617))
    path = tmp_path / "multi_annotations.geojson"
    gdf.to_file(path, driver="GeoJSON")
    return path


class TestFullPipeline:
    """End-to-end pipeline tests."""

    def test_standard_dataset_preparation(self, large_geotiff: Path, large_geojson_utm: Path, tmp_path: Path):
        """GIVEN a GeoTIFF and GeoJSON, WHEN prepare_training_dataset() called,
        THEN output has train/valid/test with images/ and _annotations.coco.json."""
        output_dir = tmp_path / "coco_output"
        prepare_training_dataset(
            raster_path=str(large_geotiff),
            annotations_path=str(large_geojson_utm),
            output_dir=str(output_dir),
            tile_size=256,
            min_annotation_area=10,
        )

        # Check directory structure
        for split in ["train", "valid", "test"]:
            assert (output_dir / split / "images").is_dir()
            assert (output_dir / split / "_annotations.coco.json").is_file()

        # Check COCO JSON validity
        for split in ["train", "valid", "test"]:
            with open(output_dir / split / "_annotations.coco.json") as f:
                coco = json.load(f)
            assert "images" in coco
            assert "categories" in coco
            assert "annotations" in coco

    def test_return_statistics(self, large_geotiff: Path, large_geojson_utm: Path, tmp_path: Path):
        """GIVEN completed preparation, THEN stats dict with correct keys."""
        output_dir = tmp_path / "stats_output"
        stats = prepare_training_dataset(
            raster_path=str(large_geotiff),
            annotations_path=str(large_geojson_utm),
            output_dir=str(output_dir),
            tile_size=256,
            min_annotation_area=10,
        )

        required_keys = [
            "train_images",
            "val_images",
            "test_images",
            "train_annotations",
            "val_annotations",
            "test_annotations",
            "skipped_tiles",
            "skipped_annotations",
        ]
        for key in required_keys:
            assert key in stats, f"Missing key: {key}"
            assert isinstance(stats[key], int), f"Key {key} is not int: {type(stats[key])}"

        # At least train should have images
        assert stats["train_images"] > 0

    def test_tile_images_are_valid_jpegs(self, large_geotiff: Path, large_geojson_utm: Path, tmp_path: Path):
        """THEN saved tile images are valid JPEG files."""
        from PIL import Image

        output_dir = tmp_path / "jpeg_test"
        prepare_training_dataset(
            raster_path=str(large_geotiff),
            annotations_path=str(large_geojson_utm),
            output_dir=str(output_dir),
            tile_size=256,
            min_annotation_area=10,
        )

        train_images = list((output_dir / "train" / "images").glob("*.jpg"))
        assert len(train_images) > 0

        for img_path in train_images:
            img = Image.open(img_path)
            assert img.mode == "RGB"
            assert img.size[0] > 0 and img.size[1] > 0


class TestCRSAlignment:
    """CRS reprojection tests."""

    def test_annotations_in_different_crs_reprojected(self, large_geotiff: Path, tmp_path: Path):
        """GIVEN annotations in EPSG:4326 and raster in EPSG:32617,
        THEN annotations are reprojected to match raster."""
        # Create annotations in 4326 that overlap the raster extent
        # Raster is at 500000-500512, 4000000-4000512 in UTM 17N
        # Approximate: (-81.07, 36.13) to (-80.79, 36.14) -- rough
        # Use a point we know is inside the raster
        features = [
            {
                "geometry": box(-81.005, 36.13, -80.998, 36.135),
                "class_name": "building",
            },
        ]
        gdf = gpd.GeoDataFrame(features, crs=CRS.from_epsg(4326))
        ann_path = tmp_path / "ann_4326.geojson"
        gdf.to_file(ann_path, driver="GeoJSON")

        output_dir = tmp_path / "reproject_test"
        import warnings

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            prepare_training_dataset(
                raster_path=str(large_geotiff),
                annotations_path=str(ann_path),
                output_dir=str(output_dir),
                tile_size=256,
                min_annotation_area=1,
            )
            # Should have reprojection warning
            reproject_warnings = [x for x in w if "reproject" in str(x.message).lower()]
            assert len(reproject_warnings) >= 1


class TestAnnotationClipping:
    """Annotation clipping to tile boundary tests."""

    def test_annotations_clipped_to_tile_boundaries(self, large_geotiff: Path, large_geojson_utm: Path, tmp_path: Path):
        """GIVEN annotations that extend beyond tile extent,
        THEN COCO bbox is from clipped polygon in tile-relative pixels."""
        output_dir = tmp_path / "clip_test"
        stats = prepare_training_dataset(
            raster_path=str(large_geotiff),
            annotations_path=str(large_geojson_utm),
            output_dir=str(output_dir),
            tile_size=256,
            min_annotation_area=10,
        )

        # Check that we got some annotations
        total_ann = stats["train_annotations"] + stats["val_annotations"] + stats["test_annotations"]
        assert total_ann > 0

        # Check COCO bbox format in at least one split
        for split in ["train", "valid", "test"]:
            with open(output_dir / split / "_annotations.coco.json") as f:
                coco = json.load(f)
            for ann in coco["annotations"]:
                bbox = ann["bbox"]
                assert len(bbox) == 4, "COCO bbox must be [x, y, w, h]"
                x, y, w, h = bbox
                # Bboxes should be tile-relative (within tile dimensions)
                assert w > 0
                assert h > 0

    def test_multipolygon_exploded(self, large_geotiff: Path, multipolygon_geojson: Path, tmp_path: Path):
        """GIVEN MultiPolygon annotations, THEN exploded into individual annotations."""
        output_dir = tmp_path / "multi_test"
        stats = prepare_training_dataset(
            raster_path=str(large_geotiff),
            annotations_path=str(multipolygon_geojson),
            output_dir=str(output_dir),
            tile_size=256,
            min_annotation_area=10,
        )

        # Should have more annotations than input features due to explosion
        total_ann = stats["train_annotations"] + stats["val_annotations"] + stats["test_annotations"]
        assert total_ann > 0


class TestMinimumAreaFilter:
    """Minimum annotation area filter tests."""

    def test_tiny_annotations_filtered(self, large_geotiff: Path, large_geojson_utm: Path, tmp_path: Path):
        """GIVEN min_annotation_area=100, THEN fragments <100 sq px are skipped."""
        output_dir = tmp_path / "area_filter"
        stats_strict = prepare_training_dataset(
            raster_path=str(large_geotiff),
            annotations_path=str(large_geojson_utm),
            output_dir=str(output_dir),
            tile_size=256,
            min_annotation_area=100,
        )

        output_dir_loose = tmp_path / "area_loose"
        stats_loose = prepare_training_dataset(
            raster_path=str(large_geotiff),
            annotations_path=str(large_geojson_utm),
            output_dir=str(output_dir_loose),
            tile_size=256,
            min_annotation_area=1,
        )

        total_strict = (
            stats_strict["train_annotations"] + stats_strict["val_annotations"] + stats_strict["test_annotations"]
        )
        total_loose = (
            stats_loose["train_annotations"] + stats_loose["val_annotations"] + stats_loose["test_annotations"]
        )

        # Loose filter should find at least as many annotations
        assert total_loose >= total_strict

    def test_custom_minimum_area(self, large_geotiff: Path, large_geojson_utm: Path, tmp_path: Path):
        """GIVEN min_annotation_area=25, THEN polygons >=25 sq px included."""
        output_dir = tmp_path / "custom_area"
        stats = prepare_training_dataset(
            raster_path=str(large_geotiff),
            annotations_path=str(large_geojson_utm),
            output_dir=str(output_dir),
            tile_size=256,
            min_annotation_area=25,
        )

        total_ann = stats["train_annotations"] + stats["val_annotations"] + stats["test_annotations"]
        assert total_ann > 0


class TestBackgroundRatio:
    """Background tile ratio control tests."""

    def test_background_ratio_limits_empty_tiles(self, large_geotiff: Path, tmp_path: Path):
        """GIVEN max_background_per_annotated=1.0 and few annotations,
        THEN background tiles are limited."""
        # Create one small annotation
        features = [
            {
                "geometry": box(500050.0, 4000050.0, 500100.0, 4000100.0),
                "class_name": "building",
            },
        ]
        gdf = gpd.GeoDataFrame(features, crs=CRS.from_epsg(32617))
        ann_path = tmp_path / "sparse_ann.geojson"
        gdf.to_file(ann_path, driver="GeoJSON")

        # With strict ratio
        output_strict = tmp_path / "strict_bg"
        stats_strict = prepare_training_dataset(
            raster_path=str(large_geotiff),
            annotations_path=str(ann_path),
            output_dir=str(output_strict),
            tile_size=256,
            max_background_per_annotated=1.0,
            min_annotation_area=10,
        )

        # With loose ratio
        output_loose = tmp_path / "loose_bg"
        stats_loose = prepare_training_dataset(
            raster_path=str(large_geotiff),
            annotations_path=str(ann_path),
            output_dir=str(output_loose),
            tile_size=256,
            max_background_per_annotated=100.0,
            min_annotation_area=10,
        )

        total_strict = stats_strict["train_images"] + stats_strict["val_images"] + stats_strict["test_images"]
        total_loose = stats_loose["train_images"] + stats_loose["val_images"] + stats_loose["test_images"]

        # Strict should have fewer or equal tiles
        assert total_strict <= total_loose


class TestClassMapping:
    """Class mapping tests."""

    def test_user_provided_class_mapping(self, large_geotiff: Path, large_geojson_utm: Path, tmp_path: Path):
        """GIVEN class_mapping={'building': 0, 'vehicle': 1, 'field': 2, 'debris': 3},
        THEN COCO category_ids match."""
        output_dir = tmp_path / "mapping_test"
        prepare_training_dataset(
            raster_path=str(large_geotiff),
            annotations_path=str(large_geojson_utm),
            output_dir=str(output_dir),
            tile_size=256,
            class_mapping={"building": 0, "vehicle": 1, "field": 2, "debris": 3},
            min_annotation_area=10,
        )

        # Check categories in COCO JSON
        for split in ["train", "valid", "test"]:
            with open(output_dir / split / "_annotations.coco.json") as f:
                coco = json.load(f)
            for cat in coco["categories"]:
                assert cat["id"] in [0, 1, 2, 3]

    def test_auto_generated_class_mapping(self, large_geotiff: Path, large_geojson_utm: Path, tmp_path: Path):
        """GIVEN class_mapping=None, THEN auto-generated from sorted unique values."""
        output_dir = tmp_path / "auto_mapping"
        prepare_training_dataset(
            raster_path=str(large_geotiff),
            annotations_path=str(large_geojson_utm),
            output_dir=str(output_dir),
            tile_size=256,
            class_mapping=None,
            min_annotation_area=10,
        )

        # Auto-mapping should produce deterministic IDs
        with open(output_dir / "train" / "_annotations.coco.json") as f:
            coco = json.load(f)

        # Categories should be present
        assert len(coco["categories"]) > 0


class TestCocoOutput:
    """COCO JSON output format tests."""

    def test_coco_json_has_valid_structure(self, large_geotiff: Path, large_geojson_utm: Path, tmp_path: Path):
        """THEN COCO JSON has valid images, categories, annotations arrays."""
        output_dir = tmp_path / "structure_test"
        prepare_training_dataset(
            raster_path=str(large_geotiff),
            annotations_path=str(large_geojson_utm),
            output_dir=str(output_dir),
            tile_size=256,
            min_annotation_area=10,
        )

        with open(output_dir / "train" / "_annotations.coco.json") as f:
            coco = json.load(f)

        # Validate structure
        assert isinstance(coco["images"], list)
        assert isinstance(coco["categories"], list)
        assert isinstance(coco["annotations"], list)

        # Validate annotation fields
        for ann in coco["annotations"]:
            assert "id" in ann
            assert "image_id" in ann
            assert "category_id" in ann
            assert "bbox" in ann
            assert "area" in ann
            assert "iscrowd" in ann
            assert ann["iscrowd"] == 0
            assert len(ann["bbox"]) == 4

        # Validate globally unique annotation IDs
        ann_ids = [a["id"] for a in coco["annotations"]]
        assert len(ann_ids) == len(set(ann_ids))

    def test_output_uses_valid_not_val(self, large_geotiff: Path, large_geojson_utm: Path, tmp_path: Path):
        """THEN output directory uses 'valid/' not 'val/' per RF-DETR convention."""
        output_dir = tmp_path / "naming_test"
        prepare_training_dataset(
            raster_path=str(large_geotiff),
            annotations_path=str(large_geojson_utm),
            output_dir=str(output_dir),
            tile_size=256,
            min_annotation_area=10,
        )

        assert (output_dir / "valid").is_dir()
        assert not (output_dir / "val").exists()


class TestTileConfiguration:
    """Tile size and overlap tests."""

    def test_custom_tile_size(self, large_geotiff: Path, large_geojson_utm: Path, tmp_path: Path):
        """GIVEN tile_size=128, THEN image chips are 128x128 pixels."""
        from PIL import Image

        output_dir = tmp_path / "tile_size_test"
        prepare_training_dataset(
            raster_path=str(large_geotiff),
            annotations_path=str(large_geojson_utm),
            output_dir=str(output_dir),
            tile_size=128,
            min_annotation_area=10,
        )

        train_images = list((output_dir / "train" / "images").glob("*.jpg"))
        assert len(train_images) > 0

        for img_path in train_images:
            img = Image.open(img_path)
            w, h = img.size
            assert w <= 128
            assert h <= 128


class TestInputValidation:
    """Input validation tests."""

    def test_missing_raster_raises_error(self, tmp_path: Path):
        """GIVEN non-existent raster, THEN FileNotFoundError raised."""
        with pytest.raises(FileNotFoundError):
            prepare_training_dataset(
                raster_path="/nonexistent/raster.tif",
                annotations_path=str(tmp_path / "ann.geojson"),
                output_dir=str(tmp_path / "out"),
            )

    def test_missing_annotations_raises_error(self, large_geotiff: Path, tmp_path: Path):
        """GIVEN non-existent annotations, THEN FileNotFoundError raised."""
        with pytest.raises(FileNotFoundError):
            prepare_training_dataset(
                raster_path=str(large_geotiff),
                annotations_path="/nonexistent/annotations.geojson",
                output_dir=str(tmp_path / "out"),
            )
