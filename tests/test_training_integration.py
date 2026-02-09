"""End-to-end integration test for the full training pipeline (Proposal 006, Task 006.7).

Tests the complete flow: synthetic GeoTIFF + GeoJSON -> prepare_training_dataset()
-> verify COCO directory -> train() with mocked adapter.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import geopandas as gpd
import numpy as np
import rasterio
from pyproj import CRS
from rasterio.transform import from_bounds
from shapely.geometry import box

from detr_geo.training import (
    AUGMENTATION_PRESETS,
    detections_to_coco,
    detections_to_yolo,
    prepare_training_dataset,
    train,
)


class TestEndToEndPipeline:
    """Full pipeline integration tests."""

    def test_full_pipeline_synthetic_data(self, tmp_path: Path):
        """End-to-end: GeoTIFF + GeoJSON -> prepare_training_dataset() -> verify COCO.

        Uses a 512x512 synthetic raster and 4 annotations to exercise the
        full pipeline including tiling, CRS alignment, annotation clipping,
        spatial splitting, and COCO output.
        """
        # Create synthetic raster
        raster_path = tmp_path / "raster.tif"
        width, height = 512, 512
        west, south = 500000.0, 4000000.0
        east, north = west + width, south + height
        transform = from_bounds(west, south, east, north, width, height)

        rng = np.random.RandomState(42)
        data = rng.randint(0, 255, (3, height, width), dtype=np.uint8)

        with rasterio.open(
            raster_path,
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

        # Create synthetic annotations
        features = [
            {"geometry": box(500050.0, 4000050.0, 500150.0, 4000150.0), "class_name": "building"},
            {"geometry": box(500200.0, 4000200.0, 500350.0, 4000350.0), "class_name": "vehicle"},
            {"geometry": box(500300.0, 4000050.0, 500450.0, 4000150.0), "class_name": "building"},
            {"geometry": box(500100.0, 4000300.0, 500250.0, 4000450.0), "class_name": "field"},
        ]
        gdf = gpd.GeoDataFrame(features, crs=CRS.from_epsg(32617))
        ann_path = tmp_path / "annotations.geojson"
        gdf.to_file(ann_path, driver="GeoJSON")

        # Run full dataset preparation
        output_dir = tmp_path / "coco_dataset"
        stats = prepare_training_dataset(
            raster_path=str(raster_path),
            annotations_path=str(ann_path),
            output_dir=str(output_dir),
            tile_size=256,
            overlap_ratio=0.0,
            class_mapping={"building": 0, "vehicle": 1, "field": 2},
            min_annotation_area=25,
            max_background_per_annotated=3.0,
            split_method="block",
            split_ratios=(0.8, 0.15, 0.05),
            seed=42,
        )

        # Verify directory structure
        assert (output_dir / "train").is_dir()
        assert (output_dir / "valid").is_dir()
        assert (output_dir / "test").is_dir()

        for split in ["train", "valid", "test"]:
            assert (output_dir / split / "images").is_dir()
            assert (output_dir / split / "_annotations.coco.json").is_file()

        # Verify COCO JSON validity
        for split in ["train", "valid", "test"]:
            with open(output_dir / split / "_annotations.coco.json") as f:
                coco = json.load(f)

            assert "images" in coco
            assert "categories" in coco
            assert "annotations" in coco

            # Verify annotation fields
            for ann in coco["annotations"]:
                assert "id" in ann
                assert "image_id" in ann
                assert "category_id" in ann
                assert "bbox" in ann
                assert "area" in ann
                assert "iscrowd" in ann
                assert ann["iscrowd"] == 0
                assert len(ann["bbox"]) == 4

            # Verify globally unique annotation IDs within split
            ann_ids = [a["id"] for a in coco["annotations"]]
            assert len(ann_ids) == len(set(ann_ids))

            # Verify image files exist
            for img in coco["images"]:
                img_path = output_dir / split / "images" / img["file_name"]
                assert img_path.exists(), f"Missing image: {img_path}"

        # Verify stats
        assert stats["train_images"] > 0
        total_images = stats["train_images"] + stats["val_images"] + stats["test_images"]
        assert total_images > 0
        total_annotations = stats["train_annotations"] + stats["val_annotations"] + stats["test_annotations"]
        assert total_annotations > 0

    def test_pipeline_to_training(self, tmp_path: Path):
        """End-to-end: prepare_training_dataset() -> train() with mocked adapter.

        Verifies the COCO directory produced by preparation can be passed
        directly to the train() wrapper.
        """
        # Create synthetic data
        raster_path = tmp_path / "raster.tif"
        width, height = 256, 256
        transform = from_bounds(500000, 4000000, 500256, 4000256, width, height)

        rng = np.random.RandomState(42)
        data = rng.randint(0, 255, (3, height, width), dtype=np.uint8)

        with rasterio.open(
            raster_path,
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

        features = [
            {"geometry": box(500050.0, 4000050.0, 500150.0, 4000150.0), "class_name": "building"},
        ]
        gdf = gpd.GeoDataFrame(features, crs=CRS.from_epsg(32617))
        ann_path = tmp_path / "annotations.geojson"
        gdf.to_file(ann_path, driver="GeoJSON")

        # Step 1: Prepare dataset
        output_dir = tmp_path / "dataset"
        prepare_training_dataset(
            raster_path=str(raster_path),
            annotations_path=str(ann_path),
            output_dir=str(output_dir),
            tile_size=128,
            min_annotation_area=10,
        )

        # Step 2: Train with mock adapter
        mock_adapter = MagicMock()
        mock_adapter.train.return_value = {"mAP": 0.85, "epochs": 50}

        result = train(
            adapter=mock_adapter,
            dataset_dir=str(output_dir),
            epochs=50,
            augmentation_preset=None,  # skip augmentation for test speed
            batch_size=4,
        )

        # Verify train was called with the dataset dir
        mock_adapter.train.assert_called_once()
        call_args = mock_adapter.train.call_args
        assert call_args[0][0] == str(output_dir)
        assert call_args[1]["epochs"] == 50
        assert call_args[1]["batch_size"] == 4
        assert result == {"mAP": 0.85, "epochs": 50}

    def test_all_training_imports(self):
        """Verify all public API elements are importable."""
        from detr_geo.training import (
            detections_to_coco,
            detections_to_yolo,
            prepare_training_dataset,
            train,
        )

        # All should be callable/constructable
        assert callable(prepare_training_dataset)
        assert callable(detections_to_coco)
        assert callable(detections_to_yolo)
        assert callable(train)
        assert isinstance(AUGMENTATION_PRESETS, dict)
        assert len(AUGMENTATION_PRESETS) == 3

    def test_detection_export_round_trip(self, tmp_path: Path):
        """Verify detection export produces files usable for feedback loop."""
        # Create detections
        gdf = gpd.GeoDataFrame(
            {
                "geometry": [box(10, 10, 50, 50), box(60, 60, 120, 100)],
                "class_id": [0, 1],
                "class_name": ["building", "vehicle"],
                "confidence": [0.95, 0.80],
            }
        )

        # Export to COCO
        coco_path = tmp_path / "detections.json"
        detections_to_coco(gdf, str(coco_path), image_width=200, image_height=200)
        assert coco_path.exists()

        with open(coco_path) as f:
            coco = json.load(f)
        assert len(coco["annotations"]) == 2

        # Export to YOLO
        yolo_dir = tmp_path / "yolo"
        detections_to_yolo(gdf, str(yolo_dir), image_name="detections", image_width=200, image_height=200)
        yolo_file = yolo_dir / "detections.txt"
        assert yolo_file.exists()

        lines = yolo_file.read_text().strip().split("\n")
        assert len(lines) == 2
