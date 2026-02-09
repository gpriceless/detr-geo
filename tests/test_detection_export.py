"""Tests for detection export functions (Proposal 006, Task 006.2).

Tests cover all GIVEN/WHEN/THEN scenarios from
openspec/changes/006-training-pipeline/specs/annotation-export.md
"""

from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
from shapely.geometry import box

from detr_geo.training import detections_to_coco, detections_to_yolo


class TestDetectionsToCoco:
    """COCO export tests."""

    def test_standard_coco_export(self, synthetic_detection_gdf: gpd.GeoDataFrame, tmp_path: Path):
        """GIVEN a GeoDataFrame with geometry, class_id, class_name,
        WHEN detections_to_coco() called,
        THEN valid COCO JSON with images, categories, annotations arrays."""
        output_path = tmp_path / "test_coco.json"
        detections_to_coco(
            synthetic_detection_gdf,
            str(output_path),
            image_width=200,
            image_height=200,
        )

        assert output_path.exists()
        with open(output_path) as f:
            coco = json.load(f)

        assert "images" in coco
        assert "categories" in coco
        assert "annotations" in coco
        assert len(coco["annotations"]) == 3

    def test_coco_bbox_format(self, synthetic_detection_gdf: gpd.GeoDataFrame, tmp_path: Path):
        """THEN annotation bboxes are in COCO format [x, y, width, height]."""
        output_path = tmp_path / "test_bbox.json"
        detections_to_coco(
            synthetic_detection_gdf,
            str(output_path),
            image_width=200,
            image_height=200,
        )

        with open(output_path) as f:
            coco = json.load(f)

        for ann in coco["annotations"]:
            bbox = ann["bbox"]
            assert len(bbox) == 4
            x, y, w, h = bbox
            assert w > 0
            assert h > 0

    def test_coco_category_mapping(self, synthetic_detection_gdf: gpd.GeoDataFrame, tmp_path: Path):
        """GIVEN detections with class_ids [0, 1] and class_names,
        THEN categories array has matching id and name fields."""
        output_path = tmp_path / "test_cats.json"
        detections_to_coco(
            synthetic_detection_gdf,
            str(output_path),
            image_width=200,
            image_height=200,
        )

        with open(output_path) as f:
            coco = json.load(f)

        cats = {c["id"]: c["name"] for c in coco["categories"]}
        assert 0 in cats
        assert 1 in cats
        assert cats[0] == "building"
        assert cats[1] == "vehicle"

    def test_coco_annotation_fields(self, synthetic_detection_gdf: gpd.GeoDataFrame, tmp_path: Path):
        """THEN each annotation has id, image_id, category_id, bbox, area, iscrowd."""
        output_path = tmp_path / "test_fields.json"
        detections_to_coco(
            synthetic_detection_gdf,
            str(output_path),
            image_width=200,
            image_height=200,
        )

        with open(output_path) as f:
            coco = json.load(f)

        for ann in coco["annotations"]:
            assert "id" in ann
            assert "image_id" in ann
            assert "category_id" in ann
            assert "bbox" in ann
            assert "area" in ann
            assert "iscrowd" in ann
            assert ann["iscrowd"] == 0

    def test_coco_globally_unique_annotation_ids(self, synthetic_detection_gdf: gpd.GeoDataFrame, tmp_path: Path):
        """THEN annotation IDs are globally unique."""
        output_path = tmp_path / "test_unique_ids.json"
        detections_to_coco(
            synthetic_detection_gdf,
            str(output_path),
            image_width=200,
            image_height=200,
        )

        with open(output_path) as f:
            coco = json.load(f)

        ids = [ann["id"] for ann in coco["annotations"]]
        assert len(ids) == len(set(ids))

    def test_empty_gdf_produces_valid_coco(self, empty_detection_gdf: gpd.GeoDataFrame, tmp_path: Path):
        """GIVEN empty GeoDataFrame, THEN valid COCO JSON with empty annotations."""
        output_path = tmp_path / "test_empty.json"
        detections_to_coco(
            empty_detection_gdf,
            str(output_path),
            image_width=200,
            image_height=200,
        )

        with open(output_path) as f:
            coco = json.load(f)

        assert coco["annotations"] == []
        assert len(coco["images"]) == 1  # image entry still present

    def test_coco_with_image_path(self, synthetic_detection_gdf: gpd.GeoDataFrame, tmp_path: Path):
        """GIVEN image_path, THEN file_name in images array matches."""
        output_path = tmp_path / "test_imgpath.json"
        detections_to_coco(
            synthetic_detection_gdf,
            str(output_path),
            image_path="/path/to/my_image.tif",
            image_width=200,
            image_height=200,
        )

        with open(output_path) as f:
            coco = json.load(f)

        assert coco["images"][0]["file_name"] == "my_image.tif"

    def test_coco_export_creates_parent_dirs(self, synthetic_detection_gdf: gpd.GeoDataFrame, tmp_path: Path):
        """GIVEN output_path with non-existent parent dirs,
        THEN dirs are created."""
        output_path = tmp_path / "nested" / "dir" / "coco.json"
        detections_to_coco(
            synthetic_detection_gdf,
            str(output_path),
            image_width=200,
            image_height=200,
        )
        assert output_path.exists()


class TestDetectionsToYolo:
    """YOLO export tests."""

    def test_standard_yolo_export(self, synthetic_detection_gdf: gpd.GeoDataFrame, tmp_path: Path):
        """GIVEN a GeoDataFrame with geometry and class_id,
        WHEN detections_to_yolo() called,
        THEN .txt file with correct format."""
        detections_to_yolo(
            synthetic_detection_gdf,
            str(tmp_path),
            image_name="test_image",
            image_width=200,
            image_height=200,
        )

        txt_path = tmp_path / "test_image.txt"
        assert txt_path.exists()

        lines = txt_path.read_text().strip().split("\n")
        assert len(lines) == 3

    def test_yolo_format_values(self, synthetic_detection_gdf: gpd.GeoDataFrame, tmp_path: Path):
        """THEN each line is: class_id center_x center_y width height,
        AND values are normalized to [0, 1]."""
        detections_to_yolo(
            synthetic_detection_gdf,
            str(tmp_path),
            image_name="test_norm",
            image_width=200,
            image_height=200,
        )

        txt_path = tmp_path / "test_norm.txt"
        lines = txt_path.read_text().strip().split("\n")

        for line in lines:
            parts = line.split()
            assert len(parts) == 5
            int(parts[0])
            cx, cy, w, h = [float(p) for p in parts[1:]]
            # All values should be in [0, 1]
            assert 0.0 <= cx <= 1.0
            assert 0.0 <= cy <= 1.0
            assert 0.0 <= w <= 1.0
            assert 0.0 <= h <= 1.0

    def test_yolo_class_ids(self, synthetic_detection_gdf: gpd.GeoDataFrame, tmp_path: Path):
        """THEN class_ids match the GeoDataFrame."""
        detections_to_yolo(
            synthetic_detection_gdf,
            str(tmp_path),
            image_name="test_cls",
            image_width=200,
            image_height=200,
        )

        txt_path = tmp_path / "test_cls.txt"
        lines = txt_path.read_text().strip().split("\n")
        class_ids = [int(line.split()[0]) for line in lines]
        assert class_ids == [0, 1, 0]

    def test_empty_gdf_produces_empty_file(self, empty_detection_gdf: gpd.GeoDataFrame, tmp_path: Path):
        """GIVEN empty GeoDataFrame, THEN valid but empty .txt file."""
        detections_to_yolo(
            empty_detection_gdf,
            str(tmp_path),
            image_name="test_empty",
            image_width=200,
            image_height=200,
        )

        txt_path = tmp_path / "test_empty.txt"
        assert txt_path.exists()
        assert txt_path.read_text() == ""

    def test_yolo_center_calculation(self, tmp_path: Path):
        """Verify center_x, center_y calculation for a known box."""
        gdf = gpd.GeoDataFrame(
            {
                "geometry": [box(0, 0, 100, 100)],
                "class_id": [0],
            }
        )
        detections_to_yolo(
            gdf,
            str(tmp_path),
            image_name="test_center",
            image_width=200,
            image_height=200,
        )

        txt_path = tmp_path / "test_center.txt"
        parts = txt_path.read_text().strip().split()
        cx, cy, w, h = [float(p) for p in parts[1:]]
        # box(0,0,100,100) in 200x200: center=(50,50), size=(100,100)
        assert abs(cx - 0.25) < 0.001
        assert abs(cy - 0.25) < 0.001
        assert abs(w - 0.5) < 0.001
        assert abs(h - 0.5) < 0.001

    def test_yolo_creates_output_dir(self, synthetic_detection_gdf, tmp_path: Path):
        """GIVEN non-existent output directory, THEN it is created."""
        out_dir = tmp_path / "nested" / "yolo"
        detections_to_yolo(
            synthetic_detection_gdf,
            str(out_dir),
            image_name="test",
            image_width=200,
            image_height=200,
        )
        assert (out_dir / "test.txt").exists()
