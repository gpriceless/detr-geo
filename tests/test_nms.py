"""Tests for cross-tile NMS and edge-zone filtering."""

from __future__ import annotations

import numpy as np
import pytest

from detr_geo.tiling import (
    compute_iou,
    cross_tile_nms,
    edge_zone_filter,
    offset_detections,
)

# ---------------------------------------------------------------------------
# Tests: compute_iou
# ---------------------------------------------------------------------------


class TestComputeIoU:
    """Test IoU computation."""

    def test_identical_boxes_iou_is_1(self):
        """Identical boxes -> IoU = 1.0."""
        box = np.array([10.0, 10.0, 50.0, 50.0])
        assert compute_iou(box, box) == pytest.approx(1.0)

    def test_non_overlapping_boxes_iou_is_0(self):
        """Non-overlapping boxes -> IoU = 0.0."""
        box1 = np.array([0.0, 0.0, 10.0, 10.0])
        box2 = np.array([20.0, 20.0, 30.0, 30.0])
        assert compute_iou(box1, box2) == 0.0

    def test_partial_overlap(self):
        """Known partial overlap returns correct IoU."""
        box1 = np.array([0.0, 0.0, 20.0, 20.0])  # area = 400
        box2 = np.array([10.0, 10.0, 30.0, 30.0])  # area = 400
        # Intersection: (10,10)-(20,20) = 100
        # Union: 400 + 400 - 100 = 700
        expected_iou = 100.0 / 700.0
        assert compute_iou(box1, box2) == pytest.approx(expected_iou, abs=1e-6)

    def test_box_inside_another(self):
        """Small box fully inside larger box."""
        outer = np.array([0.0, 0.0, 100.0, 100.0])  # area = 10000
        inner = np.array([25.0, 25.0, 75.0, 75.0])  # area = 2500
        # Intersection = 2500, union = 10000
        expected_iou = 2500.0 / 10000.0
        assert compute_iou(outer, inner) == pytest.approx(expected_iou)

    def test_touching_edges_no_area(self):
        """Boxes sharing an edge have zero intersection area."""
        box1 = np.array([0.0, 0.0, 10.0, 10.0])
        box2 = np.array([10.0, 0.0, 20.0, 10.0])
        assert compute_iou(box1, box2) == 0.0

    def test_zero_area_box(self):
        """Zero-area box returns IoU = 0.0."""
        box1 = np.array([5.0, 5.0, 5.0, 5.0])  # zero area
        box2 = np.array([0.0, 0.0, 10.0, 10.0])
        assert compute_iou(box1, box2) == 0.0


# ---------------------------------------------------------------------------
# Tests: cross_tile_nms
# ---------------------------------------------------------------------------


class TestCrossTileNMS:
    """Test class-aware NMS."""

    def test_same_class_duplicate_suppression(self):
        """Same-class high-IoU pair: lower score suppressed."""
        boxes = np.array(
            [
                [0.0, 0.0, 10.0, 10.0],
                [1.0, 1.0, 11.0, 11.0],  # ~68% overlap with first
            ],
            dtype=np.float32,
        )
        scores = np.array([0.9, 0.7], dtype=np.float32)
        class_ids = np.array([1, 1], dtype=np.int32)

        keep = cross_tile_nms(boxes, scores, class_ids, iou_threshold=0.5)
        assert keep[0]  # Higher confidence survives
        assert not keep[1]  # Lower confidence suppressed

    def test_different_class_coexistence(self):
        """Different classes at same location both survive."""
        boxes = np.array(
            [
                [0.0, 0.0, 10.0, 10.0],
                [0.0, 0.0, 10.0, 10.0],  # Identical location
            ],
            dtype=np.float32,
        )
        scores = np.array([0.9, 0.7], dtype=np.float32)
        class_ids = np.array([1, 2], dtype=np.int32)  # Different classes

        keep = cross_tile_nms(boxes, scores, class_ids, iou_threshold=0.5)
        assert keep[0]
        assert keep[1]

    def test_below_iou_threshold_both_survive(self):
        """Same class but IoU below threshold -> both survive."""
        boxes = np.array(
            [
                [0.0, 0.0, 10.0, 10.0],
                [20.0, 20.0, 30.0, 30.0],  # No overlap
            ],
            dtype=np.float32,
        )
        scores = np.array([0.9, 0.7], dtype=np.float32)
        class_ids = np.array([1, 1], dtype=np.int32)

        keep = cross_tile_nms(boxes, scores, class_ids, iou_threshold=0.5)
        assert keep.all()

    def test_confidence_ordered_suppression(self):
        """Highest confidence always survives in a cluster."""
        boxes = np.array(
            [
                [0.0, 0.0, 10.0, 10.0],
                [0.5, 0.5, 10.5, 10.5],
                [1.0, 1.0, 11.0, 11.0],
            ],
            dtype=np.float32,
        )
        scores = np.array([0.5, 0.9, 0.7], dtype=np.float32)
        class_ids = np.array([1, 1, 1], dtype=np.int32)

        keep = cross_tile_nms(boxes, scores, class_ids, iou_threshold=0.5)
        # The detection with score 0.9 (index 1) should always survive
        assert keep[1]

    def test_custom_iou_threshold(self):
        """Custom NMS threshold changes suppression behavior."""
        boxes = np.array(
            [
                [0.0, 0.0, 10.0, 10.0],
                [6.0, 6.0, 16.0, 16.0],  # ~16/284 = 0.056 IoU
            ],
            dtype=np.float32,
        )
        scores = np.array([0.9, 0.7], dtype=np.float32)
        class_ids = np.array([1, 1], dtype=np.int32)

        # With high threshold, both survive
        keep_high = cross_tile_nms(boxes, scores, class_ids, iou_threshold=0.5)
        assert keep_high.all()

        # With very low threshold, lower score suppressed
        keep_low = cross_tile_nms(boxes, scores, class_ids, iou_threshold=0.01)
        assert keep_low[0]
        assert not keep_low[1]

    def test_empty_input(self):
        """Zero detections returns empty mask."""
        boxes = np.array([], dtype=np.float32).reshape(0, 4)
        scores = np.array([], dtype=np.float32)
        class_ids = np.array([], dtype=np.int32)

        keep = cross_tile_nms(boxes, scores, class_ids)
        assert len(keep) == 0

    def test_single_detection_survives(self):
        """Single detection always survives."""
        boxes = np.array([[10.0, 10.0, 50.0, 50.0]], dtype=np.float32)
        scores = np.array([0.8], dtype=np.float32)
        class_ids = np.array([0], dtype=np.int32)

        keep = cross_tile_nms(boxes, scores, class_ids)
        assert len(keep) == 1
        assert keep[0]


# ---------------------------------------------------------------------------
# Tests: edge_zone_filter
# ---------------------------------------------------------------------------


class TestEdgeZoneFilter:
    """Test edge-zone filtering."""

    def test_center_detection_survives(self):
        """Detection centered in tile always survives."""
        # Detection at center of a 100x100 tile starting at (0, 0)
        boxes = np.array([[40.0, 40.0, 60.0, 60.0]], dtype=np.float32)
        scores = np.array([0.9], dtype=np.float32)
        tile_windows = [(0, 0, 100, 100)]
        tile_indices = np.array([0])

        keep = edge_zone_filter(boxes, scores, tile_windows, tile_indices)
        assert keep[0]

    def test_edge_detection_suppressed_by_center_duplicate(self):
        """Edge detection suppressed when center duplicate exists."""
        # Detection 1: at the edge of tile 0 (center x=5, within 10% buffer of 100-wide tile)
        # Detection 2: centered in tile 1
        boxes = np.array(
            [
                [0.0, 40.0, 10.0, 60.0],  # Edge of tile 0 (center_x=5, buffer=10)
                [0.0, 40.0, 10.0, 60.0],  # Same box, but from tile 1 where it's centered
            ],
            dtype=np.float32,
        )
        scores = np.array([0.8, 0.85], dtype=np.float32)
        tile_windows = [
            (0, 0, 100, 100),  # Tile 0: detection is at left edge
            (-50, 0, 100, 100),  # Tile 1: detection is centered (local center at 55)
        ]
        tile_indices = np.array([0, 1])

        keep = edge_zone_filter(
            boxes,
            scores,
            tile_windows,
            tile_indices,
            edge_buffer_ratio=0.1,
            iou_threshold=0.5,
        )
        # Detection 0 is at edge, detection 1 is centered
        # Detection 0 should be suppressed by detection 1
        assert not keep[0]
        assert keep[1]

    def test_edge_detection_without_center_duplicate_survives(self):
        """Edge detection without any center duplicate survives."""
        # One detection at edge, no overlapping center detection
        boxes = np.array([[0.0, 40.0, 10.0, 60.0]], dtype=np.float32)
        scores = np.array([0.8], dtype=np.float32)
        tile_windows = [(0, 0, 100, 100)]
        tile_indices = np.array([0])

        keep = edge_zone_filter(boxes, scores, tile_windows, tile_indices)
        assert keep[0]

    def test_empty_input(self):
        """Empty input returns empty mask."""
        boxes = np.array([], dtype=np.float32).reshape(0, 4)
        scores = np.array([], dtype=np.float32)
        tile_windows = []
        tile_indices = np.array([], dtype=np.int32)

        keep = edge_zone_filter(boxes, scores, tile_windows, tile_indices)
        assert len(keep) == 0


# ---------------------------------------------------------------------------
# Tests: offset_detections
# ---------------------------------------------------------------------------


class TestOffsetDetections:
    """Test tile-local to global coordinate offset."""

    def test_applies_offset_correctly(self):
        """Boxes are shifted by the tile offset."""
        detection = {
            "bbox": [[10.0, 20.0, 30.0, 40.0]],
            "confidence": [0.9],
            "class_id": [1],
        }
        shifted = offset_detections(detection, offset_x=100, offset_y=200)
        assert shifted["bbox"][0] == [110.0, 220.0, 130.0, 240.0]
        assert shifted["confidence"] == [0.9]
        assert shifted["class_id"] == [1]

    def test_zero_offset_no_change(self):
        """Zero offset doesn't change coordinates."""
        detection = {
            "bbox": [[10.0, 20.0, 30.0, 40.0]],
            "confidence": [0.9],
            "class_id": [1],
        }
        shifted = offset_detections(detection, offset_x=0, offset_y=0)
        assert shifted["bbox"][0] == [10.0, 20.0, 30.0, 40.0]

    def test_multiple_boxes_offset(self):
        """Multiple boxes all shifted correctly."""
        detection = {
            "bbox": [
                [0.0, 0.0, 10.0, 10.0],
                [20.0, 20.0, 50.0, 50.0],
            ],
            "confidence": [0.9, 0.8],
            "class_id": [1, 2],
        }
        shifted = offset_detections(detection, offset_x=50, offset_y=100)
        assert shifted["bbox"][0] == [50.0, 100.0, 60.0, 110.0]
        assert shifted["bbox"][1] == [70.0, 120.0, 100.0, 150.0]

    def test_empty_detection(self):
        """Empty detection remains empty after offset."""
        detection = {
            "bbox": [],
            "confidence": [],
            "class_id": [],
        }
        shifted = offset_detections(detection, offset_x=100, offset_y=200)
        assert shifted["bbox"] == []
        assert shifted["confidence"] == []
        assert shifted["class_id"] == []
