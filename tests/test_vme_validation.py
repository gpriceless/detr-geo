"""Tests for scripts/validate_vme.py -- VME evaluation and validation.

All tests use synthetic data -- no actual model inference.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from validate_vme import (
    SUCCESS_CRITERIA,
    VME_CLASSES,
    build_confusion_matrix,
    check_success_criteria,
    compute_ap,
    compute_iou,
    compute_map,
    generate_report,
    match_predictions_to_gt,
    parse_args,
)

# ---------------------------------------------------------------------------
# IoU Computation Tests
# ---------------------------------------------------------------------------


class TestComputeIoU:
    """Tests for compute_iou()."""

    def test_perfect_overlap(self) -> None:
        """GIVEN identical boxes WHEN IoU computed THEN returns 1.0."""
        box = [10, 10, 50, 50]
        assert compute_iou(box, box) == pytest.approx(1.0)

    def test_no_overlap(self) -> None:
        """GIVEN non-overlapping boxes WHEN IoU computed THEN returns 0.0."""
        box1 = [0, 0, 10, 10]
        box2 = [20, 20, 10, 10]
        assert compute_iou(box1, box2) == 0.0

    def test_partial_overlap(self) -> None:
        """GIVEN partially overlapping boxes WHEN IoU computed THEN correct."""
        # box1: [0,0] to [10,10], area=100
        # box2: [5,5] to [15,15], area=100
        # intersection: [5,5] to [10,10], area=25
        # union: 100+100-25=175
        # IoU = 25/175 = 0.1429
        box1 = [0, 0, 10, 10]
        box2 = [5, 5, 10, 10]
        iou = compute_iou(box1, box2)
        assert iou == pytest.approx(25.0 / 175.0, abs=1e-4)

    def test_box_inside_other(self) -> None:
        """GIVEN box inside another WHEN IoU computed THEN correct."""
        # outer: [0,0] to [100,100], area=10000
        # inner: [25,25] to [75,75], area=2500
        # intersection = 2500
        # union = 10000+2500-2500=10000
        # IoU = 2500/10000 = 0.25
        outer = [0, 0, 100, 100]
        inner = [25, 25, 50, 50]
        iou = compute_iou(outer, inner)
        assert iou == pytest.approx(0.25, abs=1e-4)

    def test_zero_area_box(self) -> None:
        """GIVEN zero-area box WHEN IoU computed THEN returns 0."""
        box1 = [10, 10, 0, 0]
        box2 = [10, 10, 50, 50]
        assert compute_iou(box1, box2) == 0.0

    def test_adjacent_boxes(self) -> None:
        """GIVEN touching but non-overlapping boxes WHEN IoU computed THEN 0."""
        box1 = [0, 0, 10, 10]
        box2 = [10, 0, 10, 10]  # starts exactly where box1 ends
        assert compute_iou(box1, box2) == 0.0


# ---------------------------------------------------------------------------
# Prediction Matching Tests
# ---------------------------------------------------------------------------


class TestMatchPredictionsToGT:
    """Tests for match_predictions_to_gt()."""

    def test_perfect_matches(self) -> None:
        """GIVEN predictions matching GT WHEN matched THEN all TP."""
        preds = [
            {"bbox": [10, 10, 30, 20], "confidence": 0.9, "category_id": 0},
            {"bbox": [50, 50, 40, 30], "confidence": 0.8, "category_id": 1},
        ]
        gt = [
            {"bbox": [10, 10, 30, 20], "category_id": 0},
            {"bbox": [50, 50, 40, 30], "category_id": 1},
        ]
        result = match_predictions_to_gt(preds, gt, iou_threshold=0.5)
        assert result["tp"] == 2
        assert result["fp"] == 0
        assert result["fn"] == 0
        assert result["precision"] == pytest.approx(1.0)
        assert result["recall"] == pytest.approx(1.0)

    def test_no_matches(self) -> None:
        """GIVEN non-overlapping predictions WHEN matched THEN all FP."""
        preds = [
            {"bbox": [200, 200, 30, 20], "confidence": 0.9, "category_id": 0},
        ]
        gt = [
            {"bbox": [10, 10, 30, 20], "category_id": 0},
        ]
        result = match_predictions_to_gt(preds, gt, iou_threshold=0.5)
        assert result["tp"] == 0
        assert result["fp"] == 1
        assert result["fn"] == 1

    def test_class_mismatch(self) -> None:
        """GIVEN overlapping boxes with different classes WHEN matched THEN no match."""
        preds = [
            {"bbox": [10, 10, 30, 20], "confidence": 0.9, "category_id": 0},
        ]
        gt = [
            {"bbox": [10, 10, 30, 20], "category_id": 1},  # Different class
        ]
        result = match_predictions_to_gt(preds, gt, iou_threshold=0.5)
        assert result["tp"] == 0
        assert result["fp"] == 1
        assert result["fn"] == 1

    def test_empty_predictions(self) -> None:
        """GIVEN no predictions WHEN matched THEN all FN."""
        gt = [
            {"bbox": [10, 10, 30, 20], "category_id": 0},
        ]
        result = match_predictions_to_gt([], gt, iou_threshold=0.5)
        assert result["tp"] == 0
        assert result["fp"] == 0
        assert result["fn"] == 1

    def test_empty_ground_truth(self) -> None:
        """GIVEN no GT WHEN matched THEN all FP."""
        preds = [
            {"bbox": [10, 10, 30, 20], "confidence": 0.9, "category_id": 0},
        ]
        result = match_predictions_to_gt(preds, [], iou_threshold=0.5)
        assert result["tp"] == 0
        assert result["fp"] == 1
        assert result["fn"] == 0

    def test_confidence_ordering(self) -> None:
        """GIVEN multiple predictions WHEN matched THEN highest confidence first."""
        # Two preds for one GT -- highest confidence should be matched
        preds = [
            {"bbox": [10, 10, 30, 20], "confidence": 0.5, "category_id": 0},
            {"bbox": [12, 12, 30, 20], "confidence": 0.9, "category_id": 0},
        ]
        gt = [
            {"bbox": [10, 10, 30, 20], "category_id": 0},
        ]
        result = match_predictions_to_gt(preds, gt, iou_threshold=0.5)
        assert result["tp"] == 1
        assert result["fp"] == 1  # Lower confidence pred unmatched


# ---------------------------------------------------------------------------
# Average Precision Tests
# ---------------------------------------------------------------------------


class TestComputeAP:
    """Tests for compute_ap()."""

    def test_perfect_detection(self) -> None:
        """GIVEN perfect predictions WHEN AP computed THEN returns 1.0."""
        preds = [
            {"bbox": [10, 10, 30, 20], "confidence": 0.9, "category_id": 0},
        ]
        gt = [
            {"bbox": [10, 10, 30, 20], "category_id": 0},
        ]
        ap = compute_ap(preds, gt, iou_threshold=0.5)
        assert ap == pytest.approx(1.0)

    def test_no_detections(self) -> None:
        """GIVEN no predictions with GT present WHEN AP computed THEN returns 0."""
        gt = [
            {"bbox": [10, 10, 30, 20], "category_id": 0},
        ]
        ap = compute_ap([], gt, iou_threshold=0.5)
        assert ap == 0.0

    def test_no_gt_no_preds(self) -> None:
        """GIVEN no GT and no predictions WHEN AP computed THEN returns 1.0."""
        ap = compute_ap([], [], iou_threshold=0.5)
        assert ap == 1.0

    def test_all_false_positives(self) -> None:
        """GIVEN predictions with no matching GT WHEN AP computed THEN 0."""
        preds = [
            {"bbox": [200, 200, 30, 20], "confidence": 0.9, "category_id": 0},
        ]
        gt = [
            {"bbox": [10, 10, 30, 20], "category_id": 0},
        ]
        ap = compute_ap(preds, gt, iou_threshold=0.5)
        assert ap == 0.0


# ---------------------------------------------------------------------------
# mAP Computation Tests
# ---------------------------------------------------------------------------


class TestComputeMAP:
    """Tests for compute_map()."""

    def test_perfect_map(self) -> None:
        """GIVEN perfect predictions WHEN mAP computed THEN returns 1.0."""
        preds = [
            {"bbox": [10, 10, 30, 20], "confidence": 0.9, "category_id": 0},
            {"bbox": [50, 50, 40, 30], "confidence": 0.8, "category_id": 1},
        ]
        gt = [
            {"bbox": [10, 10, 30, 20], "category_id": 0},
            {"bbox": [50, 50, 40, 30], "category_id": 1},
        ]
        result = compute_map(preds, gt, iou_threshold=0.5, class_ids=[0, 1])
        assert result["mAP"] == pytest.approx(1.0)

    def test_single_class_map(self) -> None:
        """GIVEN single-class predictions WHEN mAP computed THEN correct."""
        preds = [
            {"bbox": [10, 10, 30, 20], "confidence": 0.9, "category_id": 0},
        ]
        gt = [
            {"bbox": [10, 10, 30, 20], "category_id": 0},
        ]
        result = compute_map(preds, gt, iou_threshold=0.5, class_ids=[0])
        assert result["mAP"] == pytest.approx(1.0)
        assert "AP_Car" in result

    def test_empty_predictions_map(self) -> None:
        """GIVEN no predictions WHEN mAP computed THEN returns 0."""
        gt = [
            {"bbox": [10, 10, 30, 20], "category_id": 0},
        ]
        result = compute_map([], gt, iou_threshold=0.5, class_ids=[0])
        assert result["mAP"] == 0.0


# ---------------------------------------------------------------------------
# Confusion Matrix Tests
# ---------------------------------------------------------------------------


class TestBuildConfusionMatrix:
    """Tests for build_confusion_matrix()."""

    def test_perfect_classification(self) -> None:
        """GIVEN correct predictions WHEN confusion matrix built THEN diagonal."""
        preds = [
            {"bbox": [10, 10, 30, 20], "confidence": 0.9, "category_id": 0},
            {"bbox": [50, 50, 40, 30], "confidence": 0.8, "category_id": 1},
        ]
        gt = [
            {"bbox": [10, 10, 30, 20], "category_id": 0},
            {"bbox": [50, 50, 40, 30], "category_id": 1},
        ]
        result = build_confusion_matrix(preds, gt, class_ids=[0, 1])
        assert result["per_class"]["Car"]["tp"] == 1
        assert result["per_class"]["Bus"]["tp"] == 1

    def test_confusion_between_classes(self) -> None:
        """GIVEN misclassification WHEN confusion matrix built THEN off-diagonal."""
        # Prediction says Bus, but GT is Car
        preds = [
            {"bbox": [10, 10, 30, 20], "confidence": 0.9, "category_id": 1},
        ]
        gt = [
            {"bbox": [10, 10, 30, 20], "category_id": 0},
        ]
        result = build_confusion_matrix(preds, gt, class_ids=[0, 1])
        # Car was missed (FN), Bus has an FP
        assert result["per_class"]["Car"]["fn"] == 1
        assert result["per_class"]["Bus"]["fp"] == 1

    def test_empty_confusion_matrix(self) -> None:
        """GIVEN no preds or GT WHEN confusion matrix built THEN zeros."""
        result = build_confusion_matrix([], [], class_ids=[0, 1, 2])
        for class_name in VME_CLASSES.values():
            assert result["per_class"][class_name]["tp"] == 0
            assert result["per_class"][class_name]["fp"] == 0
            assert result["per_class"][class_name]["fn"] == 0


# ---------------------------------------------------------------------------
# Success Criteria Tests
# ---------------------------------------------------------------------------


class TestCheckSuccessCriteria:
    """Tests for check_success_criteria()."""

    def test_all_criteria_passed(self) -> None:
        """GIVEN high mAP and fast training WHEN checked THEN all pass."""
        metrics = {"map_0.5": {"mAP": 0.67}}
        result = check_success_criteria(metrics, training_hours=3.5)
        assert result["mAP@0.5 > 0.60"]["passed"] is True
        assert result["Training time < 4 hours"]["passed"] is True

    def test_map_criterion_failed(self) -> None:
        """GIVEN low mAP WHEN checked THEN mAP criterion fails."""
        metrics = {"map_0.5": {"mAP": 0.45}}
        result = check_success_criteria(metrics)
        assert result["mAP@0.5 > 0.60"]["passed"] is False
        assert result["mAP@0.5 > 0.60"]["value"] == 0.45

    def test_training_time_failed(self) -> None:
        """GIVEN slow training WHEN checked THEN time criterion fails."""
        metrics = {"map_0.5": {"mAP": 0.70}}
        result = check_success_criteria(metrics, training_hours=5.5)
        assert result["Training time < 4 hours"]["passed"] is False

    def test_no_metrics_empty_result(self) -> None:
        """GIVEN no metrics WHEN checked THEN empty result."""
        result = check_success_criteria({})
        assert len(result) == 0


# ---------------------------------------------------------------------------
# Report Generation Tests
# ---------------------------------------------------------------------------


class TestGenerateReport:
    """Tests for generate_report()."""

    def test_generates_markdown_file(self, tmp_path: Path) -> None:
        """GIVEN metrics WHEN report generated THEN Markdown file created."""
        metrics = {
            "map_0.5": {"mAP": 0.67, "AP_Car": 0.71, "AP_Bus": 0.58, "AP_Truck": 0.63},
            "map_0.75": {"mAP": 0.45},
            "confusion": {
                "per_class": {
                    "Car": {"precision": 0.85, "recall": 0.90, "f1": 0.87, "tp": 90, "fp": 16, "fn": 10},
                    "Bus": {"precision": 0.75, "recall": 0.80, "f1": 0.77, "tp": 40, "fp": 13, "fn": 10},
                    "Truck": {"precision": 0.78, "recall": 0.82, "f1": 0.80, "tp": 50, "fp": 14, "fn": 11},
                },
            },
            "success_criteria": {
                "mAP@0.5 > 0.60": {"passed": True, "value": 0.67, "threshold": 0.60},
            },
        }
        config = {"checkpoint": "best.pth", "model": "Medium"}
        report_path = generate_report(metrics, config, str(tmp_path))

        assert Path(report_path).exists()
        content = Path(report_path).read_text()
        assert "VME Fine-Tuning Evaluation Report" in content
        assert "0.67" in content
        assert "Car" in content
        assert "PASS" in content

    def test_empty_metrics_report(self, tmp_path: Path) -> None:
        """GIVEN empty metrics WHEN report generated THEN file still created."""
        report_path = generate_report({}, {"checkpoint": "none"}, str(tmp_path))
        assert Path(report_path).exists()


# ---------------------------------------------------------------------------
# CLI Tests
# ---------------------------------------------------------------------------


class TestParseArgs:
    """Tests for parse_args()."""

    def test_required_checkpoint(self) -> None:
        """GIVEN --checkpoint WHEN parsed THEN stored correctly."""
        args = parse_args(["--checkpoint", "best.pth"])
        assert args.checkpoint == "best.pth"

    def test_default_values(self) -> None:
        """GIVEN minimal args WHEN parsed THEN defaults set."""
        args = parse_args(["--checkpoint", "best.pth"])
        assert args.dataset_dir == "vme_dataset"
        assert args.output_dir == "evaluation_results"
        assert args.model == "medium"
        assert args.confidence == 0.3

    def test_compare_baseline_flag(self) -> None:
        """GIVEN --compare_baseline WHEN parsed THEN flag is True."""
        args = parse_args(["--checkpoint", "best.pth", "--compare_baseline"])
        assert args.compare_baseline is True

    def test_real_world_test_path(self) -> None:
        """GIVEN --real_world_test WHEN parsed THEN path stored."""
        args = parse_args(["--checkpoint", "best.pth", "--real_world_test", "images/"])
        assert args.real_world_test == "images/"


# ---------------------------------------------------------------------------
# Constants Tests
# ---------------------------------------------------------------------------


class TestConstants:
    """Tests for module constants."""

    def test_vme_classes(self) -> None:
        """VME_CLASSES should have 3 vehicle classes."""
        assert len(VME_CLASSES) == 3
        assert VME_CLASSES[0] == "Car"
        assert VME_CLASSES[1] == "Bus"
        assert VME_CLASSES[2] == "Truck"

    def test_success_criteria(self) -> None:
        """SUCCESS_CRITERIA should define all required thresholds."""
        assert "mAP_0.5" in SUCCESS_CRITERIA
        assert "parking_lot_recall" in SUCCESS_CRITERIA
        assert "mean_confidence" in SUCCESS_CRITERIA
        assert "training_hours" in SUCCESS_CRITERIA
        assert SUCCESS_CRITERIA["mAP_0.5"] == 0.60
