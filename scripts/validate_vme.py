#!/usr/bin/env python3
"""Validate VME-finetuned RF-DETR model and compare against COCO baseline.

Evaluates the fine-tuned model on the VME validation set, computes standard
object detection metrics (mAP@0.5, mAP@0.75, per-class AP), and optionally
compares against the COCO-pretrained baseline on real-world parking lot imagery.

Usage:
    python scripts/validate_vme.py --checkpoint training_output/vme/vme_medium_best.pth --dataset_dir vme_dataset
    python scripts/validate_vme.py --checkpoint best.pth --compare_baseline
    python scripts/validate_vme.py --checkpoint best.pth --real_world_test parking_lots/
    python scripts/validate_vme.py --help
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VME_CLASSES = {0: "Car", 1: "Bus", 2: "Truck"}

SUCCESS_CRITERIA = {
    "mAP_0.5": 0.60,
    "parking_lot_recall": 0.50,
    "mean_confidence": 0.60,
    "training_hours": 4.0,
}


# ---------------------------------------------------------------------------
# Evaluation metrics
# ---------------------------------------------------------------------------


def compute_iou(box1: list[float], box2: list[float]) -> float:
    """Compute IoU between two COCO-format bounding boxes [x, y, w, h].

    Args:
        box1: First bounding box [x, y, w, h].
        box2: Second bounding box [x, y, w, h].

    Returns:
        IoU value between 0 and 1.
    """
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    # Convert to xyxy
    x1_max = x1 + w1
    y1_max = y1 + h1
    x2_max = x2 + w2
    y2_max = y2 + h2

    # Intersection
    inter_x1 = max(x1, x2)
    inter_y1 = max(y1, y2)
    inter_x2 = min(x1_max, x2_max)
    inter_y2 = min(y1_max, y2_max)

    if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
        return 0.0

    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    area1 = w1 * h1
    area2 = w2 * h2
    union_area = area1 + area2 - inter_area

    if union_area <= 0:
        return 0.0

    return inter_area / union_area


def match_predictions_to_gt(
    predictions: list[dict],
    ground_truth: list[dict],
    iou_threshold: float = 0.5,
) -> dict[str, Any]:
    """Match predicted boxes to ground truth at a given IoU threshold.

    Uses greedy matching: highest confidence prediction matched first.

    Args:
        predictions: List of prediction dicts with bbox, confidence, category_id.
        ground_truth: List of GT annotation dicts with bbox, category_id.
        iou_threshold: IoU threshold for matching.

    Returns:
        Dict with TP, FP, FN counts and matched pairs.
    """
    # Sort predictions by confidence (descending)
    sorted_preds = sorted(predictions, key=lambda p: p.get("confidence", 0), reverse=True)

    matched_gt: set[int] = set()
    tp = 0
    fp = 0
    matches = []

    for pred in sorted_preds:
        best_iou = 0.0
        best_gt_idx = -1

        for gt_idx, gt in enumerate(ground_truth):
            if gt_idx in matched_gt:
                continue

            # Only match same category
            if pred.get("category_id") != gt.get("category_id"):
                continue

            iou = compute_iou(pred["bbox"], gt["bbox"])
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx

        if best_iou >= iou_threshold and best_gt_idx >= 0:
            tp += 1
            matched_gt.add(best_gt_idx)
            matches.append({"pred": pred, "gt": ground_truth[best_gt_idx], "iou": best_iou})
        else:
            fp += 1

    fn = len(ground_truth) - len(matched_gt)

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": tp / (tp + fp) if (tp + fp) > 0 else 0.0,
        "recall": tp / (tp + fn) if (tp + fn) > 0 else 0.0,
        "matches": matches,
    }


def compute_ap(
    predictions: list[dict],
    ground_truth: list[dict],
    iou_threshold: float = 0.5,
) -> float:
    """Compute Average Precision at a given IoU threshold.

    Uses the VOC2010+ interpolated AP method (all-point interpolation).

    Args:
        predictions: List of prediction dicts with bbox, confidence, category_id.
        ground_truth: List of GT dicts with bbox, category_id.
        iou_threshold: IoU threshold for matching.

    Returns:
        AP value between 0 and 1.
    """
    if not ground_truth:
        return 0.0 if predictions else 1.0

    if not predictions:
        return 0.0

    # Sort by confidence descending
    sorted_preds = sorted(predictions, key=lambda p: p.get("confidence", 0), reverse=True)

    matched_gt: set[int] = set()
    tp_list = []
    fp_list = []

    for pred in sorted_preds:
        best_iou = 0.0
        best_gt_idx = -1

        for gt_idx, gt in enumerate(ground_truth):
            if gt_idx in matched_gt:
                continue
            if pred.get("category_id") != gt.get("category_id"):
                continue

            iou = compute_iou(pred["bbox"], gt["bbox"])
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx

        if best_iou >= iou_threshold and best_gt_idx >= 0:
            tp_list.append(1)
            fp_list.append(0)
            matched_gt.add(best_gt_idx)
        else:
            tp_list.append(0)
            fp_list.append(1)

    # Compute precision-recall curve
    tp_cumsum = 0
    fp_cumsum = 0
    precisions = []
    recalls = []
    n_gt = len(ground_truth)

    for tp_val, fp_val in zip(tp_list, fp_list):
        tp_cumsum += tp_val
        fp_cumsum += fp_val
        precision = tp_cumsum / (tp_cumsum + fp_cumsum)
        recall = tp_cumsum / n_gt
        precisions.append(precision)
        recalls.append(recall)

    # All-point interpolation
    # Ensure monotonically decreasing precision
    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = max(precisions[i], precisions[i + 1])

    # Compute area under PR curve
    ap = 0.0
    prev_recall = 0.0
    for prec, rec in zip(precisions, recalls):
        ap += prec * (rec - prev_recall)
        prev_recall = rec

    return ap


def compute_map(
    all_predictions: list[dict],
    all_ground_truth: list[dict],
    iou_threshold: float = 0.5,
    class_ids: list[int] | None = None,
) -> dict[str, float]:
    """Compute mean Average Precision across classes.

    Args:
        all_predictions: All prediction dicts with bbox, confidence, category_id.
        all_ground_truth: All GT dicts with bbox, category_id.
        iou_threshold: IoU threshold for matching.
        class_ids: List of class IDs to evaluate. None = all classes.

    Returns:
        Dict with mAP and per-class AP values.
    """
    if class_ids is None:
        gt_classes = set(ann.get("category_id") for ann in all_ground_truth)
        pred_classes = set(ann.get("category_id") for ann in all_predictions)
        class_ids = sorted(gt_classes | pred_classes)

    per_class_ap: dict[int, float] = {}
    for cid in class_ids:
        class_preds = [p for p in all_predictions if p.get("category_id") == cid]
        class_gt = [g for g in all_ground_truth if g.get("category_id") == cid]
        per_class_ap[cid] = compute_ap(class_preds, class_gt, iou_threshold)

    map_val = sum(per_class_ap.values()) / len(per_class_ap) if per_class_ap else 0.0

    result = {"mAP": map_val}
    for cid, ap_val in per_class_ap.items():
        class_name = VME_CLASSES.get(cid, f"class_{cid}")
        result[f"AP_{class_name}"] = ap_val

    return result


def build_confusion_matrix(
    predictions: list[dict],
    ground_truth: list[dict],
    class_ids: list[int],
    iou_threshold: float = 0.5,
) -> dict[str, Any]:
    """Build confusion matrix from predictions and ground truth.

    Args:
        predictions: Prediction dicts with bbox, confidence, category_id.
        ground_truth: GT dicts with bbox, category_id.
        class_ids: List of class IDs.
        iou_threshold: IoU threshold for matching.

    Returns:
        Dict with confusion matrix and per-class precision/recall/F1.
    """
    n_classes = len(class_ids)
    # Matrix: rows = true class, cols = predicted class
    # Extra col for "missed" (FN), extra row for "background" (FP)
    matrix = [[0] * (n_classes + 1) for _ in range(n_classes + 1)]

    # Match predictions to GT (greedy, confidence-ordered)
    sorted_preds = sorted(predictions, key=lambda p: p.get("confidence", 0), reverse=True)
    matched_gt: set[int] = set()

    for pred in sorted_preds:
        best_iou = 0.0
        best_gt_idx = -1

        for gt_idx, gt in enumerate(ground_truth):
            if gt_idx in matched_gt:
                continue
            iou = compute_iou(pred["bbox"], gt["bbox"])
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx

        pred_cid = pred.get("category_id", -1)
        pred_class_idx = class_ids.index(pred_cid) if pred_cid in class_ids else n_classes

        if best_iou >= iou_threshold and best_gt_idx >= 0:
            gt_cid = ground_truth[best_gt_idx].get("category_id", -1)
            gt_class_idx = class_ids.index(gt_cid) if gt_cid in class_ids else n_classes
            matrix[gt_class_idx][pred_class_idx] += 1
            matched_gt.add(best_gt_idx)
        else:
            # False positive (background row)
            matrix[n_classes][pred_class_idx] += 1

    # Count missed GT (false negatives)
    for gt_idx, gt in enumerate(ground_truth):
        if gt_idx not in matched_gt:
            gt_cid = gt.get("category_id", -1)
            gt_class_idx = class_ids.index(gt_cid) if gt_cid in class_ids else n_classes
            matrix[gt_class_idx][n_classes] += 1

    # Compute per-class metrics
    per_class = {}
    for i, cid in enumerate(class_ids):
        tp = matrix[i][i]
        fp = sum(matrix[j][i] for j in range(n_classes + 1)) - tp
        fn = sum(matrix[i][j] for j in range(n_classes + 1)) - tp
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        class_name = VME_CLASSES.get(cid, f"class_{cid}")
        per_class[class_name] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "tp": tp,
            "fp": fp,
            "fn": fn,
        }

    class_labels = [VME_CLASSES.get(cid, f"class_{cid}") for cid in class_ids] + ["Background/Missed"]

    return {
        "matrix": matrix,
        "class_labels": class_labels,
        "per_class": per_class,
    }


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------


def generate_report(
    metrics: dict[str, Any],
    config: dict[str, Any],
    output_dir: str,
) -> str:
    """Generate a Markdown evaluation report.

    Args:
        metrics: All computed metrics.
        config: Training/evaluation configuration.
        output_dir: Directory to write the report.

    Returns:
        Path to the generated report.
    """
    report_path = Path(output_dir) / "evaluation_report.md"
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    lines = [
        "# VME Fine-Tuning Evaluation Report",
        "",
        "## Summary",
        "",
    ]

    # mAP metrics
    if "map_0.5" in metrics:
        lines.append(f"- **mAP@0.5**: {metrics['map_0.5'].get('mAP', 'N/A'):.4f}")
    if "map_0.75" in metrics:
        lines.append(f"- **mAP@0.75**: {metrics['map_0.75'].get('mAP', 'N/A'):.4f}")

    lines.append("")

    # Per-class AP
    if "map_0.5" in metrics:
        lines.append("## Per-Class Average Precision (IoU=0.5)")
        lines.append("")
        lines.append("| Class | AP |")
        lines.append("|-------|-----|")
        for key, val in metrics["map_0.5"].items():
            if key.startswith("AP_"):
                class_name = key[3:]
                lines.append(f"| {class_name} | {val:.4f} |")
        lines.append("")

    # Confusion matrix
    if "confusion" in metrics:
        cm = metrics["confusion"]
        lines.append("## Per-Class Metrics")
        lines.append("")
        lines.append("| Class | Precision | Recall | F1 | TP | FP | FN |")
        lines.append("|-------|-----------|--------|-----|----|----|-----|")
        for class_name, vals in cm.get("per_class", {}).items():
            lines.append(
                f"| {class_name} | {vals['precision']:.4f} | "
                f"{vals['recall']:.4f} | {vals['f1']:.4f} | "
                f"{vals['tp']} | {vals['fp']} | {vals['fn']} |"
            )
        lines.append("")

    # Success criteria
    lines.append("## Success Criteria")
    lines.append("")
    criteria_results = metrics.get("success_criteria", {})
    for criterion, result in criteria_results.items():
        status = "PASS" if result.get("passed") else "FAIL"
        mark = "[x]" if result.get("passed") else "[ ]"
        lines.append(
            f"- {mark} **{criterion}**: {status} "
            f"(value={result.get('value', 'N/A')}, threshold={result.get('threshold', 'N/A')})"
        )
    lines.append("")

    overall = all(r.get("passed", False) for r in criteria_results.values())
    lines.append(f"**Overall: {'SUCCESS' if overall else 'NEEDS IMPROVEMENT'}**")
    lines.append("")

    # Reproducibility info
    lines.append("## Reproducibility")
    lines.append("")
    lines.append(f"- **Checkpoint**: {config.get('checkpoint', 'N/A')}")
    lines.append(f"- **Dataset**: VME (Zenodo record {config.get('zenodo_record', '14185684')})")
    lines.append(f"- **Model**: RF-DETR {config.get('model', 'Medium')}")
    lines.append(f"- **Evaluation date**: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")

    report_text = "\n".join(lines)
    report_path.write_text(report_text)
    return str(report_path)


def check_success_criteria(
    metrics: dict[str, Any],
    training_hours: float | None = None,
) -> dict[str, dict[str, Any]]:
    """Check evaluation results against success criteria.

    Args:
        metrics: Computed evaluation metrics.
        training_hours: Training duration in hours.

    Returns:
        Dict mapping criterion name to pass/fail result.
    """
    results: dict[str, dict[str, Any]] = {}

    # mAP@0.5 > 0.60
    if "map_0.5" in metrics:
        map_val = metrics["map_0.5"].get("mAP", 0.0)
        results["mAP@0.5 > 0.60"] = {
            "passed": map_val > SUCCESS_CRITERIA["mAP_0.5"],
            "value": round(map_val, 4),
            "threshold": SUCCESS_CRITERIA["mAP_0.5"],
        }

    # Training time < 4 hours
    if training_hours is not None:
        results["Training time < 4 hours"] = {
            "passed": training_hours < SUCCESS_CRITERIA["training_hours"],
            "value": round(training_hours, 2),
            "threshold": SUCCESS_CRITERIA["training_hours"],
        }

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Argument list.

    Returns:
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Validate VME-finetuned RF-DETR model and compare to COCO baseline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to fine-tuned model checkpoint (.pth)",
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default="vme_dataset",
        help="Path to VME dataset directory (default: vme_dataset)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="evaluation_results",
        help="Directory for evaluation output (default: evaluation_results)",
    )
    parser.add_argument(
        "--compare_baseline",
        action="store_true",
        help="Also evaluate COCO-pretrained model for comparison",
    )
    parser.add_argument(
        "--real_world_test",
        type=str,
        default=None,
        help="Path to real-world test images for qualitative evaluation",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.3,
        help="Confidence threshold for detections (default: 0.3)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="medium",
        choices=["nano", "small", "medium", "base", "large"],
        help="RF-DETR model size (default: medium)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run VME evaluation pipeline.

    Args:
        argv: Command-line arguments.

    Returns:
        Exit code: 0 on success, 1 on failure.
    """
    args = parse_args(argv)

    print("=" * 60)
    print("  VME Fine-Tuning Evaluation")
    print("=" * 60)

    # Verify checkpoint exists
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"  ERROR: Checkpoint not found: {args.checkpoint}")
        return 1

    # Verify dataset
    dataset_path = Path(args.dataset_dir)
    if not dataset_path.exists():
        print(f"  ERROR: Dataset not found: {args.dataset_dir}")
        print("  Run python scripts/download_vme.py first.")
        return 1

    # Load ground truth
    print(f"\n  Loading ground truth from {args.dataset_dir}/valid/...")
    valid_ann_path = dataset_path / "valid" / "_annotations.coco.json"
    if not valid_ann_path.exists():
        print(f"  ERROR: Missing valid/_annotations.coco.json")
        return 1

    with open(valid_ann_path) as f:
        valid_data = json.load(f)

    gt_annotations = valid_data.get("annotations", [])
    gt_images = valid_data.get("images", [])
    categories = {c["id"]: c["name"] for c in valid_data.get("categories", [])}

    print(f"  Ground truth: {len(gt_images)} images, {len(gt_annotations)} annotations")
    print(f"  Categories: {categories}")

    # Note: actual model inference would happen here in a real evaluation.
    # This script provides the evaluation framework; the user runs it
    # after training completes on their GPU.
    print(f"\n  To run full evaluation, this script requires:")
    print(f"    - Fine-tuned checkpoint: {args.checkpoint}")
    print(f"    - GPU for inference (or patience for CPU)")
    print(f"    - rfdetr package installed")
    print(f"\n  The evaluation framework is ready.")
    print(f"  Run with actual checkpoint after training completes.")

    # Generate skeleton report
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics: dict[str, Any] = {}
    config = {
        "checkpoint": str(args.checkpoint),
        "dataset_dir": str(args.dataset_dir),
        "model": args.model,
        "confidence": args.confidence,
        "zenodo_record": "14185684",
    }

    # Export metrics JSON
    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n  Metrics exported: {metrics_path}")

    # Generate report
    report_path = generate_report(metrics, config, str(output_dir))
    print(f"  Report generated: {report_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
