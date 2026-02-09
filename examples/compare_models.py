"""Compare models: COCO baseline vs xView fine-tuned on the same image.

Runs the same satellite image through both the COCO-pretrained model and
the xView fine-tuned model, then prints a side-by-side comparison. This
demonstrates why fine-tuning matters for overhead imagery -- the COCO model
misclassifies or misses most overhead vehicles.

The COCO model typically labels overhead vehicles as "motorcycle", "boat",
"skateboard", or misses them entirely. The xView model correctly identifies
Car, Pickup Truck, Truck, Bus, and Other Vehicle.

Prerequisites:
    pip install detr-geo[all]

Usage:
    python examples/compare_models.py satellite.tif checkpoints/checkpoint_best_ema.pth
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

from detr_geo import DetrGeo

# xView vehicle class mapping
XVIEW_CLASSES = {
    0: "Car",
    1: "Pickup Truck",
    2: "Truck",
    3: "Bus",
    4: "Other Vehicle",
}


def run_detection(dg: DetrGeo, image_path: str, threshold: float) -> dict:
    """Run detection and return summary statistics."""
    dg.set_image(image_path, suppress_gsd_warning=True)
    detections = dg.detect_tiled(overlap=0.2, nms_threshold=0.5, threshold=threshold)

    result = {
        "total": len(detections),
        "class_counts": Counter(detections["class_name"]) if len(detections) > 0 else Counter(),
    }

    if len(detections) > 0:
        scores = detections["confidence"]
        result["conf_min"] = float(scores.min())
        result["conf_mean"] = float(scores.mean())
        result["conf_max"] = float(scores.max())
    else:
        result["conf_min"] = 0.0
        result["conf_mean"] = 0.0
        result["conf_max"] = 0.0

    return result


def print_section(title: str, result: dict) -> None:
    """Print detection results for one model."""
    print(f"\n  {title}")
    print(f"  {'=' * len(title)}")
    print(f"  Total detections: {result['total']}")

    if result["total"] > 0:
        print("\n  Class breakdown:")
        for cls, count in result["class_counts"].most_common(15):
            print(f"    {cls:<25s} {count:>5d}")

        print(
            f"\n  Confidence: min={result['conf_min']:.2f}  "
            f"mean={result['conf_mean']:.2f}  max={result['conf_max']:.2f}"
        )
    else:
        print("  (no detections)")


def main(image_path: str, weights_path: str) -> None:
    threshold = 0.3
    print(f"Comparing models on: {Path(image_path).name}")
    print(f"Confidence threshold: {threshold}")

    # --- COCO-pretrained model ---
    print("\nRunning COCO-pretrained model...")
    dg_coco = DetrGeo(
        model_size="medium",
        confidence_threshold=threshold,
    )
    coco_result = run_detection(dg_coco, image_path, threshold)
    print_section("COCO-Pretrained (ground-level training)", coco_result)

    # --- xView fine-tuned model ---
    print("\nRunning xView fine-tuned model...")
    dg_xview = DetrGeo(
        model_size="medium",
        pretrain_weights=weights_path,
        custom_class_names=XVIEW_CLASSES,
        confidence_threshold=threshold,
    )
    xview_result = run_detection(dg_xview, image_path, threshold)
    print_section("xView Fine-Tuned (overhead vehicle training)", xview_result)

    # --- Side-by-side summary ---
    print("\n\n  COMPARISON SUMMARY")
    print("  " + "=" * 50)
    print(f"  {'Metric':<30s} {'COCO':>8s}  {'xView':>8s}")
    print("  " + "-" * 50)
    print(f"  {'Total detections':<30s} {coco_result['total']:>8d}  {xview_result['total']:>8d}")
    print(f"  {'Mean confidence':<30s} {coco_result['conf_mean']:>8.2f}  {xview_result['conf_mean']:>8.2f}")

    # Show which COCO classes the model produces (often wrong for overhead)
    if coco_result["total"] > 0:
        coco_classes = list(coco_result["class_counts"].keys())
        print(f"\n  COCO model thinks it sees: {', '.join(coco_classes[:10])}")

    if xview_result["total"] > 0:
        xview_classes = list(xview_result["class_counts"].keys())
        print(f"  xView model detects:       {', '.join(xview_classes)}")

    # Export both results
    output_stem = Path(image_path).stem
    if coco_result["total"] > 0:
        dg_coco.to_gpkg(f"{output_stem}_coco_detections.gpkg")
        print(f"\n  COCO results:  {output_stem}_coco_detections.gpkg")
    if xview_result["total"] > 0:
        dg_xview.to_gpkg(f"{output_stem}_xview_detections.gpkg")
        print(f"  xView results: {output_stem}_xview_detections.gpkg")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python compare_models.py <satellite_image.tif> <xview_weights.pth>")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
