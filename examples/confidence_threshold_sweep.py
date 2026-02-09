"""Confidence threshold sweep: find the right detection threshold.

Runs detection once at a low threshold, then filters the results at
multiple thresholds to show how detection count and average confidence
change. Helps you pick the right threshold for your use case:

  - Lower threshold (0.1-0.3): more detections, more false positives
  - Higher threshold (0.5-0.7): fewer detections, higher precision
  - Very high threshold (0.8+): only the most confident detections

Outputs a CSV with per-threshold statistics and optionally exports
GeoPackage files at each threshold for visual comparison in QGIS.

Prerequisites:
    pip install detr-geo[rfdetr]

Usage:
    python examples/confidence_threshold_sweep.py image.tif
    python examples/confidence_threshold_sweep.py image.tif weights.pth --export-all
"""

from __future__ import annotations

import argparse
import csv
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

# Thresholds to evaluate
THRESHOLDS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep confidence thresholds to find the right cutoff.")
    parser.add_argument("image", help="Path to GeoTIFF")
    parser.add_argument(
        "weights",
        nargs="?",
        default=None,
        help="Path to fine-tuned weights (omit for COCO-pretrained)",
    )
    parser.add_argument(
        "--export-all",
        action="store_true",
        help="Export a GeoPackage at each threshold for visual comparison",
    )
    parser.add_argument(
        "--output",
        default=".",
        help="Output directory (default: current directory)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    image_stem = Path(args.image).stem

    # Initialize model
    if args.weights:
        dg = DetrGeo(
            model_size="medium",
            pretrain_weights=args.weights,
            custom_class_names=XVIEW_CLASSES,
            confidence_threshold=0.05,  # Very low -- we filter manually
        )
        model_name = "xView fine-tuned"
    else:
        dg = DetrGeo(
            model_size="medium",
            confidence_threshold=0.05,
        )
        model_name = "COCO-pretrained"

    print(f"Model: {model_name}")
    print(f"Image: {Path(args.image).name}")

    # Run detection once at the lowest threshold
    dg.set_image(args.image, suppress_gsd_warning=True)
    print("\nRunning detection at threshold=0.05 (collecting all candidates)...")

    all_detections = dg.detect_tiled(
        overlap=0.2,
        nms_threshold=0.5,
        threshold=0.05,
    )

    print(f"  Raw detections (threshold=0.05): {len(all_detections)}")

    if len(all_detections) == 0:
        print("  No detections found at any threshold.")
        return

    # Sweep thresholds by filtering the single detection run
    print(f"\n  {'Threshold':>10s}  {'Detections':>10s}  {'Mean Conf':>10s}  {'Classes':>8s}")
    print("  " + "-" * 45)

    sweep_results = []
    for thresh in THRESHOLDS:
        filtered = all_detections[all_detections["confidence"] >= thresh]
        n = len(filtered)
        mean_conf = float(filtered["confidence"].mean()) if n > 0 else 0.0
        n_classes = filtered["class_name"].nunique() if n > 0 else 0

        sweep_results.append(
            {
                "threshold": thresh,
                "detections": n,
                "mean_confidence": round(mean_conf, 4),
                "unique_classes": n_classes,
            }
        )

        print(f"  {thresh:>10.1f}  {n:>10d}  {mean_conf:>10.3f}  {n_classes:>8d}")

        # Optionally export at each threshold
        if args.export_all and n > 0:
            export_path = output_dir / f"{image_stem}_thresh_{thresh:.1f}.gpkg"
            filtered.to_file(str(export_path), driver="GPKG")

    # Write CSV summary
    csv_path = output_dir / f"{image_stem}_threshold_sweep.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["threshold", "detections", "mean_confidence", "unique_classes"])
        writer.writeheader()
        writer.writerows(sweep_results)

    print(f"\n  Sweep results: {csv_path}")

    if args.export_all:
        print(f"  GeoPackages exported to {output_dir}/")
        print("  Load all in QGIS to visually compare thresholds.")

    # Recommendation
    print("\n  Recommendation:")
    print("    - Counting objects? Use 0.3-0.4 (higher recall)")
    print("    - Precise locations? Use 0.5-0.6 (balanced)")
    print("    - High-stakes decisions? Use 0.7+ (high precision)")


if __name__ == "__main__":
    main()
