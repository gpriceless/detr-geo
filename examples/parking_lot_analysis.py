"""Parking lot analysis: count and classify vehicles from satellite imagery.

Detects vehicles in a parking lot or commercial area, breaks them down by
class (Car, Pickup Truck, Truck, Bus, Other Vehicle), and exports a GeoJSON
with per-class counts as properties. Useful for occupancy monitoring,
fleet tracking, or commercial real estate analysis.

The xView fine-tuned model is recommended for this use case. The COCO
baseline model was not trained on overhead imagery and will produce
unreliable results.

Prerequisites:
    pip install detr-geo[all]

Usage:
    python examples/parking_lot_analysis.py parking_lot.tif weights.pth
    python examples/parking_lot_analysis.py parking_lot.tif weights.pth --output results/
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from detr_geo import DetrGeo
from detr_geo.io import load_raster_metadata

# xView vehicle class mapping
XVIEW_CLASSES = {
    0: "Car",
    1: "Pickup Truck",
    2: "Truck",
    3: "Bus",
    4: "Other Vehicle",
}


def analyze_parking_lot(
    image_path: str,
    weights_path: str,
    output_dir: str = ".",
    threshold: float = 0.3,
) -> None:
    """Run vehicle detection and produce a parking lot analysis report."""

    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    # Inspect the raster
    meta = load_raster_metadata(image_path)
    image_name = Path(image_path).stem
    print(f"Analyzing: {Path(image_path).name}")
    print(f"  Size: {meta.width} x {meta.height} pixels")
    if meta.gsd:
        area_km2 = (meta.width * meta.gsd * meta.height * meta.gsd) / 1e6
        print(f"  GSD: {meta.gsd:.3f} m/px")
        print(f"  Coverage: {area_km2:.3f} km2")

    # Load xView fine-tuned model
    dg = DetrGeo(
        model_size="medium",
        pretrain_weights=weights_path,
        custom_class_names=XVIEW_CLASSES,
        confidence_threshold=threshold,
    )
    dg.set_image(image_path, suppress_gsd_warning=True)

    # Detect vehicles with tiled processing
    detections = dg.detect_tiled(
        overlap=0.2,
        nms_threshold=0.5,
        threshold=threshold,
    )

    total = len(detections)
    print(f"\n  Total vehicles detected: {total}")

    if total == 0:
        print("  No vehicles found. Try lowering the confidence threshold.")
        return

    # Per-class breakdown
    counts = Counter(detections["class_name"])
    print("\n  Vehicle breakdown:")
    for cls, count in sorted(counts.items(), key=lambda x: -x[1]):
        pct = 100.0 * count / total
        print(f"    {cls:<15s} {count:>5d}  ({pct:5.1f}%)")

    # Confidence summary
    scores = detections["confidence"]
    print(f"\n  Confidence: min={scores.min():.2f}  mean={scores.mean():.2f}  max={scores.max():.2f}")

    # High-confidence subset
    high_conf = detections[detections["confidence"] > 0.7]
    print(f"  High confidence (>0.7): {len(high_conf)} / {total}")

    # Export detections as GeoJSON with summary properties
    geojson_path = output_dir_path / f"{image_name}_vehicles.geojson"
    dg.to_geojson(str(geojson_path))
    print(f"\n  Detections: {geojson_path}")

    # Export as GeoPackage for GIS tools
    gpkg_path = output_dir_path / f"{image_name}_vehicles.gpkg"
    dg.to_gpkg(str(gpkg_path))
    print(f"  GeoPackage: {gpkg_path}")

    # Write a JSON summary report
    summary = {
        "image": str(Path(image_path).name),
        "total_vehicles": total,
        "high_confidence_vehicles": len(high_conf),
        "confidence_threshold": threshold,
        "class_counts": dict(counts),
        "confidence_stats": {
            "min": round(float(scores.min()), 3),
            "mean": round(float(scores.mean()), 3),
            "max": round(float(scores.max()), 3),
        },
    }
    if meta.gsd:
        summary["gsd_meters"] = round(meta.gsd, 4)
        summary["coverage_km2"] = round(area_km2, 4)

    summary_path = output_dir_path / f"{image_name}_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Summary:    {summary_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Count and classify vehicles in a parking lot from satellite imagery.")
    parser.add_argument("image", help="Path to GeoTIFF")
    parser.add_argument("weights", help="Path to xView fine-tuned weights (.pth)")
    parser.add_argument(
        "--output",
        default=".",
        help="Output directory (default: current directory)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.3,
        help="Confidence threshold (default: 0.3)",
    )
    args = parser.parse_args()

    analyze_parking_lot(args.image, args.weights, args.output, args.threshold)


if __name__ == "__main__":
    main()
