"""Change detection: compare vehicle presence between two dates.

Takes two GeoTIFFs of the same area at different times (e.g., before and
after an event, weekday vs weekend, or monthly monitoring), runs vehicle
detection on both, and reports what changed.

Outputs:
  - Per-image detection counts and class breakdowns
  - Change summary: how many more/fewer vehicles of each type
  - Two GeoPackage files for visual comparison in QGIS
  - A combined GeoPackage with both dates as separate layers

Useful for:
  - Parking lot occupancy tracking (weekday vs weekend)
  - Construction site monitoring (equipment mobilization)
  - Event impact analysis (before vs after)
  - Seasonal traffic pattern analysis

Prerequisites:
    pip install detr-geo[all]

Usage:
    python examples/monitor_change_detection.py before.tif after.tif weights.pth
    python examples/monitor_change_detection.py \
        monday.tif friday.tif weights.pth --output monitoring/
"""

from __future__ import annotations

import argparse
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


def detect_image(dg: DetrGeo, image_path: str, threshold: float) -> tuple[object, Counter]:
    """Run detection on one image and return (GeoDataFrame, class_counts)."""
    dg.set_image(image_path, suppress_gsd_warning=True)
    detections = dg.detect_tiled(
        overlap=0.2,
        nms_threshold=0.5,
        threshold=threshold,
    )
    counts = Counter(detections["class_name"]) if len(detections) > 0 else Counter()
    return detections, counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare detections between two dates of the same area.")
    parser.add_argument("before", help="Path to 'before' GeoTIFF")
    parser.add_argument("after", help="Path to 'after' GeoTIFF")
    parser.add_argument("weights", help="Path to xView fine-tuned weights (.pth)")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.3,
        help="Confidence threshold (default: 0.3)",
    )
    parser.add_argument(
        "--output",
        default=".",
        help="Output directory (default: current directory)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    before_name = Path(args.before).stem
    after_name = Path(args.after).stem

    # Initialize model
    dg = DetrGeo(
        model_size="medium",
        pretrain_weights=args.weights,
        custom_class_names=XVIEW_CLASSES,
        confidence_threshold=args.threshold,
    )

    # Detect in "before" image
    print(f"Processing 'before': {Path(args.before).name}")
    before_gdf, before_counts = detect_image(dg, args.before, args.threshold)
    before_total = len(before_gdf)
    print(f"  Detected {before_total} vehicles")

    # Detect in "after" image
    print(f"Processing 'after':  {Path(args.after).name}")
    after_gdf, after_counts = detect_image(dg, args.after, args.threshold)
    after_total = len(after_gdf)
    print(f"  Detected {after_total} vehicles")

    # Compute changes
    all_classes = sorted(set(list(before_counts.keys()) + list(after_counts.keys())))

    print("\n" + "=" * 65)
    print("  CHANGE DETECTION SUMMARY")
    print("=" * 65)
    print(f"\n  {'Class':<15s}  {'Before':>8s}  {'After':>8s}  {'Change':>8s}  {'%':>8s}")
    print("  " + "-" * 55)

    for cls in all_classes:
        b = before_counts.get(cls, 0)
        a = after_counts.get(cls, 0)
        change = a - b
        if b > 0:
            pct = 100.0 * change / b
            pct_str = f"{pct:+.1f}%"
        elif a > 0:
            pct_str = "new"
        else:
            pct_str = "--"

        change_str = f"{change:+d}" if change != 0 else "0"
        print(f"  {cls:<15s}  {b:>8d}  {a:>8d}  {change_str:>8s}  {pct_str:>8s}")

    # Total row
    total_change = after_total - before_total
    if before_total > 0:
        total_pct = f"{100.0 * total_change / before_total:+.1f}%"
    elif after_total > 0:
        total_pct = "new"
    else:
        total_pct = "--"

    print("  " + "-" * 55)
    total_change_str = f"{total_change:+d}" if total_change != 0 else "0"
    print(f"  {'TOTAL':<15s}  {before_total:>8d}  {after_total:>8d}  {total_change_str:>8s}  {total_pct:>8s}")

    # Interpretation
    print("\n  Interpretation:")
    if total_change > 0:
        print(f"    {total_change} more vehicles detected in the 'after' image.")
    elif total_change < 0:
        print(f"    {abs(total_change)} fewer vehicles detected in the 'after' image.")
    else:
        print("    Vehicle count is the same in both images.")

    # Export results
    # Individual GeoPackages
    if before_total > 0:
        before_path = output_dir / f"{before_name}_detections.gpkg"
        before_gdf.to_file(str(before_path), driver="GPKG")
        print(f"\n  Before: {before_path}")

    if after_total > 0:
        after_path = output_dir / f"{after_name}_detections.gpkg"
        after_gdf.to_file(str(after_path), driver="GPKG")
        print(f"  After:  {after_path}")

    # Combined GeoPackage with both dates as layers
    if before_total > 0 or after_total > 0:
        combined_path = output_dir / f"change_{before_name}_vs_{after_name}.gpkg"
        if before_total > 0:
            before_gdf.to_file(str(combined_path), driver="GPKG", layer=f"before_{before_name}")
        if after_total > 0:
            after_gdf.to_file(str(combined_path), driver="GPKG", layer=f"after_{after_name}")
        print(f"  Combined: {combined_path}")
        print(f"    Layers: before_{before_name}, after_{after_name}")
        print("    Open in QGIS to overlay both dates and compare visually.")


if __name__ == "__main__":
    main()
