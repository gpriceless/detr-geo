"""Export for GIS: full pipeline from GeoTIFF to analysis-ready GeoPackage.

Produces a GeoPackage with a complete attribute table designed for GIS
workflows. Each detection polygon includes:
  - class_name and class_id
  - confidence score
  - centroid coordinates (in the raster's CRS)
  - area in square meters (computed via UTM projection)
  - source image filename

The GeoPackage preserves the original CRS and is ready to load directly
into QGIS, ArcGIS, or PostGIS. Also exports a GeoJSON version reprojected
to WGS84 for web maps.

Prerequisites:
    pip install detr-geo[all]

Usage:
    python examples/export_for_gis.py satellite.tif weights.pth
    python examples/export_for_gis.py satellite.tif weights.pth --output results/analysis.gpkg
"""

from __future__ import annotations

import argparse
from pathlib import Path

from detr_geo import DetrGeo
from detr_geo.export import compute_areas
from detr_geo.io import load_raster_metadata

# xView vehicle class mapping
XVIEW_CLASSES = {
    0: "Car",
    1: "Pickup Truck",
    2: "Truck",
    3: "Bus",
    4: "Other Vehicle",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Export detections as analysis-ready GeoPackage for QGIS/ArcGIS.")
    parser.add_argument("image", help="Path to GeoTIFF")
    parser.add_argument("weights", help="Path to xView fine-tuned weights (.pth)")
    parser.add_argument(
        "--output",
        default=None,
        help="Output GeoPackage path (default: <image_stem>_analysis.gpkg)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.3,
        help="Confidence threshold (default: 0.3)",
    )
    parser.add_argument(
        "--layer",
        default="vehicle_detections",
        help="GeoPackage layer name (default: vehicle_detections)",
    )
    args = parser.parse_args()

    # Inspect raster
    meta = load_raster_metadata(args.image)
    print(f"Input: {Path(args.image).name}")
    print(f"  CRS: {meta.crs}")
    print(f"  Size: {meta.width} x {meta.height}")
    if meta.gsd:
        print(f"  GSD: {meta.gsd:.3f} m/px")

    # Load model and detect
    dg = DetrGeo(
        model_size="medium",
        pretrain_weights=args.weights,
        custom_class_names=XVIEW_CLASSES,
        confidence_threshold=args.threshold,
    )
    dg.set_image(args.image, suppress_gsd_warning=True)

    detections = dg.detect_tiled(
        overlap=0.2,
        nms_threshold=0.5,
        threshold=args.threshold,
    )

    print(f"\n  Detected {len(detections)} vehicles")

    if len(detections) == 0:
        print("  No detections to export.")
        return

    # Enrich the GeoDataFrame with additional attributes for GIS analysis
    gdf = detections.copy()

    # Add area in square meters (uses UTM projection for geographic CRS)
    gdf["area_m2"] = compute_areas(gdf)

    # Add source image reference
    gdf["source_image"] = Path(args.image).name

    # Round confidence for cleaner attribute tables
    gdf["confidence"] = gdf["confidence"].round(4)

    # Round centroids
    if "centroid_x" in gdf.columns:
        gdf["centroid_x"] = gdf["centroid_x"].round(6)
        gdf["centroid_y"] = gdf["centroid_y"].round(6)

    # Round area
    gdf["area_m2"] = gdf["area_m2"].round(2)

    # Store detections back for export methods
    dg._detections = gdf

    # Determine output paths
    image_stem = Path(args.image).stem
    if args.output:
        gpkg_path = args.output
    else:
        gpkg_path = f"{image_stem}_analysis.gpkg"

    geojson_path = gpkg_path.replace(".gpkg", ".geojson")

    # Export GeoPackage (preserves original CRS)
    gdf.to_file(gpkg_path, driver="GPKG", layer=args.layer)
    print(f"\n  GeoPackage: {gpkg_path}")
    print(f"    Layer: {args.layer}")
    print(f"    CRS: {gdf.crs}")
    print(f"    Features: {len(gdf)}")

    # Export GeoJSON (reprojected to WGS84)
    dg.to_geojson(geojson_path)
    print(f"\n  GeoJSON (WGS84): {geojson_path}")

    # Print attribute summary
    print("\n  Attribute table preview:")
    preview_cols = ["class_name", "confidence", "area_m2"]
    available_cols = [c for c in preview_cols if c in gdf.columns]
    print(gdf[available_cols].head(10).to_string(index=False))

    # Per-class area statistics
    print("\n  Per-class summary:")
    for cls in sorted(gdf["class_name"].unique()):
        cls_data = gdf[gdf["class_name"] == cls]
        print(
            f"    {cls:<15s}  n={len(cls_data):>5d}  "
            f"mean_area={cls_data['area_m2'].mean():>8.1f} m2  "
            f"mean_conf={cls_data['confidence'].mean():.2f}"
        )

    print(f"\n  Ready to open in QGIS: Layer > Add Layer > Add Vector Layer > {gpkg_path}")


if __name__ == "__main__":
    main()
