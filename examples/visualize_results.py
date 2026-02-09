"""Visualize detections: static plots, interactive maps, and vector export.

Shows three ways to view and share detection results:
  1. Static matplotlib plot with bounding boxes on the source image
  2. Interactive leafmap for exploring detections on a satellite basemap
  3. Vector export to GeoJSON and GeoPackage for GIS tools

Prerequisites:
    pip install detr-geo[all]

Usage:
    python examples/visualize_results.py path/to/geotiff.tif
"""

from __future__ import annotations

import sys
from pathlib import Path

from detr_geo import DetrGeo


def main(image_path: str) -> None:
    dg = DetrGeo(model_size="medium", confidence_threshold=0.4)
    dg.set_image(image_path, suppress_gsd_warning=True)

    # detect() loads the full image into memory -- needed for show_detections().
    # For large rasters, use detect_tiled() and skip the matplotlib plot.
    detections = dg.detect()
    print(f"Detected {len(detections)} objects")

    if len(detections) == 0:
        print("No detections to visualize.")
        return

    output_stem = Path(image_path).stem

    # --- 1. Static matplotlib plot ---
    # Draws bounding boxes on the source image with class labels and confidence.
    # Requires detect() (not detect_tiled) because it uses the in-memory image.
    fig, ax = dg.show_detections(
        figsize=(15, 12),
        min_confidence=0.5,
        show_labels=True,
        save_path=f"{output_stem}_detections.png",
        dpi=200,
    )
    print(f"Saved static plot to {output_stem}_detections.png")

    # --- 2. Export to GeoJSON (WGS84) and GeoPackage (original CRS) ---
    dg.to_geojson(f"{output_stem}_detections.geojson")
    print(f"Exported GeoJSON (WGS84) to {output_stem}_detections.geojson")

    dg.to_gpkg(f"{output_stem}_detections.gpkg", layer="detections")
    print(f"Exported GeoPackage to {output_stem}_detections.gpkg")

    # --- 3. Work with the GeoDataFrame directly ---
    gdf = dg.detections
    high_conf = gdf[gdf["confidence"] > 0.7]
    print(f"\n{len(high_conf)} detections above 0.7 confidence:")
    if len(high_conf) > 0:
        print(high_conf[["class_name", "confidence"]].to_string(index=False))

    # --- 4. Interactive web map (Jupyter / browser) ---
    # Uncomment in a Jupyter notebook:
    #
    # m = dg.show_map(basemap="SATELLITE", min_confidence=0.5)
    # m  # displays the interactive map


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python visualize_results.py <image.tif>")
        sys.exit(1)
    main(sys.argv[1])
