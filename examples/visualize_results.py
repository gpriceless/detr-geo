"""Visualize detections: static matplotlib plots and GeoJSON/GeoPackage export.

Shows how to overlay bounding boxes on source imagery using show_detections(),
display results on an interactive web map with show_map(), and export to
standard geospatial vector formats.

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

    # detect() loads the full image into memory, which is needed for
    # show_detections(). For large rasters, use detect_tiled() instead
    # and skip show_detections() (it requires the in-memory image).
    detections = dg.detect()
    print(f"Detected {len(detections)} objects")

    if len(detections) == 0:
        print("No detections to visualize.")
        return

    output_stem = Path(image_path).stem

    # --- Static matplotlib visualization ---
    # Draws bounding boxes on the source image with class labels.
    # Requires detect() (not detect_tiled) because it needs the full image.
    fig, ax = dg.show_detections(
        figsize=(15, 12),
        min_confidence=0.5,
        show_labels=True,
        save_path=f"{output_stem}_detections.png",
        dpi=200,
    )
    print(f"Saved static plot to {output_stem}_detections.png")

    # --- Export to GeoJSON ---
    # GeoJSON auto-reprojects to WGS84 (EPSG:4326) per the spec.
    # Coordinates become longitude/latitude regardless of the input CRS.
    dg.to_geojson(f"{output_stem}_detections.geojson")
    print(f"Exported GeoJSON (WGS84) to {output_stem}_detections.geojson")

    # --- Export to GeoPackage ---
    # GeoPackage preserves the original CRS from the raster.
    # Recommended for GIS workflows (QGIS, ArcGIS, PostGIS import).
    dg.to_gpkg(f"{output_stem}_detections.gpkg", layer="detections")
    print(f"Exported GeoPackage to {output_stem}_detections.gpkg")

    # --- Working with the GeoDataFrame directly ---
    gdf = dg.detections
    high_conf = gdf[gdf["confidence"] > 0.7]
    print(f"\n{len(high_conf)} detections above 0.7 confidence:")
    print(high_conf[["class_name", "confidence"]].to_string(index=False))

    # --- Interactive web map (Jupyter / browser) ---
    # Uncomment the lines below in a Jupyter notebook:
    #
    # m = dg.show_map(basemap="SATELLITE", min_confidence=0.5)
    # m  # displays the interactive map in Jupyter


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python visualize_results.py <image.tif>")
        sys.exit(1)
    main(sys.argv[1])
