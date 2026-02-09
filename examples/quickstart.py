"""Quickstart: detect objects in a GeoTIFF and export georeferenced results.

Three lines from GeoTIFF to GeoPackage. This is the smallest useful
detr-geo script -- load imagery, detect, export.

The output is a GeoPackage with polygon geometries in the raster's CRS,
ready to open in QGIS, ArcGIS, or load into PostGIS.

Prerequisites:
    pip install detr-geo[rfdetr]

Usage:
    python examples/quickstart.py path/to/geotiff.tif
"""

from __future__ import annotations

import sys
from pathlib import Path

from detr_geo import DetrGeo


def main(image_path: str) -> None:
    # Load model (downloads COCO-pretrained weights on first run, ~2 GB)
    dg = DetrGeo(model_size="medium")

    # Point at a GeoTIFF -- CRS and transform are read automatically
    dg.set_image(image_path)

    # Detect objects. For large rasters, use detect_tiled() instead.
    detections = dg.detect(threshold=0.4)

    # Results are a GeoDataFrame with geometry in the raster's CRS
    print(f"Detected {len(detections)} objects in {Path(image_path).name}")
    if len(detections) > 0:
        print(detections[["class_name", "confidence"]].head(10))

    # Export to GeoPackage (preserves CRS) and GeoJSON (auto-reprojects to WGS84)
    output_stem = Path(image_path).stem
    dg.to_gpkg(f"{output_stem}_detections.gpkg")
    dg.to_geojson(f"{output_stem}_detections.geojson")
    print(f"\nExported to {output_stem}_detections.gpkg and .geojson")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python quickstart.py <geotiff.tif>")
        sys.exit(1)
    main(sys.argv[1])
