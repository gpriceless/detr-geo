"""Quickstart: detect objects in a GeoTIFF and export georeferenced results.

Minimal example showing the core detr-geo workflow in under 15 lines.
Uses COCO-pretrained weights -- no custom model needed.

Prerequisites:
    pip install detr-geo[rfdetr]

Usage:
    python examples/quickstart.py path/to/geotiff.tif
"""

from detr_geo import DetrGeo

# Load model (downloads COCO-pretrained weights on first run)
dg = DetrGeo(model_size="medium")

# Point at a GeoTIFF -- CRS and transform are read automatically
dg.set_image("path/to/geotiff.tif")

# Run detection (single pass, image must fit in memory)
detections = dg.detect(threshold=0.4)

# Results are a GeoDataFrame with geometry in the raster's CRS
print(f"Detected {len(detections)} objects")
print(detections[["class_name", "confidence"]].head(10))

# Export to GeoJSON (auto-reprojects to WGS84) and GeoPackage (keeps CRS)
dg.to_geojson("detections.geojson")
dg.to_gpkg("detections.gpkg")
