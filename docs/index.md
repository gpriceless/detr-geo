# detr-geo

[![CI](https://github.com/gpriceless/detr-geo/actions/workflows/ci.yml/badge.svg)](https://github.com/gpriceless/detr-geo/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![PyPI](https://img.shields.io/pypi/v/detr-geo.svg)](https://pypi.org/project/detr-geo/)

**Object detection for satellite and aerial imagery.** Feed in a GeoTIFF, get back georeferenced vector data — GeoJSON, GeoPackage, or Shapefile — with full CRS, coordinates, and attribute tables ready for QGIS, ArcGIS, or PostGIS.

detr-geo wraps [RF-DETR](https://github.com/roboflow/rf-detr) and adds everything a geospatial workflow needs: automatic tiling with cross-tile NMS deduplication, multispectral band mapping, 16-bit imagery normalization, nodata handling, and spatial-aware training dataset preparation. Includes a training pipeline for fine-tuning on your own data, with xView-trained vehicle detection weights available as an example.

## Quick Start

```python
from detr_geo import DetrGeo

dg = DetrGeo(model_size="medium")
dg.set_image("parking_lot.tif")
detections = dg.detect_tiled(overlap=0.2, threshold=0.3)
dg.to_gpkg("vehicles.gpkg")

print(f"{len(detections)} vehicles found")
print(detections[["class_name", "confidence", "geometry"]].head())
```

## Installation

```bash
pip install detr-geo
```

With RF-DETR inference support:

```bash
pip install "detr-geo[rfdetr]"
```

All optional dependencies:

```bash
pip install "detr-geo[all]"
```

## What detr-geo does that raw RF-DETR cannot

| Capability | RF-DETR alone | detr-geo |
|---|---|---|
| **Input** | Single PIL image, <= 704px | GeoTIFF of any size, any CRS, 8/16-bit |
| **Output** | Pixel bounding boxes | Georeferenced polygons with CRS |
| **Large imagery** | Fails or requires manual slicing | Automatic tiling, overlap, cross-tile NMS |
| **Multispectral** | RGB only | Band presets for NAIP, Sentinel-2, WorldView |
| **Nodata** | Not handled | Skip empty tiles, fill partial tiles |
| **Export** | NumPy arrays | GeoJSON, GeoPackage, Shapefile |
| **Training** | Generic COCO format | Spatial splitting to prevent geospatial data leakage |
| **Overhead imagery** | Trained on ground-level photos | Fine-tuning pipeline + example xView vehicle weights |

## xView Fine-Tuned Model

The COCO-pretrained model was trained on ground-level photography and has never seen vehicles from above. The xView fine-tuned model fixes this — trained on the [xView dataset](http://xviewdataset.org/) (satellite imagery at 0.3m GSD), it detects 5 overhead vehicle classes: Car, Pickup Truck, Truck, Bus, and Other Vehicle.

```python
from detr_geo import DetrGeo

dg = DetrGeo(
    model_size="medium",
    pretrain_weights="checkpoints/checkpoint_best_ema.pth",
    custom_class_names={0: "Car", 1: "Pickup Truck", 2: "Truck", 3: "Bus", 4: "Other Vehicle"},
)
dg.set_image("satellite_scene.tif")
detections = dg.detect_tiled(overlap=0.2, threshold=0.3)
dg.to_gpkg("vehicle_detections.gpkg")
```

Download the xView weights (optional):

```bash
huggingface-cli download gpriceless/detr-geo-xview \
    checkpoint_best_ema.pth --local-dir checkpoints/
```

## Guides

- [Geospatial Guide](geospatial-guide.md) — tiling, CRS, band mapping, nodata handling
- [Fine-Tuning Guide](fine-tuning-guide.md) — training your own model on aerial imagery
- [Testing Guide](testing-guide.md) — running and writing tests
- [API Reference](api-reference.md) — full Python API docs
