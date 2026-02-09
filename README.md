# detr-geo

[![CI](https://github.com/gpriceless/detr-geo/actions/workflows/ci.yml/badge.svg)](https://github.com/gpriceless/detr-geo/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gpriceless/detr-geo/blob/main/notebooks/quickstart.ipynb)

**Object detection for satellite and aerial imagery.** Feed in a GeoTIFF, get back georeferenced vector data -- GeoJSON, GeoPackage, or Shapefile -- with full CRS, coordinates, and attribute tables ready for QGIS, ArcGIS, or PostGIS.

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

Inspired by [samgeo](https://github.com/opengeos/segment-geospatial), but for bounding-box detection rather than segmentation.

---

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

---

## xView Fine-Tuned Model

The COCO-pretrained model was trained on ground-level photography. It has never seen a car from above, and it shows -- overhead vehicles get labeled as "motorcycle", "skateboard", or "boat".

The xView fine-tuned model fixes this. Trained on the [xView dataset](http://xviewdataset.org/) (satellite imagery at 0.3m GSD), it detects 5 overhead vehicle classes:

| Class | Examples |
|---|---|
| Car | Sedans, SUVs, hatchbacks |
| Pickup Truck | Pickup trucks, utility pickups |
| Truck | Semi trucks, cargo trucks, tankers |
| Bus | Transit buses, school buses, coaches |
| Other Vehicle | Construction equipment, engineering vehicles |

```python
from detr_geo import DetrGeo

dg = DetrGeo(
    model_size="medium",
    pretrain_weights="checkpoints/checkpoint_best_ema.pth",
    custom_class_names={
        0: "Car",
        1: "Pickup Truck",
        2: "Truck",
        3: "Bus",
        4: "Other Vehicle",
    },
)

dg.set_image("satellite_scene.tif")
detections = dg.detect_tiled(overlap=0.2, threshold=0.3)

# Per-class counts
print(detections["class_name"].value_counts())

# Export for QGIS
dg.to_gpkg("vehicle_detections.gpkg")
```

Download the xView fine-tuned weights (optional):

```bash
huggingface-cli download gpriceless/detr-geo-xview \
    checkpoint_best_ema.pth --local-dir checkpoints/
```

---

## Installation

### Prerequisites

detr-geo requires GDAL and rasterio. On most systems:

```bash
# Ubuntu/Debian
sudo apt install gdal-bin libgdal-dev

# macOS (with Homebrew)
brew install gdal

# conda (recommended for geospatial Python)
conda install -c conda-forge rasterio geopandas
```

### Install detr-geo

```bash
pip install detr-geo          # Core geospatial tools only (no model, no viz)
pip install detr-geo[rfdetr]  # With RF-DETR model + PyTorch
pip install detr-geo[viz]     # With leafmap + matplotlib visualization
pip install detr-geo[all]     # Everything
```

**Core** installs rasterio, geopandas, pyproj, shapely, and numpy. This is sufficient for processing pre-computed detection results without running inference.

**[rfdetr]** adds the RF-DETR model, PyTorch, and supervision. Downloads ~2 GB of model weights on first use.

**[viz]** adds leafmap (interactive maps) and matplotlib (static plots).

### GPU Support

RF-DETR runs on CPU but is 10-50x faster on GPU:

```bash
# CUDA (NVIDIA)
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install detr-geo[all]

# MPS (Apple Silicon) -- PyTorch detects automatically
pip install detr-geo[all]
```

---

## Core Workflow

### 1. Load imagery

```python
dg = DetrGeo(model_size="medium")
dg.set_image("orthomosaic.tif")
```

detr-geo reads the CRS and affine transform from your raster automatically. It supports GeoTIFF, COG, and any rasterio-readable format.

### 2. Detect objects

For small images (fits in memory):

```python
detections = dg.detect(threshold=0.4)
```

For large rasters (orthomosaics, satellite scenes):

```python
detections = dg.detect_tiled(
    overlap=0.2,           # 20% overlap prevents boundary artifacts
    nms_threshold=0.5,     # deduplicate across tiles
    nodata_threshold=0.5,  # skip tiles that are >50% empty
)
```

### 3. Export georeferenced results

```python
dg.to_gpkg("detections.gpkg")         # GeoPackage -- recommended
dg.to_geojson("detections.geojson")   # GeoJSON (auto-reprojects to WGS84)
dg.to_shp("detections.shp")           # Shapefile (legacy)
```

Or work with the GeoDataFrame directly:

```python
gdf = dg.detections
cars = gdf[gdf["class_name"] == "Car"]
confident = gdf[gdf["confidence"] > 0.7]
```

### 4. Visualize

```python
# Static matplotlib plot
dg.show_detections(figsize=(15, 12), save_path="output.png")

# Interactive leafmap (Jupyter)
m = dg.show_map(basemap="SATELLITE")
```

---

## Band Selection

RF-DETR expects 3-channel RGB input. Different sensors store RGB in different bands. detr-geo maps them automatically:

```python
dg.set_image("naip.tif", bands="rgb")               # Default: bands 1-2-3
dg.set_image("sentinel2.tif", bands="sentinel2_rgb") # Sentinel-2: bands 4-3-2
dg.set_image("worldview.tif", bands="worldview_rgb") # WorldView: bands 5-3-2
dg.set_image("custom.tif", bands=(4, 3, 2))          # Any 3-band combo (1-indexed)
```

---

## Fine-Tuning on Custom Data

Train RF-DETR on your own overhead imagery. detr-geo handles the full pipeline from GeoTIFF + vector annotations to trained model:

```python
from detr_geo import prepare_training_dataset, train, DetrGeo

# 1. Tile raster, align CRS, clip annotations, split spatially
prepare_training_dataset(
    raster_path="ortho.tif",
    annotations_path="labels.geojson",
    output_dir="training_data/",
    tile_size=576,
    split_method="block",       # spatial splitting prevents leakage
    split_ratios=(0.8, 0.15, 0.05),
)

# 2. Train with geospatial augmentations (rotation, flip -- no "up" in overhead)
dg = DetrGeo(model_size="medium")
train(
    adapter=dg._adapter,
    dataset_dir="training_data/",
    epochs=50,
    batch_size=8,
    augmentation_preset="satellite_default",
)

# 3. Run inference with your trained weights
dg = DetrGeo(
    model_size="medium",
    pretrain_weights="output/checkpoint_best_ema.pth",
    custom_class_names={0: "Building", 1: "Road", 2: "Tree"},
)
```

See the [Fine-Tuning Guide](docs/fine-tuning-guide.md) for hardware requirements, monitoring, xView training details, and troubleshooting.

---

## Model Sizes

| Size | Resolution | Parameters | Best For |
|------|-----------|------------|----------|
| nano | 384px | ~15M | CPU inference, quick prototyping |
| small | 512px | ~22M | Balanced speed and accuracy |
| **medium** | **576px** | **~25M** | **Default. Best accuracy/speed tradeoff** |
| base | 560px | ~29M | Higher accuracy, more VRAM |
| large | 704px | ~30M | Maximum accuracy, GPU recommended |

---

## Real-World Use Cases

**Parking lot occupancy** -- Count vehicles across a commercial property from a single satellite capture. Export per-class counts (cars, trucks, buses) as a GeoJSON layer for the facilities team.

**Construction site monitoring** -- Detect heavy equipment in weekly drone surveys. Compare detections across dates to track mobilization and demobilization.

**Fleet asset tracking** -- Process satellite imagery of a logistics yard to locate trucks and trailers. Output GeoPackage layers with confidence scores for the dispatch team.

**Urban planning** -- Tile a city-wide orthomosaic and detect all vehicles. Aggregate detections by census tract for traffic density analysis.

**Disaster response** -- Rapidly scan post-event imagery for vehicles (potential rescue targets) in flood zones or collapsed structures.

---

## Known Limitations

- **Memory**: `detect()` loads the full raster. Use `detect_tiled()` for rasters larger than ~5000x5000 pixels.
- **Geographic CRS tiling**: Tiles in EPSG:4326 vary in ground size by latitude. Use a projected CRS for best results.
- **Per-tile normalization**: In tiled mode, each tile is normalized independently. For consistent 16-bit results, pre-compute stretch parameters with `detr_geo.io.compute_scene_stretch_params`.
- **Batch inference**: Tiles are processed sequentially on GPU. True tensor batching is planned.

---

## Output Formats

| Format | Method | Notes |
|--------|--------|-------|
| GeoJSON | `to_geojson()` | Auto-reprojects to WGS84 per spec |
| GeoPackage | `to_gpkg()` | Recommended. Preserves CRS, supports layers |
| Shapefile | `to_shp()` | Legacy. 10-char field names, 2 GB limit |
| GeoDataFrame | `.detections` | In-memory geopandas for further analysis |

---

## License

**Code**: MIT

**xView fine-tuned weights**: CC BY-NC-SA 4.0 (following the xView dataset license). COCO-pretrained weights are unrestricted.

---

## Links

- [API Reference](docs/api-reference.md) -- full method documentation
- [Geospatial Guide](docs/geospatial-guide.md) -- tiling, CRS, bands, 16-bit, nodata
- [Fine-Tuning Guide](docs/fine-tuning-guide.md) -- train on your own overhead imagery
- [Examples](examples/) -- runnable scripts for common workflows
- [RF-DETR](https://github.com/roboflow/rf-detr) -- the underlying detection architecture
- [xView Dataset](http://xviewdataset.org/) -- satellite imagery used for fine-tuning
- [samgeo](https://github.com/opengeos/segment-geospatial) -- the project that inspired this one

## Acknowledgments

- [RF-DETR](https://github.com/roboflow/rf-detr) by Roboflow -- transformer-based object detection
- [xView](http://xviewdataset.org/) by DIUx -- satellite annotations for fine-tuning
- [samgeo](https://github.com/opengeos/segment-geospatial) by Qiusheng Wu -- the geospatial ML workflow model
