# detr-geo

[![CI](https://github.com/gpriceless/detr-geo/actions/workflows/ci.yml/badge.svg)](https://github.com/gpriceless/detr-geo/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**Run object detection on geospatial imagery and get georeferenced vector results.** detr-geo wraps the [RF-DETR](https://github.com/roboflow/rf-detr) object detection model with a geospatial pipeline: it reads GeoTIFFs, handles tiling for large rasters, performs detection, and exports results as GeoJSON, GeoPackage, or Shapefile with proper CRS and coordinates.

Includes xView fine-tuned weights for overhead vehicle detection -- distinguishing Car, Pickup, Truck, Bus, and Other vehicles from satellite imagery at 0.3 m GSD.

Inspired by [samgeo](https://github.com/opengeos/segment-geospatial), but for bounding-box detection rather than segmentation.

<!-- Comparison images: COCO baseline vs xView fine-tuned on the same parking lot
     See scripts/output/ for generated comparison images. -->

## Quick Start

```python
from detr_geo import DetrGeo

dg = DetrGeo(model_size="medium")
dg.set_image("path/to/geotiff.tif")
detections = dg.detect()
dg.to_geojson("detections.geojson")
```

## Using the xView Fine-Tuned Model

The COCO-pretrained model was trained on ground-level photos and struggles with overhead imagery. The xView fine-tuned model detects vehicles from satellite and aerial perspectives:

```python
from detr_geo import DetrGeo

dg = DetrGeo(
    model_size="medium",
    pretrain_weights="checkpoints/checkpoint_best_ema.pth",
    custom_class_names={
        0: "Car",
        1: "Pickup",
        2: "Truck",
        3: "Bus",
        4: "Other",
    },
)

dg.set_image("satellite_image.tif")
detections = dg.detect_tiled(overlap=0.2, threshold=0.3)
dg.to_gpkg("vehicle_detections.gpkg")
```

<!-- Download xView fine-tuned weights:
     huggingface-cli download gpriceless/detr-geo-xview \
         checkpoint_best_ema.pth - -local-dir checkpoints/ -->

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

**Core** installs rasterio, geopandas, pyproj, shapely, and numpy. This is sufficient if you want to process pre-computed detection results without running inference.

**[rfdetr]** adds the RF-DETR model, PyTorch, and supervision. Requires ~2 GB disk space for model weights on first use.

**[viz]** adds leafmap (interactive maps) and matplotlib (static plots).

### GPU Support

RF-DETR runs on CPU but is significantly faster on GPU:

```bash
# CUDA (NVIDIA)
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install detr-geo[all]

# MPS (Apple Silicon) -- PyTorch detects automatically
pip install detr-geo[all]
```

## Features

- **Tiled detection** with configurable overlap and cross-tile NMS for rasters of any size
- **Fine-tuning pipeline** -- prepare datasets, train, and run inference with custom weights
- **Band presets** for NAIP, Sentinel-2, WorldView, and custom multispectral sensors
- **16-bit imagery** support with percentile stretching
- **Nodata handling** -- skip empty tiles, fill partial tiles
- **Export** to GeoJSON (auto-reproject to WGS84), GeoPackage, and Shapefile
- **Visualization** via interactive leafmap and static matplotlib
- **CRS-aware** -- reads and preserves coordinate reference systems end-to-end
- **Spatial dataset splitting** to prevent data leakage during training

## Tiled Detection

Large rasters are split into overlapping tiles, each processed independently, then merged with class-aware NMS:

```python
dg = DetrGeo(model_size="medium")
dg.set_image("large_orthomosaic.tif")

detections = dg.detect_tiled(
    overlap=0.2,           # 20% tile overlap
    nms_threshold=0.5,     # IoU threshold for deduplication
    nodata_threshold=0.5,  # Skip tiles that are >50% nodata
)

dg.to_gpkg("detections.gpkg")
```

## Fine-Tuning

Prepare a custom dataset and fine-tune RF-DETR on your own geospatial data:

```python
from detr_geo import prepare_training_dataset, SpatialSplitter, train, DetrGeo

# Prepare COCO-format dataset from GeoTIFF + GeoJSON
prepare_training_dataset(
    raster_path="ortho.tif",
    annotations_path="labels.geojson",
    output_dir="training_data/",
    tile_size=576,
    split_method="block",
    split_ratios=(0.8, 0.15, 0.05),
)

# Fine-tune (requires GPU)
dg = DetrGeo(model_size="medium")
train(
    adapter=dg._adapter,
    dataset_dir="training_data/",
    epochs=50,
    batch_size=8,
    augmentation_preset="satellite_default",
)
```

See the [Fine-Tuning Guide](docs/fine-tuning-guide.md) for a complete walkthrough including hardware requirements, monitoring, and troubleshooting.

## Model Sizes

| Size | Resolution | Parameters | Best For |
|------|-----------|------------|----------|
| nano | 384px | ~15M | CPU inference, quick tests |
| small | 512px | ~22M | Balanced speed/accuracy |
| medium | 576px | ~25M | Default. Good accuracy |
| base | 560px | ~29M | Higher accuracy |
| large | 704px | ~30M | Maximum accuracy, GPU recommended |

## Output Formats

| Format | Method | Notes |
|--------|--------|-------|
| GeoJSON | `to_geojson()` | Auto-reprojects to WGS84 per spec |
| GeoPackage | `to_gpkg()` | Recommended. Preserves CRS, supports layers |
| Shapefile | `to_shp()` | Legacy. 10-char field names, 2 GB limit |
| GeoDataFrame | `.detections` | In-memory geopandas object for further analysis |

## Band Selection

detr-geo maps sensor bands to RGB for the model:

```python
dg.set_image("naip.tif", bands="rgb")              # Default: bands 1-2-3
dg.set_image("sentinel2.tif", bands="sentinel2_rgb") # Sentinel-2: bands 4-3-2
dg.set_image("worldview.tif", bands="worldview_rgb") # WorldView: bands 5-3-2
dg.set_image("multispectral.tif", bands=(4, 3, 2))   # Custom (1-indexed)
```

## Visualization

```python
# Interactive leafmap (Jupyter/browser)
dg.show_map(basemap="SATELLITE")

# Static matplotlib plot (saved to file)
dg.show_detections(figsize=(15, 12))
```

## Known Limitations

- **Memory**: `detect()` loads the full raster into memory. Use `detect_tiled()` for rasters larger than ~5000x5000 pixels.
- **Cloud-native**: No direct COG/HTTP or STAC support yet. Download rasters locally first.
- **Geographic CRS tiling**: Tiling in geographic CRS (EPSG:4326) produces tiles of varying ground size at different latitudes. Use projected CRS for best results.
- **Batch inference**: Tiles are processed sequentially on GPU. True tensor batching is planned.
- **Per-tile normalization**: In tiled mode, each tile is normalized independently. For consistent results on 16-bit imagery, pre-compute scene stretch parameters using `detr_geo.io.compute_scene_stretch_params`.

## License

**Code**: MIT

**xView fine-tuned weights**: CC BY-NC-SA 4.0 (following the xView dataset license). The weights are derived from the xView dataset and inherit its non-commercial license. COCO-pretrained weights are unrestricted.

## Links

- [RF-DETR](https://github.com/roboflow/rf-detr) -- the underlying detection model
- [samgeo](https://github.com/opengeos/segment-geospatial) -- the inspiration for this project
- [xView Dataset](http://xviewdataset.org/) -- satellite imagery dataset used for fine-tuning
- [Geospatial Guide](docs/geospatial-guide.md) -- detailed guide for geospatial analysts
- [API Reference](docs/api-reference.md) -- full API documentation
- [Fine-Tuning Guide](docs/fine-tuning-guide.md) -- train on custom overhead imagery
- [VME Fine-Tuning](docs/fine-tuning-vme.md) -- reproduce the VME training run
- [Examples](examples/) -- runnable example scripts

## Acknowledgments

- [RF-DETR](https://github.com/roboflow/rf-detr) by Roboflow -- the transformer-based detection architecture
- [xView](http://xviewdataset.org/) by DIUx -- satellite imagery and annotations for fine-tuning
- [samgeo](https://github.com/opengeos/segment-geospatial) by Qiusheng Wu -- the geospatial ML workflow that inspired this project
