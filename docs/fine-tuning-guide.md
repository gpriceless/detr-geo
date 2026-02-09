# Fine-Tuning Guide

Practical guide for fine-tuning RF-DETR on custom overhead imagery datasets using detr-geo.

---

## Why Fine-Tune?

RF-DETR ships with COCO-pretrained weights trained on ground-level photography. From overhead, objects look completely different: cars are rectangles, buildings are rooftops, and there is no horizon. The COCO model frequently misclassifies overhead vehicles as "motorcycle", "bicycle", or "boat" because it has never seen a car from above.

Fine-tuning on labeled overhead imagery solves this. The xView fine-tuned model achieves reliable vehicle detection at 0.3 m GSD where the COCO model fails almost entirely.

---

## Prerequisites

### Hardware

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU | RTX 2070 (8 GB VRAM) | RTX 3090 / A100 (24+ GB VRAM) |
| CPU | 4 cores | 8+ cores |
| RAM | 16 GB | 32 GB |
| Storage | 10 GB free | 20 GB free |

Training on CPU is technically possible but impractically slow (hours per epoch vs. minutes on GPU).

### Software

```bash
pip install detr-geo[all]
```

This installs the core library, RF-DETR, PyTorch, and visualization tools. Verify GPU access:

```python
import torch
print(torch.cuda.is_available())       # True for NVIDIA
print(torch.backends.mps.is_available()) # True for Apple Silicon
```

---

## Dataset Preparation

detr-geo converts GeoTIFF rasters with vector annotations into COCO-format training datasets. The pipeline handles CRS alignment, tiling, annotation clipping, and spatial splitting automatically.

### Input Requirements

1. **Raster**: A georeferenced GeoTIFF with at least 3 bands (RGB). Projected CRS (UTM) is ideal; geographic CRS (WGS84) works but produces tiles of varying ground size at different latitudes.

2. **Annotations**: A vector file (GeoJSON, GeoPackage, or Shapefile) with polygon or bounding box geometries and a class attribute. The CRS can differ from the raster -- detr-geo reprojects automatically.

### Example: xView Dataset

The [xView](http://xviewdataset.org/) dataset provides 1 million labeled objects across 60 classes in satellite imagery at 0.3 m GSD. For vehicle detection, the relevant classes are:

| xView Class ID | Label | detr-geo Class ID |
|----------------|-------|-------------------|
| 17 | Small Car | 0 (Car) |
| 18 | Bus | 3 (Bus) |
| 20 | Pickup Truck | 1 (Pickup) |
| 21 | Utility Truck | 2 (Truck) |
| 23 | Truck | 2 (Truck) |
| 25 | Engineering Vehicle | 4 (Other) |
| 29 | Passenger Vehicle | 0 (Car) |

### Running the Preparation Pipeline

```python
from detr_geo import prepare_training_dataset, SpatialSplitter

stats = prepare_training_dataset(
    raster_path="satellite_scene.tif",
    annotations_path="labels.geojson",
    output_dir="training_data/",
    tile_size=576,                  # Match model native resolution
    overlap_ratio=0.0,              # No overlap for training tiles
    min_annotation_area=100,        # Skip tiny annotations (<100 px^2)
    split_method="block",           # Spatial block splitting
    split_ratios=(0.8, 0.15, 0.05),
    seed=42,
)

print(stats)
# {'train_tiles': 1200, 'train_annotations': 8500,
#  'val_tiles': 225, 'val_annotations': 1600,
#  'test_tiles': 75, 'test_annotations': 530,
#  'buffer_tiles': 50}
```

### Output Structure

```
training_data/
  train/
    images/
      tile_0000.jpg
      tile_0001.jpg
      ...
    _annotations.coco.json
  valid/
    images/
      tile_0000.jpg
      ...
    _annotations.coco.json
  test/
    images/
      ...
    _annotations.coco.json
```

### Spatial Splitting

The `SpatialSplitter` prevents spatial data leakage. With block splitting, geographically contiguous regions are assigned entirely to one split, with buffer tiles discarded between blocks. This prevents the model from memorizing scene-specific patterns visible in nearby tiles.

```python
splitter = SpatialSplitter(
    method="block",           # Contiguous spatial blocks
    ratios=(0.8, 0.15, 0.05),
    buffer_tiles=1,           # 1 tile gap between blocks
    seed=42,
)
```

Random splitting is available but not recommended for geospatial data because neighboring tiles share visual context (same lighting, shadows, terrain), inflating validation metrics.

---

## Training Configuration

### Basic Training

```python
from detr_geo import DetrGeo, train

# Create an adapter (the adapter manages the RF-DETR model)
dg = DetrGeo(model_size="medium")

result = train(
    adapter=dg._adapter,
    dataset_dir="training_data/",
    epochs=50,
    batch_size=8,                           # Reduce to 4 for 8 GB VRAM
    augmentation_preset="satellite_default", # Rotation + flip + jitter
    augmentation_factor=2,                   # 2x data via augmentation
)
```

### VRAM and Batch Size

| GPU VRAM | Recommended batch_size | model_size |
|----------|----------------------|------------|
| 8 GB | 2--4 | medium |
| 12 GB | 4--6 | medium |
| 24 GB | 8--12 | medium or base |
| 48 GB+ | 16+ | large |

If you encounter CUDA out-of-memory errors, reduce `batch_size` first. If that is not enough, switch to a smaller `model_size`.

### Augmentation Presets

Overhead imagery has no canonical "up" direction, so vertical flip and 90-degree rotation are critical augmentations that standard pipelines omit.

| Preset | Best For | Jitter Intensity |
|--------|----------|-----------------|
| `satellite_default` | Satellite imagery (0.3--0.5 m GSD) | Low |
| `aerial_default` | Aerial photography (0.1--0.3 m GSD) | Medium |
| `drone_default` | Drone imagery (<0.1 m GSD) | High |

Higher jitter intensity compensates for the greater illumination variability in drone and aerial imagery compared to consistent satellite captures.

### Learning Rate

The default learning rate from RF-DETR works well for most cases. Override only if you see training instability (loss spikes or NaN):

```python
train(
    adapter=dg._adapter,
    dataset_dir="training_data/",
    epochs=50,
    learning_rate=1e-5,  # Lower than default for fine-tuning
)
```

---

## Monitoring Training

Training progress is logged to the console. Key metrics to watch:

- **Training loss**: Should decrease steadily. Spikes may indicate a learning rate that is too high.
- **Validation mAP**: The primary metric. Should increase over epochs and plateau.
- **EMA checkpoint**: RF-DETR maintains an exponential moving average of weights. The EMA checkpoint (`checkpoint_best_ema.pth`) typically outperforms the standard checkpoint.

### Typical Training Timeline (RTX 3090, medium model)

| Epoch Range | Expected mAP | Notes |
|-------------|-------------|-------|
| 1--5 | 0.05--0.15 | Model adjusting from COCO to overhead |
| 5--15 | 0.15--0.35 | Rapid improvement phase |
| 15--30 | 0.35--0.45 | Diminishing returns |
| 30--50 | 0.45--0.50 | Fine refinement, watch for overfitting |

Stop early if validation mAP plateaus for 10+ epochs.

---

## Using Fine-Tuned Weights

After training, the best checkpoint is saved as `checkpoint_best_ema.pth` in the output directory. Load it for inference:

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

dg.set_image("satellite_scene.tif")
detections = dg.detect_tiled(overlap=0.2, threshold=0.3)
dg.to_gpkg("vehicle_detections.gpkg")
```

The `custom_class_names` mapping must match the class IDs used during training. Without it, detections would be labeled with COCO class names (e.g., "person", "bicycle") because the model outputs integer class IDs that default to the COCO lookup table.

---

## VME as a Secondary Example

The VME (Vehicles in the Middle East) dataset is a smaller alternative to xView, with 3 vehicle classes (Car, Bus, Truck) from Maxar satellite imagery. See [docs/fine-tuning-vme.md](fine-tuning-vme.md) for a complete reproduction guide.

Key differences from xView:

| | xView | VME |
|-|-------|-----|
| Classes | 5 vehicle classes | 3 vehicle classes |
| Annotations | ~100K vehicles | ~30K vehicles |
| GSD | 0.3 m | 0.3--0.5 m |
| Source | DIUx / US DoD | Maxar |
| Training time | ~4 hours (RTX 3090) | ~3 hours (RTX 2070) |
| Best mAP | ~0.45 at epoch 13 | ~0.35 at epoch 30 |

---

## Troubleshooting

### CUDA out of memory

Reduce `batch_size`. If batch_size=1 still fails, use a smaller `model_size` (try "small" or "nano").

### Training loss is NaN

Lower the learning rate by 10x. Check that your annotations are valid (no degenerate polygons, correct CRS).

### Validation mAP stays at 0

Ensure the `valid/` directory contains tiles with annotations. Check that `class_mapping` during dataset preparation matches expectations.

### Model detects nothing after fine-tuning

Verify that `custom_class_names` in `DetrGeo()` matches the class IDs from training. A mismatch causes the model to output detections with wrong labels that get filtered out.

### Weights file not found

The `pretrain_weights` path must point to the actual `.pth` file. Common locations after training:
- `output/checkpoint_best_ema.pth` (EMA, typically best)
- `output/checkpoint_best.pth` (standard best)
- `output/checkpoint_latest.pth` (last epoch)
