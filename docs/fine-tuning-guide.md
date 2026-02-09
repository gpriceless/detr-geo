# Fine-Tuning Guide

Train RF-DETR to detect objects in overhead imagery. Out of the box, RF-DETR uses COCO weights trained on ground-level photography -- it has never seen a car from above. Fine-tuning on labeled satellite or aerial data transforms it into a reliable overhead detector.

This guide covers the full pipeline: dataset preparation, spatial splitting, training configuration, and deployment of custom weights. The xView vehicle detection model was trained using this exact workflow.

---

## Why Fine-Tune?

The COCO-pretrained model was trained on 80 object classes photographed from ground level. From an overhead perspective, objects look completely different:

- Cars become small rectangles instead of side-profile shapes
- Buildings appear as rooftops, not facades
- There is no horizon, no perspective convergence
- Shadows extend away from objects at consistent angles

The result: COCO RF-DETR labels overhead vehicles as "motorcycle", "skateboard", "boat", or simply misses them. Fine-tuning on even a few thousand labeled overhead examples solves this. The xView fine-tuned model reliably detects vehicles at 0.3m GSD where the COCO model fails almost entirely.

---

## Prerequisites

### Hardware

| Component | Minimum | Recommended |
|---|---|---|
| GPU | RTX 2070 (8 GB VRAM) | RTX 3090 / A100 (24+ GB) |
| CPU | 4 cores | 8+ cores |
| RAM | 16 GB | 32 GB |
| Storage | 10 GB free | 20 GB free |

Training on CPU is technically possible but impractically slow -- hours per epoch instead of minutes.

### Software

```bash
pip install detr-geo[all]
```

Verify GPU access:

```python
import torch
print(torch.cuda.is_available())         # NVIDIA GPU
print(torch.backends.mps.is_available())  # Apple Silicon
```

---

## Step 1: Prepare Your Dataset

detr-geo converts a GeoTIFF and vector annotations into a COCO-format training dataset. It handles CRS alignment, tiling, annotation clipping, and spatial splitting automatically.

### What You Need

1. **A georeferenced raster** (GeoTIFF). RGB, any CRS, 8-bit or 16-bit. Projected CRS (UTM) is ideal.

2. **Vector annotations** (GeoJSON, GeoPackage, or Shapefile) with polygon or bounding box geometries and a class attribute. The CRS can differ from the raster -- detr-geo reprojects automatically.

### Run the Pipeline

```python
from detr_geo import prepare_training_dataset

stats = prepare_training_dataset(
    raster_path="satellite_scene.tif",
    annotations_path="labels.geojson",
    output_dir="training_data/",
    tile_size=576,                  # Match model native resolution
    overlap_ratio=0.0,              # No overlap for training tiles
    min_annotation_area=100,        # Skip tiny annotations (<100 px^2)
    split_method="block",           # Spatial splitting (see below)
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

### Key Parameters

| Parameter | What it does |
|---|---|
| `tile_size` | Tile edge length in pixels. Match the model's native resolution (576 for medium) |
| `overlap_ratio` | Overlap between training tiles. 0.0 is standard. Non-zero creates more tiles but improves boundary object coverage |
| `min_annotation_area` | Discard annotations smaller than this many square pixels. Removes noise |
| `max_background_per_annotated` | Ratio of empty tiles to annotated tiles. Controls class balance |
| `class_mapping` | Maps class attribute values to integer IDs. Auto-generated if not provided |

---

## Step 2: Understand Spatial Splitting

This is the most important difference between geospatial ML and standard computer vision. If you skip this section, your model will appear to work well but will fail on new data.

### The Problem with Random Splitting

In standard CV, you randomly split images into train/val/test. In geospatial data, neighboring tiles share visual context: same lighting angle, same shadow direction, same terrain texture, same atmospheric conditions. If tile A is in training and neighboring tile B is in validation, the model can partially memorize the scene rather than learning general features.

This inflates validation metrics. Your mAP looks great, but the model fails on imagery from a different date or location.

### Block Splitting

detr-geo's `SpatialSplitter` assigns contiguous spatial blocks to each split, with buffer tiles discarded between blocks:

```python
from detr_geo import SpatialSplitter

splitter = SpatialSplitter(
    method="block",           # Contiguous spatial blocks
    ratios=(0.8, 0.15, 0.05),
    buffer_tiles=1,           # 1-tile gap between blocks
    seed=42,
)
```

The buffer zone between blocks prevents spatial autocorrelation from leaking across splits. The validation set truly tests generalization.

### When Random Is Acceptable

Random splitting is available (`split_method="random"`) and appropriate when:
- You have multiple independent scenes (different dates, different locations)
- Each scene is its own raster, and you split at the scene level
- You understand and accept the inflated validation metrics

---

## Step 3: Train

### Basic Training

```python
from detr_geo import DetrGeo, train

dg = DetrGeo(model_size="medium")

result = train(
    adapter=dg._adapter,
    dataset_dir="training_data/",
    epochs=50,
    batch_size=8,
    augmentation_preset="satellite_default",
    augmentation_factor=2,  # 2x data via augmentation
)
```

### VRAM and Batch Size

| GPU VRAM | batch_size | model_size |
|---|---|---|
| 8 GB | 2--4 | medium |
| 12 GB | 4--6 | medium |
| 24 GB | 8--12 | medium or base |
| 48 GB+ | 16+ | large |

If you hit CUDA out-of-memory errors, reduce `batch_size` first. If batch_size=1 still fails, use a smaller `model_size`.

### Augmentation Presets

Overhead imagery has no canonical "up" direction -- a car looks the same from north or south. Standard augmentation pipelines omit vertical flip and 90-degree rotation because they break ground-level photos. For overhead imagery, these augmentations are critical.

| Preset | Best For | Color Jitter |
|---|---|---|
| `"satellite_default"` | Satellite (0.3--0.5m GSD) | Low -- satellite imagery has consistent illumination |
| `"aerial_default"` | Aerial photography (0.1--0.3m GSD) | Medium |
| `"drone_default"` | Drone (<0.1m GSD) | High -- drone lighting varies widely |

All presets enable: 90-degree rotation, horizontal flip, vertical flip, brightness/contrast/saturation jitter.

### Learning Rate

The RF-DETR default works well for most fine-tuning. Override only if training is unstable:

```python
train(
    adapter=dg._adapter,
    dataset_dir="training_data/",
    epochs=50,
    learning_rate=1e-5,  # Lower for fine-tuning stability
)
```

### Resuming Training

If training is interrupted:

```python
train(
    adapter=dg._adapter,
    dataset_dir="training_data/",
    epochs=50,
    resume="output/checkpoint_latest.pth",
)
```

---

## Step 4: Monitor Training

Key metrics to watch during training:

| Metric | What to look for |
|---|---|
| Training loss | Steady decrease. Spikes = learning rate too high |
| Validation mAP | Primary metric. Should increase and plateau |
| EMA checkpoint | RF-DETR maintains exponential moving average weights. EMA typically outperforms standard |

### Typical Training Timeline (RTX 3090, medium model)

| Epoch | Expected mAP | Phase |
|---|---|---|
| 1--5 | 0.05--0.15 | Adjusting from COCO to overhead |
| 5--15 | 0.15--0.35 | Rapid improvement |
| 15--30 | 0.35--0.45 | Diminishing returns |
| 30--50 | 0.45--0.50 | Fine refinement, watch for overfitting |

Stop early if validation mAP has not improved for 10+ epochs.

---

## Step 5: Deploy Fine-Tuned Weights

After training, the best checkpoint is saved as `checkpoint_best_ema.pth`. Load it for inference:

```python
from detr_geo import DetrGeo

dg = DetrGeo(
    model_size="medium",
    pretrain_weights="output/checkpoint_best_ema.pth",
    custom_class_names={
        0: "Car",
        1: "Pickup Truck",
        2: "Truck",
        3: "Bus",
        4: "Other Vehicle",
    },
)

dg.set_image("new_satellite_scene.tif")
detections = dg.detect_tiled(overlap=0.2, threshold=0.3)
dg.to_gpkg("vehicle_detections.gpkg")
```

The `custom_class_names` dict must match the class IDs used during training. Without it, detections are labeled with COCO names (e.g., "person", "bicycle") because the model outputs integer IDs that default to the COCO lookup table.

---

## xView Training Example

The [xView dataset](http://xviewdataset.org/) provides ~1 million labeled objects across 60 classes in satellite imagery at 0.3m GSD. The detr-geo xView model was trained on 5 vehicle classes:

| xView Class | detr-geo Class | ID |
|---|---|---|
| Small Car, Passenger Vehicle | Car | 0 |
| Pickup Truck | Pickup Truck | 1 |
| Utility Truck, Truck | Truck | 2 |
| Bus | Bus | 3 |
| Engineering Vehicle | Other Vehicle | 4 |

Training details:
- Model: RF-DETR Medium (576px)
- Dataset: ~100K vehicle annotations from xView
- Epochs: 30 (best mAP at epoch 13)
- Batch size: 8 on RTX 3090
- Augmentation: `satellite_default` with 2x factor
- Training time: ~4 hours
- Best mAP: ~0.45

See [docs/fine-tuning-vme.md](fine-tuning-vme.md) for a smaller-scale reproduction guide using the VME (Vehicles in the Middle East) dataset.

---

## Active Learning Loop

Fine-tuning does not have to be a one-shot process. detr-geo supports an active learning workflow:

1. **Detect** on new imagery with your current model
2. **Export** detections to COCO format for review
3. **Correct** annotations in your labeling tool
4. **Retrain** on the expanded dataset

```python
from detr_geo import detections_to_coco

# Export detections as COCO annotations for review
detections_to_coco(
    dg.detections,
    output_path="review/annotations.json",
    image_path="scene.tif",
)
```

Each iteration improves the model on your specific imagery domain.

---

## Troubleshooting

### CUDA out of memory

Reduce `batch_size`. If batch_size=1 still fails, use a smaller `model_size` ("small" or "nano").

### Training loss is NaN

Lower the learning rate by 10x. Check that annotations are valid (no degenerate polygons, correct CRS alignment).

### Validation mAP stays at 0

Ensure `valid/` contains tiles with annotations. Verify `class_mapping` matches your annotation schema.

### Model detects nothing after fine-tuning

Check that `custom_class_names` matches the class IDs from training. A mismatch causes detections to be labeled with wrong names and filtered out.

### Weights file not found

`pretrain_weights` must point to the actual `.pth` file. Common locations:
- `output/checkpoint_best_ema.pth` -- EMA weights (typically best)
- `output/checkpoint_best.pth` -- standard best
- `output/checkpoint_latest.pth` -- last epoch
