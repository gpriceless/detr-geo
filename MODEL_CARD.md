---
language: en
license: cc-by-nc-sa-4.0
tags:
  - object-detection
  - satellite-imagery
  - geospatial
  - rf-detr
  - xview
  - vehicle-detection
datasets:
  - xview
metrics:
  - map
base_model: roboflow/rf-detr-medium
---

# detr-geo-xview

RF-DETR Medium fine-tuned on the [xView dataset](http://xviewdataset.org/) for overhead vehicle detection in satellite imagery. Part of the [detr-geo](https://github.com/gpriceless/detr-geo) library.

## Model Description

The COCO-pretrained RF-DETR base model was trained on ground-level photography. It performs poorly on overhead satellite imagery — vehicles get misclassified as "motorcycle", "skateboard", or "boat". This fine-tuned model fixes that.

**Key facts:**
- Base model: RF-DETR Medium (576×576 input resolution)
- Fine-tuned on: xView satellite imagery at 0.3m GSD
- Task: Object detection — 5 overhead vehicle classes
- License: CC BY-NC-SA 4.0 (weights only; follows xView dataset license)

## Intended Use

Detecting vehicles in satellite and aerial imagery. Designed for use with the [detr-geo](https://github.com/gpriceless/detr-geo) library, which handles tiling, cross-tile NMS deduplication, CRS-aware output, and GeoPackage/GeoJSON export.

**Intended uses:**
- Counting vehicles in parking lots from commercial satellite imagery
- Traffic density analysis from overhead imagery
- Vehicle detection in disaster response workflows
- Urban planning and land use studies

**Not intended for:**
- Ground-level or drone photography (use base RF-DETR or retrain)
- Military targeting or surveillance
- Any use prohibited by the xView dataset license (CC BY-NC-SA 4.0)

## Training Data

Fine-tuned on the [xView Detection Challenge](http://xviewdataset.org/) dataset (Lam et al., 2018).

- **Source**: WorldView-3 satellite imagery, 0.3m GSD
- **Region**: Worldwide (diverse geographies)
- **Original classes**: 60 fine-grained object categories
- **Remapped to 5 vehicle classes** for this model (see class mapping below)
- **Split**: ~32,000 training tiles, ~4,000 validation tiles (576×576px, 20% overlap)

**Class mapping from xView:**

| Model Class | xView Source Classes |
|---|---|
| Car | Passenger Vehicle, Small Car, Van |
| Pickup | Pickup Truck |
| Truck | Truck, Cargo Truck, Truck Tractor, Semi, Truck Tractor w/ Box Trailer |
| Bus | Bus, Passenger Car, Trolley Bus |
| Other | Engineering Vehicle, Motor Vehicle, Reach Stacker, other vehicle types |

## Classes

| ID | Class | Examples |
|---|---|---|
| 0 | Car | Sedans, SUVs, hatchbacks |
| 1 | Pickup | Pickup trucks, utility pickups |
| 2 | Truck | Semi trucks, cargo trucks, tankers |
| 3 | Bus | Transit buses, school buses, coaches |
| 4 | Other | Construction equipment, engineering vehicles |

## Training Hyperparameters

| Parameter | Value |
|---|---|
| Base model | RF-DETR Medium |
| Input resolution | 576×576 px |
| Epochs | 27 (early stopping; best checkpoint at epoch 13) |
| Batch size | 8 |
| Gradient accumulation | 2 (effective batch size 16) |
| Learning rate | 1e-5 |
| Optimizer | AdamW (RF-DETR default) |
| Augmentation preset | satellite_default |
| Random seed | 42 |
| Hardware | A100 PCIe 80GB |

Training stopped at epoch 27 when validation mAP had not improved since epoch 13. The EMA (Exponential Moving Average) checkpoint from epoch 13 is the recommended checkpoint.

## Evaluation Results

Evaluated on xView validation split (COCO-style evaluation).

**Best checkpoint (epoch 13 EMA):**

| Metric | Value |
|---|---|
| mAP@0.50:0.95 (all) | 0.208 |
| mAP@0.50 (all) | 0.449 |
| mAP@0.75 (all) | 0.160 |
| AP — small objects | 0.176 |
| AP — medium objects | 0.321 |
| AP — large objects | 0.030 |
| AR@500 (all) | 0.408 |

**Notes on metrics:**
- "Small" objects here are vehicles in the COCO area sense (<32² px); most satellite vehicles fall in this category.
- AP for large objects is low (0.030) as large vehicles (semi trucks, buses) are rare in xView.
- mAP@0.50 of 0.449 is practically meaningful for tiled satellite inference with NMS deduplication.

## Usage

Install detr-geo:

```bash
pip install detr-geo
```

Download weights:

```bash
huggingface-cli download gpriceless/detr-geo-xview \
    checkpoint_best_ema.pth --local-dir checkpoints/
```

Run inference:

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

# Per-class counts
print(detections["class_name"].value_counts())

# Export to GeoPackage (with spatial reference)
dg.to_gpkg("vehicle_detections.gpkg")
```

> **Important:** Always pass `custom_class_names` when loading fine-tuned weights. Without it, class IDs map to COCO labels ("person", "motorcycle", etc.) instead of the correct vehicle classes.

## Limitations

- **GSD sensitivity**: Optimized for 0.3m GSD (WorldView-3). Performance degrades at >1.0m GSD. The detr-geo library warns at >1.0m and errors at >5.0m.
- **Nadir angle**: Trained on near-nadir imagery. Off-nadir captures (>30°) may reduce accuracy.
- **Occlusion**: Vehicles under tree cover or in shadows are frequently missed.
- **Class imbalance**: Large vehicles (Truck, Bus) are underrepresented in xView; AP for these classes is lower than for Car.
- **No nighttime support**: Not trained on SAR or nighttime imagery.
- **Geographic bias**: xView covers diverse regions but performance may vary in areas underrepresented in the training set.

## License

The model **weights** are licensed under **CC BY-NC-SA 4.0**, following the xView dataset license.

- You may use, share, and adapt these weights for non-commercial purposes.
- Derivative models must carry the same CC BY-NC-SA 4.0 license.
- Commercial use is not permitted.

The [detr-geo](https://github.com/gpriceless/detr-geo) **code** is separately licensed under MIT.

## Citation

If you use this model, please cite the xView dataset:

```bibtex
@article{lam2018xview,
  title={xView: Objects in Context in Overhead Imagery},
  author={Lam, Darius and Kuzma, Richard and McGee, Kevin and Dooley, Samuel and Laielli, Michael and Klaric, Matthew and Bulatov, Yaroslav and McCord, Brendan},
  journal={arXiv preprint arXiv:1802.07856},
  year={2018}
}
```

And RF-DETR:

```bibtex
@software{rfdetr2024,
  title={RF-DETR: Real-Time Object Detection with Transformers},
  author={Roboflow},
  year={2024},
  url={https://github.com/roboflow/rf-detr}
}
```

## Acknowledgements

- [xView Dataset](http://xviewdataset.org/) by DIUx — training data
- [RF-DETR](https://github.com/roboflow/rf-detr) by Roboflow — base model architecture
