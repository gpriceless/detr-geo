# Fine-Tuning RF-DETR on VME Dataset for Overhead Vehicle Detection

## Overview

This guide walks through fine-tuning an RF-DETR Medium model on the VME
(Vehicles in the Middle East) satellite imagery dataset to enable accurate
vehicle detection from overhead/satellite imagery at 30-50cm GSD.

**Why?** COCO-pretrained RF-DETR fails on overhead imagery because it was
trained on ground-level photography. From above, cars look completely different.
VME provides 100K+ vehicle annotations from Maxar satellite imagery at the
right resolution and viewing angle.

## Prerequisites

### Hardware
| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU | RTX 2070 (8GB VRAM) | RTX 3090 (24GB VRAM) |
| CPU | 4 cores | 8+ cores |
| RAM | 16 GB | 32 GB |
| Storage | 5 GB free | 10 GB free |
| Internet | Stable for 1.6 GB download | - |

### Software
```bash
pip install detr-geo[all]
# Or individually:
pip install detr-geo rfdetr torch torchvision requests tqdm
```

## Quick Start

```bash
# Step 1: Download VME dataset from Zenodo (1.6 GB)
python scripts/download_vme.py --output_dir vme_dataset --accept_license

# Step 2: Fine-tune RF-DETR Medium (~3-4 hours on RTX 2070)
python scripts/train_vme.py --dataset_dir vme_dataset --output_dir training_output/vme

# Step 3: Evaluate results
python scripts/validate_vme.py \
  --checkpoint training_output/vme/vme_medium_best.pth \
  --dataset_dir vme_dataset \
  --output_dir evaluation_results
```

## Step-by-Step Guide

### 1. Download the VME Dataset

The VME dataset is hosted on Zenodo (record 14185684) under CC BY-NC-ND 4.0 license.

```bash
python scripts/download_vme.py --output_dir vme_dataset --accept_license
```

This will:
- Download VME_CDSI_datasets.zip (1.6 GB) with progress bar
- Extract to vme_dataset/ with train/ and valid/ splits
- Validate COCO JSON structure and category IDs
- Report dataset statistics

**Resume support:** If download is interrupted, re-run the same command to resume.

**Verify-only mode:** Check an existing download without re-downloading:
```bash
python scripts/download_vme.py --output_dir vme_dataset --verify_only
```

### 2. Fine-Tune RF-DETR

```bash
python scripts/train_vme.py \
  --dataset_dir vme_dataset \
  --model medium \
  --epochs 30 \
  --batch_size 2 \
  --grad_accumulation_steps 8 \
  --learning_rate 1e-5 \
  --output_dir training_output/vme
```

**Default configuration (optimized for 8GB VRAM):**
| Parameter | Default | Rationale |
|-----------|---------|-----------|
| model | medium | Best accuracy/speed balance for satellite |
| batch_size | 2 | Fits in 8GB VRAM |
| grad_accumulation_steps | 8 | Effective batch size = 16 |
| learning_rate | 1e-5 | 10x lower than scratch (fine-tuning) |
| epochs | 30 | DINOv2 backbone converges fast |
| augmentation | satellite_default | 90deg rotation, vertical flip, color jitter |

**For larger GPUs (24GB+):**
```bash
python scripts/train_vme.py \
  --dataset_dir vme_dataset \
  --batch_size 8 \
  --grad_accumulation_steps 2
```

**Resume training:**
```bash
python scripts/train_vme.py \
  --dataset_dir vme_dataset \
  --resume training_output/vme/vme_medium_epoch10.pth
```

### 3. Evaluate Results

```bash
python scripts/validate_vme.py \
  --checkpoint training_output/vme/vme_medium_best.pth \
  --dataset_dir vme_dataset \
  --output_dir evaluation_results \
  --compare_baseline
```

This computes:
- mAP@0.5 and mAP@0.75 on VME validation set
- Per-class AP (Car, Bus, Truck)
- Confusion matrix
- Comparison to COCO-pretrained baseline
- Success criteria validation

### 4. Test on Real-World Imagery

After training, use the fine-tuned model with detr_geo:

```python
from detr_geo import DetrGeo

# Load fine-tuned model
detector = DetrGeo(
    model_size="medium",
    pretrain_weights="training_output/vme/vme_medium_best.pth",
)

# Detect vehicles in satellite imagery
detector.set_image("satellite_image.tif")
gdf = detector.detect_tiled(
    overlap=0.25,
    confidence_threshold=0.5,
)

# Export results
detector.to_geojson("vehicles.geojson")
```

## Success Criteria

| Criterion | Threshold | Expected |
|-----------|-----------|----------|
| mAP@0.5 on VME validation | > 60% | ~65-70% |
| Parking lot detection recall | > 50% | ~80-90% |
| Mean detection confidence | > 60% | ~70% |
| Training time on RTX 2070 | < 4 hours | ~3-4 hours |
| Improvement over COCO baseline | > 10x mAP | ~12-15x |

## VME Dataset Details

- **Source:** Zenodo record 14185684
- **License:** CC BY-NC-ND 4.0 (Non-commercial, No Derivatives)
- **Size:** 1.6 GB (ZIP), ~2 GB extracted
- **Tiles:** 4,303 at 512x512 pixels
- **Annotations:** 100,000+ vehicle bounding boxes
- **Classes:** Car, Bus, Truck
- **GSD:** 30-50cm (Maxar satellite imagery)
- **Coverage:** 54 cities across 12 countries in the Middle East
- **Format:** COCO JSON with train/valid splits

## Troubleshooting

### CUDA Out of Memory
```
RuntimeError: CUDA out of memory
```
Try:
- `--batch_size 1` (reduces VRAM usage)
- `--model small` (smaller model, less VRAM)
- `--grad_accumulation_steps 16` (maintain effective batch size)

### Download Interrupted
Re-run the download command -- it will resume from where it left off:
```bash
python scripts/download_vme.py --output_dir vme_dataset --accept_license
```

### Low mAP After Training
- Train for more epochs: `--epochs 50`
- Lower learning rate: `--learning_rate 5e-6`
- Check dataset integrity: `python scripts/download_vme.py --verify_only`

### Category ID Mismatch
If the VME dataset uses 1-indexed category IDs:
```bash
python scripts/download_vme.py --output_dir vme_dataset --verify_only --remap_categories
```

## License and Distribution

The VME dataset is licensed CC BY-NC-ND 4.0:
- **Non-Commercial:** Research use only
- **No Derivatives:** Fine-tuned weights cannot be redistributed
- **Attribution:** Cite the VME paper (Al-Emadi et al., 2025)

This means:
- You CAN use scripts to train your own model for research
- You CANNOT distribute the resulting model weights
- You CANNOT use the trained model commercially

## Citation

```bibtex
@article{alemadi2025vme,
  title={VME: Vehicles in the Middle East Dataset for Object Detection in Satellite Imagery},
  author={Al-Emadi, Nasser and others},
  year={2025}
}
```
