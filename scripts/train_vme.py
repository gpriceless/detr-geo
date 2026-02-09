#!/usr/bin/env python3
"""Fine-tune RF-DETR Medium on the VME dataset for overhead vehicle detection.

Uses detr_geo's training wrapper to fine-tune an RF-DETR model on the VME
(Vehicles in the Middle East) COCO-format dataset. Designed for 8GB VRAM
GPUs with batch_size=2 and gradient accumulation.

Prerequisite: Run scripts/download_vme.py first to download the VME dataset.

Usage:
    python scripts/train_vme.py --dataset_dir vme_dataset
    python scripts/train_vme.py --dataset_dir vme_dataset --epochs 30 --output_dir training_output/vme
    python scripts/train_vme.py --resume training_output/vme/vme_medium_epoch10.pth
    python scripts/train_vme.py --help
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VME_CLASSES = {0: "Car", 1: "Bus", 2: "Truck"}
VME_NUM_CLASSES = 3

# Default training configuration optimized for RTX 2070 (8GB VRAM)
DEFAULT_CONFIG = {
    "model": "medium",
    "epochs": 30,
    "batch_size": 2,
    "grad_accumulation_steps": 8,
    "learning_rate": 1e-5,
    "augmentation_preset": "satellite_default",
    "save_interval": 5,
    "val_interval": 1,
    "early_stopping_patience": 5,
}


# ---------------------------------------------------------------------------
# Dataset verification
# ---------------------------------------------------------------------------


def verify_dataset(dataset_dir: str) -> dict[str, Any]:
    """Verify VME dataset is present and properly structured.

    Supports two layouts:
    1. Standard COCO layout (after reorganization):
       - train/images/, train/_annotations.coco.json
       - valid/images/, valid/_annotations.coco.json
    2. Original VME layout:
       - satellite_images/, annotations_HBB/{train,val}.json

    Args:
        dataset_dir: Path to the VME dataset directory.

    Returns:
        Dict with dataset info (class mapping, image counts, paths).

    Raises:
        FileNotFoundError: If dataset directory or required files are missing.
        ValueError: If COCO JSON is invalid.
    """
    ds = Path(dataset_dir)

    if not ds.exists():
        raise FileNotFoundError(
            f"Dataset not found at {dataset_dir}. "
            "Run python scripts/download_vme.py first."
        )

    # Check for standard COCO layout first (preferred)
    train_dir = ds / "train"
    valid_dir = ds / "valid"
    train_ann = train_dir / "_annotations.coco.json"
    valid_ann = valid_dir / "_annotations.coco.json"

    if train_ann.exists() and valid_ann.exists():
        # Standard COCO layout
        train_images_dir = train_dir / "images" if (train_dir / "images").exists() else train_dir
        valid_images_dir = valid_dir / "images" if (valid_dir / "images").exists() else valid_dir
    else:
        # Fall back to original VME layout
        image_dir = ds / "satellite_images"
        ann_dir = ds / "annotations_HBB"

        if not image_dir.exists() or not ann_dir.exists():
            raise FileNotFoundError(
                f"Dataset at {dataset_dir} is missing expected structure. "
                "Expected either train/valid/ subdirs or satellite_images/ + annotations_HBB/. "
                "Run python scripts/download_vme.py first."
            )

        train_ann = ann_dir / "train.json"
        valid_ann = ann_dir / "val.json"
        train_images_dir = image_dir
        valid_images_dir = image_dir

        if not train_ann.exists() or not valid_ann.exists():
            raise FileNotFoundError(
                f"Missing annotation files in {dataset_dir}/annotations_HBB/."
            )

    # Parse annotations to get class mapping
    with open(train_ann) as f:
        train_data = json.load(f)

    categories = {c["id"]: c["name"] for c in train_data.get("categories", [])}
    num_train_images = len(train_data.get("images", []))
    num_train_annotations = len(train_data.get("annotations", []))

    with open(valid_ann) as f:
        valid_data = json.load(f)

    num_valid_images = len(valid_data.get("images", []))
    num_valid_annotations = len(valid_data.get("annotations", []))

    # Count actual image files
    train_images = list(train_images_dir.glob("*.jpg")) + list(train_images_dir.glob("*.png"))
    valid_images = list(valid_images_dir.glob("*.jpg")) + list(valid_images_dir.glob("*.png"))

    return {
        "categories": categories,
        "num_classes": len(categories),
        "train_images": num_train_images,
        "train_annotations": num_train_annotations,
        "valid_images": num_valid_images,
        "valid_annotations": num_valid_annotations,
        "train_image_files": len(train_images),
        "valid_image_files": len(valid_images),
    }


def detect_device() -> dict[str, Any]:
    """Detect available compute device and report capabilities.

    Returns:
        Dict with device name, type, and memory info.
    """
    result: dict[str, Any] = {"device": "cpu", "type": "cpu", "warning": None}

    try:
        import torch

        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            vram_bytes = torch.cuda.get_device_properties(0).total_memory
            vram_gb = vram_bytes / (1024**3)
            result["device"] = "cuda:0"
            result["type"] = "cuda"
            result["gpu_name"] = gpu_name
            result["vram_gb"] = round(vram_gb, 1)

            if vram_gb < 6:
                result["warning"] = (
                    f"GPU has only {vram_gb:.1f} GB VRAM. "
                    "Try batch_size=1 or use RF-DETR Small model."
                )
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            result["device"] = "mps"
            result["type"] = "mps"
        else:
            result["warning"] = (
                "No GPU detected. Training on CPU will be very slow "
                "(est. 40+ hours for 30 epochs)."
            )
    except ImportError:
        result["warning"] = "PyTorch not installed. Required for training."

    return result


def build_training_config(args: argparse.Namespace) -> dict[str, Any]:
    """Build training configuration from CLI arguments.

    Args:
        args: Parsed command-line arguments.

    Returns:
        Training configuration dict.
    """
    config = {
        "dataset_dir": args.dataset_dir,
        "model": args.model,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "grad_accumulation_steps": args.grad_accumulation_steps,
        "learning_rate": args.learning_rate,
        "augmentation_preset": args.augmentation_preset,
        "save_interval": args.save_interval,
        "val_interval": args.val_interval,
        "early_stopping_patience": args.early_stopping_patience,
        "output_dir": args.output_dir,
        "resume": args.resume,
        "seed": args.seed,
    }
    return config


def run_training(config: dict[str, Any]) -> dict[str, Any]:
    """Execute VME fine-tuning using detr_geo training pipeline.

    This function:
    1. Initializes the RF-DETR adapter with the specified model
    2. Calls detr_geo.training.train() with VME-specific configuration
    3. Saves checkpoints to the output directory
    4. Returns training metrics

    Args:
        config: Training configuration dict.

    Returns:
        Training results dict with metrics and checkpoint paths.

    Raises:
        RuntimeError: If training fails.
    """
    from detr_geo._adapter import RFDETRAdapter
    from detr_geo.training import train, AUGMENTATION_PRESETS

    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save config for reproducibility
    config_path = output_dir / "training_config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"  Config saved: {config_path}")

    # Initialize adapter
    print(f"  Loading RF-DETR {config['model'].capitalize()} model...")
    adapter = RFDETRAdapter(
        model_size=config["model"],
        pretrain_weights=config.get("resume"),
    )
    print(f"  Model resolution: {adapter.resolution}px")
    print(f"  Device: {adapter._device}")

    # Build train kwargs
    train_kwargs: dict[str, Any] = {
        "epochs": config["epochs"],
        "batch_size": config["batch_size"],
    }

    # Pass grad accumulation if supported
    if config.get("grad_accumulation_steps"):
        train_kwargs["grad_accum_steps"] = config["grad_accumulation_steps"]

    # Learning rate
    if config.get("learning_rate"):
        train_kwargs["learning_rate"] = config["learning_rate"]

    # Resume checkpoint
    resume_path = config.get("resume")

    # Augmentation preset
    aug_preset = config.get("augmentation_preset", "satellite_default")

    print(f"\n  Training configuration:")
    print(f"    Model: {config['model']}")
    print(f"    Epochs: {config['epochs']}")
    print(f"    Batch size: {config['batch_size']}")
    print(f"    Grad accumulation: {config.get('grad_accumulation_steps', 1)}")
    print(f"    Effective batch size: {config['batch_size'] * config.get('grad_accumulation_steps', 1)}")
    print(f"    Learning rate: {config.get('learning_rate', 'default')}")
    print(f"    Augmentation: {aug_preset}")
    print(f"    Output: {config['output_dir']}")

    if resume_path:
        print(f"    Resume from: {resume_path}")

    # Record start time
    t0 = time.time()

    # Run training
    print(f"\n  Starting training...")
    result = train(
        adapter=adapter,
        dataset_dir=config["dataset_dir"],
        epochs=config["epochs"],
        augmentation_preset=aug_preset,
        batch_size=config["batch_size"],
        learning_rate=config.get("learning_rate"),
        resume=resume_path,
        seed=config.get("seed", 42),
        output_dir=str(output_dir),
        checkpoint_interval=config.get("save_interval", 5),
        **{k: v for k, v in train_kwargs.items()
           if k not in ("epochs", "batch_size", "learning_rate")},
    )

    elapsed = time.time() - t0
    elapsed_hours = elapsed / 3600

    print(f"\n  Training complete!")
    print(f"  Duration: {elapsed_hours:.1f} hours ({elapsed:.0f} seconds)")

    # Save training summary
    summary = {
        "config": config,
        "duration_seconds": elapsed,
        "duration_hours": round(elapsed_hours, 2),
        "result": result if isinstance(result, dict) else {},
    }
    summary_path = output_dir / "training_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"  Summary saved: {summary_path}")

    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Argument list (defaults to sys.argv[1:]).

    Returns:
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Fine-tune RF-DETR on the VME dataset for overhead vehicle detection.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default="vme_dataset",
        help="Path to VME dataset directory (default: vme_dataset)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_CONFIG["model"],
        choices=["nano", "small", "medium", "base", "large"],
        help=f"RF-DETR model size (default: {DEFAULT_CONFIG['model']})",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_CONFIG["epochs"],
        help=f"Number of training epochs (default: {DEFAULT_CONFIG['epochs']})",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=DEFAULT_CONFIG["batch_size"],
        help=f"Training batch size (default: {DEFAULT_CONFIG['batch_size']})",
    )
    parser.add_argument(
        "--grad_accumulation_steps",
        type=int,
        default=DEFAULT_CONFIG["grad_accumulation_steps"],
        help=f"Gradient accumulation steps (default: {DEFAULT_CONFIG['grad_accumulation_steps']})",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=DEFAULT_CONFIG["learning_rate"],
        help=f"Learning rate (default: {DEFAULT_CONFIG['learning_rate']})",
    )
    parser.add_argument(
        "--augmentation_preset",
        type=str,
        default=DEFAULT_CONFIG["augmentation_preset"],
        choices=["satellite_default", "aerial_default", "drone_default"],
        help=f"Augmentation preset (default: {DEFAULT_CONFIG['augmentation_preset']})",
    )
    parser.add_argument(
        "--save_interval",
        type=int,
        default=DEFAULT_CONFIG["save_interval"],
        help=f"Save checkpoint every N epochs (default: {DEFAULT_CONFIG['save_interval']})",
    )
    parser.add_argument(
        "--val_interval",
        type=int,
        default=DEFAULT_CONFIG["val_interval"],
        help=f"Validate every N epochs (default: {DEFAULT_CONFIG['val_interval']})",
    )
    parser.add_argument(
        "--early_stopping_patience",
        type=int,
        default=DEFAULT_CONFIG["early_stopping_patience"],
        help=f"Early stopping patience (default: {DEFAULT_CONFIG['early_stopping_patience']})",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="training_output/vme",
        help="Directory for checkpoints and logs (default: training_output/vme)",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint for resuming training",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run VME fine-tuning pipeline.

    Args:
        argv: Command-line arguments (defaults to sys.argv[1:]).

    Returns:
        Exit code: 0 on success, 1 on failure.
    """
    args = parse_args(argv)

    print("=" * 60)
    print("  VME Fine-Tuning: RF-DETR for Overhead Vehicle Detection")
    print("=" * 60)

    # -------------------------------------------------------------------
    # Device detection
    # -------------------------------------------------------------------
    print("\n[1/4] Detecting hardware...")
    device_info = detect_device()
    print(f"  Device: {device_info['device']}")
    if "gpu_name" in device_info:
        print(f"  GPU: {device_info['gpu_name']} ({device_info.get('vram_gb', '?')} GB)")
    if device_info.get("warning"):
        print(f"  WARNING: {device_info['warning']}")
        if device_info["type"] == "cpu":
            try:
                answer = input("  Continue on CPU? (yes/no): ").strip().lower()
                if answer not in ("yes", "y"):
                    print("  Training cancelled.")
                    return 1
            except EOFError:
                pass  # Non-interactive, continue anyway

    # -------------------------------------------------------------------
    # Dataset verification
    # -------------------------------------------------------------------
    print(f"\n[2/4] Verifying dataset at {args.dataset_dir}...")
    try:
        ds_info = verify_dataset(args.dataset_dir)
    except (FileNotFoundError, ValueError) as exc:
        print(f"  ERROR: {exc}")
        return 1

    print(f"  Classes: {ds_info['categories']}")
    print(f"  Train: {ds_info['train_images']} images, {ds_info['train_annotations']} annotations")
    print(f"  Valid: {ds_info['valid_images']} images, {ds_info['valid_annotations']} annotations")

    # -------------------------------------------------------------------
    # Build config
    # -------------------------------------------------------------------
    print(f"\n[3/4] Preparing training configuration...")
    config = build_training_config(args)

    # Estimate training time
    # Rough estimate: ~8 minutes per epoch on RTX 2070 for VME
    if device_info["type"] == "cuda":
        est_minutes_per_epoch = 8
        est_hours = (args.epochs * est_minutes_per_epoch) / 60
        print(f"  Estimated training time: ~{est_hours:.1f} hours")
    else:
        print(f"  Estimated training time: very long (CPU)")

    # -------------------------------------------------------------------
    # Run training
    # -------------------------------------------------------------------
    print(f"\n[4/4] Starting training...")
    try:
        summary = run_training(config)
    except Exception as exc:
        print(f"\n  ERROR: Training failed: {exc}")

        # Provide specific OOM guidance
        error_str = str(exc).lower()
        if "cuda" in error_str and ("memory" in error_str or "oom" in error_str):
            print(
                "\n  GPU out of memory. Try:"
                "\n    --batch_size 1"
                "\n    --model small"
                "\n    --grad_accumulation_steps 16"
            )

        return 1

    # -------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print(f"  TRAINING COMPLETE")
    print(f"{'=' * 60}")
    duration = summary.get("duration_hours", 0)
    print(f"  Duration: {duration:.1f} hours")
    print(f"  Output: {args.output_dir}")
    print(f"  Config: {args.output_dir}/training_config.json")
    print(f"  Summary: {args.output_dir}/training_summary.json")

    # Report checkpoint files
    output_path = Path(args.output_dir)
    checkpoints = sorted(output_path.glob("*.pth")) if output_path.exists() else []
    if checkpoints:
        print(f"  Checkpoints ({len(checkpoints)}):")
        for ckpt in checkpoints:
            size_mb = ckpt.stat().st_size / (1024 * 1024)
            print(f"    {ckpt.name} ({size_mb:.0f} MB)")
    else:
        print(f"  Checkpoints: see {args.output_dir}/ (RF-DETR default location)")

    print(f"\n  Next steps:")
    best = next((c for c in checkpoints if "best" in c.name), None)
    ckpt_hint = best or (checkpoints[-1] if checkpoints else f"{args.output_dir}/checkpoint.pth")
    print(f"    python scripts/validate_vme.py --checkpoint {ckpt_hint}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
