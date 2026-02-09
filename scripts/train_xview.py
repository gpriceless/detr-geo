#!/usr/bin/env python3
"""Fine-tune RF-DETR Medium on the xView dataset for overhead vehicle detection.

Uses detr_geo's training wrapper to fine-tune an RF-DETR model on xView
COCO-format dataset with 5 vehicle classes (Car, Pickup, Truck, Bus, Other).
Designed for A100 80GB with batch_size=8, RTX 4090 (24GB) with batch_size=4,
or RTX 2070 (8GB) with batch_size=2 and gradient accumulation.

CRITICAL LESSONS FROM VME + xView TRAINING:
1. Class name mapping: Fine-tuned models output integer class IDs that get
   wrongly mapped to COCO labels. MUST use custom_class_names parameter.
2. Checkpoint frequency: Save EVERY EPOCH. VME training crashed at epoch
   25/30 with no intermediate checkpoints. xView attempt 3 crashed at
   batch 150 of epoch 0 with zero checkpoints saved.
3. Best model is EMA: VME's best was checkpoint_best_ema.pth, not the
   regular checkpoint. Track and save EMA weights.
4. Resume can be worse: Resumed VME training epochs 25-30 were worse than
   the pre-crash best (0.277 vs 0.282 mAP). The earlier checkpoint may win.
5. OOM after long training: VME OOM'd after 365 minutes. xView attempt 4
   ran at 84% VRAM on RTX 4090 then crashed. Memory management is critical.
6. Silent crashes: xView attempt 3 died silently at batch 150 with no
   error logged. Must add OOM try/except and memory monitoring.
7. DataLoader workers: Never kill processes that look like duplicates --
   they may be DataLoader workers. Always check pstree first.

Dataset: xView Detection Challenge (Lam et al., 2018)
License: CC BY-NC-SA 4.0 (Attribution-NonCommercial-ShareAlike)

Prerequisite:
    1. Download xView: python scripts/download_xview.py
    2. Preprocess: python scripts/preprocess_xview.py
    3. Process: python scripts/process_xview.py (tile + COCO conversion)

Usage:
    python scripts/train_xview.py --dataset_dir xview_coco/
    python scripts/train_xview.py --dataset_dir xview_coco/ --epochs 50 --output_dir training_output/xview
    python scripts/train_xview.py --dataset_dir xview_coco/ --batch_size 8 --lr 1e-5
    python scripts/train_xview.py --resume training_output/xview/checkpoint_best_ema.pth
    python scripts/train_xview.py --help
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# xView 5-class vehicle mapping (MUST match preprocess_xview.py output)
# This is the custom_class_names dict that prevents the COCO label bug.
XVIEW_CLASSES: dict[int, str] = {
    0: "Car",
    1: "Pickup",
    2: "Truck",
    3: "Bus",
    4: "Other",
}
XVIEW_NUM_CLASSES = 5

# Default training configuration optimized for A100 80GB VRAM
# (also works on smaller GPUs with auto batch size adjustment)
DEFAULT_CONFIG = {
    "model": "medium",
    "epochs": 50,
    "batch_size": 8,           # A100 80GB default; auto-adjusted for smaller GPUs
    "grad_accumulation_steps": 2,
    "learning_rate": 1e-5,
    "augmentation_preset": "satellite_default",
    "save_interval": 1,        # Save EVERY epoch (lesson: lost 6+ hours of training)
    "val_interval": 1,
    "early_stopping_patience": 10,
}


# ---------------------------------------------------------------------------
# Dataset verification
# ---------------------------------------------------------------------------


def verify_dataset(dataset_dir: str) -> dict[str, Any]:
    """Verify xView COCO-format dataset is present and properly structured.

    Expects standard COCO layout:
    - train/images/, train/_annotations.coco.json
    - valid/images/, valid/_annotations.coco.json

    Args:
        dataset_dir: Path to the COCO-format dataset directory.

    Returns:
        Dict with dataset info (class mapping, image counts, paths).

    Raises:
        FileNotFoundError: If dataset directory or required files are missing.
        ValueError: If COCO JSON is invalid or class count is wrong.
    """
    ds = Path(dataset_dir)

    if not ds.exists():
        raise FileNotFoundError(
            f"Dataset not found at {dataset_dir}. "
            "Run preprocessing pipeline first."
        )

    # Check for standard COCO layout
    train_dir = ds / "train"
    valid_dir = ds / "valid"
    train_ann = train_dir / "_annotations.coco.json"
    valid_ann = valid_dir / "_annotations.coco.json"

    if not train_ann.exists():
        raise FileNotFoundError(
            f"Missing train annotations: {train_ann}. "
            "Expected COCO-format layout with train/_annotations.coco.json."
        )
    if not valid_ann.exists():
        raise FileNotFoundError(
            f"Missing valid annotations: {valid_ann}. "
            "Expected COCO-format layout with valid/_annotations.coco.json."
        )

    train_images_dir = train_dir / "images" if (train_dir / "images").exists() else train_dir
    valid_images_dir = valid_dir / "images" if (valid_dir / "images").exists() else valid_dir

    # Parse annotations
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
    train_images = (
        list(train_images_dir.glob("*.jpg"))
        + list(train_images_dir.glob("*.png"))
        + list(train_images_dir.glob("*.jpeg"))
    )
    valid_images = (
        list(valid_images_dir.glob("*.jpg"))
        + list(valid_images_dir.glob("*.png"))
        + list(valid_images_dir.glob("*.jpeg"))
    )

    # Compute annotation statistics per category
    train_cat_counts: dict[int, int] = {}
    for ann in train_data.get("annotations", []):
        cid = ann.get("category_id", -1)
        train_cat_counts[cid] = train_cat_counts.get(cid, 0) + 1

    return {
        "categories": categories,
        "num_classes": len(categories),
        "train_images": num_train_images,
        "train_annotations": num_train_annotations,
        "valid_images": num_valid_images,
        "valid_annotations": num_valid_annotations,
        "train_image_files": len(train_images),
        "valid_image_files": len(valid_images),
        "train_cat_counts": train_cat_counts,
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
                "(est. 100+ hours for 50 epochs)."
            )
    except ImportError:
        result["warning"] = "PyTorch not installed. Required for training."

    return result


def estimate_batch_size(vram_gb: float) -> int:
    """Estimate optimal batch size based on available VRAM.

    RF-DETR Medium (576x576) VRAM usage observations:
    - RTX 4090 (24GB): batch_size=4 uses ~20.6GB (84%) -- crashed OOM
    - A100 80GB: batch_size=8 should use ~35-40GB (safe margin)
    - Per-batch overhead: ~5GB base + ~4GB per sample (approximate)

    Args:
        vram_gb: Available VRAM in GB.

    Returns:
        Recommended batch size.
    """
    if vram_gb >= 70:
        return 8       # A100 80GB: plenty of room
    elif vram_gb >= 40:
        return 6       # A100 40GB: comfortable
    elif vram_gb >= 20:
        return 4       # RTX 4090 24GB: tight but feasible (was 84% at batch=4)
    elif vram_gb >= 16:
        return 3       # RTX 4080/3090 Ti
    elif vram_gb >= 10:
        return 2       # RTX 3080/2080
    else:
        return 1       # 8GB GPUs: need gradient accumulation


def clear_gpu_cache() -> None:
    """Clear GPU memory cache to prevent OOM during long training.

    Lesson from VME: OOM after 365 minutes of training.
    Lesson from xView attempt 3: Silent crash at 84% VRAM usage.
    """
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
    except ImportError:
        pass


def log_gpu_memory(label: str = "") -> dict[str, float]:
    """Log current GPU memory usage.

    Args:
        label: Optional label for the log message.

    Returns:
        Dict with allocated_gb, reserved_gb, and free_gb.
    """
    result: dict[str, float] = {}
    try:
        import torch
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / (1024**3)
            reserved = torch.cuda.memory_reserved() / (1024**3)
            total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            free = total - reserved
            result = {
                "allocated_gb": round(allocated, 2),
                "reserved_gb": round(reserved, 2),
                "total_gb": round(total, 1),
                "free_gb": round(free, 2),
                "usage_pct": round(reserved / total * 100, 1),
            }
            prefix = f"  [{label}] " if label else "  "
            print(
                f"{prefix}GPU Memory: {allocated:.1f}GB allocated, "
                f"{reserved:.1f}GB reserved / {total:.1f}GB total "
                f"({result['usage_pct']:.0f}% used, {free:.1f}GB free)"
            )
    except ImportError:
        pass
    return result


def train_with_oom_protection(
    train_fn: Any,
    config: dict[str, Any],
    max_retries: int = 3,
) -> dict[str, Any]:
    """Run training with automatic OOM recovery.

    If CUDA OOM occurs, reduces batch size by half and retries.
    Saves a crash log each time for debugging.

    Args:
        train_fn: Callable that takes config dict and runs training.
        config: Training configuration dict.
        max_retries: Maximum number of retries with reduced batch size.

    Returns:
        Training result dict.

    Raises:
        RuntimeError: If all retries exhausted.
    """
    import torch

    current_batch_size = config["batch_size"]
    output_dir = Path(config.get("output_dir", "training_output/xview"))

    for attempt in range(max_retries + 1):
        try:
            config["batch_size"] = current_batch_size
            print(f"\n  Training attempt {attempt + 1}/{max_retries + 1} "
                  f"(batch_size={current_batch_size})")
            log_gpu_memory("pre-training")
            return train_fn(config)

        except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
            error_str = str(exc).lower()
            is_oom = (
                isinstance(exc, torch.cuda.OutOfMemoryError)
                or "out of memory" in error_str
                or "cuda" in error_str and "memory" in error_str
            )

            if not is_oom:
                raise  # Not an OOM error, re-raise

            # Log the OOM event
            print(f"\n  OOM DETECTED at batch_size={current_batch_size}")
            log_gpu_memory("oom-crash")

            # Save crash log
            crash_log = {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "attempt": attempt + 1,
                "batch_size": current_batch_size,
                "error": str(exc)[:500],
            }
            crash_path = output_dir / f"oom_crash_attempt_{attempt + 1}.json"
            output_dir.mkdir(parents=True, exist_ok=True)
            try:
                with open(crash_path, "w") as f:
                    json.dump(crash_log, f, indent=2)
                print(f"  Crash log saved: {crash_path}")
            except Exception:
                pass

            # Clear GPU cache
            clear_gpu_cache()

            if attempt < max_retries:
                new_batch_size = max(1, current_batch_size // 2)
                if new_batch_size == current_batch_size:
                    raise RuntimeError(
                        f"OOM at batch_size=1. Cannot reduce further. "
                        f"Try model='small' or reduce image resolution."
                    ) from exc

                print(f"  Reducing batch_size: {current_batch_size} -> {new_batch_size}")
                print(f"  Increasing grad_accumulation to compensate...")
                # Compensate effective batch size
                old_effective = current_batch_size * config.get("grad_accumulation_steps", 1)
                config["grad_accumulation_steps"] = max(
                    1, old_effective // new_batch_size
                )
                current_batch_size = new_batch_size
                print(f"  New effective batch size: "
                      f"{new_batch_size} x {config['grad_accumulation_steps']} = "
                      f"{new_batch_size * config['grad_accumulation_steps']}")

                # Wait briefly for GPU memory to settle
                time.sleep(5)
                clear_gpu_cache()
            else:
                raise RuntimeError(
                    f"Training failed after {max_retries + 1} attempts. "
                    f"Last batch_size={current_batch_size}. Error: {exc}"
                ) from exc

    raise RuntimeError("Should not reach here")


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
    """Execute xView fine-tuning using detr_geo training pipeline.

    This function:
    1. Initializes the RF-DETR adapter with custom_class_names
       (CRITICAL: prevents COCO label mapping bug from VME)
    2. Calls detr_geo.training.train() with xView-specific configuration
    3. Logs GPU memory usage before, during, and after training
    4. Clears GPU cache before and after training to prevent OOM

    Note on checkpointing: RF-DETR's internal training loop handles
    checkpoint saving. The checkpoint_interval and output_dir are passed
    as kwargs to adapter.train() which forwards them to rfdetr's
    model.train(). If rfdetr does not support these kwargs, checkpoints
    are saved at rfdetr's defaults (typically every epoch or at best).

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

    # -------------------------------------------------------------------
    # CRITICAL: Initialize adapter with custom_class_names
    # This prevents the class name mapping bug where fine-tuned models
    # output integer class IDs that get wrongly mapped to COCO labels.
    # VME lesson: without this, class 0 maps to "person" instead of "Car".
    # -------------------------------------------------------------------
    print(f"  Loading RF-DETR {config['model'].capitalize()} model...")
    adapter = RFDETRAdapter(
        model_size=config["model"],
        pretrain_weights=config.get("resume"),
        custom_class_names=XVIEW_CLASSES,  # CRITICAL: prevents COCO label bug
    )
    print(f"  Model resolution: {adapter.resolution}px")
    print(f"  Device: {adapter._device}")
    print(f"  Custom class names: {XVIEW_CLASSES}")

    # Log GPU memory before training
    log_gpu_memory("model-loaded")

    # Clear GPU cache before training (VME OOM lesson)
    clear_gpu_cache()

    # Resume checkpoint
    resume_path = config.get("resume")

    # Augmentation preset
    aug_preset = config.get("augmentation_preset", "satellite_default")

    print(f"\n  Training configuration:")
    print(f"    Model: {config['model']}")
    print(f"    Classes: {XVIEW_NUM_CLASSES} ({', '.join(XVIEW_CLASSES.values())})")
    print(f"    Epochs: {config['epochs']}")
    print(f"    Batch size: {config['batch_size']}")
    print(f"    Grad accumulation: {config.get('grad_accumulation_steps', 1)}")
    eff_bs = config["batch_size"] * config.get("grad_accumulation_steps", 1)
    print(f"    Effective batch size: {eff_bs}")
    print(f"    Learning rate: {config.get('learning_rate', 'default')}")
    print(f"    Augmentation: {aug_preset}")
    print(f"    Checkpoint interval: every {config.get('save_interval', 1)} epochs")
    print(f"    Output: {config['output_dir']}")

    if resume_path:
        print(f"    Resume from: {resume_path}")
        print(f"    NOTE: Resumed training may produce worse results than")
        print(f"    pre-crash best checkpoint (VME lesson: 0.277 vs 0.282 mAP).")
        print(f"    Compare both checkpoints after training completes.")

    # Record start time
    t0 = time.time()

    # Run training
    # Note: detr_geo.training.train() accepts: adapter, dataset_dir, epochs,
    # augmentation_preset, augmentation_factor, batch_size, learning_rate,
    # resume, seed, and **kwargs (forwarded to adapter.train()).
    # We pass output_dir and checkpoint_interval as kwargs -- rfdetr's
    # model.train() may accept them as save_dir and checkpoint_freq.
    print(f"\n  Starting training...")
    print(f"  Checkpoints will be saved every {config.get('save_interval', 1)} epoch(s).")
    print(f"  Best EMA checkpoint is typically the winner (VME lesson).")

    log_gpu_memory("pre-train")

    result = train(
        adapter=adapter,
        dataset_dir=config["dataset_dir"],
        epochs=config["epochs"],
        augmentation_preset=aug_preset,
        batch_size=config["batch_size"],
        learning_rate=config.get("learning_rate"),
        resume=resume_path,
        seed=config.get("seed", 42),
        # kwargs forwarded to adapter.train() -> rfdetr model.train()
        output_dir=str(output_dir),
        checkpoint_freq=config.get("save_interval", 1),
        grad_accum_steps=config.get("grad_accumulation_steps", 1),
    )

    elapsed = time.time() - t0
    elapsed_hours = elapsed / 3600

    # Log and clear GPU cache after training
    log_gpu_memory("post-train")
    clear_gpu_cache()

    print(f"\n  Training complete!")
    print(f"  Duration: {elapsed_hours:.1f} hours ({elapsed:.0f} seconds)")

    # Save training summary
    summary = {
        "config": config,
        "duration_seconds": elapsed,
        "duration_hours": round(elapsed_hours, 2),
        "num_classes": XVIEW_NUM_CLASSES,
        "class_names": XVIEW_CLASSES,
        "result": result if isinstance(result, dict) else {},
    }
    summary_path = output_dir / "training_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"  Summary saved: {summary_path}")

    # Save class names for inference
    class_names_path = output_dir / "class_names.json"
    with open(class_names_path, "w") as f:
        json.dump(XVIEW_CLASSES, f, indent=2)
    print(f"  Class names saved: {class_names_path}")

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
        description=(
            "Fine-tune RF-DETR on the xView dataset for overhead vehicle "
            "detection with 5 classes (Car, Pickup, Truck, Bus, Other)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default="xview_coco",
        help="Path to COCO-format xView dataset directory (default: xview_coco)",
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
        "--learning_rate", "--lr",
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
        help=f"Early stopping patience in epochs (default: {DEFAULT_CONFIG['early_stopping_patience']})",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="training_output/xview",
        help="Directory for checkpoints and logs (default: training_output/xview)",
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
    """Run xView fine-tuning pipeline.

    Args:
        argv: Command-line arguments (defaults to sys.argv[1:]).

    Returns:
        Exit code: 0 on success, 1 on failure.
    """
    args = parse_args(argv)

    print("=" * 60)
    print("  xView Fine-Tuning: RF-DETR for Overhead Vehicle Detection")
    print("  5 Classes: Car, Pickup, Truck, Bus, Other")
    print("=" * 60)

    # -------------------------------------------------------------------
    # Device detection
    # -------------------------------------------------------------------
    print("\n[1/4] Detecting hardware...")
    device_info = detect_device()
    print(f"  Device: {device_info['device']}")
    if "gpu_name" in device_info:
        print(f"  GPU: {device_info['gpu_name']} ({device_info.get('vram_gb', '?')} GB)")

        # Auto-adjust batch size based on VRAM
        recommended_bs = estimate_batch_size(device_info.get("vram_gb", 0))
        if args.batch_size > recommended_bs:
            print(
                f"  NOTE: batch_size={args.batch_size} may OOM on "
                f"{device_info.get('vram_gb', '?')} GB VRAM. "
                f"Recommended: {recommended_bs}"
            )

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
    print(f"  Num classes: {ds_info['num_classes']}")
    print(f"  Train: {ds_info['train_images']} images, {ds_info['train_annotations']} annotations")
    print(f"  Valid: {ds_info['valid_images']} images, {ds_info['valid_annotations']} annotations")

    # Show per-class annotation counts
    if ds_info.get("train_cat_counts"):
        print(f"  Train annotations per class:")
        for cid, count in sorted(ds_info["train_cat_counts"].items()):
            class_name = ds_info["categories"].get(cid, f"class_{cid}")
            print(f"    {class_name} (id={cid}): {count:,}")

    # Verify class count matches expectation
    if ds_info["num_classes"] != XVIEW_NUM_CLASSES:
        print(
            f"\n  WARNING: Dataset has {ds_info['num_classes']} classes, "
            f"expected {XVIEW_NUM_CLASSES}. This may be fine if background "
            f"class is included or excluded."
        )

    # -------------------------------------------------------------------
    # Build config
    # -------------------------------------------------------------------
    print(f"\n[3/4] Preparing training configuration...")
    config = build_training_config(args)

    # Estimate training time
    if device_info["type"] == "cuda":
        vram = device_info.get("vram_gb", 8)
        # Rough estimate based on dataset size and GPU
        # xView attempt 4 on RTX 4090: ~31 min/epoch with ~32K train images
        images_per_epoch = ds_info["train_images"]
        if vram >= 70:
            est_minutes_per_epoch = max(3, images_per_epoch / 800)   # A100 80GB: ~800 img/min
        elif vram >= 40:
            est_minutes_per_epoch = max(4, images_per_epoch / 650)   # A100 40GB
        elif vram >= 24:
            est_minutes_per_epoch = max(5, images_per_epoch / 500)   # RTX 4090
        elif vram >= 10:
            est_minutes_per_epoch = max(8, images_per_epoch / 300)   # RTX 3080
        else:
            est_minutes_per_epoch = max(12, images_per_epoch / 150)  # RTX 2070
        est_hours = (args.epochs * est_minutes_per_epoch) / 60
        print(f"  Estimated training time: ~{est_hours:.1f} hours")
        print(f"  ({est_minutes_per_epoch:.1f} min/epoch x {args.epochs} epochs)")
    else:
        print(f"  Estimated training time: very long (CPU)")

    # -------------------------------------------------------------------
    # Run training (with OOM protection)
    # -------------------------------------------------------------------
    print(f"\n[4/4] Starting training...")

    # Log initial GPU state
    log_gpu_memory("before-training")

    try:
        # Use OOM protection wrapper for automatic batch size reduction
        summary = train_with_oom_protection(
            train_fn=run_training,
            config=config,
            max_retries=3,
        )
    except Exception as exc:
        print(f"\n  ERROR: Training failed: {exc}")

        # Log GPU state at crash time
        log_gpu_memory("crash")

        # Provide specific OOM guidance
        error_str = str(exc).lower()
        if "cuda" in error_str and ("memory" in error_str or "oom" in error_str):
            print(
                "\n  GPU out of memory after automatic batch size reduction. Try:"
                "\n    --batch_size 1 --grad_accumulation_steps 16"
                "\n    --model small"
                "\n"
                "\n  Note: xView is larger than VME. If VME worked with"
                "\n  batch_size=2, try batch_size=1 for xView."
            )

        # Remind about existing checkpoints
        output_path = Path(args.output_dir)
        if output_path.exists():
            checkpoints = sorted(output_path.glob("*.pth"))
            if checkpoints:
                print(f"\n  Existing checkpoints found ({len(checkpoints)}):")
                for ckpt in checkpoints[-3:]:
                    size_mb = ckpt.stat().st_size / (1024 * 1024)
                    print(f"    {ckpt.name} ({size_mb:.0f} MB)")
                print(f"\n  To resume: --resume {checkpoints[-1]}")
                print(f"  VME lesson: the pre-crash best checkpoint may still be")
                print(f"  the winner. Compare checkpoint_best_ema.pth with resumed results.")

        return 1

    # -------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print(f"  TRAINING COMPLETE")
    print(f"{'=' * 60}")
    duration = summary.get("duration_hours", 0)
    print(f"  Duration: {duration:.1f} hours")
    print(f"  Classes: {XVIEW_NUM_CLASSES} ({', '.join(XVIEW_CLASSES.values())})")
    print(f"  Output: {args.output_dir}")
    print(f"  Config: {args.output_dir}/training_config.json")
    print(f"  Summary: {args.output_dir}/training_summary.json")
    print(f"  Class names: {args.output_dir}/class_names.json")

    # Report checkpoint files
    output_path = Path(args.output_dir)
    checkpoints = sorted(output_path.glob("*.pth")) if output_path.exists() else []
    if checkpoints:
        print(f"  Checkpoints ({len(checkpoints)}):")
        for ckpt in checkpoints:
            size_mb = ckpt.stat().st_size / (1024 * 1024)
            print(f"    {ckpt.name} ({size_mb:.0f} MB)")

        # Highlight best EMA checkpoint (VME lesson)
        ema_checkpoints = [c for c in checkpoints if "ema" in c.name.lower()]
        best_checkpoints = [c for c in checkpoints if "best" in c.name.lower()]
        best_ema = [c for c in checkpoints if "best" in c.name.lower() and "ema" in c.name.lower()]

        if best_ema:
            print(f"\n  RECOMMENDED: Use {best_ema[0].name}")
            print(f"  (VME lesson: EMA checkpoint typically outperforms regular)")
        elif ema_checkpoints:
            print(f"\n  NOTE: EMA checkpoint(s) available. These often outperform regular checkpoints.")
        elif best_checkpoints:
            print(f"\n  Best checkpoint: {best_checkpoints[0].name}")
    else:
        print(f"  Checkpoints: see {args.output_dir}/ (RF-DETR default location)")

    # Usage instructions with custom_class_names
    print(f"\n  Usage with fine-tuned model:")
    best = best_ema[0] if best_ema else (best_checkpoints[0] if best_checkpoints else (checkpoints[-1] if checkpoints else f"{args.output_dir}/checkpoint.pth"))
    print(f"    from detr_geo import DetrGeo")
    print(f"    model = DetrGeo(")
    print(f'        model="medium",')
    print(f'        weights="{best}",')
    print(f"        custom_class_names={XVIEW_CLASSES},  # CRITICAL: prevents COCO label bug")
    print(f"    )")
    print(f"    results = model.predict(image)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
