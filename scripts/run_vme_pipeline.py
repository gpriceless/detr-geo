#!/usr/bin/env python3
"""VME Fine-Tuning Pipeline - Orchestrate download, training, and validation.

This script runs the complete VME fine-tuning workflow with checkpoints,
progress tracking, time estimates, and resume capability. Designed for
RTX 2070 8GB VRAM GPUs.

RESOURCE USAGE
--------------
- During download: Network + CPU. Computer fully usable.
- During extraction: CPU only. Computer fully usable.
- During training: GPU at 90-100%, ~6-7GB VRAM. Computer usable for
  light tasks (browsing, docs). Avoid other GPU tasks (gaming, video editing).
- During validation: GPU for inference, quick pass. Computer usable.
- Memory: ~4GB system RAM used by Python process.
- Disk: Need ~5GB free (1.6GB dataset + checkpoints + logs).

FAQ
---
Q: Can I use my computer while this runs?
A: Yes! Download and extraction are light. During training, your computer will
   be usable but slightly slower. Avoid GPU-heavy tasks like gaming or video
   editing during training. Browsing, docs, and coding work fine.

Q: Should I run this overnight?
A: Training is the bottleneck (~2-3 hours on RTX 2070). If you have 3+ hours
   free, run now and monitor. Otherwise, use `--overnight` before bed - it
   will run at low priority and notify you when done.

Q: What if it crashes?
A: Re-run with `--resume`. The script tracks progress and will skip completed
   steps. Training also saves checkpoints every 5 epochs.

Usage:
    python scripts/run_vme_pipeline.py                    # Full pipeline
    python scripts/run_vme_pipeline.py --dry-run          # Show plan without running
    python scripts/run_vme_pipeline.py --resume           # Continue from last checkpoint
    python scripts/run_vme_pipeline.py --step train       # Run only training step
    python scripts/run_vme_pipeline.py --overnight        # Optimize for unattended run
    python scripts/run_vme_pipeline.py --help             # Show all options
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent

# Time estimates for RTX 2070 8GB (in minutes)
TIME_ESTIMATES = {
    "download": {"min": 5, "max": 15, "description": "1.6 GB download, depends on internet speed"},
    "extract": {"min": 2, "max": 5, "description": "CPU-bound extraction and validation"},
    "train": {"min": 120, "max": 180, "description": "30 epochs on RTX 2070, GPU at 90-100%"},
    "validate": {"min": 5, "max": 10, "description": "Quick inference pass on validation set"},
}

STEPS = ["download", "extract", "train", "validate"]

STATE_FILE = "pipeline_state.json"
LOG_FILE = "pipeline.log"

# Box drawing characters for nice output
BOX_TL = "\u2554"  # Top-left
BOX_TR = "\u2557"  # Top-right
BOX_BL = "\u255a"  # Bottom-left
BOX_BR = "\u255d"  # Bottom-right
BOX_H = "\u2550"   # Horizontal
BOX_V = "\u2551"   # Vertical
BOX_ML = "\u2560"  # Middle-left
BOX_MR = "\u2563"  # Middle-right

# Status indicators
STATUS_DONE = "[DONE]"
STATUS_RUNNING = "[RUNNING]"
STATUS_PENDING = "[PENDING]"
STATUS_FAILED = "[FAILED]"
STATUS_SKIPPED = "[SKIPPED]"


# ---------------------------------------------------------------------------
# State management
# ---------------------------------------------------------------------------

def get_state_path(output_dir: str) -> Path:
    """Get path to pipeline state file."""
    return Path(output_dir) / STATE_FILE


def get_log_path(output_dir: str) -> Path:
    """Get path to pipeline log file."""
    return Path(output_dir) / LOG_FILE


def load_state(output_dir: str) -> dict[str, Any]:
    """Load pipeline state from checkpoint file.

    Returns default state if file doesn't exist.
    """
    state_path = get_state_path(output_dir)
    if state_path.exists():
        try:
            with open(state_path) as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            pass

    return {
        "version": 1,
        "started_at": None,
        "completed_at": None,
        "steps": {
            "download": {"status": "pending", "started_at": None, "completed_at": None, "error": None},
            "extract": {"status": "pending", "started_at": None, "completed_at": None, "error": None},
            "train": {"status": "pending", "started_at": None, "completed_at": None, "error": None},
            "validate": {"status": "pending", "started_at": None, "completed_at": None, "error": None},
        },
        "config": {},
    }


def save_state(state: dict[str, Any], output_dir: str) -> None:
    """Save pipeline state to checkpoint file."""
    state_path = get_state_path(output_dir)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    with open(state_path, "w") as f:
        json.dump(state, f, indent=2, default=str)


def is_step_complete(state: dict[str, Any], step: str) -> bool:
    """Check if a step is already complete."""
    return state.get("steps", {}).get(step, {}).get("status") == "completed"


def mark_step_started(state: dict[str, Any], step: str) -> None:
    """Mark a step as started."""
    if "steps" not in state:
        state["steps"] = {}
    if step not in state["steps"]:
        state["steps"][step] = {}
    state["steps"][step]["status"] = "running"
    state["steps"][step]["started_at"] = datetime.datetime.now().isoformat()
    state["steps"][step]["error"] = None


def mark_step_completed(state: dict[str, Any], step: str, duration_seconds: float = 0) -> None:
    """Mark a step as completed."""
    state["steps"][step]["status"] = "completed"
    state["steps"][step]["completed_at"] = datetime.datetime.now().isoformat()
    state["steps"][step]["duration_seconds"] = duration_seconds


def mark_step_failed(state: dict[str, Any], step: str, error: str) -> None:
    """Mark a step as failed."""
    state["steps"][step]["status"] = "failed"
    state["steps"][step]["error"] = error
    state["steps"][step]["failed_at"] = datetime.datetime.now().isoformat()


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

class PipelineLogger:
    """Logger that writes to both console and file."""

    def __init__(self, log_path: Path, verbose: bool = False):
        self.log_path = log_path
        self.verbose = verbose
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(self.log_path, "a")
        self._write_header()

    def _write_header(self) -> None:
        """Write session header to log."""
        timestamp = datetime.datetime.now().isoformat()
        self._file.write(f"\n{'=' * 60}\n")
        self._file.write(f"Pipeline session started: {timestamp}\n")
        self._file.write(f"{'=' * 60}\n\n")
        self._file.flush()

    def log(self, message: str, level: str = "INFO") -> None:
        """Log a message."""
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        log_line = f"[{timestamp}] [{level}] {message}"

        # Always write to file
        self._file.write(log_line + "\n")
        self._file.flush()

        # Print to console based on verbosity
        if level == "ERROR":
            print(f"  ERROR: {message}")
        elif level == "WARNING":
            print(f"  Warning: {message}")
        elif self.verbose or level in ("INFO", "STEP"):
            print(f"  {message}")

    def subprocess_output(self, line: str) -> None:
        """Log subprocess output."""
        self._file.write(line)
        self._file.flush()
        if self.verbose:
            print(line, end="")

    def close(self) -> None:
        """Close log file."""
        self._file.close()


# ---------------------------------------------------------------------------
# GPU detection
# ---------------------------------------------------------------------------

def detect_gpu() -> dict[str, Any]:
    """Detect GPU and report info."""
    result = {
        "available": False,
        "name": "None",
        "vram_gb": 0,
        "cuda_version": None,
    }

    try:
        import torch
        if torch.cuda.is_available():
            result["available"] = True
            result["name"] = torch.cuda.get_device_name(0)
            vram_bytes = torch.cuda.get_device_properties(0).total_memory
            result["vram_gb"] = round(vram_bytes / (1024**3), 1)
            result["cuda_version"] = torch.version.cuda
    except ImportError:
        pass

    return result


def check_disk_space(path: str, required_gb: float = 5.0) -> tuple[bool, float]:
    """Check available disk space.

    Returns:
        Tuple of (has_enough, available_gb)
    """
    check_path = path
    while not os.path.exists(check_path):
        check_path = os.path.dirname(check_path) or "/"

    stat = shutil.disk_usage(check_path)
    available_gb = stat.free / (1024**3)
    return (available_gb >= required_gb, round(available_gb, 1))


# ---------------------------------------------------------------------------
# Time formatting
# ---------------------------------------------------------------------------

def format_duration(minutes: float) -> str:
    """Format duration in human-readable form."""
    if minutes < 60:
        return f"{minutes:.0f} minutes"
    hours = minutes / 60
    if hours < 2:
        return f"{hours:.1f} hour"
    return f"{hours:.1f} hours"


def format_eta(start_time: float, progress: float) -> str:
    """Calculate and format ETA based on progress."""
    if progress <= 0:
        return "calculating..."

    elapsed = time.time() - start_time
    total_estimated = elapsed / progress
    remaining = total_estimated - elapsed

    if remaining < 60:
        return f"{remaining:.0f}s remaining"
    elif remaining < 3600:
        return f"{remaining/60:.0f}m remaining"
    else:
        return f"{remaining/3600:.1f}h remaining"


def get_total_time_estimate() -> tuple[int, int]:
    """Get total estimated time in minutes (min, max)."""
    min_total = sum(e["min"] for e in TIME_ESTIMATES.values())
    max_total = sum(e["max"] for e in TIME_ESTIMATES.values())
    return (min_total, max_total)


# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------

def print_box(lines: list[str], width: int = 64) -> None:
    """Print text in a box."""
    inner_width = width - 2
    print(f"{BOX_TL}{BOX_H * inner_width}{BOX_TR}")
    for line in lines:
        padded = line.center(inner_width)
        print(f"{BOX_V}{padded}{BOX_V}")
    print(f"{BOX_BL}{BOX_H * inner_width}{BOX_BR}")


def print_header(gpu_info: dict[str, Any]) -> None:
    """Print pipeline header with system info."""
    min_time, max_time = get_total_time_estimate()

    lines = [
        "VME Fine-Tuning Pipeline - detr_geo",
        "",
    ]

    if gpu_info["available"]:
        lines.append(f"GPU: {gpu_info['name']} ({gpu_info['vram_gb']} GB)")
    else:
        lines.append("GPU: Not detected (CPU training will be slow)")

    lines.append(f"Estimated total time: {format_duration(min_time)} - {format_duration(max_time)}")
    lines.append("Disk space required: ~5 GB")

    print_box(lines)


def print_plan(state: dict[str, Any], step_filter: str | None = None) -> None:
    """Print execution plan showing all steps."""
    print()

    step_num = 1
    total_steps = len(STEPS) if step_filter in (None, "all") else 1

    for step in STEPS:
        if step_filter and step_filter != "all" and step_filter != step:
            continue

        est = TIME_ESTIMATES[step]
        step_state = state.get("steps", {}).get(step, {})
        status = step_state.get("status", "pending")

        # Determine status display
        if status == "completed":
            status_str = STATUS_DONE
            duration = step_state.get("duration_seconds", 0)
            if duration:
                status_str += f" ({duration/60:.1f}m)"
        elif status == "running":
            status_str = STATUS_RUNNING
        elif status == "failed":
            status_str = STATUS_FAILED
        else:
            status_str = STATUS_PENDING

        # Step header
        step_title = {
            "download": "Download VME Dataset",
            "extract": "Extract & Validate",
            "train": "Fine-tune RF-DETR Medium",
            "validate": "Validate on Test Set",
        }.get(step, step.title())

        print(f"Step {step_num}/{total_steps}: {step_title}")
        print(f"  Status: {status_str}")
        print(f"  Estimate: {format_duration(est['min'])} - {format_duration(est['max'])}")

        # Extra notes for training
        if step == "train":
            print("  Note: GPU will be at 90-100%. Computer usable for light tasks.")

        if status == "failed" and step_state.get("error"):
            print(f"  Error: {step_state['error'][:100]}")

        print()
        step_num += 1


# ---------------------------------------------------------------------------
# Step execution
# ---------------------------------------------------------------------------

def run_subprocess(
    cmd: list[str],
    logger: PipelineLogger,
    cwd: str | None = None,
    env: dict[str, str] | None = None,
) -> tuple[int, str]:
    """Run a subprocess and capture output.

    Returns:
        Tuple of (return_code, combined_output)
    """
    full_env = os.environ.copy()
    if env:
        full_env.update(env)

    logger.log(f"Running: {' '.join(cmd)}", "DEBUG")

    output_lines = []

    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            cwd=cwd or str(PROJECT_ROOT),
            env=full_env,
            text=True,
            bufsize=1,
        )

        for line in process.stdout:
            output_lines.append(line)
            logger.subprocess_output(line)

        process.wait()
        return (process.returncode, "".join(output_lines))

    except Exception as e:
        error_msg = f"Subprocess error: {e}"
        logger.log(error_msg, "ERROR")
        return (1, error_msg)


def run_download_step(
    args: argparse.Namespace,
    state: dict[str, Any],
    logger: PipelineLogger,
) -> bool:
    """Run the download step."""
    logger.log("Starting download step", "STEP")

    cmd = [
        sys.executable,
        str(SCRIPT_DIR / "download_vme.py"),
        "--output_dir", args.dataset_dir,
        "--accept_license",
        "--skip_existing",
    ]

    returncode, output = run_subprocess(cmd, logger)

    if returncode != 0:
        logger.log(f"Download failed with code {returncode}", "ERROR")
        return False

    logger.log("Download step completed", "STEP")
    return True


def run_extract_step(
    args: argparse.Namespace,
    state: dict[str, Any],
    logger: PipelineLogger,
) -> bool:
    """Run the extraction step (part of download script with --verify_only)."""
    logger.log("Starting extraction/validation step", "STEP")

    # Check if dataset already exists
    dataset_path = Path(args.dataset_dir)
    train_dir = dataset_path / "train"
    valid_dir = dataset_path / "valid"

    if train_dir.exists() and valid_dir.exists():
        logger.log("Dataset already extracted, validating...", "INFO")

        cmd = [
            sys.executable,
            str(SCRIPT_DIR / "download_vme.py"),
            "--output_dir", args.dataset_dir,
            "--verify_only",
        ]

        returncode, output = run_subprocess(cmd, logger)

        if returncode != 0:
            logger.log("Dataset validation failed", "ERROR")
            return False
    else:
        logger.log("Dataset directories not found, extraction may still be needed", "WARNING")

    logger.log("Extract/validation step completed", "STEP")
    return True


def run_train_step(
    args: argparse.Namespace,
    state: dict[str, Any],
    logger: PipelineLogger,
) -> bool:
    """Run the training step."""
    logger.log("Starting training step", "STEP")
    logger.log("This will take 2-3 hours on RTX 2070. GPU at 90-100%.", "INFO")

    cmd = [
        sys.executable,
        str(SCRIPT_DIR / "train_vme.py"),
        "--dataset_dir", args.dataset_dir,
        "--model", "medium",
        "--epochs", str(args.epochs),
        "--batch_size", "1",
        "--grad_accumulation_steps", "16",
        "--output_dir", str(Path(args.output_dir) / "training"),
    ]

    if args.resume:
        # Look for existing checkpoint
        training_dir = Path(args.output_dir) / "training"
        checkpoints = sorted(training_dir.glob("*.pth")) if training_dir.exists() else []
        if checkpoints:
            cmd.extend(["--resume", str(checkpoints[-1])])
            logger.log(f"Resuming from checkpoint: {checkpoints[-1]}", "INFO")

    returncode, output = run_subprocess(cmd, logger)

    if returncode != 0:
        logger.log(f"Training failed with code {returncode}", "ERROR")

        # Check for OOM
        if "cuda" in output.lower() and ("memory" in output.lower() or "oom" in output.lower()):
            logger.log("GPU out of memory. Try reducing batch_size or use smaller model.", "ERROR")

        return False

    logger.log("Training step completed", "STEP")
    return True


def run_validate_step(
    args: argparse.Namespace,
    state: dict[str, Any],
    logger: PipelineLogger,
) -> bool:
    """Run the validation step."""
    logger.log("Starting validation step", "STEP")

    # Find best checkpoint
    training_dir = Path(args.output_dir) / "training"
    best_checkpoint = training_dir / "vme_medium_best.pth"

    if not best_checkpoint.exists():
        # Fall back to any checkpoint
        checkpoints = sorted(training_dir.glob("*.pth")) if training_dir.exists() else []
        if checkpoints:
            best_checkpoint = checkpoints[-1]
        else:
            logger.log("No checkpoint found for validation", "ERROR")
            return False

    cmd = [
        sys.executable,
        str(SCRIPT_DIR / "validate_vme.py"),
        "--checkpoint", str(best_checkpoint),
        "--dataset_dir", args.dataset_dir,
        "--output_dir", str(Path(args.output_dir) / "evaluation"),
        "--model", "medium",
    ]

    returncode, output = run_subprocess(cmd, logger)

    if returncode != 0:
        logger.log(f"Validation failed with code {returncode}", "ERROR")
        return False

    logger.log("Validation step completed", "STEP")
    return True


# ---------------------------------------------------------------------------
# Overnight mode
# ---------------------------------------------------------------------------

def setup_overnight_mode() -> None:
    """Configure process for overnight/background execution."""
    # Set low CPU priority (nice)
    try:
        os.nice(10)
    except (OSError, AttributeError):
        pass  # nice not available on all platforms

    # Reduce Python GC frequency for long runs
    import gc
    gc.set_threshold(10000, 50, 50)


def send_notification(title: str, message: str) -> None:
    """Send desktop notification if available."""
    try:
        # Linux (notify-send)
        subprocess.run(
            ["notify-send", "-u", "normal", title, message],
            capture_output=True,
            timeout=5,
        )
    except (subprocess.SubprocessError, FileNotFoundError):
        pass  # Notification not available, that's fine


# ---------------------------------------------------------------------------
# Signal handling
# ---------------------------------------------------------------------------

class GracefulExit:
    """Handle graceful shutdown on SIGINT/SIGTERM."""

    def __init__(self):
        self.should_exit = False
        self.state = None
        self.output_dir = None

        signal.signal(signal.SIGINT, self._handler)
        signal.signal(signal.SIGTERM, self._handler)

    def _handler(self, signum, frame):
        print("\n\n  Interrupt received. Saving state...")
        self.should_exit = True

        if self.state and self.output_dir:
            save_state(self.state, self.output_dir)
            print(f"  State saved. Re-run with --resume to continue.")

        sys.exit(1)

    def set_state(self, state: dict[str, Any], output_dir: str) -> None:
        """Set state for saving on interrupt."""
        self.state = state
        self.output_dir = output_dir


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="VME Fine-Tuning Pipeline - orchestrate download, training, and validation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("Usage:")[1] if "Usage:" in __doc__ else "",
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help="Continue from last checkpoint, skipping completed steps",
    )
    parser.add_argument(
        "--step",
        type=str,
        choices=["download", "extract", "train", "validate", "all"],
        default="all",
        help="Run specific step only (default: all)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would happen without running anything",
    )
    parser.add_argument(
        "--overnight",
        action="store_true",
        help="Optimize for unattended run (low priority, notifications)",
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default="vme_dataset",
        help="Directory for VME dataset (default: vme_dataset)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="vme_pipeline_output",
        help="Directory for all outputs (default: vme_pipeline_output)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
        help="Number of training epochs (default: 30)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show all subprocess output",
    )
    parser.add_argument(
        "--yes", "-y",
        action="store_true",
        help="Skip confirmation prompt",
    )

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the VME fine-tuning pipeline."""
    args = parse_args(argv)

    # Setup graceful exit handler
    graceful = GracefulExit()

    # Detect hardware
    gpu_info = detect_gpu()

    # Check disk space
    has_space, available_gb = check_disk_space(args.output_dir)

    # Load or initialize state
    state = load_state(args.output_dir)
    state["config"] = {
        "dataset_dir": args.dataset_dir,
        "output_dir": args.output_dir,
        "epochs": args.epochs,
    }
    graceful.set_state(state, args.output_dir)

    # Print header
    print()
    print_header(gpu_info)

    # Warnings
    if not has_space:
        print(f"\n  WARNING: Only {available_gb} GB free. Need ~5 GB.")
        print("  Pipeline may fail during extraction or training.")

    if not gpu_info["available"]:
        print("\n  WARNING: No GPU detected. Training on CPU will be extremely slow")
        print("  (estimated 40+ hours). Consider using a GPU-equipped machine.")

    # Show plan
    print_plan(state, args.step if args.step != "all" else None)

    # Dry run exits here
    if args.dry_run:
        print("  [DRY RUN] Would execute the steps above.")
        print(f"  [DRY RUN] State would be saved to: {get_state_path(args.output_dir)}")
        print(f"  [DRY RUN] Logs would be written to: {get_log_path(args.output_dir)}")
        return 0

    # Confirmation prompt
    if not args.yes and not args.resume:
        try:
            answer = input("Press Enter to start, or Ctrl+C to cancel... ")
        except (EOFError, KeyboardInterrupt):
            print("\n  Cancelled.")
            return 1

    # Setup overnight mode if requested
    if args.overnight:
        print("\n  Overnight mode enabled:")
        print("    - Process priority lowered")
        print("    - Desktop notification when complete")
        print("    - Full logs saved")
        setup_overnight_mode()
        args.verbose = True

    # Initialize logging
    logger = PipelineLogger(get_log_path(args.output_dir), verbose=args.verbose)
    logger.log("Pipeline started", "INFO")
    logger.log(f"GPU: {gpu_info['name'] if gpu_info['available'] else 'None'}", "INFO")
    logger.log(f"Output directory: {args.output_dir}", "INFO")

    # Mark pipeline started
    if not state.get("started_at"):
        state["started_at"] = datetime.datetime.now().isoformat()
    save_state(state, args.output_dir)

    # Step execution map
    step_runners = {
        "download": run_download_step,
        "extract": run_extract_step,
        "train": run_train_step,
        "validate": run_validate_step,
    }

    # Determine which steps to run
    if args.step == "all":
        steps_to_run = STEPS
    else:
        steps_to_run = [args.step]

    # Run steps
    pipeline_start = time.time()
    success = True

    for step in steps_to_run:
        # Skip if resuming and already complete
        if args.resume and is_step_complete(state, step):
            print(f"\n  Skipping {step} (already complete)")
            logger.log(f"Skipping {step} (already complete)", "INFO")
            continue

        # Check for graceful exit
        if graceful.should_exit:
            break

        # Run step
        print(f"\n  Starting {step}...")
        step_start = time.time()
        mark_step_started(state, step)
        save_state(state, args.output_dir)

        runner = step_runners[step]
        step_success = runner(args, state, logger)

        step_duration = time.time() - step_start

        if step_success:
            mark_step_completed(state, step, step_duration)
            print(f"  {step} completed in {step_duration/60:.1f} minutes")
        else:
            mark_step_failed(state, step, "Step execution failed")
            print(f"  {step} FAILED after {step_duration/60:.1f} minutes")
            success = False
            break

        save_state(state, args.output_dir)

    # Pipeline complete
    pipeline_duration = time.time() - pipeline_start
    state["completed_at"] = datetime.datetime.now().isoformat()
    state["total_duration_seconds"] = pipeline_duration
    save_state(state, args.output_dir)

    logger.log(f"Pipeline finished in {pipeline_duration/60:.1f} minutes", "INFO")
    logger.close()

    # Final summary
    print()
    print("=" * 64)
    if success:
        print("  PIPELINE COMPLETE")
    else:
        print("  PIPELINE FAILED")
    print("=" * 64)
    print(f"  Duration: {pipeline_duration/60:.1f} minutes ({pipeline_duration/3600:.2f} hours)")
    print(f"  State: {get_state_path(args.output_dir)}")
    print(f"  Logs: {get_log_path(args.output_dir)}")

    if success:
        training_dir = Path(args.output_dir) / "training"
        eval_dir = Path(args.output_dir) / "evaluation"
        print(f"\n  Outputs:")
        print(f"    Training: {training_dir}")
        print(f"    Evaluation: {eval_dir}")
        print(f"\n  Next steps:")
        print(f"    - Review evaluation report: {eval_dir}/evaluation_report.md")
        print(f"    - Test on your own images with the trained model")
    else:
        print(f"\n  To resume from failure:")
        print(f"    python scripts/run_vme_pipeline.py --resume")

    # Send notification if overnight mode
    if args.overnight:
        status = "completed successfully" if success else "failed"
        send_notification(
            "VME Pipeline Complete",
            f"Fine-tuning {status} in {pipeline_duration/3600:.1f} hours",
        )

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
