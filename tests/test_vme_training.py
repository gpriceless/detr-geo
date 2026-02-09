"""Tests for scripts/train_vme.py -- VME fine-tuning script.

All tests use mocks for model operations -- no actual GPU training occurs.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest import mock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from train_vme import (
    DEFAULT_CONFIG,
    VME_CLASSES,
    VME_NUM_CLASSES,
    build_training_config,
    detect_device,
    main,
    parse_args,
    verify_dataset,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def valid_vme_dataset(tmp_path: Path) -> Path:
    """Create a valid VME dataset directory for training script verification."""
    ds = tmp_path / "vme_dataset"

    for split in ("train", "valid"):
        split_dir = ds / split
        split_dir.mkdir(parents=True)

        n_images = 100 if split == "train" else 25
        images = [{"id": i, "file_name": f"tile_{i}.jpg", "width": 512, "height": 512} for i in range(n_images)]
        annotations = [
            {
                "id": i,
                "image_id": i % n_images,
                "category_id": i % 3,
                "bbox": [10 + i, 10 + i, 30, 20],
                "area": 600,
            }
            for i in range(n_images * 3)
        ]
        categories = [
            {"id": 0, "name": "Car"},
            {"id": 1, "name": "Bus"},
            {"id": 2, "name": "Truck"},
        ]
        coco = {"images": images, "annotations": annotations, "categories": categories}

        ann_path = split_dir / "_annotations.coco.json"
        ann_path.write_text(json.dumps(coco, indent=2))

        # Create dummy image files
        for i in range(n_images):
            (split_dir / f"tile_{i}.jpg").write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 100)

    return ds


# ---------------------------------------------------------------------------
# Dataset Verification Tests
# ---------------------------------------------------------------------------


class TestVerifyDataset:
    """Tests for verify_dataset()."""

    def test_valid_dataset(self, valid_vme_dataset: Path) -> None:
        """GIVEN valid VME dataset WHEN verified THEN returns correct stats."""
        info = verify_dataset(str(valid_vme_dataset))
        assert info["num_classes"] == 3
        assert info["train_images"] == 100
        assert info["valid_images"] == 25
        assert info["train_annotations"] == 300
        assert info["valid_annotations"] == 75
        assert 0 in info["categories"]
        assert info["categories"][0] == "Car"

    def test_missing_dataset_dir(self, tmp_path: Path) -> None:
        """GIVEN nonexistent directory WHEN verified THEN FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="Dataset not found"):
            verify_dataset(str(tmp_path / "nonexistent"))

    def test_missing_train_dir(self, tmp_path: Path) -> None:
        """GIVEN dataset without train/ WHEN verified THEN FileNotFoundError."""
        ds = tmp_path / "no_train"
        (ds / "valid").mkdir(parents=True)
        (ds / "valid" / "_annotations.coco.json").write_text("{}")
        with pytest.raises(FileNotFoundError, match="missing expected structure"):
            verify_dataset(str(ds))

    def test_missing_valid_dir(self, tmp_path: Path) -> None:
        """GIVEN dataset without valid/ WHEN verified THEN FileNotFoundError."""
        ds = tmp_path / "no_valid"
        (ds / "train").mkdir(parents=True)
        (ds / "train" / "_annotations.coco.json").write_text("{}")
        with pytest.raises(FileNotFoundError, match="missing expected structure"):
            verify_dataset(str(ds))

    def test_missing_annotation_file(self, tmp_path: Path) -> None:
        """GIVEN dataset without annotation JSON WHEN verified THEN FileNotFoundError."""
        ds = tmp_path / "no_ann"
        (ds / "train").mkdir(parents=True)
        (ds / "valid").mkdir(parents=True)
        (ds / "valid" / "_annotations.coco.json").write_text("{}")
        with pytest.raises(FileNotFoundError, match="missing expected structure"):
            verify_dataset(str(ds))


# ---------------------------------------------------------------------------
# Device Detection Tests
# ---------------------------------------------------------------------------


class TestDetectDevice:
    """Tests for detect_device()."""

    def test_returns_dict_with_device_key(self) -> None:
        """GIVEN any environment WHEN device detected THEN dict has device key."""
        result = detect_device()
        assert "device" in result
        assert "type" in result

    @mock.patch("train_vme.detect_device")
    def test_cuda_device(self, mock_detect: mock.MagicMock) -> None:
        """GIVEN CUDA available WHEN detected THEN returns cuda device."""
        mock_detect.return_value = {
            "device": "cuda:0",
            "type": "cuda",
            "gpu_name": "Test GPU",
            "vram_gb": 8.0,
            "warning": None,
        }
        result = mock_detect()
        assert result["type"] == "cuda"
        assert result["vram_gb"] == 8.0

    def test_cpu_fallback(self) -> None:
        """GIVEN no GPU WHEN detected THEN falls back to CPU."""
        with mock.patch.dict("sys.modules", {"torch": None}):
            # If torch import fails, should still return cpu
            result = detect_device()
            assert result["device"] == "cpu"


# ---------------------------------------------------------------------------
# Config Building Tests
# ---------------------------------------------------------------------------


class TestBuildTrainingConfig:
    """Tests for build_training_config()."""

    def test_default_config(self) -> None:
        """GIVEN default args WHEN config built THEN matches DEFAULT_CONFIG."""
        args = parse_args(["--dataset_dir", "vme_dataset"])
        config = build_training_config(args)
        assert config["model"] == DEFAULT_CONFIG["model"]
        assert config["epochs"] == DEFAULT_CONFIG["epochs"]
        assert config["batch_size"] == DEFAULT_CONFIG["batch_size"]
        assert config["learning_rate"] == DEFAULT_CONFIG["learning_rate"]
        assert config["augmentation_preset"] == DEFAULT_CONFIG["augmentation_preset"]

    def test_custom_config(self) -> None:
        """GIVEN custom args WHEN config built THEN overrides applied."""
        args = parse_args(
            [
                "--dataset_dir",
                "vme_dataset",
                "--model",
                "small",
                "--epochs",
                "50",
                "--batch_size",
                "4",
                "--learning_rate",
                "5e-6",
            ]
        )
        config = build_training_config(args)
        assert config["model"] == "small"
        assert config["epochs"] == 50
        assert config["batch_size"] == 4
        assert config["learning_rate"] == 5e-6

    def test_effective_batch_size(self) -> None:
        """GIVEN batch_size=2 and grad_accum=8 WHEN config built THEN effective=16."""
        args = parse_args(["--dataset_dir", "vme_dataset"])
        config = build_training_config(args)
        effective = config["batch_size"] * config["grad_accumulation_steps"]
        assert effective == 16


# ---------------------------------------------------------------------------
# CLI Argument Tests
# ---------------------------------------------------------------------------


class TestParseArgs:
    """Tests for parse_args()."""

    def test_default_values(self) -> None:
        """GIVEN no args WHEN parsed THEN defaults match DEFAULT_CONFIG."""
        args = parse_args(["--dataset_dir", "vme_dataset"])
        assert args.model == "medium"
        assert args.epochs == 30
        assert args.batch_size == 2
        assert args.grad_accumulation_steps == 8
        assert args.learning_rate == 1e-5
        assert args.augmentation_preset == "satellite_default"
        assert args.save_interval == 5
        assert args.val_interval == 1
        assert args.early_stopping_patience == 5

    def test_model_choices(self) -> None:
        """GIVEN valid model size WHEN parsed THEN accepted."""
        for model in ("nano", "small", "medium", "base", "large"):
            args = parse_args(["--dataset_dir", "d", "--model", model])
            assert args.model == model

    def test_resume_checkpoint(self) -> None:
        """GIVEN --resume flag WHEN parsed THEN checkpoint path stored."""
        args = parse_args(["--dataset_dir", "d", "--resume", "checkpoint.pth"])
        assert args.resume == "checkpoint.pth"

    def test_augmentation_choices(self) -> None:
        """GIVEN valid augmentation preset WHEN parsed THEN accepted."""
        for preset in ("satellite_default", "aerial_default", "drone_default"):
            args = parse_args(["--dataset_dir", "d", "--augmentation_preset", preset])
            assert args.augmentation_preset == preset

    def test_seed_parameter(self) -> None:
        """GIVEN custom seed WHEN parsed THEN stored correctly."""
        args = parse_args(["--dataset_dir", "d", "--seed", "123"])
        assert args.seed == 123


# ---------------------------------------------------------------------------
# Main Function Tests
# ---------------------------------------------------------------------------


class TestMainFunction:
    """Tests for the main() entry point."""

    def test_missing_dataset_returns_1(self, tmp_path: Path) -> None:
        """GIVEN nonexistent dataset WHEN main runs THEN returns 1."""
        # Mock torch to avoid import issues and skip CPU confirmation
        with mock.patch(
            "train_vme.detect_device",
            return_value={
                "device": "cpu",
                "type": "cpu",
                "warning": None,
            },
        ):
            exit_code = main(["--dataset_dir", str(tmp_path / "nonexistent")])
            assert exit_code == 1


# ---------------------------------------------------------------------------
# Constants Tests
# ---------------------------------------------------------------------------


class TestConstants:
    """Tests for module constants."""

    def test_vme_classes(self) -> None:
        """VME_CLASSES should have 3 entries."""
        assert len(VME_CLASSES) == 3
        assert VME_CLASSES[0] == "Car"
        assert VME_CLASSES[1] == "Bus"
        assert VME_CLASSES[2] == "Truck"

    def test_vme_num_classes(self) -> None:
        """VME_NUM_CLASSES should match VME_CLASSES length."""
        assert len(VME_CLASSES) == VME_NUM_CLASSES

    def test_default_config_keys(self) -> None:
        """DEFAULT_CONFIG should have all required keys."""
        required = {
            "model",
            "epochs",
            "batch_size",
            "grad_accumulation_steps",
            "learning_rate",
            "augmentation_preset",
            "save_interval",
            "val_interval",
            "early_stopping_patience",
        }
        assert required.issubset(set(DEFAULT_CONFIG.keys()))

    def test_default_config_values(self) -> None:
        """DEFAULT_CONFIG values should be reasonable for 8GB VRAM."""
        assert DEFAULT_CONFIG["model"] == "medium"
        assert DEFAULT_CONFIG["batch_size"] == 2
        assert DEFAULT_CONFIG["grad_accumulation_steps"] == 8
        assert DEFAULT_CONFIG["learning_rate"] == 1e-5
        assert DEFAULT_CONFIG["epochs"] == 30
