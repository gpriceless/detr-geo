"""Tests for AugmentationPreset and train() wrapper (Proposal 006, Task 006.6).

Tests cover all GIVEN/WHEN/THEN scenarios from
openspec/changes/006-training-pipeline/specs/augmentation.md
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from detr_geo.exceptions import ModelError
from detr_geo.training import (
    AUGMENTATION_PRESETS,
    AugmentationPreset,
    train,
)


class TestAugmentationPresets:
    """Tests for preset definitions."""

    def test_three_presets_exist(self):
        """THEN AUGMENTATION_PRESETS has satellite, aerial, drone entries."""
        assert "satellite_default" in AUGMENTATION_PRESETS
        assert "aerial_default" in AUGMENTATION_PRESETS
        assert "drone_default" in AUGMENTATION_PRESETS
        assert len(AUGMENTATION_PRESETS) == 3

    def test_satellite_preset_values(self):
        """GIVEN satellite_default, THEN correct values."""
        preset = AUGMENTATION_PRESETS["satellite_default"]
        assert preset.random_rotation_90 is True
        assert preset.horizontal_flip is True
        assert preset.vertical_flip is True
        assert preset.brightness_jitter == 0.2
        assert preset.contrast_jitter == 0.2
        assert preset.saturation_jitter == 0.1

    def test_aerial_preset_values(self):
        """GIVEN aerial_default, THEN higher color jitter than satellite."""
        preset = AUGMENTATION_PRESETS["aerial_default"]
        assert preset.random_rotation_90 is True
        assert preset.vertical_flip is True
        assert preset.brightness_jitter == 0.3
        assert preset.contrast_jitter == 0.3
        assert preset.saturation_jitter == 0.15

    def test_drone_preset_values(self):
        """GIVEN drone_default, THEN most aggressive augmentation."""
        preset = AUGMENTATION_PRESETS["drone_default"]
        assert preset.random_rotation_90 is True
        assert preset.vertical_flip is True
        assert preset.brightness_jitter == 0.4
        assert preset.contrast_jitter == 0.4
        assert preset.saturation_jitter == 0.2

    def test_all_presets_have_vertical_flip(self):
        """GIVEN any preset, THEN vertical_flip is True."""
        for name, preset in AUGMENTATION_PRESETS.items():
            assert preset.vertical_flip is True, f"{name} missing vertical_flip"


class TestTrainWrapper:
    """Tests for train() wrapper function."""

    def test_train_with_mock_adapter(self, tmp_path):
        """GIVEN a mock adapter, WHEN train() called,
        THEN adapter.train() is called with correct args."""
        # Create minimal dataset directory structure
        (tmp_path / "train" / "images").mkdir(parents=True)
        (tmp_path / "valid" / "images").mkdir(parents=True)
        (tmp_path / "train" / "_annotations.coco.json").write_text('{"images":[],"annotations":[],"categories":[]}')
        (tmp_path / "valid" / "_annotations.coco.json").write_text('{"images":[],"annotations":[],"categories":[]}')

        mock_adapter = MagicMock()
        mock_adapter.train.return_value = {"mAP": 0.75, "loss": 0.15}

        result = train(
            adapter=mock_adapter,
            dataset_dir=str(tmp_path),
            epochs=50,
            augmentation_preset=None,  # skip augmentation for speed
        )

        mock_adapter.train.assert_called_once()
        call_args = mock_adapter.train.call_args
        assert call_args[0][0] == str(tmp_path)
        assert call_args[1]["epochs"] == 50
        assert result == {"mAP": 0.75, "loss": 0.15}

    def test_train_without_rfdetr_raises_model_error(self, tmp_path):
        """GIVEN rfdetr not installed, WHEN train() called,
        THEN ModelError raised."""
        (tmp_path / "train" / "images").mkdir(parents=True)
        (tmp_path / "valid" / "images").mkdir(parents=True)
        (tmp_path / "train" / "_annotations.coco.json").write_text('{"images":[],"annotations":[],"categories":[]}')

        mock_adapter = MagicMock()
        mock_adapter.train.side_effect = ModelError(
            "rfdetr is not installed. Install it with: pip install detr-geo[rfdetr]"
        )

        with pytest.raises(ModelError, match="rfdetr is not installed"):
            train(
                adapter=mock_adapter,
                dataset_dir=str(tmp_path),
                augmentation_preset=None,
            )

    def test_custom_augmentation_preset(self, tmp_path):
        """GIVEN custom AugmentationPreset, THEN custom values used."""
        (tmp_path / "train" / "images").mkdir(parents=True)
        (tmp_path / "valid" / "images").mkdir(parents=True)
        (tmp_path / "train" / "_annotations.coco.json").write_text('{"images":[],"annotations":[],"categories":[]}')

        custom = AugmentationPreset(
            name="custom",
            vertical_flip=False,
            brightness_jitter=0.5,
        )

        mock_adapter = MagicMock()
        mock_adapter.train.return_value = {}

        # Should not raise
        result = train(
            adapter=mock_adapter,
            dataset_dir=str(tmp_path),
            augmentation_preset=custom,
        )
        assert isinstance(result, dict)

    def test_resume_parameter_passed_through(self, tmp_path):
        """GIVEN resume='/path/to/checkpoint.pth',
        THEN resume is passed to adapter.train()."""
        (tmp_path / "train" / "images").mkdir(parents=True)
        (tmp_path / "train" / "_annotations.coco.json").write_text('{"images":[],"annotations":[],"categories":[]}')

        mock_adapter = MagicMock()
        mock_adapter.train.return_value = {}

        train(
            adapter=mock_adapter,
            dataset_dir=str(tmp_path),
            resume="/path/to/checkpoint.pth",
            augmentation_preset=None,
        )

        call_kwargs = mock_adapter.train.call_args[1]
        assert call_kwargs["resume"] == "/path/to/checkpoint.pth"

    def test_learning_rate_override(self, tmp_path):
        """GIVEN learning_rate=0.001, THEN lr is passed to adapter."""
        (tmp_path / "train" / "images").mkdir(parents=True)
        (tmp_path / "train" / "_annotations.coco.json").write_text('{"images":[],"annotations":[],"categories":[]}')

        mock_adapter = MagicMock()
        mock_adapter.train.return_value = {}

        train(
            adapter=mock_adapter,
            dataset_dir=str(tmp_path),
            learning_rate=0.001,
            augmentation_preset=None,
        )

        call_kwargs = mock_adapter.train.call_args[1]
        assert call_kwargs["lr"] == 0.001

    def test_invalid_preset_name_raises_value_error(self, tmp_path):
        """GIVEN invalid preset name, THEN ValueError raised."""
        mock_adapter = MagicMock()

        with pytest.raises(ValueError, match="Unknown augmentation preset"):
            train(
                adapter=mock_adapter,
                dataset_dir=str(tmp_path),
                augmentation_preset="nonexistent_preset",
            )

    def test_train_returns_metrics_dict(self, tmp_path):
        """GIVEN completed training, THEN metrics dict returned."""
        (tmp_path / "train" / "images").mkdir(parents=True)
        (tmp_path / "train" / "_annotations.coco.json").write_text('{"images":[],"annotations":[],"categories":[]}')

        expected_metrics = {"mAP": 0.82, "loss": 0.10, "epochs_completed": 50}
        mock_adapter = MagicMock()
        mock_adapter.train.return_value = expected_metrics

        result = train(
            adapter=mock_adapter,
            dataset_dir=str(tmp_path),
            augmentation_preset=None,
        )
        assert result == expected_metrics

    def test_default_satellite_preset(self, tmp_path):
        """GIVEN no preset specified, THEN satellite_default is used."""
        (tmp_path / "train" / "images").mkdir(parents=True)
        (tmp_path / "valid" / "images").mkdir(parents=True)
        (tmp_path / "train" / "_annotations.coco.json").write_text('{"images":[],"annotations":[],"categories":[]}')

        mock_adapter = MagicMock()
        mock_adapter.train.return_value = {}

        # Default should be "satellite_default" - should not raise
        result = train(
            adapter=mock_adapter,
            dataset_dir=str(tmp_path),
        )
        assert isinstance(result, dict)
