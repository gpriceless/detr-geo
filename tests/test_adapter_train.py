"""Tests for RFDETRAdapter.train() method (Proposal 006, Task 006.5).

Tests verify the adapter pattern: training.py calls adapter.train(),
which delegates to the underlying model.train().
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from detr_geo._adapter import RFDETRAdapter
from detr_geo.exceptions import ModelError


class TestAdapterTrain:
    """Tests for RFDETRAdapter.train() method."""

    def test_train_delegates_to_model(self):
        """GIVEN a loaded adapter, WHEN train() is called,
        THEN it delegates to model.train() with correct args."""
        adapter = RFDETRAdapter(model_size="nano")

        # Mock the model
        mock_model = MagicMock()
        mock_model.train.return_value = {"mAP": 0.75}
        adapter._model = mock_model

        result = adapter.train("/path/to/dataset", epochs=50, batch_size=8)

        mock_model.train.assert_called_once_with(dataset_dir="/path/to/dataset", epochs=50, batch_size=8)
        assert result == {"mAP": 0.75}

    @pytest.mark.skipif(
        __import__("tests.conftest", fromlist=["rfdetr_installed"]).rfdetr_installed(),
        reason="Test assumes rfdetr is NOT installed",
    )
    def test_train_ensures_model_loaded(self):
        """GIVEN an adapter with no model loaded, WHEN train() called,
        THEN _ensure_model is called (which would load model or error)."""
        adapter = RFDETRAdapter(model_size="nano")

        # rfdetr is not installed, so _ensure_model should raise ModelError
        with pytest.raises(ModelError, match="rfdetr is not installed"):
            adapter.train("/path/to/dataset")

    def test_train_passes_kwargs(self):
        """GIVEN extra keyword arguments, WHEN train() called,
        THEN they are passed through to model.train()."""
        adapter = RFDETRAdapter(model_size="nano")
        mock_model = MagicMock()
        mock_model.train.return_value = {}
        adapter._model = mock_model

        adapter.train(
            "/path/to/dataset",
            epochs=100,
            lr=0.001,
            resume="/path/to/checkpoint.pth",
        )

        mock_model.train.assert_called_once_with(
            dataset_dir="/path/to/dataset",
            epochs=100,
            lr=0.001,
            resume="/path/to/checkpoint.pth",
        )

    def test_train_returns_model_result(self):
        """GIVEN model.train() returns metrics dict,
        WHEN adapter.train() completes,
        THEN the same dict is returned."""
        adapter = RFDETRAdapter(model_size="nano")
        expected = {"mAP": 0.82, "loss": 0.15, "epochs_completed": 50}
        mock_model = MagicMock()
        mock_model.train.return_value = expected
        adapter._model = mock_model

        result = adapter.train("/data")
        assert result == expected
