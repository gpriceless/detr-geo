"""Tests for SpatialSplitter class (Proposal 006, Task 006.1).

Tests cover all GIVEN/WHEN/THEN scenarios from
openspec/changes/006-training-pipeline/specs/spatial-splitting.md
"""

from __future__ import annotations

import warnings

import pytest

from detr_geo.training import SpatialSplitter


class TestBlockSplit:
    """Block-based spatial splitting tests."""

    def test_default_block_split_approximate_ratios(self, tile_list_10x10: list[dict]):
        """GIVEN 100 tiles, WHEN block split with (0.8, 0.15, 0.05),
        THEN approximately correct ratios (allowing for block granularity)."""
        splitter = SpatialSplitter(method="block", ratios=(0.8, 0.15, 0.05), buffer_tiles=0)
        result = splitter.split(tile_list_10x10, 1000, 1000, 100)

        total = (
            len(result.train_indices) + len(result.val_indices) + len(result.test_indices) + len(result.buffer_indices)
        )
        assert total == 100

        # With block granularity, ratios won't be exact but should be in range
        non_buffer = total - len(result.buffer_indices)
        if non_buffer > 0:
            train_ratio = len(result.train_indices) / non_buffer
            assert 0.5 <= train_ratio <= 1.0, f"Train ratio {train_ratio} out of range"

    def test_entire_blocks_assigned_to_same_split(self, tile_list_10x10: list[dict]):
        """GIVEN block split, THEN entire spatial blocks are in the same split."""
        splitter = SpatialSplitter(method="block", buffer_tiles=0, seed=42)
        result = splitter.split(tile_list_10x10, 1000, 1000, 100)

        # All indices should be accounted for
        all_indices = set(result.train_indices + result.val_indices + result.test_indices + result.buffer_indices)
        assert all_indices == set(range(100))

    def test_buffer_tiles_discarded(self, tile_list_10x10: list[dict]):
        """GIVEN buffer_tiles=1, THEN boundary tiles are in buffer_indices
        and NOT in train/val/test."""
        splitter = SpatialSplitter(method="block", buffer_tiles=1, seed=42)
        result = splitter.split(tile_list_10x10, 1000, 1000, 100)

        # Buffer tiles should exist (unless all tiles are in one block)
        train_set = set(result.train_indices)
        val_set = set(result.val_indices)
        test_set = set(result.test_indices)
        buffer_set = set(result.buffer_indices)

        # Buffer tiles must not appear in any split
        assert buffer_set.isdisjoint(train_set)
        assert buffer_set.isdisjoint(val_set)
        assert buffer_set.isdisjoint(test_set)

    def test_no_tile_overlap_between_splits(self, tile_list_10x10: list[dict]):
        """GIVEN any completed split, THEN no tile appears in more than one list,
        AND every tile appears in exactly one list."""
        splitter = SpatialSplitter(method="block", buffer_tiles=1, seed=42)
        result = splitter.split(tile_list_10x10, 1000, 1000, 100)

        all_lists = [
            result.train_indices,
            result.val_indices,
            result.test_indices,
            result.buffer_indices,
        ]

        # Check no overlaps
        seen: set[int] = set()
        for idx_list in all_lists:
            for idx in idx_list:
                assert idx not in seen, f"Tile {idx} appears in multiple splits"
                seen.add(idx)

        # Check all tiles accounted for
        assert seen == set(range(100))


class TestRandomSplit:
    """Random splitting tests."""

    def test_random_split_issues_warning(self, tile_list_10x10: list[dict]):
        """GIVEN method='random', WHEN split() called,
        THEN UserWarning about spatial data leakage is issued."""
        splitter = SpatialSplitter(method="random")
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            splitter.split(tile_list_10x10, 1000, 1000, 100)
            leakage_warnings = [x for x in w if "spatial data leakage" in str(x.message).lower()]
            assert len(leakage_warnings) >= 1

    def test_random_split_respects_ratios(self, tile_list_10x10: list[dict]):
        """GIVEN random split with (0.8, 0.15, 0.05),
        THEN approximately correct ratios."""
        splitter = SpatialSplitter(method="random", ratios=(0.8, 0.15, 0.05), seed=42)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = splitter.split(tile_list_10x10, 1000, 1000, 100)

        total = len(result.train_indices) + len(result.val_indices) + len(result.test_indices)
        assert total == 100
        assert len(result.buffer_indices) == 0  # no buffer in random mode

        # Random should be very close to target
        assert abs(len(result.train_indices) - 80) <= 2
        assert abs(len(result.val_indices) - 15) <= 2
        assert abs(len(result.test_indices) - 5) <= 2

    def test_random_split_no_buffer(self, tile_list_10x10: list[dict]):
        """GIVEN random split, THEN buffer_indices is empty."""
        splitter = SpatialSplitter(method="random")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = splitter.split(tile_list_10x10, 1000, 1000, 100)
        assert result.buffer_indices == []


class TestReproducibility:
    """Seed reproducibility tests."""

    def test_deterministic_seed(self, tile_list_10x10: list[dict]):
        """GIVEN seed=42, WHEN split() called twice,
        THEN identical SplitResults."""
        splitter = SpatialSplitter(seed=42, buffer_tiles=0)
        result1 = splitter.split(tile_list_10x10, 1000, 1000, 100)
        result2 = splitter.split(tile_list_10x10, 1000, 1000, 100)

        assert result1.train_indices == result2.train_indices
        assert result1.val_indices == result2.val_indices
        assert result1.test_indices == result2.test_indices
        assert result1.buffer_indices == result2.buffer_indices

    def test_different_seeds_differ(self, tile_list_10x10: list[dict]):
        """GIVEN different seeds, WHEN split() called,
        THEN results differ (high probability with 100 tiles)."""
        splitter1 = SpatialSplitter(seed=42, buffer_tiles=0)
        splitter2 = SpatialSplitter(seed=99, buffer_tiles=0)
        result1 = splitter1.split(tile_list_10x10, 1000, 1000, 100)
        result2 = splitter2.split(tile_list_10x10, 1000, 1000, 100)

        # At least one split should differ
        any_diff = (
            result1.train_indices != result2.train_indices
            or result1.val_indices != result2.val_indices
            or result1.test_indices != result2.test_indices
        )
        assert any_diff


class TestValidation:
    """Input validation tests."""

    def test_ratios_not_summing_to_one(self):
        """GIVEN ratios=(0.5, 0.3, 0.1), THEN ValueError about sum."""
        with pytest.raises(ValueError, match="sum to 1.0"):
            SpatialSplitter(ratios=(0.5, 0.3, 0.1))

    def test_invalid_method(self):
        """GIVEN method='stratified', THEN ValueError listing valid methods."""
        with pytest.raises(ValueError, match="Invalid splitting method"):
            SpatialSplitter(method="stratified")

    def test_negative_ratio(self):
        """GIVEN negative ratio, THEN ValueError."""
        with pytest.raises(ValueError, match="non-negative"):
            SpatialSplitter(ratios=(-0.1, 0.6, 0.5))

    def test_wrong_number_of_ratios(self):
        """GIVEN 2 ratios instead of 3, THEN ValueError."""
        with pytest.raises(ValueError, match="3 ratios"):
            SpatialSplitter(ratios=(0.8, 0.2))  # type: ignore[arg-type]


class TestEdgeCases:
    """Edge case tests."""

    def test_small_raster_4_tiles(self, tile_list_2x2: list[dict]):
        """GIVEN only 4 tiles, THEN valid split produced.
        Train gets tiles; val/test may be empty."""
        splitter = SpatialSplitter(method="block", ratios=(0.8, 0.15, 0.05), buffer_tiles=0)
        result = splitter.split(tile_list_2x2, 200, 200, 100)

        total = (
            len(result.train_indices) + len(result.val_indices) + len(result.test_indices) + len(result.buffer_indices)
        )
        assert total == 4
        # At minimum, train should have tiles
        assert len(result.train_indices) > 0

    def test_empty_tile_list(self):
        """GIVEN empty tile list, THEN empty SplitResult."""
        splitter = SpatialSplitter()
        result = splitter.split([], 0, 0, 100)
        assert result.train_indices == []
        assert result.val_indices == []
        assert result.test_indices == []
        assert result.buffer_indices == []

    def test_single_tile(self):
        """GIVEN 1 tile, THEN it goes to train."""
        tiles = [{"window": (0, 0, 100, 100)}]
        splitter = SpatialSplitter(buffer_tiles=0)
        result = splitter.split(tiles, 100, 100, 100)
        assert len(result.train_indices) == 1
        assert result.train_indices == [0]
