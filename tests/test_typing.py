"""Tests for type aliases."""

from __future__ import annotations


def test_detection_result_importable():
    """DetectionResult should be importable from detr_geo._typing."""
    from detr_geo._typing import DetectionResult

    assert DetectionResult is not None


def test_tile_info_importable():
    """TileInfo should be importable from detr_geo._typing."""
    from detr_geo._typing import TileInfo

    assert TileInfo is not None


def test_pixel_bbox_importable():
    """PixelBBox should be importable from detr_geo._typing."""
    from detr_geo._typing import PixelBBox

    assert PixelBBox is not None


def test_geo_bbox_importable():
    """GeoBBox should be importable from detr_geo._typing."""
    from detr_geo._typing import GeoBBox

    assert GeoBBox is not None


def test_tile_window_importable():
    """TileWindow should be importable from detr_geo._typing."""
    from detr_geo._typing import TileWindow

    assert TileWindow is not None


def test_image_array_importable():
    """ImageArray should be importable from detr_geo._typing."""
    from detr_geo._typing import ImageArray

    assert ImageArray is not None


def test_model_size_importable():
    """ModelSize should be importable from detr_geo._typing."""
    from detr_geo._typing import ModelSize

    assert ModelSize is not None


def test_detection_result_as_dict():
    """DetectionResult should be instantiable as a dict."""
    from detr_geo._typing import DetectionResult

    result: DetectionResult = {
        "bbox": [[10.0, 20.0, 30.0, 40.0]],
        "confidence": [0.95],
        "class_id": [1],
    }
    assert result["bbox"] == [[10.0, 20.0, 30.0, 40.0]]
    assert result["confidence"] == [0.95]
    assert result["class_id"] == [1]


def test_tile_info_as_dict():
    """TileInfo should be instantiable as a dict."""
    from detr_geo._typing import TileInfo

    tile: TileInfo = {
        "window": (0, 0, 256, 256),
        "global_offset_x": 0,
        "global_offset_y": 0,
        "nodata_fraction": 0.0,
    }
    assert tile["window"] == (0, 0, 256, 256)
    assert tile["nodata_fraction"] == 0.0
