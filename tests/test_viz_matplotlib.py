"""Tests for matplotlib visualization functions."""

from __future__ import annotations

import tempfile
from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import box

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

from detr_geo.viz import (  # noqa: E402
    _draw_box,
    _filter_gdf,
    get_class_colors,
    show_detections,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def make_test_image(h: int = 200, w: int = 200) -> np.ndarray:
    """Create a synthetic test image."""
    return np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)


def make_test_gdf(n: int = 5) -> gpd.GeoDataFrame:
    """Create a synthetic GeoDataFrame with pixel-space bounding boxes."""
    rng = np.random.RandomState(42)
    geometries = []
    class_names = []
    confidences = []
    class_options = ["building", "vehicle", "tree"]

    for i in range(n):
        x1 = rng.uniform(10, 100)
        y1 = rng.uniform(10, 100)
        x2 = x1 + rng.uniform(20, 50)
        y2 = y1 + rng.uniform(20, 50)
        geometries.append(box(x1, y1, x2, y2))
        class_names.append(class_options[i % len(class_options)])
        confidences.append(rng.uniform(0.3, 0.99))

    return gpd.GeoDataFrame(
        {
            "geometry": geometries,
            "class_name": class_names,
            "confidence": confidences,
            "class_id": list(range(n)),
        }
    )


@pytest.fixture
def tmp_dir():
    with tempfile.TemporaryDirectory() as d:
        yield d


# ---------------------------------------------------------------------------
# Tests: get_class_colors
# ---------------------------------------------------------------------------


class TestGetClassColors:
    """Test class color assignment."""

    def test_returns_distinct_colors(self):
        """Each class gets a distinct color."""
        names = ["building", "vehicle", "tree"]
        colors = get_class_colors(names)
        assert len(colors) == 3
        # All colors are unique
        color_values = list(colors.values())
        assert len(set(color_values)) == 3

    def test_returns_rgb_tuples(self):
        """All returned colors are RGB tuples in [0, 1]."""
        names = ["a", "b", "c"]
        colors = get_class_colors(names)
        for _name, rgb in colors.items():
            assert isinstance(rgb, tuple)
            assert len(rgb) == 3
            for val in rgb:
                assert 0.0 <= val <= 1.0

    def test_deterministic(self):
        """Same input produces same output."""
        names = ["building", "vehicle", "tree"]
        colors1 = get_class_colors(names)
        colors2 = get_class_colors(names)
        assert colors1 == colors2

    def test_20_distinct_colors(self):
        """Up to 20 classes get distinct colors."""
        names = [f"class_{i}" for i in range(20)]
        colors = get_class_colors(names)
        color_values = list(colors.values())
        assert len(set(color_values)) == 20

    def test_user_override(self):
        """User can override specific class colors."""
        names = ["building", "vehicle"]
        colors = get_class_colors(names, user_colors={"building": "red"})
        # "building" should be pure red
        assert colors["building"] == pytest.approx((1.0, 0.0, 0.0), abs=0.01)
        # "vehicle" should get a tab20 color (not red)
        assert colors["vehicle"] != colors["building"]

    def test_hex_color_override(self):
        """Hex color strings are accepted."""
        names = ["building"]
        colors = get_class_colors(names, user_colors={"building": "#00FF00"})
        assert colors["building"] == pytest.approx((0.0, 1.0, 0.0), abs=0.01)

    def test_partial_override(self):
        """Non-overridden classes still get auto colors."""
        names = ["building", "vehicle", "tree"]
        colors = get_class_colors(names, user_colors={"building": "red"})
        assert len(colors) == 3
        assert "vehicle" in colors
        assert "tree" in colors

    def test_empty_class_names(self):
        """Empty class list returns empty dict."""
        colors = get_class_colors([])
        assert colors == {}


# ---------------------------------------------------------------------------
# Tests: _filter_gdf
# ---------------------------------------------------------------------------


class TestFilterGdf:
    """Test GeoDataFrame filtering."""

    def test_min_confidence_filter(self):
        """min_confidence reduces detections."""
        gdf = make_test_gdf(10)
        filtered = _filter_gdf(gdf, min_confidence=0.7)
        assert len(filtered) <= len(gdf)
        assert (filtered["confidence"] >= 0.7).all()

    def test_top_n_filter(self):
        """top_n limits to N detections."""
        gdf = make_test_gdf(10)
        filtered = _filter_gdf(gdf, top_n=5)
        assert len(filtered) == 5

    def test_top_n_fewer_than_available(self):
        """top_n greater than available returns all."""
        gdf = make_test_gdf(3)
        filtered = _filter_gdf(gdf, top_n=10)
        assert len(filtered) == 3

    def test_class_filter(self):
        """classes filter limits to specified classes."""
        gdf = make_test_gdf(10)
        filtered = _filter_gdf(gdf, classes=["building"])
        assert (filtered["class_name"] == "building").all()

    def test_empty_gdf_returns_empty(self):
        """Empty GeoDataFrame stays empty."""
        gdf = gpd.GeoDataFrame({"geometry": [], "confidence": [], "class_name": []})
        filtered = _filter_gdf(gdf, min_confidence=0.5)
        assert len(filtered) == 0


# ---------------------------------------------------------------------------
# Tests: show_detections
# ---------------------------------------------------------------------------


class TestShowDetections:
    """Test matplotlib detection rendering."""

    def test_creates_figure_without_error(self):
        """show_detections runs without error."""
        image = make_test_image()
        gdf = make_test_gdf(5)
        # Should not raise
        show_detections(image, gdf)

    def test_save_to_file(self, tmp_dir):
        """save_path creates an image file on disk."""
        image = make_test_image()
        gdf = make_test_gdf(3)
        path = str(Path(tmp_dir) / "detections.png")
        show_detections(image, gdf, save_path=path)
        assert Path(path).exists()
        assert Path(path).stat().st_size > 0

    def test_min_confidence_filter(self, tmp_dir):
        """min_confidence reduces displayed detections."""
        image = make_test_image()
        gdf = make_test_gdf(10)
        path = str(Path(tmp_dir) / "filtered.png")
        # Should not raise
        show_detections(image, gdf, min_confidence=0.9, save_path=path)
        assert Path(path).exists()

    def test_top_n_filter(self, tmp_dir):
        """top_n limits detections."""
        image = make_test_image()
        gdf = make_test_gdf(10)
        path = str(Path(tmp_dir) / "topn.png")
        show_detections(image, gdf, top_n=3, save_path=path)
        assert Path(path).exists()

    def test_class_filter(self, tmp_dir):
        """classes filter limits to specific classes."""
        image = make_test_image()
        gdf = make_test_gdf(10)
        path = str(Path(tmp_dir) / "class_filter.png")
        show_detections(image, gdf, classes=["building"], save_path=path)
        assert Path(path).exists()

    def test_class_colors_override(self, tmp_dir):
        """class_colors override applies without error."""
        image = make_test_image()
        gdf = make_test_gdf(5)
        path = str(Path(tmp_dir) / "custom_colors.png")
        show_detections(
            image,
            gdf,
            class_colors={"building": "red", "vehicle": "blue"},
            save_path=path,
        )
        assert Path(path).exists()

    def test_empty_gdf_shows_image_only(self, tmp_dir):
        """Empty GeoDataFrame renders just the image."""
        image = make_test_image()
        gdf = gpd.GeoDataFrame({"geometry": [], "class_name": [], "confidence": []})
        path = str(Path(tmp_dir) / "empty.png")
        show_detections(image, gdf, save_path=path)
        assert Path(path).exists()

    def test_show_labels_false(self, tmp_dir):
        """show_labels=False doesn't raise."""
        image = make_test_image()
        gdf = make_test_gdf(3)
        path = str(Path(tmp_dir) / "no_labels.png")
        show_detections(image, gdf, show_labels=False, save_path=path)
        assert Path(path).exists()

    def test_custom_figsize(self, tmp_dir):
        """Custom figsize doesn't raise."""
        image = make_test_image()
        gdf = make_test_gdf(3)
        path = str(Path(tmp_dir) / "custom_size.png")
        show_detections(image, gdf, figsize=(20, 16), save_path=path)
        assert Path(path).exists()

    def test_custom_dpi(self, tmp_dir):
        """Custom DPI saves file."""
        image = make_test_image()
        gdf = make_test_gdf(3)
        path = str(Path(tmp_dir) / "high_dpi.png")
        show_detections(image, gdf, save_path=path, dpi=300)
        assert Path(path).exists()

    def test_float_image_0_1_range(self, tmp_dir):
        """Float image in [0, 1] renders correctly."""
        image = np.random.rand(200, 200, 3).astype(np.float32)
        gdf = make_test_gdf(3)
        path = str(Path(tmp_dir) / "float_image.png")
        show_detections(image, gdf, save_path=path)
        assert Path(path).exists()


# ---------------------------------------------------------------------------
# Tests: _draw_box
# ---------------------------------------------------------------------------


class TestDrawBox:
    """Test bounding box drawing helper."""

    def test_draws_without_error(self):
        """_draw_box runs without error on a matplotlib axes."""
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        _draw_box(ax, (10, 10, 50, 50), "test 0.95", (1.0, 0.0, 0.0), 0.95)
        plt.close(fig)

    def test_no_label_mode(self):
        """show_label=False doesn't add text."""
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        _draw_box(ax, (10, 10, 50, 50), "test 0.95", (0.0, 1.0, 0.0), 0.95, show_label=False)
        # Check that no text objects were added
        texts = ax.texts
        assert len(texts) == 0
        plt.close(fig)

    def test_label_mode_adds_text(self):
        """show_label=True adds text to axes."""
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        _draw_box(ax, (10, 10, 50, 50), "building 0.95", (0.0, 0.0, 1.0), 0.95, show_label=True)
        texts = ax.texts
        assert len(texts) == 1
        assert "building" in texts[0].get_text()
        plt.close(fig)
