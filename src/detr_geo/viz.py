"""Visualization utilities for detection results.

This module provides matplotlib-based static plot rendering and leafmap-based
interactive map visualization for geospatial detection results.

Both renderers share a common class coloring system (get_class_colors) that
uses matplotlib's tab20 colormap with user override support.

leafmap is imported lazily -- users who only need static plots never need
to install it.
"""

from __future__ import annotations

import json
import warnings
from typing import Any

import geopandas as gpd
from numpy.typing import NDArray
from pyproj import CRS

# ---------------------------------------------------------------------------
# Class Coloring
# ---------------------------------------------------------------------------


def get_class_colors(
    class_names: list[str],
    user_colors: dict[str, str] | None = None,
) -> dict[str, tuple[float, float, float]]:
    """Generate consistent colors for each class.

    Uses matplotlib's tab20 colormap as default.
    User can override specific class colors.

    Args:
        class_names: List of unique class names.
        user_colors: Optional mapping of class_name -> color string
            (e.g., "red", "#FF0000").

    Returns:
        Dict mapping class_name -> RGB tuple (0-1 range).
    """
    import matplotlib
    import matplotlib.colors as mcolors

    cmap = matplotlib.colormaps.get_cmap("tab20")

    colors: dict[str, tuple[float, float, float]] = {}

    for i, name in enumerate(class_names):
        if user_colors and name in user_colors:
            # Convert user color string to RGB tuple
            rgb = mcolors.to_rgb(user_colors[name])
            colors[name] = (float(rgb[0]), float(rgb[1]), float(rgb[2]))
        else:
            # Use tab20 colormap, cycling if > 20 classes
            rgba = cmap(i % 20)
            colors[name] = (float(rgba[0]), float(rgba[1]), float(rgba[2]))

    return colors


# ---------------------------------------------------------------------------
# Filtering Helpers
# ---------------------------------------------------------------------------


def _filter_gdf(
    gdf: gpd.GeoDataFrame,
    min_confidence: float = 0.0,
    top_n: int | None = None,
    classes: list[str] | None = None,
) -> gpd.GeoDataFrame:
    """Apply confidence, class, and top-N filtering to a GeoDataFrame.

    Args:
        gdf: Input GeoDataFrame with 'confidence' and 'class_name' columns.
        min_confidence: Minimum confidence threshold.
        top_n: Show only top N detections by confidence.
        classes: Filter to specific class names.

    Returns:
        Filtered GeoDataFrame.
    """
    if len(gdf) == 0:
        return gdf

    filtered = gdf.copy()

    # Confidence filter
    if min_confidence > 0.0 and "confidence" in filtered.columns:
        filtered = filtered[filtered["confidence"] >= min_confidence]

    # Class filter
    if classes is not None and "class_name" in filtered.columns:
        filtered = filtered[filtered["class_name"].isin(classes)]

    # Top-N filter (by confidence, descending)
    if top_n is not None and len(filtered) > top_n:
        filtered = filtered.nlargest(top_n, "confidence")

    return filtered


# ---------------------------------------------------------------------------
# Matplotlib Renderer
# ---------------------------------------------------------------------------


def _draw_box(
    ax: Any,
    bbox: tuple[float, float, float, float],
    label: str,
    color: tuple[float, float, float],
    confidence: float,
    show_label: bool = True,
) -> None:
    """Draw a single bounding box with label on a matplotlib axes.

    Args:
        ax: Matplotlib Axes object.
        bbox: (x1, y1, x2, y2) bounding box coordinates.
        label: Text label (e.g., "building 0.95").
        color: RGB color tuple (0-1 range).
        confidence: Confidence score (used for line width scaling).
        show_label: Whether to display the text label.
    """
    import matplotlib.patches as patches

    x1, y1, x2, y2 = bbox
    width = x2 - x1
    height = y2 - y1

    # Line width proportional to confidence (1.0 to 3.0 range)
    linewidth = 1.0 + confidence * 2.0

    rect = patches.Rectangle(
        (x1, y1),
        width,
        height,
        linewidth=linewidth,
        edgecolor=color,
        facecolor=(*color, 0.15),  # Semi-transparent fill
    )
    ax.add_patch(rect)

    if show_label:
        ax.text(
            x1,
            y1 - 2,
            label,
            fontsize=7,
            color="white",
            backgroundcolor=(*color, 0.7),
            verticalalignment="bottom",
            clip_on=True,
        )


def show_detections(
    image: NDArray,
    gdf: gpd.GeoDataFrame,
    figsize: tuple[int, int] = (12, 10),
    min_confidence: float = 0.0,
    top_n: int | None = None,
    classes: list[str] | None = None,
    class_colors: dict[str, str] | None = None,
    show_labels: bool = True,
    save_path: str | None = None,
    dpi: int = 150,
) -> tuple:
    """Show detections as bounding boxes on source imagery using matplotlib.

    Args:
        image: Source image as numpy array shape (H, W, 3) in [0, 1] or [0, 255].
        gdf: GeoDataFrame with geometry, class_name, confidence columns.
            Uses bounding box pixel coordinates from geometry bounds.
        figsize: Figure size in inches.
        min_confidence: Minimum confidence threshold for display.
        top_n: Show only top N detections by confidence.
        classes: Filter to specific class names.
        class_colors: Override colors for specific classes.
        show_labels: Whether to draw class labels on boxes.
        save_path: If provided, save figure to this path.
        dpi: DPI for saved figure.
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    # Display image
    ax.imshow(image)
    ax.set_axis_off()

    # Filter detections
    filtered = _filter_gdf(gdf, min_confidence=min_confidence, top_n=top_n, classes=classes)

    if len(filtered) > 0 and "class_name" in filtered.columns:
        # Get class colors
        unique_classes = sorted(filtered["class_name"].unique().tolist())
        colors = get_class_colors(unique_classes, user_colors=class_colors)

        # Draw each detection
        for _, row in filtered.iterrows():
            bounds = row.geometry.bounds  # (minx, miny, maxx, maxy)
            class_name = row.get("class_name", "unknown")
            confidence = row.get("confidence", 0.0)
            color = colors.get(class_name, (1.0, 0.0, 0.0))

            label = f"{class_name} {confidence:.2f}"
            _draw_box(ax, bounds, label, color, confidence, show_label=show_labels)

    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)

    return fig, ax


# ---------------------------------------------------------------------------
# Leafmap Renderer
# ---------------------------------------------------------------------------


def _rgb_to_hex(rgb: tuple[float, float, float]) -> str:
    """Convert RGB tuple (0-1 range) to hex color string.

    Args:
        rgb: RGB tuple with values in [0, 1].

    Returns:
        Hex color string like "#FF0000".
    """
    r = int(rgb[0] * 255)
    g = int(rgb[1] * 255)
    b = int(rgb[2] * 255)
    return f"#{r:02x}{g:02x}{b:02x}"


def _gdf_to_styled_geojson(
    gdf: gpd.GeoDataFrame,
    class_colors: dict[str, tuple[float, float, float]],
) -> dict:
    """Convert GeoDataFrame to GeoJSON dict with style properties for leafmap.

    Each feature gets properties: fill_color, stroke_color, opacity (from confidence).
    Popup text includes class_name, confidence, area_m2 if available.

    Args:
        gdf: GeoDataFrame with detection polygons.
        class_colors: Mapping of class_name -> RGB tuple.

    Returns:
        GeoJSON dict with styled features.
    """
    # Ensure WGS84 for GeoJSON
    if gdf.crs is not None and not gdf.crs.equals(CRS.from_epsg(4326)):
        gdf = gdf.to_crs(epsg=4326)

    features = []
    for _, row in gdf.iterrows():
        class_name = row.get("class_name", "unknown")
        confidence = row.get("confidence", 0.0)
        color = class_colors.get(class_name, (1.0, 0.0, 0.0))
        hex_color = _rgb_to_hex(color)

        # Opacity proportional to confidence (0.3 to 0.8 range)
        opacity = 0.3 + confidence * 0.5

        # Build popup text
        popup_parts = [
            f"Class: {class_name}",
            f"Confidence: {confidence:.3f}",
        ]
        if "area_m2" in gdf.columns:
            area = row.get("area_m2", None)
            if area is not None:
                popup_parts.append(f"Area: {area:.1f} m2")

        props = {
            "class_name": class_name,
            "confidence": float(confidence),
            "fill_color": hex_color,
            "stroke_color": hex_color,
            "opacity": float(opacity),
            "popup": "<br>".join(popup_parts),
        }

        feature = {
            "type": "Feature",
            "geometry": json.loads(gpd.GeoSeries([row.geometry]).to_json())["features"][0]["geometry"],
            "properties": props,
        }
        features.append(feature)

    return {
        "type": "FeatureCollection",
        "features": features,
    }


def show_map(
    gdf: gpd.GeoDataFrame,
    basemap: str = "SATELLITE",
    min_confidence: float = 0.0,
    classes: list[str] | None = None,
    class_colors: dict[str, str] | None = None,
    max_detections: int = 1000,
    map_object: Any | None = None,
    **kwargs: Any,
) -> Any:
    """Show detections on interactive leafmap with satellite basemap.

    Adds detection polygons as a GeoJSON layer with class-based coloring,
    confidence-based opacity, and click-to-inspect popups.

    Args:
        gdf: GeoDataFrame with geometry (in any CRS -- will reproject to 4326 for display).
        basemap: One of "SATELLITE", "ROADMAP", "TERRAIN".
        min_confidence: Minimum confidence for display.
        classes: Filter to specific class names.
        class_colors: Override colors for specific classes.
        max_detections: Maximum detections to display (prevents slow rendering).
        map_object: Existing leafmap.Map to add to. If None, creates new map.
        **kwargs: Additional arguments passed to leafmap.Map.

    Returns:
        leafmap.Map object for further customization or display.

    Raises:
        ImportError: If leafmap is not installed, with pip install instructions.
    """
    try:
        import leafmap
    except ImportError as err:
        raise ImportError(
            "leafmap is required for interactive map visualization. "
            "Install it with: pip install leafmap\n"
            "Or install detr_geo with viz extras: pip install detr-geo[viz]"
        ) from err

    # Filter detections
    filtered = _filter_gdf(gdf, min_confidence=min_confidence, classes=classes)

    # Enforce max_detections limit
    if len(filtered) > max_detections:
        warnings.warn(
            f"Displaying top {max_detections} of {len(filtered)} detections. Increase max_detections to show more.",
            RuntimeWarning,
            stacklevel=2,
        )
        filtered = filtered.nlargest(max_detections, "confidence")

    # Reproject to WGS84 for leafmap display
    if filtered.crs is not None and not filtered.crs.equals(CRS.from_epsg(4326)):
        filtered = filtered.to_crs(epsg=4326)

    # Get class colors
    if len(filtered) > 0 and "class_name" in filtered.columns:
        unique_classes = sorted(filtered["class_name"].unique().tolist())
    else:
        unique_classes = []
    colors = get_class_colors(unique_classes, user_colors=class_colors)

    # Create or reuse map
    if map_object is not None:
        m = map_object
    else:
        # Map basemap names to leafmap basemaps
        basemap_map = {
            "SATELLITE": "Esri.WorldImagery",
            "ROADMAP": "OpenStreetMap",
            "TERRAIN": "Stamen.Terrain",
        }
        lm_basemap = basemap_map.get(basemap.upper(), "Esri.WorldImagery")

        # Center map on detections if available
        center_kwargs: dict[str, Any] = {}
        if len(filtered) > 0:
            bounds = filtered.total_bounds  # [minx, miny, maxx, maxy]
            center_lat = (bounds[1] + bounds[3]) / 2
            center_lon = (bounds[0] + bounds[2]) / 2
            center_kwargs["center"] = (center_lat, center_lon)
            center_kwargs["zoom"] = 15

        m = leafmap.Map(**center_kwargs, **kwargs)
        m.add_basemap(lm_basemap)

    # Add detection layer
    if len(filtered) > 0:
        geojson_data = _gdf_to_styled_geojson(filtered, colors)
        m.add_geojson(
            geojson_data,
            layer_name="Detections",
        )

    return m
