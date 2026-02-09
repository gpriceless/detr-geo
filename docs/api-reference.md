# API Reference

Complete API documentation for detr-geo v0.1.0.

---

## DetrGeo

```python
from detr_geo import DetrGeo
```

The main class. Three lines to go from GeoTIFF to georeferenced detections:

```python
dg = DetrGeo(model_size="medium")
dg.set_image("scene.tif")
detections = dg.detect_tiled()
```

### Constructor

```python
DetrGeo(
    model_size="medium",
    device=None,
    lazy_load=True,
    confidence_threshold=0.5,
    pretrain_weights=None,
    custom_class_names=None,
)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `model_size` | `str` | `"medium"` | `"nano"`, `"small"`, `"medium"`, `"base"`, or `"large"` |
| `device` | `str or None` | `None` | `"cuda"`, `"mps"`, or `"cpu"`. Auto-detects if None |
| `lazy_load` | `bool` | `True` | Defer weight download until first prediction |
| `confidence_threshold` | `float` | `0.5` | Minimum confidence for detections (0.0 to 1.0) |
| `pretrain_weights` | `str or None` | `None` | Path to custom `.pth` weights file |
| `custom_class_names` | `dict[int, str] or None` | `None` | Class ID to label mapping for fine-tuned models |

**COCO-pretrained (default):**

```python
dg = DetrGeo(model_size="nano", device="cpu", confidence_threshold=0.3)
```

**xView fine-tuned vehicle detection:**

```python
dg = DetrGeo(
    model_size="medium",
    pretrain_weights="checkpoints/checkpoint_best_ema.pth",
    custom_class_names={
        0: "Car",
        1: "Pickup Truck",
        2: "Truck",
        3: "Bus",
        4: "Other Vehicle",
    },
)
```

---

### Properties

#### `.crs`

```python
dg.crs -> CRS | None
```

The coordinate reference system of the loaded image. `None` before `set_image()` or in pixel-only mode. Settable:

```python
dg.crs = "EPSG:32618"          # EPSG string
dg.crs = CRS.from_epsg(4326)   # pyproj CRS object
```

Raises `CRSError` on invalid input.

#### `.resolution`

```python
dg.resolution -> int
```

The model's native input size in pixels. Determined by `model_size`:

| Size | Resolution |
|------|-----------|
| nano | 384 |
| small | 512 |
| medium | 576 |
| base | 560 |
| large | 704 |

#### `.detections`

```python
dg.detections -> GeoDataFrame | None
```

The most recent detection results. `None` until `detect()` or `detect_tiled()` is called.

**Columns in georeferenced mode:**

| Column | Type | Description |
|---|---|---|
| `geometry` | Polygon | Georeferenced bounding box (4-corner polygon) |
| `class_id` | int | Numeric class ID |
| `class_name` | str | Human-readable label |
| `confidence` | float | Detection confidence (0.0--1.0) |
| `centroid_x` | float | Polygon centroid X |
| `centroid_y` | float | Polygon centroid Y |

In pixel-only mode (no CRS), returns a DataFrame with `x1, y1, x2, y2` pixel columns instead of geometry.

---

### Methods

#### `set_image()`

```python
dg.set_image(source, bands="rgb", georeferenced=True, suppress_gsd_warning=False)
```

Load a geospatial raster. Reads metadata (CRS, transform, dimensions) without loading pixel data into memory.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `source` | `str or Path` | -- | Path to GeoTIFF, COG URL, S3 URI, or pystac.Item |
| `bands` | `tuple or str` | `"rgb"` | Band selection preset or 1-indexed tuple |
| `georeferenced` | `bool` | `True` | Read CRS/transform from raster |
| `suppress_gsd_warning` | `bool` | `False` | Silence GSD range warnings |

**Band presets:** `"rgb"` (1,2,3), `"naip_rgb"` (1,2,3), `"sentinel2_rgb"` (4,3,2), `"worldview_rgb"` (5,3,2)

```python
dg.set_image("sentinel2.tif", bands="sentinel2_rgb")
dg.set_image("custom.tif", bands=(8, 4, 3))
dg.set_image("no_crs.tif", georeferenced=False)
```

**Raises:** `FileNotFoundError`, `MissingCRSError`, `BandError`

---

#### `detect()`

```python
dg.detect(threshold=None, classes=None) -> GeoDataFrame
```

Single-pass detection. Loads the entire image into memory.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `threshold` | `float or None` | `None` | Confidence override (uses constructor default if None) |
| `classes` | `list[str] or None` | `None` | Filter to these class names only |

```python
detections = dg.detect(threshold=0.4, classes=["car", "truck"])
```

**Raises:** `DetrGeoError` if no image set. Issues `ResourceWarning` if raster exceeds ~1 GB.

---

#### `detect_tiled()`

```python
dg.detect_tiled(
    tile_size=None,
    overlap=0.2,
    nms_threshold=0.5,
    nodata_threshold=0.5,
    threshold=None,
    classes=None,
    batch_size=None,
) -> GeoDataFrame
```

Tiled detection for large rasters. Splits the image into overlapping tiles, runs detection on each, offsets boxes to global coordinates, and deduplicates with cross-tile class-aware NMS.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `tile_size` | `int or None` | `None` | Tile edge length in pixels. `None` = model native resolution |
| `overlap` | `float` | `0.2` | Fractional overlap between tiles (0.0--0.49) |
| `nms_threshold` | `float` | `0.5` | IoU threshold for cross-tile deduplication |
| `nodata_threshold` | `float` | `0.5` | Skip tiles with nodata fraction above this |
| `threshold` | `float or None` | `None` | Confidence override |
| `classes` | `list[str] or None` | `None` | Filter to specific class names |
| `batch_size` | `int or None` | `None` | Tiles per batch. Auto-detected from GPU memory |

```python
detections = dg.detect_tiled(
    overlap=0.3,
    nms_threshold=0.4,
    threshold=0.3,
)
```

**Raises:** `DetrGeoError` if no image set.

---

#### `to_geojson()`

```python
dg.to_geojson(path, simplify_tolerance=None)
```

Export to GeoJSON. Automatically reprojects to WGS84 (EPSG:4326) per the GeoJSON specification.

**Raises:** `ExportError` if no detections.

#### `to_gpkg()`

```python
dg.to_gpkg(path, layer="detections")
```

Export to GeoPackage. Preserves the original CRS. Supports multiple named layers.

**Raises:** `ExportError` if no detections.

#### `to_shp()`

```python
dg.to_shp(path)
```

Export to Shapefile. Issues a warning about legacy format limitations (10-char field names, 2 GB limit, sidecar files).

**Raises:** `ExportError` if no detections.

---

#### `show_map()`

```python
dg.show_map(basemap="SATELLITE", **kwargs) -> leafmap.Map
```

Interactive web map with detection polygons on a satellite basemap. Best in Jupyter.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `basemap` | `str` | `"SATELLITE"` | `"SATELLITE"`, `"ROADMAP"`, or `"TERRAIN"` |

**Raises:** `DetrGeoError` if no detections. `ImportError` if leafmap not installed.

#### `show_detections()`

```python
dg.show_detections(figsize=(12, 10), **kwargs) -> (Figure, Axes)
```

Static matplotlib plot with bounding boxes on the source image. Only works after `detect()` (not `detect_tiled()`), because it needs the in-memory image.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `figsize` | `tuple[int, int]` | `(12, 10)` | Figure size in inches |

**Raises:** `DetrGeoError` if no detections or no image.

---

## I/O Module

```python
from detr_geo.io import (
    load_raster_metadata,
    BandSelector,
    normalize_to_float32,
    compute_scene_stretch_params,
    read_tile,
    compute_gsd,
    check_gsd,
    compute_nodata_fraction,
    fill_nodata,
)
```

### load_raster_metadata()

```python
load_raster_metadata(source) -> RasterMetadata
```

Read raster metadata without loading pixel data. Returns a dataclass:

| Field | Type | Description |
|---|---|---|
| `crs` | `CRS or None` | Coordinate reference system |
| `transform` | `Affine` | Rasterio affine transform |
| `width`, `height` | `int` | Dimensions in pixels |
| `count` | `int` | Number of bands |
| `dtype` | `str` | Pixel data type |
| `nodata` | `float or None` | Nodata value |
| `has_alpha` | `bool` | Whether an alpha band exists |
| `bounds` | `tuple` | `(left, bottom, right, top)` |
| `gsd` | `float or None` | Ground sample distance in meters/pixel |

```python
meta = load_raster_metadata("scene.tif")
print(f"{meta.width}x{meta.height}, CRS: {meta.crs}, GSD: {meta.gsd}m")
```

### BandSelector

```python
BandSelector(bands="rgb")
```

Maps sensor bands to RGB. Presets:

| Preset | Bands | Sensor |
|---|---|---|
| `"rgb"` | 1, 2, 3 | Generic RGB |
| `"naip_rgb"` | 1, 2, 3 | NAIP |
| `"sentinel2_rgb"` | 4, 3, 2 | Sentinel-2 |
| `"worldview_rgb"` | 5, 3, 2 | WorldView |

### normalize_to_float32()

```python
normalize_to_float32(data, stretch="percentile", percentiles=(2.0, 98.0),
                     stretch_params=None, nodata_mask=None) -> (NDArray, StretchParams)
```

Normalize raster data to float32 in [0, 1] for model input. Three stretch modes:

- `"percentile"` (default) -- 2nd-to-98th percentile. Handles satellite 16-bit well.
- `"minmax"` -- full value range. Sensitive to outliers.
- `"none"` -- passthrough. Data must already be [0, 1].

### compute_scene_stretch_params()

```python
compute_scene_stretch_params(raster_path, bands, percentiles=(2.0, 98.0),
                             sample_tiles=20, tile_size=512) -> StretchParams
```

Sample random tiles from a raster to compute consistent normalization parameters. Apply the result to every tile for uniform brightness across a tiled detection:

```python
params = compute_scene_stretch_params("large_scene.tif", bands=[1, 2, 3])
normalized, _ = normalize_to_float32(tile_data, stretch_params=params)
```

### read_tile()

```python
read_tile(source, window, bands=None) -> (NDArray, NDArray | None)
```

Memory-efficient windowed read from a raster.

- `window`: `(col_off, row_off, width, height)` in pixels
- `bands`: 1-indexed band list. `None` reads all.
- Returns: `(data (bands, H, W), nodata_mask (H, W) or None)`

### compute_gsd()

```python
compute_gsd(transform, crs) -> float | None
```

Ground sample distance in meters/pixel from the affine transform and CRS.

### check_gsd()

```python
check_gsd(gsd, strict=False)
```

Warn if GSD is outside the optimal 0.1--0.5 m/px range. If `strict=True`, raises `BandError` above 5.0 m/px.

---

## CRS Module

```python
from detr_geo.crs import pixel_to_geo, get_transformer, validate_crs, auto_utm_crs, has_rotation
```

### pixel_to_geo()

```python
pixel_to_geo(bbox, transform) -> Polygon
```

Convert a pixel bounding box `(x_min, y_min, x_max, y_max)` to a CRS-space Shapely polygon. Transforms all 4 corners independently, handling rotated affine transforms correctly.

### get_transformer()

```python
get_transformer(src_crs, dst_crs) -> Transformer
```

Cached pyproj Transformer with `always_xy=True` to avoid axis order surprises.

### auto_utm_crs()

```python
auto_utm_crs(longitude, latitude) -> CRS
```

Return the appropriate UTM zone for a point. EPSG:326xx (north) or EPSG:327xx (south).

### validate_crs()

```python
validate_crs(crs, georeferenced=True) -> CRS | None
```

Validate and parse a CRS value. Raises `MissingCRSError` or `CRSError`.

---

## Tiling Module

```python
from detr_geo.tiling import (
    generate_tile_grid,
    cross_tile_nms,
    recommended_overlap,
    detection_range,
    process_tiles,
)
```

### generate_tile_grid()

```python
generate_tile_grid(raster_width, raster_height, tile_size, overlap_ratio=0.2) -> list[TileInfo]
```

Generate a grid of overlapping tiles covering the raster.

### cross_tile_nms()

```python
cross_tile_nms(boxes, scores, class_ids, iou_threshold=0.5) -> NDArray[bool]
```

Class-aware non-maximum suppression. Same-class detections with IoU above threshold are suppressed. Different-class overlaps are preserved.

### recommended_overlap()

```python
recommended_overlap(object_size_pixels, object_size_fraction=0.3, tile_size=576) -> float
```

Estimate the overlap ratio needed to avoid missing objects at tile boundaries, given expected object size in pixels.

### detection_range()

```python
detection_range(tile_size, gsd, overlap=0.2) -> (float, float)
```

Compute the detectable object size range in meters for a given tile size and GSD:

```python
min_m, max_m = detection_range(tile_size=576, gsd=0.3)
# (3.0, 138.2) -- objects from 3m to 138m are detectable
```

---

## Export Module

```python
from detr_geo.export import (
    build_geodataframe,
    build_dataframe_pixel,
    compute_areas,
    export_geojson,
    export_gpkg,
    export_shp,
)
```

### compute_areas()

```python
compute_areas(gdf, equal_area_crs=None) -> Series
```

Compute polygon areas in square meters. Auto-detects UTM zone for geographic CRS data. Override with an explicit equal-area CRS if needed.

```python
areas_m2 = compute_areas(dg.detections)
```

---

## Visualization Module

```python
from detr_geo.viz import show_detections, show_map, get_class_colors
```

### show_detections()

```python
show_detections(image, gdf, figsize=(12, 10), min_confidence=0.0,
                top_n=None, classes=None, class_colors=None,
                show_labels=True, save_path=None, dpi=150)
```

Render bounding boxes on the source image using matplotlib.

| Parameter | Description |
|---|---|
| `image` | Source image `(H, W, 3)` in [0, 1] or [0, 255] |
| `min_confidence` | Filter detections below this score |
| `top_n` | Limit to top N detections by confidence |
| `classes` | Filter to specific class names |
| `class_colors` | Override colors: `{"Car": "red", "Bus": "blue"}` |
| `save_path` | Save figure to file |
| `dpi` | Resolution for saved output |

### show_map()

```python
show_map(gdf, basemap="SATELLITE", min_confidence=0.0, classes=None,
         class_colors=None, max_detections=1000, map_object=None) -> leafmap.Map
```

Interactive web map. Auto-reprojects to WGS84. Click polygons to see class, confidence, and area.

---

## Training Module

```python
from detr_geo import (
    prepare_training_dataset,
    SpatialSplitter,
    train,
    detections_to_coco,
    detections_to_yolo,
    AUGMENTATION_PRESETS,
    AugmentationPreset,
)
```

### prepare_training_dataset()

```python
prepare_training_dataset(
    raster_path, annotations_path, output_dir,
    tile_size=576, overlap_ratio=0.0, class_mapping=None,
    min_annotation_area=100, max_background_per_annotated=3.0,
    bands="rgb", split_method="block", split_ratios=(0.8, 0.15, 0.05),
    seed=42,
) -> dict[str, int]
```

Convert a GeoTIFF + vector annotations into a COCO-format training dataset. Handles:
- CRS alignment between raster and annotations
- Tiling to model-sized chips
- Clipping annotations to tile boundaries
- Spatial block splitting with buffer zones
- Background tile ratio control

Returns a stats dict with tile and annotation counts per split.

### SpatialSplitter

```python
SpatialSplitter(method="block", ratios=(0.8, 0.15, 0.05), buffer_tiles=1, seed=42)
```

Spatial-aware train/val/test splitting. The `"block"` method assigns contiguous regions to splits with buffer zones between them, preventing the model from memorizing scene-specific lighting or shadow patterns that leak across nearby tiles.

### train()

```python
train(adapter, dataset_dir, epochs=50, augmentation_preset="satellite_default",
      augmentation_factor=2, batch_size=8, learning_rate=None, resume=None,
      seed=42, **kwargs) -> dict
```

Fine-tune RF-DETR on a COCO-format dataset. Applies geospatial augmentations: 90-degree rotation, horizontal flip, vertical flip, and color jitter tuned for overhead imagery.

### AugmentationPreset / AUGMENTATION_PRESETS

Built-in presets for overhead imagery:

| Preset | Best For | Jitter |
|---|---|---|
| `"satellite_default"` | Satellite (0.3--0.5m GSD) | Low |
| `"aerial_default"` | Aerial (0.1--0.3m GSD) | Medium |
| `"drone_default"` | Drone (<0.1m GSD) | High |

All presets enable rotation and flipping -- critical for overhead imagery where there is no canonical "up" direction.

### detections_to_coco() / detections_to_yolo()

Convert detection GeoDataFrames to COCO JSON or YOLO TXT format. Useful for the active learning loop: detect, review, correct, retrain.

---

## Exceptions

All exceptions inherit from `DetrGeoError`:

```python
from detr_geo import DetrGeoError, CRSError, MissingCRSError, TilingError, ModelError, BandError, ExportError
```

| Exception | When |
|---|---|
| `DetrGeoError` | Base class. Catch-all. |
| `CRSError` | Invalid CRS string or incompatible transform |
| `MissingCRSError` | Raster has no CRS in georeferenced mode |
| `TilingError` | Invalid tile size or overlap |
| `ModelError` | Unsupported model size, rfdetr not installed, inference failure |
| `BandError` | Invalid band index, wrong channel count |
| `ExportError` | Write failure, no detections to export |

All exceptions store additional context:

```python
try:
    dg.set_image("file.tif")
except DetrGeoError as e:
    print(e)           # Human-readable message
    print(e.context)   # Dict with additional details
```

---

## Type Aliases

Defined in `detr_geo._typing`:

| Type | Definition |
|---|---|
| `PixelBBox` | `tuple[float, float, float, float]` -- (x_min, y_min, x_max, y_max) |
| `GeoBBox` | `tuple[float, float, float, float]` -- (west, south, east, north) |
| `TileWindow` | `tuple[int, int, int, int]` -- (col_off, row_off, width, height) |
| `DetectionResult` | `TypedDict` with bbox, confidence, class_id lists |
| `TileInfo` | `TypedDict` with window, offsets, nodata_fraction |
| `ImageArray` | `NDArray[float32]` -- (bands, height, width) |
| `ModelSize` | `str` -- "nano", "small", "medium", "base", "large" |
