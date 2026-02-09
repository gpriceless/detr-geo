# API Reference

Complete API documentation for detr-geo v0.1.0.

---

## DetrGeo Class

**Module**: `detr_geo.core`

The primary user-facing interface. Manages the full workflow from loading imagery to exporting georeferenced detection results.

### Constructor

```python
DetrGeo(
    model_size: str = "medium",
    device: str | None = None,
    lazy_load: bool = True,
    confidence_threshold: float = 0.5,
    pretrain_weights: str | None = None,
    custom_class_names: dict[int, str] | None = None,
)
```

**Parameters**:
- `model_size`: One of `"nano"`, `"small"`, `"medium"`, `"base"`, `"large"`. Determines the model's native resolution and capacity. Default: `"medium"` (576px).
- `device`: Compute device string. `"cuda"` for NVIDIA GPU, `"mps"` for Apple Silicon, `"cpu"` for CPU. If `None`, auto-detects the best available device.
- `lazy_load`: If `True` (default), model weights are not downloaded until the first prediction. Allows creating instances without rfdetr installed.
- `confidence_threshold`: Default minimum confidence for detections. Predictions below this score are discarded. Range: 0.0 to 1.0.
- `pretrain_weights`: Path to a custom pretrained weights file (`.pth`). If `None`, uses the default COCO-pretrained weights for the selected model size. Used for loading fine-tuned models (e.g., from VME or xView training).
- `custom_class_names`: Override the COCO class name mapping for fine-tuned models. A dict mapping class_id (int) to label (str), e.g., `{0: "Car", 1: "Bus", 2: "Truck"}`. When provided, detection results use these labels instead of the default COCO names. Required when using `pretrain_weights` from a model trained on non-COCO classes.

**Examples**:
```python
# COCO-pretrained (default)
dg = DetrGeo(model_size="nano", device="cpu", confidence_threshold=0.3)

# Fine-tuned vehicle detection
dg = DetrGeo(
    model_size="medium",
    pretrain_weights="checkpoints/checkpoint_best_ema.pth",
    custom_class_names={0: "Car", 1: "Pickup", 2: "Truck", 3: "Bus", 4: "Other"},
)
```

---

### Properties

#### `crs`

```python
@property
def crs(self) -> CRS | None
```

The coordinate reference system of the loaded image. Returns `None` before `set_image()` is called or when in pixel-only mode.

Can be set manually:
```python
dg.crs = "EPSG:32618"         # EPSG string
dg.crs = CRS.from_epsg(4326)  # pyproj CRS object
```

**Raises**: `CRSError` if the assigned value is not a valid CRS.

#### `resolution`

```python
@property
def resolution(self) -> int
```

The model's native square input resolution in pixels. Determined by `model_size`:

| Size | Resolution |
|------|-----------|
| nano | 384 |
| small | 512 |
| medium | 576 |
| base | 560 |
| large | 704 |

#### `detections`

```python
@property
def detections(self) -> GeoDataFrame | None
```

The most recent detection results. Returns `None` if no detection has been run. After `detect()` or `detect_tiled()`, returns a GeoDataFrame with columns:

| Column | Type | Description |
|--------|------|-------------|
| `geometry` | Polygon | Georeferenced bounding box polygon |
| `class_id` | int | COCO class ID |
| `class_name` | str | Human-readable class name |
| `confidence` | float | Detection confidence (0.0-1.0) |
| `centroid_x` | float | Polygon centroid X coordinate |
| `centroid_y` | float | Polygon centroid Y coordinate |

If in pixel-only mode (no CRS), returns a plain DataFrame with `x1, y1, x2, y2` pixel coordinates instead of geometry.

---

### Methods

#### `set_image()`

```python
def set_image(
    source: str,
    bands: tuple[int, ...] | str = "rgb",
    georeferenced: bool = True,
    suppress_gsd_warning: bool = False,
) -> None
```

Load a geospatial raster for detection.

**Parameters**:
- `source`: Path to a raster file (GeoTIFF, etc.).
- `bands`: Band selection. Either a preset string or a tuple of 1-indexed band numbers.
  - Presets: `"rgb"` (1,2,3), `"naip_rgb"` (1,2,3), `"sentinel2_rgb"` (4,3,2), `"worldview_rgb"` (5,3,2)
  - Custom: `(4, 3, 2)` for any 3-band combination
- `georeferenced`: If `True`, reads and stores CRS/transform from the raster. If `False`, operates in pixel-only mode.
- `suppress_gsd_warning`: If `True`, suppress `GSDWarning` when imagery GSD is outside the optimal range (0.1--0.5 m/px). Useful when you know the GSD does not match standard aerial/satellite ranges but want to proceed anyway.

**Raises**:
- `FileNotFoundError`: If `source` does not exist.
- `MissingCRSError`: If `georeferenced=True` but the raster has no embedded CRS.
- `BandError`: If requested band indices exceed available bands.

**Example**:
```python
dg.set_image("sentinel2.tif", bands="sentinel2_rgb")
dg.set_image("custom.tif", bands=(8, 4, 3))
dg.set_image("no_crs.tif", georeferenced=False)
```

#### `detect()`

```python
def detect(
    threshold: float | None = None,
    classes: list[str] | None = None,
) -> GeoDataFrame
```

Run object detection on the loaded image (single-pass, full image).

**Parameters**:
- `threshold`: Confidence threshold override. If `None`, uses the threshold from the constructor.
- `classes`: Filter results to only these class names. If `None`, returns all classes.

**Returns**: GeoDataFrame (or DataFrame in pixel-only mode) with detection results.

**Raises**: `DetrGeoError` if no image has been set via `set_image()`.

**Note**: This method loads the entire image into memory. For large rasters, use `detect_tiled()`.

**Example**:
```python
detections = dg.detect(threshold=0.5, classes=["car", "truck"])
```

#### `detect_tiled()`

```python
def detect_tiled(
    tile_size: int | None = None,
    overlap: float = 0.2,
    nms_threshold: float = 0.5,
    nodata_threshold: float = 0.5,
    threshold: float | None = None,
    classes: list[str] | None = None,
    batch_size: int | None = None,
) -> GeoDataFrame
```

Run tiled object detection on a large image with cross-tile NMS deduplication.

**Parameters**:
- `tile_size`: Tile size in pixels. If `None`, uses the model's native resolution.
- `overlap`: Fractional overlap between adjacent tiles. Range: 0.0 to 0.49. Default: 0.2.
- `nms_threshold`: IoU threshold for non-maximum suppression across tiles. Detections of the same class with IoU above this value are deduplicated. Default: 0.5.
- `nodata_threshold`: Maximum fraction of nodata pixels before a tile is skipped. Default: 0.5 (skip tiles that are >50% nodata).
- `threshold`: Confidence threshold override. If `None`, uses the constructor threshold.
- `classes`: Filter results to specific class names.
- `batch_size`: Number of tiles per inference batch. If `None`, auto-detected from GPU memory. CPU always uses 1.

**Returns**: GeoDataFrame with merged, deduplicated detection results.

**Raises**: `DetrGeoError` if no image has been set.

**Example**:
```python
detections = dg.detect_tiled(
    overlap=0.3,
    nms_threshold=0.4,
    nodata_threshold=0.3,
    threshold=0.3,
)
```

#### `to_geojson()`

```python
def to_geojson(
    path: str,
    simplify_tolerance: float | None = None,
) -> None
```

Export detections to GeoJSON. Auto-reprojects to WGS84 (EPSG:4326) per the GeoJSON specification.

**Parameters**:
- `path`: Output file path.
- `simplify_tolerance`: Geometry simplification tolerance (not yet implemented in v0.1.0).

**Raises**: `ExportError` if no detections exist or export fails.

#### `to_gpkg()`

```python
def to_gpkg(
    path: str,
    layer: str = "detections",
) -> None
```

Export detections to GeoPackage format. Preserves the original CRS.

**Parameters**:
- `path`: Output file path.
- `layer`: Layer name within the GeoPackage. Default: `"detections"`.

**Raises**: `ExportError` if no detections exist or export fails.

#### `to_shp()`

```python
def to_shp(path: str) -> None
```

Export detections to Shapefile format. Issues a warning about legacy format limitations (10-character field names, 2 GB size limit).

**Parameters**:
- `path`: Output file path.

**Raises**: `ExportError` if no detections exist or export fails.

#### `show_map()`

```python
def show_map(
    basemap: str = "SATELLITE",
    **kwargs,
) -> leafmap.Map
```

Display detections on an interactive leafmap. Requires `leafmap` to be installed.

**Parameters**:
- `basemap`: One of `"SATELLITE"`, `"ROADMAP"`, `"TERRAIN"`.
- `**kwargs`: Additional arguments passed to `leafmap.Map`.

**Returns**: A `leafmap.Map` object.

**Raises**:
- `DetrGeoError` if no detections exist.
- `ImportError` if leafmap is not installed.

#### `show_detections()`

```python
def show_detections(
    figsize: tuple[int, int] = (12, 10),
    **kwargs,
) -> None
```

Display detections as bounding boxes on the source image using matplotlib. Only works after `detect()` (not `detect_tiled()`), because it requires the in-memory image.

**Parameters**:
- `figsize`: Figure size as (width, height) in inches.
- `**kwargs`: Additional arguments passed to matplotlib.

**Raises**: `DetrGeoError` if no detections or no image available.

---

## Module: detr_geo.io

Functions for reading geospatial rasters, selecting bands, and normalizing pixel values.

### BandSelector

```python
class BandSelector(bands: tuple[int, ...] | str = "rgb")
```

Selects and reorders bands from multi-band raster data.

**Presets**:

| Name | Bands | Sensor |
|------|-------|--------|
| `"rgb"` | 1, 2, 3 | Generic RGB |
| `"naip_rgb"` | 1, 2, 3 | NAIP |
| `"sentinel2_rgb"` | 4, 3, 2 | Sentinel-2 |
| `"worldview_rgb"` | 5, 3, 2 | WorldView |

**Properties**:
- `band_indices -> list[int]`: The 1-indexed band indices to read.

**Methods**:
- `select(data, num_bands) -> tuple[NDArray, NDArray | None]`: Select bands from a `(bands, H, W)` array. Returns `(rgb_array, alpha_mask)`.

### normalize_to_float32()

```python
def normalize_to_float32(
    data: NDArray,
    stretch: str = "percentile",
    percentiles: tuple[float, float] = (2.0, 98.0),
    stretch_params: StretchParams | None = None,
    nodata_mask: NDArray | None = None,
) -> tuple[NDArray[np.float32], StretchParams]
```

Normalize raster data to float32 in [0, 1] for model input.

**Parameters**:
- `data`: Array of shape `(bands, H, W)` of any numeric dtype.
- `stretch`: One of `"percentile"`, `"minmax"`, or `"none"`.
- `percentiles`: Low/high percentiles for percentile stretch.
- `stretch_params`: Pre-computed parameters for consistent normalization.
- `nodata_mask`: Boolean mask `(H, W)` where `True` = nodata pixel.

**Returns**: Tuple of `(normalized float32 array, StretchParams used)`.

### compute_scene_stretch_params()

```python
def compute_scene_stretch_params(
    raster_path: str,
    bands: list[int],
    percentiles: tuple[float, float] = (2.0, 98.0),
    sample_tiles: int = 20,
    tile_size: int = 512,
) -> StretchParams
```

Sample random tiles from a raster to compute per-scene stretch parameters. Use the returned `StretchParams` with `normalize_to_float32(stretch_params=params)` for consistent normalization across tiles.

### load_raster_metadata()

```python
def load_raster_metadata(source: str | Path) -> RasterMetadata
```

Load raster metadata without reading pixel data.

**Returns**: `RasterMetadata` dataclass with fields:
- `crs: CRS | None`
- `transform: Affine`
- `width: int`, `height: int`
- `count: int` (band count)
- `dtype: str`
- `nodata: float | None`
- `has_alpha: bool`
- `bounds: tuple[float, float, float, float]`
- `gsd: float | None` -- Ground sample distance in meters per pixel (computed from transform and CRS)

### compute_gsd()

```python
def compute_gsd(transform: Affine, crs: CRS | None) -> float | None
```

Compute ground sample distance in meters per pixel from a raster's affine transform and CRS. Returns `None` if the CRS is missing or the computation fails.

For projected CRS (units in meters), the GSD is derived directly from the transform pixel size. For geographic CRS (degrees), it is converted using the latitude at the raster center.

### check_gsd()

```python
def check_gsd(gsd: float | None, strict: bool = False) -> None
```

Check whether imagery GSD falls within the optimal range for vehicle detection (0.1--0.5 m/px). Issues a `GSDWarning` if the GSD is outside this range. If `strict=True` and GSD exceeds 5.0 m/px, raises `BandError` because objects are too few pixels to detect.

### read_tile()

```python
def read_tile(
    source: str | Path,
    window: tuple[int, int, int, int],
    bands: list[int] | None = None,
) -> tuple[NDArray, NDArray | None]
```

Read a tile from a raster using windowed (memory-efficient) read. Supports boundless windows (padding beyond raster extent).

**Parameters**:
- `source`: Path to raster file.
- `window`: `(col_off, row_off, width, height)` pixel coordinates.
- `bands`: 1-indexed band indices. `None` reads all bands.

**Returns**: `(data array (bands, H, W), nodata_mask (H, W) or None)`.

### compute_nodata_fraction()

```python
def compute_nodata_fraction(
    data: NDArray,
    nodata_value: float | None,
    alpha_mask: NDArray | None = None,
) -> float
```

Compute the fraction of nodata pixels in a tile. Priority: alpha_mask > nodata_value > return 0.0.

### fill_nodata()

```python
def fill_nodata(data: NDArray, nodata_mask: NDArray) -> NDArray
```

Fill nodata pixels with per-band mean of valid pixels. Returns a copy.

---

## Module: detr_geo.crs

CRS handling and pixel-to-geographic coordinate transforms.

### pixel_to_geo()

```python
def pixel_to_geo(
    bbox: PixelBBox,
    transform: Affine,
) -> Polygon
```

Convert a pixel bounding box to a CRS-space polygon. Uses 4-corner conversion for correctness with rotated affine transforms.

**Parameters**:
- `bbox`: `(x_min, y_min, x_max, y_max)` in pixel coordinates.
- `transform`: Rasterio affine transform.

**Returns**: Shapely Polygon with 4 corners.

### get_transformer()

```python
@lru_cache(maxsize=32)
def get_transformer(src_crs: str, dst_crs: str) -> Transformer
```

Get a cached pyproj Transformer. Uses `always_xy=True` to avoid axis order confusion.

### validate_crs()

```python
def validate_crs(
    crs: CRS | str | None,
    georeferenced: bool = True,
) -> CRS | None
```

Validate a CRS value.

**Raises**:
- `MissingCRSError`: If `crs` is `None` and `georeferenced=True`.
- `CRSError`: If the CRS string is invalid.

### auto_utm_crs()

```python
def auto_utm_crs(longitude: float, latitude: float) -> CRS
```

Determine the appropriate UTM zone CRS for a given lon/lat point. Returns EPSG:326xx (north) or EPSG:327xx (south).

### has_rotation()

```python
def has_rotation(transform: Affine) -> bool
```

Check if an affine transform includes rotation/shear terms (`b != 0` or `d != 0`).

---

## Module: detr_geo.tiling

Tile grid generation, NMS, and the tiled detection pipeline.

### generate_tile_grid()

```python
def generate_tile_grid(
    raster_width: int,
    raster_height: int,
    tile_size: int,
    overlap_ratio: float = 0.2,
) -> list[TileInfo]
```

Generate a tile grid covering the entire raster.

**Parameters**:
- `raster_width`, `raster_height`: Raster dimensions in pixels.
- `tile_size`: Size of each square tile in pixels.
- `overlap_ratio`: Fractional overlap (0.0 to 0.49).

**Returns**: List of `TileInfo` dicts with `window`, `global_offset_x`, `global_offset_y`, `nodata_fraction`.

**Raises**: `TilingError` if `overlap_ratio >= 0.5` or `tile_size <= 0`.

### cross_tile_nms()

```python
def cross_tile_nms(
    boxes: NDArray,
    scores: NDArray,
    class_ids: NDArray,
    iou_threshold: float = 0.5,
) -> NDArray
```

Class-aware non-maximum suppression across tiles. Only suppresses same-class detections with IoU above the threshold.

**Parameters**:
- `boxes`: `(N, 4)` array in `[x1, y1, x2, y2]` format (global pixel coordinates).
- `scores`: `(N,)` confidence scores.
- `class_ids`: `(N,)` class IDs.
- `iou_threshold`: IoU above which to suppress.

**Returns**: Boolean mask `(N,)` where `True` = detection survives.

### recommended_overlap()

```python
def recommended_overlap(
    object_size_pixels: float,
    object_size_fraction: float = 0.3,
    tile_size: int = 576,
) -> float
```

Recommend overlap ratio based on expected object size relative to tile.

### detection_range()

```python
def detection_range(
    tile_size: int,
    gsd: float,
    overlap: float = 0.2,
) -> tuple[float, float]
```

Compute the effective detection size range `(min_meters, max_meters)` for a given tile size and ground sample distance.

### process_tiles()

```python
def process_tiles(
    raster_path: str,
    adapter: RFDETRAdapter,
    tile_size: int,
    overlap: float = 0.2,
    nms_threshold: float = 0.5,
    nodata_threshold: float = 0.5,
    threshold: float | None = None,
    batch_size: int | None = None,
    bands: tuple[int, ...] | str = "rgb",
    show_progress: bool = True,
) -> tuple[NDArray, NDArray, NDArray]
```

Run the full tiled detection pipeline: generate grid, read tiles, normalize, detect, offset, NMS.

**Returns**: `(boxes, scores, class_ids)` in global pixel coordinates.

---

## Module: detr_geo.export

GeoDataFrame construction and vector format export.

### build_geodataframe()

```python
def build_geodataframe(
    boxes: NDArray,
    scores: NDArray,
    class_ids: NDArray,
    class_names: dict[int, str] | None,
    transform: Affine,
    crs: CRS,
) -> GeoDataFrame
```

Convert detection arrays into a georeferenced GeoDataFrame.

### build_dataframe_pixel()

```python
def build_dataframe_pixel(
    boxes: NDArray,
    scores: NDArray,
    class_ids: NDArray,
    class_names: dict[int, str] | None = None,
) -> DataFrame
```

Build a plain DataFrame with pixel-space coordinates (`x1, y1, x2, y2`).

### compute_areas()

```python
def compute_areas(
    gdf: GeoDataFrame,
    equal_area_crs: CRS | None = None,
) -> Series
```

Compute polygon areas in square meters. Auto-detects UTM zone for geographic CRS.

### export_geojson()

```python
def export_geojson(
    gdf: GeoDataFrame,
    path: str,
    coordinate_precision: int = 6,
) -> None
```

Export to GeoJSON. Auto-reprojects to WGS84.

### export_gpkg()

```python
def export_gpkg(
    gdf: GeoDataFrame,
    path: str,
    layer: str = "detections",
) -> None
```

Export to GeoPackage.

### export_shp()

```python
def export_shp(gdf: GeoDataFrame, path: str) -> None
```

Export to Shapefile. Issues legacy format warning.

---

## Module: detr_geo.viz

Visualization with matplotlib and leafmap.

### show_detections()

```python
def show_detections(
    image: NDArray,
    gdf: GeoDataFrame,
    figsize: tuple[int, int] = (12, 10),
    min_confidence: float = 0.0,
    top_n: int | None = None,
    classes: list[str] | None = None,
    class_colors: dict[str, str] | None = None,
    show_labels: bool = True,
    save_path: str | None = None,
    dpi: int = 150,
) -> None
```

Render detections as bounding boxes on the source image using matplotlib.

**Parameters**:
- `image`: Source image as `(H, W, 3)` array in [0, 1] or [0, 255].
- `gdf`: GeoDataFrame with `geometry`, `class_name`, `confidence` columns.
- `min_confidence`: Minimum confidence for display.
- `top_n`: Limit to top N detections by confidence.
- `classes`: Filter to specific class names.
- `class_colors`: Override colors for specific classes (e.g., `{"building": "red"}`).
- `show_labels`: Draw class name and confidence on each box.
- `save_path`: If provided, save the figure to this path.
- `dpi`: Resolution for saved figure.

### show_map()

```python
def show_map(
    gdf: GeoDataFrame,
    basemap: str = "SATELLITE",
    min_confidence: float = 0.0,
    classes: list[str] | None = None,
    class_colors: dict[str, str] | None = None,
    max_detections: int = 1000,
    map_object: leafmap.Map | None = None,
    **kwargs,
) -> leafmap.Map
```

Display detections on an interactive leafmap with satellite basemap.

**Parameters**:
- `gdf`: GeoDataFrame with detection polygons (any CRS -- auto-reprojects to 4326).
- `basemap`: One of `"SATELLITE"`, `"ROADMAP"`, `"TERRAIN"`.
- `max_detections`: Cap on displayed detections to prevent slow rendering.
- `map_object`: Existing leafmap.Map to add detections to.

**Returns**: leafmap.Map object.

### get_class_colors()

```python
def get_class_colors(
    class_names: list[str],
    user_colors: dict[str, str] | None = None,
) -> dict[str, tuple[float, float, float]]
```

Generate consistent class colors using matplotlib's tab20 colormap. User can override specific classes.

---

## Module: detr_geo._adapter

RF-DETR model adapter (internal, but useful for advanced users).

### RFDETRAdapter

```python
class RFDETRAdapter(
    model_size: str = "medium",
    device: str | None = None,
    pretrain_weights: str | None = None,
    confidence_threshold: float = 0.5,
    custom_class_names: dict[int, str] | None = None,
)
```

**Properties**:
- `resolution -> int`: Native input resolution.
- `block_size -> int`: Dimension divisibility constraint.
- `patch_size -> int`: Vision transformer patch size.
- `num_windows -> int`: Attention mechanism window count.
- `class_names -> dict[int, str]`: Class ID to name mapping (empty until model is loaded).
- `num_select -> int`: Maximum detections per image (default 300).
- `is_loaded -> bool`: Whether model weights have been loaded.

**Methods**:
- `predict_tile(image, threshold) -> DetectionResult`: Run inference on a single PIL Image.
- `predict_batch(images, threshold) -> list[DetectionResult]`: Run inference on multiple images (currently sequential).
- `model_info() -> dict`: Model configuration summary.
- `valid_tile_sizes(min_size, max_size) -> list[int]`: All tile sizes divisible by block_size.
- `auto_batch_size() -> int`: Estimate optimal batch size from GPU memory.
- `optimize(batch_size)`: Optimize model for repeated inference.
- `remove_optimization()`: Clear optimized state.

### Module Functions

- `detect_device(preferred) -> str`: Auto-detect compute device (cuda > mps > cpu).
- `validate_tile_input(image, block_size)`: Validate image before inference.
- `prepare_tile_image(tile_array) -> PIL.Image`: Convert `(3, H, W)` float32 array to PIL RGB.

---

## Module: detr_geo.training

Training pipeline for fine-tuning RF-DETR on custom geospatial datasets. Provides spatial-aware dataset splitting, COCO/YOLO annotation export, augmentation presets for overhead imagery, and a training wrapper.

### SpatialSplitter

```python
class SpatialSplitter(
    method: str = "block",
    ratios: tuple[float, float, float] = (0.8, 0.15, 0.05),
    buffer_tiles: int = 1,
    seed: int = 42,
)
```

Split tiles into train/val/test sets with spatial awareness to prevent data leakage.

**Parameters**:
- `method`: `"block"` assigns contiguous spatial blocks with buffer zones between splits (recommended). `"random"` assigns randomly with a leakage warning.
- `ratios`: `(train, val, test)` fractions. Must sum to 1.0.
- `buffer_tiles`: Number of buffer tiles discarded between blocks (block method only). Prevents spatial autocorrelation across splits.
- `seed`: Random seed for reproducibility.

**Methods**:
- `split(tiles, raster_width, raster_height, tile_size) -> SplitResult`: Assign tile indices to splits. Returns a `SplitResult` with `train_indices`, `val_indices`, `test_indices`, and `buffer_indices`.

### prepare_training_dataset()

```python
def prepare_training_dataset(
    raster_path: str,
    annotations_path: str,
    output_dir: str,
    tile_size: int = 576,
    overlap_ratio: float = 0.0,
    class_mapping: dict[str, int] | None = None,
    min_annotation_area: int = 100,
    max_background_per_annotated: float = 3.0,
    bands: tuple[int, ...] | str = "rgb",
    split_method: str = "block",
    split_ratios: tuple[float, float, float] = (0.8, 0.15, 0.05),
    seed: int = 42,
) -> dict[str, int]
```

Convert a GeoTIFF and vector annotations (GeoJSON/GeoPackage/Shapefile) into a COCO-format training dataset. Tiles the raster, aligns CRS between raster and annotations, clips annotations to tile boundaries, splits with spatial awareness, and writes the output.

**Parameters**:
- `raster_path`: Path to the GeoTIFF raster.
- `annotations_path`: Path to vector annotations (GeoJSON, GeoPackage, or Shapefile).
- `output_dir`: Output directory for the COCO-format dataset.
- `tile_size`: Tile size in pixels. Should match the model's native resolution.
- `overlap_ratio`: Fractional overlap between tiles during preparation (0.0 is typical for training).
- `class_mapping`: Maps class attribute values to integer category IDs. Auto-generated if `None`.
- `min_annotation_area`: Minimum annotation area in square pixels. Smaller annotations are discarded.
- `max_background_per_annotated`: Controls the ratio of background (empty) tiles to annotated tiles.
- `bands`: Band selection preset or tuple.
- `split_method`: `"block"` or `"random"`.
- `split_ratios`: `(train, val, test)` fractions.
- `seed`: Random seed.

**Returns**: Dict with tile and annotation counts per split.

**Raises**: `CRSError` if CRS alignment fails. `FileNotFoundError` if inputs are missing.

### detections_to_coco()

```python
def detections_to_coco(
    gdf: GeoDataFrame,
    output_path: str,
    image_path: str | None = None,
    image_width: int | None = None,
    image_height: int | None = None,
) -> None
```

Export a detection GeoDataFrame to COCO annotation JSON format. Converts detection polygons to pixel-space `[x, y, width, height]` bounding boxes. Useful for the active learning loop: detect, review, correct, retrain.

### detections_to_yolo()

```python
def detections_to_yolo(
    gdf: GeoDataFrame,
    output_dir: str,
    image_name: str = "image",
    image_width: int | None = None,
    image_height: int | None = None,
) -> None
```

Export a detection GeoDataFrame to YOLO annotation format. Writes `.txt` files with normalized `class_id center_x center_y width height` lines.

### AugmentationPreset

```python
@dataclass
class AugmentationPreset:
    name: str
    random_rotation_90: bool = True
    horizontal_flip: bool = True
    vertical_flip: bool = True
    brightness_jitter: float = 0.2
    contrast_jitter: float = 0.2
    saturation_jitter: float = 0.1
```

Geospatial augmentation preset for overhead imagery. Adds operations like vertical flip and 90-degree rotation that standard augmentation pipelines omit (overhead images have no canonical "up" direction).

### AUGMENTATION_PRESETS

```python
AUGMENTATION_PRESETS: dict[str, AugmentationPreset]
```

Built-in presets:

| Preset | Brightness | Contrast | Saturation | Use Case |
|--------|-----------|----------|------------|----------|
| `"satellite_default"` | 0.2 | 0.2 | 0.1 | Satellite imagery (0.3--0.5 m GSD) |
| `"aerial_default"` | 0.3 | 0.3 | 0.15 | Aerial photography (0.1--0.3 m GSD) |
| `"drone_default"` | 0.4 | 0.4 | 0.2 | Drone imagery (<0.1 m GSD, high variability) |

All presets enable rotation, horizontal flip, and vertical flip.

### train()

```python
def train(
    adapter: RFDETRAdapter,
    dataset_dir: str,
    epochs: int = 50,
    augmentation_preset: str | AugmentationPreset | None = "satellite_default",
    augmentation_factor: int = 2,
    batch_size: int = 8,
    learning_rate: float | None = None,
    resume: str | None = None,
    seed: int = 42,
    **kwargs,
) -> dict
```

Train an RF-DETR model on a COCO-format dataset with geospatial augmentation.

**Parameters**:
- `adapter`: An `RFDETRAdapter` instance.
- `dataset_dir`: Path to COCO-format dataset with `train/` and `valid/` subdirectories.
- `epochs`: Number of training epochs.
- `augmentation_preset`: Preset name, `AugmentationPreset` instance, or `None` to skip pre-augmentation.
- `augmentation_factor`: Total copies including original (2 = 1 original + 1 augmented).
- `batch_size`: Training batch size.
- `learning_rate`: Learning rate override. `None` uses the model default.
- `resume`: Path to checkpoint for resuming training.
- `seed`: Random seed.
- `**kwargs`: Additional arguments passed to the adapter's `train()` method.

**Returns**: Training metrics dict from the adapter.

**Raises**: `ModelError` if rfdetr is not installed or training fails. `ValueError` if the preset name is invalid.

---

## Exception Types

**Module**: `detr_geo.exceptions`

All exceptions inherit from `DetrGeoError`, enabling catch-all handling.

| Exception | Parent | Raised When |
|-----------|--------|-------------|
| `DetrGeoError` | `Exception` | Base class for all detr-geo errors |
| `CRSError` | `DetrGeoError` | Invalid CRS string, incompatible transforms |
| `MissingCRSError` | `CRSError` | Raster has no CRS in georeferenced mode |
| `TilingError` | `DetrGeoError` | Invalid tile size, overlap out of range |
| `ModelError` | `DetrGeoError` | Unsupported model size, rfdetr not installed, inference failure |
| `BandError` | `DetrGeoError` | Invalid band index, wrong channel count, normalization failure |
| `ExportError` | `DetrGeoError` | Export file write failure, no detections to export |

All exceptions accept keyword arguments stored in `.context`:

```python
try:
    dg.set_image("file.tif")
except DetrGeoError as e:
    print(e)           # Human-readable message
    print(e.context)   # Additional context dict
```

---

## Type Definitions

**Module**: `detr_geo._typing`

| Type | Definition | Description |
|------|-----------|-------------|
| `PixelBBox` | `tuple[float, float, float, float]` | Pixel bounding box: (x_min, y_min, x_max, y_max) |
| `GeoBBox` | `tuple[float, float, float, float]` | Geographic bounding box: (west, south, east, north) |
| `TileWindow` | `tuple[int, int, int, int]` | Rasterio window: (col_off, row_off, width, height) |
| `DetectionResult` | `TypedDict` | `{"bbox": list[list[float]], "confidence": list[float], "class_id": list[int]}` |
| `TileInfo` | `TypedDict` | `{"window": TileWindow, "global_offset_x": int, "global_offset_y": int, "nodata_fraction": float}` |
| `ImageArray` | `NDArray[np.float32]` | Image array: (bands, height, width) float32 [0, 1] |
| `ModelSize` | `str` | One of: "nano", "small", "medium", "base", "large" |
