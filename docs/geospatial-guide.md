# Geospatial Guide

Everything you need to know about the spatial side of detr-geo: how tiling works, what happens to your coordinates, how to handle different sensors, and how to avoid the most common gotchas in geospatial ML.

If you are an ML engineer who has not worked with geospatial data before, start here. If you are a GIS analyst who has not used object detection models, this guide will also cover the model constraints you need to understand.

---

## The Core Problem detr-geo Solves

Object detection models have a fixed input size. RF-DETR's "medium" model expects a 576x576 pixel image. A typical satellite scene is 10,000+ pixels on each side. A drone orthomosaic can be 50,000+.

You cannot resize the image down -- a car that is 15 pixels wide in the original becomes 1 pixel wide after downscaling and disappears. You cannot feed the whole image in -- the model physically cannot accept it.

detr-geo solves this by cutting the raster into model-sized tiles, running detection on each tile, and merging the results back into a single georeferenced layer. The coordinates in the output are real-world geographic coordinates, not pixel offsets.

---

## How Tiling Works

### The Grid

`detect_tiled()` generates a grid of overlapping square tiles across the raster:

```
+------+------+------+
|      |//////|      |
|  T1  |/ T2 /|  T3  |
|      |//////|      |
+------+------+------+
|//////|//////|//////|
|/ T4 /|/ T5 /|/ T6 /|
|//////|//////|//////|
+------+------+------+
```

The `//////` regions show where tiles overlap. Every object near a tile boundary appears in full in at least one tile, preventing boundary artifacts.

### Overlap

The overlap ratio (default 0.2 = 20%) controls how much adjacent tiles share. For a 576px tile with 0.2 overlap, the stride between tile origins is `576 * (1 - 0.2) = 461` pixels.

**Why does this matter?** Without overlap, an object straddling two tiles is split in half in both. Each half is likely too small to be detected. With 20% overlap, any object up to 20% of the tile width (115 pixels for a 576px tile) is guaranteed to appear complete in at least one tile.

**Choosing overlap:**

| Overlap | When to use |
|---|---|
| 0.1 (10%) | Small objects relative to tile size. Fastest. |
| 0.2 (20%) | Good default for most scenarios |
| 0.3 (30%) | Large objects (buildings, ships) relative to tile |

Estimate from expected object size:

```python
from detr_geo.tiling import recommended_overlap

# Vehicles ~15 pixels wide in 576px tiles
overlap = recommended_overlap(object_size_pixels=15, tile_size=576)
```

### Cross-Tile NMS

Because tiles overlap, the same object is often detected in multiple tiles. Cross-tile NMS (non-maximum suppression) deduplicates these:

1. All detections from all tiles are collected in global pixel coordinates
2. For each class independently, detections are sorted by confidence
3. If two same-class detections overlap above the IoU threshold (default 0.5), the lower-confidence one is removed

This is class-aware: a car and a building at the same location both survive. Two "car" detections of the same vehicle do not.

### Detection Size Range

The tile size and ground sample distance (GSD) together determine what objects are detectable:

```python
from detr_geo.tiling import detection_range

# 576px tiles at 0.3m GSD
min_m, max_m = detection_range(tile_size=576, gsd=0.3)
# min_m ~ 3.0m (smallest detectable object)
# max_m ~ 138m (largest detectable object)
```

Objects smaller than ~10 pixels are invisible to the model. Objects larger than ~80% of the tile dominate the image and confuse the detector.

### Tile Size

By default, detr-geo uses the model's native resolution. This is almost always correct -- the model was trained at this resolution and performs best with it.

Custom tile sizes must be divisible by the model's block size:

```python
dg = DetrGeo(model_size="base")
valid = dg._adapter.valid_tile_sizes(min_size=256, max_size=1024)
# Returns sizes divisible by 56 (base model block size)
```

---

## CRS: What Happens to Your Coordinates

### The Pipeline

1. **Load**: detr-geo reads the CRS and affine transform from the raster metadata via rasterio. This tells the library where each pixel is on Earth.

2. **Detect**: The model operates entirely in pixel space. It outputs bounding boxes as `[x1, y1, x2, y2]` pixel coordinates relative to the tile (or image) origin.

3. **Georeference**: After detection, pixel bounding boxes are converted to geographic polygons using the raster's affine transform. All four corners of each bounding box are transformed independently -- this handles rotated or sheared imagery correctly.

4. **Output**: The resulting GeoDataFrame carries the same CRS as the input raster. If you export to GeoJSON, it is automatically reprojected to WGS84 (EPSG:4326) per the GeoJSON spec.

### CRS Recommendations

**Use projected CRS when possible.** UTM, State Plane, or national grids have meters as the native unit. Pixel sizes correspond to real-world distances uniformly across the image, and area/distance calculations are straightforward.

**Geographic CRS works but has tradeoffs.** With WGS84 (EPSG:4326), a pixel covers fewer meters east-west at higher latitudes. For mid-latitude work the distortion is minor, but for high-latitude regions it becomes significant.

**No CRS?** You can still process the image by setting `georeferenced=False`. Output will be in pixel coordinates only:

```python
dg.set_image("no_crs.tif", georeferenced=False)
detections = dg.detect()
# detections has x1, y1, x2, y2 columns instead of geometry
```

If you know what CRS the raster should have, assign it manually:

```python
dg.set_image("no_crs.tif", georeferenced=False)
dg.crs = "EPSG:32618"  # UTM zone 18N
```

### Area Computation

detr-geo provides CRS-aware area computation:

```python
from detr_geo.export import compute_areas

areas_m2 = compute_areas(dg.detections)
```

- Projected CRS (meters): area computed directly from geometry
- Geographic CRS (degrees): auto-detects UTM zone and reprojects before computing
- Override: `compute_areas(gdf, equal_area_crs=CRS.from_epsg(6933))`

---

## Band Selection for Different Sensors

RF-DETR expects 3-channel RGB input. Different sensors store RGB in different band positions. detr-geo provides presets for common sensors so you do not have to remember band numbers.

### Quick Reference

| Sensor | Preset | Bands (1-indexed) |
|---|---|---|
| Generic RGB | `"rgb"` | 1, 2, 3 |
| NAIP (US aerial) | `"naip_rgb"` | 1, 2, 3 |
| Sentinel-2 (ESA) | `"sentinel2_rgb"` | 4, 3, 2 |
| WorldView (Maxar) | `"worldview_rgb"` | 5, 3, 2 |

### Examples

```python
# NAIP -- RGB is already in bands 1-2-3
dg.set_image("naip_scene.tif", bands="rgb")

# Sentinel-2 L2A -- true color is B4 (Red), B3 (Green), B2 (Blue)
dg.set_image("sentinel2.tif", bands="sentinel2_rgb")

# WorldView multispectral -- true color is B5, B3, B2
dg.set_image("worldview.tif", bands="worldview_rgb")

# Landsat 8/9 -- true color is B4, B3, B2
dg.set_image("landsat.tif", bands=(4, 3, 2))

# False color (NIR-R-G) for vegetation emphasis
dg.set_image("sentinel2.tif", bands=(8, 4, 3))
```

The order matters: first band maps to Red, second to Green, third to Blue in the model input.

### Sentinel-2 Band Resolution

Sentinel-2 bands have different spatial resolutions (10m, 20m, 60m). The `sentinel2_rgb` preset uses bands 4, 3, 2 -- all at 10m. If you mix bands from different resolution groups, resample to a common resolution before loading.

### Single-Band Rasters

Panchromatic or single-band rasters are automatically triplicated to 3 channels:

```python
dg.set_image("pan.tif", bands=(1,))
# Warning: Single-band raster detected. Triplicating to 3 channels.
```

---

## 16-Bit Imagery

### The Problem

RF-DETR was designed for 8-bit RGB images (0--255). Satellite sensors produce 12-bit or 16-bit imagery with pixel values in the thousands. A Sentinel-2 vegetation pixel might have a value of 3000. This must be mapped to a 0--1 range before the model can process it.

### Percentile Stretch (Default)

detr-geo uses 2nd-to-98th percentile stretching by default:

1. For each band, find the 2nd percentile value (`low`) and 98th percentile (`high`)
2. Map `low` to 0.0 and `high` to 1.0
3. Clip values outside this range

This is equivalent to QGIS's "cumulative count cut" or ArcGIS's "percentage clip" stretch. It handles most imagery well by ignoring extreme outliers at both ends.

### Other Stretch Modes

```python
from detr_geo.io import normalize_to_float32

# Percentile (default) -- best for satellite imagery
normalized, params = normalize_to_float32(data, stretch="percentile")

# Min-max -- uses full value range, sensitive to outliers
normalized, params = normalize_to_float32(data, stretch="minmax")

# No stretch -- assumes data is already [0, 1]
normalized, params = normalize_to_float32(data, stretch="none")
```

### Consistent Scene Normalization

When processing a large raster in tiles, each tile is normalized independently by default. This can cause brightness differences between adjacent tiles.

For consistent results, compute stretch parameters once for the whole scene and apply them to every tile:

```python
from detr_geo.io import compute_scene_stretch_params, normalize_to_float32

# Sample the scene to compute consistent parameters
params = compute_scene_stretch_params(
    "large_scene.tif",
    bands=[1, 2, 3],
    percentiles=(2.0, 98.0),
    sample_tiles=20,
)

# Apply to each tile
normalized, _ = normalize_to_float32(tile_data, stretch_params=params)
```

Note: `detect_tiled()` does not use scene-level stretch parameters automatically. For 16-bit imagery in tiled mode, use the lower-level tiling functions with pre-computed stretch parameters.

---

## Nodata Handling

### How Nodata Is Detected

1. **Raster nodata value**: Read from raster metadata (e.g., `nodata=0` or `nodata=-9999`). A pixel is nodata when all bands equal this value.
2. **Alpha band**: If the raster has 4 bands and 3 are requested, the 4th is treated as alpha. Pixels with alpha=0 are nodata.

Priority: alpha mask > nodata value > assume no nodata.

### Tile Skipping

Tiles above the nodata threshold are skipped entirely, saving inference time:

```python
dg.detect_tiled(nodata_threshold=0.5)  # Skip tiles >50% nodata
```

- `0.0` -- process all tiles regardless
- `0.5` -- skip tiles that are more than half empty (default)
- `1.0` -- only process completely filled tiles

### Nodata Fill

Tiles with partial nodata (below the threshold) are filled with the per-band mean of valid pixels before detection. This prevents nodata regions from producing false detections.

---

## Export Formats

### GeoPackage (.gpkg) -- Recommended

The modern standard for vector geospatial data. Single file, no size limit, preserves CRS, supports multiple layers, works with every modern GIS tool.

```python
dg.to_gpkg("results.gpkg", layer="vehicles")
```

### GeoJSON (.geojson)

Web-friendly, human-readable. Auto-reprojects to WGS84 per the GeoJSON specification. Good for web maps, Leaflet, Mapbox, or sharing with non-GIS users.

```python
dg.to_geojson("results.geojson")
```

### Shapefile (.shp) -- Legacy

Use only when required by a legacy system. Limitations: 10-character field names, 2 GB file size, 4 sidecar files (.dbf, .prj, .shx, .cpg), no null values.

```python
dg.to_shp("results.shp")
```

### GeoDataFrame (in-memory)

For further analysis without writing to disk:

```python
gdf = dg.detections

# Filter
cars = gdf[gdf["class_name"] == "Car"]
confident = gdf[gdf["confidence"] > 0.8]

# Spatial query
from shapely.geometry import box
aoi = box(-122.1, 37.0, -121.9, 37.1)
in_aoi = gdf[gdf.intersects(aoi)]

# Area computation
from detr_geo.export import compute_areas
areas = compute_areas(gdf)
```

---

## Visualization

### Static Plot (matplotlib)

```python
dg.show_detections(
    figsize=(15, 12),
    min_confidence=0.5,
    top_n=50,
    classes=["Car", "Truck"],
    class_colors={"Car": "cyan", "Truck": "red"},
    show_labels=True,
    save_path="output.png",
    dpi=300,
)
```

Works after `detect()` only (requires the in-memory image). For tiled results, export to GeoPackage and visualize in QGIS.

### Interactive Map (leafmap)

```python
m = dg.show_map(
    basemap="SATELLITE",
    min_confidence=0.5,
    classes=["Car"],
    max_detections=1000,
)
m  # Display in Jupyter
```

Features:
- Auto-reprojects detections to WGS84 for map display
- Click polygons for class, confidence, and area
- Opacity scales with confidence

---

## Working with Large Rasters

### Memory Strategy

`detect()` loads the full raster into memory. A 20,000 x 20,000, 3-band uint16 raster is ~2.4 GB. Use `detect_tiled()` instead -- it reads one tile at a time via rasterio windowed reads and never loads the full image.

### Recommended Workflow

```python
dg = DetrGeo(model_size="medium", device="cuda")
dg.set_image("city_orthomosaic.tif")

detections = dg.detect_tiled(
    overlap=0.2,
    nms_threshold=0.5,
    nodata_threshold=0.5,
    threshold=0.3,          # Lower threshold (filter later)
)

# Post-filter to high confidence
final = detections[detections["confidence"] > 0.7]
final.to_file("results.gpkg", driver="GPKG")
```

### Performance Factors

| Factor | Impact |
|---|---|
| Model size | nano is ~5x faster than large on GPU |
| GPU vs CPU | GPU is 10--50x faster |
| Nodata | Skipping empty tiles saves inference time |
| Tile size | Larger tiles = fewer tiles, but each is slower |

---

## Common Gotchas

### "All detections are at 0,0"

The raster has no CRS or the CRS is not being read:

```python
from detr_geo.io import load_raster_metadata
meta = load_raster_metadata("file.tif")
print(meta.crs, meta.transform)
```

If `crs` is None, set `georeferenced=False` or assign a CRS manually.

### "Detections are offset from the basemap"

CRS mismatch or datum issue. Verify the raster's embedded CRS matches the actual data. Some rasters have incorrect CRS metadata.

### "Duplicate detections of the same object"

NMS is not aggressive enough. Try lowering `nms_threshold` from 0.5 to 0.3. Also check that overlap is sufficient -- if objects span the non-overlapping gap between tiles, they appear in both tiles but NMS cannot match them.

### "16-bit imagery looks washed out"

Adjust the percentile stretch:

```python
from detr_geo.io import normalize_to_float32
normalized, _ = normalize_to_float32(data, percentiles=(1.0, 99.0))
```

Or use scene-level stretch parameters for consistency.

### "GeoJSON coordinates differ from my input CRS"

Correct behavior. GeoJSON requires WGS84 (EPSG:4326), so detr-geo reprojects automatically. Use GeoPackage if you need the original CRS preserved.

### "Band index out of range"

Band indices are 1-based (matching rasterio convention), not 0-based. Band 1 is the first band.

### "Memory error on large raster"

Use `detect_tiled()` instead of `detect()`.
