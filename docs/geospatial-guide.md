# Geospatial Analyst Guide

This guide covers the geospatial concepts and practical details you need to use detr-geo effectively with real satellite and aerial imagery. It assumes familiarity with GIS concepts like coordinate reference systems, raster data, and vector formats.

---

## How Tiling Works

### Why Tile?

Object detection models have a fixed input size. RF-DETR's "medium" model, for example, expects 576x576 pixel images. A typical satellite scene or orthomosaic is thousands to tens of thousands of pixels on each side. Tiling cuts the raster into model-sized chunks, runs detection on each, and merges the results.

### Overlap

Adjacent tiles overlap by a configurable ratio (default 0.2 = 20%). This means for a 576px tile, the stride between tiles is `576 * (1 - 0.2) = 461` pixels.

Why overlap? Objects at tile boundaries would be cut in half without it. A building that straddles two tiles would appear as a partial object in both, likely below the detection threshold. With 20% overlap, the entire building is visible in at least one tile.

**Choosing overlap**:
- 0.2 (20%): Good default for most object sizes
- 0.3 (30%): Better for large objects (buildings, ships) relative to tile size
- 0.1 (10%): Faster, but objects up to 10% of tile size may be missed at boundaries

Use the `recommended_overlap()` function to estimate from expected object size:

```python
from detr_geo.tiling import recommended_overlap

# Objects about 30 pixels wide in 576px tiles
overlap = recommended_overlap(object_size_pixels=30, tile_size=576)
# Returns ~0.17
```

### What Tile Size Should I Use?

By default, detr-geo uses the model's native resolution as the tile size. This is almost always the right choice. The model was trained at this resolution and performs best with it.

If you need custom tile sizes, they must be divisible by the model's block size:

```python
dg = DetrGeo(model_size="base")
valid_sizes = dg._adapter.valid_tile_sizes(min_size=256, max_size=1024)
# Returns sizes divisible by 56 (base model's block size)
```

### Detection Size Range

The combination of tile size and ground sample distance (GSD) determines what object sizes are detectable:

```python
from detr_geo.tiling import detection_range

# 576px tiles at 0.3m GSD (30cm aerial imagery)
min_m, max_m = detection_range(tile_size=576, gsd=0.3)
# min_m = 3.0 meters (smallest detectable)
# max_m = 138.2 meters (largest detectable)
```

Objects smaller than ~10 pixels are too small for the model. Objects larger than ~80% of the tile dominate the image and may confuse the detector.

---

## Band Selection for Different Sensors

RF-DETR expects 3-channel (RGB) input. Different sensors store RGB information in different bands. detr-geo provides presets for common sensors.

### NAIP (US aerial imagery)

NAIP stores RGB in bands 1-2-3 (with an optional NIR band 4):

```python
dg.set_image("naip_scene.tif", bands="rgb")       # or "naip_rgb"
```

### Sentinel-2 (ESA)

Sentinel-2 Level-2A has 12 bands. True color (RGB) is bands 4-3-2:

```python
dg.set_image("sentinel2_L2A.tif", bands="sentinel2_rgb")
```

Note: Sentinel-2 bands have different resolutions (10m, 20m, 60m). The `sentinel2_rgb` preset uses bands 4, 3, and 2 which are all 10m. If you use bands from different resolution groups, you must resample them to the same resolution before loading.

### WorldView (Maxar)

WorldView multispectral has 8 bands. True color is bands 5-3-2 (red-green-blue):

```python
dg.set_image("worldview_ms.tif", bands="worldview_rgb")
```

### Custom Sensors

For any sensor, specify 1-indexed band numbers as a tuple:

```python
# Landsat 8/9 true color: bands 4-3-2
dg.set_image("landsat_scene.tif", bands=(4, 3, 2))

# False color composite (NIR-R-G) for vegetation analysis
dg.set_image("sentinel2.tif", bands=(8, 4, 3))
```

The order matters -- the first band maps to Red, second to Green, third to Blue in the model input.

### Single-Band Rasters

Panchromatic or single-band rasters (e.g., SAR, DEM-derived hillshade) are automatically triplicated to 3 channels with a warning:

```python
dg.set_image("panchromatic.tif", bands=(1,))
# Warning: Single-band raster detected. Triplicating to 3 channels.
```

---

## CRS Handling

### What Happens to Your Coordinates

1. **Input**: detr-geo reads the CRS and affine transform from your raster's metadata via rasterio.

2. **Detection**: The model operates entirely in pixel space. Bounding boxes are pixel coordinates [x1, y1, x2, y2] relative to the image origin.

3. **Georeferencing**: After detection, pixel bounding boxes are converted to geographic polygons using the raster's affine transform. All four corners are transformed individually (not just opposite corners), which correctly handles rotated or sheared imagery.

4. **Output**: The resulting GeoDataFrame carries the same CRS as the input raster. GeoJSON export auto-reprojects to WGS84 (EPSG:4326).

### CRS Recommendations

**Best**: Use rasters in a projected CRS (UTM, State Plane, national grid). These have meters as the unit, which means pixel sizes correspond to real-world distances uniformly across the image.

**Acceptable**: Geographic CRS (WGS84, EPSG:4326). The library works, but be aware that at higher latitudes, a pixel covers less ground in the east-west direction. For most mid-latitude work, this distortion is minor.

**Avoid**: Any raster without an embedded CRS. You can still process it by setting `georeferenced=False`, but the output will be in pixel coordinates only -- no geographic meaning.

### Overriding CRS

If your raster has no CRS but you know what it should be:

```python
dg = DetrGeo()
dg.set_image("no_crs_raster.tif", georeferenced=False)
dg.crs = "EPSG:32618"  # Manually assign UTM zone 18N
```

### Area Computation

detr-geo provides area computation that automatically handles CRS:

```python
from detr_geo.export import compute_areas

areas_m2 = compute_areas(dg.detections)
```

- If your data is in a projected CRS (meters), area is computed directly.
- If your data is in geographic CRS (degrees), it auto-detects the appropriate UTM zone and reprojects before computing area.
- You can override with an explicit equal-area CRS: `compute_areas(gdf, equal_area_crs=CRS.from_epsg(6933))`.

---

## 16-Bit Imagery Handling

### The Problem

RF-DETR expects 8-bit RGB images (0-255 values). Satellite sensors produce 12-bit or 16-bit imagery with much wider value ranges. A Sentinel-2 pixel might have a value of 3000 for a vegetation surface -- this must be mapped to the 0-255 range before the model can use it.

### Percentile Stretch (Default)

detr-geo uses 2nd-to-98th percentile stretching by default. This means:
1. For each band, find the 2nd percentile value (call it `low`) and 98th percentile value (call it `high`).
2. Map `low` to 0.0 and `high` to 1.0.
3. Clip values outside this range.

This is the same technique as QGIS's "cumulative count cut" or ArcGIS's "percentage clip" renderer. It handles most imagery well by ignoring extreme outliers.

### Alternative Stretch Modes

```python
from detr_geo.io import normalize_to_float32

# Percentile stretch (default) -- best for satellite imagery
normalized, params = normalize_to_float32(data, stretch="percentile")

# Min-max stretch -- uses full value range, sensitive to outliers
normalized, params = normalize_to_float32(data, stretch="minmax")

# No stretch -- assumes data is already in [0, 1]
normalized, params = normalize_to_float32(data, stretch="none")
```

### Consistent Scene Normalization

For large rasters processed in tiles, you should compute stretch parameters once for the entire scene, then apply them to every tile. This prevents brightness variations between adjacent tiles:

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

Note: The current `detect_tiled()` method does not use scene-level stretch parameters automatically. This is a known limitation. For 16-bit imagery processed in tiled mode, consider using the lower-level tiling functions directly with pre-computed stretch parameters.

---

## Nodata Handling

### How Nodata Is Detected

1. **Raster nodata value**: Read from the raster metadata (e.g., `nodata=0` or `nodata=-9999`). A pixel is "nodata" when all bands equal this value.

2. **Alpha band**: If the raster has 4 bands and 3 are requested (RGBA), the 4th band is treated as an alpha mask. Pixels with alpha=0 are nodata.

Priority: alpha mask > nodata value > assume no nodata.

### Tile Skipping

Tiles with nodata above the threshold (default 50%) are skipped entirely:

```python
dg.detect_tiled(nodata_threshold=0.5)  # Skip tiles that are >50% nodata
```

Set to 0.0 to process all tiles, or 1.0 to only process completely valid tiles.

### Nodata Fill

Tiles with partial nodata (below the threshold) are filled before detection. The fill value is the per-band mean of valid pixels in that tile. This prevents the nodata area from having values that confuse the detector.

**Limitation**: Mean fill produces a flat, unrealistic patch. For imagery where nodata regions are significant (e.g., cloud masks with large gaps), detection quality near the nodata boundary may be reduced.

---

## Export Formats

### GeoJSON (`.geojson`)

**When to use**: Web visualization, Leaflet/Mapbox integration, sharing results.

```python
dg.to_geojson("results.geojson")
```

- Automatically reprojects to WGS84 (EPSG:4326) per the GeoJSON specification.
- Text-based format, human-readable.
- Coordinate precision defaults to 6 decimal places (~10 cm).
- No field name length limits.

### GeoPackage (`.gpkg`) -- Recommended

**When to use**: Most workflows. Best all-around format.

```python
dg.to_gpkg("results.gpkg", layer="buildings")
```

- SQLite-based single file. No sidecar files.
- Preserves the original CRS (no forced reprojection).
- Supports multiple layers in one file.
- No field name or file size limits.
- Works with QGIS, ArcGIS, PostGIS, and all modern GIS tools.

### Shapefile (`.shp`) -- Legacy

**When to use**: Only when required by a legacy system or data standard.

```python
dg.to_shp("results.shp")
# Warning: Shapefile format has legacy limitations...
```

- Field names truncated to 10 characters (`class_name` becomes `class_name` but `centroid_x` becomes `centroid_x` -- both happen to be under 10, but be careful with custom fields).
- 2 GB file size limit.
- Requires 4 sidecar files (.dbf, .prj, .shx, .cpg).
- No null value support.
- detr-geo issues a warning when using this format.

### Working with the GeoDataFrame Directly

For further analysis without exporting:

```python
gdf = dg.detections  # geopandas.GeoDataFrame

# Standard geopandas operations
buildings = gdf[gdf["class_name"] == "building"]
high_confidence = gdf[gdf["confidence"] > 0.8]
total_area = compute_areas(buildings).sum()

# Spatial operations
from shapely.geometry import box
aoi = box(-122.1, 37.0, -121.9, 37.1)
in_aoi = gdf[gdf.intersects(aoi)]
```

---

## Visualization

### Static Matplotlib Plot

`show_detections()` draws bounding boxes on the source image. Works in scripts and Jupyter notebooks.

```python
dg.show_detections(
    figsize=(15, 12),        # Figure size in inches
    min_confidence=0.5,      # Only show confident detections
    top_n=50,                # Limit to top 50
    classes=["building"],    # Filter to specific classes
    class_colors={"building": "red", "vehicle": "blue"},
    show_labels=True,        # Show class name and confidence
    save_path="output.png",  # Save to file
    dpi=300,                 # Print-quality resolution
)
```

Limitations:
- Only works after `detect()` (not `detect_tiled()`), because it needs the in-memory image.
- Bounding boxes are drawn in pixel coordinates from the geometry bounds.

### Interactive Leafmap

`show_map()` creates an interactive web map with detection polygons overlaid on a satellite basemap. Best in Jupyter notebooks.

```python
m = dg.show_map(
    basemap="SATELLITE",     # or "ROADMAP", "TERRAIN"
    min_confidence=0.5,
    classes=["building"],
    class_colors={"building": "red"},
    max_detections=1000,     # Limit for rendering performance
)
m  # Display in Jupyter
```

Features:
- Detections are auto-reprojected to WGS84 for map display.
- Click on a polygon to see class, confidence, and area.
- Opacity scales with confidence (higher confidence = more opaque).
- You can pass an existing leafmap Map object to add detections as a layer.

---

## Working with Large Rasters

### Memory Management

`detect()` loads the full raster into memory. For a 20,000 x 20,000, 3-band, uint16 raster, that is ~2.4 GB. Use `detect_tiled()` instead, which reads one tile at a time via rasterio's windowed reads.

### Processing Speed

Factors that affect speed:
- **Model size**: nano is ~5x faster than large on GPU
- **Tile size**: Larger tiles = fewer tiles but each takes longer
- **GPU vs CPU**: GPU is 10-50x faster depending on model size
- **Nodata**: Tiles with >50% nodata are skipped (saves inference time)

### Recommended Workflow for Large Scenes

```python
from detr_geo import DetrGeo

dg = DetrGeo(model_size="medium", device="cuda")
dg.set_image("large_ortho.tif")

detections = dg.detect_tiled(
    tile_size=None,         # Use model's native resolution (576)
    overlap=0.2,            # 20% overlap
    nms_threshold=0.5,      # Standard NMS
    nodata_threshold=0.5,   # Skip mostly-empty tiles
    threshold=0.3,          # Lower confidence threshold (filter later)
    batch_size=None,        # Auto-detect from GPU memory
)

# Filter to high-confidence results
final = detections[detections["confidence"] > 0.7]

# Export
final.to_file("results.gpkg", driver="GPKG")
```

---

## Common Gotchas

### 1. "All my detections are at 0,0"

Your raster has no CRS or the CRS is not being read. Check:
```python
from detr_geo.io import load_raster_metadata
meta = load_raster_metadata("your_file.tif")
print(meta.crs, meta.transform)
```

If `crs` is None, either set `georeferenced=False` for pixel-only mode or manually assign a CRS.

### 2. "Detections look offset from the basemap"

This usually means a CRS mismatch or datum issue. Verify that the CRS in your raster matches the actual coordinate system of the data. Some rasters have incorrect CRS metadata.

### 3. "The same object appears multiple times"

This happens when NMS is not working effectively. Common causes:
- The NMS threshold is too high (try lowering from 0.5 to 0.3)
- Objects are assigned different class labels in different tiles
- Overlap ratio is too small for the object size

### 4. "16-bit imagery looks washed out or too dark"

The percentile stretch may not suit your data. Try:
- Adjusting percentiles: `normalize_to_float32(data, percentiles=(1.0, 99.0))`
- Using min-max stretch: `normalize_to_float32(data, stretch="minmax")`
- Pre-computing scene-level stretch parameters

### 5. "Shapefile field names are truncated"

Use GeoPackage instead: `dg.to_gpkg("output.gpkg")`.

### 6. "Memory error on large raster"

Use `detect_tiled()` instead of `detect()`. The single-image `detect()` loads the entire raster into memory.

### 7. "Band index out of range"

Band indices are 1-based (matching rasterio convention), not 0-based. Band 1 is the first band, not band 0.

### 8. "GeoJSON coordinates are in a different CRS than my input"

This is correct behavior. The GeoJSON specification requires WGS84 (EPSG:4326). detr-geo automatically reprojects when exporting to GeoJSON and issues a warning. Use GeoPackage if you need the original CRS.
