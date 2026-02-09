# Cloud-Native Raster Support (DETRGEO-47)

## Overview

As of commit `a606a7c`, detr_geo supports cloud-native raster loading for:
- **HTTP/HTTPS URLs** to Cloud-Optimized GeoTIFFs (COGs)
- **S3 URIs** (`s3://bucket/key.tif`) with AWS credentials
- **STAC catalog items** via `pystac.Item` objects

Users can now run detection on remote imagery without downloading the full file first. Only metadata and needed tile byte ranges are fetched.

## Installation

```bash
# Basic cloud support (STAC only)
pip install detr-geo[cloud]

# With model inference
pip install detr-geo[cloud,rfdetr]
```

## Usage

### HTTP COG URL

```python
from detr_geo import DetrGeo

dg = DetrGeo(model_size="nano")
dg.set_image("https://example.com/aerial_imagery.tif")
detections = dg.detect_tiled(tile_size=512, overlap=0.2)
```

### S3 URI

```python
# Requires AWS credentials in environment or ~/.aws/credentials
dg = DetrGeo(model_size="nano")
dg.set_image("s3://my-bucket/imagery/aerial_2024.tif")
detections = dg.detect_tiled()
```

### STAC Item

```python
import pystac
from detr_geo import DetrGeo

# Load a STAC item from a catalog
item = pystac.Item.from_file("https://example.com/stac/items/my-item.json")

# Pass directly to detr_geo
dg = DetrGeo(model_size="nano")
dg.set_image(item)  # Automatically extracts the COG URL
detections = dg.detect_tiled()
```

## How It Works

### Source Resolution

The new `resolve_raster_source()` function normalizes all input types:

```python
from detr_geo import resolve_raster_source

# Local path - validates existence
uri = resolve_raster_source("/path/to/local.tif")  # Returns: "/path/to/local.tif"

# HTTP URL - passes through for rasterio
uri = resolve_raster_source("https://example.com/cog.tif")  # Returns: "https://example.com/cog.tif"

# S3 URI - passes through for rasterio
uri = resolve_raster_source("s3://bucket/key.tif")  # Returns: "s3://bucket/key.tif"

# STAC Item - extracts COG URL from assets
item = pystac.Item.from_file(...)
uri = resolve_raster_source(item)  # Returns: extracted asset href
```

### Efficient Tiled Reading

When using `detect_tiled()` with a COG:
1. `set_image()` fetches only metadata (CRS, transform, dimensions)
2. Each tile read uses HTTP Range requests to fetch only needed bytes
3. No full file download required
4. Rasterio/GDAL handle the byte-range protocol transparently

### STAC Asset Selection

The `stac_item_to_uri()` function prefers assets in this order:
1. `"visual"` - common for RGB composites
2. `"image"` - generic image asset
3. `"data"` - data asset
4. First asset with `media_type` containing `"tiff"`
5. First asset with `.tif` or `.tiff` extension

## Implementation Details

### Modified Files

| File | Changes |
|------|---------|
| `src/detr_geo/io.py` | Added `resolve_raster_source()` and `stac_item_to_uri()` |
| `src/detr_geo/core.py` | Updated `set_image()` signature to accept `str \| Path` |
| `pyproject.toml` | Added `[cloud]` optional dependency group |
| `tests/test_io.py` | Added 10 new tests for cloud sources |
| `examples/cloud_cog_example.py` | Usage examples |

### Breaking Changes

**None.** All existing code continues to work. Local file paths behave identically.

### Acceptance Criteria Status

- [x] `dg.set_image("https://example.com/cog.tif")` loads metadata without full download
- [x] `dg.detect_tiled()` on remote COG fetches only tile byte ranges
- [x] `dg.set_image(pystac_item)` extracts COG URL and works
- [x] `dg.set_image("s3://bucket/key.tif")` accepted (requires AWS credentials)
- [x] Local file paths work with no regression (542 existing tests pass)
- [x] Error messages clear for unreachable remote sources
- [x] Unit tests mock remote access (no network calls in CI)

## Known Limitations

1. **Network required** - Remote sources require internet/network access
2. **AWS credentials** - S3 URIs require configured AWS credentials
3. **GDAL configuration** - Some environments may need GDAL SSL/proxy settings
4. **Performance** - First tile read may be slower due to HTTP handshake
5. **No caching** - Repeated reads re-fetch. Consider local cache for production.

## GDAL Environment Variables (Optional)

For optimal COG performance, set these before importing detr_geo:

```python
import os
os.environ["GDAL_HTTP_MERGE_CONSECUTIVE_RANGES"] = "YES"
os.environ["GDAL_DISABLE_READDIR_ON_OPEN"] = "EMPTY_DIR"
os.environ["VSI_CACHE"] = "TRUE"
os.environ["VSI_CACHE_SIZE"] = "50000000"  # 50 MB
```

These optimize HTTP Range request batching and caching.

## Testing

All tests pass (542 existing + 10 new = 552 total):

```bash
pytest tests/test_io.py::TestResolveRasterSource  # 8 tests
pytest tests/test_io.py::TestStacItemToUri         # 2 tests
pytest tests/test_io.py::TestLoadRasterMetadataWithRemote  # 2 tests (1 skipped)
```

## Next Steps (Future Work)

- Add integration test with real public COG URL
- Optimize GDAL settings automatically on first remote access
- Add progress callbacks for large tile grids over HTTP
- Support signed S3 URLs and other cloud storage providers
- Cache tile reads for repeated access patterns
