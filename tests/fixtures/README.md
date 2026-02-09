# Test Fixtures

This directory contains small GeoTIFF fixtures for integration testing.

## Fixtures

| File | Size | Description |
|------|------|-------------|
| `rgb_uint8.tif` | ~43 KB | 128x128 RGB uint8, EPSG:4326 (geographic), ~0.3m GSD |
| `rgb_uint8_utm.tif` | ~43 KB | 128x128 RGB uint8, EPSG:32610 (UTM Zone 10N), 1m GSD |
| `single_band.tif` | ~15 KB | 128x128 single-band uint8 (grayscale), EPSG:4326 |
| `rgb_uint16.tif` | ~82 KB | 128x128 RGB uint16 (satellite-like), EPSG:4326 |
| `with_nodata.tif` | ~40 KB | 128x128 RGB uint8 with nodata regions, EPSG:4326 |

**Total:** ~223 KB (well within 500 KB budget)

## Fixture Properties

All fixtures are:
- 128x128 pixels (small enough to commit, large enough to test tiling)
- Compressed with DEFLATE + predictor=2
- Synthetic imagery resembling parking lots with vehicle-like features
- Georeferenced with real CRS and transforms

## Regenerating Fixtures

If fixtures need to be regenerated (e.g., to change resolution or content):

```bash
python scripts/generate_test_fixtures.py
```

This script creates deterministic fixtures using fixed random seeds, so repeated runs produce identical files.

## Provenance

Fixtures are **synthetic** (not real imagery). They are generated programmatically to avoid licensing and size constraints of real satellite/aerial imagery. The pixel values and spatial patterns are designed to exercise the full I/O pipeline (normalization, band selection, tiling, nodata handling) without requiring model inference.

## Usage in Tests

Fixtures are exposed via pytest fixtures in `conftest.py`:

```python
def test_something(real_geotiff_rgb_uint8):
    # real_geotiff_rgb_uint8 is a Path to rgb_uint8.tif
    ...
```

Tests using these fixtures should be marked with `@pytest.mark.integration`.
