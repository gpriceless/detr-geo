# Testing Guide

This document covers how to run the detr-geo test suite, what the tests cover, how to add new tests, and how to test with real geospatial data.

---

## Running the Test Suite

### Quick Run

```bash
# From the project root
.venv/bin/python -m pytest tests/ -v
```

### Expected Output

```
359 passed, 14 warnings
```

The 14 warnings are expected and intentional:
- CPU inference warnings (tests run without GPU)
- GeoJSON reprojection warnings (auto-reproject to WGS84)
- Image dimension alignment warnings (edge case tests)

### Running Specific Test Modules

```bash
# CRS tests only
.venv/bin/python -m pytest tests/test_crs.py -v

# Tiling tests only
.venv/bin/python -m pytest tests/test_tile_grid.py tests/test_nms.py -v

# Export tests only
.venv/bin/python -m pytest tests/test_export.py tests/test_geodataframe.py -v

# Visualization tests
.venv/bin/python -m pytest tests/test_viz_matplotlib.py tests/test_viz_leafmap.py -v
```

### Running by Test Class or Name

```bash
# Specific test class
.venv/bin/python -m pytest tests/test_crs.py::TestPixelToGeo -v

# Specific test
.venv/bin/python -m pytest tests/test_crs.py::TestPixelToGeo::test_rotated_affine_produces_non_axis_aligned_polygon -v

# Pattern matching
.venv/bin/python -m pytest tests/ -k "nms" -v
```

### No rfdetr Required

The entire test suite runs without RF-DETR or PyTorch installed. All model interactions are mocked. This is a deliberate design choice to keep the test suite fast and CI-friendly.

---

## What the Tests Cover

### Test Files and Their Scope

| File | Tests | Module | What It Verifies |
|------|-------|--------|------------------|
| `test_package.py` | 5 | `__init__` | Import works, version is semver, no star imports |
| `test_exceptions.py` | 12 | `exceptions` | Inheritance hierarchy, catch-all, message storage |
| `test_typing.py` | 10 | `_typing` | All type aliases importable, dict instantiation |
| `test_adapter.py` | 48 | `_adapter` | Variant mapping, normalization, validation, prediction |
| `test_device.py` | 9 | `_adapter` | Device detection with mocked torch |
| `test_input_validation.py` | 21 | `_adapter` | Channel count, dtype, value range, dimension checks |
| `test_batch_inference.py` | 7 | `_adapter` | Batch prediction, auto batch size, optimization |
| `test_weight_loading.py` | 10 | `_adapter` | Lazy loading, custom weights, missing rfdetr |
| `test_band_selector.py` | 12 | `io` | Presets, custom bands, triplication, alpha detection |
| `test_normalization.py` | 24 | `io` | Percentile, minmax, nodata mask, per-band stretch |
| `test_io.py` | 41 | `io` | Metadata loading, windowed reads, nodata fraction, fill |
| `test_crs.py` | 35 | `crs` | 4-corner transform, rotation detection, auto-UTM |
| `test_tile_grid.py` | 18 | `tiling` | Grid generation, coverage, overlap constraints |
| `test_nms.py` | 27 | `tiling` | IoU, NMS, edge zone filter, offset detections |
| `test_geodataframe.py` | 27 | `export` | GeoDataFrame construction, pixel DataFrame, areas |
| `test_export.py` | 47 | `export` | GeoJSON, GeoPackage, Shapefile round-trips |
| `test_viz_matplotlib.py` | 20 | `viz` | Class colors, filtering, drawing, file saving |
| `test_viz_leafmap.py` | 15 | `viz` | Hex conversion, styled GeoJSON, map creation |
| `test_core.py` | 15 | `core` | Instantiation, properties, method signatures |
| `test_core_integration.py` | 15 | `core` | End-to-end workflows with mocked modules |

### What Is Well-Covered

- **Exception hierarchy**: Every exception class tested for correct inheritance
- **Band selection**: All presets, custom indices, edge cases (0-index, out-of-range)
- **Normalization**: uint8, uint16, float32, float64 inputs; all stretch modes; nodata exclusion
- **CRS transforms**: Identity, north-up, rotated affines; UTM zones in both hemispheres
- **Tile grid**: Full coverage verification on regular and irregular raster sizes
- **NMS**: Class-aware suppression, confidence ordering, threshold sensitivity
- **Export**: Round-trip fidelity for all three formats; CRS preservation; empty GeoDataFrame handling
- **Visualization**: File generation, filtering, color consistency

### What Is NOT Covered

The test suite has intentional gaps that should be understood:

1. **No real RF-DETR inference**: All model predictions are mocked. There are no tests that actually run the neural network on image data. This means the full pipeline (raster -> tiles -> model -> detections -> export) has never been tested end-to-end in an automated way.

2. **No real geospatial data**: All test rasters are synthetic (random pixel values, simple transforms). No test uses actual satellite or aerial imagery with realistic spectral values, nodata patterns, or geospatial extents.

3. **No large raster tests**: The largest test raster is 1000x1000 pixels (in the tile coverage test). There are no tests for memory behavior with large rasters.

4. **No multi-CRS integration test**: The pipeline has not been tested with rasters in uncommon CRS (e.g., Lambert Conformal Conic, Albers Equal Area, local grids).

5. **No rotated affine full-pipeline test**: `pixel_to_geo` is tested with rotated affines, but the full set_image -> detect -> export pipeline is not.

6. **No performance benchmarks**: No tests measure execution time or memory usage.

---

## How to Add Tests

### Test File Organization

Tests are organized by module. To add tests for a function in `io.py`, add them to `test_io.py` or `test_band_selector.py` / `test_normalization.py` (which split io.py by concern).

### Test Structure Pattern

Follow the existing pattern using pytest classes:

```python
"""Tests for [feature description]."""

from __future__ import annotations

import numpy as np
import pytest

from detr_geo.module_name import function_to_test
from detr_geo.exceptions import ExpectedError


class TestFeatureName:
    """Test [what this group covers]."""

    def test_happy_path(self):
        """[What should happen in normal use]."""
        result = function_to_test(valid_input)
        assert result == expected_value

    def test_edge_case(self):
        """[What happens at boundaries]."""
        result = function_to_test(edge_input)
        assert result == expected_edge_value

    def test_error_case(self):
        """[What error is raised for invalid input]."""
        with pytest.raises(ExpectedError, match="descriptive message"):
            function_to_test(invalid_input)
```

### Creating Test Rasters

Use the helper in `test_io.py`:

```python
from tests.test_io import create_test_geotiff

def test_my_feature(tmp_dir):
    path = create_test_geotiff(
        str(Path(tmp_dir) / "test.tif"),
        width=256,
        height=256,
        bands=3,
        dtype="uint16",       # Test with 16-bit data
        crs="EPSG:32618",     # UTM zone 18N
        nodata=0.0,           # Nodata value
    )
    # Use path in your test
```

### Mocking RF-DETR

For tests that involve the model, use the established mocking pattern:

```python
from unittest.mock import MagicMock, patch

def test_with_mocked_model():
    adapter = RFDETRAdapter("nano", device="cpu")

    # Create mock model
    mock_model = MagicMock()
    mock_det = MagicMock()
    mock_det.xyxy = np.array([[10, 20, 30, 40]])
    mock_det.confidence = np.array([0.9])
    mock_det.class_id = np.array([0])
    mock_det.__len__ = lambda self: 1
    mock_model.predict.return_value = mock_det

    mock_rfdetr = MagicMock()
    mock_rfdetr.RFDETRNano.return_value = mock_model

    with patch.dict("sys.modules", {"rfdetr": mock_rfdetr}):
        result = adapter.predict_tile(Image.new("RGB", (384, 384)))

    assert result["bbox"] == [[10.0, 20.0, 30.0, 40.0]]
```

### Using Temporary Directories

For tests that write files, use pytest's `tmp_path` or a custom fixture:

```python
def test_export(tmp_path):
    output = tmp_path / "results.gpkg"
    export_gpkg(gdf, str(output))
    assert output.exists()
```

---

## Testing with Real Geospatial Data

The current test suite uses only synthetic data. For validating behavior with real imagery, follow these procedures.

### Setting Up a Test Data Directory

Create a `tests/data/` directory (gitignored) with real raster files:

```
tests/
  data/
    naip_sample.tif        # Small NAIP cutout (e.g., 2000x2000)
    sentinel2_sample.tif   # Sentinel-2 L2A scene subset
    rotated_aerial.tif     # Aerial photo with non-north-up orientation
    nodata_edges.tif       # Raster with nodata at boundaries
    no_crs.tif             # Raster without embedded CRS
```

### Manual Integration Test Procedure

These tests require RF-DETR installed and optionally a GPU.

```python
"""Manual integration test -- run with real model."""

from detr_geo import DetrGeo
from detr_geo.io import load_raster_metadata

# 1. Verify metadata reading
meta = load_raster_metadata("tests/data/naip_sample.tif")
print(f"CRS: {meta.crs}")
print(f"Size: {meta.width}x{meta.height}")
print(f"Bands: {meta.count}, dtype: {meta.dtype}")
print(f"Nodata: {meta.nodata}")

# 2. Single-image detection
dg = DetrGeo(model_size="nano", device="cpu")
dg.set_image("tests/data/naip_sample.tif")
detections = dg.detect(threshold=0.3)
print(f"Detections: {len(detections)}")
print(detections[["class_name", "confidence"]].head(10))

# 3. Tiled detection
dg2 = DetrGeo(model_size="nano")
dg2.set_image("tests/data/naip_sample.tif")
tiled = dg2.detect_tiled(overlap=0.2, threshold=0.3)
print(f"Tiled detections: {len(tiled)}")

# 4. Export round-trip
dg2.to_gpkg("/tmp/test_output.gpkg")
import geopandas as gpd
reimported = gpd.read_file("/tmp/test_output.gpkg")
print(f"Reimported: {len(reimported)} features, CRS: {reimported.crs}")

# 5. Visualization
dg.show_detections(save_path="/tmp/test_detections.png")
print("Saved detection plot to /tmp/test_detections.png")
```

### What to Validate

When running manual tests, check:

1. **Detection count is reasonable**: Are you getting 0 detections (model not working) or millions (something wrong)?
2. **Spatial alignment**: Open the exported GeoPackage in QGIS alongside the source raster. Do detections overlay correctly?
3. **CRS correctness**: Is the output CRS the same as the input CRS?
4. **Tile boundary artifacts**: For tiled detection, are there duplicate detections at tile boundaries (NMS not working) or missing detections (overlap too small)?
5. **16-bit stretch quality**: For satellite imagery, does the normalization produce reasonable-looking RGB output?
6. **Memory usage**: Monitor memory during tiled processing. Does it stay constant or grow?

### Adding Automated Integration Tests

If you have test data files available, add skip-if-missing tests:

```python
import os
import pytest

NAIP_PATH = "tests/data/naip_sample.tif"

@pytest.mark.skipif(
    not os.path.exists(NAIP_PATH),
    reason="Test data not available"
)
def test_real_naip_detection():
    """Integration test with real NAIP imagery."""
    dg = DetrGeo(model_size="nano", device="cpu")
    dg.set_image(NAIP_PATH)
    detections = dg.detect(threshold=0.3)
    assert len(detections) >= 0
    assert detections.crs is not None
```

Mark GPU-required tests:

```python
@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA not available"
)
def test_gpu_inference():
    ...
```

---

## Test Configuration

### pyproject.toml Settings

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
```

### Adding Custom Markers

To categorize tests (e.g., for CI vs. local runs):

```python
# conftest.py
import pytest

def pytest_configure(config):
    config.addinivalue_line("markers", "integration: requires real test data")
    config.addinivalue_line("markers", "gpu: requires CUDA GPU")
    config.addinivalue_line("markers", "slow: takes more than 10 seconds")
```

Then run subsets:

```bash
# Skip slow/integration tests in CI
pytest tests/ -m "not integration and not gpu and not slow"

# Run only integration tests locally
pytest tests/ -m "integration"
```
