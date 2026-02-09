# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-02-08

### Added
- Core geospatial detection pipeline: GeoTIFF input, georeferenced output
- RF-DETR model adapter with five model sizes (nano, small, medium, base, large)
- `DetrGeo` class with `set_image()`, `detect()`, and `detect_tiled()` methods
- Tiled inference engine with configurable tile size and overlap
- Cross-tile NMS (two-stage: per-tile class-agnostic + cross-tile class-aware)
- Band selection with sensor presets (RGB, NAIP, Sentinel-2, WorldView, custom)
- 16-bit satellite imagery normalization via percentile stretching
- Nodata-aware processing (skip empty tiles or fill with value)
- CRS handling: `pixel_to_geo()`, `geo_to_pixel()`, auto-UTM, validation
- Export formats: GeoJSON (auto-reproject to WGS84), GeoPackage, Shapefile
- Interactive leafmap visualization with basemap layers
- Static matplotlib visualization with georeferenced context
- Training pipeline with `prepare_training_dataset()` and `train()` wrapper
- Spatial-aware dataset splitting via `SpatialSplitter` (block, grid, random, sequential)
- COCO and YOLO annotation export from detection results
- Augmentation presets for satellite, aerial, and drone imagery
- Fine-tuning support via `pretrain_weights` and `custom_class_names` parameters
- VME fine-tuning scripts and reproduction guide (3-class: Car, Bus, Truck)
- xView fine-tuning scripts (5-class: Car, Pickup, Truck, Bus, Other vehicle)
- xView fine-tuned weights (best EMA checkpoint, epoch 13/50)
- Three-model comparison tests (COCO baseline vs VME vs xView)
- GSD sensitivity analysis and stress tests for tiling boundaries
- RunPod training infrastructure (setup scripts, monitoring, checkpoint sync)
- Comprehensive test suite (500+ tests across 31 test files)
- Exception hierarchy: `DetrGeoError` and 5 domain-specific subtypes
- Optional dependency groups: core, [rfdetr], [viz], [all]
- CI/CD with GitHub Actions: lint, test matrix (Python 3.10-3.12), build verification
- Ruff linting and formatting configuration

### Fixed
- Single-band grayscale GeoTIFF crash in tiled pipeline (DETRGEO-83)
- `show_detections()` closing figure immediately due to missing plt.show() (DETRGEO-53)
- Deprecated `predict_batch()` warnings from rfdetr (DETRGEO-52)
- TD-006: Tests assuming rfdetr absent now skip correctly when rfdetr IS installed

[0.1.0]: https://github.com/gpriceless/detr-geo/releases/tag/v0.1.0
