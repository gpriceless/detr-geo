"""Tiled detection for large rasters with overlap and cross-tile NMS.

Satellite orthomosaics and aerial surveys are typically thousands of pixels
on each side. Loading the full image into memory for detection would require
tens of gigabytes. Tiled detection solves this by:

  1. Splitting the raster into overlapping tiles (e.g. 576x576 pixels)
  2. Running detection on each tile independently
  3. Offsetting bounding boxes back to global pixel coordinates
  4. Running cross-tile NMS to remove duplicates from overlapping regions

The overlap parameter controls how much adjacent tiles share. Without
overlap, objects at tile boundaries are split across two tiles and may
fall below the confidence threshold in both. A 20% overlap ensures that
any object fully visible in the overlap zone appears complete in at least
one tile.

Cross-tile NMS is class-aware: a car and a building at the same location
both survive, but two "car" detections with high IoU are deduplicated.

Prerequisites:
    pip install detr-geo[rfdetr]

Usage:
    python examples/tiled_detection.py large_orthomosaic.tif
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

from detr_geo import DetrGeo
from detr_geo.io import load_raster_metadata
from detr_geo.tiling import detection_range, recommended_overlap


def main(image_path: str) -> None:
    # Check raster metadata before loading
    meta = load_raster_metadata(image_path)
    print(f"Raster: {Path(image_path).name}")
    print(f"  Size:   {meta.width} x {meta.height} pixels")
    print(f"  Bands:  {meta.count}")
    print(f"  CRS:    {meta.crs}")
    print(f"  GSD:    {meta.gsd:.3f} m/px" if meta.gsd else "  GSD:    unknown")

    # Initialize model
    dg = DetrGeo(model_size="medium", confidence_threshold=0.3)
    dg.set_image(image_path, suppress_gsd_warning=True)

    # Calculate detection size range at this GSD
    if meta.gsd:
        min_m, max_m = detection_range(
            tile_size=dg.resolution,
            gsd=meta.gsd,
            overlap=0.2,
        )
        print(f"\n  Detectable object size: {min_m:.1f}m to {max_m:.1f}m")

    # For objects roughly 5m across (vehicles), what overlap do we need?
    if meta.gsd:
        object_px = 5.0 / meta.gsd  # 5-meter object in pixels
        suggested_overlap = recommended_overlap(
            object_size_pixels=object_px,
            tile_size=dg.resolution,
        )
        print(f"  Suggested overlap for 5m objects: {suggested_overlap:.2f}")

    # Run tiled detection
    # tile_size=None uses the model's native resolution (576 for medium)
    # overlap=0.2 means each tile shares 20% of its edge with neighbors
    # nms_threshold=0.5 removes same-class detections with IoU > 0.5
    # nodata_threshold=0.5 skips tiles that are >50% nodata
    print(f"\nRunning tiled detection...")
    t0 = time.time()

    detections = dg.detect_tiled(
        tile_size=None,           # model's native resolution
        overlap=0.2,              # 20% overlap between tiles
        nms_threshold=0.5,        # IoU threshold for deduplication
        nodata_threshold=0.5,     # skip mostly-empty tiles
        threshold=0.3,            # confidence threshold
    )

    elapsed = time.time() - t0
    print(f"  Detected {len(detections)} objects in {elapsed:.1f}s")

    if len(detections) > 0:
        # Per-class summary
        class_counts = detections["class_name"].value_counts()
        print(f"\n  Class counts:")
        for cls, count in class_counts.items():
            print(f"    {cls:20s}  {count}")

        # Confidence distribution
        scores = detections["confidence"]
        print(f"\n  Confidence: min={scores.min():.2f}  "
              f"median={scores.median():.2f}  max={scores.max():.2f}")

    # Export results
    output_stem = Path(image_path).stem
    gpkg_path = f"{output_stem}_detections.gpkg"
    dg.to_gpkg(gpkg_path)
    print(f"\nExported to {gpkg_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python tiled_detection.py <large_raster.tif>")
        sys.exit(1)
    main(sys.argv[1])
