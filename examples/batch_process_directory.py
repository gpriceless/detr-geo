"""Batch process: run detection on all GeoTIFFs in a directory.

Scans a directory for GeoTIFF files, runs tiled detection on each, and
writes output Shapefiles (or GeoPackage/GeoJSON) alongside each input.
Prints a summary table at the end.

Supports both COCO-pretrained and xView fine-tuned models. For overhead
vehicle detection, pass --weights and the xView checkpoint.

Prerequisites:
    pip install detr-geo[rfdetr]

Usage:
    # COCO-pretrained (generic objects)
    python examples/batch_process_directory.py /path/to/geotiffs/

    # xView fine-tuned (overhead vehicles)
    python examples/batch_process_directory.py /path/to/geotiffs/ \
        --weights checkpoints/checkpoint_best_ema.pth \
        --format gpkg

    # Custom threshold and output format
    python examples/batch_process_directory.py /path/to/geotiffs/ \
        --threshold 0.4 --format geojson --output /path/to/results/
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from detr_geo import DetrGeo

# xView vehicle class mapping
XVIEW_CLASSES = {
    0: "Car",
    1: "Pickup Truck",
    2: "Truck",
    3: "Bus",
    4: "Other Vehicle",
}

# Common GeoTIFF extensions
TIFF_EXTENSIONS = {".tif", ".tiff", ".geotiff"}


def find_geotiffs(directory: str) -> list[Path]:
    """Find all GeoTIFF files in a directory (non-recursive)."""
    dir_path = Path(directory)
    files = []
    for ext in TIFF_EXTENSIONS:
        files.extend(dir_path.glob(f"*{ext}"))
        files.extend(dir_path.glob(f"*{ext.upper()}"))
    return sorted(set(files))


def process_one(
    dg: DetrGeo,
    image_path: Path,
    output_dir: Path,
    output_format: str,
    threshold: float,
) -> dict:
    """Process a single GeoTIFF and return results summary."""
    dg.set_image(str(image_path), suppress_gsd_warning=True)

    t0 = time.time()
    detections = dg.detect_tiled(
        overlap=0.2,
        nms_threshold=0.5,
        threshold=threshold,
    )
    elapsed = time.time() - t0

    # Export
    output_stem = output_dir / image_path.stem
    if output_format == "gpkg":
        output_path = f"{output_stem}_detections.gpkg"
        dg.to_gpkg(output_path)
    elif output_format == "geojson":
        output_path = f"{output_stem}_detections.geojson"
        dg.to_geojson(output_path)
    elif output_format == "shp":
        output_path = f"{output_stem}_detections.shp"
        dg.to_shp(output_path)
    else:
        output_path = f"{output_stem}_detections.gpkg"
        dg.to_gpkg(output_path)

    return {
        "file": image_path.name,
        "detections": len(detections),
        "elapsed": elapsed,
        "output": output_path,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run object detection on all GeoTIFFs in a directory.")
    parser.add_argument("directory", help="Directory containing GeoTIFF files")
    parser.add_argument(
        "--weights",
        default=None,
        help="Path to fine-tuned weights (.pth). Omit for COCO-pretrained.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output directory. Default: same as input directory.",
    )
    parser.add_argument(
        "--format",
        choices=["gpkg", "geojson", "shp"],
        default="gpkg",
        help="Output vector format (default: gpkg)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.3,
        help="Confidence threshold (default: 0.3)",
    )
    parser.add_argument(
        "--model-size",
        default="medium",
        help="Model size: nano, small, medium, base, large (default: medium)",
    )
    args = parser.parse_args()

    # Find GeoTIFFs
    geotiffs = find_geotiffs(args.directory)
    if not geotiffs:
        print(f"No GeoTIFF files found in {args.directory}")
        return

    print(f"Found {len(geotiffs)} GeoTIFF files in {args.directory}\n")

    # Output directory
    output_dir = Path(args.output) if args.output else Path(args.directory)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize model once (weights are loaded on first prediction)
    if args.weights:
        dg = DetrGeo(
            model_size=args.model_size,
            pretrain_weights=args.weights,
            custom_class_names=XVIEW_CLASSES,
            confidence_threshold=args.threshold,
        )
        print(f"Model: {args.model_size} with xView fine-tuned weights\n")
    else:
        dg = DetrGeo(
            model_size=args.model_size,
            confidence_threshold=args.threshold,
        )
        print(f"Model: {args.model_size} with COCO-pretrained weights\n")

    # Process each file
    results = []
    for i, geotiff in enumerate(geotiffs, 1):
        print(f"[{i}/{len(geotiffs)}] Processing {geotiff.name}...", end=" ")
        try:
            result = process_one(dg, geotiff, output_dir, args.format, args.threshold)
            results.append(result)
            print(f"{result['detections']} detections in {result['elapsed']:.1f}s")
        except Exception as e:
            print(f"FAILED: {e}")
            results.append(
                {
                    "file": geotiff.name,
                    "detections": -1,
                    "elapsed": 0,
                    "output": "ERROR",
                }
            )

    # Summary table
    print("\n" + "=" * 70)
    print(f"{'File':<35s} {'Detections':>10s} {'Time (s)':>10s}")
    print("-" * 70)

    total_detections = 0
    total_time = 0.0
    for r in results:
        det_str = str(r["detections"]) if r["detections"] >= 0 else "ERROR"
        print(f"{r['file']:<35s} {det_str:>10s} {r['elapsed']:>10.1f}")
        if r["detections"] >= 0:
            total_detections += r["detections"]
            total_time += r["elapsed"]

    print("-" * 70)
    successful = sum(1 for r in results if r["detections"] >= 0)
    print(f"{'TOTAL':<35s} {total_detections:>10d} {total_time:>10.1f}")
    print(f"\n{successful}/{len(geotiffs)} files processed successfully")
    print(f"Output format: {args.format}")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()
