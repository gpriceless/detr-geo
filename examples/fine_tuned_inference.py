"""Fine-tuned inference: detect vehicles from satellite imagery.

Uses the xView fine-tuned model to distinguish 5 overhead vehicle classes
(Car, Pickup Truck, Truck, Bus, Other Vehicle) from satellite and aerial
imagery at 0.3m GSD.

The COCO-pretrained model was trained on ground-level photos and
misclassifies overhead vehicles as motorcycles, boats, or skateboards.
The xView fine-tuned weights solve this by training on labeled satellite
imagery from the xView dataset.

Prerequisites:
    pip install detr-geo[all]

    # Download xView fine-tuned weights:
    # huggingface-cli download gpriceless/detr-geo-xview \
    #     checkpoint_best_ema.pth --local-dir checkpoints/

Usage:
    python examples/fine_tuned_inference.py satellite.tif checkpoints/checkpoint_best_ema.pth
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

from detr_geo import DetrGeo

# xView vehicle class mapping -- must match the training configuration
XVIEW_CLASSES = {
    0: "Car",
    1: "Pickup Truck",
    2: "Truck",
    3: "Bus",
    4: "Other Vehicle",
}


def main(image_path: str, weights_path: str) -> None:
    # Load the xView fine-tuned model
    dg = DetrGeo(
        model_size="medium",
        pretrain_weights=weights_path,
        custom_class_names=XVIEW_CLASSES,
        confidence_threshold=0.3,
    )

    dg.set_image(image_path, suppress_gsd_warning=True)

    # Satellite scenes are almost always too large for a single pass.
    # Tiled detection handles any raster size.
    detections = dg.detect_tiled(
        overlap=0.2,
        nms_threshold=0.5,
        threshold=0.3,
    )

    # Summary
    print(f"Detected {len(detections)} vehicles in {Path(image_path).name}")

    if len(detections) > 0:
        # Per-class breakdown
        counts = Counter(detections["class_name"])
        print("\nVehicle counts:")
        for cls, count in counts.most_common():
            print(f"  {cls:<15s} {count:>5d}")

        # Confidence statistics
        scores = detections["confidence"]
        print(f"\nConfidence: min={scores.min():.2f}  mean={scores.mean():.2f}  max={scores.max():.2f}")

    # Export to GeoPackage (recommended) and GeoJSON (web-friendly)
    output_stem = Path(image_path).stem
    dg.to_gpkg(f"{output_stem}_vehicles.gpkg")
    dg.to_geojson(f"{output_stem}_vehicles.geojson")
    print(f"\nExported to {output_stem}_vehicles.gpkg and .geojson")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python fine_tuned_inference.py <image.tif> <weights.pth>")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
