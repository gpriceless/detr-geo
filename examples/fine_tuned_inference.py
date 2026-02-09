"""Fine-tuned inference: detect vehicles in overhead imagery with custom weights.

Shows how to load an xView fine-tuned model that distinguishes between
Car, Pickup, Truck, Bus, and Other vehicle classes from satellite and
aerial imagery at 0.3m GSD.

The COCO-pretrained model was trained on ground-level photos and does not
recognize vehicles from an overhead perspective. Fine-tuned weights solve
this by training on labeled satellite imagery (xView dataset).

Prerequisites:
    pip install detr-geo[all]

    # Download xView fine-tuned weights from HuggingFace:
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


def main(image_path: str, weights_path: str) -> None:
    # xView fine-tuned model: 5 overhead vehicle classes
    dg = DetrGeo(
        model_size="medium",
        pretrain_weights=weights_path,
        custom_class_names={
            0: "Car",
            1: "Pickup",
            2: "Truck",
            3: "Bus",
            4: "Other",
        },
        confidence_threshold=0.3,
    )

    dg.set_image(image_path, suppress_gsd_warning=True)

    # Use tiled detection -- satellite scenes are almost always too large
    # for a single pass through the model
    detections = dg.detect_tiled(
        overlap=0.2,
        nms_threshold=0.5,
        threshold=0.3,
    )

    # Summarize results
    print(f"Detected {len(detections)} vehicles in {Path(image_path).name}")

    if len(detections) > 0:
        counts = Counter(detections["class_name"])
        print("\nClass breakdown:")
        for cls, count in counts.most_common():
            print(f"  {cls:10s}  {count:4d}")

        scores = detections["confidence"]
        print(f"\nConfidence: min={scores.min():.2f}  "
              f"mean={scores.mean():.2f}  max={scores.max():.2f}")

    # Export
    output_stem = Path(image_path).stem
    dg.to_gpkg(f"{output_stem}_vehicles.gpkg")
    dg.to_geojson(f"{output_stem}_vehicles.geojson")
    print(f"\nExported to {output_stem}_vehicles.gpkg and .geojson")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python fine_tuned_inference.py <image.tif> <weights.pth>")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
