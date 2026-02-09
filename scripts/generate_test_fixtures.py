#!/usr/bin/env python3
"""Generate small GeoTIFF fixtures for integration testing.

Creates realistic synthetic test fixtures with various properties:
- RGB uint8 in EPSG:4326 (geographic CRS)
- RGB uint8 in EPSG:32610 (UTM projected CRS)
- Single-band grayscale
- RGB uint16 (satellite-like)
- With nodata regions

All fixtures are 128x128 pixels and compressed to stay under 100KB each.
"""

from pathlib import Path

import numpy as np
import rasterio
from pyproj import CRS
from rasterio.enums import ColorInterp
from rasterio.transform import from_bounds

# Output directory
FIXTURES_DIR = Path(__file__).parent.parent / "tests" / "fixtures"


def generate_rgb_uint8_geographic():
    """Create a 128x128 RGB uint8 GeoTIFF in EPSG:4326."""
    output_path = FIXTURES_DIR / "rgb_uint8.tif"

    # Define a small geographic extent (~0.3m GSD at mid-latitudes)
    # Center: approximately San Jose, CA area
    width, height = 128, 128
    west, south = -121.9, 37.33
    # ~0.0001 degrees ≈ 11m at this latitude, so ~0.00001 degrees/pixel ≈ 1m/pixel
    east = west + (width * 0.00001)
    north = south + (height * 0.00001)

    transform = from_bounds(west, south, east, north, width, height)

    # Generate synthetic imagery resembling a parking lot
    # Band 1 (Red): ~100-180 (pavement)
    # Band 2 (Green): ~100-180 (pavement)
    # Band 3 (Blue): ~90-150 (pavement)
    rng = np.random.RandomState(42)

    # Base pavement texture
    red = rng.randint(100, 180, (height, width), dtype=np.uint8)
    green = rng.randint(100, 180, (height, width), dtype=np.uint8)
    blue = rng.randint(90, 150, (height, width), dtype=np.uint8)

    # Add some "vehicles" (darker rectangles)
    for _ in range(5):
        x = rng.randint(10, width - 20)
        y = rng.randint(10, height - 20)
        w, h = rng.randint(8, 15), rng.randint(5, 10)
        # Vehicles are darker
        red[y:y+h, x:x+w] = rng.randint(40, 80, (h, w))
        green[y:y+h, x:x+w] = rng.randint(40, 80, (h, w))
        blue[y:y+h, x:x+w] = rng.randint(40, 80, (h, w))

    data = np.stack([red, green, blue])

    with rasterio.open(
        output_path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=3,
        dtype="uint8",
        crs=CRS.from_epsg(4326),
        transform=transform,
        compress="deflate",
        predictor=2,
        tiled=True,
        blockxsize=128,
        blockysize=128,
    ) as dst:
        dst.write(data)
        dst.colorinterp = [ColorInterp.red, ColorInterp.green, ColorInterp.blue]

    print(f"✓ Created {output_path} ({output_path.stat().st_size / 1024:.1f} KB)")


def generate_rgb_uint8_utm():
    """Create a 128x128 RGB uint8 GeoTIFF in EPSG:32610 (UTM Zone 10N)."""
    output_path = FIXTURES_DIR / "rgb_uint8_utm.tif"

    # UTM Zone 10N coordinates (San Jose area)
    # 1m GSD
    width, height = 128, 128
    easting, northing = 586000, 4135000

    transform = from_bounds(
        easting, northing, easting + width, northing + height, width, height
    )

    # Similar synthetic imagery
    rng = np.random.RandomState(43)

    red = rng.randint(100, 180, (height, width), dtype=np.uint8)
    green = rng.randint(100, 180, (height, width), dtype=np.uint8)
    blue = rng.randint(90, 150, (height, width), dtype=np.uint8)

    # Add vehicles
    for _ in range(5):
        x = rng.randint(10, width - 20)
        y = rng.randint(10, height - 20)
        w, h = rng.randint(8, 15), rng.randint(5, 10)
        red[y:y+h, x:x+w] = rng.randint(40, 80, (h, w))
        green[y:y+h, x:x+w] = rng.randint(40, 80, (h, w))
        blue[y:y+h, x:x+w] = rng.randint(40, 80, (h, w))

    data = np.stack([red, green, blue])

    with rasterio.open(
        output_path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=3,
        dtype="uint8",
        crs=CRS.from_epsg(32610),
        transform=transform,
        compress="deflate",
        predictor=2,
        tiled=True,
        blockxsize=128,
        blockysize=128,
    ) as dst:
        dst.write(data)
        dst.colorinterp = [ColorInterp.red, ColorInterp.green, ColorInterp.blue]

    print(f"✓ Created {output_path} ({output_path.stat().st_size / 1024:.1f} KB)")


def generate_single_band():
    """Create a 128x128 single-band uint8 GeoTIFF (grayscale)."""
    output_path = FIXTURES_DIR / "single_band.tif"

    width, height = 128, 128
    west, south = -121.9, 37.33
    east = west + (width * 0.00001)
    north = south + (height * 0.00001)

    transform = from_bounds(west, south, east, north, width, height)

    # Grayscale synthetic data
    rng = np.random.RandomState(44)
    gray = rng.randint(100, 180, (height, width), dtype=np.uint8)

    # Add darker rectangles
    for _ in range(5):
        x = rng.randint(10, width - 20)
        y = rng.randint(10, height - 20)
        w, h = rng.randint(8, 15), rng.randint(5, 10)
        gray[y:y+h, x:x+w] = rng.randint(40, 80, (h, w))

    with rasterio.open(
        output_path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=1,
        dtype="uint8",
        crs=CRS.from_epsg(4326),
        transform=transform,
        compress="deflate",
        predictor=2,
    ) as dst:
        dst.write(gray, 1)
        dst.colorinterp = [ColorInterp.gray]

    print(f"✓ Created {output_path} ({output_path.stat().st_size / 1024:.1f} KB)")


def generate_uint16():
    """Create a 128x128 RGB uint16 GeoTIFF (satellite-like)."""
    output_path = FIXTURES_DIR / "rgb_uint16.tif"

    width, height = 128, 128
    west, south = -121.9, 37.33
    east = west + (width * 0.00001)
    north = south + (height * 0.00001)

    transform = from_bounds(west, south, east, north, width, height)

    # uint16 imagery typically uses 0-10000 range for reflectance
    rng = np.random.RandomState(45)

    red = rng.randint(2000, 4000, (height, width), dtype=np.uint16)
    green = rng.randint(2000, 4000, (height, width), dtype=np.uint16)
    blue = rng.randint(1800, 3500, (height, width), dtype=np.uint16)

    # Add darker features
    for _ in range(5):
        x = rng.randint(10, width - 20)
        y = rng.randint(10, height - 20)
        w, h = rng.randint(8, 15), rng.randint(5, 10)
        red[y:y+h, x:x+w] = rng.randint(800, 1500, (h, w))
        green[y:y+h, x:x+w] = rng.randint(800, 1500, (h, w))
        blue[y:y+h, x:x+w] = rng.randint(800, 1500, (h, w))

    data = np.stack([red, green, blue])

    with rasterio.open(
        output_path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=3,
        dtype="uint16",
        crs=CRS.from_epsg(4326),
        transform=transform,
        compress="deflate",
        predictor=2,
        tiled=True,
        blockxsize=128,
        blockysize=128,
    ) as dst:
        dst.write(data)
        dst.colorinterp = [ColorInterp.red, ColorInterp.green, ColorInterp.blue]

    print(f"✓ Created {output_path} ({output_path.stat().st_size / 1024:.1f} KB)")


def generate_with_nodata():
    """Create a 128x128 RGB uint8 GeoTIFF with nodata regions."""
    output_path = FIXTURES_DIR / "with_nodata.tif"

    width, height = 128, 128
    west, south = -121.9, 37.33
    east = west + (width * 0.00001)
    north = south + (height * 0.00001)

    transform = from_bounds(west, south, east, north, width, height)

    rng = np.random.RandomState(46)

    red = rng.randint(100, 180, (height, width), dtype=np.uint8)
    green = rng.randint(100, 180, (height, width), dtype=np.uint8)
    blue = rng.randint(90, 150, (height, width), dtype=np.uint8)

    # Create a nodata region (top-right corner and some scattered areas)
    # Use 0 as nodata value
    nodata_value = 0

    # Top-right corner nodata
    red[0:32, 96:128] = nodata_value
    green[0:32, 96:128] = nodata_value
    blue[0:32, 96:128] = nodata_value

    # Scattered nodata patches
    for _ in range(3):
        x = rng.randint(10, 80)
        y = rng.randint(40, 110)
        w, h = rng.randint(5, 15), rng.randint(5, 15)
        red[y:y+h, x:x+w] = nodata_value
        green[y:y+h, x:x+w] = nodata_value
        blue[y:y+h, x:x+w] = nodata_value

    data = np.stack([red, green, blue])

    with rasterio.open(
        output_path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=3,
        dtype="uint8",
        crs=CRS.from_epsg(4326),
        transform=transform,
        compress="deflate",
        predictor=2,
        nodata=nodata_value,
        tiled=True,
        blockxsize=128,
        blockysize=128,
    ) as dst:
        dst.write(data)
        dst.colorinterp = [ColorInterp.red, ColorInterp.green, ColorInterp.blue]

    print(f"✓ Created {output_path} ({output_path.stat().st_size / 1024:.1f} KB)")


def generate_geographic_crs():
    """Create a 128x128 RGB uint8 GeoTIFF in EPSG:4326 (already covered by rgb_uint8).

    This is an alias for clarity in testing.
    """
    # The rgb_uint8.tif already uses EPSG:4326, so this is redundant
    # But we keep this function for documentation purposes
    pass


if __name__ == "__main__":
    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)

    print("Generating test fixtures...")
    generate_rgb_uint8_geographic()
    generate_rgb_uint8_utm()
    generate_single_band()
    generate_uint16()
    generate_with_nodata()

    print(f"\nAll fixtures created in {FIXTURES_DIR}")

    # Report total size
    total_size = sum(f.stat().st_size for f in FIXTURES_DIR.glob("*.tif"))
    print(f"Total size: {total_size / 1024:.1f} KB")

    if total_size > 500 * 1024:
        print("WARNING: Total fixture size exceeds 500 KB budget!")
    else:
        print("✓ Within 500 KB size budget")
