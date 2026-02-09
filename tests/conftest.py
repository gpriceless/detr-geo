"""Shared test fixtures for detr_geo test suite."""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
import rasterio
from pyproj import CRS
from rasterio.transform import from_bounds
from shapely.geometry import MultiPolygon, box

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FIXTURES_DIR = Path(__file__).parent / "fixtures"

# ---------------------------------------------------------------------------
# Test Helper Functions
# ---------------------------------------------------------------------------


def rfdetr_installed() -> bool:
    """Check if rfdetr package is available.

    Used to skip tests that assume rfdetr is NOT installed when it actually is.
    """
    try:
        import rfdetr  # noqa: F401

        return True
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# Training Pipeline Fixtures (Proposal 006)
# ---------------------------------------------------------------------------


@pytest.fixture
def synthetic_geotiff(tmp_path: Path) -> Path:
    """Create a synthetic 256x256 3-band uint8 GeoTIFF in EPSG:32617.

    The raster has ~1.0 m/pixel GSD, centered near (500000, 4000000) in
    UTM Zone 17N. Contains random pixel values for realistic testing.
    """
    tiff_path = tmp_path / "synthetic.tif"
    width, height = 256, 256
    # ~1m GSD in UTM: 256m x 256m extent
    west, south = 500000.0, 4000000.0
    east, north = west + width, south + height
    transform = from_bounds(west, south, east, north, width, height)

    rng = np.random.RandomState(42)
    data = rng.randint(0, 255, (3, height, width), dtype=np.uint8)

    with rasterio.open(
        tiff_path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=3,
        dtype="uint8",
        crs=CRS.from_epsg(32617),
        transform=transform,
    ) as dst:
        dst.write(data)

    return tiff_path


@pytest.fixture
def synthetic_geojson_utm(tmp_path: Path) -> Path:
    """Create a synthetic GeoJSON with 5 polygons in EPSG:32617 (same CRS as raster).

    Includes test cases:
    - Polygon fully inside a tile
    - Polygon straddling tile boundaries
    - MultiPolygon geometry
    - Tiny polygon for min_area filter testing
    - Large polygon spanning multiple tiles
    """
    features = []

    # 1. Polygon fully inside first tile (within 0-100 pixel region)
    features.append(
        {
            "geometry": box(500020.0, 4000020.0, 500060.0, 4000060.0),
            "class_name": "building",
        }
    )

    # 2. Polygon straddling tile boundary (around pixel col 128)
    features.append(
        {
            "geometry": box(500110.0, 4000050.0, 500150.0, 4000090.0),
            "class_name": "building",
        }
    )

    # 3. MultiPolygon - two separate small polygons
    mp = MultiPolygon(
        [
            box(500170.0, 4000170.0, 500190.0, 4000190.0),
            box(500200.0, 4000200.0, 500220.0, 4000220.0),
        ]
    )
    features.append(
        {
            "geometry": mp,
            "class_name": "vehicle",
        }
    )

    # 4. Tiny polygon (should be filtered by min_area)
    features.append(
        {
            "geometry": box(500010.0, 4000010.0, 500013.0, 4000013.0),  # 3x3 = 9 sq px
            "class_name": "debris",
        }
    )

    # 5. Large polygon spanning most of the raster
    features.append(
        {
            "geometry": box(500050.0, 4000050.0, 500200.0, 4000200.0),
            "class_name": "field",
        }
    )

    gdf = gpd.GeoDataFrame(features, crs=CRS.from_epsg(32617))
    geojson_path = tmp_path / "annotations_utm.geojson"
    gdf.to_file(geojson_path, driver="GeoJSON")
    return geojson_path


@pytest.fixture
def synthetic_geojson_4326(tmp_path: Path) -> Path:
    """Create a synthetic GeoJSON with polygons in EPSG:4326 (different CRS from raster).

    These polygons correspond to the same geographic area as the UTM fixtures
    but expressed in lon/lat coordinates. Used to test CRS reprojection.
    """
    # Approximate lon/lat for UTM 17N (500000, 4000000)
    # Using rough conversion: the exact values depend on the projection
    # For testing, we use approximate center coords
    # UTM 17N (500000, 4000000) ~ (-81.0, 36.1) approximately
    features = [
        {
            "geometry": box(-81.0005, 36.1002, -81.0001, 36.1005),
            "class_name": "building",
        },
        {
            "geometry": box(-81.0003, 36.1001, -80.9998, 36.1004),
            "class_name": "vehicle",
        },
    ]
    gdf = gpd.GeoDataFrame(features, crs=CRS.from_epsg(4326))
    geojson_path = tmp_path / "annotations_4326.geojson"
    gdf.to_file(geojson_path, driver="GeoJSON")
    return geojson_path


@pytest.fixture
def synthetic_detection_gdf() -> gpd.GeoDataFrame:
    """Create a synthetic detection GeoDataFrame for export testing.

    Matches the pattern used by detr_geo.export: geometry column with
    Polygon bounding boxes, class_id, class_name, and confidence columns.
    All geometries are in pixel coordinates (not geographic).
    """
    return gpd.GeoDataFrame(
        {
            "geometry": [
                box(10, 10, 50, 50),
                box(60, 60, 120, 100),
                box(150, 30, 190, 70),
            ],
            "class_id": [0, 1, 0],
            "class_name": ["building", "vehicle", "building"],
            "confidence": [0.95, 0.80, 0.72],
        }
    )


@pytest.fixture
def empty_detection_gdf() -> gpd.GeoDataFrame:
    """Create an empty detection GeoDataFrame for edge case testing."""
    return gpd.GeoDataFrame(
        {
            "geometry": [],
            "class_id": [],
            "class_name": [],
            "confidence": [],
        }
    )


@pytest.fixture
def tile_list_10x10() -> list[dict]:
    """Generate a 10x10 grid of tiles for splitter testing.

    Each tile is 100x100 pixels, covering a 1000x1000 raster.
    """
    tiles = []
    for row in range(10):
        for col in range(10):
            tiles.append(
                {
                    "window": (col * 100, row * 100, 100, 100),
                    "global_offset_x": col * 100,
                    "global_offset_y": row * 100,
                    "nodata_fraction": 0.0,
                }
            )
    return tiles


@pytest.fixture
def tile_list_2x2() -> list[dict]:
    """Generate a 2x2 grid of tiles for small raster edge case testing."""
    tiles = []
    for row in range(2):
        for col in range(2):
            tiles.append(
                {
                    "window": (col * 100, row * 100, 100, 100),
                    "global_offset_x": col * 100,
                    "global_offset_y": row * 100,
                    "nodata_fraction": 0.0,
                }
            )
    return tiles


# ---------------------------------------------------------------------------
# Real GeoTIFF Fixtures (Integration Tests - DETRGEO-51)
# ---------------------------------------------------------------------------


@pytest.fixture
def real_geotiff_rgb_uint8() -> Path:
    """Path to a real 128x128 RGB uint8 GeoTIFF in EPSG:4326."""
    path = FIXTURES_DIR / "rgb_uint8.tif"
    if not path.exists():
        pytest.skip("Real test fixture rgb_uint8.tif not available")
    return path


@pytest.fixture
def real_geotiff_rgb_uint8_utm() -> Path:
    """Path to a real 128x128 RGB uint8 GeoTIFF in EPSG:32610 (UTM)."""
    path = FIXTURES_DIR / "rgb_uint8_utm.tif"
    if not path.exists():
        pytest.skip("Real test fixture rgb_uint8_utm.tif not available")
    return path


@pytest.fixture
def real_geotiff_single_band() -> Path:
    """Path to a real 128x128 single-band uint8 GeoTIFF."""
    path = FIXTURES_DIR / "single_band.tif"
    if not path.exists():
        pytest.skip("Real test fixture single_band.tif not available")
    return path


@pytest.fixture
def real_geotiff_rgb_uint16() -> Path:
    """Path to a real 128x128 RGB uint16 GeoTIFF."""
    path = FIXTURES_DIR / "rgb_uint16.tif"
    if not path.exists():
        pytest.skip("Real test fixture rgb_uint16.tif not available")
    return path


@pytest.fixture
def real_geotiff_with_nodata() -> Path:
    """Path to a real 128x128 RGB uint8 GeoTIFF with nodata regions."""
    path = FIXTURES_DIR / "with_nodata.tif"
    if not path.exists():
        pytest.skip("Real test fixture with_nodata.tif not available")
    return path
