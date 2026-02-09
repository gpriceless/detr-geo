"""Example: Using detr_geo with Cloud-Optimized GeoTIFFs (COGs)

This example demonstrates cloud-native raster support in detr_geo:
- Loading imagery from HTTP URLs without downloading the full file
- Working with S3 URIs (requires AWS credentials)
- Processing STAC catalog items

Requirements:
    pip install detr-geo[cloud,rfdetr]
"""

from detr_geo import resolve_raster_source

# Example 1: Load a COG from HTTP URL
# ------------------------------------
# This only fetches metadata initially, not the full image
print("Example 1: HTTP COG URL")
cog_url = "https://example.com/path/to/cog.tif"

# Note: This is a mock example. For real use, replace with actual COG URL
# dg = DetrGeo(model="nano")
# dg.set_image(cog_url)
# detections = dg.detect_tiled(tile_size=512, overlap=0.2)
print(f"  Would load: {cog_url}")
print("  Only metadata is fetched initially")
print("  Tiled detection fetches only needed tile byte ranges\n")


# Example 2: Load from S3 URI
# ----------------------------
# Requires AWS credentials configured (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY)
print("Example 2: S3 URI")
s3_uri = "s3://my-bucket/imagery/aerial_2024.tif"

# dg = DetrGeo(model="nano")
# dg.set_image(s3_uri)
# detections = dg.detect_tiled(tile_size=512)
print(f"  Would load: {s3_uri}")
print("  Requires AWS credentials in environment or ~/.aws/credentials\n")


# Example 3: Load from STAC Item
# -------------------------------
print("Example 3: STAC Item")
print("  Requires: pip install pystac")

# Example using pystac:
# import pystac
#
# # Load a STAC item from a catalog
# item_url = "https://planetarycomputer.microsoft.com/api/stac/v1/collections/naip/items/tx_m_2609719_se_14_060_20201217"
# item = pystac.Item.from_file(item_url)
#
# # Pass the STAC item directly to detr_geo
# dg = DetrGeo(model="nano")
# dg.set_image(item)  # Automatically extracts the COG URL from the item
# detections = dg.detect_tiled()

print("  STAC items can be passed directly to set_image()")
print("  detr_geo automatically extracts the visual/image asset URL\n")


# Example 4: Check raster source resolution
# ------------------------------------------
print("Example 4: Resolve raster sources")
local_path = "/path/to/local/file.tif"
http_url = "https://example.com/cog.tif"
s3_uri = "s3://bucket/key.tif"

# All of these resolve to rasterio-openable strings
print(f"  HTTP URL: {http_url} -> {resolve_raster_source(http_url)}")
print(f"  S3 URI: {s3_uri} -> {resolve_raster_source(s3_uri)}")

print("\nAll source types work identically with set_image() and detect_tiled()")
print("Cloud sources use HTTP Range requests for efficient tile reads")
