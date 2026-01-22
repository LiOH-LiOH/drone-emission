import json
import os
import requests
from datetime import datetime
from xml.etree import ElementTree as ET

from utils.coordinates import wgs84_to_3857
from utils.tools import create_retry_session

# WFS endpoint
WFS_URL = "https://tubvsig-so2sat-vm1.srv.mwn.de/geoserver/ows"

# The server uses EPSG:3857 (Web Mercator) as its native CRS
# We need to convert WGS84 (EPSG:4326) coordinates to EPSG:3857


def convert_bbox_to_3857(bbox_4326):
    """Convert bbox from EPSG:4326 to EPSG:3857 using shared utility"""
    min_lon, min_lat, max_lon, max_lat = bbox_4326
    min_x, min_y = wgs84_to_3857(min_lon, min_lat)
    max_x, max_y = wgs84_to_3857(max_lon, max_lat)
    return (min_x, min_y, max_x, max_y)


def get_capabilities():
    """Get WFS capabilities to see available layers"""
    params = {"service": "WFS", "version": "2.0.0", "request": "GetCapabilities"}

    print("Fetching WFS capabilities...")
    response = requests.get(WFS_URL, params=params, timeout=60)
    response.raise_for_status()

    # Parse XML to find layer names
    root = ET.fromstring(response.content)

    # Find all FeatureType elements (layers)
    namespaces = {
        "wfs": "http://www.opengis.net/wfs/2.0",
        "ows": "http://www.opengis.net/ows/1.1",
    }

    layers = []
    for feature_type in root.findall(".//wfs:FeatureType", namespaces):
        name = feature_type.find("wfs:Name", namespaces)
        title = feature_type.find("wfs:Title", namespaces)
        if name is not None:
            layer_info = {
                "name": name.text,
                "title": title.text if title is not None else name.text,
            }
            layers.append(layer_info)

    return layers


def get_feature_count(layer_name, bbox):
    """Get the count of features in the bounding box (bbox should be in EPSG:3857)"""
    bbox_str = f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}"  # x_min, y_min, x_max, y_max

    params = {
        "service": "WFS",
        "version": "2.0.0",
        "request": "GetFeature",
        "typeName": layer_name,
        "bbox": bbox_str,
        "srsName": "EPSG:3857",  # Server's native CRS
        "resultType": "hits",
    }

    session = create_retry_session()
    response = session.get(WFS_URL, params=params, timeout=120)

    if response.status_code != 200:
        print(f"  Count error {response.status_code}: {response.text[:300]}")
        response.raise_for_status()

    # Parse the response to get the count
    root = ET.fromstring(response.content)
    count = root.get("numberMatched") or root.get("numberOfFeatures")
    return int(count) if count else 0


def get_features(
    layer_name, bbox, max_features=10000, output_format="application/json"
):
    """
    Get features from WFS within a bounding box

    Args:
        layer_name: Name of the WFS layer
        bbox: Tuple in EPSG:3857 (min_x, min_y, max_x, max_y)
        max_features: Maximum features to retrieve per request
        output_format: Output format (application/json, json, etc.)

    Note: This layer doesn't support pagination (startIndex) because it lacks a primary key.
          For large areas, use grid-based downloading instead.
    """
    # Server uses EPSG:3857 (Web Mercator) as native CRS
    bbox_str = f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}"  # x_min, y_min, x_max, y_max

    params = {
        "service": "WFS",
        "version": "2.0.0",
        "request": "GetFeature",
        "typeName": layer_name,
        "bbox": bbox_str,
        "srsName": "EPSG:3857",  # Server's native CRS
        "outputFormat": output_format,
        "count": max_features,
        # NOTE: startIndex removed - this layer doesn't support pagination
    }

    print(f"  Fetching up to {max_features:,} features...")

    session = create_retry_session()
    response = session.get(WFS_URL, params=params, timeout=300)

    # If request fails, show the error details
    if response.status_code != 200:
        print(f"  Error {response.status_code}: {response.text[:500]}")
        response.raise_for_status()

    return response.json() if "json" in output_format.lower() else response.text


def download_all_features(
    city_name, layer_name, bbox, output_file="buildings.geojson", max_features=100000
):
    """
    Download features from a single bbox query (no pagination).

    Note: This server doesn't support pagination, so for large areas use download_by_grid() instead.

    Args:
        layer_name: WFS layer name
        bbox: Bounding box tuple in EPSG:3857
        output_file: Output filename
        max_features: Maximum features to retrieve
    """
    print(f"\n{'=' * 60}")
    print(f"Downloading from layer: {layer_name}")
    print(f"Bounding box: {bbox}")
    print(f"Output file: {output_file}")
    print("=" * 60)

    # First get the total count
    print("\nCounting features in area...")
    total_count = get_feature_count(layer_name, bbox)
    print(f"Total features in area: {total_count:,}")

    if total_count == 0:
        print("No features found in this area.")
        return None

    if total_count > max_features:
        print(
            f"\nWARNING: Area contains {total_count:,} features, but max is {max_features:,}"
        )
        print("Consider using download_by_grid() for large areas.")

    # Download features
    print(f"\nDownloading up to {max_features:,} features...")
    start_time = datetime.now()

    try:
        data = get_features(layer_name, bbox, max_features=max_features)

        if "features" not in data:
            print("No features in response")
            return None

        features = data["features"]
        elapsed = (datetime.now() - start_time).seconds
        print(f"Downloaded {len(features):,} features in {elapsed}s")

    except Exception as e:
        print(f"Error downloading: {e}")
        return None

    # Combine into single GeoJSON
    result = {
        "type": "FeatureCollection",
        "name": f"GlobalBuildingAtlas_{city_name}",
        "crs": {"type": "name", "properties": {"name": "urn:ogc:def:crs:EPSG::3857"}},
        "features": features,
    }

    # Save to file
    print(f"\nSaving to {output_file}...")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(result, f)

    file_size_mb = os.path.getsize(output_file) / (1024 * 1024)
    print(f"Saved {len(features):,} features ({file_size_mb:.1f} MB)")

    return result


def download_by_grid(
    city_name, layer_name, bbox_3857, output_dir="tiles", grid_size_meters=5000
):
    """
    Download large areas by dividing into a grid (required since server doesn't support pagination)

    Args:
        layer_name: WFS layer name
        bbox_3857: Full bounding box in EPSG:3857 (meters)
        output_dir: Directory for output file
        grid_size_meters: Size of each grid cell in meters (default 5km)
    """
    import math

    os.makedirs(output_dir, exist_ok=True)

    min_x, min_y, max_x, max_y = bbox_3857

    # Calculate grid based on meters
    cols = math.ceil((max_x - min_x) / grid_size_meters)
    rows = math.ceil((max_y - min_y) / grid_size_meters)
    total_tiles = cols * rows

    print(f"\n{'=' * 60}")
    print(f"Grid-based download for layer: {layer_name}")
    print(f"Dividing into {cols}x{rows} = {total_tiles} tiles")
    print(f"Grid cell size: {grid_size_meters}m")
    print("=" * 60)

    all_features = []
    tile_num = 0
    start_time = datetime.now()

    for row in range(rows):
        for col in range(cols):
            tile_num += 1
            tile_bbox = (
                min_x + col * grid_size_meters,
                min_y + row * grid_size_meters,
                min(min_x + (col + 1) * grid_size_meters, max_x),
                min(min_y + (row + 1) * grid_size_meters, max_y),
            )

            print(f"\nTile {tile_num}/{total_tiles}")

            try:
                count = get_feature_count(layer_name, tile_bbox)
                if count == 0:
                    print("  No features, skipping...")
                    continue

                print(f"  Found {count:,} features")

                # Download tile (max 100k per tile should be enough for 5km grid)
                data = get_features(layer_name, tile_bbox, max_features=100000)
                if "features" in data:
                    all_features.extend(data["features"])
                    elapsed = (datetime.now() - start_time).seconds
                    print(
                        f"  Total collected: {len(all_features):,} ({elapsed}s elapsed)"
                    )

            except Exception as e:
                print(f"  Error: {e}")
                continue

    # Save combined result
    output_file = os.path.join(output_dir, f"{city_name.lower()}_all_buildings.geojson")
    result = {
        "type": "FeatureCollection",
        "crs": {"type": "name", "properties": {"name": "urn:ogc:def:crs:EPSG::3857"}},
        "features": all_features,
    }

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(result, f)

    file_size_mb = os.path.getsize(output_file) / (1024 * 1024)
    print(f"\n{'=' * 60}")
    print(f"Download complete!")
    print(f"Total features: {len(all_features):,}")
    print(f"File size: {file_size_mb:.1f} MB")
    print(f"Saved to: {output_file}")

    return result


def fetch_region_buildings(region_name, bbox_4326, save_path):
    bbox = convert_bbox_to_3857(bbox_4326)

    return download_all_features(
        region_name, "lod1_global", bbox, save_path, max_features=500000
    )


# Shenzhen bounding box in WGS84 (EPSG:4326)
# Longitude: 113.75°E to 114.65°E
# Latitude: 22.45°N to 22.85°N
SHENZHEN_BBOX_4326 = (113.75, 22.45, 114.65, 22.85)
SHENZHEN_BBOX = convert_bbox_to_3857(SHENZHEN_BBOX_4326)

# For testing with a smaller area (Futian District center)
SHENZHEN_SMALL_BBOX_4326 = (114.03, 22.52, 114.08, 22.56)
SHENZHEN_SMALL_BBOX = convert_bbox_to_3857(SHENZHEN_SMALL_BBOX_4326)


def test_download(city_name, bbox_3857):
    print("=" * 60)
    print(f"GlobalBuildingAtlas WFS Query - {city_name}")
    print("=" * 60)

    # Step 1: Get available layers
    try:
        layers = get_capabilities()
    except Exception as e:
        print(f"Error connecting to WFS: {e}")
        print("\nTroubleshooting:")
        print("1. Check your internet connection")
        print("2. The server may be temporarily down")
        print("3. Try again later")
        return

    print(f"\nFound {len(layers)} available layers:")
    for i, layer in enumerate(layers, 1):
        print(f"  {i}. {layer['name']}")

    # Step 2: Find building layers
    # GlobalBuildingAtlas uses "lod1" (Level of Detail 1) and "polygon" naming
    building_layers = [
        l
        for l in layers
        if "lod1" in l["name"].lower()
        or "polygon" in l["name"].lower()
        or "global3d" in l["name"].lower()
    ]

    if not building_layers:
        print("\nNo building-specific layers found automatically.")
        print("Please check the layer names above and modify the script.")
        return

    print(f"\nBuilding-related layers found:")
    for layer in building_layers:
        print(f"  - {layer['name']}")

    # Prefer the main global LoD1 layer, or Asia-specific polygon layer
    preferred_order = ["lod1_global", "asia_oceania", "lod1_global_sipeo", "polygon"]
    layer_name = building_layers[0]["name"]  # Default to first

    for preferred in preferred_order:
        matching = [l for l in building_layers if preferred in l["name"].lower()]
        if matching:
            layer_name = matching[0]["name"]
            break

    print(f"\nSelected layer: {layer_name}")

    # Step 3: Test with small sample first
    print(f"\n{'=' * 60}")
    print(f"Download area {city_name}...")
    print(f"Bbox (EPSG:3857): {bbox_3857}")

    # Try different output formats (from GetCapabilities)
    output_formats = ["application/json", "json"]
    sample = None

    for fmt in output_formats:
        try:
            print(f"\n  Trying format: {fmt}")
            sample = get_features(
                layer_name, bbox_3857, max_features=100, output_format=fmt
            )

            if isinstance(sample, dict) and "features" in sample:
                print(f"  Success with format: {fmt}")
                break
            else:
                print(f"  Format {fmt} returned unexpected structure: {type(sample)}")
                sample = None

        except Exception as e:
            print(f"  Format {fmt} failed: {e}")
            sample = None
            continue

    if sample and "features" in sample and sample["features"]:
        print(f"\nSuccess! Got {len(sample['features'])} sample features.")
        print("\nSample feature properties:")
        props = sample["features"][0].get("properties", {})
        for key, value in list(props.items())[:10]:
            print(f"  {key}: {value}")

        # Save sample
        with open(f"{city_name.lower()}_sample.geojson", "w") as f:
            json.dump(sample, f, indent=2)
        print(f"\nSample saved to: {city_name.lower()}_sample.geojson")

    else:
        print("\nFailed to fetch sample with all formats.")
        print("The server may require a different query approach.")
        print("\nTry accessing via QGIS or check the WFS capabilities manually:")
        print(f"  {WFS_URL}?service=WFS&version=2.0.0&request=GetCapabilities")
        return

    # Step 4: Offer full download options
    print(f"\n{'=' * 60}")
    print("NEXT STEPS - Choose one option:")
    print("=" * 60)

    print(
        f"""
Option 1: Download all Shenzhen buildings (may be large!)
    download_all_features('{layer_name}', SHENZHEN_BBOX)

Option 2: Download by grid tiles (more reliable for large areas)
    download_by_grid('{layer_name}', SHENZHEN_BBOX)

Option 3: Download a smaller test area first
    download_all_features('{layer_name}', SHENZHEN_SMALL_BBOX, 'futian_test.geojson')

To run, uncomment one of these lines at the bottom of this script.
"""
    )


if __name__ == "__main__":
    test_download("Shenzhen", SHENZHEN_SMALL_BBOX)
