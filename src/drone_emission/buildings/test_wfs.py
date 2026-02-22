"""
Test script to probe the GlobalBuildingAtlas WFS endpoint.

Uses a small bbox (Futian district center, ~2km x 2km) to test:
  1. GetCapabilities - available layers and supported operations
  2. GetFeature count - how many features in the test area
  3. GetFeature download - fetch a small batch and inspect schema
  4. Pagination support - test if startIndex works
  5. Grid subdivision - simulate grid-based download for large areas
  6. Output format support - test which formats the server accepts

Run from src/drone_emission/:
    python -m buildings.test_wfs
"""

import json
import math
import time
from datetime import datetime
from xml.etree import ElementTree as ET

import requests

from utils.coordinates import wgs84_to_3857
from utils.tools import create_retry_session

WFS_URL = "https://tubvsig-so2sat-vm1.srv.mwn.de/geoserver/ows"

# Small test area: Futian district center (~2km x 2km)
TEST_BBOX_4326 = (114.03, 22.53, 114.05, 22.55)
TEST_BBOX_3857 = wgs84_to_3857(TEST_BBOX_4326[0], TEST_BBOX_4326[1]) + wgs84_to_3857(
    TEST_BBOX_4326[2], TEST_BBOX_4326[3]
)

LAYER_NAME = "lod1_global"


def header(title):
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def test_capabilities():
    """Test 1: GetCapabilities - list layers and supported operations."""
    header("Test 1: GetCapabilities")

    params = {"service": "WFS", "version": "2.0.0", "request": "GetCapabilities"}
    session = create_retry_session()
    resp = session.get(WFS_URL, params=params, timeout=60)

    print(f"Status: {resp.status_code}")
    if resp.status_code != 200:
        print(f"Error: {resp.text[:500]}")
        return False

    root = ET.fromstring(resp.content)
    ns = {
        "wfs": "http://www.opengis.net/wfs/2.0",
        "ows": "http://www.opengis.net/ows/1.1",
    }

    # List layers
    layers = []
    for ft in root.findall(".//wfs:FeatureType", ns):
        name = ft.find("wfs:Name", ns)
        title = ft.find("wfs:Title", ns)
        crs = ft.find("wfs:DefaultCRS", ns)
        if name is not None:
            layers.append(
                {
                    "name": name.text,
                    "title": title.text if title is not None else "",
                    "crs": crs.text if crs is not None else "",
                }
            )

    print(f"\nAvailable layers ({len(layers)}):")
    for layer in layers:
        marker = " <--" if "lod1" in layer["name"].lower() else ""
        print(f"  {layer['name']:40s} CRS: {layer['crs']}{marker}")

    # List supported operations
    ops = root.findall(".//ows:Operation", ns)
    print(f"\nSupported operations ({len(ops)}):")
    for op in ops:
        print(f"  {op.get('name')}")

    # List output formats
    output_formats = root.findall(
        ".//ows:Operation[@name='GetFeature']//ows:Parameter[@name='outputFormat']//ows:Value",
        ns,
    )
    if output_formats:
        print(f"\nGetFeature output formats:")
        for fmt in output_formats:
            print(f"  {fmt.text}")

    return True


def test_feature_count():
    """Test 2: Count features in the test bbox."""
    header("Test 2: Feature Count")

    bbox_str = ",".join(str(v) for v in TEST_BBOX_3857)
    params = {
        "service": "WFS",
        "version": "2.0.0",
        "request": "GetFeature",
        "typeName": LAYER_NAME,
        "bbox": bbox_str,
        "srsName": "EPSG:3857",
        "resultType": "hits",
    }

    print(f"Test bbox (WGS84): {TEST_BBOX_4326}")
    print(f"Test bbox (3857):  {TEST_BBOX_3857}")

    session = create_retry_session()
    t0 = time.time()
    resp = session.get(WFS_URL, params=params, timeout=120)
    elapsed = time.time() - t0

    print(f"Status: {resp.status_code} ({elapsed:.1f}s)")
    if resp.status_code != 200:
        print(f"Error: {resp.text[:500]}")
        return 0

    root = ET.fromstring(resp.content)
    count = root.get("numberMatched") or root.get("numberOfFeatures")
    count = int(count) if count else 0
    print(f"Feature count: {count:,}")
    return count


def test_fetch_features(max_features=10):
    """Test 3: Fetch a small batch and inspect the schema."""
    header(f"Test 3: Fetch {max_features} Features")

    bbox_str = ",".join(str(v) for v in TEST_BBOX_3857)
    params = {
        "service": "WFS",
        "version": "2.0.0",
        "request": "GetFeature",
        "typeName": LAYER_NAME,
        "bbox": bbox_str,
        "srsName": "EPSG:3857",
        "outputFormat": "application/json",
        "count": max_features,
    }

    session = create_retry_session()
    t0 = time.time()
    resp = session.get(WFS_URL, params=params, timeout=120)
    elapsed = time.time() - t0

    print(f"Status: {resp.status_code} ({elapsed:.1f}s)")
    if resp.status_code != 200:
        print(f"Error: {resp.text[:500]}")
        return None

    data = resp.json()
    features = data.get("features", [])
    print(f"Features returned: {len(features)}")

    if not features:
        print("No features returned.")
        return None

    # Inspect schema
    sample = features[0]
    props = sample.get("properties", {})
    geom = sample.get("geometry", {})

    print(f"\nGeometry type: {geom.get('type')}")
    print(f"Coordinate sample: {str(geom.get('coordinates', []))[:120]}...")

    print(f"\nProperties schema ({len(props)} fields):")
    for key, value in props.items():
        vtype = type(value).__name__
        print(f"  {key:20s}  {vtype:8s}  {str(value)[:60]}")

    # Check for height field
    height_fields = [k for k in props if "height" in k.lower() or k in ("h", "pred_h")]
    if height_fields:
        print(f"\nHeight field(s) found: {height_fields}")
    else:
        print("\nWARNING: No obvious height field found in properties.")

    return data


def test_pagination():
    """Test 4: Check if startIndex pagination works."""
    header("Test 4: Pagination Support (startIndex)")

    bbox_str = ",".join(str(v) for v in TEST_BBOX_3857)
    base_params = {
        "service": "WFS",
        "version": "2.0.0",
        "request": "GetFeature",
        "typeName": LAYER_NAME,
        "bbox": bbox_str,
        "srsName": "EPSG:3857",
        "outputFormat": "application/json",
        "count": 5,
    }

    session = create_retry_session()

    # Page 1: no startIndex
    print("Page 1 (startIndex=0)...")
    t0 = time.time()
    resp1 = session.get(WFS_URL, params=base_params, timeout=120)
    print(f"  Status: {resp1.status_code} ({time.time() - t0:.1f}s)")

    if resp1.status_code != 200:
        print(f"  Error: {resp1.text[:300]}")
        return False

    data1 = resp1.json()
    ids1 = [f.get("id") for f in data1.get("features", [])]
    print(f"  Got {len(ids1)} features: {ids1}")

    # Page 2: startIndex=5
    print("Page 2 (startIndex=5)...")
    params2 = {**base_params, "startIndex": 5}
    t0 = time.time()
    resp2 = session.get(WFS_URL, params=params2, timeout=120)
    print(f"  Status: {resp2.status_code} ({time.time() - t0:.1f}s)")

    if resp2.status_code != 200:
        print(f"  Error (pagination likely unsupported): {resp2.text[:300]}")
        print("\n  RESULT: Pagination NOT supported. Grid-based download required.")
        return False

    data2 = resp2.json()
    ids2 = [f.get("id") for f in data2.get("features", [])]
    print(f"  Got {len(ids2)} features: {ids2}")

    # Check for overlap
    overlap = set(ids1) & set(ids2)
    if overlap:
        print(f"\n  WARNING: Pages overlap ({len(overlap)} shared IDs). Pagination may be broken.")
        print("  RESULT: Pagination NOT reliable. Grid-based download required.")
        return False

    print("\n  RESULT: Pagination works. Can use startIndex for large downloads.")
    return True


def test_output_formats():
    """Test 5: Check which output formats are accepted."""
    header("Test 5: Output Format Support")

    bbox_str = ",".join(str(v) for v in TEST_BBOX_3857)
    formats_to_test = [
        "application/json",
        "json",
        "application/geo+json",
        "text/xml; subtype=gml/3.2",
        "csv",
    ]

    session = create_retry_session()

    for fmt in formats_to_test:
        params = {
            "service": "WFS",
            "version": "2.0.0",
            "request": "GetFeature",
            "typeName": LAYER_NAME,
            "bbox": bbox_str,
            "srsName": "EPSG:3857",
            "outputFormat": fmt,
            "count": 2,
        }

        t0 = time.time()
        try:
            resp = session.get(WFS_URL, params=params, timeout=60)
            elapsed = time.time() - t0
            ok = resp.status_code == 200
            size = len(resp.content)
            print(f"  {fmt:40s}  {'OK' if ok else 'FAIL':4s}  {resp.status_code}  {size:>8,} bytes  {elapsed:.1f}s")
        except Exception as e:
            print(f"  {fmt:40s}  ERR   {e}")


def test_grid_subdivision():
    """Test 6: Simulate grid-based download on a small area."""
    header("Test 6: Grid Subdivision (dry run)")

    # Use the test bbox, subdivide into ~1km grid
    grid_size_m = 1000
    min_x, min_y, max_x, max_y = TEST_BBOX_3857

    cols = math.ceil((max_x - min_x) / grid_size_m)
    rows = math.ceil((max_y - min_y) / grid_size_m)
    total = cols * rows

    print(f"Bbox size: {max_x - min_x:.0f}m x {max_y - min_y:.0f}m")
    print(f"Grid: {cols} x {rows} = {total} tiles ({grid_size_m}m each)")

    session = create_retry_session()
    total_features = 0
    tile_counts = []

    for row in range(rows):
        for col in range(cols):
            tile_bbox = (
                min_x + col * grid_size_m,
                min_y + row * grid_size_m,
                min(min_x + (col + 1) * grid_size_m, max_x),
                min(min_y + (row + 1) * grid_size_m, max_y),
            )
            bbox_str = ",".join(str(v) for v in tile_bbox)
            params = {
                "service": "WFS",
                "version": "2.0.0",
                "request": "GetFeature",
                "typeName": LAYER_NAME,
                "bbox": bbox_str,
                "srsName": "EPSG:3857",
                "resultType": "hits",
            }

            t0 = time.time()
            resp = session.get(WFS_URL, params=params, timeout=120)
            elapsed = time.time() - t0

            if resp.status_code == 200:
                root = ET.fromstring(resp.content)
                count = int(root.get("numberMatched") or root.get("numberOfFeatures") or 0)
                total_features += count
                tile_counts.append(count)
                print(f"  Tile [{row},{col}]: {count:>6,} features ({elapsed:.1f}s)")
            else:
                print(f"  Tile [{row},{col}]: ERROR {resp.status_code} ({elapsed:.1f}s)")
                tile_counts.append(-1)

    print(f"\nTotal features across tiles: {total_features:,}")
    if tile_counts:
        valid = [c for c in tile_counts if c >= 0]
        print(f"Max per tile: {max(valid):,}")
        print(f"Avg per tile: {sum(valid) / len(valid):,.0f}")

    # Estimate for full Shenzhen
    sz_bbox_4326 = (113.75, 22.45, 114.65, 22.85)
    sz_3857 = wgs84_to_3857(sz_bbox_4326[0], sz_bbox_4326[1]) + wgs84_to_3857(
        sz_bbox_4326[2], sz_bbox_4326[3]
    )
    sz_w = sz_3857[2] - sz_3857[0]
    sz_h = sz_3857[3] - sz_3857[1]
    sz_cols = math.ceil(sz_w / 3000)
    sz_rows = math.ceil(sz_h / 3000)

    print(f"\nFull Shenzhen estimate (3km grid):")
    print(f"  Size: {sz_w / 1000:.1f}km x {sz_h / 1000:.1f}km")
    print(f"  Grid: {sz_cols} x {sz_rows} = {sz_cols * sz_rows} tiles")


def main():
    print("GlobalBuildingAtlas WFS Test Suite")
    print(f"Endpoint: {WFS_URL}")
    print(f"Layer: {LAYER_NAME}")
    print(f"Test bbox (WGS84): {TEST_BBOX_4326}")

    results = {}

    # Test 1: Capabilities
    try:
        results["capabilities"] = test_capabilities()
    except Exception as e:
        print(f"FAILED: {e}")
        results["capabilities"] = False

    # Test 2: Feature count
    try:
        results["count"] = test_feature_count()
    except Exception as e:
        print(f"FAILED: {e}")
        results["count"] = 0

    # Test 3: Fetch features
    try:
        results["fetch"] = test_fetch_features(10)
    except Exception as e:
        print(f"FAILED: {e}")
        results["fetch"] = None

    # Test 4: Pagination
    try:
        results["pagination"] = test_pagination()
    except Exception as e:
        print(f"FAILED: {e}")
        results["pagination"] = False

    # Test 5: Output formats
    try:
        test_output_formats()
    except Exception as e:
        print(f"FAILED: {e}")

    # Test 6: Grid subdivision
    try:
        test_grid_subdivision()
    except Exception as e:
        print(f"FAILED: {e}")

    # Summary
    header("Summary")
    print(f"Capabilities:  {'OK' if results.get('capabilities') else 'FAIL'}")
    print(f"Feature count: {results.get('count', 0):,}")
    print(f"Fetch data:    {'OK' if results.get('fetch') else 'FAIL'}")
    print(f"Pagination:    {'Supported' if results.get('pagination') else 'NOT supported (use grid)'}")


if __name__ == "__main__":
    main()
