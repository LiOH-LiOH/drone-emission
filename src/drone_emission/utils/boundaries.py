"""Boundary helpers shared across pipeline stages.

`enlarged_boundaries` is derived from `boundaries_dict` and the configured
delivery radius — it must always be recomputed at use time. A previous
implementation cached it in `region_stats.json`, but the cache silently went
stale whenever `regions.json` was updated, leading to roadnets/buildings
fetched against a too-small bbox.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from typing import Dict, Tuple

import numpy as np

EARTH_RADIUS_KM = 6371.0
_GRAPHML_NS = "{http://graphml.graphdrawing.org/xmlns}"


def enlarge_boundaries(boundaries_dict: Dict[str, float], enlarge_km: float) -> Dict[str, float]:
    """Pad the bbox outward by `enlarge_km` on every side."""
    center_lat = (boundaries_dict["min_lat"] + boundaries_dict["max_lat"]) / 2
    km_per_deg_lat = EARTH_RADIUS_KM * np.pi / 180
    km_per_deg_lng = km_per_deg_lat * np.cos(np.radians(center_lat))
    return {
        "min_lat": boundaries_dict["min_lat"] - enlarge_km / km_per_deg_lat,
        "max_lat": boundaries_dict["max_lat"] + enlarge_km / km_per_deg_lat,
        "min_lng": boundaries_dict["min_lng"] - enlarge_km / km_per_deg_lng,
        "max_lng": boundaries_dict["max_lng"] + enlarge_km / km_per_deg_lng,
    }


def resolve_enlarged_boundaries(region_info, exp_config) -> Dict[str, float]:
    """Always-fresh enlarged boundaries from `boundaries_dict + delivery_radius_km`.

    Also writes the result back to region_info as a stat so visualization
    tooling that still reads it sees the current value, not a stale one.
    Returns None when boundaries_dict is missing (rare; handled by callers).
    """
    boundaries_dict = region_info.to_dict().get("boundaries_dict")
    if not boundaries_dict:
        return None
    enlarge_km = exp_config.get("delivery_radius_km", 5.0)
    large = enlarge_boundaries(boundaries_dict, enlarge_km)
    region_info.set_stat("enlarged_boundaries", large)
    return large


def bbox_from_graphml(path: str) -> Tuple[float, float, float, float]:
    """Stream a GraphML and return (min_lat, max_lat, min_lng, max_lng).

    Avoids the full NetworkX/OSMnx load — only `<key>` declarations and
    `<node>/<data>` tags are touched. Peak memory is independent of city size.
    """
    y_key = x_key = None
    lat_min = lng_min = float("inf")
    lat_max = lng_max = float("-inf")
    n_nodes = 0
    cur_y = cur_x = None
    in_node = False

    for event, elem in ET.iterparse(path, events=("start", "end")):
        tag = elem.tag
        if event == "start":
            if tag == _GRAPHML_NS + "key" and elem.get("for") == "node":
                name = elem.get("attr.name")
                if name == "y":
                    y_key = elem.get("id")
                elif name == "x":
                    x_key = elem.get("id")
            elif tag == _GRAPHML_NS + "node":
                in_node = True
                cur_y = cur_x = None
        else:
            if tag == _GRAPHML_NS + "data" and in_node:
                k = elem.get("key")
                if k == y_key:
                    try: cur_y = float(elem.text)
                    except (TypeError, ValueError): pass
                elif k == x_key:
                    try: cur_x = float(elem.text)
                    except (TypeError, ValueError): pass
                elem.clear()
            elif tag == _GRAPHML_NS + "node":
                if cur_y is not None and cur_x is not None:
                    if cur_y < lat_min: lat_min = cur_y
                    if cur_y > lat_max: lat_max = cur_y
                    if cur_x < lng_min: lng_min = cur_x
                    if cur_x > lng_max: lng_max = cur_x
                    n_nodes += 1
                in_node = False
                elem.clear()
            elif tag == _GRAPHML_NS + "edge":
                elem.clear()

    if n_nodes == 0:
        raise RuntimeError(f"no nodes parsed from {path}")
    return lat_min, lat_max, lng_min, lng_max


def graphml_covers_bbox(graphml_path: str, expected: Dict[str, float],
                        tolerance_deg: float = 1e-3) -> bool:
    """True if the graphml's node bbox encloses `expected` within tolerance.

    A small tolerance absorbs nodes that fall a few meters inside the requested
    bbox due to OSMnx clipping behavior.
    """
    lat_min, lat_max, lng_min, lng_max = bbox_from_graphml(graphml_path)
    return (
        lat_min <= expected["min_lat"] + tolerance_deg and
        lat_max >= expected["max_lat"] - tolerance_deg and
        lng_min <= expected["min_lng"] + tolerance_deg and
        lng_max >= expected["max_lng"] - tolerance_deg
    )
