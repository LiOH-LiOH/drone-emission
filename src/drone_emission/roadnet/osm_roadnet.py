import os
import warnings
from typing import Any, Dict, Optional, Tuple

import networkx as nx
import osmnx as ox

# Configure osmnx settings
# Place cache at project root (./cache/osmnx) regardless of CWD
_PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
ox.settings.use_cache = True
ox.settings.cache_folder = os.path.join(_PROJECT_ROOT, "cache", "osmnx")
ox.settings.log_console = False


def fetch_bikeable_roadnet(bbox_4326: Dict[str, float], save_path: str) -> None:
    """
    Fetch road network suitable for cycling in Chinese cities.

    This includes:
    - All roads where bicycles are allowed (explicit or implicit)
    - Regular car roads (which bikes can use in China)
    - Dedicated bike lanes and paths
    - Excludes only motorways and roads explicitly forbidding bikes
    """

    # Custom filter: include all roads EXCEPT those explicitly forbidding bikes
    # or high-speed motorways where bikes are prohibited
    custom_filter = (
        '["highway"]'
        '["area"!~"yes"]'
        '["highway"!~"motorway|motorway_link|proposed|construction|abandoned|platform|raceway"]'
        '["bicycle"!~"no"]'
        '["service"!~"private"]'
    )

    # osmnx 2.x uses bbox as (left, bottom, right, top) = (west, south, east, north)
    bbox = (
        bbox_4326["min_lng"],  # west
        bbox_4326["min_lat"],  # south
        bbox_4326["max_lng"],  # east
        bbox_4326["max_lat"],  # north
    )

    G = ox.graph_from_bbox(
        bbox=bbox,
        custom_filter=custom_filter,
        simplify=False,
        retain_all=True,
    )

    G = ox.distance.add_edge_lengths(G)

    ox.save_graphml(G, save_path)
    print(f"Road network saved to {save_path}")
