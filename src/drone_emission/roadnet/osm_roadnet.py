import warnings
from typing import Any, Dict, Optional, Tuple

import networkx as nx
import osmnx as ox

# Configure osmnx settings
# Cache folder is set by utils.config.configure_cache() at startup
ox.settings.use_cache = True
ox.settings.log_console = False
ox.settings.requests_timeout = 60  # HTTP connect/read timeout (seconds)
ox.settings.overpass_settings = '[out:json][timeout:180]'  # Overpass server-side query timeout
# Default overpass-api.de is unreachable from some remotes (e.g. hku-sto behind GFW).
# Use the French OSM mirror which routes more reliably; can be overridden via env var.
# osmnx 2.x uses settings.overpass_url and appends "/interpreter" itself.
import os as _os
ox.settings.overpass_url = _os.environ.get(
    "OSMNX_OVERPASS_URL",
    "https://overpass.openstreetmap.fr/api",
)
# OSM-France blocks the default "OSMnx Python package..." User-Agent and any UA
# containing "python-requests". "OverpassQL" identifies us as a direct Overpass
# client and is on their allow list. Override with OSMNX_USER_AGENT.
ox.settings.http_user_agent = _os.environ.get("OSMNX_USER_AGENT", "OverpassQL")


BIKEABLE_ROAD_FILTER = (
    '["highway"]'
    '["area"!~"yes"]'
    '["highway"!~"motorway|motorway_link|proposed|construction|abandoned|platform|raceway"]'
    '["bicycle"!~"no"]'
    '["service"!~"private"]'
)
ALL_HIGHWAY_FILTER = '["highway"]["area"!~"yes"]'


def _fetch_roadnet(
    bbox_4326: Dict[str, float],
    save_path: str,
    custom_filter: str,
) -> None:
    """Fetch an unsimplified OSM road graph using ``custom_filter``."""
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


def fetch_bikeable_roadnet(bbox_4326: Dict[str, float], save_path: str) -> None:
    """Fetch roads suitable for cycling in Chinese cities."""
    _fetch_roadnet(bbox_4326, save_path, BIKEABLE_ROAD_FILTER)


def fetch_all_highway_roadnet(
    bbox_4326: Dict[str, float],
    save_path: str,
) -> None:
    """Fetch every non-area OSM highway for a local comparison map."""
    _fetch_roadnet(bbox_4326, save_path, ALL_HIGHWAY_FILTER)
