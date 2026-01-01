import warnings
from typing import Any, Dict, Optional, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import osmnx as ox

# Configure osmnx settings
ox.settings.use_cache = True
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


def find_nearest_node(
    G: nx.MultiDiGraph, point: Tuple[float, float]
) -> Tuple[int, float]:
    lat, lon = point
    nearest_node, distance = ox.distance.nearest_nodes(
        G, X=lon, Y=lat, return_dist=True
    )
    return nearest_node, distance


def find_shortest_path(
    G: nx.MultiDiGraph,
    origin: Tuple[float, float],
    destination: Tuple[float, float],
    weight: str = "length",
) -> Dict[str, Any]:
    """
    Find the shortest path between two points on the road network.

    This function:
    1. Finds the nearest intersections (nodes) to the origin and destination
    2. Computes the shortest path between these nodes
    3. Returns the path, total distance, and related information

    Parameters
    ----------
    G : networkx.MultiDiGraph
        The road network graph (from fetch_roadnet_by_bbox)
    origin : tuple
        (latitude, longitude) of the starting point
    destination : tuple
        (latitude, longitude) of the ending point
    weight : str, optional
        Edge attribute to minimize. Options:
        - 'length': shortest distance (default)
        - 'travel_time': fastest route (requires speed data)

    Returns
    -------
    result : dict
        Dictionary containing:
        - 'path_nodes': list of node IDs in the path
        - 'path_coordinates': list of (lat, lon) tuples for the path
        - 'total_distance': total path distance in meters
        - 'origin_node': nearest node to origin
        - 'destination_node': nearest node to destination
        - 'origin_snap_distance': distance from origin to nearest node (meters)
        - 'destination_snap_distance': distance from destination to nearest node (meters)
        - 'edge_details': list of edge attributes along the path
        - 'success': boolean indicating if path was found
        - 'message': status message

    Example
    -------
    >>> G = fetch_roadnet_by_bbox(39.92, 39.90, 116.40, 116.38)
    >>> origin = (39.915, 116.385)
    >>> destination = (39.918, 116.395)
    >>> result = find_shortest_path(G, origin, destination)
    >>> print(f"Total distance: {result['total_distance']:.2f} meters")
    """

    result = {
        "path_nodes": [],
        "path_coordinates": [],
        "total_distance": 0.0,
        "origin_node": None,
        "destination_node": None,
        "origin_snap_distance": 0.0,
        "destination_snap_distance": 0.0,
        "edge_details": [],
        "success": False,
        "message": "",
    }

    try:
        # Step 1: Find nearest nodes to origin and destination
        origin_node, origin_dist = find_nearest_node(G, origin)
        dest_node, dest_dist = find_nearest_node(G, destination)

        result["origin_node"] = origin_node
        result["destination_node"] = dest_node
        result["origin_snap_distance"] = origin_dist
        result["destination_snap_distance"] = dest_dist

        print(
            f"Origin point {origin} -> Nearest node: {origin_node} (distance: {origin_dist:.2f}m)"
        )
        print(
            f"Destination point {destination} -> Nearest node: {dest_node} (distance: {dest_dist:.2f}m)"
        )

        # Step 2: Check if origin and destination are the same node
        if origin_node == dest_node:
            result["success"] = True
            result["message"] = "Origin and destination are at the same intersection"
            result["path_nodes"] = [origin_node]
            node_data = G.nodes[origin_node]
            result["path_coordinates"] = [(node_data["y"], node_data["x"])]
            return result

        # Step 3: Find shortest path
        path_nodes = ox.shortest_path(G, origin_node, dest_node, weight=weight)

        if path_nodes is None:
            result["message"] = "No path found between origin and destination"
            return result

        result["path_nodes"] = path_nodes

        # Step 4: Extract path coordinates
        path_coordinates = []
        for node in path_nodes:
            node_data = G.nodes[node]
            path_coordinates.append((node_data["y"], node_data["x"]))
        result["path_coordinates"] = path_coordinates

        # Step 5: Calculate total distance and extract edge details
        total_distance = 0.0
        edge_details = []

        for i in range(len(path_nodes) - 1):
            u, v = path_nodes[i], path_nodes[i + 1]

            # Get edge data (MultiDiGraph may have multiple edges)
            edge_data = G.get_edge_data(u, v)

            if edge_data:
                # Get the first edge (key=0) or the one with minimum weight
                min_key = min(
                    edge_data.keys(),
                    key=lambda k: edge_data[k].get(weight, float("inf")),
                )
                edge = edge_data[min_key]

                edge_length = edge.get("length", 0)
                total_distance += edge_length

                edge_details.append(
                    {
                        "from_node": u,
                        "to_node": v,
                        "length": edge_length,
                        "highway": edge.get("highway", "unknown"),
                        "name": edge.get("name", "unnamed"),
                        "oneway": edge.get("oneway", False),
                    }
                )

        result["total_distance"] = total_distance
        result["edge_details"] = edge_details
        result["success"] = True
        result["message"] = "Path found successfully"

        print(f"Shortest path found!")
        print(f"  - Number of nodes: {len(path_nodes)}")
        print(
            f"  - Total distance: {total_distance:.2f} meters ({total_distance/1000:.3f} km)"
        )

    except nx.NetworkXNoPath:
        result["message"] = (
            "No path exists between origin and destination (disconnected graph)"
        )
    except Exception as e:
        result["message"] = f"Error finding path: {str(e)}"
        warnings.warn(result["message"])

    return result
