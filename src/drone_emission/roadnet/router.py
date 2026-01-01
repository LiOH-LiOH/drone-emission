"""Road network routing functionality."""

import warnings
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
import osmnx as ox
from haversine import Unit, haversine
from scipy.spatial import cKDTree


class RoadNetRouter:
    """Router for finding shortest paths on a road network.

    This class holds a road network graph and provides efficient routing methods.
    Precomputes a KD-Tree for fast nearest node lookups.

    Example
    -------
    >>> G = ox.load_graphml("roadnet.graphml")
    >>> router = RoadNetRouter(G)
    >>> result = router.find_shortest_path((22.5, 114.0), (22.6, 114.1))
    >>> print(f"Distance: {result['total_distance']:.2f}m")
    """

    def __init__(self, G: nx.MultiDiGraph, weight: str = "length"):
        """Initialize the router with a road network graph.

        Args:
            G: OSMnx road network graph
            weight: Default edge weight for path finding ('length' or 'travel_time')
        """
        self.G = G
        self.weight = weight

        # Precompute node coordinates and KD-Tree for fast nearest neighbor lookup
        self._node_ids = list(G.nodes())
        self._node_coords = np.array(
            [(G.nodes[n]["y"], G.nodes[n]["x"]) for n in self._node_ids]
        )  # (lat, lng)
        self._kdtree = cKDTree(self._node_coords)

        print(f"RoadNetRouter initialized with {len(self._node_ids)} nodes")

    def find_nearest_node(self, point: Tuple[float, float]) -> Tuple[int, float]:
        """Find the nearest node to a given point.

        Args:
            point: (latitude, longitude) tuple

        Returns:
            Tuple of (node_id, distance_in_meters)
        """
        lat, lon = point
        # Query KD-Tree to get index of nearest node
        _, idx = self._kdtree.query([lat, lon])

        node_id = self._node_ids[idx]

        # Calculate actual distance using haversine
        node_lat, node_lon = self._node_coords[idx]
        dist_meters = haversine((lat, lon), (node_lat, node_lon), unit=Unit.METERS)

        return node_id, dist_meters

    def find_shortest_path(
        self,
        origin: Tuple[float, float],
        destination: Tuple[float, float],
        weight: Optional[str] = None,
        verbose: bool = False,
    ) -> Dict[str, Any]:
        """Find the shortest path between two points.

        Args:
            origin: (latitude, longitude) of starting point
            destination: (latitude, longitude) of ending point
            weight: Edge weight to minimize ('length' or 'travel_time')
            verbose: If True, print progress information

        Returns:
            Dictionary containing path information
        """
        weight = weight or self.weight

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
            # Step 1: Find nearest nodes
            origin_node, origin_dist = self.find_nearest_node(origin)
            dest_node, dest_dist = self.find_nearest_node(destination)

            result["origin_node"] = origin_node
            result["destination_node"] = dest_node
            result["origin_snap_distance"] = origin_dist
            result["destination_snap_distance"] = dest_dist

            if verbose:
                print(f"Origin {origin} -> Node {origin_node} ({origin_dist:.2f}m)")
                print(
                    f"Destination {destination} -> Node {dest_node} ({dest_dist:.2f}m)"
                )

            # Step 2: Check if same node
            if origin_node == dest_node:
                result["success"] = True
                result["message"] = (
                    "Origin and destination are at the same intersection"
                )
                result["path_nodes"] = [origin_node]
                node_data = self.G.nodes[origin_node]
                result["path_coordinates"] = [(node_data["y"], node_data["x"])]
                return result

            # Step 3: Find shortest path
            path_nodes = ox.shortest_path(self.G, origin_node, dest_node, weight=weight)

            if path_nodes is None:
                result["message"] = "No path found between origin and destination"
                return result

            result["path_nodes"] = path_nodes

            # Step 4: Extract path coordinates
            path_coordinates = [
                (self.G.nodes[n]["y"], self.G.nodes[n]["x"]) for n in path_nodes
            ]
            result["path_coordinates"] = path_coordinates

            # Step 5: Calculate total distance
            total_distance = 0.0
            edge_details = []

            for i in range(len(path_nodes) - 1):
                u, v = path_nodes[i], path_nodes[i + 1]
                edge_data = self.G.get_edge_data(u, v)

                if edge_data:
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

            if verbose:
                print(f"Shortest path: {len(path_nodes)} nodes, {total_distance:.2f}m")

        except nx.NetworkXNoPath:
            result["message"] = "No path exists (disconnected graph)"
        except Exception as e:
            result["message"] = f"Error finding path: {str(e)}"
            warnings.warn(result["message"])

        return result

    def find_shortest_paths_batch(
        self,
        od_pairs: List[Tuple[Tuple[float, float], Tuple[float, float]]],
        weight: Optional[str] = None,
        verbose: bool = False,
    ) -> List[Dict[str, Any]]:
        """Find shortest paths for multiple origin-destination pairs.

        Args:
            od_pairs: List of (origin, destination) tuples
            weight: Edge weight to minimize
            verbose: If True, print progress

        Returns:
            List of path result dictionaries
        """
        results = []
        total = len(od_pairs)

        for i, (origin, destination) in enumerate(od_pairs):
            if verbose and (i + 1) % 100 == 0:
                print(f"Processing path {i + 1}/{total}")

            result = self.find_shortest_path(origin, destination, weight, verbose=False)
            results.append(result)

        if verbose:
            success_count = sum(1 for r in results if r["success"])
            print(f"Completed: {success_count}/{total} paths found successfully")

        return results

    def get_path_distance(
        self,
        origin: Tuple[float, float],
        destination: Tuple[float, float],
        weight: Optional[str] = None,
    ) -> float:
        """Get only the distance of shortest path (faster for distance-only queries).

        Args:
            origin: (latitude, longitude) of starting point
            destination: (latitude, longitude) of ending point
            weight: Edge weight to minimize

        Returns:
            Total distance in meters, or -1 if no path found
        """
        weight = weight or self.weight

        origin_node, _ = self.find_nearest_node(origin)
        dest_node, _ = self.find_nearest_node(destination)

        if origin_node == dest_node:
            return 0.0

        try:
            distance = nx.shortest_path_length(
                self.G, origin_node, dest_node, weight=weight
            )
            return distance
        except nx.NetworkXNoPath:
            return -1.0

    @property
    def num_nodes(self) -> int:
        """Number of nodes in the graph."""
        return len(self._node_ids)

    @property
    def num_edges(self) -> int:
        """Number of edges in the graph."""
        return self.G.number_of_edges()
