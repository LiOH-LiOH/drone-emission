"""Compatibility bridge for visualization that requires NetworkX graphs.

Loads NetworkX graphs on demand and discards after use, keeping the
main routing path memory-efficient via SparseGraph.
"""

import networkx as nx
import osmnx as ox


def load_networkx_graph(graphml_path: str) -> nx.MultiDiGraph:
    """Load a full NetworkX graph from GraphML for visualization."""
    return ox.load_graphml(graphml_path)


def load_subgraph_for_visualization(
    graphml_path: str,
    bounds: dict,
    margin_deg: float = 0.005,
) -> nx.MultiDiGraph:
    """Load a geographic subgraph for local visualization.

    Parameters
    ----------
    graphml_path : str
        Path to the full GraphML file.
    bounds : dict
        Keys: min_lat, max_lat, min_lng, max_lng.
    margin_deg : float
        Extra margin in degrees around the bounds.

    Returns
    -------
    nx.MultiDiGraph
        Subgraph containing only nodes within the bounds.
    """
    G = ox.load_graphml(graphml_path)
    min_lat = bounds["min_lat"] - margin_deg
    max_lat = bounds["max_lat"] + margin_deg
    min_lng = bounds["min_lng"] - margin_deg
    max_lng = bounds["max_lng"] + margin_deg

    nodes_in_bounds = [
        n
        for n, d in G.nodes(data=True)
        if min_lat <= d["y"] <= max_lat and min_lng <= d["x"] <= max_lng
    ]
    return G.subgraph(nodes_in_bounds).copy()
