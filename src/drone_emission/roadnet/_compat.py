"""Compatibility bridge for visualization that requires NetworkX graphs.

Loads NetworkX graphs on demand and discards after use, keeping the
main routing path memory-efficient via SparseGraph.
"""

import networkx as nx
import osmnx as ox


def load_networkx_graph(graphml_path: str) -> nx.MultiDiGraph:
    """Load a full NetworkX graph from GraphML for visualization."""
    return ox.load_graphml(graphml_path)
