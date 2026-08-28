"""Memory-efficient sparse graph representation for road network routing.

Converts a GraphML road network into a scipy CSR sparse matrix with
KD-tree spatial indexing. The NetworkX graph is loaded temporarily
during construction and then freed.
"""

import os

import numpy as np
from lxml import etree as ET
from scipy.sparse import csr_matrix
from scipy.spatial import cKDTree


def _clear_element(element) -> None:
    """Release parsed XML nodes and already-consumed siblings."""
    element.clear()
    parent = element.getparent()
    if parent is not None:
        while element.getprevious() is not None:
            del parent[0]


class SparseGraph:
    """CSR sparse matrix + KD-tree representation of a road network.

    Attributes
    ----------
    _dist_matrix : csr_matrix
        Edge weights in meters (float32). Shape (N, N).
    _node_coords : ndarray
        Node coordinates as (lat, lng), shape (N, 2), float64.
    _kdtree : cKDTree
        Spatial index over _node_coords.
    _idx_to_osmid : ndarray
        Maps matrix index -> OSM node ID (int64).
    _osmid_to_idx : dict
        Maps OSM node ID -> matrix index.
    """

    def __init__(
        self,
        dist_matrix: csr_matrix,
        node_coords: np.ndarray,
        idx_to_osmid: np.ndarray,
        osmid_to_idx: dict,
    ):
        self._dist_matrix = dist_matrix
        self._node_coords = node_coords
        self._kdtree = cKDTree(node_coords)
        self._idx_to_osmid = idx_to_osmid
        self._osmid_to_idx = osmid_to_idx

    @classmethod
    def from_graphml(cls, path: str) -> "SparseGraph":
        """Stream GraphML into CSR without materializing a NetworkX graph."""
        print(f"Loading GraphML: {path}")
        key_names = {}
        node_count = 0
        edge_count = 0
        for _, element in ET.iterparse(path, events=("end",)):
            tag = element.tag.rsplit("}", 1)[-1]
            if tag == "key":
                key_names[element.attrib["id"]] = element.attrib.get("attr.name")
            elif tag == "node":
                node_count += 1
            elif tag == "edge":
                edge_count += 1
            _clear_element(element)
        x_key = next((key for key, name in key_names.items() if name == "x"), None)
        y_key = next((key for key, name in key_names.items() if name == "y"), None)
        length_key = next(
            (key for key, name in key_names.items() if name == "length"), None
        )
        if x_key is None or y_key is None or length_key is None:
            raise ValueError("GraphML is missing x, y, or length key definitions")

        node_ids = np.empty(node_count, dtype=np.int64)
        coordinates = np.empty((node_count, 2), dtype=np.float64)
        offset = 0
        for _, element in ET.iterparse(path, events=("end",)):
            tag = element.tag.rsplit("}", 1)[-1]
            if tag == "data":
                continue
            if tag != "node":
                _clear_element(element)
                continue
            data = {
                child.attrib.get("key"): child.text
                for child in element
                if child.tag.rsplit("}", 1)[-1] == "data"
            }
            node_ids[offset] = int(element.attrib["id"])
            coordinates[offset] = (float(data[y_key]), float(data[x_key]))
            offset += 1
            _clear_element(element)
        order = np.argsort(node_ids, kind="stable")
        idx_to_osmid = node_ids[order]
        node_coords = coordinates[order]
        osmid_to_idx = {
            int(osmid): index for index, osmid in enumerate(idx_to_osmid)
        }

        rows = np.empty(edge_count, dtype=np.int64)
        cols = np.empty(edge_count, dtype=np.int64)
        weights = np.empty(edge_count, dtype=np.float32)
        offset = 0
        for _, element in ET.iterparse(path, events=("end",)):
            tag = element.tag.rsplit("}", 1)[-1]
            if tag == "data":
                continue
            if tag != "edge":
                _clear_element(element)
                continue
            length = next(
                (
                    child.text
                    for child in element
                    if child.tag.rsplit("}", 1)[-1] == "data"
                    and child.attrib.get("key") == length_key
                ),
                "0",
            )
            rows[offset] = osmid_to_idx[int(element.attrib["source"])]
            cols[offset] = osmid_to_idx[int(element.attrib["target"])]
            weights[offset] = max(float(length), 0.001)
            offset += 1
            _clear_element(element)

        if edge_count:
            edge_order = np.lexsort((cols, rows))
            sorted_rows = rows[edge_order]
            sorted_cols = cols[edge_order]
            sorted_weights = weights[edge_order]
            starts = np.r_[
                0,
                np.flatnonzero(
                    (sorted_rows[1:] != sorted_rows[:-1])
                    | (sorted_cols[1:] != sorted_cols[:-1])
                )
                + 1,
            ]
            rows = sorted_rows[starts]
            cols = sorted_cols[starts]
            weights = np.minimum.reduceat(sorted_weights, starts)
        dist_matrix = csr_matrix(
            (weights, (rows, cols)), shape=(node_count, node_count)
        )

        graph = cls(dist_matrix, node_coords, idx_to_osmid, osmid_to_idx)

        print(
            f"  Streamed GraphML: {node_count:,} nodes, {edge_count:,} raw edges"
        )
        print(f"  SparseGraph: {graph.num_nodes:,} nodes, {graph.num_edges:,} edges")
        print(f"  Total road length: {graph.total_length / 1000:.1f} km")
        print(
            f"  Matrix memory: {dist_matrix.data.nbytes / 1e6:.1f} MB"
            f" + coords: {node_coords.nbytes / 1e6:.1f} MB"
        )
        return graph

    @classmethod
    def from_npz(cls, path: str) -> "SparseGraph":
        """Load a previously serialized sparse routing graph."""
        with np.load(path, allow_pickle=False) as payload:
            shape = tuple(int(value) for value in payload["shape"])
            dist_matrix = csr_matrix(
                (
                    payload["data"],
                    payload["indices"],
                    payload["indptr"],
                ),
                shape=shape,
            )
            node_coords = payload["node_coords"]
            idx_to_osmid = payload["idx_to_osmid"]
        osmid_to_idx = {
            int(osmid): index for index, osmid in enumerate(idx_to_osmid)
        }
        return cls(dist_matrix, node_coords, idx_to_osmid, osmid_to_idx)

    def to_npz(self, path: str) -> None:
        """Atomically serialize the arrays required for routing and plotting."""
        temporary = f"{path}.tmp"
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(temporary, "wb") as handle:
            np.savez(
                handle,
                data=self._dist_matrix.data,
                indices=self._dist_matrix.indices,
                indptr=self._dist_matrix.indptr,
                shape=np.asarray(self._dist_matrix.shape, dtype=np.int64),
                node_coords=self._node_coords,
                idx_to_osmid=self._idx_to_osmid,
            )
        os.replace(temporary, path)

    def find_nearest_node(self, lat: float, lng: float) -> int:
        """Find nearest graph node. Returns matrix index."""
        _, idx = self._kdtree.query([lat, lng])
        return int(idx)

    def find_nearest_nodes_batch(self, points: np.ndarray) -> np.ndarray:
        """Find nearest graph nodes for an array of (lat, lng) points.

        Parameters
        ----------
        points : ndarray, shape (M, 2)
            Each row is (lat, lng).

        Returns
        -------
        indices : ndarray of int, shape (M,)
            Matrix indices of nearest nodes.
        """
        _, indices = self._kdtree.query(points)
        return indices.astype(np.intp)

    def get_node_coords(self, idx: int) -> tuple:
        """Return (lat, lng) for a matrix index."""
        return float(self._node_coords[idx, 0]), float(self._node_coords[idx, 1])

    def get_node_osmid(self, idx: int) -> int:
        """Return OSM ID for a matrix index."""
        return int(self._idx_to_osmid[idx])

    def get_node_osmids_batch(self, indices: np.ndarray) -> np.ndarray:
        """Return OSM IDs for an array of matrix indices."""
        return self._idx_to_osmid[indices]

    @property
    def num_nodes(self) -> int:
        return self._dist_matrix.shape[0]

    @property
    def num_edges(self) -> int:
        return self._dist_matrix.nnz

    def get_local_edges(self, bounds: dict, margin_deg: float = 0.002):
        """Return edge endpoint coordinates within a geographic bounding box.

        Parameters
        ----------
        bounds : dict
            Keys: min_lat, max_lat, min_lng, max_lng.
        margin_deg : float
            Extra margin in degrees around the bounds.

        Returns
        -------
        edge_coords : ndarray, shape (M, 4)
            Each row is (from_lat, from_lng, to_lat, to_lng).
        """
        min_lat = bounds["min_lat"] - margin_deg
        max_lat = bounds["max_lat"] + margin_deg
        min_lng = bounds["min_lng"] - margin_deg
        max_lng = bounds["max_lng"] + margin_deg

        # Find nodes within bounds
        in_bounds = (
            (self._node_coords[:, 0] >= min_lat)
            & (self._node_coords[:, 0] <= max_lat)
            & (self._node_coords[:, 1] >= min_lng)
            & (self._node_coords[:, 1] <= max_lng)
        )
        local_indices = np.where(in_bounds)[0]
        if len(local_indices) == 0:
            return np.empty((0, 4), dtype=np.float64)

        # Extract edges where source is in local area
        mat = self._dist_matrix
        rows_list, cols_list = [], []
        for idx in local_indices:
            start, end = mat.indptr[idx], mat.indptr[idx + 1]
            cols = mat.indices[start:end]
            # Keep only edges where dest is also in bounds
            dest_in = in_bounds[cols]
            rows_list.append(np.full(dest_in.sum(), idx, dtype=np.intp))
            cols_list.append(cols[dest_in])

        if not rows_list:
            return np.empty((0, 4), dtype=np.float64)

        all_rows = np.concatenate(rows_list)
        all_cols = np.concatenate(cols_list)

        edge_coords = np.column_stack([
            self._node_coords[all_rows, 0],
            self._node_coords[all_rows, 1],
            self._node_coords[all_cols, 0],
            self._node_coords[all_cols, 1],
        ])
        return edge_coords

    @property
    def total_length(self) -> float:
        """Total road length in meters."""
        return float(self._dist_matrix.sum())
