"""Tile-based batch routing using scipy sparse Dijkstra.

Divides the road network into geographic tiles and routes OD pairs
within local subgraphs for memory-efficient, fast computation.
"""

import time
from collections import defaultdict

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

from .sparse_graph import SparseGraph

# Maximum air distance (meters) for OD pairs in this project
_MAX_AIR_DISTANCE_M = 3000
# Typical detour ratio upper bound
_MAX_DETOUR_RATIO = 1.8
# Dijkstra distance limit (meters): air distance * detour ratio, with margin
_DIJKSTRA_LIMIT = _MAX_AIR_DISTANCE_M * _MAX_DETOUR_RATIO + 2000  # ~7400m
# Max sources per Dijkstra batch to bound memory (~200MB per batch)
_MAX_SOURCES_PER_BATCH = 500


class TileRouter:
    """Tile-based router using scipy sparse Dijkstra on local subgraphs.

    The road network is divided into tiles of `tile_size_km` x `tile_size_km`.
    For each OD query, a 3x3 block of tiles around the origin is used as
    the local subgraph, providing ~6km x 6km coverage.

    Parameters
    ----------
    graph : SparseGraph
        The sparse road network.
    tile_size_km : float
        Tile edge length in kilometers.
    """

    def __init__(self, graph: SparseGraph, tile_size_km: float = 2.0):
        self._graph = graph
        self._tile_size_km = tile_size_km

        # Convert tile size to approximate degrees
        coords = graph._node_coords
        mid_lat = np.mean(coords[:, 0])
        # 1 degree latitude ~ 111 km
        self._tile_lat = tile_size_km / 111.0
        # 1 degree longitude ~ 111 * cos(lat) km
        self._tile_lng = tile_size_km / (111.0 * np.cos(np.radians(mid_lat)))

        # Assign each node to a tile
        self._node_tiles = self._compute_node_tiles(coords)
        # Build tile -> node indices mapping
        self._tile_to_nodes = self._build_tile_index()

        n_tiles = len(self._tile_to_nodes)
        avg_nodes = graph.num_nodes / max(n_tiles, 1)
        print(
            f"TileRouter: {n_tiles} tiles ({tile_size_km}km), "
            f"avg {avg_nodes:.0f} nodes/tile"
        )

    def _compute_node_tiles(self, coords: np.ndarray) -> np.ndarray:
        """Compute (tile_row, tile_col) for each node."""
        lat_min = coords[:, 0].min()
        lng_min = coords[:, 1].min()
        tile_rows = ((coords[:, 0] - lat_min) / self._tile_lat).astype(np.int32)
        tile_cols = ((coords[:, 1] - lng_min) / self._tile_lng).astype(np.int32)
        return np.column_stack([tile_rows, tile_cols])

    def _build_tile_index(self) -> dict:
        """Build mapping from tile (row, col) -> array of node indices."""
        tile_to_nodes = defaultdict(list)
        for idx in range(len(self._node_tiles)):
            key = (int(self._node_tiles[idx, 0]), int(self._node_tiles[idx, 1]))
            tile_to_nodes[key].append(idx)
        # Convert to numpy arrays
        return {k: np.array(v, dtype=np.intp) for k, v in tile_to_nodes.items()}

    def _get_expanded_tile_nodes(self, tile_row: int, tile_col: int) -> np.ndarray:
        """Get all node indices in the 3x3 block centered on (tile_row, tile_col)."""
        nodes = []
        for dr in range(-1, 2):
            for dc in range(-1, 2):
                key = (tile_row + dr, tile_col + dc)
                if key in self._tile_to_nodes:
                    nodes.append(self._tile_to_nodes[key])
        if not nodes:
            return np.array([], dtype=np.intp)
        return np.concatenate(nodes)

    def _extract_submatrix(self, node_indices: np.ndarray) -> csr_matrix:
        """Extract the CSR submatrix for the given node indices."""
        mat = self._graph._dist_matrix
        return mat[node_indices][:, node_indices]

    def compute_od_distances_batch(
        self,
        origins: np.ndarray,
        destinations: np.ndarray,
        show_progress: bool = True,
    ) -> np.ndarray:
        """Compute road distances for arrays of origin/destination coordinates.

        Parameters
        ----------
        origins : ndarray, shape (N, 2)
            Origin coordinates as (lat, lng).
        destinations : ndarray, shape (N, 2)
            Destination coordinates as (lat, lng).
        show_progress : bool
            Print progress information.

        Returns
        -------
        distances : ndarray, shape (N,)
            Road distances in meters. -1.0 for unreachable pairs.
        """
        n_pairs = len(origins)
        if show_progress:
            print(f"Computing road distances for {n_pairs:,} OD pairs...")

        start_time = time.time()

        # Step 1: Snap to nearest nodes
        if show_progress:
            print("  Snapping to nearest nodes...")
        origin_indices = self._graph.find_nearest_nodes_batch(origins)
        dest_indices = self._graph.find_nearest_nodes_batch(destinations)

        # Step 2: Group OD pairs by origin tile
        if show_progress:
            print("  Grouping OD pairs by tile...")
        tile_groups = defaultdict(list)
        for i in range(n_pairs):
            o_idx = origin_indices[i]
            tile_key = (
                int(self._node_tiles[o_idx, 0]),
                int(self._node_tiles[o_idx, 1]),
            )
            tile_groups[tile_key].append(i)

        # Step 3: Route within each tile group
        distances = np.full(n_pairs, -1.0, dtype=np.float64)
        n_tiles_total = len(tile_groups)
        n_tiles_done = 0
        n_resolved = 0

        for tile_key, pair_indices in tile_groups.items():
            pair_indices = np.array(pair_indices, dtype=np.intp)

            # Get expanded tile nodes (3x3 block)
            tile_nodes = self._get_expanded_tile_nodes(*tile_key)
            if len(tile_nodes) == 0:
                n_tiles_done += 1
                continue

            # Build local index mapping: global_idx -> local_idx
            global_to_local = np.full(
                self._graph.num_nodes, -1, dtype=np.intp
            )
            global_to_local[tile_nodes] = np.arange(len(tile_nodes))

            # Map OD pairs to local indices
            local_origins = global_to_local[origin_indices[pair_indices]]
            local_dests = global_to_local[dest_indices[pair_indices]]

            # Check which destinations are within this tile block
            in_tile = local_dests >= 0
            if in_tile.sum() == 0:
                n_tiles_done += 1
                continue

            # Extract submatrix
            sub_matrix = self._extract_submatrix(tile_nodes)

            # Get unique source nodes in this tile
            active_pairs = pair_indices[in_tile]
            active_local_origins = local_origins[in_tile]
            active_local_dests = local_dests[in_tile]

            unique_sources = np.unique(active_local_origins)

            # Batch sources to limit memory
            for batch_start in range(0, len(unique_sources), _MAX_SOURCES_PER_BATCH):
                batch_sources = unique_sources[
                    batch_start : batch_start + _MAX_SOURCES_PER_BATCH
                ]

                # Run Dijkstra for this batch of sources
                dist_result = dijkstra(
                    sub_matrix,
                    directed=True,
                    indices=batch_sources,
                    limit=_DIJKSTRA_LIMIT,
                )
                # dist_result shape: (len(batch_sources), len(tile_nodes))

                # Map batch source local indices to rows in dist_result
                source_to_row = {s: r for r, s in enumerate(batch_sources)}

                # Extract distances for pairs whose origin is in this batch
                for j in range(len(active_pairs)):
                    src_local = active_local_origins[j]
                    if src_local not in source_to_row:
                        continue
                    row = source_to_row[src_local]
                    dst_local = active_local_dests[j]
                    d = dist_result[row, dst_local]
                    if np.isfinite(d):
                        distances[active_pairs[j]] = d
                        n_resolved += 1

            n_tiles_done += 1
            if show_progress and (n_tiles_done % 50 == 0 or n_tiles_done == n_tiles_total):
                elapsed = time.time() - start_time
                print(
                    f"  Tiles: {n_tiles_done}/{n_tiles_total}, "
                    f"resolved: {n_resolved:,}/{n_pairs:,}, "
                    f"elapsed: {elapsed:.1f}s"
                )

        # Step 4: Handle unresolved pairs (destination outside 3x3 tile block)
        unresolved_mask = distances < 0
        # Some pairs are legitimately unreachable (disconnected components).
        # Try cross-tile routing for pairs where origin != destination node.
        same_node = origin_indices == dest_indices
        distances[same_node & unresolved_mask] = 0.0

        unresolved_cross = unresolved_mask & ~same_node
        n_unresolved = unresolved_cross.sum()

        if n_unresolved > 0 and show_progress:
            print(
                f"  Unresolved after tile routing: {n_unresolved:,} pairs "
                f"({100 * n_unresolved / n_pairs:.1f}%)"
            )

        if n_unresolved > 0:
            # For unresolved pairs, use merged neighborhood routing
            self._resolve_cross_tile(
                origin_indices,
                dest_indices,
                unresolved_cross,
                distances,
                show_progress,
            )

        total_time = time.time() - start_time
        valid = distances >= 0
        if show_progress:
            print(
                f"  Done: {valid.sum():,}/{n_pairs:,} valid paths "
                f"({100 * valid.sum() / n_pairs:.1f}%) in {total_time:.1f}s"
            )

        return distances

    def _resolve_cross_tile(
        self,
        origin_indices: np.ndarray,
        dest_indices: np.ndarray,
        mask: np.ndarray,
        distances: np.ndarray,
        show_progress: bool,
    ):
        """Resolve unresolved pairs by batching on (origin_tile, dest_tile) groups."""
        indices = np.where(mask)[0]
        if len(indices) == 0:
            return

        if show_progress:
            print(f"  Cross-tile fallback for {len(indices):,} pairs...")

        # Group unresolved pairs by (origin_tile, dest_tile)
        tile_pair_groups = defaultdict(list)
        for i in indices:
            o_idx = origin_indices[i]
            d_idx = dest_indices[i]
            o_tile = (int(self._node_tiles[o_idx, 0]), int(self._node_tiles[o_idx, 1]))
            d_tile = (int(self._node_tiles[d_idx, 0]), int(self._node_tiles[d_idx, 1]))
            tile_pair_groups[(o_tile, d_tile)].append(i)

        resolved = 0
        n_groups = len(tile_pair_groups)
        n_done = 0

        for (o_tile, d_tile), pair_indices in tile_pair_groups.items():
            pair_indices = np.array(pair_indices, dtype=np.intp)

            # Merge 3x3 neighborhoods of both tiles
            merged_nodes_set = set()
            for tile_key in [o_tile, d_tile]:
                for dr in range(-1, 2):
                    for dc in range(-1, 2):
                        key = (tile_key[0] + dr, tile_key[1] + dc)
                        if key in self._tile_to_nodes:
                            merged_nodes_set.update(self._tile_to_nodes[key].tolist())

            merged_nodes = np.array(sorted(merged_nodes_set), dtype=np.intp)
            if len(merged_nodes) == 0:
                n_done += 1
                continue

            global_to_local = np.full(self._graph.num_nodes, -1, dtype=np.intp)
            global_to_local[merged_nodes] = np.arange(len(merged_nodes))

            local_origins = global_to_local[origin_indices[pair_indices]]
            local_dests = global_to_local[dest_indices[pair_indices]]

            # Filter to pairs where both endpoints are in the subgraph
            valid = (local_origins >= 0) & (local_dests >= 0)
            if valid.sum() == 0:
                n_done += 1
                continue

            active_pairs = pair_indices[valid]
            active_local_origins = local_origins[valid]
            active_local_dests = local_dests[valid]

            sub_matrix = self._extract_submatrix(merged_nodes)

            unique_sources = np.unique(active_local_origins)
            for batch_start in range(0, len(unique_sources), _MAX_SOURCES_PER_BATCH):
                batch_sources = unique_sources[
                    batch_start : batch_start + _MAX_SOURCES_PER_BATCH
                ]
                dist_result = dijkstra(
                    sub_matrix,
                    directed=True,
                    indices=batch_sources,
                    limit=_DIJKSTRA_LIMIT,
                )
                source_to_row = {s: r for r, s in enumerate(batch_sources)}
                for j in range(len(active_pairs)):
                    src_local = active_local_origins[j]
                    if src_local not in source_to_row:
                        continue
                    row = source_to_row[src_local]
                    dst_local = active_local_dests[j]
                    d = dist_result[row, dst_local]
                    if np.isfinite(d):
                        distances[active_pairs[j]] = d
                        resolved += 1

            n_done += 1
            if show_progress and (n_done % 20 == 0 or n_done == n_groups):
                print(
                    f"  Cross-tile groups: {n_done}/{n_groups}, "
                    f"resolved: {resolved}/{len(indices)}"
                )

        if show_progress:
            print(f"  Cross-tile resolved: {resolved}/{len(indices)}")

    def find_shortest_path(
        self, origin: tuple, destination: tuple
    ) -> dict:
        """Find shortest path between two (lat, lng) points.

        Returns a dict compatible with the old RoadNetRouter format:
        path_nodes (OSM IDs), path_coordinates, total_distance, success, etc.
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

        o_idx = self._graph.find_nearest_node(origin[0], origin[1])
        d_idx = self._graph.find_nearest_node(destination[0], destination[1])

        result["origin_node"] = int(self._graph._idx_to_osmid[o_idx])
        result["destination_node"] = int(self._graph._idx_to_osmid[d_idx])

        # Compute snap distances (approximate meters)
        o_coord = self._graph._node_coords[o_idx]
        d_coord = self._graph._node_coords[d_idx]
        result["origin_snap_distance"] = _haversine_approx(
            origin[0], origin[1], o_coord[0], o_coord[1]
        )
        result["destination_snap_distance"] = _haversine_approx(
            destination[0], destination[1], d_coord[0], d_coord[1]
        )

        if o_idx == d_idx:
            result["success"] = True
            result["message"] = "Origin and destination are at the same intersection"
            result["path_nodes"] = [result["origin_node"]]
            result["path_coordinates"] = [(o_coord[0], o_coord[1])]
            return result

        # Build local subgraph covering both points
        o_tile = (int(self._node_tiles[o_idx, 0]), int(self._node_tiles[o_idx, 1]))
        d_tile = (int(self._node_tiles[d_idx, 0]), int(self._node_tiles[d_idx, 1]))

        merged_nodes_set = set()
        for tile_key in [o_tile, d_tile]:
            for dr in range(-2, 3):
                for dc in range(-2, 3):
                    key = (tile_key[0] + dr, tile_key[1] + dc)
                    if key in self._tile_to_nodes:
                        merged_nodes_set.update(self._tile_to_nodes[key].tolist())

        merged_nodes = np.array(sorted(merged_nodes_set), dtype=np.intp)
        global_to_local = np.full(self._graph.num_nodes, -1, dtype=np.intp)
        global_to_local[merged_nodes] = np.arange(len(merged_nodes))

        local_o = global_to_local[o_idx]
        local_d = global_to_local[d_idx]
        if local_o < 0 or local_d < 0:
            result["message"] = "Origin or destination not in subgraph"
            return result

        sub_matrix = self._extract_submatrix(merged_nodes)

        # Run Dijkstra with predecessor tracking
        dist_result, predecessors = dijkstra(
            sub_matrix,
            directed=True,
            indices=np.array([local_o]),
            limit=_DIJKSTRA_LIMIT,
            return_predecessors=True,
        )

        d = dist_result[0, local_d]
        if not np.isfinite(d):
            result["message"] = "No path found between origin and destination"
            return result

        # Reconstruct path by walking predecessors
        path_local = []
        current = local_d
        while current != local_o:
            path_local.append(current)
            pred = predecessors[0, current]
            if pred < 0:
                result["message"] = "Path reconstruction failed"
                return result
            current = pred
        path_local.append(local_o)
        path_local.reverse()

        # Convert to global indices -> OSM IDs and coordinates
        path_global = merged_nodes[path_local]
        path_osmids = self._graph._idx_to_osmid[path_global].tolist()
        path_coords = [
            (float(self._graph._node_coords[g, 0]), float(self._graph._node_coords[g, 1]))
            for g in path_global
        ]

        result["path_nodes"] = path_osmids
        result["path_coordinates"] = path_coords
        result["total_distance"] = float(d)
        result["success"] = True
        result["message"] = "Path found successfully"

        return result


def _haversine_approx(lat1, lng1, lat2, lng2):
    """Fast approximate haversine distance in meters."""
    dlat = np.radians(lat2 - lat1)
    dlng = np.radians(lng2 - lng1)
    avg_lat = np.radians((lat1 + lat2) / 2)
    x = dlng * np.cos(avg_lat) * 6371000
    y = dlat * 6371000
    return float(np.sqrt(x * x + y * y))
