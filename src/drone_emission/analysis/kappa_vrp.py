"""VRP experiment for the limiting e-bike bundling distance factor.

The experiment intentionally keeps the production emission pipeline unchanged.
It estimates a separate distance ratio from same-restaurant, closed e-bike
routes on a directed road graph.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import time
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.sparse.csgraph import dijkstra
from sklearn.cluster import KMeans

from roadnet.sparse_graph import SparseGraph


OD_COLUMNS = (
    "from_lat",
    "from_lng",
    "to_lat",
    "to_lng",
    "annual_sales",
    "road_distance_m",
)


@dataclass(frozen=True)
class KappaVrpSettings:
    """Configuration for the standalone kappa VRP experiment."""

    region: str = "changsha"
    level: str = "city"
    window_km: float = 2.0
    n_restaurants: int = 15
    destination_clusters: int = 24
    horizon_hours: float = 2.0
    operating_days_per_year: int = 300
    lunch_order_share: float = 0.30
    controlled_rates_per_hour: tuple[float, ...] = (5, 10, 20, 40, 60)
    matching_windows_min: tuple[float, ...] = (3, 6, 12)
    preparation_time_min: float = 10.0
    sla_values_min: tuple[float, ...] = (30, 45)
    speed_values_kmh: tuple[float, ...] = (20, 30)
    pickup_service_min: float = 1.0
    drop_service_values_min: tuple[float, ...] = (1, 2, 3)
    max_bundle_size: int = 12
    plateau_improvement_pct: float = 1.0
    plateau_consecutive_steps: int = 2
    primary_replicates: int = 20
    sensitivity_replicates: int = 10
    geometric_replicates: int = 5
    random_seed: int = 42
    exact_batch_limit: int = 12
    solver_time_limit_ms: int = 250
    configured_kappa: float = 1 / math.sqrt(1.69)

    @classmethod
    def from_mapping(cls, raw: dict) -> "KappaVrpSettings":
        """Build settings from the ``[kappa_vrp]`` TOML mapping."""
        tuple_fields = {
            "controlled_rates_per_hour",
            "matching_windows_min",
            "sla_values_min",
            "speed_values_kmh",
            "drop_service_values_min",
        }
        values = dict(raw)
        for field in tuple_fields:
            if field in values:
                values[field] = tuple(values[field])
        return cls(**values)


@dataclass(frozen=True)
class Scenario:
    """One demand and service-policy cell."""

    name: str
    rate_per_hour: float | None
    matching_window_min: float | None
    sla_min: float | None
    speed_kmh: float
    drop_service_min: float
    replicates: int
    constrained: bool = True


@dataclass(frozen=True)
class RouteOrder:
    """One simulated order in a restaurant dispatch batch."""

    order_id: int
    destination_index: int
    placed_min: float
    ready_min: float
    deadline_min: float


@dataclass
class BatchSolution:
    """Distance-minimum set of routes for one dispatch batch."""

    distance_m: float
    routes: list[list[int]]
    solver: str
    status: str
    solve_ms: float


def _stable_seed(base: int, *parts: object) -> int:
    text = "|".join([str(base), *(str(part) for part in parts)])
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") % (2**32)


def _file_fingerprint(path: Path) -> dict:
    stat = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        digest.update(handle.read(1024 * 1024))
    return {
        "path": str(path),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "first_mb_sha256": digest.hexdigest(),
    }


def _git_sha(project_root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            text=True,
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def _local_xy_km(lat: np.ndarray, lng: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lat0 = float(np.mean(lat))
    x = (lng - float(np.min(lng))) * 111.32 * math.cos(math.radians(lat0))
    y = (lat - float(np.min(lat))) * 110.574
    return x, y


def select_dense_restaurant_window(
    od_df: pd.DataFrame,
    window_km: float,
    n_restaurants: int,
) -> tuple[pd.DataFrame, dict]:
    """Select the deterministic highest-demand restaurant-origin window."""
    origins = (
        od_df.groupby(["from_lat", "from_lng"], as_index=False)["annual_sales"]
        .sum()
        .sort_values(["annual_sales", "from_lat", "from_lng"], ascending=[False, True, True])
        .reset_index(drop=True)
    )
    x, y = _local_xy_km(
        origins["from_lat"].to_numpy(),
        origins["from_lng"].to_numpy(),
    )
    demand = origins["annual_sales"].to_numpy(dtype=float)
    best = None
    for x0 in np.unique(x):
        x_mask = (x >= x0) & (x <= x0 + window_km)
        if not x_mask.any():
            continue
        for y0 in np.unique(y[x_mask]):
            mask = x_mask & (y >= y0) & (y <= y0 + window_km)
            score = float(demand[mask].sum())
            candidate = (score, -float(x0), -float(y0), mask)
            if best is None or candidate[:3] > best[:3]:
                best = candidate
    if best is None:
        raise ValueError("No restaurant origins were available for window selection")

    selected = (
        origins.loc[best[3]]
        .sort_values(["annual_sales", "from_lat", "from_lng"], ascending=[False, True, True])
        .head(n_restaurants)
        .reset_index(drop=True)
    )
    if selected.empty:
        raise ValueError("The selected restaurant window contains no restaurants")
    selected["restaurant_id"] = np.arange(len(selected), dtype=int)
    meta = {
        "window_km": float(window_km),
        "window_demand": float(best[0]),
        "selected_restaurants": int(len(selected)),
        "selected_annual_sales": float(selected["annual_sales"].sum()),
        "min_lat": float(selected["from_lat"].min()),
        "max_lat": float(selected["from_lat"].max()),
        "min_lng": float(selected["from_lng"].min()),
        "max_lng": float(selected["from_lng"].max()),
    }
    return selected, meta


def compress_destination_catchments(
    od_df: pd.DataFrame,
    restaurants: pd.DataFrame,
    n_clusters: int,
    seed: int,
) -> tuple[pd.DataFrame, dict]:
    """Compress full catchments into weighted representative destinations."""
    id_lookup = {
        (row.from_lat, row.from_lng): int(row.restaurant_id)
        for row in restaurants.itertuples()
    }
    subset = od_df[
        od_df.set_index(["from_lat", "from_lng"]).index.isin(id_lookup)
    ].copy()
    subset["restaurant_id"] = [
        id_lookup[(lat, lng)]
        for lat, lng in zip(subset["from_lat"], subset["from_lng"])
    ]

    records = []
    input_mass = 0.0
    represented_mass = 0.0
    for restaurant_id, group in subset.groupby("restaurant_id", sort=True):
        group = group[
            np.isfinite(group["road_distance_m"])
            & (group["road_distance_m"] >= 0)
            & np.isfinite(group["annual_sales"])
            & (group["annual_sales"] > 0)
        ].copy()
        if group.empty:
            continue
        input_mass += float(group["annual_sales"].sum())
        coords = group[["to_lat", "to_lng"]].to_numpy(dtype=float)
        weights = group["annual_sales"].to_numpy(dtype=float)
        k = min(int(n_clusters), len(group))
        if k == len(group):
            labels = np.arange(len(group), dtype=int)
            centres = coords
        else:
            model = KMeans(
                n_clusters=k,
                random_state=_stable_seed(seed, "cluster", restaurant_id),
                n_init=10,
            )
            labels = model.fit_predict(coords, sample_weight=weights)
            centres = model.cluster_centers_

        for cluster_id in range(k):
            positions = np.flatnonzero(labels == cluster_id)
            if positions.size == 0:
                continue
            centre = centres[cluster_id]
            delta = coords[positions] - centre
            nearest = positions[int(np.argmin(np.einsum("ij,ij->i", delta, delta)))]
            row = group.iloc[nearest]
            mass = float(weights[positions].sum())
            represented_mass += mass
            records.append(
                {
                    "restaurant_id": int(restaurant_id),
                    "destination_id": int(cluster_id),
                    "to_lat": float(row["to_lat"]),
                    "to_lng": float(row["to_lng"]),
                    "annual_sales": mass,
                    "source_od_count": int(positions.size),
                }
            )
    destinations = pd.DataFrame(records)
    if destinations.empty:
        raise ValueError("No valid representative destinations were created")
    meta = {
        "input_valid_annual_sales": input_mass,
        "represented_annual_sales": represented_mass,
        "represented_share_pct": (
            represented_mass / input_mass * 100.0 if input_mass else 0.0
        ),
        "representative_destinations": int(len(destinations)),
        "clusters_per_restaurant": int(n_clusters),
    }
    return destinations, meta


def build_directed_distance_blocks(
    graphml_path: Path,
    restaurants: pd.DataFrame,
    destinations: pd.DataFrame,
    source_chunk: int = 48,
) -> tuple[dict[int, np.ndarray], dict]:
    """Build one directed restaurant/customer distance matrix per restaurant."""
    graph = SparseGraph.from_graphml(str(graphml_path))
    blocks = {}
    unreachable_pairs = 0
    total_pairs = 0
    for restaurant in restaurants.itertuples():
        rid = int(restaurant.restaurant_id)
        dest = (
            destinations[destinations["restaurant_id"] == rid]
            .sort_values("destination_id")
            .reset_index(drop=True)
        )
        if dest.empty:
            continue
        coords = np.vstack(
            [
                np.array([[restaurant.from_lat, restaurant.from_lng]], dtype=float),
                dest[["to_lat", "to_lng"]].to_numpy(dtype=float),
            ]
        )
        nodes = graph.find_nearest_nodes_batch(coords)
        matrix = np.full((len(nodes), len(nodes)), np.inf, dtype=float)
        for start in range(0, len(nodes), source_chunk):
            stop = min(start + source_chunk, len(nodes))
            result = dijkstra(
                graph._dist_matrix,
                directed=True,
                indices=nodes[start:stop],
            )
            matrix[start:stop, :] = result[:, nodes]
        np.fill_diagonal(matrix, 0.0)
        total_pairs += int(matrix.size - len(matrix))
        unreachable_pairs += int((~np.isfinite(matrix)).sum())
        blocks[rid] = matrix
    meta = {
        "graph_nodes": int(graph.num_nodes),
        "graph_edges": int(graph.num_edges),
        "directed_pairs": total_pairs,
        "unreachable_pairs": unreachable_pairs,
    }
    return blocks, meta


def filter_unreachable_destinations(
    restaurants: pd.DataFrame,
    destinations: pd.DataFrame,
    blocks: dict[int, np.ndarray],
) -> tuple[pd.DataFrame, dict[int, np.ndarray], dict]:
    """Remove representatives without a finite standalone closed route."""
    kept_frames = []
    kept_blocks = {}
    removed_mass = 0.0
    total_mass = float(destinations["annual_sales"].sum())
    for restaurant in restaurants.itertuples():
        rid = int(restaurant.restaurant_id)
        block = blocks.get(rid)
        dest = (
            destinations[destinations["restaurant_id"] == rid]
            .sort_values("destination_id")
            .reset_index(drop=True)
        )
        if block is None or dest.empty:
            removed_mass += float(dest["annual_sales"].sum())
            continue
        keep = np.isfinite(block[0, 1:]) & np.isfinite(block[1:, 0])
        removed_mass += float(dest.loc[~keep, "annual_sales"].sum())
        kept = dest.loc[keep].copy().reset_index(drop=True)
        if kept.empty:
            continue
        indices = np.r_[0, np.flatnonzero(keep) + 1]
        kept["destination_id"] = np.arange(len(kept), dtype=int)
        kept_frames.append(kept)
        kept_blocks[rid] = block[np.ix_(indices, indices)]
    if not kept_frames:
        raise ValueError("No destination has a finite directed standalone cycle")
    result = pd.concat(kept_frames, ignore_index=True)
    meta = {
        "removed_unreachable_annual_sales": removed_mass,
        "retained_reachable_share_pct": (
            (total_mass - removed_mass) / total_mass * 100.0 if total_mass else 0.0
        ),
    }
    return result, kept_blocks, meta


def _generate_orders(
    destinations: pd.DataFrame,
    restaurant_sales: float,
    settings: KappaVrpSettings,
    scenario: Scenario,
    replicate: int,
    restaurant_id: int,
) -> list[RouteOrder]:
    group = (
        destinations[destinations["restaurant_id"] == restaurant_id]
        .sort_values("destination_id")
        .reset_index(drop=True)
    )
    if group.empty:
        return []
    if scenario.rate_per_hour is None:
        daily = restaurant_sales / settings.operating_days_per_year
        rate = daily * settings.lunch_order_share / settings.horizon_hours
        rate_key = "local"
    else:
        rate = float(scenario.rate_per_hour)
        rate_key = f"{rate:g}"
    rng = np.random.default_rng(
        _stable_seed(settings.random_seed, rate_key, replicate, restaurant_id)
    )
    count = int(rng.poisson(rate * settings.horizon_hours))
    if count == 0:
        return []
    placed = np.sort(rng.uniform(0.0, settings.horizon_hours * 60.0, count))
    probabilities = group["annual_sales"].to_numpy(dtype=float)
    probabilities /= probabilities.sum()
    sampled = rng.choice(len(group), size=count, replace=True, p=probabilities)
    orders = []
    for order_id, (placed_min, destination_index) in enumerate(
        zip(placed, sampled)
    ):
        ready = float(placed_min + settings.preparation_time_min)
        deadline = (
            float(placed_min + scenario.sla_min)
            if scenario.sla_min is not None
            else math.inf
        )
        orders.append(
            RouteOrder(
                order_id=order_id,
                destination_index=int(destination_index) + 1,
                placed_min=float(placed_min),
                ready_min=ready,
                deadline_min=deadline,
            )
        )
    return orders


def _dispatch_batches(
    orders: list[RouteOrder],
    matching_window_min: float | None,
) -> list[tuple[float, list[RouteOrder]]]:
    if not orders:
        return []
    if matching_window_min is None:
        return [(max(order.ready_min for order in orders), orders)]
    groups: dict[float, list[RouteOrder]] = {}
    for order in orders:
        dispatch = (
            math.ceil(order.ready_min / matching_window_min) * matching_window_min
        )
        groups.setdefault(float(dispatch), []).append(order)
    return sorted(groups.items())


def _enumerate_feasible_routes(
    orders: list[RouteOrder],
    matrix: np.ndarray,
    max_bundle: int,
    dispatch_min: float,
    speed_kmh: float,
    pickup_service_min: float,
    drop_service_min: float,
    constrained: bool,
) -> tuple[dict[int, float], dict[int, list[int]]]:
    n = len(orders)
    full = 1 << n
    state_cost: dict[tuple[int, int], float] = {}
    state_path: dict[tuple[int, int], tuple[int, ...]] = {}
    route_cost: dict[int, float] = {}
    route_path: dict[int, list[int]] = {}
    metres_per_min = speed_kmh * 1000.0 / 60.0
    departure = dispatch_min + pickup_service_min

    for j, order in enumerate(orders):
        distance = float(matrix[0, order.destination_index])
        arrival = departure + distance / metres_per_min
        if not math.isfinite(distance):
            continue
        if constrained and arrival > order.deadline_min + 1e-9:
            continue
        mask = 1 << j
        state_cost[(mask, j)] = distance
        state_path[(mask, j)] = (j,)

    for mask in range(1, full):
        size = mask.bit_count()
        if size > max_bundle:
            continue
        for last in range(n):
            key = (mask, last)
            if key not in state_cost:
                continue
            distance = state_cost[key]
            path = state_path[key]
            return_distance = matrix[orders[last].destination_index, 0]
            closed = distance + return_distance
            if math.isfinite(closed) and (
                mask not in route_cost or closed < route_cost[mask]
            ):
                route_cost[mask] = float(closed)
                route_path[mask] = list(path)
            if size == max_bundle:
                continue
            elapsed_before_next = (
                departure
                + distance / metres_per_min
                + size * drop_service_min
            )
            for nxt, order in enumerate(orders):
                bit = 1 << nxt
                if mask & bit:
                    continue
                leg = matrix[
                    orders[last].destination_index,
                    order.destination_index,
                ]
                if not math.isfinite(leg):
                    continue
                arrival = elapsed_before_next + leg / metres_per_min
                if constrained and arrival > order.deadline_min + 1e-9:
                    continue
                new_mask = mask | bit
                new_distance = distance + float(leg)
                new_key = (new_mask, nxt)
                if (
                    new_key not in state_cost
                    or new_distance < state_cost[new_key] - 1e-9
                ):
                    state_cost[new_key] = new_distance
                    state_path[new_key] = (*path, nxt)
    return route_cost, route_path


def solve_batch_exact(
    orders: list[RouteOrder],
    matrix: np.ndarray,
    max_bundle: int,
    dispatch_min: float,
    speed_kmh: float,
    pickup_service_min: float,
    drop_service_min: float,
    constrained: bool,
) -> BatchSolution:
    """Solve a small same-depot CVRPTW exactly by route/set partition DP."""
    started = time.perf_counter()
    route_cost, route_path = _enumerate_feasible_routes(
        orders,
        matrix,
        max_bundle,
        dispatch_min,
        speed_kmh,
        pickup_service_min,
        drop_service_min,
        constrained,
    )
    n = len(orders)
    full = (1 << n) - 1
    best = np.full(1 << n, np.inf)
    choice = np.zeros(1 << n, dtype=np.int64)
    best[0] = 0.0
    for covered in range(1 << n):
        if not math.isfinite(best[covered]) or covered == full:
            continue
        first = next(i for i in range(n) if not (covered & (1 << i)))
        remaining = full ^ covered
        submask = remaining
        while submask:
            if (
                submask & (1 << first)
                and submask.bit_count() <= max_bundle
                and submask in route_cost
            ):
                new = covered | submask
                candidate = best[covered] + route_cost[submask]
                if candidate < best[new] - 1e-9:
                    best[new] = candidate
                    choice[new] = submask
            submask = (submask - 1) & remaining
    if not math.isfinite(best[full]):
        return BatchSolution(
            distance_m=math.inf,
            routes=[],
            solver="exact_dp",
            status="infeasible",
            solve_ms=(time.perf_counter() - started) * 1000,
        )
    routes = []
    cursor = full
    while cursor:
        mask = int(choice[cursor])
        if mask == 0:
            raise RuntimeError("Exact route reconstruction failed")
        routes.append(route_path[mask])
        cursor ^= mask
    return BatchSolution(
        distance_m=float(best[full]),
        routes=routes,
        solver="exact_dp",
        status="optimal",
        solve_ms=(time.perf_counter() - started) * 1000,
    )


def solve_batch_ortools(
    orders: list[RouteOrder],
    matrix: np.ndarray,
    max_bundle: int,
    dispatch_min: float,
    speed_kmh: float,
    pickup_service_min: float,
    drop_service_min: float,
    constrained: bool,
    time_limit_ms: int,
) -> BatchSolution:
    """Solve a larger batch with the OR-Tools routing search."""
    from ortools.constraint_solver import pywrapcp, routing_enums_pb2

    started = time.perf_counter()
    n = len(orders)
    vehicles = n
    manager = pywrapcp.RoutingIndexManager(n + 1, vehicles, 0)
    routing = pywrapcp.RoutingModel(manager)
    scale = 100

    def distance_callback(from_index, to_index):
        src = manager.IndexToNode(from_index)
        dst = manager.IndexToNode(to_index)
        if src == 0:
            src_matrix = 0
        else:
            src_matrix = orders[src - 1].destination_index
        if dst == 0:
            dst_matrix = 0
        else:
            dst_matrix = orders[dst - 1].destination_index
        value = matrix[src_matrix, dst_matrix]
        if not math.isfinite(value):
            return 10**12
        return int(round(float(value) * scale))

    distance_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(distance_index)

    def demand_callback(index):
        return 0 if manager.IndexToNode(index) == 0 else 1

    demand_index = routing.RegisterUnaryTransitCallback(demand_callback)
    routing.AddDimensionWithVehicleCapacity(
        demand_index,
        0,
        [int(max_bundle)] * vehicles,
        True,
        "Capacity",
    )

    if constrained:
        metres_per_min = speed_kmh * 1000.0 / 60.0

        def time_callback(from_index, to_index):
            src = manager.IndexToNode(from_index)
            service = pickup_service_min if src == 0 else drop_service_min
            travel = distance_callback(from_index, to_index) / scale / metres_per_min
            return int(round((service + travel) * scale))

        time_index = routing.RegisterTransitCallback(time_callback)
        horizon = int(
            round(
                (
                    max(order.deadline_min for order in orders)
                    - dispatch_min
                    + 60.0
                )
                * scale
            )
        )
        routing.AddDimension(time_index, 0, horizon, True, "Time")
        time_dimension = routing.GetDimensionOrDie("Time")
        for i, order in enumerate(orders, start=1):
            upper = max(0, int(round((order.deadline_min - dispatch_min) * scale)))
            time_dimension.CumulVar(manager.NodeToIndex(i)).SetRange(0, upper)

    params = pywrapcp.DefaultRoutingSearchParameters()
    params.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PARALLEL_CHEAPEST_INSERTION
    )
    params.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    )
    params.time_limit.FromMilliseconds(int(time_limit_ms))
    solution = routing.SolveWithParameters(params)
    if solution is None:
        return BatchSolution(
            distance_m=math.inf,
            routes=[],
            solver="ortools",
            status=f"failed_{routing.status()}",
            solve_ms=(time.perf_counter() - started) * 1000,
        )

    routes = []
    total = 0.0
    for vehicle in range(vehicles):
        index = routing.Start(vehicle)
        route = []
        while not routing.IsEnd(index):
            node = manager.IndexToNode(index)
            next_index = solution.Value(routing.NextVar(index))
            total += distance_callback(index, next_index) / scale
            if node:
                route.append(node - 1)
            index = next_index
        if route:
            routes.append(route)
    status = "success" if routing.status() == 1 else f"status_{routing.status()}"
    return BatchSolution(
        distance_m=float(total),
        routes=routes,
        solver="ortools",
        status=status,
        solve_ms=(time.perf_counter() - started) * 1000,
    )


def solve_batch(
    orders: list[RouteOrder],
    matrix: np.ndarray,
    max_bundle: int,
    dispatch_min: float,
    scenario: Scenario,
    settings: KappaVrpSettings,
) -> BatchSolution:
    """Dispatch to the exact or scalable solver according to batch size."""
    if len(orders) <= settings.exact_batch_limit:
        return solve_batch_exact(
            orders=orders,
            matrix=matrix,
            max_bundle=max_bundle,
            dispatch_min=dispatch_min,
            speed_kmh=scenario.speed_kmh,
            pickup_service_min=settings.pickup_service_min,
            drop_service_min=scenario.drop_service_min,
            constrained=scenario.constrained,
        )
    return solve_batch_ortools(
        orders=orders,
        matrix=matrix,
        max_bundle=max_bundle,
        dispatch_min=dispatch_min,
        speed_kmh=scenario.speed_kmh,
        pickup_service_min=settings.pickup_service_min,
        drop_service_min=scenario.drop_service_min,
        constrained=scenario.constrained,
        time_limit_ms=settings.solver_time_limit_ms,
    )


def _standalone_distance(orders: Iterable[RouteOrder], matrix: np.ndarray) -> float:
    return float(
        sum(
            matrix[0, order.destination_index]
            + matrix[order.destination_index, 0]
            for order in orders
        )
    )


def _code_compatible_distance(
    orders: Iterable[RouteOrder],
    matrix: np.ndarray,
) -> float:
    return float(
        sum(2.0 * matrix[0, order.destination_index] for order in orders)
    )


def run_scenario_curve(
    scenario: Scenario,
    settings: KappaVrpSettings,
    restaurants: pd.DataFrame,
    destinations: pd.DataFrame,
    blocks: dict[int, np.ndarray],
    capacities: Iterable[int],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run all replications and capacities for one scenario."""
    result_rows = []
    route_rows = []
    restaurant_sales = restaurants.set_index("restaurant_id")["annual_sales"].to_dict()
    for replicate in range(scenario.replicates):
        generated = {}
        incumbents: dict[tuple[int, int], BatchSolution] = {}
        for restaurant in restaurants.itertuples():
            rid = int(restaurant.restaurant_id)
            if rid not in blocks:
                continue
            orders = _generate_orders(
                destinations,
                float(restaurant_sales[rid]),
                settings,
                scenario,
                replicate,
                rid,
            )
            generated[rid] = _dispatch_batches(
                orders,
                scenario.matching_window_min,
            )
        for capacity in capacities:
            numerator = 0.0
            denominator = 0.0
            code_denominator = 0.0
            orders_total = 0
            exact_orders = 0
            failed_orders = 0
            solve_ms = 0.0
            routes_total = 0
            for rid, batches in generated.items():
                matrix = blocks[rid]
                for batch_id, (dispatch, orders) in enumerate(batches):
                    if not orders:
                        continue
                    standalone = _standalone_distance(orders, matrix)
                    code_standalone = _code_compatible_distance(orders, matrix)
                    solution = solve_batch(
                        orders,
                        matrix,
                        int(capacity),
                        dispatch,
                        scenario,
                        settings,
                    )
                    incumbent_key = (rid, batch_id)
                    incumbent = incumbents.get(incumbent_key)
                    if (
                        incumbent is not None
                        and incumbent.distance_m < solution.distance_m
                    ):
                        solution = BatchSolution(
                            distance_m=incumbent.distance_m,
                            routes=incumbent.routes,
                            solver=f"{incumbent.solver}_incumbent",
                            status="reused_from_lower_capacity",
                            solve_ms=solution.solve_ms,
                        )
                    if not math.isfinite(solution.distance_m):
                        # Preserve every order in the accounting even when the
                        # constrained search times out or the SLA is impossible.
                        # The standalone fallback is explicitly counted below.
                        failed_orders += len(orders)
                        solution = BatchSolution(
                            distance_m=standalone,
                            routes=[[i] for i in range(len(orders))],
                            solver="standalone_fallback",
                            status="served_without_sla_certificate",
                            solve_ms=solution.solve_ms,
                        )
                    incumbents[incumbent_key] = solution
                    numerator += solution.distance_m
                    denominator += standalone
                    code_denominator += code_standalone
                    orders_total += len(orders)
                    solve_ms += solution.solve_ms
                    routes_total += len(solution.routes)
                    if solution.solver.startswith("exact_dp"):
                        exact_orders += len(orders)
                    route_rows.append(
                        {
                            "scenario": scenario.name,
                            "replicate": replicate,
                            "capacity": int(capacity),
                            "restaurant_id": rid,
                            "batch_id": batch_id,
                            "dispatch_min": dispatch,
                            "orders": len(orders),
                            "routes": len(solution.routes),
                            "route_distance_m": solution.distance_m,
                            "standalone_distance_m": standalone,
                            "solver": solution.solver,
                            "solver_status": solution.status,
                            "solve_ms": solution.solve_ms,
                        }
                    )
            result_rows.append(
                {
                    "scenario": scenario.name,
                    "rate_per_hour": scenario.rate_per_hour,
                    "matching_window_min": scenario.matching_window_min,
                    "sla_min": scenario.sla_min,
                    "speed_kmh": scenario.speed_kmh,
                    "drop_service_min": scenario.drop_service_min,
                    "replicates": scenario.replicates,
                    "constrained": scenario.constrained,
                    "replicate": replicate,
                    "capacity": int(capacity),
                    "route_distance_m": numerator,
                    "standalone_distance_m": denominator,
                    "code_standalone_distance_m": code_denominator,
                    "kappa": numerator / denominator if denominator else np.nan,
                    "code_compatible_kappa": (
                        numerator / code_denominator if code_denominator else np.nan
                    ),
                    "orders": orders_total,
                    "failed_orders": failed_orders,
                    "routes": routes_total,
                    "mean_bundle_size": (
                        orders_total / routes_total if routes_total else np.nan
                    ),
                    "exact_order_share_pct": (
                        exact_orders / orders_total * 100.0 if orders_total else 0.0
                    ),
                    "solve_ms": solve_ms,
                }
            )
    return pd.DataFrame(result_rows), pd.DataFrame(route_rows)


def summarize_curve(results: pd.DataFrame) -> pd.DataFrame:
    """Aggregate a scenario curve with pooled distance ratios."""
    rows = []
    keys = [
        "scenario",
        "capacity",
        "rate_per_hour",
        "matching_window_min",
        "sla_min",
        "speed_kmh",
        "drop_service_min",
        "constrained",
    ]
    for values, group in results.groupby(keys, dropna=False, sort=True):
        kappas = group["kappa"].dropna()
        numerator = float(group["route_distance_m"].sum())
        denominator = float(group["standalone_distance_m"].sum())
        code_denominator = float(group["code_standalone_distance_m"].sum())
        row = dict(zip(keys, values))
        row.update(
            {
                "kappa": numerator / denominator if denominator else np.nan,
                "kappa_ci_low": float(kappas.quantile(0.025)) if len(kappas) else np.nan,
                "kappa_ci_high": float(kappas.quantile(0.975)) if len(kappas) else np.nan,
                "code_compatible_kappa": (
                    numerator / code_denominator if code_denominator else np.nan
                ),
                "orders": int(group["orders"].sum()),
                "failed_orders": int(group["failed_orders"].sum()),
                "mean_bundle_size": (
                    float(group["orders"].sum() / group["routes"].sum())
                    if group["routes"].sum()
                    else np.nan
                ),
                "exact_order_share_pct": (
                    float(
                        np.average(
                            group["exact_order_share_pct"],
                            weights=np.maximum(group["orders"], 1),
                        )
                    )
                    if len(group)
                    else 0.0
                ),
                "solve_seconds": float(group["solve_ms"].sum() / 1000.0),
                "replicates": int(group["replicate"].nunique()),
            }
        )
        rows.append(row)
    summary = pd.DataFrame(rows).sort_values(["scenario", "capacity"])
    summary["improvement_pct"] = (
        -summary.groupby("scenario")["kappa"].pct_change() * 100.0
    )
    return summary.reset_index(drop=True)


def detect_plateau(
    curve: pd.DataFrame,
    threshold_pct: float,
    consecutive_steps: int,
) -> dict:
    """Find the first capacity followed by consecutive sub-threshold gains."""
    ordered = curve.sort_values("capacity").reset_index(drop=True)
    capacities = ordered["capacity"].to_numpy(dtype=int)
    kappas = ordered["kappa"].to_numpy(dtype=float)
    gains = (kappas[:-1] - kappas[1:]) / kappas[:-1] * 100.0
    for start in range(0, len(gains) - consecutive_steps + 1):
        window = gains[start : start + consecutive_steps]
        if np.all(np.isfinite(window)) and np.all(window < threshold_pct):
            return {
                "reached": True,
                "bundle_size_star": int(capacities[start]),
                "future_extreme_kappa": float(kappas[start]),
                "threshold_pct": float(threshold_pct),
                "consecutive_steps": int(consecutive_steps),
                "following_improvements_pct": [float(value) for value in window],
            }
    return {
        "reached": False,
        "bundle_size_star": None,
        "future_extreme_kappa": None,
        "threshold_pct": float(threshold_pct),
        "consecutive_steps": int(consecutive_steps),
        "max_tested_bundle_size": int(capacities.max()) if len(capacities) else None,
    }


def _scenario_definitions(
    settings: KappaVrpSettings,
    smoke: bool,
) -> tuple[list[Scenario], int]:
    primary_reps = 1 if smoke else settings.primary_replicates
    geometric_reps = 1 if smoke else settings.geometric_replicates
    baseline = Scenario(
        name="baseline_operational",
        rate_per_hour=None,
        matching_window_min=6,
        sla_min=30,
        speed_kmh=30,
        drop_service_min=2,
        replicates=primary_reps,
    )
    limit = Scenario(
        name="constrained_best_case",
        rate_per_hour=60,
        matching_window_min=12,
        sla_min=45,
        speed_kmh=30,
        drop_service_min=1,
        replicates=primary_reps,
    )
    geometric = Scenario(
        name="geometric_lower_bound",
        rate_per_hour=60,
        matching_window_min=None,
        sla_min=None,
        speed_kmh=30,
        drop_service_min=0,
        replicates=geometric_reps,
        constrained=False,
    )
    return [baseline, limit, geometric], primary_reps


def _sensitivity_definitions(
    settings: KappaVrpSettings,
    smoke: bool,
) -> list[Scenario]:
    reps = 1 if smoke else settings.sensitivity_replicates
    scenarios = []
    for rate in settings.controlled_rates_per_hour:
        scenarios.append(
            Scenario(
                name=f"sensitivity_rate_{rate:g}",
                rate_per_hour=float(rate),
                matching_window_min=12,
                sla_min=45,
                speed_kmh=30,
                drop_service_min=1,
                replicates=reps,
            )
        )
    for window in settings.matching_windows_min:
        scenarios.append(
            Scenario(
                name=f"sensitivity_window_{window:g}",
                rate_per_hour=60,
                matching_window_min=float(window),
                sla_min=45,
                speed_kmh=30,
                drop_service_min=1,
                replicates=reps,
            )
        )
    for sla in settings.sla_values_min:
        scenarios.append(
            Scenario(
                name=f"sensitivity_sla_{sla:g}",
                rate_per_hour=60,
                matching_window_min=12,
                sla_min=float(sla),
                speed_kmh=30,
                drop_service_min=1,
                replicates=reps,
            )
        )
    for speed in settings.speed_values_kmh:
        scenarios.append(
            Scenario(
                name=f"sensitivity_speed_{speed:g}",
                rate_per_hour=60,
                matching_window_min=12,
                sla_min=45,
                speed_kmh=float(speed),
                drop_service_min=1,
                replicates=reps,
            )
        )
    for service in settings.drop_service_values_min:
        scenarios.append(
            Scenario(
                name=f"sensitivity_drop_{service:g}",
                rate_per_hour=60,
                matching_window_min=12,
                sla_min=45,
                speed_kmh=30,
                drop_service_min=float(service),
                replicates=reps,
            )
        )
    unique = {}
    for scenario in scenarios:
        unique[scenario.name] = scenario
    return list(unique.values())


def run_experiment(
    exp_config: dict,
    project_root: Path,
    smoke: bool = False,
    skip_sensitivities: bool = False,
) -> dict:
    """Run the configured city-level kappa VRP experiment."""
    settings = KappaVrpSettings.from_mapping(exp_config.get("kappa_vrp", {}))
    region_dir = (
        Path(exp_config["data_dir"]) / settings.level / settings.region.lower()
    )
    od_path = region_dir / f"{settings.region.lower()}_od_pairs.csv"
    graphml_path = region_dir / f"{settings.region.lower()}_roadnet.graphml"
    if not od_path.exists() or not graphml_path.exists():
        raise FileNotFoundError(
            f"Required remote assets not found: {od_path}, {graphml_path}"
        )
    output_dir = region_dir / "kappa_vrp_experiment"
    figure_dir = Path(exp_config["figures_dir"]) / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    started = time.time()
    cache_payload = {
        "od_size": od_path.stat().st_size,
        "od_mtime_ns": od_path.stat().st_mtime_ns,
        "graph_size": graphml_path.stat().st_size,
        "graph_mtime_ns": graphml_path.stat().st_mtime_ns,
        "window_km": settings.window_km,
        "n_restaurants": settings.n_restaurants,
        "destination_clusters": settings.destination_clusters,
        "seed": settings.random_seed,
    }
    cache_key = hashlib.sha256(
        json.dumps(cache_payload, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]
    cache_dir = (
        Path(exp_config["cache_dir"])
        / "kappa_vrp"
        / settings.region.lower()
        / cache_key
    )
    restaurant_cache = cache_dir / "restaurants.csv"
    destination_cache = cache_dir / "destinations.csv"
    block_cache = cache_dir / "distance_blocks.npz"
    metadata_cache = cache_dir / "preprocessing.json"
    if all(
        path.exists()
        for path in (
            restaurant_cache,
            destination_cache,
            block_cache,
            metadata_cache,
        )
    ):
        print(f"Loading kappa VRP preprocessing cache: {cache_dir}")
        restaurants = pd.read_csv(restaurant_cache)
        destinations = pd.read_csv(destination_cache)
        archive = np.load(block_cache)
        blocks = {
            int(name.removeprefix("restaurant_")): archive[name]
            for name in archive.files
        }
        with metadata_cache.open(encoding="utf-8") as handle:
            preprocessing_meta = json.load(handle)
        window_meta = preprocessing_meta["window"]
        compression_meta = preprocessing_meta["compression"]
        graph_meta = preprocessing_meta["road_graph"]
        reachable_meta = preprocessing_meta["reachability"]
    else:
        print(f"Loading {od_path} with columns {OD_COLUMNS}")
        od_df = pd.read_csv(od_path, usecols=list(OD_COLUMNS))
        od_df = od_df[
            np.isfinite(od_df["annual_sales"])
            & (od_df["annual_sales"] > 0)
            & np.isfinite(od_df["from_lat"])
            & np.isfinite(od_df["from_lng"])
            & np.isfinite(od_df["to_lat"])
            & np.isfinite(od_df["to_lng"])
        ].copy()
        restaurants, window_meta = select_dense_restaurant_window(
            od_df,
            settings.window_km,
            settings.n_restaurants,
        )
        destinations, compression_meta = compress_destination_catchments(
            od_df,
            restaurants,
            settings.destination_clusters,
            settings.random_seed,
        )
        blocks, graph_meta = build_directed_distance_blocks(
            graphml_path,
            restaurants,
            destinations,
        )
        destinations, blocks, reachable_meta = filter_unreachable_destinations(
            restaurants,
            destinations,
            blocks,
        )
        cache_dir.mkdir(parents=True, exist_ok=True)
        restaurants.to_csv(restaurant_cache, index=False)
        destinations.to_csv(destination_cache, index=False)
        np.savez_compressed(
            block_cache,
            **{
                f"restaurant_{restaurant_id}": matrix
                for restaurant_id, matrix in blocks.items()
            },
        )
        preprocessing_meta = {
            "cache_key": cache_key,
            "cache_payload": cache_payload,
            "window": window_meta,
            "compression": compression_meta,
            "road_graph": graph_meta,
            "reachability": reachable_meta,
        }
        with metadata_cache.open("w", encoding="utf-8") as handle:
            json.dump(preprocessing_meta, handle, indent=2)
        print(f"Kappa VRP preprocessing cache saved to {cache_dir}")

    capacities = list(
        range(1, (4 if smoke else settings.max_bundle_size) + 1)
    )
    scenario_results = []
    route_results = []
    core_scenarios, _ = _scenario_definitions(settings, smoke)
    for scenario in core_scenarios:
        print(f"Running {scenario.name}: capacities={capacities}")
        results, routes = run_scenario_curve(
            scenario,
            settings,
            restaurants,
            destinations,
            blocks,
            capacities,
        )
        scenario_results.append(results)
        route_results.append(routes)

    core_results = pd.concat(scenario_results, ignore_index=True)
    core_summary = summarize_curve(core_results)
    limit_curve = core_summary[
        core_summary["scenario"] == "constrained_best_case"
    ]
    plateau = detect_plateau(
        limit_curve,
        settings.plateau_improvement_pct,
        settings.plateau_consecutive_steps,
    )

    sensitivity_results = pd.DataFrame()
    sensitivity_summary = pd.DataFrame()
    if not skip_sensitivities and not smoke:
        sensitivity_capacity = (
            int(plateau["bundle_size_star"])
            if plateau["reached"]
            else settings.max_bundle_size
        )
        frames = []
        route_frames = []
        for scenario in _sensitivity_definitions(settings, smoke):
            print(f"Running {scenario.name}: capacity={sensitivity_capacity}")
            results, routes = run_scenario_curve(
                scenario,
                settings,
                restaurants,
                destinations,
                blocks,
                [sensitivity_capacity],
            )
            frames.append(results)
            route_frames.append(routes)
        if frames:
            sensitivity_results = pd.concat(frames, ignore_index=True)
            sensitivity_summary = summarize_curve(sensitivity_results)
            route_results.extend(route_frames)

    route_frame = pd.concat(
        [frame for frame in route_results if not frame.empty],
        ignore_index=True,
    )
    core_results.to_csv(output_dir / "replicate_results.csv", index=False)
    core_summary.to_csv(output_dir / "kappa_by_capacity.csv", index=False)
    route_frame.to_csv(
        output_dir / "route_results.csv.gz",
        index=False,
        compression="gzip",
    )
    if not sensitivity_results.empty:
        sensitivity_results.to_csv(
            output_dir / "sensitivity_replicates.csv",
            index=False,
        )
        sensitivity_summary.to_csv(
            output_dir / "sensitivity_summary.csv",
            index=False,
        )

    plateau.update(
        {
            "configured_baseline_kappa": settings.configured_kappa,
            "definition": (
                "pooled directed closed-route distance divided by pooled "
                "standalone restaurant-customer-restaurant distance"
            ),
            "scenario": asdict(
                next(
                    scenario
                    for scenario in core_scenarios
                    if scenario.name == "constrained_best_case"
                )
            ),
        }
    )
    with (output_dir / "plateau.json").open("w", encoding="utf-8") as handle:
        json.dump(plateau, handle, indent=2)

    manifest = {
        "git_sha": _git_sha(project_root),
        "created_unix": time.time(),
        "elapsed_seconds": time.time() - started,
        "smoke": smoke,
        "settings": asdict(settings),
        "preprocessing_cache": {
            "key": cache_key,
            "path": str(cache_dir),
        },
        "input": {
            "od": _file_fingerprint(od_path),
            "graphml": _file_fingerprint(graphml_path),
        },
        "window": window_meta,
        "compression": compression_meta,
        "road_graph": graph_meta,
        "reachability": reachable_meta,
        "plateau": plateau,
    }
    with (output_dir / "run_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    restaurants.to_csv(output_dir / "selected_restaurants.csv", index=False)
    destinations.to_csv(
        output_dir / "representative_destinations.csv",
        index=False,
    )

    from visualization.analysis import plot_kappa_vrp_saturation

    figure_png = figure_dir / "kappa_vrp_saturation.png"
    figure_pdf = figure_dir / "kappa_vrp_saturation.pdf"
    plot_kappa_vrp_saturation(
        core_summary,
        plateau=plateau,
        configured_kappa=settings.configured_kappa,
        save_path=str(figure_png),
        for_latex=False,
        plot_title=False,
    )
    plot_kappa_vrp_saturation(
        core_summary,
        plateau=plateau,
        configured_kappa=settings.configured_kappa,
        save_path=str(figure_pdf),
        for_latex=True,
        plot_title=False,
    )
    return {
        "output_dir": str(output_dir),
        "figure_png": str(figure_png),
        "figure_pdf": str(figure_pdf),
        "plateau": plateau,
        "manifest": manifest,
    }
