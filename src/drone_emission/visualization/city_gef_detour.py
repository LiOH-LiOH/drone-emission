"""Formal city gef detour renderers used only by the manifest registry."""
from .palette import opacity as paper_alpha

from .prepared_bundle import prepared_plot, prepared_data, evaluate_kde, fit_polynomial, kde_with_factor

from .palette import color as paper_color

import json
import os
import pickle
import time

import pandas as pd


CITY_CONTEXT_FIGURES = {
    "shanghai_city_context": {
        "region_key": "shanghai",
        "display_name": "Shanghai",
        "power_region_key": "shanghai",
    },
    "chongqing_city_context": {
        "region_key": "chongqing",
        "display_name": "Chongqing",
        "power_region_key": "chongqing",
    },
    "chengdu_city_context": {
        "region_key": "chengdu",
        "display_name": "Chengdu",
        "power_region_key": "sichuan",
    },
    "beijing_city_context": {
        "region_key": "beijing",
        "display_name": "Beijing",
        "power_region_key": "beijing",
    },
    "guangzhou_city_context": {
        "region_key": "guangzhou",
        "display_name": "Guangzhou",
        "power_region_key": "guangdong",
    },
    "tianjin_city_context": {
        "region_key": "tianjin",
        "display_name": "Tianjin",
        "power_region_key": "tianjin",
    },
    "dali_city_context": {
        "region_key": "dalibaizuzizhizhou",
        "display_name": "Dali",
        "power_region_key": "yunnan",
    },
    "kunming_city_context": {
        "region_key": "kunming",
        "display_name": "Kunming",
        "power_region_key": "yunnan",
    },
}
CITY_GEF_CONTEXT_DETOUR_CONTEXTS = (
    CITY_CONTEXT_FIGURES["chengdu_city_context"],
    CITY_CONTEXT_FIGURES["beijing_city_context"],
    CITY_CONTEXT_FIGURES["shanghai_city_context"],
    CITY_CONTEXT_FIGURES["guangzhou_city_context"],
)
CITY_CONTEXT_MARKERS = {
    "chengdu": "o",
    "beijing": "s",
    "shanghai": "D",
    "guangzhou": "^",
}
CITY_GEF_COLOR_BIN_EDGES = tuple(value / 10.0 for value in range(1, 9))
CITY_GEF_SUMMARY_ROW_HSPACE = 0.32
CITY_GEF_BOTTOM_SCATTER_SIZE = 18
CITY_GEF_BOTTOM_SCATTER_ALPHA = paper_alpha("city_gef_detour.opacity_0_45")

REDUCTION_DETOUR_CACHE_VERSION = 8
REDUCTION_DETOUR_CHUNK_ROWS = 250_000
MERCHANT_SNAP_QUANTILE = 0.99
MIDDLE_DISTANCE_MIN_KM = 0.2
MIDDLE_DISTANCE_MAX_KM = 3.5
POWER_GENERATION_2022_100M_KWH = {
    "shanghai": {
        "total": 955.0,
        "Hydro": 0.0,
        "Thermal": 912.60,
        "Wind": 24.56,
        "Solar": 17.82,
    },
    "chongqing": {
        "total": 998.0,
        "Hydro": 203.11,
        "Thermal": 755.15,
        "Wind": 34.02,
        "Solar": 5.56,
    },
    "beijing": {
        "total": 467.0,
        "Hydro": 9.43,
        "Thermal": 443.57,
        "Wind": 4.95,
        "Solar": 9.14,
    },
    "sichuan": {
        "total": 4846.0,
        "Hydro": 3886.56,
        "Thermal": 794.61,
        "Wind": 134.71,
        "Solar": 30.28,
    },
    "yunnan": {
        "total": 4017.0,
        "Hydro": 3282.91,
        "Thermal": 464.53,
        "Wind": 213.81,
        "Solar": 55.40,
    },
    "guangdong": {
        "total": 6366.0,
        "Hydro": 344.41,
        "Thermal": 4469.17,
        "Wind": 268.66,
        "Solar": 134.89,
    },
    "tianjin": {
        "total": 765.0,
        "Hydro": 0.14,
        "Thermal": 715.72,
        "Wind": 25.51,
        "Solar": 23.52,
    },
}
POWER_MIX_COLORS = {
    "Hydro": paper_color("city_gef_detour.4c78a8"),
    "Thermal": paper_color("city_gef_detour.8c564b"),
    "Wind": paper_color("city_gef_detour.59a14f"),
    "Solar": paper_color("city_gef_detour.f2cf5b"),
    "Other": paper_color("city_gef_detour.9e9e9e"),
}
POWER_MIX_GRAYSCALE_COLORS = {
    "Hydro": paper_color("city_gef_detour.2b2b2b"),
    "Thermal": paper_color("city_gef_detour.606060"),
    "Wind": paper_color("city_gef_detour.969696"),
    "Solar": paper_color("city_gef_detour.c7c7c7"),
    "Other": paper_color("city_gef_detour.e6e6e6"),
}
CITY_GEF_GRAYSCALE_VEHICLE_COLORS = {
    "E-bike": paper_color("city_gef_detour.bdbdbd"),
    "Drone": paper_color("city_gef_detour.636363"),
}
CITY_GEF_GRAYSCALE_BIN_COLORS = (
    paper_color("scale.gef.bin0"),
    paper_color("scale.gef.bin1"),
    paper_color("scale.gef.bin2"),
    paper_color("scale.gef.bin3"),
    paper_color("scale.gef.bin4"),
    paper_color("scale.gef.bin5"),
    paper_color("scale.gef.bin6"),
)
CITY_GEF_GRAYSCALE_BOTTOM_SCATTER_ALPHA = paper_alpha("city_gef_detour.opacity_0_78")


def _get_ebike_kappa(exp_config):
    """Return the configured baseline e-bike distance factor."""
    emission_params = (exp_config or {}).get("emission_params", {})
    ebike_params = emission_params.get("ebike", {})
    return float(ebike_params.get("kappa", 1.0))


def _file_cache_signature(path):
    """Return file metadata used to invalidate a figure-data cache."""
    stat = os.stat(path)
    return {
        "path": os.path.abspath(path),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def _unique_finite_coordinates(coordinates):
    """Return sorted unique finite latitude-longitude coordinate pairs."""
    import numpy as np

    if not isinstance(coordinates, np.ndarray):
        coordinates = list(coordinates)
    coordinates = np.asarray(coordinates, dtype=np.float64)
    if coordinates.ndim != 2 or coordinates.shape[1] != 2:
        raise ValueError(
            "merchant_coordinates must contain (latitude, longitude) pairs"
        )
    coordinates = coordinates[np.isfinite(coordinates).all(axis=1)]
    if not coordinates.size:
        return np.empty((0, 2), dtype=np.float64)
    return np.unique(coordinates, axis=0)


def _merchant_snap_outliers(
    merchant_coordinates,
    snap_distances_m,
    quantile=MERCHANT_SNAP_QUANTILE,
):
    """Return merchants whose snap distance is strictly above the quantile."""
    import numpy as np

    coordinates = np.asarray(merchant_coordinates, dtype=np.float64)
    if coordinates.ndim != 2 or coordinates.shape[1] != 2:
        raise ValueError(
            "merchant_coordinates must contain (latitude, longitude) pairs"
        )
    if not np.isfinite(coordinates).all():
        raise ValueError("merchant_coordinates must be finite")
    if len(np.unique(coordinates, axis=0)) != len(coordinates):
        raise ValueError("merchant_coordinates must be unique")
    snap_distances_m = np.asarray(snap_distances_m, dtype=np.float64)
    if len(coordinates) != len(snap_distances_m):
        raise ValueError(
            "merchant_coordinates and snap_distances_m must have equal length"
        )
    if not np.isfinite(snap_distances_m).all():
        raise ValueError("snap_distances_m must be finite")
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must be between zero and one")
    threshold_m = float(
        np.quantile(snap_distances_m, quantile, method="linear")
    )
    excluded = {
        tuple(coordinate)
        for coordinate, distance in zip(coordinates, snap_distances_m)
        if distance > threshold_m
    }
    return excluded, threshold_m


def _merchant_snap_filter_from_graphml(merchant_coordinates, graphml_path):
    """Return merchant origins strictly above their city's road-snap P99."""
    import numpy as np

    from drone_emission.roadnet.sparse_graph import SparseGraph

    coordinates = _unique_finite_coordinates(merchant_coordinates)
    if not coordinates.size:
        raise ValueError("merchant_coordinates must contain finite coordinates")

    graph = SparseGraph.from_graphml(graphml_path)
    nearest_indices = graph.find_nearest_nodes_batch(coordinates)
    nearest_coordinates = graph._node_coords[nearest_indices]
    latitude_1 = np.radians(coordinates[:, 0])
    latitude_2 = np.radians(nearest_coordinates[:, 0])
    delta_latitude = latitude_2 - latitude_1
    delta_longitude = np.radians(
        nearest_coordinates[:, 1] - coordinates[:, 1]
    )
    haversine_term = (
        np.sin(delta_latitude / 2.0) ** 2
        + np.cos(latitude_1)
        * np.cos(latitude_2)
        * np.sin(delta_longitude / 2.0) ** 2
    )
    snap_distances_m = 2.0 * 6_371_000.0 * np.arcsin(
        np.sqrt(np.clip(haversine_term, 0.0, 1.0))
    )
    excluded, threshold_m = _merchant_snap_outliers(
        coordinates, snap_distances_m
    )
    return excluded, snap_distances_m, threshold_m


def _middle_distance_domain(centres, values, max_km):
    """Return the shared 0.2--3.5 km middle-panel domain and finite mask."""
    import numpy as np

    centres = np.asarray(centres, dtype=float)
    values = np.asarray(values, dtype=float)
    if not np.isfinite(float(max_km)) or float(max_km) <= 0:
        raise ValueError("max_km must be finite and positive")
    x_min = MIDDLE_DISTANCE_MIN_KM
    x_max = MIDDLE_DISTANCE_MAX_KM
    visible = (
        (centres >= x_min)
        & (centres <= x_max)
        & np.isfinite(centres)
        & np.isfinite(values)
    )
    return x_min, x_max, visible


def _filter_rows_by_merchant(od_chunk, excluded_merchants):
    """Exclude every OD row belonging to listed merchant origins."""
    import numpy as np

    if not excluded_merchants:
        return od_chunk, np.zeros(len(od_chunk), dtype=bool), 0.0
    origins = pd.MultiIndex.from_arrays(
        [od_chunk["from_lat"], od_chunk["from_lng"]]
    )
    excluded_index = pd.MultiIndex.from_tuples(
        sorted(excluded_merchants), names=["from_lat", "from_lng"]
    )
    excluded_mask = origins.isin(excluded_index)
    sales = pd.to_numeric(
        od_chunk.loc[excluded_mask, "annual_sales"], errors="coerce"
    ).to_numpy(dtype=float)
    filtered_sales = float(np.sum(sales[np.isfinite(sales)]))
    return od_chunk.loc[~excluded_mask].copy(), excluded_mask, filtered_sales


def _new_merchant_filter_bin_audit(max_air_distance_km, bin_width_km):
    """Create streaming accumulators for pre/post-filter bin diagnostics."""
    import numpy as np

    max_km = float(
        np.ceil(max_air_distance_km / bin_width_km) * bin_width_km
    )
    edges = np.arange(0.0, max_km + bin_width_km / 2, bin_width_km)
    n_bins = len(edges) - 1

    def state():
        return {
            "merchant_coordinates": [set() for _ in range(n_bins)],
            "od_count": np.zeros(n_bins, dtype=np.int64),
            "annual_sales": np.zeros(n_bins, dtype=np.float64),
            "ebike_emissions_g": np.zeros(n_bins, dtype=np.float64),
            "drone_emissions_g": np.zeros(n_bins, dtype=np.float64),
            "optimal_drone_sales": np.zeros(n_bins, dtype=np.float64),
        }

    return {"edges": edges, "before": state(), "after": state()}


def _update_merchant_filter_bin_audit(audit, state_name, od_chunk):
    """Accumulate one raw or filtered city chunk into an audit state."""
    import numpy as np

    valid = od_chunk[
        (od_chunk["road_distance_m"] >= 0)
        & (od_chunk["distance_km"] > 0)
    ]
    if valid.empty:
        return
    edges = audit["edges"]
    bin_indices = np.clip(
        np.digitize(valid["distance_km"].to_numpy(dtype=float), edges) - 1,
        0,
        len(edges) - 2,
    )
    state = audit[state_name]
    sales = valid["annual_sales"].to_numpy(dtype=float)
    ebike = valid["ebike_lifecycle_g"].to_numpy(dtype=float)
    drone = valid["drone_lifecycle_g"].to_numpy(dtype=float)
    coordinates = valid[["from_lat", "from_lng"]].to_numpy(dtype=float)
    for index in np.unique(bin_indices):
        mask = bin_indices == index
        index = int(index)
        finite_coordinates = coordinates[mask]
        finite_coordinates = finite_coordinates[
            np.isfinite(finite_coordinates).all(axis=1)
        ]
        state["merchant_coordinates"][index].update(
            map(tuple, finite_coordinates)
        )
        state["od_count"][index] += int(mask.sum())
        state["annual_sales"][index] += float(np.sum(sales[mask]))
        state["ebike_emissions_g"][index] += float(
            np.sum(ebike[mask] * sales[mask])
        )
        state["drone_emissions_g"][index] += float(
            np.sum(drone[mask] * sales[mask])
        )
        state["optimal_drone_sales"][index] += float(
            np.sum(sales[mask][drone[mask] < ebike[mask]])
        )


def _finalize_merchant_filter_bin_audit(audit):
    """Return JSON-ready pre/post-filter counts, sales, and middle metrics."""
    edges = audit["edges"]
    records = []
    for index in range(len(edges) - 1):
        record = {
            "bin_left_air_km": float(edges[index]),
            "bin_right_air_km": float(edges[index + 1]),
            "bin_center_air_km": float(
                (edges[index] + edges[index + 1]) / 2.0
            ),
        }
        for state_name in ("before", "after"):
            state = audit[state_name]
            sales = float(state["annual_sales"][index])
            ebike = float(state["ebike_emissions_g"][index])
            drone = float(state["drone_emissions_g"][index])
            populated = bool(state["od_count"][index])
            record[state_name] = {
                "merchant_count": len(state["merchant_coordinates"][index]),
                "od_count": int(state["od_count"][index]),
                "annual_sales": sales,
                "optimal_drone_share_pct": (
                    float(state["optimal_drone_sales"][index] / sales * 100.0)
                    if populated and sales > 0
                    else None
                ),
                "sales_weighted_ebike_minus_drone_g_per_order": (
                    float((ebike - drone) / sales)
                    if populated and sales > 0
                    else None
                ),
                "lifecycle_reduction_pct": (
                    float((ebike - drone) / ebike * 100.0)
                    if populated and ebike > 0
                    else None
                ),
            }
        records.append(record)
    return records


def _canonical_binned_cache_meta(metadata):
    """City-keyed bins do not depend on the requesting panel's city order."""
    result = dict(metadata)
    result["city_keys"] = sorted(result.get("city_keys", []))
    if "merchant_filter" in result:
        result["merchant_filter"] = dict(result["merchant_filter"])
        result["merchant_filter"]["region_keys"] = sorted(
            result["merchant_filter"].get("region_keys", [])
        )
    return result


def _build_reduction_detour_binned_by_city(
    data_dir,
    selected,
    figure_name,
    exp_config=None,
    absolute_reduction_metric="total_t",
    use_binned_cache=False,
    bin_width_km=0.1,
    cache_namespace=None,
):
    """Return binned reduction-detour data for selected city rows."""
    from dataclasses import asdict, replace

    import numpy as np

    from drone_emission.emission.cal_single_delivery import create_ebike_params, get_emission_params
    from drone_emission.sensitivity.vectorized_lca import EbikeScenario, compute_ebike_per_order_g
    from drone_emission.utils.config import get_cache_dir
    from drone_emission.visualization.city_gef_support import (
        _bin_reduction_and_detour_chunks,
    )

    if exp_config is None:
        print(f"{figure_name}: exp_config missing; cannot apply configured kappa")
        return None

    ebike_kappa = _get_ebike_kappa(exp_config)
    ebike_params = create_ebike_params(exp_config)
    emission_cfg = get_emission_params(exp_config)
    scenario = EbikeScenario(
        kappa=ebike_kappa,
        rider_weight=emission_cfg["rider_weight"],
        payload_kg=emission_cfg["payload_kg"],
    )

    selected_keys = selected["region_key"].tolist()
    gef_map = dict(zip(selected["region_key"], selected["gef"]))
    label_map = dict(zip(selected["region_key"], selected["region_name_en"]))
    labels = {key: label_map.get(key, key) for key in selected_keys}
    od_paths = {
        key: os.path.join(data_dir, "city", key, f"{key}_od_pairs.csv")
        for key in selected_keys
    }
    graphml_paths = {
        key: os.path.join(data_dir, "city", key, f"{key}_roadnet.graphml")
        for key in selected_keys
    }

    cache_identity = cache_namespace or figure_name
    cache_path = None
    cache_meta = None
    available_keys = []
    source_files = {}
    for key in selected_keys:
        od_path = od_paths[key]
        if not os.path.exists(od_path):
            print(f"{figure_name}: missing {od_path} — skipping {key}")
            continue
        gef = gef_map.get(key)
        if gef is None or pd.isna(gef):
            print(f"{figure_name}: {key} missing province GEF — skipping")
            continue
        available_keys.append(key)
        source_files[key] = _file_cache_signature(od_path)
        graphml_path = graphml_paths[key]
        if not os.path.exists(graphml_path):
            raise FileNotFoundError(
                f"{figure_name}: missing {graphml_path}; "
                "run the declared prerequisite stage first"
            )
        source_files[f"{key}_roadnet"] = _file_cache_signature(graphml_path)

    if not available_keys:
        print(f"{figure_name}: no city OD data available")
        return None

    if use_binned_cache:
        cache_path = os.path.join(
            get_cache_dir("visualization", "paper"),
            f"{cache_identity}_binned.pkl",
        )
        cache_meta = {
            "version": REDUCTION_DETOUR_CACHE_VERSION,
            "figure_name": cache_identity,
            "city_keys": available_keys,
            "source_files": source_files,
            "gef": {key: float(gef_map[key]) for key in available_keys},
            "kappa": ebike_kappa,
            "ebike_params": asdict(ebike_params),
            "scenario": asdict(scenario),
            "bin_width_km": bin_width_km,
            "absolute_reduction_metric": absolute_reduction_metric,
            "merchant_filter": {
                "region_keys": available_keys,
                "rule": "independently exclude each city's merchants with road-snap distance > city P99",
                "quantile": MERCHANT_SNAP_QUANTILE,
                "quantile_method": "linear",
                "scope": "all middle-panel OD rows for excluded merchants",
                "middle_distance_domain_km": [
                    MIDDLE_DISTANCE_MIN_KM,
                    MIDDLE_DISTANCE_MAX_KM,
                ],
            },
        }
        if os.path.exists(cache_path):
            try:
                with open(cache_path, "rb") as f:
                    cache = pickle.load(f)
                cached_meta = _canonical_binned_cache_meta(cache.get("meta", {}))
                expected_meta = _canonical_binned_cache_meta(cache_meta)
                if cached_meta == expected_meta:
                    print(f"{figure_name}: loading binned data cache {cache_path}")
                    return cache["binned_by_city"], labels
                print(f"{figure_name}: binned data cache is stale; rebuilding")
            except Exception as exc:
                print(
                    f"{figure_name}: failed to load binned data cache ({exc}); "
                    "rebuilding"
                )

    binned_by_city = {}
    required = {
        "road_distance_m",
        "distance_km",
        "drone_lifecycle_g",
        "ebike_lifecycle_g",
        "annual_sales",
    }
    for key in available_keys:
        od_path = od_paths[key]
        columns = set(pd.read_csv(od_path, nrows=0).columns)
        city_required = set(required)
        city_required.update({"from_lat", "from_lng"})
        if not city_required.issubset(columns):
            print(f"{figure_name}: {key} missing columns — skipping")
            continue
        max_air_distance_km = 0.0
        merchant_coordinates = set()
        first_pass_columns = ["distance_km", "from_lat", "from_lng"]
        first_pass_dtypes = {
            "distance_km": "float64",
            "from_lat": "float64",
            "from_lng": "float64",
        }
        for distance_chunk in pd.read_csv(
            od_path,
            usecols=first_pass_columns,
            dtype=first_pass_dtypes,
            chunksize=REDUCTION_DETOUR_CHUNK_ROWS,
        ):
            distances = distance_chunk["distance_km"].to_numpy(copy=False)
            valid_distances = distances[np.isfinite(distances) & (distances > 0)]
            if valid_distances.size:
                max_air_distance_km = max(
                    max_air_distance_km, float(valid_distances.max())
                )
            coordinates = distance_chunk[["from_lat", "from_lng"]].to_numpy(
                dtype=float
            )
            coordinates = coordinates[np.isfinite(coordinates).all(axis=1)]
            merchant_coordinates.update(map(tuple, coordinates))
        if max_air_distance_km <= 0:
            print(f"{figure_name}: {key} has no valid air distances — skipping")
            continue

        if not merchant_coordinates:
            raise ValueError(
                f"{figure_name}: {key} has no valid merchant origins"
            )
        sorted_merchants = _unique_finite_coordinates(merchant_coordinates)
        excluded_merchants, snap_distances_m, snap_threshold_m = (
            _merchant_snap_filter_from_graphml(
                sorted_merchants, graphml_paths[key]
            )
        )
        merchant_filter = {
            "region_key": key,
            "rule": "exclude merchants with road-snap distance > P99",
            "quantile": MERCHANT_SNAP_QUANTILE,
            "quantile_method": "linear",
            "scope": "all middle-panel OD rows for excluded merchants",
            "snap_distance_p99_m": snap_threshold_m,
            "merchant_count": len(merchant_coordinates),
            "retained_merchant_count": (
                len(merchant_coordinates) - len(excluded_merchants)
            ),
            "excluded_merchant_count": len(excluded_merchants),
            "maximum_snap_distance_m": float(snap_distances_m.max()),
            "filtered_od_count": 0,
            "filtered_annual_sales": 0.0,
        }
        merchant_bin_audit = _new_merchant_filter_bin_audit(
            max_air_distance_km, bin_width_km
        )

        gef = gef_map[key]
        ebike_pp = replace(ebike_params, e_use=float(gef))
        detour_totals = {
            "weighted_road_m": 0.0,
            "weighted_air_m": 0.0,
            "valid_path_count": 0,
            "invalid_path_count": 0,
        }

        def prepared_chunks():
            for od_chunk in pd.read_csv(
                od_path,
                usecols=sorted(city_required),
                dtype={column: "float64" for column in city_required},
                chunksize=REDUCTION_DETOUR_CHUNK_ROWS,
            ):
                road_m = od_chunk["road_distance_m"].to_numpy(copy=False)
                air_km = od_chunk["distance_km"].to_numpy(copy=False)
                annual_sales = od_chunk["annual_sales"].to_numpy(copy=False)
                valid_detour = (
                    np.isfinite(road_m)
                    & np.isfinite(air_km)
                    & np.isfinite(annual_sales)
                    & (road_m >= 0)
                    & (air_km > 0)
                    & (annual_sales > 0)
                )
                detour_totals["weighted_road_m"] += float(
                    np.sum(road_m[valid_detour] * annual_sales[valid_detour])
                )
                detour_totals["weighted_air_m"] += float(
                    np.sum(
                        air_km[valid_detour]
                        * 1000.0
                        * annual_sales[valid_detour]
                    )
                )
                detour_totals["valid_path_count"] += int(valid_detour.sum())
                detour_totals["invalid_path_count"] += int((~valid_detour).sum())
                od_chunk["ebike_lifecycle_g"] = compute_ebike_per_order_g(
                    road_m / 1000.0, ebike_pp, scenario
                )
                _update_merchant_filter_bin_audit(
                    merchant_bin_audit, "before", od_chunk
                )
                od_chunk, filtered_mask, filtered_sales = (
                    _filter_rows_by_merchant(od_chunk, excluded_merchants)
                )
                merchant_filter["filtered_od_count"] += int(
                    filtered_mask.sum()
                )
                merchant_filter["filtered_annual_sales"] += filtered_sales
                _update_merchant_filter_bin_audit(
                    merchant_bin_audit, "after", od_chunk
                )
                yield od_chunk

        (
            centres,
            reduction_pct,
            reduction_abs,
            detour_by_bin,
            max_km,
            oracle_drone_share_pct,
        ) = (
            _bin_reduction_and_detour_chunks(
                prepared_chunks(),
                bin_width_km,
                max_air_distance_km,
                absolute_reduction_metric,
            )
        )
        city_detour_ratio = (
            detour_totals["weighted_road_m"]
            / detour_totals["weighted_air_m"]
            if detour_totals["weighted_air_m"] > 0
            else np.nan
        )
        merchant_filter["distance_bins"] = (
            _finalize_merchant_filter_bin_audit(merchant_bin_audit)
        )
        binned_by_city[key] = {
            "centres": centres,
            "reduction_pct": reduction_pct,
            "reduction_abs": reduction_abs,
            "detour_by_bin": detour_by_bin,
            "max_km": max_km,
            "oracle_drone_share_pct": oracle_drone_share_pct,
            "city_detour_ratio": city_detour_ratio,
            "valid_path_count": detour_totals["valid_path_count"],
            "invalid_path_count": detour_totals["invalid_path_count"],
            "merchant_filter": merchant_filter,
        }

    if use_binned_cache and binned_by_city:
        if cache_path and cache_meta:
            with open(cache_path, "wb") as f:
                pickle.dump(
                    {"meta": cache_meta, "binned_by_city": binned_by_city},
                    f,
                    protocol=pickle.HIGHEST_PROTOCOL,
                )
            print(f"{figure_name}: wrote binned data cache {cache_path}")
    if binned_by_city:
        print(
            f"{figure_name}: rebuilt binned data at kappa={ebike_kappa} "
            f"for {len(binned_by_city)} cities"
        )
        return binned_by_city, labels

    print(f"{figure_name}: no city OD data available")
    return None


def _plot_compact_city_context_summary_panels(
    fig,
    subplot_spec,
    binned_by_city,
    labels,
    contexts,
    panel_start_letter="b",
    legend_ax=None,
    right_label_offset=0.055,
    grayscale=False,
):
    """Draw compact power-mix and distance panels for the city summary."""
    import numpy as np
    from matplotlib.patches import Patch

    power_mix_colors = (
        POWER_MIX_GRAYSCALE_COLORS if grayscale else POWER_MIX_COLORS
    )
    power_mix_handles = [
        Patch(
            facecolor=power_mix_colors[name],
            edgecolor=paper_color("city_gef_detour.777777") if grayscale else "none",
            linewidth=0.4 if grayscale else 0.0,
            label=name,
        )
        for name in ("Hydro", "Thermal", "Wind", "Solar", "Other")
    ]
    if legend_ax is None:
        middle = subplot_spec.subgridspec(
            2,
            1,
            height_ratios=[0.12, 0.88],
            hspace=0.08,
        )
        legend_ax = fig.add_subplot(middle[0])
        plot_spec = middle[1]
        legend_ax.legend(
            handles=power_mix_handles,
            loc="center",
            ncol=5,
            frameon=False,
            fontsize=7,
            handlelength=1.0,
            handletextpad=0.35,
            columnspacing=0.75,
            borderaxespad=0.0,
        )
    else:
        plot_spec = subplot_spec
        legend_ax.legend(
            handles=power_mix_handles,
            loc="center left",
            ncol=1,
            frameon=False,
            fontsize=7,
            title="Power mix",
            title_fontsize=7.5,
            handlelength=1.05,
            handletextpad=0.45,
            labelspacing=0.60,
            borderaxespad=0.0,
        )
        legend_ax.get_legend().get_title().set_fontweight("bold")
    legend_ax.set_axis_off()

    outer = plot_spec.subgridspec(1, len(contexts), wspace=0.24)
    visible_by_city = {}
    absolute_values = []
    for context in contexts:
        region_key = context["region_key"]
        binned = binned_by_city[region_key]
        centres = np.asarray(binned["centres"], dtype=float)
        reduction_abs = np.asarray(binned["reduction_abs"], dtype=float)
        x_min, x_max, visible = _middle_distance_domain(
            centres,
            reduction_abs,
            binned["max_km"],
        )
        visible_by_city[region_key] = (centres, reduction_abs, visible, x_min, x_max)
        absolute_values.extend(reduction_abs[visible].tolist())
        if visible.sum() >= 2:
            slope, intercept = fit_polynomial(
                centres[visible], reduction_abs[visible], deg=1
            )
            absolute_values.extend(
                [slope * x_min + intercept, slope * x_max + intercept]
            )

    finite_absolute = np.asarray(absolute_values, dtype=float)
    finite_absolute = finite_absolute[np.isfinite(finite_absolute)]
    if finite_absolute.size:
        absolute_min = min(0.0, float(finite_absolute.min()))
        absolute_max = max(0.0, float(finite_absolute.max()))
        absolute_pad = max((absolute_max - absolute_min) * 0.08, 0.5)
        shared_absolute_ylim = (
            absolute_min - absolute_pad,
            absolute_max + absolute_pad,
        )
    else:
        shared_absolute_ylim = (-1.0, 1.0)

    panel_records = []
    panel_start_ord = ord(panel_start_letter)
    for i, context in enumerate(contexts):
        region_key = context["region_key"]
        display_name = context["display_name"]
        power_region_key = context.get("power_region_key", region_key)
        block = outer[0, i].subgridspec(
            2,
            1,
            height_ratios=[0.34, 1.0],
            hspace=0.12,
        )
        ax_power = fig.add_subplot(block[0])
        ax_distance = fig.add_subplot(block[1])
        ax_absolute = ax_distance.twinx()
        ax_absolute.patch.set_visible(False)

        generation = POWER_GENERATION_2022_100M_KWH[power_region_key]
        total = float(generation["total"])
        components = ["Hydro", "Thermal", "Wind", "Solar"]
        values = [float(generation[name]) for name in components]
        remainder = total - sum(values)
        if remainder / total > 0.005:
            components.append("Other")
            values.append(remainder)
        shares = [value / total * 100.0 for value in values]
        left = 0.0
        for name, share in zip(components, shares):
            ax_power.barh(
                [0],
                [share],
                left=[left],
                height=0.52,
                color=power_mix_colors[name],
                edgecolor=paper_color("ink.background"),
                linewidth=0.6 if grayscale else 0.45,
                zorder=3,
            )
            if share >= 12.0:
                ax_power.text(
                    left + share / 2.0,
                    0,
                    f"{share:.0f}%",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color=paper_color("ink.background") if name in {"Hydro", "Thermal"} else paper_color("city_gef_detour.222222"),
                    fontweight="bold",
                    zorder=4,
                )
            left += share
        ax_power.set_xlim(0, 100)
        ax_power.set_ylim(-0.55, 0.55)
        ax_power.set_axis_off()

        power_letter = chr(panel_start_ord + i)
        ax_power.text(
            0.0,
            1.18,
            power_letter,
            transform=ax_power.transAxes,
            fontsize=9,
            fontweight="bold",
            ha="left",
            va="bottom",
        )
        city_label_artist = ax_power.text(
            0.15,
            1.18,
            display_name,
            transform=ax_power.transAxes,
            fontsize=8,
            fontweight="bold",
            ha="left",
            va="bottom",
        )

        binned = binned_by_city[region_key]
        centres, reduction_abs, visible, x_min, x_max = visible_by_city[region_key]
        drone_share = np.asarray(
            binned.get("oracle_drone_share_pct", np.full_like(centres, np.nan)),
            dtype=float,
        )
        bar_visible = visible & np.isfinite(drone_share)
        ax_distance.bar(
            centres[bar_visible],
            drone_share[bar_visible],
            width=0.156,
            color=paper_color("city_gef_detour.c7c7c7") if grayscale else paper_color("city_gef_detour.9ecae1"),
            edgecolor=paper_color("ink.background"),
            linewidth=0.25,
            zorder=2,
        )
        ax_distance.set_xlim(x_min, x_max)
        ax_distance.set_ylim(0, 105)
        ax_distance.set_xticks([0.2, 2.0, 3.5])
        ax_distance.set_yticks([0, 50, 100])
        ax_distance.tick_params(axis="both", labelsize=7, length=2.5, pad=1.5)
        ax_distance.yaxis.set_ticks_position("right")
        ax_distance.grid(True, axis="y", alpha=paper_alpha("city_gef_detour.alpha_0_18"), linewidth=0.45)
        ax_distance.set_axisbelow(True)
        ax_distance.spines["top"].set_visible(False)
        ax_distance.spines["left"].set_visible(False)
        if i != len(contexts) - 1:
            ax_distance.tick_params(axis="y", right=False, labelright=False)
            ax_distance.spines["right"].set_visible(False)

        ax_absolute.scatter(
            centres[visible],
            reduction_abs[visible],
            s=7,
            color=paper_color("city_gef_detour.555555"),
            alpha=paper_alpha("city_gef_detour.alpha_0_72"),
            linewidth=0,
            zorder=5,
        )
        if visible.sum() >= 2:
            slope, intercept = fit_polynomial(
                centres[visible], reduction_abs[visible], deg=1
            )
            x_fit = np.array([x_min, x_max], dtype=float)
            ax_absolute.plot(
                x_fit,
                slope * x_fit + intercept,
                color=paper_color("city_gef_detour.111111"),
                linewidth=0.9,
                alpha=paper_alpha("city_gef_detour.alpha_0_82"),
                zorder=6,
            )
        ax_absolute.axhline(0, color=paper_color("city_gef_detour.555555"), linewidth=0.4, alpha=paper_alpha("city_gef_detour.alpha_0_3"))
        ax_absolute.set_ylim(*shared_absolute_ylim)
        ax_absolute.tick_params(
            axis="y",
            labelsize=7,
            labelcolor=paper_color("city_gef_detour.555555"),
            length=2.5,
            pad=1.5,
        )
        ax_absolute.yaxis.set_ticks_position("left")
        ax_absolute.spines["top"].set_visible(False)
        ax_absolute.spines["right"].set_visible(False)
        if i != 0:
            ax_absolute.tick_params(axis="y", left=False, labelleft=False)
            ax_absolute.spines["left"].set_visible(False)

        panel_records.append(
            {
                "context": context,
                "panel_letter": power_letter,
                "power_panel_letter": power_letter,
                "reduction_panel_letter": None,
                "panel_anchor_points": None,
                "power_mix_ax": ax_power,
                "power_legend_ax": legend_ax,
                "reduction_ax": ax_distance,
                "absolute_ax": ax_absolute,
                "shared_absolute_ylim": shared_absolute_ylim,
                "city_label_artist": city_label_artist,
            }
        )

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for record in panel_records:
        label_bbox = record["city_label_artist"].get_window_extent(renderer)
        label_center_y = (label_bbox.y0 + label_bbox.y1) / 2.0
        label_start_point = fig.transFigure.inverted().transform(
            (label_bbox.x0, label_center_y)
        )
        record["city_label_position"] = (
            float(label_start_point[0]),
            float(label_start_point[1]),
        )

    middle_pos = plot_spec.get_position(fig)
    fig.text(
        middle_pos.x0 - 0.055,
        middle_pos.y0 + middle_pos.height * 0.32,
        "E-bike - Drone (g CO$_2$e/order)",
        rotation=90,
        fontsize=7.5,
        ha="center",
        va="center",
        color=paper_color("city_gef_detour.555555"),
    )
    fig.text(
        middle_pos.x1 + right_label_offset,
        middle_pos.y0 + middle_pos.height * 0.32,
        "Optimal drone share (%)",
        rotation=90,
        fontsize=7.5,
        ha="center",
        va="center",
        color=paper_color("city_gef_detour.444444") if grayscale else paper_color("city_gef_detour.3182bd"),
    )
    fig.text(
        (middle_pos.x0 + middle_pos.x1) / 2.0,
        middle_pos.y0 - 0.030,
        "Air distance (km)",
        fontsize=7.5,
        ha="center",
        va="top",
    )
    return panel_records


def _create_city_gef_summary_legend_rail(
    fig,
    contexts,
    row_hspace=CITY_GEF_SUMMARY_ROW_HSPACE,
    grayscale=False,
):
    """Create three vertically aligned legend sections beside the summary."""
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    rail = fig.add_gridspec(
        3,
        1,
        left=0.855,
        right=0.985,
        bottom=0.055,
        top=0.985,
        hspace=row_hspace,
    )
    ax_top = fig.add_subplot(rail[0])
    ax_middle = fig.add_subplot(rail[1])
    ax_bottom = fig.add_subplot(rail[2])
    for ax in (ax_top, ax_middle, ax_bottom):
        ax.set_axis_off()

    if grayscale:
        vehicle_handles = [
            Patch(
                facecolor=CITY_GEF_GRAYSCALE_VEHICLE_COLORS["E-bike"],
                edgecolor=paper_color("city_gef_detour.222222"),
                linewidth=0.45,
                label="E-bike",
            ),
            Patch(
                facecolor=CITY_GEF_GRAYSCALE_VEHICLE_COLORS["Drone"],
                edgecolor=paper_color("city_gef_detour.222222"),
                linewidth=0.45,
                label="Drone",
            ),
        ]
        component_handles = [
            Patch(
                facecolor=paper_color("city_gef_detour.bdbdbd"),
                edgecolor=paper_color("city_gef_detour.222222"),
                linewidth=0.45,
                hatch="////",
                label="Production + EOL",
            ),
            Patch(
                facecolor=paper_color("city_gef_detour.bdbdbd"),
                edgecolor=paper_color("city_gef_detour.222222"),
                linewidth=0.45,
                label="Use phase",
            ),
        ]
    else:
        vehicle_handles = [
            Patch(facecolor=paper_color("city_gef_detour.4c8cbf"), edgecolor="none", label="E-bike"),
            Patch(facecolor=paper_color("city_gef_detour.e05c2b"), edgecolor="none", label="Drone"),
        ]
        component_handles = [
            Patch(
                facecolor=paper_color("city_gef_detour.777777"),
                alpha=paper_alpha("city_gef_detour.alpha_0_35"),
                edgecolor=paper_color("city_gef_detour.555555"),
                label="Production + EOL",
            ),
            Patch(
                facecolor=paper_color("city_gef_detour.777777"),
                alpha=paper_alpha("city_gef_detour.alpha_0_9"),
                edgecolor=paper_color("city_gef_detour.555555"),
                label="Use phase",
            ),
        ]
    vehicle_legend = ax_top.legend(
        handles=vehicle_handles,
        loc="upper left",
        bbox_to_anchor=(0.0, 0.98),
        frameon=False,
        fontsize=7,
        title="Vehicle",
        title_fontsize=7.5,
        handlelength=1.05,
        handletextpad=0.45,
        labelspacing=0.60,
        borderaxespad=0.0,
    )
    vehicle_legend.get_title().set_fontweight("bold")
    ax_top.add_artist(vehicle_legend)
    component_legend = ax_top.legend(
        handles=component_handles,
        loc="upper left",
        bbox_to_anchor=(0.0, 0.68),
        frameon=False,
        fontsize=7,
        title="Component",
        title_fontsize=7.5,
        handlelength=1.05,
        handletextpad=0.45,
        labelspacing=0.60,
        borderaxespad=0.0,
    )
    component_legend.get_title().set_fontweight("bold")
    ax_top.add_artist(component_legend)

    city_entries = [
        {
            "region_key": context["region_key"],
            "display_name": context["display_name"],
            "marker": CITY_CONTEXT_MARKERS.get(context["region_key"], "o"),
        }
        for context in contexts
    ]
    city_handles = [
        Line2D(
            [0],
            [0],
            marker=entry["marker"],
            markersize=4.8,
            markerfacecolor="none",
            markeredgecolor=paper_color("city_gef_detour.111111"),
            markeredgewidth=0.9,
            linestyle="None",
            label=entry["display_name"],
        )
        for entry in city_entries
    ]
    city_legend = ax_top.legend(
        handles=city_handles,
        loc="lower left",
        bbox_to_anchor=(0.0, 0.02),
        frameon=False,
        fontsize=7,
        title="City",
        title_fontsize=7.5,
        handlelength=1.05,
        handletextpad=0.45,
        labelspacing=0.45,
        borderaxespad=0.0,
    )
    city_legend.get_title().set_fontweight("bold")

    ax_bottom.text(
        0.0,
        0.93,
        "GEF",
        transform=ax_bottom.transAxes,
        fontsize=7.5,
        fontweight="bold",
        ha="left",
        va="top",
    )
    cbar_ax = ax_bottom.inset_axes([0.08, 0.12, 0.25, 0.68])
    return {
        "top": ax_top,
        "middle": ax_middle,
        "bottom": ax_bottom,
        "colorbar": cbar_ax,
        "city_entries": city_entries,
    }


def _report_number(value):
    """Return a JSON-safe scalar with NaN/Inf represented as None."""
    import numpy as np

    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        value = float(value)
        return value if np.isfinite(value) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def _report_array(values):
    """Return a JSON-safe list from an array-like object."""
    return [_report_number(value) for value in values]


def _summary_stats(values):
    """Return compact summary statistics for an array-like numeric series."""
    import numpy as np

    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "q1": None,
            "q3": None,
            "min": None,
            "max": None,
            "whisker_low": None,
            "whisker_high": None,
        }
    q1, median, q3 = np.percentile(arr, [25, 50, 75])
    iqr = q3 - q1
    low_limit = q1 - 1.5 * iqr
    high_limit = q3 + 1.5 * iqr
    whisker_low = arr[arr >= low_limit].min()
    whisker_high = arr[arr <= high_limit].max()
    return {
        "count": int(arr.size),
        "mean": _report_number(arr.mean()),
        "median": _report_number(median),
        "q1": _report_number(q1),
        "q3": _report_number(q3),
        "min": _report_number(arr.min()),
        "max": _report_number(arr.max()),
        "whisker_low": _report_number(whisker_low),
        "whisker_high": _report_number(whisker_high),
    }


def _report_axis(ax):
    """Return figure-fraction position and data scale for an axis."""
    pos = ax.get_position()
    return {
        "bbox_figure_fraction": {
            "x0": _report_number(pos.x0),
            "y0": _report_number(pos.y0),
            "x1": _report_number(pos.x1),
            "y1": _report_number(pos.y1),
            "width": _report_number(pos.width),
            "height": _report_number(pos.height),
        },
        "xlim_data": _report_array(ax.get_xlim()),
        "ylim_data": _report_array(ax.get_ylim()),
        "xlabel": ax.get_xlabel(),
        "ylabel": ax.get_ylabel(),
        "xticks_data": _report_array(ax.get_xticks()),
        "yticks_data": _report_array(ax.get_yticks()),
    }


def _city_gef_components_report(df, axes, grayscale=False):
    """Return exact plotted data for the GEF component panel."""
    from drone_emission.visualization.city_gef_support import (
        _prepare_city_gef_component_data,
    )

    ax, ax_dist = axes
    try:
        component_data = _prepare_city_gef_component_data(
            df,
            grayscale=grayscale,
        )
    except ValueError as error:
        return {"status": "skipped", "reason": str(error)}

    bar_width = component_data["bin_width"] * 0.28
    bars = [
        {
            "vehicle": bar["vehicle"],
            "color": bar["color"],
            "bin_label_right_edge": bar["bin_label_right_edge"],
            "x_center": _report_number(bar["x_center"]),
            "width": _report_number(bar_width),
            "production_eol_segment": {
                "bottom": 0.0,
                "height_g_co2e_per_order": _report_number(
                    bar["production_eol"]
                ),
                "top": _report_number(bar["production_eol"]),
                "alpha": paper_alpha("city_gef_detour.alpha_1_0") if grayscale else paper_alpha("city_gef_detour.alpha_0_38"),
                "hatch": "////" if grayscale else None,
                "edgecolor": paper_color("city_gef_detour.222222") if grayscale else paper_color("ink.background"),
            },
            "use_phase_segment": {
                "bottom": _report_number(bar["production_eol"]),
                "height_g_co2e_per_order": _report_number(
                    bar["use_phase"]
                ),
                "top": _report_number(bar["total"]),
                "alpha": paper_alpha("city_gef_detour.alpha_1_0") if grayscale else paper_alpha("city_gef_detour.alpha_0_88"),
                "hatch": None,
                "edgecolor": paper_color("city_gef_detour.222222") if grayscale else paper_color("ink.background"),
            },
            "total_label": {
                "x": _report_number(bar["x_center"]),
                "y": _report_number(bar["total"]),
                "text": f'{bar["total"]:.1f}',
                "rotation": 90,
            },
            "orders_sum": _report_number(bar["orders_sum"]),
            "city_count": int(bar["city_count"]),
        }
        for bar in component_data["bars"]
    ]
    bins = [
        {
            key: _report_number(value) if isinstance(value, float) else value
            for key, value in bin_record.items()
            if key != "index"
        }
        for bin_record in component_data["bins"]
    ]

    return {
        "status": "drawn",
        "panel_letter": "a",
        "panel_label": {
            "text": "a",
            "x_figure_fraction": _report_number(ax.get_position().x0 - 0.035),
            "y_figure_fraction": _report_number(ax.get_position().y1 + 0.006),
        },
        "axes": {
            "stacked_bars": _report_axis(ax),
            "gef_density": _report_axis(ax_dist),
        },
        "gef_column": component_data["gef_col"],
        "bin_width": component_data["bin_width"],
        "bin_edges": _report_array(component_data["bin_edges"]),
        "bins": bins,
        "bars": bars,
        "density_curve": [
            {"x": _report_number(x), "y": _report_number(y)}
            for x, y in zip(
                component_data["density_curve_x"],
                component_data["density_curve_y"],
            )
        ],
        "city_rug_marks": [
            {"x": _report_number(x), "y": 0.0}
            for x in component_data["density_x"]
        ],
    }


def _city_context_summary_report(
    binned_by_city,
    labels,
    panel_records,
    bar_metric="reduction_pct",
    grayscale=False,
):
    """Return exact plotted data for the grouped city context row."""
    import numpy as np

    if bar_metric not in {"reduction_pct", "oracle_drone_share_pct"}:
        raise ValueError(
            "bar_metric must be 'reduction_pct' or 'oracle_drone_share_pct'"
        )

    reports = []
    for record in panel_records:
        context = record["context"]
        region_key = context["region_key"]
        binned = binned_by_city[region_key]
        centres = np.asarray(binned["centres"], dtype=float)
        reduction_pct = np.asarray(binned["reduction_pct"], dtype=float)
        reduction_abs = np.asarray(binned["reduction_abs"], dtype=float)
        oracle_drone_share_pct = np.asarray(
            binned.get("oracle_drone_share_pct", np.full_like(centres, np.nan)),
            dtype=float,
        )
        if bar_metric == "oracle_drone_share_pct":
            bar_values = oracle_drone_share_pct
            bar_color = paper_color("city_gef_detour.c7c7c7") if grayscale else paper_color("city_gef_detour.9ecae1")
            bar_label = "oracle_drone_share_pct"
        else:
            bar_values = reduction_pct
            bar_color = None
            bar_label = "reduction_pct"
        detour_by_bin = binned.get("detour_by_bin") or [np.array([]) for _ in centres]
        max_km = float(binned["max_km"])
        x_min, x_max, visible = _middle_distance_domain(
            centres,
            reduction_abs,
            max_km,
        )
        x_abs = centres[visible]
        y_abs = reduction_abs[visible]

        linear_fit = None
        if x_abs.size >= 2:
            slope, intercept = fit_polynomial(x_abs, y_abs, deg=1)
            x_fit = np.array([x_min, x_max], dtype=float)
            linear_fit = {
                "slope": _report_number(slope),
                "intercept": _report_number(intercept),
                "line_points": [
                    {
                        "x": _report_number(x),
                        "y": _report_number(slope * x + intercept),
                    }
                    for x in x_fit
                ],
                "equation": (
                    f"y = {slope:.1f}x "
                    f"{'+' if intercept >= 0 else '-'} {abs(intercept):.1f}"
                ),
            }

        show_saturation_fit = region_key not in {"dalibaizuzizhizhou", "chengdu"}
        saturation_fit = None
        if show_saturation_fit and bar_metric == "reduction_pct":
            fit_mask = visible & np.isfinite(centres) & np.isfinite(reduction_pct)
            x_fit_data = centres[fit_mask]
            y_fit_data = reduction_pct[fit_mask]
            if x_fit_data.size >= 4:
                try:
                    from scipy.optimize import curve_fit

                    x0 = float(x_min)

                    def saturation_model(x, y0, amplitude, rate):
                        return y0 + amplitude * (1.0 - np.exp(-rate * (x - x0)))

                    initial = [
                        float(y_fit_data[0]),
                        float(y_fit_data[-1] - y_fit_data[0]),
                        0.8,
                    ]
                    params, _ = curve_fit(
                        saturation_model,
                        x_fit_data,
                        y_fit_data,
                        p0=initial,
                        bounds=([-np.inf, -np.inf, 0.0], [np.inf, np.inf, np.inf]),
                        maxfev=10000,
                    )
                    x_fit = np.linspace(x_min, x_max, 160)
                    y_fit = saturation_model(x_fit, *params)
                    y0, amplitude, rate = params
                    saturation_fit = {
                        "y0": _report_number(y0),
                        "amplitude": _report_number(amplitude),
                        "rate": _report_number(rate),
                        "x0": _report_number(x0),
                        "line_points": [
                            {"x": _report_number(x), "y": _report_number(y)}
                            for x, y in zip(x_fit, y_fit)
                        ],
                        "equation": (
                            f"y = {y0:.1f} "
                            f"{'+' if amplitude >= 0 else '-'} {abs(amplitude):.1f}"
                            f"(1 - e^(-{rate:.2f}(x - {x0:.1f})))"
                        ),
                    }
                except Exception as exc:
                    saturation_fit = {"status": "fit_failed", "error": str(exc)}

        generation = POWER_GENERATION_2022_100M_KWH[context.get("power_region_key", region_key)]
        total = generation["total"]
        components = ["Hydro", "Thermal", "Wind", "Solar"]
        values = [generation[name] for name in components]
        remainder = total - sum(values)
        if remainder / total > 0.005:
            components.append("Other")
            values.append(remainder)

        reports.append(
            {
                "panel_letter": record["panel_letter"],
                "panel_label": {
                    "text": record["power_panel_letter"],
                    "x_figure_fraction": _report_number(
                        record["power_mix_ax"].get_position().x0 - 0.010
                    ),
                    "y_figure_fraction": _report_number(
                        record["power_mix_ax"].get_position().y1 + 0.035
                    ),
                },
                "reduction_panel_label": (
                    {
                        "text": record["reduction_panel_letter"],
                        "x_figure_fraction": _report_number(
                            record["reduction_ax"].get_position().x0 - 0.010
                        ),
                        "y_figure_fraction": _report_number(
                            record["reduction_ax"].get_position().y1 + 0.012
                        ),
                    }
                    if record["reduction_panel_letter"] is not None
                    else None
                ),
                "city_label_position": {
                    "text": context["display_name"],
                    "x_figure_fraction": _report_number(
                        record.get(
                            "city_label_position",
                            (record["power_mix_ax"].get_position().x0 + 0.006, 0),
                        )[0]
                    ),
                    "y_figure_fraction": _report_number(
                        record.get(
                            "city_label_position",
                            (0, record["power_mix_ax"].get_position().y1 + 0.035),
                        )[1]
                    ),
                },
                "panel_anchor_points": (
                    {
                        key: {
                            "x_figure_fraction": _report_number(point[0]),
                            "y_figure_fraction": _report_number(point[1]),
                        }
                        for key, point in record["panel_anchor_points"].items()
                    }
                    if record["panel_anchor_points"] is not None
                    else None
                ),
                "region_key": region_key,
                "city_label": labels.get(region_key, context["display_name"]),
                "display_name": context["display_name"],
                "city_detour_ratio": _report_number(
                    binned.get("city_detour_ratio")
                ),
                "merchant_filter": binned.get("merchant_filter"),
                "middle_distance_domain_km": [x_min, x_max],
                "valid_path_count": int(binned.get("valid_path_count", 0)),
                "invalid_path_count": int(binned.get("invalid_path_count", 0)),
                "axes": {
                    "power_mix": _report_axis(record["power_mix_ax"]),
                    "power_legend": _report_axis(record["power_legend_ax"]),
                    "bar_axis": _report_axis(record["reduction_ax"]),
                    "reduction_bars": _report_axis(record["reduction_ax"]),
                    "absolute_scatter": _report_axis(record["absolute_ax"]),
                },
                "bar_metric": bar_label,
                "power_mix": [
                    {
                        "component": name,
                        "value_100m_kwh": _report_number(value),
                        "share_pct": _report_number(value / total * 100.0),
                        "color": (
                            POWER_MIX_GRAYSCALE_COLORS[name]
                            if grayscale
                            else POWER_MIX_COLORS[name]
                        ),
                    }
                    for name, value in zip(components, values)
                ],
                "distance_bins": [
                    {
                        "x_center_air_km": _report_number(center),
                        "bar_width_km": 0.2 * 0.78,
                        "bar_metric": bar_label,
                        "bar_height": _report_number(bar_value),
                        "bar_bottom": 0.0,
                        "bar_top": _report_number(bar_value),
                        "bar_color": (
                            bar_color
                            if bar_color is not None
                            else (
                                paper_color("city_gef_detour.2ca02c")
                                if (np.isfinite(pct) and pct >= 0)
                                else paper_color("city_gef_detour.d62728")
                            )
                        ),
                        "oracle_drone_share_pct": _report_number(oracle_share),
                        "reduction_pct_height": _report_number(pct),
                        "reduction_bar_bottom": 0.0,
                        "reduction_bar_top": _report_number(pct),
                        "reduction_bar_color": (
                            paper_color("city_gef_detour.2ca02c")
                            if (np.isfinite(pct) and pct >= 0)
                            else paper_color("city_gef_detour.d62728")
                        ),
                        "absolute_scatter_y_g_co2e_per_order": _report_number(abs_value),
                        "visible_in_panel": bool(is_visible),
                        "detour_ratio_summary": _summary_stats(detours),
                    }
                    for (
                        center,
                        pct,
                        oracle_share,
                        bar_value,
                        abs_value,
                        detours,
                        is_visible,
                    )
                    in zip(
                        centres,
                        reduction_pct,
                        oracle_drone_share_pct,
                        bar_values,
                        reduction_abs,
                        detour_by_bin,
                        visible,
                    )
                ],
                "absolute_scatter_points": [
                    {"x_air_km": _report_number(x), "y_g_co2e_per_order": _report_number(y)}
                    for x, y in zip(x_abs, y_abs)
                ],
                "absolute_linear_fit": linear_fit,
                "reduction_saturation_fit": saturation_fit,
                "shared_absolute_ylim": (
                    _report_array(record["shared_absolute_ylim"])
                    if record.get("shared_absolute_ylim") is not None
                    else None
                ),
            }
        )
    return reports


def _add_city_gef_xaxis_markers(fig, axes, selected, contexts):
    """Add selected city province-GEF markers to the top panel x-axis."""
    _, ax_dist = axes
    marker_records = []

    for context in contexts:
        region_key = context["region_key"]
        marker = CITY_CONTEXT_MARKERS.get(region_key, "o")
        city_rows = selected[selected["region_key"] == region_key]
        if city_rows.empty or "gef" not in city_rows.columns:
            continue

        gef = pd.to_numeric(city_rows["gef"], errors="coerce").dropna()
        if gef.empty:
            continue

        x = float(gef.iloc[0])
        y = 0.0
        ax_dist.scatter(
            [x],
            [y],
            s=22,
            marker=marker,
            facecolors="none",
            edgecolors=paper_color("city_gef_detour.111111"),
            linewidths=1.0,
            zorder=8,
            clip_on=False,
        )
        display_point = ax_dist.transData.transform((x, y))
        fig_point = fig.transFigure.inverted().transform(display_point)
        marker_records.append(
            {
                "region_key": region_key,
                "display_name": context["display_name"],
                "power_region_key": context.get("power_region_key", region_key),
                "province": _report_number(city_rows.iloc[0].get("province")),
                "gef": _report_number(x),
                "marker": marker,
                "x_data": _report_number(x),
                "y_data": _report_number(y),
                "x_figure_fraction": _report_number(fig_point[0]),
                "y_figure_fraction": _report_number(fig_point[1]),
            }
        )

    return marker_records


@prepared_plot()
def plot_city_gef_distribution(
    df: pd.DataFrame,
    save_path: str = None,
    figsize: tuple = (3.7, 3.8),
    for_latex: bool = False,
    plot_title: bool = False,
):
    """Plot city GEF density above selected-city power mixes."""
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.patches import Patch
    from scipy.stats import gaussian_kde

    from drone_emission.visualization._utils import (
        _add_title_suffix,
        write_figure_report,
    )

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [
                "Arial",
                "Helvetica",
                "DejaVu Sans",
                "sans-serif",
            ],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )

    required = {"region_key"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(
            "City GEF distribution requires columns: " + ", ".join(missing)
        )

    gef_col = next(
        (
            candidate
            for candidate in ("grid_emission_factor", "gef")
            if candidate in df.columns
        ),
        None,
    )
    if gef_col is None:
        raise ValueError(
            "City GEF distribution requires grid_emission_factor or gef"
        )

    plot_df = df.copy()
    if "region_level" in plot_df.columns:
        plot_df = plot_df[plot_df["region_level"].eq("city")].copy()
    if plot_df.empty:
        raise ValueError("City GEF distribution has no city rows")

    plot_df["region_key"] = plot_df["region_key"].astype(str)
    duplicated = sorted(
        plot_df.loc[
            plot_df["region_key"].duplicated(keep=False), "region_key"
        ].unique()
    )
    if duplicated:
        raise ValueError(
            "City GEF distribution has duplicate region keys: "
            + ", ".join(duplicated)
        )

    plot_df[gef_col] = pd.to_numeric(plot_df[gef_col], errors="coerce")
    invalid_mask = ~np.isfinite(plot_df[gef_col].to_numpy(dtype=float))
    if invalid_mask.any():
        invalid_keys = sorted(plot_df.loc[invalid_mask, "region_key"])
        raise ValueError(
            "City GEF distribution has missing or non-finite GEF values: "
            + ", ".join(invalid_keys)
        )

    x_min, x_max = 0.1, 0.8
    outside_mask = ~plot_df[gef_col].between(x_min, x_max, inclusive="both")
    if outside_mask.any():
        outside = [
            f"{row.region_key}={getattr(row, gef_col):.4f}"
            for row in plot_df.loc[outside_mask, ["region_key", gef_col]].itertuples(
                index=False
            )
        ]
        raise ValueError(
            "City GEF distribution values fall outside the fixed 0.1-0.8 domain: "
            + ", ".join(outside)
        )

    density_values = plot_df[gef_col].to_numpy(dtype=float)
    if np.unique(density_values).size < 2:
        raise ValueError("City GEF distribution requires at least two unique values")
    curve_x = np.linspace(x_min, x_max, 240)
    curve_y, kde_factor = kde_with_factor(density_values, curve_x, bw_method="scott")

    indexed = plot_df.set_index("region_key", drop=False)
    marker_rows = []
    for context in CITY_GEF_CONTEXT_DETOUR_CONTEXTS:
        region_key = context["region_key"]
        if region_key not in indexed.index:
            raise ValueError(
                f"City GEF distribution is missing selected city: {region_key}"
            )
        row = indexed.loc[region_key]
        marker_rows.append(
            {
                "region_key": region_key,
                "display_name": context["display_name"],
                "gef": float(row[gef_col]),
                "gef_source": (
                    str(row["gef_source"])
                    if "gef_source" in plot_df.columns
                    and pd.notna(row["gef_source"])
                    else None
                ),
                "marker": CITY_CONTEXT_MARKERS[region_key],
            }
        )
    marker_rows.sort(key=lambda item: (item["gef"], item["region_key"]))

    layout_style = {
        "density_height_ratio": 0.828,
        "power_mix_height_ratio": 1.0,
        "panel_hspace": 0.218,
        "target_density_height_inches": 1.42,
    }
    fig = plt.figure(figsize=figsize)
    outer = fig.add_gridspec(
        2,
        1,
        height_ratios=[
            layout_style["density_height_ratio"],
            layout_style["power_mix_height_ratio"],
        ],
        hspace=layout_style["panel_hspace"],
        left=0.17,
        right=0.98,
        bottom=0.055,
        top=0.97,
    )
    ax = fig.add_subplot(outer[0])
    ax.fill_between(
        curve_x,
        0.0,
        curve_y,
        color=paper_color("city_gef_detour.afafaf"),
        alpha=paper_alpha("city_gef_detour.alpha_0_35"),
        linewidth=0,
        zorder=1,
    )
    ax.plot(curve_x, curve_y, color=paper_color("city_gef_detour.4f4f4f"), linewidth=1.0, zorder=2)
    for marker_row in marker_rows:
        ax.scatter(
            [marker_row["gef"]],
            [0.0],
            s=24,
            marker=marker_row["marker"],
            facecolors=paper_color("ink.background"),
            edgecolors=paper_color("city_gef_detour.111111"),
            linewidths=0.9,
            zorder=4,
            clip_on=False,
        )

    ax.set_xlim(x_min, x_max)
    density_ymax = float(curve_y.max()) * 1.08
    ax.set_ylim(0.0, density_ymax)
    density_yticks = (
        [0.0, 1.0, 2.0]
        if density_ymax >= 2.0
        else [float(value) for value in np.linspace(0.0, density_ymax, 3)]
    )
    ax.set_yticks(density_yticks)
    ax.set_xticks(np.arange(x_min, x_max + 0.001, 0.1))
    ax.set_xlabel("Grid Emission Factor (kg CO$_2$e/kWh)", fontsize=7.5)
    ax.set_ylabel("Density", fontsize=7.5)
    ax.tick_params(axis="both", labelsize=7, length=2.5, pad=1.5)
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)

    power_mix_style = {
        "row_hspace": 0.12,
        "column_wspace": 0.03,
        "label_bar_width_ratios": [0.23, 0.77],
        "bar_height": 0.78,
        "city_label_fontsize": 8.5,
        "percentage_label_fontsize": 8.0,
        "marker_size": 28.0,
        "marker_linewidth": 1.0,
        "legend_title_fontsize": 8.0,
        "legend_fontsize": 7.5,
    }
    lower = outer[1].subgridspec(
        5,
        2,
        height_ratios=[1.0, 1.0, 1.0, 1.0, 0.60],
        width_ratios=power_mix_style["label_bar_width_ratios"],
        hspace=power_mix_style["row_hspace"],
        wspace=power_mix_style["column_wspace"],
    )
    context_by_key = {
        context["region_key"]: context
        for context in CITY_GEF_CONTEXT_DETOUR_CONTEXTS
    }
    power_mix_rows = []
    for index, marker_row in enumerate(marker_rows):
        context = context_by_key[marker_row["region_key"]]
        power_region_key = context.get(
            "power_region_key", marker_row["region_key"]
        )
        generation = POWER_GENERATION_2022_100M_KWH[power_region_key]
        total = float(generation["total"])
        components = ["Hydro", "Thermal", "Wind", "Solar"]
        values = [float(generation[name]) for name in components]
        remainder = total - sum(values)
        if remainder / total > 0.005:
            components.append("Other")
            values.append(remainder)

        label_ax = fig.add_subplot(lower[index, 0])
        bar_ax = fig.add_subplot(lower[index, 1])
        label_ax.set_xlim(0, 1)
        label_ax.set_ylim(0, 1)
        label_ax.scatter(
            [-0.38],
            [0.5],
            s=power_mix_style["marker_size"],
            marker=marker_row["marker"],
            facecolors=paper_color("ink.background"),
            edgecolors=paper_color("city_gef_detour.111111"),
            linewidths=power_mix_style["marker_linewidth"],
            clip_on=False,
        )
        label_ax.text(
            -0.18,
            0.5,
            marker_row["display_name"],
            fontsize=power_mix_style["city_label_fontsize"],
            fontweight="bold",
            ha="left",
            va="center",
        )
        label_ax.set_axis_off()

        left = 0.0
        component_records = []
        for name, value in zip(components, values):
            share = value / total * 100.0
            bar_ax.barh(
                [0],
                [share],
                left=[left],
                height=power_mix_style["bar_height"],
                color=POWER_MIX_COLORS[name],
                edgecolor=paper_color("ink.background"),
                linewidth=0.45,
                zorder=2,
            )
            label_visible = share >= 12.0
            if label_visible:
                bar_ax.text(
                    left + share / 2.0,
                    0,
                    f"{share:.0f}%",
                    fontsize=power_mix_style[
                        "percentage_label_fontsize"
                    ],
                    fontweight="bold",
                    color=(
                        paper_color("ink.background")
                        if name in {"Hydro", "Thermal"}
                        else paper_color("city_gef_detour.222222")
                    ),
                    ha="center",
                    va="center",
                    zorder=3,
                )
            component_records.append(
                {
                    "component": name,
                    "value_100m_kwh": float(value),
                    "share_pct": float(share),
                    "color": POWER_MIX_COLORS[name],
                    "bar_left_pct": float(left),
                    "bar_right_pct": float(left + share),
                    "percentage_label": (
                        f"{share:.0f}%" if label_visible else None
                    ),
                }
            )
            left += share
        bar_ax.set_xlim(0, 100)
        bar_ax.set_ylim(-0.55, 0.55)
        bar_ax.set_axis_off()
        power_mix_rows.append(
            {
                "region_key": marker_row["region_key"],
                "display_name": marker_row["display_name"],
                "marker": marker_row["marker"],
                "power_region_key": power_region_key,
                "generation_total_100m_kwh": total,
                "components": component_records,
                "label_ax": label_ax,
                "bar_ax": bar_ax,
            }
        )

    legend_ax = fig.add_subplot(lower[4, :])
    legend_ax.set_axis_off()
    legend_components = ("Hydro", "Thermal", "Wind", "Solar", "Other")
    legend_handles = [
        Patch(
            facecolor=POWER_MIX_COLORS[name],
            edgecolor="none",
            label=name,
        )
        for name in legend_components
    ]
    legend_ax.text(
        -0.085,
        0.5,
        "Power mix",
        fontsize=power_mix_style["legend_title_fontsize"],
        fontweight="bold",
        ha="left",
        va="center",
        transform=legend_ax.transAxes,
    )
    legend_ax.legend(
        handles=legend_handles,
        loc="center left",
        bbox_to_anchor=(0.12, 0.5),
        ncol=5,
        frameon=False,
        fontsize=power_mix_style["legend_fontsize"],
        handlelength=0.75,
        handletextpad=0.25,
        columnspacing=0.55,
        borderaxespad=0.0,
    )
    if plot_title:
        ax.set_title(
            "Distribution of City Grid Emission Factors",
            fontsize=8,
            fontweight="bold",
        )

    output_path = save_path
    if output_path:
        if for_latex and not output_path.endswith(".pdf"):
            output_path = output_path.rsplit(".", 1)[0] + ".pdf"
        if plot_title:
            output_path = _add_title_suffix(output_path)
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        fig.canvas.draw()
        fig.savefig(
            output_path,
            dpi=300,
            bbox_inches=fig.bbox_inches,
            pad_inches=0.0,
        )

        source_counts = (
            plot_df["gef_source"]
            .fillna("missing")
            .value_counts()
            .sort_index()
            .to_dict()
            if "gef_source" in plot_df.columns
            else {}
        )
        selected_keys = {item["region_key"] for item in marker_rows}
        report = {
            "schema_version": 2,
            "figure_id": "city_gef_distribution",
            "core_conclusion": (
                "The tracked city cohort provides the national GEF distribution "
                "context for four representative cities, whose contrasting "
                "power mixes provide descriptive context for their positions."
            ),
            "figure_size_inches": [float(figsize[0]), float(figsize[1])],
            "weighting": "equal_city",
            "city_count": int(len(plot_df)),
            "unique_gef_count": int(np.unique(density_values).size),
            "gef_column": gef_col,
            "gef_source_counts": {
                str(key): int(value) for key, value in source_counts.items()
            },
            "gef_summary": {
                "min": float(density_values.min()),
                "max": float(density_values.max()),
                "mean": float(density_values.mean()),
                "median": float(np.median(density_values)),
            },
            "kde": {
                "implementation": "scipy.stats.gaussian_kde",
                "bandwidth_rule": "scott",
                "covariance_factor": kde_factor,
                "curve": [
                    {"gef": float(x), "density": float(y)}
                    for x, y in zip(curve_x, curve_y)
                ],
                "shade_semantics": "area_under_density_curve",
            },
            "axis": {
                "xlim": [x_min, x_max],
                "xlabel": ax.get_xlabel(),
                "ylabel": ax.get_ylabel(),
                "yticks": [float(value) for value in ax.get_yticks()],
                "axes": _report_axis(ax),
                "height_inches": float(
                    ax.get_position().height * figsize[1]
                ),
                "grid": False,
            },
            "selected_city_markers": marker_rows,
            "layout": {
                "orientation": "vertical",
                "density_panel_position": "top",
                "power_mix_panel_position": "bottom",
                "panel_gap": "whitespace",
                "embedded_panel_letters": False,
                "density_height_ratio": layout_style[
                    "density_height_ratio"
                ],
                "power_mix_height_ratio": layout_style[
                    "power_mix_height_ratio"
                ],
                "panel_hspace": layout_style["panel_hspace"],
                "target_density_height_inches": layout_style[
                    "target_density_height_inches"
                ],
            },
            "power_mix": {
                "city_order": [
                    row["region_key"] for row in power_mix_rows
                ],
                "style": power_mix_style,
                "percentage_label_threshold_pct": 12.0,
                "axis_visible": False,
                "rows": [
                    {
                        key: value
                        for key, value in row.items()
                        if key not in {"label_ax", "bar_ax"}
                    }
                    | {
                        "axes": {
                            "label": _report_axis(row["label_ax"]),
                            "bar": _report_axis(row["bar_ax"]),
                        }
                    }
                    for row in power_mix_rows
                ],
                "legend": {
                    "title": "Power mix",
                    "placement": "below_rows_single_line",
                    "components": [
                        {
                            "component": name,
                            "color": POWER_MIX_COLORS[name],
                        }
                        for name in legend_components
                    ],
                    "axis": _report_axis(legend_ax),
                },
            },
            "observations": [
                {
                    "region_key": str(row.region_key),
                    "gef": float(getattr(row, gef_col)),
                    "gef_source": (
                        str(getattr(row, "gef_source"))
                        if "gef_source" in plot_df.columns
                        and pd.notna(getattr(row, "gef_source"))
                        else None
                    ),
                    "selected_city": str(row.region_key) in selected_keys,
                }
                for row in plot_df.itertuples(index=False)
            ],
            "omitted_elements": [
                "all-city rug marks",
                "standalone city legend",
                "power-mix percentage axis",
                "city GEF value labels",
                "panel letter",
                "sample-size annotation",
            ],
        }
        report_path = output_path.rsplit(".", 1)[0] + "_report.json"
        write_figure_report(report_path, report)
        print(f"City GEF distribution report saved to {report_path}")
        print(f"City GEF distribution saved to {output_path}")
    else:
        plt.show()
    plt.close(fig)
    return output_path


@prepared_plot()
def plot_city_gef_lifecycle_components(
    df: pd.DataFrame,
    save_path: str = None,
    figsize: tuple = (3.7, 3.8),
    for_latex: bool = False,
    plot_title: bool = False,
):
    """Plot GEF-binned lifecycle components without the density strip."""
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.offsetbox import (
        AnchoredOffsetbox,
        DrawingArea,
        HPacker,
        VPacker,
    )
    from matplotlib.patches import Rectangle
    from matplotlib.text import Text

    from drone_emission.visualization._utils import (
        _add_title_suffix,
        write_figure_report,
    )
    from drone_emission.visualization.city_gef_support import (
        _prepare_city_gef_component_data,
    )

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [
                "Arial",
                "Helvetica",
                "DejaVu Sans",
                "sans-serif",
            ],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )
    component_data = _prepare_city_gef_component_data(df)
    style = {
        "x_domain": [0.1, 0.8],
        "bar_width_gef": 0.032,
        "axis_label_fontsize": 8.5,
        "tick_fontsize": 8.0,
        "total_label_fontsize": 8.0,
        "total_label_rotation": 90,
        "legend_title_fontsize": 8.0,
        "legend_fontsize": 7.5,
        "grid_alpha": paper_alpha("city_gef_detour.opacity_0_18"),
        "panel_hspace": 0.245,
        "legend_height_ratio": 0.208,
        "delta_height_ratio": 0.278,
        "delta_panel_hspace": 0.046,
        "target_bar_axis_height_inches": 1.81,
        "y_tick_step": 20.0,
        "delta_ylim": [-3.0, 7.0],
        "delta_point_size": 24.0,
        "delta_point_edge_width": 0.9,
        "delta_linewidth": 0.8,
        "delta_label_fontsize": 7.0,
        "legend_column_width_points": [60.0, 100.0, 65.0],
    }

    fig = plt.figure(figsize=figsize)
    layout = fig.add_gridspec(
        2,
        1,
        height_ratios=[1.0, style["legend_height_ratio"]],
        hspace=style["panel_hspace"],
        left=0.18,
        right=0.98,
        bottom=0.055,
        top=0.97,
    )
    chart_layout = layout[0].subgridspec(
        2,
        1,
        height_ratios=[
            style["delta_height_ratio"],
            1.0 - style["delta_height_ratio"],
        ],
        hspace=style["delta_panel_hspace"],
    )
    delta_ax = fig.add_subplot(chart_layout[0])
    ax = fig.add_subplot(chart_layout[1], sharex=delta_ax)
    legend_ax = fig.add_subplot(layout[1])
    legend_ax.set_axis_off()
    legend_position = legend_ax.get_position()
    legend_ax.set_position(
        [0.0, legend_position.y0, 1.0, legend_position.height]
    )

    delta_records = []
    for bin_record in component_data["bins"]:
        bin_bars = [
            bar
            for bar in component_data["bars"]
            if bar["bin_index"] == bin_record["index"]
        ]
        totals = {bar["vehicle"]: bar["total"] for bar in bin_bars}
        delta = float(totals["E-bike"] - totals["Drone"])
        x = float(bin_record["bin_center"])
        delta_ax.vlines(
            x,
            0.0,
            delta,
            color=paper_color("city_gef_detour.666666"),
            linewidth=style["delta_linewidth"],
            zorder=2,
        )
        delta_ax.scatter(
            [x],
            [delta],
            s=style["delta_point_size"],
            facecolors=paper_color("ink.background"),
            edgecolors=paper_color("city_gef_detour.222222"),
            linewidths=style["delta_point_edge_width"],
            zorder=3,
        )
        label_offset = 0.35 if delta >= 0 else -0.35
        delta_ax.text(
            x,
            delta + label_offset,
            f"{delta:+.1f}",
            ha="center",
            va="bottom" if delta >= 0 else "top",
            fontsize=style["delta_label_fontsize"],
            color=paper_color("city_gef_detour.333333"),
            zorder=4,
        )
        delta_records.append(
            {
                "bin_label_right_edge": bin_record["label_right_edge"],
                "x_center": x,
                "ebike_total_g_co2e_per_order": float(
                    totals["E-bike"]
                ),
                "drone_total_g_co2e_per_order": float(totals["Drone"]),
                "ebike_minus_drone_g_co2e_per_order": delta,
                "display_label": f"{delta:+.1f}",
            }
        )

    delta_ax.axhline(
        0.0,
        color=paper_color("city_gef_detour.999999"),
        linewidth=0.65,
        zorder=1,
    )
    delta_ax.set_xlim(*style["x_domain"])
    delta_ax.set_ylim(*style["delta_ylim"])
    delta_ax.tick_params(
        axis="both",
        which="both",
        bottom=False,
        left=False,
        labelbottom=False,
        labelleft=False,
    )
    for spine in delta_ax.spines.values():
        spine.set_visible(False)
    delta_ax.text(
        0.98,
        0.92,
        "$\\Delta$ = E-bike $-$ Drone",
        transform=delta_ax.transAxes,
        fontsize=style["delta_label_fontsize"],
        ha="right",
        va="top",
        color=paper_color("city_gef_detour.333333"),
    )

    max_total = max(bar["total"] for bar in component_data["bars"])
    for spec in component_data["vehicle_specs"]:
        bars = [
            bar
            for bar in component_data["bars"]
            if bar["vehicle"] == spec["label"]
        ]
        positions = np.array([bar["x_center"] for bar in bars], dtype=float)
        production_eol = np.array(
            [bar["production_eol"] for bar in bars], dtype=float
        )
        use_phase = np.array(
            [bar["use_phase"] for bar in bars], dtype=float
        )
        ax.bar(
            positions,
            production_eol,
            width=style["bar_width_gef"],
            color=spec["color"],
            alpha=paper_alpha("city_gef_detour.alpha_0_38"),
            edgecolor=paper_color("ink.background"),
            linewidth=0.6,
            zorder=3,
        )
        ax.bar(
            positions,
            use_phase,
            width=style["bar_width_gef"],
            bottom=production_eol,
            color=spec["color"],
            alpha=paper_alpha("city_gef_detour.alpha_0_88"),
            edgecolor=paper_color("ink.background"),
            linewidth=0.6,
            zorder=3,
        )
        for bar in bars:
            ax.text(
                bar["x_center"],
                bar["total"],
                f'{bar["total"]:.1f}',
                ha="center",
                va="bottom",
                fontsize=style["total_label_fontsize"],
                color=paper_color("ink.0_25"),
                rotation=style["total_label_rotation"],
                zorder=5,
            )

    ax.set_xlim(*style["x_domain"])
    ax.set_xticks(np.arange(0.1, 0.81, 0.1))
    bar_ymax = max_total * 1.22
    ax.set_ylim(0.0, bar_ymax)
    ax.set_yticks(
        np.arange(0.0, bar_ymax + 1e-9, style["y_tick_step"])
    )
    ax.set_xlabel(
        "Grid Emission Factor (kg CO$_2$e/kWh)",
        fontsize=style["axis_label_fontsize"],
    )
    ax.set_ylabel(
        "Lifecycle emission\n(g CO$_2$e/order)",
        fontsize=style["axis_label_fontsize"],
    )
    ax.tick_params(
        axis="both",
        labelsize=style["tick_fontsize"],
        length=2.5,
        pad=1.5,
    )
    ax.grid(
        True,
        axis="y",
        alpha=style["grid_alpha"],
        linestyle="--",
        linewidth=0.45,
    )
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)

    def _legend_cell(
        text,
        width,
        facecolor=None,
        alpha=paper_alpha("city_gef_detour.alpha_1_0"),
        fontweight="normal",
    ):
        cell = DrawingArea(width, 13.0, 0.0, 0.0)
        text_x = 0.0
        if facecolor is not None:
            cell.add_artist(
                Rectangle(
                    (0.0, 2.5),
                    8.0,
                    8.0,
                    facecolor=facecolor,
                    edgecolor=paper_color("city_gef_detour.555555"),
                    linewidth=0.45,
                    alpha=alpha,
                )
            )
            text_x = 12.0
        cell.add_artist(
            Text(
                text_x,
                6.5,
                text,
                fontsize=(
                    style["legend_title_fontsize"]
                    if fontweight == "bold"
                    else style["legend_fontsize"]
                ),
                fontweight=fontweight,
                verticalalignment="center",
            )
        )
        return cell

    column_widths = style["legend_column_width_points"]
    vehicle_by_name = {
        spec["label"]: spec for spec in component_data["vehicle_specs"]
    }
    legend_rows = [
        HPacker(
            children=[
                _legend_cell(
                    "Vehicle",
                    column_widths[0],
                    fontweight="bold",
                ),
                _legend_cell(
                    "E-bike",
                    column_widths[1],
                    facecolor=vehicle_by_name["E-bike"]["color"],
                    alpha=paper_alpha("city_gef_detour.alpha_0_88"),
                ),
                _legend_cell(
                    "Drone",
                    column_widths[2],
                    facecolor=vehicle_by_name["Drone"]["color"],
                    alpha=paper_alpha("city_gef_detour.alpha_0_88"),
                ),
            ],
            align="center",
            pad=0.0,
            sep=0.0,
        ),
        HPacker(
            children=[
                _legend_cell(
                    "Component",
                    column_widths[0],
                    fontweight="bold",
                ),
                _legend_cell(
                    "Production + EOL",
                    column_widths[1],
                    facecolor=paper_color("city_gef_detour.777777"),
                    alpha=paper_alpha("city_gef_detour.alpha_0_38"),
                ),
                _legend_cell(
                    "Use phase",
                    column_widths[2],
                    facecolor=paper_color("city_gef_detour.777777"),
                    alpha=paper_alpha("city_gef_detour.alpha_0_88"),
                ),
            ],
            align="center",
            pad=0.0,
            sep=0.0,
        ),
    ]
    legend_table = VPacker(
        children=legend_rows,
        align="left",
        pad=0.0,
        sep=5.0,
    )
    legend_ax.add_artist(
        AnchoredOffsetbox(
            loc="center",
            child=legend_table,
            pad=0.0,
            frameon=False,
            bbox_to_anchor=(0.5, 0.5),
            bbox_transform=legend_ax.transAxes,
            borderpad=0.0,
        )
    )

    if plot_title:
        ax.set_title(
            "Lifecycle Emission Components by Grid Emission Factor",
            fontsize=9,
            fontweight="bold",
        )

    output_path = save_path
    if output_path:
        if for_latex and not output_path.endswith(".pdf"):
            output_path = output_path.rsplit(".", 1)[0] + ".pdf"
        if plot_title:
            output_path = _add_title_suffix(output_path)
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        fig.canvas.draw()
        fig.savefig(
            output_path,
            dpi=300,
            bbox_inches=fig.bbox_inches,
            pad_inches=0.0,
        )
        report_bars = [
            {
                "vehicle": bar["vehicle"],
                "color": bar["color"],
                "bin_label_right_edge": bar["bin_label_right_edge"],
                "x_center": float(bar["x_center"]),
                "width": style["bar_width_gef"],
                "production_eol_segment": {
                    "bottom": 0.0,
                    "height_g_co2e_per_order": float(
                        bar["production_eol"]
                    ),
                    "top": float(bar["production_eol"]),
                    "alpha": paper_alpha("city_gef_detour.alpha_0_38"),
                },
                "use_phase_segment": {
                    "bottom": float(bar["production_eol"]),
                    "height_g_co2e_per_order": float(bar["use_phase"]),
                    "top": float(bar["total"]),
                    "alpha": paper_alpha("city_gef_detour.alpha_0_88"),
                },
                "total_label": {
                    "text": f'{bar["total"]:.1f}',
                    "rotation": style["total_label_rotation"],
                },
                "orders_sum": float(bar["orders_sum"]),
                "city_count": int(bar["city_count"]),
            }
            for bar in component_data["bars"]
        ]
        omitted_elements = [
            "GEF density strip",
            "city rug marks",
            "panel letter",
        ]
        if not plot_title:
            omitted_elements.append("title")
        report = {
            "schema_version": 2,
            "figure_id": "city_gef_lifecycle_components",
            "core_conclusion": (
                "Sales-weighted per-order lifecycle emissions vary across "
                "GEF bins, while the descriptive e-bike minus drone "
                "difference changes from positive to negative at higher GEF."
            ),
            "figure_size_inches": [float(figsize[0]), float(figsize[1])],
            "aggregation": (
                "sum of lifecycle-component emissions divided by sum of "
                "annual orders within each GEF bin"
            ),
            "gef_column": component_data["gef_col"],
            "input_city_count": component_data["input_city_count"],
            "valid_city_count": component_data["valid_city_count"],
            "excluded_city_count": component_data["excluded_city_count"],
            "bin_width": component_data["bin_width"],
            "bin_edges": [
                float(value) for value in component_data["bin_edges"]
            ],
            "populated_bins": [
                {
                    key: value
                    for key, value in bin_record.items()
                    if key != "index"
                }
                for bin_record in component_data["bins"]
            ],
            "bars": report_bars,
            "axis": _report_axis(ax),
            "layout": {
                "bar_axis_height_inches": float(
                    ax.get_position().height * figsize[1]
                ),
                "delta_axis_height_inches": float(
                    delta_ax.get_position().height * figsize[1]
                ),
                "legend_axis_height_inches": float(
                    legend_ax.get_position().height * figsize[1]
                ),
            },
            "delta_strip": {
                "definition": "E-bike total minus Drone total",
                "unit": "g CO2e/order",
                "ylim": style["delta_ylim"],
                "height_ratio": style["delta_height_ratio"],
                "axis": _report_axis(delta_ax),
                "values": delta_records,
                "encoding": {
                    "type": "neutral_lollipop",
                    "zero_line_color": paper_color("city_gef_detour.999999"),
                    "stem_color": paper_color("city_gef_detour.666666"),
                    "point_face_color": paper_color("city_gef_detour.ffffff"),
                    "point_edge_color": paper_color("city_gef_detour.222222"),
                    "point_edge_width": style[
                        "delta_point_edge_width"
                    ],
                    "label_precision_decimals": 1,
                    "direction_labels": False,
                },
            },
            "legend_axis": _report_axis(legend_ax),
            "legends": {
                "placement": "below_chart_centered_table",
                "layout": "two_rows_three_columns",
                "center_figure_fraction_x": 0.5,
                "column_width_points": column_widths,
                "vehicle": ["E-bike", "Drone"],
                "component": ["Production + EOL", "Use phase"],
            },
            "style": style,
            "omitted_elements": omitted_elements,
        }
        report_path = output_path.rsplit(".", 1)[0] + "_report.json"
        write_figure_report(report_path, report)
        print(f"City GEF lifecycle components report saved to {report_path}")
        print(f"City GEF lifecycle components saved to {output_path}")
    else:
        plt.show()
    plt.close(fig)
    return output_path


@prepared_plot()
def plot_city_distance_drone_share_emission_fit(
    binned_by_city,
    contexts,
    save_path: str = None,
    figsize: tuple = (7.8, 2.4),
    for_latex: bool = False,
    plot_title: bool = False,
):
    """Plot four-city distance response bars, points, and linear fits."""
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import numpy as np

    from drone_emission.visualization._utils import (
        _add_title_suffix,
        write_figure_report,
    )

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [
                "Arial",
                "Helvetica",
                "DejaVu Sans",
                "sans-serif",
            ],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )
    contexts = list(contexts)
    if len(contexts) != 4:
        raise ValueError("distance response figure requires exactly four cities")
    region_keys = [context["region_key"] for context in contexts]
    if len(set(region_keys)) != len(region_keys):
        raise ValueError("distance response city keys must be unique")
    missing_keys = [key for key in region_keys if key not in binned_by_city]
    if missing_keys:
        raise ValueError(
            "distance response figure missing binned cities: "
            + ", ".join(missing_keys)
        )

    style = {
        "x_domain_km": [MIDDLE_DISTANCE_MIN_KM, MIDDLE_DISTANCE_MAX_KM],
        "x_ticks_km": [0.2, 2.0, 3.5],
        "bar_ylim_pct": [0.0, 105.0],
        "bar_yticks_pct": [0.0, 50.0, 100.0],
        "bar_width_km": 0.156,
        "bar_color": paper_color("city_gef_detour.9ecae1"),
        "scatter_color": paper_color("city_gef_detour.555555"),
        "scatter_size": 14.0,
        "scatter_alpha": paper_alpha("city_gef_detour.opacity_0_72"),
        "fit_color": paper_color("city_gef_detour.111111"),
        "fit_linewidth": 1.0,
        "fit_alpha": paper_alpha("city_gef_detour.opacity_0_82"),
        "zero_line_color": paper_color("city_gef_detour.555555"),
        "zero_linewidth": 0.45,
        "tick_fontsize": 8.0,
        "axis_label_fontsize": 8.0,
        "city_label_fontsize": 8.0,
        "grid_alpha": paper_alpha("city_gef_detour.opacity_0_18"),
        "panel_wspace": 0.24,
    }

    prepared = []
    absolute_values = []
    for context in contexts:
        region_key = context["region_key"]
        binned = binned_by_city[region_key]
        required_fields = (
            "centres",
            "reduction_abs",
            "oracle_drone_share_pct",
            "max_km",
        )
        missing_fields = [name for name in required_fields if name not in binned]
        if missing_fields:
            raise ValueError(
                f"{region_key} missing binned fields: "
                + ", ".join(missing_fields)
            )
        centres = np.asarray(binned["centres"], dtype=float)
        reduction_abs = np.asarray(binned["reduction_abs"], dtype=float)
        drone_share = np.asarray(
            binned["oracle_drone_share_pct"], dtype=float
        )
        if not (
            centres.shape == reduction_abs.shape == drone_share.shape
        ):
            raise ValueError(
                f"{region_key} binned arrays must have matching shapes"
            )
        x_min, x_max, visible = _middle_distance_domain(
            centres,
            reduction_abs,
            binned["max_km"],
        )
        if int(visible.sum()) < 2:
            raise ValueError(
                f"{region_key} requires at least two visible distance bins"
            )
        fit_x_data = centres[visible]
        fit_y_data = reduction_abs[visible]
        slope, intercept = fit_polynomial(fit_x_data, fit_y_data, deg=1)
        fit_x = np.array(
            [float(fit_x_data.min()), float(fit_x_data.max())],
            dtype=float,
        )
        fit_y = slope * fit_x + intercept
        absolute_values.extend(fit_y_data.tolist())
        absolute_values.extend(fit_y.tolist())
        prepared.append(
            {
                "context": context,
                "binned": binned,
                "centres": centres,
                "reduction_abs": reduction_abs,
                "drone_share": drone_share,
                "visible": visible,
                "bar_visible": visible & np.isfinite(drone_share),
                "x_domain": [x_min, x_max],
                "fit_slope": float(slope),
                "fit_intercept": float(intercept),
                "fit_x": fit_x,
                "fit_y": fit_y,
            }
        )

    finite_absolute = np.asarray(absolute_values, dtype=float)
    finite_absolute = finite_absolute[np.isfinite(finite_absolute)]
    absolute_min = min(0.0, float(finite_absolute.min()))
    absolute_max = max(0.0, float(finite_absolute.max()))
    absolute_pad = max((absolute_max - absolute_min) * 0.08, 0.5)
    shared_absolute_ylim = [
        absolute_min - absolute_pad,
        absolute_max + absolute_pad,
    ]

    fig = plt.figure(figsize=figsize)
    grid = fig.add_gridspec(
        1,
        4,
        left=0.09,
        right=0.91,
        bottom=0.23,
        top=0.83,
        wspace=style["panel_wspace"],
    )
    plot_records = []
    for index, item in enumerate(prepared):
        context = item["context"]
        ax_share = fig.add_subplot(grid[0, index])
        ax_absolute = ax_share.twinx()
        ax_absolute.patch.set_visible(False)
        ax_absolute.set_zorder(ax_share.get_zorder() + 1)

        ax_share.bar(
            item["centres"][item["bar_visible"]],
            item["drone_share"][item["bar_visible"]],
            width=style["bar_width_km"],
            color=style["bar_color"],
            edgecolor=paper_color("ink.background"),
            linewidth=0.25,
            zorder=2,
        )
        ax_absolute.scatter(
            item["centres"][item["visible"]],
            item["reduction_abs"][item["visible"]],
            s=style["scatter_size"],
            color=style["scatter_color"],
            alpha=style["scatter_alpha"],
            linewidth=0,
            zorder=5,
        )
        ax_absolute.plot(
            item["fit_x"],
            item["fit_y"],
            color=style["fit_color"],
            linewidth=style["fit_linewidth"],
            alpha=style["fit_alpha"],
            zorder=6,
        )
        ax_absolute.axhline(
            0.0,
            color=style["zero_line_color"],
            linewidth=style["zero_linewidth"],
            alpha=paper_alpha("city_gef_detour.alpha_0_3"),
            zorder=3,
        )

        ax_share.set_xlim(*style["x_domain_km"])
        ax_share.set_xticks(style["x_ticks_km"])
        ax_share.set_ylim(*style["bar_ylim_pct"])
        ax_share.set_yticks(style["bar_yticks_pct"])
        ax_share.yaxis.set_ticks_position("right")
        ax_share.tick_params(
            axis="both",
            labelsize=style["tick_fontsize"],
            length=2.5,
            pad=1.5,
        )
        ax_share.tick_params(axis="y", labelcolor=paper_color("city_gef_detour.3182bd"))
        ax_share.grid(
            True,
            axis="y",
            alpha=style["grid_alpha"],
            linewidth=0.45,
        )
        ax_share.set_axisbelow(True)
        ax_share.spines["top"].set_visible(False)
        ax_share.spines["left"].set_visible(False)
        if index != len(prepared) - 1:
            ax_share.tick_params(axis="y", right=False, labelright=False)
            ax_share.spines["right"].set_visible(False)

        ax_absolute.set_ylim(*shared_absolute_ylim)
        ax_absolute.yaxis.set_ticks_position("left")
        ax_absolute.tick_params(
            axis="y",
            labelsize=style["tick_fontsize"],
            labelcolor=style["scatter_color"],
            length=2.5,
            pad=1.5,
        )
        ax_absolute.tick_params(axis="x", bottom=False, labelbottom=False)
        ax_absolute.spines["top"].set_visible(False)
        ax_absolute.spines["right"].set_visible(False)
        ax_absolute.spines["bottom"].set_visible(False)
        if index != 0:
            ax_absolute.tick_params(axis="y", left=False, labelleft=False)
            ax_absolute.spines["left"].set_visible(False)

        ax_share.set_title(
            context["display_name"],
            fontsize=style["city_label_fontsize"],
            fontweight="bold",
            pad=5.0,
        )
        plot_records.append(
            {
                **item,
                "share_ax": ax_share,
                "absolute_ax": ax_absolute,
            }
        )

    fig.text(
        0.018,
        0.53,
        "E-bike $-$ Drone (g CO$_2$e/order)",
        rotation=90,
        fontsize=style["axis_label_fontsize"],
        ha="center",
        va="center",
        color=style["scatter_color"],
    )
    fig.text(
        0.982,
        0.53,
        "Optimal drone share (%)",
        rotation=90,
        fontsize=style["axis_label_fontsize"],
        ha="center",
        va="center",
        color=paper_color("city_gef_detour.3182bd"),
    )
    fig.text(
        0.5,
        0.06,
        "Air distance (km)",
        fontsize=style["axis_label_fontsize"],
        ha="center",
        va="center",
    )
    if plot_title:
        fig.suptitle(
            "Distance Response of Drone Share and Lifecycle Emissions",
            fontsize=9,
            fontweight="bold",
            y=0.98,
        )

    output_path = save_path
    if output_path:
        if for_latex and not output_path.endswith(".pdf"):
            output_path = output_path.rsplit(".", 1)[0] + ".pdf"
        if plot_title:
            output_path = _add_title_suffix(output_path)
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        fig.canvas.draw()
        fig.savefig(
            output_path,
            dpi=300,
            bbox_inches=fig.bbox_inches,
            pad_inches=0.0,
        )
        city_reports = []
        for item in plot_records:
            visible_indices = np.flatnonzero(item["visible"])
            bar_indices = np.flatnonzero(item["bar_visible"])
            binned = item["binned"]
            city_reports.append(
                {
                    "region_key": item["context"]["region_key"],
                    "display_name": item["context"]["display_name"],
                    "gef_kg_co2e_per_kwh": _report_number(
                        item["context"].get("gef")
                    ),
                    "valid_path_count": int(
                        binned.get("valid_path_count", 0)
                    ),
                    "invalid_path_count": int(
                        binned.get("invalid_path_count", 0)
                    ),
                    "merchant_filter": binned.get("merchant_filter"),
                    "axes": {
                        "optimal_drone_share": _report_axis(
                            item["share_ax"]
                        ),
                        "ebike_minus_drone": _report_axis(
                            item["absolute_ax"]
                        ),
                    },
                    "bars": [
                        {
                            "x_air_km": _report_number(
                                item["centres"][bin_index]
                            ),
                            "width_km": style["bar_width_km"],
                            "optimal_drone_share_pct": _report_number(
                                item["drone_share"][bin_index]
                            ),
                        }
                        for bin_index in bar_indices
                    ],
                    "scatter_points": [
                        {
                            "x_air_km": _report_number(
                                item["centres"][bin_index]
                            ),
                            "ebike_minus_drone_g_co2e_per_order": (
                                _report_number(
                                    item["reduction_abs"][bin_index]
                                )
                            ),
                        }
                        for bin_index in visible_indices
                    ],
                    "linear_fit": {
                        "weighting": "equal weight per visible distance bin",
                        "slope_g_co2e_per_order_per_km": _report_number(
                            item["fit_slope"]
                        ),
                        "intercept_g_co2e_per_order": _report_number(
                            item["fit_intercept"]
                        ),
                        "line_points": [
                            {
                                "x_air_km": _report_number(x),
                                "y_g_co2e_per_order": _report_number(y),
                            }
                            for x, y in zip(item["fit_x"], item["fit_y"])
                        ],
                    },
                }
            )
        report = {
            "schema_version": 1,
            "figure_id": "city_distance_drone_share_emission_fit",
            "core_conclusion": (
                "Optimal drone share and the signed e-bike minus drone "
                "lifecycle-emission difference generally rise with air "
                "distance, with heterogeneous levels and slopes across cities."
            ),
            "figure_size_inches": [float(figsize[0]), float(figsize[1])],
            "archetype": "quantitative_grid",
            "city_order": region_keys,
            "data_contract": {
                "bin_width_km": 0.2,
                "distance_domain_km": style["x_domain_km"],
                "within_bin_aggregation": "sales weighted",
                "fit_weighting": "equal weight per visible distance bin",
                "fit_line_domain": "minimum to maximum visible bin center",
                "merchant_filter": (
                    "independently exclude each city's merchants with "
                    "road-snap distance above the city P99"
                ),
            },
            "shared_absolute_ylim": shared_absolute_ylim,
            "style": style,
            "cities": city_reports,
            "omitted_elements": [
                "power mix bars",
                "panel letters",
                "legend",
                "fit equation",
                "R-squared",
                "confidence interval",
            ] + ([] if plot_title else ["title"]),
        }
        report_path = output_path.rsplit(".", 1)[0] + "_report.json"
        write_figure_report(report_path, report)
        print(f"City distance response report saved to {report_path}")
        print(f"City distance response saved to {output_path}")
    else:
        plt.show()
    plt.close(fig)
    return output_path


def _render_city_distance_drone_share_emission_fit(
    data_dir,
    save_path,
    for_latex,
    plot_title,
    city_df,
    exp_config,
    figure_name="city_distance_drone_share_emission_fit",
):
    """Build and render the standalone four-city distance response figure."""
    base_contexts = CITY_GEF_CONTEXT_DETOUR_CONTEXTS
    region_keys = [context["region_key"] for context in base_contexts]
    selected = city_df[city_df["region_key"].isin(region_keys)].copy()
    selected["_context_order"] = selected["region_key"].map(
        {key: index for index, key in enumerate(region_keys)}
    )
    selected = selected.sort_values("_context_order")
    selected_keys = selected["region_key"].astype(str).tolist()
    missing_keys = [key for key in region_keys if key not in selected_keys]
    if missing_keys:
        raise ValueError(
            f"{figure_name} missing cohort cities: " + ", ".join(missing_keys)
        )

    gef_by_key = dict(
        zip(
            selected["region_key"],
            pd.to_numeric(selected["gef"], errors="coerce"),
        )
    )
    missing_gef = [key for key in region_keys if pd.isna(gef_by_key.get(key))]
    if missing_gef:
        raise ValueError(
            f"{figure_name} missing city GEF: " + ", ".join(missing_gef)
        )
    contexts = tuple(
        {
            **context,
            "gef": float(gef_by_key[context["region_key"]]),
        }
        for context in sorted(
            base_contexts,
            key=lambda item: (
                float(gef_by_key[item["region_key"]]),
                item["region_key"],
            ),
        )
    )
    built = _build_reduction_detour_binned_by_city(
        data_dir=data_dir,
        selected=selected,
        figure_name=figure_name,
        exp_config=exp_config,
        absolute_reduction_metric="per_order_g",
        use_binned_cache=True,
        bin_width_km=0.2,
        cache_namespace="city_gef_context_detour_summary",
    )
    if built is None:
        raise RuntimeError(f"{figure_name} could not build city distance bins")
    binned_by_city, _ = built
    missing_binned = [
        context["region_key"]
        for context in contexts
        if context["region_key"] not in binned_by_city
    ]
    if missing_binned:
        raise RuntimeError(
            f"{figure_name} missing binned city data: "
            + ", ".join(missing_binned)
        )
    return plot_city_distance_drone_share_emission_fit(
        binned_by_city=binned_by_city,
        contexts=contexts,
        save_path=save_path,
        for_latex=for_latex,
        plot_title=plot_title,
    )


def _add_city_detour_reduction_markers(
    fig,
    axes,
    df,
    contexts,
    reduction_col,
    panel_letters=("f", "g"),
    marker_size=28,
):
    """Add selected city markers to the bottom detour-reduction scatter panels."""
    ax_low, _, ax_high = axes
    feature_var = "detour_ratio"
    group_var = "grid_emission_factor"
    threshold = 0.3
    axis_by_group = {
        "low_gef": ax_low,
        "high_gef": ax_high,
    }
    panel_letter_by_group = {
        "low_gef": panel_letters[0],
        "high_gef": panel_letters[1],
    }
    marker_records = []

    needed = [feature_var, reduction_col, group_var]
    if "region_key" not in df.columns or any(col not in df.columns for col in needed):
        return marker_records

    for context in contexts:
        region_key = context["region_key"]
        marker = CITY_CONTEXT_MARKERS.get(region_key, "o")
        city_rows = df[df["region_key"] == region_key].copy()
        if city_rows.empty:
            continue

        for col in needed:
            city_rows[col] = pd.to_numeric(city_rows[col], errors="coerce")
        city_rows = city_rows.dropna(subset=needed)
        if city_rows.empty:
            continue

        row = city_rows.iloc[0]
        x = float(row[feature_var])
        y = float(row[reduction_col])
        gef = float(row[group_var])
        axis_name = "low_gef" if gef <= threshold else "high_gef"
        ax = axis_by_group[axis_name]
        ax.scatter(
            [x],
            [y],
            s=marker_size,
            marker=marker,
            facecolors="none",
            edgecolors=paper_color("city_gef_detour.111111"),
            linewidths=1.1,
            zorder=35,
            clip_on=False,
        )
        display_point = ax.transData.transform((x, y))
        fig_point = fig.transFigure.inverted().transform(display_point)
        marker_records.append(
            {
                "region_key": region_key,
                "display_name": context["display_name"],
                "axis_name": axis_name,
                "panel_letter": panel_letter_by_group[axis_name],
                "x_detour_ratio": _report_number(x),
                "y_reduction_pct": _report_number(y),
                "grid_emission_factor": _report_number(gef),
                "marker": marker,
                "x_figure_fraction": _report_number(fig_point[0]),
                "y_figure_fraction": _report_number(fig_point[1]),
            }
        )

    return marker_records


def _detour_reduction_scatter_report(
    df,
    reduction_col,
    axes,
    panel_letters=("f", "g"),
    compact=False,
    grayscale=False,
    scatter_size=None,
    scatter_alpha=None,
    fit_linewidth=None,
    show_panel_letters=True,
):
    """Return exact plotted data for the bottom detour-reduction scatter panels."""
    import numpy as np
    from scipy import stats

    ax_low, cbar_ax, ax_high = axes
    feature_var = "detour_ratio"
    group_var = "grid_emission_factor"
    threshold = 0.3
    valid_mask = df[[feature_var, reduction_col, group_var]].notna().all(axis=1)
    plot_df = df.loc[valid_mask].copy()
    low_mask = plot_df[group_var] <= threshold
    masks = [low_mask, ~low_mask]
    labels = (f"Low GEF ($\\leq$ {threshold})", f"High GEF (> {threshold})")
    line_colors = (
        [paper_color("city_gef_detour.4d4d4d"), paper_color("city_gef_detour.4d4d4d")]
        if grayscale
        else [paper_color("city_gef_detour.006837"), paper_color("city_gef_detour.a50f15")]
    )
    axis_names = ["low_gef", "high_gef"]
    axis_objs = [ax_low, ax_high]

    x_data_min = plot_df[feature_var].min()
    x_data_max = plot_df[feature_var].max()
    y_data_min = plot_df[reduction_col].min()
    y_data_max = plot_df[reduction_col].max()
    x_pad = (x_data_max - x_data_min) * 0.03 or 0.1
    y_pad = (y_data_max - y_data_min) * 0.06 or 1.0
    x_global_min = x_data_min - x_pad
    x_global_max = x_data_max + x_pad

    panels = []
    for axis_name, ax, mask, label, color, panel_letter in zip(
        axis_names, axis_objs, masks, labels, line_colors, panel_letters
    ):
        panel_df = plot_df.loc[mask]
        points = []
        for _, row in panel_df.iterrows():
            points.append(
                {
                    "region_key": _report_number(row.get("region_key")),
                    "region_name_en": _report_number(row.get("region_name_en")),
                    "region_name": _report_number(row.get("region_name")),
                    "x_detour_ratio": _report_number(row[feature_var]),
                    "y_reduction_pct": _report_number(row[reduction_col]),
                    "grid_emission_factor": _report_number(row[group_var]),
                }
            )

        regression = None
        x = panel_df[feature_var].to_numpy(dtype=float)
        y = panel_df[reduction_col].to_numpy(dtype=float)
        if len(x) >= 3 and np.sum((x - np.mean(x)) ** 2) > 0:
            slope, intercept, r_value, p_value, stderr = stats.linregress(x, y)
            x_line = np.linspace(x_global_min, x_global_max, 100)
            y_line = slope * x_line + intercept
            n = len(x)
            se = np.sqrt(np.sum((y - (slope * x + intercept)) ** 2) / (n - 2))
            t_val = stats.t.ppf(0.975, n - 2)
            x_mean = np.mean(x)
            ci = t_val * se * np.sqrt(
                1.0 / n + (x_line - x_mean) ** 2 / np.sum((x - x_mean) ** 2)
            )
            regression = {
                "slope": _report_number(slope),
                "intercept": _report_number(intercept),
                "r_value": _report_number(r_value),
                "p_value": _report_number(p_value),
                "stderr": _report_number(stderr),
                "line_color": color,
                "line_points": [
                    {
                        "x": _report_number(xv),
                        "y": _report_number(yv),
                        "ci_low": _report_number(yv - civ),
                        "ci_high": _report_number(yv + civ),
                    }
                    for xv, yv, civ in zip(x_line, y_line, ci)
                ],
            }

        panels.append(
            {
                "panel_letter": panel_letter,
                "axis_name": axis_name,
                "label_text": label,
                "sample_size": int(len(panel_df)),
                "axes": _report_axis(ax),
                "panel_label": {
                    "text": panel_letter if show_panel_letters else None,
                    "x_axes_fraction": 0.0 if compact else -0.10,
                    "y_axes_fraction": 1.01 if compact else 1.02,
                },
                "group_label": {
                    "text": label,
                    "x_axes_fraction": 0.03,
                    "y_axes_fraction": 0.96,
                },
                "scatter_style": {
                    "size": (
                        scatter_size
                        if scatter_size is not None
                        else CITY_GEF_BOTTOM_SCATTER_SIZE
                        if compact
                        else 24
                    ),
                    "alpha": (
                        scatter_alpha
                        if scatter_alpha is not None
                        else CITY_GEF_GRAYSCALE_BOTTOM_SCATTER_ALPHA
                        if grayscale and compact
                        else CITY_GEF_BOTTOM_SCATTER_ALPHA
                        if compact
                        else paper_alpha("city_gef_detour.alpha_0_68")
                    ),
                    "edgecolors": "none",
                    "zorder": 10,
                },
                "points": points,
                "regression": regression,
                "fit_linewidth": (
                    fit_linewidth
                    if fit_linewidth is not None
                    else 1.35
                    if compact
                    else 2.5
                ),
            }
        )

    return {
        "axes": {
            "colorbar": _report_axis(cbar_ax),
        },
        "feature_var": feature_var,
        "result_var": reduction_col,
        "group_var": group_var,
        "threshold": threshold,
        "panels": panels,
    }


@prepared_plot()
def plot_city_detour_reduction_by_gef(
    df: pd.DataFrame,
    contexts=CITY_GEF_CONTEXT_DETOUR_CONTEXTS,
    save_path: str = None,
    figsize: tuple = (7.8, 2.6),
    for_latex: bool = False,
    plot_title: bool = False,
):
    """Plot city detour-reduction relationships split by grid intensity."""
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.colors import to_hex

    from drone_emission.visualization._utils import (
        _add_title_suffix,
        write_figure_report,
    )
    from drone_emission.visualization.city_gef_support import (
        _plot_grouped_scatter_split_panels_axes,
        _weighted_reduction_column,
    )

    feature_var = "detour_ratio"
    group_var = "grid_emission_factor"
    threshold = 0.3
    required = {"region_key", feature_var, group_var}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(
            "city_detour_reduction_by_gef missing columns: "
            + ", ".join(missing)
        )

    plot_df = df.copy()
    reduction_col = _weighted_reduction_column(plot_df)
    numeric_columns = [feature_var, reduction_col, group_var]
    for column in numeric_columns:
        plot_df[column] = pd.to_numeric(plot_df[column], errors="coerce")
    finite_mask = np.isfinite(plot_df[numeric_columns]).all(axis=1)
    valid_df = plot_df.loc[finite_mask].copy()
    if valid_df.empty:
        raise ValueError("city_detour_reduction_by_gef has no valid city rows")

    context_keys = [str(context["region_key"]) for context in contexts]
    expected_keys = set(CITY_CONTEXT_MARKERS)
    if len(context_keys) != 4 or set(context_keys) != expected_keys:
        raise ValueError(
            "city_detour_reduction_by_gef requires exactly Chengdu, "
            "Guangzhou, Beijing, and Shanghai contexts"
        )
    valid_keys = set(valid_df["region_key"].astype(str))
    missing_city_keys = sorted(expected_keys.difference(valid_keys))
    if missing_city_keys:
        raise ValueError(
            "city_detour_reduction_by_gef missing valid selected cities: "
            + ", ".join(missing_city_keys)
        )

    group_masks = {
        "low_gef": valid_df[group_var] <= threshold,
        "high_gef": valid_df[group_var] > threshold,
    }
    for group_name, mask in group_masks.items():
        group = valid_df.loc[mask]
        if len(group) < 3 or group[feature_var].nunique() < 2:
            raise ValueError(
                "city_detour_reduction_by_gef requires at least three rows "
                f"and two distinct detour ratios in {group_name}"
            )

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [
                "Arial",
                "Helvetica",
                "DejaVu Sans",
                "sans-serif",
            ],
            "pdf.fonttype": 42,
            "axes.linewidth": 0.8,
        }
    )
    fig = plt.figure(figsize=figsize)
    root = fig.add_gridspec(
        1,
        1,
        left=0.085,
        right=0.965,
        bottom=0.19,
        top=0.90,
    )
    axes = _plot_grouped_scatter_split_panels_axes(
        fig=fig,
        subplot_spec=root[0],
        df=valid_df,
        feature_var=feature_var,
        result_var=reduction_col,
        group_var=group_var,
        threshold=threshold,
        plot_title=False,
        panel_letters=(None, None),
        compact=True,
        horizontal_colorbar=False,
        colorbar_boundaries=CITY_GEF_COLOR_BIN_EDGES,
        colorbar_label="kg CO$_2$e/kWh",
        scatter_size=20,
        scatter_alpha=paper_alpha("city_gef_detour.opacity_0_45"),
        fit_linewidth=1.5,
        show_panel_letters=False,
        axis_label_fontsize=8,
        tick_fontsize=7.8,
        group_label_fontsize=8,
        colorbar_fontsize=8,
    )
    if axes is None:
        plt.close(fig)
        raise RuntimeError("city_detour_reduction_by_gef could not draw panels")
    ax_low, cbar_ax, ax_high = axes
    ax_low.set_ylabel("Optimal lifecycle reduction (%)", fontsize=8)

    panel_bbox = ax_low.get_position()
    cbar_bbox = cbar_ax.get_position()
    target_height = panel_bbox.height * 0.72
    target_y0 = panel_bbox.y0 + (panel_bbox.height - target_height) / 2.0
    cbar_ax.set_position(
        [cbar_bbox.x0, target_y0, cbar_bbox.width, target_height]
    )
    cbar_ax.set_title("GEF", fontsize=8, pad=3)
    cbar_ax.yaxis.labelpad = 3

    fig.canvas.draw()
    selected_city_markers = _add_city_detour_reduction_markers(
        fig=fig,
        axes=axes,
        df=valid_df,
        contexts=contexts,
        reduction_col=reduction_col,
        panel_letters=(None, None),
        marker_size=34,
    )
    if len(selected_city_markers) != 4:
        plt.close(fig)
        raise RuntimeError(
            "city_detour_reduction_by_gef could not draw all four city markers"
        )

    if plot_title:
        fig.suptitle(
            "Detour and Lifecycle Reduction by Grid Emission Factor",
            fontsize=9,
            fontweight="bold",
        )

    output_path = save_path
    if output_path:
        if for_latex and not output_path.endswith(".pdf"):
            output_path = output_path.rsplit(".", 1)[0] + ".pdf"
        if plot_title:
            output_path = _add_title_suffix(output_path)
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        fig.canvas.draw()
        fig.savefig(
            output_path,
            dpi=300,
            bbox_inches=fig.bbox_inches,
            pad_inches=0.0,
        )

        panel_report = _detour_reduction_scatter_report(
            valid_df,
            reduction_col,
            axes,
            panel_letters=(None, None),
            compact=True,
            scatter_size=20,
            scatter_alpha=paper_alpha("city_gef_detour.opacity_0_45"),
            fit_linewidth=1.5,
            show_panel_letters=False,
        )
        cmap = ax_low.collections[0].cmap
        colorbar_bbox = _report_axis(cbar_ax)
        report = {
            "schema_version": 1,
            "figure_id": "city_detour_reduction_by_gef",
            "core_conclusion": (
                "Detour ratio is positively associated with lifecycle "
                "emission reduction in both GEF groups, while high-GEF "
                "cities are shifted toward lower reduction at a given "
                "detour ratio."
            ),
            "archetype": "quantitative_grid",
            "figure_size_inches": [float(figsize[0]), float(figsize[1])],
            "data_contract": {
                "feature_variable": feature_var,
                "result_variable": reduction_col,
                "group_variable": group_var,
                "threshold_kg_co2e_per_kwh": threshold,
                "low_group_rule": "grid_emission_factor <= 0.3",
                "high_group_rule": "grid_emission_factor > 0.3",
                "input_city_count": int(len(plot_df)),
                "valid_city_count": int(len(valid_df)),
                "excluded_city_count": int(len(plot_df) - len(valid_df)),
                "exclusion_rule": (
                    "exclude rows with non-finite detour ratio, weighted "
                    "lifecycle reduction, or grid emission factor"
                ),
                "selected_city_detour_source": (
                    "OD-derived sales-weighted road-to-air ratio using the "
                    "city_gef_context_detour_summary cache contract"
                ),
            },
            "shared_axes": {
                "xlim_data": _report_array(ax_low.get_xlim()),
                "ylim_data": _report_array(ax_low.get_ylim()),
                "right_y_ticks_and_label_hidden": True,
            },
            "panels": panel_report["panels"],
            "colorbar": {
                "placement": "center_between_low_and_high_gef_panels",
                "orientation": "vertical",
                "title": "GEF",
                "side_label": "kg CO2e/kWh",
                "boundaries": _report_array(CITY_GEF_COLOR_BIN_EDGES),
                "colors_low_to_high": [
                    to_hex(cmap(index)) for index in range(cmap.N)
                ],
                "height_fraction_of_panel": _report_number(
                    cbar_ax.get_position().height / ax_low.get_position().height
                ),
                "axes": colorbar_bbox,
            },
            "selected_city_markers": selected_city_markers,
            "style": {
                "ordinary_scatter_size_pt2": 20.0,
                "ordinary_scatter_alpha": paper_alpha("city_gef_detour.opacity_0_45"),
                "selected_city_marker_size_pt2": 34.0,
                "selected_city_marker_face": "none",
                "selected_city_marker_edge": paper_color("city_gef_detour.111111"),
                "selected_city_marker_edge_width_pt": 1.1,
                "fit_linewidth_pt": 1.5,
                "confidence_level": 0.95,
                "confidence_band_alpha": paper_alpha("city_gef_detour.opacity_0_11"),
                "ordinary_text_size_pt": 8.0,
            },
            "omitted_elements": [
                "panel letters",
                "selected-city legend",
                "fit equation",
                "R-squared",
                "p-value annotation",
            ] + ([] if plot_title else ["title"]),
        }
        report_path = output_path.rsplit(".", 1)[0] + "_report.json"
        write_figure_report(report_path, report)
        print(f"City detour-reduction by GEF report saved to {report_path}")
        print(f"City detour-reduction by GEF saved to {output_path}")
    else:
        plt.show()
    plt.close(fig)
    return output_path


def _render_city_detour_reduction_by_gef(
    data_dir,
    save_path,
    for_latex,
    plot_title,
    df_enriched,
    city_df,
    exp_config,
    figure_name="city_detour_reduction_by_gef",
):
    """Build and render the standalone GEF-split detour-reduction figure."""
    expected_cohort_keys = set(city_df["region_key"].astype(str))
    enriched_keys = set(df_enriched["region_key"].astype(str))
    if (
        enriched_keys != expected_cohort_keys
        or len(df_enriched) != len(expected_cohort_keys)
    ):
        raise RuntimeError(
            f"{figure_name} enriched statistics do not match the locked "
            "paper cohort"
        )

    contexts = CITY_GEF_CONTEXT_DETOUR_CONTEXTS
    region_keys = [context["region_key"] for context in contexts]
    selected = city_df[city_df["region_key"].isin(region_keys)].copy()
    selected_keys = set(selected["region_key"].astype(str))
    missing_keys = sorted(set(region_keys).difference(selected_keys))
    if missing_keys:
        raise ValueError(
            f"{figure_name} missing cohort cities: " + ", ".join(missing_keys)
        )

    built = _build_reduction_detour_binned_by_city(
        data_dir=data_dir,
        selected=selected,
        figure_name=figure_name,
        exp_config=exp_config,
        absolute_reduction_metric="per_order_g",
        use_binned_cache=True,
        bin_width_km=0.2,
        cache_namespace="city_gef_context_detour_summary",
    )
    if built is None:
        raise RuntimeError(f"{figure_name} could not build city distance bins")
    binned_by_city, _ = built
    missing_binned = sorted(set(region_keys).difference(binned_by_city))
    if missing_binned:
        raise RuntimeError(
            f"{figure_name} missing binned city data: "
            + ", ".join(missing_binned)
        )

    summary_df = df_enriched.copy()
    if "detour_ratio" not in summary_df:
        summary_df["detour_ratio"] = float("nan")
    for context in contexts:
        region_key = context["region_key"]
        city_detour = binned_by_city[region_key].get("city_detour_ratio")
        if city_detour is None or pd.isna(city_detour):
            raise ValueError(f"{figure_name}: {region_key} OD detour unavailable")
        city_mask = summary_df["region_key"].astype(str).eq(region_key)
        if not city_mask.any():
            raise ValueError(f"{figure_name}: {region_key} summary row missing")
        summary_df.loc[city_mask, "detour_ratio"] = float(city_detour)

    return plot_city_detour_reduction_by_gef(
        df=summary_df,
        contexts=contexts,
        save_path=save_path,
        for_latex=for_latex,
        plot_title=plot_title,
    )


def _write_city_gef_context_detour_report(
    save_path,
    figure_name,
    figure_size,
    top_report,
    city_reports,
    bottom_report,
    connector_lines,
    legend_rail=None,
    render_style="color",
    merchant_filters=None,
):
    """Write JSON and Markdown reports for the city GEF context figure."""
    report_base = save_path.rsplit(".", 1)[0] + "_report"
    json_path = report_base + ".json"
    md_path = report_base + ".md"
    payload = {
        "figure_name": figure_name,
        "figure_path": save_path,
        "render_style": render_style,
        "figure_size_inches": {
            "width": figure_size[0],
            "height": figure_size[1],
        },
        "coordinate_convention": (
            "All x/y values are in Matplotlib data coordinates for the stated axis; "
            "axis bbox values are figure-fraction coordinates."
        ),
        "panels": {
            "top_city_gef_scatter_components": top_report,
            "middle_city_context": city_reports,
            "bottom_detour_ratio_vs_reduction_by_gef": bottom_report,
        },
        "connector_lines": connector_lines,
        "legend_rail": legend_rail,
        "middle_analysis_domain_km": [
            MIDDLE_DISTANCE_MIN_KM,
            MIDDLE_DISTANCE_MAX_KM,
        ],
        "merchant_filters": merchant_filters or {},
    }
    if render_style == "grayscale":
        payload["visual_encoding"] = {
            "vehicle_colors": CITY_GEF_GRAYSCALE_VEHICLE_COLORS,
            "production_eol_hatch": "////",
            "use_phase_hatch": None,
            "power_mix_colors": POWER_MIX_GRAYSCALE_COLORS,
            "gef_bin_colors_low_to_high": CITY_GEF_GRAYSCALE_BIN_COLORS,
            "gef_gray_direction": "higher GEF uses darker gray",
            "regression_linestyle": "solid",
        }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, allow_nan=False)

    merchant_filter_summaries = {
        region_key: {
            key: value
            for key, value in merchant_filter.items()
            if key != "distance_bins"
        }
        for region_key, merchant_filter in (merchant_filters or {}).items()
    }
    lines = [
        f"# {figure_name} Figure Report",
        "",
        f"- Figure path: `{save_path}`",
        f"- JSON data path: `{json_path}`",
        f"- Render style: `{render_style}`",
        f"- Figure size: {figure_size[0]} x {figure_size[1]} inches",
        "- Coordinate convention: data coordinates for plotted elements; figure-fraction coordinates for axes.",
        f"- Connector line groups: {len(connector_lines)}",
        f"- Legend rail orientation: {(legend_rail or {}).get('orientation')}",
        f"- GEF bin edges: {(legend_rail or {}).get('gef_bin_edges')}",
        f"- City marker entries: {(legend_rail or {}).get('city_markers')}",
        f"- Common middle-panel analysis domain: {MIDDLE_DISTANCE_MIN_KM}-{MIDDLE_DISTANCE_MAX_KM} km bin centres",
        f"- Merchant filters: {merchant_filter_summaries}",
        "",
        "## Per-city merchant filter audit",
        "",
    ]
    if not merchant_filters:
        lines.append("- Not applied.")
    else:
        for region_key, merchant_filter in merchant_filters.items():
            lines.extend(
                [
                    f"### {region_key}",
                    "",
                    f"- Rule: {merchant_filter.get('rule')}",
                    f"- Quantile: {merchant_filter.get('quantile')}",
                    f"- Quantile method: {merchant_filter.get('quantile_method')}",
                    f"- Actual P99: {merchant_filter.get('snap_distance_p99_m')} m",
                    f"- Merchants: {merchant_filter.get('merchant_count')} total, "
                    f"{merchant_filter.get('excluded_merchant_count')} excluded",
                    f"- Filtered OD rows: {merchant_filter.get('filtered_od_count')}",
                    f"- Filtered annual sales: {merchant_filter.get('filtered_annual_sales')}",
                    "",
                    "| Air-distance bin (km) | State | Merchants | OD rows | Annual sales | Optimal drone share (%) | Sales-weighted E-bike - Drone (g/order) | Lifecycle reduction (%) |",
                    "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
                ]
            )
            for bin_record in merchant_filter.get("distance_bins", []):
                bin_label = (
                    f"{bin_record['bin_left_air_km']:.1f}-"
                    f"{bin_record['bin_right_air_km']:.1f}"
                )
                for state_name in ("before", "after"):
                    state = bin_record[state_name]
                    lines.append(
                        f"| {bin_label} | {state_name} | "
                        f"{state['merchant_count']} | {state['od_count']} | "
                        f"{state['annual_sales']} | "
                        f"{state['optimal_drone_share_pct']} | "
                        f"{state['sales_weighted_ebike_minus_drone_g_per_order']} | "
                        f"{state['lifecycle_reduction_pct']} |"
                    )
            lines.append("")
    lines.extend(
        [
        "",
        "## Top: city_gef_scatter_components",
        "",
        f"- GEF column: `{top_report.get('gef_column')}`",
        f"- Bin edges: {top_report.get('bin_edges')}",
        f"- Bar count: {len(top_report.get('bars', []))}",
        f"- Density points: {len(top_report.get('density_curve', []))}",
        f"- City rug marks: {len(top_report.get('city_rug_marks', []))}",
        f"- Selected city GEF markers: {len(top_report.get('selected_city_gef_markers', []))}",
        "",
        "Each bar entry in JSON records `x_center`, `width`, production+EOL segment bottom/height/top, use-phase segment bottom/height/top, and total label location.",
        "",
        "## Middle: selected_city_context",
        "",
        ]
    )
    for city in city_reports:
        visible_bins = [
            item for item in city["distance_bins"]
            if item["visible_in_panel"]
        ]
        lines.extend(
            [
                f"### Panel {city['panel_letter']}: {city['display_name']}",
                "",
                f"- Region key: `{city['region_key']}`",
                f"- OD-derived detour ratio: {city['city_detour_ratio']}",
                f"- Valid/invalid paths: {city['valid_path_count']}/{city['invalid_path_count']}",
                f"- Common fit/display domain: {city['middle_distance_domain_km']} km bin centres",
                f"- Visible distance-bin count: {len(visible_bins)}",
                f"- E-bike - Drone linear slope: {(city.get('absolute_linear_fit') or {}).get('slope')} g CO2e/order/km",
                f"- Power mix components: {', '.join(item['component'] for item in city['power_mix'])}",
                f"- Bar metric: `{city['bar_metric']}`",
                "- JSON records every distance bin with air-distance x center, "
                "plotted bar height, optimal drone share, reduction metrics, "
                "absolute scatter y value, and compact detour-ratio summary "
                "statistics.",
                "",
            ]
        )
    lines.extend(
        [
            "## Bottom: scatter_detour_ratio_vs_reduction_by_gef",
            "",
            f"- Feature variable: `{bottom_report['feature_var']}`",
            f"- Result variable: `{bottom_report['result_var']}`",
            f"- Group variable: `{bottom_report['group_var']}`",
            f"- Threshold: {bottom_report['threshold']}",
            f"- Selected city scatter markers: {len(bottom_report.get('selected_city_scatter_markers', []))}",
            f"- Major city labels: {len(bottom_report.get('major_city_labels', []))}",
        ]
    )
    for panel in bottom_report["panels"]:
        lines.extend(
            [
                f"- Panel {panel['panel_letter']} ({panel['axis_name']}): {len(panel['points'])} scatter points.",
            ]
        )
    lines.extend(
        [
            "",
            "The JSON file is the authoritative element-level record for writing: it includes every scatter point coordinate, every bar x/height/bottom/top, fitted line points, confidence intervals, axes limits, tick positions, and axes locations.",
            "",
        ]
    )
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"City GEF context detour summary report saved to {md_path}")
    print(f"City GEF context detour summary report data saved to {json_path}")
    return md_path, json_path


def _render_city_gef_context_detour_summary(
    data_dir,
    out,
    for_latex,
    plot_title,
    df_enriched,
    city_df,
    exp_config,
    figure_name="city_gef_context_detour_summary",
    grayscale=False,
):
    """Render GEF components, selected city contexts, and detour-reduction scatter."""
    contexts = CITY_GEF_CONTEXT_DETOUR_CONTEXTS
    region_keys = [context["region_key"] for context in contexts]
    selected = city_df[city_df["region_key"].isin(region_keys)].copy()
    selected["_context_order"] = selected["region_key"].map(
        {key: i for i, key in enumerate(region_keys)}
    )
    selected = selected.sort_values("_context_order")
    gef_by_key = dict(
        zip(
            selected["region_key"],
            pd.to_numeric(selected["gef"], errors="coerce"),
        )
    )
    context_by_key = {
        context["region_key"]: context for context in contexts
    }

    def _context_gef_sort_key(context):
        region_key = context["region_key"]
        gef = gef_by_key.get(region_key)
        return (
            float("inf") if gef is None or pd.isna(gef) else float(gef),
            region_key,
        )

    contexts = tuple(
        sorted(
            context_by_key.values(),
            key=_context_gef_sort_key,
        )
    )
    if selected.empty:
        print(f"{figure_name} skipped — selected city rows missing")
        return

    built = _build_reduction_detour_binned_by_city(
        data_dir=data_dir,
        selected=selected,
        figure_name=figure_name,
        exp_config=exp_config,
        absolute_reduction_metric="per_order_g",
        use_binned_cache=True,
        bin_width_km=0.2,
    )
    if built is None:
        return

    binned_by_city, labels = built
    available_contexts = [
        context for context in contexts
        if context["region_key"] in binned_by_city
    ]
    missing_names = [
        context["display_name"] for context in contexts
        if context["region_key"] not in binned_by_city
    ]
    if missing_names:
        print(
            f"{figure_name}: missing binned data for "
            f"{', '.join(missing_names)}"
        )
    if not available_contexts:
        print(f"{figure_name} skipped — no selected city binned data available")
        return

    import matplotlib.pyplot as plt
    from matplotlib.colors import to_hex

    from drone_emission.visualization._utils import _add_title_suffix
    from drone_emission.visualization.city_gef_support import (
        _plot_city_gef_scatter_components_axes,
        _plot_grouped_scatter_split_panels_axes,
        _weighted_reduction_column,
    )

    summary_df = df_enriched.copy()
    if "detour_ratio" not in summary_df.columns:
        summary_df["detour_ratio"] = float("nan")
    for context in available_contexts:
        region_key = context["region_key"]
        city_detour = binned_by_city[region_key].get("city_detour_ratio")
        if city_detour is None or pd.isna(city_detour):
            print(f"{figure_name}: {region_key} OD detour unavailable")
            continue
        city_mask = summary_df["region_key"] == region_key
        existing = pd.to_numeric(
            summary_df.loc[city_mask, "detour_ratio"], errors="coerce"
        ).dropna()
        if not existing.empty and abs(float(existing.iloc[0]) - city_detour) > 1e-6:
            print(
                f"{figure_name}: {region_key} OD detour {city_detour:.6f} "
                f"differs from stats {float(existing.iloc[0]):.6f}"
            )
        summary_df.loc[city_mask, "detour_ratio"] = city_detour

    reduction_col = _weighted_reduction_column(summary_df)

    figure_size = (7.8, 8.4)
    fig = plt.figure(figsize=figure_size)
    outer = fig.add_gridspec(
        3,
        1,
        height_ratios=[1.0, 1.0, 1.0],
        hspace=CITY_GEF_SUMMARY_ROW_HSPACE,
        left=0.09,
        right=0.81,
        bottom=0.055,
        top=0.985,
    )
    legend_rail = _create_city_gef_summary_legend_rail(
        fig,
        contexts=available_contexts,
        row_hspace=CITY_GEF_SUMMARY_ROW_HSPACE,
        grayscale=grayscale,
    )

    top_axes = _plot_city_gef_scatter_components_axes(
        fig=fig,
        subplot_spec=outer[0],
        df=summary_df,
        plot_title=plot_title,
        compact=True,
        show_legend=False,
        grayscale=grayscale,
    )
    if top_axes is None:
        plt.close(fig)
        return
    ax_top, _ = top_axes
    top_pos = ax_top.get_position()
    fig.text(
        top_pos.x0 + 0.004,
        top_pos.y1 - 0.004,
        "a",
        fontsize=9,
        fontweight="bold",
        ha="left",
        va="bottom",
    )
    top_city_gef_markers = _add_city_gef_xaxis_markers(
        fig,
        top_axes,
        selected,
        available_contexts,
    )

    city_panel_records = _plot_compact_city_context_summary_panels(
        fig=fig,
        subplot_spec=outer[1],
        binned_by_city=binned_by_city,
        labels=labels,
        contexts=available_contexts,
        panel_start_letter="b",
        legend_ax=legend_rail["middle"],
        right_label_offset=0.035,
        grayscale=grayscale,
    )

    bottom_start_ord = ord("b") + len(available_contexts)
    bottom_panel_letters = (
        chr(bottom_start_ord),
        chr(bottom_start_ord + 1),
    )
    bottom_axes = _plot_grouped_scatter_split_panels_axes(
        fig=fig,
        subplot_spec=outer[2],
        df=summary_df,
        feature_var="detour_ratio",
        result_var=reduction_col,
        group_var="grid_emission_factor",
        threshold=0.3,
        plot_title=False,
        panel_letters=bottom_panel_letters,
        compact=True,
        horizontal_colorbar=False,
        external_colorbar_ax=legend_rail["colorbar"],
        colorbar_boundaries=CITY_GEF_COLOR_BIN_EDGES,
        colorbar_label="kg CO$_2$e/kWh",
        scatter_size=CITY_GEF_BOTTOM_SCATTER_SIZE,
        scatter_alpha=(
            CITY_GEF_GRAYSCALE_BOTTOM_SCATTER_ALPHA
            if grayscale
            else CITY_GEF_BOTTOM_SCATTER_ALPHA
        ),
        grayscale=grayscale,
    )
    if bottom_axes is None:
        plt.close(fig)
        return
    bottom_axes[0].set_ylabel("Optimal lifecycle reduction (%)")
    bottom_city_markers = _add_city_detour_reduction_markers(
        fig,
        bottom_axes,
        summary_df,
        available_contexts,
        reduction_col,
        panel_letters=bottom_panel_letters,
    )
    bottom_major_city_labels = []
    connector_lines = []

    if plot_title:
        fig.suptitle(
            "City Grid Intensity, Context, and Detour-reduction Relationships",
            fontsize=9,
        )

    save_path = os.path.join(out, f"{figure_name}.png")
    if for_latex and not save_path.endswith(".pdf"):
        save_path = save_path.rsplit(".", 1)[0] + ".pdf"
    if plot_title:
        save_path = _add_title_suffix(save_path)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.canvas.draw()
    fig.savefig(save_path, dpi=300, bbox_inches=fig.bbox_inches, pad_inches=0.0)
    top_report = _city_gef_components_report(
        summary_df,
        top_axes,
        grayscale=grayscale,
    )
    top_report["selected_city_gef_markers"] = top_city_gef_markers
    city_reports = _city_context_summary_report(
        binned_by_city,
        labels,
        city_panel_records,
        bar_metric="oracle_drone_share_pct",
        grayscale=grayscale,
    )
    bottom_report = _detour_reduction_scatter_report(
        summary_df,
        reduction_col,
        bottom_axes,
        panel_letters=bottom_panel_letters,
        compact=True,
        grayscale=grayscale,
    )
    bottom_report["selected_city_scatter_markers"] = bottom_city_markers
    bottom_report["major_city_labels"] = bottom_major_city_labels
    gef_cmap = bottom_axes[0].collections[0].cmap
    legend_rail_report = {
        "orientation": "vertical",
        "sections": {
            "top_vehicle_component": _report_axis(legend_rail["top"]),
            "middle_power_mix": _report_axis(legend_rail["middle"]),
            "bottom_gef": _report_axis(legend_rail["bottom"]),
            "gef_colorbar": _report_axis(legend_rail["colorbar"]),
        },
        "gef_bin_edges": _report_array(CITY_GEF_COLOR_BIN_EDGES),
        "gef_bin_colors": [to_hex(gef_cmap(index)) for index in range(gef_cmap.N)],
        "city_markers": legend_rail["city_entries"],
        "render_style": "grayscale" if grayscale else "color",
        "production_eol_hatch": "////" if grayscale else None,
        "use_phase_hatch": None,
    }
    _write_city_gef_context_detour_report(
        save_path=save_path,
        figure_name=figure_name,
        figure_size=figure_size,
        top_report=top_report,
        city_reports=city_reports,
        bottom_report=bottom_report,
        connector_lines=connector_lines,
        legend_rail=legend_rail_report,
        render_style="grayscale" if grayscale else "color",
        merchant_filters={
            region_key: binned["merchant_filter"]
            for region_key, binned in binned_by_city.items()
            if binned.get("merchant_filter") is not None
        },
    )
    plt.close(fig)
    print(f"City GEF context detour summary saved to {save_path}")
