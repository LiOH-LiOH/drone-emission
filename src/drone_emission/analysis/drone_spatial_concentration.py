"""Analyze the spatial concentration of drone-favorable delivery orders.

The primary metric is the Multi-scale Pilot Capture Score (MPCS): the mean
maximum annual-order share captured by one circular pilot area with radii of
2, 3, and 5 km. An order is captured only when both its merchant origin and
customer destination are inside the circle.

Usage:
    cd src/drone_emission
    ../../.venv/bin/python -m analysis.drone_spatial_concentration \
        --env remote --workers 8

This module only computes and persists the city-level concentration cache.
Render figures from that cache with ``tools.plot_drone_spatial_concentration``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from pyproj import Transformer
from scipy.ndimage import maximum_filter
from scipy.signal import fftconvolve
from scipy.spatial import cKDTree

from utils import load_config
from utils.tools import get_epsg_from_bounds


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RADII_KM = (2.0, 3.0, 5.0)
SENSITIVITY_MARGINS_PCT = (0.0, 5.0, 10.0)
GRID_RESOLUTION_M = 500.0
REFINEMENT_STEPS_M = (250.0, 100.0, 50.0)
DEFAULT_HOTSPOTS = 32
EXPANDED_HOTSPOTS = 128
CONVERGENCE_TOLERANCE_PCT = 0.5
MAX_SCREENING_CELLS = 4_000_000
ABSOLUTE_REDUCTION_COLUMN = "oracle_absolute_reduction_t_co2e"
RELATIVE_REDUCTION_COLUMN = "oracle_relative_reduction_pct"
SUPER_LARGE_CITIES = (
    "Beijing",
    "Shanghai",
    "Shenzhen",
    "Chongqing",
    "Guangzhou",
    "Chengdu",
    "Tianjin",
)

REQUIRED_OD_COLUMNS = (
    "from_lat",
    "from_lng",
    "to_lat",
    "to_lng",
    "annual_sales",
    "drone_lifecycle_g",
    "ebike_lifecycle_g",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finite_numeric(df: pd.DataFrame, columns: Iterable[str]) -> np.ndarray:
    return np.column_stack(
        [pd.to_numeric(df[column], errors="coerce").to_numpy() for column in columns]
    )


def _eligible_mask(od_df: pd.DataFrame, minimum_reduction_pct: float) -> np.ndarray:
    """Return valid, positive-weight OD rows meeting an emission advantage."""
    missing = [column for column in REQUIRED_OD_COLUMNS if column not in od_df]
    if missing:
        raise ValueError(f"OD data missing required columns: {missing}")

    coordinates = _finite_numeric(
        od_df, ("from_lat", "from_lng", "to_lat", "to_lng")
    )
    weights = pd.to_numeric(od_df["annual_sales"], errors="coerce").to_numpy()
    drone = pd.to_numeric(od_df["drone_lifecycle_g"], errors="coerce").to_numpy()
    ebike = pd.to_numeric(od_df["ebike_lifecycle_g"], errors="coerce").to_numpy()
    valid = (
        np.isfinite(coordinates).all(axis=1)
        & np.isfinite(weights)
        & (weights > 0)
        & np.isfinite(drone)
        & np.isfinite(ebike)
        & (ebike > 0)
    )
    reduction_pct = np.full(len(od_df), np.nan, dtype=float)
    reduction_pct[valid] = (ebike[valid] - drone[valid]) / ebike[valid] * 100.0
    if minimum_reduction_pct <= 0:
        return valid & (reduction_pct > 0)
    return valid & (reduction_pct >= minimum_reduction_pct)


def _project_od(od_df: pd.DataFrame, eligible: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    selected = od_df.loc[eligible]
    bounds = {
        "min_lat": float(
            min(selected["from_lat"].min(), selected["to_lat"].min())
        ),
        "max_lat": float(
            max(selected["from_lat"].max(), selected["to_lat"].max())
        ),
        "min_lng": float(
            min(selected["from_lng"].min(), selected["to_lng"].min())
        ),
        "max_lng": float(
            max(selected["from_lng"].max(), selected["to_lng"].max())
        ),
    }
    epsg = get_epsg_from_bounds(bounds)
    transformer = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
    origin_x, origin_y = transformer.transform(
        selected["from_lng"].to_numpy(dtype=float),
        selected["from_lat"].to_numpy(dtype=float),
    )
    destination_x, destination_y = transformer.transform(
        selected["to_lng"].to_numpy(dtype=float),
        selected["to_lat"].to_numpy(dtype=float),
    )
    return (
        np.column_stack([origin_x, origin_y]),
        np.column_stack([destination_x, destination_y]),
    )


def _disk_kernel(radius_cells: int) -> np.ndarray:
    offsets = np.arange(-radius_cells, radius_cells + 1)
    xx, yy = np.meshgrid(offsets, offsets)
    return ((xx * xx + yy * yy) <= radius_cells * radius_cells).astype(float)


def _screen_hotspots(
    midpoints: np.ndarray,
    weights: np.ndarray,
    radius_m: float,
    max_hotspots: int,
) -> tuple[np.ndarray, float]:
    """Screen promising centers using a weighted midpoint raster convolution."""
    span = np.maximum(np.ptp(midpoints, axis=0), GRID_RESOLUTION_M)
    resolution = GRID_RESOLUTION_M
    estimated_cells = np.prod(np.ceil(span / resolution) + 3)
    if estimated_cells > MAX_SCREENING_CELLS:
        resolution *= math.sqrt(estimated_cells / MAX_SCREENING_CELLS)

    minimum = midpoints.min(axis=0) - resolution
    cell_indices = np.floor((midpoints - minimum) / resolution).astype(int)
    shape = tuple((cell_indices.max(axis=0) + 2).astype(int)[::-1])
    raster = np.zeros(shape, dtype=float)
    np.add.at(raster, (cell_indices[:, 1], cell_indices[:, 0]), weights)

    radius_cells = max(1, int(math.ceil(radius_m / resolution)))
    screened = fftconvolve(raster, _disk_kernel(radius_cells), mode="same")
    local_maximum = screened == maximum_filter(
        screened, size=2 * radius_cells + 1, mode="constant"
    )
    peak_rows, peak_cols = np.nonzero(local_maximum & (screened > 0))
    if peak_rows.size:
        order = np.argsort(screened[peak_rows, peak_cols])[::-1][:max_hotspots]
        peak_rows = peak_rows[order]
        peak_cols = peak_cols[order]
        candidates = np.column_stack(
            [
                minimum[0] + (peak_cols + 0.5) * resolution,
                minimum[1] + (peak_rows + 0.5) * resolution,
            ]
        )
    else:
        candidates = np.empty((0, 2), dtype=float)

    centroid = np.average(midpoints, axis=0, weights=weights).reshape(1, 2)
    candidates = np.vstack([candidates, centroid])
    return np.unique(np.round(candidates, decimals=6), axis=0), resolution


def _captured_weight(
    center: np.ndarray,
    radius_m: float,
    midpoint_tree: cKDTree,
    origins: np.ndarray,
    destinations: np.ndarray,
    weights: np.ndarray,
) -> float:
    candidate_indices = midpoint_tree.query_ball_point(center, radius_m)
    if not candidate_indices:
        return 0.0
    indices = np.asarray(candidate_indices, dtype=int)
    radius_sq = radius_m * radius_m
    origin_delta = origins[indices] - center
    destination_delta = destinations[indices] - center
    inside = (
        np.einsum("ij,ij->i", origin_delta, origin_delta) <= radius_sq
    ) & (
        np.einsum("ij,ij->i", destination_delta, destination_delta) <= radius_sq
    )
    return float(weights[indices][inside].sum())


def _refine_center(
    initial_center: np.ndarray,
    radius_m: float,
    midpoint_tree: cKDTree,
    origins: np.ndarray,
    destinations: np.ndarray,
    weights: np.ndarray,
) -> tuple[np.ndarray, float, float]:
    center = np.asarray(initial_center, dtype=float)
    best_weight = _captured_weight(
        center, radius_m, midpoint_tree, origins, destinations, weights
    )
    weight_at_100m = best_weight
    directions = np.array(
        [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1), (0, 0), (0, 1),
            (1, -1), (1, 0), (1, 1),
        ],
        dtype=float,
    )
    for step_m in REFINEMENT_STEPS_M:
        for _ in range(12):
            trial_centers = center + directions * step_m
            trial_weights = np.array(
                [
                    _captured_weight(
                        trial,
                        radius_m,
                        midpoint_tree,
                        origins,
                        destinations,
                        weights,
                    )
                    for trial in trial_centers
                ]
            )
            best_index = int(np.argmax(trial_weights))
            if trial_weights[best_index] <= best_weight:
                break
            center = trial_centers[best_index]
            best_weight = float(trial_weights[best_index])
        if step_m == 100.0:
            weight_at_100m = best_weight
    return center, best_weight, weight_at_100m


def _best_capture_for_radius(
    origins: np.ndarray,
    destinations: np.ndarray,
    weights: np.ndarray,
    radius_km: float,
    total_weight: float,
    max_hotspots: int = DEFAULT_HOTSPOTS,
) -> dict:
    radius_m = radius_km * 1000.0
    endpoint_distance_sq = np.einsum(
        "ij,ij->i", destinations - origins, destinations - origins
    )
    feasible = endpoint_distance_sq <= (2.0 * radius_m) ** 2
    if not feasible.any():
        return {
            "capture_pct": 0.0,
            "center_x_m": np.nan,
            "center_y_m": np.nan,
            "screening_resolution_m": GRID_RESOLUTION_M,
            "refinement_delta_pct": 0.0,
            "hotspot_count": 0,
        }

    feasible_origins = origins[feasible]
    feasible_destinations = destinations[feasible]
    feasible_weights = weights[feasible]
    midpoints = (feasible_origins + feasible_destinations) / 2.0
    midpoint_tree = cKDTree(midpoints)
    candidates, screening_resolution = _screen_hotspots(
        midpoints, feasible_weights, radius_m, max_hotspots
    )

    best_center = candidates[0]
    best_weight = -1.0
    best_weight_at_100m = -1.0
    for candidate in candidates:
        center, captured, captured_at_100m = _refine_center(
            candidate,
            radius_m,
            midpoint_tree,
            feasible_origins,
            feasible_destinations,
            feasible_weights,
        )
        if captured > best_weight:
            best_center = center
            best_weight = captured
            best_weight_at_100m = captured_at_100m

    capture_pct = best_weight / total_weight * 100.0
    refinement_delta_pct = (
        best_weight - best_weight_at_100m
    ) / total_weight * 100.0
    if (
        refinement_delta_pct > CONVERGENCE_TOLERANCE_PCT
        and max_hotspots < EXPANDED_HOTSPOTS
    ):
        return _best_capture_for_radius(
            origins,
            destinations,
            weights,
            radius_km,
            total_weight,
            max_hotspots=EXPANDED_HOTSPOTS,
        )
    return {
        "capture_pct": float(capture_pct),
        "center_x_m": float(best_center[0]),
        "center_y_m": float(best_center[1]),
        "screening_resolution_m": float(screening_resolution),
        "refinement_delta_pct": float(refinement_delta_pct),
        "hotspot_count": int(len(candidates)),
    }


def compute_city_pilot_capture(
    od_df: pd.DataFrame,
    radii_km: tuple[float, ...] = DEFAULT_RADII_KM,
    minimum_reduction_pct: float = 0.0,
) -> dict:
    """Compute sales-weighted pilot capture metrics for one city.

    Args:
        od_df: City OD data with coordinates, annual sales, and lifecycle emissions.
        radii_km: Circular pilot-area radii in kilometres.
        minimum_reduction_pct: Required drone reduction relative to e-bike.

    Returns:
        Dictionary containing eligible demand and capture metrics.
    """
    eligible = _eligible_mask(od_df, minimum_reduction_pct)
    valid_weights = (
        pd.to_numeric(od_df["annual_sales"], errors="coerce")
        .replace([np.inf, -np.inf], 0)
        .fillna(0)
    )
    total_orders = float(valid_weights.clip(lower=0).sum())
    eligible_weights = valid_weights.to_numpy(dtype=float)[eligible]
    eligible_orders = float(eligible_weights.sum())
    result = {
        "minimum_reduction_pct": float(minimum_reduction_pct),
        "eligible_od_count": int(eligible.sum()),
        "eligible_annual_orders": eligible_orders,
        "total_annual_orders": total_orders,
        "eligible_order_share_pct": (
            eligible_orders / total_orders * 100.0 if total_orders > 0 else np.nan
        ),
    }
    if eligible_orders <= 0:
        for radius_km in radii_km:
            key = f"capture_{int(radius_km)}km"
            result[f"{key}_pct"] = np.nan
            result[f"{key}_center_x_m"] = np.nan
            result[f"{key}_center_y_m"] = np.nan
            result[f"{key}_refinement_delta_pct"] = np.nan
        result["pilot_concentration_score_pct"] = np.nan
        return result

    origins, destinations = _project_od(od_df, eligible)
    capture_values = []
    for radius_km in radii_km:
        capture = _best_capture_for_radius(
            origins,
            destinations,
            eligible_weights,
            radius_km,
            eligible_orders,
        )
        key = f"capture_{int(radius_km)}km"
        result[f"{key}_pct"] = capture["capture_pct"]
        result[f"{key}_center_x_m"] = capture["center_x_m"]
        result[f"{key}_center_y_m"] = capture["center_y_m"]
        result[f"{key}_screening_resolution_m"] = capture[
            "screening_resolution_m"
        ]
        result[f"{key}_refinement_delta_pct"] = capture[
            "refinement_delta_pct"
        ]
        result[f"{key}_hotspot_count"] = capture["hotspot_count"]
        capture_values.append(capture["capture_pct"])
    result["pilot_concentration_score_pct"] = float(np.mean(capture_values))
    return result


def _city_od_path(data_dir: Path, row: dict) -> Path:
    region_key = str(row["region_key"])
    region_level = str(row.get("region_level", "city"))
    return data_dir / region_level / region_key / f"{region_key}_od_pairs.csv"


def _analyze_city(task: tuple[dict, str]) -> dict:
    row, data_dir_text = task
    started = time.perf_counter()
    data_dir = Path(data_dir_text)
    od_path = _city_od_path(data_dir, row)
    base = {
        "region_key": row.get("region_key"),
        "region_name_cn": row.get("region_name_cn"),
        "region_name_en": row.get("region_name_en"),
        "region_level": row.get("region_level"),
        "od_path": str(od_path),
    }
    if not od_path.exists():
        return {
            **base,
            "status": "skipped",
            "skip_reason": "missing_od_file",
            "runtime_seconds": time.perf_counter() - started,
        }
    try:
        od_df = pd.read_csv(
            od_path, usecols=lambda column: column in REQUIRED_OD_COLUMNS
        )
        missing = [column for column in REQUIRED_OD_COLUMNS if column not in od_df]
        if missing:
            raise ValueError(f"missing columns: {missing}")
        scenario_results = {
            margin: compute_city_pilot_capture(
                od_df, minimum_reduction_pct=margin
            )
            for margin in SENSITIVITY_MARGINS_PCT
        }
        primary = scenario_results[0.0]
        if not np.isfinite(primary["pilot_concentration_score_pct"]):
            return {
                **base,
                "status": "skipped",
                "skip_reason": "no_drone_favorable_orders",
                "eligible_od_count": primary["eligible_od_count"],
                "eligible_annual_orders": primary["eligible_annual_orders"],
                "total_annual_orders": primary["total_annual_orders"],
                "eligible_order_share_pct": primary["eligible_order_share_pct"],
                "od_file_size_bytes": od_path.stat().st_size,
                "runtime_seconds": time.perf_counter() - started,
            }
        result = {
            **base,
            "status": "ok",
            "skip_reason": "",
            **primary,
            "od_file_size_bytes": od_path.stat().st_size,
        }
        for margin in (5.0, 10.0):
            suffix = f"margin_{int(margin)}pct"
            scenario = scenario_results[margin]
            result[f"pilot_concentration_score_{suffix}"] = scenario[
                "pilot_concentration_score_pct"
            ]
            result[f"eligible_annual_orders_{suffix}"] = scenario[
                "eligible_annual_orders"
            ]
            for radius in DEFAULT_RADII_KM:
                key = f"capture_{int(radius)}km_pct"
                result[f"{key}_{suffix}"] = scenario[key]
        result["runtime_seconds"] = time.perf_counter() - started
        return result
    except Exception as exc:
        return {
            **base,
            "status": "skipped",
            "skip_reason": f"{type(exc).__name__}: {exc}",
            "runtime_seconds": time.perf_counter() - started,
        }


def _assign_classes_and_representatives(results: pd.DataFrame) -> pd.DataFrame:
    output = results.copy()
    output["concentration_class"] = ""
    output["pilot_typology"] = ""
    output["is_representative"] = False
    output["representative_proxy"] = False
    output["representative_typology"] = ""
    valid = output["status"].eq("ok") & output[
        "pilot_concentration_score_pct"
    ].notna()
    successful = output.loc[valid].copy()
    if successful.empty:
        return output

    score = successful["pilot_concentration_score_pct"]
    percentile = score.rank(method="average", pct=True)
    successful["concentration_class"] = np.select(
        [percentile <= 1 / 3, percentile <= 2 / 3],
        ["Low", "Medium"],
        default="High",
    )
    score_median = float(score.median())
    volume_median = float(successful["eligible_annual_orders"].median())
    concentration_side = np.where(
        score >= score_median, "High concentration", "Low concentration"
    )
    volume_side = np.where(
        successful["eligible_annual_orders"] >= volume_median,
        "High volume",
        "Low volume",
    )
    successful["pilot_typology"] = [
        f"{left} / {right}" for left, right in zip(concentration_side, volume_side)
    ]

    score_scale = max(float(score.quantile(0.75) - score.quantile(0.25)), 1e-9)
    log_volume = np.log10(successful["eligible_annual_orders"].clip(lower=1.0))
    volume_scale = max(
        float(log_volume.quantile(0.75) - log_volume.quantile(0.25)), 1e-9
    )
    chosen_indices: set[int] = set()
    typologies = (
        "High concentration / High volume",
        "High concentration / Low volume",
        "Low concentration / High volume",
        "Low concentration / Low volume",
    )
    for typology in typologies:
        unused = ~successful.index.isin(chosen_indices)
        candidates = successful[successful["pilot_typology"].eq(typology) & unused]
        proxy = False
        if candidates.empty:
            candidates = successful.loc[unused]
            if candidates.empty:
                break
            proxy = True
            target_score = score.quantile(
                0.75 if typology.startswith("High") else 0.25
            )
            target_volume = log_volume.quantile(
                0.75 if typology.endswith("High volume") else 0.25
            )
        else:
            target_score = candidates["pilot_concentration_score_pct"].median()
            target_volume = np.log10(
                candidates["eligible_annual_orders"].clip(lower=1.0)
            ).median()
        candidate_log_volume = np.log10(
            candidates["eligible_annual_orders"].clip(lower=1.0)
        )
        distance = np.sqrt(
            ((candidates["pilot_concentration_score_pct"] - target_score) / score_scale) ** 2
            + ((candidate_log_volume - target_volume) / volume_scale) ** 2
        )
        ordered = candidates.assign(_distance=distance).sort_values(
            ["_distance", "region_key"]
        )
        representative_index = int(ordered.index[0])
        chosen_indices.add(representative_index)
        successful.loc[representative_index, "is_representative"] = True
        successful.loc[representative_index, "representative_proxy"] = proxy
        successful.loc[representative_index, "representative_typology"] = typology

    assignment_columns = [
        "concentration_class",
        "pilot_typology",
        "is_representative",
        "representative_proxy",
        "representative_typology",
    ]
    output.loc[successful.index, assignment_columns] = successful[
        assignment_columns
    ]
    return output


def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(value) else float(value)
    if pd.isna(value):
        return None
    return value


def load_representative_payloads(
    results: pd.DataFrame,
    data_dir: str | Path,
) -> list[dict]:
    """Load eligible OD rows for cached representative-city metrics."""
    data_dir = Path(data_dir)
    payloads = []
    representatives = results[results["is_representative"].eq(True)]
    for _, row in representatives.iterrows():
        od_path = _city_od_path(data_dir, row.to_dict())
        od_df = pd.read_csv(od_path, usecols=list(REQUIRED_OD_COLUMNS))
        eligible = _eligible_mask(od_df, 0.0)
        payloads.append({"metrics": row.to_dict(), "od_df": od_df.loc[eligible].copy()})
    return payloads


def _write_csv_atomic(frame: pd.DataFrame, path: Path) -> None:
    """Write a CSV without exposing a partial concentration cache."""
    temporary = path.with_name(f".{path.name}.tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def _write_json_atomic(payload: dict, path: Path) -> None:
    """Write JSON without exposing a partial analysis report."""
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as file_obj:
        json.dump(_json_safe(payload), file_obj, ensure_ascii=False, indent=2)
        file_obj.write("\n")
    temporary.replace(path)


def _attach_oracle_reduction_metrics(
    results: pd.DataFrame,
    city_stats: pd.DataFrame,
) -> pd.DataFrame:
    """Attach comparable absolute and relative Oracle reductions to results."""
    required = {
        "region_key",
        "total_ebike_lifecycle",
        "oracle_min_lifecycle_kg",
        "oracle_min_reduction_vs_ebike_pct",
    }
    missing = sorted(required.difference(city_stats.columns))
    if missing:
        raise ValueError(f"City statistics missing reduction columns: {missing}")

    metrics = city_stats[list(required)].copy()
    ebike_kg = pd.to_numeric(metrics["total_ebike_lifecycle"], errors="coerce")
    oracle_kg = pd.to_numeric(metrics["oracle_min_lifecycle_kg"], errors="coerce")
    metrics[ABSOLUTE_REDUCTION_COLUMN] = (ebike_kg - oracle_kg) / 1_000.0
    metrics[RELATIVE_REDUCTION_COLUMN] = pd.to_numeric(
        metrics["oracle_min_reduction_vs_ebike_pct"], errors="coerce"
    )
    metrics = metrics[
        ["region_key", ABSOLUTE_REDUCTION_COLUMN, RELATIVE_REDUCTION_COLUMN]
    ]
    return results.merge(metrics, on="region_key", how="left", validate="one_to_one")


def run_drone_spatial_concentration(
    stats_path: str | Path,
    data_dir: str | Path,
    results_dir: str | Path,
    workers: int | None = None,
    force: bool = False,
) -> dict:
    """Compute and persist the all-city spatial concentration cache."""
    started = time.perf_counter()
    stats_path = Path(stats_path).resolve()
    data_dir = Path(data_dir).resolve()
    results_dir = Path(results_dir).resolve()
    csv_path = results_dir / "city_drone_spatial_concentration.csv"
    report_path = results_dir / "drone_spatial_concentration_report.json"
    if csv_path.exists() and not force:
        raise FileExistsError(
            f"Concentration cache already exists: {csv_path}. "
            "Use --force to recompute it."
        )
    results_dir.mkdir(parents=True, exist_ok=True)
    stats = pd.read_csv(stats_path)
    city_stats = stats[stats["region_level"].eq("city")].copy()
    city_stats = city_stats.sort_values("region_key").reset_index(drop=True)
    tasks = [(row, str(data_dir)) for row in city_stats.to_dict("records")]
    worker_count = workers or max(1, (os.cpu_count() or 2) - 1)

    rows = []
    if worker_count == 1:
        rows = [_analyze_city(task) for task in tasks]
    else:
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            future_map = {
                executor.submit(_analyze_city, task): task[0]["region_key"]
                for task in tasks
            }
            for future in as_completed(future_map):
                rows.append(future.result())
    results = pd.DataFrame(rows).sort_values("region_key").reset_index(drop=True)
    results = _assign_classes_and_representatives(results)
    results = _attach_oracle_reduction_metrics(results, city_stats)

    _write_csv_atomic(results, csv_path)
    successful = results[results["status"].eq("ok")].copy()
    skipped = results[~results["status"].eq("ok")].copy()

    correlations = {}
    for margin in (5, 10):
        column = f"pilot_concentration_score_margin_{margin}pct"
        paired = successful[["pilot_concentration_score_pct", column]].dropna()
        correlations[str(margin)] = (
            float(paired.corr(method="spearman").iloc[0, 1])
            if len(paired) >= 3
            else None
        )
    runtime_seconds = time.perf_counter() - started
    scatter_sample = successful.copy()
    scatter_sample[ABSOLUTE_REDUCTION_COLUMN] = pd.to_numeric(
        scatter_sample[ABSOLUTE_REDUCTION_COLUMN], errors="coerce"
    )
    scatter_sample[RELATIVE_REDUCTION_COLUMN] = pd.to_numeric(
        scatter_sample[RELATIVE_REDUCTION_COLUMN], errors="coerce"
    )
    scatter_mask = (
        np.isfinite(scatter_sample[ABSOLUTE_REDUCTION_COLUMN])
        & (scatter_sample[ABSOLUTE_REDUCTION_COLUMN] > 0)
        & np.isfinite(scatter_sample[RELATIVE_REDUCTION_COLUMN])
    )
    scatter_excluded = scatter_sample.loc[~scatter_mask]
    scatter_sample = scatter_sample.loc[scatter_mask]
    labeled_cities = scatter_sample[
        scatter_sample["region_name_en"].isin(SUPER_LARGE_CITIES)
    ]
    report = {
        "schema_version": 2,
        "analysis": "drone_spatial_concentration",
        "metric": {
            "name": "Multi-scale Pilot Capture Score",
            "abbreviation": "MPCS",
            "radii_km": list(DEFAULT_RADII_KM),
            "aggregation": "equal arithmetic mean of capture percentages",
            "capture_rule": "both merchant origin and customer destination inside one circle",
            "weight": "annual_sales",
            "primary_eligibility": "drone_lifecycle_g < ebike_lifecycle_g",
            "sensitivity_minimum_reduction_pct": [5, 10],
        },
        "solver": {
            "screening_grid_resolution_m": GRID_RESOLUTION_M,
            "refinement_steps_m": list(REFINEMENT_STEPS_M),
            "initial_hotspots": DEFAULT_HOTSPOTS,
            "expanded_hotspots": EXPANDED_HOTSPOTS,
            "convergence_tolerance_percentage_points": CONVERGENCE_TOLERANCE_PCT,
            "workers": worker_count,
        },
        "source": {
            "stats_path": str(stats_path),
            "stats_sha256": _sha256(stats_path),
            "data_dir": str(data_dir),
        },
        "coverage": {
            "configured_city_count": int(len(city_stats)),
            "successful_city_count": int(len(successful)),
            "skipped_city_count": int(len(skipped)),
            "skipped": skipped[["region_key", "skip_reason"]].to_dict("records"),
        },
        "classification": {
            "concentration": "sample percentile-rank tertiles",
            "typology": "sample medians of MPCS and eligible annual orders",
            "scatter_color": "MPCS sample percentile-rank tertiles",
            "representatives": successful.loc[
                successful["is_representative"].eq(True),
                [
                    "region_key",
                    "region_name_en",
                    "pilot_typology",
                    "representative_typology",
                    "representative_proxy",
                ],
            ].to_dict("records"),
        },
        "reduction_concentration_scatter": {
            "x": {
                "metric": ABSOLUTE_REDUCTION_COLUMN,
                "unit": "t CO2e per year",
                "scale": "log",
            },
            "y": {
                "metric": RELATIVE_REDUCTION_COLUMN,
                "unit": "percent",
            },
            "plotted_city_count": int(len(scatter_sample)),
            "excluded_scored_city_count": int(len(scatter_excluded)),
            "excluded_scored_cities": scatter_excluded[
                ["region_key", "region_name_en"]
            ].to_dict("records"),
            "median_absolute_reduction_t_co2e": (
                float(scatter_sample[ABSOLUTE_REDUCTION_COLUMN].median())
                if not scatter_sample.empty else None
            ),
            "median_relative_reduction_pct": (
                float(scatter_sample[RELATIVE_REDUCTION_COLUMN].median())
                if not scatter_sample.empty else None
            ),
            "labeled_super_large_cities": labeled_cities[
                [
                    "region_key",
                    "region_name_en",
                    ABSOLUTE_REDUCTION_COLUMN,
                    RELATIVE_REDUCTION_COLUMN,
                    "pilot_concentration_score_pct",
                    "concentration_class",
                ]
            ].to_dict("records"),
        },
        "sensitivity_spearman_vs_primary": correlations,
        "performance": {
            "runtime_seconds": runtime_seconds,
            "target_seconds": 900,
            "target_met": runtime_seconds <= 900,
        },
        "outputs": {
            "metrics_csv": str(csv_path),
            "report_json": str(report_path),
        },
    }
    _write_json_atomic(report, report_path)
    print(
        "Drone spatial concentration cities: "
        f"{len(successful)}/{len(city_stats)} successful; "
        f"runtime={runtime_seconds:.1f}s"
    )
    print(f"Drone spatial concentration report saved to {report_path}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze spatial concentration of drone-favorable city orders"
    )
    parser.add_argument("--env", default=None, help="Configuration environment")
    parser.add_argument("--stats", default=None, help="Region statistics CSV")
    parser.add_argument(
        "--results-dir",
        "--output-dir",
        dest="results_dir",
        default=None,
        help="Persistent concentration-cache directory",
    )
    parser.add_argument("--workers", type=int, default=None, help="Parallel city workers")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute and replace an existing concentration cache",
    )
    args = parser.parse_args()

    config = load_config(PROJECT_ROOT / "experiment.toml", env=args.env)
    data_dir = Path(config["data_dir"])
    stats_path = Path(args.stats) if args.stats else data_dir / "all_region_stats.csv"
    results_dir = (
        Path(args.results_dir)
        if args.results_dir
        else data_dir / "analysis" / "drone_spatial_concentration"
    )
    run_drone_spatial_concentration(
        stats_path=stats_path,
        data_dir=data_dir,
        results_dir=results_dir,
        workers=args.workers,
        force=args.force,
    )


if __name__ == "__main__":
    main()
