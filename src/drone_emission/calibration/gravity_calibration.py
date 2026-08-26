"""End-to-end Beijing joint gravity-model calibration workflow."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil

import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree
from scipy.special import logsumexp

from .gravity_model import (
    _group_observations,
    building_order_fit,
    CandidateCSR,
    CURRENT_PARAMETERS,
    clustered_covariance,
    evaluate_model,
    fit_parameters,
    merchant_spatial_fit,
    utility_features,
)


SCHEMA_VERSION = 1
DELIVERY_RADIUS_KM = 5.0
BUILDING_MATCH_RADIUS_M = 50.0
BEIJING_METRIC_CRS = "EPSG:32650"
REQUIRED_CORRECTED_COLUMNS = {
    "shop_id",
    "source_file",
    "pick_lng_gcj02",
    "pick_lat_gcj02",
    "deliver_lng_gcj02",
    "deliver_lat_gcj02",
    "pick_lng_wgs84",
    "pick_lat_wgs84",
    "deliver_lng_wgs84",
    "deliver_lat_wgs84",
}
DATE_COLUMNS = (
    "order_date",
    "date",
)
TIMESTAMP_COLUMNS = (
    "create_time",
    "created_at",
    "order_time",
    "delivery_time",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_shop_is_unseen(shop_id: object) -> bool:
    """Assign exactly one reproducible hash bucket in five to unseen shops."""
    digest = hashlib.sha256(str(shop_id).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % 5 == 0


def _extract_dates(orders: pd.DataFrame) -> pd.Series:
    for column in DATE_COLUMNS:
        if column in orders:
            values = pd.to_datetime(orders[column], errors="coerce")
            if values.notna().all():
                return values.dt.normalize()
    extracted = orders["source_file"].astype(str).str.extract(
        r"((?:19|20)\d{2}[-_]?(?:0[1-9]|1[0-2])[-_]?(?:0[1-9]|[12]\d|3[01]))",
        expand=False,
    ).str.replace("-", "", regex=False).str.replace("_", "", regex=False)
    values = pd.to_datetime(extracted, format="%Y%m%d", errors="coerce")
    if values.notna().all():
        return values.dt.normalize()
    for column in TIMESTAMP_COLUMNS:
        if column not in orders:
            continue
        raw = orders[column]
        numeric = pd.to_numeric(raw, errors="coerce")
        if numeric.notna().all():
            scale = float(numeric.abs().median())
            unit = "ms" if scale >= 1e11 else "s"
            values = pd.to_datetime(numeric, unit=unit, errors="coerce")
        else:
            values = pd.to_datetime(raw, errors="coerce")
        if values.notna().all():
            return values.dt.normalize()
    raise ValueError(
        "Could not derive every order date from a date column, source_file, or timestamp"
    )


def assign_evaluation_split(orders: pd.DataFrame) -> pd.Series:
    """Return mutually exclusive training, unseen-shop, and temporal labels."""
    dates = _extract_dates(orders)
    distinct = sorted(pd.unique(dates))
    if len(distinct) != 29:
        raise ValueError(f"Expected exactly 29 order dates, found {len(distinct)}")
    last_five = set(distinct[-5:])
    unseen = orders["shop_id"].map(stable_shop_is_unseen).to_numpy(dtype=bool)
    temporal = dates.isin(last_five).to_numpy() & ~unseen
    labels = np.full(len(orders), "train_seen_first_24_days", dtype=object)
    labels[unseen] = "test_unseen_shops_all_29_days"
    labels[temporal] = "test_seen_shops_last_5_days"
    return pd.Series(labels, index=orders.index, dtype="string")


def validate_corrected_inputs(
    orders_path: Path, buildings_path: Path, transform_path: Path
) -> dict:
    """Validate schemas and create immutable input fingerprints."""
    for path in (orders_path, buildings_path, transform_path):
        if not Path(path).is_file():
            raise FileNotFoundError(f"Calibration input does not exist: {path}")
    transform = json.loads(Path(transform_path).read_text(encoding="utf-8"))
    if transform.get("schema_version") != 1:
        raise ValueError("Unsupported coordinate-transform schema")
    raw_orders = transform.get("inputs", {}).get("orders", {})
    if raw_orders.get("file_count") != 29 or len(raw_orders.get("files", [])) != 29:
        raise ValueError("Coordinate transform is not tied to the audited 29-file set")
    return {
        "corrected_orders": {
            "path": str(Path(orders_path).resolve()),
            "sha256": sha256_file(orders_path),
        },
        "buildings": {
            "path": str(Path(buildings_path).resolve()),
            "sha256": sha256_file(buildings_path),
        },
        "coordinate_transform": {
            "path": str(Path(transform_path).resolve()),
            "sha256": sha256_file(transform_path),
            "raw_orders_sha256": raw_orders["sha256"],
        },
    }


def _load_orders(path: Path, transform_path: Path) -> pd.DataFrame:
    orders = pd.read_csv(path)
    missing = REQUIRED_CORRECTED_COLUMNS - set(orders)
    if missing:
        raise ValueError("Corrected orders are missing: " + ", ".join(sorted(missing)))
    transform = json.loads(Path(transform_path).read_text(encoding="utf-8"))
    expected_files = {row["path"] for row in transform["inputs"]["orders"]["files"]}
    if set(orders["source_file"].astype(str).unique()) != expected_files:
        raise ValueError("Corrected order source_file values do not match the transform")
    for column in REQUIRED_CORRECTED_COLUMNS - {"shop_id", "source_file"}:
        orders[column] = pd.to_numeric(orders[column], errors="coerce")
    orders["order_date"] = _extract_dates(orders)
    orders["split"] = assign_evaluation_split(orders)
    orders["_order_index"] = np.arange(len(orders), dtype=np.int64)
    return orders


def _stable_building_id(frame: pd.DataFrame) -> pd.Series:
    for column in ("building_id", "GBAID", "gba_id", "id", "fid", "osm_id"):
        if column in frame and frame[column].notna().all():
            return frame[column].astype(str)
    return pd.Series([f"source-row-{index:012d}" for index in range(len(frame))])


def _load_buildings(path: Path):
    import geopandas as gpd

    buildings = gpd.read_file(path)
    if buildings.empty or "geometry" not in buildings:
        raise ValueError("Building input contains no geometry")
    if "volume_m3" not in buildings:
        raise ValueError("Building input is missing raw volume_m3")
    if buildings.crs is None:
        raise ValueError("Building input has no CRS")
    geometry_valid = buildings.geometry.notna()
    excluded_invalid_geometry = int((~geometry_valid).sum())
    buildings = buildings.loc[geometry_valid].copy()
    volumes = pd.to_numeric(buildings["volume_m3"], errors="coerce")
    buildings["volume_m3"] = volumes.to_numpy(float)
    buildings["valid_volume"] = np.isfinite(volumes) & (volumes > 0)
    invalid_volume_count = int((~buildings["valid_volume"]).sum())
    buildings["stable_building_id"] = _stable_building_id(buildings).to_numpy()
    buildings = buildings.sort_values("stable_building_id", kind="mergesort").reset_index(drop=True)
    buildings["building_index"] = np.arange(len(buildings), dtype=np.int64)
    metric = buildings.to_crs(BEIJING_METRIC_CRS)
    metric_centroids = metric.geometry.centroid
    centroids = metric_centroids.to_crs("EPSG:4326")
    buildings = buildings.to_crs("EPSG:4326")
    buildings["centroid_lng_wgs84"] = centroids.x.to_numpy(float)
    buildings["centroid_lat_wgs84"] = centroids.y.to_numpy(float)
    metric["building_index"] = buildings["building_index"].to_numpy()
    metric["stable_building_id"] = buildings["stable_building_id"].to_numpy()
    metric["centroid_x"] = metric_centroids.x.to_numpy(float)
    metric["centroid_y"] = metric_centroids.y.to_numpy(float)
    metric["valid_volume"] = buildings["valid_volume"].to_numpy(bool)
    return buildings, metric, {
        "excluded_invalid_geometry": excluded_invalid_geometry,
        "invalid_volume_buildings": invalid_volume_count,
    }


def match_orders_to_buildings(orders: pd.DataFrame, buildings_metric) -> pd.DataFrame:
    """Match deliveries by containment first, then nearest boundary within 50 m."""
    import geopandas as gpd

    finite = np.isfinite(
        orders[["deliver_lng_wgs84", "deliver_lat_wgs84"]].to_numpy(float)
    ).all(axis=1)
    points = gpd.GeoDataFrame(
        orders.loc[finite, ["_order_index"]].copy(),
        geometry=gpd.points_from_xy(
            orders.loc[finite, "deliver_lng_wgs84"],
            orders.loc[finite, "deliver_lat_wgs84"],
        ),
        crs="EPSG:4326",
    ).to_crs(BEIJING_METRIC_CRS)
    right = buildings_metric[
        [
            "building_index", "stable_building_id", "centroid_x", "centroid_y",
            "valid_volume", "geometry",
        ]
    ]
    contained = gpd.sjoin(points, right, how="inner", predicate="intersects")
    contained["match_method"] = pd.Series(dtype="string")
    contained["match_distance_m"] = pd.Series(dtype=float)
    if not contained.empty:
        contained["centroid_distance_m"] = np.hypot(
            contained.geometry.x - contained["centroid_x"],
            contained.geometry.y - contained["centroid_y"],
        )
        contained = contained.sort_values(
            ["_order_index", "centroid_distance_m", "stable_building_id"],
            kind="mergesort",
        ).drop_duplicates("_order_index", keep="first")
        contained["match_method"] = "contains"
        contained["match_distance_m"] = 0.0
    matched_indices = set(contained["_order_index"].astype(int))
    unmatched = points.loc[~points["_order_index"].isin(matched_indices)]
    nearest = gpd.sjoin_nearest(
        unmatched, right, how="inner", max_distance=BUILDING_MATCH_RADIUS_M,
        distance_col="match_distance_m",
    )
    nearest["match_method"] = pd.Series(dtype="string")
    if not nearest.empty:
        nearest["centroid_distance_m"] = np.hypot(
            nearest.geometry.x - nearest["centroid_x"],
            nearest.geometry.y - nearest["centroid_y"],
        )
        nearest = nearest.sort_values(
            ["_order_index", "match_distance_m", "centroid_distance_m", "stable_building_id"],
            kind="mergesort",
        ).drop_duplicates("_order_index", keep="first")
        nearest["match_method"] = "nearest_within_50m"
    columns = [
        "_order_index", "building_index", "stable_building_id", "match_method",
        "match_distance_m", "valid_volume",
    ]
    return pd.concat([contained[columns], nearest[columns]], ignore_index=True)


def _candidate_query(tree: BallTree, latitude: float, longitude: float) -> tuple[np.ndarray, np.ndarray]:
    indices, angular = tree.query_radius(
        np.radians([[latitude, longitude]]),
        r=DELIVERY_RADIUS_KM / 6371.0,
        return_distance=True,
        sort_results=True,
    )
    indices = indices[0].astype(np.int64, copy=False)
    distances = angular[0] * 6371.0
    valid = np.isfinite(distances) & (distances > 0) & (distances <= DELIVERY_RADIUS_KM)
    return indices[valid], distances[valid]


def _query_shop_batch(
    tree: BallTree, rows: list[tuple[float, float]], workers: int
) -> list[tuple[np.ndarray, np.ndarray]]:
    def query(row):
        return _candidate_query(tree, row[0], row[1])

    if workers == 1:
        return [query(row) for row in rows]
    with ThreadPoolExecutor(max_workers=workers) as executor:
        return list(executor.map(query, rows))


def build_candidate_cache(
    shops: pd.DataFrame,
    buildings: pd.DataFrame,
    cache_dir: Path,
    workers: int = 1,
) -> dict:
    """Build two-pass memory-mapped CSR candidate attributes on disk."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    building_coords = np.radians(
        buildings[["centroid_lat_wgs84", "centroid_lng_wgs84"]].to_numpy(float)
    )
    tree = BallTree(building_coords, metric="haversine")
    indptr = np.zeros(len(shops) + 1, dtype=np.int64)
    coordinates = list(
        shops[["pick_lat_wgs84", "pick_lng_wgs84"]].itertuples(
            index=False, name=None
        )
    )
    batch_size = 256
    for batch_start in range(0, len(coordinates), batch_size):
        results = _query_shop_batch(
            tree, coordinates[batch_start:batch_start + batch_size], workers
        )
        for offset, (indices, _) in enumerate(results):
            index = batch_start + offset
            indptr[index + 1] = indptr[index] + len(indices)
    np.save(cache_dir / "candidate_indptr.npy", indptr)
    length = int(indptr[-1])
    arrays = {
        "candidate_building_index.npy": np.lib.format.open_memmap(
            cache_dir / "candidate_building_index.npy", mode="w+", dtype="int64", shape=(length,)
        ),
        "candidate_log_volume.npy": np.lib.format.open_memmap(
            cache_dir / "candidate_log_volume.npy", mode="w+", dtype="float64", shape=(length,)
        ),
        "candidate_log_distance.npy": np.lib.format.open_memmap(
            cache_dir / "candidate_log_distance.npy", mode="w+", dtype="float64", shape=(length,)
        ),
        "candidate_distance_km.npy": np.lib.format.open_memmap(
            cache_dir / "candidate_distance_km.npy", mode="w+", dtype="float64", shape=(length,)
        ),
    }
    volumes = buildings["volume_m3"].to_numpy(float)
    building_ids = buildings["building_index"].to_numpy(np.int64)
    for batch_start in range(0, len(coordinates), batch_size):
        results = _query_shop_batch(
            tree, coordinates[batch_start:batch_start + batch_size], workers
        )
        for offset, (indices, distances) in enumerate(results):
            index = batch_start + offset
            start, stop = int(indptr[index]), int(indptr[index + 1])
            arrays["candidate_building_index.npy"][start:stop] = building_ids[indices]
            arrays["candidate_log_volume.npy"][start:stop] = np.log(volumes[indices])
            arrays["candidate_log_distance.npy"][start:stop] = np.log(distances)
            arrays["candidate_distance_km.npy"][start:stop] = distances
    for array in arrays.values():
        array.flush()
    return {
        "shop_count": int(len(shops)),
        "candidate_pair_count": length,
        "maximum_candidates_per_shop": int(np.diff(indptr).max(initial=0)),
    }


def _candidate_positions(candidates: CandidateCSR, orders: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    positions = np.full(len(orders), -1, dtype=np.int64)
    actual_distance = np.full(len(orders), np.nan, dtype=float)
    for shop_index, group in orders.groupby("shop_index", sort=False):
        start, stop = candidates.bounds(int(shop_index))
        building_indices = np.asarray(candidates.building_index[start:stop])
        lookup = {int(value): start + offset for offset, value in enumerate(building_indices)}
        group_positions = group["building_index"].map(lookup).fillna(-1).to_numpy(np.int64)
        positions[group.index] = group_positions
        valid = group_positions >= 0
        actual_distance[group.index[valid]] = candidates.distance_km[group_positions[valid]]
    return positions, actual_distance


def prepare_calibration_data(
    orders_path: Path,
    buildings_path: Path,
    transform_path: Path,
    cache_dir: Path,
    input_fingerprints: dict,
    workers: int = 1,
) -> dict:
    """Match orders, build candidates, exclusions, splits, and sufficient statistics."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    orders = _load_orders(orders_path, transform_path)
    buildings, buildings_metric, building_quality = _load_buildings(buildings_path)
    buildings_metric[["building_index", "centroid_x", "centroid_y"]].to_csv(
        cache_dir / "building_centroids.csv.gz", index=False, compression="gzip"
    )
    candidate_buildings = buildings.loc[buildings["valid_volume"]].copy().reset_index(drop=True)
    coordinate_valid = np.isfinite(
        orders[[
            "pick_lng_wgs84", "pick_lat_wgs84", "deliver_lng_wgs84", "deliver_lat_wgs84"
        ]].to_numpy(float)
    ).all(axis=1)
    matches = match_orders_to_buildings(orders, buildings_metric)
    orders = orders.merge(matches, on="_order_index", how="left", validate="one_to_one")
    shop_rows = (
        orders.loc[coordinate_valid, ["shop_id", "pick_lng_wgs84", "pick_lat_wgs84"]]
        .sort_values("shop_id", key=lambda values: values.astype(str), kind="mergesort")
        .drop_duplicates("shop_id", keep="first")
        .reset_index(drop=True)
    )
    shop_rows["shop_index"] = np.arange(len(shop_rows), dtype=np.int64)
    orders = orders.merge(shop_rows[["shop_id", "shop_index"]], on="shop_id", how="left")
    candidate_summary = build_candidate_cache(
        shop_rows, candidate_buildings, cache_dir, workers=workers
    )
    candidates = CandidateCSR.load(cache_dir)
    matched = orders["building_index"].notna()
    work = orders.loc[matched & coordinate_valid].copy()
    work["valid_volume"] = work["valid_volume"].fillna(False).astype(bool)
    work["building_index"] = work["building_index"].astype(np.int64)
    work["shop_index"] = work["shop_index"].astype(np.int64)
    matched_orders = work.reset_index(drop=True)
    candidate_ready = matched_orders.loc[matched_orders["valid_volume"]].copy().reset_index(drop=True)
    positions, actual_distance = _candidate_positions(candidates, candidate_ready)
    candidate_ready["candidate_position"] = positions
    candidate_ready["actual_distance_km"] = actual_distance
    in_candidate = (positions >= 0) & np.isfinite(actual_distance) & (actual_distance > 0)
    valid = candidate_ready.loc[in_candidate].copy()
    observations = (
        valid.groupby(["split", "shop_index", "candidate_position"], observed=True, as_index=False)
        .size().rename(columns={"size": "count"})
    )
    observations.to_csv(cache_dir / "observations.csv.gz", index=False, compression="gzip")
    match_source = matched_orders[[
        "_order_index", "split", "shop_index", "building_index", "match_method",
        "match_distance_m", "valid_volume",
    ]]
    match_source.to_csv(cache_dir / "matched_orders.csv.gz", index=False, compression="gzip")
    shop_rows.to_csv(cache_dir / "shops.csv", index=False)
    exclusion = {
        "raw_corrected_orders": int(len(orders)),
        "excluded_invalid_coordinates": int((~coordinate_valid).sum()),
        "excluded_no_building_within_50m": int((coordinate_valid & ~matched).sum()),
        "orders_matched_within_50m": int((coordinate_valid & matched).sum()),
        "excluded_matched_invalid_building_volume": int((~matched_orders["valid_volume"]).sum()),
        "excluded_true_building_outside_5km_or_invalid_distance": int(len(candidate_ready) - in_candidate.sum()),
        "retained_valid_orders": int(len(valid)),
        "retained_unique_shops": int(valid["shop_id"].nunique()),
        "all_geometry_valid_buildings": int(len(buildings)),
        "valid_candidate_buildings": int(len(candidate_buildings)),
        "excluded_nonpositive_or_invalid_building_volume": building_quality["invalid_volume_buildings"],
        "excluded_invalid_building_geometry": building_quality["excluded_invalid_geometry"],
        "match_contains": int((valid["match_method"] == "contains").sum()),
        "match_nearest_within_50m": int((valid["match_method"] == "nearest_within_50m").sum()),
    }
    split_counts = valid["split"].value_counts().sort_index().astype(int).to_dict()
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "input_fingerprints": input_fingerprints,
        "rules": {
            "building_match": "contains_else_nearest_boundary",
            "building_match_radius_m": BUILDING_MATCH_RADIUS_M,
            "candidate_radius_km": DELIVERY_RADIUS_KM,
            "distance": "WGS84 Haversine centroid distance",
            "volume": "finite_positive_raw_volume_m3_no_truncation",
            "split": "sha256_shop_80_20_then_seen_last_5_of_29_days",
        },
        "candidate_summary": candidate_summary,
        "exclusion_flow": exclusion,
        "split_counts": split_counts,
    }
    (cache_dir / "prepared_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return manifest


def _prepared_matches(cache_dir: Path, fingerprints: dict) -> bool:
    path = Path(cache_dir) / "prepared_manifest.json"
    if not path.is_file():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return False
    required = (
        "candidate_indptr.npy", "candidate_building_index.npy", "candidate_log_volume.npy",
        "candidate_log_distance.npy", "candidate_distance_km.npy", "observations.csv.gz",
        "building_centroids.csv.gz",
    )
    return payload.get("input_fingerprints") == fingerprints and all(
        (Path(cache_dir) / name).is_file() for name in required
    )


def _split_observations(observations: pd.DataFrame, split: str) -> pd.DataFrame:
    selected = observations.loc[observations["split"] == split].copy()
    return selected[["shop_index", "candidate_position", "count"]]


def _parameter_vector(payload: dict) -> np.ndarray:
    return np.array([payload["parameters"][name] for name in ("alpha", "beta", "gamma")])


def _fixed_model(parameters: np.ndarray, name: str, message: str) -> dict:
    return {
        "parameters": {
            "alpha": float(parameters[0]),
            "beta": float(parameters[1]),
            "gamma": float(parameters[2]),
        },
        "success": True,
        "status": 0,
        "message": message,
        "iterations": 0,
        "function_evaluations": 0,
        "negative_log_likelihood": None,
        "gradient": None,
        "gradient_norm": None,
        "hessian": None,
        "hessian_eigenvalues": None,
        "hessian_condition_number": None,
        "free_parameters": [],
        "fixed_parameters": {
            "alpha": float(parameters[0]),
            "beta": float(parameters[1]),
            "gamma": float(parameters[2]),
        },
        "name": name,
    }


def _evaluate_all(candidates, observations, model_vectors, output_dir):
    metric_rows = []
    distribution_frames = []
    current_nll = {}
    uniform_nll = {}
    for split in ("test_unseen_shops_all_29_days", "test_seen_shops_last_5_days"):
        test = _split_observations(observations, split)
        if test.empty:
            raise ValueError(f"Evaluation split is empty: {split}")
        for model_name, parameters in model_vectors.items():
            metrics, distribution = evaluate_model(parameters, candidates, test)
            if model_name == "current_parameters":
                current_nll[split] = metrics["nll_per_order"]
            if model_name == "uniform_buildings":
                uniform_nll[split] = metrics["nll_per_order"]
            metric_rows.append({"split": split, "model": model_name, **metrics})
            distribution.insert(0, "model", model_name)
            distribution.insert(0, "split", split)
            distribution_frames.append(distribution)
    metrics_frame = pd.DataFrame(metric_rows)
    metrics_frame["delta_nll_vs_current"] = metrics_frame.apply(
        lambda row: row["nll_per_order"] - current_nll[row["split"]], axis=1
    )
    metrics_frame["mcfadden_pseudo_r2"] = metrics_frame.apply(
        lambda row: 1.0 - row["nll_per_order"] / uniform_nll[row["split"]], axis=1
    )
    metrics_frame.to_csv(output_dir / "model_metrics.csv", index=False)
    metrics_frame.to_csv(output_dir / "model_metrics_table.csv", index=False)
    pd.concat(distribution_frames, ignore_index=True).to_csv(
        output_dir / "distance_distribution_source.csv.gz", index=False, compression="gzip"
    )
    return metrics_frame


def _write_spatial_fit_diagnostics(
    candidates: CandidateCSR,
    observations: pd.DataFrame,
    training_parameters: np.ndarray,
    final_parameters: np.ndarray,
    output_dir: Path,
) -> dict:
    """Write full-sample spatial fit and small held-out validation summaries."""
    full = observations.groupby(
        ["shop_index", "candidate_position"], as_index=False
    )["count"].sum()
    merchant_frames = []
    merchant_summaries = []
    full_frame, full_summary = merchant_spatial_fit(
        final_parameters, candidates, full, "full_valid_sample"
    )
    merchant_frames.append(full_frame)
    merchant_summaries.append(full_summary)
    for split in ("test_unseen_shops_all_29_days", "test_seen_shops_last_5_days"):
        test = _split_observations(observations, split)
        frame, summary = merchant_spatial_fit(
            training_parameters, candidates, test, split
        )
        merchant_frames.append(frame)
        merchant_summaries.append(summary)
    merchant_source = pd.concat(merchant_frames, ignore_index=True)
    merchant_source.to_csv(
        output_dir / "merchant_spatial_fit_source.csv.gz",
        index=False,
        compression="gzip",
    )
    building_source, building_summary = building_order_fit(
        final_parameters, candidates, full
    )
    building_source.to_csv(
        output_dir / "building_order_fit_source.csv.gz",
        index=False,
        compression="gzip",
    )
    summary_frame = pd.DataFrame(merchant_summaries)
    for key, value in building_summary.items():
        summary_frame.loc[summary_frame["scope"] == "full_valid_sample", key] = value
    summary_frame.to_csv(output_dir / "spatial_fit_metrics.csv", index=False)
    return {
        "primary_scope": "full_valid_sample",
        "merchant_metrics": merchant_summaries,
        "building_metrics": building_summary,
        "definitions": {
            "merchant_mean_distance_r2_equal_weighted": (
                "1 - sum_i((observed_mean_i - predicted_mean_i)^2) / "
                "sum_i((observed_mean_i - mean_observed)^2); each merchant has equal weight"
            ),
            "merchant_mean_distance_r2_order_weighted": (
                "The same predictive R-squared with merchant order counts as weights"
            ),
            "building_order_count_r2_raw": (
                "Predictive R-squared of observed versus exact expected raw order counts "
                "over every building in an evaluated merchant's 5 km candidate set"
            ),
            "building_order_count_r2_log1p": (
                "The same building-level predictive R-squared after log1p transformation"
            ),
            "wasserstein_m": (
                "Exact first Wasserstein distance between each merchant's observed and "
                "model-predicted delivery-distance distributions"
            ),
        },
    }


def _merge_endpoint_centroids(
    endpoint_source: pd.DataFrame, centroids: pd.DataFrame
) -> pd.DataFrame:
    """Attach canonical metric centroids without accumulating merge suffixes."""
    coordinate_columns = [
        column
        for column in endpoint_source.columns
        if column.startswith("centroid_x") or column.startswith("centroid_y")
    ]
    endpoint_source = endpoint_source.drop(columns=coordinate_columns)
    return endpoint_source.merge(
        centroids[["building_index", "centroid_x", "centroid_y"]],
        on="building_index",
        how="left",
        validate="one_to_one",
    )


def _write_full_fit_plot_sources(
    candidates: CandidateCSR,
    full: pd.DataFrame,
    parameters: np.ndarray,
    cache_dir: Path,
    output_dir: Path,
) -> None:
    """Write full-sample distance and endpoint sources for audit plots."""
    observed_bins = np.zeros(101, dtype=float)
    predicted_bins = np.zeros(101, dtype=float)
    for shop_index, (positions, counts) in _group_observations(full).items():
        start, stop = candidates.bounds(shop_index)
        distances = np.asarray(candidates.distance_km[start:stop])
        observed_indices = np.floor(distances[positions - start] / 0.05).astype(int)
        observed_bins += np.bincount(
            np.clip(observed_indices, 0, 100), weights=counts, minlength=101
        )
        utilities = utility_features(candidates, start, stop) @ parameters
        probabilities = np.exp(utilities - logsumexp(utilities))
        predicted_indices = np.floor(distances / 0.05).astype(int)
        predicted_bins += np.bincount(
            np.clip(predicted_indices, 0, 100),
            weights=probabilities * float(counts.sum()),
            minlength=101,
        )
    distance_source = pd.concat(
        [
            pd.DataFrame({
                "series": "observed",
                "distance_bin_km": np.arange(101) * 0.05,
                "order_weight": observed_bins,
            }),
            pd.DataFrame({
                "series": "joint_three_parameter",
                "distance_bin_km": np.arange(101) * 0.05,
                "order_weight": predicted_bins,
            }),
        ], ignore_index=True,
    )
    distance_source.to_csv(
        output_dir / "full_order_distance_source.csv.gz",
        index=False,
        compression="gzip",
    )
    endpoint_path = output_dir / "endpoint_building_order_source.csv.gz"
    if endpoint_path.is_file():
        endpoint_source = pd.read_csv(endpoint_path)
    elif (output_dir / "building_order_fit_source.csv.gz").is_file():
        endpoint_source = pd.read_csv(output_dir / "building_order_fit_source.csv.gz")
    else:
        endpoint_source, _ = building_order_fit(parameters, candidates, full)
    centroids = pd.read_csv(cache_dir / "building_centroids.csv.gz")
    endpoint_source = _merge_endpoint_centroids(endpoint_source, centroids)
    endpoint_source.to_csv(
        output_dir / "endpoint_building_order_source.csv.gz",
        index=False,
        compression="gzip",
    )


def estimate_peak_memory_gb(buildings_path: Path, corrected_orders_path: Path) -> float:
    """Conservative preparation estimate used only to choose worker count."""
    input_gb = (Path(buildings_path).stat().st_size + Path(corrected_orders_path).stat().st_size) / 2**30
    return max(4.0, 7.0 * input_gb + 2.0)


def choose_worker_count(requested: int | None, memory_limit_gb: float, peak_gb: float) -> int:
    if memory_limit_gb <= 0:
        raise ValueError("memory_limit_gb must be positive")
    ceiling = min(32, os.cpu_count() or 1)
    wanted = ceiling if requested is None else max(1, min(32, requested))
    per_worker_gb = max(0.25, peak_gb * 0.08)
    return max(1, min(wanted, int(max(memory_limit_gb - peak_gb, per_worker_gb) / per_worker_gb)))


def run_joint_calibration(
    corrected_orders_path: Path,
    buildings_path: Path,
    transform_path: Path,
    output_dir: Path,
    memory_limit_gb: float = 64.0,
    workers: int | None = None,
    rebuild_prepared: bool = False,
    no_basemap: bool = False,
) -> dict:
    """Prepare/restore, evaluate, refit all data, and write audit artifacts."""
    fingerprints = validate_corrected_inputs(
        corrected_orders_path, buildings_path, transform_path
    )
    run_hash = hashlib.sha256(
        json.dumps(fingerprints, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]
    output_root = Path(output_dir)
    output_dir = output_root if output_root.name == run_hash else output_root / run_hash
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = output_dir / "cache" / run_hash
    if rebuild_prepared and cache_dir.exists():
        shutil.rmtree(cache_dir)
    peak_gb = estimate_peak_memory_gb(buildings_path, corrected_orders_path)
    if peak_gb >= memory_limit_gb:
        raise MemoryError(
            f"Estimated preparation peak {peak_gb:.1f} GB is not below the "
            f"{memory_limit_gb:.1f} GB task limit"
        )
    worker_count = choose_worker_count(workers, memory_limit_gb, peak_gb)
    if _prepared_matches(cache_dir, fingerprints):
        prepared = json.loads((cache_dir / "prepared_manifest.json").read_text())
        prepared["restored_from_cache"] = True
    else:
        prepared = prepare_calibration_data(
            corrected_orders_path,
            buildings_path,
            transform_path,
            cache_dir,
            fingerprints,
            workers=worker_count,
        )
        prepared["restored_from_cache"] = False
    candidates = CandidateCSR.load(cache_dir)
    observations = pd.read_csv(cache_dir / "observations.csv.gz")
    train = _split_observations(observations, "train_seen_first_24_days")
    current_model = _fixed_model(
        CURRENT_PARAMETERS.copy(), "current_parameters", "locked current parameters"
    )
    uniform_model = _fixed_model(
        np.zeros(3, dtype=float), "uniform_buildings", "uniform building selection"
    )
    volume_fit = fit_parameters(
        candidates,
        train,
        initial=np.array([CURRENT_PARAMETERS[0]], dtype=float),
        free_parameters=("alpha",),
        fixed_parameters={"beta": 0.0, "gamma": 0.0},
    )
    tanner_fit = fit_parameters(
        candidates,
        train,
        initial=np.array(CURRENT_PARAMETERS[1:], dtype=float),
        free_parameters=("beta", "gamma"),
        fixed_parameters={"alpha": 0.0},
    )
    train_fit = fit_parameters(candidates, train)
    training_models = {
        "current_parameters": current_model,
        "uniform_buildings": uniform_model,
        "building_volume_only": volume_fit,
        "tanner_distance_only": tanner_fit,
        "joint_three_parameter": train_fit,
    }
    model_vectors = {
        name: _parameter_vector(payload) for name, payload in training_models.items()
    }
    metrics_frame = _evaluate_all(
        candidates, observations, model_vectors, output_dir
    )
    full = observations.groupby(
        ["shop_index", "candidate_position"], as_index=False
    )["count"].sum()
    final_fit = fit_parameters(candidates, full, initial=_parameter_vector(train_fit))
    inference = clustered_covariance(_parameter_vector(final_fit), candidates, full)
    _write_full_fit_plot_sources(
        candidates,
        full,
        _parameter_vector(final_fit),
        cache_dir,
        output_dir,
    )
    spatial_fit = _write_spatial_fit_diagnostics(
        candidates,
        observations,
        _parameter_vector(train_fit),
        _parameter_vector(final_fit),
        output_dir,
    )
    parameter_table = pd.DataFrame(inference["parameters"])
    parameter_table.to_csv(output_dir / "parameter_estimates.csv", index=False)
    parameter_table.to_csv(output_dir / "parameter_table.csv", index=False)
    flow = pd.DataFrame(
        [{"stage": key, "count": value} for key, value in prepared["exclusion_flow"].items()]
    )
    flow.to_csv(output_dir / "data_flow.csv", index=False)
    from .gravity_plots import render_calibration_audits

    result = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "candidate",
        "run_hash": run_hash,
        "output_dir": str(output_dir.resolve()),
        "input_fingerprints": fingerprints,
        "memory": {
            "limit_gb": memory_limit_gb,
            "estimated_peak_gb": peak_gb,
            "selected_workers": worker_count,
        },
        "prepared": prepared,
        "training_models": training_models,
        "final_fit": final_fit,
        "clustered_inference": inference,
        "spatial_fit": spatial_fit,
        "metrics": metrics_frame.to_dict(orient="records"),
        "limitations": [
            "GBA building geometries and volumes post-date the 2020 Beijing orders.",
            "No threshold, weighting, truncation, sample, or parameter sensitivity analysis was applied.",
        ],
    }
    result_path = output_dir / "gravity_calibration.json"
    result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    render_calibration_audits(
        output_dir, result, cache_dir, no_basemap=no_basemap
    )
    result["artifacts"] = {
        path.name: {"sha256": sha256_file(path), "bytes": path.stat().st_size}
        for path in sorted(output_dir.iterdir()) if path.is_file() and path.name != result_path.name
    }
    result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    return result


def refresh_spatial_fit_diagnostics(
    result_path: Path, no_basemap: bool = False
) -> dict:
    """Add spatial-fit metrics to an existing converged calibration run."""
    result_path = Path(result_path)
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if result.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported calibration result schema")
    if not result.get("final_fit", {}).get("success"):
        raise ValueError("Spatial-fit diagnostics require a converged calibration result")
    output_dir = result_path.parent
    cache_dir = output_dir / "cache" / result["run_hash"]
    candidates = CandidateCSR.load(cache_dir)
    observations = pd.read_csv(cache_dir / "observations.csv.gz")
    centroid_path = cache_dir / "building_centroids.csv.gz"
    if not centroid_path.is_file():
        buildings_path = Path(result["input_fingerprints"]["buildings"]["path"])
        _, buildings_metric, _ = _load_buildings(buildings_path)
        buildings_metric[["building_index", "centroid_x", "centroid_y"]].to_csv(
            centroid_path, index=False, compression="gzip"
        )
    training_fit = result.get("training_models", {}).get("joint_three_parameter")
    if not training_fit or not training_fit.get("success"):
        raise ValueError("Spatial-fit diagnostics require the converged training-only fit")
    result["spatial_fit"] = _write_spatial_fit_diagnostics(
        candidates,
        observations,
        _parameter_vector(training_fit),
        _parameter_vector(result["final_fit"]),
        output_dir,
    )
    full = observations.groupby(
        ["shop_index", "candidate_position"], as_index=False
    )["count"].sum()
    _write_full_fit_plot_sources(
        candidates,
        full,
        _parameter_vector(result["final_fit"]),
        cache_dir,
        output_dir,
    )
    from .gravity_plots import render_spatial_fit_audit
    from .gravity_plots import _plot_endpoint_grid_heatmap, _plot_full_order_distance_distribution

    render_spatial_fit_audit(output_dir, result)
    _plot_full_order_distance_distribution(output_dir)
    _plot_endpoint_grid_heatmap(
        output_dir, cache_dir, no_basemap=no_basemap
    )
    result["artifacts"] = {
        path.name: {"sha256": sha256_file(path), "bytes": path.stat().st_size}
        for path in sorted(output_dir.iterdir())
        if path.is_file() and path.name != result_path.name
    }
    result_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return result


def accept_calibration_result(result_path: Path, experiment_path: Path, summary_path: Path) -> dict:
    """Accept one converged, fingerprinted result into tracked configuration."""
    result_path, experiment_path, summary_path = map(Path, (result_path, experiment_path, summary_path))
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if result.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported calibration result schema")
    if not result.get("final_fit", {}).get("success"):
        raise ValueError("Cannot accept a calibration result that did not converge")
    required_metric_fields = {
        "split", "model", "order_count", "nll_per_order", "perplexity",
        "mcfadden_pseudo_r2", "delta_nll_vs_current", "top_1", "top_10",
        "top_1_percent", "ks", "wasserstein_1_km", "observed_mean_km",
        "observed_median_km", "observed_p95_km", "predicted_mean_km",
        "predicted_median_km", "predicted_p95_km",
    }
    metrics = result.get("metrics")
    if not isinstance(metrics, list) or len(metrics) != 10:
        raise ValueError("Calibration result does not contain both tests for five models")
    if any(required_metric_fields - set(row) for row in metrics):
        raise ValueError("Calibration result metric schema is incomplete")
    if len(result.get("clustered_inference", {}).get("parameters", [])) != 3:
        raise ValueError("Calibration result is missing clustered parameter inference")
    for label, record in result.get("input_fingerprints", {}).items():
        path = Path(record["path"])
        if not path.is_file() or sha256_file(path) != record["sha256"]:
            raise ValueError(f"Input fingerprint changed since calibration: {label}")
    parameters = result["final_fit"]["parameters"]
    values = np.array([parameters[name] for name in ("alpha", "beta", "gamma")], float)
    if not np.isfinite(values).all() or values[0] < 0 or values[1] < 0 or values[2] <= 0:
        raise ValueError("Final parameters violate the gravity-model constraints")
    text = experiment_path.read_text(encoding="utf-8")
    replacements = {"alpha": values[0], "tanner_a": values[1], "tanner_b": values[2]}
    for key, value in replacements.items():
        pattern = rf"(?m)^(\s*{re.escape(key)}\s*=\s*)[^#\n]+"
        text, count = re.subn(pattern, rf"\g<1>{value:.10g} ", text, count=1)
        if count != 1:
            raise ValueError(f"Could not uniquely update {key} in experiment.toml")
    experiment_path.write_text(text, encoding="utf-8")
    summary = {
        "schema_version": SCHEMA_VERSION,
        "accepted_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_result_sha256": sha256_file(result_path),
        "run_hash": result["run_hash"],
        "input_fingerprints": result["input_fingerprints"],
        "parameters": parameters,
        "clustered_inference": result["clustered_inference"],
        "final_fit": result["final_fit"],
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    return summary
