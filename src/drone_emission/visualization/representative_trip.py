"""Shared representative-trip selection for the Fig. 1 input panels."""

from __future__ import annotations

import hashlib
import heapq
import json
import math
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from drone_emission.analysis.weighted_quantile import weighted_csv_quantile
from drone_emission.utils.atomic import atomic_open


_ALGORITHM_VERSION = 1
_CHUNK_ROWS = 250_000
_QUANTILE = 0.75
_DETOUR_QUANTILE = 0.5
_NUMERIC_COLUMNS = (
    "road_distance_m",
    "distance_km",
    "annual_sales",
    "from_lat",
    "from_lng",
    "to_lat",
    "to_lng",
    "drone_emission_kg",
    "ebike_emission_kg",
    "ebike_kappa",
    "drone_lifecycle_production_g",
    "ebike_lifecycle_production_g",
    "drone_lifecycle_eol_g",
    "ebike_lifecycle_eol_g",
)


@dataclass(frozen=True)
class RepresentativeTrip:
    """One selected OD row and its reproducible selection provenance."""

    row: pd.Series
    selection: dict


def _source_signature(path: Path) -> dict:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def _selection_cache_path(path: Path, cache_dir: str | Path) -> Path:
    source_key = hashlib.sha256(str(path.resolve()).encode("utf-8")).hexdigest()[:16]
    return Path(cache_dir) / "beijing_p75_trip" / source_key / "selection.json"


def _load_cached_selection(cache_path: Path, identity: dict) -> RepresentativeTrip | None:
    try:
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
        if payload.get("identity") != identity:
            return None
        return RepresentativeTrip(
            row=pd.Series(payload["row"]),
            selection=payload["selection"],
        )
    except (FileNotFoundError, json.JSONDecodeError, KeyError, TypeError, ValueError):
        return None


def _weighted_detour_quantile(
    csv_path: Path,
    *,
    quantile: float,
    chunk_rows: int = _CHUNK_ROWS,
) -> tuple[float, int, float]:
    """Return an exact annual-sales-weighted quantile of road/air detour."""
    total_weight = 0.0
    valid_rows = 0
    with tempfile.TemporaryDirectory(prefix="weighted-detour-") as temp_dir:
        run_paths: list[Path] = []
        for run_index, chunk in enumerate(
            pd.read_csv(
                csv_path,
                usecols=["road_distance_m", "distance_km", "annual_sales"],
                chunksize=chunk_rows,
            )
        ):
            road = pd.to_numeric(chunk["road_distance_m"], errors="coerce").to_numpy(
                dtype=float
            )
            air = pd.to_numeric(chunk["distance_km"], errors="coerce").to_numpy(
                dtype=float
            )
            weights = pd.to_numeric(chunk["annual_sales"], errors="coerce").to_numpy(
                dtype=float
            )
            valid = (
                np.isfinite(road)
                & np.isfinite(air)
                & np.isfinite(weights)
                & (road > 0)
                & (air > 0)
                & (weights > 0)
            )
            detour = road[valid] / (air[valid] * 1000.0)
            valid_weights = weights[valid]
            if not len(detour):
                continue
            order = np.argsort(detour, kind="stable")
            run = np.column_stack((detour[order], valid_weights[order]))
            run_path = Path(temp_dir) / f"run_{run_index:06d}.npy"
            np.save(run_path, run, allow_pickle=False)
            run_paths.append(run_path)
            total_weight += float(valid_weights.sum())
            valid_rows += int(len(detour))

        if not run_paths or total_weight <= 0:
            raise ValueError(f"{csv_path} contains no positive weighted detour values")

        runs = [np.load(path, mmap_mode="r") for path in run_paths]
        heap = [(float(run[0, 0]), index, 0) for index, run in enumerate(runs)]
        heapq.heapify(heap)
        target = quantile * total_weight
        cumulative = 0.0
        selected = float(heap[0][0])
        while heap:
            selected = float(heap[0][0])
            equal_weight = 0.0
            while heap and heap[0][0] == selected:
                _, run_index, row_index = heapq.heappop(heap)
                run = runs[run_index]
                equal_weight += float(run[row_index, 1])
                next_index = row_index + 1
                if next_index < len(run):
                    heapq.heappush(
                        heap,
                        (float(run[next_index, 0]), run_index, next_index),
                    )
            cumulative += equal_weight
            if cumulative >= target:
                return selected, valid_rows, total_weight

    raise RuntimeError("weighted detour quantile merge ended without a result")


def _json_scalar(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        numeric = float(value)
        return numeric if math.isfinite(numeric) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def select_beijing_p75_trip(
    od_csv: str | Path,
    *,
    cache_dir: str | Path,
) -> RepresentativeTrip:
    """Select one complete Beijing OD at weighted road-distance P75."""
    source = Path(od_csv)
    header = pd.read_csv(source, nrows=0).columns
    missing = sorted(set(_NUMERIC_COLUMNS) - set(header))
    if missing:
        raise ValueError(f"{source} is missing representative-trip columns: {missing}")

    identity = {
        "source": _source_signature(source),
        "algorithm_version": _ALGORITHM_VERSION,
        "road_quantile": _QUANTILE,
        "detour_quantile": _DETOUR_QUANTILE,
        "weight_column": "annual_sales",
    }
    cache_path = _selection_cache_path(source, cache_dir)
    cached = _load_cached_selection(cache_path, identity)
    if cached is not None:
        return cached

    road_quantile = weighted_csv_quantile(
        source,
        value_column="road_distance_m",
        weight_column="annual_sales",
        quantile=_QUANTILE,
        cache_dir=Path(cache_dir) / "weighted_quantile",
    )
    detour_median, detour_valid_rows, detour_total_weight = (
        _weighted_detour_quantile(source, quantile=_DETOUR_QUANTILE)
    )

    best_key = None
    best_row = None
    row_offset = 0
    eligible_rows = 0
    for chunk in pd.read_csv(source, usecols=list(_NUMERIC_COLUMNS), chunksize=_CHUNK_ROWS):
        source_indices = np.arange(row_offset, row_offset + len(chunk), dtype=np.int64)
        row_offset += len(chunk)
        numeric = chunk.copy()
        for column in _NUMERIC_COLUMNS:
            numeric[column] = pd.to_numeric(numeric[column], errors="coerce")
        values = numeric.loc[:, list(_NUMERIC_COLUMNS)].to_numpy(dtype=float)
        complete = np.isfinite(values).all(axis=1)
        complete &= numeric["road_distance_m"].to_numpy(dtype=float) > 0
        complete &= numeric["distance_km"].to_numpy(dtype=float) > 0
        complete &= numeric["annual_sales"].to_numpy(dtype=float) > 0
        complete &= numeric["drone_emission_kg"].to_numpy(dtype=float) >= 0
        complete &= numeric["ebike_emission_kg"].to_numpy(dtype=float) >= 0
        eligible_rows += int(complete.sum())
        if not complete.any():
            continue
        candidates = numeric.loc[complete].copy()
        candidates["_source_row_index"] = source_indices[complete]
        candidates["_detour_ratio"] = (
            candidates["road_distance_m"]
            / (candidates["distance_km"] * 1000.0)
        )
        candidates["_road_delta"] = (
            candidates["road_distance_m"] - road_quantile.value
        ).abs()
        candidates["_detour_delta"] = (
            candidates["_detour_ratio"] - detour_median
        ).abs()
        candidate = candidates.sort_values(
            ["_road_delta", "_detour_delta", "_source_row_index"],
            kind="stable",
        ).iloc[0]
        key = (
            float(candidate["_road_delta"]),
            float(candidate["_detour_delta"]),
            int(candidate["_source_row_index"]),
        )
        if best_key is None or key < best_key:
            best_key = key
            best_row = candidate

    if best_row is None:
        raise ValueError(f"{source} contains no complete representative-trip rows")

    row = {column: _json_scalar(best_row[column]) for column in _NUMERIC_COLUMNS}
    source_row_index = int(best_row["_source_row_index"])
    selected_detour = float(best_row["_detour_ratio"])
    fingerprint_input = {
        "source": identity["source"],
        "source_row_index": source_row_index,
        "road_distance_m": row["road_distance_m"],
        "distance_km": row["distance_km"],
    }
    selection_id = hashlib.sha256(
        json.dumps(fingerprint_input, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()
    selection = {
        "selection_id": selection_id,
        "source_city": "Beijing",
        "source_row_index": source_row_index,
        "input_signature": identity["source"],
        "rule": (
            "minimum absolute road-distance difference from annual-sales-weighted "
            "P75, then minimum absolute detour-ratio difference from the "
            "annual-sales-weighted median, then source row index"
        ),
        "road_distance_target": {
            "source_column": "road_distance_m",
            "weight_column": "annual_sales",
            "quantile": _QUANTILE,
            "value_m": float(road_quantile.value),
            "valid_rows": int(road_quantile.valid_rows),
            "total_annual_sales": float(road_quantile.total_weight),
        },
        "detour_tiebreak_target": {
            "definition": "road_distance_m / (distance_km * 1000)",
            "weight_column": "annual_sales",
            "quantile": _DETOUR_QUANTILE,
            "value": detour_median,
            "valid_rows": detour_valid_rows,
            "total_annual_sales": detour_total_weight,
        },
        "complete_candidate_rows": eligible_rows,
        "selected_trip": {
            "air_distance_km": float(row["distance_km"]),
            "road_distance_m": float(row["road_distance_m"]),
            "road_target_difference_m": float(best_key[0]),
            "detour_ratio": selected_detour,
            "detour_target_difference": float(best_key[1]),
            "annual_sales": float(row["annual_sales"]),
            "ebike_kappa": float(row["ebike_kappa"]),
            "origin_lat_lng": [float(row["from_lat"]), float(row["from_lng"])],
            "destination_lat_lng": [float(row["to_lat"]), float(row["to_lng"])],
        },
    }
    payload = {"identity": identity, "row": row, "selection": selection}
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with atomic_open(cache_path, mode="w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    return RepresentativeTrip(row=pd.Series(row), selection=selection)
