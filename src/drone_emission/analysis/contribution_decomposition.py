"""Bounded-memory contribution decomposition for the locked paper cohort."""

from __future__ import annotations

from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import time
import traceback

import numpy as np
import pandas as pd

from drone_emission.emission.cal_single_delivery import (
    create_drone_params,
    create_ebike_params,
    get_emission_params,
)
from drone_emission.paper_cohort import cohort_hash, locked_region_keys
from drone_emission.sensitivity.vectorized_lca import (
    DroneScenario,
    EbikeScenario,
    compute_drone_per_order_parts_g,
    compute_ebike_per_order_parts_g,
)


ANALYSIS_VERSION = "1"
DEFAULT_CHUNK_ROWS = 250_000
CONSERVATION_RTOL = 1e-10
COMPONENT_KEYS = (
    "embodied_balance",
    "avoided_ebike_road_use",
    "drone_fixed_flight_penalty",
    "drone_cruise_penalty",
)
REQUIRED_OD_COLUMNS = ("distance_km", "road_distance_m", "annual_sales")
SUMMARY_COLUMNS = (
    "region_key",
    "region_name_en",
    "province",
    "gef",
    "gef_source",
    "baseline_ebike_lifecycle_kg",
    "total_od_rows",
    "valid_od_rows",
    "invalid_od_rows",
    "invalid_distance_km_rows",
    "invalid_road_distance_rows",
    "invalid_annual_sales_rows",
    "drone_assigned_rows",
    "drone_assigned_ratio_pct",
    *(f"{key}_kg" for key in COMPONENT_KEYS),
    "decomposition_net_kg",
    "oracle_net_kg",
    "max_abs_identity_error_g_per_order",
    "aggregate_identity_error_kg",
    "chunk_rows",
    "od_sha256",
    "semantic_sha256",
    "analysis_version",
    "paper_cohort_sha256",
    "generated_at_utc",
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(payload: dict) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(fd)
    try:
        frame.to_csv(temporary, index=False)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _atomic_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(fd)
    try:
        with open(temporary, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _status(
    city: dict,
    status: str,
    elapsed_seconds: float,
    message: str = "",
) -> dict:
    return {
        "region_key": city["region_key"],
        "region_name_en": city["region_name_en"],
        "status": status,
        "elapsed_seconds": round(elapsed_seconds, 3),
        "message": message,
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def _resolve_od_path(data_dir: Path, city: dict) -> Path:
    names = (str(city["region_key"]), str(city["region_name_en"]).lower())
    candidates = []
    for name in names:
        candidates.extend(
            (
                data_dir / "city" / name / f"{name}_od_pairs.csv",
                data_dir / "city" / name / f"{str(city['region_key'])}_od_pairs.csv",
            )
        )
    for path in dict.fromkeys(candidates):
        if path.is_file():
            return path
    raise FileNotFoundError(
        f"OD input missing for {city['region_key']}: "
        + ", ".join(str(path) for path in dict.fromkeys(candidates))
    )


def _model_inputs(config: dict, gef: float):
    emission = get_emission_params(config)
    ebike = replace(create_ebike_params(config), e_use=gef)
    drone = replace(create_drone_params(config), e_use=gef)
    ebike_scenario = EbikeScenario(
        kappa=float(ebike.kappa),
        rider_weight=float(emission["rider_weight"]),
        payload_kg=float(emission["payload_kg"]),
    )
    drone_scenario = DroneScenario(payload_kg=float(emission["payload_kg"]))
    return ebike, drone, ebike_scenario, drone_scenario


def _semantic_sha256(city: dict, config: dict) -> str:
    ebike, drone, ebike_scenario, drone_scenario = _model_inputs(
        config, float(city["gef"])
    )
    return _canonical_sha256(
        {
            "analysis_version": ANALYSIS_VERSION,
            "paper_cohort_sha256": cohort_hash(),
            "city": {
                key: city[key]
                for key in (
                    "region_key",
                    "region_name_en",
                    "province",
                    "gef",
                    "gef_source",
                    "baseline_ebike_lifecycle_kg",
                )
            },
            "ebike_params": asdict(ebike),
            "drone_params": asdict(drone),
            "ebike_scenario": asdict(ebike_scenario),
            "drone_scenario": asdict(drone_scenario),
        }
    )


def aggregate_od_contributions(
    od_path: str | Path,
    *,
    ebike_params,
    drone_params,
    ebike_scenario: EbikeScenario,
    drone_scenario: DroneScenario,
    chunk_rows: int = DEFAULT_CHUNK_ROWS,
) -> dict:
    """Reduce one OD CSV to a four-component ledger in bounded memory."""
    if chunk_rows <= 0:
        raise ValueError("contribution chunk_rows must be positive")
    od_path = Path(od_path)
    columns = set(pd.read_csv(od_path, nrows=0).columns)
    missing = sorted(set(REQUIRED_OD_COLUMNS) - columns)
    if missing:
        raise ValueError(f"OD input missing required columns: {missing}")

    component_partials = {key: [] for key in COMPONENT_KEYS}
    oracle_partials: list[float] = []
    identity_partials: list[float] = []
    counts = {
        "total_od_rows": 0,
        "valid_od_rows": 0,
        "invalid_distance_km_rows": 0,
        "invalid_road_distance_rows": 0,
        "invalid_annual_sales_rows": 0,
        "drone_assigned_rows": 0,
    }
    max_abs_error_g = 0.0
    max_abs_net_g = 0.0

    for chunk in pd.read_csv(
        od_path,
        usecols=list(REQUIRED_OD_COLUMNS),
        chunksize=chunk_rows,
    ):
        numeric = {
            column: pd.to_numeric(chunk[column], errors="coerce").to_numpy(
                dtype=np.float64, copy=False
            )
            for column in REQUIRED_OD_COLUMNS
        }
        air_km = numeric["distance_km"]
        road_m = numeric["road_distance_m"]
        annual_sales = numeric["annual_sales"]
        invalid_air = ~np.isfinite(air_km) | (air_km <= 0)
        invalid_road = ~np.isfinite(road_m) | (road_m <= 0)
        invalid_sales = ~np.isfinite(annual_sales) | (annual_sales <= 0)
        invalid = invalid_air | invalid_road | invalid_sales
        valid = ~invalid

        counts["total_od_rows"] += len(chunk)
        counts["invalid_distance_km_rows"] += int(invalid_air.sum())
        counts["invalid_road_distance_rows"] += int(invalid_road.sum())
        counts["invalid_annual_sales_rows"] += int(invalid_sales.sum())
        counts["valid_od_rows"] += int(valid.sum())
        if not valid.any():
            continue

        air_km = air_km[valid]
        road_km = road_m[valid] / 1000.0
        weight = annual_sales[valid]
        ebike = compute_ebike_per_order_parts_g(
            road_km, ebike_params, ebike_scenario
        )
        drone = compute_drone_per_order_parts_g(
            air_km, drone_params, drone_scenario
        )
        assigned = drone["total_g"] < ebike["total_g"]
        counts["drone_assigned_rows"] += int(assigned.sum())
        if not assigned.any():
            continue

        selected_weight = weight[assigned]
        components = {
            "embodied_balance": (
                ebike["production_g"]
                + ebike["end_of_life_g"]
                - drone["production_g"]
                - drone["end_of_life_g"]
            ),
            "avoided_ebike_road_use": ebike["use_g"],
            "drone_fixed_flight_penalty": -drone["fixed_use_g"],
            "drone_cruise_penalty": -drone["cruise_use_g"],
        }
        net_g = ebike["total_g"] - drone["total_g"]
        identity_error_g = sum(components.values()) - net_g
        selected_net = net_g[assigned]
        selected_error = identity_error_g[assigned]
        max_abs_error_g = max(
            max_abs_error_g, float(np.max(np.abs(selected_error)))
        )
        max_abs_net_g = max(max_abs_net_g, float(np.max(np.abs(selected_net))))
        for key, values in components.items():
            component_partials[key].append(
                float(np.sum(values[assigned] * selected_weight, dtype=np.float64) / 1000.0)
            )
        oracle_partials.append(
            float(np.sum(selected_net * selected_weight, dtype=np.float64) / 1000.0)
        )
        identity_partials.append(
            float(np.sum(selected_error * selected_weight, dtype=np.float64) / 1000.0)
        )

    if counts["valid_od_rows"] == 0:
        raise ValueError("OD input contains no valid rows")

    components_kg = {
        key: float(math.fsum(values)) for key, values in component_partials.items()
    }
    decomposition_net_kg = float(math.fsum(components_kg.values()))
    oracle_net_kg = float(math.fsum(oracle_partials))
    aggregate_error_kg = float(math.fsum(identity_partials))
    per_order_limit_g = CONSERVATION_RTOL * max(1.0, max_abs_net_g)
    aggregate_limit_kg = CONSERVATION_RTOL * max(1.0, abs(oracle_net_kg))
    if max_abs_error_g > per_order_limit_g:
        raise ArithmeticError(
            "per-order contribution identity failed: "
            f"error_g={max_abs_error_g}, limit_g={per_order_limit_g}"
        )
    if abs(decomposition_net_kg - oracle_net_kg) > aggregate_limit_kg:
        raise ArithmeticError(
            "aggregate contribution identity failed: "
            f"difference_kg={decomposition_net_kg - oracle_net_kg}, "
            f"limit_kg={aggregate_limit_kg}"
        )

    return {
        **counts,
        "invalid_od_rows": counts["total_od_rows"] - counts["valid_od_rows"],
        "drone_assigned_ratio_pct": (
            counts["drone_assigned_rows"] / counts["valid_od_rows"] * 100.0
        ),
        "component_kg": components_kg,
        "decomposition_net_kg": decomposition_net_kg,
        "oracle_net_kg": oracle_net_kg,
        "max_abs_identity_error_g_per_order": max_abs_error_g,
        "aggregate_identity_error_kg": aggregate_error_kg,
    }


def _summary_is_current(
    summary_path: Path,
    *,
    od_sha256: str,
    semantic_sha256: str,
) -> bool:
    if not summary_path.is_file():
        return False
    try:
        frame = pd.read_csv(summary_path)
    except (OSError, pd.errors.ParserError):
        return False
    if len(frame) != 1 or not set(SUMMARY_COLUMNS).issubset(frame.columns):
        return False
    row = frame.iloc[0]
    fingerprints_match = bool(
        str(row["od_sha256"]) == od_sha256
        and str(row["semantic_sha256"]) == semantic_sha256
        and str(row["analysis_version"]) == ANALYSIS_VERSION
        and str(row["paper_cohort_sha256"]) == cohort_hash()
    )
    if not fingerprints_match:
        return False
    numeric_columns = [
        *(f"{key}_kg" for key in COMPONENT_KEYS),
        "decomposition_net_kg",
        "oracle_net_kg",
        "valid_od_rows",
        "drone_assigned_rows",
    ]
    numeric = pd.to_numeric(row[numeric_columns], errors="coerce")
    if numeric.isna().any() or not np.isfinite(numeric.to_numpy(dtype=float)).all():
        return False
    component_net = math.fsum(float(row[f"{key}_kg"]) for key in COMPONENT_KEYS)
    oracle_net = float(row["oracle_net_kg"])
    if abs(component_net - oracle_net) > CONSERVATION_RTOL * max(1.0, abs(oracle_net)):
        return False
    return bool(
        int(row["valid_od_rows"]) >= 0
        and 0 <= int(row["drone_assigned_rows"]) <= int(row["valid_od_rows"])
    )


def analyze_city(
    city: dict,
    config: dict,
    *,
    force: bool = False,
    chunk_rows: int = DEFAULT_CHUNK_ROWS,
) -> dict:
    """Analyze one city and persist an independently reusable summary."""
    started = time.monotonic()
    try:
        od_path = _resolve_od_path(Path(config["data_dir"]), city)
        output_dir = od_path.parent / "analysis" / "contribution_decomposition"
    except FileNotFoundError as exc:
        output_dir = (
            Path(config["data_dir"])
            / "city"
            / city["region_key"]
            / "analysis"
            / "contribution_decomposition"
        )
        result = _status(
            city,
            "error",
            time.monotonic() - started,
            f"FileNotFoundError: {exc}",
        )
        _atomic_json(result, output_dir / "status.json")
        return result
    status_path = output_dir / "status.json"
    _atomic_json(_status(city, "running", 0.0), status_path)
    try:
        summary_path = output_dir / "summary.csv"
        od_sha256 = _sha256_file(od_path)
        semantic_sha256 = _semantic_sha256(city, config)
        if not force and _summary_is_current(
            summary_path,
            od_sha256=od_sha256,
            semantic_sha256=semantic_sha256,
        ):
            result = _status(city, "cached", time.monotonic() - started)
            result.update(
                {
                    "od_sha256": od_sha256,
                    "semantic_sha256": semantic_sha256,
                }
            )
            _atomic_json(result, status_path)
            return result

        ebike, drone, ebike_scenario, drone_scenario = _model_inputs(
            config, float(city["gef"])
        )
        aggregate = aggregate_od_contributions(
            od_path,
            ebike_params=ebike,
            drone_params=drone,
            ebike_scenario=ebike_scenario,
            drone_scenario=drone_scenario,
            chunk_rows=chunk_rows,
        )
        row = {
            "region_key": city["region_key"],
            "region_name_en": city["region_name_en"],
            "province": city["province"],
            "gef": float(city["gef"]),
            "gef_source": city["gef_source"],
            "baseline_ebike_lifecycle_kg": float(
                city["baseline_ebike_lifecycle_kg"]
            ),
            **{key: aggregate[key] for key in SUMMARY_COLUMNS if key in aggregate},
            **{f"{key}_kg": aggregate["component_kg"][key] for key in COMPONENT_KEYS},
            "od_sha256": od_sha256,
            "semantic_sha256": semantic_sha256,
            "analysis_version": ANALYSIS_VERSION,
            "paper_cohort_sha256": cohort_hash(),
            "chunk_rows": int(chunk_rows),
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        _atomic_csv(pd.DataFrame([row]).loc[:, list(SUMMARY_COLUMNS)], summary_path)
        result = _status(
            city,
            "ok",
            time.monotonic() - started,
            f"valid_rows={aggregate['valid_od_rows']:,}",
        )
        result.update(
            {
                "od_sha256": od_sha256,
                "semantic_sha256": semantic_sha256,
            }
        )
    except Exception as exc:  # city isolation is required for resumability
        result = _status(
            city,
            "error",
            time.monotonic() - started,
            f"{type(exc).__name__}: {exc}",
        )
        traceback.print_exc()
    _atomic_json(result, status_path)
    return result


def _city_inputs(config: dict) -> list[dict]:
    stats_path = Path(config["data_dir"]) / "all_region_stats.csv"
    required = {
        "region_key",
        "region_name_en",
        "province",
        "region_level",
        "total_ebike_lifecycle",
        "gef",
        "gef_source",
    }
    stats = pd.read_csv(stats_path)
    missing = sorted(required - set(stats.columns))
    if missing:
        raise ValueError(f"city stats missing contribution columns: {missing}")
    stats = stats[stats["region_level"].eq("city")].copy()
    expected = set(locked_region_keys())
    actual = set(stats["region_key"].astype(str))
    if len(stats) != len(expected) or actual != expected:
        raise ValueError(
            "contribution city stats do not match locked cohort: "
            f"rows={len(stats)}, missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )
    for column in ("total_ebike_lifecycle", "gef"):
        stats[column] = pd.to_numeric(stats[column], errors="coerce")
    if stats[["total_ebike_lifecycle", "gef"]].isna().any().any():
        raise ValueError("contribution city stats contain non-numeric baseline or GEF")
    if (stats["total_ebike_lifecycle"] <= 0).any():
        raise ValueError("contribution city stats contain non-positive baselines")
    return [
        {
            "region_key": str(row.region_key),
            "region_name_en": str(row.region_name_en),
            "province": str(row.province),
            "gef": float(row.gef),
            "gef_source": str(row.gef_source),
            "baseline_ebike_lifecycle_kg": float(row.total_ebike_lifecycle),
        }
        for row in stats.sort_values("region_key").itertuples(index=False)
    ]


def consolidate(
    config: dict,
    cities: list[dict],
    results: list[dict],
) -> tuple[Path, bool]:
    """Write partial/final cohort tables from this run's city statuses."""
    result_by_key = {row["region_key"]: row for row in results}
    frames = []
    statuses = []
    complete = True
    for city in cities:
        result = result_by_key.get(city["region_key"], {})
        try:
            od_path = _resolve_od_path(Path(config["data_dir"]), city)
            summary_path = (
                od_path.parent / "analysis" / "contribution_decomposition" / "summary.csv"
            )
        except FileNotFoundError:
            summary_path = Path()
        current = (
            result.get("status") in {"ok", "cached"}
            and summary_path.is_file()
            and _summary_is_current(
                summary_path,
                od_sha256=str(result.get("od_sha256", "")),
                semantic_sha256=str(result.get("semantic_sha256", "")),
            )
        )
        if current:
            frame = pd.read_csv(summary_path)
            current = (
                len(frame) == 1
                and str(frame.iloc[0]["region_key"]) == city["region_key"]
                and str(frame.iloc[0]["paper_cohort_sha256"]) == cohort_hash()
            )
            if current:
                frames.append(frame.loc[:, list(SUMMARY_COLUMNS)])
        complete = complete and current
        statuses.append({**result, "summary_current": bool(current)})

    data_dir = Path(config["data_dir"])
    final_path = data_dir / "all_city_contribution_decomposition.csv"
    partial_path = data_dir / "all_city_contribution_decomposition.partial.csv"
    status_path = data_dir / "all_city_contribution_decomposition_status.csv"
    _atomic_csv(pd.DataFrame(statuses), status_path)
    if not frames:
        if final_path.exists():
            final_path.unlink()
        raise RuntimeError("no current city contribution summaries found")
    merged = pd.concat(frames, ignore_index=True).sort_values("region_key")
    merged = merged.loc[:, list(SUMMARY_COLUMNS)]
    expected = set(locked_region_keys())
    keys = set(merged["region_key"].astype(str))
    complete = complete and len(merged) == len(expected) and keys == expected
    if complete:
        _atomic_csv(merged, final_path)
        if partial_path.exists():
            partial_path.unlink()
        return final_path, True
    if final_path.exists():
        final_path.unlink()
    _atomic_csv(merged, partial_path)
    return partial_path, False


def run_contribution_analysis(
    config: dict,
    *,
    force: bool = False,
    chunk_rows: int = DEFAULT_CHUNK_ROWS,
) -> tuple[Path, list[dict]]:
    """Analyze the locked cohort serially and consolidate its compact table."""
    if chunk_rows <= 0:
        raise ValueError("--contribution-chunk-rows must be positive")
    final_path = Path(config["data_dir"]) / "all_city_contribution_decomposition.csv"
    if final_path.exists():
        final_path.unlink()
    cities = _city_inputs(config)
    results = [
        analyze_city(city, config, force=force, chunk_rows=chunk_rows)
        for city in cities
    ]
    output, complete = consolidate(config, cities, results)
    if not complete:
        failures = [
            f"{row['region_key']}:{row.get('status', 'missing')}"
            for row in results
            if row.get("status") not in {"ok", "cached"}
        ]
        raise RuntimeError(
            f"contribution decomposition is partial: {output}; "
            + ", ".join(failures)
        )
    return output, results
