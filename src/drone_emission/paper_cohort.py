"""Locked paper-city cohort and post-aggregate eligibility audit."""

from __future__ import annotations

import hashlib
from importlib.resources import files
import json
import math
import os
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd

from .paths import PROJECT_ROOT


COHORT_FILENAME = "paper_cohort.json"
REQUIRED_CITY_STATS = (
    "total_ebike_lifecycle",
    "total_drone_lifecycle",
    "oracle_min_lifecycle_kg",
    "oracle_min_reduction_vs_ebike_pct",
    "total_odpair_annual_sales",
    "detour_ratio",
)
POSITIVE_CITY_STATS = {
    "total_ebike_lifecycle",
    "total_drone_lifecycle",
    "oracle_min_lifecycle_kg",
    "total_odpair_annual_sales",
    "detour_ratio",
}


def region_keys_sha256(region_keys: list[str] | tuple[str, ...] | set[str]) -> str:
    """Hash a city set using a stable newline-delimited representation."""
    payload = "".join(f"{key}\n" for key in sorted(region_keys))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_cohort_contract() -> dict:
    """Load and validate the tracked paper-cohort contract."""
    contract = json.loads(
        files("drone_emission").joinpath(COHORT_FILENAME).read_text(encoding="utf-8")
    )
    if contract.get("schema_version") != 1:
        raise ValueError("Unsupported paper cohort schema")
    expected = int(contract.get("expected_city_count", 0))
    if expected <= 0:
        raise ValueError("paper cohort expected_city_count must be positive")
    fallback = contract.get("grid_emission_factor_fallback", {})
    if float(fallback.get("kg_co2e_per_kwh", 0.0)) <= 0:
        raise ValueError("paper cohort GEF fallback must be positive")
    return contract


def configured_city_keys() -> list[str]:
    """Return city keys from the tracked region registry."""
    payload = json.loads((PROJECT_ROOT / "regions.json").read_text(encoding="utf-8"))
    return sorted(
        key for key, region in payload.items() if region.get("region_level") == "city"
    )


def locked_region_keys(contract: dict | None = None) -> list[str]:
    """Resolve and verify the exact locked paper-city set."""
    contract = contract or load_cohort_contract()
    excluded = set(contract.get("excluded_regions", {}))
    keys = [key for key in configured_city_keys() if key not in excluded]
    expected_count = int(contract["expected_city_count"])
    if len(keys) != expected_count:
        raise ValueError(
            "paper cohort count changed: "
            f"expected={expected_count}, actual={len(keys)}"
        )
    actual_hash = region_keys_sha256(keys)
    expected_hash = contract["included_region_keys_sha256"]
    if actual_hash != expected_hash:
        raise ValueError(
            "paper cohort city set changed: "
            f"expected_sha256={expected_hash}, actual_sha256={actual_hash}"
        )
    return keys


def cohort_hash() -> str:
    return load_cohort_contract()["included_region_keys_sha256"]


def _city_paths(data_root: Path, region_key: str, region: dict) -> tuple[Path, Path]:
    name = str(region["region_name_en"]).lower()
    city_root = data_root / "city" / name
    return city_root / "region_stats.json", city_root / f"{name}_od_pairs.csv"


def _finite(value: object) -> bool:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(number)


def _valid_stat(name: str, value: object) -> bool:
    if not _finite(value):
        return False
    return name not in POSITIVE_CITY_STATS or float(value) > 0


def _scan_weighted_paths(path: Path) -> tuple[int, float]:
    """Count valid weighted paths only when city metadata are insufficient."""
    valid_count = 0
    annual_sales_sum = 0.0
    for chunk in pd.read_csv(
        path,
        usecols=["road_distance_m", "distance_km", "annual_sales"],
        chunksize=250_000,
    ):
        road = pd.to_numeric(chunk["road_distance_m"], errors="coerce").to_numpy()
        air = pd.to_numeric(chunk["distance_km"], errors="coerce").to_numpy()
        sales = pd.to_numeric(chunk["annual_sales"], errors="coerce").to_numpy()
        valid = (
            np.isfinite(road)
            & np.isfinite(air)
            & np.isfinite(sales)
            & (road >= 0)
            & (air > 0)
            & (sales > 0)
        )
        valid_count += int(valid.sum())
        annual_sales_sum += float(sales[valid].sum())
    return valid_count, annual_sales_sum


def audit_city_cohort(config: dict) -> dict:
    """Audit every configured city after aggregate without changing city data."""
    contract = load_cohort_contract()
    threshold = int(contract["minimum_valid_weighted_od_pairs"])
    registry = json.loads((PROJECT_ROOT / "regions.json").read_text(encoding="utf-8"))
    data_root = Path(config["data_dir"])
    rows: list[dict] = []
    candidate: list[str] = []

    for region_key in configured_city_keys():
        region = registry[region_key]
        stats_path, od_path = _city_paths(data_root, region_key, region)
        reason = None
        stats: dict = {}
        if not stats_path.is_file() or not od_path.is_file():
            reason = "missing_city_input"
        else:
            try:
                stats = json.loads(stats_path.read_text(encoding="utf-8")).get(
                    "stats", {}
                )
            except (OSError, json.JSONDecodeError):
                reason = "invalid_region_stats"
        counts = []
        for count_name in ("valid_path_count", "odpair_count"):
            try:
                counts.append(int(stats.get(count_name, 0)))
            except (TypeError, ValueError):
                continue
        valid_count = max(counts, default=0)
        annual_sales_sum = stats.get("total_odpair_annual_sales")
        if reason is None and (
            valid_count < threshold or not _finite(annual_sales_sum)
        ):
            try:
                scanned_count, scanned_sales = _scan_weighted_paths(od_path)
                valid_count = max(valid_count, scanned_count)
                annual_sales_sum = scanned_sales
                stats = dict(stats, total_odpair_annual_sales=scanned_sales)
            except (OSError, ValueError, KeyError, pd.errors.ParserError):
                reason = "invalid_od_input"
        missing_stats = [
            name for name in REQUIRED_CITY_STATS if not _valid_stat(name, stats.get(name))
        ]
        if reason is None and valid_count < threshold:
            reason = "insufficient_od_pairs"
        if reason is None and missing_stats:
            reason = "incomplete_city_result"
        eligible = reason is None
        if eligible:
            candidate.append(region_key)
        rows.append(
            {
                "region_key": region_key,
                "eligible": eligible,
                "reason": reason,
                "valid_weighted_od_pairs": valid_count,
                "missing_required_stats": missing_stats,
                "region_stats_path": str(stats_path),
                "od_path": str(od_path),
            }
        )

    locked = locked_region_keys(contract)
    payload = {
        "schema_version": 1,
        "cohort_id": contract["cohort_id"],
        "minimum_valid_weighted_od_pairs": threshold,
        "candidate_region_keys": candidate,
        "candidate_region_keys_sha256": region_keys_sha256(candidate),
        "locked_region_keys": locked,
        "locked_region_keys_sha256": contract["included_region_keys_sha256"],
        "matches_locked": candidate == locked,
        "added_since_lock": sorted(set(candidate) - set(locked)),
        "missing_since_lock": sorted(set(locked) - set(candidate)),
        "cities": rows,
    }
    return payload


def write_cohort_audit(config: dict, payload: dict) -> Path:
    """Write the candidate audit below the configured work root."""
    target = Path(config["work_dir"]) / "cohort" / "candidate.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    os.close(fd)
    try:
        Path(temporary).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, target)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return target


def assert_locked_cohort(config: dict) -> Path:
    """Audit and fail when current city data no longer match the paper lock."""
    payload = audit_city_cohort(config)
    path = write_cohort_audit(config, payload)
    if not payload["matches_locked"]:
        raise RuntimeError(
            "paper cohort audit differs from the tracked lock: "
            f"added={payload['added_since_lock']}, "
            f"missing={payload['missing_since_lock']}. "
            f"Review {path} and explicitly accept a revised cohort before analyze."
        )
    return path


def accept_candidate_cohort(payload: dict) -> Path:
    """Explicitly replace the tracked lock with a reviewed audit candidate."""
    contract = load_cohort_contract()
    excluded = {
        row["region_key"]: row["reason"]
        for row in payload["cities"]
        if not row["eligible"]
    }
    contract["expected_city_count"] = len(payload["candidate_region_keys"])
    contract["included_region_keys_sha256"] = payload[
        "candidate_region_keys_sha256"
    ]
    contract["excluded_regions"] = dict(sorted(excluded.items()))
    fallback = contract["grid_emission_factor_fallback"]
    fallback["affected_region_keys"] = sorted(
        set(fallback["affected_region_keys"]) & set(payload["candidate_region_keys"])
    )
    target = PROJECT_ROOT / "src" / "drone_emission" / COHORT_FILENAME
    temporary = target.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(contract, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(target)
    return target
