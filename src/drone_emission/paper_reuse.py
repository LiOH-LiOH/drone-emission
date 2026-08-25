"""Audited one-directory reuse for low-cost paper validation."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile

import numpy as np
import pandas as pd

from .paper_cohort import cohort_hash, load_cohort_contract, locked_region_keys


REUSE_SCHEMA_VERSION = 1
GLOBAL_DATA_ARTIFACTS = (
    "all_region_stats.json",
    "all_region_stats.csv",
    "all_region_stats_enriched.csv",
    "all_sensitivity_named.csv",
    "all_city_distance_threshold_summary.csv",
    "all_city_distance_threshold_status.csv",
    "all_threshold_resilience.csv",
    "all_threshold_resilience_drivers.csv",
)
WORK_ARTIFACTS = (
    "analysis/city_attribution/r2_city_attribution_data.csv",
    "analysis/city_attribution/r2_city_attribution_analysis_report.json",
)
CACHE_ARTIFACTS = (
    "drone_spatial_concentration/city_drone_spatial_concentration.csv",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    os.close(fd)
    try:
        shutil.copy2(source, temporary)
        os.replace(temporary, target)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _artifact_record(source: Path, destination: str) -> dict:
    record = {
        "path": destination,
        "size_bytes": source.stat().st_size,
        "sha256": _sha256(source),
    }
    if source.suffix == ".csv":
        frame = pd.read_csv(source)
        record["row_count"] = len(frame)
        if "region_key" in frame.columns:
            keys = sorted(frame["region_key"].dropna().astype(str).unique())
            record["region_keys_sha256"] = hashlib.sha256(
                "".join(f"{key}\n" for key in keys).encode("utf-8")
            ).hexdigest()
            record["region_count"] = len(keys)
    return record


def capture_reuse_directory(config: dict, output_root: str | Path) -> Path:
    """Capture the prepared cohort view and five-city overlay once."""
    output_root = Path(output_root).resolve()
    payload_root = output_root / "payload"
    payload_root.mkdir(parents=True, exist_ok=True)
    data_root = Path(config["data_dir"])
    cache_root = Path(config["cache_dir"])
    work_root = Path(config["work_dir"])
    contract = load_cohort_contract()
    affected = contract["grid_emission_factor_fallback"]["affected_region_keys"]
    fallback_value = float(
        contract["grid_emission_factor_fallback"]["kg_co2e_per_kwh"]
    )
    records: list[dict] = []

    for region_key in affected:
        city_root = data_root / "city" / region_key
        od = pd.read_csv(
            city_root / f"{region_key}_od_pairs.csv",
            usecols=["grid_emission_factor"],
        )
        values = pd.to_numeric(od["grid_emission_factor"], errors="coerce")
        if len(values) == 0 or not np.allclose(values, fallback_value):
            raise ValueError(f"fallback-GEF overlay is stale for {region_key}")
        stats = json.loads(
            (city_root / "region_stats.json").read_text(encoding="utf-8")
        ).get("stats", {})
        if not np.isclose(float(stats.get("grid_emission_factor", -1)), fallback_value):
            raise ValueError(f"fallback-GEF city stats are stale for {region_key}")
        if stats.get("gef_source") != "fallback_0.5366":
            raise ValueError(f"fallback-GEF provenance is missing for {region_key}")

    declarations: list[tuple[Path, str]] = []
    declarations.extend((data_root / name, f"data:{name}") for name in GLOBAL_DATA_ARTIFACTS)
    declarations.extend((cache_root / name, f"cache:{name}") for name in CACHE_ARTIFACTS)
    declarations.extend((work_root / name, f"work:{name}") for name in WORK_ARTIFACTS)
    for region_key in affected:
        city_root = data_root / "city" / region_key
        for relative in (
            f"{region_key}_od_pairs.csv",
            "region_stats.json",
            f"{region_key}_named.csv",
            "analysis/distance_threshold/summary.csv",
            "analysis/distance_threshold/status.json",
            "analysis/threshold_resilience/summary.csv",
            "analysis/threshold_resilience/drivers.csv",
            "analysis/threshold_resilience/metadata.json",
            "analysis/threshold_resilience/status.json",
        ):
            declarations.append(
                (city_root / relative, f"data:city/{region_key}/{relative}")
            )

    missing = [str(source) for source, _ in declarations if not source.is_file()]
    if missing:
        raise FileNotFoundError(
            "cannot capture incomplete paper reuse directory:\n- "
            + "\n- ".join(missing)
        )

    for source, destination in declarations:
        relative = destination.replace(":", "/", 1)
        target = payload_root / relative
        _atomic_copy(source, target)
        records.append(_artifact_record(target, destination))

    manifest = {
        "schema_version": REUSE_SCHEMA_VERSION,
        "cohort_sha256": cohort_hash(),
        "city_count": len(locked_region_keys()),
        "grid_emission_factor_fallback": contract[
            "grid_emission_factor_fallback"
        ],
        "artifacts": records,
    }
    manifest_path = output_root / "manifest.json"
    temporary = manifest_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    temporary.replace(manifest_path)
    return manifest_path


def restore_reuse_directory(config: dict, reuse_root: str | Path) -> dict:
    """Verify and restore an explicit reuse directory into isolated roots."""
    reuse_root = Path(reuse_root).resolve()
    manifest_path = reuse_root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"paper reuse manifest missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != REUSE_SCHEMA_VERSION:
        raise ValueError("unsupported paper reuse schema")
    if manifest.get("cohort_sha256") != cohort_hash():
        raise ValueError("paper reuse cohort hash does not match tracked cohort")
    contract_fallback = load_cohort_contract()["grid_emission_factor_fallback"]
    if manifest.get("grid_emission_factor_fallback") != contract_fallback:
        raise ValueError("paper reuse GEF fallback contract is stale")

    roots = {
        "data": Path(config["data_dir"]).resolve(),
        "cache": Path(config["cache_dir"]).resolve(),
        "work": Path(config["work_dir"]).resolve(),
    }
    restored = []
    for record in manifest.get("artifacts", []):
        declaration = str(record["path"])
        prefix, separator, relative = declaration.partition(":")
        if not separator or prefix not in roots:
            raise ValueError(f"invalid paper reuse destination: {declaration}")
        source = reuse_root / "payload" / prefix / relative
        if not source.is_file():
            raise FileNotFoundError(f"paper reuse artifact missing: {source}")
        if source.stat().st_size != int(record["size_bytes"]):
            raise ValueError(f"paper reuse size mismatch: {source}")
        if _sha256(source) != record["sha256"]:
            raise ValueError(f"paper reuse SHA-256 mismatch: {source}")
        target = (roots[prefix] / relative).absolute()
        if roots[prefix] != target and roots[prefix] not in target.parents:
            raise ValueError(f"paper reuse destination escapes root: {target}")
        _atomic_copy(source, target)
        restored.append(str(target))
    return {"manifest": str(manifest_path), "restored": restored}


def validate_reused_region_sets(config: dict) -> None:
    """Require every nationwide reused CSV to cover exactly the paper cohort."""
    expected = set(locked_region_keys())
    for name in GLOBAL_DATA_ARTIFACTS:
        path = Path(config["data_dir"]) / name
        if path.suffix != ".csv" or not path.is_file():
            continue
        frame = pd.read_csv(path)
        if "region_key" not in frame.columns:
            continue
        actual = set(frame["region_key"].dropna().astype(str))
        if actual != expected:
            raise ValueError(
                f"reused {name} has wrong cohort: "
                f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
            )


def _materialize(path: Path) -> None:
    """Replace a staged symlink with an independent file copy."""
    if not path.is_symlink():
        return
    source = path.resolve()
    _atomic_copy(source, path)


def _refresh_spatial_subset(config: dict, affected: set[str]) -> Path:
    """Recompute only fallback-GEF cities, then reclassify the full cohort."""
    from .analysis.drone_spatial_concentration import (
        _analyze_city,
        _assign_classes_and_representatives,
        _attach_oracle_reduction_metrics,
    )

    output = (
        Path(config["cache_dir"])
        / "drone_spatial_concentration"
        / "city_drone_spatial_concentration.csv"
    )
    if not output.is_file():
        raise FileNotFoundError(
            f"legacy spatial-concentration seed missing: {output}"
        )
    existing = pd.read_csv(output)
    stats = pd.read_csv(Path(config["data_dir"]) / "all_region_stats.csv")
    stats = stats[stats["region_key"].astype(str).isin(locked_region_keys())].copy()
    refreshed_rows = [
        _analyze_city((row, str(config["data_dir"])))
        for row in stats[stats["region_key"].astype(str).isin(affected)].to_dict("records")
    ]
    retained = existing[
        existing["region_key"].astype(str).isin(set(locked_region_keys()) - affected)
    ].copy()
    merged = pd.concat([retained, pd.DataFrame(refreshed_rows)], ignore_index=True)
    merged = merged.sort_values("region_key").reset_index(drop=True)
    merged = _assign_classes_and_representatives(merged)
    reduction_columns = (
        "oracle_absolute_reduction_t_co2e",
        "oracle_relative_reduction_pct",
    )
    merged = merged.drop(columns=list(reduction_columns), errors="ignore")
    merged = _attach_oracle_reduction_metrics(merged, stats)
    if set(merged["region_key"].astype(str)) != set(locked_region_keys()):
        raise RuntimeError("partial spatial refresh did not cover the locked cohort")
    temporary = output.with_suffix(".csv.tmp")
    merged.to_csv(temporary, index=False)
    temporary.replace(output)
    return output


def migrate_fallback_gef_overlay(config: dict) -> dict:
    """Refresh only the five fallback-GEF cities inside an isolated data view."""
    from .analysis_workflow import _filter_analysis_view, _run_city_attribution
    from .tools.consolidate_sensitivity import consolidate as consolidate_named
    from .tools.consolidate_stats import consolidate as consolidate_stats
    from .tools.enrich_stats import enrich
    from .tools.recompute_lifecycle_parts import _process_region
    from .tools.rerun_named_scenarios import _process as process_named
    from .tools.run_distance_threshold_analysis import (
        analyze_region as analyze_distance,
        consolidate as consolidate_distance,
    )
    from .tools.run_threshold_resilience import (
        analyze_region as analyze_resilience,
        consolidate as consolidate_resilience,
    )
    from .utils.region_config import init_registry
    from .paths import PROJECT_ROOT

    contract = load_cohort_contract()
    affected = set(contract["grid_emission_factor_fallback"]["affected_region_keys"])
    study = set(locked_region_keys())
    if not affected <= study:
        raise ValueError("fallback-GEF overlay includes a city outside the cohort")
    data_root = Path(config["data_dir"])
    for region_key in affected:
        city_root = data_root / "city" / region_key
        _materialize(city_root / f"{region_key}_od_pairs.csv")
        _materialize(city_root / "region_stats.json")

    registry = init_registry(
        str(PROJECT_ROOT / "regions.json"), data_dir=str(data_root)
    )
    lifecycle = []
    for region_key in sorted(affected):
        result = _process_region(
            region_key,
            registry[region_key],
            config,
            dry_run=False,
            write_od=True,
            compare_existing=True,
        )
        if result.status != "ok":
            raise RuntimeError(f"GEF lifecycle refresh failed: {region_key}:{result.status}")
        lifecycle.append(region_key)

    consolidate_stats(str(data_root))
    enrich(str(data_root), config["sales_path"], config["gef_path"])
    _filter_analysis_view(data_root, study)

    for region_key in sorted(affected):
        key, status, _ = process_named(registry[region_key], config)
        if status != "ok":
            raise RuntimeError(f"GEF named-scenario refresh failed: {key}:{status}")
    consolidate_named(str(data_root), level_filter="city")
    _filter_analysis_view(data_root, study)

    for region_key in sorted(affected):
        result = analyze_distance(
            region_key, registry[region_key], config, force=True
        )
        if result["status"] != "ok":
            raise RuntimeError(
                f"GEF distance-threshold refresh failed: {region_key}:{result['status']}"
            )
    distance_path, distance_complete = consolidate_distance(config)
    if not distance_complete:
        raise RuntimeError(f"cohort distance threshold is partial: {distance_path}")

    for region_key in sorted(affected):
        result = analyze_resilience(
            region_key, registry[region_key], config, force=True
        )
        if result["status"] != "ok":
            raise RuntimeError(
                f"GEF threshold-resilience refresh failed: {region_key}:{result['status']}"
            )
    consolidate_resilience(config)
    _refresh_spatial_subset(config, affected)
    attribution = _run_city_attribution(config, force=True)
    _filter_analysis_view(data_root, study)
    return {
        "affected_region_keys": sorted(affected),
        "lifecycle_refreshed": lifecycle,
        "distance_threshold": str(distance_path),
        "city_attribution": str(attribution),
    }


def write_migration_difference_report(
    old_stats_path: str | Path,
    new_stats_path: str | Path,
    output_path: str | Path,
) -> Path:
    """Record the 314/0.5600 to 313/0.5366 numerical migration."""
    old = pd.read_csv(old_stats_path)
    new = pd.read_csv(new_stats_path)
    contract = load_cohort_contract()
    affected = set(contract["grid_emission_factor_fallback"]["affected_region_keys"])
    metrics = (
        "total_ebike_lifecycle",
        "total_drone_lifecycle",
        "oracle_min_lifecycle_kg",
        "oracle_min_reduction_vs_ebike_pct",
        "oracle_drone_order_ratio_pct",
    )
    old = old[old["region_level"].eq("city")].set_index("region_key")
    new = new[new["region_level"].eq("city")].set_index("region_key")
    city_changes = {}
    for region_key in sorted(affected):
        city_changes[region_key] = {
            name: {
                "old": float(pd.to_numeric(old.loc[region_key, name])),
                "new": float(pd.to_numeric(new.loc[region_key, name])),
            }
            for name in metrics
        }
    additive = (
        "total_ebike_lifecycle",
        "total_drone_lifecycle",
        "oracle_min_lifecycle_kg",
    )
    totals = {
        name: {
            "old_locked_314": float(pd.to_numeric(old[name], errors="coerce").sum()),
            "new_locked_313": float(pd.to_numeric(new[name], errors="coerce").sum()),
        }
        for name in additive
    }
    unchanged = sorted(set(new.index) - affected)
    unchanged_max_abs_diff = {}
    for name in metrics:
        old_values = pd.to_numeric(old.loc[unchanged, name], errors="coerce").to_numpy()
        new_values = pd.to_numeric(new.loc[unchanged, name], errors="coerce").to_numpy()
        difference = np.abs(old_values - new_values)
        finite = difference[np.isfinite(difference)]
        maximum = float(finite.max()) if len(finite) else 0.0
        unchanged_max_abs_diff[name] = maximum
        if not np.allclose(old_values, new_values, rtol=0.0, atol=1e-9, equal_nan=True):
            raise RuntimeError(f"unaffected city metric changed during GEF overlay: {name}")
    payload = {
        "schema_version": 1,
        "old_city_count": int(len(old.dropna(subset=["total_ebike_lifecycle"]))),
        "new_city_count": int(len(new)),
        "excluded_region_keys": sorted(set(old.index) - set(new.index)),
        "old_fallback_kg_co2e_per_kwh": 0.5600,
        "new_fallback_kg_co2e_per_kwh": contract[
            "grid_emission_factor_fallback"
        ]["kg_co2e_per_kwh"],
        "affected_city_metrics": city_changes,
        "national_totals_kg_co2e": totals,
        "unaffected_city_count": len(unchanged),
        "unaffected_max_abs_diff": unchanged_max_abs_diff,
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output_path)
    return output_path
