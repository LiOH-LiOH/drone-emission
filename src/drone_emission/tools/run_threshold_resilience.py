"""Run fixed-threshold probabilistic lifecycle-emission robustness analysis.

Examples:
    python -m tools.run_threshold_resilience --env remote --region beijing --save-samples
    python -m tools.run_threshold_resilience --env remote --consolidate
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src" / "drone_emission"))

from emission.cal_single_delivery import (  # noqa: E402
    create_drone_params,
    create_ebike_params,
    get_emission_params,
)
from sensitivity.threshold_resilience import (  # noqa: E402
    ANALYSIS_VERSION,
    aggregate_fixed_threshold,
    evaluate_samples,
    mode_sample,
    resolve_distributions,
    sample_parameters,
    spearman_drivers,
    summarize_samples,
)
from utils import load_config  # noqa: E402
from utils.grid_emission_factor import fetch_factor, normalize_province_name  # noqa: E402
from utils.province import get_province_from_district  # noqa: E402
from utils.region_config import RegionConfig, init_registry  # noqa: E402

REGIONS_JSON = PROJECT_ROOT / "regions.json"
OD_COLUMNS = ("distance_km", "road_distance_m", "annual_sales")
SUMMARY_FILENAME = "summary.csv"
DRIVERS_FILENAME = "drivers.csv"
SAMPLES_FILENAME = "samples.csv.gz"
METADATA_FILENAME = "metadata.json"
STATUS_FILENAME = "status.json"
BASELINE_TOLERANCE_PP = 0.1


def _city_regions(config: dict) -> dict[str, RegionConfig]:
    return init_registry(
        str(REGIONS_JSON), data_dir=config["data_dir"]
    ).list_by_level("city")


def _lookup_region(
    regions: dict[str, RegionConfig], requested: str
) -> tuple[str, RegionConfig]:
    requested_lower = requested.lower()
    for key, region in regions.items():
        if key.lower() == requested_lower or region.region_name_en.lower() == requested_lower:
            return key, region
    raise KeyError(f"city region not found: {requested}")


def _city_dir(config: dict, region: RegionConfig) -> Path:
    return Path(config["data_dir"]) / "city" / region.region_name_en.lower()


def _od_path(config: dict, region: RegionConfig) -> Path:
    name = region.region_name_en.lower()
    return _city_dir(config, region) / f"{name}_od_pairs.csv"


def _threshold_path(config: dict, region: RegionConfig) -> Path:
    return _city_dir(config, region) / "analysis" / "distance_threshold" / "summary.csv"


def _output_dir(config: dict, region: RegionConfig) -> Path:
    return _city_dir(config, region) / "analysis" / "threshold_resilience"


def _atomic_csv(frame: pd.DataFrame, path: Path, *, compression=None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = ".tmp.gz" if compression == "gzip" else ".tmp"
    fd, temporary = tempfile.mkstemp(prefix=f".{path.stem}.", suffix=suffix, dir=path.parent)
    os.close(fd)
    try:
        frame.to_csv(temporary, index=False, compression=compression)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _atomic_json(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(fd)
    try:
        with open(temporary, "w", encoding="utf-8") as handle:
            json.dump(data, handle, ensure_ascii=False, indent=2, sort_keys=True)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _resolve_province(region: RegionConfig, config: dict) -> str:
    province = region.get_stat("province")
    if province:
        return normalize_province_name(province)
    return normalize_province_name(
        get_province_from_district(region.region_name_cn, config["sales_path"])
    )


def _read_current_threshold(
    threshold_path: Path,
    od_path: Path,
    distance_basis: str,
) -> pd.Series:
    if not threshold_path.exists():
        raise FileNotFoundError(
            f"distance-threshold summary missing: {threshold_path}; run "
            "tools.run_distance_threshold_analysis first"
        )
    summary = pd.read_csv(threshold_path)
    required = {
        "distance_basis",
        "optimal_threshold_km",
        "annual_reduction_pct",
        "input_size_bytes",
        "input_mtime_ns",
    }
    missing = sorted(required - set(summary.columns))
    if missing:
        raise ValueError(f"distance-threshold summary missing columns: {missing}")
    selected = summary[summary["distance_basis"] == distance_basis]
    if len(selected) != 1:
        raise ValueError(
            f"expected one {distance_basis!r} threshold row, found {len(selected)}"
        )
    row = selected.iloc[0]
    stat = od_path.stat()
    if (
        int(row["input_size_bytes"]) != stat.st_size
        or int(row["input_mtime_ns"]) != stat.st_mtime_ns
    ):
        raise ValueError(
            "distance-threshold summary is stale relative to the OD CSV; rerun "
            "tools.run_distance_threshold_analysis"
        )
    return row


def _config_digest(section: dict) -> str:
    canonical = json.dumps(section, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _analysis_config_digest(config: dict) -> str:
    """Fingerprint effective model inputs that are not city-specific."""
    gef_path = Path(config["gef_path"])
    payload = {
        "threshold_resilience": config.get("threshold_resilience", {}) or {},
        "emission_params": config.get("emission_params", {}) or {},
        "gef_sha256": _file_digest(gef_path),
    }
    return _config_digest(payload)


def _cache_is_fresh(
    config: dict,
    region: RegionConfig,
    *,
    require_samples: bool,
    config_digest: str | None = None,
) -> tuple[bool, str]:
    """Validate every input needed to reuse or consolidate one city result."""
    od_path = _od_path(config, region)
    if not od_path.exists():
        return False, "OD CSV missing"
    threshold_path = _threshold_path(config, region)
    try:
        _read_current_threshold(
            threshold_path,
            od_path,
            str((config.get("threshold_resilience", {}) or {}).get("distance_basis", "road")),
        )
    except (FileNotFoundError, ValueError) as exc:
        return False, str(exc)

    output_dir = _output_dir(config, region)
    required_paths = [
        output_dir / SUMMARY_FILENAME,
        output_dir / DRIVERS_FILENAME,
        output_dir / METADATA_FILENAME,
    ]
    if require_samples:
        required_paths.append(output_dir / SAMPLES_FILENAME)
    missing = [path.name for path in required_paths if not path.exists()]
    if missing:
        return False, f"output missing: {', '.join(missing)}"

    try:
        with (output_dir / METADATA_FILENAME).open(encoding="utf-8") as handle:
            metadata = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        return False, f"invalid metadata: {exc}"

    stat = od_path.stat()
    expected_config_digest = config_digest or _analysis_config_digest(config)
    checks = (
        (metadata.get("analysis_version") == ANALYSIS_VERSION, "analysis version changed"),
        (metadata.get("config_sha256") == expected_config_digest, "analysis configuration changed"),
        (metadata.get("od_input_size_bytes") == stat.st_size, "OD size changed"),
        (metadata.get("od_input_mtime_ns") == stat.st_mtime_ns, "OD timestamp changed"),
        (
            metadata.get("threshold_summary_sha256") == _file_digest(threshold_path),
            "distance-threshold summary changed",
        ),
    )
    for valid, message in checks:
        if not valid:
            return False, message
    return True, "current result is reusable"


def inventory_regions(
    config: dict, *, require_samples: bool, force: bool
) -> pd.DataFrame:
    """Classify all registered cities and order runnable work largest-first."""
    rows = []
    config_digest = _analysis_config_digest(config)
    distance_basis = str(
        (config.get("threshold_resilience", {}) or {}).get("distance_basis", "road")
    )
    for region_key, region in _city_regions(config).items():
        od_path = _od_path(config, region)
        threshold_path = _threshold_path(config, region)
        size = od_path.stat().st_size if od_path.exists() else 0
        status = "ready"
        message = "prerequisites current"
        if not od_path.exists():
            status, message = "missing_od", f"OD CSV missing: {od_path}"
        elif not threshold_path.exists():
            status, message = (
                "missing_threshold",
                f"distance-threshold summary missing: {threshold_path}",
            )
        else:
            try:
                _read_current_threshold(threshold_path, od_path, distance_basis)
            except ValueError as exc:
                status = (
                    "stale_threshold"
                    if "stale relative" in str(exc)
                    else "invalid_threshold"
                )
                message = str(exc)
            if status == "ready" and not force:
                fresh, cache_message = _cache_is_fresh(
                    config,
                    region,
                    require_samples=require_samples,
                    config_digest=config_digest,
                )
                if fresh:
                    status, message = "cached", cache_message
                else:
                    message = cache_message
        rows.append(
            {
                "region_key": region_key,
                "status": status,
                "od_size_bytes": size,
                "region_name_cn": region.region_name_cn,
                "region_name_en": region.region_name_en,
                "message": message,
            }
        )
    frame = pd.DataFrame(rows)
    frame["_runnable"] = frame["status"].isin(("ready", "cached")).astype(int)
    frame = frame.sort_values(
        ["_runnable", "od_size_bytes", "region_key"],
        ascending=[False, False, True],
    ).drop(columns="_runnable")
    return frame.reset_index(drop=True)


def build_status_report(
    config: dict, preflight_path: Path, *, require_samples: bool
) -> pd.DataFrame:
    """Combine preflight exclusions with worker status for one batch run."""
    preflight = pd.read_csv(preflight_path)
    regions = _city_regions(config)
    config_digest = _analysis_config_digest(config)
    rows = []
    for row in preflight.to_dict("records"):
        region_key = str(row["region_key"])
        initial_status = str(row["status"])
        final_status = initial_status
        message = str(row["message"])
        if initial_status in ("ready", "cached"):
            status_path = _output_dir(config, regions[region_key]) / STATUS_FILENAME
            try:
                with status_path.open(encoding="utf-8") as handle:
                    worker_status = json.load(handle)
                final_status = str(worker_status.get("status", "error"))
                message = str(worker_status.get("message", "worker status missing message"))
            except (OSError, json.JSONDecodeError) as exc:
                final_status, message = "error", f"invalid worker status: {exc}"
            fresh, cache_message = _cache_is_fresh(
                config,
                regions[region_key],
                require_samples=require_samples,
                config_digest=config_digest,
            )
            if final_status in ("ok", "cached") and not fresh:
                final_status, message = "error", cache_message
        row["initial_status"] = initial_status
        row["status"] = final_status
        row["message"] = message
        rows.append(row)
    return pd.DataFrame(rows)


def analyze_region(
    region_key: str,
    region: RegionConfig,
    config: dict,
    *,
    save_samples: bool = False,
    force: bool = False,
) -> dict:
    started = time.monotonic()
    output_dir = _output_dir(config, region)
    status_path = output_dir / STATUS_FILENAME
    output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_json(
        {
            "region_key": region_key,
            "status": "running",
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        },
        status_path,
    )

    try:
        od_path = _od_path(config, region)
        if not od_path.exists():
            raise FileNotFoundError(f"OD CSV missing: {od_path}")

        cfg = config.get("threshold_resilience", {}) or {}
        distance_basis = str(cfg.get("distance_basis", "road"))
        if distance_basis != "road":
            raise ValueError("threshold resilience currently requires distance_basis='road'")
        threshold_row = _read_current_threshold(
            _threshold_path(config, region), od_path, distance_basis
        )
        threshold_km = float(threshold_row["optimal_threshold_km"])

        summary_path = output_dir / SUMMARY_FILENAME
        metadata_path = output_dir / METADATA_FILENAME
        config_digest = _analysis_config_digest(config)
        if not force:
            fresh, _ = _cache_is_fresh(
                config,
                region,
                require_samples=save_samples,
                config_digest=config_digest,
            )
            if fresh:
                result = {
                    "region_key": region_key,
                    "status": "cached",
                    "elapsed_seconds": round(time.monotonic() - started, 3),
                    "message": str(summary_path),
                    "updated_at_utc": datetime.now(timezone.utc).isoformat(),
                }
                _atomic_json(result, status_path)
                return result

        frame = pd.read_csv(
            od_path,
            usecols=list(OD_COLUMNS),
            dtype={column: "float64" for column in OD_COLUMNS},
        )
        aggregates = aggregate_fixed_threshold(
            road_distance_km=frame["road_distance_m"].to_numpy(copy=False) / 1000.0,
            straight_distance_km=frame["distance_km"].to_numpy(copy=False),
            annual_sales=frame["annual_sales"].to_numpy(copy=False),
            threshold_km=threshold_km,
        )
        del frame

        province = _resolve_province(region, config)
        province_gef = fetch_factor(province, config["gef_path"])
        ebike_params = create_ebike_params(config)
        drone_params = create_drone_params(config)
        rider_weight = float(get_emission_params(config)["rider_weight"])
        specs = resolve_distributions(config, ebike_params, drone_params)

        tolerance = float(cfg.get("positive_tolerance_pct", 1e-9))
        baseline_eval = evaluate_samples(
            mode_sample(specs), aggregates, ebike_params, drone_params,
            province_gef, rider_weight, tolerance,
        )
        baseline_reduction = float(baseline_eval.iloc[0]["reduction_pct"])
        threshold_baseline = float(threshold_row["annual_reduction_pct"])
        baseline_gap = abs(baseline_reduction - threshold_baseline)
        if baseline_gap > BASELINE_TOLERANCE_PP:
            raise AssertionError(
                "resilience baseline does not reproduce threshold analysis within "
                f"{BASELINE_TOLERANCE_PP} pp: resilience={baseline_reduction:.6f}, "
                f"threshold={threshold_baseline:.6f}, gap={baseline_gap:.6f}"
            )

        n_samples = int(cfg.get("n_samples", 10000))
        seed = int(cfg.get("seed", 42))
        independent = sample_parameters(specs, n_samples, seed)
        independent_eval = evaluate_samples(
            independent, aggregates, ebike_params, drone_params,
            province_gef, rider_weight, tolerance,
        )
        summaries = [
            summarize_samples(
                independent_eval, "independent", aggregates, baseline_reduction
            )
        ]
        drivers = spearman_drivers(independent_eval)

        evaluated_frames = []
        if save_samples:
            saved_independent = independent_eval.copy()
            saved_independent.insert(0, "sample_id", range(n_samples))
            saved_independent.insert(0, "scenario", "independent")
            evaluated_frames.append(saved_independent)

        stress_rho = None
        if bool(cfg.get("correlated_stress", True)):
            stress_rho = float(cfg.get("stress_spearman_rho", 0.5))
            stress = sample_parameters(
                specs, n_samples, seed + 1, stress_spearman_rho=stress_rho
            )
            stress_eval = evaluate_samples(
                stress, aggregates, ebike_params, drone_params,
                province_gef, rider_weight, tolerance,
            )
            summaries.append(
                summarize_samples(
                    stress_eval, "correlated_adverse_stress", aggregates,
                    baseline_reduction,
                )
            )
            if save_samples:
                saved_stress = stress_eval.copy()
                saved_stress.insert(0, "sample_id", range(n_samples))
                saved_stress.insert(0, "scenario", "correlated_adverse_stress")
                evaluated_frames.append(saved_stress)

        summary = pd.DataFrame(summaries)
        for column, value in reversed(
            (
                ("region_key", region_key),
                ("region_name_cn", region.region_name_cn),
                ("region_name_en", region.region_name_en),
                ("province", province),
                ("province_gef_kg_per_kwh", province_gef),
                ("distance_basis", distance_basis),
            )
        ):
            summary.insert(0, column, value)
        drivers.insert(0, "region_key", region_key)

        _atomic_csv(summary, summary_path)
        _atomic_csv(drivers, output_dir / DRIVERS_FILENAME)
        if save_samples:
            _atomic_csv(
                pd.concat(evaluated_frames, ignore_index=True),
                output_dir / SAMPLES_FILENAME,
                compression="gzip",
            )

        stat = od_path.stat()
        metadata = {
            "analysis_version": ANALYSIS_VERSION,
            "region_key": region_key,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "config_sha256": config_digest,
            "od_path": str(od_path),
            "od_input_size_bytes": stat.st_size,
            "od_input_mtime_ns": stat.st_mtime_ns,
            "threshold_summary_path": str(_threshold_path(config, region)),
            "threshold_summary_sha256": _file_digest(_threshold_path(config, region)),
            "distance_basis": distance_basis,
            "fixed_threshold_km": threshold_km,
            "province_gef_kg_per_kwh": province_gef,
            "n_samples_per_scenario": n_samples,
            "seed_independent": seed,
            "seed_correlated_stress": seed + 1 if stress_rho is not None else None,
            "stress_spearman_rho": stress_rho,
            "positive_tolerance_pct": tolerance,
            "baseline_threshold_reduction_pct": threshold_baseline,
            "baseline_resilience_reduction_pct": baseline_reduction,
            "baseline_gap_pp": baseline_gap,
            "probability_scope": (
                "Conditional on the current OD distribution, road network, provincial "
                "grid-emission factor, fixed baseline road-distance threshold, and the "
                "configured triangular parameter distributions."
            ),
            "distributions": {
                name: [spec.minimum, spec.mode, spec.maximum]
                for name, spec in specs.items()
            },
        }
        _atomic_json(metadata, metadata_path)

        result = {
            "region_key": region_key,
            "status": "ok",
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "message": str(summary_path),
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        }
    except Exception as exc:
        traceback.print_exc()
        result = {
            "region_key": region_key,
            "status": "error",
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "message": f"{type(exc).__name__}: {exc}",
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        }
    _atomic_json(result, status_path)
    return result


def consolidate(config: dict, *, require_samples: bool = False) -> tuple[Path, Path]:
    summaries = []
    drivers = []
    config_digest = _analysis_config_digest(config)
    for _, region in _city_regions(config).items():
        output_dir = _output_dir(config, region)
        summary_path = output_dir / SUMMARY_FILENAME
        drivers_path = output_dir / DRIVERS_FILENAME
        fresh, _ = _cache_is_fresh(
            config,
            region,
            require_samples=require_samples,
            config_digest=config_digest,
        )
        if fresh:
            summaries.append(pd.read_csv(summary_path))
            drivers.append(pd.read_csv(drivers_path))
    if not summaries:
        raise RuntimeError("no threshold-resilience outputs found")
    data_dir = Path(config["data_dir"])
    summary_out = data_dir / "all_threshold_resilience.csv"
    drivers_out = data_dir / "all_threshold_resilience_drivers.csv"
    _atomic_csv(pd.concat(summaries, ignore_index=True), summary_out)
    _atomic_csv(pd.concat(drivers, ignore_index=True), drivers_out)
    return summary_out, drivers_out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fixed road-threshold emission-reduction resilience"
    )
    parser.add_argument("--env", default=None)
    parser.add_argument("--region", action="append", default=[])
    parser.add_argument("--save-samples", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--consolidate", action="store_true")
    parser.add_argument("--list-regions", action="store_true")
    parser.add_argument("--inventory-out", type=Path)
    parser.add_argument("--status-report-out", type=Path)
    parser.add_argument("--preflight-file", type=Path)
    args = parser.parse_args()

    config = load_config(PROJECT_ROOT / "experiment.toml", env=args.env)
    regions = _city_regions(config)
    if args.list_regions:
        for region_key in sorted(regions):
            print(region_key)
        return
    if args.inventory_out:
        inventory = inventory_regions(
            config, require_samples=args.save_samples, force=args.force
        )
        _atomic_csv(inventory, args.inventory_out)
        print(f"Wrote {args.inventory_out}")
        return
    if args.status_report_out:
        if not args.preflight_file:
            parser.error("--status-report-out requires --preflight-file")
        report = build_status_report(
            config, args.preflight_file, require_samples=args.save_samples
        )
        _atomic_csv(report, args.status_report_out)
        print(f"Wrote {args.status_report_out}")
        return
    if args.consolidate:
        summary, drivers = consolidate(config, require_samples=args.save_samples)
        print(f"Wrote {summary}")
        print(f"Wrote {drivers}")
        return
    if not args.region:
        parser.error("provide at least one --region, or use --consolidate")

    failures = []
    for requested in args.region:
        region_key, region = _lookup_region(regions, requested)
        result = analyze_region(
            region_key, region, config,
            save_samples=args.save_samples,
            force=args.force,
        )
        print(
            f"{region_key}: {result['status']} ({result['elapsed_seconds']:.1f}s) "
            f"{result['message']}"
        )
        if result["status"] == "error":
            failures.append(region_key)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
