"""Run the standalone city distance-threshold lifecycle analysis.

This command intentionally does not participate in the main pipeline.  The
remote shell wrapper invokes it once per city so each process can have an
independent memory limit.

Examples:
    python -m tools.run_distance_threshold_analysis --env remote --region beijing
    python -m tools.run_distance_threshold_analysis --env remote --consolidate
    python -m tools.run_distance_threshold_analysis --list-regions
"""

from __future__ import annotations

import argparse
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

from drone_emission.analysis.distance_threshold import (  # noqa: E402
    ANALYSIS_VERSION,
    analyze_city_inputs,
    validate_inputs,
)
from drone_emission.utils import load_config  # noqa: E402
from drone_emission.utils.region_config import RegionConfig, init_registry  # noqa: E402

REGIONS_JSON = PROJECT_ROOT / "regions.json"
REQUIRED_COLUMNS = (
    "distance_km",
    "road_distance_m",
    "annual_sales",
    "ebike_lifecycle_g",
    "drone_lifecycle_g",
)
SUMMARY_COLUMNS = (
    "region_key",
    "region_name_cn",
    "region_name_en",
    "distance_basis",
    "policy_case",
    "optimal_threshold_km",
    "max_observed_distance_km",
    "all_ebike_lifecycle_kg",
    "optimal_lifecycle_kg",
    "annual_reduction_kg",
    "annual_reduction_pct",
    "drone_annual_orders",
    "drone_order_ratio_pct",
    "n_od_pairs_total",
    "n_od_pairs_optimized",
    "n_od_pairs_unreachable",
    "unreachable_annual_orders",
    "total_annual_orders",
    "analysis_version",
    "input_size_bytes",
    "input_mtime_ns",
    "generated_at_utc",
)
CURVE_COLUMNS = (
    "region_key",
    "region_name_en",
    "distance_basis",
    "threshold_km",
    "policy_case",
    "mixed_lifecycle_kg",
    "annual_reduction_kg",
    "annual_reduction_pct",
    "drone_annual_orders",
    "drone_order_ratio_pct",
    "is_exact_optimum",
    "analysis_version",
    "input_size_bytes",
    "input_mtime_ns",
)


def _city_regions(config: dict) -> dict[str, RegionConfig]:
    registry = init_registry(str(REGIONS_JSON), data_dir=config["data_dir"])
    regions = registry.list_by_level("city")
    requested = set(config.get("study_region_keys", ()))
    if requested:
        regions = {key: region for key, region in regions.items() if key in requested}
    return regions


def _lookup_region(
    regions: dict[str, RegionConfig], requested: str
) -> tuple[str, RegionConfig]:
    requested_lower = requested.lower()
    for key, region in regions.items():
        if key.lower() == requested_lower:
            return key, region
        if region.region_name_en.lower() == requested_lower:
            return key, region
    raise KeyError(f"city region not found: {requested}")


def _od_path(config: dict, region: RegionConfig) -> Path:
    name = region.region_name_en.lower()
    return Path(config["data_dir"]) / "city" / name / f"{name}_od_pairs.csv"


def _output_dir(config: dict, region: RegionConfig) -> Path:
    return (
        Path(config["data_dir"])
        / "city"
        / region.region_name_en.lower()
        / "analysis"
        / "distance_threshold"
    )


def _fingerprint(path: Path) -> dict:
    stat = path.stat()
    return {
        "analysis_version": ANALYSIS_VERSION,
        "input_size_bytes": int(stat.st_size),
        "input_mtime_ns": int(stat.st_mtime_ns),
    }


def _atomic_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(fd)
    try:
        df.to_csv(temporary, index=False)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _atomic_json(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(fd)
    try:
        with open(temporary, "w", encoding="utf-8") as handle:
            json.dump(data, handle, ensure_ascii=False, indent=2)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _summary_is_current(summary_path: Path, fingerprint: dict) -> bool:
    if not summary_path.exists():
        return False
    try:
        summary = pd.read_csv(
            summary_path,
            usecols=(
                "distance_basis",
                "analysis_version",
                "input_size_bytes",
                "input_mtime_ns",
            ),
        )
    except (OSError, ValueError, pd.errors.ParserError):
        return False
    if set(summary["distance_basis"]) != {"road", "straight"} or len(summary) != 2:
        return False
    return bool(
        (summary["analysis_version"].astype(str) == fingerprint["analysis_version"]).all()
        and (summary["input_size_bytes"] == fingerprint["input_size_bytes"]).all()
        and (summary["input_mtime_ns"] == fingerprint["input_mtime_ns"]).all()
    )


def _status(
    region_key: str,
    region: RegionConfig,
    status: str,
    elapsed_seconds: float,
    message: str = "",
) -> dict:
    return {
        "region_key": region_key,
        "region_name_en": region.region_name_en,
        "status": status,
        "elapsed_seconds": round(elapsed_seconds, 3),
        "message": message,
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def analyze_region(
    region_key: str,
    region: RegionConfig,
    config: dict,
    *,
    force: bool = False,
    include_curve: bool = False,
    n_uniform: int = 201,
) -> dict:
    """Analyze one city and write its dedicated outputs."""

    started = time.monotonic()
    output_dir = _output_dir(config, region)
    summary_path = output_dir / "summary.csv"
    curve_path = output_dir / "curve.csv"
    status_path = output_dir / "status.json"
    od_path = _od_path(config, region)

    _atomic_json(
        _status(region_key, region, "running", 0.0),
        status_path,
    )

    if not od_path.exists():
        result = _status(
            region_key,
            region,
            "missing_input",
            time.monotonic() - started,
            str(od_path),
        )
        _atomic_json(result, status_path)
        return result

    fingerprint = _fingerprint(od_path)
    if not force and not include_curve and _summary_is_current(
        summary_path, fingerprint
    ):
        result = _status(
            region_key, region, "skipped", time.monotonic() - started
        )
        _atomic_json(result, status_path)
        return result

    try:
        frame = pd.read_csv(
            od_path,
            usecols=list(REQUIRED_COLUMNS),
            dtype={column: "float64" for column in REQUIRED_COLUMNS},
        )
        inputs = validate_inputs(
            road_distance_km=frame["road_distance_m"].to_numpy(copy=False) / 1000.0,
            straight_distance_km=frame["distance_km"].to_numpy(copy=False),
            annual_sales=frame["annual_sales"].to_numpy(copy=False),
            ebike_lifecycle_g=frame["ebike_lifecycle_g"].to_numpy(copy=False),
            drone_lifecycle_g=frame["drone_lifecycle_g"].to_numpy(copy=False),
        )
        del frame

        summaries, curves = analyze_city_inputs(
            inputs,
            include_curve=include_curve,
            n_uniform=n_uniform,
        )
        generated_at = datetime.now(timezone.utc).isoformat()
        for row in summaries:
            row.update(
                {
                    "region_key": region_key,
                    "region_name_cn": region.region_name_cn,
                    "region_name_en": region.region_name_en,
                    **fingerprint,
                    "generated_at_utc": generated_at,
                }
            )
        summary_df = pd.DataFrame(summaries).loc[:, list(SUMMARY_COLUMNS)]
        _atomic_csv(summary_df, summary_path)

        if include_curve:
            for row in curves:
                row.update(
                    {
                        "region_key": region_key,
                        "region_name_en": region.region_name_en,
                        **fingerprint,
                    }
                )
            curve_df = pd.DataFrame(curves).loc[:, list(CURVE_COLUMNS)]
            _atomic_csv(curve_df, curve_path)

        result = _status(
            region_key,
            region,
            "ok",
            time.monotonic() - started,
            f"rows={len(inputs.annual_sales):,}",
        )
    except Exception as exc:  # per-city isolation is intentional
        result = _status(
            region_key,
            region,
            "error",
            time.monotonic() - started,
            f"{type(exc).__name__}: {exc}",
        )
        traceback.print_exc()

    _atomic_json(result, status_path)
    return result


def consolidate(config: dict) -> tuple[Path, bool]:
    """Rebuild the cross-city table from current per-city summaries."""

    regions = _city_regions(config)
    frames: list[pd.DataFrame] = []
    statuses: list[dict] = []
    complete = True

    for region_key, region in regions.items():
        od_path = _od_path(config, region)
        summary_path = _output_dir(config, region) / "summary.csv"
        status_path = _output_dir(config, region) / "status.json"
        if status_path.exists():
            try:
                with status_path.open(encoding="utf-8") as handle:
                    status = json.load(handle)
            except (OSError, json.JSONDecodeError):
                status = {}
        else:
            status = {}

        current = False
        if od_path.exists() and status.get("status") in {"ok", "skipped"}:
            current = _summary_is_current(summary_path, _fingerprint(od_path))
        if current:
            frames.append(pd.read_csv(summary_path))
        else:
            complete = False

        status.update(
            {
                "region_key": region_key,
                "region_name_en": region.region_name_en,
                "summary_current": current,
            }
        )
        statuses.append(status)

    data_dir = Path(config["data_dir"])
    final_path = data_dir / "all_city_distance_threshold_summary.csv"
    partial_path = data_dir / "all_city_distance_threshold_summary.partial.csv"
    status_path = data_dir / "all_city_distance_threshold_status.csv"
    _atomic_csv(pd.DataFrame(statuses), status_path)

    if not frames:
        raise RuntimeError("no current city distance-threshold summaries found")
    merged = pd.concat(frames, ignore_index=True).loc[:, list(SUMMARY_COLUMNS)]
    if complete:
        _atomic_csv(merged, final_path)
        if partial_path.exists():
            partial_path.unlink()
        return final_path, True
    _atomic_csv(merged, partial_path)
    return partial_path, False


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Standalone city distance-threshold lifecycle analysis"
    )
    parser.add_argument("--env", default=None)
    parser.add_argument("--region", action="append", default=[])
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--consolidate", action="store_true")
    parser.add_argument("--list-regions", action="store_true")
    args = parser.parse_args()

    config = load_config(PROJECT_ROOT / "experiment.toml", env=args.env)
    regions = _city_regions(config)

    if args.list_regions:
        for region_key in regions:
            print(region_key)
        return

    if args.consolidate:
        output, complete = consolidate(config)
        print(f"Wrote {'complete' if complete else 'partial'} summary: {output}")
        if not complete:
            raise SystemExit(1)
        return

    if not args.region:
        parser.error("provide at least one --region, or use --consolidate")

    failed = False
    for requested in args.region:
        try:
            region_key, region = _lookup_region(regions, requested)
        except KeyError as exc:
            print(str(exc), file=sys.stderr)
            failed = True
            continue
        result = analyze_region(
            region_key,
            region,
            config,
            force=args.force,
        )
        print(
            f"{region_key}: {result['status']} "
            f"({result['elapsed_seconds']:.1f}s) {result['message']}",
            flush=True,
        )
        failed |= result["status"] not in {"ok", "skipped"}
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
