"""Orchestrate retained paper analyses without modifying city result files."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import pandas as pd

from .paths import PROJECT_ROOT


DERIVED_GLOBAL_NAMES = {
    "all_sensitivity_named.csv",
    "all_city_distance_threshold_summary.csv",
    "all_city_distance_threshold_summary.partial.csv",
    "all_city_distance_threshold_status.csv",
    "all_threshold_resilience.csv",
    "all_threshold_resilience_drivers.csv",
}


def analysis_root(config: dict) -> Path:
    return Path(config["work_dir"]) / "analysis"


def analysis_data_dir(config: dict) -> Path:
    """Return the compatibility data view used by retained analyses and plots."""
    return analysis_root(config) / "data"


def _is_derived(relative: Path) -> bool:
    return (
        relative.name in DERIVED_GLOBAL_NAMES
        or relative.name.endswith("_named.csv")
        or "analysis" in relative.parts
    )


def _stage_file(source: Path, target: Path, *, copy_file: bool) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        return
    if copy_file:
        shutil.copy2(source, target)
    else:
        target.symlink_to(source.resolve())


def prepare_analysis_data_view(config: dict) -> Path:
    """Mirror inputs into work while copying every file an analysis may replace.

    Large immutable city inputs (notably OD, building and road-network files) are
    represented by read-only-use symlinks. Existing derived files are copied so
    direct writers and atomic replacements can never reach the configured data
    root through a symlink.
    """
    source_root = Path(config["data_dir"]).resolve()
    target_root = analysis_data_dir(config)
    target_root.mkdir(parents=True, exist_ok=True)
    if source_root == target_root.resolve():
        raise ValueError("analysis work root must be independent of data root")

    for source in source_root.iterdir():
        if source.is_file():
            relative = source.relative_to(source_root)
            _stage_file(source, target_root / relative, copy_file=_is_derived(relative))

    for level in ("city", "district"):
        level_root = source_root / level
        if not level_root.is_dir():
            continue
        for region_root in sorted(path for path in level_root.iterdir() if path.is_dir()):
            for source in region_root.rglob("*"):
                if not source.is_file():
                    continue
                relative = source.relative_to(source_root)
                _stage_file(
                    source,
                    target_root / relative,
                    copy_file=_is_derived(relative),
                )
    for directory in sorted(path for path in source_root.iterdir() if path.is_dir()):
        if directory.name in {"city", "district"}:
            continue
        for source in directory.rglob("*"):
            if not source.is_file():
                continue
            relative = source.relative_to(source_root)
            _stage_file(
                source,
                target_root / relative,
                copy_file=_is_derived(relative),
            )
    return target_root


def _view_config(config: dict) -> dict:
    view = dict(config)
    view["data_dir"] = str(prepare_analysis_data_view(config))
    # Global inputs remain authoritative and read-only to analysis routines.
    view["sales_path"] = config["sales_path"]
    view["gef_path"] = config["gef_path"]
    return view


def _run_named_scenarios(config: dict, *, force: bool) -> Path:
    from .tools.consolidate_sensitivity import consolidate
    from .tools.rerun_named_scenarios import _process, _select_regions
    from .utils.region_config import init_registry

    output = Path(config["data_dir"]) / "all_sensitivity_named.csv"
    if output.exists() and not force:
        return output
    registry = init_registry(
        str(PROJECT_ROOT / "regions.json"), data_dir=config["data_dir"]
    )
    failures = []
    regions = _select_regions(registry, config["data_dir"], "city", None, False)
    for _, region in regions:
        region_key, status, _ = _process(region, config)
        if status != "ok":
            failures.append(f"{region_key}:{status}")
    if failures:
        raise RuntimeError("named-scenario analysis incomplete: " + ", ".join(failures))
    consolidate(config["data_dir"], level_filter="city")
    if not output.exists():
        raise RuntimeError("named-scenario consolidation produced no output")
    return output


def _run_distance_threshold(config: dict, *, force: bool) -> Path:
    from .tools.run_distance_threshold_analysis import (
        _city_regions,
        analyze_region,
        consolidate,
    )

    output = Path(config["data_dir"]) / "all_city_distance_threshold_summary.csv"
    if output.exists() and not force:
        return output
    failures = []
    for region_key, region in _city_regions(config).items():
        result = analyze_region(region_key, region, config, force=force)
        if result["status"] not in {"ok", "skipped"}:
            failures.append(f"{region_key}:{result['status']}")
    if failures:
        raise RuntimeError("distance-threshold analysis incomplete: " + ", ".join(failures))
    output, complete = consolidate(config)
    if not complete:
        raise RuntimeError(f"distance-threshold consolidation is partial: {output}")
    return output


def _run_threshold_resilience(config: dict, *, force: bool) -> Path:
    from .tools.run_threshold_resilience import (
        _city_regions,
        analyze_region,
        consolidate,
    )

    output = Path(config["data_dir"]) / "all_threshold_resilience.csv"
    if output.exists() and not force:
        return output
    failures = []
    regions = _city_regions(config)
    for region_key, region in regions.items():
        result = analyze_region(region_key, region, config, force=force)
        if result["status"] not in {"ok", "cached"}:
            failures.append(f"{region_key}:{result['status']}")
    if failures:
        raise RuntimeError("threshold-resilience analysis incomplete: " + ", ".join(failures))
    output, _ = consolidate(config)
    summary = pd.read_csv(output, usecols=["region_key"])
    if summary["region_key"].nunique() != len(regions):
        raise RuntimeError("threshold-resilience consolidation omitted cities")
    return output


def _run_spatial_concentration(config: dict, *, force: bool) -> Path:
    from .analysis.drone_spatial_concentration import run_drone_spatial_concentration

    results_dir = Path(config["cache_dir"]) / "drone_spatial_concentration"
    output = results_dir / "city_drone_spatial_concentration.csv"
    if output.exists() and not force:
        return output
    run_drone_spatial_concentration(
        stats_path=Path(config["data_dir"]) / "all_region_stats.csv",
        data_dir=config["data_dir"],
        results_dir=results_dir,
        workers=int(os.environ.get("DRONE_EMISSION_ANALYSIS_WORKERS", "1")),
        force=force,
    )
    return output


def _run_city_attribution(config: dict, *, force: bool) -> Path:
    from .analysis.city_attribution import (
        build_r2_city_attribution,
        write_r2_city_attribution_report,
    )

    root = analysis_root(config) / "city_attribution"
    report = root / "r2_city_attribution_analysis_report.json"
    prepared = root / "r2_city_attribution_data.csv"
    if report.exists() and prepared.exists() and not force:
        return report
    frame, payload = build_r2_city_attribution(
        Path(config["data_dir"]) / "all_region_stats_enriched.csv",
        config["data_dir"],
    )
    root.mkdir(parents=True, exist_ok=True)
    frame.to_csv(prepared, index=False)
    write_r2_city_attribution_report(payload, report)
    return report


def analyze(config: dict, *, force: bool = False) -> dict[str, str]:
    """Build or restore every retained non-plot paper analysis."""
    view = _view_config(config)
    outputs = {
        "named_scenarios": _run_named_scenarios(view, force=force),
        "distance_threshold": _run_distance_threshold(view, force=force),
        "threshold_resilience": _run_threshold_resilience(view, force=force),
        "spatial_concentration": _run_spatial_concentration(config, force=force),
        "city_attribution": _run_city_attribution(config, force=force),
    }
    payload = {name: str(path) for name, path in outputs.items()}
    payload["analysis_data_dir"] = str(analysis_data_dir(config))
    status = analysis_root(config) / "status.json"
    status.parent.mkdir(parents=True, exist_ok=True)
    status.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload
