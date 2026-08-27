"""Orchestrate retained paper analyses without modifying city result files."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd

from .paper_cohort import cohort_hash, locked_region_keys
from .paths import PROJECT_ROOT


DERIVED_GLOBAL_NAMES = {
    "all_region_stats.json",
    "all_region_stats.csv",
    "all_region_stats_enriched.csv",
    "all_sensitivity_named.csv",
    "all_city_distance_threshold_summary.csv",
    "all_city_distance_threshold_summary.partial.csv",
    "all_city_distance_threshold_status.csv",
    "all_city_contribution_decomposition.csv",
    "all_city_contribution_decomposition.partial.csv",
    "all_city_contribution_decomposition_status.csv",
    "all_threshold_resilience.csv",
    "all_threshold_resilience_drivers.csv",
}

COHORT_FILTERED_TABLES = DERIVED_GLOBAL_NAMES - {"all_region_stats.json"}


def analysis_root(config: dict) -> Path:
    return Path(config["work_dir"]) / "analysis"


def analysis_data_dir(config: dict) -> Path:
    """Return the cohort-versioned data view used by analyses and plots."""
    return analysis_root(config) / f"cohort-{cohort_hash()[:12]}" / "data"


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

    study_keys = set(locked_region_keys())
    for level in ("city", "district"):
        level_root = source_root / level
        if not level_root.is_dir():
            continue
        for region_root in sorted(path for path in level_root.iterdir() if path.is_dir()):
            if level == "city" and region_root.name not in study_keys:
                continue
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
    _filter_analysis_view(target_root, study_keys)
    return target_root


def _atomic_frame(frame: pd.DataFrame, target: Path) -> None:
    temporary = target.with_suffix(target.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(target)


def _filter_analysis_view(target_root: Path, study_keys: set[str]) -> None:
    """Make every nationwide analysis input match the locked paper cohort."""
    for name in sorted(COHORT_FILTERED_TABLES):
        path = target_root / name
        if not path.is_file():
            continue
        try:
            frame = pd.read_csv(path)
        except (OSError, pd.errors.ParserError):
            continue
        if "region_key" not in frame.columns:
            continue
        frame = frame[frame["region_key"].astype(str).isin(study_keys)].copy()
        _atomic_frame(frame, path)

    stats_json = target_root / "all_region_stats.json"
    if stats_json.is_file():
        payload = json.loads(stats_json.read_text(encoding="utf-8"))
        filtered = {key: value for key, value in payload.items() if key in study_keys}
        temporary = stats_json.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(filtered, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(stats_json)

    partial = target_root / "all_city_distance_threshold_summary.partial.csv"
    complete = target_root / "all_city_distance_threshold_summary.csv"
    if partial.is_file() and not complete.is_file():
        frame = pd.read_csv(partial)
        keys = set(frame["region_key"].astype(str))
        bases = set(frame.get("distance_basis", pd.Series(dtype=str)).astype(str))
        if keys == study_keys and bases == {"road", "straight"} and len(frame) == 2 * len(study_keys):
            _atomic_frame(frame, complete)

    metadata = {
        "schema_version": 1,
        "cohort_sha256": cohort_hash(),
        "city_count": len(study_keys),
        "region_keys": sorted(study_keys),
    }
    metadata_path = target_root / "paper_cohort_view.json"
    temporary = metadata_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    temporary.replace(metadata_path)


def _view_config(config: dict) -> dict:
    view = dict(config)
    view["data_dir"] = str(prepare_analysis_data_view(config))
    # Global inputs remain authoritative and read-only to analysis routines.
    view["sales_path"] = config["sales_path"]
    view["gef_path"] = config["gef_path"]
    view["study_region_keys"] = locked_region_keys()
    view["paper_cohort_sha256"] = cohort_hash()
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
    study_keys = set(config.get("study_region_keys", ()))
    if study_keys:
        regions = [(key, region) for key, region in regions if key in study_keys]
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


def _run_contribution_decomposition(
    config: dict,
    *,
    force: bool,
    chunk_rows: int,
) -> Path:
    from .analysis.contribution_decomposition import run_contribution_analysis

    output, _ = run_contribution_analysis(
        config,
        force=force,
        chunk_rows=chunk_rows,
    )
    return output


def _run_threshold_resilience(config: dict, *, force: bool) -> Path:
    from .tools.run_threshold_resilience import (
        _city_regions,
        analyze_region,
        build_execution_context,
        consolidate,
        inventory_regions,
    )

    output = Path(config["data_dir"]) / "all_threshold_resilience.csv"
    regions = _city_regions(config)
    if output.exists() and not force:
        inventory = inventory_regions(
            config, require_samples=False, force=False
        )
        all_city_caches_are_fresh = (
            len(inventory) == len(regions)
            and (inventory["status"] == "cached").all()
        )
        if all_city_caches_are_fresh:
            summary = pd.read_csv(output, usecols=["region_key"])
            if summary["region_key"].nunique() == len(regions):
                return output
    failures = []
    execution_context = build_execution_context(config)
    for region_key, region in regions.items():
        result = analyze_region(
            region_key,
            region,
            config,
            force=force,
            execution_context=execution_context,
        )
        if result["status"] not in {"ok", "cached"}:
            failures.append(f"{region_key}:{result['status']}")
    if failures:
        raise RuntimeError("threshold-resilience analysis incomplete: " + ", ".join(failures))
    output, _ = consolidate(config)
    summary = pd.read_csv(output, usecols=["region_key"])
    if summary["region_key"].nunique() != len(regions):
        raise RuntimeError("threshold-resilience consolidation omitted cities")
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


def analyze(
    config: dict,
    *,
    force: bool = False,
    reuse_from: str | Path | None = None,
    only: list[str] | None = None,
    contribution_chunk_rows: int = 250_000,
) -> dict[str, str]:
    """Build or restore every retained non-plot paper analysis."""
    runners = {
        "named-scenarios": lambda view: _run_named_scenarios(view, force=force),
        "distance-threshold": lambda view: _run_distance_threshold(view, force=force),
        "threshold-resilience": lambda view: _run_threshold_resilience(view, force=force),
        "city-attribution": lambda view: _run_city_attribution(config, force=force),
        "contribution-decomposition": lambda view: _run_contribution_decomposition(
            view,
            force=force,
            chunk_rows=contribution_chunk_rows,
        ),
    }
    selected = list(runners) if only is None else list(dict.fromkeys(only))
    unknown = sorted(set(selected) - set(runners))
    if unknown:
        raise ValueError(
            f"unknown analysis selection(s): {unknown}; "
            f"choose from {sorted(runners)}"
        )
    view = _view_config(config)
    reuse_payload = None
    if reuse_from is not None:
        if force:
            raise ValueError("--force cannot be combined with --reuse-from")
        from .paper_reuse import (
            restore_reuse_directory,
            validate_reused_region_sets,
        )

        reuse_payload = restore_reuse_directory(view, reuse_from)
        validate_reused_region_sets(view)
        outputs = {
            "named_scenarios": Path(view["data_dir"]) / "all_sensitivity_named.csv",
            "distance_threshold": Path(view["data_dir"])
            / "all_city_distance_threshold_summary.csv",
            "threshold_resilience": Path(view["data_dir"])
            / "all_threshold_resilience.csv",
            "city_attribution": analysis_root(config)
            / "city_attribution"
            / "r2_city_attribution_analysis_report.json",
            "contribution_decomposition": Path(view["data_dir"])
            / "all_city_contribution_decomposition.csv",
        }
        selected_outputs = {name.replace("-", "_") for name in selected}
        outputs = {
            name: path for name, path in outputs.items() if name in selected_outputs
        }
        missing = [str(path) for path in outputs.values() if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                "audited reuse directory did not restore required analysis files:\n- "
                + "\n- ".join(missing)
            )
    else:
        outputs = {name.replace("-", "_"): runners[name](view) for name in selected}
    payload = {name: str(path) for name, path in outputs.items()}
    payload["analysis_data_dir"] = str(analysis_data_dir(config))
    payload["paper_cohort_sha256"] = cohort_hash()
    payload["paper_city_count"] = len(locked_region_keys())
    if reuse_payload is not None:
        payload["reuse_manifest"] = reuse_payload["manifest"]
    status = analysis_root(config) / "status.json"
    status.parent.mkdir(parents=True, exist_ok=True)
    status.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def require_current_analysis(config: dict) -> dict:
    """Require an analyze result produced for the tracked paper cohort."""
    status = analysis_root(config) / "status.json"
    if not status.is_file():
        raise FileNotFoundError(
            f"Missing analysis status: {status}. Run `python -m drone_emission analyze` first."
        )
    payload = json.loads(status.read_text(encoding="utf-8"))
    if payload.get("paper_cohort_sha256") != cohort_hash():
        raise RuntimeError(
            "analysis status belongs to a different paper cohort; rerun analyze"
        )
    return payload
