"""Orchestrate retained paper analyses without modifying city result files."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path

import pandas as pd

from .paper_cohort import cohort_hash, experiment_excluded_region_keys, locked_region_keys
from .paths import PROJECT_ROOT


ANALYSIS_STATUS_SCHEMA_VERSION = 2
INPUT_MANIFEST_SCHEMA_VERSION = 1
AUTHORITATIVE_AGGREGATES = (
    "all_region_stats.json",
    "all_region_stats.csv",
    "all_region_stats_enriched.csv",
)
ANALYSIS_MODULES = (
    "named-scenarios",
    "pathway-scenarios",
    "distance-threshold",
    "threshold-resilience",
    "city-attribution",
    "contribution-decomposition",
    "beijing-delivery-modes",
    "four-city-delivery-modes",
    "cost-time-substitution",
    "objective-tradeoffs",
)
DERIVED_GLOBAL_NAMES = {
    *AUTHORITATIVE_AGGREGATES,
    "all_sensitivity_named.csv",
    "all_city_distance_threshold_summary.csv",
    "all_city_distance_threshold_summary.partial.csv",
    "all_city_distance_threshold_status.csv",
    "all_city_contribution_decomposition.csv",
    "all_city_contribution_decomposition.partial.csv",
    "all_city_contribution_decomposition_status.csv",
    "all_pathway_scenario_city.csv",
    "all_pathway_curve_points.csv",
    "all_pathway_scenarios_report.json",
    "all_threshold_resilience.csv",
    "all_threshold_resilience_drivers.csv",
}
COHORT_FILTERED_TABLES = {
    name for name in DERIVED_GLOBAL_NAMES if name.endswith(".csv")
}


def analysis_root(config: dict) -> Path:
    return Path(config["work_dir"]) / "analysis"


def analysis_data_dir(config: dict) -> Path:
    """Return the cohort-versioned data view used by analyses and plots."""
    return analysis_root(config) / f"cohort-{cohort_hash()[:12]}" / "data"


def analysis_input_manifest_path(config: dict) -> Path:
    return analysis_data_dir(config).parent / "input_manifest.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(payload: dict) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _atomic_json(payload: dict, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(target)


def _atomic_copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    os.close(descriptor)
    try:
        shutil.copy2(source, temporary)
        os.replace(temporary, target)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _region_input_paths(config: dict) -> list[tuple[str, Path, Path]]:
    registry = json.loads((PROJECT_ROOT / "regions.json").read_text(encoding="utf-8"))
    data_root = Path(config["data_dir"]).resolve()
    paths = []
    for region_key in locked_region_keys():
        region = registry[region_key]
        directory = str(region.get("region_name_en", region_key)).lower()
        city_root = data_root / "city" / directory
        paths.append(
            (
                region_key,
                city_root / "region_stats.json",
                city_root / f"{directory}_od_pairs.csv",
            )
        )
    return paths


def build_input_manifest(config: dict) -> dict:
    """Fingerprint authoritative analysis inputs without writing any state."""
    data_root = Path(config["data_dir"]).resolve()
    for name in AUTHORITATIVE_AGGREGATES:
        if not name.endswith(".csv"):
            continue
        keys = pd.read_csv(data_root / name, usecols=["region_key"])["region_key"]
        forbidden = set(keys.astype(str)) & experiment_excluded_region_keys()
        if forbidden:
            raise ValueError(
                f"{name} still contains excluded cities {sorted(forbidden)}; "
                "refresh aggregate or use an audited cohort migration first"
            )
    hashed: list[dict] = []
    metadata: list[dict] = []
    hash_inputs = [
        (f"data:{name}", data_root / name) for name in AUTHORITATIVE_AGGREGATES
    ]
    for region_key, stats_path, od_path in _region_input_paths(config):
        hash_inputs.append((f"city:{region_key}/region_stats.json", stats_path))
        if not od_path.is_file():
            raise FileNotFoundError(f"Missing analysis OD input: {od_path}")
        stat = od_path.stat()
        metadata.append(
            {
                "path": f"city:{region_key}/{od_path.name}",
                "size_bytes": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
            }
        )
    hash_inputs.extend(
        (
            (
                "project:paper_cohort.json",
                PROJECT_ROOT / "src" / "drone_emission" / "paper_cohort.json",
            ),
            ("project:experiment.toml", PROJECT_ROOT / "experiment.toml"),
            ("data:gef", Path(config["gef_path"]).resolve()),
        )
    )
    sales_path = Path(config["sales_path"]).resolve()
    if not sales_path.is_file():
        raise FileNotFoundError(f"Missing analysis sales input: {sales_path}")
    sales_stat = sales_path.stat()
    metadata.append(
        {
            "path": "data:combined_data.csv",
            "size_bytes": sales_stat.st_size,
            "mtime_ns": sales_stat.st_mtime_ns,
        }
    )
    for logical_path, path in hash_inputs:
        if not path.is_file():
            raise FileNotFoundError(f"Missing analysis input: {path}")
        stat = path.stat()
        hashed.append(
            {
                "path": logical_path,
                "size_bytes": stat.st_size,
                "sha256": _sha256(path),
            }
        )
    body = {
        "schema_version": INPUT_MANIFEST_SCHEMA_VERSION,
        "cohort_sha256": cohort_hash(),
        "city_count": len(locked_region_keys()),
        "sha256_inputs": sorted(hashed, key=lambda item: item["path"]),
        "stat_inputs": sorted(metadata, key=lambda item: item["path"]),
    }
    return {**body, "manifest_sha256": _canonical_sha256(body)}


def _is_analysis_product(relative: Path) -> bool:
    return (
        relative.name in DERIVED_GLOBAL_NAMES
        or relative.name.endswith("_named.csv")
        or "analysis" in relative.parts
    )


def _stage_symlink(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        return
    target.symlink_to(source.resolve())


def _write_view_metadata(target_root: Path, manifest: dict) -> None:
    metadata = {
        "schema_version": 2,
        "cohort_sha256": cohort_hash(),
        "city_count": len(locked_region_keys()),
        "region_keys": sorted(locked_region_keys()),
        "input_manifest_sha256": manifest["manifest_sha256"],
        "input_manifest": str(target_root.parent / "input_manifest.json"),
    }
    _atomic_json(metadata, target_root / "paper_cohort_view.json")


def prepare_analysis_data_view(config: dict, manifest: dict | None = None) -> Path:
    """Refresh inputs; never import analysis products from the official root."""
    source_root = Path(config["data_dir"]).resolve()
    target_root = analysis_data_dir(config)
    target_root.mkdir(parents=True, exist_ok=True)
    if source_root == target_root.resolve():
        raise ValueError("analysis work root must be independent of data root")
    manifest = manifest or build_input_manifest(config)
    for name in AUTHORITATIVE_AGGREGATES:
        _atomic_copy(source_root / name, target_root / name)
    for source in source_root.iterdir():
        if not source.is_file() or source.name in AUTHORITATIVE_AGGREGATES:
            continue
        relative = source.relative_to(source_root)
        if not _is_analysis_product(relative):
            _stage_symlink(source, target_root / relative)
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
                if not _is_analysis_product(relative):
                    _stage_symlink(source, target_root / relative)
    for directory in sorted(path for path in source_root.iterdir() if path.is_dir()):
        if directory.name in {"city", "district"}:
            continue
        for source in directory.rglob("*"):
            if not source.is_file():
                continue
            relative = source.relative_to(source_root)
            if not _is_analysis_product(relative):
                _stage_symlink(source, target_root / relative)
    _atomic_json(manifest, analysis_input_manifest_path(config))
    _write_view_metadata(target_root, manifest)
    return target_root


def _atomic_frame(frame: pd.DataFrame, target: Path) -> None:
    temporary = target.with_suffix(target.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(target)


def _filter_analysis_view(target_root: Path, study_keys: set[str]) -> None:
    """Filter derived nationwide outputs to the locked paper cohort."""
    for name in sorted(COHORT_FILTERED_TABLES - set(AUTHORITATIVE_AGGREGATES)):
        path = target_root / name
        if not path.is_file():
            continue
        try:
            frame = pd.read_csv(path)
        except (OSError, pd.errors.ParserError):
            continue
        if "region_key" not in frame.columns:
            continue
        cohort_rows = frame[frame["region_key"].astype(str).isin(study_keys)]
        if len(cohort_rows) != len(frame):
            _atomic_frame(cohort_rows.copy(), path)
    partial = target_root / "all_city_distance_threshold_summary.partial.csv"
    complete = target_root / "all_city_distance_threshold_summary.csv"
    if partial.is_file() and not complete.is_file():
        frame = pd.read_csv(partial)
        keys = set(frame["region_key"].astype(str))
        bases = set(frame.get("distance_basis", pd.Series(dtype=str)).astype(str))
        if (
            keys == study_keys
            and bases == {"road", "straight"}
            and len(frame) == 2 * len(study_keys)
        ):
            _atomic_frame(frame, complete)


def _view_config(config: dict, manifest: dict | None = None) -> dict:
    view = dict(config)
    view["data_dir"] = str(prepare_analysis_data_view(config, manifest))
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
    registry = init_registry(str(PROJECT_ROOT / "regions.json"), data_dir=config["data_dir"])
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
    named = pd.read_csv(output)
    if "region_key" not in named.columns:
        raise ValueError("named-scenario output has no region_key column")
    named = named[
        named["region_key"].astype(str).isin(set(config["study_region_keys"]))
    ].copy()
    _atomic_frame(named, output)
    return output


def _run_distance_threshold(config: dict, *, force: bool) -> Path:
    from .tools.run_distance_threshold_analysis import _city_regions, analyze_region, consolidate
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


def _run_contribution_decomposition(config: dict, *, force: bool, chunk_rows: int) -> Path:
    from .analysis.contribution_decomposition import run_contribution_analysis
    output, _ = run_contribution_analysis(config, force=force, chunk_rows=chunk_rows)
    return output


def _run_pathway_scenarios(config: dict, *, force: bool) -> Path:
    from .analysis.pathway_scenarios import run_pathway_analysis
    _, _, report = run_pathway_analysis(config, force=force)
    return report


def _run_threshold_resilience(config: dict, *, force: bool) -> Path:
    from .tools.run_threshold_resilience import (
        _city_regions, analyze_region, build_execution_context, consolidate, inventory_regions,
    )
    output = Path(config["data_dir"]) / "all_threshold_resilience.csv"
    regions = _city_regions(config)
    if output.exists() and not force:
        inventory = inventory_regions(config, require_samples=False, force=False)
        if len(inventory) == len(regions) and (inventory["status"] == "cached").all():
            summary = pd.read_csv(output, usecols=["region_key"])
            if summary["region_key"].nunique() == len(regions):
                return output
    failures = []
    execution_context = build_execution_context(config)
    for region_key, region in regions.items():
        result = analyze_region(
            region_key, region, config, force=force, execution_context=execution_context
        )
        if result["status"] not in {"ok", "cached"}:
            failures.append(f"{region_key}:{result['status']}")
    if failures:
        raise RuntimeError("threshold-resilience analysis incomplete: " + ", ".join(failures))
    output, _ = consolidate(config)
    if pd.read_csv(output, usecols=["region_key"])["region_key"].nunique() != len(regions):
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
        region_keys=set(config["study_region_keys"]),
    )
    root.mkdir(parents=True, exist_ok=True)
    frame.to_csv(prepared, index=False)
    write_r2_city_attribution_report(payload, report)
    return report


def _run_beijing_delivery_modes(config: dict, *, force: bool) -> Path:
    from .analysis.beijing_delivery_modes import run_beijing_delivery_mode_analysis

    output, _ = run_beijing_delivery_mode_analysis(config, force=force)
    return output


def _run_four_city_delivery_modes(config: dict, *, force: bool) -> Path:
    from .analysis.four_city_delivery_modes import (
        run_four_city_delivery_mode_analysis,
    )

    output, _ = run_four_city_delivery_mode_analysis(config, force=force)
    return output


def _run_cost_time_substitution(config: dict, *, force: bool) -> Path:
    from .analysis.cost_time_substitution import run_cost_time_analysis

    return run_cost_time_analysis(config, force=force)[0]


def _run_objective_tradeoffs(config: dict, *, force: bool) -> Path:
    from .analysis.objective_tradeoffs import run_objective_tradeoffs

    return run_objective_tradeoffs(config, force=force)[0]


def _module_outputs(config: dict) -> dict[str, tuple[Path, ...]]:
    data = analysis_data_dir(config)
    work = analysis_root(config)
    return {
        "objective-tradeoffs": tuple(
            work / "objective_tradeoffs" / name
            for name in ("city_strategies.csv", "cohort_strategies.csv", "report.json", "summary.md")
        ),
        "cost-time-substitution": tuple(
            work / "cost_time_substitution" / name
            for name in ("city_strategies.csv", "cohort_strategies.csv", "report.json")
        ),
        "named-scenarios": (data / "all_sensitivity_named.csv",),
        "pathway-scenarios": (
            data / "all_pathway_scenario_city.csv",
            data / "all_pathway_curve_points.csv",
            data / "all_pathway_scenarios_report.json",
        ),
        "distance-threshold": (
            data / "all_city_distance_threshold_summary.csv",
            data / "all_city_distance_threshold_status.csv",
        ),
        "threshold-resilience": (
            data / "all_threshold_resilience.csv",
            data / "all_threshold_resilience_drivers.csv",
        ),
        "city-attribution": (
            work / "city_attribution" / "r2_city_attribution_data.csv",
            work / "city_attribution" / "r2_city_attribution_analysis_report.json",
        ),
        "contribution-decomposition": (
            data / "all_city_contribution_decomposition.csv",
            data / "all_city_contribution_decomposition_status.csv",
        ),
        "beijing-delivery-modes": (
            work
            / "beijing_delivery_modes"
            / "beijing_all_mode_lifecycle.csv",
            work
            / "beijing_delivery_modes"
            / "beijing_all_mode_lifecycle_report.json",
        ),
        "four-city-delivery-modes": (
            work
            / "four_city_delivery_modes"
            / "four_city_mode_oracle_summary.csv",
            work
            / "four_city_delivery_modes"
            / "four_city_mode_oracle_report.json",
        ),
    }


def _output_signatures(paths: tuple[Path, ...]) -> list[dict]:
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("analysis module omitted output(s):\n- " + "\n- ".join(missing))
    return [
        {"path": str(path.resolve()), "size_bytes": path.stat().st_size, "sha256": _sha256(path)}
        for path in paths
    ]


def _new_status(config: dict, manifest: dict) -> dict:
    return {
        "schema_version": ANALYSIS_STATUS_SCHEMA_VERSION,
        "input_manifest_sha256": manifest["manifest_sha256"],
        "input_manifest": str(analysis_input_manifest_path(config)),
        "analysis_data_dir": str(analysis_data_dir(config)),
        "paper_cohort_sha256": cohort_hash(),
        "paper_city_count": len(locked_region_keys()),
        "modules": {},
        "complete": False,
    }


def _load_status(path: Path) -> dict | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def analyze(
    config: dict,
    *,
    force: bool = False,
    reuse_from: str | Path | None = None,
    only: list[str] | None = None,
    contribution_chunk_rows: int = 250_000,
) -> dict:
    """Build or restore retained analyses with incrementally durable status."""
    runners = {
        "objective-tradeoffs": lambda view: _run_objective_tradeoffs(view, force=force),
        "cost-time-substitution": lambda view: _run_cost_time_substitution(
            view, force=force
        ),
        "named-scenarios": lambda view: _run_named_scenarios(view, force=force),
        "pathway-scenarios": lambda view: _run_pathway_scenarios(view, force=force),
        "distance-threshold": lambda view: _run_distance_threshold(view, force=force),
        "threshold-resilience": lambda view: _run_threshold_resilience(view, force=force),
        "city-attribution": lambda view: _run_city_attribution(view, force=force),
        "contribution-decomposition": lambda view: _run_contribution_decomposition(
            view, force=force, chunk_rows=contribution_chunk_rows
        ),
        "beijing-delivery-modes": lambda view: _run_beijing_delivery_modes(
            view, force=force
        ),
        "four-city-delivery-modes": lambda view: _run_four_city_delivery_modes(
            view, force=force
        ),
    }
    selected = list(runners) if only is None else list(dict.fromkeys(only))
    unknown = sorted(set(selected) - set(runners))
    if unknown:
        raise ValueError(
            f"unknown analysis selection(s): {unknown}; "
            f"choose from {sorted(runners)}"
        )
    if reuse_from is not None and force:
        raise ValueError("--force cannot be combined with --reuse-from")
    manifest = build_input_manifest(config)
    view = _view_config(config, manifest)
    status_path = analysis_root(config) / "status.json"
    previous = _load_status(status_path)
    previous_is_v2 = bool(
        previous and previous.get("schema_version") == ANALYSIS_STATUS_SCHEMA_VERSION
    )
    if previous is not None and not previous_is_v2 and not force:
        raise RuntimeError(
            "analysis status schema is stale; "
            "run `python -m drone_emission analyze --force`"
        )
    fingerprint_changed = previous_is_v2 and (
        previous.get("input_manifest_sha256") != manifest["manifest_sha256"]
    )
    if fingerprint_changed and not force and reuse_from is None:
        raise RuntimeError(
            "analysis inputs changed; old module results are invalid. "
            "Run `python -m drone_emission analyze --force`."
        )
    status = (
        previous
        if previous_is_v2 and not fingerprint_changed
        else _new_status(config, manifest)
    )
    if fingerprint_changed:
        _atomic_json(status, status_path)
    reuse_payload = None
    if reuse_from is not None:
        from .paper_reuse import restore_reuse_directory, validate_reused_region_sets
        reuse_payload = restore_reuse_directory(view, reuse_from)
        validate_reused_region_sets(view)
        reuse_manifest = build_input_manifest(view)
        _atomic_json(reuse_manifest, analysis_input_manifest_path(config))
        _write_view_metadata(analysis_data_dir(config), reuse_manifest)
        status = _new_status(config, reuse_manifest)
        status["reuse"] = {
            "manifest": reuse_payload["manifest"],
            "restored": reuse_payload["restored"],
        }
    module_paths = _module_outputs(config)
    outputs: dict[str, str] = {}
    for name in selected:
        # A forced or retried module must stop being renderable before it runs.
        status["modules"].pop(name, None)
        status["complete"] = False
        _atomic_json(status, status_path)
        # Existing audited reuse bundles predate the transport-cost module.
        # Compute it from the explicitly restored view, not legacy output paths.
        primary = (
            Path(runners[name](view))
            if reuse_from is None or name in {"cost-time-substitution", "objective-tradeoffs"}
            else module_paths[name][0]
        )
        status["modules"][name] = {
            "status": "complete",
            "primary_output": str(primary),
            "outputs": _output_signatures(module_paths[name]),
        }
        status["complete"] = set(status["modules"]) == set(ANALYSIS_MODULES)
        _atomic_json(status, status_path)
        outputs[name.replace("-", "_")] = str(primary)
    status["complete"] = (
        set(status["modules"]) == set(ANALYSIS_MODULES)
        and all(item.get("status") == "complete" for item in status["modules"].values())
    )
    _atomic_json(status, status_path)
    return {**outputs, **status}


def _verify_output_signature(record: dict) -> bool:
    path = Path(record.get("path", ""))
    return (
        path.is_file()
        and path.stat().st_size == record.get("size_bytes")
        and _sha256(path) == record.get("sha256")
    )


def require_current_analysis(
    config: dict,
    *,
    required_modules: set[str] | None = None,
) -> dict:
    """Validate the analysis modules required by a render selection."""
    selected_modules = (
        set(ANALYSIS_MODULES)
        if required_modules is None
        else set(required_modules)
    )
    unknown = selected_modules - set(ANALYSIS_MODULES)
    if unknown:
        raise ValueError(f"unknown analysis modules: {sorted(unknown)}")
    status_path = analysis_root(config) / "status.json"
    if not status_path.is_file():
        raise FileNotFoundError(
            f"Missing analysis status: {status_path}. Run aggregate, cohort, and analyze first."
        )
    payload = json.loads(status_path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != ANALYSIS_STATUS_SCHEMA_VERSION:
        raise RuntimeError(
            "analysis status schema is stale; run `python -m drone_emission analyze --force`"
        )
    is_explicit_reuse = bool(payload.get("reuse"))
    manifest_config = (
        dict(config, data_dir=str(analysis_data_dir(config)))
        if is_explicit_reuse
        else config
    )
    current_manifest = build_input_manifest(manifest_config)
    if payload.get("input_manifest_sha256") != current_manifest["manifest_sha256"]:
        raise RuntimeError(
            "analysis inputs changed; rerun aggregate/cohort and "
            "`python -m drone_emission analyze --force`"
        )
    stored_manifest_path = Path(payload.get("input_manifest", ""))
    if not stored_manifest_path.is_file():
        raise RuntimeError("analysis input manifest is missing; rerun analyze --force")
    stored_manifest = json.loads(stored_manifest_path.read_text(encoding="utf-8"))
    if stored_manifest != current_manifest:
        raise RuntimeError("analysis input manifest is stale or changed; rerun analyze --force")
    view_metadata_path = analysis_data_dir(config) / "paper_cohort_view.json"
    if not view_metadata_path.is_file():
        raise RuntimeError("analysis view metadata is missing; rerun analyze --force")
    view_metadata = json.loads(view_metadata_path.read_text(encoding="utf-8"))
    if (
        view_metadata.get("schema_version") != 2
        or view_metadata.get("input_manifest_sha256") != current_manifest["manifest_sha256"]
        or view_metadata.get("cohort_sha256") != cohort_hash()
    ):
        raise RuntimeError("analysis view metadata is stale; rerun analyze --force")
    if not is_explicit_reuse:
        for name in AUTHORITATIVE_AGGREGATES:
            official = Path(config["data_dir"]) / name
            staged = analysis_data_dir(config) / name
            if not staged.is_file() or _sha256(official) != _sha256(staged):
                raise RuntimeError(
                    f"analysis view aggregate is stale: {name}; rerun analyze --force"
                )
    modules = payload.get("modules", {})
    missing = sorted(selected_modules - set(modules))
    if missing:
        raise RuntimeError(
            "analysis is incomplete"
            + f" (missing: {', '.join(missing)})"
            + "; run analyze --force"
        )
    if required_modules is None and (
        not payload.get("complete") or set(modules) != set(ANALYSIS_MODULES)
    ):
        raise RuntimeError("analysis is incomplete; run analyze --force")
    for name in selected_modules:
        module = modules[name]
        if module.get("status") != "complete" or not module.get("outputs"):
            raise RuntimeError(f"analysis module is incomplete: {name}; rerun analyze --force")
        if not all(_verify_output_signature(record) for record in module["outputs"]):
            raise RuntimeError(
                f"analysis module output changed or is missing: {name}; rerun analyze --force"
            )
    return payload
