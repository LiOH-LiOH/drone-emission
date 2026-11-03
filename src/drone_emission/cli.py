"""Single public command surface for end-to-end paper reproduction."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from .paper_workflow import (
    compose,
    figure_required_outputs,
    load_manifest,
    promote,
    render,
)
from .paths import PROJECT_ROOT


def _config(env: str | None) -> dict:
    from drone_emission.utils import load_config

    return load_config(str(PROJECT_ROOT / "experiment.toml"), env=env)


def _split_ids(values: list[str] | None) -> list[str] | None:
    if not values:
        return None
    return [item for value in values for item in value.split(",") if item]


def _input_problems(config: dict) -> list[str]:
    required = [
        PROJECT_ROOT / "experiment.toml",
        PROJECT_ROOT / "regions.json",
        Path(config["sales_path"]),
        Path(config["gef_path"]),
    ]
    return [str(path) for path in required if not path.exists()]


def _require_clean_source_commit() -> None:
    result = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    if result.stdout.strip():
        raise RuntimeError("audited batch requires a clean committed worktree")


def command_inputs(args: argparse.Namespace) -> None:
    config = _config(args.env)
    problems = _input_problems(config)
    payload = {
        "ok": not problems,
        "data_root": config["data_dir"],
        "figures_root": config["figures_dir"],
        "cache_root": config["cache_dir"],
        "work_root": config["work_dir"],
        "missing": problems,
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    if problems:
        raise SystemExit(2)


def command_calibrate(args: argparse.Namespace) -> None:
    from drone_emission.calibration.gravity_calibration import (
        accept_calibration_result,
        run_joint_calibration,
    )

    result = run_joint_calibration(
        corrected_orders_path=args.orders_corrected,
        buildings_path=args.buildings,
        transform_path=args.transform,
        output_dir=args.output,
        memory_limit_gb=args.memory_limit_gb,
        workers=args.workers,
        rebuild_prepared=args.rebuild_prepared,
        no_basemap=args.no_basemap,
    )
    payload = {
        "output_dir": result["output_dir"],
        "result": str(Path(result["output_dir"]) / "gravity_calibration.json"),
        "run_hash": result["run_hash"],
        "success": result["final_fit"]["success"],
        "accepted": False,
    }
    if args.accept_result:
        summary = accept_calibration_result(
            Path(result["output_dir"]) / "gravity_calibration.json",
            PROJECT_ROOT / "experiment.toml",
            PROJECT_ROOT
            / "src"
            / "drone_emission"
            / "calibration"
            / "gravity_calibration_summary.json",
        )
        payload["accepted"] = True
        payload["accepted_summary"] = summary
    print(json.dumps(payload, indent=2, ensure_ascii=False))


def command_calibrate_external(args: argparse.Namespace) -> None:
    from drone_emission.calibration.external_validation import run_external_validation

    result = run_external_validation(
        args.bundle, args.buildings, args.parameters, args.output, args.memory_limit_gb,
        by_date=args.by_date,
    )
    print(json.dumps({
        "result": str(Path(result["output_dir"]) / "external_validation.json"),
        "status": result["status"], "run_hash": result["run_hash"],
        "retained_orders": result["prepared"]["exclusion_flow"]["retained_valid_orders"],
    }, indent=2, ensure_ascii=False))


def command_calibrate_external_buildings(args: argparse.Namespace) -> None:
    from drone_emission.calibration.external_validation import prepare_external_buildings

    print(json.dumps(prepare_external_buildings(args.bundle, args.gba_tiles, args.output), indent=2))


def command_calibrate_spatial_fit(args: argparse.Namespace) -> None:
    from drone_emission.calibration.gravity_calibration import (
        refresh_spatial_fit_diagnostics,
    )

    result = refresh_spatial_fit_diagnostics(
        args.result, no_basemap=args.no_basemap
    )
    print(
        json.dumps(
            {
                "result": str(args.result.resolve()),
                "run_hash": result["run_hash"],
                "spatial_fit": result["spatial_fit"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )


def command_calibrate_coordinates(args: argparse.Namespace) -> None:
    """Restore and audit the translated Beijing order coordinates."""
    config = _config(args.env)
    orders = args.orders or Path(
        os.environ.get(
            "DRONE_EMISSION_BEIJING_OD_ROOT",
            Path(config["data_dir"])
            / "raw"
            / "即时配送数据"
            / "疫情期间北京饿了么"
            / "eleme_round1_train_20200313"
            / "order",
        )
    )
    poi = args.poi or (
        Path(config["data_dir"])
        / "raw"
        / "即时配送数据"
        / "北京市poi202209"
        / "北京市_餐饮服务_202209.shp"
    )
    output = args.output or (
        Path(config["work_dir"]) / "calibration" / "coordinate_restore"
    )
    from drone_emission.calibration.coordinate_restore import (
        apply_coordinate_transform,
        run_coordinate_restore,
    )

    if args.apply_transform:
        if args.transform is None:
            raise ValueError("--transform is required with --apply-transform")
        output_path = output if output.suffix else output / "corrected_orders.csv.gz"
        result = apply_coordinate_transform(orders, args.transform, output_path)
        print(json.dumps({"output": str(output_path), **result}, indent=2, ensure_ascii=False))
        return

    result = run_coordinate_restore(
        orders_path=orders,
        poi_path=poi,
        output_dir=output,
        poi_datum=args.poi_datum,
        match_radius_m=args.match_radius_m,
        write_orders=args.write_orders,
        no_basemap=args.no_basemap,
        heatmap_gridsize=args.heatmap_gridsize,
        cache_dir=Path(config["cache_dir"]) / "coordinate_restore",
    )
    print(json.dumps({"output": str(output), **result}, indent=2, ensure_ascii=False))


def command_cities(args: argparse.Namespace) -> None:
    from .paper_cohort import require_experiment_region

    for region_key in args.region or ():
        require_experiment_region(region_key)
    config = _config(args.env)
    jobs = getattr(args, "jobs", None)
    cohort = getattr(args, "cohort", None)
    run_dir = getattr(args, "run_dir", None)
    resume = bool(getattr(args, "resume", False))
    start_at = getattr(args, "start_at", None)
    source_data_root = getattr(args, "source_data_root", None)
    staging_data_root = getattr(args, "staging_data_root", None)
    repair_keys = getattr(args, "repair", None)
    batch_requested = any(
        (
            jobs is not None,
            cohort is not None,
            run_dir is not None,
            resume,
            source_data_root is not None,
            staging_data_root is not None,
            bool(repair_keys),
        )
    )
    if args.region and batch_requested:
        raise ValueError("--region cannot be combined with audited batch options")
    if start_at == "road-clean" and not args.region and not batch_requested:
        raise ValueError("--start-at road-clean requires --region or audited batch options")
    if start_at and not args.region and not repair_keys and start_at != "road-clean":
        raise ValueError("--start-at requires at least one --region")
    if start_at and batch_requested and not repair_keys and start_at != "road-clean":
        raise ValueError("--start-at cannot be combined with audited batch options")
    if batch_requested:
        if jobs is None or cohort is None or run_dir is None:
            raise ValueError(
                "audited batch mode requires --jobs, --cohort, and --run-dir"
            )
        from drone_emission.city_batch import run_city_batch
        from drone_emission.city_staging import (
            prepare_isolated_run,
            prepare_road_clean_run,
        )
        from drone_emission.paper_cohort import cohort_hash, locked_region_keys
        from drone_emission.utils.region_config import RegionRegistry

        if not args.no_fetch:
            raise ValueError("audited isolated batch requires --no-fetch")
        if repair_keys and not resume:
            raise ValueError("--repair requires --resume and the original --run-dir")
        manifest_path = run_dir / "batch_manifest.json"
        manifest = (
            json.loads(manifest_path.read_text(encoding="utf-8"))
            if resume and manifest_path.is_file()
            else None
        )
        failed_record = run_dir / "failed_cities.txt"
        recorded_failures = (
            set(failed_record.read_text(encoding="utf-8").split())
            if failed_record.is_file()
            else set()
        )
        if resume and recorded_failures and not repair_keys:
            raise ValueError(
                "failed cities require explicit --repair entries; plain --resume "
                "only continues interrupted work"
            )
        if repair_keys and not set(repair_keys) <= recorded_failures:
            raise ValueError("--repair may select only failures recorded by this batch")
        recorded_env = (manifest or {}).get("environment_overrides", {})
        source_root = Path(
            source_data_root
            or recorded_env.get("DRONE_EMISSION_SOURCE_DATA_ROOT", "")
        )
        staging_root = Path(
            staging_data_root
            or recorded_env.get("DRONE_EMISSION_DATA_ROOT", "")
        )
        if not str(source_root) or str(source_root) == "." or not str(staging_root) or str(staging_root) == ".":
            raise ValueError(
                "audited batch requires --source-data-root and --staging-data-root"
            )
        registry = RegionRegistry(str(PROJECT_ROOT / "regions.json"))
        locked_keys = locked_region_keys()
        region_keys = locked_keys
        if manifest is None:
            _require_clean_source_commit()
            if start_at == "road-clean":
                preflight = prepare_road_clean_run(
                    source_root=source_root,
                    staging_root=staging_root,
                    regions=dict(registry.items()),
                    region_keys=locked_keys,
                )
                region_keys = preflight["affected_region_keys"]
            else:
                preflight = prepare_isolated_run(
                    source_root=source_root,
                    staging_root=staging_root,
                    regions=dict(registry.items()),
                    region_keys=region_keys,
                    exp_config=config,
                )
            resource_plan = preflight["resource_plan"]
        else:
            resource_plan = manifest["resource_plan"]
            region_keys = manifest["region_keys"]
        if not region_keys:
            print("Road-distance scan found no affected cities")
            return
        environment_overrides = {
            "DRONE_EMISSION_SOURCE_DATA_ROOT": str(source_root.resolve()),
            "DRONE_EMISSION_DATA_ROOT": str(staging_root.resolve()),
            "DRONE_EMISSION_SALES_PATH": str((staging_root / "combined_data.csv").resolve()),
            "DRONE_EMISSION_GEF_PATH": str((staging_root / "gef_china.csv").resolve()),
        }
        command_factory = None
        if start_at:
            def command_factory(region_key: str) -> list[str]:
                return [
                    os.environ.get("DRONE_EMISSION_PYTHON", sys.executable),
                    "-m", "drone_emission", "--env", args.env or "local",
                    "cities", "--region", region_key, "--start-at", start_at,
                    "--no-fetch",
                ]
        from drone_emission.city_staging import (
            validate_road_clean_city,
            validate_staged_city,
        )

        def result_validator(region_key: str) -> None:
            validator = (
                validate_road_clean_city
                if start_at == "road-clean"
                else validate_staged_city
            )
            validator(staging_root, registry[region_key])

        exit_code = run_city_batch(
            region_keys=region_keys,
            environment=args.env or "local",
            jobs=jobs,
            run_dir=run_dir,
            force=args.force,
            no_fetch=args.no_fetch,
            cohort_hash=cohort_hash(),
            resume=resume,
            resource_plan=resource_plan,
            canary_keys=[
                key for key in ("anqing", "beijing", "chongqing")
                if key in region_keys
            ],
            memory_budget_gb=args.memory_budget_gb,
            environment_overrides=environment_overrides,
            repair_keys=repair_keys,
            command_factory=command_factory,
            result_validator=result_validator,
            batch_mode=start_at or "full",
        )
        if exit_code:
            raise SystemExit(exit_code)
        from drone_emission.city_staging import (
            audit_road_clean_results,
            audit_staged_results,
        )

        auditor = (
            audit_road_clean_results
            if start_at == "road-clean"
            else audit_staged_results
        )
        auditor(
            source_root=source_root,
            staging_root=staging_root,
            regions=dict(registry.items()),
            region_keys=region_keys,
            status_path=run_dir / "city_status.tsv",
        )
        return

    problems = _input_problems(config)
    if problems:
        raise FileNotFoundError("Missing city input(s): " + ", ".join(problems))

    from drone_emission.main import (
        has_city_results,
        run_all,
        run_region_analysis,
        run_region_from_road_clean,
        run_region_from_roadnet,
    )
    from drone_emission.utils.config import configure_cache
    from drone_emission.utils.region_config import init_registry

    if args.no_fetch:
        config["fetch_buildings"] = False
        config["fetch_roadnet"] = False
        config["strict_no_fetch"] = True
    configure_cache(config)
    registry = init_registry(
        str(PROJECT_ROOT / "regions.json"), data_dir=config["data_dir"]
    )
    if args.region:
        for region_key in args.region:
            if start_at == "roadnet":
                run_region_from_roadnet(region_key, config)
                registry.save_region(region_key)
                continue
            if start_at == "road-clean":
                run_region_from_road_clean(region_key, config)
                registry.save_region(region_key)
                continue
            if (
                not args.force
                and region_key in registry
                and has_city_results(registry[region_key], config)
            ):
                print(f"Skipping {region_key} (restored completed result)")
                continue
            run_region_analysis(region_key, config)
            registry.save_region(region_key)
    else:
        run_all(config, level="city", skip_completed=not args.force)


def command_aggregate(args: argparse.Namespace) -> None:
    config = _config(args.env)
    data_dir = Path(config["data_dir"])
    outputs = [
        data_dir / "all_region_stats.json",
        data_dir / "all_region_stats.csv",
        data_dir / "all_region_stats_enriched.csv",
    ]
    if not args.force and all(path.exists() for path in outputs):
        import csv
        from .paper_cohort import experiment_excluded_region_keys

        excluded = experiment_excluded_region_keys()
        with outputs[1].open(newline="") as handle:
            old_keys = {row["region_key"] for row in csv.DictReader(handle)}
        if not old_keys.intersection(excluded):
            print("Restored existing nationwide aggregates")
            return
    from drone_emission.tools.consolidate_stats import consolidate
    from drone_emission.tools.enrich_stats import enrich

    consolidate(config["data_dir"])
    enrich(config["data_dir"], config["sales_path"], config["gef_path"])


def command_cohort(args: argparse.Namespace) -> None:
    """Audit post-aggregate city eligibility against the tracked paper lock."""
    config = _config(args.env)
    from .paper_cohort import (
        accept_candidate_cohort,
        audit_city_cohort,
        write_cohort_audit,
    )

    write_contract = getattr(args, "write_contract", None)
    shrink_only = bool(getattr(args, "shrink_only", False))
    if write_contract is not None and not shrink_only:
        raise ValueError("--write-contract requires --shrink-only")
    payload = audit_city_cohort(
        config, restrict_to_locked=shrink_only
    )
    path = write_cohort_audit(config, payload)
    summary = {
        "audit": str(path),
        "candidate_city_count": len(payload["candidate_region_keys"]),
        "candidate_sha256": payload["candidate_region_keys_sha256"],
        "matches_locked": payload["matches_locked"],
        "added_since_lock": payload["added_since_lock"],
        "missing_since_lock": payload["missing_since_lock"],
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if args.accept:
        accepted = accept_candidate_cohort(payload)
        print(f"Accepted reviewed cohort into {accepted}")
        return
    if write_contract is not None:
        accepted = accept_candidate_cohort(payload, target=write_contract)
        print(f"Wrote candidate cohort contract to {accepted}")
        return
    if not payload["matches_locked"]:
        raise SystemExit(2)


def command_cost_pilot(args: argparse.Namespace) -> None:
    """Run an explicitly selected city's full-cost capacity pilot only."""
    from .analysis.full_cost_pilot import run_full_cost_pilot

    if getattr(args, "objectives", False):
        from .analysis.objective_tradeoffs import run_objective_tradeoffs

        paths = run_objective_tradeoffs(_config(args.env), region_key=args.region)
    else:
        paths = run_full_cost_pilot(_config(args.env), args.region)
    print(json.dumps({"region": args.region, "outputs": [str(p) for p in paths]}, indent=2))


def command_analyze(args: argparse.Namespace) -> None:
    """Build or restore all retained non-plot paper analyses."""
    config = _config(args.env)
    required = [
        Path(config["data_dir"]) / "all_region_stats.csv",
        Path(config["data_dir"]) / "all_region_stats_enriched.csv",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing aggregate input(s): " + ", ".join(missing))
    from .paper_cohort import assert_locked_cohort
    assert_locked_cohort(config)
    from .analysis_workflow import analyze

    print(
        json.dumps(
            analyze(
                config,
                force=args.force,
                reuse_from=args.reuse_from,
                only=_split_ids(getattr(args, "analysis_only", None)),
                contribution_chunk_rows=getattr(
                    args, "contribution_chunk_rows", 250_000
                ),
            ),
            indent=2,
        )
    )


def command_prepare_reuse(args: argparse.Namespace) -> None:
    """Build the one audited reuse directory without changing official data."""
    config = _config(args.env)
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"paper reuse output is not empty: {output}")
    from .paper_cohort import assert_locked_cohort
    from .analysis_workflow import _view_config
    from .paper_reuse import (
        capture_reuse_directory,
        migrate_fallback_gef_overlay,
        write_migration_difference_report,
    )

    assert_locked_cohort(config)
    view = _view_config(config)
    migration = migrate_fallback_gef_overlay(view)
    manifest = capture_reuse_directory(view, output)
    difference = write_migration_difference_report(
        Path(config["data_dir"]) / "all_region_stats.csv",
        Path(view["data_dir"]) / "all_region_stats.csv",
        output / "migration_difference.json",
    )
    print(
        json.dumps(
            {
                "manifest": str(manifest),
                "difference_report": str(difference),
                "migration": migration,
            },
            indent=2,
        )
    )


def command_render(args: argparse.Namespace) -> None:
    staged = render(
        _config(args.env),
        only=_split_ids(getattr(args, "only", None)),
        requests=_split_ids(getattr(args, "request", None)),
        force=bool(getattr(args, "force", False)),
        prepare_only=bool(getattr(args, "prepare_only", False)),
        bundle_dir=getattr(args, "bundle_dir", None),
        from_bundle=getattr(args, "from_bundle", None),
        palette=getattr(args, "palette", None),
    )
    if getattr(args, "prepare_only", False):
        print("Prepared plot inputs are ready for render --from-bundle.")
        return
    if staged:
        print("Staged direct figures:")
        for path in staged:
            print(path)
    print(
        "Run `python -m drone_emission --env local compose` "
        "to refresh Inkscape figures."
    )


def command_compose(args: argparse.Namespace) -> None:
    if getattr(args, "preview_only", False):
        if args.prepare_links or args.force:
            raise ValueError("--preview-only cannot prepare links or force composition")
        from .paper_preview import refresh_previews
        from .paper_workflow import _tool_path, select_figures

        refresh_previews(
            _config(args.env)["figures_dir"],
            select_figures(_split_ids(getattr(args, "only", None))),
            rasterizer=_tool_path("pdftoppm"),
        )
        return
    paths = compose(
        _config(args.env),
        only=_split_ids(getattr(args, "only", None)),
        prepare_links=bool(getattr(args, "prepare_links", False)),
        force=bool(getattr(args, "force", False)),
    )
    if args.prepare_links:
        for path in paths:
            print(f"Linked subfigure ready: {path}")
        print(f"Prepared {len(paths)} linked subfigure(s).")
    elif paths:
        print(f"Composed {len(paths)} formal figure(s).")
    else:
        print("All selected compositions are fresh.")


def command_promote(args: argparse.Namespace) -> None:
    operations = promote(_config(args.env), yes=args.yes, dry_run=args.dry_run)
    prefix = "Would promote" if args.dry_run else "Promoted"
    for source, target in operations:
        print(f"{prefix}: {source} -> {target}")


def command_city_promote(args: argparse.Namespace) -> None:
    from drone_emission.city_staging import promote_city_results
    from drone_emission.paper_cohort import locked_region_keys
    from drone_emission.utils.region_config import RegionRegistry

    registry = RegionRegistry(str(PROJECT_ROOT / "regions.json"))
    region_keys = locked_region_keys()
    extra = {}
    if args.run_dir is not None:
        manifest_path = args.run_dir / "batch_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("batch_mode") != "road-clean":
            raise ValueError("--run-dir partial promotion requires a road-clean batch")
        recorded = manifest.get("environment_overrides", {})
        expected_source = Path(
            recorded.get("DRONE_EMISSION_SOURCE_DATA_ROOT", "")
        ).resolve()
        expected_staging = Path(
            recorded.get("DRONE_EMISSION_DATA_ROOT", "")
        ).resolve()
        if args.source_data_root.resolve() != expected_source:
            raise ValueError("source data root does not match the road-clean batch")
        if args.staging_data_root.resolve() != expected_staging:
            raise ValueError("staging data root does not match the road-clean batch")
        region_keys = manifest["region_keys"]
        preflight = json.loads(
            (args.staging_data_root / "preflight.json").read_text(encoding="utf-8")
        )
        scans = {row["region_key"]: row for row in preflight["cities"]}
        for key in region_keys:
            region = registry[key]
            name = region.region_name_en.lower()
            source_od = (
                args.source_data_root / region.region_level / name / f"{name}_od_pairs.csv"
            )
            source_stat = source_od.stat()
            scan = scans[key]
            if (
                source_stat.st_size != scan["source_size_bytes"]
                or source_stat.st_mtime_ns != scan["source_mtime_ns"]
            ):
                raise RuntimeError(f"source OD changed after road-clean preflight: {key}")
        extra["promoted_suffixes"] = ("_od_pairs.csv",)
    operations = promote_city_results(
        source_root=args.source_data_root,
        staging_root=args.staging_data_root,
        regions=dict(registry.items()),
        region_keys=region_keys,
        yes=args.yes,
        **extra,
    )
    print(json.dumps({"promoted": len(operations)}, indent=2))


def command_status(args: argparse.Namespace) -> None:
    config = _config(args.env)
    manifest = load_manifest()
    figures_root = Path(config["figures_dir"])
    status = []
    for figure in manifest["figures"]:
        staging = figures_root / "paper" / f"{figure['id']}.pdf"
        status.append(
            {
                "id": figure["id"],
                "mode": figure["mode"],
                "staged": staging.exists(),
                "subfigures_ready": all(
                    (figures_root / "paper" / "subfigure" / name).exists()
                    for name in figure_required_outputs(figure, manifest)
                ),
            }
        )
    print(json.dumps(status, indent=2))


def command_reproduce(args: argparse.Namespace) -> None:
    command_inputs(args)
    command_cities(args)
    command_aggregate(args)
    command_cohort(args)
    command_analyze(args)
    command_render(args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m drone_emission")
    parser.add_argument("--env", default=None, help="experiment.toml environment")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("inputs", help="validate explicit input roots").set_defaults(func=command_inputs)
    calibrate = subparsers.add_parser(
        "calibrate", help="jointly calibrate the Beijing building-choice gravity model"
    )
    calibrate.add_argument("--orders-corrected", type=Path, required=True)
    calibrate.add_argument("--buildings", type=Path, required=True)
    calibrate.add_argument("--transform", type=Path, required=True)
    calibrate.add_argument("--output", type=Path, required=True)
    calibrate.add_argument("--memory-limit-gb", type=float, default=64.0)
    calibrate.add_argument("--workers", type=int)
    calibrate.add_argument("--rebuild-prepared", action="store_true")
    calibrate.add_argument("--no-basemap", action="store_true")
    calibrate.add_argument("--accept-result", action="store_true")
    calibrate.set_defaults(func=command_calibrate)

    external = subparsers.add_parser(
        "calibrate-external", help="validate locked Beijing parameters on restored Meituan orders"
    )
    external.add_argument("--bundle", type=Path, required=True)
    external.add_argument("--buildings", type=Path, required=True)
    external.add_argument(
        "--parameters", type=Path,
        default=PROJECT_ROOT / "src/drone_emission/calibration/gravity_calibration_summary.json",
    )
    external.add_argument("--output", type=Path, required=True)
    external.add_argument("--memory-limit-gb", type=float, default=12.0)
    external.add_argument("--by-date", action="store_true", help="also evaluate each date separately")
    external.set_defaults(func=command_calibrate_external)

    external_buildings = subparsers.add_parser(
        "calibrate-external-buildings", help="prepare complete building choice sets for Meituan validation"
    )
    external_buildings.add_argument("--bundle", type=Path, required=True)
    external_buildings.add_argument("--gba-tiles", type=Path, nargs="+", required=True)
    external_buildings.add_argument("--output", type=Path, required=True)
    external_buildings.set_defaults(func=command_calibrate_external_buildings)

    spatial_fit = subparsers.add_parser(
        "calibrate-spatial-fit",
        help="add full-sample spatial-fit diagnostics to a converged calibration",
    )
    spatial_fit.add_argument("--result", type=Path, required=True)
    spatial_fit.add_argument("--no-basemap", action="store_true")
    spatial_fit.set_defaults(func=command_calibrate_spatial_fit)

    coordinate_parser = subparsers.add_parser(
        "calibrate-coordinates",
        help="restore and audit translated Beijing order coordinates",
    )
    coordinate_parser.add_argument(
        "--orders", type=Path, help="order CSV/TXT file or directory"
    )
    coordinate_parser.add_argument(
        "--poi", type=Path, help="restaurant POI shapefile, GeoPackage, or CSV"
    )
    coordinate_parser.add_argument(
        "--apply-transform", action="store_true",
        help="only apply an audited transform; do not load POIs or make figures",
    )
    coordinate_parser.add_argument("--transform", type=Path)
    coordinate_parser.add_argument(
        "--output", type=Path, help="output directory"
    )
    coordinate_parser.add_argument(
        "--poi-datum", choices=("gcj02", "wgs84"), default="gcj02"
    )
    coordinate_parser.add_argument("--match-radius-m", type=float, default=100.0)
    coordinate_parser.add_argument("--write-orders", action="store_true")
    coordinate_parser.add_argument("--no-basemap", action="store_true")
    coordinate_parser.add_argument("--heatmap-gridsize", type=int, default=100)
    coordinate_parser.set_defaults(func=command_calibrate_coordinates)

    cities = subparsers.add_parser("cities", help="run or restore city experiments")
    cities.add_argument("--region", action="append")
    cities.add_argument("--force", action="store_true")
    cities.add_argument("--no-fetch", action="store_true")
    cities.add_argument("--jobs", type=int)
    cities.add_argument("--cohort", choices=("locked",))
    cities.add_argument("--run-dir", type=Path)
    cities.add_argument("--resume", action="store_true")
    cities.add_argument("--start-at", choices=("roadnet", "road-clean"))
    cities.add_argument("--source-data-root", type=Path)
    cities.add_argument("--staging-data-root", type=Path)
    cities.add_argument("--memory-budget-gb", type=int, choices=(84,), default=84)
    cities.add_argument("--repair", action="append")
    cities.set_defaults(func=command_cities)

    aggregate = subparsers.add_parser("aggregate", help="build nationwide compatible tables")
    aggregate.add_argument("--force", action="store_true")
    aggregate.set_defaults(func=command_aggregate)

    cohort = subparsers.add_parser("cohort", help="audit the locked paper-city cohort")
    cohort_output = cohort.add_mutually_exclusive_group()
    cohort_output.add_argument("--accept", action="store_true")
    cohort.add_argument(
        "--shrink-only",
        action="store_true",
        help="audit only the currently locked cities; never re-admit exclusions",
    )
    cohort_output.add_argument(
        "--write-contract",
        type=Path,
        help="write the audited candidate contract without editing source",
    )
    cohort.set_defaults(func=command_cohort)

    analyze = subparsers.add_parser("analyze", help="build retained paper analyses")
    analyze.add_argument("--force", action="store_true")
    analyze.add_argument("--reuse-from", type=Path)
    analyze.add_argument(
        "--only",
        action="append",
        dest="analysis_only",
        help="analysis ID; comma-separated accepted",
    )
    analyze.add_argument(
        "--contribution-chunk-rows",
        type=int,
        default=250_000,
        help="OD rows per contribution-decomposition chunk",
    )
    analyze.set_defaults(func=command_analyze)

    cost_pilot = subparsers.add_parser(
        "cost-pilot", help="estimate citywide full costs for one explicit cohort city"
    )
    cost_pilot.add_argument("--region", required=True)
    cost_pilot.add_argument("--objectives", action="store_true", help="cross-evaluate all objectives for this city")
    cost_pilot.set_defaults(func=command_cost_pilot)

    prepare_reuse = subparsers.add_parser(
        "prepare-reuse", help="build one audited paper-analysis reuse directory"
    )
    prepare_reuse.add_argument("--output", type=Path, required=True)
    prepare_reuse.set_defaults(func=command_prepare_reuse)

    render_parser = subparsers.add_parser("render", help="render locked formal figure sources")
    render_selection = render_parser.add_mutually_exclusive_group()
    render_selection.add_argument(
        "--only",
        action="append",
        help="formal manifest figure ID; comma-separated accepted",
    )
    render_selection.add_argument(
        "--request",
        action="append",
        help="formal render request ID; comma-separated accepted",
    )
    render_parser.add_argument(
        "--force",
        action="store_true",
        help="ignore fresh render-request receipts",
    )
    render_parser.set_defaults(func=command_render)
    render_parser.add_argument("--prepare-only", action="store_true", help="save portable plot inputs without exporting figures")
    render_parser.add_argument("--bundle-dir", type=Path, help="destination for prepared plot inputs")
    render_parser.add_argument("--from-bundle", type=Path, help="strictly redraw from portable prepared inputs")
    render_parser.add_argument("--palette", type=Path, help="current paper palette TOML")

    compose_parser = subparsers.add_parser(
        "compose",
        help="compose linked subfigures through the formal Inkscape SVGs",
    )
    compose_parser.add_argument(
        "--only",
        action="append",
        help="Inkscape manifest figure ID; comma-separated accepted",
    )
    compose_parser.add_argument(
        "--prepare-links",
        action="store_true",
        help="convert subfigure PDFs to linked SVGs without exporting formal figures",
    )
    compose_parser.add_argument(
        "--force",
        action="store_true",
        help="refresh linked SVGs and formal PDFs regardless of modification time",
    )
    compose_parser.set_defaults(func=command_compose)
    compose_parser.add_argument(
        "--preview-only", action="store_true",
        help="refresh previews for selected staged figures without recomposing",
    )

    promote_parser = subparsers.add_parser("promote", help="publish all staged PDFs to docs/paper")
    promote_parser.add_argument("--yes", action="store_true")
    promote_parser.add_argument("--dry-run", action="store_true")
    promote_parser.set_defaults(func=command_promote)

    city_promote = subparsers.add_parser(
        "city-promote", help="transactionally publish all audited city results"
    )
    city_promote.add_argument("--source-data-root", type=Path, required=True)
    city_promote.add_argument("--staging-data-root", type=Path, required=True)
    city_promote.add_argument(
        "--run-dir", type=Path, help="bind partial promotion to a road-clean batch"
    )
    city_promote.add_argument("--yes", action="store_true")
    city_promote.set_defaults(func=command_city_promote)

    subparsers.add_parser("status", help="show manifest and staging status").set_defaults(func=command_status)

    reproduce = subparsers.add_parser("reproduce", help="run inputs, cities, aggregate, analyze, and render")
    reproduce.add_argument("--region", action="append")
    reproduce.add_argument("--only", action="append")
    reproduce.add_argument("--force", action="store_true")
    reproduce.add_argument("--no-fetch", action="store_true")
    reproduce.add_argument("--reuse-from", type=Path)
    reproduce.set_defaults(accept=False)
    reproduce.set_defaults(func=command_reproduce)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
