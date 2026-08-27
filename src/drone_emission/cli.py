"""Single public command surface for end-to-end paper reproduction."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .paper_workflow import load_manifest, promote, render
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
    config = _config(args.env)
    problems = _input_problems(config)
    if problems:
        raise FileNotFoundError("Missing city input(s): " + ", ".join(problems))

    jobs = getattr(args, "jobs", None)
    cohort = getattr(args, "cohort", None)
    run_dir = getattr(args, "run_dir", None)
    resume = bool(getattr(args, "resume", False))
    start_at = getattr(args, "start_at", None)
    batch_requested = any(
        (
            jobs is not None,
            cohort is not None,
            run_dir is not None,
            resume,
        )
    )
    if args.region and batch_requested:
        raise ValueError("--region cannot be combined with audited batch options")
    if start_at and not args.region:
        raise ValueError("--start-at requires at least one --region")
    if start_at and batch_requested:
        raise ValueError("--start-at cannot be combined with audited batch options")
    if batch_requested:
        if jobs is None or cohort is None or run_dir is None:
            raise ValueError(
                "audited batch mode requires --jobs, --cohort, and --run-dir"
            )
        from drone_emission.city_batch import run_city_batch
        from drone_emission.paper_cohort import cohort_hash, locked_region_keys

        exit_code = run_city_batch(
            region_keys=locked_region_keys(),
            environment=args.env or "local",
            jobs=jobs,
            run_dir=run_dir,
            force=args.force,
            no_fetch=args.no_fetch,
            cohort_hash=cohort_hash(),
            resume=resume,
        )
        if exit_code:
            raise SystemExit(exit_code)
        return

    from drone_emission.main import (
        has_city_results,
        run_all,
        run_region_analysis,
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

    payload = audit_city_cohort(config)
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
    if not payload["matches_locked"]:
        raise SystemExit(2)


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
            analyze(config, force=args.force, reuse_from=args.reuse_from), indent=2
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
        _atomic_copy,
        capture_reuse_directory,
        migrate_fallback_gef_overlay,
        write_migration_difference_report,
    )

    assert_locked_cohort(config)
    view = _view_config(config)
    spatial_target = (
        Path(view["cache_dir"])
        / "drone_spatial_concentration"
        / "city_drone_spatial_concentration.csv"
    )
    _atomic_copy(args.legacy_spatial.resolve(), spatial_target)
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
    config = _config(args.env)
    from .analysis_workflow import (
        analysis_data_dir,
        prepare_analysis_data_view,
        require_current_analysis,
    )

    require_current_analysis(config)
    prepare_analysis_data_view(config)
    config = dict(config, data_dir=str(analysis_data_dir(config)))
    staged = render(config, only=_split_ids(args.only))
    if staged:
        print("Staged direct figures:")
        for path in staged:
            print(path)
    print("Inkscape figures require manual assembly from manifest subfigures.")


def command_promote(args: argparse.Namespace) -> None:
    operations = promote(_config(args.env), yes=args.yes, dry_run=args.dry_run)
    prefix = "Would promote" if args.dry_run else "Promoted"
    for source, target in operations:
        print(f"{prefix}: {source} -> {target}")


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
                    for name in figure.get("required_subfigures", [])
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
    cities.add_argument("--start-at", choices=("roadnet",))
    cities.set_defaults(func=command_cities)

    aggregate = subparsers.add_parser("aggregate", help="build nationwide compatible tables")
    aggregate.add_argument("--force", action="store_true")
    aggregate.set_defaults(func=command_aggregate)

    cohort = subparsers.add_parser("cohort", help="audit the locked paper-city cohort")
    cohort.add_argument("--accept", action="store_true")
    cohort.set_defaults(func=command_cohort)

    analyze = subparsers.add_parser("analyze", help="build retained paper analyses")
    analyze.add_argument("--force", action="store_true")
    analyze.add_argument("--reuse-from", type=Path)
    analyze.set_defaults(func=command_analyze)

    prepare_reuse = subparsers.add_parser(
        "prepare-reuse", help="build one audited paper-analysis reuse directory"
    )
    prepare_reuse.add_argument("--legacy-spatial", type=Path, required=True)
    prepare_reuse.add_argument("--output", type=Path, required=True)
    prepare_reuse.set_defaults(func=command_prepare_reuse)

    render_parser = subparsers.add_parser("render", help="render locked formal figure sources")
    render_parser.add_argument("--only", action="append", help="formal manifest ID; comma-separated accepted")
    render_parser.set_defaults(func=command_render)

    promote_parser = subparsers.add_parser("promote", help="publish all staged PDFs to docs/paper")
    promote_parser.add_argument("--yes", action="store_true")
    promote_parser.add_argument("--dry-run", action="store_true")
    promote_parser.set_defaults(func=command_promote)

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
