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
    config = _config(args.env)
    raw_root = Path(
        os.environ.get(
            "DRONE_EMISSION_BEIJING_OD_ROOT",
            Path(config["data_dir"]) / "raw" / "即时配送数据" / "疫情期间北京饿了么" / "eleme_round1_train_20200313" / "order",
        )
    )
    if not raw_root.is_dir() or not any(raw_root.glob("*.txt")):
        raise FileNotFoundError(
            "Beijing raw OD files are unavailable; locked experiment.toml and "
            "calibration CSV remain the default. Set DRONE_EMISSION_BEIJING_OD_ROOT."
        )
    from drone_emission.calibration import cali_elm

    cali_elm.elm_data_dir = str(raw_root)
    distances = cali_elm.load_all_eleme_distances(
        min_orders_per_shop=50, max_distance_km=10.0
    )
    a_value, b_value, r_squared = cali_elm.fit_tanner_impedance(distances)
    output = Path(config["work_dir"]) / "calibration" / "tanner_fit.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {"tanner_a": a_value, "tanner_b": b_value, "r_squared": r_squared},
            indent=2,
        ),
        encoding="utf-8",
    )
    print(output)


def command_cities(args: argparse.Namespace) -> None:
    config = _config(args.env)
    problems = _input_problems(config)
    if problems:
        raise FileNotFoundError("Missing city input(s): " + ", ".join(problems))
    from drone_emission.main import has_city_results, run_all, run_region_analysis
    from drone_emission.utils.config import configure_cache
    from drone_emission.utils.region_config import init_registry

    if args.no_fetch:
        config["fetch_buildings"] = False
        config["fetch_roadnet"] = False
    configure_cache(config)
    registry = init_registry(
        str(PROJECT_ROOT / "regions.json"), data_dir=config["data_dir"]
    )
    if args.region:
        for region_key in args.region:
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
    subparsers.add_parser("calibrate", help="rebuild calibration when Beijing raw OD is available").set_defaults(func=command_calibrate)

    cities = subparsers.add_parser("cities", help="run or restore city experiments")
    cities.add_argument("--region", action="append")
    cities.add_argument("--force", action="store_true")
    cities.add_argument("--no-fetch", action="store_true")
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
