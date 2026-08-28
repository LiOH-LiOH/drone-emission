"""Isolated city-run input preparation, resource planning, and promotion."""

from __future__ import annotations

import csv
import json
import math
import os
import shutil
from pathlib import Path
from typing import Iterable

from drone_emission.roadnet.api import validate_od_checkpoint
from drone_emission.utils.atomic import atomic_open, atomic_replace
from drone_emission.utils.boundaries import graphml_covers_bbox, resolve_enlarged_boundaries


RESOURCE_GB = {"light": 21, "medium": 42, "heavy": 84}
PROMOTED_SUFFIXES = (
    "_sales.csv",
    "_buildings_preprocessed.gpkg",
    "_od_pairs.csv",
)
REQUIRED_FINAL_OD_COLUMNS = {
    "road_distance_m",
    "drone_emission_kg",
    "ebike_emission_kg",
    "drone_lifecycle_g",
    "ebike_lifecycle_g",
    "drone_time_min",
    "ebike_time_min",
    "grid_emission_factor",
    "ebike_kappa",
    "drone_lifecycle_production_g",
    "drone_lifecycle_use_g",
    "drone_lifecycle_eol_g",
    "ebike_lifecycle_production_g",
    "ebike_lifecycle_use_g",
    "ebike_lifecycle_eol_g",
}
NETWORK_STAT_KEYS = (
    "orientation_entropy",
    "dead_end_proportion",
    "four_way_proportion",
    "avg_street_length",
)


def _city_paths(root: Path, region) -> tuple[Path, str]:
    name = region.region_name_en.lower()
    return root / region.region_level / name, name


def _first_existing(paths: Iterable[Path]) -> Path | None:
    return next((path for path in paths if path.is_file()), None)


def classify_resources(*, building_bytes: int, graphml_bytes: int, od_bytes: int) -> str:
    """Apply the frozen heavy/medium thresholds from the rerun contract."""
    gib = 1024**3
    if graphml_bytes >= 2 * gib or building_bytes >= 1.5 * gib or od_bytes >= 4 * gib:
        return "heavy"
    if graphml_bytes >= 1 * gib or building_bytes >= 0.75 * gib or od_bytes >= 2 * gib:
        return "medium"
    return "light"


def prepare_isolated_run(
    *,
    source_root: Path,
    staging_root: Path,
    regions: dict,
    region_keys: list[str],
    exp_config: dict,
) -> dict:
    """Preflight every locked input, link immutable inputs, and freeze a plan."""
    source_root = source_root.resolve()
    staging_root = staging_root.resolve()
    if source_root == staging_root:
        raise ValueError("source and staging data roots must differ")
    if staging_root.is_relative_to(source_root) or source_root.is_relative_to(
        staging_root
    ):
        raise ValueError("source and staging data roots must not contain one another")
    if not source_root.is_dir():
        raise FileNotFoundError(f"source data root does not exist: {source_root}")
    if staging_root.exists() and any(staging_root.iterdir()):
        raise FileExistsError(
            f"new isolated staging root must be empty: {staging_root}"
        )
    staging_root.mkdir(parents=True, exist_ok=True)
    problems: list[dict] = []
    plan: list[dict] = []
    for shared in ("combined_data.csv", "gef_china.csv"):
        source = source_root / shared
        if not source.is_file():
            problems.append(
                {"region_key": "_shared", "input": shared, "error": "missing"}
            )
            continue
        target = staging_root / shared
        if target.exists() or target.is_symlink():
            if target.resolve() != source:
                problems.append(
                    {"region_key": "_shared", "input": shared, "error": "wrong_link"}
                )
        else:
            target.symlink_to(source)

    for region_key in region_keys:
        region = regions[region_key]
        source_dir, name = _city_paths(source_root, region)
        target_dir, _ = _city_paths(staging_root, region)
        target_dir.mkdir(parents=True, exist_ok=True)
        building = _first_existing(
            (
                source_dir / f"{name}_buildings.gpkg",
                source_dir / f"{name}_buildings.geojson",
            )
        )
        graphml = source_dir / f"{name}_roadnet.graphml"
        old_od = source_dir / f"{name}_od_pairs.csv"
        source_stats = source_dir / "region_stats.json"
        if building is None:
            problems.append({"region_key": region_key, "input": "buildings", "error": "missing"})
        if not graphml.is_file():
            problems.append({"region_key": region_key, "input": "graphml", "error": "missing"})
        elif graphml.stat().st_size == 0:
            problems.append({"region_key": region_key, "input": "graphml", "error": "empty"})
        else:
            try:
                expected = resolve_enlarged_boundaries(region, exp_config)
                if expected and not graphml_covers_bbox(graphml, expected):
                    problems.append({"region_key": region_key, "input": "graphml", "error": "bbox"})
            except Exception as error:
                problems.append({"region_key": region_key, "input": "graphml", "error": f"unreadable:{error}"})
        if building is not None:
            target = target_dir / building.name
            if not target.exists() and not target.is_symlink():
                target.symlink_to(building)
        if graphml.is_file():
            target = target_dir / graphml.name
            if not target.exists() and not target.is_symlink():
                target.symlink_to(graphml)
        try:
            source_payload = json.loads(source_stats.read_text(encoding="utf-8"))
            source_network_stats = source_payload["stats"]
            seed_stats = {
                key: source_network_stats[key] for key in NETWORK_STAT_KEYS
            }
            if not all(math.isfinite(float(value)) for value in seed_stats.values()):
                raise ValueError("network metrics must be finite")
        except (OSError, ValueError, KeyError, TypeError) as error:
            problems.append(
                {
                    "region_key": region_key,
                    "input": "network_stats",
                    "error": f"missing_or_invalid:{error}",
                }
            )
        else:
            with atomic_open(
                target_dir / "region_stats.json", "w", encoding="utf-8"
            ) as handle:
                json.dump(
                    {
                        "stats": seed_stats,
                        "cache": {},
                        "seed_provenance": str(source_stats.resolve()),
                    },
                    handle,
                    ensure_ascii=False,
                    indent=2,
                )
                handle.write("\n")
        sizes = {
            "building_bytes": building.stat().st_size if building else 0,
            "graphml_bytes": graphml.stat().st_size if graphml.is_file() else 0,
            "old_od_bytes": old_od.stat().st_size if old_od.is_file() else 0,
        }
        resource_class = classify_resources(
            building_bytes=sizes["building_bytes"],
            graphml_bytes=sizes["graphml_bytes"],
            od_bytes=sizes["old_od_bytes"],
        )
        plan.append(
            {
                "region_key": region_key,
                **sizes,
                "resource_class": resource_class,
                "reservation_gb": RESOURCE_GB[resource_class],
            }
        )
    disk_free_bytes = shutil.disk_usage(staging_root).free
    estimated_required_bytes = sum(
        item["building_bytes"] + 3 * item["old_od_bytes"] for item in plan
    )
    if disk_free_bytes < int(estimated_required_bytes * 1.1):
        problems.append(
            {
                "region_key": "_batch",
                "input": "disk",
                "error": "insufficient_space",
            }
        )
    payload = {
        "schema_version": 1,
        "source_data_root": str(source_root),
        "staging_data_root": str(staging_root),
        "city_count": len(region_keys),
        "ok": not problems,
        "problems": problems,
        "disk_free_bytes": disk_free_bytes,
        "estimated_required_bytes": estimated_required_bytes,
        "resource_plan": plan,
    }
    with atomic_open(staging_root / "preflight.json", "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    if problems:
        raise RuntimeError(f"strict offline preflight failed for {len(problems)} input(s)")
    return payload


def _promotion_sources(staging_root: Path, region) -> list[tuple[Path, Path]]:
    staging_dir, name = _city_paths(staging_root, region)
    result = [
        (staging_dir / f"{name}{suffix}", Path(f"{name}{suffix}"))
        for suffix in PROMOTED_SUFFIXES
    ]
    result.append((staging_dir / "region_stats.json", Path("region_stats.json")))
    return result


def promote_city_results(
    *,
    source_root: Path,
    staging_root: Path,
    regions: dict,
    region_keys: list[str],
    yes: bool,
) -> list[dict]:
    """Transactionally replace all derived city artifacts after explicit consent."""
    if not yes:
        raise PermissionError("city promotion requires explicit --yes confirmation")
    acceptance_path = staging_root / "acceptance_report.json"
    if not acceptance_path.is_file():
        raise FileNotFoundError("city promotion requires acceptance_report.json")
    acceptance = json.loads(acceptance_path.read_text(encoding="utf-8"))
    if not acceptance.get("ok") or acceptance.get("audited_count") != len(region_keys):
        raise RuntimeError("city promotion requires a complete successful acceptance audit")
    operations: list[tuple[Path, Path]] = []
    missing = []
    for key in region_keys:
        target_dir, _ = _city_paths(source_root, regions[key])
        for staged, relative in _promotion_sources(staging_root, regions[key]):
            target = target_dir / relative
            if not staged.is_file() or staged.is_symlink():
                missing.append(str(staged))
            operations.append((staged, target))
    if missing:
        raise FileNotFoundError("promotion manifest is incomplete: " + ", ".join(missing[:10]))

    backups: list[tuple[Path, Path]] = []
    published: list[Path] = []
    temporaries: list[Path] = []
    try:
        for staged, target in operations:
            target.parent.mkdir(parents=True, exist_ok=True)
            backup = target.with_name(f".{target.name}.pre-rerun.bak")
            if target.exists() or target.is_symlink():
                os.replace(target, backup)
                backups.append((backup, target))
            temporary = target.with_name(f".{target.name}.{os.getpid()}.promote.tmp")
            temporaries.append(temporary)
            shutil.copy2(staged, temporary)
            atomic_replace(temporary, target)
            temporaries.remove(temporary)
            published.append(target)
    except BaseException:
        for temporary in temporaries:
            temporary.unlink(missing_ok=True)
        for target in reversed(published):
            target.unlink(missing_ok=True)
        for backup, target in reversed(backups):
            os.replace(backup, target)
        raise
    for backup, _ in backups:
        backup.unlink(missing_ok=True)
    return [{"source": str(source), "target": str(target)} for source, target in operations]


def audit_staged_results(
    *,
    source_root: Path,
    staging_root: Path,
    regions: dict,
    region_keys: list[str],
    status_path: Path | None = None,
) -> dict:
    """Build the post-city conservation, completeness, and difference report."""
    resources = {}
    if status_path and status_path.is_file():
        with status_path.open(encoding="utf-8", newline="") as handle:
            resources = {
                row["region_key"]: row
                for row in csv.DictReader(handle, delimiter="\t")
            }
    rows = []
    problems = []
    for key in region_keys:
        staged_dir, name = _city_paths(staging_root, regions[key])
        source_dir, _ = _city_paths(source_root, regions[key])
        od_path = staged_dir / f"{name}_od_pairs.csv"
        stats_path = staged_dir / "region_stats.json"
        try:
            checkpoint = validate_od_checkpoint(od_path)
            missing_columns = sorted(REQUIRED_FINAL_OD_COLUMNS - set(checkpoint["columns"]))
            if missing_columns:
                raise ValueError(f"missing final OD columns: {missing_columns}")
            stats = json.loads(stats_path.read_text(encoding="utf-8"))["stats"]
            if int(stats.get("od_allocation_annual_sales_error", 1)) != 0:
                raise ValueError("OD allocation error is nonzero")
        except Exception as error:
            problems.append({"region_key": key, "error": str(error)})
            continue
        old_od = source_dir / f"{name}_od_pairs.csv"
        old_bytes = old_od.stat().st_size if old_od.is_file() else None
        old_annual_sales = None
        if old_od.is_file():
            try:
                import pandas as pd

                old_annual_sales = sum(
                    int(pd.to_numeric(chunk["annual_sales"], errors="coerce").fillna(0).sum())
                    for chunk in pd.read_csv(old_od, usecols=["annual_sales"], chunksize=250_000)
                )
            except (OSError, ValueError, KeyError):
                old_annual_sales = None
        rows.append(
            {
                "region_key": key,
                "rows": checkpoint["rows"],
                "annual_sales": checkpoint["annual_sales"],
                "new_od_bytes": od_path.stat().st_size,
                "old_od_bytes": old_bytes,
                "od_size_ratio": (
                    od_path.stat().st_size / old_bytes if old_bytes else None
                ),
                "old_annual_sales": old_annual_sales,
                "annual_sales_delta": (
                    checkpoint["annual_sales"] - old_annual_sales
                    if old_annual_sales is not None else None
                ),
                "allocation_error": 0,
                "resource_class": resources.get(key, {}).get("resource_class"),
                "peak_rss_gb": resources.get(key, {}).get("peak_rss_gb"),
                "duration_s": resources.get(key, {}).get("duration_s"),
            }
        )
    report = {
        "schema_version": 1,
        "ok": not problems and len(rows) == len(region_keys),
        "city_count": len(region_keys),
        "audited_count": len(rows),
        "problems": problems,
        "cities": rows,
    }
    with atomic_open(staging_root / "acceptance_report.json", "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    if not report["ok"]:
        raise RuntimeError(f"staged acceptance audit failed for {len(problems)} city/cities")
    return report


def validate_staged_city(staging_root: Path, region) -> dict:
    """Reject a city success marker unless all scientific artifacts are complete."""
    staged_dir, name = _city_paths(staging_root, region)
    od_path = staged_dir / f"{name}_od_pairs.csv"
    checkpoint = validate_od_checkpoint(od_path)
    missing_columns = sorted(REQUIRED_FINAL_OD_COLUMNS - set(checkpoint["columns"]))
    if missing_columns:
        raise ValueError(f"missing final OD columns: {missing_columns}")
    for suffix in ("_sales.csv", "_buildings_preprocessed.gpkg"):
        path = staged_dir / f"{name}{suffix}"
        if not path.is_file() or path.is_symlink() or path.stat().st_size == 0:
            raise FileNotFoundError(f"missing generated city artifact: {path}")
    stats_path = staged_dir / "region_stats.json"
    stats = json.loads(stats_path.read_text(encoding="utf-8"))["stats"]
    if int(stats.get("od_allocation_annual_sales_error", 1)) != 0:
        raise ValueError("OD allocation error is nonzero")
    if int(stats.get("total_odpair_annual_sales", -1)) != checkpoint["annual_sales"]:
        raise ValueError("region stats and OD annual sales disagree")
    return checkpoint
