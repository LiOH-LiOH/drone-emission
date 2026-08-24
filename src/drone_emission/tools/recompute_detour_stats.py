"""Repair weighted city detour statistics from existing OD files.

The command is dry-run by default. Pass ``--write`` to atomically update the
selected per-city ``region_stats.json`` files. ``--patch-summary`` updates only
the selected rows and distance fields in the existing consolidated JSON/CSV;
it deliberately does not run a full consolidation.

Examples:
    python -m tools.recompute_detour_stats --env remote \
        --region beijing --region shanghai
    python -m tools.recompute_detour_stats --env remote \
        --region beijing --region shanghai --write --patch-summary --enrich
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]

from drone_emission.analysis.city_attribution import compute_weighted_detour_stats  # noqa: E402
from drone_emission.utils import load_config  # noqa: E402


PATCH_FIELDS = (
    "avg_road_distance_m",
    "avg_straight_distance_m",
    "detour_ratio",
    "valid_path_count",
    "invalid_path_count",
    "detour_stats_source",
    "detour_stats_source_path",
    "detour_stats_source_mtime_ns",
    "detour_stats_repaired_at",
)


def _atomic_json(path: Path, payload: dict) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2, allow_nan=False)
        file.write("\n")
    os.replace(tmp_path, path)


def _atomic_csv(path: Path, df: pd.DataFrame) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp_path, index=False, quoting=csv.QUOTE_MINIMAL)
    os.replace(tmp_path, path)


def _backup_files(data_dir: Path, paths: list[Path]) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = data_dir / "backups" / f"detour_stats_{timestamp}"
    backup_dir.mkdir(parents=True, exist_ok=False)
    for path in paths:
        if not path.exists():
            continue
        relative = path.relative_to(data_dir)
        destination = backup_dir / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
    return backup_dir


def _region_paths(data_dir: Path, region_key: str) -> tuple[Path, Path]:
    region_dir = data_dir / "city" / region_key
    return (
        region_dir / f"{region_key}_od_pairs.csv",
        region_dir / "region_stats.json",
    )


def _load_region_stats(path: Path) -> dict:
    if not path.exists():
        return {"stats": {}, "cache": {}}
    with open(path, "r", encoding="utf-8") as file:
        payload = json.load(file)
    payload.setdefault("stats", {})
    payload.setdefault("cache", {})
    return payload


def _patch_summary_csv(
    summary_path: Path,
    computed_by_region: dict[str, dict],
) -> None:
    df = pd.read_csv(summary_path)
    for field in PATCH_FIELDS:
        if field not in df.columns:
            df[field] = pd.NA
    for region_key, computed in computed_by_region.items():
        mask = df["region_key"] == region_key
        if int(mask.sum()) != 1:
            raise ValueError(
                f"Expected one {region_key} row in {summary_path}, found {int(mask.sum())}"
            )
        for field in PATCH_FIELDS:
            df.loc[mask, field] = computed[field]
    _atomic_csv(summary_path, df)


def _patch_summary_json(
    summary_path: Path,
    computed_by_region: dict[str, dict],
) -> None:
    with open(summary_path, "r", encoding="utf-8") as file:
        payload = json.load(file)
    for region_key, computed in computed_by_region.items():
        if region_key not in payload:
            raise KeyError(f"Missing {region_key} in {summary_path}")
        stats_payload = payload[region_key].setdefault("stats", {})
        for field in PATCH_FIELDS:
            stats_payload[field] = computed[field]
    _atomic_json(summary_path, payload)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Repair annual-sales-weighted city detour statistics."
    )
    parser.add_argument("--env", type=str, default=None)
    parser.add_argument(
        "--region",
        action="append",
        required=True,
        help="City region key; repeat for multiple cities.",
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="Write per-city region_stats.json; otherwise dry-run only.",
    )
    parser.add_argument(
        "--patch-summary",
        action="store_true",
        help="Patch selected rows in all_region_stats.json/csv (requires --write).",
    )
    parser.add_argument(
        "--enrich",
        action="store_true",
        help="Rebuild all_region_stats_enriched.csv after summary patch.",
    )
    args = parser.parse_args()
    if args.patch_summary and not args.write:
        parser.error("--patch-summary requires --write")
    if args.enrich and not args.patch_summary:
        parser.error("--enrich requires --patch-summary")

    config = load_config(str(PROJECT_ROOT / "experiment.toml"), env=args.env)
    data_dir = Path(config["data_dir"])
    regions = list(dict.fromkeys(args.region))
    computed_by_region = {}
    stats_payloads = {}
    stats_paths = {}

    repair_timestamp = datetime.now().astimezone().isoformat(timespec="seconds")
    for region_key in regions:
        od_path, stats_path = _region_paths(data_dir, region_key)
        if not od_path.exists():
            raise FileNotFoundError(od_path)
        computed = compute_weighted_detour_stats(od_path)
        computed.update({
            "detour_stats_source": "annual_sales_weighted_existing_od",
            "detour_stats_source_path": computed["source_path"],
            "detour_stats_source_mtime_ns": computed["source_mtime_ns"],
            "detour_stats_repaired_at": repair_timestamp,
        })
        payload = _load_region_stats(stats_path)
        old_values = {field: payload["stats"].get(field) for field in PATCH_FIELDS}
        computed_by_region[region_key] = computed
        stats_payloads[region_key] = payload
        stats_paths[region_key] = stats_path
        print(
            f"{region_key}: detour_ratio {old_values['detour_ratio']} -> "
            f"{computed['detour_ratio']:.9f}; valid_paths={computed['valid_path_count']:,}"
        )

    if not args.write:
        print("Dry-run complete; no files changed.")
        return

    summary_json = data_dir / "all_region_stats.json"
    summary_csv = data_dir / "all_region_stats.csv"
    enriched_csv = data_dir / "all_region_stats_enriched.csv"
    backup_targets = list(stats_paths.values())
    if args.patch_summary:
        backup_targets.extend([summary_json, summary_csv, enriched_csv])
    backup_dir = _backup_files(data_dir, backup_targets)
    print(f"Backup saved to {backup_dir}")

    for region_key, payload in stats_payloads.items():
        for field in PATCH_FIELDS:
            payload["stats"][field] = computed_by_region[region_key][field]
        _atomic_json(stats_paths[region_key], payload)
        print(f"Updated {stats_paths[region_key]}")

    if args.patch_summary:
        _patch_summary_json(summary_json, computed_by_region)
        _patch_summary_csv(summary_csv, computed_by_region)
        print(f"Patched {summary_json}")
        print(f"Patched {summary_csv}")

    if args.enrich:
        from drone_emission.tools.enrich_stats import enrich

        enrich(
            data_dir=str(data_dir),
            sales_path=config["sales_path"],
            gef_path=config["gef_path"],
        )


if __name__ == "__main__":
    main()
