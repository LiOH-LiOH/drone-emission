"""Contracts for isolated resource planning and transactional promotion."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from drone_emission import city_staging


class Region:
    def __init__(self, name):
        self.region_name_en = name
        self.region_level = "city"


def _promotion_fixture(tmp_path: Path):
    source = tmp_path / "source"
    staging = tmp_path / "staging"
    regions = {"alpha": Region("Alpha"), "beta": Region("Beta")}
    for key, region in regions.items():
        name = region.region_name_en.lower()
        source_dir = source / "city" / name
        staging_dir = staging / "city" / name
        source_dir.mkdir(parents=True)
        staging_dir.mkdir(parents=True)
        for suffix in city_staging.PROMOTED_SUFFIXES:
            (source_dir / f"{name}{suffix}").write_text(f"old-{key}-{suffix}")
            (staging_dir / f"{name}{suffix}").write_text(f"new-{key}-{suffix}")
        (source_dir / "region_stats.json").write_text("old-stats")
        (staging_dir / "region_stats.json").write_text("new-stats")
    staging.mkdir(exist_ok=True)
    (staging / "acceptance_report.json").write_text(
        json.dumps({"ok": True, "audited_count": 2})
    )
    return source, staging, regions


def test_resource_thresholds_match_frozen_contract():
    gib = 1024**3
    assert city_staging.classify_resources(
        building_bytes=0, graphml_bytes=2 * gib, od_bytes=0
    ) == "heavy"
    assert city_staging.classify_resources(
        building_bytes=0, graphml_bytes=1 * gib, od_bytes=0
    ) == "medium"
    assert city_staging.classify_resources(
        building_bytes=1, graphml_bytes=1, od_bytes=1
    ) == "light"


def test_promotion_requires_explicit_confirmation(tmp_path):
    source, staging, regions = _promotion_fixture(tmp_path)
    with pytest.raises(PermissionError, match="--yes"):
        city_staging.promote_city_results(
            source_root=source,
            staging_root=staging,
            regions=regions,
            region_keys=list(regions),
            yes=False,
        )


def test_partial_promotion_failure_rolls_back(tmp_path, monkeypatch):
    source, staging, regions = _promotion_fixture(tmp_path)
    original = city_staging.atomic_replace
    calls = 0

    def fail_second(temporary, target):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("injected publication failure")
        original(temporary, target)

    monkeypatch.setattr(city_staging, "atomic_replace", fail_second)
    with pytest.raises(OSError, match="injected"):
        city_staging.promote_city_results(
            source_root=source,
            staging_root=staging,
            regions=regions,
            region_keys=list(regions),
            yes=True,
        )
    assert (source / "city" / "alpha" / "alpha_sales.csv").read_text().startswith("old-")
    assert not list(source.rglob("*.pre-rerun.bak"))


def test_successful_promotion_removes_backups(tmp_path):
    source, staging, regions = _promotion_fixture(tmp_path)
    operations = city_staging.promote_city_results(
        source_root=source,
        staging_root=staging,
        regions=regions,
        region_keys=list(regions),
        yes=True,
    )
    assert len(operations) == 8
    assert (source / "city" / "alpha" / "alpha_sales.csv").read_text().startswith("new-")
    assert not list(source.rglob("*.pre-rerun.bak"))


def _write_routed_city(root: Path, region: Region, road_distances: list[float]):
    name = region.region_name_en.lower()
    city_dir = root / "city" / name
    city_dir.mkdir(parents=True)
    sales = [12 * (index + 1) for index in range(len(road_distances))]
    pd.DataFrame(
        {
            "from_lat": [1.0] * len(road_distances),
            "from_lng": [1.0] * len(road_distances),
            "to_lat": [1.1] * len(road_distances),
            "to_lng": [1.1] * len(road_distances),
            "distance_km": [1.0] * len(road_distances),
            "annual_sales": sales,
            "road_distance_m": road_distances,
        }
    ).to_csv(city_dir / f"{name}_od_pairs.csv", index=False)
    (city_dir / "region_stats.json").write_text(
        json.dumps(
            {
                "stats": {
                    "odpair_count": len(road_distances),
                    "total_odpair_annual_sales": sum(sales),
                },
                "cache": {},
            }
        ),
        encoding="utf-8",
    )


def test_road_clean_preflight_stages_only_affected_cities(tmp_path):
    source = tmp_path / "source"
    staging = tmp_path / "staging"
    source.mkdir()
    (source / "combined_data.csv").write_text("x\n1\n", encoding="utf-8")
    (source / "gef_china.csv").write_text("x\n1\n", encoding="utf-8")
    regions = {"alpha": Region("Alpha"), "beta": Region("Beta")}
    _write_routed_city(source, regions["alpha"], [1000.0, -1.0])
    _write_routed_city(source, regions["beta"], [1000.0, 2000.0])

    result = city_staging.prepare_road_clean_run(
        source_root=source,
        staging_root=staging,
        regions=regions,
        region_keys=["alpha", "beta"],
    )

    assert result["affected_region_keys"] == ["alpha"]
    assert (staging / "city" / "alpha" / "alpha_od_pairs.csv").is_file()
    assert not (staging / "city" / "beta").exists()


def test_minimal_promotion_replaces_only_od_and_stats(tmp_path):
    source, staging, regions = _promotion_fixture(tmp_path)
    (staging / "acceptance_report.json").write_text(
        json.dumps({"ok": True, "audited_count": 1})
    )

    operations = city_staging.promote_city_results(
        source_root=source,
        staging_root=staging,
        regions=regions,
        region_keys=["alpha"],
        yes=True,
        promoted_suffixes=("_od_pairs.csv",),
    )

    assert len(operations) == 2
    assert (source / "city" / "alpha" / "alpha_od_pairs.csv").read_text().startswith(
        "new-"
    )
    assert (source / "city" / "alpha" / "alpha_sales.csv").read_text().startswith(
        "old-"
    )
