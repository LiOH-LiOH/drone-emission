"""Contracts for isolated resource planning and transactional promotion."""

from __future__ import annotations

import json
from pathlib import Path

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
