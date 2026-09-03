"""Focused contracts for strict no-fetch and roadnet-stage city recovery."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from drone_emission import main
from drone_emission.buildings import api as buildings_api
from drone_emission.cli import build_parser
from drone_emission.roadnet import api as roadnet_api


class FakeRegion:
    def __init__(self, name: str = "TestCity"):
        self.name = name
        self.stats = {"orientation_entropy": 1.0}

    def to_dict(self):
        return {
            "region_name_en": self.name,
            "region_level": "city",
        }

    def set_stat(self, key, value):
        self.stats[key] = value

    def get_stat(self, key, default=None):
        return self.stats.get(key, default)


class FakeGraph:
    num_nodes = 3
    num_edges = 2
    total_length = 12.0


def _roadnet_path(tmp_path: Path, name: str = "testcity") -> Path:
    path = tmp_path / "city" / name / f"{name}_roadnet.graphml"
    path.parent.mkdir(parents=True)
    path.write_text("graph", encoding="utf-8")
    return path


def _config(tmp_path: Path, strict: bool = True) -> dict:
    return {
        "data_dir": str(tmp_path),
        "fetch_roadnet": False,
        "strict_no_fetch": strict,
    }


def test_strict_no_fetch_rejects_a_missing_roadnet(tmp_path, monkeypatch):
    monkeypatch.setattr(
        roadnet_api, "resolve_enlarged_boundaries", lambda *_: {"min_lat": 0}
    )
    monkeypatch.setattr(
        roadnet_api,
        "_fetch_bikeable_roadnet",
        lambda **_: pytest.fail("strict no-fetch attempted a download"),
    )

    with pytest.raises(FileNotFoundError, match="--no-fetch"):
        roadnet_api.load_roadnet(FakeRegion(), _config(tmp_path))


def test_strict_no_fetch_rejects_missing_buildings(tmp_path, monkeypatch):
    region = FakeRegion()
    region.to_dict = lambda: {
        "region_name_en": "TestCity",
        "region_level": "city",
        "boundaries_dict": {"min_lat": 0},
    }
    monkeypatch.setattr(
        buildings_api,
        "resolve_enlarged_boundaries",
        lambda *_: {"min_lat": 0, "min_lng": 0, "max_lat": 1, "max_lng": 1},
    )

    with pytest.raises(FileNotFoundError, match="--no-fetch"):
        buildings_api.load_buildings(region, _config(tmp_path))


def test_strict_no_fetch_rejects_a_stale_roadnet(tmp_path, monkeypatch):
    _roadnet_path(tmp_path)
    monkeypatch.setattr(
        roadnet_api, "resolve_enlarged_boundaries", lambda *_: {"min_lat": 0}
    )
    monkeypatch.setattr(roadnet_api, "graphml_covers_bbox", lambda *_: False)
    monkeypatch.setattr(
        roadnet_api,
        "_fetch_bikeable_roadnet",
        lambda **_: pytest.fail("strict no-fetch attempted a download"),
    )

    with pytest.raises(RuntimeError, match="--no-fetch"):
        roadnet_api.load_roadnet(FakeRegion(), _config(tmp_path))


def test_strict_no_fetch_rejects_a_bbox_parse_error(tmp_path, monkeypatch):
    _roadnet_path(tmp_path)
    monkeypatch.setattr(
        roadnet_api, "resolve_enlarged_boundaries", lambda *_: {"min_lat": 0}
    )

    def fail_parse(*_):
        raise ValueError("bad graphml")

    monkeypatch.setattr(roadnet_api, "graphml_covers_bbox", fail_parse)
    with pytest.raises(RuntimeError, match="Could not verify"):
        roadnet_api.load_roadnet(FakeRegion(), _config(tmp_path))


def test_non_strict_stale_roadnet_still_fetches(tmp_path, monkeypatch):
    _roadnet_path(tmp_path)
    monkeypatch.setattr(
        roadnet_api, "resolve_enlarged_boundaries", lambda *_: {"min_lat": 0}
    )
    monkeypatch.setattr(roadnet_api, "graphml_covers_bbox", lambda *_: False)

    class FetchReached(RuntimeError):
        pass

    def record_fetch(**_):
        raise FetchReached

    monkeypatch.setattr(roadnet_api, "_fetch_bikeable_roadnet", record_fetch)
    with pytest.raises(FetchReached):
        roadnet_api.load_roadnet(FakeRegion(), _config(tmp_path, strict=False))


def test_roadnet_read_refreshes_od_statistics(tmp_path, monkeypatch):
    roadnet_path = _roadnet_path(tmp_path)
    od_path = roadnet_path.with_name("testcity_od_pairs.csv")
    pd.DataFrame(
        {
            "from_lat": [1.0, 2.0],
            "from_lng": [1.0, 2.0],
            "to_lat": [1.1, 2.1],
            "to_lng": [1.1, 2.1],
            "distance_km": [1.0, 2.0],
            "annual_sales": [12, 24],
            "road_distance_m": [1200.0, 2500.0],
        }
    ).to_csv(od_path, index=False)
    monkeypatch.setattr(
        roadnet_api, "resolve_enlarged_boundaries", lambda *_: {"min_lat": 0}
    )
    monkeypatch.setattr(roadnet_api, "graphml_covers_bbox", lambda *_: True)
    monkeypatch.setattr(roadnet_api, "_load_sparse_graph", lambda *_: FakeGraph())
    monkeypatch.setattr(roadnet_api, "_make_tile_router", lambda *_: object())

    region = FakeRegion()
    roadnet_api.load_roadnet(region, _config(tmp_path))

    assert region.stats["odpair_count"] == 2
    assert region.stats["total_odpair_annual_sales"] == 36


def test_road_cleanup_removes_every_nonpositive_or_nonfinite_distance(tmp_path):
    path = tmp_path / "od.csv"
    pd.DataFrame(
        {
            "from_lat": [1.0] * 5,
            "from_lng": [1.0] * 5,
            "to_lat": [1.1] * 5,
            "to_lng": [1.1] * 5,
            "distance_km": [1.0] * 5,
            "annual_sales": [12, 24, 36, 48, 60],
            "road_distance_m": [1000.0, 0.0, -1.0, float("nan"), float("inf")],
        }
    ).to_csv(path, index=False)
    region = FakeRegion()

    audit = roadnet_api.clean_routed_od_checkpoint(
        path, region_info=region, chunk_rows=2
    )

    cleaned = pd.read_csv(path)
    assert cleaned["road_distance_m"].tolist() == [1000.0]
    assert cleaned["annual_sales"].tolist() == [12]
    assert audit["input_rows"] == 5
    assert audit["kept_rows"] == 1
    assert audit["nonfinite_rows"] == 2
    assert audit["nonpositive_rows"] == 2
    assert audit["excluded_annual_sales"] == 168
    assert region.stats["odpair_count"] == 1
    assert region.stats["total_odpair_annual_sales"] == 12
    roadnet_api.validate_clean_routed_od_checkpoint(path)


def test_failed_road_cleanup_preserves_original_checkpoint(tmp_path):
    path = tmp_path / "od.csv"
    pd.DataFrame(
        {
            "from_lat": [1.0],
            "from_lng": [1.0],
            "to_lat": [1.1],
            "to_lng": [1.1],
            "distance_km": [1.0],
            "annual_sales": [12],
            "road_distance_m": [-1.0],
        }
    ).to_csv(path, index=False)
    original = path.read_bytes()

    with pytest.raises(ValueError, match="No finite positive"):
        roadnet_api.clean_routed_od_checkpoint(path, region_info=FakeRegion())

    assert path.read_bytes() == original


def test_resume_from_roadnet_runs_only_remaining_stages(tmp_path, monkeypatch):
    calls = []
    region = FakeRegion()
    monkeypatch.setattr(main, "get_registry", lambda: {"testcity": region})
    monkeypatch.setattr(main, "load_roadnet", lambda **_: calls.append("roadnet"))
    monkeypatch.setattr(main, "cal_emission", lambda **_: calls.append("emission"))
    city_dir = tmp_path / "city" / "testcity"
    city_dir.mkdir(parents=True)
    pd.DataFrame(
        {
            "from_lat": [1.0], "from_lng": [1.0],
            "to_lat": [1.1], "to_lng": [1.1],
            "distance_km": [1.0], "annual_sales": [12],
        }
    ).to_csv(city_dir / "testcity_od_pairs.csv", index=False)

    main.run_region_from_roadnet("testcity", {"data_dir": str(tmp_path)})

    assert calls == ["roadnet", "emission"]


def test_resume_from_road_clean_skips_roadnet(tmp_path, monkeypatch):
    calls = []
    region = FakeRegion()
    monkeypatch.setattr(main, "get_registry", lambda: {"testcity": region})
    monkeypatch.setattr(
        roadnet_api,
        "clean_routed_od_checkpoint",
        lambda *_args, **_kwargs: calls.append("road-clean"),
    )
    monkeypatch.setattr(main, "cal_emission", lambda **_: calls.append("emission"))

    main.run_region_from_road_clean("testcity", {"data_dir": str(tmp_path)})

    assert calls == ["road-clean", "emission"]


def test_cli_exposes_roadnet_stage_resume():
    args = build_parser().parse_args(
        [
            "--env",
            "remote",
            "cities",
            "--region",
            "chongqing",
            "--start-at",
            "roadnet",
            "--no-fetch",
        ]
    )
    assert args.region == ["chongqing"]
    assert args.start_at == "roadnet"
    assert args.no_fetch is True


def test_cli_exposes_audited_road_clean_batch():
    args = build_parser().parse_args(
        [
            "--env",
            "remote",
            "cities",
            "--jobs",
            "4",
            "--cohort",
            "locked",
            "--run-dir",
            "/tmp/run",
            "--source-data-root",
            "/tmp/source",
            "--staging-data-root",
            "/tmp/staging",
            "--start-at",
            "road-clean",
            "--no-fetch",
        ]
    )
    assert args.region is None
    assert args.start_at == "road-clean"
