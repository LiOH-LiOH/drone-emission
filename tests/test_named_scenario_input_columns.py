"""Regression tests for bounded named-scenario OD input loading."""

from __future__ import annotations

import pandas as pd

from drone_emission.tools import rerun_named_scenarios


class _Region:
    def to_dict(self):
        return {
            "region_key": "test",
            "region_name_en": "Test",
            "region_level": "city",
        }

    def get_stat(self, _key):
        return "Guangdong"


def test_process_reads_only_named_scenario_columns(tmp_path, monkeypatch):
    city_dir = tmp_path / "city" / "test"
    city_dir.mkdir(parents=True)
    pd.DataFrame(
        {
            "road_distance_m": [1000.0],
            "distance_km": [0.8],
            "annual_sales": [12.0],
            "large_unused_column": ["unused"],
        }
    ).to_csv(city_dir / "test_od_pairs.csv", index=False)

    observed = {}

    def fake_scenarios(frame, **_kwargs):
        observed["columns"] = frame.columns.tolist()
        return pd.DataFrame({"strategy": ["all_ebike"]})

    monkeypatch.setattr(
        rerun_named_scenarios, "run_named_scenarios_for_region", fake_scenarios
    )
    monkeypatch.setattr(rerun_named_scenarios, "fetch_factor", lambda *_args: 0.5)
    monkeypatch.setattr(
        rerun_named_scenarios, "create_ebike_params", lambda _config: object()
    )
    monkeypatch.setattr(
        rerun_named_scenarios, "create_drone_params", lambda _config: object()
    )
    monkeypatch.setattr(
        rerun_named_scenarios,
        "get_emission_params",
        lambda _config: {"rider_weight": 70.0, "payload_kg": 1.0},
    )

    key, status, _elapsed = rerun_named_scenarios._process(
        _Region(),
        {"data_dir": str(tmp_path), "gef_path": str(tmp_path / "gef.csv")},
    )

    assert (key, status) == ("test", "ok")
    assert observed["columns"] == [
        "road_distance_m",
        "distance_km",
        "annual_sales",
    ]


def test_process_preserves_legacy_optional_annual_sales(tmp_path, monkeypatch):
    city_dir = tmp_path / "city" / "test"
    city_dir.mkdir(parents=True)
    pd.DataFrame(
        {
            "road_distance_m": [1000.0],
            "distance_km": [0.8],
            "unused": [1],
        }
    ).to_csv(city_dir / "test_od_pairs.csv", index=False)

    observed = {}

    def fake_scenarios(frame, **_kwargs):
        observed["columns"] = frame.columns.tolist()
        return pd.DataFrame({"strategy": ["all_ebike"]})

    monkeypatch.setattr(
        rerun_named_scenarios, "run_named_scenarios_for_region", fake_scenarios
    )
    monkeypatch.setattr(rerun_named_scenarios, "fetch_factor", lambda *_args: 0.5)
    monkeypatch.setattr(
        rerun_named_scenarios, "create_ebike_params", lambda _config: object()
    )
    monkeypatch.setattr(
        rerun_named_scenarios, "create_drone_params", lambda _config: object()
    )
    monkeypatch.setattr(
        rerun_named_scenarios,
        "get_emission_params",
        lambda _config: {"rider_weight": 70.0, "payload_kg": 1.0},
    )

    _key, status, _elapsed = rerun_named_scenarios._process(
        _Region(),
        {"data_dir": str(tmp_path), "gef_path": str(tmp_path / "gef.csv")},
    )

    assert status == "ok"
    assert observed["columns"] == ["road_distance_m", "distance_km"]
