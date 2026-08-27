"""Regression tests for bounded-memory full contribution aggregation."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from drone_emission.analysis.contribution_decomposition import (
    COMPONENT_KEYS,
    _model_inputs,
    aggregate_od_contributions,
    analyze_city,
)
from drone_emission.utils import load_config
from drone_emission.visualization import contribution_break_even as contribution
from drone_emission.visualization.contribution_break_even import (
    CONTRIBUTION_COMPONENT_KEYS,
    prepare_contribution_decomposition_data,
)


def test_chunked_full_contribution_matches_in_memory_path(
    tmp_path: Path,
    monkeypatch,
) -> None:
    data_dir = tmp_path / "data"
    city_dir = data_dir / "city" / "example"
    city_dir.mkdir(parents=True)
    pd.DataFrame(
        {
            "distance_km": [0.4, 0.8, 1.5, 2.4, 3.0],
            "road_distance_m": [500.0, 1100.0, 2100.0, 3300.0, 4200.0],
            "annual_sales": [10.0, 20.0, 30.0, 40.0, 50.0],
        }
    ).to_csv(city_dir / "example_od_pairs.csv", index=False)
    city_stats = pd.DataFrame(
        {
            "region_key": ["example"],
            "region_name_en": ["Example"],
            "region_level": ["city"],
            "province": ["Example"],
            "total_ebike_lifecycle": [10_000.0],
            "gef": [0.42],
        }
    )
    config = load_config(
        str(Path(__file__).parents[1] / "experiment.toml"), env="local"
    )
    monkeypatch.setattr(contribution, "CONTRIBUTION_FULL_CHUNK_ROWS", 2)

    chunked = prepare_contribution_decomposition_data(
        city_stats,
        str(data_dir),
        exp_config=config,
        max_sample_rows=None,
    )
    in_memory = prepare_contribution_decomposition_data(
        city_stats,
        str(data_dir),
        exp_config=config,
        max_sample_rows=1_000,
    )

    assert chunked["sampling"]["mode"] == "full_real_data"
    assert chunked["national"]["drone_assigned_sampled_rows"] == in_memory[
        "national"
    ]["drone_assigned_sampled_rows"]
    for key in CONTRIBUTION_COMPONENT_KEYS:
        assert np.isclose(
            chunked["national"][f"{key}_kg"],
            in_memory["national"][f"{key}_kg"],
            rtol=0.0,
            atol=1e-12,
        )
    assert np.isclose(
        chunked["national"]["sum_check"]["aggregate_error_kg"],
        in_memory["national"]["sum_check"]["aggregate_error_kg"],
        rtol=0.0,
        atol=1e-12,
    )


def test_analysis_reducer_is_stable_across_chunk_boundaries(tmp_path: Path) -> None:
    od_path = tmp_path / "od.csv"
    pd.DataFrame(
        {
            "distance_km": [0.4, 0.8, 1.5, 2.4, 3.0],
            "road_distance_m": [500.0, 1100.0, 2100.0, 3300.0, 4200.0],
            "annual_sales": [10.0, 20.0, 30.0, 40.0, 1e9],
        }
    ).to_csv(od_path, index=False)
    config = load_config(
        str(Path(__file__).parents[1] / "experiment.toml"), env="local"
    )
    ebike, drone, ebike_scenario, drone_scenario = _model_inputs(config, 0.42)

    def reduce(chunk_rows: int) -> dict:
        return aggregate_od_contributions(
            od_path,
            ebike_params=ebike,
            drone_params=drone,
            ebike_scenario=ebike_scenario,
            drone_scenario=drone_scenario,
            chunk_rows=chunk_rows,
        )

    expected = reduce(100)
    for chunk_rows in (1, 2, 3):
        actual = reduce(chunk_rows)
        assert actual["valid_od_rows"] == expected["valid_od_rows"]
        assert actual["drone_assigned_rows"] == expected["drone_assigned_rows"]
        for key in COMPONENT_KEYS:
            assert np.isclose(
                actual["component_kg"][key],
                expected["component_kg"][key],
                rtol=1e-12,
                atol=1e-9,
            )
        assert np.isclose(
            actual["oracle_net_kg"],
            expected["oracle_net_kg"],
            rtol=1e-12,
            atol=1e-9,
        )


def test_analysis_reducer_reports_invalid_rows(tmp_path: Path) -> None:
    od_path = tmp_path / "invalid.csv"
    pd.DataFrame(
        {
            "distance_km": [1.0, "bad", 1.0, 1.0, "bad"],
            "road_distance_m": [1200.0, 1200.0, -1.0, 1200.0, "bad"],
            "annual_sales": [10.0, 10.0, 10.0, 0.0, "bad"],
        }
    ).to_csv(od_path, index=False)
    config = load_config(
        str(Path(__file__).parents[1] / "experiment.toml"), env="local"
    )
    ebike, drone, ebike_scenario, drone_scenario = _model_inputs(config, 0.42)
    result = aggregate_od_contributions(
        od_path,
        ebike_params=ebike,
        drone_params=drone,
        ebike_scenario=ebike_scenario,
        drone_scenario=drone_scenario,
        chunk_rows=2,
    )

    assert result["total_od_rows"] == 5
    assert result["valid_od_rows"] == 1
    assert result["invalid_od_rows"] == 4
    assert result["invalid_distance_km_rows"] == 2
    assert result["invalid_road_distance_rows"] == 2
    assert result["invalid_annual_sales_rows"] == 2


def test_analysis_reducer_requires_road_distance(tmp_path: Path) -> None:
    od_path = tmp_path / "missing-road.csv"
    pd.DataFrame(
        {"distance_km": [1.0], "annual_sales": [10.0]}
    ).to_csv(od_path, index=False)
    config = load_config(
        str(Path(__file__).parents[1] / "experiment.toml"), env="local"
    )
    ebike, drone, ebike_scenario, drone_scenario = _model_inputs(config, 0.42)

    with pytest.raises(ValueError, match="road_distance_m"):
        aggregate_od_contributions(
            od_path,
            ebike_params=ebike,
            drone_params=drone,
            ebike_scenario=ebike_scenario,
            drone_scenario=drone_scenario,
            chunk_rows=2,
        )


def test_city_cache_detects_same_size_content_change(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    city_dir = data_dir / "city" / "example"
    city_dir.mkdir(parents=True)
    od_path = city_dir / "example_od_pairs.csv"
    pd.DataFrame(
        {
            "distance_km": [1.0],
            "road_distance_m": [1200.0],
            "annual_sales": [10.0],
        }
    ).to_csv(od_path, index=False)
    config = load_config(
        str(Path(__file__).parents[1] / "experiment.toml"), env="local"
    )
    config = dict(config, data_dir=str(data_dir))
    city = {
        "region_key": "example",
        "region_name_en": "Example",
        "province": "Example",
        "gef": 0.42,
        "gef_source": "province_table",
        "baseline_ebike_lifecycle_kg": 10_000.0,
    }

    assert analyze_city(city, config, chunk_rows=1)["status"] == "ok"
    assert analyze_city(city, config, chunk_rows=1)["status"] == "cached"
    original = od_path.read_text(encoding="utf-8")
    changed = original.replace("1.0,", "2.0,", 1)
    assert len(changed) == len(original)
    od_path.write_text(changed, encoding="utf-8")
    assert analyze_city(city, config, chunk_rows=1)["status"] == "ok"
