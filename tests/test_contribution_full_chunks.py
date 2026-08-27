"""Regression tests for bounded-memory full contribution aggregation."""

from pathlib import Path

import numpy as np
import pandas as pd

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
