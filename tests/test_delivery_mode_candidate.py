import importlib.util
from pathlib import Path

import pandas as pd

from drone_emission.emission import DELIVERY_MODES


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "archive"
    / "code"
    / "plot_delivery_mode_lifecycle_comparison.py"
)
SPEC = importlib.util.spec_from_file_location("delivery_mode_candidate", SCRIPT_PATH)
CANDIDATE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CANDIDATE)


def test_candidate_uses_weighted_p75_and_sorts_all_modes(tmp_path):
    od_path = tmp_path / "beijing_od.csv"
    pd.DataFrame(
        {
            "road_distance_m": [1000.0, 2000.0, 3000.0, 4000.0],
            "annual_sales": [1.0, 1.0, 7.0, 1.0],
        }
    ).to_csv(od_path, index=False)
    config = {
        "emission_params": {
            "payload_kg": 1.0,
            "rider_weight": 70.0,
            "ebike": {"velocity": 30.0},
            "drone": {
                "v_level": 60.0,
                "v_vertical": 20.0,
                "h_takeoff": 100.0,
                "hover_time": 10.0,
            },
        }
    }

    rows, distance_km, quantile = CANDIDATE.build_comparison(
        od_csv=od_path,
        exp_config=config,
        quantile_cache_dir=tmp_path / "cache",
    )

    assert distance_km == 3.0
    assert quantile.quantile == 0.75
    assert {mode for mode, _ in rows} == set(DELIVERY_MODES)
    assert len(rows) == 8
    assert [value for _, value in rows] == sorted(value for _, value in rows)
