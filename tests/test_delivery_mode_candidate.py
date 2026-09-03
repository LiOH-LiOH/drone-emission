from pathlib import Path

import pandas as pd

from drone_emission.emission import DELIVERY_MODES
from drone_emission.visualization import delivery_modes as renderer


def test_candidate_uses_weighted_p75_and_sorts_displayed_modes(tmp_path):
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

    rows, distance_km, quantile = renderer.build_comparison(
        od_csv=od_path,
        exp_config=config,
        quantile_cache_dir=tmp_path / "cache",
    )

    assert distance_km == 3.0
    assert quantile.quantile == 0.75
    assert {mode for mode, _ in rows} == set(DELIVERY_MODES) - {"walking"}
    assert len(rows) == 7
    assert [value for _, value in rows] == sorted(value for _, value in rows)

    report = renderer.build_delivery_mode_figure_report(
        rows=rows,
        distance_km=distance_km,
        quantile=quantile,
        figure_path=tmp_path / "comparison.pdf",
        od_csv=od_path,
    )
    assert "walking" not in report["displayed_modes"]
    assert report["excluded_modes"] == [
        {"mode": "walking", "reason": "excluded from display by request"}
    ]
    assert len(report["emissions_g_per_order"]) == 7
    assert report["formal_manifest_registered"] is True
