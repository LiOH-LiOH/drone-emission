import pandas as pd

from drone_emission.visualization.distance_threshold import (
    _prepare_distance_threshold_all_city_data,
)


def test_road_policy_uses_full_city_baseline_for_oracle_comparison() -> None:
    summary = pd.DataFrame(
        {
            "region_key": ["example", "example"],
            "region_name_en": ["Example", "Example"],
            "distance_basis": ["road", "straight"],
            "optimal_threshold_km": [2.0, 1.5],
            "annual_reduction_kg": [10.0, 12.0],
            "annual_reduction_pct": [20.0, 24.0],
            "n_od_pairs_total": [12, 12],
            "n_od_pairs_unreachable": [2, 2],
            "unreachable_annual_orders": [10.0, 10.0],
            "total_annual_orders": [90.0, 90.0],
        }
    )
    oracle = pd.DataFrame(
        {
            "region_key": ["example"],
            "region_level": ["city"],
            "total_ebike_lifecycle": [100.0],
            "oracle_min_lifecycle_kg": [85.0],
        }
    )

    wide = _prepare_distance_threshold_all_city_data(summary, oracle)

    row = wide.iloc[0]
    assert row["road_annual_reduction_pct"] == 20.0
    assert row["road_full_domain_reduction_pct"] == 10.0
    assert row["oracle_annual_reduction_pct"] == 15.0
    assert row["road_oracle_gap_pp"] == 5.0
