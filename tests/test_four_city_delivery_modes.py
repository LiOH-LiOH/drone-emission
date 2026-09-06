import json
from pathlib import Path

import pandas as pd
import pytest

import drone_emission.analysis_workflow as analysis_workflow
from drone_emission.analysis.four_city_delivery_modes import (
    BASELINE_MODES,
    CITY_ORDER,
    PETROL_WTW_EF_KG_PER_MJ,
    calculate_city_mode_oracle,
    run_four_city_delivery_mode_analysis,
)
from drone_emission.emission import lifecycle_emission_per_order
from drone_emission.visualization.delivery_modes import (
    FOUR_CITY_LEGEND_FONTSIZE,
    FOUR_CITY_LEGEND_HEIGHT_MM,
    FOUR_CITY_LEGEND_WIDTH_MM,
    FOUR_CITY_PANEL_HEIGHT_MM,
    FOUR_CITY_PANEL_WIDTH_MM,
    load_four_city_mode_data,
    plot_four_city_mode_relative_reduction,
    plot_four_city_mode_replacement_rate,
    plot_four_city_mode_shared_legend,
)


def _write_od(path: Path, *, gef: float = 0.5, kappa: float = 0.75) -> None:
    pd.DataFrame(
        {
            "distance_km": [1.0, 2.0, 3.0],
            "road_distance_m": [1000.0, 2000.0, 3000.0],
            "annual_sales": [10.0, 30.0, 60.0],
            "drone_lifecycle_g": [5.0, 20.0, 10.0],
            "ebike_lifecycle_g": [10.0, 20.0, 5.0],
            "ebike_kappa": [kappa] * 3,
            "grid_emission_factor": [gef] * 3,
        }
    ).to_csv(path, index=False)


def _stats(region_key: str = "beijing", name: str = "Beijing") -> pd.Series:
    return pd.Series(
        {
            "region_key": region_key,
            "region_name_en": name,
            "total_ebike_lifecycle": 1.0,
            "oracle_min_lifecycle_kg": 0.95,
            "oracle_drone_order_ratio_pct": 10.0,
            "oracle_min_reduction_vs_ebike_pct": 5.0,
        }
    )


def test_oracle_is_sales_weighted_and_ties_stay_with_ground(tmp_path):
    od_path = tmp_path / "beijing_od_pairs.csv"
    _write_od(od_path)

    frame, report = calculate_city_mode_oracle(od_path, _stats(), {})

    ebike = frame.set_index("baseline_mode").loc["pedelec_ebike"]
    assert ebike["annual_orders"] == pytest.approx(100.0)
    assert ebike["drone_replaced_annual_orders"] == pytest.approx(10.0)
    assert ebike["drone_replacement_rate_pct"] == pytest.approx(10.0)
    assert ebike["baseline_lifecycle_kg"] == pytest.approx(1.0)
    assert ebike["oracle_mixed_lifecycle_kg"] == pytest.approx(0.95)
    assert ebike["absolute_reduction_kg"] == pytest.approx(0.05)
    assert ebike["relative_reduction_pct"] == pytest.approx(5.0)
    assert report["ground_distance_kappa"] == pytest.approx(0.75)


def test_common_kappa_scales_every_non_ebike_ground_mode(tmp_path):
    od_path = tmp_path / "beijing_od_pairs.csv"
    _write_od(od_path, gef=0.5, kappa=0.75)
    frame, _ = calculate_city_mode_oracle(od_path, _stats(), {})
    rows = frame.set_index("baseline_mode")
    weighted_effective_km = (1.0 * 10 + 2.0 * 30 + 3.0 * 60) * 0.75

    for mode in BASELINE_MODES[1:]:
        slope = lifecycle_emission_per_order(
            mode,
            1.0,
            exp_config={},
            grid_ef_kg_per_kwh=0.5,
            fuel_wtw_ef_kg_per_mj=PETROL_WTW_EF_KG_PER_MJ,
        )
        assert rows.loc[mode, "baseline_lifecycle_kg"] == pytest.approx(
            slope * weighted_effective_km / 1000.0
        )


def test_city_grid_factor_changes_electric_car_but_not_fuel_modes(tmp_path):
    low_path = tmp_path / "low.csv"
    high_path = tmp_path / "high.csv"
    _write_od(low_path, gef=0.2)
    _write_od(high_path, gef=0.8)
    low, _ = calculate_city_mode_oracle(low_path, _stats(), {})
    high, _ = calculate_city_mode_oracle(high_path, _stats(), {})
    low = low.set_index("baseline_mode")
    high = high.set_index("baseline_mode")

    assert high.loc[
        "battery_electric_car", "baseline_lifecycle_kg"
    ] > low.loc["battery_electric_car", "baseline_lifecycle_kg"]
    for mode in ("fuel_moped", "fuel_car"):
        assert high.loc[mode, "baseline_lifecycle_kg"] == pytest.approx(
            low.loc[mode, "baseline_lifecycle_kg"]
        )


def test_road_unreachable_rows_are_audited_and_excluded(tmp_path):
    od_path = tmp_path / "beijing_od_pairs.csv"
    _write_od(od_path)
    od = pd.read_csv(od_path)
    od.loc[2, "road_distance_m"] = -1.0
    od.to_csv(od_path, index=False)

    frame, report = calculate_city_mode_oracle(od_path, _stats(), {})

    ebike = frame.set_index("baseline_mode").loc["pedelec_ebike"]
    assert ebike["annual_orders"] == pytest.approx(40.0)
    assert report["input_od_rows"] == 3
    assert report["valid_od_rows"] == 2
    assert report["invalid_od_rows"] == 1
    assert report["excluded_positive_annual_orders"] == pytest.approx(60.0)
    comparison = report["main_experiment_comparison"]
    assert comparison["exact_match_required"] is False
    assert comparison["metrics"]["baseline_lifecycle_kg"]["matches"] is False


def test_runner_writes_sixteen_plot_ready_rows(tmp_path):
    data_dir = tmp_path / "data"
    stats_rows = []
    for city in CITY_ORDER:
        city_dir = data_dir / "city" / city
        city_dir.mkdir(parents=True)
        _write_od(city_dir / f"{city}_od_pairs.csv")
        stats_rows.append(
            {
                **_stats(city, city.title()).to_dict(),
                "region_level": "city",
            }
        )
    pd.DataFrame(stats_rows).to_csv(data_dir / "all_region_stats.csv", index=False)
    config = {"data_dir": str(data_dir), "work_dir": str(tmp_path / "work")}

    csv_path, report_path = run_four_city_delivery_mode_analysis(config)
    frame = load_four_city_mode_data(csv_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))

    assert len(frame) == 16
    assert frame["region_key"].drop_duplicates().tolist() == list(CITY_ORDER)
    assert frame.groupby("region_key")["baseline_mode"].apply(list).iloc[0] == list(
        BASELINE_MODES
    )
    assert report["city_order"] == list(CITY_ORDER)
    assert report["baseline_mode_order"] == list(BASELINE_MODES)


def test_analysis_workflow_registers_four_city_outputs(tmp_path):
    paths = analysis_workflow._module_outputs(
        {"work_dir": str(tmp_path / "work")}
    )["four-city-delivery-modes"]

    assert "four-city-delivery-modes" in analysis_workflow.ANALYSIS_MODULES
    assert [path.name for path in paths] == [
        "four_city_mode_oracle_summary.csv",
        "four_city_mode_oracle_report.json",
    ]


def test_four_city_candidate_reports_lock_panel_encodings(tmp_path):
    rows = []
    for city_index, city in enumerate(CITY_ORDER):
        for mode_index, mode in enumerate(BASELINE_MODES):
            rows.append(
                {
                    "region_key": city,
                    "region_name_en": city.title(),
                    "baseline_mode": mode,
                    "drone_replacement_rate_pct": 20 + city_index + mode_index,
                    "absolute_reduction_kt": 10 + 5 * city_index + mode_index,
                    "relative_reduction_pct": 15 + city_index + mode_index,
                }
            )
    frame = pd.DataFrame(rows)
    report = {"analysis_version": "test"}
    cases = (
        (
            plot_four_city_mode_replacement_rate,
            "replacement.pdf",
            "drone_replacement_rate_pct",
        ),
        (
            plot_four_city_mode_relative_reduction,
            "reduction.pdf",
            "relative_reduction_pct",
        ),
    )
    for plotter, filename, metric in cases:
        output = plotter(frame, report, save_path=tmp_path / filename)
        figure_report = json.loads(
            output.with_name(f"{output.stem}_report.json").read_text()
        )
        assert figure_report["figure_size_mm"] == {
            "width": FOUR_CITY_PANEL_WIDTH_MM,
            "height": FOUR_CITY_PANEL_HEIGHT_MM,
        }
        encoding = figure_report["visual_encoding"]
        assert encoding["orientation"] == "vertical"
        assert encoding["x_grouping"] == "city"
        assert encoding["within_group_order"] == "baseline_mode"
        assert encoding["city_label_rotation_deg"] == 0
        assert encoding["legend_location"] == "separate_shared_asset"
        assert encoding["font_size_pt"] == {
            "axis_label": 8.0,
            "tick_and_city": 7.0,
            "bar_value": 6.3,
        }
        assert encoding["axes_bounds_fraction"]["bottom"] == 0.21
        expected_secondary = (
            "absolute_reduction_kt" if metric == "relative_reduction_pct" else None
        )
        assert encoding["relative_reduction_secondary_axis"] == expected_secondary
        if metric == "relative_reduction_pct":
            marker = encoding["absolute_reduction_marker"]
            assert marker["shape"] == "filled_circle"
            assert marker["connected"] is False
            assert marker["labels"] is False
            assert (
                encoding["bar_label_collision_avoidance"]
                == "raise_above_secondary_axis_marker"
            )


def test_four_city_shared_legend_report_locks_single_row_layout(tmp_path):
    frame = pd.DataFrame()
    output = plot_four_city_mode_shared_legend(
        frame,
        {"analysis_version": "test"},
        save_path=tmp_path / "legend.pdf",
    )
    report = json.loads(
        output.with_name(f"{output.stem}_report.json").read_text()
    )
    assert report["figure_size_mm"] == {
        "width": FOUR_CITY_LEGEND_WIDTH_MM,
        "height": FOUR_CITY_LEGEND_HEIGHT_MM,
    }
    assert report["visual_encoding"]["layout"] == "single_row"
    assert report["visual_encoding"]["font_size_pt"] == FOUR_CITY_LEGEND_FONTSIZE
    assert report["visual_encoding"]["items"][-1] == "Absolute reduction"
    assert len(report["visual_encoding"]["items"]) == 5
