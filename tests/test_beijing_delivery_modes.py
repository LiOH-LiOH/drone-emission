import json
from pathlib import Path

import pandas as pd
import pytest

import drone_emission.analysis_workflow as analysis_workflow
from drone_emission.analysis.beijing_delivery_modes import (
    BEIJING_GRID_EF_KG_PER_KWH,
    PETROL_WTW_EF_KG_PER_MJ,
    calculate_beijing_delivery_mode_totals,
    run_beijing_delivery_mode_analysis,
)
from drone_emission.emission import DELIVERY_MODES, lifecycle_emission_per_order
from drone_emission.visualization import delivery_modes as renderer


@pytest.fixture
def emission_config():
    return {
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


def _write_od(path: Path) -> None:
    pd.DataFrame(
        {
            "distance_km": [1.0, 2.0, 9.0],
            "road_distance_m": [1500.0, 3000.0, float("nan")],
            "annual_sales": [10.0, 30.0, 5.0],
        }
    ).to_csv(path, index=False)


def test_totals_match_direct_weighted_model_for_every_mode(
    tmp_path, emission_config
):
    od_path = tmp_path / "beijing_od_pairs.csv"
    _write_od(od_path)

    frame, report = calculate_beijing_delivery_mode_totals(
        od_path,
        emission_config,
        chunk_rows=1,
    )

    assert list(frame["mode"]) == list(DELIVERY_MODES)
    assert set(frame["annual_orders"]) == {40.0}
    assert report["od_filter"]["valid_rows"] == 2
    assert report["od_filter"]["excluded_positive_annual_orders"] == 5.0
    assert report["input_signature"]["source"] == str(od_path)
    assert report["input_signature"]["size_bytes"] == od_path.stat().st_size
    assert report["input_signature"]["mtime_ns"] == od_path.stat().st_mtime_ns
    for row in frame.itertuples(index=False):
        distances = [1.0, 2.0] if row.mode == "drone" else [1.5, 3.0]
        expected_g = sum(
            lifecycle_emission_per_order(
                row.mode,
                distance,
                exp_config=emission_config,
                grid_ef_kg_per_kwh=BEIJING_GRID_EF_KG_PER_KWH,
                fuel_wtw_ef_kg_per_mj=PETROL_WTW_EF_KG_PER_MJ,
            )
            * weight
            for distance, weight in zip(distances, [10.0, 30.0])
        )
        assert row.annual_lifecycle_kg == pytest.approx(expected_g / 1000.0)
        assert row.annual_lifecycle_kt == pytest.approx(expected_g / 1e9)


def test_chunk_size_does_not_change_analysis(tmp_path, emission_config):
    od_path = tmp_path / "beijing_od_pairs.csv"
    _write_od(od_path)
    one, _ = calculate_beijing_delivery_mode_totals(
        od_path, emission_config, chunk_rows=1
    )
    all_rows, _ = calculate_beijing_delivery_mode_totals(
        od_path, emission_config, chunk_rows=10
    )
    pd.testing.assert_frame_equal(one, all_rows)


def test_runner_persists_plot_ready_data_and_figure_report(
    tmp_path, emission_config
):
    data_dir = tmp_path / "data"
    city_dir = data_dir / "city" / "beijing"
    city_dir.mkdir(parents=True)
    _write_od(city_dir / "beijing_od_pairs.csv")
    config = {
        **emission_config,
        "data_dir": str(data_dir),
        "work_dir": str(tmp_path / "work"),
    }

    csv_path, report_path = run_beijing_delivery_mode_analysis(config)
    assert csv_path.is_file()
    assert report_path.is_file()
    assert "input_signature" in report_path.read_text(encoding="utf-8")
    mode_order = [
        "pedal_bicycle",
        "pedelec_ebike",
        "drone",
        "fuel_moped",
        "battery_electric_car",
        "fuel_car",
    ]
    plot_frame = renderer.load_beijing_mode_data(
        csv_path,
        mode_order=mode_order,
    )
    assert len(plot_frame) == len(DELIVERY_MODES) - 2
    assert {"walking", "electric_moped"}.isdisjoint(plot_frame["mode"])
    assert plot_frame["mode"].tolist() == mode_order
    figure_report = renderer.build_beijing_mode_figure_report(
        plot_frame,
        json.loads(report_path.read_text(encoding="utf-8")),
        figure_path=tmp_path / "beijing.pdf",
    )
    assert {"walking", "electric_moped"}.isdisjoint(
        figure_report["displayed_modes"]
    )
    assert {"walking", "electric_moped"}.isdisjoint(
        figure_report["analysis"]["coefficients"]
    )
    assert not any(
        key.startswith("walking_")
        for key in figure_report["analysis"]["boundary"]
    )
    assert figure_report["figure_size_mm"] == {"width": 88.0, "height": 80.0}
    assert len(figure_report["plotted_values"]) == len(DELIVERY_MODES) - 2
    assert figure_report["formal_manifest_registered"] is True


def test_analysis_workflow_registers_both_retained_outputs(tmp_path):
    config = {"work_dir": str(tmp_path / "work")}
    paths = analysis_workflow._module_outputs(config)["beijing-delivery-modes"]

    assert "beijing-delivery-modes" in analysis_workflow.ANALYSIS_MODULES
    assert [path.name for path in paths] == [
        "beijing_all_mode_lifecycle.csv",
        "beijing_all_mode_lifecycle_report.json",
    ]
