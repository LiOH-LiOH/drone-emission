import json

import pandas as pd
import pytest

from drone_emission.visualization.city_gef_detour import (
    plot_city_gef_lifecycle_components,
)


def _component_frame():
    return pd.DataFrame(
        [
            ("city_a", 0.15, 100.0, 2.0, 1.0, 0.5, 1.5, 0.2, 0.1),
            ("city_b", 0.18, 300.0, 9.0, 3.0, 0.0, 10.42, 0.6, 0.2),
            ("city_c", 0.55, 200.0, 8.0, 2.0, 0.4, 10.1, 0.3, 0.1),
            ("city_d", 0.75, 400.0, 20.0, 4.0, 0.8, 25.08, 0.4, 0.2),
        ],
        columns=[
            "region_key",
            "grid_emission_factor",
            "total_odpair_annual_sales",
            "total_ebike_use_phase_emission_kg",
            "total_ebike_production_emission_kg",
            "total_ebike_end_of_life_emission_kg",
            "total_drone_use_phase_emission_kg",
            "total_drone_production_emission_kg",
            "total_drone_end_of_life_emission_kg",
        ],
    ).assign(region_level="city")


def test_city_gef_lifecycle_components_writes_weighted_report(tmp_path):
    output = tmp_path / "city_gef_lifecycle_components.pdf"

    result = plot_city_gef_lifecycle_components(
        _component_frame(),
        save_path=str(output),
        for_latex=True,
        plot_title=False,
    )

    report = json.loads(
        (tmp_path / "city_gef_lifecycle_components_report.json").read_text()
    )
    assert result == str(output)
    assert output.is_file()
    assert report["schema_version"] == 2
    assert report["figure_size_inches"] == [3.7, 4.2]
    assert report["input_city_count"] == 4
    assert report["valid_city_count"] == 4
    assert report["excluded_city_count"] == 0
    assert report["bin_width"] == 0.1
    assert [item["label_right_edge"] for item in report["populated_bins"]] == [
        "0.2",
        "0.6",
        "0.8",
    ]
    assert report["style"]["x_domain"] == [0.1, 0.8]
    assert report["style"]["bar_width_gef"] == 0.032
    assert report["style"]["total_label_rotation"] == 90
    assert report["style"]["panel_hspace"] == 0.20
    assert report["style"]["delta_height_ratio"] == 0.24
    assert report["style"]["delta_ylim"] == [-3.0, 7.0]
    assert report["style"]["delta_point_size"] == 24.0
    assert report["style"]["delta_point_edge_width"] == 0.9
    assert report["legends"] == {
        "placement": "below_chart_centered_table",
        "layout": "two_rows_three_columns",
        "center_figure_fraction_x": 0.5,
        "column_width_points": [60.0, 100.0, 65.0],
        "vehicle": ["E-bike", "Drone"],
        "component": ["Production + EOL", "Use phase"],
    }
    assert "GEF density strip" in report["omitted_elements"]
    assert "panel letter" in report["omitted_elements"]

    ebike_low = next(
        bar
        for bar in report["bars"]
        if bar["vehicle"] == "E-bike"
        and bar["bin_label_right_edge"] == "0.2"
    )
    assert ebike_low["orders_sum"] == 400.0
    assert ebike_low["production_eol_segment"][
        "height_g_co2e_per_order"
    ] == pytest.approx(11.25)
    assert ebike_low["use_phase_segment"][
        "height_g_co2e_per_order"
    ] == pytest.approx(27.5)
    assert ebike_low["total_label"]["text"] == "38.8"
    delta_values = report["delta_strip"]["values"]
    assert [item["display_label"] for item in delta_values] == [
        "+6.2",
        "-0.5",
        "-2.2",
    ]
    assert report["delta_strip"]["definition"] == (
        "E-bike total minus Drone total"
    )
    assert report["delta_strip"]["encoding"]["type"] == (
        "neutral_lollipop"
    )
    assert report["delta_strip"]["encoding"]["point_face_color"] == (
        "#FFFFFF"
    )
    assert report["delta_strip"]["encoding"]["point_edge_color"] == (
        "#222222"
    )
    assert report["delta_strip"]["encoding"]["point_edge_width"] == 0.9
    assert report["delta_strip"]["encoding"]["direction_labels"] is False


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda frame: frame.drop(
                columns=["total_drone_use_phase_emission_kg"]
            ),
            "require columns",
        ),
        (
            lambda frame: frame.assign(total_odpair_annual_sales=0.0),
            "no valid city rows",
        ),
        (
            lambda frame: frame.assign(grid_emission_factor=float("nan")),
            "require grid_emission_factor or gef",
        ),
    ],
)
def test_city_gef_lifecycle_components_rejects_invalid_inputs(
    mutate,
    message,
):
    with pytest.raises(ValueError, match=message):
        plot_city_gef_lifecycle_components(mutate(_component_frame()))
