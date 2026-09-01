import json

import pandas as pd
import pytest

from drone_emission.visualization.city_gef_detour import (
    plot_city_gef_distribution,
)


def _distribution_frame():
    return pd.DataFrame(
        [
            ("chengdu", 0.15, "province_table"),
            ("guangzhou", 0.36, "province_table"),
            ("beijing", 0.52, "province_table"),
            ("shanghai", 0.66, "province_table"),
            ("city_e", 0.24, "province_table"),
            ("city_f", 0.44, "province_table"),
            ("city_g", 0.58, "fallback_0.5366"),
            ("city_h", 0.74, "province_table"),
        ],
        columns=["region_key", "grid_emission_factor", "gef_source"],
    ).assign(region_level="city")


def test_city_gef_distribution_writes_equal_city_kde_report(tmp_path):
    output = tmp_path / "city_gef_distribution.pdf"

    result = plot_city_gef_distribution(
        _distribution_frame(),
        save_path=str(output),
        for_latex=True,
        plot_title=False,
    )

    report = json.loads(
        (tmp_path / "city_gef_distribution_report.json").read_text()
    )
    assert result == str(output)
    assert output.is_file()
    assert report["schema_version"] == 2
    assert report["figure_size_inches"] == [3.7, 3.8]
    assert report["weighting"] == "equal_city"
    assert report["city_count"] == 8
    assert report["kde"]["bandwidth_rule"] == "scott"
    assert len(report["kde"]["curve"]) == 240
    assert report["axis"]["xlim"] == [0.1, 0.8]
    assert len(report["axis"]["yticks"]) == 3
    assert report["axis"]["yticks"][0] == 0.0
    assert report["axis"]["yticks"][-1] <= report["axis"]["axes"][
        "ylim_data"
    ][1]
    assert report["axis"]["height_inches"] == pytest.approx(
        1.42, abs=0.03
    )
    assert [item["region_key"] for item in report["selected_city_markers"]] == [
        "chengdu",
        "guangzhou",
        "beijing",
        "shanghai",
    ]
    assert [item["marker"] for item in report["selected_city_markers"]] == [
        "o",
        "^",
        "s",
        "D",
    ]
    assert report["layout"] == {
        "orientation": "vertical",
        "density_panel_position": "top",
        "power_mix_panel_position": "bottom",
        "panel_gap": "whitespace",
        "embedded_panel_letters": False,
        "density_height_ratio": 0.828,
        "power_mix_height_ratio": 1.0,
        "panel_hspace": 0.218,
        "target_density_height_inches": 1.42,
    }
    power_mix = report["power_mix"]
    assert power_mix["city_order"] == [
        "chengdu",
        "guangzhou",
        "beijing",
        "shanghai",
    ]
    assert power_mix["percentage_label_threshold_pct"] == 12.0
    assert power_mix["axis_visible"] is False
    assert power_mix["style"] == {
        "row_hspace": 0.12,
        "column_wspace": 0.03,
        "label_bar_width_ratios": [0.23, 0.77],
        "bar_height": 0.78,
        "city_label_fontsize": 8.5,
        "percentage_label_fontsize": 8.0,
        "marker_size": 28.0,
        "marker_linewidth": 1.0,
        "legend_title_fontsize": 8.0,
        "legend_fontsize": 7.5,
    }
    assert [row["power_region_key"] for row in power_mix["rows"]] == [
        "sichuan",
        "guangdong",
        "beijing",
        "shanghai",
    ]
    assert [row["marker"] for row in power_mix["rows"]] == [
        "o",
        "^",
        "s",
        "D",
    ]
    assert [
        item["component"] for item in power_mix["legend"]["components"]
    ] == ["Hydro", "Thermal", "Wind", "Solar", "Other"]
    assert power_mix["legend"]["title"] == "Power mix"
    first_bar_axis = power_mix["rows"][0]["axes"]["bar"]
    assert first_bar_axis["bbox_figure_fraction"]["height"] * 3.8 == pytest.approx(
        0.34, abs=0.02
    )
    legend_axis = power_mix["legend"]["axis"]
    assert legend_axis["bbox_figure_fraction"]["height"] * 3.8 == pytest.approx(
        0.20, abs=0.02
    )
    for row in power_mix["rows"]:
        for component in row["components"]:
            expected = (
                f'{component["share_pct"]:.0f}%'
                if component["share_pct"] >= 12.0
                else None
            )
            assert component["percentage_label"] == expected
    assert len(report["observations"]) == 8


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda frame: frame.assign(
                grid_emission_factor=frame["grid_emission_factor"].mask(
                    frame["region_key"].eq("city_h")
                )
            ),
            "missing or non-finite",
        ),
        (
            lambda frame: pd.concat([frame, frame.iloc[[0]]], ignore_index=True),
            "duplicate region keys",
        ),
        (
            lambda frame: frame.assign(
                grid_emission_factor=frame["grid_emission_factor"].mask(
                    frame["region_key"].eq("city_h"), 0.81
                )
            ),
            "outside the fixed 0.1-0.8 domain",
        ),
        (
            lambda frame: frame[frame["region_key"].ne("shanghai")],
            "missing selected city",
        ),
    ],
)
def test_city_gef_distribution_rejects_invalid_inputs(mutate, message):
    with pytest.raises(ValueError, match=message):
        plot_city_gef_distribution(mutate(_distribution_frame()))
