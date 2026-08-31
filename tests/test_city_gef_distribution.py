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
    assert report["figure_size_inches"] == [3.7, 2.1]
    assert report["weighting"] == "equal_city"
    assert report["city_count"] == 8
    assert report["kde"]["bandwidth_rule"] == "scott"
    assert len(report["kde"]["curve"]) == 240
    assert report["axis"]["xlim"] == [0.1, 0.8]
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
