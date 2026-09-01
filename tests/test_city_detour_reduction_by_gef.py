import json

import pandas as pd
import pytest

from drone_emission.visualization.city_gef_detour import (
    plot_city_detour_reduction_by_gef,
)


def _contexts():
    return [
        {"region_key": "chengdu", "display_name": "Chengdu"},
        {"region_key": "beijing", "display_name": "Beijing"},
        {"region_key": "shanghai", "display_name": "Shanghai"},
        {"region_key": "guangzhou", "display_name": "Guangzhou"},
    ]


def _city_frame():
    return pd.DataFrame(
        {
            "region_key": [
                "chengdu",
                "low_two",
                "low_three",
                "guangzhou",
                "beijing",
                "shanghai",
                "high_four",
            ],
            "region_name_en": [
                "Chengdu",
                "Low Two",
                "Low Three",
                "Guangzhou",
                "Beijing",
                "Shanghai",
                "High Four",
            ],
            "detour_ratio": [1.35, 1.55, 1.75, 1.45, 1.65, 1.85, 2.05],
            "oracle_min_reduction_vs_ebike_pct": [
                25.0,
                31.0,
                38.0,
                8.0,
                14.0,
                21.0,
                27.0,
            ],
            "grid_emission_factor": [
                0.18,
                0.22,
                0.28,
                0.42,
                0.52,
                0.62,
                0.72,
            ],
        }
    )


def test_city_detour_reduction_by_gef_writes_centered_colorbar_report(
    tmp_path,
):
    output = tmp_path / "city_detour_reduction_by_gef.pdf"

    result = plot_city_detour_reduction_by_gef(
        _city_frame(),
        contexts=_contexts(),
        save_path=str(output),
        for_latex=True,
        plot_title=False,
    )

    report = json.loads(
        (tmp_path / "city_detour_reduction_by_gef_report.json").read_text()
    )
    assert result == str(output)
    assert output.is_file()
    assert report["schema_version"] == 1
    assert report["figure_size_inches"] == [7.8, 2.6]
    assert report["archetype"] == "quantitative_grid"
    assert report["data_contract"]["valid_city_count"] == 7
    assert [panel["sample_size"] for panel in report["panels"]] == [3, 4]
    assert all(panel["panel_label"]["text"] is None for panel in report["panels"])
    assert all(panel["regression"] is not None for panel in report["panels"])

    low_bbox = report["panels"][0]["axes"]["bbox_figure_fraction"]
    colorbar_bbox = report["colorbar"]["axes"]["bbox_figure_fraction"]
    high_bbox = report["panels"][1]["axes"]["bbox_figure_fraction"]
    assert low_bbox["x1"] < colorbar_bbox["x0"]
    assert colorbar_bbox["x1"] < high_bbox["x0"]
    assert report["colorbar"]["height_fraction_of_panel"] == pytest.approx(
        0.72
    )
    assert report["colorbar"]["boundaries"] == pytest.approx(
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    )
    assert report["style"]["ordinary_scatter_size_pt2"] == 20.0
    assert report["style"]["ordinary_scatter_alpha"] == 0.45
    assert report["style"]["selected_city_marker_size_pt2"] == 34.0
    assert len(report["selected_city_markers"]) == 4
    assert "selected-city legend" in report["omitted_elements"]


def test_city_detour_reduction_by_gef_rejects_incomplete_evidence():
    frame = _city_frame()
    frame.loc[frame["grid_emission_factor"] <= 0.3, "detour_ratio"] = 1.5
    with pytest.raises(ValueError, match="two distinct detour ratios"):
        plot_city_detour_reduction_by_gef(frame, contexts=_contexts())

    missing_city = _city_frame().query("region_key != 'shanghai'")
    with pytest.raises(ValueError, match="missing valid selected cities"):
        plot_city_detour_reduction_by_gef(
            missing_city,
            contexts=_contexts(),
        )
