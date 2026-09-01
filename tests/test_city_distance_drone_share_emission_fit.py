import json

import numpy as np
import pytest

from drone_emission.visualization.city_gef_detour import (
    plot_city_distance_drone_share_emission_fit,
)


def _contexts():
    return [
        {"region_key": "chengdu", "display_name": "Chengdu", "gef": 0.15},
        {
            "region_key": "guangzhou",
            "display_name": "Guangzhou",
            "gef": 0.45,
        },
        {"region_key": "beijing", "display_name": "Beijing", "gef": 0.55},
        {
            "region_key": "shanghai",
            "display_name": "Shanghai",
            "gef": 0.65,
        },
    ]


def _binned_by_city():
    centres = np.array([0.1, 0.3, 1.0, 2.0, 3.3, 3.7])
    records = {}
    for index, context in enumerate(_contexts()):
        slope = 4.0 + index
        intercept = 1.0 - index
        records[context["region_key"]] = {
            "centres": centres,
            "reduction_abs": intercept + slope * centres,
            "oracle_drone_share_pct": np.array(
                [5.0, 10.0, 35.0, 60.0, 90.0, 98.0]
            ),
            "max_km": 4.0,
            "valid_path_count": 100 + index,
            "invalid_path_count": index,
            "merchant_filter": {
                "rule": "exclude merchants with road-snap distance > P99",
                "excluded_merchant_count": index,
            },
        }
    return records


def test_city_distance_response_writes_shared_scale_report(tmp_path):
    output = tmp_path / "city_distance_drone_share_emission_fit.pdf"

    result = plot_city_distance_drone_share_emission_fit(
        _binned_by_city(),
        _contexts(),
        save_path=str(output),
        for_latex=True,
        plot_title=False,
    )

    report = json.loads(
        (
            tmp_path
            / "city_distance_drone_share_emission_fit_report.json"
        ).read_text()
    )
    assert result == str(output)
    assert output.is_file()
    assert report["schema_version"] == 1
    assert report["figure_size_inches"] == [7.8, 2.4]
    assert report["archetype"] == "quantitative_grid"
    assert report["city_order"] == [
        "chengdu",
        "guangzhou",
        "beijing",
        "shanghai",
    ]
    assert report["data_contract"]["bin_width_km"] == 0.2
    assert report["data_contract"]["distance_domain_km"] == [0.2, 3.5]
    assert report["data_contract"]["fit_weighting"] == (
        "equal weight per visible distance bin"
    )
    assert report["style"]["scatter_size"] == 14.0
    assert report["style"]["fit_linewidth"] == 1.0
    assert report["style"]["bar_width_km"] == 0.156
    assert "power mix bars" in report["omitted_elements"]
    assert "panel letters" in report["omitted_elements"]
    assert "legend" in report["omitted_elements"]

    shared_ylim = report["shared_absolute_ylim"]
    for city in report["cities"]:
        assert len(city["bars"]) == 4
        assert len(city["scatter_points"]) == 4
        assert city["axes"]["ebike_minus_drone"]["ylim_data"] == (
            shared_ylim
        )
        assert [
            point["x_air_km"]
            for point in city["linear_fit"]["line_points"]
        ] == [0.3, 3.3]

    chengdu = report["cities"][0]
    assert chengdu["linear_fit"][
        "slope_g_co2e_per_order_per_km"
    ] == pytest.approx(4.0)
    assert chengdu["linear_fit"][
        "intercept_g_co2e_per_order"
    ] == pytest.approx(1.0)


def test_city_distance_response_requires_four_complete_cities():
    contexts = _contexts()
    with pytest.raises(ValueError, match="exactly four cities"):
        plot_city_distance_drone_share_emission_fit(
            _binned_by_city(), contexts[:3]
        )

    incomplete = _binned_by_city()
    incomplete.pop("shanghai")
    with pytest.raises(ValueError, match="missing binned cities: shanghai"):
        plot_city_distance_drone_share_emission_fit(incomplete, contexts)


def test_city_distance_response_requires_two_visible_fit_bins():
    binned = _binned_by_city()
    binned["chengdu"]["reduction_abs"] = np.array(
        [1.0, np.nan, np.nan, np.nan, 2.0, np.nan]
    )
    with pytest.raises(ValueError, match="at least two visible distance bins"):
        plot_city_distance_drone_share_emission_fit(binned, _contexts())
