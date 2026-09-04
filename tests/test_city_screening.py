"""Tests for the retained city reduction-potential scatter inputs."""

import json

import numpy as np
import pandas as pd
import pytest

from drone_emission.visualization.city_screening import (
    ABSOLUTE_REDUCTION_COL,
    GDP_PER_CAPITA_COL,
    RELATIVE_REDUCTION_COL,
    _reduction_metrics,
    plot_city_reduction_potential_scatter,
)


def test_reduction_metrics_derive_values_and_filter_nonpositive_cities() -> None:
    source = pd.DataFrame(
        {
            "region_key": ["positive", "negative", "province"],
            "region_level": ["city", "city", "province"],
            "total_ebike_lifecycle": [2_000.0, 1_000.0, 9_000.0],
            "oracle_min_lifecycle_kg": [1_250.0, 1_100.0, 1_000.0],
            RELATIVE_REDUCTION_COL: [37.5, -10.0, 88.0],
        }
    )

    result = _reduction_metrics(source)

    assert result["region_key"].tolist() == ["positive"]
    assert np.isclose(result.iloc[0][ABSOLUTE_REDUCTION_COL], 0.75)
    assert np.isclose(result.iloc[0][RELATIVE_REDUCTION_COL], 37.5)


def test_reduction_metrics_require_lifecycle_inputs() -> None:
    try:
        _reduction_metrics(pd.DataFrame({RELATIVE_REDUCTION_COL: [1.0]}))
    except ValueError as exc:
        assert "oracle_min_lifecycle_kg" in str(exc)
        assert "total_ebike_lifecycle" in str(exc)
    else:
        raise AssertionError("missing lifecycle inputs were accepted")


def test_scatter_writes_exact_size_and_traceable_report(tmp_path) -> None:
    city_names = ("Alpha", "Beta", "Gamma")
    city_count = len(city_names)
    source = pd.DataFrame(
        {
            "region_key": [name.casefold() for name in city_names],
            "region_name_en": city_names,
            "region_level": ["city"] * city_count,
            "total_ebike_lifecycle": np.linspace(2_000.0, 9_000.0, city_count),
            "oracle_min_lifecycle_kg": np.linspace(1_000.0, 4_000.0, city_count),
            RELATIVE_REDUCTION_COL: np.linspace(5.0, 40.0, city_count),
            GDP_PER_CAPITA_COL: [30_000.0, 100_000.0, 300_000.0],
        }
    )
    output = tmp_path / "city_reduction_potential_scatter.pdf"

    result = plot_city_reduction_potential_scatter(
        source,
        save_path=str(output),
        for_latex=True,
        plot_title=False,
    )

    assert result == str(output)
    assert output.is_file()
    report = json.loads(
        output.with_name("city_reduction_potential_scatter_report.json").read_text()
    )
    assert report["figure_size_inches"] == {"width": 1.965, "height": 3.43}
    assert report["data"]["plotted_city_count"] == city_count
    assert report["axes"]["x_scale"] == "log"
    assert report["axes"]["y_limits"] == [0.0, 50.0]
    assert report["data"]["color_field"] == GDP_PER_CAPITA_COL
    assert [point[GDP_PER_CAPITA_COL] for point in report["data"]["points"]] == [
        30_000.0,
        100_000.0,
        300_000.0,
    ]
    assert report["color_encoding"] == {
        "field": GDP_PER_CAPITA_COL,
        "colormap": "viridis",
        "normalization": "log",
        "range_cny_per_person": [30_000.0, 300_000.0],
        "point_alpha": 0.68,
    }
    assert report["colorbar"]["orientation"] == "horizontal"
    assert report["colorbar"]["label"] == "GDP per capita (10^4 CNY person-1)"
    assert report["colorbar"]["display_scale_cny_per_unit"] == 10_000.0
    assert report["colorbar"]["ticks_cny_per_person"] == [
        50_000.0,
        100_000.0,
        200_000.0,
    ]
    assert report["colorbar"]["tick_labels"] == ["5", "10", "20"]
    assert report["colorbar"]["bbox_figure_fraction"]["y0"] == pytest.approx(0.11)
    assert report["colorbar"]["bbox_figure_fraction"]["y0"] < report["axes"][
        "bbox_figure_fraction"
    ]["y0"]
    assert report["highlighted_cities"] == []
    assert report["city_name_labels_drawn"] is False
    assert report["highlight_rings_drawn"] is False
    assert report["label_leaders_drawn"] is False


@pytest.mark.parametrize("gdp_value", [None, 0.0, -1.0, np.inf])
def test_scatter_rejects_invalid_gdp(tmp_path, gdp_value) -> None:
    source = pd.DataFrame(
        {
            "region_key": ["invalid-gdp"],
            "region_level": ["city"],
            "total_ebike_lifecycle": [2_000.0],
            "oracle_min_lifecycle_kg": [1_000.0],
            RELATIVE_REDUCTION_COL: [25.0],
            GDP_PER_CAPITA_COL: [gdp_value],
        }
    )

    with pytest.raises(ValueError, match="invalid-gdp"):
        plot_city_reduction_potential_scatter(
            source,
            save_path=str(tmp_path / "invalid.pdf"),
            for_latex=True,
        )


def test_scatter_requires_gdp_column(tmp_path) -> None:
    source = pd.DataFrame(
        {
            "region_key": ["missing-gdp"],
            "region_level": ["city"],
            "total_ebike_lifecycle": [2_000.0],
            "oracle_min_lifecycle_kg": [1_000.0],
            RELATIVE_REDUCTION_COL: [25.0],
        }
    )

    with pytest.raises(ValueError, match=GDP_PER_CAPITA_COL):
        plot_city_reduction_potential_scatter(
            source,
            save_path=str(tmp_path / "missing.pdf"),
            for_latex=True,
        )
