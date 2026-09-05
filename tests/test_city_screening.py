"""Tests for the retained city reduction-potential scatter inputs."""

import json

import numpy as np
import pandas as pd
import pytest

from drone_emission.visualization.city_screening import (
    ABSOLUTE_REDUCTION_COL,
    POPULATION_COL,
    BACKGROUND_COLORS,
    _population_classes,
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
            POPULATION_COL: [30.0, 100.0, 1300.0],
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
    assert report["data"]["color_field"] == POPULATION_COL
    assert [point[POPULATION_COL] for point in report["data"]["points"]] == [
        30.0,
        100.0,
        1300.0,
    ]
    assert report["color_encoding"]["normalization"] == "discrete"
    assert report["color_encoding"]["official_city_size_classification"] is False
    assert report["color_encoding"]["thresholds_10k"] == [50, 100, 500, 1000]
    assert [p["population_class_index"] for p in report["data"]["points"]] == [0, 2, 4]
    assert report["background_regions"]["colors"] == BACKGROUND_COLORS
    assert report["background_regions"]["split_x"] == pytest.approx(3.0)
    assert report["background_regions"]["split_y"] == pytest.approx(22.5)
    assert report["colorbar"]["orientation"] == "horizontal"
    assert report["colorbar"]["label"] == "Population (million)"
    assert report["colorbar"]["tick_labels"] == ["<0.5", "0.5–1", "1–5", "5–10", "≥10"]
    assert report["colorbar"]["bbox_figure_fraction"]["y0"] == pytest.approx(0.11)
    assert report["colorbar"]["bbox_figure_fraction"]["y0"] < report["axes"][
        "bbox_figure_fraction"
    ]["y0"]
    assert report["highlighted_cities"] == []
    assert report["city_name_labels_drawn"] is False
    assert report["highlight_rings_drawn"] is False
    assert report["label_leaders_drawn"] is False


@pytest.mark.parametrize("population_value", [None, 0.0, -1.0, np.inf])
def test_scatter_rejects_invalid_population(tmp_path, population_value) -> None:
    source = pd.DataFrame(
        {
            "region_key": ["invalid-population"],
            "region_level": ["city"],
            "total_ebike_lifecycle": [2_000.0],
            "oracle_min_lifecycle_kg": [1_000.0],
            RELATIVE_REDUCTION_COL: [25.0],
            POPULATION_COL: [population_value],
        }
    )

    with pytest.raises(ValueError, match="invalid-population"):
        plot_city_reduction_potential_scatter(
            source,
            save_path=str(tmp_path / "invalid.pdf"),
            for_latex=True,
        )


def test_scatter_requires_population_column(tmp_path) -> None:
    source = pd.DataFrame(
        {
            "region_key": ["missing-population"],
            "region_level": ["city"],
            "total_ebike_lifecycle": [2_000.0],
            "oracle_min_lifecycle_kg": [1_000.0],
            RELATIVE_REDUCTION_COL: [25.0],
        }
    )

    with pytest.raises(ValueError, match=POPULATION_COL):
        plot_city_reduction_potential_scatter(
            source,
            save_path=str(tmp_path / "missing.pdf"),
            for_latex=True,
        )


def test_population_class_thresholds_are_left_closed() -> None:
    values = [1, 49.9, 50, 99.9, 100, 499.9, 500, 999.9, 1000, 10000]
    assert _population_classes(values).tolist() == [0, 0, 1, 1, 2, 2, 3, 3, 4, 4]
