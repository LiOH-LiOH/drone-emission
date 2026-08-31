"""Tests for the retained city reduction-potential scatter inputs."""

import json

import numpy as np
import pandas as pd

from drone_emission.analysis.constants import ORACLE_REDUCTION_LABEL_CITIES
from drone_emission.visualization.city_screening import (
    ABSOLUTE_REDUCTION_COL,
    RELATIVE_REDUCTION_COL,
    _reduction_metrics,
    plot_city_reduction_potential_scatter,
)
from drone_emission.visualization.city_priorities import (
    ORACLE_REDUCTION_LABEL_CITIES as MAP_LABEL_CITIES,
)


def test_scatter_and_map_share_highlighted_cities() -> None:
    assert ORACLE_REDUCTION_LABEL_CITIES == MAP_LABEL_CITIES
    assert ORACLE_REDUCTION_LABEL_CITIES == (
        "Beijing",
        "Shanghai",
        "Guangzhou",
        "Hangzhou",
        "Shenzhen",
        "Wuhan",
        "Chongqing",
        "Chengdu",
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
    city_count = len(ORACLE_REDUCTION_LABEL_CITIES)
    source = pd.DataFrame(
        {
            "region_key": [name.casefold() for name in ORACLE_REDUCTION_LABEL_CITIES],
            "region_name_en": ORACLE_REDUCTION_LABEL_CITIES,
            "region_level": ["city"] * city_count,
            "total_ebike_lifecycle": np.linspace(2_000.0, 9_000.0, city_count),
            "oracle_min_lifecycle_kg": np.linspace(1_000.0, 4_000.0, city_count),
            RELATIVE_REDUCTION_COL: np.linspace(5.0, 40.0, city_count),
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
    assert report["highlighted_cities"] == list(ORACLE_REDUCTION_LABEL_CITIES)
    assert report["label_leaders_drawn"] is False
