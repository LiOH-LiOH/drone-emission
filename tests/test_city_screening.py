"""Tests for the retained city reduction-potential scatter inputs."""

import numpy as np
import pandas as pd

from drone_emission.visualization.city_screening import (
    ABSOLUTE_REDUCTION_COL,
    RELATIVE_REDUCTION_COL,
    _reduction_metrics,
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
