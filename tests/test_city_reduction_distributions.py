"""Regression tests for map-city highlights in reduction distributions."""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd


os.environ.setdefault("MPLBACKEND", "Agg")
HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.normpath(os.path.join(HERE, "..", "src", "drone_emission"))
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from visualization.analysis import (  # noqa: E402
    ORACLE_REDUCTION_LABEL_CITIES,
    _city_reduction_highlight_offsets,
    plot_city_annual_absolute_reduction_distribution,
    plot_city_relative_lifecycle_reduction_distribution,
)


def _city_frame() -> pd.DataFrame:
    cities = [*ORACLE_REDUCTION_LABEL_CITIES, "Foshan", "Nanjing"]
    relative_pct = np.array(
        [13.2, 10.7, 20.1, 13.0, 14.1, 19.2, 23.9, 35.0, 7.0, 16.0]
    )
    absolute_t = np.array(
        [3274.0, 2136.0, 2519.0, 1318.0, 2223.0, 727.0, 1576.0, 2012.0, 40.0, 100.0]
    )
    reduction_kg = absolute_t * 1e3
    ebike_kg = reduction_kg / (relative_pct / 100.0)
    return pd.DataFrame(
        {
            "region_level": "city",
            "region_name_en": cities,
            "region_key": [city.casefold() for city in cities],
            "total_ebike_lifecycle": ebike_kg,
            "oracle_min_lifecycle_kg": ebike_kg - reduction_kg,
        }
    )


def _load_report(figure_path: str) -> dict:
    report_path = figure_path.rsplit(".", 1)[0] + "_report.json"
    with open(report_path, encoding="utf-8") as report_file:
        return json.load(report_file)


def _assert_highlight_report(report: dict) -> None:
    highlights = report["highlighted_cities"]
    assert highlights["requested_cities"] == list(ORACLE_REDUCTION_LABEL_CITIES)
    assert highlights["matched_count"] == 8
    assert highlights["missing_cities"] == []
    assert [record["city"] for record in highlights["records"]] == list(
        ORACLE_REDUCTION_LABEL_CITIES
    )
    assert report["sample_audit"]["eligible_city_count"] == 10
    assert report["summary"]["count"] == 10


class CityReductionDistributionTest(unittest.TestCase):
    def test_both_distributions_report_the_same_eight_map_cities(self) -> None:
        city_frame = _city_frame()
        with tempfile.TemporaryDirectory() as tmp_dir:
            relative_path = plot_city_relative_lifecycle_reduction_distribution(
                city_frame,
                save_path=os.path.join(tmp_dir, "relative.png"),
                for_latex=True,
            )
            absolute_path = plot_city_annual_absolute_reduction_distribution(
                city_frame,
                save_path=os.path.join(tmp_dir, "absolute.png"),
                for_latex=True,
            )

            relative_report = _load_report(relative_path)
            absolute_report = _load_report(absolute_path)
            _assert_highlight_report(relative_report)
            _assert_highlight_report(absolute_report)
            self.assertEqual(
                relative_report["highlighted_cities"]["unit"],
                "percent",
            )
            self.assertEqual(
                absolute_report["highlighted_cities"]["unit"],
                "t CO2e per year",
            )

    def test_highlight_offsets_are_deterministic_and_inside_violin(self) -> None:
        values = np.array([10.0, 10.2, 10.4, 20.0])
        grid = np.linspace(0.0, 30.0, 101)
        half_width = np.full_like(grid, 0.3)

        first = _city_reduction_highlight_offsets(values, grid, half_width)
        second = _city_reduction_highlight_offsets(values, grid, half_width)

        np.testing.assert_allclose(first, second)
        self.assertTrue(np.all(np.abs(first) <= 0.78 * 0.3))
        self.assertGreater(len(np.unique(first[:3])), 1)


if __name__ == "__main__":
    unittest.main()
