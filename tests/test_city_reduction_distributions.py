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

from drone_emission.visualization.analysis import (  # noqa: E402
    ORACLE_REDUCTION_LABEL_CITIES,
    _city_reduction_box_statistics,
    _city_reduction_violin_highlight_offsets,
    _draw_city_reduction_violin,
    _city_reduction_highlight_offsets,
    _city_reduction_outlier_offsets,
    plot_city_annual_absolute_reduction_distribution,
    plot_city_annual_absolute_reduction_violin_distribution,
    plot_city_relative_lifecycle_reduction_distribution,
    plot_city_relative_lifecycle_reduction_violin_distribution,
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
            self.assertEqual(
                relative_report["distribution"]["method"],
                "Vertical Tukey boxplot",
            )
            self.assertEqual(
                absolute_report["box_statistics"]["calculation_space"],
                "log10",
            )
            self.assertEqual(relative_report["figure_size_inches"], [1.65, 2.04])
            self.assertEqual(absolute_report["figure_size_inches"], [1.65, 2.04])
            self.assertTrue(relative_report["distribution"]["outliers_displayed"])
            self.assertTrue(absolute_report["distribution"]["outliers_displayed"])
            self.assertEqual(relative_report["outliers"]["count"], 1)
            self.assertEqual(
                relative_report["outliers"]["count"],
                len(relative_report["outliers"]["records"]),
            )

    def test_highlight_offsets_are_deterministic_and_symmetric(self) -> None:
        values = np.array([10.0, 10.2, 10.4, 20.0])

        first = _city_reduction_highlight_offsets(values, (0.0, 50.0))
        second = _city_reduction_highlight_offsets(values, (0.0, 50.0))

        np.testing.assert_allclose(first, second)
        self.assertTrue(np.all(np.abs(first) <= 0.38))
        self.assertTrue(np.any(first > 0))
        self.assertTrue(np.any(first < 0))
        self.assertGreater(len(np.unique(first[:3])), 1)

    def test_violin_variants_keep_vertical_geometry_and_new_names(self) -> None:
        city_frame = _city_frame()
        with tempfile.TemporaryDirectory() as tmp_dir:
            relative_path = (
                plot_city_relative_lifecycle_reduction_violin_distribution(
                    city_frame,
                    save_path=os.path.join(tmp_dir, "relative_violin.png"),
                    for_latex=True,
                )
            )
            absolute_path = (
                plot_city_annual_absolute_reduction_violin_distribution(
                    city_frame,
                    save_path=os.path.join(tmp_dir, "absolute_violin.png"),
                    for_latex=True,
                )
            )

            relative_report = _load_report(relative_path)
            absolute_report = _load_report(absolute_path)
            _assert_highlight_report(relative_report)
            _assert_highlight_report(absolute_report)
            self.assertEqual(relative_report["figure_size_inches"], [1.65, 2.04])
            self.assertEqual(absolute_report["figure_size_inches"], [1.65, 2.04])
            self.assertEqual(
                relative_report["figure_name"],
                "city_relative_lifecycle_reduction_violin_distribution",
            )
            self.assertEqual(
                absolute_report["figure_name"],
                "city_annual_absolute_reduction_violin_distribution",
            )
            self.assertEqual(
                relative_report["distribution"]["method"],
                "Vertical violin plot with Gaussian KDE",
            )
            for report in (relative_report, absolute_report):
                self.assertNotIn("outliers", report)
                self.assertNotIn("outliers_displayed", report["distribution"])
                self.assertEqual(
                    set(report["box_statistics"]),
                    {"calculation_space", "q1", "median", "q3"},
                )
            self.assertEqual(
                relative_report["highlighted_cities"]["marker"]["layout"],
                "Q1-Q3 highlights alternate left and right clear of center, "
                "clipped to 95% of local violin half-width",
            )
            self.assertEqual(
                absolute_report["highlighted_cities"]["marker"]["layout"],
                "compact symmetric offsets clipped to 95% of local violin "
                "half-width",
            )

    def test_violin_highlight_offsets_are_compact_and_balanced(self) -> None:
        cases = (
            (
                np.array([13.2, 10.7, 20.1, 13.0, 14.1, 19.2, 23.9, 35.0]),
                (0.0, 50.0),
                False,
            ),
            (
                np.array(
                    [
                        3274.0,
                        2136.0,
                        2519.0,
                        1318.0,
                        2223.0,
                        727.0,
                        1576.0,
                        2012.0,
                    ]
                ),
                (0.03, 5000.0),
                True,
            ),
        )

        for values, limits, log10_space in cases:
            first = _city_reduction_violin_highlight_offsets(
                values,
                limits,
                log10_space=log10_space,
            )
            second = _city_reduction_violin_highlight_offsets(
                values,
                limits,
                log10_space=log10_space,
            )

            np.testing.assert_allclose(first, second)
            self.assertTrue(np.all(np.abs(first) <= 0.28))
            positive = np.sort(first[first > 0])
            negative = np.sort(-first[first < 0])
            np.testing.assert_allclose(positive, negative)

    def test_violin_highlight_offsets_respect_linear_local_width(self) -> None:
        values = np.array([1.0, 1.02, 1.5, 2.0, 3.0])
        density_grid = np.array([1.0, 2.0, 3.0])
        half_width = np.array([0.0, 0.32, 0.0])

        offsets = _city_reduction_violin_highlight_offsets(
            values,
            (1.0, 3.0),
            density_grid=density_grid,
            half_width=half_width,
        )
        local_half_width = np.interp(values, density_grid, half_width)

        self.assertTrue(
            np.all(np.abs(offsets) <= 0.95 * local_half_width + 1e-12)
        )
        self.assertEqual(offsets[0], 0.0)
        self.assertGreater(abs(offsets[1]), 0.0)

    def test_violin_highlight_offsets_respect_log_local_width(self) -> None:
        values = np.array([1.0, 1.02, 3.0, 10.0, 100.0])
        density_grid = np.array([0.0, 1.0, 2.0])
        half_width = np.array([0.0, 0.32, 0.0])

        offsets = _city_reduction_violin_highlight_offsets(
            values,
            (1.0, 100.0),
            log10_space=True,
            density_grid=density_grid,
            half_width=half_width,
        )
        local_half_width = np.interp(
            np.log10(values),
            density_grid,
            half_width,
        )

        self.assertTrue(
            np.all(np.abs(offsets) <= 0.95 * local_half_width + 1e-12)
        )
        self.assertEqual(offsets[0], 0.0)
        self.assertGreater(abs(offsets[1]), 0.0)

    def test_violin_summary_avoidance_is_scoped_and_alternating(self) -> None:
        values = np.array([5.0, 9.0, 13.0, 13.5, 14.0, 20.0])
        density_grid = np.array([0.0, 25.0])
        half_width = np.array([0.2, 0.2])

        baseline = _city_reduction_violin_highlight_offsets(
            values,
            (0.0, 50.0),
            density_grid=density_grid,
            half_width=half_width,
        )
        offsets = _city_reduction_violin_highlight_offsets(
            values,
            (0.0, 50.0),
            density_grid=density_grid,
            half_width=half_width,
            center_avoidance_interval=(9.0, 14.0),
        )

        self.assertEqual(baseline[1], 0.0)
        self.assertEqual(baseline[3], 0.0)
        np.testing.assert_allclose(offsets[[0, 5]], baseline[[0, 5]])
        self.assertTrue(np.all(np.abs(offsets[1:5]) >= 0.06))
        np.testing.assert_array_equal(
            np.sign(offsets[1:5]),
            np.array([-1.0, 1.0, -1.0, 1.0]),
        )
        self.assertTrue(np.all(np.abs(offsets) <= 0.19 + 1e-12))

    def test_violin_draws_no_neutral_outlier_markers(self) -> None:
        import matplotlib.pyplot as plt
        from matplotlib.collections import PathCollection

        fig, ax = plt.subplots()
        try:
            box_report, _, _ = _draw_city_reduction_violin(
                ax,
                np.array([10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 49.0]),
                value_limits=(0.0, 50.0),
                y_label="Reduction (%)",
                highlight_values=[12.0],
            )
            point_collections = [
                collection
                for collection in ax.collections
                if isinstance(collection, PathCollection)
            ]

            self.assertEqual(len(point_collections), 2)
            self.assertEqual(
                set(box_report),
                {"calculation_space", "q1", "median", "q3"},
            )
        finally:
            plt.close(fig)

    def test_tukey_whiskers_use_the_requested_calculation_space(self) -> None:
        linear_stats, linear_report = _city_reduction_box_statistics(
            [1.0, 2.0, 3.0, 4.0, 100.0]
        )
        self.assertEqual(linear_stats["whishi"], 4.0)
        self.assertEqual(linear_stats["fliers"], [100.0])
        self.assertEqual(linear_report["calculation_space"], "linear")
        self.assertTrue(linear_report["outliers_displayed"])
        self.assertEqual(linear_report["outlier_count"], 1)

        log_stats, log_report = _city_reduction_box_statistics(
            [1.0, 10.0, 100.0, 1000.0, 1e9],
            log10_space=True,
        )
        self.assertEqual(log_stats["q1"], 10.0)
        self.assertEqual(log_stats["med"], 100.0)
        self.assertEqual(log_stats["q3"], 1000.0)
        self.assertEqual(log_stats["whishi"], 1000.0)
        self.assertEqual(log_stats["fliers"], [1e9])
        self.assertEqual(log_report["calculation_space"], "log10")

    def test_outlier_offsets_are_narrow_and_deterministic(self) -> None:
        values = np.array([30.0, 30.1, 30.2, 40.0])

        first = _city_reduction_outlier_offsets(values, (0.0, 50.0))
        second = _city_reduction_outlier_offsets(values, (0.0, 50.0))

        np.testing.assert_allclose(first, second)
        self.assertTrue(np.all(np.abs(first) <= 0.11))
        self.assertTrue(np.any(first < 0))
        self.assertTrue(np.any(first > 0))


if __name__ == "__main__":
    unittest.main()
