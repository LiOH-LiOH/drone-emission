"""Targeted tests for the 5-km captured drone-order scatter."""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.normpath(os.path.join(HERE, "..", "src", "drone_emission"))
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from visualization.analysis import (  # noqa: E402
    PILOT_CAPTURED_5KM_ORDERS_COL,
    _prepare_city_reduction_5km_order_scatter,
    plot_city_reduction_5km_drone_orders_scatter,
)


def _city_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "region_key": ["city_a", "city_b", "city_c"],
            "region_name_en": ["City A", "City B", "City C"],
            "eligible_annual_orders": [1_000.0, 2_000.0, 3_000.0],
            "capture_5km_pct": [25.0, 50.0, 0.0],
            "oracle_absolute_reduction_t_co2e": [10.0, 100.0, 1_000.0],
            "oracle_relative_reduction_pct": [5.0, 10.0, 15.0],
        }
    )


class CityReduction5kmOrderScatterTests(unittest.TestCase):
    def test_prepares_captured_orders_and_drops_nonpositive_values(self) -> None:
        prepared = _prepare_city_reduction_5km_order_scatter(_city_frame())

        self.assertEqual(prepared["region_key"].tolist(), ["city_a", "city_b"])
        np.testing.assert_allclose(
            prepared[PILOT_CAPTURED_5KM_ORDERS_COL],
            [250.0, 1_000.0],
        )

    def test_missing_capture_column_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "capture_5km_pct"):
            _prepare_city_reduction_5km_order_scatter(
                _city_frame().drop(columns="capture_5km_pct")
            )

    def test_renders_png_and_single_value_pdf(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            png_path = output_dir / "orders.png"
            rendered_png = plot_city_reduction_5km_drone_orders_scatter(
                _city_frame(),
                save_path=str(png_path),
            )
            self.assertEqual(rendered_png, str(png_path))
            self.assertGreater(png_path.stat().st_size, 0)

            pdf_source_path = output_dir / "single.png"
            rendered_pdf = plot_city_reduction_5km_drone_orders_scatter(
                _city_frame().iloc[[0]],
                save_path=str(pdf_source_path),
                for_latex=True,
            )
            pdf_path = output_dir / "single.pdf"
            self.assertEqual(rendered_pdf, str(pdf_path))
            self.assertGreater(pdf_path.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
