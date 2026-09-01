"""Narrow tests for the formal Beijing endpoint-grid heatmap."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from drone_emission.visualization import gravity_endpoint


def _source() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "grid_y": [0, 0, 1],
            "grid_x": [0, 1, 1],
            "observed_orders": [12.0, 0.0, 3.0],
            "expected_orders": [10.0, 1.0, 4.0],
            "x_min_m": [440_000.0, 441_000.0, 441_000.0],
            "y_min_m": [4_420_000.0, 4_420_000.0, 4_421_000.0],
        }
    )


def test_endpoint_grid_heatmap_writes_full_width_pdf_and_report(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(gravity_endpoint, "_add_basemap", lambda *args, **kwargs: None)
    source_path = tmp_path / "endpoint_grid.csv"
    _source().to_csv(source_path, index=False)
    output = tmp_path / "beijing_endpoint_grid_heatmap.pdf"

    result = gravity_endpoint.plot_beijing_endpoint_grid_heatmap(
        pd.read_csv(source_path),
        save_path=str(output),
        for_latex=True,
        plot_title=False,
        basemap_cache_dir=tmp_path / "basemap",
        source_path=source_path,
        calibration_metadata={
            "run_hash": "a1303469cb38697a",
            "source_result_sha256": "accepted-result-hash",
        },
    )

    assert result == str(output)
    assert output.is_file()
    report = json.loads(
        output.with_name("beijing_endpoint_grid_heatmap_report.json").read_text()
    )
    assert report["layout"] == {
        "native_size_mm": [180.0, 83.0],
        "panel_labels": False,
        "panel_titles": False,
        "figure_title": False,
        "tight_crop": False,
    }
    assert report["grid"]["crs"] == "EPSG:32650"
    assert report["grid"]["cell_size_m"] == 1_000.0
    assert report["grid"]["observed_order_total"] == 15.0
    assert report["grid"]["expected_order_total"] == 15.0
    assert report["encoding"] == {
        "colormap": "magma",
        "normalization": "shared_logarithmic",
        "vmin_orders": 1.0,
        "vmax_orders": 12.0,
        "zero_cells": "transparent",
    }
    assert [panel["title"] for panel in report["panels"]] == [None, None]
    assert report["basemap"]["attribution_placement"] == "lower left of each map"
    assert report["calibration"]["run_hash"] == "a1303469cb38697a"
    assert len(report["source"]["sha256"]) == 64


def test_endpoint_grid_heatmap_rejects_unaligned_coordinates() -> None:
    source = _source()
    source.loc[1, "x_min_m"] += 100.0

    with pytest.raises(ValueError, match="do not match the 1 km grid"):
        gravity_endpoint._prepare_endpoint_grid(source)


def test_endpoint_grid_heatmap_requires_positive_orders() -> None:
    source = _source()
    source[["observed_orders", "expected_orders"]] = 0.0

    with pytest.raises(ValueError, match="no positive order totals"):
        gravity_endpoint._prepare_endpoint_grid(source)
