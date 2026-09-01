"""Formal spatial validation for the accepted Beijing gravity model."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from drone_emission.calibration.coordinate_restore import _add_basemap
from drone_emission.visualization._utils import _add_title_suffix, write_figure_report


GRID_SIZE_M = 1_000.0
GRID_CRS = "EPSG:32650"
BASEMAP_PROVIDER = "OpenStreetMap.HOT"
BASEMAP_ATTRIBUTION = "© OpenStreetMap contributors"
DEFAULT_FIGSIZE = (180.0 / 25.4, 83.0 / 25.4)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _prepare_endpoint_grid(source: pd.DataFrame) -> dict:
    required = {
        "grid_x",
        "grid_y",
        "observed_orders",
        "expected_orders",
        "x_min_m",
        "y_min_m",
    }
    missing = sorted(required.difference(source.columns))
    if missing:
        raise ValueError("Endpoint grid source is missing: " + ", ".join(missing))
    if source.empty:
        raise ValueError("Endpoint grid source is empty")

    grid = source.loc[:, sorted(required)].copy()
    for column in required:
        grid[column] = pd.to_numeric(grid[column], errors="coerce")
    if grid.isna().any().any() or not np.isfinite(grid.to_numpy(float)).all():
        raise ValueError("Endpoint grid source contains non-finite values")
    if (grid[["observed_orders", "expected_orders"]] < 0).any().any():
        raise ValueError("Endpoint grid order totals must be non-negative")
    if grid.duplicated(["grid_y", "grid_x"]).any():
        raise ValueError("Endpoint grid source contains duplicate grid cells")

    grid_x = grid["grid_x"].to_numpy(float)
    grid_y = grid["grid_y"].to_numpy(float)
    if (
        (grid_x < 0).any()
        or (grid_y < 0).any()
        or not np.equal(grid_x, np.floor(grid_x)).all()
        or not np.equal(grid_y, np.floor(grid_y)).all()
    ):
        raise ValueError("Endpoint grid indices must be non-negative integers")
    grid["grid_x"] = grid_x.astype(int)
    grid["grid_y"] = grid_y.astype(int)

    x_origin = grid["x_min_m"] - grid["grid_x"] * GRID_SIZE_M
    y_origin = grid["y_min_m"] - grid["grid_y"] * GRID_SIZE_M
    if not np.allclose(x_origin, x_origin.iloc[0]) or not np.allclose(
        y_origin, y_origin.iloc[0]
    ):
        raise ValueError("Endpoint grid coordinates do not match the 1 km grid")

    max_x = int(grid["grid_x"].max()) + 1
    max_y = int(grid["grid_y"].max()) + 1
    observed = np.zeros((max_y, max_x), dtype=float)
    expected = np.zeros_like(observed)
    observed[grid["grid_y"], grid["grid_x"]] = grid["observed_orders"]
    expected[grid["grid_y"], grid["grid_x"]] = grid["expected_orders"]
    positive = np.concatenate((observed[observed > 0], expected[expected > 0]))
    if not len(positive):
        raise ValueError("Endpoint grid source contains no positive order totals")

    x_min = float(x_origin.iloc[0])
    y_min = float(y_origin.iloc[0])
    return {
        "grid": grid,
        "observed": observed,
        "expected": expected,
        "x_edges": x_min + np.arange(max_x + 1, dtype=float) * GRID_SIZE_M,
        "y_edges": y_min + np.arange(max_y + 1, dtype=float) * GRID_SIZE_M,
        "extent": [
            x_min,
            x_min + max_x * GRID_SIZE_M,
            y_min,
            y_min + max_y * GRID_SIZE_M,
        ],
        "vmax": float(positive.max()),
    }


def plot_beijing_endpoint_grid_heatmap(
    source: pd.DataFrame,
    save_path: str | None = None,
    figsize: tuple = DEFAULT_FIGSIZE,
    for_latex: bool = False,
    plot_title: bool = False,
    *,
    basemap_cache_dir: str | Path,
    source_path: str | Path | None = None,
    calibration_metadata: dict | None = None,
):
    """Compare observed and modelled Beijing endpoints on aligned 1 km grids."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    from matplotlib.patheffects import withStroke

    prepared = _prepare_endpoint_grid(source)
    norm = LogNorm(vmin=1.0, vmax=max(1.000001, prepared["vmax"]))

    fig = plt.figure(figsize=figsize, facecolor="white")
    layout = fig.add_gridspec(
        1,
        3,
        width_ratios=(1.0, 1.0, 0.035),
        left=0.012,
        right=0.94,
        bottom=0.035,
        top=0.97,
        wspace=0.035,
    )
    axes = [fig.add_subplot(layout[0, 0])]
    axes.append(fig.add_subplot(layout[0, 1], sharex=axes[0], sharey=axes[0]))
    colorbar_ax = fig.add_subplot(layout[0, 2])
    image = None
    cache_dir = Path(basemap_cache_dir)
    for axis, values in zip(
        axes,
        (prepared["observed"], prepared["expected"]),
    ):
        axis.set_xlim(prepared["extent"][0], prepared["extent"][1])
        axis.set_ylim(prepared["extent"][2], prepared["extent"][3])
        _add_basemap(axis, cache_dir, crs=GRID_CRS)
        image = axis.pcolormesh(
            prepared["x_edges"],
            prepared["y_edges"],
            np.ma.masked_less(values, norm.vmin),
            cmap="magma",
            norm=norm,
            alpha=0.82,
            shading="flat",
            linewidth=0,
            zorder=3,
        )
        axis.text(
            0.008,
            0.008,
            BASEMAP_ATTRIBUTION,
            transform=axis.transAxes,
            fontsize=4.5,
            color="0.25",
            ha="left",
            va="bottom",
            path_effects=[withStroke(linewidth=1.0, foreground="white")],
            zorder=10,
        )
        axis.set_aspect("equal", adjustable="box")
        axis.set_axis_off()
    fig.colorbar(
        image,
        cax=colorbar_ax,
        label="Orders per 1 km grid cell (log scale)",
    )
    colorbar_ax.tick_params(labelsize=6, length=2, pad=1.5)
    colorbar_ax.yaxis.label.set_size(7)
    if plot_title:
        fig.suptitle("Beijing endpoint-order spatial validation", fontsize=9)

    output_path = None
    if save_path:
        output_path = Path(save_path)
        if for_latex and output_path.suffix.lower() != ".pdf":
            output_path = output_path.with_suffix(".pdf")
        if plot_title:
            output_path = Path(_add_title_suffix(str(output_path)))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, facecolor="white")

        grid = prepared["grid"]
        locked_source = Path(source_path) if source_path is not None else None
        report = {
            "schema_version": 1,
            "figure_name": "beijing_endpoint_grid_heatmap",
            "figure_path": str(output_path),
            "core_conclusion": (
                "Observed and accepted gravity-model endpoint orders are shown "
                "on aligned 1-km Beijing grids with one shared logarithmic scale."
            ),
            "claim_boundary": (
                "The maps describe the accepted Beijing calibration sample and "
                "do not validate destination allocation outside Beijing."
            ),
            "source": {
                "path": str(locked_source) if locked_source is not None else None,
                "sha256": _sha256(locked_source) if locked_source is not None else None,
            },
            "calibration": calibration_metadata or {},
            "grid": {
                "crs": GRID_CRS,
                "cell_size_m": GRID_SIZE_M,
                "source_cell_count": int(len(grid)),
                "array_shape_yx": [
                    int(prepared["observed"].shape[0]),
                    int(prepared["observed"].shape[1]),
                ],
                "extent_m": [float(value) for value in prepared["extent"]],
                "observed_positive_cell_count": int(
                    np.count_nonzero(prepared["observed"] > 0)
                ),
                "expected_positive_cell_count": int(
                    np.count_nonzero(prepared["expected"] > 0)
                ),
                "observed_order_total": float(grid["observed_orders"].sum()),
                "expected_order_total": float(grid["expected_orders"].sum()),
            },
            "encoding": {
                "colormap": "magma",
                "normalization": "shared_logarithmic",
                "vmin_orders": 1.0,
                "vmax_orders": float(prepared["vmax"]),
                "values_below_one_order": "transparent",
            },
            "panels": [
                {"position": "left", "data": "observed_orders", "title": None},
                {
                    "position": "right",
                    "data": "expected_orders",
                    "title": None,
                },
            ],
            "basemap": {
                "provider": BASEMAP_PROVIDER,
                "attribution": BASEMAP_ATTRIBUTION,
                "attribution_placement": "lower left of each map",
            },
            "layout": {
                "native_size_mm": [
                    float(figsize[0] * 25.4),
                    float(figsize[1] * 25.4),
                ],
                "panel_labels": False,
                "panel_titles": False,
                "figure_title": bool(plot_title),
                "tight_crop": False,
            },
        }
        report_path = output_path.with_name(output_path.stem + "_report.json")
        write_figure_report(str(report_path), report)
        print(f"Beijing endpoint grid heatmap report saved to {report_path}")
        print(f"Beijing endpoint grid heatmap saved to {output_path}")
    else:
        plt.show()
    plt.close(fig)
    return str(output_path) if output_path is not None else None


def _render_beijing_endpoint_grid_heatmap(
    out: str,
    cache_dir: str | Path,
    for_latex: bool,
    plot_title: bool,
) -> None:
    """Render the locked accepted-run endpoint grid as a formal subfigure."""
    calibration_dir = Path(__file__).resolve().parent.parent / "calibration"
    summary_path = calibration_dir / "gravity_calibration_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    run_hash = str(summary["run_hash"])
    source_path = calibration_dir / f"beijing_gravity_endpoint_grid_{run_hash}.csv"
    if not source_path.is_file():
        raise FileNotFoundError(
            "Missing accepted Beijing endpoint grid source: "
            f"{source_path}. Restore it from calibration run {run_hash}."
        )
    plot_beijing_endpoint_grid_heatmap(
        pd.read_csv(source_path),
        save_path=str(Path(out) / "beijing_endpoint_grid_heatmap.pdf"),
        for_latex=for_latex,
        plot_title=plot_title,
        basemap_cache_dir=Path(cache_dir) / "beijing_endpoint_grid_heatmap",
        source_path=source_path,
        calibration_metadata={
            "run_hash": run_hash,
            "source_result_sha256": summary.get("source_result_sha256"),
        },
    )
