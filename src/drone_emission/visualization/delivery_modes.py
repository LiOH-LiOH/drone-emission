"""Formal delivery-mode lifecycle comparison subfigures."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import font_manager
from matplotlib.ticker import MaxNLocator

from drone_emission.analysis.weighted_quantile import weighted_csv_quantile
from drone_emission.emission import DELIVERY_MODES, lifecycle_emission_per_order
from drone_emission.visualization._utils import write_figure_report


BEIJING_GRID_EF_KG_PER_KWH = 0.558
PETROL_WTW_EF_KG_PER_MJ = 0.08901666666666667
MODE_LABELS = {
    "pedal_bicycle": "Pedal bicycle",
    "pedelec_ebike": "Pedelec e-bike",
    "electric_moped": "Electric moped",
    "fuel_moped": "Fuel moped",
    "battery_electric_car": "Battery-electric car",
    "fuel_car": "Fuel car",
    "drone": "Drone",
}
COMPARISON_MODE_LABELS = {**MODE_LABELS, "pedelec_ebike": "E-bike"}
MODE_COLORS = {
    "pedal_bicycle": "#4f8f62",
    "pedelec_ebike": "#76a96d",
    "electric_moped": "#5f9fbd",
    "fuel_moped": "#c58a52",
    "battery_electric_car": "#4678a8",
    "fuel_car": "#a95f43",
    "drone": "#7b6aa8",
}
COMPARISON_EXCLUDED_MODES = ("walking", "electric_moped", "pedal_bicycle")
COMPARISON_DISPLAY_MODES = tuple(
    mode for mode in DELIVERY_MODES if mode not in COMPARISON_EXCLUDED_MODES
)
COMPARISON_BOTTOM_TO_TOP_MODES = (
    "drone",
    "pedelec_ebike",
    "fuel_moped",
    "battery_electric_car",
    "fuel_car",
)
BEIJING_EXCLUDED_MODES = ("walking", "electric_moped")
BEIJING_DISPLAY_MODES = tuple(
    mode for mode in DELIVERY_MODES if mode not in BEIJING_EXCLUDED_MODES
)
BEIJING_MODE_ORDER = (
    "pedal_bicycle",
    "pedelec_ebike",
    "drone",
    "fuel_moped",
    "battery_electric_car",
    "fuel_car",
)
SOURCE_MODES = tuple(DELIVERY_MODES)
REQUIRED_COLUMNS = {
    "mode",
    "distance_basis",
    "annual_orders",
    "weighted_mean_distance_km",
    "weighted_mean_g_per_order",
    "annual_lifecycle_kg",
    "annual_lifecycle_kt",
}
COMPARISON_WIDTH_MM = 180.0
COMPARISON_HEIGHT_MM = 60.0
BEIJING_WIDTH_MM = 88.0
BEIJING_HEIGHT_MM = 80.0
FONT_FAMILY = "Arial"
AXIS_LABEL_FONTSIZE = 7.0
STANDARD_FONTSIZE = 6.0


def _set_plot_style() -> None:
    font_manager.findfont(
        font_manager.FontProperties(family=FONT_FAMILY),
        fallback_to_default=False,
    )
    mpl.rcParams.update(
        {
            "font.family": FONT_FAMILY,
            "font.sans-serif": [FONT_FAMILY],
            "mathtext.fontset": "custom",
            "mathtext.rm": FONT_FAMILY,
            "mathtext.it": f"{FONT_FAMILY}:italic",
            "mathtext.bf": f"{FONT_FAMILY}:bold",
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": STANDARD_FONTSIZE,
            "axes.linewidth": 0.8,
        }
    )


def build_comparison(
    *,
    od_csv: str | Path,
    exp_config: dict,
    grid_ef_kg_per_kwh: float = BEIJING_GRID_EF_KG_PER_KWH,
    fuel_wtw_ef_kg_per_mj: float = PETROL_WTW_EF_KG_PER_MJ,
    quantile_cache_dir: str | Path | None = None,
):
    """Calculate displayed modes at Beijing's sales-weighted road-distance P75."""
    quantile = weighted_csv_quantile(
        od_csv,
        value_column="road_distance_m",
        weight_column="annual_sales",
        quantile=0.75,
        cache_dir=quantile_cache_dir,
    )
    distance_km = quantile.value / 1000.0
    values_by_mode = {
        mode: float(
            lifecycle_emission_per_order(
                mode,
                distance_km,
                exp_config=exp_config,
                grid_ef_kg_per_kwh=grid_ef_kg_per_kwh,
                fuel_wtw_ef_kg_per_mj=fuel_wtw_ef_kg_per_mj,
            )
        )
        for mode in COMPARISON_DISPLAY_MODES
    }
    rows = [
        (mode, values_by_mode[mode])
        for mode in COMPARISON_BOTTOM_TO_TOP_MODES
    ]
    return rows, distance_km, quantile


def build_delivery_mode_figure_report(
    *,
    rows,
    distance_km: float,
    quantile,
    figure_path: str | Path,
    od_csv: str | Path,
) -> dict:
    source = Path(od_csv)
    stat = source.stat()
    return {
        "schema_version": 3,
        "figure_name": "delivery_mode_lifecycle_comparison",
        "figure_path": str(figure_path),
        "formal_manifest_registered": True,
        "figure_size_mm": {
            "width": COMPARISON_WIDTH_MM,
            "height": COMPARISON_HEIGHT_MM,
        },
        "functional_unit": "g CO2e per unbundled order",
        "displayed_modes": [mode for mode, _ in rows],
        "excluded_modes": [
            {"mode": mode, "reason": "excluded from display by request"}
            for mode in COMPARISON_EXCLUDED_MODES
        ],
        "input_signature": {
            "source": source.name,
            "size_bytes": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
        },
        "distance": {
            "definition": "common one-way travelled distance; round trip is 2d",
            "source_column": "road_distance_m",
            "weight_column": "annual_sales",
            "quantile": 0.75,
            "method": "smallest value with cumulative weight >= q * total weight",
            "value_km": distance_km,
            "valid_rows": quantile.valid_rows,
            "total_annual_sales": quantile.total_weight,
        },
        "scenario": {
            "grid_ef_kg_per_kwh": BEIJING_GRID_EF_KG_PER_KWH,
            "fuel_wtw_ef_kg_per_mj": PETROL_WTW_EF_KG_PER_MJ,
            "orders_per_trip": 1,
        },
        "boundary": {
            "included": [
                "vehicle and battery manufacturing",
                "assembly and disposal",
                "source-included fluids",
                "vehicle delivery at point of purchase",
                "energy use and upstream supply",
            ],
            "excluded": [
                "road and charging infrastructure",
                "platform and warehouse operations",
                "shared-vehicle operations and rebalancing",
            ],
        },
        "sources": {
            "ground_modes": (
                "https://www.itf-oecd.org/"
                "good-to-go-environmental-performance-new-mobility"
            ),
            "fuel_factor": (
                "https://www.gov.uk/government/publications/"
                "greenhouse-gas-reporting-conversion-factors-2026"
            ),
        },
        "emissions_g_per_order": [
            {"mode": mode, "value": value} for mode, value in rows
        ],
    }


def plot_delivery_mode_lifecycle_comparison(
    *,
    od_csv: str | Path,
    exp_config: dict,
    save_path: str | Path,
    quantile_cache_dir: str | Path | None = None,
    figsize: tuple[float, float] = (
        COMPARISON_WIDTH_MM / 25.4,
        COMPARISON_HEIGHT_MM / 25.4,
    ),
    for_latex: bool = False,
    plot_title: bool = False,
) -> Path:
    """Plot lifecycle emissions for five powered modes at one common distance."""
    del for_latex
    rows, distance_km, quantile = build_comparison(
        od_csv=od_csv,
        exp_config=exp_config,
        quantile_cache_dir=quantile_cache_dir,
    )
    _set_plot_style()
    output = Path(save_path)
    if plot_title:
        output = output.with_name(f"{output.stem}_t{output.suffix}")
    output.parent.mkdir(parents=True, exist_ok=True)
    labels = [COMPARISON_MODE_LABELS[mode] for mode, _ in rows]
    values = [value for _, value in rows]
    colors = [MODE_COLORS[mode] for mode, _ in rows]
    fig, ax = plt.subplots(figsize=figsize, facecolor="white")
    bars = ax.barh(labels, values, color=colors, height=0.62)
    max_value = max(values)
    label_offset = max(max_value * 0.012, 0.8)
    for bar, value in zip(bars, values):
        ax.text(
            value + label_offset,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.1f}",
            va="center",
            ha="left",
            fontsize=STANDARD_FONTSIZE,
            color="0.2",
        )
    ax.set_xlim(0, max_value * 1.15 + label_offset)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.set_xlabel(
        "Lifecycle emissions (g CO$_2$e/order)",
        fontsize=AXIS_LABEL_FONTSIZE,
    )
    ax.tick_params(axis="both", labelsize=STANDARD_FONTSIZE, length=2.5)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.grid(axis="x", color="0.9", linewidth=0.5)
    ax.set_axisbelow(True)
    if plot_title:
        ax.set_title(
            f"Delivery modes at the same {distance_km:.2f}-km one-way distance",
            fontsize=AXIS_LABEL_FONTSIZE,
            loc="left",
        )
    fig.tight_layout(pad=0.8)
    fig.savefig(output, dpi=300, facecolor="white")
    plt.close(fig)
    write_figure_report(
        output.with_name(f"{output.stem}_report.json"),
        build_delivery_mode_figure_report(
            rows=rows,
            distance_km=distance_km,
            quantile=quantile,
            figure_path=output,
            od_csv=od_csv,
        ),
    )
    return output


def load_beijing_mode_data(
    path: str | Path,
    *,
    mode_order: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Load the retained table and return the six displayed modes."""
    frame = pd.read_csv(path)
    missing = sorted(REQUIRED_COLUMNS - set(frame.columns))
    if missing:
        raise ValueError(f"analysis CSV is missing columns: {missing}")
    unknown = sorted(set(frame["mode"].astype(str)) - set(SOURCE_MODES))
    if unknown:
        raise ValueError(f"analysis CSV contains unknown modes: {unknown}")
    if len(frame) != len(SOURCE_MODES) or frame["mode"].nunique() != len(SOURCE_MODES):
        raise ValueError("analysis CSV must contain each delivery mode exactly once")
    frame = frame.copy()
    frame["annual_lifecycle_kt"] = pd.to_numeric(
        frame["annual_lifecycle_kt"], errors="raise"
    )
    frame = frame[frame["mode"].isin(BEIJING_DISPLAY_MODES)].copy()
    if mode_order is None:
        return frame.sort_values(
            ["annual_lifecycle_kt", "mode"], kind="stable"
        ).reset_index(drop=True)
    order = {mode: index for index, mode in enumerate(mode_order)}
    frame["_display_order"] = frame["mode"].map(order)
    return (
        frame.sort_values("_display_order", kind="stable")
        .drop(columns="_display_order")
        .reset_index(drop=True)
    )


def build_beijing_mode_figure_report(
    frame: pd.DataFrame,
    analysis_report: dict,
    *,
    figure_path: str | Path,
) -> dict:
    displayed_modes = frame["mode"].astype(str).tolist()
    coefficients = analysis_report.get("coefficients", {})
    boundary = {
        key: value
        for key, value in analysis_report.get("boundary", {}).items()
        if not key.startswith("walking_")
    }
    return {
        "schema_version": 1,
        "figure_name": "beijing_all_mode_lifecycle",
        "figure_path": str(figure_path),
        "formal_manifest_registered": True,
        "figure_size_mm": {
            "width": BEIJING_WIDTH_MM,
            "height": BEIJING_HEIGHT_MM,
        },
        "functional_unit": analysis_report.get("functional_unit"),
        "displayed_modes": displayed_modes,
        "excluded_modes": [
            {"mode": mode, "reason": "excluded from display by request"}
            for mode in BEIJING_EXCLUDED_MODES
        ],
        "analysis": {
            "analysis_version": analysis_report.get("analysis_version"),
            "calculation_method": analysis_report.get("calculation_method"),
            "input_signature": analysis_report.get("input_signature"),
            "od_filter": analysis_report.get("od_filter"),
            "distance": analysis_report.get("distance"),
            "scenario": analysis_report.get("scenario"),
            "boundary": boundary,
            "coefficients": {
                mode: coefficients[mode]
                for mode in displayed_modes
                if mode in coefficients
            },
        },
        "plotted_values": frame.to_dict(orient="records"),
    }


def plot_beijing_all_mode_lifecycle(
    frame: pd.DataFrame,
    analysis_report: dict,
    *,
    save_path: str | Path,
    figsize: tuple[float, float] = (
        BEIJING_WIDTH_MM / 25.4,
        BEIJING_HEIGHT_MM / 25.4,
    ),
    for_latex: bool = False,
    plot_title: bool = False,
) -> Path:
    """Plot annual lifecycle totals for six all-Beijing mode scenarios."""
    del for_latex
    _set_plot_style()
    output = Path(save_path)
    if plot_title:
        output = output.with_name(f"{output.stem}_t{output.suffix}")
    output.parent.mkdir(parents=True, exist_ok=True)
    labels = [MODE_LABELS[mode] for mode in frame["mode"]]
    values = frame["annual_lifecycle_kt"].to_numpy(dtype=float)
    colors = [MODE_COLORS[mode] for mode in frame["mode"]]
    fig, ax = plt.subplots(figsize=figsize, facecolor="white")
    bars = ax.barh(labels, values, color=colors, height=0.68)
    max_value = float(values.max())
    label_offset = max(max_value * 0.012, 0.05)
    for bar, value in zip(bars, values):
        ax.text(
            value + label_offset,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.1f}",
            va="center",
            ha="left",
            fontsize=STANDARD_FONTSIZE,
            color="0.2",
        )
    ax.set_xlim(0, max_value * 1.15 + label_offset)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.set_xlabel(
        "Annual lifecycle emissions (kt CO$_2$e yr$^{-1}$)",
        fontsize=AXIS_LABEL_FONTSIZE,
    )
    ax.tick_params(axis="both", labelsize=STANDARD_FONTSIZE, length=2.5)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.grid(axis="x", color="0.9", linewidth=0.5)
    ax.set_axisbelow(True)
    if plot_title:
        ax.set_title(
            "All Beijing deliveries served by one transport mode",
            fontsize=AXIS_LABEL_FONTSIZE,
            loc="left",
        )
    fig.tight_layout(pad=0.8)
    fig.savefig(output, dpi=300, facecolor="white")
    plt.close(fig)
    write_figure_report(
        output.with_name(f"{output.stem}_report.json"),
        build_beijing_mode_figure_report(
            frame,
            analysis_report,
            figure_path=output,
        ),
    )
    return output


def load_analysis_report(path: str | Path) -> dict:
    """Load retained Beijing delivery-mode analysis metadata."""
    return json.loads(Path(path).read_text(encoding="utf-8"))
