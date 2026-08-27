"""City-level climate-mitigation screening visualization."""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

from drone_emission.analysis.constants import (
    CITY_REDUCTION_SCATTER_LABELED_CITIES,
)

ABSOLUTE_REDUCTION_COL = "annual_absolute_reduction_t_co2e"
RELATIVE_REDUCTION_COL = "oracle_min_reduction_vs_ebike_pct"


def _reduction_metrics(df: pd.DataFrame) -> pd.DataFrame:
    required = {
        "total_ebike_lifecycle",
        "oracle_min_lifecycle_kg",
        RELATIVE_REDUCTION_COL,
    }
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"City reduction scatter missing columns: {missing}")

    frame = df.copy()
    if "region_level" in frame:
        frame = frame[frame["region_level"].eq("city")].copy()
    ebike_kg = pd.to_numeric(frame["total_ebike_lifecycle"], errors="coerce")
    mixed_kg = pd.to_numeric(frame["oracle_min_lifecycle_kg"], errors="coerce")
    frame[ABSOLUTE_REDUCTION_COL] = (ebike_kg - mixed_kg) / 1_000.0
    frame[RELATIVE_REDUCTION_COL] = pd.to_numeric(
        frame[RELATIVE_REDUCTION_COL], errors="coerce"
    )
    valid = (
        np.isfinite(frame[ABSOLUTE_REDUCTION_COL])
        & (frame[ABSOLUTE_REDUCTION_COL] > 0)
        & np.isfinite(frame[RELATIVE_REDUCTION_COL])
    )
    result = frame.loc[valid].copy()
    if result.empty:
        raise ValueError("No finite positive city reduction results to plot")
    return result


def plot_city_reduction_potential_scatter(
    df: pd.DataFrame,
    save_path: str = None,
    figsize: tuple = (130.5 / 25.4, 100 / 25.4),
    for_latex: bool = False,
):
    """Plot absolute against relative city emission-reduction potential."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [
                "Arial",
                "Helvetica",
                "DejaVu Sans",
                "sans-serif",
            ],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )
    plot_df = _reduction_metrics(df)
    absolute_median = float(plot_df[ABSOLUTE_REDUCTION_COL].median())
    relative_median = float(plot_df[RELATIVE_REDUCTION_COL].median())

    fig, ax = plt.subplots(figsize=figsize)
    fig.subplots_adjust(left=0.14, right=0.98, bottom=0.16, top=0.98)
    ax.scatter(
        plot_df[ABSOLUTE_REDUCTION_COL],
        plot_df[RELATIVE_REDUCTION_COL],
        s=30,
        alpha=0.72,
        color="#737373",
        edgecolor="white",
        linewidth=0.35,
        zorder=3,
    )
    ax.axhline(relative_median, color="#666666", linestyle="--", linewidth=1)
    ax.axvline(absolute_median, color="#666666", linestyle="--", linewidth=1)
    ax.set_xscale("log")

    name_col = "region_name_en" if "region_name_en" in plot_df else "region_key"
    labeled_city_order = tuple(
        city_name for city_name, _ in CITY_REDUCTION_SCATTER_LABELED_CITIES
    )
    labeled = plot_df[plot_df[name_col].isin(labeled_city_order)].copy()
    labeled["_label_order"] = pd.Categorical(
        labeled[name_col], categories=labeled_city_order, ordered=True
    )
    labeled = labeled.sort_values("_label_order")
    ax.scatter(
        labeled[ABSOLUTE_REDUCTION_COL],
        labeled[RELATIVE_REDUCTION_COL],
        s=30,
        facecolors="none",
        edgecolors="#555555",
        linewidth=0.8,
        zorder=5,
    )
    label_texts = [
        ax.text(
            row[ABSOLUTE_REDUCTION_COL],
            row[RELATIVE_REDUCTION_COL],
            row[name_col],
            fontsize=7.5,
            color="#202020",
            zorder=6,
        )
        for _, row in labeled.iterrows()
    ]
    if label_texts:
        from adjustText import adjust_text

        adjust_text(
            label_texts,
            x=labeled[ABSOLUTE_REDUCTION_COL].to_numpy(dtype=float),
            y=labeled[RELATIVE_REDUCTION_COL].to_numpy(dtype=float),
            ax=ax,
            expand=(1.12, 1.22),
            force_text=(0.15, 0.25),
            force_static=(0.08, 0.15),
            ensure_inside_axes=True,
            arrowprops=None,
        )

    ax.set_xlabel(
        "Annual absolute emission reduction (t CO$_2$e yr$^{-1}$)",
        fontsize=9,
    )
    ax.set_ylabel("Relative lifecycle emission reduction (%)", fontsize=9)
    ax.tick_params(axis="both", labelsize=8)
    ax.legend(
        [Line2D([0], [0], color="#666666", linestyle="--", linewidth=1)],
        ["Median values"],
        loc="upper left",
        frameon=False,
        fontsize=8,
    )
    ax.spines[["top", "right"]].set_visible(False)

    if not save_path:
        fig.show()
        return None
    if for_latex and not save_path.endswith(".pdf"):
        save_path = save_path.rsplit(".", 1)[0] + ".pdf"
    save_dir = os.path.dirname(save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    fig.savefig(
        save_path,
        dpi=300,
        bbox_inches=fig.bbox_inches,
        pad_inches=0.0,
    )
    plt.close(fig)
    print(f"City reduction-potential scatter saved to {save_path}")
    return save_path
