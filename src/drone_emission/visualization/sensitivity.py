"""Exploratory plots from the consolidated sensitivity outputs.

Reads ``data/all_sensitivity_oat.csv``, ``data/all_sensitivity_shapley.csv``,
``data/all_sensitivity_sobol.csv``, and ``data/all_sensitivity_named.csv``.
Produces candidate figures into ``figures/sensitivity/``.

Each plot function follows the project's standard signature
``(..., save_path=None, dpi=150, for_latex=False, plot_title=False)``:

* ``for_latex=True`` swaps the file extension to ``.pdf``
* ``plot_title=True`` appends ``_t`` to the filename and renders titles
  + supplementary text. Otherwise titles are suppressed (for clean
  paper-figure exports).

Run with:
    cd src/drone_emission && ../../.venv/bin/python -m visualization.sensitivity
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

from ._utils import _add_title_suffix

OAT_FEATURES = (
    "drone_energy_mult", "payload_kg", "drone_lifespan_yr",
    "drone_cruise_kmh", "drone_body_kg", "drone_hover_s",
    "drone_battery_lifespan_yr", "headwind_mps",
    "drone_battery_mass",
    "drone_production_co2_per_kg",
    "drone_battery_co2_per_kwh",
    "drone_utilization_rate",
)

# Curated subset for supplementary OAT diagnostics. The main paper robustness
# figure now uses class-specific Sobol panels below.
OAT_FEATURES_STORY = (
    "drone_energy_mult", "drone_body_kg",
    "drone_hover_s", "headwind_mps",
    "drone_cruise_kmh", "drone_battery_lifespan_yr", "payload_kg",
)

OAT_FEATURE_LABELS = {
    "kappa": "Bundling factor (κ)",
    "gef_mult": "Grid emission factor",
    "drone_energy_mult": "Drone energy use",
    "payload_kg": "Payload mass",
    "drone_lifespan_yr": "Drone frame lifespan",
    "ebike_lifespan_yr": "E-bike lifespan",
    "drone_cruise_kmh": "Drone cruise speed",
    "drone_body_kg": "Drone empty mass",
    "drone_hover_s": "Hover time",
    "drone_battery_lifespan_yr": "Drone battery lifespan",
    "headwind_mps": "Headwind speed",
    "drone_battery_mass": "Drone battery mass",
    "drone_production_co2_per_kg": "Drone body production CO₂/kg",
    "drone_battery_co2_per_kwh": "Drone battery production CO₂/kWh",
    "drone_utilization_rate": "Drone utilization rate",
}

# Feature → class mapping for drone-technology robustness.
OAT_FEATURE_MECHANISM = {
    "drone_body_kg":               "production_eol",
    "drone_battery_mass":          "production_eol",
    "drone_production_co2_per_kg": "production_eol",
    "drone_battery_co2_per_kwh":   "production_eol",
    "drone_energy_mult":           "operation",
    "drone_cruise_kmh":            "operation",
    "drone_hover_s":               "operation",
    "headwind_mps":                "operation",
    "payload_kg":                  "operation",
    "drone_lifespan_yr":           "amortization",
    "drone_battery_lifespan_yr":   "amortization",
    "drone_utilization_rate":      "amortization",
}
OAT_MECHANISM_COLORS = {
    "operation":       "#4C72B0",
    "production_eol":  "#55A467",
    "amortization":    "#8172B2",
    "anchor":          "#DD8452",  # legacy MC baseline color
}
OAT_MECHANISM_LABELS = {
    "operation":      "Drone operation",
    "production_eol": "Production and end-of-life",
    "amortization":   "Amortization",
    "anchor":         "Baseline",
}

MECHANISM_ORDER = ("M1_routing", "M2_bundling", "M3_production", "M4_gef")
MECHANISM_LABELS = {
    "M1_routing": "Routing geometry",
    "M2_bundling": "E-bike bundling",
    "M3_production": "Production amortization",
    "M4_gef": "Grid carbon intensity",
    "residual_all_off": "Residual (use-phase asymmetry)",
}
MECHANISM_COLORS = {
    "M1_routing": "#4C72B0",
    "M2_bundling": "#DD8452",
    "M3_production": "#55A467",
    "M4_gef": "#C44E52",
    "residual_all_off": "#8172B2",
}

NAMED_STRATEGY_ORDER = (
    "all_ebike", "all_drone", "optimal_mixed", "conservative_2km",
    "optimal_mixed_2030", "optimal_mixed_2060",
)


# ---------------------------------------------------------------------------
# Save-time conventions: matches the rest of visualization/*.py
# ---------------------------------------------------------------------------

def _finalize(save_path: str | None, dpi: int = 300, for_latex: bool = False, plot_title: bool = False):
    """Standard save logic. dpi defaults to 300 per viz-conventions."""
    if save_path is None:
        plt.show()
        return None
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    if for_latex and not save_path.endswith(".pdf"):
        save_path = save_path.rsplit(".", 1)[0] + ".pdf"
    if plot_title:
        save_path = _add_title_suffix(save_path)
    plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"  wrote {save_path}")
    return save_path


def _resolve_path(out_dir: str, basename: str, for_latex: bool) -> str:
    ext = "pdf" if for_latex else "png"
    return os.path.join(out_dir, f"{basename}.{ext}")


# ---------------------------------------------------------------------------
# OAT plots
# ---------------------------------------------------------------------------

def plot_oat_tornado(
    oat: pd.DataFrame,
    save_path: str | None = None,
    figsize: tuple = (8, 5.5),
    for_latex: bool = False,
    plot_title: bool = False,
    feature_whitelist: Iterable[str] | None = None,
    metric: str = "vs_scenario",
):
    """OAT tornado: median absolute swing per feature with IQR whiskers.

    Bars are coloured by the drone-technology class each feature maps to.
    ``feature_whitelist`` restricts the bars to a named subset (in any order;
    bars are still sorted by descending median swing).
    If ``None``, every feature present in ``oat`` is plotted.

    Args:
        oat: Long-format OAT DataFrame from ``all_sensitivity_oat.csv``.
        save_path: Path to save figure. Shows plot if None.
        figsize: Figure size tuple. Defaults to (8, 5.5); pass a taller size
            for the full drone-tech tornado.
        for_latex: If True, save as PDF.
        plot_title: If True, show title.
        feature_whitelist: Optional ordered iterable of feature names to
            include. Use ``OAT_FEATURES_STORY`` for a compact diagnostic figure.
        metric: Which reduction column to compute swings against. Use
            ``"vs_scenario"`` (default, §4 substitution-efficiency reading)
            or ``"vs_anchor"`` (§5 headline-reduction reading, kept for
            cross-reference).
    """
    metric_col = {
        "vs_scenario": "reduction_vs_scenario_ebike_pct",
        "vs_anchor":   "reduction_vs_anchor_ebike_pct",
    }[metric]

    features = (
        list(feature_whitelist)
        if feature_whitelist is not None
        else list(OAT_FEATURES)
    )
    base = oat[oat["scenario_id"] == "baseline"][["region_key", metric_col]]
    base = base.rename(columns={metric_col: "baseline_red"})
    diffs: Dict[str, list[float]] = {}
    for feature in features:
        sub = oat[oat["feature_varied"] == feature]
        if sub.empty:
            continue
        merged = sub.merge(base, on="region_key", how="inner")
        merged["delta"] = merged[metric_col] - merged["baseline_red"]
        per_city_max = merged.groupby("region_key")["delta"].agg(lambda s: s.abs().max())
        diffs[feature] = per_city_max.tolist()

    sorted_features = sorted(diffs.keys(), key=lambda f: -np.median(diffs[f]))
    medians = [np.median(diffs[f]) for f in sorted_features]
    p25 = [np.percentile(diffs[f], 25) for f in sorted_features]
    p75 = [np.percentile(diffs[f], 75) for f in sorted_features]
    bar_colors = [
        OAT_MECHANISM_COLORS[OAT_FEATURE_MECHANISM.get(f, "operation")]
        for f in sorted_features
    ]

    # Honour caller's figsize. Default (8, 5.5) suits the story tornado.
    fig, ax = plt.subplots(figsize=figsize)
    y = np.arange(len(sorted_features))
    ax.barh(y, medians, height=0.65, color=bar_colors, alpha=0.9, edgecolor="white")
    ax.errorbar(medians, y, xerr=[
        np.array(medians) - np.array(p25),
        np.array(p75) - np.array(medians),
    ], fmt="none", color="black", capsize=3, alpha=0.6)
    ax.set_yticks(y)
    ax.set_yticklabels([OAT_FEATURE_LABELS.get(f, f) for f in sorted_features])
    ax.invert_yaxis()
    ax.set_xlabel("Median absolute swing in reduction (pp), with IQR across cities")

    # Class legend (only show classes actually present)
    present_mechs = []
    for f in sorted_features:
        m = OAT_FEATURE_MECHANISM.get(f, "operation")
        if m not in present_mechs:
            present_mechs.append(m)
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=OAT_MECHANISM_COLORS[m], alpha=0.9)
        for m in present_mechs
    ]
    labels = [OAT_MECHANISM_LABELS[m] for m in present_mechs]
    ax.legend(handles, labels, loc="lower right", frameon=False, fontsize=9,
              title="Sensitivity class", title_fontsize=9)

    if plot_title:
        ax.set_title("OAT sensitivity tornado")
    ax.grid(axis="x", linestyle=":", alpha=0.5)
    plt.tight_layout()
    return _finalize(save_path, for_latex=for_latex, plot_title=plot_title)


def _oat_panels(
    oat: pd.DataFrame,
    value_col: str,
    ylabel: str,
    save_path: str | None,
    dpi: int,
    for_latex: bool,
    plot_title: bool,
    features: Iterable[str] | None = None,
):
    feature_list = list(features) if features is not None else list(OAT_FEATURES)
    n = len(feature_list)
    ncols = 3
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(13, 2.8 * nrows), sharey=True)
    axes = np.atleast_2d(axes).flatten()

    # Effective baseline values per feature: read from the CSV's baseline rows
    # (which carry the experiment.toml-resolved value, not a hardcoded default).
    base_row = oat[oat["scenario_id"] == "baseline"].iloc[0]

    for ax, feature in zip(axes, feature_list):
        sub_dev = oat[oat["feature_varied"] == feature]
        sub_base = oat[oat["scenario_id"] == "baseline"].copy()
        sub_base["feature_varied"] = feature  # so it joins under this panel
        sub = pd.concat([sub_dev, sub_base], ignore_index=True)
        levels = sorted(sub[feature].unique())
        med = [sub[sub[feature] == lv][value_col].median() for lv in levels]
        p25 = [sub[sub[feature] == lv][value_col].quantile(0.25) for lv in levels]
        p75 = [sub[sub[feature] == lv][value_col].quantile(0.75) for lv in levels]
        ax.plot(levels, med, marker="o", color="#4C72B0", label="median")
        ax.fill_between(levels, p25, p75, color="#4C72B0", alpha=0.2, label="IQR")
        ax.axvline(base_row[feature], linestyle="--", color="grey", alpha=0.6)
        if plot_title:
            ax.set_title(OAT_FEATURE_LABELS.get(feature, feature))
        ax.set_xlabel(OAT_FEATURE_LABELS.get(feature, feature))
        ax.grid(linestyle=":", alpha=0.5)
    # Hide unused axes
    for ax in axes[n:]:
        ax.axis("off")
    # Y-label only on left column
    for i, ax in enumerate(axes):
        if i % ncols == 0 and i < n:
            ax.set_ylabel(ylabel)
    if plot_title:
        fig.suptitle(ylabel + " — OAT response (median ± IQR across cities)")
    fig.tight_layout()
    return _finalize(save_path, dpi, for_latex, plot_title)


def plot_oat_reduction_panels(oat, save_path=None, dpi=150, for_latex=False, plot_title=False):
    return _oat_panels(oat, "reduction_vs_anchor_ebike_pct", "Reduction vs anchor (%)",
                        save_path, dpi, for_latex, plot_title)


def plot_oat_threshold_panels(oat, save_path=None, dpi=150, for_latex=False, plot_title=False):
    oat2 = oat.copy()
    oat2["optimal_threshold_km"] = oat2["optimal_threshold_km"].replace([np.inf, -np.inf], np.nan)
    return _oat_panels(oat2, "optimal_threshold_km", "Optimal threshold (km)",
                        save_path, dpi, for_latex, plot_title)


def plot_oat_drone_ratio_panels(oat, save_path=None, dpi=150, for_latex=False, plot_title=False):
    return _oat_panels(oat, "optimal_drone_ratio_pct", "Optimal drone share (%)",
                        save_path, dpi, for_latex, plot_title)


def plot_oat_beneficial_share_panels(
    oat: pd.DataFrame,
    save_path: str | None = None,
    dpi: int = 150,
    for_latex: bool = False,
    plot_title: bool = False,
    features: Iterable[str] | None = None,
):
    feature_list = list(features) if features is not None else list(OAT_FEATURES)
    n = len(feature_list)
    ncols = 3
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(13, 2.8 * nrows), sharey=True)
    axes = np.atleast_2d(axes).flatten()
    base_row = oat[oat["scenario_id"] == "baseline"].iloc[0]
    for ax, feature in zip(axes, feature_list):
        sub_dev = oat[oat["feature_varied"] == feature]
        sub_base = oat[oat["scenario_id"] == "baseline"].copy()
        sub_base["feature_varied"] = feature
        sub = pd.concat([sub_dev, sub_base], ignore_index=True)
        levels = sorted(sub[feature].unique())
        share = [sub[sub[feature] == lv]["beneficial"].mean() * 100 for lv in levels]
        ax.plot(levels, share, marker="o", color="#55A467")
        ax.axvline(base_row[feature], linestyle="--", color="grey", alpha=0.6)
        ax.set_ylim(0, 105)
        if plot_title:
            ax.set_title(OAT_FEATURE_LABELS.get(feature, feature))
        ax.set_xlabel(OAT_FEATURE_LABELS.get(feature, feature))
        ax.grid(linestyle=":", alpha=0.5)
    for ax in axes[n:]:
        ax.axis("off")
    for i, ax in enumerate(axes):
        if i % ncols == 0 and i < n:
            ax.set_ylabel("% of cities with reduction > 0")
    if plot_title:
        fig.suptitle("Beneficial-city share — OAT response")
    fig.tight_layout()
    return _finalize(save_path, dpi, for_latex, plot_title)


# ---------------------------------------------------------------------------
# Shapley plots
# ---------------------------------------------------------------------------

def _pivot_shapley(shapley: pd.DataFrame) -> pd.DataFrame:
    return shapley.pivot_table(
        index=["region_key", "province"],
        columns="mechanism",
        values="shapley_pct_pts",
        aggfunc="first",
    ).reset_index()


def plot_shapley_aggregate_stacked(
    shapley: pd.DataFrame,
    save_path: str | None = None,
    dpi: int = 150,
    for_latex: bool = False,
    plot_title: bool = False,
):
    pivot = _pivot_shapley(shapley)
    medians = {m: pivot[m].median() for m in (*MECHANISM_ORDER, "residual_all_off")}
    fig, ax = plt.subplots(figsize=(6, 5))
    bottom = 0
    for m in (*MECHANISM_ORDER, "residual_all_off"):
        v = medians[m]
        ax.bar(["Median across cities"], [v], bottom=bottom,
               color=MECHANISM_COLORS[m], label=MECHANISM_LABELS[m])
        bottom += v
    ax.axhline(pivot["total_baseline"].median(), linestyle="--", color="black",
                label=f"Median baseline reduction = {pivot['total_baseline'].median():.1f}%")
    ax.set_ylabel("Reduction (%)")
    if plot_title:
        ax.set_title("Shapley decomposition of drone reduction\n(median across cities)")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    return _finalize(save_path, dpi, for_latex, plot_title)


def plot_shapley_boxplot(
    shapley: pd.DataFrame,
    save_path: str | None = None,
    dpi: int = 150,
    for_latex: bool = False,
    plot_title: bool = False,
):
    pivot = _pivot_shapley(shapley)
    data = [pivot[m].dropna().values for m in MECHANISM_ORDER]
    fig, ax = plt.subplots(figsize=(8, 5))
    bp = ax.boxplot(data, tick_labels=[MECHANISM_LABELS[m] for m in MECHANISM_ORDER],
                     showfliers=False, patch_artist=True)
    for patch, m in zip(bp["boxes"], MECHANISM_ORDER):
        patch.set_facecolor(MECHANISM_COLORS[m])
        patch.set_alpha(0.7)
    ax.axhline(0, linestyle=":", color="black", alpha=0.5)
    ax.set_ylabel("Shapley contribution (pct pts)")
    if plot_title:
        ax.set_title("Shapley contribution by mechanism (across all cities)")
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    plt.setp(ax.get_xticklabels(), rotation=15, ha="right")
    fig.tight_layout()
    return _finalize(save_path, dpi, for_latex, plot_title)


def plot_shapley_per_city_stacked(
    shapley: pd.DataFrame,
    save_path_top: str | None = None,
    save_path_bottom: str | None = None,
    top_n: int = 50,
    dpi: int = 150,
    for_latex: bool = False,
    plot_title: bool = False,
):
    pivot = _pivot_shapley(shapley)
    pivot = pivot.sort_values("total_baseline", ascending=False)
    top = pivot.head(top_n)
    bot = pivot.tail(top_n)
    out_paths = []
    for label, sub, save_path in (("top", top, save_path_top), ("bottom", bot, save_path_bottom)):
        fig, ax = plt.subplots(figsize=(max(8, len(sub) * 0.18), 5))
        x = np.arange(len(sub))
        bottom_neg = np.zeros(len(sub))
        bottom_pos = np.zeros(len(sub))
        for m in (*MECHANISM_ORDER, "residual_all_off"):
            vals = sub[m].fillna(0).to_numpy()
            pos_mask = vals >= 0
            neg_mask = ~pos_mask
            ax.bar(x[pos_mask], vals[pos_mask], bottom=bottom_pos[pos_mask],
                   color=MECHANISM_COLORS[m], label=MECHANISM_LABELS[m] if label == "top" else None)
            ax.bar(x[neg_mask], vals[neg_mask], bottom=bottom_neg[neg_mask],
                   color=MECHANISM_COLORS[m])
            bottom_pos[pos_mask] += vals[pos_mask]
            bottom_neg[neg_mask] += vals[neg_mask]
        ax.set_xticks(x)
        ax.set_xticklabels(sub["region_key"], rotation=90, fontsize=6)
        ax.axhline(0, linestyle="-", color="black", linewidth=0.6)
        ax.set_ylabel("Shapley contribution (pct pts)")
        if plot_title:
            ax.set_title(f"Per-city Shapley decomposition — {label} {top_n} cities by total reduction")
        if label == "top":
            ax.legend(loc="upper right", fontsize=7)
        fig.tight_layout()
        out_paths.append(_finalize(save_path, dpi, for_latex, plot_title))
    return out_paths


def plot_shapley_dominance_scatter(
    shapley: pd.DataFrame,
    save_path: str | None = None,
    dpi: int = 150,
    for_latex: bool = False,
    plot_title: bool = False,
):
    pivot = _pivot_shapley(shapley)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(pivot["M1_routing"], pivot["M4_gef"], c=pivot["total_baseline"],
                cmap="viridis", s=14, alpha=0.7)
    ax.axhline(0, linestyle=":", color="black", alpha=0.4)
    ax.axvline(0, linestyle=":", color="black", alpha=0.4)
    ax.set_xlabel("M1 routing Shapley (pct pts)")
    ax.set_ylabel("M4 GEF Shapley (pct pts)")
    cbar = fig.colorbar(ax.collections[0], ax=ax, label="total baseline reduction (%)")
    if plot_title:
        ax.set_title("City-level mechanism mix (color = total reduction)")
    ax.grid(linestyle=":", alpha=0.5)
    fig.tight_layout()
    return _finalize(save_path, dpi, for_latex, plot_title)


# ---------------------------------------------------------------------------
# Sobol + Monte Carlo plots (global SA, replaces Shapley in main text)
# ---------------------------------------------------------------------------

# Feature → 3-class mapping for the Sobol aggregation figures.
SOBOL_MECHANISM_GROUPS = {
    "operation": (
        "drone_energy_mult", "drone_cruise_kmh", "drone_hover_s",
        "headwind_mps", "payload_kg",
    ),
    "production_eol": (
        "drone_body_kg", "drone_battery_mass",
        "drone_production_co2_per_kg", "drone_battery_co2_per_kwh",
    ),
    "amortization": (
        "drone_lifespan_yr", "drone_battery_lifespan_yr",
        "drone_utilization_rate",
    ),
}


def _pivot_mc_summary(sobol: pd.DataFrame) -> pd.DataFrame | None:
    """Pivot the long-form mc_summary rows to one row per region."""
    mc = sobol[sobol["row_kind"] == "mc_summary"]
    if mc.empty:
        return None
    wide = mc.pivot_table(index="region_key", columns="feature", values="value")
    old_anchor_col = "anchor_kappa1_reduction_pct"
    old_baseline_kappa_col = "anchor_baseline_kappa_reduction_pct"
    baseline_col = "baseline_drone_tech_reduction_pct"
    if baseline_col not in wide.columns and old_baseline_kappa_col in wide.columns:
        wide[baseline_col] = wide[old_baseline_kappa_col]
    if baseline_col not in wide.columns and old_anchor_col in wide.columns:
        wide[baseline_col] = wide[old_anchor_col]
    needed = ["mc_median_reduction_pct", "mc_p2.5", "mc_p97.5",
              baseline_col, "mc_mean_reduction_pct"]
    missing = [c for c in needed if c not in wide.columns]
    if missing:
        print(f"  _pivot_mc_summary: missing columns {missing}")
        return None
    return wide


def plot_sobol_indices(
    sobol: pd.DataFrame,
    save_path: str | None = None,
    figsize: tuple = (8, 5.5),
    dpi: int = 150,
    for_latex: bool = False,
    plot_title: bool = False,
):
    """Sobol S1 and ST per feature, median across cities with IQR whiskers.

    Bars are colored by drone-technology class. S1 is plotted as a dark inset
    over the lighter ST so the interaction share is visually the gap between
    the two. Numeric ST median is annotated to the right of each bar.

    Args:
        sobol: Long-format Sobol DataFrame from ``all_sensitivity_sobol.csv``.
        save_path: Path to save figure. Shows plot if None.
        figsize: Figure size tuple.
        for_latex: If True, save as PDF.
        plot_title: If True, show title.
    """
    si = sobol[sobol["row_kind"] == "sobol_index"].copy()
    if si.empty:
        print("  plot_sobol_indices: no sobol_index rows; skipping")
        return None
    feature_to_class = {f: c for c, feats in SOBOL_MECHANISM_GROUPS.items() for f in feats}
    si = si[si["feature"].isin(feature_to_class)].copy()
    if si.empty:
        print("  plot_sobol_indices: no drone-tech features; skipping")
        return None

    grouped = si.groupby("feature")
    feature_medians = grouped["ST"].median().sort_values(ascending=True)
    features = list(feature_medians.index)
    st_median = feature_medians.values
    st_p25 = grouped["ST"].quantile(0.25).reindex(features).values
    st_p75 = grouped["ST"].quantile(0.75).reindex(features).values
    # Negative S1 is a finite-N estimator artifact; clamp at 0 for display.
    s1_median = np.clip(grouped["S1"].median().reindex(features).values, 0, None)

    bar_colors = [
        OAT_MECHANISM_COLORS[OAT_FEATURE_MECHANISM.get(f, "operation")]
        for f in features
    ]

    fig, ax = plt.subplots(figsize=figsize)
    y = np.arange(len(features))
    ax.barh(y, st_median, color=bar_colors, alpha=0.45, edgecolor="black",
            linewidth=1.0)
    ax.barh(y, s1_median, color=bar_colors, alpha=0.95, edgecolor="black",
            linewidth=1.0)
    err_lower = np.clip(st_median - st_p25, 0, None)
    err_upper = np.clip(st_p75 - st_median, 0, None)
    ax.errorbar(st_median, y, xerr=[err_lower, err_upper], fmt="none",
                ecolor="black", capsize=3, linewidth=1.0)
    # Numeric ST labels to the right of each bar
    x_max = float(np.max(st_p75)) if len(st_p75) else 1.0
    label_pad = x_max * 0.015
    for yi, st in zip(y, st_median):
        ax.text(st + err_upper[yi] + label_pad, yi, f"{st:.2f}",
                va="center", ha="left", fontsize=8)
    ax.set_yticks(y)
    ax.set_yticklabels([OAT_FEATURE_LABELS.get(f, f) for f in features])
    ax.set_xlabel("Sobol index (variance share of city-level reduction)")
    ax.axvline(0, color="black", linewidth=0.5)
    ax.set_xlim(0, x_max * 1.18)

    legend_handles = [
        Patch(facecolor=OAT_MECHANISM_COLORS[m], edgecolor="black",
              label=OAT_MECHANISM_LABELS[m])
        for m in ("operation", "production_eol", "amortization")
    ]
    legend_handles.append(Patch(facecolor="lightgray", edgecolor="black", alpha=0.45,
                                 label="$S_T$ (incl. interactions)"))
    legend_handles.append(Patch(facecolor="lightgray", edgecolor="black", alpha=0.95,
                                 label="$S_1$ (main effect)"))
    ax.legend(handles=legend_handles, loc="lower right", fontsize=8, frameon=True)

    if plot_title:
        n_cities = si["region_key"].nunique()
        ax.set_title(f"Sobol global SA: drone-technology variance decomposition "
                     f"(n={n_cities} cities, median ± IQR)")
    fig.tight_layout()
    return _finalize(save_path, dpi, for_latex, plot_title)


def plot_sobol_by_mechanism(
    sobol: pd.DataFrame,
    save_path: str | None = None,
    figsize: tuple = (7, 3.5),
    dpi: int = 150,
    for_latex: bool = False,
    plot_title: bool = False,
):
    """Sobol total-order indices aggregated to drone-technology classes.

    Sums ST across features within each class per city, then plots median
    + IQR across cities.

    Args:
        sobol: Long-format Sobol DataFrame from ``all_sensitivity_sobol.csv``.
        save_path: Path to save figure. Shows plot if None.
        figsize: Figure size tuple.
        for_latex: If True, save as PDF.
        plot_title: If True, show title.
    """
    si = sobol[sobol["row_kind"] == "sobol_index"].copy()
    if si.empty:
        print("  plot_sobol_by_mechanism: no sobol_index rows; skipping")
        return None

    feature_to_class = {f: c for c, feats in SOBOL_MECHANISM_GROUPS.items() for f in feats}
    si["class"] = si["feature"].map(feature_to_class)
    si = si.dropna(subset=["class"])
    if si.empty:
        print("  plot_sobol_by_mechanism: no drone-tech features; skipping")
        return None

    # Sum ST per (region, class), then aggregate across regions.
    per_city = si.groupby(["region_key", "class"])["ST"].sum().reset_index()
    grouped = per_city.groupby("class")["ST"]
    medians = grouped.median()
    p25 = grouped.quantile(0.25)
    p75 = grouped.quantile(0.75)
    order = ("operation", "production_eol", "amortization")
    medians = medians.reindex(order)
    p25 = p25.reindex(order)
    p75 = p75.reindex(order)

    fig, ax = plt.subplots(figsize=figsize)
    y = np.arange(len(order))
    colors = [OAT_MECHANISM_COLORS[m] for m in order]
    ax.barh(y, medians.values, color=colors, edgecolor="black", linewidth=1.0)
    err_lower = np.clip(medians.values - p25.values, 0, None)
    err_upper = np.clip(p75.values - medians.values, 0, None)
    ax.errorbar(medians.values, y, xerr=[err_lower, err_upper], fmt="none",
                ecolor="black", capsize=3, linewidth=1.0)
    for yi, m in zip(y, medians.values):
        ax.text(m + err_upper[yi] + 0.01, yi, f"{m:.2f}",
                va="center", ha="left", fontsize=9)
    ax.set_yticks(y)
    ax.set_yticklabels([OAT_MECHANISM_LABELS[m] for m in order])
    ax.set_xlabel("Aggregated total-order Sobol index $\\sum S_T$")
    ax.axvline(0, color="black", linewidth=0.5)
    ax.set_xlim(0, float(p75.max()) * 1.25)

    if plot_title:
        n_cities = si["region_key"].nunique()
        ax.set_title(f"Sobol aggregated to drone-technology classes "
                     f"(n={n_cities} cities, median ± IQR)")
    fig.tight_layout()
    return _finalize(save_path, dpi, for_latex, plot_title)


def plot_drone_tech_sensitivity_panels(
    sobol: pd.DataFrame,
    save_path: str | None = None,
    figsize: tuple = (12, 7.8),
    dpi: int = 150,
    for_latex: bool = False,
    plot_title: bool = False,
):
    """Four-panel Sobol summary for drone-technology uncertainty classes.

    Args:
        sobol: Long-format Sobol DataFrame from ``all_sensitivity_sobol.csv``.
        save_path: Path to save figure. Shows plot if None.
        figsize: Figure size tuple.
        for_latex: If True, save as PDF.
        plot_title: If True, show title.
    """
    si = sobol[sobol["row_kind"] == "sobol_index"].copy()
    if si.empty:
        print("  plot_drone_tech_sensitivity_panels: no sobol_index rows; skipping")
        return None

    feature_to_class = {f: c for c, feats in SOBOL_MECHANISM_GROUPS.items() for f in feats}
    si = si[si["feature"].isin(feature_to_class)].copy()
    si["class"] = si["feature"].map(feature_to_class)
    if si.empty:
        print("  plot_drone_tech_sensitivity_panels: no drone-tech features; skipping")
        return None

    class_order = ("operation", "production_eol", "amortization")
    grouped = si.groupby("feature")
    st_median = grouped["ST"].median()
    st_p25 = grouped["ST"].quantile(0.25)
    st_p75 = grouped["ST"].quantile(0.75)
    s1_median = np.clip(grouped["S1"].median(), 0, None)

    per_city_class = si.groupby(["region_key", "class"])["ST"].sum().reset_index()
    class_grouped = per_city_class.groupby("class")["ST"]
    class_median = class_grouped.median().reindex(class_order)
    class_p25 = class_grouped.quantile(0.25).reindex(class_order)
    class_p75 = class_grouped.quantile(0.75).reindex(class_order)

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 2, width_ratios=(1.0, 1.35), hspace=0.32, wspace=0.28)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    panel_axes = {
        "operation": (ax_b, "B"),
        "production_eol": (ax_c, "C"),
        "amortization": (ax_d, "D"),
    }
    panel_notes = {
        "operation": "Energy, headwind, and hover terms dominate use-phase variation.",
        "production_eol": "Mass terms separate from embodied-carbon coefficients.",
        "amortization": "Lower aggregate influence across cities.",
    }

    class_y = np.arange(len(class_order))
    class_vals = class_median.values.astype(float)
    class_err_lower = np.clip(class_vals - class_p25.values.astype(float), 0, None)
    class_err_upper = np.clip(class_p75.values.astype(float) - class_vals, 0, None)
    class_colors = [OAT_MECHANISM_COLORS[cls] for cls in class_order]

    ax_a.barh(class_y, class_vals, color=class_colors, alpha=0.75,
              edgecolor="black", linewidth=0.9)
    ax_a.errorbar(class_vals, class_y, xerr=[class_err_lower, class_err_upper], fmt="none",
                  ecolor="black", capsize=3, linewidth=0.9)
    ax_a.set_yticks(class_y)
    ax_a.set_yticklabels([OAT_MECHANISM_LABELS[cls] for cls in class_order], fontsize=9)
    ax_a.invert_yaxis()
    ax_a.grid(axis="x", linestyle=":", alpha=0.45)
    ax_a.axvline(0, color="black", linewidth=0.5)
    ax_a.set_xlabel("Aggregate total-order importance")
    ax_a.text(0.0, 1.04, "A", transform=ax_a.transAxes, fontsize=11,
              fontweight="bold", ha="left", va="bottom")
    ax_a.text(0.07, 1.04, "Class-level overview", transform=ax_a.transAxes,
              fontsize=10, fontweight="bold", ha="left", va="bottom")

    class_xmax = float(np.nanmax(class_p75.values.astype(float))) if len(class_p75) else 0.0
    class_label_pad = max(class_xmax * 0.02, 0.01)
    for yi, val, upper in zip(class_y, class_vals, class_err_upper):
        ax_a.text(val + upper + class_label_pad, yi, f"{val:.2f}",
                  va="center", ha="left", fontsize=8)
    ax_a.set_xlim(0, max(0.2, class_xmax * 1.25))

    feature_xmax = float(st_p75.max()) if len(st_p75) else 0.0
    feature_xlim = max(0.05, feature_xmax * 1.18)
    feature_label_pad = max(feature_xlim * 0.015, 0.004)

    for cls in class_order:
        ax, letter = panel_axes[cls]
        features = [f for f in SOBOL_MECHANISM_GROUPS[cls] if f in st_median.index]
        features.sort(key=lambda f: (-st_median[f], OAT_FEATURE_LABELS.get(f, f)))
        if not features:
            ax.axis("off")
            continue

        y = np.arange(len(features))
        st_vals = np.array([st_median[f] for f in features], dtype=float)
        s1_vals = np.array([s1_median[f] for f in features], dtype=float)
        p25_vals = np.array([st_p25[f] for f in features], dtype=float)
        p75_vals = np.array([st_p75[f] for f in features], dtype=float)
        err_lower = np.clip(st_vals - p25_vals, 0, None)
        err_upper = np.clip(p75_vals - st_vals, 0, None)
        color = OAT_MECHANISM_COLORS[cls]

        ax.barh(y, st_vals, color=color, alpha=0.42, edgecolor="black",
                linewidth=0.8, zorder=2)
        ax.barh(y, s1_vals, color=color, alpha=0.95, edgecolor="black",
                linewidth=0.8, zorder=3)
        ax.errorbar(st_vals, y, xerr=[err_lower, err_upper], fmt="none",
                    ecolor="black", capsize=3, linewidth=0.8, zorder=4)
        ax.set_yticks(y)
        ax.set_yticklabels([OAT_FEATURE_LABELS.get(f, f) for f in features], fontsize=8)
        ax.invert_yaxis()
        ax.grid(axis="x", linestyle=":", alpha=0.45)
        ax.axvline(0, color="black", linewidth=0.5)
        ax.set_xlim(0, feature_xlim)
        ax.set_xlabel("Sobol index")
        ax.text(0.0, 1.04, letter, transform=ax.transAxes, fontsize=11,
                fontweight="bold", ha="left", va="bottom")
        ax.text(0.07, 1.04, OAT_MECHANISM_LABELS[cls], transform=ax.transAxes,
                fontsize=10, fontweight="bold", ha="left", va="bottom")
        ax.text(
            0.98,
            0.02,
            panel_notes[cls],
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=7.5,
            bbox=dict(facecolor="white", edgecolor="0.75", pad=2, alpha=0.92),
        )

        for yi, st_val, upper in zip(y, st_vals, err_upper):
            ax.text(st_val + upper + feature_label_pad, yi, f"{st_val:.2f}",
                    va="center", ha="left", fontsize=7.5)

    handles = [
        Patch(facecolor="lightgray", edgecolor="black", alpha=0.42,
              label="$S_T$ (incl. interactions)"),
        Patch(facecolor="lightgray", edgecolor="black", alpha=0.95,
              label="$S_1$ (main effect)"),
    ]
    ax_a.legend(handles=handles, loc="lower right", fontsize=8, frameon=True)

    if plot_title:
        n_cities = si["region_key"].nunique()
        fig.suptitle(
            f"Drone-technology sensitivity across {n_cities} cities",
            y=0.99,
        )
    fig.tight_layout()
    return _finalize(save_path, dpi, for_latex, plot_title)


def plot_sobol_grouped(
    sobol: pd.DataFrame,
    save_path: str | None = None,
    figsize: tuple = (10.5, 5.2),
    dpi: int = 150,
    for_latex: bool = False,
    plot_title: bool = False,
):
    """Sobol indices grouped by drone-technology class, with a total overlay.

    Vertical bar chart, one bar per feature. Features are arranged in
    class groups; classes are ordered by descending ΣS_T (sum within
    class, median across cities). Within each group, features are sorted
    by descending median S_T. A polyline overlays the class totals at
    the center x-position of each group, so the reader sees both the per-
    feature breakdown and the per-class aggregate in one panel.

    Args:
        sobol: Long-format Sobol DataFrame from ``all_sensitivity_sobol.csv``.
        save_path: Path to save figure. Shows plot if None.
        figsize: Figure size tuple.
        for_latex: If True, save as PDF.
        plot_title: If True, show title.
    """
    si = sobol[sobol["row_kind"] == "sobol_index"].copy()
    if si.empty:
        print("  plot_sobol_grouped: no sobol_index rows; skipping")
        return None

    feature_to_class = {f: c for c, feats in SOBOL_MECHANISM_GROUPS.items() for f in feats}
    si["class"] = si["feature"].map(feature_to_class)
    si = si.dropna(subset=["class"])
    if si.empty:
        print("  plot_sobol_grouped: no drone-tech features; skipping")
        return None

    # Per-feature median ST + median S1 across cities.
    feat_st = si.groupby("feature")["ST"].median()
    feat_s1 = np.clip(si.groupby("feature")["S1"].median(), 0, None)
    feat_st_p25 = si.groupby("feature")["ST"].quantile(0.25)
    feat_st_p75 = si.groupby("feature")["ST"].quantile(0.75)

    # Per-class aggregate ST (sum per city, then median across cities).
    per_city_class = si.groupby(["region_key", "class"])["ST"].sum().reset_index()
    class_total = per_city_class.groupby("class")["ST"].median().sort_values(ascending=False)
    class_order = class_total.index.tolist()

    # Feature ordering: by class (sorted by total), within class by ST desc.
    feature_order: list[str] = []
    class_x_centers: list[float] = []
    group_boundaries: list[float] = [0.0]
    for cls in class_order:
        feats = [f for f in feat_st.index if feature_to_class.get(f) == cls]
        feats.sort(key=lambda f: -feat_st[f])
        start = len(feature_order)
        feature_order.extend(feats)
        end = len(feature_order)
        if end > start:
            class_x_centers.append((start + end - 1) / 2.0)
        group_boundaries.append(float(end))

    x = np.arange(len(feature_order))
    st_vals = np.array([feat_st[f] for f in feature_order])
    s1_vals = np.array([feat_s1[f] for f in feature_order])
    p25_vals = np.array([feat_st_p25[f] for f in feature_order])
    p75_vals = np.array([feat_st_p75[f] for f in feature_order])
    colors = [OAT_MECHANISM_COLORS[feature_to_class[f]] for f in feature_order]

    fig, ax = plt.subplots(figsize=figsize)

    # ST bars (light shade) and S1 bars (dark shade, inset).
    ax.bar(x, st_vals, color=colors, alpha=0.45, edgecolor="black",
           linewidth=0.8, label="_nolegend_", zorder=2)
    ax.bar(x, s1_vals, color=colors, alpha=0.95, edgecolor="black",
           linewidth=0.8, label="_nolegend_", zorder=3)

    # IQR error bars on ST
    err_lower = np.clip(st_vals - p25_vals, 0, None)
    err_upper = np.clip(p75_vals - st_vals, 0, None)
    ax.errorbar(x, st_vals, yerr=[err_lower, err_upper], fmt="none",
                ecolor="black", capsize=3, linewidth=0.8, zorder=4)

    # Numeric ST annotation above each bar (rounded to 3 dp; show "<0.01" for tiny)
    y_max = max(float(np.max(p75_vals)) if len(p75_vals) else 0.0,
                float(class_total.max()))
    label_pad = y_max * 0.015
    for xi, v in zip(x, st_vals):
        label = f"{v:.2f}" if v >= 0.005 else "<.01"
        ax.text(xi, v + label_pad, label, ha="center", va="bottom", fontsize=7.5)

    # Class-total polyline overlay.
    line_y = [float(class_total[cls]) for cls in class_order]
    ax.plot(class_x_centers, line_y, "-o", color="black", linewidth=1.6,
            markersize=7, markerfacecolor="white", markeredgewidth=1.6,
            zorder=5, label=r"Class total $\sum S_T$")

    # Annotate each class-total point
    for xc, yv in zip(class_x_centers, line_y):
        ax.annotate(f"{yv:.2f}", xy=(xc, yv), xytext=(0, 11),
                    textcoords="offset points", ha="center", fontsize=9,
                    fontweight="bold")

    # Vertical separators between class groups
    for b in group_boundaries[1:-1]:
        ax.axvline(b - 0.5, color="gray", linestyle=":", alpha=0.55, zorder=1)

    # Class labels on a second header row above the plot
    ax_top = ax.twiny()
    ax_top.set_xlim(ax.get_xlim())
    ax_top.set_xticks(class_x_centers)
    ax_top.set_xticklabels(
        [OAT_MECHANISM_LABELS[cls] for cls in class_order],
        fontsize=9,
    )
    ax_top.tick_params(axis="x", which="both", length=0, pad=4)
    ax_top.spines["top"].set_visible(False)

    # Feature labels on the bottom x-axis (rotated)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [OAT_FEATURE_LABELS.get(f, f) for f in feature_order],
        rotation=35, ha="right", fontsize=8,
    )
    ax.set_xlim(-0.7, len(feature_order) - 0.3)
    ax.set_ylim(0, y_max * 1.30)
    ax.set_ylabel("Sobol total-order index $S_T$ (variance share)")

    # Full legend: class color patches + S1/ST shading + class-total marker.
    legend_handles = [
        Patch(facecolor=OAT_MECHANISM_COLORS[m], edgecolor="black",
              label=OAT_MECHANISM_LABELS[m])
        for m in class_order
    ]
    legend_handles.append(Patch(facecolor="lightgray", edgecolor="black",
                                 alpha=0.45, label="$S_T$ (incl. interactions)"))
    legend_handles.append(Patch(facecolor="lightgray", edgecolor="black",
                                 alpha=0.95, label="$S_1$ (main effect)"))
    legend_handles.append(plt.Line2D([], [], color="black", marker="o",
                                      markerfacecolor="white", markersize=7,
                                      linewidth=1.6,
                                      label=r"Class total $\sum S_T$"))
    ax.legend(handles=legend_handles, loc="upper right", fontsize=8,
              frameon=True, ncol=1)

    if plot_title:
        n_cities = si["region_key"].nunique()
        ax.set_title(f"Sobol global SA: per-feature S_T grouped by drone-tech class, "
                     f"with class aggregates (n={n_cities} cities)")
    fig.tight_layout()
    return _finalize(save_path, dpi, for_latex, plot_title)


def plot_sobol_strip(
    sobol: pd.DataFrame,
    save_path: str | None = None,
    figsize: tuple = (7.5, 3.5),
    dpi: int = 150,
    for_latex: bool = False,
    plot_title: bool = False,
    top_n: int = 3,
):
    """Per-city ST strip plot for the top-N features by median ST.

    Each dot is one city's ST for that feature. Shows whether the dominant
    parameters are universal across cities or driven by outliers.

    Args:
        sobol: Long-format Sobol DataFrame from ``all_sensitivity_sobol.csv``.
        save_path: Path to save figure. Shows plot if None.
        figsize: Figure size tuple.
        for_latex: If True, save as PDF.
        plot_title: If True, show title.
        top_n: Number of top-ranked features (by median ST) to display.
    """
    si = sobol[sobol["row_kind"] == "sobol_index"].copy()
    if si.empty:
        print("  plot_sobol_strip: no sobol_index rows; skipping")
        return None
    feature_to_class = {f: c for c, feats in SOBOL_MECHANISM_GROUPS.items() for f in feats}
    si = si[si["feature"].isin(feature_to_class)].copy()
    if si.empty:
        print("  plot_sobol_strip: no drone-tech features; skipping")
        return None

    medians = si.groupby("feature")["ST"].median().sort_values(ascending=False)
    top_features = list(medians.head(top_n).index)
    fig, ax = plt.subplots(figsize=figsize)
    rng = np.random.default_rng(42)
    for i, f in enumerate(top_features):
        vals = si[si["feature"] == f]["ST"].to_numpy()
        jitter = rng.uniform(-0.18, 0.18, size=len(vals))
        color = OAT_MECHANISM_COLORS[OAT_FEATURE_MECHANISM.get(f, "operation")]
        ax.scatter(vals, np.full_like(vals, i) + jitter, color=color, alpha=0.45,
                   s=14, edgecolor="none")
        med = np.median(vals)
        ax.plot([med, med], [i - 0.32, i + 0.32], color="black", linewidth=2.0)
        ax.text(med, i + 0.42, f"median {med:.2f}", ha="center", fontsize=8)
    ax.set_yticks(np.arange(len(top_features)))
    ax.set_yticklabels([OAT_FEATURE_LABELS.get(f, f) for f in top_features])
    ax.set_xlabel("Per-city total-order Sobol index $S_T$")
    ax.set_xlim(left=0)
    ax.invert_yaxis()

    if plot_title:
        n_cities = si["region_key"].nunique()
        ax.set_title(f"Per-city ST for top-{top_n} features (n={n_cities} cities)")
    fig.tight_layout()
    return _finalize(save_path, dpi, for_latex, plot_title)


def plot_mc_uncertainty(
    sobol: pd.DataFrame,
    save_path: str | None = None,
    figsize: tuple = (9, 4.5),
    dpi: int = 150,
    for_latex: bool = False,
    plot_title: bool = False,
):
    """Cross-city distribution of MC reduction% vs the baseline technology case.

    Two panels framing the headline robustness finding: the Saltelli uniform
    sample gives a lower median reduction than the baseline drone-technology
    case with all sampled drone assumptions at their configured values.

    Left: CDF of per-city MC medians with the 95% MC band as a ribbon, and
    the CDF of per-city baseline reductions overlaid for direct comparison.

    Right: paired histogram of MC median and baseline reduction with their
    cross-city means annotated.

    Args:
        sobol: Long-format Sobol DataFrame from ``all_sensitivity_sobol.csv``.
        save_path: Path to save figure. Shows plot if None.
        figsize: Figure size tuple.
        for_latex: If True, save as PDF.
        plot_title: If True, show title.
    """
    wide = _pivot_mc_summary(sobol)
    if wide is None:
        print("  plot_mc_uncertainty: skipping (missing mc_summary)")
        return None

    n = len(wide)
    baseline_col = "baseline_drone_tech_reduction_pct"
    by_mc = wide.sort_values("mc_median_reduction_pct").reset_index(drop=True)
    by_anchor = wide.sort_values(baseline_col).reset_index(drop=True)
    rank = np.arange(1, n + 1) / n * 100.0
    mc_color = "#C44E52"  # fixed red for MC distribution (was OAT_MECHANISM_COLORS["grid"])
    anchor_color = OAT_MECHANISM_COLORS["anchor"]
    anchor_mean = float(wide[baseline_col].mean())
    anchor_median = float(wide[baseline_col].median())
    mc_median_overall = float(wide["mc_median_reduction_pct"].median())

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=figsize)

    # Left: CDFs of MC median and anchor, with per-city MC band as ribbon
    ax_l.fill_betweenx(rank, by_mc["mc_p2.5"], by_mc["mc_p97.5"],
                       color=mc_color, alpha=0.12,
                       label="Per-city 95% MC band")
    ax_l.plot(by_mc["mc_median_reduction_pct"], rank,
              color=mc_color, linewidth=1.8,
              label=f"MC median (cross-city median {mc_median_overall:.1f}%)")
    ax_l.plot(by_anchor[baseline_col], rank,
              color=anchor_color, linewidth=1.8, linestyle="--",
              label=f"Baseline drone tech (cross-city median {anchor_median:.1f}%)")
    ax_l.set_xlabel("Reduction vs e-bike-only (%)")
    ax_l.set_ylabel("City percentile")
    ax_l.set_xlim(left=min(-5, float(by_mc["mc_p2.5"].min()) - 2))
    ax_l.legend(loc="lower right", fontsize=8, frameon=True)

    # Right: paired histogram of MC median and baseline
    bins = np.linspace(0, max(float(wide[baseline_col].max()),
                              float(wide["mc_p97.5"].max())) + 2, 40)
    ax_r.hist(wide["mc_median_reduction_pct"].dropna(), bins=bins,
              color=mc_color, alpha=0.55, edgecolor="black", linewidth=0.4,
              label=f"MC median (mean {wide['mc_median_reduction_pct'].mean():.1f}%)")
    ax_r.hist(wide[baseline_col].dropna(), bins=bins,
              color=anchor_color, alpha=0.55, edgecolor="black", linewidth=0.4,
              label=f"Baseline drone tech (mean {anchor_mean:.1f}%)")
    ax_r.axvline(anchor_mean, color=anchor_color, linestyle=":", linewidth=1.2)
    ax_r.set_xlabel("Reduction vs e-bike-only (%)")
    ax_r.set_ylabel("Number of cities")
    ax_r.legend(loc="upper right", fontsize=8, frameon=True)

    if plot_title:
        fig.suptitle(f"Sobol/MC uncertainty across {n} cities", y=1.02)
    fig.tight_layout()
    return _finalize(save_path, dpi, for_latex, plot_title)


def plot_mc_anchor_forest(
    sobol: pd.DataFrame,
    save_path: str | None = None,
    figsize: tuple = (9, 5.0),
    dpi: int = 150,
    for_latex: bool = False,
    plot_title: bool = False,
    show_every: int = 1,
):
    """Per-city baseline reduction with the MC 95% band as vertical whiskers.

    Cities ordered along x by baseline reduction. Each city gets a dot for the
    all-baseline drone-technology reduction and a vertical [p2.5, p97.5] MC whisker. The
    line plot of the MC median across the same ordering is overlaid so the
    reader sees whether the baseline case sits inside the naive MC
    band for every city.

    Args:
        sobol: Long-format Sobol DataFrame from ``all_sensitivity_sobol.csv``.
        save_path: Path to save figure. Shows plot if None.
        figsize: Figure size tuple.
        for_latex: If True, save as PDF.
        plot_title: If True, show title.
        show_every: Plot every Nth city (after sorting by baseline) to thin a
            dense forest. Default 1 keeps every city.
    """
    wide = _pivot_mc_summary(sobol)
    if wide is None:
        return None
    baseline_col = "baseline_drone_tech_reduction_pct"
    wide = wide.sort_values(baseline_col).reset_index(drop=True)
    if show_every > 1:
        wide = wide.iloc[::show_every].reset_index(drop=True)
    n = len(wide)
    x = np.arange(n)

    fig, ax = plt.subplots(figsize=figsize)
    anchor_color = OAT_MECHANISM_COLORS["anchor"]
    mc_color = "#C44E52"  # fixed red for MC distribution (was OAT_MECHANISM_COLORS["grid"])
    ax.vlines(x, wide["mc_p2.5"], wide["mc_p97.5"],
              color=mc_color, alpha=0.30, linewidth=0.9,
              label="95% MC band")
    ax.plot(x, wide["mc_median_reduction_pct"], color=mc_color,
            linewidth=1.4, alpha=0.85, label="MC median")
    ax.scatter(x, wide[baseline_col],
               color=anchor_color, s=10, zorder=4,
               label="Baseline drone tech")
    ax.set_xlabel("City (ordered by baseline reduction)")
    ax.set_ylabel("Reduction vs e-bike-only (%)")
    ax.set_xlim(-1, n)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.legend(loc="upper left", fontsize=8, frameon=True)

    if plot_title:
        ax.set_title(f"Per-city MC band vs baseline drone tech (n={n} cities)")
    fig.tight_layout()
    return _finalize(save_path, dpi, for_latex, plot_title)


def plot_mc_anchor_scatter(
    sobol: pd.DataFrame,
    save_path: str | None = None,
    figsize: tuple = (5.5, 5.5),
    dpi: int = 150,
    for_latex: bool = False,
    plot_title: bool = False,
):
    """Scatter of baseline reduction (x) vs MC median reduction (y), per city.

    y = x diagonal drawn for reference. Cities below the diagonal have an MC
    median lower than the all-baseline drone-technology reduction.

    Args:
        sobol: Long-format Sobol DataFrame from ``all_sensitivity_sobol.csv``.
        save_path: Path to save figure. Shows plot if None.
        figsize: Figure size tuple.
        for_latex: If True, save as PDF.
        plot_title: If True, show title.
    """
    wide = _pivot_mc_summary(sobol)
    if wide is None:
        return None

    baseline_col = "baseline_drone_tech_reduction_pct"
    anchor = wide[baseline_col].to_numpy()
    mc_med = wide["mc_median_reduction_pct"].to_numpy()
    lo = float(min(np.nanmin(anchor), np.nanmin(mc_med)))
    hi = float(max(np.nanmax(anchor), np.nanmax(mc_med)))
    pad = (hi - lo) * 0.05

    fig, ax = plt.subplots(figsize=figsize)
    mc_color = "#C44E52"  # fixed red for MC distribution (was OAT_MECHANISM_COLORS["grid"])
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color="black",
            linewidth=0.8, linestyle="--", alpha=0.7, label="$y = x$")
    ax.scatter(anchor, mc_med, color=mc_color, alpha=0.55, s=18,
               edgecolor="none")
    ax.set_xlabel("Baseline drone-tech reduction (%)")
    ax.set_ylabel("MC median reduction (%)")
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="upper left", fontsize=9, frameon=True)

    # Quick fit / annotation: median gap
    gap = float(np.nanmedian(anchor - mc_med))
    ax.text(0.03, 0.85,
            f"n = {len(wide)} cities\nMedian gap = {gap:.1f} pp",
            transform=ax.transAxes, fontsize=9,
            bbox=dict(facecolor="white", edgecolor="black", alpha=0.85, pad=4))

    if plot_title:
        ax.set_title("MC median vs baseline drone tech per city")
    fig.tight_layout()
    return _finalize(save_path, dpi, for_latex, plot_title)


# ---------------------------------------------------------------------------
# Named-scenario plots
# ---------------------------------------------------------------------------

def plot_named_boxplot(
    named: pd.DataFrame,
    save_path: str | None = None,
    dpi: int = 150,
    for_latex: bool = False,
    plot_title: bool = False,
):
    fig, ax = plt.subplots(figsize=(11, 6))
    positions = []
    labels = []
    data = []
    colors = []
    pos = 0
    for kappa in sorted(named["kappa"].unique()):
        for strat in NAMED_STRATEGY_ORDER:
            sub = named[(named["kappa"] == kappa) & (named["strategy"] == strat)]
            data.append(sub["reduction_vs_anchor_ebike_pct"].dropna().to_numpy())
            positions.append(pos)
            labels.append(f"{strat}\nκ={kappa}")
            colors.append("#4C72B0" if np.isclose(kappa, 0.7) else "#DD8452")
            pos += 1
        pos += 1

    bp = ax.boxplot(data, positions=positions, widths=0.7, showfliers=False, patch_artist=True)
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.7)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.axhline(0, linestyle=":", color="black", alpha=0.5)
    ax.set_ylabel("Reduction vs anchor (%)")
    if plot_title:
        ax.set_title("Named scenarios — distribution across cities")
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    fig.tight_layout()
    return _finalize(save_path, dpi, for_latex, plot_title)


def plot_named_cdf(
    named: pd.DataFrame,
    save_path: str | None = None,
    dpi: int = 150,
    for_latex: bool = False,
    plot_title: bool = False,
):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for ax, kappa in zip(axes, sorted(named["kappa"].unique())):
        for strat in NAMED_STRATEGY_ORDER:
            sub = named[(named["kappa"] == kappa) & (named["strategy"] == strat)]
            vals = np.sort(sub["reduction_vs_anchor_ebike_pct"].dropna().to_numpy())
            if len(vals) == 0:
                continue
            ys = 1.0 - np.arange(len(vals)) / len(vals)
            ax.plot(vals, ys * 100, label=strat)
        ax.set_xlabel("Reduction vs anchor (%)")
        ax.set_ylabel("% of cities reaching ≥ x")
        if plot_title:
            ax.set_title(f"κ = {kappa}")
        ax.grid(linestyle=":", alpha=0.5)
        ax.legend(fontsize=7)
    fig.tight_layout()
    return _finalize(save_path, dpi, for_latex, plot_title)


def plot_named_2030_2060_scatter(
    named: pd.DataFrame,
    save_path: str | None = None,
    dpi: int = 150,
    for_latex: bool = False,
    plot_title: bool = False,
):
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharex=True, sharey=True)
    for ax, kappa in zip(axes, sorted(named["kappa"].unique())):
        a = named[(named["kappa"] == kappa) & (named["strategy"] == "optimal_mixed_2030")][
            ["region_key", "reduction_vs_anchor_ebike_pct"]].rename(columns={"reduction_vs_anchor_ebike_pct": "y2030"})
        b = named[(named["kappa"] == kappa) & (named["strategy"] == "optimal_mixed_2060")][
            ["region_key", "reduction_vs_anchor_ebike_pct"]].rename(columns={"reduction_vs_anchor_ebike_pct": "y2060"})
        m = a.merge(b, on="region_key")
        ax.scatter(m["y2030"], m["y2060"], s=12, alpha=0.6, color="#4C72B0")
        if len(m):
            lo = min(m["y2030"].min(), m["y2060"].min(), 0)
            hi = max(m["y2030"].max(), m["y2060"].max())
            ax.plot([lo, hi], [lo, hi], color="grey", linestyle="--", alpha=0.6)
        ax.set_xlabel("2030 grid reduction (%)")
        ax.set_ylabel("2060 grid reduction (%)")
        if plot_title:
            ax.set_title(f"κ = {kappa}")
        ax.grid(linestyle=":", alpha=0.5)
    if plot_title:
        fig.suptitle("Cross-scenario: 2030 vs 2060 grid reduction per city")
    fig.tight_layout()
    return _finalize(save_path, dpi, for_latex, plot_title)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def render_all(data_dir: str, figures_dir: str,
               for_latex: bool = False, plot_title: bool = False, dpi: int = 150) -> None:
    out = os.path.join(figures_dir, "sensitivity")
    os.makedirs(out, exist_ok=True)

    oat_path = os.path.join(data_dir, "all_sensitivity_oat.csv")
    sh_path = os.path.join(data_dir, "all_sensitivity_shapley.csv")
    nm_path = os.path.join(data_dir, "all_sensitivity_named.csv")
    sb_path = os.path.join(data_dir, "all_sensitivity_sobol.csv")

    if os.path.exists(oat_path):
        oat = pd.read_csv(oat_path)
        print(f"OAT: {len(oat)} rows from {oat_path}")
        # Full tornado (all drone-tech features) — for supplementary material.
        plot_oat_tornado(oat, save_path=_resolve_path(out, "oat_tornado_full", for_latex),
                         figsize=(8, 6.5), for_latex=for_latex, plot_title=plot_title)
        # Story tornado (curated subset) — for main-text Fig. 5a.
        plot_oat_tornado(oat, save_path=_resolve_path(out, "oat_tornado_story", for_latex),
                         figsize=(8, 5.5), for_latex=for_latex, plot_title=plot_title,
                         feature_whitelist=OAT_FEATURES_STORY)
        plot_oat_reduction_panels(oat, save_path=_resolve_path(out, "oat_reduction_panels", for_latex),
                                   dpi=dpi, for_latex=for_latex, plot_title=plot_title)
        plot_oat_threshold_panels(oat, save_path=_resolve_path(out, "oat_threshold_panels", for_latex),
                                   dpi=dpi, for_latex=for_latex, plot_title=plot_title)
        plot_oat_drone_ratio_panels(oat, save_path=_resolve_path(out, "oat_drone_ratio_panels", for_latex),
                                     dpi=dpi, for_latex=for_latex, plot_title=plot_title)
        plot_oat_beneficial_share_panels(oat, save_path=_resolve_path(out, "oat_beneficial_share", for_latex),
                                          dpi=dpi, for_latex=for_latex, plot_title=plot_title)
    else:
        print(f"  missing {oat_path}; skipping OAT plots")

    if os.path.exists(sh_path):
        sh = pd.read_csv(sh_path)
        print(f"Shapley: {len(sh)} rows from {sh_path}")
        plot_shapley_aggregate_stacked(sh, save_path=_resolve_path(out, "shapley_aggregate_stacked", for_latex),
                                        dpi=dpi, for_latex=for_latex, plot_title=plot_title)
        plot_shapley_boxplot(sh, save_path=_resolve_path(out, "shapley_boxplot", for_latex),
                              dpi=dpi, for_latex=for_latex, plot_title=plot_title)
        plot_shapley_per_city_stacked(
            sh,
            save_path_top=_resolve_path(out, "shapley_per_city_top", for_latex),
            save_path_bottom=_resolve_path(out, "shapley_per_city_bottom", for_latex),
            top_n=50, dpi=dpi, for_latex=for_latex, plot_title=plot_title,
        )
        plot_shapley_dominance_scatter(sh, save_path=_resolve_path(out, "shapley_dominance_scatter", for_latex),
                                        dpi=dpi, for_latex=for_latex, plot_title=plot_title)
    else:
        print(f"  missing {sh_path}; skipping Shapley plots")

    if os.path.exists(sb_path):
        sb = pd.read_csv(sb_path)
        print(f"Sobol: {len(sb)} rows from {sb_path}")
        plot_drone_tech_sensitivity_panels(
            sb, save_path=_resolve_path(out, "drone_tech_sensitivity", for_latex),
            dpi=dpi, for_latex=for_latex, plot_title=plot_title)
        plot_sobol_indices(sb, save_path=_resolve_path(out, "sobol_indices", for_latex),
                            dpi=dpi, for_latex=for_latex, plot_title=plot_title)
        plot_sobol_by_mechanism(sb, save_path=_resolve_path(out, "sobol_by_mechanism", for_latex),
                                 dpi=dpi, for_latex=for_latex, plot_title=plot_title)
        plot_sobol_strip(sb, save_path=_resolve_path(out, "sobol_strip", for_latex),
                          dpi=dpi, for_latex=for_latex, plot_title=plot_title)
        plot_mc_uncertainty(sb, save_path=_resolve_path(out, "mc_uncertainty", for_latex),
                             dpi=dpi, for_latex=for_latex, plot_title=plot_title)
        plot_mc_anchor_forest(sb, save_path=_resolve_path(out, "mc_anchor_forest", for_latex),
                               dpi=dpi, for_latex=for_latex, plot_title=plot_title)
        plot_mc_anchor_scatter(sb, save_path=_resolve_path(out, "mc_anchor_scatter", for_latex),
                                dpi=dpi, for_latex=for_latex, plot_title=plot_title)
    else:
        print(f"  missing {sb_path}; skipping Sobol/MC plots")

    if os.path.exists(nm_path):
        nm = pd.read_csv(nm_path)
        print(f"Named: {len(nm)} rows from {nm_path}")
        plot_named_boxplot(nm, save_path=_resolve_path(out, "named_boxplot", for_latex),
                            dpi=dpi, for_latex=for_latex, plot_title=plot_title)
        plot_named_cdf(nm, save_path=_resolve_path(out, "named_cdf", for_latex),
                        dpi=dpi, for_latex=for_latex, plot_title=plot_title)
        plot_named_2030_2060_scatter(nm, save_path=_resolve_path(out, "named_2030_2060_scatter", for_latex),
                                      dpi=dpi, for_latex=for_latex, plot_title=plot_title)
    else:
        print(f"  missing {nm_path}; skipping named-scenario plots")


def _bootstrap_paths():
    here = os.path.dirname(os.path.abspath(__file__))
    src = os.path.normpath(os.path.join(here, ".."))
    if src not in sys.path:
        sys.path.insert(0, src)


def main():
    _bootstrap_paths()
    from utils import load_config

    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default=None)
    parser.add_argument("--png", action="store_true",
                         help="Force PNG output (overrides experiment.toml for_latex)")
    parser.add_argument("--titled", action="store_true",
                         help="Force titled output with _t suffix (overrides experiment.toml plot_title)")
    args = parser.parse_args()

    project_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    )
    config = load_config(os.path.join(project_root, "experiment.toml"), env=args.env)
    for_latex = (not args.png) and config.get("for_latex", False)
    plot_title = args.titled or config.get("plot_title", False)
    print(f"for_latex={for_latex}, plot_title={plot_title}")
    render_all(config["data_dir"], config["figures_dir"],
                for_latex=for_latex, plot_title=plot_title)


if __name__ == "__main__":
    main()
