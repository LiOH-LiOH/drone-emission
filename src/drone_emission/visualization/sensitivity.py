"""Exploratory plots from the consolidated sensitivity outputs.

Reads ``data/all_sensitivity_oat.csv``, ``data/all_sensitivity_shapley.csv``,
and ``data/all_sensitivity_named.csv``. Produces ~12 candidate figures into
``figures/sensitivity/``. The user picks the final paper figures from this
set later.

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
    "kappa", "gef_mult", "drone_energy_mult", "payload_kg",
    "drone_lifespan_yr", "ebike_lifespan_yr",
    "drone_cruise_kmh", "drone_body_kg", "drone_hover_s",
    "drone_battery_lifespan_yr", "headwind_mps",
)

# Curated subset for the §4 story tornado: rule = median city-level swing
# exceeds 1 pp, plus kappa regardless (narrative continuity with §1). Five
# features are excluded by this rule: payload_kg, drone_lifespan_yr,
# ebike_lifespan_yr (below 1 pp) — the remaining nine carry the story.
OAT_FEATURES_STORY = (
    "gef_mult", "drone_energy_mult", "drone_body_kg",
    "kappa", "drone_hover_s", "headwind_mps",
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
}

# Mechanism-aware colors for the 6→4 mapping (Shapley alignment).
OAT_FEATURE_MECHANISM = {
    "kappa": "bundling",
    "gef_mult": "grid",
    "drone_energy_mult": "grid",
    "payload_kg": "production",
    "drone_lifespan_yr": "production",
    "ebike_lifespan_yr": "production",
    "drone_body_kg": "production",
    "drone_battery_lifespan_yr": "production",
    "drone_cruise_kmh": "operation",
    "drone_hover_s": "operation",
    "headwind_mps": "operation",
}
OAT_MECHANISM_COLORS = {
    "bundling": "#DD8452",
    "grid": "#C44E52",
    "production": "#55A467",
    "operation": "#4C72B0",
}
OAT_MECHANISM_LABELS = {
    "bundling": "Trip bundling",
    "grid": "Grid carbon",
    "production": "Production amortization",
    "operation": "Operation / environment",
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
):
    """OAT tornado: median absolute swing per feature with IQR whiskers.

    Bars are coloured by the Shapley mechanism each feature maps to, so the
    Fig. 5 tornado and the Fig. 5 Shapley stacked bar read jointly without
    re-explanation. ``feature_whitelist`` restricts the bars to a named
    subset (in any order; bars are still sorted by descending median swing).
    If ``None``, every feature present in ``oat`` is plotted.

    Args:
        oat: Long-format OAT DataFrame from ``all_sensitivity_oat.csv``.
        save_path: Path to save figure. Shows plot if None.
        figsize: Figure size tuple. Defaults to (8, 5.5); pass a taller size
            for the full 11-feature tornado.
        for_latex: If True, save as PDF.
        plot_title: If True, show title.
        feature_whitelist: Optional ordered iterable of feature names to
            include. Use ``OAT_FEATURES_STORY`` for the main-text figure.
    """
    features = (
        list(feature_whitelist)
        if feature_whitelist is not None
        else list(OAT_FEATURES)
    )
    base = oat[oat["scenario_id"] == "baseline"][["region_key", "reduction_vs_anchor_ebike_pct"]]
    base = base.rename(columns={"reduction_vs_anchor_ebike_pct": "baseline_red"})
    diffs: Dict[str, list[float]] = {}
    for feature in features:
        sub = oat[oat["feature_varied"] == feature]
        if sub.empty:
            continue
        merged = sub.merge(base, on="region_key", how="inner")
        merged["delta"] = merged["reduction_vs_anchor_ebike_pct"] - merged["baseline_red"]
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

    # Honour caller's figsize. Default (8, 5.5) suits the 9-feature story
    # tornado; for the 11-feature full tornado pass figsize=(8, 6.5).
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

    # Mechanism legend (only show mechanisms actually present)
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
              title="Shapley mechanism", title_fontsize=9)

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


def plot_oat_gef_kappa_heatmap(
    oat: pd.DataFrame,
    save_path: str | None = None,
    dpi: int = 150,
    for_latex: bool = False,
    plot_title: bool = False,
):
    base_row = oat[oat["scenario_id"] == "baseline"].iloc[0]
    kappa_base = float(base_row["kappa"])
    gef_base = float(base_row["gef_mult"])
    kappa_levels = sorted(set(oat[oat["feature_varied"] == "kappa"]["kappa"].unique().tolist() + [kappa_base]))
    gef_levels = sorted(set(oat[oat["feature_varied"] == "gef_mult"]["gef_mult"].unique().tolist() + [gef_base]))

    M = np.full((len(kappa_levels), len(gef_levels)), np.nan)
    for i, k in enumerate(kappa_levels):
        for j, g in enumerate(gef_levels):
            if k == kappa_base and g == gef_base:
                vals = oat[oat["scenario_id"] == "baseline"]["reduction_vs_anchor_ebike_pct"]
            elif g == gef_base and k != kappa_base:
                vals = oat[(oat["feature_varied"] == "kappa") & (oat["kappa"] == k)]["reduction_vs_anchor_ebike_pct"]
            elif k == kappa_base and g != gef_base:
                vals = oat[(oat["feature_varied"] == "gef_mult") & (oat["gef_mult"] == g)]["reduction_vs_anchor_ebike_pct"]
            else:
                vals = pd.Series([], dtype=float)
            if len(vals):
                M[i, j] = vals.median()

    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(M, aspect="auto", cmap="viridis", origin="lower")
    ax.set_xticks(range(len(gef_levels)))
    ax.set_xticklabels([f"{g}" for g in gef_levels])
    ax.set_yticks(range(len(kappa_levels)))
    ax.set_yticklabels([f"{k}" for k in kappa_levels])
    ax.set_xlabel("gef_mult")
    ax.set_ylabel("kappa")
    cbar = fig.colorbar(im, ax=ax, label="median reduction-pct (vs anchor)")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.1f}", ha="center", va="center",
                         color="white" if v < (np.nanmin(M) + np.nanmax(M)) / 2 else "black", fontsize=9)
    if plot_title:
        ax.set_title("OAT cross-section: median reduction-pct (kappa × gef_mult)")
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

def plot_sobol_indices(
    sobol: pd.DataFrame,
    save_path: str | None = None,
    figsize: tuple = (8, 5.5),
    dpi: int = 150,
    for_latex: bool = False,
    plot_title: bool = False,
):
    """Sobol S1 and ST per feature, median across cities with IQR whiskers.

    Bars colored by the same Shapley-mechanism palette as the OAT tornado so
    the Fig. 5a (OAT) and Fig. 5f (Sobol) read jointly. S1 plotted as light
    bar inset over the darker ST outline so interaction share is visually
    the gap between the two.

    Args:
        sobol: Long-format Sobol DataFrame from ``all_sensitivity_sobol.csv``.
    """
    si = sobol[sobol["row_kind"] == "sobol_index"].copy()
    if si.empty:
        print("  plot_sobol_indices: no sobol_index rows; skipping")
        return None

    # Per-feature aggregates across cities
    grouped = si.groupby("feature")
    feature_medians = grouped["ST"].median().sort_values(ascending=True)
    features = list(feature_medians.index)
    st_median = feature_medians.values
    st_p25 = grouped["ST"].quantile(0.25).reindex(features).values
    st_p75 = grouped["ST"].quantile(0.75).reindex(features).values
    s1_median = grouped["S1"].median().reindex(features).values

    bar_colors = [
        OAT_MECHANISM_COLORS[OAT_FEATURE_MECHANISM.get(f, "operation")]
        for f in features
    ]

    fig, ax = plt.subplots(figsize=figsize)
    y = np.arange(len(features))
    ax.barh(y, st_median, color=bar_colors, alpha=0.45, edgecolor="black",
            linewidth=1.0, label="Total-order $S_T$ (incl. interactions)")
    ax.barh(y, s1_median, color=bar_colors, alpha=0.95, edgecolor="black",
            linewidth=1.0, label="First-order $S_1$")
    err_lower = np.clip(st_median - st_p25, 0, None)
    err_upper = np.clip(st_p75 - st_median, 0, None)
    ax.errorbar(st_median, y, xerr=[err_lower, err_upper], fmt="none",
                ecolor="black", capsize=3, linewidth=1.0)
    ax.set_yticks(y)
    ax.set_yticklabels([OAT_FEATURE_LABELS.get(f, f) for f in features])
    ax.set_xlabel("Sobol index (variance share of city-level reduction)")
    ax.axvline(0, color="black", linewidth=0.5)
    ax.set_xlim(left=min(0, np.nanmin(s1_median) - 0.02))

    legend_handles = [
        Patch(facecolor=OAT_MECHANISM_COLORS[m], edgecolor="black",
              label=OAT_MECHANISM_LABELS[m])
        for m in ("grid", "production", "bundling", "operation")
    ]
    legend_handles.append(Patch(facecolor="lightgray", edgecolor="black", alpha=0.45,
                                 label="$S_T$ (interactions)"))
    legend_handles.append(Patch(facecolor="lightgray", edgecolor="black", alpha=0.95,
                                 label="$S_1$ (main effect)"))
    ax.legend(handles=legend_handles, loc="lower right", fontsize=8, frameon=True)

    if plot_title:
        n_cities = si["region_key"].nunique()
        ax.set_title(f"Sobol global SA: 9-feature variance decomposition "
                     f"(n={n_cities} cities, median ± IQR)")
    return _finalize(save_path, dpi, for_latex, plot_title)


def plot_mc_uncertainty(
    sobol: pd.DataFrame,
    save_path: str | None = None,
    figsize: tuple = (8, 5.0),
    dpi: int = 150,
    for_latex: bool = False,
    plot_title: bool = False,
    headline_pct: float = 29.2,
):
    """City-level MC distribution of optimal-mix reduction %.

    Two panels: (left) CDF of per-city MC medians with 95% ribbon from the
    p2.5/p97.5 rows, (right) histogram of per-city anchor (κ=1.0) reductions
    with a vertical line at the §2 headline so the reader sees that the
    Sobol/MC summary reconciles with the headline value.
    """
    mc = sobol[sobol["row_kind"] == "mc_summary"].copy()
    if mc.empty:
        print("  plot_mc_uncertainty: no mc_summary rows; skipping")
        return None

    wide = mc.pivot_table(index="region_key", columns="feature", values="value")
    needed = ["mc_median_reduction_pct", "mc_p2.5", "mc_p97.5", "anchor_kappa1_reduction_pct"]
    missing = [c for c in needed if c not in wide.columns]
    if missing:
        print(f"  plot_mc_uncertainty: missing columns {missing}; skipping")
        return None

    wide = wide.sort_values("mc_median_reduction_pct").reset_index(drop=True)
    n = len(wide)
    rank = np.arange(1, n + 1) / n * 100.0

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=figsize)

    # Left: CDF of MC medians with shaded 2.5/97.5 ribbon (per-city).
    ax_l.fill_betweenx(rank, wide["mc_p2.5"], wide["mc_p97.5"],
                       color=OAT_MECHANISM_COLORS["grid"], alpha=0.15,
                       label="Per-city 95% MC band")
    ax_l.plot(wide["mc_median_reduction_pct"], rank,
              color=OAT_MECHANISM_COLORS["grid"], linewidth=1.6,
              label="MC median (ordered)")
    ax_l.axvline(headline_pct, color="black", linestyle="--", linewidth=1.0,
                 label=f"§2 headline ({headline_pct:.1f}%)")
    ax_l.set_xlabel("Reduction vs e-bike-only (%)")
    ax_l.set_ylabel("City percentile")
    ax_l.set_xlim(left=min(-5, float(wide["mc_p2.5"].min()) - 2))
    ax_l.legend(loc="lower right", fontsize=8, frameon=True)

    # Right: histogram of anchor κ=1.0 reductions; the headline lives in this
    # distribution.
    ax_r.hist(wide["anchor_kappa1_reduction_pct"].dropna(), bins=40,
              color=OAT_MECHANISM_COLORS["grid"], alpha=0.8, edgecolor="black",
              linewidth=0.4)
    ax_r.axvline(headline_pct, color="black", linestyle="--", linewidth=1.0,
                 label=f"§2 headline ({headline_pct:.1f}%)")
    anchor_mean = float(wide["anchor_kappa1_reduction_pct"].mean())
    ax_r.axvline(anchor_mean, color=OAT_MECHANISM_COLORS["bundling"],
                 linestyle=":", linewidth=1.2,
                 label=f"Anchor mean ({anchor_mean:.1f}%)")
    ax_r.set_xlabel("Anchor reduction at κ=1.0 (%)")
    ax_r.set_ylabel("Number of cities")
    ax_r.legend(loc="upper left", fontsize=8, frameon=True)

    if plot_title:
        fig.suptitle(f"Sobol/MC uncertainty across {n} cities", y=1.02)
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
            colors.append("#4C72B0" if kappa == 0.7 else "#DD8452")
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
        # Full tornado (all 11 features) — for supplementary material.
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
        plot_oat_gef_kappa_heatmap(oat, save_path=_resolve_path(out, "oat_gef_kappa_heatmap", for_latex),
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
        plot_sobol_indices(sb, save_path=_resolve_path(out, "sobol_indices", for_latex),
                            dpi=dpi, for_latex=for_latex, plot_title=plot_title)
        plot_mc_uncertainty(sb, save_path=_resolve_path(out, "mc_uncertainty", for_latex),
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
