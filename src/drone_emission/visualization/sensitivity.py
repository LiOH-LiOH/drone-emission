"""Exploratory plots from the consolidated sensitivity outputs.

Reads ``data/all_sensitivity_oat.csv``, ``data/all_sensitivity_shapley.csv``,
and ``data/all_sensitivity_named.csv``. Produces ~12 candidate figures into
``figures/sensitivity/``. The user picks the final paper figures from this
set later.

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

OAT_FEATURES = ("kappa", "gef_mult", "drone_energy_mult", "payload_kg",
                "drone_lifespan_yr", "ebike_lifespan_yr")
OAT_BASELINE = {"kappa": 1.0, "gef_mult": 1.0, "drone_energy_mult": 1.0,
                "payload_kg": 1.0, "drone_lifespan_yr": 4, "ebike_lifespan_yr": 4}

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


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _save(fig, path: str) -> None:
    _ensure_dir(os.path.dirname(path))
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path}")


# ---------------------------------------------------------------------------
# OAT plots
# ---------------------------------------------------------------------------

def plot_oat_tornado(oat: pd.DataFrame, out_dir: str) -> None:
    """Median |Δ reduction| (vs baseline) per OAT feature, sorted descending."""
    base = oat[oat["scenario_id"] == "baseline"][["region_key", "reduction_vs_anchor_ebike_pct"]]
    base = base.rename(columns={"reduction_vs_anchor_ebike_pct": "baseline_red"})
    diffs: Dict[str, list[float]] = {}
    for feature in OAT_FEATURES:
        sub = oat[oat["feature_varied"] == feature]
        merged = sub.merge(base, on="region_key", how="inner")
        merged["delta"] = merged["reduction_vs_anchor_ebike_pct"] - merged["baseline_red"]
        # Take median absolute deviation per city, then median across cities and across levels
        per_city_max = merged.groupby("region_key")["delta"].agg(lambda s: s.abs().max())
        diffs[feature] = per_city_max.tolist()

    sorted_features = sorted(diffs.keys(), key=lambda f: -np.median(diffs[f]))
    medians = [np.median(diffs[f]) for f in sorted_features]
    p25 = [np.percentile(diffs[f], 25) for f in sorted_features]
    p75 = [np.percentile(diffs[f], 75) for f in sorted_features]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    y = np.arange(len(sorted_features))
    ax.barh(y, medians, height=0.6, color="#4C72B0", alpha=0.85)
    ax.errorbar(medians, y, xerr=[
        np.array(medians) - np.array(p25),
        np.array(p75) - np.array(medians),
    ], fmt="none", color="black", capsize=3)
    ax.set_yticks(y)
    ax.set_yticklabels(sorted_features)
    ax.invert_yaxis()
    ax.set_xlabel("Max |Δ reduction-pct| across levels (median across cities, IQR)")
    ax.set_title("OAT sensitivity tornado")
    ax.grid(axis="x", linestyle=":", alpha=0.5)
    _save(fig, os.path.join(out_dir, "oat_tornado.png"))


def _oat_panels(oat: pd.DataFrame, value_col: str, ylabel: str, fname: str, out_dir: str) -> None:
    """6-panel grid showing median + IQR of `value_col` vs each feature's level."""
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharey=True)
    axes = axes.flatten()
    for ax, feature in zip(axes, OAT_FEATURES):
        sub_dev = oat[oat["feature_varied"] == feature]
        sub_base = oat[oat["scenario_id"] == "baseline"].copy()
        # Treat baseline as a level on the same x axis
        sub_base[feature] = OAT_BASELINE[feature]
        sub_base["feature_varied"] = feature
        sub = pd.concat([sub_dev, sub_base], ignore_index=True)
        levels = sorted(sub[feature].unique())
        med = [sub[sub[feature] == lv][value_col].median() for lv in levels]
        p25 = [sub[sub[feature] == lv][value_col].quantile(0.25) for lv in levels]
        p75 = [sub[sub[feature] == lv][value_col].quantile(0.75) for lv in levels]
        ax.plot(levels, med, marker="o", color="#4C72B0", label="median")
        ax.fill_between(levels, p25, p75, color="#4C72B0", alpha=0.2, label="IQR")
        ax.axvline(OAT_BASELINE[feature], linestyle="--", color="grey", alpha=0.6)
        ax.set_title(feature)
        ax.set_xlabel(feature)
        ax.grid(linestyle=":", alpha=0.5)
    axes[0].set_ylabel(ylabel)
    axes[3].set_ylabel(ylabel)
    fig.suptitle(ylabel + " — OAT response (median ± IQR across cities)")
    fig.tight_layout()
    _save(fig, os.path.join(out_dir, fname))


def plot_oat_reduction_panels(oat: pd.DataFrame, out_dir: str) -> None:
    _oat_panels(oat, "reduction_vs_anchor_ebike_pct",
                "Reduction vs anchor (%)", "oat_reduction_panels.png", out_dir)


def plot_oat_threshold_panels(oat: pd.DataFrame, out_dir: str) -> None:
    # Replace inf threshold with NaN for plotting
    oat2 = oat.copy()
    oat2["optimal_threshold_km"] = oat2["optimal_threshold_km"].replace([np.inf, -np.inf], np.nan)
    _oat_panels(oat2, "optimal_threshold_km",
                "Optimal threshold (km)", "oat_threshold_panels.png", out_dir)


def plot_oat_drone_ratio_panels(oat: pd.DataFrame, out_dir: str) -> None:
    _oat_panels(oat, "optimal_drone_ratio_pct",
                "Optimal drone share (%)", "oat_drone_ratio_panels.png", out_dir)


def plot_oat_beneficial_share_panels(oat: pd.DataFrame, out_dir: str) -> None:
    """Share of cities where reduction_vs_scenario > 0, vs each feature's level."""
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharey=True)
    axes = axes.flatten()
    for ax, feature in zip(axes, OAT_FEATURES):
        sub_dev = oat[oat["feature_varied"] == feature]
        sub_base = oat[oat["scenario_id"] == "baseline"].copy()
        sub_base[feature] = OAT_BASELINE[feature]
        sub_base["feature_varied"] = feature
        sub = pd.concat([sub_dev, sub_base], ignore_index=True)
        levels = sorted(sub[feature].unique())
        share = [sub[sub[feature] == lv]["beneficial"].mean() * 100 for lv in levels]
        ax.plot(levels, share, marker="o", color="#55A467")
        ax.axvline(OAT_BASELINE[feature], linestyle="--", color="grey", alpha=0.6)
        ax.set_ylim(0, 105)
        ax.set_title(feature)
        ax.set_xlabel(feature)
        ax.grid(linestyle=":", alpha=0.5)
    axes[0].set_ylabel("% of cities with reduction > 0")
    axes[3].set_ylabel("% of cities with reduction > 0")
    fig.suptitle("Beneficial-city share — OAT response")
    fig.tight_layout()
    _save(fig, os.path.join(out_dir, "oat_beneficial_share.png"))


def plot_oat_gef_kappa_heatmap(oat: pd.DataFrame, out_dir: str) -> None:
    """Median reduction across cities, sectioned through OAT runs of GEF and kappa.

    OAT means each varied feature is alone (others at baseline). We can still build
    a 2D heatmap by overlaying: kappa-axis varies along rows, gef-axis varies along
    columns; cells are populated only where one of them was the varied feature OR
    at baseline (single dot at center).
    """
    # Build a kappa × gef_mult matrix, populated by:
    # - kappa-feature rows at gef_mult=baseline
    # - gef_mult-feature rows at kappa=baseline
    # - baseline at center
    kappa_levels = sorted(oat[oat["feature_varied"] == "kappa"]["kappa"].unique().tolist() + [OAT_BASELINE["kappa"]])
    kappa_levels = sorted(set(kappa_levels))
    gef_levels = sorted(oat[oat["feature_varied"] == "gef_mult"]["gef_mult"].unique().tolist() + [OAT_BASELINE["gef_mult"]])
    gef_levels = sorted(set(gef_levels))

    M = np.full((len(kappa_levels), len(gef_levels)), np.nan)
    for i, k in enumerate(kappa_levels):
        for j, g in enumerate(gef_levels):
            if k == OAT_BASELINE["kappa"] and g == OAT_BASELINE["gef_mult"]:
                vals = oat[oat["scenario_id"] == "baseline"]["reduction_vs_anchor_ebike_pct"]
            elif g == OAT_BASELINE["gef_mult"] and k != OAT_BASELINE["kappa"]:
                vals = oat[(oat["feature_varied"] == "kappa") & (oat["kappa"] == k)]["reduction_vs_anchor_ebike_pct"]
            elif k == OAT_BASELINE["kappa"] and g != OAT_BASELINE["gef_mult"]:
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
    ax.set_title("OAT cross-section: median reduction-pct (kappa × gef_mult)\n(off-axis cells are blank — OAT design)")
    fig.tight_layout()
    _save(fig, os.path.join(out_dir, "oat_gef_kappa_heatmap.png"))


# ---------------------------------------------------------------------------
# Shapley plots
# ---------------------------------------------------------------------------

def _pivot_shapley(shapley: pd.DataFrame) -> pd.DataFrame:
    """One row per region, columns = mechanism (incl. residual_all_off + total_baseline)."""
    return shapley.pivot_table(
        index=["region_key", "province"],
        columns="mechanism",
        values="shapley_pct_pts",
        aggfunc="first",
    ).reset_index()


def plot_shapley_aggregate_stacked(shapley: pd.DataFrame, out_dir: str) -> None:
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
    ax.set_title("Shapley decomposition of drone reduction\n(median across cities)")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    _save(fig, os.path.join(out_dir, "shapley_aggregate_stacked.png"))


def plot_shapley_boxplot(shapley: pd.DataFrame, out_dir: str) -> None:
    pivot = _pivot_shapley(shapley)
    data = [pivot[m].dropna().values for m in MECHANISM_ORDER]
    fig, ax = plt.subplots(figsize=(8, 5))
    bp = ax.boxplot(data, labels=[MECHANISM_LABELS[m] for m in MECHANISM_ORDER],
                     showfliers=False, patch_artist=True)
    for patch, m in zip(bp["boxes"], MECHANISM_ORDER):
        patch.set_facecolor(MECHANISM_COLORS[m])
        patch.set_alpha(0.7)
    ax.axhline(0, linestyle=":", color="black", alpha=0.5)
    ax.set_ylabel("Shapley contribution (pct pts)")
    ax.set_title("Shapley contribution by mechanism (across all cities)")
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    plt.setp(ax.get_xticklabels(), rotation=15, ha="right")
    fig.tight_layout()
    _save(fig, os.path.join(out_dir, "shapley_boxplot.png"))


def plot_shapley_per_city_stacked(shapley: pd.DataFrame, out_dir: str, top_n: int = 50) -> None:
    pivot = _pivot_shapley(shapley)
    pivot = pivot.sort_values("total_baseline", ascending=False)
    top = pivot.head(top_n)
    bot = pivot.tail(top_n)
    for label, sub in (("top", top), ("bottom", bot)):
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
        ax.set_title(f"Per-city Shapley decomposition — {label} {top_n} cities by total reduction")
        if label == "top":
            ax.legend(loc="upper right", fontsize=7)
        fig.tight_layout()
        _save(fig, os.path.join(out_dir, f"shapley_per_city_{label}.png"))


def plot_shapley_dominance_scatter(shapley: pd.DataFrame, out_dir: str) -> None:
    """Each city is a point: x = M1 routing Shapley, y = M4 GEF Shapley.

    Shows the spatial story without needing a choropleth (no province polygons here).
    """
    pivot = _pivot_shapley(shapley)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(pivot["M1_routing"], pivot["M4_gef"], c=pivot["total_baseline"],
                cmap="viridis", s=14, alpha=0.7)
    ax.axhline(0, linestyle=":", color="black", alpha=0.4)
    ax.axvline(0, linestyle=":", color="black", alpha=0.4)
    ax.set_xlabel("M1 routing Shapley (pct pts)")
    ax.set_ylabel("M4 GEF Shapley (pct pts)")
    cbar = fig.colorbar(ax.collections[0], ax=ax, label="total baseline reduction (%)")
    ax.set_title("City-level mechanism mix (color = total reduction)")
    ax.grid(linestyle=":", alpha=0.5)
    fig.tight_layout()
    _save(fig, os.path.join(out_dir, "shapley_dominance_scatter.png"))


# ---------------------------------------------------------------------------
# Named-scenario plots
# ---------------------------------------------------------------------------

def plot_named_boxplot(named: pd.DataFrame, out_dir: str) -> None:
    """Boxplot of reduction-vs-anchor across cities, one box per (kappa, strategy)."""
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
        pos += 1  # gap between kappa groups

    bp = ax.boxplot(data, positions=positions, widths=0.7, showfliers=False, patch_artist=True)
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.7)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.axhline(0, linestyle=":", color="black", alpha=0.5)
    ax.set_ylabel("Reduction vs anchor (%)")
    ax.set_title("Named scenarios — distribution across cities")
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    fig.tight_layout()
    _save(fig, os.path.join(out_dir, "named_boxplot.png"))


def plot_named_cdf(named: pd.DataFrame, out_dir: str) -> None:
    """Cumulative-distribution: % of cities reaching ≥ X% reduction per (kappa, strategy)."""
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
        ax.set_title(f"κ = {kappa}")
        ax.grid(linestyle=":", alpha=0.5)
        ax.legend(fontsize=7)
    fig.tight_layout()
    _save(fig, os.path.join(out_dir, "named_cdf.png"))


def plot_named_2030_2060_scatter(named: pd.DataFrame, out_dir: str) -> None:
    """Scatter: x = optimal_mixed_2030 reduction, y = optimal_mixed_2060 reduction.

    One panel per kappa anchor.
    """
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharex=True, sharey=True)
    for ax, kappa in zip(axes, sorted(named["kappa"].unique())):
        a = named[(named["kappa"] == kappa) & (named["strategy"] == "optimal_mixed_2030")][
            ["region_key", "reduction_vs_anchor_ebike_pct"]].rename(columns={"reduction_vs_anchor_ebike_pct": "y2030"})
        b = named[(named["kappa"] == kappa) & (named["strategy"] == "optimal_mixed_2060")][
            ["region_key", "reduction_vs_anchor_ebike_pct"]].rename(columns={"reduction_vs_anchor_ebike_pct": "y2060"})
        m = a.merge(b, on="region_key")
        ax.scatter(m["y2030"], m["y2060"], s=12, alpha=0.6, color="#4C72B0")
        lo = min(m["y2030"].min(), m["y2060"].min(), 0)
        hi = max(m["y2030"].max(), m["y2060"].max())
        ax.plot([lo, hi], [lo, hi], color="grey", linestyle="--", alpha=0.6)
        ax.set_xlabel("2030 grid reduction (%)")
        ax.set_ylabel("2060 grid reduction (%)")
        ax.set_title(f"κ = {kappa}")
        ax.grid(linestyle=":", alpha=0.5)
    fig.suptitle("Cross-scenario: 2030 vs 2060 grid reduction per city")
    fig.tight_layout()
    _save(fig, os.path.join(out_dir, "named_2030_2060_scatter.png"))


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def render_all(data_dir: str, figures_dir: str) -> None:
    out = os.path.join(figures_dir, "sensitivity")
    _ensure_dir(out)
    oat_path = os.path.join(data_dir, "all_sensitivity_oat.csv")
    sh_path = os.path.join(data_dir, "all_sensitivity_shapley.csv")
    nm_path = os.path.join(data_dir, "all_sensitivity_named.csv")

    if os.path.exists(oat_path):
        oat = pd.read_csv(oat_path)
        print(f"OAT: {len(oat)} rows from {oat_path}")
        plot_oat_tornado(oat, out)
        plot_oat_reduction_panels(oat, out)
        plot_oat_threshold_panels(oat, out)
        plot_oat_drone_ratio_panels(oat, out)
        plot_oat_beneficial_share_panels(oat, out)
        plot_oat_gef_kappa_heatmap(oat, out)
    else:
        print(f"  missing {oat_path}; skipping OAT plots")

    if os.path.exists(sh_path):
        sh = pd.read_csv(sh_path)
        print(f"Shapley: {len(sh)} rows from {sh_path}")
        plot_shapley_aggregate_stacked(sh, out)
        plot_shapley_boxplot(sh, out)
        plot_shapley_per_city_stacked(sh, out, top_n=50)
        plot_shapley_dominance_scatter(sh, out)
    else:
        print(f"  missing {sh_path}; skipping Shapley plots")

    if os.path.exists(nm_path):
        nm = pd.read_csv(nm_path)
        print(f"Named: {len(nm)} rows from {nm_path}")
        plot_named_boxplot(nm, out)
        plot_named_cdf(nm, out)
        plot_named_2030_2060_scatter(nm, out)
    else:
        print(f"  missing {nm_path}; skipping named-scenario plots")


def _bootstrap_paths():
    here = os.path.dirname(os.path.abspath(__file__))
    src = os.path.normpath(os.path.join(here, ".."))
    if src not in sys.path:
        sys.path.insert(0, src)


def main():
    _bootstrap_paths()
    from utils import load_config  # noqa: E402

    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default=None)
    args = parser.parse_args()

    project_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    )
    config = load_config(os.path.join(project_root, "experiment.toml"), env=args.env)
    render_all(config["data_dir"], config["figures_dir"])


if __name__ == "__main__":
    main()
