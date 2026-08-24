"""Summarize consolidated sensitivity and pathway results across all cities.

Run after consolidate_sensitivity.

Usage:
    cd src/drone_emission
    ../../.venv/bin/python -m tools.summarize_sensitivity
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src", "drone_emission"))

from utils import load_config  # noqa: E402

DRONE_TECH_GROUPS = {
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

DRONE_TECH_FEATURES = {
    feature for features in DRONE_TECH_GROUPS.values() for feature in features
}


def _q(s: pd.Series, qs=(0.25, 0.5, 0.75)) -> tuple:
    return tuple(round(float(s.quantile(q)), 2) for q in qs)


def summarize_oat(df: pd.DataFrame) -> None:
    print("=" * 70)
    print("ONE-AT-A-TIME SENSITIVITY (OAT)")
    print("=" * 70)
    n_cities = df["region_key"].nunique()
    print(f"Cities covered: {n_cities}")
    print(f"OAT rows: {len(df)}  (≈ {len(df)/n_cities:.1f}/city)")
    print()

    # Baseline (anchor) row per city
    base = df[df["feature_varied"] == "baseline"].copy()
    print("--- Anchor baseline (configured κ, gef=1, payload=1, lifespans=4) ---")
    print(f"  cities with baseline row: {len(base)}")
    q1, q2, q3 = _q(base["reduction_vs_anchor_ebike_pct"])
    print(f"  reduction_vs_anchor_ebike_pct  P25/P50/P75: {q1} / {q2} / {q3}")
    q1, q2, q3 = _q(base["optimal_threshold_km"].replace([np.inf], np.nan).dropna())
    print(f"  optimal_threshold_km           P25/P50/P75: {q1} / {q2} / {q3}")
    q1, q2, q3 = _q(base["optimal_drone_ratio_pct"])
    print(f"  optimal_drone_ratio_pct        P25/P50/P75: {q1} / {q2} / {q3}")
    share = float((base["reduction_vs_anchor_ebike_pct"] > 0).mean()) * 100
    print(f"  beneficial-city share at anchor: {share:.1f}%")
    print()

    # Tornado: median |Δreduction-pct| per drone-technology feature
    print("--- Drone-tech tornado: median |Δreduction-pct| per feature (OAT, anchor as ref) ---")
    base_by_city = base.set_index("region_key")["reduction_vs_anchor_ebike_pct"]
    tornado = []
    for feat in df["feature_varied"].unique():
        if feat == "baseline" or feat not in DRONE_TECH_FEATURES:
            continue
        sub = df[df["feature_varied"] == feat].copy()
        sub["delta"] = (
            sub["reduction_vs_anchor_ebike_pct"]
            - sub["region_key"].map(base_by_city)
        )
        med_abs = float(sub["delta"].abs().median())
        med_signed = float(sub["delta"].median())
        tornado.append((feat, med_abs, med_signed))
    tornado.sort(key=lambda r: -r[1])
    print(f"  {'feature':<22} {'med|Δ|':>8} {'med Δ':>8}")
    for feat, ma, ms in tornado:
        print(f"  {feat:<22} {ma:>8.2f} {ms:>+8.2f}")
    print()


def summarize_shapley(df: pd.DataFrame) -> None:
    print("=" * 70)
    print("SHAPLEY MECHANISM DECOMPOSITION")
    print("=" * 70)
    n_cities = df["region_key"].nunique()
    print(f"Cities covered: {n_cities}")
    print()

    print("--- Median Shapley contribution per mechanism (pct pts of total) ---")
    print(f"  {'mechanism':<26} {'P25':>8} {'P50':>8} {'P75':>8} {'sign':>6}")
    for mech in ["M1_routing", "M2_bundling", "M3_production", "M4_gef",
                 "residual", "total_baseline_reduction", "sum_check"]:
        sub = df[df["mechanism"] == mech]
        if sub.empty:
            continue
        q1, q2, q3 = _q(sub["shapley_pct_pts"])
        sign = "(+)" if q2 > 0 else ("(-)" if q2 < 0 else "(0)")
        print(f"  {mech:<26} {q1:>+8.2f} {q2:>+8.2f} {q3:>+8.2f} {sign:>6}")
    print()


def summarize_named(df: pd.DataFrame) -> None:
    print("=" * 70)
    print("NAMED SCENARIOS")
    print("=" * 70)
    n_cities = df["region_key"].nunique()
    print(f"Cities covered: {n_cities}")
    print()

    print("--- Median reduction by (κ, strategy) ---")
    print(f"  {'κ':>4} {'strategy':<22} {'medRed_anchor':>14} {'medRed_sameκ':>14}")
    for (k, strat), sub in df.groupby(["kappa", "strategy"]):
        a = float(sub["reduction_vs_anchor_ebike_pct"].median())
        b = float(sub["reduction_vs_all_ebike_same_kappa_pct"].median())
        print(f"  {k:>4} {strat:<22} {a:>+14.2f} {b:>+14.2f}")
    print()

    comparison_kappas = sorted(df["kappa"].dropna().unique())
    comparison_kappa = comparison_kappas[0] if comparison_kappas else None

    # 2060 upside
    sub_2060 = df[(df["strategy"] == "optimal_mixed_2060")]
    if comparison_kappa is not None:
        sub_2060 = sub_2060[np.isclose(sub_2060["kappa"], comparison_kappa)]
    if comparison_kappa is not None and not sub_2060.empty:
        q1, q2, q3 = _q(sub_2060["reduction_vs_anchor_ebike_pct"])
        print(f"  optimal_mixed_2060 @ κ={comparison_kappa:g}  P25/P50/P75: {q1} / {q2} / {q3}")

    sub_2030 = df[(df["strategy"] == "optimal_mixed_2030")]
    if comparison_kappa is not None:
        sub_2030 = sub_2030[np.isclose(sub_2030["kappa"], comparison_kappa)]
    if comparison_kappa is not None and not sub_2030.empty:
        q1, q2, q3 = _q(sub_2030["reduction_vs_anchor_ebike_pct"])
        print(f"  optimal_mixed_2030 @ κ={comparison_kappa:g}  P25/P50/P75: {q1} / {q2} / {q3}")
    print()


def summarize_sobol(df: pd.DataFrame) -> None:
    print("=" * 70)
    print("SOBOL DRONE-TECH SENSITIVITY")
    print("=" * 70)
    si = df[df["row_kind"] == "sobol_index"].copy()
    si = si[si["feature"].isin(DRONE_TECH_FEATURES)]
    n_cities = si["region_key"].nunique()
    print(f"Cities covered: {n_cities}")
    print()

    print("--- Median Sobol index per drone-tech feature ---")
    med = si.groupby("feature").agg(S1=("S1", "median"), ST=("ST", "median"))
    med = med.sort_values("ST", ascending=False)
    print(f"  {'feature':<34} {'S1':>8} {'ST':>8}")
    for feat, row in med.iterrows():
        print(f"  {feat:<34} {row['S1']:>8.3f} {row['ST']:>8.3f}")
    print()

    print("--- Median class total ΣST ---")
    feature_to_group = {
        feature: group for group, features in DRONE_TECH_GROUPS.items() for feature in features
    }
    si["class"] = si["feature"].map(feature_to_group)
    per_city = si.groupby(["region_key", "class"])["ST"].sum().reset_index()
    grouped = per_city.groupby("class")["ST"].median().sort_values(ascending=False)
    for group, value in grouped.items():
        print(f"  {group:<18} {value:>8.3f}")
    print()

    mc = df[df["row_kind"] == "mc_summary"].pivot_table(
        index="region_key", columns="feature", values="value", aggfunc="first"
    )
    baseline_col = "baseline_drone_tech_reduction_pct"
    if baseline_col in mc.columns:
        q1, q2, q3 = _q(mc[baseline_col])
        print(f"  baseline_drone_tech_reduction_pct P25/P50/P75: {q1} / {q2} / {q3}")
    if "mc_median_reduction_pct" in mc.columns:
        q1, q2, q3 = _q(mc["mc_median_reduction_pct"])
        print(f"  MC median reduction pct           P25/P50/P75: {q1} / {q2} / {q3}")
    print()


def compare_outputs(oat: pd.DataFrame, shap: pd.DataFrame, named: pd.DataFrame, sobol: pd.DataFrame) -> None:
    print("=" * 70)
    print("OUTPUT CROSS-CHECK")
    print("=" * 70)

    base = oat[oat["feature_varied"] == "baseline"]
    actual_anchor = float(base["reduction_vs_anchor_ebike_pct"].median())
    actual_share = float((base["reduction_vs_anchor_ebike_pct"] > 0).mean()) * 100
    print(f"  OAT baseline median reduction:       {actual_anchor:.1f}%")
    print(f"  OAT beneficial-city share @ baseline: {actual_share:.1f}%")

    sobol_features = set(sobol.loc[sobol["row_kind"] == "sobol_index", "feature"].dropna())
    unexpected_sobol = sorted(sobol_features - DRONE_TECH_FEATURES)
    missing_sobol = sorted(DRONE_TECH_FEATURES - sobol_features)
    print(f"  Sobol sampled drone-tech features:    {len(sobol_features & DRONE_TECH_FEATURES)} / {len(DRONE_TECH_FEATURES)}")
    if unexpected_sobol:
        print(f"  Unexpected Sobol features:            {', '.join(unexpected_sobol)}")
    if missing_sobol:
        print(f"  Missing Sobol features:               {', '.join(missing_sobol)}")

    # Tornado top features
    base_by_city = base.set_index("region_key")["reduction_vs_anchor_ebike_pct"]
    tornado = []
    for feat in oat["feature_varied"].unique():
        if feat == "baseline" or feat not in DRONE_TECH_FEATURES:
            continue
        sub = oat[oat["feature_varied"] == feat].copy()
        sub["delta"] = sub["reduction_vs_anchor_ebike_pct"] - sub["region_key"].map(base_by_city)
        tornado.append((feat, float(sub["delta"].abs().median())))
    tornado.sort(key=lambda r: -r[1])
    top = ", ".join(f"{f}({v:.1f})" for f, v in tornado[:3])
    print(f"  Top-3 OAT drone-tech levers:          {top}")

    # Shapley signs / order
    shap_med = shap.groupby("mechanism")["shapley_pct_pts"].median()
    print()
    print("  Shapley median (pct pts):")
    for m in ["M1_routing", "M2_bundling", "M3_production", "M4_gef"]:
        if m in shap_med.index:
            print(f"    {m:<18} actual {shap_med[m]:+.2f}")

    # 2060 upside
    comparison_kappas = sorted(named["kappa"].dropna().unique())
    comparison_kappa = comparison_kappas[0] if comparison_kappas else None
    sub_2060 = named[named["strategy"] == "optimal_mixed_2060"]
    if comparison_kappa is not None:
        sub_2060 = sub_2060[np.isclose(sub_2060["kappa"], comparison_kappa)]
    if comparison_kappa is not None and not sub_2060.empty:
        a = float(sub_2060["reduction_vs_anchor_ebike_pct"].median())
        print()
        print(f"  optimal_mixed_2060 @ κ={comparison_kappa:g}:   {a:.1f}%")
    sub_2030 = named[named["strategy"] == "optimal_mixed_2030"]
    if comparison_kappa is not None:
        sub_2030 = sub_2030[np.isclose(sub_2030["kappa"], comparison_kappa)]
    if comparison_kappa is not None and not sub_2030.empty:
        a = float(sub_2030["reduction_vs_anchor_ebike_pct"].median())
        print(f"  optimal_mixed_2030 @ κ={comparison_kappa:g}:   {a:.1f}%")
    print()


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default=None)
    args = parser.parse_args()

    config = load_config(os.path.join(PROJECT_ROOT, "experiment.toml"), env=args.env)
    data_dir = config["data_dir"]

    oat = pd.read_csv(os.path.join(data_dir, "all_sensitivity_oat.csv"))
    shap = pd.read_csv(os.path.join(data_dir, "all_sensitivity_shapley.csv"))
    named = pd.read_csv(os.path.join(data_dir, "all_sensitivity_named.csv"))
    sobol = pd.read_csv(os.path.join(data_dir, "all_sensitivity_sobol.csv"))
    summarize_oat(oat)
    summarize_shapley(shap)
    summarize_named(named)
    summarize_sobol(sobol)
    compare_outputs(oat, shap, named, sobol)


if __name__ == "__main__":
    main()
