"""Summarize consolidated sensitivity results across all cities.

Compares actual 312-city numbers against the projected ranges in
notes/expected_sensitivity_results.md. Run after consolidate_sensitivity.

Usage:
    cd src/drone_emission
    ../../.venv/bin/python -m tools.summarize_sensitivity
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
DATA_DIR = os.path.join(PROJECT_ROOT, "data")


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

    # Tornado: median |Δreduction-pct| per feature
    print("--- Tornado: median |Δreduction-pct| per feature (OAT, anchor as ref) ---")
    base_by_city = base.set_index("region_key")["reduction_vs_anchor_ebike_pct"]
    tornado = []
    for feat in df["feature_varied"].unique():
        if feat == "baseline":
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

    # Beneficial-city share by κ
    print("--- Beneficial-city share by κ (κ-only OAT rows) ---")
    kappa_rows = df[df["feature_varied"] == "kappa"].copy()
    kappa_rows = pd.concat([kappa_rows, base.assign(feature_varied="kappa")], ignore_index=True)
    grp = kappa_rows.groupby("kappa").agg(
        n=("region_key", "nunique"),
        share_pos=("reduction_vs_scenario_ebike_pct", lambda s: float((s > 0).mean()) * 100),
        med_red=("reduction_vs_scenario_ebike_pct", "median"),
    )
    print(f"  {'κ':>4} {'n':>5} {'%>0':>7} {'medRed':>8}")
    for k, row in grp.iterrows():
        print(f"  {k:>4} {int(row['n']):>5} {row['share_pos']:>6.1f}% {row['med_red']:>8.2f}")
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

    # 2060 upside
    sub_2060 = df[(df["strategy"] == "optimal_mixed_2060") & (df["kappa"] == 0.7)]
    if not sub_2060.empty:
        q1, q2, q3 = _q(sub_2060["reduction_vs_anchor_ebike_pct"])
        print(f"  optimal_mixed_2060 @ κ=0.7  P25/P50/P75: {q1} / {q2} / {q3}")

    sub_2030 = df[(df["strategy"] == "optimal_mixed_2030") & (df["kappa"] == 0.7)]
    if not sub_2030.empty:
        q1, q2, q3 = _q(sub_2030["reduction_vs_anchor_ebike_pct"])
        print(f"  optimal_mixed_2030 @ κ=0.7  P25/P50/P75: {q1} / {q2} / {q3}")
    print()


def compare_to_projections(oat: pd.DataFrame, shap: pd.DataFrame, named: pd.DataFrame) -> None:
    print("=" * 70)
    print("ACTUAL vs PROJECTED (notes/expected_sensitivity_results.md)")
    print("=" * 70)

    base = oat[oat["feature_varied"] == "baseline"]
    actual_anchor = float(base["reduction_vs_anchor_ebike_pct"].median())
    actual_share = float((base["reduction_vs_anchor_ebike_pct"] > 0).mean()) * 100
    print(f"  Anchor median reduction:      actual {actual_anchor:.1f}%   projected ~25% (15–35%)")
    print(f"  Beneficial-city share @ anchor κ: actual {actual_share:.1f}%    projected >95%")

    # Tornado top-1 feature
    base_by_city = base.set_index("region_key")["reduction_vs_anchor_ebike_pct"]
    tornado = []
    for feat in oat["feature_varied"].unique():
        if feat == "baseline":
            continue
        sub = oat[oat["feature_varied"] == feat].copy()
        sub["delta"] = sub["reduction_vs_anchor_ebike_pct"] - sub["region_key"].map(base_by_city)
        tornado.append((feat, float(sub["delta"].abs().median())))
    tornado.sort(key=lambda r: -r[1])
    top = ", ".join(f"{f}({v:.1f})" for f, v in tornado[:3])
    print(f"  Top-3 OAT levers (med|Δ|):    actual {top}")
    print(f"                                projected: kappa, drone_energy_mult, gef_mult")

    # Shapley signs / order
    shap_med = shap.groupby("mechanism")["shapley_pct_pts"].median()
    print()
    print("  Shapley median (pct pts):")
    for m in ["M1_routing", "M2_bundling", "M3_production", "M4_gef"]:
        if m in shap_med.index:
            print(f"    {m:<18} actual {shap_med[m]:+.2f}")
    print(f"    projected: M3 ≈ +25..+30, M1 ≈ +6..+12, M2 ≈ -8..-12, M4 ≈ -20..-28")

    # 2060 upside
    sub_2060 = named[(named["strategy"] == "optimal_mixed_2060") & (named["kappa"] == 0.7)]
    if not sub_2060.empty:
        a = float(sub_2060["reduction_vs_anchor_ebike_pct"].median())
        print()
        print(f"  optimal_mixed_2060 @ κ=0.7:   actual {a:.1f}%   projected ~80–90%")
    sub_2030 = named[(named["strategy"] == "optimal_mixed_2030") & (named["kappa"] == 0.7)]
    if not sub_2030.empty:
        a = float(sub_2030["reduction_vs_anchor_ebike_pct"].median())
        print(f"  optimal_mixed_2030 @ κ=0.7:   actual {a:.1f}%   projected ~45–55%")
    print()


def main():
    oat = pd.read_csv(os.path.join(DATA_DIR, "all_sensitivity_oat.csv"))
    shap = pd.read_csv(os.path.join(DATA_DIR, "all_sensitivity_shapley.csv"))
    named = pd.read_csv(os.path.join(DATA_DIR, "all_sensitivity_named.csv"))
    summarize_oat(oat)
    summarize_shapley(shap)
    summarize_named(named)
    compare_to_projections(oat, shap, named)


if __name__ == "__main__":
    main()
