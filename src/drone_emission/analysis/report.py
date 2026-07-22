"""Generate a markdown report from cross-region analysis results.

The report is designed to be fed to an LLM for generating insights.

Usage:
    cd src/drone_emission
    python -m analysis.report --env remote
"""

import argparse
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.constants import VARIABLE_LABELS, province_name_en
from analysis.correlation import (
    FEATURE_VARIABLES,
    RESULT_VARIABLES,
    load_region_stats,
)
from utils import load_config

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)


def _fmt(val, decimals=2):
    """Format a number, handling NaN."""
    if pd.isna(val):
        return "N/A"
    if isinstance(val, float):
        return f"{val:.{decimals}f}"
    return str(val)


def _province_en(name):
    return province_name_en(name)


def _describe(series, label=None):
    """Return a markdown stats line for a numeric series."""
    s = series.dropna()
    parts = [
        f"n={len(s)}",
        f"mean={s.mean():.2f}",
        f"median={s.median():.2f}",
        f"std={s.std():.2f}",
        f"min={s.min():.2f}",
        f"max={s.max():.2f}",
    ]
    prefix = f"**{label}**: " if label else ""
    return prefix + ", ".join(parts)


def generate_report(csv_path: str) -> str:
    """Generate a markdown report string from the enriched stats CSV."""
    df = load_region_stats(csv_path)
    lines = []

    def w(line=""):
        lines.append(line)

    # ── Header ──
    w("# Cross-Region Emission Analysis Report")
    w(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    w(f"\nData source: `{os.path.basename(csv_path)}`")

    # ── 1. Coverage ──
    w("\n## 1. Coverage")
    w(f"\n- **Total cities analyzed**: {len(df)}")
    if "province" in df.columns:
        provinces = df["province"].dropna().unique()
        w(f"- **Provincial-level regions covered**: {len(provinces)}")
        w(
            "- **Provincial-level region list**: "
            f"{', '.join(sorted(_province_en(p) for p in provinces))}"
        )
    if "region_area_km2" in df.columns:
        w(f"- **Total area**: {df['region_area_km2'].sum():,.0f} km²")
    if "odpair_count" in df.columns:
        w(f"- **Total OD pairs**: {df['odpair_count'].sum():,.0f}")
    if "store_count" in df.columns:
        w(f"- **Total stores**: {df['store_count'].sum():,.0f}")

    # ── 2. Key Findings ──
    w("\n## 2. Key Emission Reduction Findings")

    col = "max_lifecycle_reduction_pct"
    if col in df.columns:
        s = df[col].dropna()
        w(f"\n### Lifecycle Emission Reduction (%)")
        w(f"\n{_describe(s)}")
        w(f"\n- **Cities with >20% reduction**: {(s > 20).sum()} ({(s > 20).sum() / len(s) * 100:.0f}%)")
        w(f"- **Cities with >30% reduction**: {(s > 30).sum()} ({(s > 30).sum() / len(s) * 100:.0f}%)")
        w(f"- **Cities with <10% reduction**: {(s < 10).sum()} ({(s < 10).sum() / len(s) * 100:.0f}%)")

    col_thresh = "optimal_threshold_km"
    if col_thresh in df.columns:
        w(f"\n### Optimal Distance Threshold (km)")
        w(f"\n{_describe(df[col_thresh].dropna())}")

    col_ratio = "optimal_drone_ratio_pct"
    if col_ratio in df.columns:
        w(f"\n### Optimal Drone Ratio (%)")
        w(f"\n{_describe(df[col_ratio].dropna())}")

    # ── 3. Top & Bottom Cities ──
    w("\n## 3. Top & Bottom Cities")

    if "max_lifecycle_reduction_pct" in df.columns:
        sorted_df = df.sort_values("max_lifecycle_reduction_pct", ascending=False)

        w("\n### Top 15 Cities (Highest Emission Reduction)")
        w("\n| Rank | City | Provincial-level region | Reduction (%) | Opt. Threshold (km) | Drone Ratio (%) | Detour Ratio | Grid EF |")
        w("|------|------|----------|---------------|---------------------|-----------------|--------------|---------|")
        for i, (_, row) in enumerate(sorted_df.head(15).iterrows(), 1):
            w(f"| {i} | {row.get('region_name', 'N/A')} "
              f"| {_province_en(row.get('province', 'N/A'))} "
              f"| {_fmt(row.get('max_lifecycle_reduction_pct'))} "
              f"| {_fmt(row.get('optimal_threshold_km'))} "
              f"| {_fmt(row.get('optimal_drone_ratio_pct'))} "
              f"| {_fmt(row.get('detour_ratio'))} "
              f"| {_fmt(row.get('grid_emission_factor'))} |")

        w("\n### Bottom 15 Cities (Lowest Emission Reduction)")
        w("\n| Rank | City | Provincial-level region | Reduction (%) | Opt. Threshold (km) | Drone Ratio (%) | Detour Ratio | Grid EF |")
        w("|------|------|----------|---------------|---------------------|-----------------|--------------|---------|")
        for i, (_, row) in enumerate(sorted_df.tail(15).iterrows(), 1):
            w(f"| {i} | {row.get('region_name', 'N/A')} "
              f"| {_province_en(row.get('province', 'N/A'))} "
              f"| {_fmt(row.get('max_lifecycle_reduction_pct'))} "
              f"| {_fmt(row.get('optimal_threshold_km'))} "
              f"| {_fmt(row.get('optimal_drone_ratio_pct'))} "
              f"| {_fmt(row.get('detour_ratio'))} "
              f"| {_fmt(row.get('grid_emission_factor'))} |")

    # ── 4. Provincial-level region summary ──
    w("\n## 4. Provincial-Level Region Summary")

    if "province" in df.columns and "max_lifecycle_reduction_pct" in df.columns:
        prov = df.groupby("province").agg(
            n_cities=("region_name", "count"),
            mean_reduction=("max_lifecycle_reduction_pct", "mean"),
            median_reduction=("max_lifecycle_reduction_pct", "median"),
            std_reduction=("max_lifecycle_reduction_pct", "std"),
            mean_detour=("detour_ratio", "mean") if "detour_ratio" in df.columns else ("max_lifecycle_reduction_pct", "count"),
            mean_gef=("grid_emission_factor", "mean") if "grid_emission_factor" in df.columns else ("max_lifecycle_reduction_pct", "count"),
        ).sort_values("mean_reduction", ascending=False)

        w("\n| Provincial-level region | Cities | Mean Reduction (%) | Median (%) | Std (%) | Mean Detour | Mean Grid EF |")
        w("|----------|--------|-------------------|------------|---------|-------------|--------------|")
        for prov_name, row in prov.iterrows():
            w(f"| {_province_en(prov_name)} | {row['n_cities']:.0f} "
              f"| {_fmt(row['mean_reduction'])} "
              f"| {_fmt(row['median_reduction'])} "
              f"| {_fmt(row['std_reduction'])} "
              f"| {_fmt(row.get('mean_detour', float('nan')))} "
              f"| {_fmt(row.get('mean_gef', float('nan')))} |")

    # ── 5. Feature Distributions ──
    w("\n## 5. Feature Distributions")

    for category, features in FEATURE_VARIABLES.items():
        w(f"\n### {category.title()}")
        for feat in features:
            if feat in df.columns:
                label = VARIABLE_LABELS.get(feat, feat)
                w(f"\n- {_describe(df[feat].dropna(), label)}")

    # ── 6. Correlations ──
    w("\n## 6. Key Correlations with Emission Reduction")

    all_features = [f for fs in FEATURE_VARIABLES.values() for f in fs]
    col = "max_lifecycle_reduction_pct"
    if col in df.columns:
        corrs = []
        for feat in all_features:
            if feat in df.columns:
                valid = df[[col, feat]].dropna()
                if len(valid) > 10:
                    r_pearson, p_pearson = pd.Series.corr(valid[col], valid[feat]), None
                    from scipy import stats as sp_stats
                    r_s, p_s = sp_stats.spearmanr(valid[col], valid[feat])
                    r_p, p_p = sp_stats.pearsonr(valid[col], valid[feat])
                    corrs.append({
                        "feature": feat,
                        "label": VARIABLE_LABELS.get(feat, feat),
                        "pearson_r": r_p,
                        "pearson_p": p_p,
                        "spearman_r": r_s,
                        "spearman_p": p_s,
                        "n": len(valid),
                    })

        corrs.sort(key=lambda x: abs(x["spearman_r"]), reverse=True)

        w(f"\nCorrelations with **{VARIABLE_LABELS.get(col, col)}** (sorted by |Spearman r|):")
        w("\n| Feature | Pearson r | p-value | Spearman r | p-value | n |")
        w("|---------|-----------|---------|------------|---------|---|")
        for c in corrs:
            sig_p = "***" if c["pearson_p"] < 0.001 else "**" if c["pearson_p"] < 0.01 else "*" if c["pearson_p"] < 0.05 else ""
            sig_s = "***" if c["spearman_p"] < 0.001 else "**" if c["spearman_p"] < 0.01 else "*" if c["spearman_p"] < 0.05 else ""
            w(f"| {c['label']} | {c['pearson_r']:.3f}{sig_p} | {c['pearson_p']:.1e} "
              f"| {c['spearman_r']:.3f}{sig_s} | {c['spearman_p']:.1e} | {c['n']} |")

    # ── 7. Percentile Breakdown ──
    w("\n## 7. Percentile Breakdown")

    if "max_lifecycle_reduction_pct" in df.columns:
        s = df["max_lifecycle_reduction_pct"].dropna().sort_values(ascending=False)
        percentiles = [10, 25, 50, 75, 90]
        w("\n| Percentile | Emission Reduction (%) |")
        w("|------------|----------------------|")
        for p in percentiles:
            w(f"| {p}th | {np.percentile(s, 100 - p):.2f} |")

    # ── 8. Full City Table ──
    w("\n## 8. Full City Data (sorted by emission reduction)")

    display_cols = [
        "region_name", "province", "max_lifecycle_reduction_pct",
        "optimal_threshold_km", "optimal_drone_ratio_pct",
        "detour_ratio", "grid_emission_factor", "region_area_km2",
        "building_density", "road_density", "store_count",
        "total_monthly_sales", "odpair_count",
    ]
    available_cols = [c for c in display_cols if c in df.columns]

    if available_cols:
        sorted_df = df.sort_values("max_lifecycle_reduction_pct", ascending=False)

        # Header
        header_labels = {
            "region_name": "City",
            "province": "Provincial-level region",
            "max_lifecycle_reduction_pct": "Reduction (%)",
            "optimal_threshold_km": "Threshold (km)",
            "optimal_drone_ratio_pct": "Drone Ratio (%)",
            "detour_ratio": "Detour Ratio",
            "grid_emission_factor": "Grid EF",
            "region_area_km2": "Area (km²)",
            "building_density": "Bldg Density",
            "road_density": "Road Density",
            "store_count": "Stores",
            "total_monthly_sales": "Monthly Sales",
            "odpair_count": "OD Pairs",
        }
        headers = [header_labels.get(c, c) for c in available_cols]
        w("\n| " + " | ".join(headers) + " |")
        w("| " + " | ".join(["---"] * len(headers)) + " |")

        for _, row in sorted_df.iterrows():
            cells = []
            for c in available_cols:
                val = row.get(c)
                if c == "province":
                    cells.append(_province_en(str(val)) if pd.notna(val) else "N/A")
                elif c in ("store_count", "odpair_count", "total_monthly_sales"):
                    cells.append(f"{val:,.0f}" if pd.notna(val) else "N/A")
                elif c == "region_area_km2":
                    cells.append(f"{val:,.0f}" if pd.notna(val) else "N/A")
                elif isinstance(val, float):
                    cells.append(_fmt(val))
                else:
                    cells.append(str(val) if pd.notna(val) else "N/A")
            w("| " + " | ".join(cells) + " |")

    w("\n---")
    w("*End of report. Provide this file to an LLM for insight generation.*")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Generate analysis report")
    parser.add_argument("--env", type=str, default=None, help="Environment name")
    parser.add_argument("--output", type=str, default=None, help="Output path override")
    args = parser.parse_args()

    exp_config_path = os.path.join(PROJECT_ROOT, "experiment.toml")
    config = load_config(exp_config_path, env=args.env)
    data_dir = config["data_dir"]

    enriched = os.path.join(data_dir, "all_region_stats_enriched.csv")
    csv_path = enriched if os.path.exists(enriched) else os.path.join(data_dir, "all_region_stats.csv")

    report = generate_report(csv_path)

    output_path = args.output or os.path.join(PROJECT_ROOT, "docs", "shared", "analysis_report.md")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        f.write(report)

    print(f"Report saved to {output_path}")
    print(f"  ({len(report.splitlines())} lines, {len(report)} chars)")


if __name__ == "__main__":
    main()
