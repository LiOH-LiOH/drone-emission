"""OLS regression analysis for cross-region emission results.

Provides standardized OLS regression, VIF multicollinearity checks,
hierarchical (nested) regression with incremental R², Lasso feature
selection, and publication-ready output for identifying which urban
features significantly predict emission reduction outcomes.

Usage:
    from drone_emission.analysis.regression import run_regression_analysis
    run_regression_analysis(csv_path, output_dir)

    from drone_emission.analysis.regression import run_hierarchical_regression
    run_hierarchical_regression(df, result_var)
"""

import os

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats as sp_stats
from statsmodels.stats.outliers_influence import variance_inflation_factor

from drone_emission.analysis.constants import VARIABLE_LABELS

# Regression focuses on the primary outcome used in the paper narrative.
HIERARCHICAL_MODELS = {
    "M1: Energy": [
        "grid_emission_factor",
    ],
    "M2: + Road structure": [
        "grid_emission_factor",
        "avg_block_length",
        "orientation_entropy",
    ],
    "M3: Full": [
        "grid_emission_factor",
        "avg_block_length",
        "orientation_entropy",
        "intersection_density",
        "store_density",
    ],
}


def compute_vif(df: pd.DataFrame, feature_vars: list) -> pd.DataFrame:
    """Compute Variance Inflation Factor for each feature.

    Args:
        df: DataFrame with feature columns.
        feature_vars: List of feature variable names.

    Returns:
        DataFrame with columns ['feature', 'VIF'], sorted by VIF descending.
    """
    available = [f for f in feature_vars if f in df.columns]
    subset = df[available].dropna()

    if len(subset) < len(available) + 1:
        print(f"Warning: only {len(subset)} observations for {len(available)} features")
        return pd.DataFrame(columns=["feature", "VIF"])

    X = sm.add_constant(subset, has_constant="add").values
    vif_data = []
    for i, feat in enumerate(available):
        # Column 0 is the intercept added above. VIFs must be calculated with
        # an intercept; omitting it severely inflates variables whose means are
        # large relative to their standard deviations.
        vif_val = variance_inflation_factor(X, i + 1)
        vif_data.append({"feature": feat, "VIF": vif_val})

    vif_df = pd.DataFrame(vif_data).sort_values("VIF", ascending=False)
    return vif_df


def run_hierarchical_regression(
    df: pd.DataFrame,
    result_var: str,
    models: dict = None,
) -> list[dict]:
    """Run nested OLS models and report incremental R².

    Fits a sequence of nested models (each a superset of the previous),
    reports R², adjusted R², delta-R², F-test for the increment, and
    standardized coefficients for each.

    Args:
        df: DataFrame with all variables.
        result_var: Dependent variable name.
        models: Ordered dict of {model_name: [feature_list]}.
                Uses HIERARCHICAL_MODELS if None.

    Returns:
        List of dicts with keys: name, features, n, r2, r2_adj,
        delta_r2, f_change, f_p, ols_result.
    """
    if models is None:
        models = HIERARCHICAL_MODELS

    result_label = VARIABLE_LABELS.get(result_var, result_var)
    print(f"\n{'=' * 72}")
    print(f"HIERARCHICAL REGRESSION: {result_label}")
    print(f"{'=' * 72}")

    # Find the common valid sample across all models (use the largest
    # feature set to determine the sample so N is constant across models).
    all_features = []
    for feats in models.values():
        all_features.extend(feats)
    all_features = list(dict.fromkeys(all_features))  # dedupe, preserve order

    available = [f for f in all_features if f in df.columns]
    missing = set(all_features) - set(available)
    if missing:
        print(f"Warning: missing columns {missing}")

    subset = df[[result_var] + available].dropna()
    n = len(subset)
    if n < len(available) + 2:
        print(f"Insufficient observations ({n}), need at least {len(available) + 2}")
        return []

    print(f"Sample size: N = {n} (common across all models)\n")

    # Standardize once on the common sample
    standardized = (subset - subset.mean()) / subset.std()
    y = standardized[result_var]

    # Fit each model
    results_list = []
    prev_r2 = 0.0
    prev_p = 0

    # Header
    print(
        f"{'Model':<28s} {'R²':>7s} {'Adj R²':>7s} {'ΔR²':>7s} "
        f"{'F-chg':>8s} {'p(F-chg)':>10s} {'N':>5s}"
    )
    print("-" * 72)

    for name, features in models.items():
        model_features = [f for f in features if f in df.columns]
        if not model_features:
            continue

        X = sm.add_constant(standardized[model_features])
        ols = sm.OLS(y, X).fit()

        r2 = ols.rsquared
        r2_adj = ols.rsquared_adj
        delta_r2 = r2 - prev_r2
        p_curr = len(model_features)

        # F-test for the incremental R²
        # F = ((R²_full - R²_reduced) / (p_full - p_reduced)) /
        #     ((1 - R²_full) / (n - p_full - 1))
        df_num = p_curr - prev_p
        df_den = n - p_curr - 1
        if df_num > 0 and df_den > 0 and (1 - r2) > 0:
            f_change = (delta_r2 / df_num) / ((1 - r2) / df_den)
            f_p = 1 - sp_stats.f.cdf(f_change, df_num, df_den)
        else:
            f_change = np.nan
            f_p = np.nan

        # Print row
        f_str = f"{f_change:>8.1f}" if not np.isnan(f_change) else "     ---"
        p_str = f"{f_p:>10.4f}" if not np.isnan(f_p) else "       ---"
        if not np.isnan(f_p) and f_p < 0.001:
            p_str = "    <0.001"
        print(
            f"  {name:<26s} {r2:>7.4f} {r2_adj:>7.4f} {delta_r2:>7.4f} "
            f"{f_str} {p_str} {n:>5d}"
        )

        results_list.append({
            "name": name,
            "features": model_features,
            "n": n,
            "r2": r2,
            "r2_adj": r2_adj,
            "delta_r2": delta_r2,
            "f_change": f_change,
            "f_p": f_p,
            "ols_result": ols,
        })

        prev_r2 = r2
        prev_p = p_curr

    # Print coefficients for the final (full) model
    if results_list:
        final = results_list[-1]
        ols = final["ols_result"]
        print(f"\n{final['name']} Coefficients:")
        print(f"  {'Variable':<35s} {'β':>8s} {'SE':>8s} {'t':>8s} {'p':>10s} {'Sig':>5s}")
        print("  " + "-" * 68)
        for var in ols.params.index:
            if var == "const":
                continue
            label = VARIABLE_LABELS.get(var, var)
            beta = ols.params[var]
            se = ols.bse[var]
            t_val = ols.tvalues[var]
            p_val = ols.pvalues[var]
            sig = (
                "***" if p_val < 0.001
                else "**" if p_val < 0.01
                else "*" if p_val < 0.05
                else "." if p_val < 0.1
                else ""
            )
            print(
                f"  {label:<35s} {beta:>+8.3f} {se:>8.3f} "
                f"{t_val:>8.3f} {p_val:>10.4f} {sig:>5s}"
            )

    return results_list
