"""OLS regression analysis for cross-region emission results.

Provides standardized OLS regression, VIF multicollinearity checks,
and publication-ready output for identifying which urban features
significantly predict emission reduction outcomes.

Usage:
    from analysis.regression import run_regression_analysis
    run_regression_analysis(csv_path, output_dir)
"""

import os

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

from analysis.constants import VARIABLE_LABELS


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

    X = subset.values
    vif_data = []
    for i, feat in enumerate(available):
        vif_val = variance_inflation_factor(X, i)
        vif_data.append({"feature": feat, "VIF": vif_val})

    vif_df = pd.DataFrame(vif_data).sort_values("VIF", ascending=False)
    return vif_df


def run_ols_regression(
    df: pd.DataFrame,
    result_var: str,
    feature_vars: list,
) -> sm.regression.linear_model.RegressionResultsWrapper | None:
    """Run OLS regression with standardized (z-score) variables.

    Args:
        df: DataFrame with all variables.
        result_var: Dependent variable name.
        feature_vars: List of independent variable names.

    Returns:
        statsmodels RegressionResults, or None if insufficient data.
    """
    all_vars = [result_var] + feature_vars
    available = [v for v in all_vars if v in df.columns]
    missing = set(all_vars) - set(available)
    if missing:
        print(f"Warning: missing columns {missing}")

    available_features = [f for f in feature_vars if f in df.columns]
    if not available_features:
        print("Error: no feature variables available")
        return None

    subset = df[[result_var] + available_features].dropna()
    if len(subset) < len(available_features) + 2:
        print(
            f"Error: insufficient observations ({len(subset)}) "
            f"for {len(available_features)} features"
        )
        return None

    # Standardize all variables (z-scores)
    standardized = (subset - subset.mean()) / subset.std()

    y = standardized[result_var]
    X = sm.add_constant(standardized[available_features])

    model = sm.OLS(y, X)
    results = model.fit()

    return results


def print_regression_table(
    results: sm.regression.linear_model.RegressionResultsWrapper,
    result_var: str,
):
    """Print publication-ready regression results table.

    Args:
        results: OLS regression results.
        result_var: Name of dependent variable (for header).
    """
    result_label = VARIABLE_LABELS.get(result_var, result_var)
    print(f"\n{'=' * 70}")
    print(f"OLS REGRESSION: {result_label}")
    print(f"{'=' * 70}")

    print(f"{'Variable':<35s} {'β':>8s} {'SE':>8s} {'t':>8s} {'p':>10s} {'Sig':>5s}")
    print("-" * 70)

    for var in results.params.index:
        if var == "const":
            label = "(Intercept)"
        else:
            label = VARIABLE_LABELS.get(var, var)

        beta = results.params[var]
        se = results.bse[var]
        t_val = results.tvalues[var]
        p_val = results.pvalues[var]

        # Significance stars
        if p_val < 0.001:
            sig = "***"
        elif p_val < 0.01:
            sig = "**"
        elif p_val < 0.05:
            sig = "*"
        elif p_val < 0.1:
            sig = "."
        else:
            sig = ""

        print(f"  {label:<33s} {beta:>8.3f} {se:>8.3f} {t_val:>8.3f} {p_val:>10.4f} {sig:>5s}")

    print("-" * 70)
    print(f"  N = {int(results.nobs)}")
    print(f"  R² = {results.rsquared:.4f}")
    print(f"  Adjusted R² = {results.rsquared_adj:.4f}")
    print(f"  F-statistic = {results.fvalue:.3f} (p = {results.f_pvalue:.2e})")
    print(f"  Significance: *** p<0.001, ** p<0.01, * p<0.05, . p<0.1")


def run_regression_analysis(
    csv_path: str,
    output_dir: str,
    feature_vars: list = None,
    result_vars: list = None,
    for_latex: bool = False,
    plot_title: bool = False,
) -> dict:
    """Run full regression analysis: VIF check, OLS, coefficient plots.

    Args:
        csv_path: Path to enriched region stats CSV.
        output_dir: Directory for output figures.
        feature_vars: Feature variables to use. Auto-detected if None.
        result_vars: Result variables to predict. Uses defaults if None.
        for_latex: If True, save figures as PDF.
        plot_title: If True, show titles on plots.

    Returns:
        Dictionary with regression results per result variable.
    """
    from analysis.correlation import RESULT_VARIABLES, get_all_feature_variables

    if result_vars is None:
        result_vars = RESULT_VARIABLES

    df = pd.read_csv(csv_path)
    # Filter to completed cities
    df = df[df["total_ebike_lifecycle"].notna()].copy()
    print(f"Loaded {len(df)} completed regions")

    # Auto-detect available features if not specified
    if feature_vars is None:
        # Start with existing correlation features
        feature_vars = get_all_feature_variables()
        # Add new enrichment features if present
        enrichment_features = [
            "store_count",
            "total_monthly_sales",
            "store_density",
            "sales_intensity",
            "grid_emission_factor",
            "latitude",
            "longitude",
            "population_10k",
            "gdp_100m",
            "gdp_per_capita",
            "pop_density",
            "sales_per_capita",
        ]
        for f in enrichment_features:
            if f in df.columns and f not in feature_vars:
                feature_vars.append(f)

    available_features = [f for f in feature_vars if f in df.columns]
    print(f"Available features: {len(available_features)}/{len(feature_vars)}")

    if len(available_features) < 2:
        print("Error: need at least 2 features for regression")
        return {}

    os.makedirs(output_dir, exist_ok=True)

    # Lazy import to avoid circular dependency
    from visualization.analysis import plot_regression_coefficients

    results = {}

    for result_var in result_vars:
        if result_var not in df.columns:
            print(f"Skipping {result_var}: not in data")
            continue

        print(f"\n{'=' * 60}")
        result_label = VARIABLE_LABELS.get(result_var, result_var)
        print(f"REGRESSION ANALYSIS: {result_label}")
        print(f"{'=' * 60}")

        # Step 1: VIF check on features with sufficient data
        valid_mask = df[[result_var] + available_features].notna().all(axis=1)
        df_valid = df.loc[valid_mask]

        if len(df_valid) < len(available_features) + 2:
            print(f"Insufficient observations ({len(df_valid)}), skipping")
            continue

        vif_df = compute_vif(df_valid, available_features)
        if not vif_df.empty:
            print("\nVariance Inflation Factors:")
            for _, row in vif_df.iterrows():
                label = VARIABLE_LABELS.get(row["feature"], row["feature"])
                flag = " *** DROP" if row["VIF"] > 10 else (" (high)" if row["VIF"] > 5 else "")
                print(f"  {label:<35s} VIF = {row['VIF']:>8.2f}{flag}")

        # Step 2: Drop highly collinear features (VIF > 10)
        if not vif_df.empty:
            keep_features = vif_df[vif_df["VIF"] <= 10]["feature"].tolist()
            dropped = set(available_features) - set(keep_features)
            if dropped:
                dropped_labels = [VARIABLE_LABELS.get(f, f) for f in dropped]
                print(f"\nDropped (VIF > 10): {', '.join(dropped_labels)}")
        else:
            keep_features = available_features

        if len(keep_features) < 1:
            print("No features remaining after VIF filter")
            continue

        # Step 3a: Full model
        print("\n--- Full Model (all non-collinear features) ---")
        full_results = run_ols_regression(df, result_var, keep_features)
        if full_results is not None:
            print_regression_table(full_results, result_var)

            # Plot coefficients (exclude intercept)
            coef_path = os.path.join(
                output_dir, f"regression_coefficients_{result_var}.pdf"
            )
            plot_regression_coefficients(
                full_results,
                result_var,
                save_path=coef_path,
                for_latex=for_latex,
                plot_title=plot_title,
            )

        # Step 3b: Parsimonious model (p < 0.05 only)
        if full_results is not None:
            sig_features = [
                var
                for var in full_results.pvalues.index
                if var != "const" and full_results.pvalues[var] < 0.05
            ]
            if sig_features and len(sig_features) < len(keep_features):
                print(f"\n--- Parsimonious Model (p < 0.05 features only) ---")
                parsi_results = run_ols_regression(df, result_var, sig_features)
                if parsi_results is not None:
                    print_regression_table(parsi_results, result_var)

                    coef_path_p = os.path.join(
                        output_dir,
                        f"regression_coefficients_{result_var}_parsimonious.pdf",
                    )
                    plot_regression_coefficients(
                        parsi_results,
                        result_var,
                        save_path=coef_path_p,
                        for_latex=for_latex,
                        plot_title=plot_title,
                    )
                    results[result_var] = {
                        "full": full_results,
                        "parsimonious": parsi_results,
                    }
                else:
                    results[result_var] = {"full": full_results}
            elif not sig_features:
                print("\nNo significant features at p < 0.05")
                results[result_var] = {"full": full_results}
            else:
                print("\nAll features significant — parsimonious = full model")
                results[result_var] = {"full": full_results}
        else:
            print("Full model failed to fit")

    return results
