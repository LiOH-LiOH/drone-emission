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
REGRESSION_RESULT_VARIABLES = [
    "oracle_min_reduction_vs_ebike_pct",
]

EXCLUDED_REGRESSION_FEATURES = {
    "detour_ratio",
}

REGRESSION_CANDIDATE_FEATURES = [
    "building_count",
    "building_volume_total",
    "roadnet_length_total",
    "intersection_density",
    "avg_block_length",
    "road_connectivity",
    "road_distance_excess",
    "dead_end_proportion",
    "four_way_proportion",
    "orientation_entropy",
    "avg_street_length",
    "region_area_km2",
    "building_density",
    "road_density",
    "building_volume_density",
    "store_count",
    "sales_intensity",
    "store_density",
    "grid_emission_factor",
]

# Hierarchical model definitions — curated to avoid within-group collinearity.
# Each successive model adds one conceptual group of predictors.
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


def filter_regression_features(feature_vars: list) -> list:
    """Remove features excluded from regression models."""
    return [f for f in feature_vars if f not in EXCLUDED_REGRESSION_FEATURES]


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


def run_lasso_regression(
    df: pd.DataFrame,
    result_var: str,
    feature_vars: list = None,
    alphas: int = 100,
    cv: int = 10,
) -> dict:
    """Run Lasso regression with cross-validation for feature selection.

    Args:
        df: DataFrame with all variables.
        result_var: Dependent variable name.
        feature_vars: Candidate features. Uses all if None.
        alphas: Number of alpha values to try in LassoCV.
        cv: Number of cross-validation folds.

    Returns:
        Dict with keys: alpha, r2, features_selected, coefficients,
        feature_names, lasso_model.
    """
    from sklearn.linear_model import LassoCV
    from sklearn.preprocessing import StandardScaler

    if feature_vars is None:
        from drone_emission.analysis.correlation import get_all_feature_variables
        feature_vars = get_all_feature_variables()

    available = [f for f in feature_vars if f in df.columns]
    subset = df[[result_var] + available].dropna()
    n = len(subset)

    if n < 10:
        print(f"Insufficient observations ({n}) for Lasso")
        return {}

    result_label = VARIABLE_LABELS.get(result_var, result_var)
    print(f"\n{'=' * 72}")
    print(f"LASSO FEATURE SELECTION: {result_label}")
    print(f"{'=' * 72}")
    print(f"Candidates: {len(available)} features, N = {n}")

    X = subset[available].values
    y = subset[result_var].values

    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).ravel()

    lasso_cv = LassoCV(alphas=alphas, cv=cv, random_state=42, max_iter=10000)
    lasso_cv.fit(X_scaled, y_scaled)

    # Selected features (non-zero coefficients)
    coefs = lasso_cv.coef_
    selected_mask = np.abs(coefs) > 1e-6
    selected_features = [available[i] for i in range(len(available)) if selected_mask[i]]
    selected_coefs = coefs[selected_mask]

    # R² on full sample
    r2 = lasso_cv.score(X_scaled, y_scaled)

    print(f"\nOptimal α = {lasso_cv.alpha_:.6f}")
    print(f"R² = {r2:.4f}")
    print(f"Selected {len(selected_features)}/{len(available)} features:\n")

    # Sort by absolute coefficient
    order = np.argsort(np.abs(selected_coefs))[::-1]
    print(f"  {'Variable':<35s} {'β':>8s}")
    print("  " + "-" * 43)
    for idx in order:
        label = VARIABLE_LABELS.get(selected_features[idx], selected_features[idx])
        print(f"  {label:<35s} {selected_coefs[idx]:>+8.3f}")

    return {
        "alpha": lasso_cv.alpha_,
        "r2": r2,
        "features_selected": selected_features,
        "coefficients": dict(zip(selected_features, selected_coefs)),
        "feature_names": available,
        "all_coefs": dict(zip(available, coefs)),
        "lasso_model": lasso_cv,
    }


def analyze_hierarchical_regression(
    csv_path: str,
    output_dir: str = None,
    for_latex: bool = False,
    plot_title: bool = False,
) -> dict:
    """Run hierarchical + Lasso regression for all result variables.

    Orchestration function that loads data, runs hierarchical nested
    models and Lasso for each target, and generates comparison plots.

    Args:
        csv_path: Path to enriched region stats CSV.
        output_dir: Directory for output figures.
        for_latex: If True, save figures as PDF.
        plot_title: If True, show titles on plots.

    Returns:
        Dict with hierarchical and lasso results per result variable.
    """
    from drone_emission.analysis.correlation import load_region_stats

    _PROJECT_ROOT = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    )
    if output_dir is None:
        output_dir = os.path.join(_PROJECT_ROOT, "figures", "analysis")
    os.makedirs(output_dir, exist_ok=True)

    # Lazy import for visualization
    from drone_emission.visualization.analysis import plot_hierarchical_r2_comparison

    df = load_region_stats(csv_path)
    print(f"Loaded {len(df)} completed regions for hierarchical regression")

    all_results = {}

    for result_var in REGRESSION_RESULT_VARIABLES:
        if result_var not in df.columns:
            print(f"Skipping {result_var}: not in data")
            continue

        # Hierarchical regression
        hier_results = run_hierarchical_regression(df, result_var)

        # Lasso regression
        lasso_result = run_lasso_regression(
            df, result_var, filter_regression_features(REGRESSION_CANDIDATE_FEATURES)
        )

        all_results[result_var] = {
            "hierarchical": hier_results,
            "lasso": lasso_result,
        }

        # Plot R² comparison for this target
        if hier_results:
            fig_path = os.path.join(
                output_dir, f"hierarchical_r2_{result_var}.pdf"
            )
            plot_hierarchical_r2_comparison(
                hier_results,
                result_var,
                lasso_r2=lasso_result.get("r2"),
                save_path=fig_path,
                for_latex=for_latex,
                plot_title=plot_title,
            )

    return all_results


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
    from drone_emission.analysis.correlation import load_region_stats

    if result_vars is None:
        result_vars = REGRESSION_RESULT_VARIABLES

    df = load_region_stats(csv_path)
    print(f"Loaded {len(df)} completed regions")

    # Auto-detect available features if not specified
    if feature_vars is None:
        feature_vars = REGRESSION_CANDIDATE_FEATURES.copy()

    feature_vars = filter_regression_features(feature_vars)

    available_features = [f for f in feature_vars if f in df.columns]
    print(f"Available features: {len(available_features)}/{len(feature_vars)}")

    if len(available_features) < 2:
        print("Error: need at least 2 features for regression")
        return {}

    os.makedirs(output_dir, exist_ok=True)

    # Lazy import to avoid circular dependency
    from drone_emission.visualization.analysis import plot_regression_coefficients

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
