"""Reproducible city-attribution analysis for manuscript Results section R2.

The analysis distinguishes pairwise descriptive samples from the fixed common
sample used by nested regression. Missing city detour statistics may be
reconstructed from existing OD files using the same annual-sales-weighted
definition as the road-network stage.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from statsmodels.stats.diagnostic import het_breuschpagan, linear_reset
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.outliers_influence import OLSInfluence

from drone_emission.analysis.regression import compute_vif, run_hierarchical_regression


R2_OUTCOME = "oracle_min_reduction_vs_ebike_pct"
R2_ABSOLUTE_OUTCOME = "oracle_min_reduction_vs_ebike_t"
R2_EQUIVALENCE_BOUND = 0.10
R2_BOOTSTRAP_REPLICATES = 5000
R2_RANDOM_SEED = 20260729

R2_HIERARCHICAL_MODELS = {
    "M1: Grid": ["grid_emission_factor"],
    "M2: + Detour": ["grid_emission_factor", "detour_ratio"],
    "M3: + Other road form": [
        "grid_emission_factor",
        "detour_ratio",
        "avg_street_length",
        "dead_end_proportion",
        "orientation_entropy",
        "four_way_proportion",
    ],
    "M4: + Socioeconomic": [
        "grid_emission_factor",
        "detour_ratio",
        "avg_street_length",
        "dead_end_proportion",
        "orientation_entropy",
        "four_way_proportion",
        "population_10k",
        "pop_density",
        "gdp_per_capita",
        "store_density",
    ],
}

R2_SCREEN_VARIABLES = (
    "grid_emission_factor",
    "detour_ratio",
    "avg_street_length",
    "dead_end_proportion",
    "four_way_proportion",
    "orientation_entropy",
    "latitude",
    "population_10k",
    "store_density",
    "pop_density",
    "gdp_per_capita",
    "building_density",
    "building_count",
    "road_density",
)

R2_MATRIX_VARIABLES = (
    R2_OUTCOME,
    "grid_emission_factor",
    "detour_ratio",
    "avg_street_length",
    "avg_block_length",
    "dead_end_proportion",
    "four_way_proportion",
    "orientation_entropy",
    "intersection_density",
    "avg_road_distance_m",
    "avg_straight_distance_m",
    "road_density",
    "roadnet_length_total",
    "population_10k",
    "pop_density",
    "gdp_per_capita",
    "building_density",
    "building_count",
    "building_volume_total",
    "store_density",
    "sales_intensity",
    "region_area_km2",
    "latitude",
    "longitude",
)


def compute_weighted_detour_stats(od_path: str | Path) -> dict:
    """Compute annual-sales-weighted city distance statistics from an OD CSV."""
    od_path = Path(od_path)
    required = ("road_distance_m", "distance_km", "annual_sales")
    od_df = pd.read_csv(od_path, usecols=list(required))
    road_m = pd.to_numeric(od_df["road_distance_m"], errors="coerce").to_numpy()
    air_km = pd.to_numeric(od_df["distance_km"], errors="coerce").to_numpy()
    annual_sales = pd.to_numeric(
        od_df["annual_sales"], errors="coerce"
    ).to_numpy()
    valid = (
        np.isfinite(road_m)
        & np.isfinite(air_km)
        & np.isfinite(annual_sales)
        & (road_m >= 0)
        & (air_km > 0)
        & (annual_sales > 0)
    )
    if not valid.any():
        raise ValueError(f"No valid weighted paths in {od_path}")

    weights = annual_sales[valid]
    weight_sum = float(weights.sum())
    avg_road_m = float(np.sum(road_m[valid] * weights) / weight_sum)
    avg_straight_m = float(
        np.sum(air_km[valid] * 1000.0 * weights) / weight_sum
    )
    return {
        "avg_road_distance_m": avg_road_m,
        "avg_straight_distance_m": avg_straight_m,
        "detour_ratio": avg_road_m / avg_straight_m,
        "valid_path_count": int(valid.sum()),
        "invalid_path_count": int((~valid).sum()),
        "annual_sales_sum": weight_sum,
        "source_path": str(od_path),
        "source_size_bytes": int(od_path.stat().st_size),
        "source_mtime_ns": int(od_path.stat().st_mtime_ns),
    }


def prepare_r2_city_data(
    enriched_csv_path: str | Path,
    data_dir: str | Path,
    supplement_missing_detour: bool = True,
    region_keys: set[str] | None = None,
) -> tuple[pd.DataFrame, list[dict]]:
    """Load city data, derive Oracle outcomes, and supplement detour statistics."""
    enriched_csv_path = Path(enriched_csv_path)
    data_dir = Path(data_dir)
    df = pd.read_csv(enriched_csv_path)
    if "region_level" in df.columns:
        df = df[df["region_level"] == "city"].copy()
    if region_keys is not None:
        df = df[df["region_key"].astype(str).isin(region_keys)].copy()
    if "total_ebike_lifecycle" in df.columns:
        df = df[pd.to_numeric(df["total_ebike_lifecycle"], errors="coerce").notna()]
    df = df.copy()

    ebike = pd.to_numeric(df["total_ebike_lifecycle"], errors="coerce")
    oracle = pd.to_numeric(df["oracle_min_lifecycle_kg"], errors="coerce")
    df[R2_OUTCOME] = np.where(ebike > 0, (1.0 - oracle / ebike) * 100.0, np.nan)
    df[R2_ABSOLUTE_OUTCOME] = (ebike - oracle) / 1000.0

    supplements = []
    if supplement_missing_detour:
        if "detour_ratio" not in df.columns:
            df["detour_ratio"] = np.nan
        for idx in df.index[df["detour_ratio"].isna()]:
            region_key = str(df.at[idx, "region_key"])
            level = str(df.at[idx, "region_level"]) if "region_level" in df else "city"
            od_path = data_dir / level / region_key / f"{region_key}_od_pairs.csv"
            if not od_path.exists():
                continue
            try:
                computed = compute_weighted_detour_stats(od_path)
            except (OSError, ValueError, KeyError) as exc:
                supplements.append({
                    "region_key": region_key,
                    "status": "failed",
                    "reason": str(exc),
                    "source_path": str(od_path),
                })
                continue
            for column in (
                "avg_road_distance_m",
                "avg_straight_distance_m",
                "detour_ratio",
            ):
                if column not in df.columns:
                    df[column] = np.nan
                df.at[idx, column] = computed[column]
            supplements.append({
                "region_key": region_key,
                "status": "supplemented",
                **computed,
            })

    return df.reset_index(drop=True), supplements


def _pairwise_correlation(
    df: pd.DataFrame,
    x_var: str,
    y_var: str,
    method: str,
) -> dict:
    pair = df[[x_var, y_var]].apply(pd.to_numeric, errors="coerce").dropna()
    pair = pair[np.isfinite(pair[x_var]) & np.isfinite(pair[y_var])]
    if len(pair) < 3 or pair[x_var].nunique() < 2 or pair[y_var].nunique() < 2:
        return {"x": x_var, "y": y_var, "method": method, "n": len(pair),
                "coefficient": None, "p_value": None}
    if method == "pearson":
        coefficient, p_value = stats.pearsonr(pair[x_var], pair[y_var])
    elif method == "spearman":
        coefficient, p_value = stats.spearmanr(pair[x_var], pair[y_var])
    else:
        raise ValueError(f"Unsupported correlation method: {method}")
    clipped = float(np.clip(coefficient, -0.999999, 0.999999))
    standard_error = 1.0 / math.sqrt(len(pair) - 3)
    fisher_value = np.arctanh(clipped)
    ci95 = np.tanh(
        [fisher_value - 1.96 * standard_error,
         fisher_value + 1.96 * standard_error]
    )
    ci90 = np.tanh(
        [fisher_value - stats.norm.ppf(0.95) * standard_error,
         fisher_value + stats.norm.ppf(0.95) * standard_error]
    )
    lower_z = (
        fisher_value - np.arctanh(-R2_EQUIVALENCE_BOUND)
    ) / standard_error
    upper_z = (
        fisher_value - np.arctanh(R2_EQUIVALENCE_BOUND)
    ) / standard_error
    p_lower = float(stats.norm.sf(lower_z))
    p_upper = float(stats.norm.cdf(upper_z))
    tost_p = max(p_lower, p_upper)
    return {
        "x": x_var,
        "y": y_var,
        "method": method,
        "n": int(len(pair)),
        "coefficient": float(coefficient),
        "p_value": float(p_value),
        "ci95": [float(value) for value in ci95],
        "equivalence": {
            "bound_abs": R2_EQUIVALENCE_BOUND,
            "ci90": [float(value) for value in ci90],
            "p_lower": p_lower,
            "p_upper": p_upper,
            "p_tost": float(tost_p),
            "equivalent_0_05": bool(tost_p < 0.05),
            "method": "Fisher-z TOST",
        },
    }


def _adjust_pvalues(records: list[dict], method: str) -> list[dict]:
    """Adjust one declared family of p-values and retain its metadata."""
    valid_indices = [
        index for index, record in enumerate(records)
        if record.get("p_value") is not None
    ]
    adjusted_values = []
    if valid_indices:
        adjusted_values = multipletests(
            [records[index]["p_value"] for index in valid_indices],
            alpha=0.05,
            method=method,
        )[1]
    adjusted_lookup = dict(zip(valid_indices, adjusted_values))
    adjusted = []
    for index, record in enumerate(records):
        item = dict(record)
        item["p_adjusted"] = (
            float(adjusted_lookup[index]) if index in adjusted_lookup else None
        )
        item["significant_adjusted_0_05"] = bool(
            item["p_adjusted"] is not None and item["p_adjusted"] < 0.05
        )
        item["adjustment_method"] = method
        item["family_size"] = len(valid_indices)
        adjusted.append(item)
    return adjusted


def _screen_correlations(df: pd.DataFrame) -> dict:
    available = [var for var in R2_SCREEN_VARIABLES if var in df.columns]
    outcome = [
        _pairwise_correlation(df, variable, R2_OUTCOME, "pearson")
        for variable in available
    ]
    gef = [
        _pairwise_correlation(df, variable, "grid_emission_factor", "pearson")
        for variable in available
        if variable != "grid_emission_factor"
    ]
    return {
        "variables": available,
        "outcome": _adjust_pvalues(outcome, "holm"),
        "grid_emission_factor": _adjust_pvalues(gef, "holm"),
        "adjustment_method": "holm",
    }


def _correlation_matrix(df: pd.DataFrame, method: str) -> dict:
    variables = [var for var in R2_MATRIX_VARIABLES if var in df.columns]
    pair_records = []
    pair_lookup = {}
    for i, first in enumerate(variables):
        for second in variables[i + 1:]:
            record = _pairwise_correlation(df, first, second, method)
            pair_records.append(record)
    pair_records = _adjust_pvalues(pair_records, "fdr_bh")
    for record in pair_records:
        pair_lookup[(record["x"], record["y"])] = record
        pair_lookup[(record["y"], record["x"])] = record

    coefficients = []
    adjusted_p = []
    sample_sizes = []
    for first in variables:
        coefficient_row = []
        p_row = []
        n_row = []
        for second in variables:
            if first == second:
                coefficient_row.append(1.0)
                p_row.append(0.0)
                n_row.append(int(pd.to_numeric(df[first], errors="coerce").notna().sum()))
                continue
            record = pair_lookup[(first, second)]
            coefficient_row.append(record["coefficient"])
            p_row.append(record["p_adjusted"])
            n_row.append(record["n"])
        coefficients.append(coefficient_row)
        adjusted_p.append(p_row)
        sample_sizes.append(n_row)
    return {
        "method": method,
        "variables": variables,
        "coefficients": coefficients,
        "p_adjusted": adjusted_p,
        "adjustment_method": "fdr_bh",
        "sample_sizes": sample_sizes,
        "family_size": pair_records[0]["family_size"] if pair_records else 0,
        "pairs": pair_records,
    }


def _result_value(result, attribute: str, name: str) -> float:
    """Read one named model result value across statsmodels result wrappers."""
    names = list(result.model.exog_names)
    values = np.asarray(getattr(result, attribute), dtype=float)
    return float(values[names.index(name)])


def _beta_tost(beta: float, standard_error: float, df_resid: float) -> dict:
    """Test whether a standardized coefficient lies within the preset bound."""
    lower_stat = (beta + R2_EQUIVALENCE_BOUND) / standard_error
    upper_stat = (beta - R2_EQUIVALENCE_BOUND) / standard_error
    p_lower = float(stats.t.sf(lower_stat, df_resid))
    p_upper = float(stats.t.cdf(upper_stat, df_resid))
    critical = float(stats.t.ppf(0.95, df_resid))
    p_tost = max(p_lower, p_upper)
    return {
        "bound_abs": R2_EQUIVALENCE_BOUND,
        "ci90": [
            float(beta - critical * standard_error),
            float(beta + critical * standard_error),
        ],
        "p_lower": p_lower,
        "p_upper": p_upper,
        "p_tost": float(p_tost),
        "equivalent_0_05": bool(p_tost < 0.05),
        "method": "two-one-sided t tests on standardized beta",
    }


def _coefficient_record(result, variable: str, include_tost: bool = True) -> dict:
    """Serialize one robust regression coefficient with its confidence interval."""
    beta = _result_value(result, "params", variable)
    standard_error = _result_value(result, "bse", variable)
    names = list(result.model.exog_names)
    interval = np.asarray(result.conf_int(alpha=0.05), dtype=float)[names.index(variable)]
    record = {
        "variable": variable,
        "beta_standardized": beta,
        "standard_error": standard_error,
        "test_statistic": _result_value(result, "tvalues", variable),
        "p_value": _result_value(result, "pvalues", variable),
        "ci95": [float(interval[0]), float(interval[1])],
    }
    if include_tost:
        record["equivalence"] = _beta_tost(
            beta, standard_error, float(result.df_resid)
        )
    return record


def _wald_block_test(result, variables: list[str]) -> dict:
    """Run one joint Wald test for a declared predictor block."""
    names = list(result.model.exog_names)
    tested = [variable for variable in variables if variable in names]
    restrictions = np.zeros((len(tested), len(names)))
    for row, variable in enumerate(tested):
        restrictions[row, names.index(variable)] = 1.0
    test = result.wald_test(restrictions, scalar=True)
    return {
        "variables": tested,
        "statistic": float(np.asarray(test.statistic).squeeze()),
        "p_value": float(np.asarray(test.pvalue).squeeze()),
        "df_num": int(len(tested)),
        "method": "HC3 robust Wald test",
    }


def _r2_common_sample(df: pd.DataFrame) -> pd.DataFrame:
    """Return the fixed complete-case sample used by every nested R2 model."""
    final_features = list(R2_HIERARCHICAL_MODELS.values())[-1]
    columns = [R2_OUTCOME, *final_features, "province", "region_key"]
    available = [column for column in columns if column in df.columns]
    sample = df[available].copy()
    numeric = [R2_OUTCOME, *final_features]
    sample[numeric] = sample[numeric].apply(pd.to_numeric, errors="coerce")
    sample = sample.dropna(subset=numeric)
    sample = sample[np.isfinite(sample[numeric]).all(axis=1)]
    return sample.reset_index(drop=True)


def _robust_hierarchical_models(df: pd.DataFrame) -> list[dict]:
    """Fit HC3-primary and province-clustered versions of M1--M4."""
    sample = _r2_common_sample(df)
    numeric = [R2_OUTCOME, *list(R2_HIERARCHICAL_MODELS.values())[-1]]
    standardized = (sample[numeric] - sample[numeric].mean()) / sample[numeric].std()
    y = standardized[R2_OUTCOME]
    previous_features: list[str] = []
    records = []
    for name, features in R2_HIERARCHICAL_MODELS.items():
        x = sm.add_constant(standardized[features], has_constant="add")
        hc3 = sm.OLS(y, x).fit(cov_type="HC3", use_t=True)
        cluster = sm.OLS(y, x).fit(
            cov_type="cluster",
            cov_kwds={
                "groups": sample["province"],
                "use_correction": True,
                "df_correction": True,
            },
            use_t=True,
        )
        added = [feature for feature in features if feature not in previous_features]
        records.append({
            "name": name,
            "n": int(len(sample)),
            "province_clusters": int(sample["province"].nunique()),
            "added_block": added,
            "hc3_block_test": _wald_block_test(hc3, added),
            "hc3_coefficients": [
                _coefficient_record(hc3, feature) for feature in features
            ],
            "province_cluster_coefficients": [
                _coefficient_record(cluster, feature, include_tost=False)
                for feature in features
            ],
        })
        previous_features = list(features)
    return records


def _fit_r2_values(y: np.ndarray, x: np.ndarray) -> float:
    """Fit least squares and return R-squared for bootstrap resamples."""
    design = np.column_stack([np.ones(len(x)), x])
    fitted = design @ np.linalg.lstsq(design, y, rcond=None)[0]
    denominator = float(np.sum((y - np.mean(y)) ** 2))
    if denominator <= 0:
        return np.nan
    return float(1.0 - np.sum((y - fitted) ** 2) / denominator)


def _bootstrap_hierarchical_delta_r2(
    df: pd.DataFrame,
    replicates: int = R2_BOOTSTRAP_REPLICATES,
) -> list[dict]:
    """Bootstrap nested-model incremental R-squared on the common sample."""
    sample = _r2_common_sample(df)
    y = sample[R2_OUTCOME].to_numpy(dtype=float)
    rng = np.random.default_rng(R2_RANDOM_SEED)
    deltas = [[] for _ in R2_HIERARCHICAL_MODELS]
    feature_sets = list(R2_HIERARCHICAL_MODELS.values())
    for _ in range(replicates):
        positions = rng.integers(0, len(sample), len(sample))
        sampled_y = y[positions]
        previous = 0.0
        for index, features in enumerate(feature_sets):
            sampled_x = sample[features].to_numpy(dtype=float)[positions]
            r2_value = _fit_r2_values(sampled_y, sampled_x)
            if np.isfinite(r2_value):
                deltas[index].append(r2_value - previous)
                previous = r2_value
    records = []
    for name, values in zip(R2_HIERARCHICAL_MODELS, deltas):
        interval = np.percentile(values, [2.5, 97.5])
        records.append({
            "name": name,
            "replicates_requested": replicates,
            "replicates_valid": int(len(values)),
            "delta_r2_ci95_percentile": [float(interval[0]), float(interval[1])],
        })
    return records


def _serialize_hierarchical(results: list[dict]) -> list[dict]:
    serialized = []
    for result in results:
        ols = result["ols_result"]
        coefficients = []
        for variable in ols.params.index:
            if variable == "const":
                continue
            coefficients.append({
                "variable": variable,
                "beta_standardized": float(ols.params[variable]),
                "standard_error": float(ols.bse[variable]),
                "t_value": float(ols.tvalues[variable]),
                "p_value": float(ols.pvalues[variable]),
            })
        serialized.append({
            "name": result["name"],
            "features": list(result["features"]),
            "n": int(result["n"]),
            "r2": float(result["r2"]),
            "adjusted_r2": float(result["r2_adj"]),
            "delta_r2": float(result["delta_r2"]),
            "f_change": float(result["f_change"]),
            "f_change_p": float(result["f_p"]),
            "coefficients": coefficients,
        })
    return serialized


def _prepare_core_sample(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Prepare the complete city sample and z-scored core mechanism variables."""
    columns = [
        R2_OUTCOME,
        "grid_emission_factor",
        "detour_ratio",
        "province",
        "region_key",
        "total_odpair_annual_sales",
    ]
    sample = df[[column for column in columns if column in df.columns]].copy()
    numeric = [R2_OUTCOME, "grid_emission_factor", "detour_ratio"]
    sample[numeric] = sample[numeric].apply(pd.to_numeric, errors="coerce")
    sample = sample.dropna(subset=[*numeric, "province"])
    sample = sample[np.isfinite(sample[numeric]).all(axis=1)].reset_index(drop=True)
    standardized = (sample[numeric] - sample[numeric].mean()) / sample[numeric].std()
    standardized["gef_x_detour"] = (
        standardized["grid_emission_factor"] * standardized["detour_ratio"]
    )
    return sample, standardized


def _fit_interaction_model(df: pd.DataFrame) -> dict:
    """Fit the continuous grid-by-detour mechanism model with robust inference."""
    sample, standardized = _prepare_core_sample(df)
    features = ["grid_emission_factor", "detour_ratio", "gef_x_detour"]
    x = sm.add_constant(standardized[features], has_constant="add")
    y = standardized[R2_OUTCOME]
    classic = sm.OLS(y, x).fit()
    hc3 = sm.OLS(y, x).fit(cov_type="HC3", use_t=True)
    cluster = sm.OLS(y, x).fit(
        cov_type="cluster",
        cov_kwds={
            "groups": sample["province"],
            "use_correction": True,
            "df_correction": True,
        },
        use_t=True,
    )
    names = list(hc3.model.exog_names)
    covariance = np.asarray(hc3.cov_params(), dtype=float)
    detour_index = names.index("detour_ratio")
    interaction_index = names.index("gef_x_detour")
    simple_slopes = []
    for gef_z, label in ((-1.0, "GEF -1 SD"), (0.0, "Mean GEF"), (1.0, "GEF +1 SD")):
        beta = (
            _result_value(hc3, "params", "detour_ratio")
            + gef_z * _result_value(hc3, "params", "gef_x_detour")
        )
        variance = (
            covariance[detour_index, detour_index]
            + gef_z ** 2 * covariance[interaction_index, interaction_index]
            + 2.0 * gef_z * covariance[detour_index, interaction_index]
        )
        standard_error = math.sqrt(max(float(variance), 0.0))
        critical = float(stats.t.ppf(0.975, hc3.df_resid))
        statistic = beta / standard_error
        simple_slopes.append({
            "label": label,
            "gef_z": gef_z,
            "beta_standardized": float(beta),
            "standard_error": float(standard_error),
            "ci95": [
                float(beta - critical * standard_error),
                float(beta + critical * standard_error),
            ],
            "p_value": float(2.0 * stats.t.sf(abs(statistic), hc3.df_resid)),
        })
    numeric = [R2_OUTCOME, "grid_emission_factor", "detour_ratio"]
    return {
        "formula": "reduction_z ~ gef_z + detour_z + gef_z:detour_z",
        "role": "separate continuous mechanism model; not an M1-M4 layer",
        "n": int(len(sample)),
        "province_clusters": int(sample["province"].nunique()),
        "r2": float(classic.rsquared),
        "adjusted_r2": float(classic.rsquared_adj),
        "standardization": {
            column: {
                "mean": float(sample[column].mean()),
                "sd": float(sample[column].std()),
            }
            for column in numeric
        },
        "hc3_intercept": _result_value(hc3, "params", "const"),
        "hc3_coefficients": [
            _coefficient_record(hc3, feature) for feature in features
        ],
        "province_cluster_coefficients": [
            _coefficient_record(cluster, feature, include_tost=False)
            for feature in features
        ],
        "simple_detour_slopes": simple_slopes,
    }


def _nonlinear_core_sensitivity(df: pd.DataFrame) -> dict:
    """Test quadratic core terms after RESET flags additive-model misspecification."""
    sample, standardized = _prepare_core_sample(df)
    standardized["grid_emission_factor_sq"] = (
        standardized["grid_emission_factor"] ** 2
    )
    standardized["detour_ratio_sq"] = standardized["detour_ratio"] ** 2
    base = ["grid_emission_factor", "detour_ratio"]
    nonlinear = ["grid_emission_factor_sq", "detour_ratio_sq", "gef_x_detour"]
    full_features = [*base, *nonlinear]
    y = standardized[R2_OUTCOME]
    reduced = sm.OLS(
        y, sm.add_constant(standardized[base], has_constant="add")
    ).fit()
    design = sm.add_constant(standardized[full_features], has_constant="add")
    classic = sm.OLS(y, design).fit()
    hc3 = sm.OLS(y, design).fit(cov_type="HC3", use_t=True)
    cluster = sm.OLS(y, design).fit(
        cov_type="cluster",
        cov_kwds={
            "groups": sample["province"],
            "use_correction": True,
            "df_correction": True,
        },
        use_t=True,
    )
    reset = linear_reset(classic, power=2, use_f=True)
    return {
        "formula": (
            "reduction_z ~ gef_z + detour_z + gef_z^2 + detour_z^2 + "
            "gef_z:detour_z"
        ),
        "n": int(len(sample)),
        "r2": float(classic.rsquared),
        "adjusted_r2": float(classic.rsquared_adj),
        "incremental_r2_over_additive": float(classic.rsquared - reduced.rsquared),
        "nonlinear_block_hc3_wald": _wald_block_test(hc3, nonlinear),
        "hc3_coefficients": [
            _coefficient_record(hc3, feature, include_tost=False)
            for feature in full_features
        ],
        "province_cluster_coefficients": [
            _coefficient_record(cluster, feature, include_tost=False)
            for feature in full_features
        ],
        "ramsey_reset_after_quadratic": {
            "f_statistic": float(np.asarray(reset.fvalue).squeeze()),
            "p_value": float(np.asarray(reset.pvalue).squeeze()),
            "power": 2,
        },
        "interpretation": (
            "Specification sensitivity only; the prespecified additive M1-M4 "
            "hierarchy remains the primary attribution model."
        ),
    }


def _bootstrap_core_coefficients(
    df: pd.DataFrame,
    replicates: int = R2_BOOTSTRAP_REPLICATES,
) -> dict:
    """Bootstrap additive and interaction-model standardized coefficients."""
    sample, _ = _prepare_core_sample(df)
    rng = np.random.default_rng(R2_RANDOM_SEED + 1)
    stored = {
        "additive": {"grid_emission_factor": [], "detour_ratio": []},
        "interaction": {
            "grid_emission_factor": [],
            "detour_ratio": [],
            "gef_x_detour": [],
        },
    }
    numeric = [R2_OUTCOME, "grid_emission_factor", "detour_ratio"]
    for _ in range(replicates):
        positions = rng.integers(0, len(sample), len(sample))
        sampled = sample.iloc[positions][numeric]
        scales = sampled.std()
        if (scales <= 0).any():
            continue
        z = (sampled - sampled.mean()) / scales
        interaction = z["grid_emission_factor"] * z["detour_ratio"]
        designs = {
            "additive": np.column_stack([
                np.ones(len(z)),
                z["grid_emission_factor"],
                z["detour_ratio"],
            ]),
            "interaction": np.column_stack([
                np.ones(len(z)),
                z["grid_emission_factor"],
                z["detour_ratio"],
                interaction,
            ]),
        }
        names = {
            "additive": ["const", "grid_emission_factor", "detour_ratio"],
            "interaction": [
                "const", "grid_emission_factor", "detour_ratio", "gef_x_detour"
            ],
        }
        for model_name, design in designs.items():
            coefficients = np.linalg.lstsq(
                design, z[R2_OUTCOME].to_numpy(dtype=float), rcond=None
            )[0]
            for variable, coefficient in zip(names[model_name], coefficients):
                if variable != "const":
                    stored[model_name][variable].append(float(coefficient))
    return {
        "replicates_requested": replicates,
        "random_seed": R2_RANDOM_SEED + 1,
        **{
            model_name: {
                variable: {
                    "replicates_valid": int(len(values)),
                    "median": float(np.median(values)),
                    "ci95_percentile": [
                        float(value) for value in np.percentile(values, [2.5, 97.5])
                    ],
                }
                for variable, values in variables.items()
            }
            for model_name, variables in stored.items()
        },
    }


def _weighted_core_sensitivity(df: pd.DataFrame) -> dict:
    """Fit annual-order-weighted additive and interaction sensitivity models."""
    sample, standardized = _prepare_core_sample(df)
    if "total_odpair_annual_sales" not in sample.columns:
        return {"available": False, "reason": "missing total_odpair_annual_sales"}
    weights = pd.to_numeric(
        sample["total_odpair_annual_sales"], errors="coerce"
    ).to_numpy(dtype=float)
    valid = np.isfinite(weights) & (weights > 0)
    weights = weights[valid] / np.mean(weights[valid])
    y = standardized.loc[valid, R2_OUTCOME]
    records = {}
    for model_name, features in {
        "additive": ["grid_emission_factor", "detour_ratio"],
        "interaction": ["grid_emission_factor", "detour_ratio", "gef_x_detour"],
    }.items():
        x = sm.add_constant(standardized.loc[valid, features], has_constant="add")
        result = sm.WLS(y, x, weights=weights).fit(cov_type="HC3", use_t=True)
        records[model_name] = {
            "n": int(valid.sum()),
            "coefficients": [
                _coefficient_record(result, feature, include_tost=False)
                for feature in features
            ],
        }
    return {
        "available": True,
        "estimand": "annual-order-weighted cross-city sensitivity",
        "weight": "total_odpair_annual_sales normalized to mean one",
        "models": records,
    }


def _leave_one_province_out(df: pd.DataFrame) -> dict:
    """Summarize coefficient stability after omitting each province in turn."""
    sample, _ = _prepare_core_sample(df)
    stored = {
        "additive": {"grid_emission_factor": [], "detour_ratio": []},
        "interaction": {
            "grid_emission_factor": [],
            "detour_ratio": [],
            "gef_x_detour": [],
        },
    }
    for province in sorted(sample["province"].unique()):
        reduced = sample[sample["province"] != province].copy()
        numeric = [R2_OUTCOME, "grid_emission_factor", "detour_ratio"]
        z = (reduced[numeric] - reduced[numeric].mean()) / reduced[numeric].std()
        z["gef_x_detour"] = z["grid_emission_factor"] * z["detour_ratio"]
        for model_name, features in {
            "additive": ["grid_emission_factor", "detour_ratio"],
            "interaction": ["grid_emission_factor", "detour_ratio", "gef_x_detour"],
        }.items():
            design = sm.add_constant(z[features], has_constant="add")
            result = sm.OLS(z[R2_OUTCOME], design).fit()
            for variable in features:
                stored[model_name][variable].append({
                    "omitted_province": str(province),
                    "beta_standardized": float(result.params[variable]),
                })
    return {
        model_name: {
            variable: {
                "minimum": float(min(item["beta_standardized"] for item in values)),
                "maximum": float(max(item["beta_standardized"] for item in values)),
                "sign_consistent": bool(
                    all(item["beta_standardized"] > 0 for item in values)
                    or all(item["beta_standardized"] < 0 for item in values)
                ),
                "estimates": values,
            }
            for variable, values in variables.items()
        }
        for model_name, variables in stored.items()
    }


def _socioeconomic_alternative_models(df: pd.DataFrame) -> dict:
    """Test socioeconomic predictors separately after the M3 road-form block."""
    baseline = list(R2_HIERARCHICAL_MODELS["M3: + Other road form"])
    socioeconomic = ["population_10k", "pop_density", "gdp_per_capita", "store_density"]
    records = []
    for variable in socioeconomic:
        numeric = [R2_OUTCOME, *baseline, variable]
        sample = df[[*numeric, "province"]].copy()
        sample[numeric] = sample[numeric].apply(pd.to_numeric, errors="coerce")
        sample = sample.dropna(subset=[*numeric, "province"])
        sample = sample[np.isfinite(sample[numeric]).all(axis=1)]
        z = (sample[numeric] - sample[numeric].mean()) / sample[numeric].std()
        x = sm.add_constant(z[[*baseline, variable]], has_constant="add")
        hc3 = sm.OLS(z[R2_OUTCOME], x).fit(cov_type="HC3", use_t=True)
        records.append({
            "variable": variable,
            "n": int(len(sample)),
            "coefficient": _coefficient_record(hc3, variable),
        })
    return {
        "baseline": "M3: Grid + detour + other road form",
        "models": records,
    }


def _geographic_sensitivity(df: pd.DataFrame) -> dict:
    """Test residual road-form associations after a quadratic geographic surface."""
    core = ["grid_emission_factor", "detour_ratio"]
    road = [
        "avg_street_length",
        "dead_end_proportion",
        "orientation_entropy",
        "four_way_proportion",
    ]
    numeric = [R2_OUTCOME, *core, *road, "latitude", "longitude"]
    sample = df[numeric].apply(pd.to_numeric, errors="coerce").dropna()
    sample = sample[np.isfinite(sample).all(axis=1)]
    z = (sample - sample.mean()) / sample.std()
    z["latitude_sq"] = z["latitude"] ** 2
    z["longitude_sq"] = z["longitude"] ** 2
    z["latitude_x_longitude"] = z["latitude"] * z["longitude"]
    geography = [
        "latitude", "longitude", "latitude_sq", "longitude_sq",
        "latitude_x_longitude",
    ]
    reduced_features = [*core, *geography]
    full_features = [*reduced_features, *road]
    reduced = sm.OLS(
        z[R2_OUTCOME], sm.add_constant(z[reduced_features], has_constant="add")
    ).fit()
    full = sm.OLS(
        z[R2_OUTCOME], sm.add_constant(z[full_features], has_constant="add")
    ).fit(cov_type="HC3", use_t=True)
    return {
        "n": int(len(sample)),
        "geographic_terms": geography,
        "road_block_partial_r2": float(full.rsquared - reduced.rsquared),
        "road_block_hc3_wald": _wald_block_test(full, road),
        "road_coefficients_hc3": [
            _coefficient_record(full, variable) for variable in road
        ],
        "interpretation_rule": (
            "Report attenuation and residual association; do not assign a binary proxy label."
        ),
    }


def _model_diagnostics(df: pd.DataFrame) -> dict:
    """Audit heteroskedasticity, functional form, and influential cities for M4."""
    sample = _r2_common_sample(df)
    features = list(R2_HIERARCHICAL_MODELS.values())[-1]
    numeric = [R2_OUTCOME, *features]
    z = (sample[numeric] - sample[numeric].mean()) / sample[numeric].std()
    result = sm.OLS(
        z[R2_OUTCOME], sm.add_constant(z[features], has_constant="add")
    ).fit()
    bp_lm, bp_lm_p, bp_f, bp_f_p = het_breuschpagan(result.resid, result.model.exog)
    reset = linear_reset(result, power=2, use_f=True)
    influence = OLSInfluence(result)
    cooks = np.asarray(influence.cooks_distance[0], dtype=float)
    threshold = 4.0 / len(sample)
    top_positions = np.argsort(cooks)[-5:][::-1]
    jb_stat, jb_p, skew, kurtosis = sm.stats.jarque_bera(result.resid)
    return {
        "model": "M4",
        "n": int(len(sample)),
        "breusch_pagan": {
            "lm_statistic": float(bp_lm),
            "lm_p_value": float(bp_lm_p),
            "f_statistic": float(bp_f),
            "f_p_value": float(bp_f_p),
        },
        "ramsey_reset": {
            "f_statistic": float(np.asarray(reset.fvalue).squeeze()),
            "p_value": float(np.asarray(reset.pvalue).squeeze()),
            "power": 2,
        },
        "jarque_bera": {
            "statistic": float(jb_stat),
            "p_value": float(jb_p),
            "skew": float(skew),
            "kurtosis": float(kurtosis),
        },
        "influence": {
            "cook_threshold_4_over_n": float(threshold),
            "count_above_threshold": int((cooks > threshold).sum()),
            "top_cities": [
                {
                    "region_key": str(sample.iloc[position]["region_key"]),
                    "cooks_distance": float(cooks[position]),
                }
                for position in top_positions
            ],
        },
    }


def _missingness_audit(df: pd.DataFrame) -> dict:
    """Compare observable attributes for cities included and excluded from core R2."""
    required = [R2_OUTCOME, "grid_emission_factor", "detour_ratio"]
    numeric_required = df[required].apply(pd.to_numeric, errors="coerce")
    included = numeric_required.notna().all(axis=1) & np.isfinite(numeric_required).all(axis=1)
    comparison_variables = [
        "population_10k", "pop_density", "gdp_per_capita", "store_density",
        "latitude", "longitude", "region_area_km2",
    ]
    comparisons = []
    for variable in comparison_variables:
        values = pd.to_numeric(df[variable], errors="coerce")
        retained = values[included].dropna()
        excluded = values[~included].dropna()
        pooled_sd = math.sqrt((retained.var() + excluded.var()) / 2.0)
        comparisons.append({
            "variable": variable,
            "included_n": int(len(retained)),
            "excluded_n": int(len(excluded)),
            "included_mean": float(retained.mean()),
            "excluded_mean": float(excluded.mean()),
            "standardized_mean_difference": (
                float((retained.mean() - excluded.mean()) / pooled_sd)
                if pooled_sd > 0 else None
            ),
        })
    return {
        "required_for_core": required,
        "included_city_count": int(included.sum()),
        "excluded_city_count": int((~included).sum()),
        "imputation": "none; grid emission factor is not imputed",
        "included_province_count": int(df.loc[included, "province"].nunique()),
        "excluded_province_count": int(df.loc[~included, "province"].nunique()),
        "observable_comparisons": comparisons,
    }


def _coefficient_lookup(records: list[dict], variable: str) -> dict:
    """Return a serialized coefficient record by variable name."""
    return next(record for record in records if record["variable"] == variable)


def _robustness_forest_summary(
    robust_models: list[dict],
    interaction: dict,
    bootstrap: dict,
    weighted: dict,
    leave_one_out: dict,
) -> dict:
    """Assemble comparable intervals for the R2 robustness forest plot."""
    additive = next(model for model in robust_models if model["name"] == "M2: + Detour")
    terms = []
    specs = (
        ("additive", "grid_emission_factor", "Grid factor"),
        ("additive", "detour_ratio", "Detour ratio"),
        ("interaction", "gef_x_detour", "GEF × detour"),
    )
    for model_name, variable, label in specs:
        source = additive if model_name == "additive" else interaction
        hc3 = _coefficient_lookup(source["hc3_coefficients"], variable)
        cluster = _coefficient_lookup(
            source["province_cluster_coefficients"], variable
        )
        weighted_record = _coefficient_lookup(
            weighted["models"][model_name]["coefficients"], variable
        )
        bootstrap_record = bootstrap[model_name][variable]
        loo_record = leave_one_out[model_name][variable]
        terms.append({
            "model": model_name,
            "variable": variable,
            "label": label,
            "methods": [
                {
                    "method": "HC3",
                    "estimate": hc3["beta_standardized"],
                    "interval": hc3["ci95"],
                    "interval_kind": "95% confidence interval",
                },
                {
                    "method": "Province cluster",
                    "estimate": cluster["beta_standardized"],
                    "interval": cluster["ci95"],
                    "interval_kind": "95% confidence interval",
                },
                {
                    "method": "City bootstrap",
                    "estimate": bootstrap_record["median"],
                    "interval": bootstrap_record["ci95_percentile"],
                    "interval_kind": "95% percentile interval",
                },
                {
                    "method": "Sales-weighted WLS",
                    "estimate": weighted_record["beta_standardized"],
                    "interval": weighted_record["ci95"],
                    "interval_kind": "HC3 95% confidence interval",
                },
                {
                    "method": "Leave one province out",
                    "estimate": hc3["beta_standardized"],
                    "interval": [loo_record["minimum"], loo_record["maximum"]],
                    "interval_kind": "range across omitted provinces",
                },
            ],
        })
    return {"terms": terms}


def _fit_landscape_model(df: pd.DataFrame) -> dict:
    variables = [R2_OUTCOME, "grid_emission_factor", "detour_ratio"]
    sample = df[variables].apply(pd.to_numeric, errors="coerce").dropna()
    sample = sample[np.isfinite(sample).all(axis=1)]
    model = sm.OLS(
        sample[R2_OUTCOME],
        sm.add_constant(sample[["grid_emission_factor", "detour_ratio"]]),
    ).fit()
    return {
        "n": int(len(sample)),
        "r2": float(model.rsquared),
        "adjusted_r2": float(model.rsquared_adj),
        "coefficients": {name: float(value) for name, value in model.params.items()},
        "p_values": {name: float(value) for name, value in model.pvalues.items()},
    }


def _fisher_group_difference(df: pd.DataFrame, threshold: float = 0.3) -> dict:
    required = ["detour_ratio", R2_OUTCOME, "grid_emission_factor"]
    valid = df[required].apply(pd.to_numeric, errors="coerce").dropna()
    valid = valid[np.isfinite(valid).all(axis=1)]
    groups = {
        "low_gef": valid[valid["grid_emission_factor"] <= threshold],
        "high_gef": valid[valid["grid_emission_factor"] > threshold],
    }
    records = {}
    for name, group in groups.items():
        r_value, p_value = stats.pearsonr(group["detour_ratio"], group[R2_OUTCOME])
        records[name] = {"n": int(len(group)), "r": float(r_value), "p": float(p_value)}
    low = records["low_gef"]
    high = records["high_gef"]
    se = math.sqrt(1.0 / (low["n"] - 3) + 1.0 / (high["n"] - 3))
    z_value = (np.arctanh(high["r"]) - np.arctanh(low["r"])) / se
    p_value = 2.0 * stats.norm.sf(abs(z_value))
    return {
        "threshold": threshold,
        "groups": records,
        "comparison": {
            "method": "independent Fisher r-to-z",
            "alternative": "two-sided",
            "z": float(z_value),
            "p": float(p_value),
            "high_is_stronger": bool(high["r"] > low["r"] and p_value < 0.05),
        },
    }


def _population_contrast(df: pd.DataFrame) -> dict:
    return {
        "absolute_reduction": _pairwise_correlation(
            df, "population_10k", R2_ABSOLUTE_OUTCOME, "pearson"
        ),
        "relative_reduction": _pairwise_correlation(
            df, "population_10k", R2_OUTCOME, "pearson"
        ),
    }


def run_r2_city_attribution(df: pd.DataFrame, supplements: list[dict]) -> dict:
    """Run the complete R2 attribution analysis and return JSON-safe results."""
    hierarchical_raw = run_hierarchical_regression(
        df,
        R2_OUTCOME,
        models=R2_HIERARCHICAL_MODELS,
    )
    hierarchical = _serialize_hierarchical(hierarchical_raw)
    robust_hierarchical = _robust_hierarchical_models(df)
    bootstrap_delta = _bootstrap_hierarchical_delta_r2(df)
    for model, robust, bootstrap in zip(
        hierarchical, robust_hierarchical, bootstrap_delta
    ):
        model["hc3_block_test"] = robust["hc3_block_test"]
        model["hc3_coefficients"] = robust["hc3_coefficients"]
        model["province_cluster_coefficients"] = robust[
            "province_cluster_coefficients"
        ]
        model["province_clusters"] = robust["province_clusters"]
        model.update(bootstrap)

    final_features = list(R2_HIERARCHICAL_MODELS.values())[-1]
    vif = compute_vif(df, final_features)
    interaction = _fit_interaction_model(df)
    bootstrap_core = _bootstrap_core_coefficients(df)
    weighted = _weighted_core_sensitivity(df)
    leave_one_out = _leave_one_province_out(df)
    return {
        "schema_version": 2,
        "analysis_id": "r2_city_attribution",
        "interpretation": (
            "Within-model mechanism attribution and cross-city association; "
            "not a causal identification design."
        ),
        "decision_rules": {
            "alpha": 0.05,
            "equivalence_bound_abs": R2_EQUIVALENCE_BOUND,
            "primary_covariance": "HC3",
            "sensitivity_covariance": "province-clustered with small-sample correction",
            "conflict_rule": (
                "Primary HC3 inference decides significance; qualify conclusions when "
                "direction or interval is unstable in sensitivity analyses."
            ),
        },
        "multiplicity": {
            "prespecified_core_and_model_blocks": "no multiplicity adjustment",
            "manuscript_attribute_screen": "Holm within each declared screen family",
            "full_correlation_matrices": "Benjamini-Hochberg FDR within each method",
        },
        "outcomes": {
            "relative": R2_OUTCOME,
            "absolute": R2_ABSOLUTE_OUTCOME,
        },
        "sample": {
            "city_rows": int(len(df)),
            "pairwise_complete_case": True,
            "hierarchical_common_sample": True,
        },
        "detour_supplements": supplements,
        "hierarchical_models": hierarchical,
        "final_model_vif": [
            {"variable": str(row.feature), "vif": float(row.VIF)}
            for row in vif.itertuples(index=False)
        ],
        "landscape_model": _fit_landscape_model(df),
        "interaction_model": interaction,
        "nonlinear_core_sensitivity": _nonlinear_core_sensitivity(df),
        "screen_correlations": _screen_correlations(df),
        "correlation_matrices": {
            "pearson": _correlation_matrix(df, "pearson"),
            "spearman": _correlation_matrix(df, "spearman"),
        },
        "gef_detour_orthogonality": _pairwise_correlation(
            df, "grid_emission_factor", "detour_ratio", "pearson"
        ),
        "gef_group_detour_difference": _fisher_group_difference(df),
        "population_contrast": _population_contrast(df),
        "socioeconomic_alternative_models": _socioeconomic_alternative_models(df),
        "geographic_sensitivity": _geographic_sensitivity(df),
        "model_diagnostics": _model_diagnostics(df),
        "bootstrap_core_models": bootstrap_core,
        "weighted_sensitivity": weighted,
        "leave_one_province_out": leave_one_out,
        "missingness_audit": _missingness_audit(df),
        "robustness_forest": _robustness_forest_summary(
            robust_hierarchical,
            interaction,
            bootstrap_core,
            weighted,
            leave_one_out,
        ),
    }


def build_r2_city_attribution(
    enriched_csv_path: str | Path,
    data_dir: str | Path,
    region_keys: set[str] | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Prepare data and run all R2 analyses used by paper subfigures."""
    df, supplements = prepare_r2_city_data(
        enriched_csv_path,
        data_dir,
        region_keys=region_keys,
    )
    analysis = run_r2_city_attribution(df, supplements)
    source = Path(enriched_csv_path)
    analysis["source"] = {
        "enriched_csv_path": str(source),
        "enriched_csv_size_bytes": int(source.stat().st_size),
        "enriched_csv_mtime_ns": int(source.stat().st_mtime_ns),
        "data_dir": str(data_dir),
        "persisted_detour_repairs": (
            df.loc[
                df.get("detour_stats_source", pd.Series(index=df.index, dtype=object)).notna(),
                [
                    column for column in (
                        "region_key",
                        "detour_ratio",
                        "detour_stats_source",
                        "detour_stats_source_path",
                        "detour_stats_source_mtime_ns",
                        "detour_stats_repaired_at",
                    )
                    if column in df.columns
                ],
            ].to_dict(orient="records")
        ),
    }
    return df, analysis


def write_r2_city_attribution_report(analysis: dict, report_path: str | Path) -> str:
    """Write the shared R2 analysis report."""
    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = report_path.with_suffix(report_path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as file:
        json.dump(
            _json_compatible(analysis),
            file,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
        file.write("\n")
    os.replace(tmp_path, report_path)
    return str(report_path)


def _json_compatible(value):
    """Replace undefined numerical diagnostics with explicit JSON nulls."""
    if isinstance(value, dict):
        return {str(key): _json_compatible(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_compatible(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value
