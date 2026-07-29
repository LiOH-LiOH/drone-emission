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

from analysis.regression import compute_vif, run_hierarchical_regression


R2_OUTCOME = "oracle_min_reduction_vs_ebike_pct"
R2_ABSOLUTE_OUTCOME = "oracle_min_reduction_vs_ebike_t"

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
) -> tuple[pd.DataFrame, list[dict]]:
    """Load city data, derive Oracle outcomes, and supplement detour statistics."""
    enriched_csv_path = Path(enriched_csv_path)
    data_dir = Path(data_dir)
    df = pd.read_csv(enriched_csv_path)
    if "region_level" in df.columns:
        df = df[df["region_level"] == "city"].copy()
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
    return {
        "x": x_var,
        "y": y_var,
        "method": method,
        "n": int(len(pair)),
        "coefficient": float(coefficient),
        "p_value": float(p_value),
    }


def _bonferroni(records: list[dict]) -> list[dict]:
    valid_count = sum(record.get("p_value") is not None for record in records)
    adjusted = []
    for record in records:
        item = dict(record)
        p_value = item.get("p_value")
        item["p_bonferroni"] = (
            min(float(p_value) * valid_count, 1.0) if p_value is not None else None
        )
        item["significant_bonferroni_0_05"] = (
            item["p_bonferroni"] < 0.05
            if item["p_bonferroni"] is not None
            else False
        )
        item["family_size"] = valid_count
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
        "outcome": _bonferroni(outcome),
        "grid_emission_factor": _bonferroni(gef),
    }


def _correlation_matrix(df: pd.DataFrame, method: str) -> dict:
    variables = [var for var in R2_MATRIX_VARIABLES if var in df.columns]
    pair_records = []
    pair_lookup = {}
    for i, first in enumerate(variables):
        for second in variables[i + 1:]:
            record = _pairwise_correlation(df, first, second, method)
            pair_records.append(record)
    pair_records = _bonferroni(pair_records)
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
            p_row.append(record["p_bonferroni"])
            n_row.append(record["n"])
        coefficients.append(coefficient_row)
        adjusted_p.append(p_row)
        sample_sizes.append(n_row)
    return {
        "method": method,
        "variables": variables,
        "coefficients": coefficients,
        "p_bonferroni": adjusted_p,
        "sample_sizes": sample_sizes,
        "family_size": pair_records[0]["family_size"] if pair_records else 0,
        "pairs": pair_records,
    }


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
    final_features = list(R2_HIERARCHICAL_MODELS.values())[-1]
    vif = compute_vif(df, final_features)
    return {
        "schema_version": 1,
        "analysis_id": "r2_city_attribution",
        "interpretation": (
            "Within-model mechanism attribution and cross-city association; "
            "not a causal identification design."
        ),
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
        "hierarchical_models": _serialize_hierarchical(hierarchical_raw),
        "final_model_vif": [
            {"variable": str(row.feature), "vif": float(row.VIF)}
            for row in vif.itertuples(index=False)
        ],
        "landscape_model": _fit_landscape_model(df),
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
    }


def build_r2_city_attribution(
    enriched_csv_path: str | Path,
    data_dir: str | Path,
) -> tuple[pd.DataFrame, dict]:
    """Prepare data and run all R2 analyses used by paper subfigures."""
    df, supplements = prepare_r2_city_data(enriched_csv_path, data_dir)
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
        json.dump(analysis, file, ensure_ascii=False, indent=2, allow_nan=False)
        file.write("\n")
    os.replace(tmp_path, report_path)
    return str(report_path)
