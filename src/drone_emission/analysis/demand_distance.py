"""Analyze city features associated with modeled average demand distance.

Usage:
    cd src/drone_emission
    python -m analysis.demand_distance \
        --csv ../../data/all_region_stats_enriched.csv \
        --output-dir ../../figures/analysis/demand_distance \
        --for-latex
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from statsmodels.stats.multitest import multipletests

from drone_emission.analysis.correlation import load_region_stats


BOOTSTRAP_REPLICATES = 5000
RANDOM_SEED = 20260806

OUTCOMES = {
    "avg_straight_distance_m": "Straight-line demand distance",
    "avg_road_distance_m": "Road demand distance",
}

FEATURE_LABELS = {
    "population_10k": "Population",
    "gdp_100m": "GDP",
    "gdp_per_capita": "GDP per capita",
    "pop_density": "Population density",
    "region_area_km2": "Region area",
    "builtup_area_2022_km2": "Built-up area",
    "urban_density_2022_per_km2": "Urban population density",
    "store_density": "Store density",
    "building_density": "Building density",
    "building_volume_density": "Building volume density",
    "road_density": "Road density",
    "intersection_density": "Intersection density",
    "avg_block_length": "Mean block length",
    "road_connectivity": "Road connectivity",
    "avg_street_length": "Mean street length",
    "dead_end_proportion": "Dead-end proportion",
    "four_way_proportion": "Four-way intersection proportion",
    "orientation_entropy": "Orientation entropy",
}

SOCIOECONOMIC_FEATURES = (
    "population_10k",
    "gdp_100m",
    "gdp_per_capita",
    "pop_density",
)

FORM_FEATURES = (
    "region_area_km2",
    "builtup_area_2022_km2",
    "urban_density_2022_per_km2",
    "store_density",
    "building_density",
    "building_volume_density",
    "road_density",
    "intersection_density",
    "avg_block_length",
    "road_connectivity",
    "avg_street_length",
    "dead_end_proportion",
    "four_way_proportion",
    "orientation_entropy",
)

SCREEN_FEATURES = (*SOCIOECONOMIC_FEATURES, *FORM_FEATURES)

HIERARCHICAL_MODELS = {
    "M1: Socioeconomic": (
        "population_10k",
        "gdp_per_capita",
    ),
    "M2: + Urban form": (
        "population_10k",
        "gdp_per_capita",
        "region_area_km2",
        "store_density",
        "building_density",
    ),
    "M3: + Road form": (
        "population_10k",
        "gdp_per_capita",
        "region_area_km2",
        "store_density",
        "building_density",
        "road_density",
        "avg_street_length",
        "dead_end_proportion",
        "four_way_proportion",
        "orientation_entropy",
    ),
}

LOG_VARIABLES = {
    *OUTCOMES,
    "population_10k",
    "gdp_per_capita",
    "region_area_km2",
    "store_density",
    "building_density",
    "road_density",
    "avg_street_length",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_demand_distance_data(csv_path: str | Path) -> pd.DataFrame:
    """Load completed city rows and derive reusable urban-form features."""
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Region statistics CSV not found: {csv_path}")

    df = load_region_stats(str(csv_path)).reset_index()
    if "region_level" not in df.columns:
        raise KeyError("Region statistics must contain region_level")
    df = df[df["region_level"].eq("city")].copy()

    required = ["region_key", *OUTCOMES]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise KeyError(f"Missing required demand-distance columns: {missing}")

    numeric = [
        column
        for column in {*OUTCOMES, *SCREEN_FEATURES}
        if column in df.columns
    ]
    df[numeric] = df[numeric].apply(pd.to_numeric, errors="coerce")
    return df.reset_index(drop=True)


def _bootstrap_spearman_ci(
    x: np.ndarray,
    y: np.ndarray,
    rng: np.random.Generator,
) -> tuple[float, float]:
    estimates = np.empty(BOOTSTRAP_REPLICATES, dtype=float)
    n = len(x)
    for index in range(BOOTSTRAP_REPLICATES):
        sample = rng.integers(0, n, size=n)
        estimate = stats.spearmanr(x[sample], y[sample]).statistic
        estimates[index] = float(estimate)
    finite = estimates[np.isfinite(estimates)]
    if len(finite) < BOOTSTRAP_REPLICATES * 0.95:
        return np.nan, np.nan
    lower, upper = np.quantile(finite, [0.025, 0.975])
    return float(lower), float(upper)


def compute_distance_correlations(df: pd.DataFrame) -> pd.DataFrame:
    """Compute pairwise Spearman screens with bootstrap intervals and FDR."""
    rng = np.random.default_rng(RANDOM_SEED)
    records: list[dict] = []
    for outcome, outcome_label in OUTCOMES.items():
        family: list[dict] = []
        for feature in SCREEN_FEATURES:
            if feature not in df.columns:
                continue
            pair = df[[feature, outcome]].dropna()
            pair = pair[np.isfinite(pair[feature]) & np.isfinite(pair[outcome])]
            record = {
                "outcome": outcome,
                "outcome_label": outcome_label,
                "feature": feature,
                "feature_label": FEATURE_LABELS[feature],
                "feature_group": (
                    "Socioeconomic"
                    if feature in SOCIOECONOMIC_FEATURES
                    else "Urban form"
                ),
                "n": int(len(pair)),
                "spearman_rho": np.nan,
                "ci95_lower": np.nan,
                "ci95_upper": np.nan,
                "p_value": np.nan,
            }
            if (
                len(pair) >= 3
                and pair[feature].nunique() >= 2
                and pair[outcome].nunique() >= 2
            ):
                x = pair[feature].to_numpy(dtype=float)
                y = pair[outcome].to_numpy(dtype=float)
                result = stats.spearmanr(x, y)
                lower, upper = _bootstrap_spearman_ci(x, y, rng)
                record.update({
                    "spearman_rho": float(result.statistic),
                    "ci95_lower": lower,
                    "ci95_upper": upper,
                    "p_value": float(result.pvalue),
                })
            family.append(record)

        valid = [
            index
            for index, row in enumerate(family)
            if np.isfinite(row["p_value"])
        ]
        adjusted = multipletests(
            [family[index]["p_value"] for index in valid],
            method="fdr_bh",
        )[1] if valid else []
        adjusted_lookup = dict(zip(valid, adjusted))
        for index, record in enumerate(family):
            q_value = (
                float(adjusted_lookup[index])
                if index in adjusted_lookup
                else np.nan
            )
            record["q_value"] = q_value
            record["significant_fdr_0_05"] = bool(
                np.isfinite(q_value) and q_value < 0.05
            )
            record["fdr_family_size"] = len(valid)
        records.extend(family)

    return pd.DataFrame.from_records(records)


def _transform_regression_sample(
    df: pd.DataFrame,
    outcome: str,
) -> tuple[pd.DataFrame, dict[str, str]]:
    final_features = list(HIERARCHICAL_MODELS.values())[-1]
    variables = [outcome, *final_features]
    missing = [variable for variable in variables if variable not in df.columns]
    if missing:
        raise KeyError(f"Missing hierarchical-regression columns: {missing}")

    sample = df[["region_key", *variables]].copy()
    sample[variables] = sample[variables].apply(pd.to_numeric, errors="coerce")
    transformations: dict[str, str] = {}
    for variable in variables:
        if variable in LOG_VARIABLES:
            sample.loc[sample[variable] <= 0, variable] = np.nan
            sample[variable] = np.log(sample[variable])
            transformations[variable] = "natural_log_then_z_score"
        else:
            transformations[variable] = "z_score"
    sample = sample.dropna(subset=variables)
    sample = sample[np.isfinite(sample[variables]).all(axis=1)].copy()
    if len(sample) < len(final_features) + 2:
        raise ValueError(
            f"Only {len(sample)} complete cities for {len(final_features)} predictors"
        )
    standard_deviations = sample[variables].std()
    constant = standard_deviations[standard_deviations <= 0].index.tolist()
    if constant:
        raise ValueError(f"Constant transformed regression variables: {constant}")
    sample[variables] = (
        sample[variables] - sample[variables].mean()
    ) / standard_deviations
    return sample.reset_index(drop=True), transformations


def _wald_block_test(result, variables: list[str]) -> dict:
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
        "df_num": len(tested),
        "method": "HC3 robust Wald test",
    }


def compute_hierarchical_regressions(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:
    """Fit common-sample hierarchical regressions for both distance outcomes."""
    coefficient_records: list[dict] = []
    report: dict = {}
    for outcome in OUTCOMES:
        sample, transformations = _transform_regression_sample(df, outcome)
        y = sample[outcome]
        previous_features: list[str] = []
        previous_r2 = 0.0
        model_records = []
        for model_name, feature_tuple in HIERARCHICAL_MODELS.items():
            features = list(feature_tuple)
            design = sm.add_constant(sample[features], has_constant="add")
            result = sm.OLS(y, design).fit(cov_type="HC3", use_t=True)
            added = [
                feature
                for feature in features
                if feature not in previous_features
            ]
            interval = pd.DataFrame(
                np.asarray(result.conf_int(alpha=0.05), dtype=float),
                index=result.model.exog_names,
                columns=["lower", "upper"],
            )
            coefficients = []
            for feature in features:
                coefficient = {
                    "outcome": outcome,
                    "model": model_name,
                    "feature": feature,
                    "feature_label": FEATURE_LABELS[feature],
                    "n": int(len(sample)),
                    "beta_standardized": float(result.params[feature]),
                    "standard_error_hc3": float(result.bse[feature]),
                    "ci95_lower": float(interval.loc[feature, "lower"]),
                    "ci95_upper": float(interval.loc[feature, "upper"]),
                    "test_statistic": float(result.tvalues[feature]),
                    "p_value": float(result.pvalues[feature]),
                    "r2": float(result.rsquared),
                    "adjusted_r2": float(result.rsquared_adj),
                    "delta_r2": float(result.rsquared - previous_r2),
                }
                coefficients.append(coefficient)
                coefficient_records.append(coefficient)
            model_records.append({
                "name": model_name,
                "features": features,
                "added_block": added,
                "n": int(len(sample)),
                "r2": float(result.rsquared),
                "adjusted_r2": float(result.rsquared_adj),
                "delta_r2": float(result.rsquared - previous_r2),
                "added_block_test": _wald_block_test(result, added),
                "coefficients": coefficients,
            })
            previous_features = features
            previous_r2 = float(result.rsquared)
        report[outcome] = {
            "common_sample_n": int(len(sample)),
            "common_sample_region_keys": sample["region_key"].astype(str).tolist(),
            "transformations": transformations,
            "covariance": "HC3",
            "models": model_records,
        }
    return pd.DataFrame.from_records(coefficient_records), report


def select_top_form_feature(correlations: pd.DataFrame) -> str:
    """Select the strongest eligible urban-form correlate of air distance."""
    eligible = correlations[
        correlations["outcome"].eq("avg_straight_distance_m")
        & correlations["feature"].isin(FORM_FEATURES)
        & correlations["spearman_rho"].notna()
        & correlations["n"].ge(30)
    ].copy()
    if eligible.empty:
        raise ValueError("No urban-form feature has at least 30 valid cities")
    eligible["abs_rho"] = eligible["spearman_rho"].abs()
    eligible = eligible.sort_values(
        ["abs_rho", "feature"], ascending=[False, True], kind="mergesort"
    )
    return str(eligible.iloc[0]["feature"])


def _json_records(df: pd.DataFrame) -> list[dict]:
    return json.loads(df.to_json(orient="records"))


def run_demand_distance_analysis(
    csv_path: str | Path,
    output_dir: str | Path,
    for_latex: bool = False,
    plot_title: bool = False,
) -> dict:
    """Run the complete demand-distance feature analysis and write outputs."""
    csv_path = Path(csv_path).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    df = prepare_demand_distance_data(csv_path)
    correlations = compute_distance_correlations(df)
    regressions, regression_report = compute_hierarchical_regressions(df)
    top_form_feature = select_top_form_feature(correlations)

    correlation_path = output_dir / "demand_distance_correlations.csv"
    regression_path = output_dir / "demand_distance_regression.csv"
    correlations.to_csv(correlation_path, index=False)
    regressions.to_csv(regression_path, index=False)

    from drone_emission.visualization.analysis import (
        plot_demand_distance_feature_correlations,
        plot_demand_distance_key_scatters,
    )

    correlation_figure = output_dir / "demand_distance_feature_correlations.png"
    scatter_figure = output_dir / "demand_distance_key_scatters.png"
    plot_demand_distance_feature_correlations(
        correlations,
        save_path=str(correlation_figure),
        for_latex=for_latex,
        plot_title=plot_title,
    )
    plot_demand_distance_key_scatters(
        df,
        correlations,
        top_form_feature,
        save_path=str(scatter_figure),
        for_latex=for_latex,
        plot_title=plot_title,
    )

    report = {
        "schema_version": 1,
        "analysis": "city_feature_demand_distance_association",
        "interpretation_scope": (
            "Sales-weighted modeled OD distance within the common 5 km delivery "
            "radius and Tanner impedance specification; not observed city "
            "delivery distance."
        ),
        "source": {
            "csv_path": str(csv_path),
            "sha256": _sha256(csv_path),
            "size_bytes": csv_path.stat().st_size,
        },
        "sample": {
            "city_rows": int(len(df)),
            "region_keys": df["region_key"].astype(str).tolist(),
            "missing_by_variable": {
                column: int(df[column].isna().sum())
                for column in [*OUTCOMES, *SCREEN_FEATURES]
                if column in df.columns
            },
        },
        "correlation_analysis": {
            "method": "Spearman rank correlation",
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
            "bootstrap_seed": RANDOM_SEED,
            "confidence_interval": "city bootstrap percentile 95% interval",
            "multiple_testing": "Benjamini-Hochberg FDR within each outcome",
            "records": _json_records(correlations),
        },
        "hierarchical_regression": regression_report,
        "regression_feature_policy": {
            "screen_only_road_features": [
                "intersection_density",
                "avg_block_length",
                "road_connectivity",
            ],
            "reason": (
                "Excluded from M3 because their log transforms are algebraically "
                "dependent on road density and one another; retained in the "
                "pairwise correlation screen."
            ),
        },
        "top_form_feature": top_form_feature,
        "outputs": {
            "correlations_csv": str(correlation_path),
            "regression_csv": str(regression_path),
            "correlation_figure": str(
                correlation_figure.with_suffix(".pdf")
                if for_latex else correlation_figure
            ),
            "scatter_figure": str(
                scatter_figure.with_suffix(".pdf")
                if for_latex else scatter_figure
            ),
        },
    }
    report_path = output_dir / "demand_distance_analysis_report.json"
    with report_path.open("w", encoding="utf-8") as file_obj:
        json.dump(report, file_obj, ensure_ascii=False, indent=2)

    primary = correlations[
        correlations["outcome"].eq("avg_straight_distance_m")
    ].sort_values("spearman_rho", key=lambda values: values.abs(), ascending=False)
    print(f"Demand-distance city sample: {len(df)}")
    print(f"Strongest urban-form correlate: {top_form_feature}")
    print(primary[["feature", "n", "spearman_rho", "q_value"]].to_string(index=False))
    print(f"Demand-distance analysis report saved to {report_path}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze city features associated with average demand distance"
    )
    parser.add_argument("--csv", required=True, help="Enriched region statistics CSV")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument(
        "--for-latex", action="store_true", help="Save figures as PDF"
    )
    parser.add_argument(
        "--plot-title", action="store_true", help="Show optional figure titles"
    )
    args = parser.parse_args()
    run_demand_distance_analysis(
        csv_path=args.csv,
        output_dir=args.output_dir,
        for_latex=args.for_latex,
        plot_title=args.plot_title,
    )


if __name__ == "__main__":
    main()
