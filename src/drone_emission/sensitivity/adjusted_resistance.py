"""Baseline-adjusted resistance to fixed-threshold parameter perturbations.

The primary quantity is the absolute lower-tail loss ``D_q = R0 - Q_q(R)``.
Conditional quantile splines remove the typical loss associated with a city's
deterministic baseline without dividing by that baseline.  Positive adjusted
scores indicate less downside than cities with a comparable baseline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from patsy import build_design_matrices, dmatrix
from scipy.stats import spearmanr
from statsmodels.regression.quantile_regression import QuantReg

from .threshold_resilience import THRESHOLD_RESILIENCE_SCENARIOS


ADJUSTED_RESISTANCE_VERSION = "1.0.0"
ADJUSTED_RESISTANCE_SPLINE_DF = 4
ADJUSTED_RESISTANCE_BOOTSTRAP_REPLICATES = 1000
ADJUSTED_RESISTANCE_BOOTSTRAP_SEED = 20260806
ADJUSTED_RESISTANCE_STABILITY_THRESHOLD = 0.80
ADJUSTED_RESISTANCE_TAIL_LEVELS = (0.05, 0.10)
ADJUSTED_RESISTANCE_CLASSES = ("high", "typical", "low")


@dataclass(frozen=True)
class _QuantileSpline:
    """Fitted conditional quartiles sharing one natural-spline design."""

    design_info: object
    models: dict[float, object]

    def predict(self, baseline: Iterable[float]) -> dict[float, np.ndarray]:
        values = np.asarray(tuple(baseline), dtype=np.float64)
        matrix = np.asarray(
            build_design_matrices([self.design_info], {"x": values})[0],
            dtype=np.float64,
        )
        return {
            quantile: np.asarray(model.predict(matrix), dtype=np.float64)
            for quantile, model in self.models.items()
        }


def _tail_suffix(tail: float) -> str:
    return f"p{int(round(tail * 100)):02d}"


def _classify(
    downside: np.ndarray,
    q25: np.ndarray,
    q75: np.ndarray,
) -> np.ndarray:
    return np.where(downside < q25, "high", np.where(downside > q75, "low", "typical"))


def _fit_quantile_spline(
    baseline: np.ndarray,
    downside: np.ndarray,
    *,
    spline_df: int,
    lower_bound: float,
    upper_bound: float,
) -> _QuantileSpline:
    if spline_df < 3:
        raise ValueError("spline_df must be at least 3")
    formula = (
        f"cr(x, df={int(spline_df)}, lower_bound={lower_bound!r}, "
        f"upper_bound={upper_bound!r})"
    )
    design = dmatrix(formula, {"x": baseline}, return_type="dataframe")
    matrix = np.asarray(design, dtype=np.float64)
    models = {}
    for quantile in (0.25, 0.50, 0.75):
        models[quantile] = QuantReg(downside, matrix).fit(
            q=quantile,
            max_iter=5000,
            p_tol=1e-8,
        )
    return _QuantileSpline(design.design_info, models)


def _validate_predictions(predicted: dict[float, np.ndarray], context: str) -> None:
    q25, q50, q75 = (predicted[value] for value in (0.25, 0.50, 0.75))
    if not all(np.isfinite(values).all() for values in (q25, q50, q75)):
        raise ValueError(f"{context} produced non-finite conditional quantiles")
    tolerance = 1e-8
    if ((q25 - q50) > tolerance).any() or ((q50 - q75) > tolerance).any():
        raise ValueError(f"{context} produced crossing conditional quantiles")
    if ((q75 - q25) <= tolerance).any():
        raise ValueError(f"{context} produced a non-positive conditional IQR")


def summarize_city_downside(
    reduction_pct: Iterable[float],
    baseline_reduction_pct: float,
) -> dict[str, float]:
    """Return absolute P05/P10 downside losses for one city scenario."""
    reduction = np.asarray(tuple(reduction_pct), dtype=np.float64)
    if reduction.ndim != 1 or reduction.size == 0:
        raise ValueError("reduction_pct must be a non-empty one-dimensional array")
    if not np.isfinite(reduction).all():
        raise ValueError("reduction_pct contains non-finite values")
    baseline = float(baseline_reduction_pct)
    if not np.isfinite(baseline):
        raise ValueError("baseline reduction must be finite")
    result = {}
    for tail in ADJUSTED_RESISTANCE_TAIL_LEVELS:
        suffix = _tail_suffix(tail)
        quantile = float(np.quantile(reduction, tail))
        result[f"reduction_{suffix}_pct"] = quantile
        result[f"downside_{suffix}_pp"] = baseline - quantile
    return result


def validate_downside_table(
    downside: pd.DataFrame,
    expected_cities: int | None = None,
) -> pd.DataFrame:
    """Validate the two-scenario city downside table."""
    required = {
        "region_key", "region_name_cn", "region_name_en", "province",
        "scenario", "n_samples", "baseline_fixed_threshold_reduction_pct",
        "reduction_p05_pct", "downside_p05_pp", "reduction_p10_pct",
        "downside_p10_pp",
    }
    missing = sorted(required - set(downside.columns))
    if missing:
        raise KeyError(f"adjusted-resistance downside table missing columns: {missing}")
    frame = downside[downside["scenario"].isin(THRESHOLD_RESILIENCE_SCENARIOS)].copy()
    if frame.duplicated(["region_key", "scenario"]).any():
        raise ValueError("downside table contains duplicate city/scenario rows")
    counts = frame.groupby("region_key")["scenario"].nunique()
    incomplete = sorted(counts[counts != len(THRESHOLD_RESILIENCE_SCENARIOS)].index)
    if incomplete:
        raise ValueError(f"cities missing one downside scenario: {incomplete}")
    if expected_cities is not None and frame["region_key"].nunique() != expected_cities:
        raise ValueError(
            f"expected {expected_cities} cities, found {frame['region_key'].nunique()}"
        )
    numeric = (
        "n_samples", "baseline_fixed_threshold_reduction_pct",
        "reduction_p05_pct", "downside_p05_pp", "reduction_p10_pct",
        "downside_p10_pp",
    )
    for column in numeric:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if not np.isfinite(frame[list(numeric)].to_numpy(dtype=float)).all():
        raise ValueError("downside table contains non-finite numeric values")
    if (frame["n_samples"] <= 0).any():
        raise ValueError("n_samples must be positive")
    baseline_spread = frame.groupby("region_key")[
        "baseline_fixed_threshold_reduction_pct"
    ].agg(lambda values: float(values.max() - values.min()))
    if (baseline_spread > 1e-12).any():
        raise ValueError("city baselines differ between scenarios")
    for suffix in ("p05", "p10"):
        expected = (
            frame["baseline_fixed_threshold_reduction_pct"]
            - frame[f"reduction_{suffix}_pct"]
        )
        if not np.allclose(expected, frame[f"downside_{suffix}_pp"], atol=1e-10):
            raise ValueError(f"downside_{suffix}_pp is inconsistent with source values")
    return frame


def derive_adjusted_resistance(
    downside: pd.DataFrame,
    *,
    expected_cities: int | None = None,
    spline_df: int = ADJUSTED_RESISTANCE_SPLINE_DF,
    bootstrap_replicates: int = ADJUSTED_RESISTANCE_BOOTSTRAP_REPLICATES,
    bootstrap_seed: int = ADJUSTED_RESISTANCE_BOOTSTRAP_SEED,
    stability_threshold: float = ADJUSTED_RESISTANCE_STABILITY_THRESHOLD,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Fit conditional resistance scores and return four auditable outputs."""
    frame = validate_downside_table(downside, expected_cities=expected_cities)
    if bootstrap_replicates < 0:
        raise ValueError("bootstrap_replicates cannot be negative")
    if not 0.0 < stability_threshold <= 1.0:
        raise ValueError("stability_threshold must lie within (0, 1]")
    frame["resistance_eligible"] = (
        frame["baseline_fixed_threshold_reduction_pct"] > 1e-9
    )
    frame["resistance_exclusion_reason"] = np.where(
        frame["resistance_eligible"], "", "structural_zero_or_nonpositive_baseline"
    )
    eligible = frame[frame["resistance_eligible"]].copy()
    city_keys = np.asarray(sorted(eligible["region_key"].unique()), dtype=object)
    lower_bound = float(eligible["baseline_fixed_threshold_reduction_pct"].min())
    upper_bound = float(eligible["baseline_fixed_threshold_reduction_pct"].max())
    if lower_bound >= upper_bound:
        raise ValueError("eligible cities must span more than one baseline value")

    diagnostics = []
    fitted: dict[tuple[str, str], _QuantileSpline] = {}
    for scenario in THRESHOLD_RESILIENCE_SCENARIOS:
        mask = frame["resistance_eligible"] & (frame["scenario"] == scenario)
        subset = frame.loc[mask]
        baseline = subset["baseline_fixed_threshold_reduction_pct"].to_numpy(float)
        for tail in ADJUSTED_RESISTANCE_TAIL_LEVELS:
            suffix = _tail_suffix(tail)
            downside_values = subset[f"downside_{suffix}_pp"].to_numpy(float)
            model = _fit_quantile_spline(
                baseline, downside_values, spline_df=spline_df,
                lower_bound=lower_bound, upper_bound=upper_bound,
            )
            predicted = model.predict(baseline)
            _validate_predictions(predicted, f"{scenario}/{suffix}")
            fitted[(scenario, suffix)] = model
            iqr = predicted[0.75] - predicted[0.25]
            score = (predicted[0.50] - downside_values) / iqr
            classes = _classify(downside_values, predicted[0.25], predicted[0.75])
            for quantile in (0.25, 0.50, 0.75):
                frame.loc[mask, f"conditional_{suffix}_q{int(quantile * 100):02d}_pp"] = (
                    predicted[quantile]
                )
            frame.loc[mask, f"adjusted_resistance_score_{suffix}"] = score
            frame.loc[mask, f"resistance_class_{suffix}"] = classes
            rho = float(spearmanr(baseline, score).statistic)
            diagnostics.append({
                "scenario": scenario,
                "tail_level": tail,
                "spline_df": spline_df,
                "n_cities": len(subset),
                "baseline_score_spearman_rho": rho,
                "below_q25_share": float(np.mean(downside_values < predicted[0.25])),
                "within_iqr_share": float(np.mean(
                    (downside_values >= predicted[0.25])
                    & (downside_values <= predicted[0.75])
                )),
                "above_q75_share": float(np.mean(downside_values > predicted[0.75])),
            })

    frame["resistance_class"] = frame["resistance_class_p05"]
    frame["tail_level_uncertain"] = (
        frame["resistance_eligible"]
        & (frame["resistance_class_p05"] != frame["resistance_class_p10"])
    )
    frame["adjusted_score_ci95_low"] = np.nan
    frame["adjusted_score_ci95_high"] = np.nan
    frame["class_stability"] = np.nan

    bootstrap_scores = {
        scenario: [] for scenario in THRESHOLD_RESILIENCE_SCENARIOS
    }
    bootstrap_classes = {
        scenario: [] for scenario in THRESHOLD_RESILIENCE_SCENARIOS
    }
    rng = np.random.default_rng(bootstrap_seed)
    successful = 0
    rearranged = 0
    for _ in range(bootstrap_replicates):
        sampled_keys = rng.choice(city_keys, size=len(city_keys), replace=True)
        replicate_ok = True
        replicate_scores = {}
        replicate_classes = {}
        replicate_rearranged = False
        for scenario in THRESHOLD_RESILIENCE_SCENARIOS:
            source = eligible[eligible["scenario"] == scenario].set_index("region_key")
            sampled = source.loc[sampled_keys]
            target = source.loc[city_keys]
            try:
                model = _fit_quantile_spline(
                    sampled["baseline_fixed_threshold_reduction_pct"].to_numpy(float),
                    sampled["downside_p05_pp"].to_numpy(float),
                    spline_df=spline_df,
                    lower_bound=lower_bound,
                    upper_bound=upper_bound,
                )
                predicted = model.predict(
                    target["baseline_fixed_threshold_reduction_pct"].to_numpy(float)
                )
                try:
                    _validate_predictions(predicted, f"bootstrap/{scenario}")
                except ValueError as error:
                    if "crossing conditional quantiles" not in str(error):
                        raise
                    ordered = np.sort(
                        np.vstack([predicted[0.25], predicted[0.50], predicted[0.75]]),
                        axis=0,
                    )
                    predicted = {
                        0.25: ordered[0], 0.50: ordered[1], 0.75: ordered[2]
                    }
                    _validate_predictions(
                        predicted, f"rearranged bootstrap/{scenario}"
                    )
                    replicate_rearranged = True
            except (ValueError, np.linalg.LinAlgError):
                replicate_ok = False
                break
            observed = target["downside_p05_pp"].to_numpy(float)
            replicate_scores[scenario] = (
                predicted[0.50] - observed
            ) / (predicted[0.75] - predicted[0.25])
            replicate_classes[scenario] = _classify(
                observed, predicted[0.25], predicted[0.75]
            )
        if not replicate_ok:
            continue
        successful += 1
        rearranged += int(replicate_rearranged)
        for scenario in THRESHOLD_RESILIENCE_SCENARIOS:
            bootstrap_scores[scenario].append(replicate_scores[scenario])
            bootstrap_classes[scenario].append(replicate_classes[scenario])

    minimum_successful = min(bootstrap_replicates, max(1, int(0.95 * bootstrap_replicates)))
    if bootstrap_replicates and successful < minimum_successful:
        raise RuntimeError(
            f"only {successful}/{bootstrap_replicates} valid bootstrap fits; "
            f"required at least {minimum_successful}"
        )
    for scenario in THRESHOLD_RESILIENCE_SCENARIOS:
        mask = frame["resistance_eligible"] & (frame["scenario"] == scenario)
        target = frame.loc[mask].set_index("region_key").loc[city_keys]
        if bootstrap_replicates:
            scores = np.asarray(bootstrap_scores[scenario], dtype=np.float64)
            classes = np.asarray(bootstrap_classes[scenario], dtype=object)
            point = target["resistance_class"].to_numpy(dtype=object)
            stability = np.mean(classes == point[None, :], axis=0)
            values = pd.DataFrame({
                "region_key": city_keys,
                "adjusted_score_ci95_low": np.quantile(scores, 0.025, axis=0),
                "adjusted_score_ci95_high": np.quantile(scores, 0.975, axis=0),
                "class_stability": stability,
            }).set_index("region_key")
            indices = frame.index[mask]
            keys = frame.loc[indices, "region_key"]
            frame.loc[indices, "adjusted_score_ci95_low"] = keys.map(
                values["adjusted_score_ci95_low"]
            ).to_numpy()
            frame.loc[indices, "adjusted_score_ci95_high"] = keys.map(
                values["adjusted_score_ci95_high"]
            ).to_numpy()
            frame.loc[indices, "class_stability"] = keys.map(
                values["class_stability"]
            ).to_numpy()
    frame["classification_uncertain"] = (
        frame["resistance_eligible"]
        & (frame["class_stability"] < stability_threshold)
    )
    frame["bootstrap_replicates_requested"] = bootstrap_replicates
    frame["bootstrap_replicates_successful"] = successful
    frame["adjusted_resistance_version"] = ADJUSTED_RESISTANCE_VERSION

    full_index = pd.MultiIndex.from_product(
        [THRESHOLD_RESILIENCE_SCENARIOS, ADJUSTED_RESISTANCE_CLASSES],
        names=["scenario", "resistance_class"],
    )
    grouped = (
        frame[frame["resistance_eligible"]]
        .groupby(["scenario", "resistance_class"], observed=False)
        .size()
        .reindex(full_index, fill_value=0)
    )
    class_summary = grouped.rename("city_count").reset_index()
    class_summary["city_share"] = class_summary["city_count"] / class_summary.groupby(
        "scenario"
    )["city_count"].transform("sum")
    scenario_stats = frame[frame["resistance_eligible"]].groupby("scenario").agg(
        median_downside_p05_pp=("downside_p05_pp", "median"),
        median_adjusted_score=("adjusted_resistance_score_p05", "median"),
        uncertain_city_count=("classification_uncertain", "sum"),
        tail_level_uncertain_city_count=("tail_level_uncertain", "sum"),
    )
    class_summary = class_summary.merge(
        scenario_stats, left_on="scenario", right_index=True, validate="many_to_one"
    )

    wide = frame.pivot(
        index=["region_key", "region_name_cn", "region_name_en", "province"],
        columns="scenario",
        values=["downside_p05_pp", "downside_p10_pp"],
    )
    wide.columns = [f"{metric}_{scenario}" for metric, scenario in wide.columns]
    amplification = wide.reset_index()
    amplification["stress_amplification_p05_pp"] = (
        amplification["downside_p05_pp_correlated_adverse_stress"]
        - amplification["downside_p05_pp_independent"]
    )
    amplification["stress_amplification_p10_pp"] = (
        amplification["downside_p10_pp_correlated_adverse_stress"]
        - amplification["downside_p10_pp_independent"]
    )
    eligible_by_city = frame.groupby("region_key")["resistance_eligible"].all()
    amplification["resistance_eligible"] = amplification["region_key"].map(eligible_by_city)
    amplification.loc[
        ~amplification["resistance_eligible"],
        ["stress_amplification_p05_pp", "stress_amplification_p10_pp"],
    ] = np.nan

    frame = frame.sort_values(["scenario", "region_key"]).reset_index(drop=True)
    diagnostics_frame = pd.DataFrame(diagnostics)
    diagnostics_frame["bootstrap_replicates_requested"] = bootstrap_replicates
    diagnostics_frame["bootstrap_replicates_successful"] = successful
    diagnostics_frame["bootstrap_replicates_rearranged"] = rearranged
    diagnostics_frame["stability_threshold"] = stability_threshold
    return frame, class_summary, diagnostics_frame, amplification
