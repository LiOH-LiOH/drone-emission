"""Baseline-relative retention metrics for threshold-resilience samples.

The deterministic fixed-threshold reduction ``R0`` is held as the city-specific
reference. Each paired Monte Carlo reduction ``R`` is summarized as ``R / R0``.
This module derives statistics only; it does not generate new samples.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from .threshold_resilience import wilson_interval


RETENTION_SCENARIOS = (
    "independent",
    "correlated_adverse_stress",
)
RETENTION_CURVE_TARGETS = tuple(round(index * 0.05, 2) for index in range(21))
RETENTION_KEY_TARGETS = (0.25, 0.50, 0.75, 1.00)

SUMMARY_IDENTIFIER_COLUMNS = (
    "region_key",
    "region_name_cn",
    "region_name_en",
    "province",
    "scenario",
    "n_samples",
    "baseline_fixed_threshold_reduction_pct",
    "prob_reduction_positive",
)


def _target_suffix(target: float) -> str:
    return f"{int(round(target * 100)):03d}pct"


def _probability_fields(
    reduction: np.ndarray,
    baseline_reduction_pct: float,
    target: float,
    positive_tolerance_pct: float,
) -> tuple[int, float, float, float]:
    if np.isclose(target, 0.0):
        success = reduction > positive_tolerance_pct
    else:
        success = reduction >= target * baseline_reduction_pct
    count = int(np.count_nonzero(success))
    probability = count / len(reduction)
    low, high = wilson_interval(count, len(reduction))
    return count, probability, low, high


def summarize_baseline_retention(
    reduction_pct: Iterable[float],
    baseline_reduction_pct: float,
    *,
    scenario: str,
    curve_targets: Iterable[float] = RETENTION_CURVE_TARGETS,
    positive_tolerance_pct: float = 1e-9,
) -> tuple[dict, pd.DataFrame]:
    """Summarize one city's baseline retention for one uncertainty scenario."""
    if scenario not in RETENTION_SCENARIOS:
        raise ValueError(f"unknown retention scenario: {scenario}")
    reduction = np.asarray(reduction_pct, dtype=np.float64)
    if reduction.ndim != 1 or reduction.size == 0:
        raise ValueError("reduction_pct must be a non-empty one-dimensional array")
    if not np.isfinite(reduction).all():
        raise ValueError("reduction_pct contains non-finite values")
    baseline = float(baseline_reduction_pct)
    if not np.isfinite(baseline):
        raise ValueError("baseline reduction must be finite")
    targets = tuple(float(value) for value in curve_targets)
    if not targets or any(
        not np.isfinite(value) or value < 0.0 or value > 1.0
        for value in targets
    ):
        raise ValueError("retention curve targets must be finite and within [0, 1]")
    if any(right <= left for left, right in zip(targets, targets[1:])):
        raise ValueError("retention curve targets must be strictly increasing")

    eligible = baseline > positive_tolerance_pct
    summary = {
        "scenario": scenario,
        "n_samples": len(reduction),
        "baseline_fixed_threshold_reduction_pct": baseline,
        "retention_eligible": eligible,
        "retention_exclusion_reason": "" if eligible else "nonpositive_baseline",
        "baseline_retention_p05": np.nan,
        "baseline_retention_p50": np.nan,
        "baseline_retention_p95": np.nan,
    }
    for target in RETENTION_KEY_TARGETS:
        suffix = _target_suffix(target)
        summary[f"retain_{suffix}_count"] = pd.NA
        summary[f"prob_retain_{suffix}_baseline"] = np.nan
        summary[f"prob_retain_{suffix}_ci95_low"] = np.nan
        summary[f"prob_retain_{suffix}_ci95_high"] = np.nan

    curve_rows = []
    if eligible:
        ratio = reduction / baseline
        quantiles = np.percentile(ratio, [5.0, 50.0, 95.0])
        summary.update({
            "baseline_retention_p05": float(quantiles[0]),
            "baseline_retention_p50": float(quantiles[1]),
            "baseline_retention_p95": float(quantiles[2]),
        })
        key_probabilities = {}
        for target in targets:
            count, probability, low, high = _probability_fields(
                reduction, baseline, target, positive_tolerance_pct
            )
            key_probabilities[target] = (count, probability, low, high)
            curve_rows.append({
                "scenario": scenario,
                "n_samples": len(reduction),
                "baseline_fixed_threshold_reduction_pct": baseline,
                "retention_eligible": True,
                "retention_target_fraction": target,
                "target_reduction_pct": target * baseline,
                "success_count": count,
                "prob_retain_target": probability,
                "prob_retain_target_ci95_low": low,
                "prob_retain_target_ci95_high": high,
            })
        for target in RETENTION_KEY_TARGETS:
            if target not in key_probabilities:
                count, probability, low, high = _probability_fields(
                    reduction, baseline, target, positive_tolerance_pct
                )
            else:
                count, probability, low, high = key_probabilities[target]
            suffix = _target_suffix(target)
            summary[f"retain_{suffix}_count"] = count
            summary[f"prob_retain_{suffix}_baseline"] = probability
            summary[f"prob_retain_{suffix}_ci95_low"] = low
            summary[f"prob_retain_{suffix}_ci95_high"] = high
    else:
        for target in targets:
            curve_rows.append({
                "scenario": scenario,
                "n_samples": len(reduction),
                "baseline_fixed_threshold_reduction_pct": baseline,
                "retention_eligible": False,
                "retention_target_fraction": target,
                "target_reduction_pct": np.nan,
                "success_count": pd.NA,
                "prob_retain_target": np.nan,
                "prob_retain_target_ci95_low": np.nan,
                "prob_retain_target_ci95_high": np.nan,
            })
    return summary, pd.DataFrame(curve_rows)


def derive_city_baseline_retention(
    city_summary: pd.DataFrame,
    samples: pd.DataFrame,
    *,
    curve_targets: Iterable[float] = RETENTION_CURVE_TARGETS,
    positive_tolerance_pct: float = 1e-9,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Derive auditable summary and curve rows for one city."""
    missing_summary = sorted(set(SUMMARY_IDENTIFIER_COLUMNS) - set(city_summary.columns))
    if missing_summary:
        raise KeyError(f"city summary missing columns: {missing_summary}")
    missing_samples = sorted({"scenario", "reduction_pct"} - set(samples.columns))
    if missing_samples:
        raise KeyError(f"city samples missing columns: {missing_samples}")
    region_keys = city_summary["region_key"].dropna().unique()
    if len(region_keys) != 1:
        raise ValueError("city summary must contain exactly one region_key")
    region_key = str(region_keys[0])
    scenario_rows = city_summary[city_summary["scenario"].isin(RETENTION_SCENARIOS)].copy()
    if set(scenario_rows["scenario"]) != set(RETENTION_SCENARIOS):
        raise ValueError(f"{region_key} summary must contain both retention scenarios")
    if scenario_rows.duplicated("scenario").any():
        raise ValueError(f"{region_key} summary contains duplicate scenarios")
    baselines = pd.to_numeric(
        scenario_rows["baseline_fixed_threshold_reduction_pct"], errors="coerce"
    ).to_numpy(dtype=np.float64)
    if not np.isfinite(baselines).all() or not np.allclose(
        baselines, baselines[0], rtol=0.0, atol=1e-12
    ):
        raise ValueError(f"{region_key} scenarios must share one deterministic baseline")
    unexpected_samples = sorted(set(samples["scenario"].dropna()) - set(RETENTION_SCENARIOS))
    if unexpected_samples:
        raise ValueError(f"{region_key} samples contain unknown scenarios: {unexpected_samples}")

    summary_outputs = []
    curve_outputs = []
    for scenario in RETENTION_SCENARIOS:
        source = scenario_rows[scenario_rows["scenario"] == scenario].iloc[0]
        expected_n = int(source["n_samples"])
        reduction = pd.to_numeric(
            samples.loc[samples["scenario"] == scenario, "reduction_pct"],
            errors="coerce",
        ).to_numpy(dtype=np.float64)
        if len(reduction) != expected_n:
            raise ValueError(
                f"{region_key}/{scenario} expected {expected_n} samples, found {len(reduction)}"
            )
        derived, curve = summarize_baseline_retention(
            reduction,
            float(source["baseline_fixed_threshold_reduction_pct"]),
            scenario=scenario,
            curve_targets=curve_targets,
            positive_tolerance_pct=positive_tolerance_pct,
        )
        identifiers = {
            column: source[column]
            for column in SUMMARY_IDENTIFIER_COLUMNS
            if column not in {"scenario", "n_samples", "baseline_fixed_threshold_reduction_pct"}
        }
        summary_outputs.append({**identifiers, **derived})
        for column, value in reversed(tuple(identifiers.items())):
            curve.insert(0, column, value)
        curve_outputs.append(curve)

    return (
        pd.DataFrame(summary_outputs),
        pd.concat(curve_outputs, ignore_index=True),
    )
