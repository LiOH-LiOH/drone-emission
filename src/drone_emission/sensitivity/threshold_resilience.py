"""Probabilistic robustness of a fixed city distance-threshold policy.

The baseline road-distance threshold is optimized elsewhere and is deliberately
held fixed here.  Each Monte Carlo draw compares that mixed policy with an
all-e-bike counterfactual evaluated under the same sampled assumptions.
Provincial grid intensity and the city's OD distribution remain fixed.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Mapping

import numpy as np
import pandas as pd
from scipy.stats import norm, spearmanr

from emission.cal_single_delivery import DroneParams, EbikeParams

from .vectorized_lca import (
    DroneScenario,
    EbikeScenario,
    compute_drone_per_order_g,
    compute_ebike_per_order_g,
)


ANALYSIS_VERSION = "1.1.0"
THRESHOLD_RESILIENCE_SCENARIOS = (
    "independent",
    "correlated_adverse_stress",
)
THRESHOLD_RESILIENCE_MEGACITIES = (
    "beijing",
    "shanghai",
    "shenzhen",
    "chongqing",
    "guangzhou",
    "chengdu",
    "tianjin",
)
THRESHOLD_RESILIENCE_QUADRANT_IDS = (
    "low_baseline_low_resilience",
    "high_baseline_low_resilience",
    "low_baseline_high_resilience",
    "high_baseline_high_resilience",
)
THRESHOLD_RESILIENCE_QUADRANT_LABELS = {
    "low_baseline_low_resilience": "Low baseline / low safety probability",
    "high_baseline_low_resilience": "High baseline / low safety probability",
    "low_baseline_high_resilience": "Low baseline / high safety probability",
    "high_baseline_high_resilience": "High baseline / high safety probability",
}
PARAMETER_ORDER = (
    "kappa",
    "payload_kg",
    "drone_hover_s",
    "headwind_mps",
    "drone_cruise_kmh",
    "drone_energy_mult",
    "ebike_energy_mult",
    "drone_lifespan_yr",
    "drone_battery_lifespan_yr",
    "ebike_lifespan_yr",
    "ebike_battery_lifespan_yr",
    "drone_utilization_rate",
    "ebike_utilization_rate",
    "drone_production_co2_per_kg",
    "drone_battery_co2_per_kwh",
    "ebike_production_co2_per_kg",
    "ebike_battery_co2_per_kwh",
)

# +1 means a larger parameter is adverse to drone substitution; -1 means a
# smaller parameter is adverse. Payload and cruise speed remain independent in
# the stress case because their comparative effects can be non-monotonic.
STRESS_DIRECTIONS = {
    "kappa": -1,
    "drone_hover_s": 1,
    "headwind_mps": 1,
    "drone_energy_mult": 1,
    "ebike_energy_mult": -1,
    "drone_lifespan_yr": -1,
    "drone_battery_lifespan_yr": -1,
    "ebike_lifespan_yr": 1,
    "ebike_battery_lifespan_yr": 1,
    "drone_utilization_rate": -1,
    "ebike_utilization_rate": 1,
    "drone_production_co2_per_kg": 1,
    "drone_battery_co2_per_kwh": 1,
    "ebike_production_co2_per_kg": -1,
    "ebike_battery_co2_per_kwh": -1,
}


@dataclass(frozen=True)
class TriangularSpec:
    minimum: float
    mode: float
    maximum: float

    def __post_init__(self) -> None:
        if not (
            np.isfinite(self.minimum)
            and np.isfinite(self.mode)
            and np.isfinite(self.maximum)
        ):
            raise ValueError("triangular distribution values must be finite")
        if self.minimum > self.mode or self.mode > self.maximum:
            raise ValueError(
                "triangular distribution requires minimum <= mode <= maximum"
            )
        if self.minimum == self.maximum:
            raise ValueError("triangular distribution must have non-zero width")

    def ppf(self, u: np.ndarray) -> np.ndarray:
        """Inverse CDF, including one-sided triangular distributions."""
        u = np.asarray(u, dtype=np.float64)
        width = self.maximum - self.minimum
        split = (self.mode - self.minimum) / width
        left = self.minimum + np.sqrt(
            np.maximum(0.0, u * width * (self.mode - self.minimum))
        )
        right = self.maximum - np.sqrt(
            np.maximum(0.0, (1.0 - u) * width * (self.maximum - self.mode))
        )
        return np.where(u < split, left, right)


@dataclass(frozen=True)
class FixedThresholdAggregates:
    threshold_km: float
    total_orders: float
    all_road_order_km: float
    ebike_orders: float
    ebike_road_order_km: float
    drone_orders: float
    drone_air_order_km: float
    n_od_pairs: int

    @property
    def drone_order_ratio_pct(self) -> float:
        return self.drone_orders / self.total_orders * 100.0


def resolve_distributions(
    config: Mapping,
    ebike_params: EbikeParams,
    drone_params: DroneParams,
) -> dict[str, TriangularSpec]:
    """Load and validate the 17 configured triangular distributions."""
    section = config.get("threshold_resilience", {}) or {}
    raw = section.get("distributions", {}) or {}
    missing = [name for name in PARAMETER_ORDER if name not in raw]
    extra = sorted(set(raw) - set(PARAMETER_ORDER))
    if missing or extra:
        raise ValueError(
            f"threshold_resilience distributions mismatch; missing={missing}, extra={extra}"
        )

    specs: dict[str, TriangularSpec] = {}
    for name in PARAMETER_ORDER:
        values = raw[name]
        if not isinstance(values, (list, tuple)) or len(values) != 3:
            raise ValueError(f"distribution {name} must be [minimum, mode, maximum]")
        specs[name] = TriangularSpec(*(float(value) for value in values))

    effective_modes = {
        "kappa": ebike_params.kappa,
        "payload_kg": config.get("emission_params", {}).get("payload_kg", 1.0),
        "drone_hover_s": drone_params.hover_time,
        "headwind_mps": drone_params.headwind_speed_mps,
        "drone_cruise_kmh": drone_params.v_level,
        "drone_energy_mult": 1.0,
        "ebike_energy_mult": 1.0,
        "drone_lifespan_yr": drone_params.lifespan,
        "drone_battery_lifespan_yr": drone_params.battery_lifespan,
        "ebike_lifespan_yr": ebike_params.lifespan,
        "ebike_battery_lifespan_yr": ebike_params.battery_lifespan,
        "drone_utilization_rate": 1.0,
        "ebike_utilization_rate": 1.0,
        "drone_production_co2_per_kg": drone_params.e_body,
        "drone_battery_co2_per_kwh": drone_params.e_battery,
        "ebike_production_co2_per_kg": ebike_params.e_body,
        "ebike_battery_co2_per_kwh": ebike_params.e_battery,
    }
    mismatched = {
        name: (specs[name].mode, float(value))
        for name, value in effective_modes.items()
        if not np.isclose(specs[name].mode, float(value), rtol=0.0, atol=1e-9)
    }
    if mismatched:
        raise ValueError(
            "distribution modes do not match effective baseline parameters: "
            f"{mismatched}"
        )
    return specs


def sample_parameters(
    specs: Mapping[str, TriangularSpec],
    n_samples: int,
    seed: int,
    *,
    stress_spearman_rho: float | None = None,
) -> pd.DataFrame:
    """Draw independent samples or a risk-aligned Gaussian-copula stress case."""
    if n_samples <= 0:
        raise ValueError("n_samples must be positive")
    rng = np.random.default_rng(seed)
    values: dict[str, np.ndarray] = {}
    if stress_spearman_rho is None:
        for name in PARAMETER_ORDER:
            spec = specs[name]
            values[name] = rng.triangular(
                spec.minimum, spec.mode, spec.maximum, size=n_samples
            )
    else:
        if not 0.0 < stress_spearman_rho < 1.0:
            raise ValueError("stress_spearman_rho must lie strictly between 0 and 1")
        # For a Gaussian copula, rho_s = 6/pi * asin(rho_p/2).
        latent_rho = 2.0 * np.sin(np.pi * stress_spearman_rho / 6.0)
        common = rng.standard_normal(n_samples)
        scale_common = np.sqrt(latent_rho)
        scale_unique = np.sqrt(1.0 - latent_rho)
        for name in PARAMETER_ORDER:
            unique = rng.standard_normal(n_samples)
            direction = STRESS_DIRECTIONS.get(name)
            if direction is None:
                z = unique
            else:
                z = direction * scale_common * common + scale_unique * unique
            u = np.clip(norm.cdf(z), np.finfo(float).eps, 1.0 - np.finfo(float).eps)
            values[name] = specs[name].ppf(u)
    return pd.DataFrame(values, columns=list(PARAMETER_ORDER))


def aggregate_fixed_threshold(
    road_distance_km,
    straight_distance_km,
    annual_sales,
    threshold_km: float,
) -> FixedThresholdAggregates:
    """Reduce the OD table to the sufficient statistics for a fixed policy."""
    road = np.asarray(road_distance_km, dtype=np.float64)
    straight = np.asarray(straight_distance_km, dtype=np.float64)
    sales = np.asarray(annual_sales, dtype=np.float64)
    if not (len(road) == len(straight) == len(sales)) or len(road) == 0:
        raise ValueError("OD arrays must have the same non-zero length")
    if not np.isfinite(threshold_km):
        raise ValueError("fixed threshold must be finite")
    if not (np.isfinite(road).all() and np.isfinite(straight).all() and np.isfinite(sales).all()):
        raise ValueError("OD arrays contain non-finite values")
    if (sales < 0).any():
        raise ValueError("annual_sales contains negative values")
    valid = (road >= 0) & (straight >= 0) & (sales > 0)
    if not valid.any():
        raise ValueError("OD data has no reachable positive-volume rows")
    road = road[valid]
    straight = straight[valid]
    sales = sales[valid]
    use_drone = road >= threshold_km
    use_ebike = ~use_drone
    total_orders = float(np.sum(sales, dtype=np.float64))
    return FixedThresholdAggregates(
        threshold_km=float(threshold_km),
        total_orders=total_orders,
        all_road_order_km=float(np.sum(sales * road, dtype=np.float64)),
        ebike_orders=float(np.sum(sales[use_ebike], dtype=np.float64)),
        ebike_road_order_km=float(
            np.sum(sales[use_ebike] * road[use_ebike], dtype=np.float64)
        ),
        drone_orders=float(np.sum(sales[use_drone], dtype=np.float64)),
        drone_air_order_km=float(
            np.sum(sales[use_drone] * straight[use_drone], dtype=np.float64)
        ),
        n_od_pairs=int(valid.sum()),
    )


def _affine_coefficients(
    samples: pd.DataFrame,
    ebike_params: EbikeParams,
    drone_params: DroneParams,
    province_gef_kwh: float,
    rider_weight: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return per-sample intercept/slope for both per-order LCA functions."""
    distances = np.array([0.0, 1.0], dtype=np.float64)
    n = len(samples)
    ebike_intercept = np.empty(n, dtype=np.float64)
    ebike_slope = np.empty(n, dtype=np.float64)
    drone_intercept = np.empty(n, dtype=np.float64)
    drone_slope = np.empty(n, dtype=np.float64)

    for i, row in enumerate(samples.itertuples(index=False)):
        ep = replace(
            ebike_params,
            e_use=province_gef_kwh,
            e_body=row.ebike_production_co2_per_kg,
            e_battery=row.ebike_battery_co2_per_kwh,
        )
        dp = replace(
            drone_params,
            e_use=province_gef_kwh,
            v_level=row.drone_cruise_kmh,
            hover_time=row.drone_hover_s,
            headwind_speed_mps=row.headwind_mps,
            e_body=row.drone_production_co2_per_kg,
            e_battery=row.drone_battery_co2_per_kwh,
        )
        ebike = compute_ebike_per_order_g(
            distances,
            ep,
            EbikeScenario(
                kappa=row.kappa,
                energy_mult=row.ebike_energy_mult,
                payload_kg=row.payload_kg,
                rider_weight=rider_weight,
                lifespan_yr=row.ebike_lifespan_yr,
                battery_lifespan_yr=row.ebike_battery_lifespan_yr,
                utilization_mult=row.ebike_utilization_rate,
            ),
        )
        drone = compute_drone_per_order_g(
            distances,
            dp,
            DroneScenario(
                energy_mult=row.drone_energy_mult,
                payload_kg=row.payload_kg,
                lifespan_yr=row.drone_lifespan_yr,
                battery_lifespan_yr=row.drone_battery_lifespan_yr,
                utilization_mult=row.drone_utilization_rate,
            ),
        )
        ebike_intercept[i], ebike_slope[i] = ebike[0], ebike[1] - ebike[0]
        drone_intercept[i], drone_slope[i] = drone[0], drone[1] - drone[0]
    return ebike_intercept, ebike_slope, drone_intercept, drone_slope


def evaluate_samples(
    samples: pd.DataFrame,
    aggregates: FixedThresholdAggregates,
    ebike_params: EbikeParams,
    drone_params: DroneParams,
    province_gef_kwh: float,
    rider_weight: float,
    positive_tolerance_pct: float = 1e-9,
) -> pd.DataFrame:
    """Evaluate a fixed threshold for all supplied parameter samples."""
    e0, e1, d0, d1 = _affine_coefficients(
        samples, ebike_params, drone_params, province_gef_kwh, rider_weight
    )
    all_ebike_g = (
        e0 * aggregates.total_orders + e1 * aggregates.all_road_order_km
    )
    mixed_g = (
        e0 * aggregates.ebike_orders
        + e1 * aggregates.ebike_road_order_km
        + d0 * aggregates.drone_orders
        + d1 * aggregates.drone_air_order_km
    )
    if (all_ebike_g <= 0).any():
        raise ValueError("sampled all-e-bike lifecycle total is not positive")
    reduction_pct = (all_ebike_g - mixed_g) / all_ebike_g * 100.0
    result = samples.copy()
    result["all_ebike_lifecycle_kg"] = all_ebike_g / 1000.0
    result["mixed_lifecycle_kg"] = mixed_g / 1000.0
    result["annual_reduction_kg"] = (all_ebike_g - mixed_g) / 1000.0
    result["reduction_pct"] = reduction_pct
    result["positive_reduction"] = reduction_pct > positive_tolerance_pct
    return result


def wilson_interval(
    successes: int,
    total: int,
    z: float = 1.959963984540054,
) -> tuple[float, float]:
    if total <= 0 or successes < 0 or successes > total:
        raise ValueError("invalid binomial counts")
    p = successes / total
    denominator = 1.0 + z * z / total
    centre = (p + z * z / (2.0 * total)) / denominator
    half = z * np.sqrt(p * (1.0 - p) / total + z * z / (4.0 * total * total)) / denominator
    low = 0.0 if successes == 0 else max(0.0, centre - half)
    high = 1.0 if successes == total else min(1.0, centre + half)
    return low, high


def derive_threshold_resilience_typology(
    summary: pd.DataFrame,
    expected_cities: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Derive separate four-quadrant city typologies for both scenarios."""
    required = {
        "region_key",
        "region_name_cn",
        "region_name_en",
        "scenario",
        "n_samples",
        "baseline_fixed_threshold_reduction_pct",
        "prob_reduction_positive",
        "prob_positive_ci95_low",
        "prob_positive_ci95_high",
    }
    missing = sorted(required - set(summary.columns))
    if missing:
        raise KeyError(f"threshold-resilience summary missing columns: {missing}")

    frame = summary[
        summary["scenario"].isin(THRESHOLD_RESILIENCE_SCENARIOS)
    ].copy()
    duplicates = frame.duplicated(["region_key", "scenario"], keep=False)
    if duplicates.any():
        keys = sorted(frame.loc[duplicates, "region_key"].unique())
        raise ValueError(f"duplicate city/scenario rows: {keys}")
    scenario_counts = frame.groupby("region_key")["scenario"].nunique()
    incomplete = sorted(
        scenario_counts[
            scenario_counts != len(THRESHOLD_RESILIENCE_SCENARIOS)
        ].index
    )
    if incomplete:
        raise ValueError(f"cities missing one resilience scenario: {incomplete}")
    n_cities = frame["region_key"].nunique()
    if expected_cities is not None and n_cities != expected_cities:
        raise ValueError(f"expected {expected_cities} cities, found {n_cities}")

    numeric_columns = (
        "n_samples",
        "baseline_fixed_threshold_reduction_pct",
        "prob_reduction_positive",
        "prob_positive_ci95_low",
        "prob_positive_ci95_high",
    )
    for column in numeric_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if not np.isfinite(frame[list(numeric_columns)].to_numpy(dtype=float)).all():
        raise ValueError("typology inputs must be finite")
    if (frame["n_samples"] <= 0).any():
        raise ValueError("n_samples must be positive")
    for column in (
        "prob_reduction_positive",
        "prob_positive_ci95_low",
        "prob_positive_ci95_high",
    ):
        if ((frame[column] < 0) | (frame[column] > 1)).any():
            raise ValueError(f"{column} must lie within [0, 1]")
    invalid_intervals = (
        (frame["prob_positive_ci95_low"] > frame["prob_reduction_positive"])
        | (frame["prob_reduction_positive"] > frame["prob_positive_ci95_high"])
    )
    if invalid_intervals.any():
        raise ValueError("probability point estimates must lie within their intervals")

    baseline_spread = frame.groupby("region_key")[
        "baseline_fixed_threshold_reduction_pct"
    ].agg(lambda values: float(values.max() - values.min()))
    inconsistent = sorted(baseline_spread[baseline_spread > 1e-12].index)
    if inconsistent:
        raise ValueError(f"city baselines differ between scenarios: {inconsistent}")

    city_baselines = frame.drop_duplicates("region_key")[
        "baseline_fixed_threshold_reduction_pct"
    ]
    baseline_median = float(city_baselines.median())
    scenario_medians = frame.groupby("scenario")[
        "prob_reduction_positive"
    ].median()
    frame["baseline_median"] = baseline_median
    frame["resilience_probability_median"] = frame["scenario"].map(
        scenario_medians
    )
    high_baseline = (
        frame["baseline_fixed_threshold_reduction_pct"] >= baseline_median
    )
    high_resilience = (
        frame["prob_reduction_positive"]
        >= frame["resilience_probability_median"]
    )
    quadrant_index = high_baseline.astype(int) + 2 * high_resilience.astype(int)
    frame["quadrant_id"] = [
        THRESHOLD_RESILIENCE_QUADRANT_IDS[index] for index in quadrant_index
    ]
    frame["quadrant_label"] = frame["quadrant_id"].map(
        THRESHOLD_RESILIENCE_QUADRANT_LABELS
    )
    frame["boundary_uncertain"] = (
        (frame["prob_positive_ci95_low"] <= frame["resilience_probability_median"])
        & (frame["prob_positive_ci95_high"] >= frame["resilience_probability_median"])
    )
    frame["is_census_megacity"] = frame["region_key"].isin(
        THRESHOLD_RESILIENCE_MEGACITIES
    )
    frame["scenario"] = pd.Categorical(
        frame["scenario"], categories=THRESHOLD_RESILIENCE_SCENARIOS, ordered=True
    )
    frame = frame.sort_values(["scenario", "region_key"]).reset_index(drop=True)
    frame["scenario"] = frame["scenario"].astype(str)

    full_index = pd.MultiIndex.from_product(
        [THRESHOLD_RESILIENCE_SCENARIOS, THRESHOLD_RESILIENCE_QUADRANT_IDS],
        names=["scenario", "quadrant_id"],
    )
    grouped = frame.groupby(
        ["scenario", "quadrant_id"], observed=False
    ).size().reindex(full_index, fill_value=0)
    typology_summary = grouped.rename("city_count").reset_index()
    typology_summary["city_share"] = (
        typology_summary["city_count"]
        / typology_summary.groupby("scenario")["city_count"].transform("sum")
    )
    typology_summary["quadrant_label"] = typology_summary["quadrant_id"].map(
        THRESHOLD_RESILIENCE_QUADRANT_LABELS
    )
    typology_summary["baseline_median"] = baseline_median
    typology_summary["resilience_probability_median"] = typology_summary[
        "scenario"
    ].map(scenario_medians)
    return frame, typology_summary


def summarize_samples(
    evaluated: pd.DataFrame,
    scenario: str,
    aggregates: FixedThresholdAggregates,
    baseline_reduction_pct: float,
) -> dict:
    reduction = evaluated["reduction_pct"].to_numpy(dtype=np.float64)
    successes = int(evaluated["positive_reduction"].sum())
    n = len(evaluated)
    low, high = wilson_interval(successes, n)
    return {
        "scenario": scenario,
        "n_samples": n,
        "fixed_threshold_km": aggregates.threshold_km,
        "fixed_drone_order_ratio_pct": aggregates.drone_order_ratio_pct,
        "baseline_fixed_threshold_reduction_pct": baseline_reduction_pct,
        "positive_count": successes,
        "prob_reduction_positive": successes / n,
        "prob_positive_ci95_low": low,
        "prob_positive_ci95_high": high,
        "reduction_mean_pct": float(np.mean(reduction)),
        "reduction_std_pct": float(np.std(reduction, ddof=1)),
        "reduction_p05_pct": float(np.percentile(reduction, 5.0)),
        "reduction_p50_pct": float(np.percentile(reduction, 50.0)),
        "reduction_p95_pct": float(np.percentile(reduction, 95.0)),
        "minimum_reduction_pct": float(np.min(reduction)),
        "maximum_reduction_pct": float(np.max(reduction)),
    }


def spearman_drivers(evaluated: pd.DataFrame) -> pd.DataFrame:
    reduction = evaluated["reduction_pct"].to_numpy(dtype=np.float64)
    rows = []
    for name in PARAMETER_ORDER:
        rho, pvalue = spearmanr(
            evaluated[name].to_numpy(dtype=np.float64), reduction
        )
        rows.append(
            {
                "parameter": name,
                "spearman_rho": float(rho),
                "spearman_pvalue": float(pvalue),
                "abs_spearman_rho": abs(float(rho)),
                "direction": "increases_reduction" if rho > 0 else "decreases_reduction",
            }
        )
    result = pd.DataFrame(rows).sort_values(
        ["abs_spearman_rho", "parameter"], ascending=[False, True]
    )
    result.insert(0, "rank", np.arange(1, len(result) + 1))
    return result.reset_index(drop=True)


def mode_sample(specs: Mapping[str, TriangularSpec]) -> pd.DataFrame:
    return pd.DataFrame(
        [{name: specs[name].mode for name in PARAMETER_ORDER}],
        columns=list(PARAMETER_ORDER),
    )
