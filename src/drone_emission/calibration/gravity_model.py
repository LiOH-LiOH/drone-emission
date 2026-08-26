"""Numerical core for conditional building-choice gravity calibration."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import logsumexp

PARAMETER_NAMES = ("alpha", "beta", "gamma")
CURRENT_PARAMETERS = np.array([1.7, 1.1131, 1.4115], dtype=float)
PARAMETER_INDEX = {name: index for index, name in enumerate(PARAMETER_NAMES)}


@dataclass(frozen=True)
class CandidateCSR:
    """Memory-mapped shop-to-building candidate attributes."""

    indptr: np.ndarray
    building_index: np.ndarray
    log_volume: np.ndarray
    log_distance: np.ndarray
    distance_km: np.ndarray

    @classmethod
    def load(cls, directory: Path, mmap_mode: str = "r") -> "CandidateCSR":
        directory = Path(directory)
        return cls(
            *(np.load(directory / name, mmap_mode=mmap_mode) for name in (
                "candidate_indptr.npy",
                "candidate_building_index.npy",
                "candidate_log_volume.npy",
                "candidate_log_distance.npy",
                "candidate_distance_km.npy",
            ))
        )

    @property
    def shop_count(self) -> int:
        return int(len(self.indptr) - 1)

    def bounds(self, shop_index: int) -> tuple[int, int]:
        return int(self.indptr[shop_index]), int(self.indptr[shop_index + 1])


def utility_features(candidates: CandidateCSR, start: int, stop: int) -> np.ndarray:
    """Return columns matching alpha, beta, gamma in the linear utility."""
    return np.column_stack(
        (
            candidates.log_volume[start:stop],
            candidates.log_distance[start:stop],
            -candidates.distance_km[start:stop],
        )
    )


def _group_observations(observations: pd.DataFrame) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    required = {"shop_index", "candidate_position", "count"}
    missing = required - set(observations)
    if missing:
        raise ValueError("Observation table is missing: " + ", ".join(sorted(missing)))
    grouped = {}
    for shop_index, group in observations.groupby("shop_index", sort=True):
        grouped[int(shop_index)] = (
            group["candidate_position"].to_numpy(dtype=np.int64),
            group["count"].to_numpy(dtype=float),
        )
    return grouped


def objective_gradient_hessian(
    parameters: np.ndarray,
    candidates: CandidateCSR,
    observations: pd.DataFrame,
    need_hessian: bool = True,
) -> tuple[float, np.ndarray, np.ndarray | None]:
    """Exact negative log-likelihood, analytic gradient, and Hessian."""
    theta = np.asarray(parameters, dtype=float)
    if theta.shape != (3,) or not np.isfinite(theta).all():
        raise ValueError("parameters must be three finite values")
    value = 0.0
    gradient = np.zeros(3, dtype=float)
    hessian = np.zeros((3, 3), dtype=float) if need_hessian else None
    for shop_index, (positions, counts) in _group_observations(observations).items():
        start, stop = candidates.bounds(shop_index)
        features = utility_features(candidates, start, stop)
        if len(features) == 0:
            raise ValueError(f"Shop {shop_index} has no candidates")
        local = positions - start
        if np.any(local < 0) or np.any(local >= len(features)):
            raise ValueError(f"Observation outside candidate slice for shop {shop_index}")
        utilities = features @ theta
        log_normalizer = float(logsumexp(utilities))
        total = float(counts.sum())
        observed_sum = np.sum(features[local] * counts[:, None], axis=0)
        probabilities = np.exp(utilities - log_normalizer)
        expected = probabilities @ features
        value += total * log_normalizer - float(observed_sum @ theta)
        gradient += total * expected - observed_sum
        if need_hessian:
            second = (features * probabilities[:, None]).T @ features
            hessian += total * (second - np.outer(expected, expected))
    return float(value), gradient, hessian


def fit_parameters(
    candidates: CandidateCSR,
    observations: pd.DataFrame,
    initial: Iterable[float] = CURRENT_PARAMETERS,
    free_parameters: Iterable[str] = PARAMETER_NAMES,
    fixed_parameters: dict[str, float] | None = None,
) -> dict:
    """Fit non-negative alpha/beta and positive gamma with L-BFGS-B."""
    free_parameters = tuple(free_parameters)
    fixed_parameters = dict(fixed_parameters or {})
    for name in free_parameters:
        if name not in PARAMETER_INDEX:
            raise ValueError(f"Unknown free parameter: {name}")
    for name in fixed_parameters:
        if name not in PARAMETER_INDEX:
            raise ValueError(f"Unknown fixed parameter: {name}")
    if set(free_parameters) & set(fixed_parameters):
        raise ValueError("A parameter cannot be both free and fixed")
    initial = np.asarray(tuple(initial), dtype=float)
    if initial.shape == (3,):
        initial_free = np.asarray(
            [initial[PARAMETER_INDEX[name]] for name in free_parameters], dtype=float
        )
    else:
        initial_free = np.asarray(initial, dtype=float)
    theta0 = CURRENT_PARAMETERS.copy()
    for name, value in fixed_parameters.items():
        theta0[PARAMETER_INDEX[name]] = float(value)

    def expand_theta(values: np.ndarray) -> np.ndarray:
        theta = theta0.copy()
        for name, value in zip(free_parameters, values):
            theta[PARAMETER_INDEX[name]] = float(value)
        return theta

    def fun(theta):
        full_theta = expand_theta(theta)
        value, gradient, _ = objective_gradient_hessian(
            full_theta, candidates, observations, need_hessian=False
        )
        free_gradient = np.asarray(
            [gradient[PARAMETER_INDEX[name]] for name in free_parameters], dtype=float
        )
        return value, free_gradient

    fitted = minimize(
        fun,
        initial_free,
        method="L-BFGS-B",
        jac=True,
        bounds=tuple(
            (1e-10, None) if name == "gamma" else (0.0, None)
            for name in free_parameters
        ),
        options={"maxiter": 1000, "ftol": 1e-12, "gtol": 1e-8},
    )
    fitted_theta = expand_theta(fitted.x)
    value, gradient, hessian = objective_gradient_hessian(
        fitted_theta, candidates, observations, need_hessian=True
    )
    eigenvalues = np.linalg.eigvalsh(hessian)
    return {
        "parameters": {
            name: float(value_) for name, value_ in zip(PARAMETER_NAMES, fitted_theta)
        },
        "free_parameters": list(free_parameters),
        "fixed_parameters": {name: float(value) for name, value in fixed_parameters.items()},
        "success": bool(fitted.success),
        "status": int(fitted.status),
        "message": str(fitted.message),
        "iterations": int(fitted.nit),
        "function_evaluations": int(fitted.nfev),
        "negative_log_likelihood": float(value),
        "gradient": gradient.tolist(),
        "gradient_norm": float(np.linalg.norm(gradient)),
        "hessian": hessian.tolist(),
        "hessian_eigenvalues": eigenvalues.tolist(),
        "hessian_condition_number": float(np.linalg.cond(hessian)),
    }


def clustered_covariance(
    parameters: Iterable[float],
    candidates: CandidateCSR,
    observations: pd.DataFrame,
) -> dict:
    """Compute a shop-clustered sandwich covariance and Wald intervals."""
    theta = np.asarray(tuple(parameters), dtype=float)
    _, _, bread = objective_gradient_hessian(theta, candidates, observations, True)
    scores = []
    for shop_index, (positions, counts) in _group_observations(observations).items():
        start, stop = candidates.bounds(shop_index)
        features = utility_features(candidates, start, stop)
        local = positions - start
        utilities = features @ theta
        probabilities = np.exp(utilities - logsumexp(utilities))
        observed = np.sum(features[local] * counts[:, None], axis=0)
        scores.append(observed - float(counts.sum()) * (probabilities @ features))
    score_matrix = np.asarray(scores)
    inverse = np.linalg.pinv(bread, hermitian=True)
    covariance = inverse @ (score_matrix.T @ score_matrix) @ inverse
    errors = np.sqrt(np.maximum(np.diag(covariance), 0.0))
    rows = []
    for name, estimate, error in zip(PARAMETER_NAMES, theta, errors):
        rows.append(
            {
                "parameter": name,
                "estimate": float(estimate),
                "standard_error": float(error),
                "ci95_low": float(estimate - 1.959963984540054 * error),
                "ci95_high": float(estimate + 1.959963984540054 * error),
            }
        )
    return {"covariance": covariance.tolist(), "parameters": rows, "cluster_count": len(scores)}


def model_definitions(fitted_parameters: Iterable[float]) -> dict[str, np.ndarray]:
    """Return the five fixed comparison models in reporting order."""
    fitted = np.asarray(tuple(fitted_parameters), dtype=float)
    return {
        "current_parameters": CURRENT_PARAMETERS.copy(),
        "uniform_buildings": np.zeros(3, dtype=float),
        "building_volume_only": np.array([fitted[0], 0.0, 0.0]),
        "tanner_distance_only": np.array([0.0, fitted[1], fitted[2]]),
        "joint_three_parameter": fitted,
    }


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, quantile: float) -> float:
    order = np.argsort(values, kind="mergesort")
    values, weights = values[order], weights[order]
    cumulative = np.cumsum(weights)
    return float(values[np.searchsorted(cumulative, quantile * cumulative[-1], side="left")])


def _distance_metrics(
    observed_values: np.ndarray,
    observed_weights: np.ndarray,
    predicted_values: np.ndarray,
    predicted_weights: np.ndarray,
) -> dict:
    """Exact weighted empirical KS, Wasserstein, and distance summaries."""
    from scipy.stats import wasserstein_distance

    grid = np.unique(np.concatenate((observed_values, predicted_values)))

    def cdf(values, weights):
        order = np.argsort(values, kind="mergesort")
        values, weights = values[order], weights[order]
        cumulative = np.cumsum(weights) / np.sum(weights)
        indices = np.searchsorted(values, grid, side="right") - 1
        return np.where(indices >= 0, cumulative[np.maximum(indices, 0)], 0.0)

    return {
        "ks": float(np.max(np.abs(cdf(observed_values, observed_weights) - cdf(predicted_values, predicted_weights)))),
        "wasserstein_1_km": float(wasserstein_distance(
            observed_values, predicted_values, observed_weights, predicted_weights
        )),
        "observed_mean_km": float(np.average(observed_values, weights=observed_weights)),
        "observed_median_km": _weighted_quantile(observed_values, observed_weights, 0.5),
        "observed_p95_km": _weighted_quantile(observed_values, observed_weights, 0.95),
        "predicted_mean_km": float(np.average(predicted_values, weights=predicted_weights)),
        "predicted_median_km": _weighted_quantile(predicted_values, predicted_weights, 0.5),
        "predicted_p95_km": _weighted_quantile(predicted_values, predicted_weights, 0.95),
    }


def _evaluate_feature_model(
    parameters: Iterable[float],
    candidates: CandidateCSR,
    observations: pd.DataFrame,
    feature_factory,
    return_components: bool = False,
    include_ranks: bool = True,
):
    """Evaluate one linear-utility choice model and its distance mixture."""
    theta = np.asarray(tuple(parameters), dtype=float)
    nll = 0.0
    top1 = top10 = top1pct = 0.0
    total_orders = float(observations["count"].sum())
    observed_distances = []
    observed_weights = []
    predicted_distances = []
    predicted_weights = []
    for shop_index, (positions, counts) in _group_observations(observations).items():
        start, stop = candidates.bounds(shop_index)
        features = feature_factory(candidates, start, stop)
        utilities = features @ theta
        logz = float(logsumexp(utilities))
        probabilities = np.exp(utilities - logz)
        local = positions - start
        nll += float(np.sum(counts * (logz - utilities[local])))
        if include_ranks:
            order = np.argsort(-utilities, kind="mergesort")
            ranks = np.empty(len(order), dtype=int)
            ranks[order] = np.arange(1, len(order) + 1)
            actual_ranks = ranks[local]
            top1 += float(np.sum(counts[actual_ranks <= 1]))
            top10 += float(np.sum(counts[actual_ranks <= min(10, len(order))]))
            cutoff = max(1, int(math.ceil(len(order) * 0.01)))
            top1pct += float(np.sum(counts[actual_ranks <= cutoff]))
        shop_total = float(counts.sum())
        distances = np.asarray(candidates.distance_km[start:stop])
        observed_distances.append(distances[local])
        observed_weights.append(counts)
        predicted_distances.append(distances)
        predicted_weights.append(probabilities * shop_total)
    observed_distances = np.concatenate(observed_distances)
    observed_weights = np.concatenate(observed_weights)
    predicted_distances = np.concatenate(predicted_distances)
    predicted_weights = np.concatenate(predicted_weights)
    metrics = {
        "order_count": int(total_orders),
        "negative_log_likelihood": float(nll),
        "nll_per_order": float(nll / total_orders),
        "perplexity": float(np.exp(nll / total_orders)),
        **_distance_metrics(
            observed_distances, observed_weights, predicted_distances, predicted_weights
        ),
    }
    if include_ranks:
        metrics.update({
            "top_1": float(top1 / total_orders),
            "top_10": float(top10 / total_orders),
            "top_1_percent": float(top1pct / total_orders),
        })
    # Plot sources use fixed 50 m bins; all reported metrics above remain exact.
    bin_width_km = 0.05
    observed_bins = np.floor(observed_distances / bin_width_km) * bin_width_km
    predicted_bins = np.floor(predicted_distances / bin_width_km) * bin_width_km
    observed_rows = pd.DataFrame(
        {"distance_bin_km": observed_bins, "observed_order_weight": observed_weights}
    ).groupby("distance_bin_km", as_index=False)["observed_order_weight"].sum()
    predicted_rows = pd.DataFrame(
        {"distance_bin_km": predicted_bins, "predicted_order_weight": predicted_weights}
    ).groupby("distance_bin_km", as_index=False)["predicted_order_weight"].sum()
    distribution = observed_rows.merge(
        predicted_rows, on="distance_bin_km", how="outer"
    ).fillna(0.0)
    if not return_components:
        return metrics, distribution
    return metrics, distribution, {
        "observed_values": observed_distances,
        "observed_weights": observed_weights,
        "predicted_values": predicted_distances,
        "predicted_weights": predicted_weights,
    }


def evaluate_model(
    parameters: Iterable[float],
    candidates: CandidateCSR,
    observations: pd.DataFrame,
    return_components: bool = False,
    include_ranks: bool = True,
):
    """Evaluate Tanner likelihood, ranks, and exact predicted distance mixture."""
    return _evaluate_feature_model(
        parameters,
        candidates,
        observations,
        utility_features,
        return_components=return_components,
        include_ranks=include_ranks,
    )


def _r_squared(
    observed: np.ndarray,
    predicted: np.ndarray,
    weights: np.ndarray | None = None,
) -> float:
    """Return predictive R-squared against the observed-mean null model."""
    observed = np.asarray(observed, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    if weights is None:
        center = float(np.mean(observed))
        residual = float(np.sum(np.square(observed - predicted)))
        total = float(np.sum(np.square(observed - center)))
    else:
        weights = np.asarray(weights, dtype=float)
        center = float(np.average(observed, weights=weights))
        residual = float(np.sum(weights * np.square(observed - predicted)))
        total = float(np.sum(weights * np.square(observed - center)))
    return float(1.0 - residual / total) if total > 0 else float("nan")


def _sorted_support_wasserstein(
    distances: np.ndarray,
    observed_weights: np.ndarray,
    predicted_weights: np.ndarray,
) -> float:
    """Exact W1 distance when candidate distances are already increasing."""
    if len(distances) <= 1:
        return 0.0
    if np.any(np.diff(distances) < 0):
        order = np.argsort(distances, kind="mergesort")
        distances = distances[order]
        observed_weights = observed_weights[order]
        predicted_weights = predicted_weights[order]
    observed_cdf = np.cumsum(observed_weights) / np.sum(observed_weights)
    predicted_cdf = np.cumsum(predicted_weights) / np.sum(predicted_weights)
    return float(
        np.sum(np.abs(observed_cdf[:-1] - predicted_cdf[:-1]) * np.diff(distances))
    )


def _merchant_spatial_fit_feature(
    parameters: Iterable[float],
    candidates: CandidateCSR,
    observations: pd.DataFrame,
    scope: str,
    feature_factory,
) -> tuple[pd.DataFrame, dict]:
    """Compute exact merchant-level distance fit from sufficient statistics."""
    theta = np.asarray(tuple(parameters), dtype=float)
    rows = []
    for shop_index, (positions, counts) in _group_observations(observations).items():
        start, stop = candidates.bounds(shop_index)
        distances = np.asarray(candidates.distance_km[start:stop])
        utilities = feature_factory(candidates, start, stop) @ theta
        probabilities = np.exp(utilities - logsumexp(utilities))
        local = positions - start
        observed_weights = np.zeros(stop - start, dtype=float)
        np.add.at(observed_weights, local, counts)
        total = float(counts.sum())
        predicted_weights = probabilities * total
        observed_mean = float(np.average(distances, weights=observed_weights))
        predicted_mean = float(np.average(distances, weights=predicted_weights))
        rows.append(
            {
                "scope": scope,
                "shop_index": shop_index,
                "order_count": int(total),
                "candidate_count": int(stop - start),
                "observed_mean_distance_km": observed_mean,
                "predicted_mean_distance_km": predicted_mean,
                "absolute_mean_error_m": abs(observed_mean - predicted_mean) * 1000.0,
                "wasserstein_m": _sorted_support_wasserstein(
                    distances, observed_weights, predicted_weights
                ) * 1000.0,
            }
        )
    frame = pd.DataFrame(rows)
    observed = frame["observed_mean_distance_km"].to_numpy(float)
    predicted = frame["predicted_mean_distance_km"].to_numpy(float)
    order_weights = frame["order_count"].to_numpy(float)
    summary = {
        "scope": scope,
        "merchant_count": int(len(frame)),
        "order_count": int(order_weights.sum()),
        "merchant_mean_distance_r2_equal_weighted": _r_squared(observed, predicted),
        "merchant_mean_distance_r2_order_weighted": _r_squared(
            observed, predicted, order_weights
        ),
        "absolute_mean_error_m_median": float(frame["absolute_mean_error_m"].median()),
        "absolute_mean_error_m_p90": float(frame["absolute_mean_error_m"].quantile(0.9)),
        "wasserstein_m_median": float(frame["wasserstein_m"].median()),
        "wasserstein_m_p90": float(frame["wasserstein_m"].quantile(0.9)),
    }
    return frame, summary


def merchant_spatial_fit(
    parameters: Iterable[float],
    candidates: CandidateCSR,
    observations: pd.DataFrame,
    scope: str,
) -> tuple[pd.DataFrame, dict]:
    """Compute Tanner merchant-level spatial fit."""
    return _merchant_spatial_fit_feature(
        parameters, candidates, observations, scope, utility_features
    )


def _building_order_fit_feature(
    parameters: Iterable[float],
    candidates: CandidateCSR,
    observations: pd.DataFrame,
    feature_factory,
) -> tuple[pd.DataFrame, dict]:
    """Aggregate exact observed and expected orders over all candidate buildings."""
    theta = np.asarray(tuple(parameters), dtype=float)
    grouped = _group_observations(observations)
    if not grouped:
        raise ValueError("Building-order fit requires at least one merchant")
    maximum_building_index = int(np.max(candidates.building_index))
    observed = np.zeros(maximum_building_index + 1, dtype=float)
    expected = np.zeros(maximum_building_index + 1, dtype=float)
    included = np.zeros(maximum_building_index + 1, dtype=bool)
    for shop_index, (positions, counts) in grouped.items():
        start, stop = candidates.bounds(shop_index)
        building_indices = np.asarray(candidates.building_index[start:stop])
        utilities = feature_factory(candidates, start, stop) @ theta
        probabilities = np.exp(utilities - logsumexp(utilities))
        included[building_indices] = True
        expected[building_indices] += probabilities * float(counts.sum())
        observed_buildings = np.asarray(candidates.building_index[positions])
        np.add.at(observed, observed_buildings, counts)
    building_indices = np.flatnonzero(included)
    frame = pd.DataFrame(
        {
            "building_index": building_indices,
            "observed_orders": observed[building_indices],
            "expected_orders": expected[building_indices],
        }
    )
    raw_r2 = _r_squared(
        frame["observed_orders"].to_numpy(float),
        frame["expected_orders"].to_numpy(float),
    )
    log_r2 = _r_squared(
        np.log1p(frame["observed_orders"].to_numpy(float)),
        np.log1p(frame["expected_orders"].to_numpy(float)),
    )
    summary = {
        "candidate_building_count": int(len(frame)),
        "observed_positive_building_count": int((frame["observed_orders"] > 0).sum()),
        "observed_order_count": int(frame["observed_orders"].sum()),
        "expected_order_count": float(frame["expected_orders"].sum()),
        "building_order_count_r2_raw": raw_r2,
        "building_order_count_r2_log1p": log_r2,
    }
    return frame, summary


def building_order_fit(
    parameters: Iterable[float],
    candidates: CandidateCSR,
    observations: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:
    """Aggregate Tanner observed and expected orders over candidate buildings."""
    return _building_order_fit_feature(
        parameters, candidates, observations, utility_features
    )
