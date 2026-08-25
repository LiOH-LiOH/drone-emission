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


def evaluate_model(
    parameters: Iterable[float],
    candidates: CandidateCSR,
    observations: pd.DataFrame,
) -> tuple[dict, pd.DataFrame]:
    """Evaluate likelihood, ranks, and exact predicted distance mixture."""
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
        features = utility_features(candidates, start, stop)
        utilities = features @ theta
        logz = float(logsumexp(utilities))
        probabilities = np.exp(utilities - logz)
        local = positions - start
        nll += float(np.sum(counts * (logz - utilities[local])))
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
        "top_1": float(top1 / total_orders),
        "top_10": float(top10 / total_orders),
        "top_1_percent": float(top1pct / total_orders),
        **_distance_metrics(
            observed_distances, observed_weights, predicted_distances, predicted_weights
        ),
    }
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
    return metrics, observed_rows.merge(
        predicted_rows, on="distance_bin_km", how="outer"
    ).fillna(0.0)
