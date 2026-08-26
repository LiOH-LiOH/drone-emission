"""Narrow tests for the Box-Cox gravity challenger and production weights."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from scipy.special import softmax

from drone_emission.calibration.boxcox_calibration import (
    TEST_SPLITS,
    accept_boxcox_result,
    boxcox_acceptance_gate,
    stable_boxcox_fold,
)
from drone_emission.calibration.gravity_calibration import sha256_file
from drone_emission.calibration.gravity_model import (
    CandidateCSR,
    fit_boxcox_parameters,
    objective_gradient_hessian_boxcox,
)
from drone_emission.demand.api import (
    generate_one_store_od_pairs,
    generate_one_store_od_pairs_vectorized,
)
from drone_emission.utils.gravity import (
    boxcox_gravity_log_weights,
    boxcox_transform,
)


def _candidates(shops: int = 15, candidates_per_shop: int = 10) -> CandidateCSR:
    rng = np.random.default_rng(73)
    length = shops * candidates_per_shop
    distances = rng.uniform(0.08, 4.9, size=length)
    volumes = np.exp(rng.normal(8.5, 1.0, size=length))
    return CandidateCSR(
        indptr=np.arange(0, length + 1, candidates_per_shop, dtype=np.int64),
        building_index=np.arange(length, dtype=np.int64),
        log_volume=np.log(volumes),
        log_distance=np.log(distances),
        distance_km=distances,
    )


def _observations(
    candidates: CandidateCSR,
    parameters: np.ndarray,
    shape: float,
) -> pd.DataFrame:
    rng = np.random.default_rng(91)
    rows = []
    for shop_index in range(candidates.shop_count):
        start, stop = candidates.bounds(shop_index)
        features = np.column_stack((
            candidates.log_volume[start:stop],
            -boxcox_transform(candidates.distance_km[start:stop], shape),
        ))
        counts = rng.multinomial(10000, softmax(features @ parameters))
        rows.extend(
            {
                "shop_index": shop_index,
                "candidate_position": start + local,
                "count": int(count),
            }
            for local, count in enumerate(counts)
            if count
        )
    return pd.DataFrame(rows)


def test_boxcox_limits_recover_power_and_exponential_distance_decay():
    distances = np.array([0.2, 1.0, 3.5])
    volumes = np.array([10.0, 20.0, 40.0])
    alpha, gamma = 0.6, 1.2
    assert boxcox_transform(distances, 0.0) == pytest.approx(np.log(distances))
    assert boxcox_transform(distances, 1.0) == pytest.approx(distances - 1.0)

    power = np.exp(boxcox_gravity_log_weights(
        volumes, distances, alpha, gamma, 0.0
    ))
    assert power == pytest.approx(volumes ** alpha * distances ** (-gamma))
    exponential = np.exp(boxcox_gravity_log_weights(
        volumes, distances, alpha, gamma, 1.0
    ))
    expected = volumes ** alpha * np.exp(-gamma * distances)
    assert exponential / expected == pytest.approx(np.full(3, np.exp(gamma)))


def test_fixed_shape_likelihood_is_convex_and_recovers_parameters():
    candidates = _candidates()
    truth = np.array([0.7, 1.35])
    shape = 0.75
    observations = _observations(candidates, truth, shape)
    point = np.array([0.55, 1.0])
    value, gradient, hessian = objective_gradient_hessian_boxcox(
        point, candidates, observations, shape
    )
    assert np.isfinite(value)
    step = 1e-5
    numerical = np.array([
        (
            objective_gradient_hessian_boxcox(
                point + np.eye(2)[index] * step,
                candidates,
                observations,
                shape,
                False,
            )[0]
            - objective_gradient_hessian_boxcox(
                point - np.eye(2)[index] * step,
                candidates,
                observations,
                shape,
                False,
            )[0]
        ) / (2 * step)
        for index in range(2)
    ])
    assert gradient == pytest.approx(numerical, rel=2e-5, abs=2e-5)
    assert np.linalg.eigvalsh(hessian).min() > 0
    fitted = fit_boxcox_parameters(candidates, observations, shape)
    assert fitted["success"]
    estimates = fitted["parameters"]
    assert [estimates["alpha"], estimates["boxcox_gamma"]] == pytest.approx(
        truth, abs=0.025
    )


def test_shop_fold_is_stable_and_uses_all_three_folds():
    first = [stable_boxcox_fold(str(value)) for value in range(200)]
    second = [stable_boxcox_fold(str(value)) for value in range(200)]
    assert first == second
    assert set(first) == {0, 1, 2}


def _metric_row(split: str, model: str, wasserstein: float) -> dict:
    return {"split": split, "model": model, "wasserstein_1_km": wasserstein}


def test_acceptance_gate_requires_every_aggregate_distance_check():
    test_metrics = pd.DataFrame([
        _metric_row(TEST_SPLITS[0], "tanner", 0.10),
        _metric_row(TEST_SPLITS[0], "boxcox", 0.08),
        _metric_row(TEST_SPLITS[1], "tanner", 0.12),
        _metric_row(TEST_SPLITS[1], "boxcox", 0.11),
    ])
    combined = {
        "tanner": {
            "wasserstein_1_km": 0.11,
            "observed_mean_km": 2.0,
            "predicted_mean_km": 2.1,
            "observed_p95_km": 4.0,
            "predicted_p95_km": 4.2,
            "nll_per_order": 5.0,
        },
        "boxcox": {
            "wasserstein_1_km": 0.09,
            "observed_mean_km": 2.0,
            "predicted_mean_km": 2.05,
            "observed_p95_km": 4.0,
            "predicted_p95_km": 4.1,
            "nll_per_order": 5.04,
        },
    }
    assert boxcox_acceptance_gate(test_metrics, combined)[
        "eligible_for_acceptance"
    ]
    combined["boxcox"]["nll_per_order"] = 5.051
    gate = boxcox_acceptance_gate(test_metrics, combined)
    assert not gate["eligible_for_acceptance"]
    assert not gate["checks"][
        "combined_nll_per_order_not_worse_by_more_than_1_percent"
    ]


def test_scalar_and_vectorized_od_generation_use_identical_boxcox_weights():
    shapely = pytest.importorskip("shapely.geometry")
    buildings = pd.DataFrame({
        "centroid": [
            shapely.Point(116.401, 39.901),
            shapely.Point(116.408, 39.904),
            shapely.Point(116.42, 39.91),
        ],
        "volume_m3": [1000.0, 3500.0, 1200.0],
    })
    parameters = {
        "impedance_type": "boxcox",
        "alpha": 0.55,
        "boxcox_gamma": 1.4,
        "boxcox_lambda": 0.75,
        "min_sales": 0,
    }
    scalar = generate_one_store_od_pairs(
        buildings, 100000, (39.9, 116.4), parameters
    )
    vectorized = generate_one_store_od_pairs_vectorized(
        buildings, 100000, (39.9, 116.4), parameters
    )
    assert [row["annual_sales"] for row in scalar] == [
        row["annual_sales"] for row in vectorized
    ]


def test_boxcox_acceptance_is_gated_and_retains_tanner_fields(tmp_path):
    source_path = tmp_path / "gravity_calibration.json"
    source_path.write_text('{"source": true}\n')
    input_path = tmp_path / "orders.csv.gz"
    input_path.write_bytes(b"orders")
    fingerprints = {
        "corrected_orders": {
            "path": str(input_path),
            "sha256": sha256_file(input_path),
        }
    }
    result = {
        "schema_version": 1,
        "source_calibration": {
            "path": str(source_path),
            "sha256": sha256_file(source_path),
            "run_hash": "source-run",
        },
        "input_fingerprints": fingerprints,
        "acceptance": {"eligible_for_acceptance": False},
        "final_fit": {
            "success": True,
            "parameters": {
                "alpha": 0.5,
                "boxcox_lambda": 0.75,
                "boxcox_gamma": 1.4,
            },
        },
        "clustered_inference": {},
    }
    result_path = tmp_path / "boxcox.json"
    result_path.write_text(json.dumps(result))
    config = tmp_path / "experiment.toml"
    original = (
        '[od_params]\nimpedance_type = "tanner"\nalpha = 0.42\n'
        "tanner_a = 0.32\ntanner_b = 1.65\n"
    )
    config.write_text(original)
    with pytest.raises(ValueError, match="did not pass"):
        accept_boxcox_result(result_path, config, tmp_path / "summary.json")
    assert config.read_text() == original

    result["acceptance"]["eligible_for_acceptance"] = True
    result_path.write_text(json.dumps(result))
    accept_boxcox_result(result_path, config, tmp_path / "summary.json")
    accepted = config.read_text()
    assert 'impedance_type = "boxcox"' in accepted
    assert "boxcox_lambda = 0.75" in accepted
    assert "boxcox_gamma = 1.4" in accepted
    assert "tanner_a = 0.32" in accepted
    assert "tanner_b = 1.65" in accepted
