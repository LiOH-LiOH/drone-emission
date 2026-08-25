"""Narrow synthetic tests for Beijing joint gravity calibration."""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd
import pytest
from scipy.special import softmax

from drone_emission.calibration.gravity_calibration import (
    accept_calibration_result,
    assign_evaluation_split,
    build_candidate_cache,
    match_orders_to_buildings,
    sha256_file,
    stable_shop_is_unseen,
)
from drone_emission.calibration.gravity_model import (
    building_order_fit,
    CandidateCSR,
    evaluate_model,
    fit_parameters,
    merchant_spatial_fit,
    objective_gradient_hessian,
)


def _write_candidates(tmp_path, shops=18, candidates_per_shop=12):
    rng = np.random.default_rng(41)
    length = shops * candidates_per_shop
    indptr = np.arange(0, length + 1, candidates_per_shop, dtype=np.int64)
    distance = rng.uniform(0.08, 4.9, size=length)
    volume = np.exp(rng.normal(9.0, 1.1, size=length))
    arrays = {
        "candidate_indptr.npy": indptr,
        "candidate_building_index.npy": np.arange(length, dtype=np.int64),
        "candidate_log_volume.npy": np.log(volume),
        "candidate_log_distance.npy": np.log(distance),
        "candidate_distance_km.npy": distance,
    }
    for name, values in arrays.items():
        np.save(tmp_path / name, values)
    return CandidateCSR.load(tmp_path, mmap_mode=None)


def _synthetic_observations(candidates, theta=np.array([0.8, 1.25, 0.65])):
    rng = np.random.default_rng(19)
    rows = []
    for shop_index in range(candidates.shop_count):
        start, stop = candidates.bounds(shop_index)
        features = np.column_stack(
            (
                candidates.log_volume[start:stop],
                candidates.log_distance[start:stop],
                -candidates.distance_km[start:stop],
            )
        )
        counts = rng.multinomial(8000, softmax(features @ theta))
        rows.extend(
            {
                "shop_index": shop_index,
                "candidate_position": start + local,
                "count": int(count),
            }
            for local, count in enumerate(counts) if count
        )
    return pd.DataFrame(rows)


def test_joint_likelihood_recovers_parameters_and_derivatives(tmp_path):
    candidates = _write_candidates(tmp_path)
    truth = np.array([0.8, 1.25, 0.65])
    observations = _synthetic_observations(candidates, truth)
    point = np.array([0.7, 1.0, 0.9])
    value, gradient, hessian = objective_gradient_hessian(
        point, candidates, observations
    )
    assert np.isfinite(value)
    step = 1e-5
    numerical_gradient = np.array([
        (
            objective_gradient_hessian(
                point + np.eye(3)[index] * step, candidates, observations, False
            )[0]
            - objective_gradient_hessian(
                point - np.eye(3)[index] * step, candidates, observations, False
            )[0]
        ) / (2 * step)
        for index in range(3)
    ])
    assert gradient == pytest.approx(numerical_gradient, rel=2e-5, abs=2e-5)
    assert np.linalg.eigvalsh(hessian).min() > 0
    fitted = fit_parameters(candidates, observations, initial=(0.5, 0.8, 1.0))
    assert fitted["success"]
    assert list(fitted["parameters"].values()) == pytest.approx(truth, abs=0.04)


def test_logsumexp_stays_finite_for_extreme_utilities(tmp_path):
    candidates = _write_candidates(tmp_path, shops=2, candidates_per_shop=4)
    observations = _synthetic_observations(candidates, np.array([0.5, 0.5, 0.5]))
    value, gradient, hessian = objective_gradient_hessian(
        np.array([1e4, 1e4, 1e4]), candidates, observations
    )
    assert np.isfinite(value)
    assert np.isfinite(gradient).all()
    assert np.isfinite(hessian).all()


def test_shop_and_time_splits_are_reproducible_and_disjoint():
    unseen = next(str(value) for value in range(1000) if stable_shop_is_unseen(value))
    seen = next(str(value) for value in range(1000) if not stable_shop_is_unseen(value))
    dates = pd.date_range("2020-02-01", periods=29, freq="D")
    orders = pd.DataFrame(
        {
            "shop_id": np.repeat([seen, unseen], len(dates)),
            "date": list(dates) * 2,
            "source_file": ["unused"] * (2 * len(dates)),
        }
    )
    labels = assign_evaluation_split(orders)
    assert labels.equals(assign_evaluation_split(orders.copy()))
    assert set(labels[orders["shop_id"] == unseen]) == {"test_unseen_shops_all_29_days"}
    seen_labels = labels[orders["shop_id"] == seen]
    assert (seen_labels == "train_seen_first_24_days").sum() == 24
    assert (seen_labels == "test_seen_shops_last_5_days").sum() == 5


def test_contains_precedes_nearest_and_ties_are_stable():
    geopandas = pytest.importorskip("geopandas")
    from shapely.geometry import Polygon

    buildings = geopandas.GeoDataFrame(
        {
            "building_index": [0, 1],
            "stable_building_id": ["b", "a"],
            "centroid_x": [5.0, 5.0],
            "centroid_y": [5.0, 5.0],
            "valid_volume": [False, True],
        },
        geometry=[
            Polygon([(0, 0), (10, 0), (10, 10), (0, 10)]),
            Polygon([(0, 0), (10, 0), (10, 10), (0, 10)]),
        ],
        crs="EPSG:32650",
    )
    longitude, latitude = geopandas.GeoSeries(
        geopandas.points_from_xy([5.0], [5.0]), crs="EPSG:32650"
    ).to_crs("EPSG:4326").iloc[0].coords[0]
    orders = pd.DataFrame(
        {"_order_index": [0], "deliver_lng_wgs84": [longitude], "deliver_lat_wgs84": [latitude]}
    )
    matched = match_orders_to_buildings(orders, buildings)
    assert matched.loc[0, "match_method"] == "contains"
    assert matched.loc[0, "stable_building_id"] == "a"
    assert bool(matched.loc[0, "valid_volume"])


def test_candidate_cache_preserves_noncontiguous_building_ids(tmp_path):
    shops = pd.DataFrame(
        {"pick_lat_wgs84": [39.9], "pick_lng_wgs84": [116.4]}
    )
    buildings = pd.DataFrame(
        {
            "building_index": [3, 9],
            "centroid_lat_wgs84": [39.901, 39.902],
            "centroid_lng_wgs84": [116.401, 116.402],
            "volume_m3": [100.0, 200.0],
        }
    )
    build_candidate_cache(shops, buildings, tmp_path)
    candidates = CandidateCSR.load(tmp_path, mmap_mode=None)
    assert set(candidates.building_index) == {3, 9}


def test_all_models_use_identical_observation_count(tmp_path):
    candidates = _write_candidates(tmp_path, shops=4, candidates_per_shop=6)
    observations = _synthetic_observations(candidates)
    counts = []
    for parameters in (
        (1.7, 1.1131, 1.4115), (0, 0, 0), (1, 0, 0), (0, 1.1131, 1.4115), (0.8, 1.25, 0.65)
    ):
        metrics, _ = evaluate_model(parameters, candidates, observations)
        counts.append(metrics["order_count"])
        assert 0 <= metrics["top_1"] <= metrics["top_10"] <= 1
        assert np.isfinite(metrics["nll_per_order"])
    assert len(set(counts)) == 1


def test_spatial_fit_uses_exact_expectations_and_keeps_observed_zeros():
    candidates = CandidateCSR(
        indptr=np.array([0, 2, 4], dtype=np.int64),
        building_index=np.array([0, 1, 1, 2], dtype=np.int64),
        log_volume=np.zeros(4),
        log_distance=np.log(np.array([1.0, 2.0, 1.0, 3.0])),
        distance_km=np.array([1.0, 2.0, 1.0, 3.0]),
    )
    observations = pd.DataFrame(
        {
            "shop_index": [0, 1],
            "candidate_position": [0, 3],
            "count": [10, 10],
        }
    )
    merchants, merchant_summary = merchant_spatial_fit(
        (0.0, 0.0, 0.0), candidates, observations, "full_valid_sample"
    )
    assert merchant_summary["merchant_mean_distance_r2_equal_weighted"] == pytest.approx(0.375)
    assert merchants["wasserstein_m"].to_list() == pytest.approx([500.0, 1000.0])
    buildings, building_summary = building_order_fit(
        (0.0, 0.0, 0.0), candidates, observations
    )
    assert buildings["building_index"].to_list() == [0, 1, 2]
    assert buildings["observed_orders"].to_list() == pytest.approx([10.0, 0.0, 10.0])
    assert buildings["expected_orders"].to_list() == pytest.approx([5.0, 10.0, 5.0])
    assert building_summary["building_order_count_r2_raw"] == pytest.approx(-1.25)
    assert building_summary["expected_order_count"] == pytest.approx(20.0)


def test_acceptance_requires_success_and_preserves_default_config(tmp_path):
    input_path = tmp_path / "orders.csv.gz"
    input_path.write_bytes(b"input")
    buildings_path = tmp_path / "buildings.gpkg"
    buildings_path.write_bytes(b"building")
    transform_path = tmp_path / "transform.json"
    transform_path.write_text("{}")
    config = tmp_path / "experiment.toml"
    original = "[od_params]\nalpha = 1.7\ntanner_a = 1.1131\ntanner_b = 1.4115\n"
    config.write_text(original)
    fingerprints = {
        label: {"path": str(path), "sha256": sha256_file(path)}
        for label, path in {
            "corrected_orders": input_path,
            "buildings": buildings_path,
            "coordinate_transform": transform_path,
        }.items()
    }
    result = {
        "schema_version": 1,
        "run_hash": "abc",
        "input_fingerprints": fingerprints,
        "final_fit": {
            "success": False,
            "parameters": {"alpha": 0.9, "beta": 1.2, "gamma": 0.7},
        },
        "clustered_inference": {},
    }
    result_path = tmp_path / "result.json"
    result_path.write_text(json.dumps(result))
    with pytest.raises(ValueError, match="did not converge"):
        accept_calibration_result(result_path, config, tmp_path / "summary.json")
    assert config.read_text() == original
    result["final_fit"]["success"] = True
    metric_template = {
        "split": "test", "model": "model", "order_count": 1,
        "nll_per_order": 1.0, "perplexity": 1.0, "mcfadden_pseudo_r2": 0.0,
        "delta_nll_vs_current": 0.0, "top_1": 0.0, "top_10": 0.0,
        "top_1_percent": 0.0, "ks": 0.0, "wasserstein_1_km": 0.0,
        "observed_mean_km": 1.0, "observed_median_km": 1.0,
        "observed_p95_km": 1.0, "predicted_mean_km": 1.0,
        "predicted_median_km": 1.0, "predicted_p95_km": 1.0,
    }
    result["metrics"] = [dict(metric_template) for _ in range(10)]
    result["clustered_inference"] = {"parameters": [{}, {}, {}]}
    result_path.write_text(json.dumps(result))
    accept_calibration_result(result_path, config, tmp_path / "summary.json")
    assert "alpha = 0.9" in config.read_text()
    assert (tmp_path / "summary.json").is_file()
