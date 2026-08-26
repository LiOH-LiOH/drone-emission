"""Box-Cox distance-decay challenger for the Beijing gravity calibration."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd

from .gravity_calibration import SCHEMA_VERSION, sha256_file
from .gravity_model import (
    CandidateCSR,
    _distance_metrics,
    building_order_fit_boxcox,
    clustered_covariance_boxcox,
    evaluate_boxcox_model,
    evaluate_model,
    fit_boxcox_parameters,
    merchant_spatial_fit_boxcox,
)


BOXCOX_SCHEMA_VERSION = 1
BOXCOX_LAMBDA_GRID = tuple(float(value) for value in np.arange(0.0, 2.01, 0.25))
CV_FOLDS = 3
TEST_SPLITS = (
    "test_unseen_shops_all_29_days",
    "test_seen_shops_last_5_days",
)


def stable_boxcox_fold(shop_id: object, folds: int = CV_FOLDS) -> int:
    """Assign one stable merchant fold for Box-Cox shape selection."""
    if folds < 2:
        raise ValueError("Box-Cox cross-validation requires at least two folds")
    payload = f"boxcox-cv:{shop_id}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % folds


def _boxcox_parameter_vector(payload: dict) -> np.ndarray:
    parameters = payload["parameters"]
    return np.array([parameters["alpha"], parameters["boxcox_gamma"]], dtype=float)


def _tanner_parameter_vector(payload: dict) -> np.ndarray:
    parameters = payload["parameters"]
    return np.array(
        [parameters["alpha"], parameters["beta"], parameters["gamma"]],
        dtype=float,
    )


def _split_observations(observations: pd.DataFrame, split: str) -> pd.DataFrame:
    selected = observations.loc[observations["split"] == split]
    return selected[["shop_index", "candidate_position", "count"]].copy()


def _pooled_distance_metrics(components: list[dict]) -> dict:
    return _distance_metrics(
        np.concatenate([item["observed_values"] for item in components]),
        np.concatenate([item["observed_weights"] for item in components]),
        np.concatenate([item["predicted_values"] for item in components]),
        np.concatenate([item["predicted_weights"] for item in components]),
    )


def _cross_validate_shape(
    candidates: CandidateCSR,
    training: pd.DataFrame,
    shop_folds: dict[int, int],
    checkpoint_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    fold_rows = []
    grid_rows = []
    warm_starts = {fold: np.array([0.4, 1.5], dtype=float) for fold in range(CV_FOLDS)}
    training_folds = training["shop_index"].map(shop_folds)
    if training_folds.isna().any():
        raise ValueError("Every training merchant must have a stable Box-Cox fold")
    for shape in BOXCOX_LAMBDA_GRID:
        components = []
        total_nll = 0.0
        total_orders = 0
        for fold in range(CV_FOLDS):
            fit_observations = training.loc[training_folds != fold]
            validation = training.loc[training_folds == fold]
            if fit_observations.empty or validation.empty:
                raise ValueError(f"Box-Cox CV fold {fold} is empty")
            checkpoint = checkpoint_dir / f"lambda_{shape:.2f}_fold_{fold}.json"
            if checkpoint.is_file():
                fit = json.loads(checkpoint.read_text(encoding="utf-8"))
                print(
                    f"Restored Box-Cox lambda={shape:g}, fold={fold}",
                    flush=True,
                )
            else:
                print(f"Fitting Box-Cox lambda={shape:g}, fold={fold}", flush=True)
                fit = fit_boxcox_parameters(
                    candidates,
                    fit_observations,
                    shape,
                    initial=warm_starts[fold],
                )
                temporary = checkpoint.with_suffix(".json.tmp")
                temporary.write_text(
                    json.dumps(fit, indent=2, ensure_ascii=False) + "\n",
                    encoding="utf-8",
                )
                temporary.replace(checkpoint)
            if not fit["success"]:
                raise RuntimeError(
                    f"Box-Cox fit failed for lambda={shape:g}, fold={fold}: "
                    f"{fit['message']}"
                )
            warm_starts[fold] = _boxcox_parameter_vector(fit)
            metrics, _, component = evaluate_boxcox_model(
                warm_starts[fold],
                candidates,
                validation,
                shape,
                return_components=True,
                include_ranks=False,
            )
            components.append(component)
            total_nll += metrics["negative_log_likelihood"]
            total_orders += metrics["order_count"]
            fold_rows.append({
                "boxcox_lambda": shape,
                "fold": fold,
                "fit_merchant_count": int(fit_observations["shop_index"].nunique()),
                "validation_merchant_count": int(validation["shop_index"].nunique()),
                "validation_order_count": metrics["order_count"],
                "alpha": fit["parameters"]["alpha"],
                "boxcox_gamma": fit["parameters"]["boxcox_gamma"],
                "nll_per_order": metrics["nll_per_order"],
                "wasserstein_1_km": metrics["wasserstein_1_km"],
                "observed_mean_km": metrics["observed_mean_km"],
                "predicted_mean_km": metrics["predicted_mean_km"],
                "observed_p95_km": metrics["observed_p95_km"],
                "predicted_p95_km": metrics["predicted_p95_km"],
            })
        pooled = _pooled_distance_metrics(components)
        grid_rows.append({
            "boxcox_lambda": shape,
            "fold_count": CV_FOLDS,
            "validation_order_count": total_orders,
            "nll_per_order": total_nll / total_orders,
            **pooled,
        })
    folds = pd.DataFrame(fold_rows)
    grid = pd.DataFrame(grid_rows)
    selected = min(
        grid.itertuples(index=False),
        key=lambda row: (
            row.wasserstein_1_km,
            row.nll_per_order,
            abs(row.boxcox_lambda - 1.0),
            row.boxcox_lambda,
        ),
    )
    grid["selected"] = grid["boxcox_lambda"] == selected.boxcox_lambda
    return folds, grid, float(selected.boxcox_lambda)


def _evaluate_tests(
    candidates: CandidateCSR,
    observations: pd.DataFrame,
    tanner_parameters: np.ndarray,
    boxcox_parameters: np.ndarray,
    shape: float,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    metric_rows = []
    distribution_frames = []
    for split in TEST_SPLITS:
        test = _split_observations(observations, split)
        for model, evaluator, parameters in (
            ("tanner", evaluate_model, tanner_parameters),
            ("boxcox", evaluate_boxcox_model, boxcox_parameters),
        ):
            if model == "boxcox":
                metrics, distribution = evaluator(
                    parameters, candidates, test, shape, include_ranks=False
                )
            else:
                metrics, distribution = evaluator(
                    parameters, candidates, test, include_ranks=False
                )
            metric_rows.append({"split": split, "model": model, **metrics})
            distribution.insert(0, "model", model)
            distribution.insert(0, "split", split)
            distribution_frames.append(distribution)
    combined = pd.concat(
        [_split_observations(observations, split) for split in TEST_SPLITS],
        ignore_index=True,
    )
    tanner_combined, _ = evaluate_model(
        tanner_parameters, candidates, combined, include_ranks=False
    )
    boxcox_combined, _ = evaluate_boxcox_model(
        boxcox_parameters, candidates, combined, shape, include_ranks=False
    )
    combined_metrics = {"tanner": tanner_combined, "boxcox": boxcox_combined}
    metric_rows.extend(
        {"split": "combined_tests", "model": model, **metrics}
        for model, metrics in combined_metrics.items()
    )
    return (
        pd.DataFrame(metric_rows),
        pd.concat(distribution_frames, ignore_index=True),
        combined_metrics,
    )


def _evaluate_full_sample(
    candidates: CandidateCSR,
    observations: pd.DataFrame,
    tanner_parameters: np.ndarray,
    boxcox_parameters: np.ndarray,
    shape: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    full = observations.groupby(
        ["shop_index", "candidate_position"], as_index=False
    )["count"].sum()
    metric_rows = []
    distribution_frames = []
    for model, evaluator, parameters in (
        ("tanner", evaluate_model, tanner_parameters),
        ("boxcox", evaluate_boxcox_model, boxcox_parameters),
    ):
        if model == "boxcox":
            metrics, distribution = evaluator(
                parameters, candidates, full, shape, include_ranks=False
            )
        else:
            metrics, distribution = evaluator(
                parameters, candidates, full, include_ranks=False
            )
        metric_rows.append({"scope": "full_valid_sample", "model": model, **metrics})
        distribution.insert(0, "model", model)
        distribution.insert(0, "scope", "full_valid_sample")
        distribution_frames.append(distribution)
    return pd.DataFrame(metric_rows), pd.concat(distribution_frames, ignore_index=True)


def boxcox_acceptance_gate(
    test_metrics: pd.DataFrame,
    combined_metrics: dict,
) -> dict:
    """Apply the pre-registered aggregate-distance challenger gate."""
    tanner = combined_metrics["tanner"]
    boxcox = combined_metrics["boxcox"]
    tolerance = 1e-12
    split_checks = {}
    for split in TEST_SPLITS:
        indexed = test_metrics.loc[test_metrics["split"] == split].set_index("model")
        split_checks[split] = bool(
            indexed.loc["boxcox", "wasserstein_1_km"]
            <= indexed.loc["tanner", "wasserstein_1_km"] + tolerance
        )
    checks = {
        "combined_wasserstein_improves_at_least_10_percent": bool(
            boxcox["wasserstein_1_km"]
            <= 0.9 * tanner["wasserstein_1_km"] + tolerance
        ),
        "each_test_split_wasserstein_not_worse": bool(all(split_checks.values())),
        "combined_mean_absolute_bias_not_worse": bool(
            abs(boxcox["predicted_mean_km"] - boxcox["observed_mean_km"])
            <= abs(tanner["predicted_mean_km"] - tanner["observed_mean_km"])
            + tolerance
        ),
        "combined_p95_absolute_bias_not_worse": bool(
            abs(boxcox["predicted_p95_km"] - boxcox["observed_p95_km"])
            <= abs(tanner["predicted_p95_km"] - tanner["observed_p95_km"])
            + tolerance
        ),
        "combined_nll_per_order_not_worse_by_more_than_1_percent": bool(
            boxcox["nll_per_order"] <= 1.01 * tanner["nll_per_order"] + tolerance
        ),
    }
    return {
        "eligible_for_acceptance": bool(all(checks.values())),
        "checks": checks,
        "split_wasserstein_not_worse": split_checks,
        "combined_wasserstein_improvement_fraction": float(
            1.0 - boxcox["wasserstein_1_km"] / tanner["wasserstein_1_km"]
        ),
        "thresholds": {
            "minimum_combined_wasserstein_improvement_fraction": 0.10,
            "maximum_nll_per_order_ratio": 1.01,
        },
    }


def _write_spatial_sources(
    output_dir: Path,
    candidates: CandidateCSR,
    observations: pd.DataFrame,
    training_parameters: np.ndarray,
    final_parameters: np.ndarray,
    shape: float,
) -> dict:
    full = observations.groupby(
        ["shop_index", "candidate_position"], as_index=False
    )["count"].sum()
    merchant_frames = []
    merchant_summaries = []
    for scope, selected, parameters in (
        ("full_valid_sample", full, final_parameters),
        *(
            (split, _split_observations(observations, split), training_parameters)
            for split in TEST_SPLITS
        ),
    ):
        frame, summary = merchant_spatial_fit_boxcox(
            parameters, candidates, selected, scope, shape
        )
        merchant_frames.append(frame)
        merchant_summaries.append(summary)
    pd.concat(merchant_frames, ignore_index=True).to_csv(
        output_dir / "boxcox_merchant_spatial_fit_source.csv.gz",
        index=False,
        compression="gzip",
    )
    buildings, building_summary = building_order_fit_boxcox(
        final_parameters, candidates, full, shape
    )
    buildings.to_csv(
        output_dir / "boxcox_building_order_fit_source.csv.gz",
        index=False,
        compression="gzip",
    )
    summary = pd.DataFrame(merchant_summaries)
    for key, value in building_summary.items():
        summary.loc[summary["scope"] == "full_valid_sample", key] = value
    summary.to_csv(output_dir / "boxcox_spatial_fit_metrics.csv", index=False)
    return {
        "merchant_metrics": merchant_summaries,
        "building_metrics": building_summary,
    }


def run_boxcox_challenger(
    calibration_result_path: Path,
    output_root: Path,
    memory_limit_gb: float = 64.0,
) -> dict:
    """Select, evaluate, and fully refit the Box-Cox gravity challenger."""
    calibration_result_path = Path(calibration_result_path).resolve()
    source = json.loads(calibration_result_path.read_text(encoding="utf-8"))
    if source.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported source calibration schema")
    if not source.get("final_fit", {}).get("success"):
        raise ValueError("Box-Cox challenger requires a converged source calibration")
    source_hash = sha256_file(calibration_result_path)
    challenge_hash = hashlib.sha256(
        json.dumps({
            "source_sha256": source_hash,
            "lambda_grid": BOXCOX_LAMBDA_GRID,
            "cv_folds": CV_FOLDS,
        }, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]
    output_root = Path(output_root)
    output_dir = (
        output_root
        if output_root.name == challenge_hash
        else output_root / challenge_hash
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "boxcox_challenger.json"
    if result_path.is_file():
        restored = json.loads(result_path.read_text(encoding="utf-8"))
        if (
            restored.get("challenge_hash") == challenge_hash
            and restored.get("source_calibration", {}).get("sha256") == source_hash
            and restored.get("final_fit", {}).get("success")
        ):
            print(f"Restored completed Box-Cox result from {result_path}", flush=True)
            return restored
    cache_dir = calibration_result_path.parent / "cache" / source["run_hash"]
    candidates = CandidateCSR.load(cache_dir)
    observations = pd.read_csv(cache_dir / "observations.csv.gz")
    candidate_pairs = int(len(candidates.building_index))
    # Exact pooled W1 temporarily materializes several values, weights, and
    # sorting arrays in addition to the memory-mapped candidate attributes.
    estimated_peak_gb = max(12.0, candidate_pairs * 128.0 / 2**30 + 4.0)
    if estimated_peak_gb >= memory_limit_gb:
        raise MemoryError(
            f"Estimated Box-Cox peak {estimated_peak_gb:.1f} GB is not below "
            f"the {memory_limit_gb:.1f} GB task limit"
        )
    shops = pd.read_csv(cache_dir / "shops.csv", dtype={"shop_id": "string"})
    shop_folds = {
        int(row.shop_index): stable_boxcox_fold(row.shop_id)
        for row in shops.itertuples(index=False)
    }
    training = _split_observations(observations, "train_seen_first_24_days")
    cv_folds, cv_grid, selected_shape = _cross_validate_shape(
        candidates,
        training,
        shop_folds,
        output_dir / ".boxcox_cv_checkpoints",
    )
    cv_folds.to_csv(output_dir / "boxcox_cv_fold_metrics.csv", index=False)
    cv_grid.to_csv(output_dir / "boxcox_cv_grid_metrics.csv", index=False)
    selected_row = cv_folds.loc[cv_folds["boxcox_lambda"] == selected_shape]
    initial = selected_row[["alpha", "boxcox_gamma"]].median().to_numpy(float)
    training_fit = fit_boxcox_parameters(
        candidates, training, selected_shape, initial=initial
    )
    if not training_fit["success"]:
        raise RuntimeError("Selected Box-Cox training fit did not converge")
    tanner_training = source["training_models"]["joint_three_parameter"]
    test_metrics, distributions, combined_metrics = _evaluate_tests(
        candidates,
        observations,
        _tanner_parameter_vector(tanner_training),
        _boxcox_parameter_vector(training_fit),
        selected_shape,
    )
    test_metrics.to_csv(output_dir / "boxcox_test_metrics.csv", index=False)
    distributions.to_csv(
        output_dir / "boxcox_test_distance_source.csv.gz",
        index=False,
        compression="gzip",
    )
    gate = boxcox_acceptance_gate(test_metrics, combined_metrics)
    gate_rows = [
        {"check": name, "passed": passed}
        for name, passed in gate["checks"].items()
    ]
    gate_rows.extend(
        {
            "check": f"{split}:wasserstein_not_worse",
            "passed": passed,
        }
        for split, passed in gate["split_wasserstein_not_worse"].items()
    )
    pd.DataFrame(gate_rows).to_csv(
        output_dir / "boxcox_acceptance_gate.csv", index=False
    )
    full = observations.groupby(
        ["shop_index", "candidate_position"], as_index=False
    )["count"].sum()
    final_fit = fit_boxcox_parameters(
        candidates,
        full,
        selected_shape,
        initial=_boxcox_parameter_vector(training_fit),
    )
    if not final_fit["success"]:
        raise RuntimeError("Full-sample Box-Cox fit did not converge")
    inference = clustered_covariance_boxcox(
        _boxcox_parameter_vector(final_fit), candidates, full, selected_shape
    )
    full_metrics, full_distributions = _evaluate_full_sample(
        candidates,
        observations,
        _tanner_parameter_vector(source["final_fit"]),
        _boxcox_parameter_vector(final_fit),
        selected_shape,
    )
    full_metrics.to_csv(output_dir / "boxcox_full_sample_metrics.csv", index=False)
    full_distributions.to_csv(
        output_dir / "boxcox_full_sample_distance_source.csv.gz",
        index=False,
        compression="gzip",
    )
    parameter_rows = [
        {
            "parameter": "boxcox_lambda",
            "estimate": selected_shape,
            "standard_error": np.nan,
            "ci95_low": np.nan,
            "ci95_high": np.nan,
            "inference": "selected_by_grouped_cross_validation",
        },
        *({**row, "inference": "conditional_on_selected_lambda"}
          for row in inference["parameters"]),
    ]
    pd.DataFrame(parameter_rows).to_csv(
        output_dir / "boxcox_parameter_estimates.csv", index=False
    )
    spatial_fit = _write_spatial_sources(
        output_dir,
        candidates,
        observations,
        _boxcox_parameter_vector(training_fit),
        _boxcox_parameter_vector(final_fit),
        selected_shape,
    )
    from .gravity_plots import render_boxcox_challenger_audit

    result = {
        "schema_version": BOXCOX_SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "candidate",
        "model_family": "boxcox",
        "challenge_hash": challenge_hash,
        "output_dir": str(output_dir.resolve()),
        "source_calibration": {
            "path": str(calibration_result_path),
            "sha256": source_hash,
            "run_hash": source["run_hash"],
        },
        "input_fingerprints": source["input_fingerprints"],
        "memory": {
            "limit_gb": float(memory_limit_gb),
            "estimated_peak_gb": estimated_peak_gb,
            "estimate_basis": "128 bytes per candidate pair plus 4 GiB fixed",
        },
        "selection": {
            "lambda_grid": list(BOXCOX_LAMBDA_GRID),
            "fold_count": CV_FOLDS,
            "fold_assignment": "sha256_boxcox_cv_shop_id_mod_3",
            "primary_metric": "pooled_order_weighted_wasserstein_1_km",
            "selected_lambda": selected_shape,
            "tie_breakers": [
                "lower_nll_per_order",
                "closer_to_lambda_1",
                "smaller_lambda",
            ],
        },
        "training_fit": training_fit,
        "test_metrics": test_metrics.to_dict(orient="records"),
        "combined_test_metrics": combined_metrics,
        "acceptance": gate,
        "final_fit": final_fit,
        "full_sample_metrics": full_metrics.to_dict(orient="records"),
        "clustered_inference": inference,
        "spatial_fit": spatial_fit,
        "limitations": [
            "The selected Box-Cox shape is global only over the pre-registered finite grid.",
            "Merchant-level heterogeneity is reported but is not an acceptance target.",
        ],
    }
    result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    render_boxcox_challenger_audit(output_dir, result)
    result["artifacts"] = {
        path.name: {"sha256": sha256_file(path), "bytes": path.stat().st_size}
        for path in sorted(output_dir.iterdir())
        if path.is_file() and path.name != result_path.name
    }
    result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    return result


def accept_boxcox_result(
    result_path: Path,
    experiment_path: Path,
    summary_path: Path,
) -> dict:
    """Accept an eligible Box-Cox challenger into tracked configuration."""
    result_path, experiment_path, summary_path = map(
        Path, (result_path, experiment_path, summary_path)
    )
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if result.get("schema_version") != BOXCOX_SCHEMA_VERSION:
        raise ValueError("Unsupported Box-Cox challenger schema")
    if not result.get("acceptance", {}).get("eligible_for_acceptance"):
        raise ValueError("Box-Cox challenger did not pass the pre-registered gate")
    if not result.get("final_fit", {}).get("success"):
        raise ValueError("Cannot accept a Box-Cox fit that did not converge")
    source = result["source_calibration"]
    if sha256_file(Path(source["path"])) != source["sha256"]:
        raise ValueError("Source gravity calibration changed after Box-Cox evaluation")
    for label, record in result["input_fingerprints"].items():
        path = Path(record["path"])
        if not path.is_file() or sha256_file(path) != record["sha256"]:
            raise ValueError(f"Input fingerprint changed since calibration: {label}")
    parameters = result["final_fit"]["parameters"]
    alpha = float(parameters["alpha"])
    shape = float(parameters["boxcox_lambda"])
    gamma = float(parameters["boxcox_gamma"])
    if not np.isfinite([alpha, shape, gamma]).all() or alpha < 0 or gamma <= 0:
        raise ValueError("Box-Cox parameters violate model constraints")
    text = experiment_path.read_text(encoding="utf-8")
    text, count = re.subn(
        r'(?m)^(\s*impedance_type\s*=\s*)"[^"]+"',
        r'\g<1>"boxcox"',
        text,
        count=1,
    )
    if count != 1:
        raise ValueError("Could not uniquely update impedance_type in experiment.toml")
    text, count = re.subn(
        r"(?m)^(\s*alpha\s*=\s*)[^#\n]+",
        rf"\g<1>{alpha:.10g} ",
        text,
        count=1,
    )
    if count != 1:
        raise ValueError("Could not uniquely update alpha in experiment.toml")
    for key, value in (("boxcox_lambda", shape), ("boxcox_gamma", gamma)):
        pattern = rf"(?m)^(\s*{key}\s*=\s*)[^#\n]+"
        replacement = rf"\g<1>{value:.10g} "
        text, count = re.subn(pattern, replacement, text, count=1)
        if count == 0:
            anchor = re.search(r"(?m)^\s*alpha\s*=.*$", text)
            if anchor is None:
                raise ValueError(f"Could not insert {key} in experiment.toml")
            text = text[:anchor.end()] + f"\n{key} = {value:.10g}" + text[anchor.end():]
    experiment_path.write_text(text, encoding="utf-8")
    summary = {
        "schema_version": SCHEMA_VERSION,
        "accepted_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_result_sha256": sha256_file(result_path),
        "run_hash": result["source_calibration"]["run_hash"],
        "model_family": "boxcox",
        "input_fingerprints": result["input_fingerprints"],
        "parameters": parameters,
        "acceptance": result["acceptance"],
        "clustered_inference": result["clustered_inference"],
        "final_fit": result["final_fit"],
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    return summary
