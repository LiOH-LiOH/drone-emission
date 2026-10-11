"""Fixed-parameter, source-audited external validation on restored Meituan orders."""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .gravity_calibration import (
    _prepared_matches, estimate_peak_memory_gb, prepare_calibration_data, sha256_file,
)
from .gravity_model import CandidateCSR, evaluate_model, merchant_spatial_fit


def prepare_external_buildings(bundle: Path, tiles: list[Path], output: Path) -> dict:
    """Crop complete GBA tiles to all pickup choice sets and delivery match areas."""
    from drone_emission.buildings.load_gba import get_gba_tile_infos, load_buildings_from_gba
    from drone_emission.demand.cal_area import preprocess_buildings

    orders, _ = load_meituan_orders(Path(bundle))
    # Conservative angular bounds for a 5 km Haversine sphere, with extra margin.
    latitude_pad = np.degrees(5.1 / 6371.0)
    max_lat = orders["pick_lat_wgs84"].abs().max() + latitude_pad
    longitude_pad = latitude_pad / np.cos(np.radians(max_lat))
    bounds = {
        "min_lng": min(orders["pick_lng_wgs84"].min() - longitude_pad,
                       orders["deliver_lng_wgs84"].min() - .002),
        "max_lng": max(orders["pick_lng_wgs84"].max() + longitude_pad,
                       orders["deliver_lng_wgs84"].max() + .002),
        "min_lat": min(orders["pick_lat_wgs84"].min() - latitude_pad,
                       orders["deliver_lat_wgs84"].min() - .002),
        "max_lat": max(orders["pick_lat_wgs84"].max() + latitude_pad,
                       orders["deliver_lat_wgs84"].max() + .002),
    }
    required = {name for name, _ in get_gba_tile_infos(bounds)}
    available = {Path(path).stem for path in tiles if Path(path).is_file()}
    if not required <= available:
        raise FileNotFoundError(f"Missing complete GBA tiles: {sorted(required - available)}")
    tiles = [Path(path) for path in tiles if Path(path).stem in required]
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    raw_path = output.with_name(output.stem + "_raw.gpkg")
    bbox = tuple(bounds[name] for name in ("min_lng", "min_lat", "max_lng", "max_lat"))
    load_buildings_from_gba([str(path) for path in tiles], bbox, str(raw_path))
    saved, frame = preprocess_buildings(str(raw_path), str(output))
    metadata = {
        "schema_version": 1, "coverage": "all_pickup_5km_choice_sets_and_delivery_50m_matches",
        "bounds_wgs84": bounds, "building_count": len(frame),
        "volume_convention": "EPSG:3857_geometry_area_times_height_existing_preprocessing",
        "orders_sha256": sha256_file(Path(bundle) / "data/all_waybill_info_meituan_0322.csv.gz"),
        "buildings_sha256": sha256_file(Path(saved)),
        "tiles": [{"path": str(path.resolve()), "size_bytes": path.stat().st_size} for path in tiles],
    }
    output.with_suffix(".coverage.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


def load_meituan_orders(bundle: Path) -> tuple[pd.DataFrame, dict]:
    """Verify restoration provenance and retain exactly one accepted row per order."""
    transform_path = bundle / "processing/coordinate_transform.json"
    transform = json.loads(transform_path.read_text(encoding="utf-8"))
    source = bundle / "data/all_waybill_info_meituan_0322.csv.gz"
    relative = str(source.relative_to(bundle))
    artifacts = transform["packaged_artifacts"]
    artifact = next(row for row in artifacts if row["path"] == relative)
    if sha256_file(source) != artifact["sha256"]:
        raise ValueError("Restored Meituan orders do not match the locked bundle hash")
    columns = ["dt", "order_id", "poi_id", "is_courier_grabbed"] + [
        f"{endpoint}_{axis}_{datum}"
        for endpoint in ("sender", "recipient")
        for datum in ("gcj02", "wgs84") for axis in ("lng", "lat")
    ]
    raw = pd.read_csv(source, usecols=columns, dtype={"order_id": str, "poi_id": str})
    orders = raw.loc[raw["is_courier_grabbed"] == 1].copy()
    if orders["order_id"].duplicated().any():
        raise ValueError("Accepted order IDs must be unique")
    if len(orders) != transform["counts"]["accepted_orders"]:
        raise ValueError("Accepted order count differs from the restoration audit")
    orders["order_date"] = pd.to_datetime(orders["dt"].astype(str), format="%Y%m%d")
    dates = sorted(orders["order_date"].dt.strftime("%Y-%m-%d").unique())
    if dates != pd.date_range("2022-10-17", "2022-10-24").strftime("%Y-%m-%d").tolist():
        raise ValueError("Expected the audited eight Meituan dates")
    orders = orders.rename(columns={
        f"{old}_{axis}_{datum}": f"{new}_{axis}_{datum}"
        for old, new in (("sender", "pick"), ("recipient", "deliver"))
        for datum in ("gcj02", "wgs84") for axis in ("lng", "lat")
    })
    orders["merchant_id"] = "meituan_baoding:" + orders["poi_id"]
    # Each actual pickup location has its own choice set; never replace a pickup
    # by the merchant's first or modal coordinate (97 merchants have >1 location).
    orders["shop_id"] = orders["merchant_id"] + ":" + orders["pick_lng_wgs84"].map(
        lambda value: f"{value:.8f}"
    ) + ":" + orders["pick_lat_wgs84"].map(lambda value: f"{value:.8f}")
    orders["source_file"] = relative
    orders["split"] = orders["order_date"].dt.strftime("%Y-%m-%d")
    orders["_order_index"] = np.arange(len(orders), dtype=np.int64)
    orders = orders.reset_index(drop=True)
    return orders, {
        "raw_waybill_count": len(raw), "accepted_order_count": len(orders),
        "excluded_rejected_waybills": len(raw) - len(orders),
        "merchant_count": int(orders["merchant_id"].nunique()),
        "pickup_location_count": int(orders["shop_id"].nunique()), "dates": dates,
    }


def run_external_validation(bundle: Path, buildings: Path, parameters: Path,
                            output: Path, memory_limit_gb: float = 12.0) -> dict:
    """Evaluate accepted Beijing parameters and uniform choice; no fitting/acceptance."""
    bundle, buildings, parameters, output = map(Path, (bundle, buildings, parameters, output))
    locked = json.loads(parameters.read_text(encoding="utf-8"))
    theta = np.array([locked["parameters"][name] for name in ("alpha", "beta", "gamma")])
    if not np.isfinite(theta).all() or (theta[:2] < 0).any() or theta[2] <= 0:
        raise ValueError("Invalid locked gravity parameters")
    if not locked.get("accepted_at_utc") or not locked.get("source_result_sha256"):
        raise ValueError("Parameter input must be an accepted calibration summary")
    source = bundle / "data/all_waybill_info_meituan_0322.csv.gz"
    transform = bundle / "processing/coordinate_transform.json"
    coverage_path = buildings.with_suffix(".coverage.json")
    coverage = json.loads(coverage_path.read_text())
    if (coverage.get("coverage") != "all_pickup_5km_choice_sets_and_delivery_50m_matches"
            or coverage.get("orders_sha256") != sha256_file(source)
            or coverage.get("buildings_sha256") != sha256_file(buildings)):
        raise ValueError("Building coverage provenance does not match these orders/buildings")
    fingerprints = {
        name: {"path": str(path.resolve()), "sha256": sha256_file(path)}
        for name, path in (("corrected_orders", source), ("buildings", buildings),
                           ("coordinate_transform", transform), ("parameters", parameters),
                           ("building_coverage", coverage_path))
    }
    fingerprints["external_protocol"] = "meituan_fixed_beijing_v1_actual_pickup_locations"
    peak = estimate_peak_memory_gb(buildings, source)
    if memory_limit_gb <= peak:
        raise MemoryError(f"Estimated preparation peak {peak:.1f} GB exceeds {memory_limit_gb:.1f} GB limit")
    run_hash = hashlib.sha256(json.dumps(fingerprints, sort_keys=True).encode()).hexdigest()[:16]
    output = output / run_hash
    output.mkdir(parents=True, exist_ok=True)
    orders, order_audit = load_meituan_orders(bundle)
    cache = output / "cache"
    if _prepared_matches(cache, fingerprints):
        prepared = json.loads((cache / "prepared_manifest.json").read_text())
    else:
        prepared = prepare_calibration_data(source, buildings, transform, cache,
                                            fingerprints, workers=1, orders=orders)
    orders[["shop_id", "merchant_id"]].drop_duplicates().to_csv(
        output / "pickup_merchant_mapping.csv", index=False
    )
    candidates = CandidateCSR.load(cache)
    evaluation_peak = peak + int(candidates.indptr[-1]) * 256 / 2**30
    if evaluation_peak >= memory_limit_gb:
        raise MemoryError(
            f"Exact evaluation estimate {evaluation_peak:.1f} GB exceeds {memory_limit_gb:.1f} GB; "
            "prepared cache is preserved for a higher-memory rerun"
        )
    observations = pd.read_csv(cache / "observations.csv.gz")
    if observations.empty:
        raise ValueError("No orders survived building matching and candidate rules")
    metrics_rows, distributions = [], []
    for scope in ["all_external_orders", *order_audit["dates"]]:
        subset = observations if scope == "all_external_orders" else observations.loc[observations["split"] == scope]
        if subset.empty:
            continue
        subset = subset.groupby(["shop_index", "candidate_position"], as_index=False)["count"].sum()
        uniform_nll = None
        for model, vector in (("uniform_buildings", np.zeros(3)), ("locked_beijing", theta)):
            metrics, distribution = evaluate_model(vector, candidates, subset)
            if model == "uniform_buildings":
                uniform_nll = metrics["nll_per_order"]
            metrics_rows.append({"scope": scope, "model": model, **metrics,
                                 "mcfadden_pseudo_r2": 1 - metrics["nll_per_order"] / uniform_nll})
            distributions.append(distribution.assign(scope=scope, model=model))
    pd.DataFrame(metrics_rows).to_csv(output / "model_metrics.csv", index=False)
    pd.concat(distributions).to_csv(output / "distance_distribution_source.csv.gz", index=False)
    full = observations.groupby(["shop_index", "candidate_position"], as_index=False)["count"].sum()
    spatial, spatial_summary = merchant_spatial_fit(theta, candidates, full, "all_external_orders")
    spatial.to_csv(output / "pickup_location_spatial_fit.csv.gz", index=False)
    result = {
        "schema_version": 1, "status": "external_validation_only",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "output_dir": str(output.resolve()), "run_hash": run_hash,
        "input_fingerprints": fingerprints, "locked_parameters": locked["parameters"],
        "order_audit": order_audit, "prepared": prepared,
        "memory": {"estimated_preparation_peak_gb": peak,
                   "estimated_evaluation_peak_gb": evaluation_peak,
                   "limit_gb": memory_limit_gb, "workers": 1},
        "metrics": metrics_rows, "pickup_location_spatial_fit": spatial_summary,
        "limitations": [
            "Only retained orders matched within 50 m to positive-volume buildings and inside the 5 km choice set are evaluated; inspect exclusion flow.",
            "Spatial-fit merchants here are actual pickup locations, not necessarily unique poi_id merchants.",
            "2020 Beijing and 2022 Baoding differ in city, platform, year and pandemic conditions.",
            "Order native datum is inferred through POI alignment, not independently certified.",
            "GBA building geometry/volume dates differ from the orders; volume follows existing preprocessing conventions.",
        ],
    }
    (output / "external_validation.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    return result
