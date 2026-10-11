"""External validation must preserve origins, provenance and locked parameters."""

import json

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import box

from drone_emission.calibration.external_validation import load_meituan_orders, run_external_validation
from drone_emission.calibration import external_validation
from drone_emission.calibration.gravity_calibration import sha256_file


def _bundle(tmp_path):
    bundle = tmp_path / "bundle"
    (bundle / "data").mkdir(parents=True)
    (bundle / "processing").mkdir()
    rows = []
    for day in range(17, 25):
        row = {"dt": 20221000 + day, "order_id": str(day), "poi_id": "same",
               "is_courier_grabbed": 1}
        for datum in ("gcj02", "wgs84"):
            row.update({f"sender_lng_{datum}": 115.45 + (day % 2) * .0005,
                        f"sender_lat_{datum}": 38.9, f"recipient_lng_{datum}": 115.451,
                        f"recipient_lat_{datum}": 38.9})
        rows.append(row)
    rows.append({**rows[0], "is_courier_grabbed": 0})
    source = bundle / "data/all_waybill_info_meituan_0322.csv.gz"
    pd.DataFrame(rows).to_csv(source, index=False)
    transform = {"packaged_artifacts": [{"path": str(source.relative_to(bundle)),
                                        "sha256": sha256_file(source)}],
                 "counts": {"accepted_orders": 8}}
    (bundle / "processing/coordinate_transform.json").write_text(json.dumps(transform))
    return bundle, source


def test_fixed_validation_preserves_locations_and_never_refits(tmp_path, monkeypatch):
    bundle, source = _bundle(tmp_path)
    orders, audit = load_meituan_orders(bundle)
    assert audit["accepted_order_count"] == 8
    assert audit["excluded_rejected_waybills"] == 1
    assert audit["merchant_count"] == 1
    assert audit["pickup_location_count"] == 2
    assert orders["pick_lng_wgs84"].nunique() == 2
    buildings = tmp_path / "buildings.gpkg"
    gpd.GeoDataFrame({"building_id": ["a", "b"], "volume_m3": [1000., 2000.]},
                     geometry=[box(115.4509, 38.8999, 115.4511, 38.9001),
                               box(115.452, 38.9, 115.4521, 38.9001)],
                     crs="EPSG:4326").to_file(buildings, driver="GPKG")
    buildings.with_suffix(".coverage.json").write_text(json.dumps({
        "coverage": "all_pickup_5km_choice_sets_and_delivery_50m_matches",
        "orders_sha256": sha256_file(source), "buildings_sha256": sha256_file(buildings),
    }))
    parameters = tmp_path / "parameters.json"
    locked = {"accepted_at_utc": "2026-01-01", "source_result_sha256": "locked",
              "parameters": {"alpha": .42, "beta": .32, "gamma": 1.65}}
    parameters.write_text(json.dumps(locked))
    previous = parameters.read_bytes()
    def forbid_fit(*args, **kwargs):
        raise AssertionError("External validation must never fit parameters")
    monkeypatch.setattr("drone_emission.calibration.gravity_calibration.fit_parameters", forbid_fit)
    evaluate = external_validation.evaluate_model
    first_calls = []
    def interrupted_evaluation(*args, **kwargs):
        first_calls.append(1)
        if len(first_calls) == 3:
            raise RuntimeError("simulated interruption")
        return evaluate(*args, **kwargs)
    monkeypatch.setattr(external_validation, "evaluate_model", interrupted_evaluation)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        run_external_validation(bundle, buildings, parameters, tmp_path / "result", by_date=True)
    resumed_calls = []
    def resumed_evaluation(*args, **kwargs):
        resumed_calls.append(1)
        return evaluate(*args, **kwargs)
    monkeypatch.setattr(external_validation, "evaluate_model", resumed_evaluation)
    result = run_external_validation(bundle, buildings, parameters, tmp_path / "result", by_date=True)
    assert len(resumed_calls) == 16  # two completed models are restored, never rerun
    assert result["status"] == "external_validation_only"
    assert result["locked_parameters"] == locked["parameters"]
    assert result["prepared"]["exclusion_flow"]["retained_valid_orders"] == 8
    assert len(result["metrics"]) == 18  # aggregate + eight dates, two locked models
    assert parameters.read_bytes() == previous
    overall = run_external_validation(bundle, buildings, parameters, tmp_path / "result")
    assert overall["evaluation_scopes"] == ["all_external_orders"]
    assert len(overall["metrics"]) == 2
    assert len(resumed_calls) == 16  # overall rerun reuses both checkpoints as well
    source.write_bytes(source.read_bytes() + b"tampered")
    with pytest.raises(ValueError, match="locked bundle hash"):
        load_meituan_orders(bundle)
