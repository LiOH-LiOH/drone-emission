"""Fixture-light tests for the independent Beijing coordinate audit."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from drone_emission.calibration import coordinate_restore as restore


def _synthetic_inputs():
    poi_gcj_lng = np.array([116.40, 116.42, 116.44])
    poi_gcj_lat = np.array([39.90, 39.92, 39.94])
    poi_wgs_lng, poi_wgs_lat = restore._convert_pairs(
        poi_gcj_lng, poi_gcj_lat, restore.gcj02_to_wgs84
    )
    poi = pd.DataFrame(
        {
            "poi_name": ["A", "B", "C"],
            "poi_category": ["food"] * 3,
            "poi_lng_gcj02": poi_gcj_lng,
            "poi_lat_gcj02": poi_gcj_lat,
            "poi_lng_wgs84": poi_wgs_lng,
            "poi_lat_wgs84": poi_wgs_lat,
        }
    )
    shops = pd.DataFrame(
        {
            "shop_id": ["a", "b", "c"],
            "original_pick_lng": poi["poi_lng_gcj02"] - restore.HISTORICAL_OFFSET[0],
            "original_pick_lat": poi["poi_lat_gcj02"] - restore.HISTORICAL_OFFSET[1],
        }
    )
    return shops, poi


def test_known_global_translation_is_recovered_without_overwrite():
    shops, poi = _synthetic_inputs()
    original = shops.copy(deep=True)
    result = restore.estimate_global_translation(shops, poi)
    corrected = restore.append_corrected_shop_coordinates(shops, poi, result)

    assert result.longitude_degrees == pytest.approx(-5.1192, abs=2e-5)
    assert result.latitude_degrees == pytest.approx(0.7810, abs=2e-5)
    pd.testing.assert_frame_equal(shops, original)
    assert {
        "pick_lng_gcj02",
        "pick_lat_gcj02",
        "pick_lng_wgs84",
        "pick_lat_wgs84",
    } <= set(corrected)
    assert np.max(result.distances_m) < 1.0
    assert set(restore.match_metrics(result.distances_m)["match_rates"]) == {
        "within_25m", "within_50m", "within_100m", "within_200m"
    }


def test_multi_file_loading_deduplicates_shops_and_tracks_sources(tmp_path):
    columns = {
        "shop_id": ["a", "a"],
        "pick_lng": [121.5, 121.5],
        "pick_lat": [39.1, 39.1],
        "deliver_lng": [121.51, 121.52],
        "deliver_lat": [39.11, 39.12],
    }
    pd.DataFrame(columns).iloc[:1].to_csv(tmp_path / "one.txt", index=False)
    pd.DataFrame(columns).iloc[1:].to_csv(tmp_path / "two.txt", index=False)
    orders, files = restore.load_orders(tmp_path)

    assert len(files) == 2
    assert len(orders) == 2
    assert set(orders["source_file"]) == {"one.txt", "two.txt"}
    assert len(restore.unique_shops(orders)) == 1
    assert len(restore.fingerprint_files(files, tmp_path)["sha256"]) == 64


def test_order_export_appends_both_datums_and_retains_input():
    orders = pd.DataFrame(
        {
            "shop_id": ["a"], "pick_lng": [121.5], "pick_lat": [39.1],
            "deliver_lng": [121.51], "deliver_lat": [39.11],
            "source_file": ["one.txt"],
        }
    )
    original = orders.copy(deep=True)
    corrected = restore.append_corrected_order_coordinates(orders, *restore.HISTORICAL_OFFSET)
    pd.testing.assert_frame_equal(orders, original)
    assert len(corrected) == len(orders)
    for prefix in ("pick", "deliver"):
        for datum in ("gcj02", "wgs84"):
            assert f"{prefix}_lng_{datum}" in corrected
            assert f"{prefix}_lat_{datum}" in corrected


def test_run_defaults_to_no_full_order_export(tmp_path, monkeypatch):
    orders_dir = tmp_path / "orders"
    orders_dir.mkdir()
    shops, poi = _synthetic_inputs()
    order_frame = pd.DataFrame(
        {
            "shop_id": shops["shop_id"],
            "pick_lng": shops["original_pick_lng"],
            "pick_lat": shops["original_pick_lat"],
            "deliver_lng": shops["original_pick_lng"] + 0.001,
            "deliver_lat": shops["original_pick_lat"] + 0.001,
        }
    )
    order_frame.iloc[:2].to_csv(orders_dir / "one.txt", index=False)
    order_frame.iloc[2:].to_csv(orders_dir / "two.txt", index=False)
    poi_csv = tmp_path / "poi.csv"
    pd.DataFrame(
        {
            "name": poi["poi_name"], "category": poi["poi_category"],
            "lng": poi["poi_lng_gcj02"], "lat": poi["poi_lat_gcj02"],
        }
    ).to_csv(poi_csv, index=False)
    monkeypatch.setattr(restore, "plot_alignment_audit", lambda *args, **kwargs: None)
    monkeypatch.setattr(restore, "plot_supply_demand_heatmap", lambda *args, **kwargs: None)
    output = tmp_path / "output"
    record = restore.run_coordinate_restore(orders_dir, poi_csv, output, no_basemap=True)

    assert record["counts"] == {"orders": 3, "unique_shops": 3, "poi": 3}
    assert (output / "corrected_shops.csv.gz").exists()
    assert not (output / "corrected_orders.csv.gz").exists()
    payload = json.loads((output / "coordinate_transform.json").read_text())
    assert payload["inputs"]["orders"]["file_count"] == 2
    assert payload["coordinate_reference"]["orders_corrected_native"] == "gcj02"


def test_write_orders_includes_row_provenance(tmp_path, monkeypatch):
    orders_dir = tmp_path / "orders"
    orders_dir.mkdir()
    shops, poi = _synthetic_inputs()
    pd.DataFrame(
        {
            "shop_id": shops["shop_id"], "pick_lng": shops["original_pick_lng"],
            "pick_lat": shops["original_pick_lat"],
            "deliver_lng": shops["original_pick_lng"],
            "deliver_lat": shops["original_pick_lat"],
        }
    ).to_csv(orders_dir / "orders.txt", index=False)
    poi_csv = tmp_path / "poi.csv"
    pd.DataFrame(
        {"name": poi["poi_name"], "lng": poi["poi_lng_gcj02"], "lat": poi["poi_lat_gcj02"]}
    ).to_csv(poi_csv, index=False)
    monkeypatch.setattr(restore, "plot_alignment_audit", lambda *args, **kwargs: None)
    monkeypatch.setattr(restore, "plot_supply_demand_heatmap", lambda *args, **kwargs: None)
    output = tmp_path / "output"
    restore.run_coordinate_restore(
        orders_dir, poi_csv, output, write_orders=True, no_basemap=True
    )
    exported = pd.read_csv(output / "corrected_orders.csv.gz")
    assert len(exported) == 3
    assert set(exported["source_file"]) == {"orders.txt"}
    assert {"pick_lng_gcj02", "deliver_lng_wgs84"} <= set(exported)


def test_missing_inputs_and_poi_name_are_explicit(tmp_path):
    with pytest.raises(FileNotFoundError, match="Input does not exist"):
        restore.load_orders(tmp_path / "missing")
    poi_path = tmp_path / "poi.csv"
    pd.DataFrame({"lng": [116.4], "lat": [39.9]}).to_csv(poi_path, index=False)
    with pytest.raises(ValueError, match="missing a name field"):
        restore.load_poi(poi_path, "gcj02")


def test_basemap_failure_is_explicit_and_writes_no_figure(tmp_path, monkeypatch):
    orders = pd.DataFrame(
        {
            "pick_lng_wgs84": [116.4, 116.41],
            "pick_lat_wgs84": [39.9, 39.91],
            "deliver_lng_wgs84": [116.405, 116.415],
            "deliver_lat_wgs84": [39.905, 39.915],
        }
    )
    monkeypatch.setattr(
        restore,
        "_add_basemap",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError("OpenStreetMap basemap retrieval failed")
        ),
    )
    pdf = tmp_path / "heatmap.pdf"
    with pytest.raises(RuntimeError, match="basemap retrieval failed"):
        restore.plot_supply_demand_heatmap(
            orders, pdf, tmp_path / "heatmap.png", tmp_path / "source.csv.gz",
            gridsize=10, no_basemap=False, cache_dir=tmp_path / "cache",
        )
    assert not pdf.exists()


def test_tracked_baseline_has_complete_schema():
    path = Path(restore.__file__).with_name("beijing_coordinate_transform.json")
    payload = json.loads(path.read_text())
    assert payload["model"] == "global_constant_translation"
    assert payload["offset"]["longitude_degrees"] == -5.1192
    assert payload["offset"]["latitude_degrees"] == 0.781
    assert len(payload["inputs"]["orders"]["sha256"]) == 64
    assert set(payload["metrics"]["match_rates"]) == {
        "within_25m", "within_50m", "within_100m", "within_200m"
    }


def test_apply_transform_validates_all_29_files_without_poi(tmp_path):
    orders_dir = tmp_path / "orders"
    orders_dir.mkdir()
    for day in range(1, 30):
        pd.DataFrame(
            {
                "shop_id": [f"shop-{day}"],
                "pick_lng": [121.5],
                "pick_lat": [39.1],
                "deliver_lng": [121.51],
                "deliver_lat": [39.11],
            }
        ).to_csv(orders_dir / f"orders_202003{day:02d}.txt", index=False)
    _, files = restore.load_orders(orders_dir)
    transform = {
        "schema_version": 1,
        "model": "global_constant_translation",
        "offset": {"longitude_degrees": -5.1192, "latitude_degrees": 0.781},
        "inputs": {"orders": restore.fingerprint_files(files, orders_dir)},
    }
    transform_path = tmp_path / "transform.json"
    transform_path.write_text(json.dumps(transform))
    output = tmp_path / "corrected_orders.csv.gz"
    record = restore.apply_coordinate_transform(orders_dir, transform_path, output)
    exported = pd.read_csv(output)
    assert record["mode"] == "apply_existing_transform"
    assert record["orders"]["file_count"] == 29
    assert len(exported) == 29
    assert {"pick_lng_gcj02", "deliver_lng_wgs84", "source_file"} <= set(exported)

    (orders_dir / "orders_20200301.txt").write_text("changed")
    with pytest.raises(ValueError, match="fingerprint"):
        restore.apply_coordinate_transform(orders_dir, transform_path, output)
