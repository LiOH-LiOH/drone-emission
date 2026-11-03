"""Verify exact subtraction, policy guards and unchanged pathway sampling."""
from argparse import Namespace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from drone_emission import cli, paper_cohort
from drone_emission.analysis.pathway_scenarios import _sample_od, _valid_od
from drone_emission.tools.migrate_cohort_exclusion import (
    EXCLUDED_KEYS, SOURCE_COHORT_SHA256, original_city_rows, sample_city_od, subtract_rows,
)


def test_lock_is_exact_old_sample_minus_two():
    keys = set(paper_cohort.locked_region_keys())
    assert len(keys) == 311
    assert paper_cohort.region_keys_sha256(keys | EXCLUDED_KEYS) == SOURCE_COHORT_SHA256
    assert paper_cohort.experiment_excluded_region_keys() == EXCLUDED_KEYS


@pytest.mark.parametrize("key", sorted(EXCLUDED_KEYS))
def test_explicit_city_rejected_before_inputs_or_cache(key, monkeypatch):
    def forbidden(*args):
        pytest.fail("excluded city must fail before reading experiment inputs")
    monkeypatch.setattr(cli, "_config", forbidden)
    with pytest.raises(ValueError, match="excluded from this experiment"):
        cli.command_cities(Namespace(region=[key]))


def test_policy_audit_does_not_inspect_city_files(tmp_path, monkeypatch):
    registry = {key: {"region_name_en": key} for key in EXCLUDED_KEYS}
    registry_path = tmp_path / "regions.json"
    import json
    registry_path.write_text(json.dumps(registry))
    monkeypatch.setattr(paper_cohort, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(paper_cohort, "configured_city_keys", lambda: sorted(EXCLUDED_KEYS))
    monkeypatch.setattr(paper_cohort, "locked_region_keys", lambda *args: [])
    result = paper_cohort.audit_city_cohort({"data_dir": str(tmp_path)})
    assert result["candidate_region_keys"] == []
    assert {row["reason"] for row in result["cities"]} == {"invalid_merchant_geography"}
    assert all(row["valid_weighted_od_pairs"] is None for row in result["cities"])


def test_subtraction_preserves_repeated_city_rows_and_values():
    source = pd.DataFrame({"region_key": [*sorted(EXCLUDED_KEYS), "beijing", "beijing"],
                           "value": [1., 2., 3.125, 4.25]})
    result = subtract_rows(source, {"beijing"})
    pd.testing.assert_frame_equal(result, source.iloc[2:])
    with pytest.raises(ValueError, match="source cohort mismatch"):
        subtract_rows(source, {"beijing", "shanghai"})


def test_original_curve_sample_indices_survive_existing_exclusions():
    frame = pd.DataFrame({"region_key": ["already_excluded", "beijing", *sorted(EXCLUDED_KEYS), "shanghai"]})
    frame["region_level"] = "city"
    original = original_city_rows(frame, {"beijing", "shanghai"})
    assert original.index.tolist() == [1, 2, 3, 4]
    retained = original.loc[~original.region_key.isin(EXCLUDED_KEYS)]
    assert retained.index.tolist() == [1, 4]


@pytest.mark.parametrize("count", [21, 1607])
def test_streamed_samples_match_existing_pandas_rule(tmp_path, count):
    frame = pd.DataFrame({"road_distance_m": np.arange(count) * 2.,
                          "distance_km": np.arange(count) / 100 + .1,
                          "annual_sales": np.arange(count) + 1.})
    invalid = pd.DataFrame({"road_distance_m": [-1., 1., 1.],
                            "distance_km": [1., 0., 1.], "annual_sales": [1., 1., 0.]})
    raw = pd.concat([frame.iloc[:11], invalid, frame.iloc[11:]], ignore_index=True)
    path = tmp_path / "od.csv"
    raw.to_csv(path, index=False)
    expected, expected_scale = _sample_od(_valid_od(raw), 71)
    actual, scale = sample_city_od(path, count, 71, chunk_rows=79)
    pd.testing.assert_frame_equal(actual, expected.reset_index(drop=True))
    assert scale == expected_scale
    with pytest.raises(ValueError, match="valid OD count changed"):
        sample_city_od(path, count + 1, 71, chunk_rows=79)


def test_cohort_acceptance_cannot_reintroduce_policy_exclusion(tmp_path):
    with pytest.raises(ValueError, match="policy exclusion"):
        paper_cohort.accept_candidate_cohort({"candidate_region_keys": list(EXCLUDED_KEYS)},
                                             target=tmp_path / "forbidden.json")
    assert not (tmp_path / "forbidden.json").exists()


@pytest.mark.parametrize("contaminated", [None, "all_region_stats.json", "all_region_stats.csv", "all_region_stats_enriched.csv"])
def test_aggregate_cache_checks_every_authoritative_file(tmp_path, monkeypatch, contaminated):
    import json
    from drone_emission.tools import consolidate_stats, enrich_stats

    for name in ("all_region_stats.json", "all_region_stats.csv", "all_region_stats_enriched.csv"):
        keys = ["beijing"] + (["aomentebiexingzhengqu"] if name == contaminated else [])
        if name.endswith(".json"):
            (tmp_path / name).write_text(json.dumps({key: {} for key in keys}))
        else:
            pd.DataFrame({"region_key": keys}).to_csv(tmp_path / name, index=False)
    calls = []
    monkeypatch.setattr(cli, "_config", lambda env: {"data_dir": str(tmp_path), "sales_path": "unused", "gef_path": "unused"})
    monkeypatch.setattr(consolidate_stats, "consolidate", lambda *args: calls.append("consolidate"))
    monkeypatch.setattr(enrich_stats, "enrich", lambda *args: calls.append("enrich"))
    cli.command_aggregate(Namespace(env="local", force=False))
    assert calls == (["consolidate", "enrich"] if contaminated else [])
