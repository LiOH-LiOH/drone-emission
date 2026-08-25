"""Fast cohort/reuse contracts for the remote validation worker."""

import hashlib
import json
import math
from pathlib import Path

import pandas as pd
import pytest

from drone_emission.paper_cohort import (
    cohort_hash,
    load_cohort_contract,
    locked_region_keys,
    region_keys_sha256,
)
from drone_emission.paper_reuse import restore_reuse_directory
from drone_emission.analysis.city_attribution import write_r2_city_attribution_report
from drone_emission.utils.grid_emission_factor import fallback_factor
from drone_emission.visualization.distance_threshold_inputs import (
    _resolve_summary_path,
)


def test_locked_paper_cohort_is_exactly_313_cities():
    contract = load_cohort_contract()
    keys = locked_region_keys(contract)
    assert len(keys) == 313
    assert "linzhi" not in keys
    assert "ganzizangzuzizhizhou" in keys
    assert region_keys_sha256(keys) == cohort_hash()
    assert contract["excluded_regions"]["linzhi"] == "insufficient_od_pairs"


def test_locked_fallback_is_2022_national_mean_for_five_cities():
    contract = load_cohort_contract()
    fallback = contract["grid_emission_factor_fallback"]
    assert fallback_factor() == pytest.approx(0.5366)
    assert len(fallback["affected_region_keys"]) == 5
    assert "linzhi" not in fallback["affected_region_keys"]


def test_distance_threshold_loader_never_accepts_partial_as_complete(tmp_path):
    partial = tmp_path / "all_city_distance_threshold_summary.partial.csv"
    partial.write_text("region_key,distance_basis\ncity,road\n", encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="partial"):
        _resolve_summary_path(tmp_path, None)


def test_explicit_reuse_directory_verifies_and_restores_hash(tmp_path):
    reuse = tmp_path / "reuse"
    source = reuse / "payload" / "data" / "all_sensitivity_named.csv"
    source.parent.mkdir(parents=True)
    pd.DataFrame({"region_key": ["beijing"], "value": [1]}).to_csv(
        source, index=False
    )
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    contract = load_cohort_contract()
    manifest = {
        "schema_version": 1,
        "cohort_sha256": cohort_hash(),
        "city_count": 313,
        "grid_emission_factor_fallback": contract[
            "grid_emission_factor_fallback"
        ],
        "artifacts": [
            {
                "path": "data:all_sensitivity_named.csv",
                "size_bytes": source.stat().st_size,
                "sha256": digest,
            }
        ],
    }
    (reuse / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    roots = {name: tmp_path / name for name in ("data", "cache", "work")}
    result = restore_reuse_directory(
        {
            "data_dir": str(roots["data"]),
            "cache_dir": str(roots["cache"]),
            "work_dir": str(roots["work"]),
        },
        reuse,
    )
    restored = roots["data"] / "all_sensitivity_named.csv"
    assert restored.read_bytes() == source.read_bytes()
    assert result["restored"] == [str(restored.absolute())]

    source.write_text("tampered\n", encoding="utf-8")
    with pytest.raises(ValueError, match="size mismatch|SHA-256 mismatch"):
        restore_reuse_directory(
            {
                "data_dir": str(roots["data"]),
                "cache_dir": str(roots["cache"]),
                "work_dir": str(roots["work"]),
            },
            reuse,
        )


def test_active_render_code_has_no_retired_sample_or_fallback_literals():
    root = Path(__file__).parents[1] / "src" / "drone_emission"
    sources = [
        root / "visualization" / "formal_registry.py",
        root / "visualization" / "contribution_renderers.py",
        root / "utils" / "grid_emission_factor.py",
    ]
    combined = "\n".join(path.read_text(encoding="utf-8") for path in sources)
    assert "CONTRIBUTION_STUDY_CITY_COUNT = 314" not in combined
    assert "fallback_0.5600" not in combined


def test_city_attribution_report_serializes_undefined_diagnostics_as_null(tmp_path):
    report = tmp_path / "attribution.json"
    write_r2_city_attribution_report(
        {"defined": 1.0, "diagnostics": [float("nan"), float("inf")]},
        report,
    )
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert math.isfinite(payload["defined"])
    assert payload["diagnostics"] == [None, None]
