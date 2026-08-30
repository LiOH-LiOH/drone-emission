"""Regression contracts for the fixed-cohort analysis view."""

import json
from pathlib import Path

import pytest
import pandas as pd

import drone_emission.analysis_workflow as workflow
import drone_emission.paper_reuse as paper_reuse
import drone_emission.analysis.pathway_scenarios as pathway
from drone_emission.analysis.city_attribution import prepare_r2_city_data
from drone_emission.paper_workflow import render
import drone_emission.visualization.formal_registry as formal_registry


def _fixture_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict:
    data = tmp_path / "official"
    work = tmp_path / "work"
    city = data / "city" / "alpha"
    city.mkdir(parents=True)
    for name in workflow.AUTHORITATIVE_AGGREGATES:
        (data / name).write_text(f"fresh-{name}\n", encoding="utf-8")
    (data / "combined_data.csv").write_text("city,sales\nalpha,1\n", encoding="utf-8")
    (data / "gef.csv").write_text("province,value\na,1\n", encoding="utf-8")
    stats = city / "region_stats.json"
    od = city / "alpha_od_pairs.csv"
    stats.write_text('{"stats": {"value": 1}}\n', encoding="utf-8")
    od.write_text("annual_sales\n1\n", encoding="utf-8")
    config = {
        "data_dir": str(data),
        "work_dir": str(work),
        "cache_dir": str(tmp_path / "cache"),
        "figures_dir": str(tmp_path / "figures"),
        "sales_path": str(data / "combined_data.csv"),
        "gef_path": str(data / "gef.csv"),
    }
    monkeypatch.setattr(workflow, "locked_region_keys", lambda: ["alpha"])
    monkeypatch.setattr(workflow, "cohort_hash", lambda: "a" * 64)
    monkeypatch.setattr(
        workflow,
        "_region_input_paths",
        lambda candidate: [
            (
                "alpha",
                Path(candidate["data_dir"]) / "city" / "alpha" / "region_stats.json",
                Path(candidate["data_dir"])
                / "city"
                / "alpha"
                / "alpha_od_pairs.csv",
            )
        ],
    )
    return config


def _complete_status(config: dict) -> dict:
    manifest = workflow.build_input_manifest(config)
    workflow.prepare_analysis_data_view(config, manifest)
    status = workflow._new_status(config, manifest)
    for name, paths in workflow._module_outputs(config).items():
        for path in paths:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(f"{name}\n", encoding="utf-8")
        status["modules"][name] = {
            "status": "complete",
            "outputs": workflow._output_signatures(paths),
        }
    status["complete"] = True
    workflow._atomic_json(status, workflow.analysis_root(config) / "status.json")
    return status


def test_prepare_atomically_refreshes_aggregates_and_keeps_od_symlink(
    tmp_path, monkeypatch
):
    config = _fixture_config(tmp_path, monkeypatch)
    official = Path(config["data_dir"])
    stale_product = official / "all_sensitivity_named.csv"
    stale_product.write_text("old analysis\n", encoding="utf-8")

    view = workflow.prepare_analysis_data_view(config)
    od_link = view / "city" / "alpha" / "alpha_od_pairs.csv"
    assert od_link.is_symlink()
    assert not (view / stale_product.name).exists()

    for name in workflow.AUTHORITATIVE_AGGREGATES:
        (official / name).write_text(f"updated-{name}\n", encoding="utf-8")
    workflow.prepare_analysis_data_view(config)
    assert od_link.is_symlink()
    assert od_link.resolve() == official / "city" / "alpha" / "alpha_od_pairs.csv"
    for name in workflow.AUTHORITATIVE_AGGREGATES:
        assert (view / name).read_bytes() == (official / name).read_bytes()


def test_render_gate_rejects_schema_v1_and_incomplete_status(tmp_path, monkeypatch):
    config = _fixture_config(tmp_path, monkeypatch)
    workflow.prepare_analysis_data_view(config)
    status_path = workflow.analysis_root(config) / "status.json"
    workflow._atomic_json({"schema_version": 1}, status_path)
    with pytest.raises(RuntimeError, match="schema is stale"):
        workflow.require_current_analysis(config)

    status = workflow._new_status(config, workflow.build_input_manifest(config))
    workflow._atomic_json(status, status_path)
    with pytest.raises(RuntimeError, match="incomplete"):
        workflow.require_current_analysis(config)


def test_render_gate_rejects_changed_aggregate_and_od_metadata(tmp_path, monkeypatch):
    config = _fixture_config(tmp_path, monkeypatch)
    _complete_status(config)
    official = Path(config["data_dir"])
    (official / "all_region_stats.csv").write_text("changed\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="inputs changed"):
        workflow.require_current_analysis(config)

    _complete_status(config)
    od = official / "city" / "alpha" / "alpha_od_pairs.csv"
    od.write_text("annual_sales\n100\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="inputs changed"):
        workflow.require_current_analysis(config)


def test_render_gate_requires_unchanged_module_outputs(tmp_path, monkeypatch):
    config = _fixture_config(tmp_path, monkeypatch)
    status = _complete_status(config)
    assert workflow.require_current_analysis(config) == status
    output = Path(status["modules"]["named-scenarios"]["outputs"][0]["path"])
    output.write_text("tampered\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="output changed"):
        workflow.require_current_analysis(config)


def test_partial_analysis_accumulates_and_changed_input_requires_force(
    tmp_path, monkeypatch
):
    config = _fixture_config(tmp_path, monkeypatch)

    def create_named(view, *, force):
        path = Path(view["data_dir"]) / "all_sensitivity_named.csv"
        path.write_text("region_key\nalpha\n", encoding="utf-8")
        return path

    def create_pathway(view, *, force):
        paths = workflow._module_outputs(config)["pathway-scenarios"]
        for path in paths:
            path.write_text("region_key\nalpha\n", encoding="utf-8")
        return paths[-1]

    monkeypatch.setattr(workflow, "_run_named_scenarios", create_named)
    monkeypatch.setattr(workflow, "_run_pathway_scenarios", create_pathway)
    first = workflow.analyze(config, only=["named-scenarios"])
    assert set(first["modules"]) == {"named-scenarios"}
    second = workflow.analyze(config, only=["pathway-scenarios"])
    assert set(second["modules"]) == {"named-scenarios", "pathway-scenarios"}

    Path(config["data_dir"], "all_region_stats.json").write_text(
        json.dumps({"changed": True}), encoding="utf-8"
    )
    with pytest.raises(RuntimeError, match="analyze --force"):
        workflow.analyze(config, only=["named-scenarios"])
    forced = workflow.analyze(config, force=True, only=["named-scenarios"])
    assert set(forced["modules"]) == {"named-scenarios"}


def test_explicit_reuse_records_an_independent_input_source(tmp_path, monkeypatch):
    config = _fixture_config(tmp_path, monkeypatch)
    reuse_manifest = tmp_path / "reuse" / "manifest.json"
    reuse_manifest.parent.mkdir()
    reuse_manifest.write_text("{}\n", encoding="utf-8")

    def restore(view, unused):
        data = Path(view["data_dir"])
        (data / "all_region_stats.csv").write_text("historic\n", encoding="utf-8")
        (data / "all_sensitivity_named.csv").write_text(
            "region_key\nalpha\n", encoding="utf-8"
        )
        return {
            "manifest": str(reuse_manifest),
            "restored": [str(data / "all_sensitivity_named.csv")],
        }

    monkeypatch.setattr(paper_reuse, "restore_reuse_directory", restore)
    monkeypatch.setattr(paper_reuse, "validate_reused_region_sets", lambda view: None)
    result = workflow.analyze(
        config, reuse_from=reuse_manifest.parent, only=["named-scenarios"]
    )
    assert result["reuse"]["manifest"] == str(reuse_manifest)
    assert result["input_manifest_sha256"] != workflow.build_input_manifest(config)[
        "manifest_sha256"
    ]
    stored = json.loads(
        workflow.analysis_input_manifest_path(config).read_text(encoding="utf-8")
    )
    assert stored["manifest_sha256"] == result["input_manifest_sha256"]


def test_render_requires_validated_input_manifest_binding():
    with pytest.raises(RuntimeError, match="validated analysis input manifest"):
        render({})


def test_authoritative_aggregate_is_filtered_only_at_analysis_boundary(
    tmp_path, monkeypatch
):
    stats = pd.DataFrame(
        {
            "region_key": ["alpha", "beta", "outside"],
            "region_level": ["city", "city", "city"],
            "total_ebike_lifecycle": [1.0, 2.0, 3.0],
            "oracle_min_lifecycle_kg": [0.5, 1.0, 1.5],
        }
    )
    monkeypatch.setattr(pathway, "locked_region_keys", lambda: ["alpha", "beta"])
    selected = pathway._cohort_city_rows(stats)
    assert selected["region_key"].tolist() == ["alpha", "beta"]

    enriched = tmp_path / "all_region_stats_enriched.csv"
    stats.to_csv(enriched, index=False)
    prepared, _ = prepare_r2_city_data(
        enriched,
        tmp_path,
        supplement_missing_detour=False,
        region_keys={"alpha", "beta"},
    )
    assert prepared["region_key"].tolist() == ["alpha", "beta"]


def test_formal_renderer_filters_authoritative_aggregate_to_cohort(
    tmp_path, monkeypatch
):
    frame = pd.DataFrame(
        {
            "region_key": ["alpha", "outside"],
            "region_level": ["city", "city"],
            "grid_emission_factor": [0.5, 0.6],
        }
    )
    stats = tmp_path / "all_region_stats.csv"
    enriched = tmp_path / "all_region_stats_enriched.csv"
    frame.to_csv(stats, index=False)
    frame.to_csv(enriched, index=False)

    class Context:
        data_root = tmp_path

        @staticmethod
        def require_file(unused, *, stage):
            return stats

    monkeypatch.setattr(
        "drone_emission.paper_cohort.locked_region_keys", lambda: ["alpha"]
    )
    selected = formal_registry._city_stats(Context())
    assert selected["region_key"].tolist() == ["alpha"]
