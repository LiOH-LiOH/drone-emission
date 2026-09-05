"""Small fixture-free contract tests intended for remote validation."""

import ast
import csv
import hashlib
import json
from pathlib import Path

from drone_emission.paper_workflow import (
    figure_required_inputs,
    load_manifest,
    request_index,
    select_figures,
)
from drone_emission.visualization.formal_registry import REQUEST_RENDERERS


def test_manifest_has_exact_locked_set():
    manifest = load_manifest()
    assert manifest["figure_count"] == 12
    assert len(manifest["figures"]) == 12
    assert {figure["mode"] for figure in manifest["figures"]} == {"inkscape"}
    assert manifest["render_request_count"] == 31
    assert len(manifest["requests"]) == 31
    assert sum(len(request["outputs"]) for request in manifest["requests"]) == 46
    assert all(
        figure_required_inputs(figure, manifest)
        for figure in manifest["figures"]
    )
    assert all(len(figure["report_inputs"]) == 1 for figure in manifest["figures"])


def test_manifest_requests_exactly_match_registry():
    requests = [
        request
        for figure in load_manifest()["figures"]
        for request in figure["render_requests"]
    ]
    assert len(requests) == 31
    assert len(requests) == len(set(requests))
    assert set(requests) == set(REQUEST_RENDERERS)


def test_manifest_rendering_is_bound_to_tracked_city_cohort():
    root = Path(__file__).parents[1]
    cohort = json.loads(
        (root / "src" / "drone_emission" / "paper_cohort.json").read_text()
    )
    assert cohort["expected_city_count"] == 313
    assert cohort["minimum_valid_weighted_od_pairs"] == 30
    assert cohort["grid_emission_factor_fallback"]["kg_co2e_per_kwh"] == 0.5366
    assert cohort["excluded_regions"]["linzhi"] == "insufficient_od_pairs"


def test_retired_visualization_interfaces_are_physically_absent():
    package = Path(__file__).parents[1] / "src" / "drone_emission"
    for name in ("analysis.py", "paper.py", "emission.py", "sensitivity.py"):
        assert not (package / "visualization" / name).exists()
    for name in (
        "plot_distance_threshold_analysis.py",
        "plot_drone_spatial_concentration.py",
        "plot_threshold_resilience.py",
    ):
        assert not (package / "tools" / name).exists()


def test_manifest_paths_are_one_to_one():
    figures = load_manifest()["figures"]
    staging = [Path(figure["staging"]).name for figure in figures]
    publish = [Path(figure["publish"]).name for figure in figures]
    assert len(set(staging)) == 12
    assert staging == publish
    assert staging == [f"{figure['id']}.pdf" for figure in figures]


def test_contribution_figure_consumes_prepared_analysis_and_population():
    request = request_index()["contribution_decomposition_city_b"]
    assert request["cache_version"] == 7
    assert request["required_inputs"] == [
        "all_city_contribution_decomposition.csv",
        "all_region_stats_enriched.csv",
    ]
    assert request["outputs"] == [
        "contribution_decomposition_city_b.pdf",
        "contribution_decomposition_city_population.pdf",
    ]
    figure = next(
        figure
        for figure in load_manifest()["figures"]
        if figure["id"] == "fig5_contribution_decomposition"
    )
    assert figure["mode"] == "inkscape"
    assert figure["inkscape_source"] == "figures/inkscape/figure17.svg"


def test_delivery_mode_figure_consumes_declared_beijing_inputs():
    request = request_index()["delivery_mode_lifecycle_comparison"]
    assert request["cache_version"] == 5
    assert request["required_inputs"] == [
        "city/beijing/beijing_od_pairs.csv",
        "project:experiment.toml",
    ]
    assert request["outputs"] == ["delivery_mode_lifecycle_comparison.pdf"]
    figure = next(
        figure
        for figure in load_manifest()["figures"]
        if figure["id"] == "fig7_delivery_mode_lifecycle_comparison"
    )
    assert figure["inkscape_source"] == "figures/inkscape/figure18.svg"


def test_four_city_delivery_mode_candidates_consume_retained_analysis():
    expected_inputs = [
        "work:analysis/four_city_delivery_modes/four_city_mode_oracle_summary.csv",
        "work:analysis/four_city_delivery_modes/four_city_mode_oracle_report.json",
    ]
    for figure_id in (
        "candidate_four_city_mode_replacement_rate",
        "candidate_four_city_mode_relative_reduction",
        "candidate_four_city_mode_shared_legend",
    ):
        request = request_index()[figure_id]
        expected_cache_version = {
            "candidate_four_city_mode_replacement_rate": 3,
            "candidate_four_city_mode_relative_reduction": 4,
            "candidate_four_city_mode_shared_legend": 2,
        }[figure_id]
        assert request["cache_version"] == expected_cache_version
        assert request["required_inputs"] == expected_inputs
        assert request["outputs"] == [f"{figure_id}.pdf"]
    figure = next(
        item
        for item in load_manifest()["figures"]
        if item["id"] == "fig7_delivery_mode_lifecycle_comparison"
    )
    assert figure["render_requests"] == [
        "delivery_mode_lifecycle_comparison",
        "candidate_four_city_mode_relative_reduction",
        "candidate_four_city_mode_replacement_rate",
        "candidate_four_city_mode_shared_legend",
    ]


def test_city_gef_distribution_consumes_cohort_statistics_only():
    request = request_index()["city_gef_distribution"]
    assert request["cache_version"] == 4
    assert request["required_inputs"] == [
        "all_region_stats.csv",
        "all_region_stats_enriched.csv",
    ]
    assert request["outputs"] == ["city_gef_distribution.pdf"]


def test_city_gef_lifecycle_components_consumes_cohort_statistics_only():
    request = request_index()["city_gef_lifecycle_components"]
    assert request["cache_version"] == 6
    assert request["required_inputs"] == [
        "all_region_stats.csv",
        "all_region_stats_enriched.csv",
    ]
    assert request["outputs"] == ["city_gef_lifecycle_components.pdf"]


def test_city_distance_response_uses_only_declared_four_city_inputs():
    request = request_index()["city_distance_drone_share_emission_fit"]
    assert request["cache_version"] == 1
    assert request["required_inputs"] == [
        "all_region_stats.csv",
        "all_region_stats_enriched.csv",
        "city/chengdu/chengdu_od_pairs.csv",
        "city/guangzhou/guangzhou_od_pairs.csv",
        "city/beijing/beijing_od_pairs.csv",
        "city/shanghai/shanghai_od_pairs.csv",
        "city/chengdu/chengdu_roadnet.graphml",
        "city/guangzhou/guangzhou_roadnet.graphml",
        "city/beijing/beijing_roadnet.graphml",
        "city/shanghai/shanghai_roadnet.graphml",
        "project:experiment.toml",
    ]
    assert request["outputs"] == [
        "city_distance_drone_share_emission_fit.pdf"
    ]


def test_city_detour_reduction_by_gef_uses_declared_cohort_and_city_inputs():
    request = request_index()["city_detour_reduction_by_gef"]
    assert request["cache_version"] == 1
    assert request["required_inputs"] == [
        "all_region_stats.csv",
        "all_region_stats_enriched.csv",
        "city/chengdu/chengdu_od_pairs.csv",
        "city/guangzhou/guangzhou_od_pairs.csv",
        "city/beijing/beijing_od_pairs.csv",
        "city/shanghai/shanghai_od_pairs.csv",
        "city/chengdu/chengdu_roadnet.graphml",
        "city/guangzhou/guangzhou_roadnet.graphml",
        "city/beijing/beijing_roadnet.graphml",
        "city/shanghai/shanghai_roadnet.graphml",
        "project:experiment.toml",
    ]
    assert request["outputs"] == ["city_detour_reduction_by_gef.pdf"]


def test_city_reduction_scatter_consumes_cohort_statistics_only():
    request = request_index()["city_reduction_potential_scatter"]
    assert request["cache_version"] == 4
    assert request["required_inputs"] == [
        "all_region_stats.csv",
        "all_region_stats_enriched.csv",
    ]
    assert request["outputs"] == ["city_reduction_potential_scatter.pdf"]


def test_pathway_renderers_consume_only_prepared_analysis():
    requests = request_index()
    expected = {
        "all_region_stats.csv",
        "all_pathway_scenario_city.csv",
        "all_pathway_curve_points.csv",
        "all_pathway_scenarios_report.json",
        "project:experiment.toml",
    }
    for request_id in (
        "regions_map_pathway_scenarios",
        "regions_map_pathway_scenario_maps",
    ):
        assert set(requests[request_id]["required_inputs"]) == expected


def test_render_selection_rejects_internal_plot_names():
    try:
        select_figures(["city_gef_context_detour_summary"])
    except ValueError as error:
        assert "manifest IDs only" in str(error)
    else:
        raise AssertionError("internal plot name was accepted")


def test_locked_beijing_distance_sources_are_complete():
    root = Path(__file__).parents[1]
    histogram = (
        root
        / "src"
        / "drone_emission"
        / "calibration"
        / "observed_delivery_distance_histogram.csv"
    )
    with histogram.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 90
    assert sum(int(row["count"]) for row in rows) == 278688
    model_source = (
        root
        / "src"
        / "drone_emission"
        / "calibration"
        / "beijing_gravity_distance_distribution_a1303469cb38697a.csv"
    )
    with model_source.open(newline="", encoding="utf-8") as handle:
        model_rows = list(csv.DictReader(handle))
    assert len(model_rows) == 202
    assert {row["series"] for row in model_rows} == {
        "observed",
        "joint_three_parameter",
    }
    assert {float(row["distance_bin_km"]) for row in model_rows} == {
        index * 0.05 for index in range(101)
    }
    request = request_index()["beijing_order_distance_distribution"]
    assert request["cache_version"] == 2
    assert request["required_inputs"] == [
        "package:calibration/observed_delivery_distance_histogram.csv",
        "package:calibration/gravity_calibration_summary.json",
        "package:calibration/beijing_gravity_distance_distribution_a1303469cb38697a.csv",
    ]
    assert request["outputs"] == ["beijing_order_distance_distribution.pdf"]
    endpoint_source = (
        root
        / "src"
        / "drone_emission"
        / "calibration"
        / "beijing_gravity_endpoint_grid_a1303469cb38697a.csv"
    )
    with endpoint_source.open(newline="", encoding="utf-8") as handle:
        endpoint_rows = list(csv.DictReader(handle))
    assert len(endpoint_rows) == 4093
    assert set(endpoint_rows[0]) == {
        "grid_y",
        "grid_x",
        "observed_orders",
        "expected_orders",
        "x_min_m",
        "y_min_m",
    }
    assert hashlib.sha256(endpoint_source.read_bytes()).hexdigest() == (
        "7b93092c94362ee670e93860f3b46860f6108e4d289f834f21c6c955ae3bd9da"
    )
    assert sum(float(row["observed_orders"]) for row in endpoint_rows) == 245144.0
    assert abs(
        sum(float(row["expected_orders"]) for row in endpoint_rows) - 245144.0
    ) < 1e-6
    endpoint_request = request_index()["beijing_endpoint_grid_heatmap"]
    assert endpoint_request["required_inputs"] == [
        "package:calibration/gravity_calibration_summary.json",
        "package:calibration/beijing_gravity_endpoint_grid_a1303469cb38697a.csv",
    ]
    assert endpoint_request["outputs"] == ["beijing_endpoint_grid_heatmap.pdf"]
    ed6 = next(
        figure
        for figure in load_manifest()["figures"]
        if figure["id"] == "extended_data_fig6_delivery_distance_distribution"
    )
    assert ed6["render_requests"] == [
        "beijing_order_distance_distribution",
        "beijing_endpoint_grid_heatmap",
    ]
    assert ed6["mode"] == "inkscape"
    assert ed6["inkscape_source"] == "figures/inkscape/figure16.svg"


def test_city_pipeline_has_no_visualization_dependency():
    package = Path(__file__).parents[1] / "src" / "drone_emission"
    paths = [package / "main.py"]
    for name in ("preprocessing", "buildings", "demand", "roadnet", "emission"):
        paths.extend((package / name).glob("*.py"))
    violations = []
    for path in paths:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            module = getattr(node, "module", "") or ""
            if module.startswith("drone_emission.visualization"):
                violations.append(f"{path}:{node.lineno}")
    assert violations == []


def test_active_package_does_not_import_archive_or_inject_paths():
    package = Path(__file__).parents[1] / "src" / "drone_emission"
    violations = []
    for path in package.rglob("*.py"):
        if "archive" in path.parts:
            continue
        source = path.read_text(encoding="utf-8")
        if "drone_emission.archive" in source or "sys.path.insert" in source:
            violations.append(str(path))
    assert violations == []


def test_active_visualization_contract_mentions_no_placeholders():
    package = Path(__file__).parents[1] / "src" / "drone_emission" / "visualization"
    violations = []
    for path in package.rglob("*.py"):
        source = path.read_text(encoding="utf-8").lower()
        if "placeholder" in source:
            violations.append(str(path))
    assert violations == []


def test_shell_declares_single_global_memory_contract():
    root = Path(__file__).parents[1]
    source = (root / "entrypoint.sh").read_text(encoding="utf-8")
    assert 'DRONE_EMISSION_MEMORY_MAX_GB:-96' in source
    assert 'DRONE_EMISSION_SYSTEM_RESERVE_GB:-12' in source
    assert 'DRONE_EMISSION_CITY_RESERVATION_GB:-21' in source
    assert source.count("systemd-run --user --scope") == 1
    assert 'MemoryMax=${MEMORY_MAX_GB}G' in source
    assert 'MemorySwapMax=0' in source
    assert "command -v jq" not in source


def test_docs_paper_preserves_deferred_fig3_outside_manifest():
    root = Path(__file__).parents[1]
    manifest = json.loads(
        (root / "src" / "drone_emission" / "paper_figures.json").read_text()
    )
    declared = {Path(item["publish"]).name for item in manifest["figures"]}
    paper_dir = root / "docs" / "paper" / "figures"
    if paper_dir.is_dir():
        existing = {path.name for path in paper_dir.glob("*.pdf")}
        assert declared <= existing
        assert existing - declared == {"fig3_city_deployment_screening.pdf"}
    else:
        assert all(
            item["publish"].startswith("docs/paper/figures/")
            for item in manifest["figures"]
        )


def test_docs_and_agent_guides_do_not_reference_retired_entry_points():
    root = Path(__file__).parents[1]
    targets = {
        root / "README.md": (
            "python -m visualization.paper",
            "visualization.paper",
            "visualization.analysis",
            "visualization.emission",
        ),
        root / "docs" / "architecture.md": (
            "python -m visualization.paper",
            "visualization.paper",
            "visualization.analysis",
            "visualization.emission",
        ),
        root / "AGENTS.md": (
            "python -m visualization.paper",
            "visualization.paper",
            "visualization.analysis",
            "visualization.emission",
            "cd src/drone_emission && python main.py --region",
            "./entrypoint.sh 4 --all",
            "./postprocess.sh local",
            "./run_sensitivity.sh",
            "python main.py --region <region> --env local --no-fetch",
        ),
    }
    violations = []
    for path, forbidden in targets.items():
        source = path.read_text(encoding="utf-8")
        for needle in forbidden:
            if needle in source:
                violations.append(f"{path}: {needle}")
    assert violations == []
