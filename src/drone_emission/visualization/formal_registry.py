"""Exact request-to-renderer registry for the locked paper manifest."""

from __future__ import annotations

from collections.abc import Callable
import json

import pandas as pd

from .render_context import RenderContext

Renderer = Callable[[RenderContext], None]


def _city_stats(context: RenderContext) -> pd.DataFrame:
    frame = pd.read_csv(context.require_file("all_region_stats.csv", stage="aggregate"))
    if "region_level" in frame.columns:
        frame = frame[frame["region_level"].eq("city")].copy()
    enriched_path = context.data_root / "all_region_stats_enriched.csv"
    if enriched_path.is_file():
        enriched = pd.read_csv(enriched_path)
        available = [
            name for name in ("region_key", "grid_emission_factor", "gef_source")
            if name in enriched.columns
        ]
        if "region_key" in available:
            frame = frame.drop(
                columns=["grid_emission_factor", "gef", "gef_source"],
                errors="ignore",
            ).merge(
                enriched[available].drop_duplicates("region_key"),
                on="region_key",
                how="left",
            )
    from drone_emission.paper_cohort import load_cohort_contract

    fallback = load_cohort_contract()["grid_emission_factor_fallback"]
    fallback_keys = set(fallback["affected_region_keys"])
    fallback_mask = frame["region_key"].astype(str).isin(fallback_keys)
    if "grid_emission_factor" not in frame:
        frame["grid_emission_factor"] = float("nan")
    if "gef_source" not in frame:
        frame["gef_source"] = "province_table"
    frame.loc[fallback_mask, "grid_emission_factor"] = float(
        fallback["kg_co2e_per_kwh"]
    )
    frame.loc[fallback_mask, "gef_source"] = "fallback_0.5366"
    frame["gef"] = pd.to_numeric(frame["grid_emission_factor"], errors="coerce")
    from drone_emission.paper_cohort import locked_region_keys

    expected = set(locked_region_keys())
    actual = set(frame["region_key"].astype(str))
    if actual != expected or len(frame) != len(expected):
        raise RuntimeError(
            "formal city statistics do not match the locked paper cohort: "
            f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
        )
    return frame


def _render_config_input(context: RenderContext, request: str) -> None:
    from . import config_city_inputs as renderer

    common = dict(
        data_dir=str(context.data_root),
        out=str(context.subfigure_dir),
        for_latex=True,
        plot_title=False,
    )
    functions = {
        "shanghai_merchant_monthly_sales": renderer._render_shanghai_merchant_monthly_sales,
        "shanghai_building_footprint_heatmap": renderer._render_shanghai_building_footprint_heatmap,
        "beijing_road_network_subfigure": renderer._render_beijing_road_network_subfigure,
        "shanghai_building_height_merchant": renderer._render_shanghai_building_height_merchant,
        "distance_comparison_single_panel": renderer._render_distance_comparison_single_panel,
    }
    extra = {}
    if request in {"beijing_road_network_subfigure", "shanghai_building_height_merchant"}:
        extra["graph_cache"] = {}
    if request == "distance_comparison_single_panel":
        extra.update(graph_cache={}, router_cache={})
    functions[request](**common, **extra)


def _render_province_gef(context: RenderContext) -> None:
    from .regions import plot_province_gef_china_map

    plot_province_gef_china_map(
        gef_path=str(context.require_file("gef_china.csv", stage="inputs")),
        save_path=str(context.output("province_gef_china_map.pdf")),
        for_latex=True,
        plot_title=False,
    )


def _render_detour_concept(context: RenderContext) -> None:
    from .route_concepts import plot_detour_ratio_concept

    plot_detour_ratio_concept(
        save_path=str(context.output("detour_ratio_concept.pdf")),
        for_latex=True,
        plot_title=False,
    )


def _render_median_phase(context: RenderContext, request: str) -> None:
    from .config_city_inputs import _render_beijing_median_trip_phase_emissions

    _render_beijing_median_trip_phase_emissions(
        data_dir=str(context.data_root),
        out=str(context.subfigure_dir),
        for_latex=True,
        plot_title=False,
        only={request},
    )


def _render_province_lifecycle(context: RenderContext) -> None:
    from .city_priorities import plot_province_total_ebike_lifecycle

    plot_province_total_ebike_lifecycle(
        _city_stats(context),
        save_path=str(context.output("province_total_ebike_lifecycle")),
        for_latex=True,
        plot_title=False,
    )


def _render_priority(context: RenderContext, request: str) -> None:
    from . import city_priorities as renderer

    frame = _city_stats(context)
    save_path = context.output(
        request if request == "city_reduction_rank_all_macro_region" else f"{request}.pdf"
    )
    function = getattr(renderer, f"plot_{request}")
    kwargs = dict(save_path=str(save_path), for_latex=True, plot_title=False)
    if request == "regions_map_oracle_reduction_rank_map_only":
        kwargs["stats_csv_path"] = str(context.data_root / "all_region_stats.csv")
        function(df_stats=frame, **kwargs)
    else:
        function(frame, **kwargs)


def _render_gef_detour(context: RenderContext) -> None:
    from .city_gef_detour import _render_city_gef_context_detour_summary

    stats = _city_stats(context)
    enriched = pd.read_csv(context.require_file("all_region_stats_enriched.csv", stage="aggregate"))
    enriched = enriched.drop(
        columns=["grid_emission_factor", "gef", "gef_source"], errors="ignore"
    ).merge(
        stats[["region_key", "grid_emission_factor", "gef", "gef_source"]],
        on="region_key",
        how="inner",
    )
    _render_city_gef_context_detour_summary(
        data_dir=str(context.data_root),
        out=str(context.subfigure_dir),
        for_latex=True,
        plot_title=False,
        df_enriched=enriched,
        city_df=stats,
        exp_config=context.experiment,
    )


def _render_contribution(context: RenderContext) -> None:
    from .contribution_renderers import render_contribution_decomposition_city_b

    render_contribution_decomposition_city_b(context)


def _render_scenario(context: RenderContext, request: str) -> None:
    from . import national_scenarios as renderer

    named = context.require_file("all_sensitivity_named.csv", stage="analyze")
    stats = context.require_file("all_region_stats.csv", stage="aggregate")
    getattr(renderer, f"plot_{request}")(
        named_csv_path=str(named), stats_csv_path=str(stats),
        data_dir=str(context.data_root), exp_config=context.experiment,
        save_path=str(context.output(f"{request}.pdf")),
        for_latex=True, plot_title=False,
    )


def _render_threshold_resilience(context: RenderContext) -> None:
    from drone_emission.paper_cohort import locked_region_keys
    from .threshold_resilience import (
        plot_threshold_resilience_national_kdes,
        plot_threshold_resilience_national_subfigures,
    )

    summary = pd.read_csv(context.require_file("all_threshold_resilience.csv", stage="analyze"))
    expected_cities = len(locked_region_keys())
    plot_threshold_resilience_national_subfigures(
        summary,
        save_path=str(context.output("threshold_resilience_national_overview.pdf")),
        for_latex=True,
        plot_title=False,
        expected_cities=expected_cities,
    )
    plot_threshold_resilience_national_kdes(
        summary,
        str(context.subfigure_dir),
        expected_cities=expected_cities,
    )


def _render_attribution(context: RenderContext) -> None:
    from .city_attribution_figures import plot_r2_population_reduction_contrast

    root = context.work_root / "analysis" / "city_attribution"
    prepared = root / "r2_city_attribution_data.csv"
    report_path = root / "r2_city_attribution_analysis_report.json"
    missing = [str(path) for path in (prepared, report_path) if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing city-attribution render input(s): " + ", ".join(missing)
            + ". Run `python -m drone_emission analyze` first."
        )
    frame = pd.read_csv(prepared)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    plot_r2_population_reduction_contrast(
        frame, report, save_dir=str(context.subfigure_dir),
        for_latex=True, plot_title=False,
    )


def _render_break_even(context: RenderContext) -> None:
    from .contribution_renderers import _render_break_even_marching_cubes_subfigures

    _render_break_even_marching_cubes_subfigures(
        data_dir=str(context.data_root), out=str(context.subfigure_dir),
        for_latex=True, plot_title=False, exp_config=context.experiment,
        only={"break_even_contribution_decomposition_marching_cubes_subfigures"},
    )


def _render_distance_threshold(context: RenderContext) -> None:
    from .distance_threshold_inputs import (
        _load_oracle_results, _load_summary, _resolve_summary_path,
    )
    from .distance_threshold import (
        plot_distance_threshold_city_landscape,
        plot_distance_threshold_oracle_performance,
    )

    summary_path = _resolve_summary_path(context.data_root, None)
    summary = _load_summary(summary_path, context.data_root)
    oracle = _load_oracle_results(context.data_root)
    plot_distance_threshold_city_landscape(
        summary, oracle, save_path=str(context.output("distance_threshold_city_landscape.pdf")),
        for_latex=True, plot_title=False,
    )
    plot_distance_threshold_oracle_performance(
        summary, oracle, save_path=str(context.output("distance_threshold_oracle_performance.pdf")),
        for_latex=True, plot_title=False,
    )


def _render_observed(context: RenderContext) -> None:
    from .observed_distance import _render_observed_delivery_distance_distribution

    _render_observed_delivery_distance_distribution(
        data_dir=str(context.data_root), out=str(context.subfigure_dir),
        for_latex=True, plot_title=False,
    )


REQUEST_RENDERERS: dict[str, Renderer] = {
    "shanghai_merchant_monthly_sales": lambda c: _render_config_input(c, "shanghai_merchant_monthly_sales"),
    "shanghai_building_footprint_heatmap": lambda c: _render_config_input(c, "shanghai_building_footprint_heatmap"),
    "province_gef_china_map": _render_province_gef,
    "beijing_road_network_subfigure": lambda c: _render_config_input(c, "beijing_road_network_subfigure"),
    "shanghai_building_height_merchant": lambda c: _render_config_input(c, "shanghai_building_height_merchant"),
    "distance_comparison_single_panel": lambda c: _render_config_input(c, "distance_comparison_single_panel"),
    "detour_ratio_concept": _render_detour_concept,
    "median_trip_production_emissions": lambda c: _render_median_phase(c, "median_trip_production_emissions"),
    "median_trip_use_phase_emissions": lambda c: _render_median_phase(c, "median_trip_use_phase_emissions"),
    "median_trip_eol_emissions": lambda c: _render_median_phase(c, "median_trip_eol_emissions"),
    "province_total_ebike_lifecycle": _render_province_lifecycle,
    "regions_map_oracle_reduction_rank_map_only": lambda c: _render_priority(c, "regions_map_oracle_reduction_rank_map_only"),
    "city_annual_absolute_reduction_distribution": lambda c: _render_priority(c, "city_annual_absolute_reduction_distribution"),
    "city_relative_lifecycle_reduction_distribution": lambda c: _render_priority(c, "city_relative_lifecycle_reduction_distribution"),
    "city_reduction_rank_all_macro_region": lambda c: _render_priority(c, "city_reduction_rank_all_macro_region"),
    "city_gef_context_detour_summary": _render_gef_detour,
    "contribution_decomposition_city_b": _render_contribution,
    "regions_map_pathway_scenarios": lambda c: _render_scenario(c, "regions_map_pathway_scenarios"),
    "threshold_resilience_national_overview": _render_threshold_resilience,
    "r2_population_reduction_contrast": _render_attribution,
    "break_even_contribution_decomposition_marching_cubes_subfigures": _render_break_even,
    "distance_threshold_policy_comparison": _render_distance_threshold,
    "regions_map_pathway_scenario_maps": lambda c: _render_scenario(c, "regions_map_pathway_scenario_maps"),
    "observed_delivery_distance_distribution": _render_observed,
}


def render_requests(context: RenderContext, requests: set[str]) -> None:
    """Render a manifest-derived request set with no implicit analysis."""
    unknown = requests.difference(REQUEST_RENDERERS)
    if unknown:
        raise ValueError(f"Unregistered formal render request(s): {sorted(unknown)}")
    for request in sorted(requests):
        REQUEST_RENDERERS[request](context)
