"""Formal contribution renderers used only by the manifest registry."""

from contextlib import contextmanager
import math
import os
import pickle
import time

import pandas as pd

from drone_emission.paper_cohort import load_cohort_contract
from drone_emission.visualization.render_context import RenderContext

_COHORT = load_cohort_contract()
CONTRIBUTION_STUDY_CITY_COUNT = int(_COHORT["expected_city_count"])
CONTRIBUTION_GEF_FALLBACK = float(
    _COHORT["grid_emission_factor_fallback"]["kg_co2e_per_kwh"]
)
CONTRIBUTION_EXPECTED_FALLBACK_GEF_REGION_KEYS = set(
    _COHORT["grid_emission_factor_fallback"]["affected_region_keys"]
)
BREAK_EVEN_MARCHING_CUBES_SUBFIGURE_GROUP = (
    "break_even_contribution_decomposition_marching_cubes_subfigures"
)
BREAK_EVEN_PHASE_CACHE_VERSION = 9


def _get_ebike_kappa(exp_config):
    """Return the configured baseline e-bike distance factor."""
    emission_params = (exp_config or {}).get("emission_params", {})
    ebike_params = emission_params.get("ebike", {})
    return float(ebike_params.get("kappa", 1.0))


def _file_cache_signature(path):
    """Return file metadata used to invalidate a figure-data cache."""
    stat = os.stat(path)
    return {
        "path": os.path.abspath(path),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def _optional_file_cache_signature(path):
    """Return cache signature for an optional file, or None when absent."""
    if not path or not os.path.exists(path):
        return None
    return _file_cache_signature(path)


def _prepared_contribution_decomposition(summary: pd.DataFrame) -> dict:
    """Validate the compact analysis table and adapt it for plotting."""
    from drone_emission.analysis.contribution_decomposition import (
        ANALYSIS_VERSION,
        COMPONENT_KEYS,
        CONSERVATION_RTOL,
        SUMMARY_COLUMNS,
    )
    from drone_emission.paper_cohort import cohort_hash, locked_region_keys

    missing = sorted(set(SUMMARY_COLUMNS) - set(summary.columns))
    if missing:
        raise ValueError(f"contribution analysis table missing columns: {missing}")
    region_keys = summary["region_key"].astype(str)
    duplicates = sorted(region_keys[region_keys.duplicated()].unique())
    actual = set(region_keys)
    expected = set(locked_region_keys())
    if len(summary) != len(expected) or actual != expected or duplicates:
        raise RuntimeError(
            "contribution analysis table does not match the locked cohort: "
            f"rows={len(summary)}, missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}, duplicates={duplicates}"
        )
    if set(summary["analysis_version"].astype(str)) != {ANALYSIS_VERSION}:
        raise RuntimeError("contribution analysis version is stale; rerun analyze")
    if set(summary["paper_cohort_sha256"].astype(str)) != {cohort_hash()}:
        raise RuntimeError("contribution analysis belongs to a different cohort")

    numeric_columns = [
        "gef",
        "baseline_ebike_lifecycle_kg",
        "valid_od_rows",
        "drone_assigned_rows",
        "decomposition_net_kg",
        "oracle_net_kg",
        "max_abs_identity_error_g_per_order",
        "aggregate_identity_error_kg",
        *(f"{key}_kg" for key in COMPONENT_KEYS),
    ]
    values = summary.copy()
    for column in numeric_columns:
        values[column] = pd.to_numeric(values[column], errors="coerce")
    if values[numeric_columns].isna().any().any():
        raise ValueError("contribution analysis table contains non-numeric values")
    component_net = values[[f"{key}_kg" for key in COMPONENT_KEYS]].sum(axis=1)
    scale = values["oracle_net_kg"].abs().clip(lower=1.0)
    if (
        (component_net - values["oracle_net_kg"]).abs()
        > CONSERVATION_RTOL * scale
    ).any():
        raise RuntimeError("contribution analysis table fails the ledger identity")

    cities = []
    for row in values.sort_values("region_key").itertuples(index=False):
        component_kg = {
            key: float(getattr(row, f"{key}_kg")) for key in COMPONENT_KEYS
        }
        baseline = float(row.baseline_ebike_lifecycle_kg)
        cities.append(
            {
                "region_key": str(row.region_key),
                "region_name_en": str(row.region_name_en),
                "province": str(row.province),
                "gef": float(row.gef),
                "baseline_ebike_lifecycle_kg": baseline,
                "drone_assigned_sampled_rows": int(row.drone_assigned_rows),
                "sampled_drone_ratio_pct": float(row.drone_assigned_ratio_pct),
                "component_kg": component_kg,
                "component_pct_points": {
                    key: component_kg[key] / baseline * 100.0 for key in COMPONENT_KEYS
                },
                "net_saving_kg": float(row.oracle_net_kg),
                "net_saving_pct_points": float(row.oracle_net_kg) / baseline * 100.0,
                "sample": {
                    "valid_rows": int(row.valid_od_rows),
                    "sampled_rows": int(row.valid_od_rows),
                    "sampling_mode": "full",
                },
                "sum_check": {
                    "max_abs_error_g_per_order": float(
                        row.max_abs_identity_error_g_per_order
                    ),
                    "aggregate_error_kg": float(row.aggregate_identity_error_kg),
                },
            }
        )

    baseline_kg = float(
        math.fsum(city["baseline_ebike_lifecycle_kg"] for city in cities)
    )
    national = {
        f"{key}_kg": float(math.fsum(city["component_kg"][key] for city in cities))
        for key in COMPONENT_KEYS
    }
    for key in COMPONENT_KEYS:
        national[f"{key}_pp"] = national[f"{key}_kg"] / baseline_kg * 100.0
    national.update(
        {
            "baseline_ebike_lifecycle_kg": baseline_kg,
            "total_net_kg": float(math.fsum(city["net_saving_kg"] for city in cities)),
            "drone_assigned_sampled_rows": sum(
                city["drone_assigned_sampled_rows"] for city in cities
            ),
            "sum_check": {
                "max_abs_error_g_per_order": max(
                    city["sum_check"]["max_abs_error_g_per_order"] for city in cities
                ),
                "aggregate_error_kg": float(
                    math.fsum(city["sum_check"]["aggregate_error_kg"] for city in cities)
                ),
            },
        }
    )
    national["total_net_pp"] = national["total_net_kg"] / baseline_kg * 100.0
    national_component_net = math.fsum(
        national[f"{key}_kg"] for key in COMPONENT_KEYS
    )
    if abs(national_component_net - national["total_net_kg"]) > (
        CONSERVATION_RTOL * max(1.0, abs(national["total_net_kg"]))
    ):
        raise RuntimeError("national contribution table fails the ledger identity")
    fallback_keys = sorted(
        values.loc[values["gef_source"].eq("fallback_0.5366"), "region_key"].astype(str)
    )
    if set(fallback_keys) != CONTRIBUTION_EXPECTED_FALLBACK_GEF_REGION_KEYS:
        raise RuntimeError(
            "contribution fallback-GEF coverage changed: "
            f"expected={sorted(CONTRIBUTION_EXPECTED_FALLBACK_GEF_REGION_KEYS)}, "
            f"actual={fallback_keys}"
        )
    table_keys = sorted(
        values.loc[values["gef_source"].eq("province_table"), "region_key"].astype(str)
    )
    return {
        "cities": cities,
        "national": national,
        "sampling": {
            "max_sample_rows_per_city": None,
            "mode": "full_real_data",
            "distance_stratification": None,
        },
        "runtime": {
            "city_count_input": len(values),
            "city_count_processed": len(cities),
            "city_count_skipped": 0,
        },
        "skipped_cities": [],
        "study_scope": {
            "label": f"All {CONTRIBUTION_STUDY_CITY_COUNT} study cities",
            "expected_city_count": CONTRIBUTION_STUDY_CITY_COUNT,
            "region_keys": sorted(expected),
        },
        "gef_coverage": {
            "province_table_city_count": len(table_keys),
            "province_table_region_keys": table_keys,
            "fallback_city_count": len(fallback_keys),
            "fallback_region_keys": fallback_keys,
            "fallback_gef_kg_co2e_per_kwh": CONTRIBUTION_GEF_FALLBACK,
        },
    }


def _attach_contribution_population(
    prepared: dict,
    enriched: pd.DataFrame,
    selected_region_keys: tuple[str, ...],
) -> dict:
    """Attach cohort population metadata after strict key and coverage checks."""
    from drone_emission.paper_cohort import locked_region_keys

    required_columns = {
        "region_key",
        "population_10k",
        "population_source_year",
    }
    missing_columns = sorted(required_columns - set(enriched.columns))
    if missing_columns:
        raise ValueError(
            "cohort population input is missing columns: "
            f"{missing_columns}; rerun aggregate"
        )

    values = enriched.copy()
    if "region_level" in values.columns:
        values = values[values["region_level"].eq("city")].copy()
    values["region_key"] = values["region_key"].astype(str)
    expected = set(locked_region_keys())
    values = values[values["region_key"].isin(expected)].copy()
    region_keys = values["region_key"]
    duplicates = sorted(region_keys[region_keys.duplicated()].unique())
    actual = set(region_keys)
    if len(values) != len(expected) or actual != expected or duplicates:
        raise RuntimeError(
            "cohort population input does not match the locked cohort: "
            f"rows={len(values)}, missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}, duplicates={duplicates}"
        )

    values["population_10k"] = pd.to_numeric(
        values["population_10k"], errors="coerce"
    )
    invalid_population = values[
        values["population_10k"].isna() | (values["population_10k"] <= 0)
    ]["region_key"].tolist()
    if invalid_population:
        raise RuntimeError(
            "cohort population input requires positive population_10k for all "
            f"study cities; invalid={sorted(invalid_population)}"
        )

    selected_mask = values["region_key"].isin(selected_region_keys)
    selected = values.loc[selected_mask].copy()
    missing_selected = sorted(set(selected_region_keys) - set(selected["region_key"]))
    missing_source_year = sorted(
        selected.loc[
            selected["population_source_year"].isna()
            | selected["population_source_year"].astype(str).str.strip().eq(""),
            "region_key",
        ].tolist()
    )
    if (
        len(selected) != len(selected_region_keys)
        or missing_selected
        or missing_source_year
    ):
        raise RuntimeError(
            "representative-city population metadata is incomplete: "
            f"rows={len(selected)}, missing={missing_selected}, "
            f"missing_source_year={missing_source_year}"
        )

    population_by_key = values.set_index("region_key")
    for city in prepared.get("cities", []):
        region_key = str(city.get("region_key", ""))
        row = population_by_key.loc[region_key]
        source_year = row["population_source_year"]
        if hasattr(source_year, "item"):
            source_year = source_year.item()
        city["population_10k"] = float(row["population_10k"])
        city["population_million"] = float(row["population_10k"]) / 100.0
        city["population_source_year"] = source_year

    minimum = values.loc[values["population_10k"].idxmin()]
    maximum = values.loc[values["population_10k"].idxmax()]
    prepared["population_coverage"] = {
        "expected_city_count": len(expected),
        "actual_city_count": len(values),
        "positive_population_city_count": int((values["population_10k"] > 0).sum()),
        "population_source_year_city_count": int(
            values["population_source_year"].notna().sum()
        ),
        "representative_city_count_expected": len(selected_region_keys),
        "representative_city_count_actual": len(selected),
        "representative_source_year_count": int(
            selected["population_source_year"].notna().sum()
        ),
        "unit_input": "10,000 persons",
        "unit_plot": "million persons",
        "unit_conversion": "population_million = population_10k / 100",
        "all_study_city_range": {
            "minimum_region_key": str(minimum["region_key"]),
            "minimum_population_10k": float(minimum["population_10k"]),
            "minimum_population_million": float(minimum["population_10k"]) / 100.0,
            "maximum_region_key": str(maximum["region_key"]),
            "maximum_population_10k": float(maximum["population_10k"]),
            "maximum_population_million": float(maximum["population_10k"]) / 100.0,
        },
    }
    return prepared


def render_contribution_decomposition_city_b(context: RenderContext) -> None:
    """Render the standalone Fig. 5 contribution and population panels."""
    from drone_emission.visualization.contribution_break_even import (
        CONTRIBUTION_CITY_B_REGION_KEYS,
        plot_contribution_decomposition_city_b,
        plot_contribution_decomposition_city_population,
    )
    path = context.require_file(
        "all_city_contribution_decomposition.csv",
        stage="analyze --only contribution-decomposition",
    )
    prepared = _prepared_contribution_decomposition(pd.read_csv(path))
    population_path = context.require_file(
        "all_region_stats_enriched.csv",
        stage="aggregate",
    )
    prepared = _attach_contribution_population(
        prepared,
        pd.read_csv(population_path),
        CONTRIBUTION_CITY_B_REGION_KEYS,
    )
    processed_region_keys = {
        str(city["region_key"]) for city in prepared.get("cities", [])
    }
    missing_selected_region_keys = sorted(
        set(CONTRIBUTION_CITY_B_REGION_KEYS) - processed_region_keys
    )
    if missing_selected_region_keys:
        raise RuntimeError(
            "contribution_decomposition_city_b representative-city coverage "
            f"is incomplete; missing={missing_selected_region_keys}"
        )

    plot_contribution_decomposition_city_b(
        prepared,
        save_path=str(context.output("contribution_decomposition_city_b.pdf")),
        figsize=(132.0 / 25.4, 190.5 / 25.4),
        for_latex=True,
        plot_title=False,
    )
    plot_contribution_decomposition_city_population(
        prepared,
        save_path=str(
            context.output("contribution_decomposition_city_population.pdf")
        ),
        figsize=(44.0 / 25.4, 190.5 / 25.4),
        for_latex=True,
        plot_title=False,
    )


def _prepare_break_even_phase_diagram(
    data_dir,
    exp_config,
):
    """Prepare lifecycle break-even / phase-diagram data with cache."""
    from drone_emission.visualization.contribution_break_even import (
        CONTRIBUTION_SAMPLE_LIMIT,
        CONTRIBUTION_SAMPLE_SEED,
        prepare_break_even_phase_diagram_data,
    )

    city_df = None
    stats_path = os.path.join(data_dir, "all_region_stats.csv")
    if os.path.exists(stats_path):
        city_df = pd.read_csv(stats_path)
        if "region_level" in city_df.columns:
            city_df = city_df[city_df["region_level"] == "city"].copy()

        has_gef = "gef" in city_df.columns and city_df["gef"].notna().any()
        if not has_gef:
            gef_path = os.path.join(data_dir, "gef_china.csv")
            if "province" in city_df.columns and os.path.exists(gef_path):
                gef_df = pd.read_csv(gef_path, encoding="gbk")
                gef_df.columns = ["province", "gef"]
                city_df = city_df.merge(gef_df, on="province", how="left")
            else:
                raise RuntimeError(
                    "Break-even render requires city GEF data. Run "
                    "`python -m drone_emission aggregate` first."
                )
    else:
        raise FileNotFoundError(
            f"Missing render input: {stats_path}. Run "
            "`python -m drone_emission aggregate` first."
        )

    from drone_emission.utils.config import get_cache_dir

    figure_name = "break_even_phase_diagram"
    gef_path = os.path.join(data_dir, "gef_china.csv")
    cache_path = os.path.join(
        get_cache_dir("visualization", "paper"),
        f"{figure_name}_prepared.pkl",
    )
    cache_meta = {
        "version": BREAK_EVEN_PHASE_CACHE_VERSION,
        "figure_name": figure_name,
        "stats_file": _optional_file_cache_signature(stats_path),
        "gef_file": _optional_file_cache_signature(gef_path),
        "kappa": _get_ebike_kappa(exp_config),
        "max_sample_rows": CONTRIBUTION_SAMPLE_LIMIT,
        "sample_seed": CONTRIBUTION_SAMPLE_SEED,
    }
    prepared = None
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "rb") as f:
                cache = pickle.load(f)
            if cache.get("meta") == cache_meta:
                print(f"{figure_name}: loading prepared data cache {cache_path}")
                prepared = cache["prepared"]
            else:
                print(f"{figure_name}: prepared data cache is stale; rebuilding")
        except Exception as exc:
            print(f"{figure_name}: failed to load prepared cache ({exc}); rebuilding")

    if prepared is None:
        prepared = prepare_break_even_phase_diagram_data(
            data_dir=data_dir,
            city_df=city_df,
            exp_config=exp_config,
        )
        with open(cache_path, "wb") as f:
            pickle.dump(
                {"meta": cache_meta, "prepared": prepared},
                f,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
        print(f"{figure_name}: wrote prepared data cache {cache_path}")

    return prepared


def _render_break_even_marching_cubes_subfigures(
    data_dir,
    out,
    for_latex,
    plot_title,
    exp_config,
    only,
):
    """Render selected standalone panels from the marching-cubes composite."""
    from drone_emission.visualization.contribution_break_even import (
        plot_break_even_contribution_decomposition_marching_cubes_subfigures,
    )

    if only is None or BREAK_EVEN_MARCHING_CUBES_SUBFIGURE_GROUP in only:
        selected_panels = ("a", "b", "c")
    else:
        selected_panels = tuple(
            panel
            for panel in ("a", "b", "c")
            if (
                "break_even_contribution_decomposition_"
                f"marching_cubes_{panel}"
            )
            in only
        )
    if not selected_panels:
        return

    break_even_prepared = _prepare_break_even_phase_diagram(
        data_dir=data_dir,
        exp_config=exp_config,
    )
    plot_break_even_contribution_decomposition_marching_cubes_subfigures(
        break_even_prepared,
        save_dir=out,
        for_latex=for_latex,
        plot_title=plot_title,
        selected_panels=selected_panels,
    )
