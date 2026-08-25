"""Formal contribution renderers renderers used only by the manifest registry."""

from contextlib import contextmanager
import json
import os
import pickle
import time

import pandas as pd


CONTRIBUTION_STUDY_CITY_COUNT = 314
CONTRIBUTION_EXPECTED_FALLBACK_GEF_REGION_KEYS = {
    "aomentebiexingzhengqu",
    "changdou",
    "lasa",
    "linzhi",
    "shannan",
    "xianggangtebiexingzhengqu",
}
BREAK_EVEN_MARCHING_CUBES_SUBFIGURE_GROUP = (
    "break_even_contribution_decomposition_marching_cubes_subfigures"
)
BREAK_EVEN_PHASE_CACHE_VERSION = 9
CONTRIBUTION_DECOMPOSITION_FULL_CACHE_VERSION = 6
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


def _render_contribution_decomposition_city_b(
    data_dir,
    out,
    for_latex,
    plot_title,
    city_df,
    exp_config,
):
    """Render the full-data selected-city contribution-decomposition figure."""
    from drone_emission.visualization.contribution_break_even import (
        CONTRIBUTION_CITY_B_REGION_KEYS,
        plot_contribution_decomposition_city_b,
    )

    input_region_keys = city_df["region_key"].astype(str)
    duplicate_input_keys = sorted(
        input_region_keys[input_region_keys.duplicated()].unique()
    )
    study_region_keys = set(input_region_keys)
    if (
        len(city_df) != CONTRIBUTION_STUDY_CITY_COUNT
        or len(study_region_keys) != CONTRIBUTION_STUDY_CITY_COUNT
        or duplicate_input_keys
    ):
        raise RuntimeError(
            "contribution_decomposition_city_b requires exactly "
            f"{CONTRIBUTION_STUDY_CITY_COUNT} unique study cities; "
            f"received rows={len(city_df)}, unique={len(study_region_keys)}, "
            f"duplicate_keys={duplicate_input_keys}"
        )

    named_path = os.path.join(data_dir, "all_sensitivity_named.csv")
    named_df = None
    if os.path.exists(named_path):
        named_df = pd.read_csv(named_path)
    else:
        print(
            "contribution_decomposition_city_b: missing "
            f"{named_path}; thresholds will fall back to sampled sweeps"
        )

    enriched_path = os.path.join(data_dir, "all_region_stats_enriched.csv")
    if not os.path.exists(enriched_path):
        raise RuntimeError(
            "contribution_decomposition_city_b missing "
            f"{enriched_path}; run tools.enrich_stats first"
        )

    enriched_df = pd.read_csv(enriched_path)
    gdp_columns = (
        "region_key",
        "gdp_100m",
        "gdp_source_year",
        "source_url_gdp",
    )
    required_gdp_columns = (*gdp_columns, "region_level")
    missing_gdp_columns = sorted(
        set(required_gdp_columns) - set(enriched_df.columns)
    )
    if missing_gdp_columns:
        raise RuntimeError(
            "contribution_decomposition_city_b enriched city stats "
            f"missing columns {missing_gdp_columns}"
        )

    gdp_df = enriched_df[list(gdp_columns)].drop_duplicates("region_key")
    gdp_df["gdp_100m"] = pd.to_numeric(gdp_df["gdp_100m"], errors="coerce")
    gdp_df["gdp_source_year"] = pd.to_numeric(
        gdp_df["gdp_source_year"],
        errors="coerce",
    )
    all_study_gdp_df = enriched_df[
        enriched_df["region_level"].astype(str).eq("city")
        & enriched_df["region_key"].astype(str).isin(study_region_keys)
    ][list(gdp_columns)].drop_duplicates("region_key")
    all_study_gdp_df["gdp_100m"] = pd.to_numeric(
        all_study_gdp_df["gdp_100m"],
        errors="coerce",
    )
    all_study_gdp_df["gdp_source_year"] = pd.to_numeric(
        all_study_gdp_df["gdp_source_year"],
        errors="coerce",
    )
    gdp_region_keys = set(all_study_gdp_df["region_key"].astype(str))
    if gdp_region_keys != study_region_keys:
        raise RuntimeError(
            "contribution_decomposition_city_b GDP coverage does not match "
            f"the {CONTRIBUTION_STUDY_CITY_COUNT}-city study scope; "
            f"missing={sorted(study_region_keys - gdp_region_keys)}, "
            f"extra={sorted(gdp_region_keys - study_region_keys)}"
        )
    missing_all_study_gdp = all_study_gdp_df[
        all_study_gdp_df["gdp_100m"].isna()
        | (all_study_gdp_df["gdp_100m"] <= 0)
        | all_study_gdp_df["gdp_source_year"].isna()
    ]
    if not missing_all_study_gdp.empty:
        missing_keys = missing_all_study_gdp["region_key"].astype(str).tolist()
        raise RuntimeError(
            "contribution_decomposition_city_b GDP or source year missing "
            f"for study cities {missing_keys}"
        )

    gdp_min_idx = all_study_gdp_df["gdp_100m"].idxmin()
    gdp_max_idx = all_study_gdp_df["gdp_100m"].idxmax()
    gdp_min_row = all_study_gdp_df.loc[gdp_min_idx]
    gdp_max_row = all_study_gdp_df.loc[gdp_max_idx]
    all_study_gdp_range = {
        "scope": f"All {CONTRIBUTION_STUDY_CITY_COUNT} study cities",
        "city_count": int(len(all_study_gdp_df)),
        "gdp_min_100m": float(gdp_min_row["gdp_100m"]),
        "gdp_max_100m": float(gdp_max_row["gdp_100m"]),
        "gdp_min_region_key": str(gdp_min_row["region_key"]),
        "gdp_max_region_key": str(gdp_max_row["region_key"]),
        "source_year_min": int(all_study_gdp_df["gdp_source_year"].min()),
        "source_year_max": int(all_study_gdp_df["gdp_source_year"].max()),
    }
    city_df = city_df.drop(
        columns=["gdp_100m", "gdp_source_year", "source_url_gdp"],
        errors="ignore",
    ).merge(gdp_df, on="region_key", how="left")

    prepared = _prepare_contribution_decomposition_full(
        city_df=city_df,
        data_dir=data_dir,
        named_df=named_df,
        exp_config=exp_config,
    )
    if prepared is None:
        raise RuntimeError(
            "contribution_decomposition_city_b preparation returned no data"
        )
    sampling = prepared.get("sampling", {})
    if (
        sampling.get("max_sample_rows_per_city") is not None
        or sampling.get("mode") != "full_real_data"
    ):
        raise RuntimeError(
            "contribution_decomposition_city_b requires full OD data; "
            f"sampling={sampling}"
        )

    processed_region_keys = {
        str(city["region_key"]) for city in prepared.get("cities", [])
    }
    if processed_region_keys != study_region_keys:
        raise RuntimeError(
            "contribution_decomposition_city_b processed-city coverage does "
            f"not match the {CONTRIBUTION_STUDY_CITY_COUNT}-city study scope; "
            f"missing={sorted(study_region_keys - processed_region_keys)}, "
            f"extra={sorted(processed_region_keys - study_region_keys)}, "
            f"skipped={prepared.get('skipped_cities', [])}"
        )
    missing_selected_region_keys = sorted(
        set(CONTRIBUTION_CITY_B_REGION_KEYS) - processed_region_keys
    )
    if missing_selected_region_keys:
        raise RuntimeError(
            "contribution_decomposition_city_b representative-city coverage "
            f"is incomplete; missing={missing_selected_region_keys}"
        )

    gdp_by_region_key = gdp_df.set_index("region_key").to_dict("index")
    for city in prepared.get("cities", []):
        region_key = str(city["region_key"])
        gdp = gdp_by_region_key[region_key]
        city["gdp_100m"] = float(gdp["gdp_100m"])
        city["gdp_source_year"] = int(gdp["gdp_source_year"])
        city["source_url_gdp"] = (
            str(gdp["source_url_gdp"])
            if pd.notna(gdp["source_url_gdp"])
            else ""
        )

    fallback_region_keys = sorted(
        city_df.loc[
            city_df["gef_source"].eq("fallback_0.5600"),
            "region_key",
        ].astype(str)
    )
    if set(fallback_region_keys) != CONTRIBUTION_EXPECTED_FALLBACK_GEF_REGION_KEYS:
        raise RuntimeError(
            "contribution_decomposition_city_b fallback-GEF coverage changed; "
            f"expected={sorted(CONTRIBUTION_EXPECTED_FALLBACK_GEF_REGION_KEYS)}, "
            f"actual={fallback_region_keys}"
        )
    table_region_keys = sorted(
        city_df.loc[
            city_df["gef_source"].eq("province_table"),
            "region_key",
        ].astype(str)
    )
    prepared["study_scope"] = {
        "label": f"All {CONTRIBUTION_STUDY_CITY_COUNT} study cities",
        "expected_city_count": CONTRIBUTION_STUDY_CITY_COUNT,
        "region_keys": sorted(study_region_keys),
    }
    prepared["gef_coverage"] = {
        "province_table_city_count": len(table_region_keys),
        "province_table_region_keys": table_region_keys,
        "fallback_city_count": len(fallback_region_keys),
        "fallback_region_keys": fallback_region_keys,
        "fallback_gef_kg_co2e_per_kwh": 0.5600,
    }
    prepared["all_study_city_gdp_range"] = all_study_gdp_range

    plot_contribution_decomposition_city_b(
        prepared,
        save_path=os.path.join(out, "contribution_decomposition_city_b.png"),
        for_latex=for_latex,
        plot_title=plot_title,
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
                print(
                    "break_even_phase_diagram: city GEF data unavailable; "
                    "panel c will show a placeholder"
                )
    else:
        print(
            f"break_even_phase_diagram: missing {stats_path}; "
            "panel c will show a placeholder"
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


def _prepare_contribution_decomposition_full(
    data_dir,
    city_df,
    named_df,
    exp_config,
):
    """Prepare full-data contribution decomposition with cache."""
    from drone_emission.visualization.contribution_break_even import (
        CONTRIBUTION_SAMPLE_SEED,
        _resolve_contribution_od_path,
        prepare_contribution_decomposition_data,
    )
    from drone_emission.utils.config import get_cache_dir

    stats_path = os.path.join(data_dir, "all_region_stats.csv")
    gef_path = os.path.join(data_dir, "gef_china.csv")
    named_path = os.path.join(data_dir, "all_sensitivity_named.csv")
    city_rows = city_df.copy()
    required_cols = {"region_key", "total_ebike_lifecycle", "gef"}
    od_files = {}
    city_gef_values = []
    if required_cols.issubset(city_rows.columns):
        for col in ("total_ebike_lifecycle", "gef"):
            city_rows[col] = pd.to_numeric(city_rows[col], errors="coerce")
        city_rows = city_rows[
            city_rows["total_ebike_lifecycle"].notna()
            & (city_rows["total_ebike_lifecycle"] > 0)
            & city_rows["gef"].notna()
        ].copy()
        city_gef_values = [
            (str(row.region_key), float(row.gef))
            for row in city_rows.sort_values("region_key")[
                ["region_key", "gef"]
            ].itertuples(index=False)
        ]
        for region_key in sorted(city_rows["region_key"].dropna().astype(str).unique()):
            od_path = _resolve_contribution_od_path(data_dir, region_key)
            if od_path:
                od_files[region_key] = _file_cache_signature(od_path)

    figure_name = "break_even_contribution_decomposition"
    cache_path = os.path.join(
        get_cache_dir("visualization", "paper"),
        f"{figure_name}_contribution_full_prepared.pkl",
    )
    cache_meta = {
        "version": CONTRIBUTION_DECOMPOSITION_FULL_CACHE_VERSION,
        "figure_name": figure_name,
        "stats_file": _optional_file_cache_signature(stats_path),
        "gef_file": _optional_file_cache_signature(gef_path),
        "named_file": _optional_file_cache_signature(named_path),
        "od_files": od_files,
        "city_gef_values": city_gef_values,
        "kappa": _get_ebike_kappa(exp_config),
        "gef_fallback": 0.5600,
        "max_sample_rows": None,
        "sample_seed": CONTRIBUTION_SAMPLE_SEED,
    }
    prepared = None
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "rb") as f:
                cache = pickle.load(f)
            if cache.get("meta") == cache_meta:
                print(f"{figure_name}: loading full contribution cache {cache_path}")
                prepared = cache["prepared"]
            else:
                print(f"{figure_name}: full contribution cache is stale; rebuilding")
        except Exception as exc:
            print(
                f"{figure_name}: failed to load full contribution cache "
                f"({exc}); rebuilding"
            )

    if prepared is None:
        prepared = prepare_contribution_decomposition_data(
            city_df=city_df,
            data_dir=data_dir,
            named_df=named_df,
            exp_config=exp_config,
            max_sample_rows=None,
        )
        if prepared is None:
            return None
        with open(cache_path, "wb") as f:
            pickle.dump(
                {"meta": cache_meta, "prepared": prepared},
                f,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
        print(f"{figure_name}: wrote full contribution cache {cache_path}")

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
