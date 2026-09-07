"""
排放分析可视化模块
"""
from .palette import opacity as paper_alpha

from .prepared_bundle import prepared_plot, prepared_data

from .palette import color as paper_color, colormap as paper_cmap

import json
import os
import time
import zlib
from functools import wraps
from typing import Dict

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from matplotlib.cm import ScalarMappable
from matplotlib.colors import BoundaryNorm, TwoSlopeNorm

from ._utils import _add_title_suffix
from .macro_regions import (
    CHINA_MACRO_REGION_SPECS,
    PROVINCE_TO_MACRO_REGION,
    normalize_province_name,
)


CONTRIBUTION_SAMPLE_LIMIT = 2000
CONTRIBUTION_SAMPLE_SEED = 42
CONTRIBUTION_FULL_CHUNK_ROWS = 250_000
CONTRIBUTION_COMPONENT_SPECS = (
    {
        "key": "embodied_balance",
        "label": "Embodied lifecycle balance",
        "short_label": "Embodied balance",
        "color": paper_color("contribution_break_even.4c78a8"),
    },
    {
        "key": "avoided_ebike_road_use",
        "label": "Avoided e-bike road operation",
        "short_label": "Avoided road operation",
        "color": paper_color("contribution_break_even.72b7b2"),
    },
    {
        "key": "drone_fixed_flight_penalty",
        "label": "Drone take-off / landing / hover",
        "short_label": "Fixed flight",
        "color": paper_color("contribution_break_even.f2a65a"),
    },
    {
        "key": "drone_cruise_penalty",
        "label": "Drone cruise",
        "short_label": "Cruise",
        "color": paper_color("contribution_break_even.e45756"),
    },
)
CONTRIBUTION_COMPONENT_KEYS = tuple(
    spec["key"] for spec in CONTRIBUTION_COMPONENT_SPECS
)
CONTRIBUTION_CITY_B_GROUPS = (
    ("north_china", ("beijing", "tianjin", "shijiazhuang", "taiyuan")),
    ("northeast_china", ("shenyang", "dalian", "changchun", "haerbin")),
    ("east_china", ("shanghai", "hangzhou", "nanjing", "ningbo")),
    ("central_china", ("zhengzhou", "wuhan", "changsha", "xiangyang")),
    ("south_china", ("guangzhou", "shenzhen", "nanning", "haikou")),
    ("southwest_china", ("chongqing", "chengdu", "guiyang", "kunming")),
    ("northwest_china", ("xian", "lanzhou", "xining", "wulumuqi")),
)
CONTRIBUTION_CITY_B_REGION_KEYS = tuple(
    region_key
    for _, region_keys in CONTRIBUTION_CITY_B_GROUPS
    for region_key in region_keys
)
CONTRIBUTION_CITY_B_DISPLAY_NAMES = {
    "haerbin": "Harbin",
    "wulumuqi": "Urumqi",
    "xian": "Xi'an",
}
CONTRIBUTION_CITY_PANEL_HEIGHT_MM = 170.0
CONTRIBUTION_CITY_AXES_BOTTOM_MM = 25.0
CONTRIBUTION_CITY_AXES_HEIGHT_MM = 125.73
CONTRIBUTION_CITY_LEGEND_ANCHOR_MM = 7.3
CONTRIBUTION_CITY_LEGEND_GAP_TARGET_MM = (2.5, 3.0)
CONTRIBUTION_CITY_FONT_FAMILY = "Arial"
BREAK_EVEN_REPRESENTATIVE_AIR_DISTANCE_KM = 2.0
BREAK_EVEN_CURVE_AIR_DISTANCES_KM = (0.5, 1.0, 2.0, 3.0, 5.0)
BREAK_EVEN_DETOUR_LIMITS = (1.0, 2.4)
BREAK_EVEN_GEF_LIMITS = (0.05, 0.80)


def _json_safe(value):
    """Convert numpy/pandas values into JSON-safe Python scalars."""
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return [_json_safe(v) for v in value.tolist()]
    if isinstance(value, (np.floating, float)):
        if not np.isfinite(value):
            return None
        return float(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if pd.isna(value):
        return None
    return value


def _report_number(value):
    """Return a JSON-safe scalar with NaN/Inf represented as None."""
    return _json_safe(value)


def _report_array(values):
    """Return a JSON-safe list from an array-like object."""
    return [_report_number(value) for value in values]


def _report_axis(ax):
    """Return figure-fraction position and data scale for an axis."""
    pos = ax.get_position()
    return {
        "bbox_figure_fraction": {
            "x0": _report_number(pos.x0),
            "y0": _report_number(pos.y0),
            "x1": _report_number(pos.x1),
            "y1": _report_number(pos.y1),
            "width": _report_number(pos.width),
            "height": _report_number(pos.height),
        },
        "xlim_data": _report_array(ax.get_xlim()),
        "ylim_data": _report_array(ax.get_ylim()),
        "xlabel": ax.get_xlabel(),
        "ylabel": ax.get_ylabel(),
        "xticks_data": _report_array(ax.get_xticks()),
        "yticks_data": _report_array(ax.get_yticks()),
    }


def _artist_bbox_mm(fig, artist) -> dict:
    """Return an artist bounding box in page millimetres from the lower left."""
    fig.canvas.draw()
    bbox_inches = artist.get_window_extent(
        renderer=fig.canvas.get_renderer()
    ).transformed(fig.dpi_scale_trans.inverted())
    return {
        "x0": _report_number(bbox_inches.x0 * 25.4),
        "y0": _report_number(bbox_inches.y0 * 25.4),
        "x1": _report_number(bbox_inches.x1 * 25.4),
        "y1": _report_number(bbox_inches.y1 * 25.4),
        "width": _report_number(bbox_inches.width * 25.4),
        "height": _report_number(bbox_inches.height * 25.4),
    }


def _selected_city_vertical_layout(figsize: tuple) -> dict:
    """Return shared physical vertical geometry for the two city panels."""
    page_height_mm = float(figsize[1]) * 25.4
    axes_top_mm = (
        CONTRIBUTION_CITY_AXES_BOTTOM_MM + CONTRIBUTION_CITY_AXES_HEIGHT_MM
    )
    if page_height_mm <= axes_top_mm:
        raise ValueError(
            "selected-city panel is too short for the locked axes geometry: "
            f"page_height_mm={page_height_mm:.3f}, axes_top_mm={axes_top_mm:.3f}"
        )
    return {
        "page_height_mm": page_height_mm,
        "axes_bottom_mm": CONTRIBUTION_CITY_AXES_BOTTOM_MM,
        "axes_top_mm": axes_top_mm,
        "axes_height_mm": CONTRIBUTION_CITY_AXES_HEIGHT_MM,
        "top_margin_mm": page_height_mm - axes_top_mm,
        "axes_bottom_fraction": CONTRIBUTION_CITY_AXES_BOTTOM_MM / page_height_mm,
        "axes_top_fraction": axes_top_mm / page_height_mm,
        "axes_height_fraction": (
            CONTRIBUTION_CITY_AXES_HEIGHT_MM / page_height_mm
        ),
        "alignment_contract": (
            "When equal-height PDF pages are top-aligned, both axes borders, "
            "y limits and all overall-plus-city row coordinates coincide."
        ),
    }


def _set_selected_city_axes_position(
    ax,
    figsize: tuple,
    *,
    left: float,
    right: float,
) -> dict:
    """Apply the shared physical vertical geometry to a city-panel axis."""
    layout = _selected_city_vertical_layout(figsize)
    ax.set_position(
        [
            left,
            layout["axes_bottom_fraction"],
            right - left,
            layout["axes_height_fraction"],
        ]
    )
    return layout


def _with_contribution_city_typography(plot_function):
    """Render a selected-city panel with explicit Arial typography."""

    @wraps(plot_function)
    def wrapped(*args, **kwargs):
        font_manager.findfont(
            font_manager.FontProperties(family=CONTRIBUTION_CITY_FONT_FAMILY),
            fallback_to_default=False,
        )
        with mpl.rc_context(
            {
                "font.family": CONTRIBUTION_CITY_FONT_FAMILY,
                "font.sans-serif": [CONTRIBUTION_CITY_FONT_FAMILY],
                "mathtext.fontset": "custom",
                "mathtext.rm": CONTRIBUTION_CITY_FONT_FAMILY,
                "mathtext.it": f"{CONTRIBUTION_CITY_FONT_FAMILY}:italic",
                "mathtext.bf": f"{CONTRIBUTION_CITY_FONT_FAMILY}:bold",
            }
        ):
            return plot_function(*args, **kwargs)

    return wrapped


def _axis_bbox_mm(ax, figsize: tuple) -> dict:
    """Return axis position in physical page millimetres."""
    position = ax.get_position()
    width_mm = float(figsize[0]) * 25.4
    height_mm = float(figsize[1]) * 25.4
    return {
        "x0": _report_number(position.x0 * width_mm),
        "y0": _report_number(position.y0 * height_mm),
        "x1": _report_number(position.x1 * width_mm),
        "y1": _report_number(position.y1 * height_mm),
        "width": _report_number(position.width * width_mm),
        "height": _report_number(position.height * height_mm),
    }


def _array_summary(values):
    """Return compact numeric summary stats for report payloads."""
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "min": None,
            "max": None,
            "q05": None,
            "q95": None,
        }
    q05, median, q95 = np.percentile(arr, [5, 50, 95])
    return {
        "count": int(arr.size),
        "mean": _report_number(arr.mean()),
        "median": _report_number(median),
        "min": _report_number(arr.min()),
        "max": _report_number(arr.max()),
        "q05": _report_number(q05),
        "q95": _report_number(q95),
    }


def _stable_seed(region_key: str, base_seed: int) -> int:
    """Return a deterministic per-city seed from a shared base seed."""
    return int((base_seed + zlib.crc32(region_key.encode("utf-8"))) % (2 ** 32))


def _resolve_contribution_od_path(data_dir: str, region_key: str) -> str | None:
    """Return the first available OD CSV path for a region key."""
    candidates = []
    for level in ("city", "district"):
        for key in (region_key, region_key.lower()):
            candidates.append(
                os.path.join(data_dir, level, key, f"{key}_od_pairs.csv")
            )
    seen = set()
    for path in candidates:
        if path in seen:
            continue
        seen.add(path)
        if os.path.exists(path):
            return path
    return None


def _distance_strata_codes(distance_values: np.ndarray, n_strata: int = 5) -> np.ndarray:
    """Return quantile-based distance strata codes for proportional sampling."""
    if len(distance_values) == 0:
        return np.zeros(0, dtype=int)
    series = pd.Series(distance_values, copy=False)
    finite = series[np.isfinite(series)]
    if finite.empty or finite.nunique() <= 1:
        return np.zeros(len(series), dtype=int)
    q = int(min(n_strata, len(finite), finite.nunique()))
    if q <= 1:
        return np.zeros(len(series), dtype=int)
    try:
        strata = pd.qcut(finite, q=q, labels=False, duplicates="drop")
    except ValueError:
        return np.zeros(len(series), dtype=int)
    codes = np.zeros(len(series), dtype=int)
    codes[finite.index.to_numpy(dtype=int)] = strata.to_numpy(dtype=int)
    return codes


def _allocate_stratified_sample_sizes(
    stratum_sizes: np.ndarray,
    sample_size: int,
) -> np.ndarray:
    """Allocate a proportional sample size across non-empty strata."""
    sizes = np.asarray(stratum_sizes, dtype=int)
    total = int(sizes.sum())
    if total <= 0 or sample_size >= total:
        return sizes.copy()

    raw = sample_size * sizes / total
    alloc = np.floor(raw).astype(int)
    nonempty = np.flatnonzero(sizes > 0)
    if sample_size >= len(nonempty):
        zero_alloc = nonempty[alloc[nonempty] == 0]
        alloc[zero_alloc] = 1

    max_keep = np.where(sizes > 0, sizes, 0)
    alloc = np.minimum(alloc, max_keep)

    remainder = int(sample_size - alloc.sum())
    fractions = raw - np.floor(raw)

    while remainder < 0:
        removable = np.flatnonzero(alloc > 1)
        if len(removable) == 0:
            removable = np.flatnonzero(alloc > 0)
        if len(removable) == 0:
            break
        order = removable[np.argsort(fractions[removable])]
        idx = int(order[0])
        alloc[idx] -= 1
        remainder += 1

    while remainder > 0:
        capacity = sizes - alloc
        candidates = np.flatnonzero(capacity > 0)
        if len(candidates) == 0:
            break
        order = candidates[np.argsort(-fractions[candidates])]
        idx = int(order[0])
        alloc[idx] += 1
        remainder -= 1

    return alloc


def _sample_city_valid_rows(
    valid_df: pd.DataFrame,
    sample_distance_km: np.ndarray,
    region_key: str,
    max_sample_rows: int | None,
    sample_seed: int,
) -> tuple[pd.DataFrame, dict]:
    """Sample valid OD rows with proportional distance stratification."""
    total_rows = len(valid_df)
    if total_rows == 0:
        return valid_df.copy(), {
            "valid_rows": 0,
            "sampled_rows": 0,
            "distance_strata": [],
            "sample_seed": int(sample_seed),
        }

    if max_sample_rows is None or total_rows <= max_sample_rows:
        sampled = valid_df.copy()
        sampled["_sample_expansion_weight"] = 1.0
        sampled["_sample_stratum"] = _distance_strata_codes(sample_distance_km)
        stratum_counts = (
            sampled["_sample_stratum"].value_counts().sort_index().to_dict()
        )
        return sampled, {
            "valid_rows": int(total_rows),
            "sampled_rows": int(total_rows),
            "distance_strata": [
                {
                    "stratum": int(code),
                    "valid_rows": int(count),
                    "sampled_rows": int(count),
                    "expansion_weight": 1.0,
                }
                for code, count in stratum_counts.items()
            ],
            "sample_seed": int(sample_seed),
            "sampling_mode": "full" if max_sample_rows is None else "sample_or_full",
        }

    strata = _distance_strata_codes(sample_distance_km)
    sampled_parts = []
    stratum_report = []
    rng = np.random.default_rng(_stable_seed(region_key, sample_seed))
    unique_codes = np.array(sorted(pd.unique(strata)), dtype=int)
    stratum_sizes = np.array([(strata == code).sum() for code in unique_codes], dtype=int)
    allocations = _allocate_stratified_sample_sizes(stratum_sizes, max_sample_rows)

    for code, stratum_size, draw_size in zip(unique_codes, stratum_sizes, allocations):
        indices = np.flatnonzero(strata == code)
        if draw_size <= 0 or len(indices) == 0:
            stratum_report.append(
                {
                    "stratum": int(code),
                    "valid_rows": int(stratum_size),
                    "sampled_rows": 0,
                    "expansion_weight": None,
                }
            )
            continue
        if draw_size >= len(indices):
            chosen = indices
        else:
            chosen = np.sort(rng.choice(indices, size=int(draw_size), replace=False))
        part = valid_df.iloc[chosen].copy()
        expansion_weight = float(stratum_size / len(chosen))
        part["_sample_expansion_weight"] = expansion_weight
        part["_sample_stratum"] = int(code)
        sampled_parts.append(part)
        stratum_report.append(
            {
                "stratum": int(code),
                "valid_rows": int(stratum_size),
                "sampled_rows": int(len(chosen)),
                "expansion_weight": expansion_weight,
            }
        )

    sampled = pd.concat(sampled_parts, ignore_index=True) if sampled_parts else valid_df.iloc[0:0].copy()
    return sampled, {
        "valid_rows": int(total_rows),
        "sampled_rows": int(len(sampled)),
        "distance_strata": stratum_report,
        "sample_seed": int(sample_seed),
    }


def _contribution_row_template(label: str) -> dict:
    """Create a zero-filled aggregate row for a plotted contribution group."""
    row = {"label": label, "sampled_drone_rows": 0}
    for key in CONTRIBUTION_COMPONENT_KEYS:
        row[f"{key}_kg"] = 0.0
        row[f"{key}_pp"] = 0.0
    row["total_net_kg"] = 0.0
    row["total_net_pp"] = 0.0
    return row


def _normalize_province_name(province: str) -> str:
    """Normalize province-level names before assigning macro-regions."""
    return normalize_province_name(province)


def _top_city_component_rows(
    city_reports: list[dict],
    national_baseline_kg: float,
    top_n: int = 20,
) -> list[dict]:
    """Return top-city contribution rows ranked by total sampled net reduction."""
    if national_baseline_kg <= 0:
        return []

    rows = []
    for city in city_reports:
        label = str(city.get("region_name_en", city.get("region_key", "")))
        row = _contribution_row_template(label)
        row["region_key"] = str(city.get("region_key", ""))
        row["sampled_drone_rows"] = int(city.get("drone_assigned_sampled_rows", 0))
        component_kg = city.get("component_kg", {}) or {}
        for key in CONTRIBUTION_COMPONENT_KEYS:
            kg_value = float(component_kg.get(key, 0.0))
            row[f"{key}_kg"] = kg_value
            row[f"{key}_pp"] = kg_value / national_baseline_kg * 100.0
        row["total_net_kg"] = float(city.get("net_saving_kg", 0.0))
        row["total_net_pp"] = row["total_net_kg"] / national_baseline_kg * 100.0
        rows.append(row)

    rows.sort(key=lambda row: (row["total_net_kg"], row["label"]), reverse=True)
    return rows[:top_n]


def _macro_region_component_rows(
    city_reports: list[dict],
    national_baseline_kg: float,
) -> list[dict]:
    """Return contribution rows with each China macro-region treated as one city."""
    if national_baseline_kg <= 0:
        return []

    rows_by_key = {}
    for spec in CHINA_MACRO_REGION_SPECS:
        row = _contribution_row_template(spec["label"])
        row.update(
            {
                "region_key": spec["key"],
                "macro_region_key": spec["key"],
                "macro_region_label": spec["full_label"],
                "macro_region_label_cn": spec["cn_label"],
                "provinces": list(spec["provinces"]),
                "city_count": 0,
                "baseline_ebike_lifecycle_kg": 0.0,
            }
        )
        rows_by_key[spec["key"]] = row

    for city in city_reports:
        province = _normalize_province_name(city.get("province", ""))
        spec = PROVINCE_TO_MACRO_REGION.get(province)
        if spec is None:
            continue
        row = rows_by_key[spec["key"]]
        row["city_count"] += 1
        row["sampled_drone_rows"] += int(city.get("drone_assigned_sampled_rows", 0))
        row["baseline_ebike_lifecycle_kg"] += float(
            city.get("baseline_ebike_lifecycle_kg", 0.0)
        )
        component_kg = city.get("component_kg", {}) or {}
        for key in CONTRIBUTION_COMPONENT_KEYS:
            row[f"{key}_kg"] += float(component_kg.get(key, 0.0))
        row["total_net_kg"] += float(city.get("net_saving_kg", 0.0))

    rows = []
    for spec in CHINA_MACRO_REGION_SPECS:
        row = rows_by_key[spec["key"]]
        regional_baseline_kg = float(row["baseline_ebike_lifecycle_kg"])
        for key in CONTRIBUTION_COMPONENT_KEYS:
            row[f"{key}_pp"] = (
                row[f"{key}_kg"] / regional_baseline_kg * 100.0
                if regional_baseline_kg > 0
                else 0.0
            )
        row["total_net_pp"] = (
            row["total_net_kg"] / regional_baseline_kg * 100.0
            if regional_baseline_kg > 0
            else 0.0
        )
        rows.append(row)
    return rows


def _selected_city_component_rows(
    city_reports: list[dict],
    selected_region_keys: tuple[str, ...] = CONTRIBUTION_CITY_B_REGION_KEYS,
) -> list[dict]:
    """Return selected city contribution rows in grouped representative order."""
    city_by_key = {
        str(city.get("region_key", "")): city
        for city in city_reports
    }
    selected = set(selected_region_keys)
    macro_region_by_key = {
        spec["key"]: spec
        for spec in CHINA_MACRO_REGION_SPECS
    }
    rows = []
    for macro_region_key, region_keys in CONTRIBUTION_CITY_B_GROUPS:
        macro_region = macro_region_by_key.get(macro_region_key, {})
        for region_key in region_keys:
            if region_key not in selected:
                continue
            city = city_by_key.get(region_key)
            if city is None:
                continue

            label = CONTRIBUTION_CITY_B_DISPLAY_NAMES.get(
                region_key,
                str(city.get("region_name_en", region_key)),
            )
            row = _contribution_row_template(label)
            city_baseline_kg = float(city.get("baseline_ebike_lifecycle_kg", 0.0))
            row.update(
                {
                    "row_type": "city",
                    "region_key": region_key,
                    "province": str(city.get("province", "")),
                    "macro_region_key": macro_region_key,
                    "macro_region_label": macro_region.get(
                        "full_label",
                        macro_region_key,
                    ),
                    "macro_region_label_cn": macro_region.get("cn_label", ""),
                    "macro_region_color": macro_region.get("color", paper_color("contribution_break_even.777777")),
                    "baseline_ebike_lifecycle_kg": city_baseline_kg,
                    "population_10k": float(city.get("population_10k", 0.0)),
                    "population_million": float(
                        city.get("population_million", 0.0)
                    ),
                    "population_source_year": city.get(
                        "population_source_year"
                    ),
                    "sampled_drone_rows": int(
                        city.get("drone_assigned_sampled_rows", 0)
                    ),
                }
            )
            component_kg = city.get("component_kg", {}) or {}
            for key in CONTRIBUTION_COMPONENT_KEYS:
                kg_value = float(component_kg.get(key, 0.0))
                row[f"{key}_kg"] = kg_value
                row[f"{key}_pp"] = (
                    kg_value / city_baseline_kg * 100.0
                    if city_baseline_kg > 0
                    else 0.0
                )
            row["total_net_kg"] = float(city.get("net_saving_kg", 0.0))
            row["total_net_pp"] = (
                row["total_net_kg"] / city_baseline_kg * 100.0
                if city_baseline_kg > 0
                else 0.0
            )
            rows.append(row)

    return rows


def _national_contribution_row(national: dict) -> dict:
    """Return the all-study-regions row for the selected-city panel."""
    row = _contribution_row_template("All study regions")
    row.update(
        {
            "row_type": "overall",
            "baseline_ebike_lifecycle_kg": float(
                national.get("baseline_ebike_lifecycle_kg", 0.0)
            ),
            "sampled_drone_rows": int(
                national.get("drone_assigned_sampled_rows", 0)
            ),
        }
    )
    for key in CONTRIBUTION_COMPONENT_KEYS:
        row[f"{key}_kg"] = float(national.get(f"{key}_kg", 0.0))
        row[f"{key}_pp"] = float(national.get(f"{key}_pp", 0.0))
    row["total_net_kg"] = float(national.get("total_net_kg", 0.0))
    row["total_net_pp"] = float(national.get("total_net_pp", 0.0))
    return row


def _full_city_contribution_aggregates(
    od_path: str,
    ebike_params,
    drone_params,
    ebike_scenario,
    drone_scenario,
    sample_seed: int,
) -> tuple[dict | None, str | None]:
    """Compatibility adapter around the analysis-layer streaming reducer."""
    from drone_emission.analysis.contribution_decomposition import (
        aggregate_od_contributions,
    )

    try:
        aggregate = aggregate_od_contributions(
            od_path,
            ebike_params=ebike_params,
            drone_params=drone_params,
            ebike_scenario=ebike_scenario,
            drone_scenario=drone_scenario,
            chunk_rows=CONTRIBUTION_FULL_CHUNK_ROWS,
        )
    except ValueError as exc:
        return None, str(exc)
    return {
        "has_road_distance_m": True,
        "sample": {
            "valid_rows": int(aggregate["valid_od_rows"]),
            "sampled_rows": int(aggregate["valid_od_rows"]),
            "distance_strata": [],
            "sample_seed": int(sample_seed),
            "sampling_mode": "full",
        },
        "assigned_rows": int(aggregate["drone_assigned_rows"]),
        "assigned_ratio_pct": float(aggregate["drone_assigned_ratio_pct"]),
        "component_kg": aggregate["component_kg"],
        "max_abs_error_g_per_order": float(
            aggregate["max_abs_identity_error_g_per_order"]
        ),
        "aggregate_error_kg": float(aggregate["aggregate_identity_error_kg"]),
    }, None


def prepare_contribution_decomposition_data(
    city_df: pd.DataFrame,
    data_dir: str,
    exp_config: dict = None,
    max_sample_rows: int | None = CONTRIBUTION_SAMPLE_LIMIT,
    sample_seed: int = CONTRIBUTION_SAMPLE_SEED,
) -> Dict:
    """Prepare the four-part lifecycle-emissions ledger for the paper figure."""
    from dataclasses import replace

    from drone_emission.emission.cal_single_delivery import (
        create_drone_params,
        create_ebike_params,
        get_emission_params,
    )
    from drone_emission.sensitivity.vectorized_lca import (
        DroneScenario,
        EbikeScenario,
        compute_drone_per_order_parts_g,
        compute_ebike_per_order_parts_g,
    )

    if exp_config is None:
        print("Contribution decomposition skipped — exp_config missing")
        return None

    start_time = time.perf_counter()
    city_rows = city_df.copy()
    if "region_level" in city_rows.columns:
        city_rows = city_rows[city_rows["region_level"] == "city"].copy()

    required_city_cols = {
        "region_key",
        "region_name_en",
        "province",
        "total_ebike_lifecycle",
        "gef",
    }
    missing_city_cols = sorted(required_city_cols - set(city_rows.columns))
    if missing_city_cols:
        print(
            "Contribution decomposition skipped — city stats missing columns "
            f"{missing_city_cols}"
        )
        return None

    for col in ("total_ebike_lifecycle", "gef", "ebike_kappa"):
        if col in city_rows.columns:
            city_rows[col] = pd.to_numeric(city_rows[col], errors="coerce")
    city_rows = city_rows[
        city_rows["total_ebike_lifecycle"].notna()
        & (city_rows["total_ebike_lifecycle"] > 0)
        & city_rows["gef"].notna()
    ].copy()
    if city_rows.empty:
        print("Contribution decomposition skipped — no eligible city rows")
        return None

    baseline_kappa = float(getattr(create_ebike_params(exp_config), "kappa", 1.0))
    emission_cfg = get_emission_params(exp_config)
    ebike_params = create_ebike_params(exp_config)
    drone_params = create_drone_params(exp_config)
    ebike_actual_scenario = EbikeScenario(
        kappa=baseline_kappa,
        rider_weight=emission_cfg["rider_weight"],
        payload_kg=emission_cfg["payload_kg"],
    )
    drone_actual_scenario = DroneScenario(
        payload_kg=emission_cfg["payload_kg"],
    )

    city_reports = []
    skipped_cities = []
    national_components_kg = {key: 0.0 for key in CONTRIBUTION_COMPONENT_KEYS}
    national_baseline_kg = 0.0
    national_assigned_rows = 0
    national_max_abs_error_g = 0.0
    national_aggregate_error_kg = 0.0

    for city_row in city_rows.sort_values("region_key").itertuples(index=False):
        region_key = str(city_row.region_key)
        od_path = _resolve_contribution_od_path(data_dir, region_key)
        if od_path is None:
            skipped_cities.append(
                {"region_key": region_key, "reason": "missing_od_csv"}
            )
            continue

        gef = float(city_row.gef)
        ebike_pp = replace(ebike_params, e_use=gef)
        drone_pp = replace(drone_params, e_use=gef)
        assignment_meta = {
            "rule": "oracle_per_od_lifecycle_minimum",
            "description": (
                "Assign each OD to drone only when drone lifecycle emissions "
                "are lower than the e-bike alternative."
            ),
        }
        if max_sample_rows is None:
            full, skip_reason = _full_city_contribution_aggregates(
                od_path,
                ebike_pp,
                drone_pp,
                ebike_actual_scenario,
                drone_actual_scenario,
                sample_seed,
            )
            if full is None:
                skipped_cities.append(
                    {"region_key": region_key, "reason": skip_reason}
                )
                continue
            city_components_kg = full["component_kg"]
            city_net_kg = float(sum(city_components_kg.values()))
            city_baseline_kg = float(city_row.total_ebike_lifecycle)
            city_reports.append(
                {
                    "region_key": region_key,
                    "region_name_en": str(city_row.region_name_en),
                    "province": str(city_row.province),
                    "gef": gef,
                    "baseline_ebike_lifecycle_kg": city_baseline_kg,
                    "assignment_rule": assignment_meta["rule"],
                    "assignment_meta": assignment_meta,
                    "od_path": od_path,
                    "has_road_distance_m": full["has_road_distance_m"],
                    "sample": full["sample"],
                    "drone_assigned_sampled_rows": full["assigned_rows"],
                    "sampled_drone_ratio_pct": full["assigned_ratio_pct"],
                    "component_kg": city_components_kg,
                    "component_pct_points": {
                        key: city_components_kg[key] / city_baseline_kg * 100.0
                        for key in CONTRIBUTION_COMPONENT_KEYS
                    },
                    "net_saving_kg": city_net_kg,
                    "net_saving_pct_points": (
                        city_net_kg / city_baseline_kg * 100.0
                    ),
                    "sum_check": {
                        "max_abs_error_g_per_order": full[
                            "max_abs_error_g_per_order"
                        ],
                        "aggregate_error_kg": full["aggregate_error_kg"],
                    },
                }
            )
            national_baseline_kg += city_baseline_kg
            for key, value in city_components_kg.items():
                national_components_kg[key] += value
            national_assigned_rows += full["assigned_rows"]
            national_max_abs_error_g = max(
                national_max_abs_error_g,
                full["max_abs_error_g_per_order"],
            )
            national_aggregate_error_kg += full["aggregate_error_kg"]
            continue

        od_df = pd.read_csv(
            od_path,
            usecols=lambda col: col in {"distance_km", "annual_sales", "road_distance_m"},
        )
        if "distance_km" not in od_df.columns or "annual_sales" not in od_df.columns:
            skipped_cities.append(
                {
                    "region_key": region_key,
                    "reason": "missing_required_od_columns",
                    "columns_present": sorted(od_df.columns.tolist()),
                }
            )
            continue

        valid = od_df.copy()
        valid["distance_km"] = pd.to_numeric(valid["distance_km"], errors="coerce")
        valid["annual_sales"] = pd.to_numeric(valid["annual_sales"], errors="coerce")
        has_road_distance = "road_distance_m" in valid.columns
        if has_road_distance:
            valid["road_distance_m"] = pd.to_numeric(
                valid["road_distance_m"], errors="coerce"
            )
            valid["_road_km"] = valid["road_distance_m"] / 1000.0
        else:
            valid["_road_km"] = valid["distance_km"]

        valid = valid[
            valid["distance_km"].notna()
            & (valid["distance_km"] > 0)
            & valid["_road_km"].notna()
            & (valid["_road_km"] > 0)
            & valid["annual_sales"].notna()
            & (valid["annual_sales"] > 0)
        ].copy()
        if valid.empty:
            skipped_cities.append(
                {"region_key": region_key, "reason": "no_valid_od_rows"}
            )
            continue

        sample_distance_km = (
            valid["_road_km"].to_numpy(dtype=float)
            if has_road_distance
            else valid["distance_km"].to_numpy(dtype=float)
        )
        sampled, sample_meta = _sample_city_valid_rows(
            valid_df=valid,
            sample_distance_km=sample_distance_km,
            region_key=region_key,
            max_sample_rows=max_sample_rows,
            sample_seed=sample_seed,
        )
        if sampled.empty:
            skipped_cities.append(
                {"region_key": region_key, "reason": "sampling_returned_no_rows"}
            )
            continue

        road_km = sampled["_road_km"].to_numpy(dtype=float)
        air_km = sampled["distance_km"].to_numpy(dtype=float)
        annual_sales = sampled["annual_sales"].to_numpy(dtype=float)
        expansion_weight = sampled["_sample_expansion_weight"].to_numpy(dtype=float)
        annual_weight = annual_sales * expansion_weight

        ebike_actual_parts = compute_ebike_per_order_parts_g(
            road_km,
            ebike_pp,
            ebike_actual_scenario,
        )
        drone_actual_parts = compute_drone_per_order_parts_g(
            air_km,
            drone_pp,
            drone_actual_scenario,
        )

        use_drone = drone_actual_parts["total_g"] < ebike_actual_parts["total_g"]
        embodied_balance_g = (
            ebike_actual_parts["production_g"]
            + ebike_actual_parts["end_of_life_g"]
            - drone_actual_parts["production_g"]
            - drone_actual_parts["end_of_life_g"]
        )
        avoided_ebike_road_use_g = ebike_actual_parts["use_g"]
        drone_fixed_flight_penalty_g = -drone_actual_parts["fixed_use_g"]
        drone_cruise_penalty_g = -drone_actual_parts["cruise_use_g"]
        net_saving_g = ebike_actual_parts["total_g"] - drone_actual_parts["total_g"]
        sum_check_g = (
            embodied_balance_g
            + avoided_ebike_road_use_g
            + drone_fixed_flight_penalty_g
            + drone_cruise_penalty_g
            - net_saving_g
        )

        assigned = sampled.loc[use_drone].copy()
        assigned["_annual_weight"] = annual_weight[use_drone]
        for key, values in (
            ("embodied_balance", embodied_balance_g),
            ("avoided_ebike_road_use", avoided_ebike_road_use_g),
            ("drone_fixed_flight_penalty", drone_fixed_flight_penalty_g),
            ("drone_cruise_penalty", drone_cruise_penalty_g),
        ):
            assigned[f"{key}_kg"] = values[use_drone] * annual_weight[use_drone] / 1000.0
        assigned["net_saving_kg"] = net_saving_g[use_drone] * annual_weight[use_drone] / 1000.0
        assigned["sum_check_error_g"] = sum_check_g[use_drone]
        assigned["gef"] = gef
        assigned["region_key"] = region_key
        assigned["region_name_en"] = str(city_row.region_name_en)
        assigned["city_baseline_kg"] = float(city_row.total_ebike_lifecycle)

        city_components_kg = {
            key: float(assigned[f"{key}_kg"].sum())
            for key in CONTRIBUTION_COMPONENT_KEYS
        }
        city_net_kg = float(sum(city_components_kg.values()))
        city_baseline_kg = float(city_row.total_ebike_lifecycle)
        city_reports.append(
            {
                "region_key": region_key,
                "region_name_en": str(city_row.region_name_en),
                "province": str(city_row.province),
                "gef": gef,
                "baseline_ebike_lifecycle_kg": city_baseline_kg,
                "assignment_rule": assignment_meta["rule"],
                "assignment_meta": assignment_meta,
                "od_path": od_path,
                "has_road_distance_m": bool(has_road_distance),
                "sample": sample_meta,
                "drone_assigned_sampled_rows": int(use_drone.sum()),
                "sampled_drone_ratio_pct": float(use_drone.mean() * 100.0),
                "component_kg": city_components_kg,
                "component_pct_points": {
                    key: (
                        city_components_kg[key] / city_baseline_kg * 100.0
                        if city_baseline_kg > 0
                        else 0.0
                    )
                    for key in CONTRIBUTION_COMPONENT_KEYS
                },
                "net_saving_kg": city_net_kg,
                "net_saving_pct_points": (
                    city_net_kg / city_baseline_kg * 100.0
                    if city_baseline_kg > 0
                    else 0.0
                ),
                "sum_check": {
                    "max_abs_error_g_per_order": float(
                        np.abs(sum_check_g[use_drone]).max()
                    )
                    if use_drone.any()
                    else 0.0,
                    "aggregate_error_kg": float(
                        assigned["sum_check_error_g"].mul(assigned["_annual_weight"]).sum()
                        / 1000.0
                    )
                    if not assigned.empty
                    else 0.0,
                },
            }
        )

        national_baseline_kg += city_baseline_kg
        for key, value in city_components_kg.items():
            national_components_kg[key] += value
        if not assigned.empty:
            national_assigned_rows += len(assigned)
            national_max_abs_error_g = max(
                national_max_abs_error_g,
                float(np.abs(assigned["sum_check_error_g"]).max()),
            )
            national_aggregate_error_kg += float(
                assigned["sum_check_error_g"].mul(assigned["_annual_weight"]).sum()
                / 1000.0
            )

    if national_baseline_kg <= 0 or not city_reports:
        print("Contribution decomposition skipped — no city contributions available")
        return None

    national = {
        f"{key}_kg": float(national_components_kg[key])
        for key in CONTRIBUTION_COMPONENT_KEYS
    }
    for key in CONTRIBUTION_COMPONENT_KEYS:
        national[f"{key}_pp"] = (
            national[f"{key}_kg"] / national_baseline_kg * 100.0
            if national_baseline_kg > 0
            else 0.0
        )
    national["baseline_ebike_lifecycle_kg"] = float(national_baseline_kg)
    national["total_net_kg"] = float(
        sum(national[f"{key}_kg"] for key in CONTRIBUTION_COMPONENT_KEYS)
    )
    national["total_net_pp"] = float(
        sum(national[f"{key}_pp"] for key in CONTRIBUTION_COMPONENT_KEYS)
    )
    national["drone_assigned_sampled_rows"] = int(national_assigned_rows)
    national["sum_check"] = {
        "max_abs_error_g_per_order": float(national_max_abs_error_g),
        "aggregate_error_kg": float(national_aggregate_error_kg),
    }

    runtime_seconds = time.perf_counter() - start_time
    return {
        "component_specs": list(CONTRIBUTION_COMPONENT_SPECS),
        "sampling": {
            "max_sample_rows_per_city": (
                None if max_sample_rows is None else int(max_sample_rows)
            ),
            "sample_seed": int(sample_seed),
            "mode": "full_real_data" if max_sample_rows is None else "stratified_sample",
            "distance_stratification": "road_distance_m if available else distance_km",
        },
        "runtime": {
            "prep_runtime_seconds": float(runtime_seconds),
            "city_count_input": int(len(city_rows)),
            "city_count_processed": int(len(city_reports)),
            "city_count_skipped": int(len(skipped_cities)),
        },
        "national": national,
        "top_cities": _top_city_component_rows(
            city_reports=city_reports,
            national_baseline_kg=national_baseline_kg,
            top_n=15,
        ),
        "macro_regions": _macro_region_component_rows(
            city_reports=city_reports,
            national_baseline_kg=national_baseline_kg,
        ),
        "cities": city_reports,
        "skipped_cities": skipped_cities,
    }


def _selected_city_y_positions(rows: list[dict]) -> np.ndarray:
    """Return the locked overall-plus-28-city row coordinates."""
    y_values = []
    current_y = 0.0
    for idx, row in enumerate(rows):
        if idx:
            previous = rows[idx - 1]
            current_y += 1.0
            if previous.get("row_type") == "overall":
                current_y += 0.85
            elif (
                row.get("macro_region_key")
                != previous.get("macro_region_key")
            ):
                current_y += 0.65
        y_values.append(current_y)
    return np.array(y_values, dtype=float)


def _set_selected_city_y_limits(ax, y: np.ndarray) -> None:
    """Apply deterministic vertical limits shared by both standalone panels."""
    if y.size:
        ax.set_ylim(float(y[-1]) + 0.72, float(y[0]) - 0.72)


def _plot_selected_city_contribution_bars(ax, rows: list[dict]):
    """Draw one-row-per-city horizontal component bars with net points."""
    if not rows:
        ax.text(
            0.5,
            0.5,
            "no sampled contributions",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=10,
            color=paper_color("contribution_break_even.777777"),
        )
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        return np.array([], dtype=float)

    y = _selected_city_y_positions(rows)
    positive_acc = np.zeros(len(rows), dtype=float)
    negative_acc = np.zeros(len(rows), dtype=float)
    extrema = [0.0]

    for spec in CONTRIBUTION_COMPONENT_SPECS:
        values = np.array([float(row[f"{spec['key']}_pp"]) for row in rows], dtype=float)
        benefits = np.clip(values, 0.0, None)
        penalties = np.clip(values, None, 0.0)
        if np.any(benefits):
            ax.barh(
                y,
                benefits,
                left=positive_acc,
                height=0.62,
                color=spec["color"],
                edgecolor=paper_color("ink.background"),
                linewidth=0.8,
            )
        if np.any(penalties):
            ax.barh(
                y,
                np.abs(penalties),
                left=negative_acc + penalties,
                height=0.62,
                color=spec["color"],
                edgecolor=paper_color("ink.background"),
                linewidth=0.8,
            )
        positive_acc += benefits
        negative_acc += penalties
        extrema.extend(positive_acc.tolist())
        extrema.extend(negative_acc.tolist())

    total_pp = np.array([float(row["total_net_pp"]) for row in rows], dtype=float)
    extrema.extend(total_pp.tolist())
    ax.scatter(
        total_pp,
        y,
        s=26,
        color=paper_color("contribution_break_even.111111"),
        edgecolor=paper_color("ink.background"),
        linewidth=0.5,
        zorder=5,
        label="Net reduction",
    )

    ax.axvline(0, color=paper_color("contribution_break_even.555555"), linewidth=0.8)
    ax.set_yticks(y, [row["label"] for row in rows])
    _set_selected_city_y_limits(ax, y)
    tick_labels = ax.get_yticklabels()
    if tick_labels and rows[0].get("row_type") == "overall":
        tick_labels[0].set_fontweight("bold")
        tick_labels[0].set_color(paper_color("contribution_break_even.222222"))
    ax.set_xlabel(
        "Contribution to all-e-bike baseline (pp)",
        fontsize=9.5,
    )
    ax.grid(axis="x", alpha=paper_alpha("contribution_break_even.alpha_0_25"))
    ax.tick_params(axis="both", labelsize=8.2)

    xmin = min(extrema)
    xmax = max(extrema)
    span = max(xmax - xmin, 1.0)
    ax.set_xlim(xmin - span * 0.10, xmax + span * 0.10)
    _annotate_selected_city_macro_regions(ax, rows, y)
    return y


def _annotate_selected_city_macro_regions(ax, rows: list[dict], y: np.ndarray):
    """Add left-side macro-region labels and brackets to selected-city rows."""
    if not rows or y.size == 0:
        return

    city_indices = [
        idx for idx, row in enumerate(rows) if row.get("row_type") == "city"
    ]
    if not city_indices:
        return

    groups = []
    start_idx = city_indices[0]
    for position, idx in enumerate(city_indices):
        is_last = position == len(city_indices) - 1
        next_idx = None if is_last else city_indices[position + 1]
        if is_last or (
            rows[next_idx].get("macro_region_key")
            != rows[start_idx].get("macro_region_key")
        ):
            groups.append((start_idx, idx))
            if next_idx is not None:
                start_idx = next_idx

    y_transform = ax.get_yaxis_transform()
    if rows[0].get("row_type") == "overall":
        separator_y = (float(y[0]) + float(y[city_indices[0]])) / 2.0
        ax.axhline(
            separator_y,
            color=paper_color("contribution_break_even.c9c9c9"),
            linewidth=0.9,
            zorder=1,
        )

    bracket_x = -0.280
    bracket_cap_x = -0.235
    label_x = -0.320
    for group_idx, (start, end) in enumerate(groups):
        row = rows[start]
        color = row.get("macro_region_color", paper_color("contribution_break_even.777777"))
        label = row.get("macro_region_label", row.get("macro_region_key", ""))
        y_top = float(y[start] - 0.38)
        y_bottom = float(y[end] + 0.38)
        ax.plot(
            [bracket_x, bracket_x],
            [y_top, y_bottom],
            transform=y_transform,
            color=color,
            linewidth=1.25,
            solid_capstyle="butt",
            clip_on=False,
            zorder=4,
        )
        for y_cap in (y_top, y_bottom):
            ax.plot(
                [bracket_x, bracket_cap_x],
                [y_cap, y_cap],
                transform=y_transform,
                color=color,
                linewidth=1.25,
                solid_capstyle="butt",
                clip_on=False,
                zorder=4,
            )
        if group_idx:
            ax.axhline(
                y_top - 0.32,
                color=paper_color("contribution_break_even.dddddd"),
                linewidth=0.7,
                zorder=1,
            )
        ax.text(
            label_x,
            (y_top + y_bottom) / 2.0,
            label,
            transform=y_transform,
            ha="right",
            va="center",
            fontsize=7.2,
            fontweight="bold",
            color=color,
            clip_on=False,
        )


def _plot_selected_city_absolute_reduction_points(
    ax,
    rows: list[dict],
    y: np.ndarray,
    zero_fraction: float,
):
    """Draw selected-city absolute net reductions on a top x-axis."""
    city_indices = [
        idx for idx, row in enumerate(rows) if row.get("row_type") == "city"
    ]
    if not city_indices:
        ax.tick_params(top=False, labeltop=False, bottom=False, labelbottom=False)
        return

    reductions_kt = np.array(
        [float(rows[idx].get("total_net_kg", 0.0)) / 1e6 for idx in city_indices],
        dtype=float,
    )
    ax.scatter(
        reductions_kt,
        y[city_indices],
        s=42,
        facecolor=paper_color("ink.background"),
        edgecolor=paper_color("contribution_break_even.1b5e20"),
        linewidth=1.1,
        zorder=6,
        label="Absolute net reduction",
    )
    ax.set_xlabel("Absolute net reduction (kt CO2e yr$^{-1}$)", fontsize=9.5)
    ax.tick_params(
        axis="x",
        top=True,
        labeltop=True,
        bottom=False,
        labelbottom=False,
        labelsize=8.2,
        colors=paper_color("contribution_break_even.1b5e20"),
    )
    ax.xaxis.label.set_color(paper_color("contribution_break_even.1b5e20"))
    ax.spines["top"].set_color(paper_color("contribution_break_even.1b5e20"))
    ax.spines["bottom"].set_visible(False)
    ax.tick_params(axis="y", left=False, labelleft=False)

    xmax = max(float(np.max(reductions_kt)), 0.0)
    positive_limit = max(xmax * 1.12, 0.1)
    zero_fraction = float(np.clip(zero_fraction, 0.05, 0.95))
    negative_limit = positive_limit * zero_fraction / (1.0 - zero_fraction)
    ax.set_xlim(-negative_limit, positive_limit)


def _write_contribution_decomposition_city_b_report(
    prepared: Dict,
    save_path: str,
    figsize: tuple,
    fig,
    legend,
    axes: tuple,
    rows: list[dict],
):
    """Write compact JSON report for the selected-city contribution figure."""
    ax_main, ax_top = axes
    report_path = save_path.rsplit(".", 1)[0] + "_report.json"
    macro_region_by_key = {
        spec["key"]: spec
        for spec in CHINA_MACRO_REGION_SPECS
    }
    selected_city_groups = []
    for macro_region_key, region_keys in CONTRIBUTION_CITY_B_GROUPS:
        spec = macro_region_by_key.get(macro_region_key, {})
        selected_city_groups.append(
            {
                "macro_region_key": macro_region_key,
                "macro_region_label": spec.get("full_label", macro_region_key),
                "macro_region_label_cn": spec.get("cn_label", ""),
                "region_keys": list(region_keys),
            }
        )
    overall_row = next(
        (row for row in rows if row.get("row_type") == "overall"),
        None,
    )
    city_rows = [row for row in rows if row.get("row_type") == "city"]
    plotted_region_keys = {row.get("region_key") for row in city_rows}
    vertical_layout = _selected_city_vertical_layout(figsize)
    xlabel_bbox_mm = _artist_bbox_mm(fig, ax_main.xaxis.label)
    legend_bbox_mm = _artist_bbox_mm(fig, legend)
    xlabel_to_legend_gap_mm = xlabel_bbox_mm["y0"] - legend_bbox_mm["y1"]
    payload = {
        "figure": "contribution_decomposition_city_b",
        "figure_path": save_path,
        "figure_size_inches": {"width": figsize[0], "height": figsize[1]},
        "figure_size_mm": {
            "width": figsize[0] * 25.4,
            "height": figsize[1] * 25.4,
        },
        "axes": {
            "bottom_relative_contribution_decomposition": _report_axis(ax_main),
            "top_absolute_net_reduction": _report_axis(ax_top),
        },
        "vertical_layout": {
            **_json_safe(vertical_layout),
            "axes_bbox_mm": _axis_bbox_mm(ax_main, figsize),
            "bottom_xlabel_bbox_mm": xlabel_bbox_mm,
            "legend_bbox_mm": legend_bbox_mm,
            "xlabel_to_legend_gap_mm": _report_number(
                xlabel_to_legend_gap_mm
            ),
        },
        "row_alignment": {
            "contract": vertical_layout["alignment_contract"],
            "row_count": len(rows),
            "y_positions": _report_array(ax_main.get_yticks()),
            "y_limits": _report_array(ax_main.get_ylim()),
        },
        "selected_region_keys": list(CONTRIBUTION_CITY_B_REGION_KEYS),
        "selected_city_groups": selected_city_groups,
        "missing_selected_region_keys": [
            region_key
            for region_key in CONTRIBUTION_CITY_B_REGION_KEYS
            if region_key not in plotted_region_keys
        ],
        "overall": _json_safe(overall_row),
        "overall_scope": prepared.get("study_scope", {}).get(
            "label",
            "All successfully processed study cities",
        ),
        "study_scope": _json_safe(prepared.get("study_scope", {})),
        "gef_coverage": _json_safe(prepared.get("gef_coverage", {})),
        "sort_order": (
            "All study regions first, followed by China macro-region order "
            "with four curated representative cities per region"
        ),
        "bar_design": (
            "An independent all-study-regions row precedes the grouped city "
            "rows. The seven China macro-regions follow the shared paper "
            "order, with colored left-side labels and brackets spanning four "
            "curated representative cities per region. Positive components "
            "stack from zero to the right, negative components stack from "
            "zero to the left, and a black point marks total net reduction "
            "on the bottom percentage-point axis. White circles with dark "
            "green outlines mark city absolute net reduction on the top kt "
            "CO2e/yr axis; the overall row is excluded from that axis."
        ),
        "decomposition_contract": {
            "scope": "OD pairs assigned to drones by the lifecycle-minimum rule",
            "denominator": "complete all-e-bike lifecycle baseline",
            "identity": (
                "embodied_balance + avoided_ebike_road_use + "
                "drone_fixed_flight_penalty + drone_cruise_penalty "
                "= e-bike lifecycle emissions - drone lifecycle emissions"
            ),
            "fixed_flight_definition": (
                "loaded and empty take-off and landing, plus loaded hover"
            ),
            "cruise_definition": "loaded outbound and empty return level flight",
            "bundling_treatment": (
                "configured e-bike kappa is embedded in avoided road operation; "
                "bundling is not an independent contribution"
            ),
            "driver_mapping": {
                "grid_carbon_intensity": "all use-phase components",
                "air_distance": (
                    "drone cruise, fixed-flight dilution and embodied "
                    "amortization through delivery time"
                ),
                "road_detour": (
                    "avoided e-bike road operation and embodied amortization "
                    "through delivery time"
                ),
            },
        },
        "component_specs": _json_safe(CONTRIBUTION_COMPONENT_SPECS),
        "legend": {
            "location": "bottom center",
            "columns": 6,
            "font_size_pt": 6.4,
            "anchor_from_page_bottom_mm": CONTRIBUTION_CITY_LEGEND_ANCHOR_MM,
            "xlabel_gap_target_mm": list(
                CONTRIBUTION_CITY_LEGEND_GAP_TARGET_MM
            ),
            "xlabel_gap_actual_mm": _report_number(
                xlabel_to_legend_gap_mm
            ),
            "labels": [
                "Embodied",
                "Avoided road",
                "Fixed flight",
                "Cruise",
                "Net reduction",
                "Absolute reduction",
            ],
        },
        "net_marker": {"label": "Net reduction", "color": paper_color("contribution_break_even.111111")},
        "absolute_marker": {
            "label": "Absolute net reduction",
            "facecolor": "white",
            "edgecolor": paper_color("contribution_break_even.1b5e20"),
            "scope": "Representative city rows only",
        },
        "export": {
            "bbox_inches": None,
            "pad_inches": 0.0,
            "source_canvas_inches": {
                "width": figsize[0],
                "height": figsize[1],
            },
        },
        "cities": _json_safe(city_rows),
        "sampling": _json_safe(prepared.get("sampling", {})),
        "runtime": _json_safe(prepared.get("runtime", {})),
        "skipped_cities": _json_safe(prepared.get("skipped_cities", [])),
    }
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, allow_nan=False)
    print(f"Contribution decomposition city-b report saved to {report_path}")


@prepared_plot()
@_with_contribution_city_typography
def plot_contribution_decomposition_city_b(
    prepared: Dict,
    save_path: str = None,
    figsize: tuple = (132.0 / 25.4, CONTRIBUTION_CITY_PANEL_HEIGHT_MM / 25.4),
    for_latex: bool = False,
    plot_title: bool = False,
):
    """Plot the lifecycle-emissions balance for selected study cities.

    Args:
        prepared: Output of ``prepare_contribution_decomposition_data``.
        save_path: Path to save figure (with extension). Shows plot if None.
        figsize: Figure size tuple.
        for_latex: If True, save as PDF.
        plot_title: If True, show title.
    """
    if not prepared:
        print("Contribution decomposition city-b skipped — prepared data missing")
        return

    city_rows = _selected_city_component_rows(prepared.get("cities", []))
    rows = [_national_contribution_row(prepared.get("national", {})), *city_rows]
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(1, 1, 1)
    vertical_layout = _set_selected_city_axes_position(
        ax,
        figsize,
        left=0.39,
        right=0.98,
    )
    y = _plot_selected_city_contribution_bars(ax, rows)
    ax_top = ax.twiny()
    ax_top.set_position(ax.get_position())
    bottom_xlim = ax.get_xlim()
    zero_fraction = (0.0 - bottom_xlim[0]) / (bottom_xlim[1] - bottom_xlim[0])
    _plot_selected_city_absolute_reduction_points(ax_top, rows, y, zero_fraction)
    ax_top.set_ylim(ax.get_ylim())

    legend_handles = [
        plt.Rectangle(
            (0, 0),
            1,
            1,
            facecolor=spec["color"],
            edgecolor=paper_color("ink.background"),
            label={
                "embodied_balance": "Embodied",
                "avoided_ebike_road_use": "Avoided road",
                "drone_fixed_flight_penalty": "Fixed flight",
                "drone_cruise_penalty": "Cruise",
            }[spec["key"]],
        )
        for spec in CONTRIBUTION_COMPONENT_SPECS
    ]
    legend_handles.append(
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=paper_color("contribution_break_even.111111"),
            markeredgecolor=paper_color("contribution_break_even.111111"),
            markersize=5,
            label="Net reduction",
        )
    )
    legend_handles.append(
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=paper_color("ink.background"),
            markeredgecolor=paper_color("contribution_break_even.1b5e20"),
            markeredgewidth=1.1,
            markersize=6,
            label="Absolute reduction",
        )
    )
    legend = fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(
            0.5,
            CONTRIBUTION_CITY_LEGEND_ANCHOR_MM
            / vertical_layout["page_height_mm"],
        ),
        ncol=6,
        frameon=False,
        fontsize=6.4,
        handlelength=0.9,
        handletextpad=0.35,
        columnspacing=0.55,
    )
    if plot_title:
        ax.set_title(
            "Lifecycle-emissions balance by representative city",
            fontsize=13,
        )

    if save_path:
        if for_latex and not save_path.endswith(".pdf"):
            save_path = save_path.rsplit(".", 1)[0] + ".pdf"
        if plot_title:
            save_path = _add_title_suffix(save_path)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(
            save_path,
            dpi=300,
        )
        _write_contribution_decomposition_city_b_report(
            prepared,
            save_path,
            figsize,
            fig,
            legend,
            axes=(ax, ax_top),
            rows=rows,
        )
        plt.close(fig)
        print(f"Contribution decomposition city-b saved to {save_path}")
    else:
        plt.show()
        plt.close(fig)


def _write_contribution_city_population_report(
    prepared: Dict,
    save_path: str,
    figsize: tuple,
    fig,
    ax,
    rows: list[dict],
    y: np.ndarray,
):
    """Write the population-panel coverage and row-alignment report."""
    report_path = save_path.rsplit(".", 1)[0] + "_report.json"
    city_rows = [row for row in rows if row.get("row_type") == "city"]
    vertical_layout = _selected_city_vertical_layout(figsize)
    city_positions = {
        row["region_key"]: float(y[idx])
        for idx, row in enumerate(rows)
        if row.get("row_type") == "city"
    }
    cities = []
    for row in city_rows:
        cities.append(
            {
                "region_key": row["region_key"],
                "label": row["label"],
                "macro_region_key": row["macro_region_key"],
                "population_10k": row["population_10k"],
                "population_million": row["population_million"],
                "population_source_year": row["population_source_year"],
                "y_position": city_positions[row["region_key"]],
            }
        )
    payload = {
        "figure": "contribution_decomposition_city_population",
        "figure_path": save_path,
        "figure_size_inches": {"width": figsize[0], "height": figsize[1]},
        "figure_size_mm": {
            "width": figsize[0] * 25.4,
            "height": figsize[1] * 25.4,
        },
        "axis": _report_axis(ax),
        "vertical_layout": {
            **_json_safe(vertical_layout),
            "axes_bbox_mm": _axis_bbox_mm(ax, figsize),
            "bottom_xlabel_bbox_mm": _artist_bbox_mm(fig, ax.xaxis.label),
        },
        "selected_region_keys": list(CONTRIBUTION_CITY_B_REGION_KEYS),
        "row_alignment": {
            "contract": vertical_layout["alignment_contract"],
            "row_count": len(rows),
            "y_positions": _report_array(y),
            "y_limits": _report_array(ax.get_ylim()),
            "city_names_repeated": False,
        },
        "lollipop_design": {
            "color": paper_color("contribution_break_even.d4a017"),
            "city_connection": "none",
            "overall_row": "min-max range across all locked study cities",
        },
        "population_coverage": _json_safe(
            prepared.get("population_coverage", {})
        ),
        "overall": {
            "label": "All study regions",
            "y_position": float(y[0]),
            "range": _json_safe(
                prepared.get("population_coverage", {}).get(
                    "all_study_city_range", {}
                )
            ),
        },
        "cities": _json_safe(cities),
        "export": {
            "bbox_inches": None,
            "pad_inches": 0.0,
            "vector_format": "PDF",
        },
    }
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, allow_nan=False)
    print(f"Contribution city population report saved to {report_path}")


@prepared_plot()
@_with_contribution_city_typography
def plot_contribution_decomposition_city_population(
    prepared: Dict,
    save_path: str = None,
    figsize: tuple = (44.0 / 25.4, CONTRIBUTION_CITY_PANEL_HEIGHT_MM / 25.4),
    for_latex: bool = False,
    plot_title: bool = False,
):
    """Plot representative-city population aligned to the contribution panel."""
    if not prepared:
        print("Contribution city population skipped — prepared data missing")
        return

    city_rows = _selected_city_component_rows(prepared.get("cities", []))
    rows = [_national_contribution_row(prepared.get("national", {})), *city_rows]
    y = _selected_city_y_positions(rows)
    if len(city_rows) != len(CONTRIBUTION_CITY_B_REGION_KEYS):
        raise RuntimeError(
            "population panel representative-city rows are incomplete: "
            f"expected={len(CONTRIBUTION_CITY_B_REGION_KEYS)}, actual={len(city_rows)}"
        )

    fig, ax = plt.subplots(figsize=figsize)
    vertical_layout = _set_selected_city_axes_position(
        ax,
        figsize,
        left=0.10,
        right=0.96,
    )
    gold = paper_color("contribution_break_even.d4a017")
    city_population = np.array(
        [float(row["population_million"]) for row in city_rows], dtype=float
    )
    city_y = y[1:]
    ax.hlines(city_y, 0.0, city_population, color=gold, linewidth=1.0, alpha=paper_alpha("contribution_break_even.alpha_0_82"))
    ax.scatter(
        city_population,
        city_y,
        s=24,
        color=gold,
        edgecolor=paper_color("ink.background"),
        linewidth=0.45,
        zorder=4,
    )

    population_range = prepared.get("population_coverage", {}).get(
        "all_study_city_range", {}
    )
    range_min = float(population_range["minimum_population_million"])
    range_max = float(population_range["maximum_population_million"])
    ax.hlines(y[0], range_min, range_max, color=gold, linewidth=2.0, zorder=3)
    ax.scatter(
        [range_min, range_max],
        [y[0], y[0]],
        s=28,
        color=gold,
        edgecolor=paper_color("ink.background"),
        linewidth=0.5,
        zorder=4,
    )

    ax.set_xlim(0.0, max(range_max, float(np.max(city_population))) * 1.08)
    ax.set_xlabel("Population (million)", fontsize=9.5)
    ax.set_yticks(y)
    ax.set_yticklabels([])
    ax.tick_params(axis="x", labelsize=8.2)
    ax.tick_params(axis="y", left=False)
    ax.grid(axis="x", alpha=paper_alpha("contribution_break_even.alpha_0_25"))
    ax.spines[["top", "right", "left"]].set_visible(False)
    _set_selected_city_y_limits(ax, y)
    if plot_title:
        ax.set_title("Population", fontsize=13)

    if save_path:
        if for_latex and not save_path.endswith(".pdf"):
            save_path = save_path.rsplit(".", 1)[0] + ".pdf"
        if plot_title:
            save_path = _add_title_suffix(save_path)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300)
        _write_contribution_city_population_report(
            prepared,
            save_path,
            figsize,
            fig,
            ax,
            rows,
            y,
        )
        plt.close(fig)
        print(f"Contribution city population saved to {save_path}")
    else:
        plt.show()
        plt.close(fig)


def _build_break_even_scenarios(exp_config: dict = None):
    """Create shared LCA params and scenarios for break-even calculations."""
    from drone_emission.emission.cal_single_delivery import (
        create_drone_params,
        create_ebike_params,
        get_emission_params,
    )
    from drone_emission.sensitivity.vectorized_lca import DroneScenario, EbikeScenario

    emission_cfg = get_emission_params(exp_config)
    ebike_params = create_ebike_params(exp_config)
    drone_params = create_drone_params(exp_config)
    ebike_scenario = EbikeScenario(
        kappa=float(getattr(ebike_params, "kappa", 1.0)),
        rider_weight=emission_cfg["rider_weight"],
        payload_kg=emission_cfg["payload_kg"],
    )
    drone_scenario = DroneScenario(
        payload_kg=emission_cfg["payload_kg"],
    )
    return ebike_params, drone_params, ebike_scenario, drone_scenario


def _break_even_zero_curve(
    delta_zero: np.ndarray,
    delta_slope: np.ndarray,
    gef_limits: tuple[float, float],
) -> np.ndarray:
    """Return the GEF at which delta_g equals zero for each detour ratio."""
    gef_min, gef_max = map(float, gef_limits)
    roots = np.full_like(delta_zero, np.nan, dtype=float)
    valid = np.abs(delta_slope) > 1e-12
    roots[valid] = -delta_zero[valid] / delta_slope[valid]
    roots[(roots < gef_min) | (roots > gef_max)] = np.nan
    return roots


def _break_even_zero_surface_raw(
    delta_zero: np.ndarray,
    delta_slope: np.ndarray,
) -> np.ndarray:
    """Return the unconstrained GEF at which delta_g equals zero."""
    roots = np.full_like(delta_zero, np.nan, dtype=float)
    valid = np.abs(delta_slope) > 1e-12
    roots[valid] = -delta_zero[valid] / delta_slope[valid]
    return roots


def _sample_break_even_observed_context(
    city_df: pd.DataFrame,
    data_dir: str,
    ebike_params,
    drone_params,
    ebike_scenario,
    drone_scenario,
    max_sample_rows: int,
    sample_seed: int,
) -> dict:
    """Sample observed OD/city contexts for the phase-diagram panel."""
    from dataclasses import replace

    from drone_emission.sensitivity.vectorized_lca import (
        compute_drone_per_order_g,
        compute_ebike_per_order_g,
    )

    empty_samples = pd.DataFrame(
        columns=[
            "region_key",
            "gef",
            "air_distance_km",
            "road_distance_km",
            "detour_ratio",
            "delta_g",
        ]
    )
    if city_df is None or city_df.empty:
        return {
            "samples": empty_samples,
            "city_reports": [],
            "skipped_cities": [],
            "annotation": "No observed city sample available",
        }

    city_rows = city_df.copy()
    if "region_level" in city_rows.columns:
        city_rows = city_rows[city_rows["region_level"] == "city"].copy()

    required_cols = {"region_key", "gef"}
    missing = sorted(required_cols - set(city_rows.columns))
    if missing:
        return {
            "samples": empty_samples,
            "city_reports": [],
            "skipped_cities": [
                {
                    "reason": "missing_city_columns",
                    "columns": missing,
                }
            ],
            "annotation": "No observed city sample available",
        }

    city_rows["gef"] = pd.to_numeric(city_rows["gef"], errors="coerce")
    city_rows = city_rows[city_rows["region_key"].notna() & city_rows["gef"].notna()]
    if city_rows.empty:
        return {
            "samples": empty_samples,
            "city_reports": [],
            "skipped_cities": [],
            "annotation": "No observed city sample available",
        }

    sample_frames = []
    city_reports = []
    skipped_cities = []

    for city_row in city_rows.sort_values("region_key").itertuples(index=False):
        region_key = str(city_row.region_key)
        od_path = _resolve_contribution_od_path(data_dir, region_key)
        if od_path is None:
            skipped_cities.append(
                {"region_key": region_key, "reason": "missing_od_csv"}
            )
            continue

        od_df = pd.read_csv(
            od_path,
            usecols=lambda col: col in {"distance_km", "road_distance_m"},
        )
        required_od_cols = {"distance_km", "road_distance_m"}
        if not required_od_cols.issubset(od_df.columns):
            skipped_cities.append(
                {
                    "region_key": region_key,
                    "reason": "missing_required_od_columns",
                    "columns_present": sorted(od_df.columns.tolist()),
                }
            )
            continue

        valid = od_df.copy()
        valid["distance_km"] = pd.to_numeric(valid["distance_km"], errors="coerce")
        valid["road_distance_m"] = pd.to_numeric(
            valid["road_distance_m"], errors="coerce"
        )
        valid["_road_km"] = valid["road_distance_m"] / 1000.0
        valid = valid[
            valid["distance_km"].notna()
            & (valid["distance_km"] > 0)
            & valid["_road_km"].notna()
            & (valid["_road_km"] > 0)
        ].copy()
        if valid.empty:
            skipped_cities.append(
                {"region_key": region_key, "reason": "no_valid_od_rows"}
            )
            continue

        sampled, sample_meta = _sample_city_valid_rows(
            valid_df=valid,
            sample_distance_km=valid["_road_km"].to_numpy(dtype=float),
            region_key=region_key,
            max_sample_rows=max_sample_rows,
            sample_seed=sample_seed,
        )
        if sampled.empty:
            skipped_cities.append(
                {"region_key": region_key, "reason": "sampling_returned_no_rows"}
            )
            continue

        gef = float(city_row.gef)
        air_distance_km = sampled["distance_km"].to_numpy(dtype=float)
        road_distance_km = sampled["_road_km"].to_numpy(dtype=float)
        delta_g = compute_drone_per_order_g(
            air_distance_km,
            replace(drone_params, e_use=gef),
            drone_scenario,
        ) - compute_ebike_per_order_g(
            road_distance_km,
            replace(ebike_params, e_use=gef),
            ebike_scenario,
        )
        sample_frames.append(
            pd.DataFrame(
                {
                    "region_key": region_key,
                    "gef": gef,
                    "air_distance_km": air_distance_km,
                    "road_distance_km": road_distance_km,
                    "detour_ratio": road_distance_km / air_distance_km,
                    "delta_g": delta_g,
                }
            )
        )
        city_reports.append(
            {
                "region_key": region_key,
                "gef": gef,
                "sample": sample_meta,
                "od_path": od_path,
            }
        )

    samples = (
        pd.concat(sample_frames, ignore_index=True)
        if sample_frames
        else empty_samples
    )
    if samples.empty:
        annotation = "No observed OD/city sample available"
    else:
        annotation = (
            f"Observed sample: {len(city_reports)} cities, "
            f"{len(samples):,} sampled OD pairs"
        )
    return {
        "samples": samples,
        "city_reports": city_reports,
        "skipped_cities": skipped_cities,
        "annotation": annotation,
    }


def _break_even_mesh_coefficients(
    air_distance_grid_km: np.ndarray,
    detour_ratios: np.ndarray,
    ebike_params,
    drone_params,
    ebike_scenario,
    drone_scenario,
) -> tuple[np.ndarray, np.ndarray]:
    """Return delta_g intercept and GEF slope over a distance-detour mesh."""
    from dataclasses import replace

    from drone_emission.sensitivity.vectorized_lca import (
        compute_drone_per_order_g,
        compute_ebike_per_order_g,
    )

    distance_axis = np.asarray(air_distance_grid_km, dtype=float)
    detour_axis = np.asarray(detour_ratios, dtype=float)
    distance_mesh, detour_mesh = np.meshgrid(distance_axis, detour_axis)
    road_distance_mesh = distance_mesh * detour_mesh

    air_distance_flat = distance_mesh.ravel()
    road_distance_flat = road_distance_mesh.ravel()
    drone_zero = compute_drone_per_order_g(
        air_distance_flat,
        replace(drone_params, e_use=0.0),
        drone_scenario,
    )
    ebike_zero = compute_ebike_per_order_g(
        road_distance_flat,
        replace(ebike_params, e_use=0.0),
        ebike_scenario,
    )
    drone_one = compute_drone_per_order_g(
        air_distance_flat,
        replace(drone_params, e_use=1.0),
        drone_scenario,
    )
    ebike_one = compute_ebike_per_order_g(
        road_distance_flat,
        replace(ebike_params, e_use=1.0),
        ebike_scenario,
    )

    delta_zero = (drone_zero - ebike_zero).reshape(distance_mesh.shape)
    delta_slope = ((drone_one - ebike_one) - (drone_zero - ebike_zero)).reshape(
        distance_mesh.shape
    )
    return delta_zero, delta_slope


def _break_even_observed_mask_data(
    observed_samples: pd.DataFrame,
    distance_limits: tuple[float, float],
    detour_limits: tuple[float, float],
    bin_width_km: float = 0.25,
) -> dict | None:
    """Return smoothed observed detour quantile bands for masking."""
    valid = observed_samples.copy()
    air_col = "air_distance_km" if "air_distance_km" in valid.columns else "distance_km"
    detour_col = "detour_ratio" if "detour_ratio" in valid.columns else "detour"
    if air_col not in valid.columns or detour_col not in valid.columns:
        return None

    distance_min, distance_max = map(float, distance_limits)
    detour_min, detour_max = map(float, detour_limits)
    valid[air_col] = pd.to_numeric(valid[air_col], errors="coerce")
    valid[detour_col] = pd.to_numeric(valid[detour_col], errors="coerce")
    valid = valid.dropna(subset=[air_col, detour_col]).copy()
    valid = valid[
        (valid[air_col] >= distance_min)
        & (valid[air_col] <= distance_max)
        & (valid[detour_col] >= detour_min)
        & (valid[detour_col] <= detour_max)
    ]
    if valid.empty:
        return None

    from scipy.ndimage import gaussian_filter1d

    x_edges = np.arange(distance_min, distance_max + bin_width_km, bin_width_km)
    if len(x_edges) < 2:
        return None
    x_centers = (x_edges[:-1] + x_edges[1:]) / 2
    mask_x = np.linspace(distance_min, distance_max, 700)
    quantile_levels = np.array([0.05, 0.10, 0.25, 0.40, 0.60, 0.75, 0.90, 0.95])
    quantile_curves = np.full((len(quantile_levels), len(x_centers)), np.nan)
    quantile_bands = [
        (None, 0, 0.66),
        (0, 1, 0.54),
        (1, 2, 0.42),
        (2, 3, 0.24),
        (3, 4, 0.00),
        (4, 5, 0.24),
        (5, 6, 0.42),
        (6, 7, 0.54),
        (7, None, 0.66),
    ]

    for x_i, (left, right) in enumerate(zip(x_edges[:-1], x_edges[1:])):
        if right >= distance_max:
            mask = (valid[air_col] >= left) & (valid[air_col] <= right)
        else:
            mask = (valid[air_col] >= left) & (valid[air_col] < right)
        values = valid.loc[mask, detour_col].to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        if values.size == 0:
            continue
        quantile_curves[:, x_i] = np.quantile(values, quantile_levels)

    observed = np.flatnonzero(np.isfinite(quantile_curves[0]))
    if observed.size <= 1:
        return None

    smooth_curves = np.full((len(quantile_levels), len(mask_x)), np.nan)
    for q_i in range(len(quantile_levels)):
        curve = quantile_curves[q_i]
        curve = np.interp(x_centers, x_centers[observed], curve[observed])
        curve = gaussian_filter1d(curve, sigma=1.0, mode="nearest")
        smooth_curves[q_i] = np.interp(mask_x, x_centers, curve)

    return {
        "mask_x": mask_x,
        "smooth_curves": smooth_curves,
        "quantile_bands": quantile_bands,
        "detour_limits": (detour_min, detour_max),
    }


def _select_break_even_gef_levels(
    city_gefs: np.ndarray,
    gef_limits: tuple[float, float],
) -> list[dict]:
    """Choose low/high GEF slices from the strongest low/high city-distribution peaks."""
    gef_min, gef_max = map(float, gef_limits)
    values = np.asarray(city_gefs, dtype=float)
    values = values[np.isfinite(values)]

    def _pick_peak_value(
        subset: np.ndarray,
        lower: float,
        upper: float,
        fallback: float,
    ) -> float:
        subset = np.asarray(subset, dtype=float)
        subset = subset[np.isfinite(subset)]
        subset = subset[(subset >= lower) & (subset < upper)]
        if subset.size == 0:
            return float(fallback)
        if subset.size == 1:
            return float(subset[0])

        bin_width = 0.025
        edges = np.arange(lower, upper + bin_width, bin_width)
        if len(edges) < 2:
            return float(np.median(subset))
        counts, _ = np.histogram(subset, bins=edges)
        peak_idx = int(np.argmax(counts))
        peak_low = edges[peak_idx]
        peak_high = edges[peak_idx + 1]
        if peak_idx == len(counts) - 1:
            mask = (subset >= peak_low) & (subset <= peak_high)
        else:
            mask = (subset >= peak_low) & (subset < peak_high)
        peak_values = np.sort(subset[mask])
        if peak_values.size == 0:
            return float(np.median(subset))
        return float(np.median(peak_values))

    if len(values) >= 8:
        low_peak = _pick_peak_value(
            values,
            lower=gef_min,
            upper=min(0.30, gef_max),
            fallback=0.14,
        )
        high_peak = _pick_peak_value(
            values,
            lower=max(0.30, gef_min),
            upper=gef_max + 1e-9,
            fallback=0.49,
        )
        selected = np.array([low_peak, high_peak], dtype=float)
    else:
        selected = np.array([0.14, 0.49], dtype=float)

    selected = np.clip(selected, gef_min, gef_max)
    return [
        {"label": "Low GEF", "gef": float(selected[0])},
        {"label": "High GEF", "gef": float(selected[1])},
    ]


@prepared_data
def _build_break_even_zero_lines(
    air_distance_grid_km: np.ndarray,
    detour_ratios: np.ndarray,
    delta_zero: np.ndarray,
    delta_slope: np.ndarray,
    gef_limits: tuple[float, float],
    gef_step: float = 0.05,
    line_points: int = 320,
) -> list[dict]:
    """Build fixed-GEF break-even curves in 3D space."""
    from scipy.interpolate import PchipInterpolator

    gef_min, gef_max = map(float, gef_limits)
    x_axis = np.asarray(air_distance_grid_km, dtype=float)
    y_axis = np.asarray(detour_ratios, dtype=float)
    gef_axis = np.arange(gef_min, gef_max + 1e-12, float(gef_step))
    gef_axis = gef_axis[gef_axis <= gef_max + 1e-12]
    delta_zero = np.asarray(delta_zero, dtype=float)
    delta_slope = np.asarray(delta_slope, dtype=float)

    def _smooth_segment(segment: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x_segment = np.asarray(segment[:, 0], dtype=float)
        y_segment = np.asarray(segment[:, 1], dtype=float)
        finite = np.isfinite(x_segment) & np.isfinite(y_segment)
        x_segment = x_segment[finite]
        y_segment = y_segment[finite]
        if len(x_segment) < 2:
            return x_segment, y_segment

        deltas = np.hypot(np.diff(x_segment), np.diff(y_segment))
        keep = np.concatenate(([True], deltas > 1e-10))
        x_segment = x_segment[keep]
        y_segment = y_segment[keep]
        if len(x_segment) < 4:
            return x_segment, y_segment

        distance = np.concatenate(
            ([0.0], np.cumsum(np.hypot(np.diff(x_segment), np.diff(y_segment))))
        )
        if distance[-1] <= 1e-10:
            return x_segment, y_segment

        line_distance = np.linspace(0.0, float(distance[-1]), line_points)
        x_line = PchipInterpolator(distance, x_segment)(line_distance)
        y_line = PchipInterpolator(distance, y_segment)(line_distance)
        return x_line, y_line

    lines = []
    for gef in gef_axis:
        delta_g = delta_zero + gef * delta_slope
        finite_delta = delta_g[np.isfinite(delta_g)]
        if not len(finite_delta) or finite_delta.min() > 0.0 or finite_delta.max() < 0.0:
            continue

        contour_fig, contour_ax = plt.subplots(figsize=(1, 1))
        try:
            contour = contour_ax.contour(
                x_axis,
                y_axis,
                delta_g,
                levels=[0.0],
            )
            segments = [
                np.asarray(segment, dtype=float)
                for segment in contour.allsegs[0]
                if len(segment) >= 2
            ]
        finally:
            plt.close(contour_fig)

        if not segments:
            continue

        segment = max(segments, key=len)
        x_line, y_line = _smooth_segment(segment)
        if len(x_line) < 2:
            continue
        lines.append(
            {
                "gef": float(gef),
                "x": np.asarray(x_line, dtype=float),
                "y": np.asarray(y_line, dtype=float),
                "z": np.full_like(x_line, float(gef), dtype=float),
            }
        )

    return lines


@prepared_data
def _build_break_even_marching_cubes_surface(
    air_distance_grid_km: np.ndarray,
    detour_ratios: np.ndarray,
    delta_zero: np.ndarray,
    delta_slope: np.ndarray,
    gef_limits: tuple[float, float],
    gef_points: int = 72,
) -> dict | None:
    """Extract the Delta g = 0 surface with skimage marching cubes."""
    from skimage.measure import marching_cubes

    x_axis = np.asarray(air_distance_grid_km, dtype=float)
    y_axis = np.asarray(detour_ratios, dtype=float)
    z_axis = np.linspace(float(gef_limits[0]), float(gef_limits[1]), int(gef_points))
    delta_zero = np.asarray(delta_zero, dtype=float)
    delta_slope = np.asarray(delta_slope, dtype=float)
    volume = (
        delta_zero[np.newaxis, :, :]
        + z_axis[:, np.newaxis, np.newaxis] * delta_slope[np.newaxis, :, :]
    )
    finite_volume = volume[np.isfinite(volume)]
    if not len(finite_volume) or finite_volume.min() > 0.0 or finite_volume.max() < 0.0:
        return None

    dz = float(z_axis[1] - z_axis[0]) if len(z_axis) > 1 else 1.0
    dy = float(y_axis[1] - y_axis[0]) if len(y_axis) > 1 else 1.0
    dx = float(x_axis[1] - x_axis[0]) if len(x_axis) > 1 else 1.0
    vertices, faces, _, _ = marching_cubes(
        volume,
        level=0.0,
        spacing=(dz, dy, dx),
        step_size=1,
        allow_degenerate=False,
    )
    if not len(vertices) or not len(faces):
        return None

    z = vertices[:, 0] + float(z_axis[0])
    y = vertices[:, 1] + float(y_axis[0])
    x = vertices[:, 2] + float(x_axis[0])
    return {
        "kind": "marching_cubes_surface",
        "x": x,
        "y": y,
        "z": z,
        "triangles": faces,
    }


def _select_sparse_break_even_overlay_curves(
    break_even_lines: list[dict],
    max_curves: int = 5,
) -> list[dict]:
    """Return a low-frequency subset of fixed-GEF break-even curves."""
    if not break_even_lines:
        return []

    sorted_lines = sorted(
        break_even_lines,
        key=lambda line: float(line.get("gef", np.nan)),
    )
    if len(sorted_lines) <= max_curves:
        return sorted_lines

    indices = np.linspace(0, len(sorted_lines) - 1, max_curves)
    unique_indices = np.unique(np.rint(indices).astype(int))
    return [sorted_lines[idx] for idx in unique_indices]


def prepare_break_even_phase_diagram_data(
    data_dir: str = None,
    city_df: pd.DataFrame = None,
    exp_config: dict = None,
    representative_air_distance_km: float = BREAK_EVEN_REPRESENTATIVE_AIR_DISTANCE_KM,
    curve_air_distances_km: tuple[float, ...] = BREAK_EVEN_CURVE_AIR_DISTANCES_KM,
    max_sample_rows: int = CONTRIBUTION_SAMPLE_LIMIT,
    sample_seed: int = CONTRIBUTION_SAMPLE_SEED,
) -> Dict:
    """Prepare break-even phase-diagram inputs for the paper figure.

    Args:
        data_dir: Root data directory for optional observed OD sampling.
        city_df: Optional city-level stats with ``region_key`` and ``gef``.
        exp_config: Experiment config dict for parameter overrides.
        representative_air_distance_km: Air distance retained for compatibility and
            surface annotation.
        curve_air_distances_km: Retained for API compatibility.
        max_sample_rows: Per-city OD sampling cap.
        sample_seed: Deterministic sampling seed.
    """
    ebike_params, drone_params, ebike_scenario, drone_scenario = (
        _build_break_even_scenarios(exp_config)
    )

    observed = {
        "samples": pd.DataFrame(
            columns=[
                "region_key",
                "gef",
                "air_distance_km",
                "road_distance_km",
                "detour_ratio",
                "delta_g",
            ]
        ),
        "city_reports": [],
        "skipped_cities": [],
        "annotation": "No observed city sample available",
    }
    if data_dir:
        observed = _sample_break_even_observed_context(
            city_df=city_df,
            data_dir=data_dir,
            ebike_params=ebike_params,
            drone_params=drone_params,
            ebike_scenario=ebike_scenario,
            drone_scenario=drone_scenario,
            max_sample_rows=max_sample_rows,
            sample_seed=sample_seed,
        )

    distance_min, distance_max = 0.2, 5.0
    detour_min, detour_max = BREAK_EVEN_DETOUR_LIMITS
    detour_ratios = np.linspace(detour_min, detour_max, 260)

    gef_min, gef_max = BREAK_EVEN_GEF_LIMITS
    grid_factors = np.linspace(gef_min, gef_max, 220)
    air_distance_grid_km = np.linspace(distance_min, distance_max, 260)
    delta_zero_grid, delta_slope_grid = _break_even_mesh_coefficients(
        air_distance_grid_km=air_distance_grid_km,
        detour_ratios=detour_ratios,
        ebike_params=ebike_params,
        drone_params=drone_params,
        ebike_scenario=ebike_scenario,
        drone_scenario=drone_scenario,
    )
    break_even_surface_raw = _break_even_zero_surface_raw(
        delta_zero=delta_zero_grid,
        delta_slope=delta_slope_grid,
    )
    break_even_surface = _break_even_zero_curve(
        delta_zero=delta_zero_grid,
        delta_slope=delta_slope_grid,
        gef_limits=BREAK_EVEN_GEF_LIMITS,
    )
    break_even_lines = _build_break_even_zero_lines(
        air_distance_grid_km=air_distance_grid_km,
        detour_ratios=detour_ratios,
        delta_zero=delta_zero_grid,
        delta_slope=delta_slope_grid,
        gef_limits=BREAK_EVEN_GEF_LIMITS,
    )
    observed_mask = _break_even_observed_mask_data(
        observed_samples=observed["samples"],
        distance_limits=(distance_min, distance_max),
        detour_limits=(detour_min, detour_max),
    )
    city_gefs = np.array(
        [report["gef"] for report in observed.get("city_reports", [])],
        dtype=float,
    )
    gef_panels = []
    for level in _select_break_even_gef_levels(city_gefs, BREAK_EVEN_GEF_LIMITS):
        gef_panels.append(
            {
                "label": level["label"],
                "gef": level["gef"],
                "delta_g": delta_zero_grid + float(level["gef"]) * delta_slope_grid,
            }
        )

    return {
        "axes": {
            "distance_limits": (float(distance_min), float(distance_max)),
            "detour_limits": (float(detour_min), float(detour_max)),
            "gef_limits": (float(gef_min), float(gef_max)),
        },
        "representative_air_distance_km": float(representative_air_distance_km),
        "curve_air_distances_km": tuple(float(v) for v in curve_air_distances_km),
        "air_distance_grid_km": air_distance_grid_km,
        "detour_ratios": detour_ratios,
        "grid_factors": grid_factors,
        "panel_a": {
            "break_even_gef": break_even_surface,
            "break_even_gef_raw": break_even_surface_raw,
            "delta_zero": delta_zero_grid,
            "delta_slope": delta_slope_grid,
            "break_even_lines": break_even_lines,
            "observed_mask": observed_mask,
        },
        "panel_b": {
            "gef_panels": gef_panels,
        },
        "panel_c": observed,
        "meta": {
            "sample_seed": int(sample_seed),
            "max_sample_rows_per_city": int(max_sample_rows),
            "ebike_kappa": float(ebike_scenario.kappa),
            "city_gefs": city_gefs,
        },
    }


def _break_even_heatmap_encoding(prepared: Dict) -> Dict:
    """Return the shared color encoding for the fixed-GEF heatmaps."""
    panel_b = prepared["panel_b"]
    finite_blocks = []
    for gef_panel in panel_b["gef_panels"]:
        values = np.asarray(gef_panel["delta_g"], dtype=float)
        finite_blocks.append(values[np.isfinite(values)])
    finite_blocks = [block for block in finite_blocks if len(block)]
    finite_delta = np.concatenate(finite_blocks) if finite_blocks else np.array([])
    max_abs = (
        float(np.nanpercentile(np.abs(finite_delta), 90))
        if len(finite_delta)
        else 1.0
    )
    max_abs = max(max_abs, 10.0)

    delta_bounds = np.linspace(-max_abs, max_abs, 9)
    delta_cmap = paper_cmap("contribution_break_even.delta", len(delta_bounds) - 1)
    delta_norm = BoundaryNorm(delta_bounds, ncolors=delta_cmap.N, clip=False)
    return {
        "bounds": delta_bounds,
        "cmap": delta_cmap,
        "norm": delta_norm,
    }


def _plot_break_even_surface_panel(
    fig,
    ax,
    prepared: Dict,
    plot_title: bool = False,
    font_scale: float = 1.0,
    show_annotations: bool = True,
    panel_mode: str = "lines",
    colorbar_ax=None,
    colorbar_ticks=None,
):
    """Draw the 3D break-even panel and return its extracted surface."""
    air_distance_grid_km = prepared["air_distance_grid_km"]
    detour_ratios = prepared["detour_ratios"]
    panel_a = prepared["panel_a"]
    distance_limits = prepared["axes"]["distance_limits"]
    detour_limits = prepared["axes"]["detour_limits"]
    gef_limits = prepared["axes"]["gef_limits"]
    break_even_lines = panel_a.get("break_even_lines", [])
    line_cmap = paper_cmap("contribution_break_even.gef")
    line_norm = plt.Normalize(vmin=gef_limits[0], vmax=gef_limits[1])
    surface_colorbar_mappable = None
    surface_mesh = None

    if panel_mode == "marching_cubes":
        surface_mesh = _build_break_even_marching_cubes_surface(
            air_distance_grid_km=air_distance_grid_km,
            detour_ratios=detour_ratios,
            delta_zero=panel_a["delta_zero"],
            delta_slope=panel_a["delta_slope"],
            gef_limits=gef_limits,
        )
        if surface_mesh:
            ax.plot_trisurf(
                surface_mesh["x"],
                surface_mesh["y"],
                surface_mesh["z"],
                triangles=surface_mesh["triangles"],
                cmap=line_cmap,
                norm=line_norm,
                linewidth=0.0,
                antialiased=False,
                shade=False,
                alpha=paper_alpha("contribution_break_even.alpha_1_0"),
                edgecolor="none",
                rasterized=True,
            )
            for overlay_line in _select_sparse_break_even_overlay_curves(
                break_even_lines,
                max_curves=5,
            ):
                ax.plot(
                    overlay_line["x"],
                    overlay_line["y"],
                    overlay_line["z"],
                    color=(0.16, 0.16, 0.16, 0.45),
                    linewidth=0.8,
                    solid_capstyle="round",
                )
            surface_colorbar_mappable = ScalarMappable(
                cmap=line_cmap,
                norm=line_norm,
            )
            surface_colorbar_mappable.set_array([])
    else:
        for line in break_even_lines:
            gef = float(line["gef"])
            line_color = line_cmap(line_norm(gef))
            ax.plot(
                line["x"],
                line["y"],
                line["z"],
                color=line_color,
                linewidth=1.55,
                alpha=paper_alpha("contribution_break_even.alpha_0_95"),
            )

    if show_annotations:
        if panel_mode == "marching_cubes":
            annotation_specs = (
                (0.90, "Break-even surface from marching cubes", 9.5, paper_color("contribution_break_even.444444")),
                (
                    0.82,
                    "Sparse fixed-GEF curves over a complete Delta g = 0 surface",
                    8.8,
                    paper_color("contribution_break_even.555555"),
                ),
                (0.05, "Surface marks Delta g = 0", 8.4, paper_color("contribution_break_even.555555")),
            )
        else:
            annotation_specs = (
                (0.90, "Break-even curves at fixed GEF", 9.5, paper_color("contribution_break_even.444444")),
                (0.82, "GEF spacing = 0.05 kg CO2e/kWh", 8.8, paper_color("contribution_break_even.555555")),
                (0.05, "Each line marks Delta g = 0", 8.4, paper_color("contribution_break_even.555555")),
            )
        for y, text, fontsize, color in annotation_specs:
            ax.text2D(
                0.02,
                y,
                text,
                transform=ax.transAxes,
                fontsize=fontsize * font_scale,
                color=color,
                bbox={
                    "boxstyle": "round,pad=0.22",
                    "facecolor": "white",
                    "alpha": paper_alpha("contribution_break_even.alpha_0_84"),
                    "edgecolor": "none",
                },
            )
    ax.set_xlim(*distance_limits)
    ax.set_ylim(*detour_limits)
    ax.set_zlim(*gef_limits)
    ax.set_xlabel("Air distance (km)", fontsize=10.5 * font_scale, labelpad=8)
    ax.set_ylabel("Detour ratio (road / air)", fontsize=10.5 * font_scale, labelpad=8)
    ax.set_zlabel("GEF (kg CO2e/kWh)", fontsize=10.5 * font_scale, labelpad=8)
    ax.set_proj_type("ortho")
    if panel_mode == "marching_cubes":
        ax.view_init(elev=20, azim=225)
    else:
        ax.view_init(elev=28, azim=-128)
    ax.grid(alpha=paper_alpha("contribution_break_even.alpha_0_2"))
    ax.xaxis.pane.set_alpha(paper_alpha("contribution_break_even.alpha_0_0"))
    ax.yaxis.pane.set_alpha(paper_alpha("contribution_break_even.alpha_0_0"))
    ax.zaxis.pane.set_alpha(paper_alpha("contribution_break_even.alpha_0_0"))
    if surface_colorbar_mappable is not None:
        if colorbar_ticks is None:
            colorbar_ticks = np.linspace(gef_limits[0], gef_limits[1], 4)
        if colorbar_ax is None:
            surface_colorbar = fig.colorbar(
                surface_colorbar_mappable,
                ax=ax,
                fraction=0.035,
                pad=0.05,
                shrink=0.62,
                ticks=colorbar_ticks,
            )
        else:
            surface_colorbar = fig.colorbar(
                surface_colorbar_mappable,
                cax=colorbar_ax,
                ticks=colorbar_ticks,
            )
        surface_colorbar.ax.set_title(
            "GEF\n(kg CO2e/kWh)",
            fontsize=8.2 * font_scale,
            pad=4,
        )
        surface_colorbar.ax.tick_params(labelsize=8.5 * font_scale)

    if plot_title:
        if panel_mode == "marching_cubes":
            title = "Break-even surface by marching cubes"
        else:
            title = "Break-even curves by GEF"
        ax.set_title(title, fontsize=12 * font_scale, pad=10)

    return surface_mesh


def _plot_break_even_heatmap_panel(
    ax,
    prepared: Dict,
    gef_panel: Dict,
    encoding: Dict,
    plot_title: bool = False,
    font_scale: float = 1.0,
    show_ylabel: bool = True,
    show_y_ticklabels: bool = True,
    show_annotation: bool = True,
    show_contour_label: bool = True,
):
    """Draw one fixed-GEF lifecycle-emission difference heatmap."""
    air_distance_grid_km = prepared["air_distance_grid_km"]
    detour_ratios = prepared["detour_ratios"]
    distance_limits = prepared["axes"]["distance_limits"]
    detour_limits = prepared["axes"]["detour_limits"]
    heatmap_detour_limits = (detour_limits[0], min(detour_limits[1], 2.4))
    X, Y = np.meshgrid(air_distance_grid_km, detour_ratios)
    delta_g = np.asarray(gef_panel["delta_g"], dtype=float)
    im = ax.imshow(
        delta_g,
        cmap=encoding["cmap"],
        aspect="auto",
        origin="lower",
        extent=[
            distance_limits[0],
            distance_limits[1],
            detour_limits[0],
            detour_limits[1],
        ],
        norm=encoding["norm"],
        interpolation="nearest",
    )
    if np.nanmin(delta_g) <= 0.0 <= np.nanmax(delta_g):
        contour = ax.contour(
            X,
            Y,
            delta_g,
            levels=[0.0],
            colors=paper_color("ink.black"),
            linewidths=1.3,
            zorder=6,
        )
        if show_contour_label:
            ax.clabel(
                contour,
                fmt={0.0: "Delta g = 0"},
                fontsize=8 * font_scale,
                inline=True,
            )
    ax.set_xlim(*distance_limits)
    ax.set_ylim(*heatmap_detour_limits)
    ax.set_xlabel("Air distance (km)", fontsize=11 * font_scale)
    ax.set_ylabel(
        "Detour ratio (road / air)" if show_ylabel else "",
        fontsize=11 * font_scale,
    )
    if not show_y_ticklabels:
        ax.tick_params(axis="y", left=False, labelleft=False)
    ax.grid(alpha=paper_alpha("contribution_break_even.alpha_0_15"))
    if plot_title:
        ax.set_title(
            f"{gef_panel['label']} ({gef_panel['gef']:.2f})",
            fontsize=11.5 * font_scale,
        )
    if show_annotation:
        ax.text(
            0.03,
            0.97,
            f"{gef_panel['label']}\nGEF = {gef_panel['gef']:.2f}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9.2 * font_scale,
            bbox={
                "boxstyle": "round,pad=0.22",
                "facecolor": "white",
                "alpha": paper_alpha("contribution_break_even.alpha_0_82"),
                "edgecolor": "none",
            },
            zorder=7,
        )
    return im


@prepared_plot()
def plot_break_even_contribution_decomposition_marching_cubes_subfigures(
    break_even_prepared: Dict,
    save_dir: str = None,
    for_latex: bool = False,
    plot_title: bool = False,
    selected_panels: tuple[str, ...] = ("a", "b", "c"),
):
    """Save standalone a-c panels from the marching-cubes composite figure.

    Args:
        break_even_prepared: Output of ``prepare_break_even_phase_diagram_data``.
        save_dir: Directory for the standalone subfigures. Shows plots if None.
        for_latex: If True, save as PDF.
        plot_title: If True, show titles.
        selected_panels: Panel letters to render from ``a``, ``b``, and ``c``.
    """
    if not break_even_prepared:
        print("Break-even marching-cubes subfigures skipped — prepared data missing")
        return {}

    valid_panels = {"a", "b", "c"}
    requested_panels = tuple(
        panel for panel in selected_panels if panel in valid_panels
    )
    if not requested_panels:
        print("Break-even marching-cubes subfigures skipped — no panels requested")
        return {}

    gef_panels = break_even_prepared["panel_b"].get("gef_panels", [])
    needs_heatmaps = any(panel in requested_panels for panel in ("b", "c"))
    if needs_heatmaps and len(gef_panels) < 2:
        print("Break-even marching-cubes subfigures skipped — GEF panels missing")
        return {}

    figure_name = "break_even_contribution_decomposition_marching_cubes"
    extension = ".pdf" if for_latex else ".png"
    width_mm = {"a": 120.0, "b": 55.0, "c": 65.0}
    height_mm = {"a": 85.0, "b": 52.0, "c": 52.0}
    font_scale = 0.72
    encoding = _break_even_heatmap_encoding(break_even_prepared)
    output_paths = {}
    panel_reports = {}

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    for panel in requested_panels:
        fig = plt.figure(
            figsize=(width_mm[panel] / 25.4, height_mm[panel] / 25.4),
            facecolor=paper_color("ink.background"),
        )
        output_path = None
        if save_dir:
            output_path = os.path.join(
                save_dir,
                f"{figure_name}_{panel}{extension}",
            )
            if plot_title:
                output_path = _add_title_suffix(output_path)

        if panel == "a":
            ax = fig.add_axes([0.06, 0.12, 0.76, 0.82], projection="3d")
            colorbar_ax = fig.add_axes([0.82, 0.22, 0.025, 0.58])
            gef_ticks = (0.05, 0.30, 0.55, 0.80)
            surface_mesh = _plot_break_even_surface_panel(
                fig,
                ax,
                break_even_prepared,
                plot_title=plot_title,
                font_scale=font_scale,
                show_annotations=False,
                panel_mode="marching_cubes",
                colorbar_ax=colorbar_ax,
                colorbar_ticks=gef_ticks,
            )
            ax.set_xlabel("Air distance (km)", fontsize=7.5, labelpad=3)
            ax.set_ylabel("Detour ratio (road / air)", fontsize=7.5, labelpad=3)
            ax.set_zlabel("GEF (kg CO2e/kWh)", fontsize=7.5, labelpad=3)
            ax.set_xticks((1, 2, 3, 4, 5))
            ax.set_yticks((1.0, 1.4, 1.8, 2.2, 2.4))
            ax.set_zticks(gef_ticks)
            ax.tick_params(axis="both", labelsize=6.5, pad=1)
            panel_reports[panel] = {
                "output_path": output_path,
                "figure_size_mm": {
                    "width": width_mm[panel],
                    "height": height_mm[panel],
                },
                "axis": _report_axis(ax),
                "colorbar_axis": _report_axis(colorbar_ax),
                "zlim_data": _report_array(ax.get_zlim()),
                "surface_method": (
                    "skimage.measure.marching_cubes on Delta g = 0 "
                    "with step_size = 1"
                ),
                "surface_vertex_count": (
                    int(len(surface_mesh["x"])) if surface_mesh else 0
                ),
                "surface_triangle_count": (
                    int(len(surface_mesh["triangles"])) if surface_mesh else 0
                ),
            }
        else:
            if panel == "b":
                ax = fig.add_axes([0.25, 0.22, 0.71, 0.72])
                gef_panel = gef_panels[0]
                show_ylabel = True
                show_y_ticklabels = True
            else:
                ax = fig.add_axes([0.08, 0.22, 0.59, 0.72])
                gef_panel = gef_panels[1]
                show_ylabel = False
                show_y_ticklabels = False

            heatmap_mappable = _plot_break_even_heatmap_panel(
                ax,
                break_even_prepared,
                gef_panel,
                encoding,
                plot_title=plot_title,
                font_scale=font_scale,
                show_ylabel=show_ylabel,
                show_y_ticklabels=show_y_ticklabels,
                show_annotation=False,
                show_contour_label=False,
            )
            ax.tick_params(axis="both", labelsize=6.5)
            if panel == "c":
                colorbar_ax = fig.add_axes([0.75, 0.22, 0.045, 0.72])
                colorbar = fig.colorbar(
                    heatmap_mappable,
                    cax=colorbar_ax,
                    boundaries=encoding["bounds"],
                    ticks=encoding["bounds"][::2],
                    extend="both",
                    spacing="proportional",
                )
                colorbar.set_label(
                    "Delta g = E_drone - E_ebike\n(g CO2e/order)",
                    fontsize=6.5,
                    labelpad=3,
                )
                colorbar.ax.tick_params(labelsize=6.0)

            panel_reports[panel] = {
                "output_path": output_path,
                "figure_size_mm": {
                    "width": width_mm[panel],
                    "height": height_mm[panel],
                },
                "axis": _report_axis(ax),
                "label": gef_panel["label"],
                "grid_emission_factor_kg_co2e_per_kwh": float(gef_panel["gef"]),
                "delta_g_summary_g_co2e_per_order": _array_summary(
                    gef_panel["delta_g"]
                ),
                "shows_shared_colorbar": panel == "c",
            }

        if output_path:
            fig.savefig(output_path, dpi=300, facecolor=paper_color("ink.background"))
            output_paths[panel] = output_path
            plt.close(fig)
            print(
                f"Break-even marching-cubes subfigure {panel} "
                f"saved to {output_path}"
            )
        else:
            plt.show()
            plt.close(fig)

    if save_dir:
        report_path = os.path.join(
            save_dir,
            "break_even_contribution_decomposition_"
            "marching_cubes_subfigures_report.json",
        )
        report = {
            "figure_name": (
                "break_even_contribution_decomposition_"
                "marching_cubes_subfigures"
            ),
            "requested_panels": list(requested_panels),
            "combined_target_size_mm": {
                "width": 120.0,
                "height": 141.0,
            },
            "external_layout": {
                "first_row": ["a"],
                "second_row": ["b", "c"],
                "row_gap_mm": 4.0,
            },
            "panel_order": ["a", "b", "c"],
            "panels": panel_reports,
            "shared_heatmap_encoding": {
                "quantity": "Delta g = E_drone - E_ebike",
                "unit": "g CO2e/order",
                "bounds": _report_array(encoding["bounds"]),
                "ticks": _report_array(encoding["bounds"][::2]),
                "negative_interpretation": "Drone has lower lifecycle emissions",
            },
            "panel_letters_drawn": False,
            "in_panel_gef_annotations_drawn": False,
        }
        with open(report_path, "w", encoding="utf-8") as report_file:
            json.dump(
                _json_safe(report),
                report_file,
                indent=2,
                ensure_ascii=False,
                allow_nan=False,
            )
        output_paths["report"] = report_path
        print(f"Break-even marching-cubes subfigure report saved to {report_path}")

    return output_paths
