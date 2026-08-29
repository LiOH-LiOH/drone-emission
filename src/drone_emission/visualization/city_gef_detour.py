"""Formal city gef detour renderers used only by the manifest registry."""

import json
import os
import pickle
import time

import pandas as pd


CITY_CONTEXT_FIGURES = {
    "shanghai_city_context": {
        "region_key": "shanghai",
        "display_name": "Shanghai",
        "power_region_key": "shanghai",
    },
    "chongqing_city_context": {
        "region_key": "chongqing",
        "display_name": "Chongqing",
        "power_region_key": "chongqing",
    },
    "chengdu_city_context": {
        "region_key": "chengdu",
        "display_name": "Chengdu",
        "power_region_key": "sichuan",
    },
    "beijing_city_context": {
        "region_key": "beijing",
        "display_name": "Beijing",
        "power_region_key": "beijing",
    },
    "guangzhou_city_context": {
        "region_key": "guangzhou",
        "display_name": "Guangzhou",
        "power_region_key": "guangdong",
    },
    "tianjin_city_context": {
        "region_key": "tianjin",
        "display_name": "Tianjin",
        "power_region_key": "tianjin",
    },
    "dali_city_context": {
        "region_key": "dalibaizuzizhizhou",
        "display_name": "Dali",
        "power_region_key": "yunnan",
    },
    "kunming_city_context": {
        "region_key": "kunming",
        "display_name": "Kunming",
        "power_region_key": "yunnan",
    },
}
CITY_GEF_CONTEXT_DETOUR_CONTEXTS = (
    CITY_CONTEXT_FIGURES["chengdu_city_context"],
    CITY_CONTEXT_FIGURES["beijing_city_context"],
    CITY_CONTEXT_FIGURES["shanghai_city_context"],
    CITY_CONTEXT_FIGURES["guangzhou_city_context"],
)
CITY_CONTEXT_MARKERS = {
    "chengdu": "o",
    "beijing": "s",
    "shanghai": "D",
    "guangzhou": "^",
}
CITY_GEF_COLOR_BIN_EDGES = tuple(value / 10.0 for value in range(1, 9))
CITY_GEF_SUMMARY_ROW_HSPACE = 0.32
CITY_GEF_BOTTOM_SCATTER_SIZE = 18
CITY_GEF_BOTTOM_SCATTER_ALPHA = 0.45

REDUCTION_DETOUR_CACHE_VERSION = 5
REDUCTION_DETOUR_CHUNK_ROWS = 250_000
POWER_GENERATION_2022_100M_KWH = {
    "shanghai": {
        "total": 955.0,
        "Hydro": 0.0,
        "Thermal": 912.60,
        "Wind": 24.56,
        "Solar": 17.82,
    },
    "chongqing": {
        "total": 998.0,
        "Hydro": 203.11,
        "Thermal": 755.15,
        "Wind": 34.02,
        "Solar": 5.56,
    },
    "beijing": {
        "total": 467.0,
        "Hydro": 9.43,
        "Thermal": 443.57,
        "Wind": 4.95,
        "Solar": 9.14,
    },
    "sichuan": {
        "total": 4846.0,
        "Hydro": 3886.56,
        "Thermal": 794.61,
        "Wind": 134.71,
        "Solar": 30.28,
    },
    "yunnan": {
        "total": 4017.0,
        "Hydro": 3282.91,
        "Thermal": 464.53,
        "Wind": 213.81,
        "Solar": 55.40,
    },
    "guangdong": {
        "total": 6366.0,
        "Hydro": 344.41,
        "Thermal": 4469.17,
        "Wind": 268.66,
        "Solar": 134.89,
    },
    "tianjin": {
        "total": 765.0,
        "Hydro": 0.14,
        "Thermal": 715.72,
        "Wind": 25.51,
        "Solar": 23.52,
    },
}
POWER_MIX_COLORS = {
    "Hydro": "#4c78a8",
    "Thermal": "#8c564b",
    "Wind": "#59a14f",
    "Solar": "#f2cf5b",
    "Other": "#9e9e9e",
}
POWER_MIX_GRAYSCALE_COLORS = {
    "Hydro": "#2B2B2B",
    "Thermal": "#606060",
    "Wind": "#969696",
    "Solar": "#C7C7C7",
    "Other": "#E6E6E6",
}
CITY_GEF_GRAYSCALE_VEHICLE_COLORS = {
    "E-bike": "#BDBDBD",
    "Drone": "#636363",
}
CITY_GEF_GRAYSCALE_BIN_COLORS = (
    "#D9D9D9",
    "#BDBDBD",
    "#A1A1A1",
    "#858585",
    "#696969",
    "#4D4D4D",
    "#313131",
)
CITY_GEF_GRAYSCALE_BOTTOM_SCATTER_ALPHA = 0.78


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


def _build_reduction_detour_binned_by_city(
    data_dir,
    selected,
    figure_name,
    exp_config=None,
    absolute_reduction_metric="total_t",
    use_binned_cache=False,
    bin_width_km=0.1,
):
    """Return binned reduction-detour data for selected city rows."""
    from dataclasses import asdict, replace

    import numpy as np

    from drone_emission.emission.cal_single_delivery import create_ebike_params, get_emission_params
    from drone_emission.sensitivity.vectorized_lca import EbikeScenario, compute_ebike_per_order_g
    from drone_emission.utils.config import get_cache_dir
    from drone_emission.visualization.city_gef_support import (
        _bin_reduction_and_detour_chunks,
    )

    if exp_config is None:
        print(f"{figure_name}: exp_config missing; cannot apply configured kappa")
        return None

    ebike_kappa = _get_ebike_kappa(exp_config)
    ebike_params = create_ebike_params(exp_config)
    emission_cfg = get_emission_params(exp_config)
    scenario = EbikeScenario(
        kappa=ebike_kappa,
        rider_weight=emission_cfg["rider_weight"],
        payload_kg=emission_cfg["payload_kg"],
    )

    selected_keys = selected["region_key"].tolist()
    gef_map = dict(zip(selected["region_key"], selected["gef"]))
    label_map = dict(zip(selected["region_key"], selected["region_name_en"]))
    labels = {key: label_map.get(key, key) for key in selected_keys}
    od_paths = {
        key: os.path.join(data_dir, "city", key, f"{key}_od_pairs.csv")
        for key in selected_keys
    }

    cache_path = None
    cache_meta = None
    available_keys = []
    source_files = {}
    for key in selected_keys:
        od_path = od_paths[key]
        if not os.path.exists(od_path):
            print(f"{figure_name}: missing {od_path} — skipping {key}")
            continue
        gef = gef_map.get(key)
        if gef is None or pd.isna(gef):
            print(f"{figure_name}: {key} missing province GEF — skipping")
            continue
        available_keys.append(key)
        source_files[key] = _file_cache_signature(od_path)

    if not available_keys:
        print(f"{figure_name}: no city OD data available")
        return None

    if use_binned_cache:
        cache_path = os.path.join(
            get_cache_dir("visualization", "paper"),
            f"{figure_name}_binned.pkl",
        )
        cache_meta = {
            "version": REDUCTION_DETOUR_CACHE_VERSION,
            "figure_name": figure_name,
            "city_keys": available_keys,
            "source_files": source_files,
            "gef": {key: float(gef_map[key]) for key in available_keys},
            "kappa": ebike_kappa,
            "ebike_params": asdict(ebike_params),
            "scenario": asdict(scenario),
            "bin_width_km": bin_width_km,
            "absolute_reduction_metric": absolute_reduction_metric,
        }
        if os.path.exists(cache_path):
            try:
                with open(cache_path, "rb") as f:
                    cache = pickle.load(f)
                if cache.get("meta") == cache_meta:
                    print(f"{figure_name}: loading binned data cache {cache_path}")
                    return cache["binned_by_city"], labels
                print(f"{figure_name}: binned data cache is stale; rebuilding")
            except Exception as exc:
                print(
                    f"{figure_name}: failed to load binned data cache ({exc}); "
                    "rebuilding"
                )

    binned_by_city = {}
    required = {
        "road_distance_m",
        "distance_km",
        "drone_lifecycle_g",
        "ebike_lifecycle_g",
        "annual_sales",
    }
    for key in available_keys:
        od_path = od_paths[key]
        columns = set(pd.read_csv(od_path, nrows=0).columns)
        if not required.issubset(columns):
            print(f"{figure_name}: {key} missing columns — skipping")
            continue
        max_air_distance_km = 0.0
        for distance_chunk in pd.read_csv(
            od_path,
            usecols=["distance_km"],
            dtype={"distance_km": "float64"},
            chunksize=REDUCTION_DETOUR_CHUNK_ROWS,
        ):
            distances = distance_chunk["distance_km"].to_numpy(copy=False)
            valid_distances = distances[np.isfinite(distances) & (distances > 0)]
            if valid_distances.size:
                max_air_distance_km = max(
                    max_air_distance_km, float(valid_distances.max())
                )
        if max_air_distance_km <= 0:
            print(f"{figure_name}: {key} has no valid air distances — skipping")
            continue

        gef = gef_map[key]
        ebike_pp = replace(ebike_params, e_use=float(gef))
        detour_totals = {
            "weighted_road_m": 0.0,
            "weighted_air_m": 0.0,
            "valid_path_count": 0,
            "invalid_path_count": 0,
        }

        def prepared_chunks():
            for od_chunk in pd.read_csv(
                od_path,
                usecols=sorted(required),
                dtype={column: "float64" for column in required},
                chunksize=REDUCTION_DETOUR_CHUNK_ROWS,
            ):
                road_m = od_chunk["road_distance_m"].to_numpy(copy=False)
                air_km = od_chunk["distance_km"].to_numpy(copy=False)
                annual_sales = od_chunk["annual_sales"].to_numpy(copy=False)
                valid_detour = (
                    np.isfinite(road_m)
                    & np.isfinite(air_km)
                    & np.isfinite(annual_sales)
                    & (road_m >= 0)
                    & (air_km > 0)
                    & (annual_sales > 0)
                )
                detour_totals["weighted_road_m"] += float(
                    np.sum(road_m[valid_detour] * annual_sales[valid_detour])
                )
                detour_totals["weighted_air_m"] += float(
                    np.sum(
                        air_km[valid_detour]
                        * 1000.0
                        * annual_sales[valid_detour]
                    )
                )
                detour_totals["valid_path_count"] += int(valid_detour.sum())
                detour_totals["invalid_path_count"] += int((~valid_detour).sum())
                od_chunk["ebike_lifecycle_g"] = compute_ebike_per_order_g(
                    road_m / 1000.0, ebike_pp, scenario
                )
                yield od_chunk

        (
            centres,
            reduction_pct,
            reduction_abs,
            detour_by_bin,
            max_km,
            oracle_drone_share_pct,
        ) = (
            _bin_reduction_and_detour_chunks(
                prepared_chunks(),
                bin_width_km,
                max_air_distance_km,
                absolute_reduction_metric,
            )
        )
        city_detour_ratio = (
            detour_totals["weighted_road_m"]
            / detour_totals["weighted_air_m"]
            if detour_totals["weighted_air_m"] > 0
            else np.nan
        )
        binned_by_city[key] = {
            "centres": centres,
            "reduction_pct": reduction_pct,
            "reduction_abs": reduction_abs,
            "detour_by_bin": detour_by_bin,
            "max_km": max_km,
            "oracle_drone_share_pct": oracle_drone_share_pct,
            "city_detour_ratio": city_detour_ratio,
            "valid_path_count": detour_totals["valid_path_count"],
            "invalid_path_count": detour_totals["invalid_path_count"],
        }

    if use_binned_cache and binned_by_city:
        if cache_path and cache_meta:
            with open(cache_path, "wb") as f:
                pickle.dump(
                    {"meta": cache_meta, "binned_by_city": binned_by_city},
                    f,
                    protocol=pickle.HIGHEST_PROTOCOL,
                )
            print(f"{figure_name}: wrote binned data cache {cache_path}")
    if binned_by_city:
        print(
            f"{figure_name}: rebuilt binned data at kappa={ebike_kappa} "
            f"for {len(binned_by_city)} cities"
        )
        return binned_by_city, labels

    print(f"{figure_name}: no city OD data available")
    return None


def _plot_compact_city_context_summary_panels(
    fig,
    subplot_spec,
    binned_by_city,
    labels,
    contexts,
    panel_start_letter="b",
    legend_ax=None,
    right_label_offset=0.055,
    grayscale=False,
):
    """Draw compact power-mix and distance panels for the city summary."""
    import numpy as np
    from matplotlib.patches import Patch

    power_mix_colors = (
        POWER_MIX_GRAYSCALE_COLORS if grayscale else POWER_MIX_COLORS
    )
    power_mix_handles = [
        Patch(
            facecolor=power_mix_colors[name],
            edgecolor="#777777" if grayscale else "none",
            linewidth=0.4 if grayscale else 0.0,
            label=name,
        )
        for name in ("Hydro", "Thermal", "Wind", "Solar", "Other")
    ]
    if legend_ax is None:
        middle = subplot_spec.subgridspec(
            2,
            1,
            height_ratios=[0.12, 0.88],
            hspace=0.08,
        )
        legend_ax = fig.add_subplot(middle[0])
        plot_spec = middle[1]
        legend_ax.legend(
            handles=power_mix_handles,
            loc="center",
            ncol=5,
            frameon=False,
            fontsize=7,
            handlelength=1.0,
            handletextpad=0.35,
            columnspacing=0.75,
            borderaxespad=0.0,
        )
    else:
        plot_spec = subplot_spec
        legend_ax.legend(
            handles=power_mix_handles,
            loc="center left",
            ncol=1,
            frameon=False,
            fontsize=7,
            title="Power mix",
            title_fontsize=7.5,
            handlelength=1.05,
            handletextpad=0.45,
            labelspacing=0.60,
            borderaxespad=0.0,
        )
        legend_ax.get_legend().get_title().set_fontweight("bold")
    legend_ax.set_axis_off()

    outer = plot_spec.subgridspec(1, len(contexts), wspace=0.24)
    visible_by_city = {}
    absolute_values = []
    for context in contexts:
        region_key = context["region_key"]
        binned = binned_by_city[region_key]
        centres = np.asarray(binned["centres"], dtype=float)
        reduction_abs = np.asarray(binned["reduction_abs"], dtype=float)
        x_max = min(5.0, float(binned["max_km"]))
        x_min = min(0.2, x_max)
        visible = (
            (centres >= x_min)
            & (centres <= x_max)
            & np.isfinite(centres)
            & np.isfinite(reduction_abs)
        )
        visible_by_city[region_key] = (centres, reduction_abs, visible, x_min, x_max)
        absolute_values.extend(reduction_abs[visible].tolist())
        if visible.sum() >= 2:
            slope, intercept = np.polyfit(
                centres[visible], reduction_abs[visible], deg=1
            )
            absolute_values.extend(
                [slope * x_min + intercept, slope * x_max + intercept]
            )

    finite_absolute = np.asarray(absolute_values, dtype=float)
    finite_absolute = finite_absolute[np.isfinite(finite_absolute)]
    if finite_absolute.size:
        absolute_min = min(0.0, float(finite_absolute.min()))
        absolute_max = max(0.0, float(finite_absolute.max()))
        absolute_pad = max((absolute_max - absolute_min) * 0.08, 0.5)
        shared_absolute_ylim = (
            absolute_min - absolute_pad,
            absolute_max + absolute_pad,
        )
    else:
        shared_absolute_ylim = (-1.0, 1.0)

    panel_records = []
    panel_start_ord = ord(panel_start_letter)
    for i, context in enumerate(contexts):
        region_key = context["region_key"]
        display_name = context["display_name"]
        power_region_key = context.get("power_region_key", region_key)
        block = outer[0, i].subgridspec(
            2,
            1,
            height_ratios=[0.34, 1.0],
            hspace=0.12,
        )
        ax_power = fig.add_subplot(block[0])
        ax_distance = fig.add_subplot(block[1])
        ax_absolute = ax_distance.twinx()
        ax_absolute.patch.set_visible(False)

        generation = POWER_GENERATION_2022_100M_KWH[power_region_key]
        total = float(generation["total"])
        components = ["Hydro", "Thermal", "Wind", "Solar"]
        values = [float(generation[name]) for name in components]
        remainder = total - sum(values)
        if remainder / total > 0.005:
            components.append("Other")
            values.append(remainder)
        shares = [value / total * 100.0 for value in values]
        left = 0.0
        for name, share in zip(components, shares):
            ax_power.barh(
                [0],
                [share],
                left=[left],
                height=0.52,
                color=power_mix_colors[name],
                edgecolor="white",
                linewidth=0.6 if grayscale else 0.45,
                zorder=3,
            )
            if share >= 12.0:
                ax_power.text(
                    left + share / 2.0,
                    0,
                    f"{share:.0f}%",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="white" if name in {"Hydro", "Thermal"} else "#222222",
                    fontweight="bold",
                    zorder=4,
                )
            left += share
        ax_power.set_xlim(0, 100)
        ax_power.set_ylim(-0.55, 0.55)
        ax_power.set_axis_off()

        power_letter = chr(panel_start_ord + i)
        ax_power.text(
            0.0,
            1.18,
            power_letter,
            transform=ax_power.transAxes,
            fontsize=9,
            fontweight="bold",
            ha="left",
            va="bottom",
        )
        city_label_artist = ax_power.text(
            0.15,
            1.18,
            display_name,
            transform=ax_power.transAxes,
            fontsize=8,
            fontweight="bold",
            ha="left",
            va="bottom",
        )

        binned = binned_by_city[region_key]
        centres, reduction_abs, visible, x_min, x_max = visible_by_city[region_key]
        drone_share = np.asarray(
            binned.get("oracle_drone_share_pct", np.full_like(centres, np.nan)),
            dtype=float,
        )
        bar_visible = visible & np.isfinite(drone_share)
        ax_distance.bar(
            centres[bar_visible],
            drone_share[bar_visible],
            width=0.156,
            color="#C7C7C7" if grayscale else "#9ecae1",
            edgecolor="white",
            linewidth=0.25,
            zorder=2,
        )
        ax_distance.set_xlim(x_min, x_max)
        ax_distance.set_ylim(0, 105)
        ax_distance.set_xticks([0.2, 2.5, 5.0])
        ax_distance.set_yticks([0, 50, 100])
        ax_distance.tick_params(axis="both", labelsize=7, length=2.5, pad=1.5)
        ax_distance.yaxis.set_ticks_position("right")
        ax_distance.grid(True, axis="y", alpha=0.18, linewidth=0.45)
        ax_distance.set_axisbelow(True)
        ax_distance.spines["top"].set_visible(False)
        ax_distance.spines["left"].set_visible(False)
        if i != len(contexts) - 1:
            ax_distance.tick_params(axis="y", right=False, labelright=False)
            ax_distance.spines["right"].set_visible(False)

        ax_absolute.scatter(
            centres[visible],
            reduction_abs[visible],
            s=7,
            color="#555555",
            alpha=0.72,
            linewidth=0,
            zorder=5,
        )
        if visible.sum() >= 2:
            slope, intercept = np.polyfit(
                centres[visible], reduction_abs[visible], deg=1
            )
            x_fit = np.array([x_min, x_max], dtype=float)
            ax_absolute.plot(
                x_fit,
                slope * x_fit + intercept,
                color="#111111",
                linewidth=0.9,
                alpha=0.82,
                zorder=6,
            )
        ax_absolute.axhline(0, color="#555555", linewidth=0.4, alpha=0.3)
        ax_absolute.set_ylim(*shared_absolute_ylim)
        ax_absolute.tick_params(
            axis="y",
            labelsize=7,
            labelcolor="#555555",
            length=2.5,
            pad=1.5,
        )
        ax_absolute.yaxis.set_ticks_position("left")
        ax_absolute.spines["top"].set_visible(False)
        ax_absolute.spines["right"].set_visible(False)
        if i != 0:
            ax_absolute.tick_params(axis="y", left=False, labelleft=False)
            ax_absolute.spines["left"].set_visible(False)

        panel_records.append(
            {
                "context": context,
                "panel_letter": power_letter,
                "power_panel_letter": power_letter,
                "reduction_panel_letter": None,
                "panel_anchor_points": None,
                "power_mix_ax": ax_power,
                "power_legend_ax": legend_ax,
                "reduction_ax": ax_distance,
                "absolute_ax": ax_absolute,
                "shared_absolute_ylim": shared_absolute_ylim,
                "city_label_artist": city_label_artist,
            }
        )

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for record in panel_records:
        label_bbox = record["city_label_artist"].get_window_extent(renderer)
        label_center_y = (label_bbox.y0 + label_bbox.y1) / 2.0
        label_start_point = fig.transFigure.inverted().transform(
            (label_bbox.x0, label_center_y)
        )
        record["city_label_position"] = (
            float(label_start_point[0]),
            float(label_start_point[1]),
        )

    middle_pos = plot_spec.get_position(fig)
    fig.text(
        middle_pos.x0 - 0.055,
        middle_pos.y0 + middle_pos.height * 0.32,
        "E-bike - Drone (g CO$_2$e/order)",
        rotation=90,
        fontsize=7.5,
        ha="center",
        va="center",
        color="#555555",
    )
    fig.text(
        middle_pos.x1 + right_label_offset,
        middle_pos.y0 + middle_pos.height * 0.32,
        "Optimal drone share (%)",
        rotation=90,
        fontsize=7.5,
        ha="center",
        va="center",
        color="#444444" if grayscale else "#3182bd",
    )
    fig.text(
        (middle_pos.x0 + middle_pos.x1) / 2.0,
        middle_pos.y0 - 0.030,
        "Air distance (km)",
        fontsize=7.5,
        ha="center",
        va="top",
    )
    return panel_records


def _create_city_gef_summary_legend_rail(
    fig,
    contexts,
    row_hspace=CITY_GEF_SUMMARY_ROW_HSPACE,
    grayscale=False,
):
    """Create three vertically aligned legend sections beside the summary."""
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    rail = fig.add_gridspec(
        3,
        1,
        left=0.855,
        right=0.985,
        bottom=0.055,
        top=0.985,
        hspace=row_hspace,
    )
    ax_top = fig.add_subplot(rail[0])
    ax_middle = fig.add_subplot(rail[1])
    ax_bottom = fig.add_subplot(rail[2])
    for ax in (ax_top, ax_middle, ax_bottom):
        ax.set_axis_off()

    if grayscale:
        vehicle_handles = [
            Patch(
                facecolor=CITY_GEF_GRAYSCALE_VEHICLE_COLORS["E-bike"],
                edgecolor="#222222",
                linewidth=0.45,
                label="E-bike",
            ),
            Patch(
                facecolor=CITY_GEF_GRAYSCALE_VEHICLE_COLORS["Drone"],
                edgecolor="#222222",
                linewidth=0.45,
                label="Drone",
            ),
        ]
        component_handles = [
            Patch(
                facecolor="#BDBDBD",
                edgecolor="#222222",
                linewidth=0.45,
                hatch="////",
                label="Production + EOL",
            ),
            Patch(
                facecolor="#BDBDBD",
                edgecolor="#222222",
                linewidth=0.45,
                label="Use phase",
            ),
        ]
    else:
        vehicle_handles = [
            Patch(facecolor="#4C8CBF", edgecolor="none", label="E-bike"),
            Patch(facecolor="#E05C2B", edgecolor="none", label="Drone"),
        ]
        component_handles = [
            Patch(
                facecolor="#777777",
                alpha=0.35,
                edgecolor="#555555",
                label="Production + EOL",
            ),
            Patch(
                facecolor="#777777",
                alpha=0.90,
                edgecolor="#555555",
                label="Use phase",
            ),
        ]
    vehicle_legend = ax_top.legend(
        handles=vehicle_handles,
        loc="upper left",
        bbox_to_anchor=(0.0, 0.98),
        frameon=False,
        fontsize=7,
        title="Vehicle",
        title_fontsize=7.5,
        handlelength=1.05,
        handletextpad=0.45,
        labelspacing=0.60,
        borderaxespad=0.0,
    )
    vehicle_legend.get_title().set_fontweight("bold")
    ax_top.add_artist(vehicle_legend)
    component_legend = ax_top.legend(
        handles=component_handles,
        loc="upper left",
        bbox_to_anchor=(0.0, 0.68),
        frameon=False,
        fontsize=7,
        title="Component",
        title_fontsize=7.5,
        handlelength=1.05,
        handletextpad=0.45,
        labelspacing=0.60,
        borderaxespad=0.0,
    )
    component_legend.get_title().set_fontweight("bold")
    ax_top.add_artist(component_legend)

    city_entries = [
        {
            "region_key": context["region_key"],
            "display_name": context["display_name"],
            "marker": CITY_CONTEXT_MARKERS.get(context["region_key"], "o"),
        }
        for context in contexts
    ]
    city_handles = [
        Line2D(
            [0],
            [0],
            marker=entry["marker"],
            markersize=4.8,
            markerfacecolor="none",
            markeredgecolor="#111111",
            markeredgewidth=0.9,
            linestyle="None",
            label=entry["display_name"],
        )
        for entry in city_entries
    ]
    city_legend = ax_top.legend(
        handles=city_handles,
        loc="lower left",
        bbox_to_anchor=(0.0, 0.02),
        frameon=False,
        fontsize=7,
        title="City",
        title_fontsize=7.5,
        handlelength=1.05,
        handletextpad=0.45,
        labelspacing=0.45,
        borderaxespad=0.0,
    )
    city_legend.get_title().set_fontweight("bold")

    ax_bottom.text(
        0.0,
        0.93,
        "GEF",
        transform=ax_bottom.transAxes,
        fontsize=7.5,
        fontweight="bold",
        ha="left",
        va="top",
    )
    cbar_ax = ax_bottom.inset_axes([0.08, 0.12, 0.25, 0.68])
    return {
        "top": ax_top,
        "middle": ax_middle,
        "bottom": ax_bottom,
        "colorbar": cbar_ax,
        "city_entries": city_entries,
    }


def _report_number(value):
    """Return a JSON-safe scalar with NaN/Inf represented as None."""
    import numpy as np

    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        value = float(value)
        return value if np.isfinite(value) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def _report_array(values):
    """Return a JSON-safe list from an array-like object."""
    return [_report_number(value) for value in values]


def _summary_stats(values):
    """Return compact summary statistics for an array-like numeric series."""
    import numpy as np

    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "q1": None,
            "q3": None,
            "min": None,
            "max": None,
            "whisker_low": None,
            "whisker_high": None,
        }
    q1, median, q3 = np.percentile(arr, [25, 50, 75])
    iqr = q3 - q1
    low_limit = q1 - 1.5 * iqr
    high_limit = q3 + 1.5 * iqr
    whisker_low = arr[arr >= low_limit].min()
    whisker_high = arr[arr <= high_limit].max()
    return {
        "count": int(arr.size),
        "mean": _report_number(arr.mean()),
        "median": _report_number(median),
        "q1": _report_number(q1),
        "q3": _report_number(q3),
        "min": _report_number(arr.min()),
        "max": _report_number(arr.max()),
        "whisker_low": _report_number(whisker_low),
        "whisker_high": _report_number(whisker_high),
    }


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


def _city_gef_components_report(df, axes, grayscale=False):
    """Return exact plotted data for the GEF component panel."""
    import numpy as np

    ax, ax_dist = axes
    gef_col = None
    for candidate in ("grid_emission_factor", "gef"):
        if candidate in df.columns and df[candidate].notna().any():
            gef_col = candidate
            break
    if gef_col is None or "total_odpair_annual_sales" not in df.columns:
        return {"status": "skipped", "reason": "missing GEF or sales column"}

    vehicle_specs = [
        {
            "label": "E-bike",
            "color": (
                CITY_GEF_GRAYSCALE_VEHICLE_COLORS["E-bike"]
                if grayscale
                else "#4C8CBF"
            ),
            "offset": -0.18,
            "use_col": "total_ebike_use_phase_emission_kg",
            "production_col": "total_ebike_production_emission_kg",
            "eol_col": "total_ebike_end_of_life_emission_kg",
        },
        {
            "label": "Drone",
            "color": (
                CITY_GEF_GRAYSCALE_VEHICLE_COLORS["Drone"]
                if grayscale
                else "#E05C2B"
            ),
            "offset": 0.18,
            "use_col": "total_drone_use_phase_emission_kg",
            "production_col": "total_drone_production_emission_kg",
            "eol_col": "total_drone_end_of_life_emission_kg",
        },
    ]
    required_cols = ["total_odpair_annual_sales"]
    for spec in vehicle_specs:
        required_cols.extend([spec["use_col"], spec["production_col"], spec["eol_col"]])

    plot_df = df.copy()
    if "region_level" in plot_df.columns:
        plot_df = plot_df[plot_df["region_level"] == "city"]
    for col in [gef_col, *required_cols]:
        plot_df[col] = pd.to_numeric(plot_df[col], errors="coerce")
    plot_df = plot_df.dropna(subset=[gef_col, *required_cols])
    valid_positive = (
        (plot_df[gef_col] > 0)
        & (plot_df[gef_col] <= 0.9)
        & (plot_df["total_odpair_annual_sales"] > 0)
    )
    for spec in vehicle_specs:
        valid_positive = (
            valid_positive
            & (plot_df[spec["use_col"]] > 0)
            & (plot_df[spec["production_col"]] >= 0)
            & (plot_df[spec["eol_col"]] >= 0)
        )
    plot_df = plot_df[valid_positive]

    bin_width = 0.1
    bin_edges = np.linspace(0.0, 0.9, 10)
    labels = [f"{right:.1f}" for right in bin_edges[1:]]
    plot_df["gef_range"] = pd.cut(
        plot_df[gef_col],
        bins=bin_edges,
        labels=labels,
        include_lowest=True,
        right=True,
    )
    x_all = (bin_edges[:-1] + bin_edges[1:]) / 2.0

    bars = []
    bins = []
    for idx, label in enumerate(labels):
        group = plot_df[plot_df["gef_range"] == label]
        if group.empty:
            continue
        orders_sum = group["total_odpair_annual_sales"].sum()
        bin_record = {
            "label_right_edge": label,
            "bin_left": _report_number(bin_edges[idx]),
            "bin_right": _report_number(bin_edges[idx + 1]),
            "bin_center": _report_number(x_all[idx]),
            "city_count": int(len(group)),
            "orders_sum": _report_number(orders_sum),
        }
        bins.append(bin_record)
        for spec in vehicle_specs:
            fixed = (
                (group[spec["production_col"]].sum() + group[spec["eol_col"]].sum())
                / orders_sum
                * 1000.0
            )
            use = group[spec["use_col"]].sum() / orders_sum * 1000.0
            x_center = x_all[idx] + spec["offset"] * bin_width
            bars.append(
                {
                    "vehicle": spec["label"],
                    "color": spec["color"],
                    "bin_label_right_edge": label,
                    "x_center": _report_number(x_center),
                    "width": _report_number(bin_width * 0.28),
                    "production_eol_segment": {
                        "bottom": 0.0,
                        "height_g_co2e_per_order": _report_number(fixed),
                        "top": _report_number(fixed),
                        "alpha": 1.0 if grayscale else 0.38,
                        "hatch": "////" if grayscale else None,
                        "edgecolor": "#222222" if grayscale else "white",
                    },
                    "use_phase_segment": {
                        "bottom": _report_number(fixed),
                        "height_g_co2e_per_order": _report_number(use),
                        "top": _report_number(fixed + use),
                        "alpha": 1.0 if grayscale else 0.88,
                        "hatch": None,
                        "edgecolor": "#222222" if grayscale else "white",
                    },
                    "total_label": {
                        "x": _report_number(x_center),
                        "y": _report_number(fixed + use),
                        "text": f"{fixed + use:.1f}",
                        "rotation": 90,
                    },
                    "orders_sum": _report_number(orders_sum),
                    "city_count": int(len(group)),
                }
            )

    density_x = plot_df[gef_col].to_numpy(dtype=float)
    if len(density_x) > 5 and np.unique(density_x).size > 1:
        from scipy.stats import gaussian_kde

        x_violin = np.linspace(bin_edges[0], bin_edges[-1], 240)
        density = gaussian_kde(density_x)(x_violin)
    else:
        x_violin = x_all
        density = np.array(
            [len(plot_df[plot_df["gef_range"] == label]) for label in labels],
            dtype=float,
        )
    if density.size and density.max() > 0:
        density = density / density.max()

    return {
        "status": "drawn",
        "panel_letter": "a",
        "panel_label": {
            "text": "a",
            "x_figure_fraction": _report_number(ax.get_position().x0 - 0.035),
            "y_figure_fraction": _report_number(ax.get_position().y1 + 0.006),
        },
        "axes": {
            "stacked_bars": _report_axis(ax),
            "gef_density": _report_axis(ax_dist),
        },
        "gef_column": gef_col,
        "bin_width": bin_width,
        "bin_edges": _report_array(bin_edges),
        "bins": bins,
        "bars": bars,
        "density_curve": [
            {"x": _report_number(x), "y": _report_number(y)}
            for x, y in zip(x_violin, density)
        ],
        "city_rug_marks": [
            {"x": _report_number(x), "y": 0.0}
            for x in density_x
        ],
    }


def _city_context_summary_report(
    binned_by_city,
    labels,
    panel_records,
    bar_metric="reduction_pct",
    grayscale=False,
):
    """Return exact plotted data for the grouped city context row."""
    import numpy as np

    if bar_metric not in {"reduction_pct", "oracle_drone_share_pct"}:
        raise ValueError(
            "bar_metric must be 'reduction_pct' or 'oracle_drone_share_pct'"
        )

    reports = []
    for record in panel_records:
        context = record["context"]
        region_key = context["region_key"]
        binned = binned_by_city[region_key]
        centres = np.asarray(binned["centres"], dtype=float)
        reduction_pct = np.asarray(binned["reduction_pct"], dtype=float)
        reduction_abs = np.asarray(binned["reduction_abs"], dtype=float)
        oracle_drone_share_pct = np.asarray(
            binned.get("oracle_drone_share_pct", np.full_like(centres, np.nan)),
            dtype=float,
        )
        if bar_metric == "oracle_drone_share_pct":
            bar_values = oracle_drone_share_pct
            bar_color = "#C7C7C7" if grayscale else "#9ecae1"
            bar_label = "oracle_drone_share_pct"
        else:
            bar_values = reduction_pct
            bar_color = None
            bar_label = "reduction_pct"
        detour_by_bin = binned.get("detour_by_bin") or [np.array([]) for _ in centres]
        max_km = float(binned["max_km"])
        x_min = min(0.2, max_km)
        x_max = min(5.0, max_km)
        visible = (centres >= x_min) & (centres <= x_max)
        finite_abs = visible & np.isfinite(centres) & np.isfinite(reduction_abs)
        x_abs = centres[finite_abs]
        y_abs = reduction_abs[finite_abs]

        linear_fit = None
        if x_abs.size >= 2:
            slope, intercept = np.polyfit(x_abs, y_abs, deg=1)
            x_fit = np.array([x_min, x_max], dtype=float)
            linear_fit = {
                "slope": _report_number(slope),
                "intercept": _report_number(intercept),
                "line_points": [
                    {
                        "x": _report_number(x),
                        "y": _report_number(slope * x + intercept),
                    }
                    for x in x_fit
                ],
                "equation": (
                    f"y = {slope:.1f}x "
                    f"{'+' if intercept >= 0 else '-'} {abs(intercept):.1f}"
                ),
            }

        show_saturation_fit = region_key not in {"dalibaizuzizhizhou", "chengdu"}
        saturation_fit = None
        if show_saturation_fit and bar_metric == "reduction_pct":
            fit_mask = visible & np.isfinite(centres) & np.isfinite(reduction_pct)
            x_fit_data = centres[fit_mask]
            y_fit_data = reduction_pct[fit_mask]
            if x_fit_data.size >= 4:
                try:
                    from scipy.optimize import curve_fit

                    x0 = float(x_min)

                    def saturation_model(x, y0, amplitude, rate):
                        return y0 + amplitude * (1.0 - np.exp(-rate * (x - x0)))

                    initial = [
                        float(y_fit_data[0]),
                        float(y_fit_data[-1] - y_fit_data[0]),
                        0.8,
                    ]
                    params, _ = curve_fit(
                        saturation_model,
                        x_fit_data,
                        y_fit_data,
                        p0=initial,
                        bounds=([-np.inf, -np.inf, 0.0], [np.inf, np.inf, np.inf]),
                        maxfev=10000,
                    )
                    x_fit = np.linspace(x_min, x_max, 160)
                    y_fit = saturation_model(x_fit, *params)
                    y0, amplitude, rate = params
                    saturation_fit = {
                        "y0": _report_number(y0),
                        "amplitude": _report_number(amplitude),
                        "rate": _report_number(rate),
                        "x0": _report_number(x0),
                        "line_points": [
                            {"x": _report_number(x), "y": _report_number(y)}
                            for x, y in zip(x_fit, y_fit)
                        ],
                        "equation": (
                            f"y = {y0:.1f} "
                            f"{'+' if amplitude >= 0 else '-'} {abs(amplitude):.1f}"
                            f"(1 - e^(-{rate:.2f}(x - {x0:.1f})))"
                        ),
                    }
                except Exception as exc:
                    saturation_fit = {"status": "fit_failed", "error": str(exc)}

        generation = POWER_GENERATION_2022_100M_KWH[context.get("power_region_key", region_key)]
        total = generation["total"]
        components = ["Hydro", "Thermal", "Wind", "Solar"]
        values = [generation[name] for name in components]
        remainder = total - sum(values)
        if remainder / total > 0.005:
            components.append("Other")
            values.append(remainder)

        reports.append(
            {
                "panel_letter": record["panel_letter"],
                "panel_label": {
                    "text": record["power_panel_letter"],
                    "x_figure_fraction": _report_number(
                        record["power_mix_ax"].get_position().x0 - 0.010
                    ),
                    "y_figure_fraction": _report_number(
                        record["power_mix_ax"].get_position().y1 + 0.035
                    ),
                },
                "reduction_panel_label": (
                    {
                        "text": record["reduction_panel_letter"],
                        "x_figure_fraction": _report_number(
                            record["reduction_ax"].get_position().x0 - 0.010
                        ),
                        "y_figure_fraction": _report_number(
                            record["reduction_ax"].get_position().y1 + 0.012
                        ),
                    }
                    if record["reduction_panel_letter"] is not None
                    else None
                ),
                "city_label_position": {
                    "text": context["display_name"],
                    "x_figure_fraction": _report_number(
                        record.get(
                            "city_label_position",
                            (record["power_mix_ax"].get_position().x0 + 0.006, 0),
                        )[0]
                    ),
                    "y_figure_fraction": _report_number(
                        record.get(
                            "city_label_position",
                            (0, record["power_mix_ax"].get_position().y1 + 0.035),
                        )[1]
                    ),
                },
                "panel_anchor_points": (
                    {
                        key: {
                            "x_figure_fraction": _report_number(point[0]),
                            "y_figure_fraction": _report_number(point[1]),
                        }
                        for key, point in record["panel_anchor_points"].items()
                    }
                    if record["panel_anchor_points"] is not None
                    else None
                ),
                "region_key": region_key,
                "city_label": labels.get(region_key, context["display_name"]),
                "display_name": context["display_name"],
                "city_detour_ratio": _report_number(
                    binned.get("city_detour_ratio")
                ),
                "valid_path_count": int(binned.get("valid_path_count", 0)),
                "invalid_path_count": int(binned.get("invalid_path_count", 0)),
                "axes": {
                    "power_mix": _report_axis(record["power_mix_ax"]),
                    "power_legend": _report_axis(record["power_legend_ax"]),
                    "bar_axis": _report_axis(record["reduction_ax"]),
                    "reduction_bars": _report_axis(record["reduction_ax"]),
                    "absolute_scatter": _report_axis(record["absolute_ax"]),
                },
                "bar_metric": bar_label,
                "power_mix": [
                    {
                        "component": name,
                        "value_100m_kwh": _report_number(value),
                        "share_pct": _report_number(value / total * 100.0),
                        "color": (
                            POWER_MIX_GRAYSCALE_COLORS[name]
                            if grayscale
                            else POWER_MIX_COLORS[name]
                        ),
                    }
                    for name, value in zip(components, values)
                ],
                "distance_bins": [
                    {
                        "x_center_air_km": _report_number(center),
                        "bar_width_km": 0.2 * 0.78,
                        "bar_metric": bar_label,
                        "bar_height": _report_number(bar_value),
                        "bar_bottom": 0.0,
                        "bar_top": _report_number(bar_value),
                        "bar_color": (
                            bar_color
                            if bar_color is not None
                            else (
                                "#2ca02c"
                                if (np.isfinite(pct) and pct >= 0)
                                else "#d62728"
                            )
                        ),
                        "oracle_drone_share_pct": _report_number(oracle_share),
                        "reduction_pct_height": _report_number(pct),
                        "reduction_bar_bottom": 0.0,
                        "reduction_bar_top": _report_number(pct),
                        "reduction_bar_color": (
                            "#2ca02c"
                            if (np.isfinite(pct) and pct >= 0)
                            else "#d62728"
                        ),
                        "absolute_scatter_y_g_co2e_per_order": _report_number(abs_value),
                        "visible_in_panel": bool(is_visible),
                        "detour_ratio_summary": _summary_stats(detours),
                    }
                    for (
                        center,
                        pct,
                        oracle_share,
                        bar_value,
                        abs_value,
                        detours,
                        is_visible,
                    )
                    in zip(
                        centres,
                        reduction_pct,
                        oracle_drone_share_pct,
                        bar_values,
                        reduction_abs,
                        detour_by_bin,
                        visible,
                    )
                ],
                "absolute_scatter_points": [
                    {"x_air_km": _report_number(x), "y_g_co2e_per_order": _report_number(y)}
                    for x, y in zip(x_abs, y_abs)
                ],
                "absolute_linear_fit": linear_fit,
                "reduction_saturation_fit": saturation_fit,
                "shared_absolute_ylim": (
                    _report_array(record["shared_absolute_ylim"])
                    if record.get("shared_absolute_ylim") is not None
                    else None
                ),
            }
        )
    return reports


def _add_city_gef_xaxis_markers(fig, axes, selected, contexts):
    """Add selected city province-GEF markers to the top panel x-axis."""
    _, ax_dist = axes
    marker_records = []

    for context in contexts:
        region_key = context["region_key"]
        marker = CITY_CONTEXT_MARKERS.get(region_key, "o")
        city_rows = selected[selected["region_key"] == region_key]
        if city_rows.empty or "gef" not in city_rows.columns:
            continue

        gef = pd.to_numeric(city_rows["gef"], errors="coerce").dropna()
        if gef.empty:
            continue

        x = float(gef.iloc[0])
        y = 0.0
        ax_dist.scatter(
            [x],
            [y],
            s=22,
            marker=marker,
            facecolors="none",
            edgecolors="#111111",
            linewidths=1.0,
            zorder=8,
            clip_on=False,
        )
        display_point = ax_dist.transData.transform((x, y))
        fig_point = fig.transFigure.inverted().transform(display_point)
        marker_records.append(
            {
                "region_key": region_key,
                "display_name": context["display_name"],
                "power_region_key": context.get("power_region_key", region_key),
                "province": _report_number(city_rows.iloc[0].get("province")),
                "gef": _report_number(x),
                "marker": marker,
                "x_data": _report_number(x),
                "y_data": _report_number(y),
                "x_figure_fraction": _report_number(fig_point[0]),
                "y_figure_fraction": _report_number(fig_point[1]),
            }
        )

    return marker_records


def _add_city_detour_reduction_markers(
    fig,
    axes,
    df,
    contexts,
    reduction_col,
    panel_letters=("f", "g"),
):
    """Add selected city markers to the bottom detour-reduction scatter panels."""
    ax_low, _, ax_high = axes
    feature_var = "detour_ratio"
    group_var = "grid_emission_factor"
    threshold = 0.3
    axis_by_group = {
        "low_gef": ax_low,
        "high_gef": ax_high,
    }
    panel_letter_by_group = {
        "low_gef": panel_letters[0],
        "high_gef": panel_letters[1],
    }
    marker_records = []

    needed = [feature_var, reduction_col, group_var]
    if "region_key" not in df.columns or any(col not in df.columns for col in needed):
        return marker_records

    for context in contexts:
        region_key = context["region_key"]
        marker = CITY_CONTEXT_MARKERS.get(region_key, "o")
        city_rows = df[df["region_key"] == region_key].copy()
        if city_rows.empty:
            continue

        for col in needed:
            city_rows[col] = pd.to_numeric(city_rows[col], errors="coerce")
        city_rows = city_rows.dropna(subset=needed)
        if city_rows.empty:
            continue

        row = city_rows.iloc[0]
        x = float(row[feature_var])
        y = float(row[reduction_col])
        gef = float(row[group_var])
        axis_name = "low_gef" if gef <= threshold else "high_gef"
        ax = axis_by_group[axis_name]
        ax.scatter(
            [x],
            [y],
            s=28,
            marker=marker,
            facecolors="none",
            edgecolors="#111111",
            linewidths=1.1,
            zorder=35,
            clip_on=False,
        )
        display_point = ax.transData.transform((x, y))
        fig_point = fig.transFigure.inverted().transform(display_point)
        marker_records.append(
            {
                "region_key": region_key,
                "display_name": context["display_name"],
                "axis_name": axis_name,
                "panel_letter": panel_letter_by_group[axis_name],
                "x_detour_ratio": _report_number(x),
                "y_reduction_pct": _report_number(y),
                "grid_emission_factor": _report_number(gef),
                "marker": marker,
                "x_figure_fraction": _report_number(fig_point[0]),
                "y_figure_fraction": _report_number(fig_point[1]),
            }
        )

    return marker_records


def _detour_reduction_scatter_report(
    df,
    reduction_col,
    axes,
    panel_letters=("f", "g"),
    compact=False,
    grayscale=False,
):
    """Return exact plotted data for the bottom detour-reduction scatter panels."""
    import numpy as np
    from scipy import stats

    ax_low, cbar_ax, ax_high = axes
    feature_var = "detour_ratio"
    group_var = "grid_emission_factor"
    threshold = 0.3
    valid_mask = df[[feature_var, reduction_col, group_var]].notna().all(axis=1)
    plot_df = df.loc[valid_mask].copy()
    low_mask = plot_df[group_var] <= threshold
    masks = [low_mask, ~low_mask]
    labels = (f"Low GEF ($\\leq$ {threshold})", f"High GEF (> {threshold})")
    line_colors = (
        ["#4D4D4D", "#4D4D4D"]
        if grayscale
        else ["#006837", "#A50F15"]
    )
    axis_names = ["low_gef", "high_gef"]
    axis_objs = [ax_low, ax_high]

    x_data_min = plot_df[feature_var].min()
    x_data_max = plot_df[feature_var].max()
    y_data_min = plot_df[reduction_col].min()
    y_data_max = plot_df[reduction_col].max()
    x_pad = (x_data_max - x_data_min) * 0.03 or 0.1
    y_pad = (y_data_max - y_data_min) * 0.06 or 1.0
    x_global_min = x_data_min - x_pad
    x_global_max = x_data_max + x_pad

    panels = []
    for axis_name, ax, mask, label, color, panel_letter in zip(
        axis_names, axis_objs, masks, labels, line_colors, panel_letters
    ):
        panel_df = plot_df.loc[mask]
        points = []
        for _, row in panel_df.iterrows():
            points.append(
                {
                    "region_key": _report_number(row.get("region_key")),
                    "region_name_en": _report_number(row.get("region_name_en")),
                    "region_name": _report_number(row.get("region_name")),
                    "x_detour_ratio": _report_number(row[feature_var]),
                    "y_reduction_pct": _report_number(row[reduction_col]),
                    "grid_emission_factor": _report_number(row[group_var]),
                }
            )

        regression = None
        x = panel_df[feature_var].to_numpy(dtype=float)
        y = panel_df[reduction_col].to_numpy(dtype=float)
        if len(x) >= 3 and np.sum((x - np.mean(x)) ** 2) > 0:
            slope, intercept, r_value, p_value, stderr = stats.linregress(x, y)
            x_line = np.linspace(x_global_min, x_global_max, 100)
            y_line = slope * x_line + intercept
            n = len(x)
            se = np.sqrt(np.sum((y - (slope * x + intercept)) ** 2) / (n - 2))
            t_val = stats.t.ppf(0.975, n - 2)
            x_mean = np.mean(x)
            ci = t_val * se * np.sqrt(
                1.0 / n + (x_line - x_mean) ** 2 / np.sum((x - x_mean) ** 2)
            )
            regression = {
                "slope": _report_number(slope),
                "intercept": _report_number(intercept),
                "r_value": _report_number(r_value),
                "p_value": _report_number(p_value),
                "stderr": _report_number(stderr),
                "line_color": color,
                "line_points": [
                    {
                        "x": _report_number(xv),
                        "y": _report_number(yv),
                        "ci_low": _report_number(yv - civ),
                        "ci_high": _report_number(yv + civ),
                    }
                    for xv, yv, civ in zip(x_line, y_line, ci)
                ],
            }

        panels.append(
            {
                "panel_letter": panel_letter,
                "axis_name": axis_name,
                "label_text": label,
                "axes": _report_axis(ax),
                "panel_label": {
                    "text": panel_letter,
                    "x_axes_fraction": 0.0 if compact else -0.10,
                    "y_axes_fraction": 1.01 if compact else 1.02,
                },
                "group_label": {
                    "text": label,
                    "x_axes_fraction": 0.03,
                    "y_axes_fraction": 0.96,
                },
                "scatter_style": {
                    "size": CITY_GEF_BOTTOM_SCATTER_SIZE if compact else 24,
                    "alpha": (
                        CITY_GEF_GRAYSCALE_BOTTOM_SCATTER_ALPHA
                        if grayscale and compact
                        else CITY_GEF_BOTTOM_SCATTER_ALPHA
                        if compact
                        else 0.68
                    ),
                    "edgecolors": "none",
                    "zorder": 10,
                },
                "points": points,
                "regression": regression,
            }
        )

    return {
        "axes": {
            "colorbar": _report_axis(cbar_ax),
        },
        "feature_var": feature_var,
        "result_var": reduction_col,
        "group_var": group_var,
        "threshold": threshold,
        "panels": panels,
    }


def _write_city_gef_context_detour_report(
    save_path,
    figure_name,
    figure_size,
    top_report,
    city_reports,
    bottom_report,
    connector_lines,
    legend_rail=None,
    render_style="color",
):
    """Write JSON and Markdown reports for the city GEF context figure."""
    report_base = save_path.rsplit(".", 1)[0] + "_report"
    json_path = report_base + ".json"
    md_path = report_base + ".md"
    payload = {
        "figure_name": figure_name,
        "figure_path": save_path,
        "render_style": render_style,
        "figure_size_inches": {
            "width": figure_size[0],
            "height": figure_size[1],
        },
        "coordinate_convention": (
            "All x/y values are in Matplotlib data coordinates for the stated axis; "
            "axis bbox values are figure-fraction coordinates."
        ),
        "panels": {
            "top_city_gef_scatter_components": top_report,
            "middle_city_context": city_reports,
            "bottom_detour_ratio_vs_reduction_by_gef": bottom_report,
        },
        "connector_lines": connector_lines,
        "legend_rail": legend_rail,
    }
    if render_style == "grayscale":
        payload["visual_encoding"] = {
            "vehicle_colors": CITY_GEF_GRAYSCALE_VEHICLE_COLORS,
            "production_eol_hatch": "////",
            "use_phase_hatch": None,
            "power_mix_colors": POWER_MIX_GRAYSCALE_COLORS,
            "gef_bin_colors_low_to_high": CITY_GEF_GRAYSCALE_BIN_COLORS,
            "gef_gray_direction": "higher GEF uses darker gray",
            "regression_linestyle": "solid",
        }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, allow_nan=False)

    lines = [
        f"# {figure_name} Figure Report",
        "",
        f"- Figure path: `{save_path}`",
        f"- JSON data path: `{json_path}`",
        f"- Render style: `{render_style}`",
        f"- Figure size: {figure_size[0]} x {figure_size[1]} inches",
        "- Coordinate convention: data coordinates for plotted elements; figure-fraction coordinates for axes.",
        f"- Connector line groups: {len(connector_lines)}",
        f"- Legend rail orientation: {(legend_rail or {}).get('orientation')}",
        f"- GEF bin edges: {(legend_rail or {}).get('gef_bin_edges')}",
        f"- City marker entries: {(legend_rail or {}).get('city_markers')}",
        "",
        "## Top: city_gef_scatter_components",
        "",
        f"- GEF column: `{top_report.get('gef_column')}`",
        f"- Bin edges: {top_report.get('bin_edges')}",
        f"- Bar count: {len(top_report.get('bars', []))}",
        f"- Density points: {len(top_report.get('density_curve', []))}",
        f"- City rug marks: {len(top_report.get('city_rug_marks', []))}",
        f"- Selected city GEF markers: {len(top_report.get('selected_city_gef_markers', []))}",
        "",
        "Each bar entry in JSON records `x_center`, `width`, production+EOL segment bottom/height/top, use-phase segment bottom/height/top, and total label location.",
        "",
        "## Middle: selected_city_context",
        "",
    ]
    for city in city_reports:
        visible_bins = [
            item for item in city["distance_bins"]
            if item["visible_in_panel"]
        ]
        lines.extend(
            [
                f"### Panel {city['panel_letter']}: {city['display_name']}",
                "",
                f"- Region key: `{city['region_key']}`",
                f"- OD-derived detour ratio: {city['city_detour_ratio']}",
                f"- Valid/invalid paths: {city['valid_path_count']}/{city['invalid_path_count']}",
                f"- Visible distance-bin count: {len(visible_bins)}",
                f"- Power mix components: {', '.join(item['component'] for item in city['power_mix'])}",
                f"- Bar metric: `{city['bar_metric']}`",
                "- JSON records every distance bin with air-distance x center, "
                "plotted bar height, optimal drone share, reduction metrics, "
                "absolute scatter y value, and compact detour-ratio summary "
                "statistics.",
                "",
            ]
        )
    lines.extend(
        [
            "## Bottom: scatter_detour_ratio_vs_reduction_by_gef",
            "",
            f"- Feature variable: `{bottom_report['feature_var']}`",
            f"- Result variable: `{bottom_report['result_var']}`",
            f"- Group variable: `{bottom_report['group_var']}`",
            f"- Threshold: {bottom_report['threshold']}",
            f"- Selected city scatter markers: {len(bottom_report.get('selected_city_scatter_markers', []))}",
            f"- Major city labels: {len(bottom_report.get('major_city_labels', []))}",
        ]
    )
    for panel in bottom_report["panels"]:
        lines.extend(
            [
                f"- Panel {panel['panel_letter']} ({panel['axis_name']}): {len(panel['points'])} scatter points.",
            ]
        )
    lines.extend(
        [
            "",
            "The JSON file is the authoritative element-level record for writing: it includes every scatter point coordinate, every bar x/height/bottom/top, fitted line points, confidence intervals, axes limits, tick positions, and axes locations.",
            "",
        ]
    )
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"City GEF context detour summary report saved to {md_path}")
    print(f"City GEF context detour summary report data saved to {json_path}")
    return md_path, json_path


def _render_city_gef_context_detour_summary(
    data_dir,
    out,
    for_latex,
    plot_title,
    df_enriched,
    city_df,
    exp_config,
    figure_name="city_gef_context_detour_summary",
    grayscale=False,
):
    """Render GEF components, selected city contexts, and detour-reduction scatter."""
    contexts = CITY_GEF_CONTEXT_DETOUR_CONTEXTS
    region_keys = [context["region_key"] for context in contexts]
    selected = city_df[city_df["region_key"].isin(region_keys)].copy()
    selected["_context_order"] = selected["region_key"].map(
        {key: i for i, key in enumerate(region_keys)}
    )
    selected = selected.sort_values("_context_order")
    gef_by_key = dict(
        zip(
            selected["region_key"],
            pd.to_numeric(selected["gef"], errors="coerce"),
        )
    )
    context_by_key = {
        context["region_key"]: context for context in contexts
    }

    def _context_gef_sort_key(context):
        region_key = context["region_key"]
        gef = gef_by_key.get(region_key)
        return (
            float("inf") if gef is None or pd.isna(gef) else float(gef),
            region_key,
        )

    contexts = tuple(
        sorted(
            context_by_key.values(),
            key=_context_gef_sort_key,
        )
    )
    if selected.empty:
        print(f"{figure_name} skipped — selected city rows missing")
        return

    built = _build_reduction_detour_binned_by_city(
        data_dir=data_dir,
        selected=selected,
        figure_name=figure_name,
        exp_config=exp_config,
        absolute_reduction_metric="per_order_g",
        use_binned_cache=True,
        bin_width_km=0.2,
    )
    if built is None:
        return

    binned_by_city, labels = built
    available_contexts = [
        context for context in contexts
        if context["region_key"] in binned_by_city
    ]
    missing_names = [
        context["display_name"] for context in contexts
        if context["region_key"] not in binned_by_city
    ]
    if missing_names:
        print(
            f"{figure_name}: missing binned data for "
            f"{', '.join(missing_names)}"
        )
    if not available_contexts:
        print(f"{figure_name} skipped — no selected city binned data available")
        return

    import matplotlib.pyplot as plt
    from matplotlib.colors import to_hex

    from drone_emission.visualization._utils import _add_title_suffix
    from drone_emission.visualization.city_gef_support import (
        _plot_city_gef_scatter_components_axes,
        _plot_grouped_scatter_split_panels_axes,
        _weighted_reduction_column,
    )

    summary_df = df_enriched.copy()
    if "detour_ratio" not in summary_df.columns:
        summary_df["detour_ratio"] = float("nan")
    for context in available_contexts:
        region_key = context["region_key"]
        city_detour = binned_by_city[region_key].get("city_detour_ratio")
        if city_detour is None or pd.isna(city_detour):
            print(f"{figure_name}: {region_key} OD detour unavailable")
            continue
        city_mask = summary_df["region_key"] == region_key
        existing = pd.to_numeric(
            summary_df.loc[city_mask, "detour_ratio"], errors="coerce"
        ).dropna()
        if not existing.empty and abs(float(existing.iloc[0]) - city_detour) > 1e-6:
            print(
                f"{figure_name}: {region_key} OD detour {city_detour:.6f} "
                f"differs from stats {float(existing.iloc[0]):.6f}"
            )
        summary_df.loc[city_mask, "detour_ratio"] = city_detour

    reduction_col = _weighted_reduction_column(summary_df)

    figure_size = (7.8, 8.4)
    fig = plt.figure(figsize=figure_size)
    outer = fig.add_gridspec(
        3,
        1,
        height_ratios=[1.0, 1.0, 1.0],
        hspace=CITY_GEF_SUMMARY_ROW_HSPACE,
        left=0.09,
        right=0.81,
        bottom=0.055,
        top=0.985,
    )
    legend_rail = _create_city_gef_summary_legend_rail(
        fig,
        contexts=available_contexts,
        row_hspace=CITY_GEF_SUMMARY_ROW_HSPACE,
        grayscale=grayscale,
    )

    top_axes = _plot_city_gef_scatter_components_axes(
        fig=fig,
        subplot_spec=outer[0],
        df=summary_df,
        plot_title=plot_title,
        compact=True,
        show_legend=False,
        grayscale=grayscale,
    )
    if top_axes is None:
        plt.close(fig)
        return
    ax_top, _ = top_axes
    top_pos = ax_top.get_position()
    fig.text(
        top_pos.x0 + 0.004,
        top_pos.y1 - 0.004,
        "a",
        fontsize=9,
        fontweight="bold",
        ha="left",
        va="bottom",
    )
    top_city_gef_markers = _add_city_gef_xaxis_markers(
        fig,
        top_axes,
        selected,
        available_contexts,
    )

    city_panel_records = _plot_compact_city_context_summary_panels(
        fig=fig,
        subplot_spec=outer[1],
        binned_by_city=binned_by_city,
        labels=labels,
        contexts=available_contexts,
        panel_start_letter="b",
        legend_ax=legend_rail["middle"],
        right_label_offset=0.035,
        grayscale=grayscale,
    )

    bottom_start_ord = ord("b") + len(available_contexts)
    bottom_panel_letters = (
        chr(bottom_start_ord),
        chr(bottom_start_ord + 1),
    )
    bottom_axes = _plot_grouped_scatter_split_panels_axes(
        fig=fig,
        subplot_spec=outer[2],
        df=summary_df,
        feature_var="detour_ratio",
        result_var=reduction_col,
        group_var="grid_emission_factor",
        threshold=0.3,
        plot_title=False,
        panel_letters=bottom_panel_letters,
        compact=True,
        horizontal_colorbar=False,
        external_colorbar_ax=legend_rail["colorbar"],
        colorbar_boundaries=CITY_GEF_COLOR_BIN_EDGES,
        colorbar_label="kg CO$_2$e/kWh",
        scatter_size=CITY_GEF_BOTTOM_SCATTER_SIZE,
        scatter_alpha=(
            CITY_GEF_GRAYSCALE_BOTTOM_SCATTER_ALPHA
            if grayscale
            else CITY_GEF_BOTTOM_SCATTER_ALPHA
        ),
        grayscale=grayscale,
    )
    if bottom_axes is None:
        plt.close(fig)
        return
    bottom_axes[0].set_ylabel("Optimal lifecycle reduction (%)")
    bottom_city_markers = _add_city_detour_reduction_markers(
        fig,
        bottom_axes,
        summary_df,
        available_contexts,
        reduction_col,
        panel_letters=bottom_panel_letters,
    )
    bottom_major_city_labels = []
    connector_lines = []

    if plot_title:
        fig.suptitle(
            "City Grid Intensity, Context, and Detour-reduction Relationships",
            fontsize=9,
        )

    save_path = os.path.join(out, f"{figure_name}.png")
    if for_latex and not save_path.endswith(".pdf"):
        save_path = save_path.rsplit(".", 1)[0] + ".pdf"
    if plot_title:
        save_path = _add_title_suffix(save_path)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.canvas.draw()
    fig.savefig(save_path, dpi=300, bbox_inches=fig.bbox_inches, pad_inches=0.0)
    top_report = _city_gef_components_report(
        summary_df,
        top_axes,
        grayscale=grayscale,
    )
    top_report["selected_city_gef_markers"] = top_city_gef_markers
    city_reports = _city_context_summary_report(
        binned_by_city,
        labels,
        city_panel_records,
        bar_metric="oracle_drone_share_pct",
        grayscale=grayscale,
    )
    bottom_report = _detour_reduction_scatter_report(
        summary_df,
        reduction_col,
        bottom_axes,
        panel_letters=bottom_panel_letters,
        compact=True,
        grayscale=grayscale,
    )
    bottom_report["selected_city_scatter_markers"] = bottom_city_markers
    bottom_report["major_city_labels"] = bottom_major_city_labels
    gef_cmap = bottom_axes[0].collections[0].cmap
    legend_rail_report = {
        "orientation": "vertical",
        "sections": {
            "top_vehicle_component": _report_axis(legend_rail["top"]),
            "middle_power_mix": _report_axis(legend_rail["middle"]),
            "bottom_gef": _report_axis(legend_rail["bottom"]),
            "gef_colorbar": _report_axis(legend_rail["colorbar"]),
        },
        "gef_bin_edges": _report_array(CITY_GEF_COLOR_BIN_EDGES),
        "gef_bin_colors": [to_hex(gef_cmap(index)) for index in range(gef_cmap.N)],
        "city_markers": legend_rail["city_entries"],
        "render_style": "grayscale" if grayscale else "color",
        "production_eol_hatch": "////" if grayscale else None,
        "use_phase_hatch": None,
    }
    _write_city_gef_context_detour_report(
        save_path=save_path,
        figure_name=figure_name,
        figure_size=figure_size,
        top_report=top_report,
        city_reports=city_reports,
        bottom_report=bottom_report,
        connector_lines=connector_lines,
        legend_rail=legend_rail_report,
        render_style="grayscale" if grayscale else "color",
    )
    plt.close(fig)
    print(f"City GEF context detour summary saved to {save_path}")
