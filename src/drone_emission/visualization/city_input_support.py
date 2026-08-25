"""Visualization functions for cross-region correlation and emission analysis."""

import json
import os
import pickle

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
})

from drone_emission.visualization._utils import (
    _add_title_suffix,
    saved_tight_axis_report,
    write_figure_report,
)
from drone_emission.visualization.macro_regions import (
    CHINA_MACRO_REGION_SPECS,
    macro_region_for_province,
)

# Import from constants module (no circular dependency)
from drone_emission.analysis.constants import (
    DRONE_SPATIAL_SCATTER_LABELED_CITIES,
    PROVINCE_NAME_EN,
    VARIABLE_LABELS,
)
from drone_emission.analysis.city_attribution import R2_ABSOLUTE_OUTCOME, R2_OUTCOME


def _select_median_trip_emission_sample(od_df: pd.DataFrame):
    """Return an OD row close to both median air and road distances."""
    required = [
        "distance_km",
        "road_distance_m",
        "drone_emission_kg",
        "ebike_emission_kg",
    ]
    valid = od_df.dropna(subset=required).copy()
    for col in required:
        valid[col] = pd.to_numeric(valid[col], errors="coerce")
    valid = valid.dropna(subset=required)
    valid = valid[
        (valid["distance_km"] > 0)
        & (valid["road_distance_m"] > 0)
        & (valid["drone_emission_kg"] >= 0)
        & (valid["ebike_emission_kg"] >= 0)
    ]
    if valid.empty:
        return None

    if "annual_sales" in valid.columns:
        valid["annual_sales"] = pd.to_numeric(valid["annual_sales"], errors="coerce")
        weighted = valid.dropna(subset=["annual_sales"])
        weighted = weighted[weighted["annual_sales"] > 0]
    else:
        weighted = pd.DataFrame()

    def _median(value_col):
        if weighted.empty:
            return valid[value_col].median()
        ordered = weighted.sort_values(value_col)
        cumulative = ordered["annual_sales"].cumsum()
        index = cumulative.searchsorted(cumulative.iloc[-1] / 2.0)
        return ordered.iloc[min(index, len(ordered) - 1)][value_col]

    target_air = _median("distance_km")
    target_road = _median("road_distance_m")
    air_scale = valid["distance_km"].quantile(0.75) - valid["distance_km"].quantile(0.25)
    road_scale = valid["road_distance_m"].quantile(0.75) - valid["road_distance_m"].quantile(0.25)
    if pd.isna(air_scale) or air_scale <= 0:
        air_scale = max(float(valid["distance_km"].median()), 1.0)
    if pd.isna(road_scale) or road_scale <= 0:
        road_scale = max(float(valid["road_distance_m"].median()), 1.0)
    valid["_median_trip_score"] = (
        (valid["distance_km"] - target_air).abs() / air_scale
        + (valid["road_distance_m"] - target_road).abs() / road_scale
    )
    return valid.sort_values("_median_trip_score").iloc[0]


def _json_safe_scalar(value):
    """Return a JSON-safe scalar with missing numeric values represented as None."""
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


def _draw_trip_phase_barplot(
    ax,
    df,
    ylabel,
    body_label,
    battery_label,
    body_color,
    battery_color,
    distance_label,
    body_colors_by_vehicle=None,
    battery_colors_by_vehicle=None,
    legend_body_color=None,
    legend_battery_color=None,
    body_hatch=None,
    battery_hatch=None,
):
    """Draw stacked body/battery phase emissions for one selected OD trip."""
    x = np.arange(len(df))
    body = df["body_g"].to_numpy(dtype=float)
    battery = df["battery_g"].to_numpy(dtype=float)
    totals = body + battery
    vehicles = df["vehicle"].tolist()
    body_colors = [
        body_colors_by_vehicle.get(vehicle, body_color)
        if body_colors_by_vehicle
        else body_color
        for vehicle in vehicles
    ]
    battery_colors = [
        battery_colors_by_vehicle.get(vehicle, battery_color)
        if battery_colors_by_vehicle
        else battery_color
        for vehicle in vehicles
    ]
    edge_color = "#2f3437"

    ax.bar(
        x,
        body,
        width=0.56,
        color=body_colors,
        edgecolor=edge_color,
        linewidth=0.8,
        hatch=body_hatch,
        label=body_label,
    )
    ax.bar(
        x,
        battery,
        width=0.56,
        bottom=body,
        color=battery_colors,
        edgecolor=edge_color,
        linewidth=0.8,
        hatch=battery_hatch,
        label=battery_label,
    )

    max_total = max(float(np.nanmax(totals)), 1e-6)
    for i, total in enumerate(totals):
        ax.text(
            x[i],
            total + max_total * 0.035,
            f"{total:.1f}",
            ha="center",
            va="bottom",
            fontsize=9,
            color="#222222",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(vehicles)
    ax.set_xlabel(distance_label)
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, max_total * 1.25)
    ax.grid(axis="y", alpha=0.25, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    legend_handles = [
        Patch(
            facecolor=legend_body_color or body_color,
            edgecolor=edge_color,
            linewidth=0.8,
            hatch=body_hatch,
            label=body_label,
        ),
        Patch(
            facecolor=legend_battery_color or battery_color,
            edgecolor=edge_color,
            linewidth=0.8,
            hatch=battery_hatch,
            label=battery_label,
        ),
    ]
    ax.legend(
        handles=legend_handles,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.15),
        ncol=2,
        fontsize=9,
    )


def _model_phase_component_shares(distance_km, vehicle, phase):
    """Return body/battery shares for a vehicle phase at the selected trip distance."""
    from drone_emission.emission.cal_single_delivery import LifeCycleEmissions

    lca = LifeCycleEmissions()
    if vehicle == "E-bike":
        p = lca.ebike_p
        prod = lca.calculate_ebike_production_emissions()
        speed = p.velocity
    elif vehicle == "Drone":
        p = lca.drone_p
        prod = lca.calculate_drone_production_emissions()
        speed = p.v_level
    else:
        raise ValueError(f"Unknown vehicle: {vehicle}")

    delivery_time_min = (2 * distance_km / speed) * 60 + p.loading_time_min
    deliveries_per_day = p.working_hours_per_day * 60 / delivery_time_min
    lifetime_deliveries = deliveries_per_day * p.working_days_per_year * p.lifespan

    if phase == "production":
        body_kg = prod["body_production_kg"] / p.lifespan
        battery_kg = prod["battery_production_kg"] / p.battery_lifespan
    elif phase == "eol":
        body_kg = prod["body_disposal_kg"] / p.lifespan
        battery_kg = prod["battery_disposal_kg"] / p.battery_lifespan
    else:
        raise ValueError(f"Unknown phase: {phase}")

    body_g = body_kg * p.lifespan / lifetime_deliveries * 1000
    battery_g = battery_kg * p.lifespan / lifetime_deliveries * 1000
    total_g = body_g + battery_g
    if total_g <= 0:
        return 0.5, 0.5
    return body_g / total_g, battery_g / total_g


def _selected_trip_phase_parts(sample, phase):
    """Return body/battery production or EOL parts for the selected OD row."""
    if sample is None:
        return None

    if phase == "production":
        drone_col = "drone_lifecycle_production_g"
        ebike_col = "ebike_lifecycle_production_g"
    elif phase == "eol":
        drone_col = "drone_lifecycle_eol_g"
        ebike_col = "ebike_lifecycle_eol_g"
    else:
        raise ValueError(f"Unknown phase: {phase}")

    required = {drone_col, ebike_col, "distance_km", "road_distance_m"}
    if not required.issubset(sample.index):
        return None

    ebike_total = pd.to_numeric(pd.Series([sample[ebike_col]]), errors="coerce").iloc[0]
    drone_total = pd.to_numeric(pd.Series([sample[drone_col]]), errors="coerce").iloc[0]
    if not np.isfinite(ebike_total) or not np.isfinite(drone_total):
        return None

    air_km = float(sample["distance_km"])
    road_km = float(sample["road_distance_m"]) / 1000.0
    ebike_distance_km = road_km
    if "ebike_kappa" in sample.index and pd.notna(sample["ebike_kappa"]):
        ebike_distance_km *= float(sample["ebike_kappa"])

    ebike_body_share, ebike_battery_share = _model_phase_component_shares(
        ebike_distance_km,
        "E-bike",
        phase,
    )
    drone_body_share, drone_battery_share = _model_phase_component_shares(
        air_km,
        "Drone",
        phase,
    )

    return pd.DataFrame(
        [
            {
                "vehicle": "E-bike",
                "body_g": float(ebike_total) * ebike_body_share,
                "battery_g": float(ebike_total) * ebike_battery_share,
                "cached_total_g": float(ebike_total),
                "distance_km": ebike_distance_km,
            },
            {
                "vehicle": "Drone",
                "body_g": float(drone_total) * drone_body_share,
                "battery_g": float(drone_total) * drone_battery_share,
                "cached_total_g": float(drone_total),
                "distance_km": air_km,
            },
        ]
    )


def _normalize_median_trip_subfigure_text(ax):
    """Apply consistent publication-scale typography to a median-trip panel."""
    ax.xaxis.label.set_fontsize(8.5)
    ax.yaxis.label.set_fontsize(8.5)
    ax.tick_params(axis="both", labelsize=8)
    if ax.title:
        ax.title.set_fontsize(9)
    for text_artist in ax.texts:
        text_artist.set_fontsize(8)
    legend = ax.get_legend()
    if legend is not None:
        for legend_text in legend.get_texts():
            legend_text.set_fontsize(8)


def _save_median_trip_lifecycle_subfigures(
    production_df,
    eol_df,
    median_trip_od_df,
    median_trip_city_label,
    distance_label,
    phase_body_colors,
    phase_battery_colors,
    save_dir,
    for_latex=False,
    plot_title=False,
    only=None,
):
    """Save production, use-phase, and EOL median-trip subfigures."""
    from drone_emission.visualization.lifecycle_panels import _draw_median_trip_emission_process

    mm_per_inch = 25.4
    extension = ".pdf" if for_latex else ".png"
    os.makedirs(save_dir, exist_ok=True)

    phase_body_color = "#5f6368"
    phase_battery_color = "#c7c9cc"
    panel_specs = (
        (
            "median_trip_production_emissions",
            production_df,
            "Production emissions\n(g CO$_2$e/order)",
        ),
        (
            "median_trip_eol_emissions",
            eol_df,
            "End-of-life emissions\n(g CO$_2$e/order)",
        ),
    )
    for figure_name, phase_df, ylabel in panel_specs:
        if only is not None and figure_name not in only:
            continue
        fig, ax = plt.subplots(figsize=(43.5 / mm_per_inch, 50.0 / mm_per_inch))
        fig.subplots_adjust(left=0.34, right=0.97, bottom=0.23, top=0.78)
        if phase_df is None:
            phase_name = "production" if "production" in figure_name else "EOL"
            ax.text(
                0.5,
                0.5,
                f"{median_trip_city_label} median trip {phase_name} data\nnot available",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=8,
                color="#777777",
            )
            ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
            for spine in ax.spines.values():
                spine.set_visible(False)
        else:
            _draw_trip_phase_barplot(
                ax,
                phase_df,
                ylabel=ylabel,
                body_label="Body",
                battery_label="Battery",
                body_color=phase_body_color,
                battery_color=phase_battery_color,
                distance_label="",
                body_colors_by_vehicle=phase_body_colors,
                battery_colors_by_vehicle=phase_battery_colors,
                legend_body_color=phase_body_color,
                legend_battery_color=phase_battery_color,
                battery_hatch="///",
            )
            legend_handles = [
                Patch(
                    facecolor=phase_body_color,
                    edgecolor="#2f3437",
                    linewidth=0.8,
                    label="Body",
                ),
                Patch(
                    facecolor=phase_battery_color,
                    edgecolor="#2f3437",
                    linewidth=0.8,
                    hatch="///",
                    label="Battery",
                ),
            ]
            ax.legend(
                handles=legend_handles,
                frameon=False,
                loc="lower center",
                bbox_to_anchor=(0.5, 1.02),
                borderaxespad=0,
                ncol=2,
                fontsize=7.5,
                handlelength=1.2,
                handletextpad=0.4,
                columnspacing=0.8,
            )
            if plot_title:
                title = (
                    "Production emissions"
                    if "production" in figure_name
                    else "End-of-life emissions"
                )
                ax.set_title(title, fontsize=9, y=1.30)
        _normalize_median_trip_subfigure_text(ax)
        output_path = os.path.join(save_dir, figure_name + extension)
        if plot_title:
            output_path = _add_title_suffix(output_path)
        fig.savefig(output_path, dpi=300, facecolor="white")
        phase_name = "production" if "production" in figure_name else "end_of_life"
        if phase_df is None:
            phase_facts = {
                "status": "unavailable",
                "reason": "Required selected-trip lifecycle columns were unavailable",
                "vehicle_components": [],
            }
        else:
            vehicle_components = []
            for _, row in phase_df.iterrows():
                body_g = float(row["body_g"])
                battery_g = float(row["battery_g"])
                cached_total_g = float(row["cached_total_g"])
                vehicle_components.append(
                    {
                        "vehicle": row["vehicle"],
                        "distance_km": float(row["distance_km"]),
                        "body_g_co2e_per_order": body_g,
                        "battery_g_co2e_per_order": battery_g,
                        "component_total_g_co2e_per_order": body_g + battery_g,
                        "cached_total_g_co2e_per_order": cached_total_g,
                        "component_sum_difference_g": (
                            body_g + battery_g - cached_total_g
                        ),
                    }
                )
            phase_facts = {
                "status": "ok",
                "vehicle_components": vehicle_components,
            }
        write_figure_report(
            output_path.rsplit(".", 1)[0] + "_report.json",
            {
                "figure_name": figure_name,
                "figure_path": output_path,
                "source_city": median_trip_city_label,
                "sample_selection": "annual-sales-weighted median trip when annual_sales is available",
                "lifecycle_phase": phase_name,
                "unit": "g CO2e per order",
                **phase_facts,
            },
        )
        plt.close(fig)
        print(f"Median-trip lifecycle subfigure saved to {output_path}")

    if only is not None and "median_trip_use_phase_emissions" not in only:
        return

    fig, ax = plt.subplots(figsize=(87.0 / mm_per_inch, 50.0 / mm_per_inch))
    fig.subplots_adjust(left=0.17, right=0.98, bottom=0.23, top=0.80)
    if median_trip_od_df is None:
        ax.text(
            0.5,
            0.5,
            f"{median_trip_city_label} median trip emission data\nnot available",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=8,
            color="#777777",
        )
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        for spine in ax.spines.values():
            spine.set_visible(False)
    else:
        use_phase_result = _draw_median_trip_emission_process(
            ax,
            median_trip_od_df,
            plot_title=plot_title,
        )
        ax.set_xlabel("Distance (km)")
        ax.set_ylabel("Use-phase emissions\n(g CO$_2$e/leg)")
        legend_handles, legend_labels = ax.get_legend_handles_labels()
        ax.legend(
            legend_handles,
            legend_labels,
            frameon=False,
            loc="lower left",
            bbox_to_anchor=(0.0, 1.02),
            borderaxespad=0,
            ncol=2,
            fontsize=8,
            handlelength=1.6,
            handletextpad=0.5,
            columnspacing=1.0,
        )
        for text_artist in ax.texts:
            if text_artist.get_text().endswith("median trip"):
                text_artist.remove()
    _normalize_median_trip_subfigure_text(ax)
    output_path = os.path.join(save_dir, "median_trip_use_phase_emissions" + extension)
    if plot_title:
        output_path = _add_title_suffix(output_path)
    fig.savefig(output_path, dpi=300, facecolor="white")
    if median_trip_od_df is None or use_phase_result is None:
        use_phase_facts = {
            "status": "unavailable",
            "reason": "Median-trip use-phase data were unavailable",
        }
    else:
        selected_sample = use_phase_result["sample"]
        use_phase_facts = {
            "status": "ok",
            "eligible_od_count": int(
                len(
                    median_trip_od_df.dropna(
                        subset=[
                            "distance_km",
                            "road_distance_m",
                            "drone_emission_kg",
                            "ebike_emission_kg",
                        ]
                    )
                )
            ),
            "selected_trip": {
                "source_row_index": _json_safe_scalar(
                    use_phase_result["sample_index"]
                ),
                "air_distance_km": float(use_phase_result["air_km"]),
                "road_distance_km": float(use_phase_result["road_km"]),
                "detour_ratio": float(
                    use_phase_result["road_km"] / use_phase_result["air_km"]
                ),
                "drone_delivery_leg_end_g_co2e": float(
                    use_phase_result["drone_y"][-1]
                ),
                "ebike_delivery_leg_end_g_co2e": float(
                    use_phase_result["ebike_y"][-1]
                ),
                "annual_sales": _json_safe_scalar(
                    selected_sample.get("annual_sales")
                ),
            },
        }
    write_figure_report(
        output_path.rsplit(".", 1)[0] + "_report.json",
        {
            "figure_name": "median_trip_use_phase_emissions",
            "figure_path": output_path,
            "source_city": median_trip_city_label,
            "sample_selection": "annual-sales-weighted median trip when annual_sales is available",
            "lifecycle_phase": "use",
            "unit": "g CO2e per delivery leg",
            **use_phase_facts,
        },
    )
    plt.close(fig)
    print(f"Median-trip use-phase subfigure saved to {output_path}")
