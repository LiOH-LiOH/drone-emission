"""
排放分析可视化模块
"""
from .palette import opacity as paper_alpha

from .palette import color as paper_color
from .prepared_bundle import prepared_data

import json
import os
import time
import zlib
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.colors import BoundaryNorm, TwoSlopeNorm

from ._utils import _add_title_suffix
from .macro_regions import (
    CHINA_MACRO_REGION_SPECS,
    PROVINCE_TO_MACRO_REGION,
    normalize_province_name,
)


def _draw_selected_trip_emission_process(
    ax,
    sample: pd.Series,
    plot_title: bool = False,
):
    """Draw cumulative use-phase emissions for one preselected OD row."""

    air_km = float(sample["distance_km"])
    road_km = float(sample["road_distance_m"]) / 1000.0
    drone_use_cycle_total = float(sample["drone_emission_kg"]) * 1000.0
    ebike_use_cycle_total = float(sample["ebike_emission_kg"]) * 1000.0

    @prepared_data
    def _drone_delivery_leg_profile(distance_km, use_cycle_total_g):
        """Scale modeled drone use-phase shares to the cached OD total."""
        try:
            from drone_emission.emission.cal_single_delivery import DroneEnergyModel

            energy = DroneEnergyModel().calculate_energy_consumption(
                distance_km,
                payload_kg=1.0,
            )
            parts = {
                "takeoff": energy["energy_takeoff_kwh"],
                "level_out": energy["energy_level_out_kwh"],
                "landing_out": energy["energy_landing_out_kwh"],
                "hover": energy["energy_hover_kwh"],
                "takeoff_return": energy["energy_takeoff_return_kwh"],
                "level_return": energy["energy_level_return_kwh"],
                "landing_return": energy["energy_landing_return_kwh"],
            }
            cycle_energy = sum(parts.values())
            if cycle_energy <= 0:
                raise ValueError("non-positive drone cycle energy")

            scale = use_cycle_total_g / cycle_energy
            takeoff_g = parts["takeoff"] * scale
            cruise_g = parts["level_out"] * scale
            landing_g = (parts["landing_out"] + parts["hover"]) * scale
            y_takeoff = takeoff_g
            y_cruise_end = takeoff_g + cruise_g
            y_leg_end = y_cruise_end + landing_g
            return (
                np.array([0.0, 0.0, distance_km, distance_km]),
                np.array([0.0, y_takeoff, y_cruise_end, y_leg_end]),
                {"takeoff_g": takeoff_g, "landing_g": landing_g},
            )
        except Exception:
            outbound_g = use_cycle_total_g / 2.0
            return (
                np.array([0.0, distance_km]),
                np.array([0.0, outbound_g]),
                {"takeoff_g": 0.0, "landing_g": 0.0},
            )

    @prepared_data
    def _ebike_delivery_leg_profile(distance_km, use_cycle_total_g):
        """Scale modeled e-bike outbound use-phase share to the cached OD total."""
        try:
            from drone_emission.emission.cal_single_delivery import EbikeEnergyModel

            energy = EbikeEnergyModel().calculate_energy_consumption(
                distance_km,
                payload_kg=1.0,
                rider_weight=70,
            )
            energy_go = energy["power_kw_go"] * energy["time_hours"]
            total_energy = energy["energy_kwh"]
            if total_energy <= 0:
                raise ValueError("non-positive e-bike cycle energy")
            outbound_g = use_cycle_total_g * energy_go / total_energy
        except Exception:
            outbound_g = use_cycle_total_g / 2.0
        return np.array([0.0, distance_km]), np.array([0.0, outbound_g])

    drone_x, drone_y, drone_parts = _drone_delivery_leg_profile(
        air_km,
        drone_use_cycle_total,
    )
    ebike_x, ebike_y = _ebike_delivery_leg_profile(
        road_km,
        ebike_use_cycle_total,
    )
    max_x = max(air_km, road_km)
    max_y = max(drone_y.max(), ebike_y.max(), 1e-6)

    drone_color = paper_color("lifecycle_panels.2b78b8")
    ebike_color = paper_color("lifecycle_panels.d95f35")
    gap_color = paper_color("lifecycle_panels.8a8f98")

    ax.plot(
        drone_x,
        drone_y,
        color=drone_color,
        linewidth=2.6,
        marker="o",
        markevery=[len(drone_x) - 1],
        markersize=6,
        label="Drone (air path)",
    )
    ax.plot(
        ebike_x,
        ebike_y,
        color=ebike_color,
        linewidth=2.6,
        marker="s",
        markevery=[len(ebike_x) - 1],
        markersize=5.5,
        label="E-bike (road route)",
    )

    ax.axvline(
        air_km,
        color=drone_color,
        linestyle="--",
        linewidth=1.0,
        alpha=paper_alpha("lifecycle_panels.alpha_0_55"),
    )
    ax.axvline(
        road_km,
        color=ebike_color,
        linestyle="--",
        linewidth=1.0,
        alpha=paper_alpha("lifecycle_panels.alpha_0_55"),
    )

    if road_km > air_km:
        ax.annotate(
            "",
            xy=(road_km, max_y * 0.10),
            xytext=(air_km, max_y * 0.10),
            arrowprops=dict(arrowstyle="<->", color=gap_color, linewidth=1.0),
        )
        ax.text(
            (air_km + road_km) / 2,
            max_y * 0.13,
            f"route detour +{road_km - air_km:.2f} km",
            ha="center",
            va="bottom",
            fontsize=9,
            color=gap_color,
        )

    if drone_parts["takeoff_g"] > 0:
        ax.text(
            air_km * 0.015,
            drone_y[1] / 2,
            "takeoff",
            ha="left",
            va="center",
            fontsize=8.5,
            color=drone_color,
        )
        ax.text(
            air_km * 0.985,
            drone_y[-2] + drone_parts["landing_g"] / 2,
            "landing + hover",
            ha="right",
            va="center",
            fontsize=8.5,
            color=drone_color,
        )

    label_offset = max_y * 0.045
    ax.text(
        air_km,
        drone_y[-1] + label_offset,
        f"{drone_y[-1]:.1f} g\n{air_km:.2f} km",
        ha="center",
        va="bottom",
        fontsize=9,
        color=drone_color,
    )
    ax.text(
        road_km,
        ebike_y[-1] + label_offset,
        f"{ebike_y[-1]:.1f} g\n{road_km:.2f} km",
        ha="center",
        va="bottom",
        fontsize=9,
        color=ebike_color,
    )

    ax.set_xlim(0, max_x * 1.08)
    ax.set_ylim(0, max_y * 1.22)
    ax.set_xlabel("Actual one-way distance traveled (km)", fontsize=12)
    ax.set_ylabel(
        "Accumulated use-phase emission (g CO$_2$e / delivery leg)",
        fontsize=12,
    )
    ax.grid(True, alpha=paper_alpha("lifecycle_panels.alpha_0_25"), linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, loc="upper left", fontsize=10)
    ax.text(
        0.98,
        0.04,
        "Beijing road-distance P75",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        color=paper_color("lifecycle_panels.555555"),
    )

    if plot_title:
        ax.set_title(
            "Beijing P75 Delivery-Leg Emission Process",
            fontsize=14,
        )
    return {
        "sample": sample,
        "sample_index": sample.name,
        "air_km": air_km,
        "road_km": road_km,
        "drone_x": drone_x,
        "drone_y": drone_y,
        "ebike_x": ebike_x,
        "ebike_y": ebike_y,
    }
