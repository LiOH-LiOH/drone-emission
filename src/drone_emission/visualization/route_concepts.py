"""Concept illustration figures for the paper."""

from __future__ import annotations

import math
import os
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patheffects as pe
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle

from drone_emission.visualization._utils import _add_title_suffix, write_figure_report

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
})

Point = tuple[float, float]

# ---------------------------------------------------------------------------
# Shared scene constants
# ---------------------------------------------------------------------------

_COLORS = {
    "background": "#f8f7f3",
    "ground": "#e9ece8",
    "ground_edge": "#c5cbc6",
    "grid": "#d7dcd8",
    "road": "#b9c0bb",
    "road_dark": "#8f9792",
    "bike": "#d95f35",
    "bike_shadow": "#8f3f2b",
    "drone": "#1f77b4",
    "drone_shadow": "#2f4f62",
    "origin": "#2f6f4e",
    "destination": "#3f4b64",
    "building_top": "#d9dde1",
    "building_left": "#b9c0c8",
    "building_right": "#c9cfd5",
    "text": "#25302d",
    "muted": "#65706b",
    "white": "#ffffff",
}

_GRID_X = [1.0, 3.5, 6.0, 8.5]   # vertical road x-positions
_GRID_Y = [1.0, 3.0, 5.0]         # horizontal road y-positions
_CRUISE_ALT = 1.3                  # drone cruise altitude

# ---------------------------------------------------------------------------
# Projection helpers
# ---------------------------------------------------------------------------

def _project(x: float, y: float, z: float = 0.0) -> Point:
    """Oblique projection: world (x, y, z) → screen (sx, sy)."""
    return x + 0.42 * y, 0.58 * y + z


def _projected(points: Iterable[Sequence[float]], z: float = 0.0) -> np.ndarray:
    return np.array([_project(p[0], p[1], p[2] if len(p) > 2 else z) for p in points])


def _path_length(points: Sequence[Sequence[float]]) -> float:
    total = 0.0
    for a, b in zip(points[:-1], points[1:]):
        total += math.dist((a[0], a[1]), (b[0], b[1]))
    return total

# ---------------------------------------------------------------------------
# Scene drawing helpers
# ---------------------------------------------------------------------------

def _add_polyline(
    ax: plt.Axes,
    points: Sequence[Sequence[float]],
    *,
    color: str,
    linewidth: float,
    zorder: int,
    alpha: float = 1.0,
    z: float = 0.0,
    linestyle: str = "-",
    capstyle: str = "round",
    clip_patch=None,
) -> None:
    xy = _projected(points, z=z)
    line, = ax.plot(xy[:, 0], xy[:, 1], color=color, linewidth=linewidth, alpha=alpha,
                    linestyle=linestyle, solid_capstyle=capstyle, zorder=zorder)
    if clip_patch is not None:
        line.set_clip_path(clip_patch)


def _add_ground(
    ax: plt.Axes,
    bounds: tuple[float, float, float, float] | None = None,
) -> "Polygon":
    if bounds is None:
        x_min, x_max, y_min, y_max = 0.0, 9.7, 0.0, 6.3
        grid_xs = np.arange(1.0, 9.6, 1.0)
        grid_ys = np.arange(0.8, 6.3, 0.9)
        grid_pad = 0.2
    else:
        x_min, x_max, y_min, y_max = bounds
        grid_xs = np.arange(x_min + 0.55, x_max, 0.75)
        grid_ys = np.arange(y_min + 0.45, y_max, 0.75)
        grid_pad = 0.12

    plane = _projected([
        (x_min, y_min),
        (x_max, y_min),
        (x_max, y_max),
        (x_min, y_max),
    ])
    patch = Polygon(plane, closed=True, facecolor=_COLORS["ground"],
                    edgecolor=_COLORS["ground_edge"], linewidth=1.2, zorder=1)
    ax.add_patch(patch)
    for x in grid_xs:
        _add_polyline(
            ax,
            [(x, y_min + grid_pad), (x, y_max - grid_pad)],
            color=_COLORS["grid"],
            linewidth=0.7,
            zorder=2,
        )
    for y in grid_ys:
        _add_polyline(
            ax,
            [(x_min + grid_pad, y), (x_max - grid_pad, y)],
            color=_COLORS["grid"],
            linewidth=0.7,
            zorder=2,
        )
    return patch


def _add_road_network(
    ax: plt.Axes,
    clip_patch=None,
    *,
    grid_x: Sequence[float] = _GRID_X,
    grid_y: Sequence[float] = _GRID_Y,
    bounds: tuple[float, float, float, float] = (0.0, 9.7, 0.0, 6.3),
    linewidth: float = 12.0,
) -> None:
    x_min, x_max, y_min, y_max = bounds
    for x in grid_x:
        _add_polyline(
            ax,
            [(x, y_min), (x, y_max)],
            color=_COLORS["road"],
            linewidth=linewidth,
            zorder=5,
            clip_patch=clip_patch,
        )
        _add_polyline(
            ax,
            [(x, y_min), (x, y_max)],
            color=_COLORS["road_dark"],
            linewidth=1.0,
            zorder=6,
            alpha=0.32,
            clip_patch=clip_patch,
        )
    for y in grid_y:
        _add_polyline(
            ax,
            [(x_min, y), (x_max, y)],
            color=_COLORS["road"],
            linewidth=linewidth,
            zorder=5,
            clip_patch=clip_patch,
        )
        _add_polyline(
            ax,
            [(x_min, y), (x_max, y)],
            color=_COLORS["road_dark"],
            linewidth=1.0,
            zorder=6,
            alpha=0.32,
            clip_patch=clip_patch,
        )


def _add_building(ax: plt.Axes, x: float, y: float, w: float, d: float, h: float) -> None:
    base = [(x, y, 0), (x + w, y, 0), (x + w, y + d, 0), (x, y + d, 0)]
    top  = [(x, y, h), (x + w, y, h), (x + w, y + d, h), (x, y + d, h)]
    p_base, p_top = _projected(base), _projected(top)
    faces = [
        ([p_base[0], p_base[1], p_top[1], p_top[0]], _COLORS["building_right"]),
        ([p_base[1], p_base[2], p_top[2], p_top[1]], _COLORS["building_left"]),
        ([p_top[0],  p_top[1],  p_top[2],  p_top[3]],  _COLORS["building_top"]),
    ]
    depth_zorder = 15 - (y + d / 2)
    for coords, color in faces:
        ax.add_patch(Polygon(coords, closed=True, facecolor=color,
                             edgecolor="#aab2ba", linewidth=0.45, zorder=depth_zorder))


def _add_buildings(
    ax: plt.Axes,
    *,
    grid_x: Sequence[float] = _GRID_X,
    grid_y: Sequence[float] = _GRID_Y,
    building_width: float = 0.65,
    building_depth: float = 0.50,
    margin: float = 0.38,
) -> None:
    blocks = [
        (grid_x[0], grid_y[0], grid_x[1], grid_y[1]),
        (grid_x[1], grid_y[0], grid_x[2], grid_y[1]),
        (grid_x[2], grid_y[0], grid_x[3], grid_y[1]),
        (grid_x[0], grid_y[1], grid_x[1], grid_y[2]),
        (grid_x[1], grid_y[1], grid_x[2], grid_y[2]),
        (grid_x[2], grid_y[1], grid_x[3], grid_y[2]),
    ]
    block_heights = [
        [0.70, 0.45, 0.55, 0.80],
        [0.50, 0.75, 0.60, 0.40],
        [0.65, 0.50, 0.80, 0.55],
        [0.55, 0.80, 0.45, 0.70],
        [0.75, 0.50, 0.65, 0.55],
        [0.45, 0.65, 0.75, 0.50],
    ]
    bw, bd = building_width, building_depth
    for (x0, y0, x1, y1), heights in zip(blocks, block_heights):
        col_gap = x1 - x0 - 2 * margin - 2 * bw
        row_gap = y1 - y0 - 2 * margin - 2 * bd
        col_xs = [x0 + margin, x0 + margin + bw + col_gap]
        row_ys = [y0 + margin, y0 + margin + bd + row_gap]
        i = 0
        for y in row_ys:
            for x in col_xs:
                _add_building(ax, x, y, bw, bd, heights[i])
                i += 1


def _add_marker(
    ax: plt.Axes,
    xy_world: Point,
    *,
    color: str,
    label: str,
    label_offset: Point,
    zorder: int = 40,
    shape: str = "circle",
    label_fontsize: float = 10.5,
    marker_scale: float = 1.0,
) -> Point:
    xy = _project(*xy_world)
    path_effects = [pe.SimplePatchShadow(offset=(1.0, -1.0), alpha=0.25), pe.Normal()]
    if shape == "square":
        side = 0.26 * marker_scale
        marker = Rectangle(
            (xy[0] - side / 2, xy[1] - side / 2),
            side,
            side,
            facecolor=color,
            edgecolor=_COLORS["white"],
            linewidth=1.4,
            zorder=zorder,
            path_effects=path_effects,
        )
    else:
        marker = Circle(
            xy,
            radius=0.15 * marker_scale,
            facecolor=color,
            edgecolor=_COLORS["white"],
            linewidth=1.6,
            zorder=zorder,
            path_effects=path_effects,
        )
    ax.add_patch(marker)
    ax.text(xy[0] + label_offset[0], xy[1] + label_offset[1], label,
            color=_COLORS["text"], fontsize=label_fontsize, fontweight="bold",
            ha="center", va="center", zorder=zorder + 2)
    return xy


def _add_arrow_head(
    ax: plt.Axes,
    points: Sequence[Sequence[float]],
    color: str,
    zorder: int,
    z: float = 0.0,
    alpha: float = 1.0,
    mutation_scale: float = 18.0,
) -> None:
    start = _project(points[-2][0], points[-2][1], z)
    end   = _project(points[-1][0], points[-1][1], z)
    ax.add_patch(FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=mutation_scale,
        linewidth=0,
        color=color,
        zorder=zorder,
        alpha=alpha,
    ))


def _add_drone_icon(ax: plt.Axes, center_world: tuple[float, float, float]) -> None:
    cx, cy = _project(center_world[0], center_world[1], center_world[2])
    ax.add_patch(Circle((cx, cy), radius=0.10, facecolor=_COLORS["drone"],
                        edgecolor=_COLORS["white"], linewidth=0.9, zorder=58))
    arm_len = 0.34
    for dx, dy in [(-arm_len, 0), (arm_len, 0), (0, -arm_len * 0.55), (0, arm_len * 0.55)]:
        ax.plot([cx, cx + dx], [cy, cy + dy], color=_COLORS["drone"], linewidth=1.2, zorder=57)
        ax.add_patch(Circle((cx + dx, cy + dy), radius=0.055, facecolor=_COLORS["white"],
                            edgecolor=_COLORS["drone"], linewidth=1.0, zorder=58))


def _add_drone_path(
    ax: plt.Axes,
    start: Point,
    end: Point,
    *,
    line_scale: float = 1.0,
) -> None:
    """Three straight segments: takeoff, cruise, land — each with a mid-segment arrow."""
    p0 = (start[0], start[1], 0.0)
    p1 = (start[0], start[1], _CRUISE_ALT)
    p2 = (end[0],   end[1],   _CRUISE_ALT)
    p3 = (end[0],   end[1],   0.0)

    for pa, pb in [(p0, p1), (p1, p2), (p2, p3)]:
        xy = _projected([pa, pb])
        ax.plot(
            xy[:, 0],
            xy[:, 1],
            color=_COLORS["drone"],
            linewidth=3.3 * line_scale,
            solid_capstyle="round",
            zorder=50,
            path_effects=[
                pe.Stroke(
                    linewidth=5.5 * line_scale,
                    foreground=_COLORS["white"],
                    alpha=0.85,
                ),
                pe.Normal(),
            ],
        )

    for pa, pb in [(p0, p1), (p1, p2), (p2, p3)]:
        pa_np, pb_np = np.array(pa), np.array(pb)
        s = _project(*(pa_np * 0.60 + pb_np * 0.40))
        e = _project(*(pa_np * 0.40 + pb_np * 0.60))
        ax.add_patch(FancyArrowPatch(
            s,
            e,
            arrowstyle="-|>",
            mutation_scale=16 * line_scale,
            linewidth=0,
            color=_COLORS["drone"],
            zorder=55,
        ))

    _add_drone_icon(ax, ((start[0] + end[0]) / 2, (start[1] + end[1]) / 2, _CRUISE_ALT))


def _add_scale_annotation(ax: plt.Axes, road_distance: float, straight_distance: float) -> None:
    ratio = road_distance / straight_distance
    text = (
        "Detour ratio\n"
        r"$= \frac{\mathrm{road\ distance}}{\mathrm{straight-line\ distance}}$"
        f"\n= {ratio:.2f} in this example"
    )
    ax.text(8.15, 6.62, text, fontsize=10.2, color=_COLORS["text"], ha="left", va="top",
            linespacing=1.25,
            bbox=dict(boxstyle="round,pad=0.45,rounding_size=0.08", facecolor="#ffffff",
                      edgecolor="#d3d8d5", alpha=0.94),
            zorder=70)

# ---------------------------------------------------------------------------
# Public plot function
# ---------------------------------------------------------------------------

def _draw_detour_ratio_concept(
    ax: "plt.Axes",
    plot_title: bool = False,
    compact: bool = False,
) -> None:
    """Draw detour ratio concept scene onto *ax*.

    The compact variant is reserved for the standalone 60 mm paper figure. It
    keeps the complete 3D building scene while reducing peripheral roads and
    non-essential labels.
    """
    ax.set_facecolor(_COLORS["background"])

    if compact:
        grid_x = [1.0, 2.8, 4.6, 6.4]
        grid_y = _GRID_Y
        scene_bounds = (0.55, 7.0, 0.55, 5.45)
        ground_patch = _add_ground(ax, bounds=scene_bounds)
        _add_road_network(
            ax,
            clip_patch=ground_patch,
            grid_x=grid_x,
            grid_y=grid_y,
            bounds=scene_bounds,
            linewidth=4.5,
        )
        _add_buildings(
            ax,
            grid_x=grid_x,
            grid_y=grid_y,
            building_width=0.58,
            building_depth=0.50,
            margin=0.22,
        )
    else:
        grid_x = _GRID_X
        grid_y = _GRID_Y
        ground_patch = _add_ground(ax)
        _add_road_network(ax, clip_patch=ground_patch)
        _add_buildings(ax)

    origin      = (grid_x[0], grid_y[0])
    destination = (grid_x[3], grid_y[2])
    bike_route  = [
        origin,
        (grid_x[1], grid_y[0]),
        (grid_x[1], grid_y[1]),
        (grid_x[2], grid_y[1]),
        (grid_x[2], grid_y[2]),
        destination,
    ]

    road_distance     = _path_length(bike_route)
    straight_distance = math.dist(origin, destination)

    # E-bike route: solid layer below buildings + transparent layer above
    route_scale = 0.6 if compact else 1.0
    _add_polyline(
        ax,
        bike_route,
        color=_COLORS["bike_shadow"],
        linewidth=7.5 * route_scale,
        zorder=7,
        alpha=0.16,
    )
    _add_polyline(
        ax,
        bike_route,
        color=_COLORS["bike"],
        linewidth=4.0 * route_scale,
        zorder=8,
    )
    arrow_scale = 9.0 if compact else 18.0
    _add_arrow_head(
        ax,
        bike_route,
        _COLORS["bike"],
        zorder=8,
        mutation_scale=arrow_scale,
    )
    _add_polyline(
        ax,
        bike_route,
        color=_COLORS["bike"],
        linewidth=4.0 * route_scale,
        zorder=25,
        alpha=0.2,
    )
    _add_arrow_head(
        ax,
        bike_route,
        _COLORS["bike"],
        zorder=25,
        alpha=0.2,
        mutation_scale=arrow_scale,
    )

    _add_drone_path(ax, origin, destination, line_scale=route_scale)

    origin_xy = _add_marker(
        ax,
        origin,
        color=_COLORS["origin"],
        label="Merchant",
        label_offset=(-0.08, -0.34) if compact else (-0.25, -0.36),
        zorder=65,
        label_fontsize=6.5 if compact else 10.5,
        marker_scale=1.1 if compact else 1.0,
    )
    dest_xy = _add_marker(
        ax,
        destination,
        color=_COLORS["destination"],
        label="Customer",
        label_offset=(0.25, 0.32) if compact else (0.90, 0.40),
        zorder=65,
        shape="square",
        label_fontsize=6.5 if compact else 10.5,
        marker_scale=1.1 if compact else 1.0,
    )

    if not compact:
        mid_bike = _project(4.75, _GRID_Y[1])
        ax.text(mid_bike[0], mid_bike[1] - 0.72, "E-bike route\non roads",
                color=_COLORS["bike"], fontsize=10.0, ha="center", va="top", fontweight="bold", zorder=60)

        mid_drone = _project(4.75, 3.0, _CRUISE_ALT)
        ax.text(mid_drone[0] + 0.3, mid_drone[1] + 0.55, "Drone route\non air space",
                color=_COLORS["drone"], fontsize=10.5, ha="center", va="bottom", fontweight="bold", zorder=60)

        # Segment labels on the side of each drone flight phase
        tkoff_mid = _project(_GRID_X[0], _GRID_Y[0], _CRUISE_ALT / 2)
        ax.text(tkoff_mid[0] + 0.18, tkoff_mid[1], "takeoff",
                color=_COLORS["drone"], fontsize=8.5, ha="left", va="center", zorder=60)

        cruise_mid = _project(4.75, 3.0, _CRUISE_ALT)
        ax.text(cruise_mid[0], cruise_mid[1] - 0.28, "cruise",
                color=_COLORS["drone"], fontsize=8.5, ha="center", va="top", zorder=60)

        land_mid = _project(_GRID_X[3], _GRID_Y[2], _CRUISE_ALT / 2)
        ax.text(land_mid[0] - 0.18, land_mid[1], "land",
                color=_COLORS["drone"], fontsize=8.5, ha="right", va="center", zorder=60)

    ax.plot([origin_xy[0], dest_xy[0]], [origin_xy[1], dest_xy[1]],
            color=_COLORS["drone"], linewidth=0.8 if compact else 1.2,
            alpha=0.28, linestyle=":", zorder=18)

    straight_mid = (np.array(origin_xy) + np.array(dest_xy)) / 2
    straight_offset = (-1.25, 0.55) if compact else (-1.8, 1.4)
    ax.text(straight_mid[0] + straight_offset[0], straight_mid[1] + straight_offset[1],
            "Straight-line distance", color=_COLORS["drone"],
            fontsize=6.5 if compact else 9.5, ha="center", va="bottom",
            zorder=60)

    road_label = _project((grid_x[2] + grid_x[3]) / 2, grid_y[2])
    ax.text(road_label[0], road_label[1] - 0.28, "Road distance",
            color=_COLORS["bike"], fontsize=6.5 if compact else 9.5,
            ha="center", va="top", zorder=60)

    if plot_title and not compact:
        _add_scale_annotation(ax, road_distance, straight_distance)
        ax.text(0.28, 6.62, "Reduced detours favor drones",
                fontsize=12.8, fontweight="bold", color=_COLORS["text"], ha="left", va="top", zorder=80)
        ax.text(0.28, 6.32, "Abstract road network example",
                fontsize=9.4, color=_COLORS["muted"], ha="left", va="top", zorder=80)

    ax.set_aspect("equal")
    if compact:
        ax.set_xlim(0.4, 9.6)
        ax.set_ylim(-0.2, 4.5)
    else:
        ax.set_xlim(-0.25, 12.75)
        ax.set_ylim(-0.25, 4.8)
    ax.axis("off")


def plot_detour_ratio_concept(
    save_path: str = None,
    figsize: tuple = (60.0 / 25.4, (60.0 / 25.4) / (9.2 / 4.7)),
    for_latex: bool = False,
    plot_title: bool = False,
):
    """Abstract vector figure illustrating the delivery detour ratio concept.

    Args:
        save_path: Path to save figure. Shows plot if None.
        figsize: Figure size tuple.
        for_latex: If True, save as PDF.
        plot_title: Retained for the standard paper-figure interface. The
            compact figure does not show a title or formula annotation.
    """
    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes((0.0, 0.0, 1.0, 1.0))
    fig.patch.set_facecolor(_COLORS["background"])
    _draw_detour_ratio_concept(ax, plot_title=plot_title, compact=True)

    if save_path:
        if for_latex and not save_path.endswith(".pdf"):
            save_path = save_path.rsplit(".", 1)[0] + ".pdf"
        if plot_title:
            save_path = _add_title_suffix(save_path)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(
            save_path,
            dpi=300,
            facecolor=fig.get_facecolor(),
            bbox_inches=None,
            pad_inches=0,
        )
        grid_x = [1.0, 2.8, 4.6, 6.4]
        grid_y = list(_GRID_Y)
        origin = (grid_x[0], grid_y[0])
        destination = (grid_x[3], grid_y[2])
        bike_route = [
            origin,
            (grid_x[1], grid_y[0]),
            (grid_x[1], grid_y[1]),
            (grid_x[2], grid_y[1]),
            (grid_x[2], grid_y[2]),
            destination,
        ]
        road_distance = _path_length(bike_route)
        straight_distance = math.dist(origin, destination)
        write_figure_report(
            save_path.rsplit(".", 1)[0] + "_report.json",
            {
                "figure_name": "detour_ratio_concept",
                "figure_path": save_path,
                "panel_type": "conceptual_route_geometry",
                "metric_definition": "road distance divided by straight-line distance",
                "example_geometry": {
                    "origin": list(origin),
                    "destination": list(destination),
                    "road_route": [list(point) for point in bike_route],
                    "straight_route": [list(origin), list(destination)],
                    "road_distance_world_units": road_distance,
                    "straight_distance_world_units": straight_distance,
                    "detour_ratio": road_distance / straight_distance,
                },
                "transport_modes": {
                    "road_route": "E-bike",
                    "straight_route": "Drone",
                },
                "interpretation_policy": "Concept geometry only; no empirical observations",
            },
        )
        print(f"Detour ratio concept saved to {save_path}")
    else:
        plt.show()

    plt.close(fig)
