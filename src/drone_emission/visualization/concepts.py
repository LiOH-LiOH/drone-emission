"""Concept illustration figures for the paper."""

from __future__ import annotations

import math
import os
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patheffects as pe
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon

from visualization._utils import _add_title_suffix

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


def _add_ground(ax: plt.Axes) -> "Polygon":
    plane = _projected([(0, 0), (9.7, 0), (9.7, 6.3), (0, 6.3)])
    patch = Polygon(plane, closed=True, facecolor=_COLORS["ground"],
                    edgecolor=_COLORS["ground_edge"], linewidth=1.2, zorder=1)
    ax.add_patch(patch)
    for x in np.arange(1.0, 9.6, 1.0):
        _add_polyline(ax, [(x, 0.2), (x, 6.1)], color=_COLORS["grid"], linewidth=0.7, zorder=2)
    for y in np.arange(0.8, 6.3, 0.9):
        _add_polyline(ax, [(0.2, y), (9.5, y)], color=_COLORS["grid"], linewidth=0.7, zorder=2)
    return patch


def _add_road_network(ax: plt.Axes, clip_patch=None) -> None:
    for x in _GRID_X:
        _add_polyline(ax, [(x, 0.0), (x, 6.3)], color=_COLORS["road"], linewidth=12.0, zorder=5, clip_patch=clip_patch)
        _add_polyline(ax, [(x, 0.0), (x, 6.3)], color=_COLORS["road_dark"], linewidth=1.0, zorder=6, alpha=0.32, clip_patch=clip_patch)
    for y in _GRID_Y:
        _add_polyline(ax, [(0.0, y), (9.7, y)], color=_COLORS["road"], linewidth=12.0, zorder=5, clip_patch=clip_patch)
        _add_polyline(ax, [(0.0, y), (9.7, y)], color=_COLORS["road_dark"], linewidth=1.0, zorder=6, alpha=0.32, clip_patch=clip_patch)


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


def _add_buildings(ax: plt.Axes) -> None:
    blocks = [
        (_GRID_X[0], _GRID_Y[0], _GRID_X[1], _GRID_Y[1]),
        (_GRID_X[1], _GRID_Y[0], _GRID_X[2], _GRID_Y[1]),
        (_GRID_X[2], _GRID_Y[0], _GRID_X[3], _GRID_Y[1]),
        (_GRID_X[0], _GRID_Y[1], _GRID_X[1], _GRID_Y[2]),
        (_GRID_X[1], _GRID_Y[1], _GRID_X[2], _GRID_Y[2]),
        (_GRID_X[2], _GRID_Y[1], _GRID_X[3], _GRID_Y[2]),
    ]
    block_heights = [
        [0.70, 0.45, 0.55, 0.80],
        [0.50, 0.75, 0.60, 0.40],
        [0.65, 0.50, 0.80, 0.55],
        [0.55, 0.80, 0.45, 0.70],
        [0.75, 0.50, 0.65, 0.55],
        [0.45, 0.65, 0.75, 0.50],
    ]
    bw, bd, margin = 0.65, 0.50, 0.38
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


def _add_marker(ax: plt.Axes, xy_world: Point, *, color: str, label: str,
                label_offset: Point, zorder: int = 40) -> Point:
    xy = _project(*xy_world)
    ax.add_patch(Circle(xy, radius=0.15, facecolor=color, edgecolor=_COLORS["white"],
                        linewidth=1.6, zorder=zorder,
                        path_effects=[pe.SimplePatchShadow(offset=(1.0, -1.0), alpha=0.25), pe.Normal()]))
    ax.text(xy[0] + label_offset[0], xy[1] + label_offset[1], label,
            color=_COLORS["text"], fontsize=10.5, fontweight="bold",
            ha="center", va="center", zorder=zorder + 2)
    return xy


def _add_arrow_head(ax: plt.Axes, points: Sequence[Sequence[float]], color: str,
                    zorder: int, z: float = 0.0, alpha: float = 1.0) -> None:
    start = _project(points[-2][0], points[-2][1], z)
    end   = _project(points[-1][0], points[-1][1], z)
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=18,
                                 linewidth=0, color=color, zorder=zorder, alpha=alpha))


def _add_drone_icon(ax: plt.Axes, center_world: tuple[float, float, float]) -> None:
    cx, cy = _project(center_world[0], center_world[1], center_world[2])
    ax.add_patch(Circle((cx, cy), radius=0.10, facecolor=_COLORS["drone"],
                        edgecolor=_COLORS["white"], linewidth=0.9, zorder=58))
    arm_len = 0.34
    for dx, dy in [(-arm_len, 0), (arm_len, 0), (0, -arm_len * 0.55), (0, arm_len * 0.55)]:
        ax.plot([cx, cx + dx], [cy, cy + dy], color=_COLORS["drone"], linewidth=1.2, zorder=57)
        ax.add_patch(Circle((cx + dx, cy + dy), radius=0.055, facecolor=_COLORS["white"],
                            edgecolor=_COLORS["drone"], linewidth=1.0, zorder=58))


def _add_drone_path(ax: plt.Axes, start: Point, end: Point) -> None:
    """Three straight segments: takeoff, cruise, land — each with a mid-segment arrow."""
    p0 = (start[0], start[1], 0.0)
    p1 = (start[0], start[1], _CRUISE_ALT)
    p2 = (end[0],   end[1],   _CRUISE_ALT)
    p3 = (end[0],   end[1],   0.0)

    for pa, pb in [(p0, p1), (p1, p2), (p2, p3)]:
        xy = _projected([pa, pb])
        ax.plot(xy[:, 0], xy[:, 1], color=_COLORS["drone"], linewidth=3.3,
                solid_capstyle="round", zorder=50,
                path_effects=[pe.Stroke(linewidth=5.5, foreground=_COLORS["white"], alpha=0.85), pe.Normal()])

    for pa, pb in [(p0, p1), (p1, p2), (p2, p3)]:
        pa_np, pb_np = np.array(pa), np.array(pb)
        s = _project(*(pa_np * 0.60 + pb_np * 0.40))
        e = _project(*(pa_np * 0.40 + pb_np * 0.60))
        ax.add_patch(FancyArrowPatch(s, e, arrowstyle="-|>", mutation_scale=16,
                                     linewidth=0, color=_COLORS["drone"], zorder=55))

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

def _draw_detour_ratio_concept(ax: "plt.Axes", plot_title: bool = False) -> None:
    """Draw detour ratio concept scene onto *ax*."""
    ax.set_facecolor(_COLORS["background"])

    ground_patch = _add_ground(ax)
    _add_road_network(ax, clip_patch=ground_patch)
    _add_buildings(ax)

    origin      = (_GRID_X[0], _GRID_Y[0])
    destination = (_GRID_X[3], _GRID_Y[2])
    bike_route  = [
        origin,
        (_GRID_X[1], _GRID_Y[0]),
        (_GRID_X[1], _GRID_Y[1]),
        (_GRID_X[2], _GRID_Y[1]),
        (_GRID_X[2], _GRID_Y[2]),
        destination,
    ]

    road_distance     = _path_length(bike_route)
    straight_distance = math.dist(origin, destination)

    # E-bike route: solid layer below buildings + transparent layer above
    _add_polyline(ax, bike_route, color=_COLORS["bike_shadow"], linewidth=7.5, zorder=7, alpha=0.16)
    _add_polyline(ax, bike_route, color=_COLORS["bike"], linewidth=4.0, zorder=8)
    _add_arrow_head(ax, bike_route, _COLORS["bike"], zorder=8)
    _add_polyline(ax, bike_route, color=_COLORS["bike"], linewidth=4.0, zorder=25, alpha=0.2)
    _add_arrow_head(ax, bike_route, _COLORS["bike"], zorder=25, alpha=0.2)

    _add_drone_path(ax, origin, destination)

    origin_xy = _add_marker(ax, origin,      color=_COLORS["origin"],      label="Merchant", label_offset=(-0.25, -0.36), zorder=65)
    dest_xy   = _add_marker(ax, destination, color=_COLORS["destination"], label="Customer",  label_offset=(0.90, 0.40),  zorder=65)

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
            color=_COLORS["drone"], linewidth=1.2, alpha=0.28, linestyle=":", zorder=18)

    straight_mid = (np.array(origin_xy) + np.array(dest_xy)) / 2
    ax.text(straight_mid[0] - 1.8, straight_mid[1] + 1.4, "Straight-line distance",
            color=_COLORS["drone"], fontsize=9.5, ha="center", va="bottom", zorder=60)

    road_label = _project(7.25, _GRID_Y[2])
    ax.text(road_label[0], road_label[1] - 0.28, "Road distance",
            color=_COLORS["bike"], fontsize=9.5, ha="center", va="top", zorder=60)

    if plot_title:
        _add_scale_annotation(ax, road_distance, straight_distance)
        ax.text(0.28, 6.62, "Reduced detours favor drones",
                fontsize=12.8, fontweight="bold", color=_COLORS["text"], ha="left", va="top", zorder=80)
        ax.text(0.28, 6.32, "Abstract road network example",
                fontsize=9.4, color=_COLORS["muted"], ha="left", va="top", zorder=80)

    ax.set_aspect("equal")
    ax.set_xlim(-0.25, 12.75)
    ax.set_ylim(-0.25, 4.8)
    ax.axis("off")


def plot_detour_ratio_concept(
    save_path: str = None,
    figsize: tuple = (8.2, 5.2),
    for_latex: bool = False,
    plot_title: bool = False,
):
    """Abstract vector figure illustrating the delivery detour ratio concept.

    Args:
        save_path: Path to save figure. Shows plot if None.
        figsize: Figure size tuple.
        for_latex: If True, save as PDF.
        plot_title: If True, show title and formula annotation.
    """
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=False)
    fig.patch.set_facecolor(_COLORS["background"])
    _draw_detour_ratio_concept(ax, plot_title=plot_title)

    if save_path:
        if for_latex and not save_path.endswith(".pdf"):
            save_path = save_path.rsplit(".", 1)[0] + ".pdf"
        if plot_title:
            save_path = _add_title_suffix(save_path)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight", pad_inches=0.05,
                    facecolor=fig.get_facecolor())
        print(f"Detour ratio concept saved to {save_path}")
    else:
        plt.show()

    plt.close(fig)


# ---------------------------------------------------------------------------
# Emission model LCA flowchart
# ---------------------------------------------------------------------------

def plot_emission_model_flowchart(
    save_path: str = None,
    figsize: tuple = (10.5, 5.8),
    for_latex: bool = False,
    plot_title: bool = False,
):
    """LCA flowchart (poster/paper style) for e-bike and drone delivery emission model.

    2×3 grid: rows = vehicle type (e-bike, drone),
    columns = lifecycle phase (production, use phase, end-of-life).
    Each item rendered as a centred block; row labels use programmatic icons.

    Args:
        save_path: Path to save figure. Shows plot if None.
        figsize: Figure size tuple.
        for_latex: If True, save as PDF.
        plot_title: If True, show title.
    """
    # --- Colours (aligned with shared _COLORS) ---
    BIKE_C   = "#d95f35"
    BIKE_LT  = "#fdf1ec"
    BIKE_BK  = "#e07858"   # item block border, e-bike
    DRONE_C  = "#1f77b4"
    DRONE_LT = "#edf4fb"
    DRONE_BK = "#4a8ec0"   # item block border, drone
    HDR_BG   = "#2c3d4f"
    HDR_FG   = "#ffffff"
    DIV_C    = "#c0cad2"
    ARROW_C  = "#5a6872"
    TXT      = "#1c2830"
    STXT     = "#5a6878"
    NOTE_C   = "#8898a8"
    BLK_BG   = "#ffffff"   # item block fill

    W, H = figsize
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")

    # --- Layout (1 coord ≈ 1 inch) ---
    LBL  = (0.04, 0.64)
    PROD = (0.72, 3.10)    # 2.38 wide
    USE  = (3.50, 7.34)    # 3.84 wide
    DISP = (7.72, 10.12)   # 2.40 wide
    ARR1 = (PROD[1], USE[0])
    ARR2 = (USE[1],  DISP[0])
    HROW = (5.44, 5.76)
    EROW = (2.96, 5.20)    # 2.24 tall
    DROW = (0.26, 2.76)    # 2.50 tall

    CX_LBL  = (LBL[0]  + LBL[1])  / 2
    CX_PROD = (PROD[0] + PROD[1]) / 2
    CX_USE  = (USE[0]  + USE[1])  / 2
    CX_DISP = (DISP[0] + DISP[1]) / 2

    # Block geometry
    BH  = 0.255   # standard item block height
    BG  = 0.058   # gap between blocks
    BW  = {       # block widths by column
        "prod": PROD[1] - PROD[0] - 0.30,
        "use":  USE[1]  - USE[0]  - 0.40,
        "disp": DISP[1] - DISP[0] - 0.30,
    }
    # Drone USE sub-columns
    CX_DIV  = CX_USE + 0.08           # divider x
    CX_USEL = (USE[0] + CX_DIV) / 2   # left sub-col centre
    CX_USER = (CX_DIV + USE[1]) / 2   # right sub-col centre
    BW_SUB  = (USE[1] - USE[0]) / 2 - 0.34

    # --- Private helpers ---
    def _box(x0, x1, y0, y1, fc, ec, lw=1.4, r=0.12):
        ax.add_patch(FancyBboxPatch(
            (x0, y0), x1 - x0, y1 - y0,
            boxstyle=f"round,pad=0,rounding_size={r}",
            facecolor=fc, edgecolor=ec, linewidth=lw, zorder=2,
        ))

    def _t(x, y, s, **kw):
        kw.setdefault("zorder", 5)
        ax.text(x, y, s, **kw)

    def _cell_hdr(cx, y, label, c):
        """Centred bold cell header; returns y below."""
        _t(cx, y, label, fontsize=9.5, fontweight="bold", color=c,
           ha="center", va="top")
        return y - 0.30

    def _sub_hdr(cx, y, label):
        """Centred sub-section header (smaller, dark); returns y below."""
        _t(cx, y, label, fontsize=8.5, fontweight="bold", color=TXT,
           ha="center", va="top")
        return y - 0.26

    def _blk(cx, y_top, w, h, label, bg, brd, fs=9.0, sub=None):
        """Draw one centred item block. Returns y_bottom."""
        x0, y_bot = cx - w / 2, y_top - h
        ax.add_patch(FancyBboxPatch(
            (x0, y_bot), w, h,
            boxstyle="round,pad=0,rounding_size=0.06",
            facecolor=bg, edgecolor=brd, linewidth=0.9, zorder=4,
        ))
        if sub:
            _t(cx, (y_top + y_bot) / 2 + h * 0.14, label,
               fontsize=fs, color=TXT, ha="center", va="center")
            _t(cx, (y_top + y_bot) / 2 - h * 0.16, sub,
               fontsize=fs - 1.4, color=STXT, ha="center", va="center",
               style="italic")
        else:
            _t(cx, (y_top + y_bot) / 2, label,
               fontsize=fs, color=TXT, ha="center", va="center")
        return y_bot

    def _blk_col(cx, y, items, w, bg, brd, bh=None, gap=None):
        """Stack item blocks downward from y. Items: str or (label, sub) tuple."""
        h   = bh  if bh  is not None else BH
        gap_ = gap if gap is not None else BG
        for item in items:
            if isinstance(item, tuple):
                y = _blk(cx, y, w, h, item[0], bg, brd, sub=item[1])
            else:
                y = _blk(cx, y, w, h, item, bg, brd)
            y -= gap_
        return y + gap_

    def _arrow(ym, xgap):
        ax.add_patch(FancyArrowPatch(
            (xgap[0] + 0.05, ym), (xgap[1] - 0.05, ym),
            arrowstyle="-|>", mutation_scale=12,
            color=ARROW_C, linewidth=1.4, zorder=6,
        ))

    # --- Programmatic icons ---
    def _icon_ebike(cx, cy, s, color):
        """Simplified bicycle silhouette centred at (cx, cy), scale s."""
        lw = 1.3
        wr = s * 0.28
        wL = (cx - s * 0.52, cy - s * 0.15)
        wR = (cx + s * 0.52, cy - s * 0.15)
        for w in [wL, wR]:
            ax.add_patch(Circle(w, wr, fill=False,
                                edgecolor=color, linewidth=lw, zorder=10))
        bb = (cx,           wL[1])
        sc = (cx - s*0.08,  cy + s*0.20)
        ht = (wR[0],        cy + s*0.08)
        for p1, p2 in [(wL, bb), (wL, sc), (bb, sc), (sc, ht), (sc, wR), (ht, wR)]:
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]],
                    color=color, lw=lw, solid_capstyle="round", zorder=10)
        # seat
        sy = sc[1] + s * 0.06
        ax.plot([sc[0] - s*0.13, sc[0] + s*0.07], [sy, sy],
                color=color, lw=lw * 1.5, solid_capstyle="round", zorder=10)
        # handlebar stem + bar
        ax.plot([ht[0], ht[0]], [ht[1], ht[1] + s*0.11],
                color=color, lw=lw, solid_capstyle="round", zorder=10)
        hy = ht[1] + s * 0.11
        ax.plot([ht[0] - s*0.08, ht[0] + s*0.05], [hy, hy],
                color=color, lw=lw * 1.2, solid_capstyle="round", zorder=10)

    def _icon_drone(cx, cy, s, color):
        """Simplified quadcopter viewed from above, centred at (cx, cy), scale s."""
        lw = 1.3
        arm = s * 0.46
        pr  = s * 0.20
        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            tip = (cx + dx * arm, cy + dy * arm)
            ax.plot([cx, tip[0]], [cy, tip[1]],
                    color=color, lw=lw * 1.5, solid_capstyle="round", zorder=10)
            ax.add_patch(Circle(tip, pr, fill=False,
                                edgecolor=color, linewidth=lw, zorder=10))
        ax.add_patch(Circle((cx, cy), s * 0.13,
                            facecolor=color, edgecolor="none", zorder=11))

    # ===== Column headers =====
    for (x0, x1), lbl in [
        (PROD, "PRODUCTION"), (USE, "USE PHASE"), (DISP, "END-OF-LIFE")
    ]:
        _box(x0, x1, HROW[0], HROW[1], HDR_BG, HDR_BG, r=0.09)
        _t((x0 + x1) / 2, (HROW[0] + HROW[1]) / 2, lbl,
           color=HDR_FG, fontsize=11.0, fontweight="bold", ha="center", va="center")

    # ===== Row labels with icons =====
    for (y0, y1), lbl, icon_fn, c in [
        (EROW, "E-bike", _icon_ebike, BIKE_C),
        (DROW, "Drone",  _icon_drone, DRONE_C),
    ]:
        _box(LBL[0], LBL[1], y0 + 0.08, y1 - 0.08, "white", c, lw=2.0, r=0.10)
        ym = (y0 + y1) / 2
        # Icon in upper portion, label below
        icon_s = 0.30
        icon_fn(CX_LBL, ym + 0.28, icon_s, c)
        _t(CX_LBL, ym - 0.32, lbl,
           color=c, fontsize=9.0, fontweight="bold", ha="center", va="center")

    # ===== E-BIKE ROW =====
    y0, y1 = EROW
    C, LT, BK = BIKE_C, BIKE_LT, BIKE_BK

    _box(PROD[0], PROD[1], y0, y1, LT, C, lw=1.5)
    cy = y1 - 0.16
    cy = _cell_hdr(CX_PROD, cy, "Manufacturing", C)
    cy -= 0.04
    _blk_col(CX_PROD, cy, [
        "Frame manufacturing",
        "Battery manufacturing",
        "5 packs × 0.24 kWh",
        "Battery lifespan: 2 yr",
        "Vehicle lifespan: 4 yr",
    ], BW["prod"], BLK_BG, BK)

    _box(USE[0], USE[1], y0, y1, LT, C, lw=1.5)
    cy = y1 - 0.16
    cy = _cell_hdr(CX_USE, cy, "Round-trip delivery", C)
    _t(CX_USE, cy, "Laden outbound  ·  Unladen return",
       fontsize=8.0, color=STXT, ha="center", va="top", style="italic")
    cy -= 0.22
    _blk_col(CX_USE, cy, [
        "Rolling resistance",
        "Aerodynamic drag",
        "Road grade",
        "Acceleration",
        "Auxiliary systems",
    ], BW["use"], BLK_BG, BK)

    _box(DISP[0], DISP[1], y0, y1, LT, C, lw=1.5)
    eol_h = ((y1 - y0) - 0.16 - 0.30 - 0.12 - 0.14 - BG) / 2  # fill cell height
    cy = y1 - 0.16
    cy = _cell_hdr(CX_DISP, cy, "End-of-life processing", C)
    cy -= 0.12
    _blk_col(CX_DISP, cy, [
        ("Frame (body)", "material recovery"),
        ("Battery packs", "chemical recycling"),
    ], BW["disp"], BLK_BG, BK, bh=eol_h)

    # ===== DRONE ROW =====
    y0, y1 = DROW
    C, LT, BK = DRONE_C, DRONE_LT, DRONE_BK

    _box(PROD[0], PROD[1], y0, y1, LT, C, lw=1.5)
    cy = y1 - 0.16
    cy = _cell_hdr(CX_PROD, cy, "Manufacturing", C)
    cy -= 0.04
    _blk_col(CX_PROD, cy, [
        "Airframe manufacturing",
        "Battery manufacturing",
        "4 packs × 0.389 kWh",
        "Battery lifespan: 2 yr",
        "Vehicle lifespan: 4 yr",
    ], BW["prod"], BLK_BG, BK)

    _box(USE[0], USE[1], y0, y1, LT, C, lw=1.5)
    ax.plot([CX_DIV, CX_DIV], [y0 + 0.20, y1 - 0.20],
            color=DIV_C, linewidth=0.9, zorder=3)
    cy = y1 - 0.16
    cy = _cell_hdr(CX_USE, cy, "Round-trip delivery", C)
    cy -= 0.08
    cy_l = _sub_hdr(CX_USEL, cy, "Flight segments:")
    cy_r = _sub_hdr(CX_USER, cy, "Per-segment power:")
    _blk_col(CX_USEL, cy_l, [
        "Vertical takeoff",
        "Level cruise",
        "Hover at delivery",
        "Level cruise",
        "Vertical landing",
    ], BW_SUB, BLK_BG, BK)
    _blk_col(CX_USER, cy_r, [
        "Induced rotor",
        "Blade profile drag",
        "Body drag",
        "Climb / descent",
        "Auxiliary systems",
    ], BW_SUB, BLK_BG, BK)

    _box(DISP[0], DISP[1], y0, y1, LT, C, lw=1.5)
    eol_h_d = ((y1 - y0) - 0.16 - 0.30 - 0.12 - 0.14 - BG) / 2
    cy = y1 - 0.16
    cy = _cell_hdr(CX_DISP, cy, "End-of-life processing", C)
    cy -= 0.12
    _blk_col(CX_DISP, cy, [
        ("Airframe", "material recovery"),
        ("Battery packs", "chemical recycling"),
    ], BW["disp"], BLK_BG, BK, bh=eol_h_d)

    # ===== Phase arrows =====
    for (ry0, ry1) in [EROW, DROW]:
        ym = (ry0 + ry1) / 2
        _arrow(ym, ARR1)
        _arrow(ym, ARR2)

    # ===== Lifecycle allocation note =====
    _t(W / 2, 0.13,
       "Production & end-of-life emissions amortised over vehicle lifetime deliveries  ·  "
       "Use-phase emissions per delivery trip",
       fontsize=7.5, color=NOTE_C, ha="center", va="bottom", style="italic")

    if plot_title:
        ax.set_title(
            "Life Cycle Emission Model: E-Bike and Drone Last-Mile Delivery",
            fontsize=12.0, fontweight="bold", pad=5, color=TXT,
        )

    plt.tight_layout(pad=0.15)

    if save_path:
        if for_latex and not save_path.endswith(".pdf"):
            save_path = save_path.rsplit(".", 1)[0] + ".pdf"
        if plot_title:
            save_path = _add_title_suffix(save_path)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight", pad_inches=0.05,
                    facecolor="white")
        print(f"Emission model flowchart saved to {save_path}")
    else:
        plt.show()

    plt.close(fig)


# ---------------------------------------------------------------------------
# Combined: detour ratio concept + lifecycle distance/route-factor heatmap
# ---------------------------------------------------------------------------

def plot_detour_and_lifecycle(
    od_df=None,
    save_path: str = None,
    figsize: tuple = (10.5, 10.5),
    for_latex: bool = False,
    plot_title: bool = False,
):
    """Two-panel figure: detour ratio concept (a) above lifecycle heatmap (b).

    Args:
        od_df: OD pairs DataFrame (distance_km, road_distance_m, annual_sales).
               When provided, draws iso-density contours on the heatmap panel.
        save_path: Path to save figure. Shows plot if None.
        figsize: Figure size tuple.
        for_latex: If True, save as PDF.
        plot_title: If True, show titles.
    """
    from visualization.emission import _draw_lifecycle_distance_route_factor

    fig = plt.figure(figsize=figsize, constrained_layout=True)
    fig.set_constrained_layout_pads(hspace=0.02, h_pad=0.02, wspace=0.02, w_pad=0.02)
    fig.patch.set_facecolor(_COLORS["background"])

    gs = fig.add_gridspec(2, 2, width_ratios=[1, 0.03])
    ax_a = fig.add_subplot(gs[0, :])     # spans full width (heatmap + colorbar columns)
    ax_b = fig.add_subplot(gs[1, 0])     # heatmap
    ax_cbar = fig.add_subplot(gs[1, 1])  # colorbar

    _draw_detour_ratio_concept(ax_a, plot_title=plot_title)
    _draw_lifecycle_distance_route_factor(ax_b, fig, od_df=od_df, plot_title=plot_title, cax=ax_cbar)
    ax_b.set_box_aspect(0.5)  # 2:1 w:h

    for ax, label in zip([ax_a, ax_b], ["a", "b"]):
        ax.text(0.0, 1.02, label, transform=ax.transAxes,
                fontsize=14, fontweight="bold", va="bottom", ha="left")

    if save_path:
        if for_latex and not save_path.endswith(".pdf"):
            save_path = save_path.rsplit(".", 1)[0] + ".pdf"
        if plot_title:
            save_path = _add_title_suffix(save_path)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
        print(f"Detour and lifecycle figure saved to {save_path}")
    else:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# Combined three-panel: concept (a) + lifecycle heatmap (b) + distance maps (c)
# ---------------------------------------------------------------------------

def plot_detour_lifecycle_distance(
    od_df=None,
    graph=None,
    router=None,
    save_path: str = None,
    figsize: tuple = (15.0, 10.5),
    for_latex: bool = False,
    plot_title: bool = False,
):
    """Four-panel figure: detour concept (a, top-left), lifecycle heatmap (b, bottom-left),
    one distance comparison map (c, top-right), emission reduction vs drone ratio (d, bottom-right).

    Panel c renders as a placeholder grey box when graph/router are unavailable
    (e.g. local debugging without Shanghai road data). Panel d renders a placeholder
    when od_df lacks cached lifecycle columns.

    Args:
        od_df: Shanghai OD DataFrame. Used for heatmap overlay, distance map sampling,
               and emission reduction curve.
        graph: SparseGraph for distance comparison road network.
        router: TileRouter for distance comparison shortest paths.
        save_path: Path to save figure. Shows plot if None.
        figsize: Figure size tuple.
        for_latex: If True, save as PDF.
        plot_title: If True, show titles.
    """
    from visualization.emission import (
        _draw_emission_reduction_vs_drone_ratio,
        _draw_lifecycle_distance_route_factor,
    )

    fig = plt.figure(figsize=figsize, constrained_layout=True)
    fig.set_constrained_layout_pads(hspace=0.02, h_pad=0.02, wspace=0.04, w_pad=0.04)
    fig.patch.set_facecolor(_COLORS["background"])

    gs = fig.add_gridspec(2, 3, width_ratios=[1, 0.03, 0.55])
    ax_a   = fig.add_subplot(gs[0, 0:2])   # concept: spans data + colorbar columns
    ax_b   = fig.add_subplot(gs[1, 0])      # lifecycle heatmap
    ax_cbar = fig.add_subplot(gs[1, 1])     # colorbar
    ax_c   = fig.add_subplot(gs[0, 2])      # distance map (top-right)
    ax_d   = fig.add_subplot(gs[1, 2])      # emission reduction vs drone ratio (bottom-right)

    _draw_detour_ratio_concept(ax_a, plot_title=plot_title)
    _draw_lifecycle_distance_route_factor(ax_b, fig, od_df=od_df, plot_title=plot_title, cax=ax_cbar)
    ax_b.set_box_aspect(0.5)

    if graph is not None and router is not None and od_df is not None:
        from visualization.roadnet import _compute_distance_samples, _draw_distance_panel
        sample_paths = _compute_distance_samples(od_df, router, n_samples=1)
        _draw_distance_panel(ax_c, sample_paths[0], graph, show_legend=True)
        if for_latex:
            ax_c.set_rasterization_zorder(12)
    else:
        ax_c.set_facecolor("#e8e8e8")
        ax_c.text(0.5, 0.5, "road data\nnot available", transform=ax_c.transAxes,
                  ha="center", va="center", fontsize=10, color="#888888")
        ax_c.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

    required = {"road_distance_m", "drone_lifecycle_g", "ebike_lifecycle_g"}
    if od_df is not None and required.issubset(od_df.columns):
        _draw_emission_reduction_vs_drone_ratio(ax_d, od_df, plot_title=plot_title)
        ax_d.set_box_aspect(0.5 / 0.55)
    else:
        ax_d.set_facecolor("#e8e8e8")
        ax_d.text(0.5, 0.5, "lifecycle data\nnot available", transform=ax_d.transAxes,
                  ha="center", va="center", fontsize=10, color="#888888")
        ax_d.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

    ax_a.text(0.0, 1.02, "a", transform=ax_a.transAxes,
              fontsize=14, fontweight="bold", va="bottom", ha="left")
    ax_c.text(0.0, 1.02, "b", transform=ax_c.transAxes,
              fontsize=14, fontweight="bold", va="bottom", ha="left")
    ax_b.text(0.0, 1.02, "c", transform=ax_b.transAxes,
              fontsize=14, fontweight="bold", va="bottom", ha="left")
    ax_d.text(0.0, 1.02, "d", transform=ax_d.transAxes,
              fontsize=14, fontweight="bold", va="bottom", ha="left")

    if save_path:
        if for_latex and not save_path.endswith(".pdf"):
            save_path = save_path.rsplit(".", 1)[0] + ".pdf"
        if plot_title:
            save_path = _add_title_suffix(save_path)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
        print(f"Detour, lifecycle, and distance figure saved to {save_path}")
    else:
        plt.show()
    plt.close(fig)
