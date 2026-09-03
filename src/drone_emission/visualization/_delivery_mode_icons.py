"""Original vector pictograms for the delivery-mode comparison figure."""

from __future__ import annotations

from collections.abc import Callable

from matplotlib.lines import Line2D
from matplotlib.offsetbox import DrawingArea
from matplotlib.patches import (
    Circle,
    FancyBboxPatch,
    PathPatch,
    Polygon,
    Rectangle,
)
from matplotlib.path import Path


ICON_SIZE_MM = 7.0
ICON_EDGE_COLOR = "#303030"
ICON_DETAIL_COLOR = "#ffffff"
ICON_SEMANTICS = {
    "pedelec_ebike": "generic pedelec bicycle with a frame-mounted battery",
    "drone": "generic top-view quadrotor carrying a parcel",
    "fuel_moped": "generic step-through moped with a fuel-drop mark",
    "battery_electric_car": "generic car with a battery and lightning mark",
    "fuel_car": "generic car with a fuel-drop mark",
}


def _line(area: DrawingArea, points, *, color=ICON_EDGE_COLOR, width=1.05):
    area.add_artist(
        Line2D(
            [point[0] for point in points],
            [point[1] for point in points],
            color=color,
            linewidth=width,
            solid_capstyle="round",
            solid_joinstyle="round",
        )
    )


def _wheel(area: DrawingArea, x: float, y: float, radius: float) -> None:
    area.add_artist(
        Circle(
            (x, y),
            radius,
            facecolor="white",
            edgecolor=ICON_EDGE_COLOR,
            linewidth=1.05,
        )
    )


def _fuel_drop(center_x: float, center_y: float, scale: float) -> PathPatch:
    vertices = [
        (center_x, center_y + 1.15 * scale),
        (center_x - 0.95 * scale, center_y - 0.05 * scale),
        (center_x - 0.62 * scale, center_y - 0.95 * scale),
        (center_x, center_y - 1.15 * scale),
        (center_x + 0.62 * scale, center_y - 0.95 * scale),
        (center_x + 0.95 * scale, center_y - 0.05 * scale),
        (center_x, center_y + 1.15 * scale),
        (center_x, center_y + 1.15 * scale),
    ]
    codes = [
        Path.MOVETO,
        Path.CURVE4,
        Path.CURVE4,
        Path.CURVE4,
        Path.CURVE4,
        Path.CURVE4,
        Path.CURVE4,
        Path.CLOSEPOLY,
    ]
    return PathPatch(
        Path(vertices, codes),
        facecolor=ICON_DETAIL_COLOR,
        edgecolor=ICON_EDGE_COLOR,
        linewidth=0.45,
    )


def _draw_pedelec(area: DrawingArea, color: str, size: float) -> None:
    left = (0.24 * size, 0.28 * size)
    right = (0.76 * size, 0.28 * size)
    radius = 0.17 * size
    _wheel(area, *left, radius)
    _wheel(area, *right, radius)
    crank = (0.48 * size, 0.30 * size)
    seat = (0.40 * size, 0.58 * size)
    handle = (0.68 * size, 0.62 * size)
    _line(area, [left, crank, seat, right, crank], color=color, width=1.45)
    _line(area, [seat, (0.36 * size, 0.66 * size)])
    _line(area, [(0.32 * size, 0.66 * size), (0.43 * size, 0.66 * size)])
    _line(area, [right, handle, (0.73 * size, 0.69 * size)])
    _line(area, [(0.68 * size, 0.69 * size), (0.78 * size, 0.69 * size)])
    area.add_artist(
        FancyBboxPatch(
            (0.43 * size, 0.39 * size),
            0.20 * size,
            0.11 * size,
            boxstyle="round,pad=0.03",
            facecolor=color,
            edgecolor=ICON_EDGE_COLOR,
            linewidth=0.65,
        )
    )


def _draw_drone(area: DrawingArea, color: str, size: float) -> None:
    center = (0.50 * size, 0.52 * size)
    rotors = (
        (0.22 * size, 0.72 * size),
        (0.78 * size, 0.72 * size),
        (0.22 * size, 0.32 * size),
        (0.78 * size, 0.32 * size),
    )
    for rotor in rotors:
        _line(area, [center, rotor], width=1.25)
        area.add_artist(
            Circle(
                rotor,
                0.14 * size,
                facecolor=color,
                edgecolor=ICON_EDGE_COLOR,
                linewidth=0.85,
            )
        )
        area.add_artist(
            Circle(rotor, 0.045 * size, facecolor="white", edgecolor="none")
        )
    area.add_artist(
        FancyBboxPatch(
            (0.37 * size, 0.39 * size),
            0.26 * size,
            0.26 * size,
            boxstyle="round,pad=0.05",
            facecolor=color,
            edgecolor=ICON_EDGE_COLOR,
            linewidth=1.0,
        )
    )
    area.add_artist(
        Rectangle(
            (0.42 * size, 0.39 * size),
            0.16 * size,
            0.12 * size,
            facecolor=ICON_DETAIL_COLOR,
            edgecolor=ICON_EDGE_COLOR,
            linewidth=0.45,
        )
    )


def _draw_moped(area: DrawingArea, color: str, size: float) -> None:
    left = (0.25 * size, 0.27 * size)
    right = (0.76 * size, 0.27 * size)
    radius = 0.16 * size
    _wheel(area, *left, radius)
    _wheel(area, *right, radius)
    area.add_artist(
        Polygon(
            [
                (0.29 * size, 0.31 * size),
                (0.41 * size, 0.50 * size),
                (0.64 * size, 0.48 * size),
                (0.72 * size, 0.31 * size),
                (0.54 * size, 0.32 * size),
                (0.45 * size, 0.39 * size),
            ],
            closed=True,
            facecolor=color,
            edgecolor=ICON_EDGE_COLOR,
            linewidth=0.9,
        )
    )
    area.add_artist(
        FancyBboxPatch(
            (0.35 * size, 0.52 * size),
            0.25 * size,
            0.08 * size,
            boxstyle="round,pad=0.02",
            facecolor=color,
            edgecolor=ICON_EDGE_COLOR,
            linewidth=0.7,
        )
    )
    _line(
        area,
        [(0.64 * size, 0.47 * size), (0.68 * size, 0.69 * size)],
    )
    _line(
        area,
        [(0.64 * size, 0.69 * size), (0.76 * size, 0.69 * size)],
    )
    area.add_artist(_fuel_drop(0.51 * size, 0.43 * size, 0.055 * size))


def _draw_car(area: DrawingArea, color: str, size: float, *, electric: bool) -> None:
    area.add_artist(
        Polygon(
            [
                (0.12 * size, 0.31 * size),
                (0.18 * size, 0.50 * size),
                (0.33 * size, 0.56 * size),
                (0.43 * size, 0.70 * size),
                (0.68 * size, 0.70 * size),
                (0.80 * size, 0.54 * size),
                (0.90 * size, 0.49 * size),
                (0.92 * size, 0.31 * size),
            ],
            closed=True,
            facecolor=color,
            edgecolor=ICON_EDGE_COLOR,
            linewidth=1.0,
        )
    )
    for x in (0.30 * size, 0.75 * size):
        area.add_artist(
            Circle(
                (x, 0.30 * size),
                0.12 * size,
                facecolor=ICON_EDGE_COLOR,
                edgecolor="white",
                linewidth=0.35,
            )
        )
        area.add_artist(
            Circle(
                (x, 0.30 * size),
                0.045 * size,
                facecolor="white",
                edgecolor="none",
            )
        )
    if electric:
        area.add_artist(
            FancyBboxPatch(
                (0.44 * size, 0.43 * size),
                0.24 * size,
                0.16 * size,
                boxstyle="round,pad=0.03",
                facecolor=ICON_DETAIL_COLOR,
                edgecolor=ICON_EDGE_COLOR,
                linewidth=0.45,
            )
        )
        area.add_artist(
            Rectangle(
                (0.68 * size, 0.48 * size),
                0.025 * size,
                0.06 * size,
                facecolor=ICON_DETAIL_COLOR,
                edgecolor=ICON_EDGE_COLOR,
                linewidth=0.3,
            )
        )
        area.add_artist(
            Polygon(
                [
                    (0.57 * size, 0.57 * size),
                    (0.51 * size, 0.50 * size),
                    (0.56 * size, 0.50 * size),
                    (0.52 * size, 0.44 * size),
                    (0.63 * size, 0.52 * size),
                    (0.58 * size, 0.52 * size),
                ],
                closed=True,
                facecolor=ICON_EDGE_COLOR,
                edgecolor="none",
            )
        )
    else:
        area.add_artist(_fuel_drop(0.56 * size, 0.51 * size, 0.085 * size))


def _draw_battery_car(area: DrawingArea, color: str, size: float) -> None:
    _draw_car(area, color, size, electric=True)


def _draw_fuel_car(area: DrawingArea, color: str, size: float) -> None:
    _draw_car(area, color, size, electric=False)


_DRAWERS: dict[str, Callable[[DrawingArea, str, float], None]] = {
    "pedelec_ebike": _draw_pedelec,
    "drone": _draw_drone,
    "fuel_moped": _draw_moped,
    "battery_electric_car": _draw_battery_car,
    "fuel_car": _draw_fuel_car,
}


def make_delivery_mode_icon(
    mode: str,
    color: str,
    *,
    size_mm: float = ICON_SIZE_MM,
) -> DrawingArea:
    """Return one author-created, PDF-safe vector delivery-mode icon."""
    try:
        drawer = _DRAWERS[mode]
    except KeyError as exc:
        raise ValueError(f"No delivery-mode icon is defined for {mode!r}") from exc
    size_pt = size_mm * 72.0 / 25.4
    area = DrawingArea(size_pt, size_pt, 0, 0, clip=False)
    drawer(area, color, size_pt)
    return area


def icon_modes() -> tuple[str, ...]:
    """Return modes with an original delivery-mode pictogram."""
    return tuple(_DRAWERS)
