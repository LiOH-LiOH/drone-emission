"""Interactive label tuning for selected paper figures.

Run:
    cd src/drone_emission
    ../../.venv/bin/python -m visualization.paper_label_tuner --env local

Drag labels with the mouse, then click "Save Layout".
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import numpy as np
import pandas as pd


def _bootstrap_paths():
    here = os.path.dirname(os.path.abspath(__file__))
    src = os.path.normpath(os.path.join(here, ".."))
    if src not in sys.path:
        sys.path.insert(0, src)


def _raw_offset_from_effective(effective_offset, highlight_size):
    """Invert the map-only label expansion from raw offset to display offset."""
    effective = np.array(effective_offset, dtype=float)
    norm = np.linalg.norm(effective)
    if norm <= 1e-9:
        return (0.0, 0.0)

    radius_points = np.sqrt(float(highlight_size)) / 2.0
    text_gap_points = 6.0
    raw_length = max(norm - radius_points - text_gap_points, 1.0)
    raw = effective / norm * raw_length
    return (float(raw[0]), float(raw[1]))


class _LabelDragController:
    """Mouse drag and button controller for map label annotations."""

    def __init__(self, fig, ax, label_artists, save_path):
        self.fig = fig
        self.ax = ax
        self.label_artists = label_artists
        self.save_path = save_path
        fig.subplots_adjust(bottom=0.085)
        self.active_city = None
        self.active_start_mouse = None
        self.active_start_offset = None
        self.initial_offsets = {
            city: tuple(item["annotation"].get_position())
            for city, item in label_artists.items()
        }
        self.current_offsets = dict(self.initial_offsets)

        canvas = fig.canvas
        canvas.mpl_connect("button_press_event", self.on_press)
        canvas.mpl_connect("motion_notify_event", self.on_motion)
        canvas.mpl_connect("button_release_event", self.on_release)

        self.save_button = Button(
            fig.add_axes((0.58, 0.015, 0.14, 0.045)),
            "Save Layout",
        )
        self.reset_button = Button(fig.add_axes((0.73, 0.015, 0.10, 0.045)), "Reset")
        self.quit_button = Button(fig.add_axes((0.84, 0.015, 0.08, 0.045)), "Quit")
        self.save_button.on_clicked(self.save)
        self.reset_button.on_clicked(self.reset)
        self.quit_button.on_clicked(self.quit)

    def on_press(self, event):
        if event.button != 1 or event.x is None or event.y is None:
            return
        for city, item in self.label_artists.items():
            contains, _ = item["annotation"].contains(event)
            if not contains:
                renderer = self.fig.canvas.get_renderer()
                bbox = item["annotation"].get_window_extent(renderer=renderer)
                contains = bbox.expanded(1.5, 1.8).contains(event.x, event.y)
            if contains:
                self.active_city = city
                self.active_start_mouse = (event.x, event.y)
                self.active_start_offset = item["annotation"].get_position()
                print(f"Dragging {city}")
                return

    def on_motion(self, event):
        if self.active_city is None or event.x is None or event.y is None:
            return
        dx_points = (event.x - self.active_start_mouse[0]) * 72.0 / self.fig.dpi
        dy_points = (event.y - self.active_start_mouse[1]) * 72.0 / self.fig.dpi
        new_offset = (
            self.active_start_offset[0] + dx_points,
            self.active_start_offset[1] + dy_points,
        )
        annotation = self.label_artists[self.active_city]["annotation"]
        annotation.set_position(new_offset)
        self.current_offsets[self.active_city] = new_offset
        self.fig.canvas.draw_idle()

    def on_release(self, _event):
        if self.active_city is not None:
            print(f"{self.active_city}: {self.current_offsets[self.active_city]}")
        self.active_city = None
        self.active_start_mouse = None
        self.active_start_offset = None

    def _raw_offsets(self):
        return {
            city: _raw_offset_from_effective(
                self.current_offsets.get(city, item["annotation"].get_position()),
                item["highlight_size"],
            )
            for city, item in self.label_artists.items()
        }

    def save(self, _event=None):
        from visualization.analysis import save_oracle_reduction_label_layout

        path = save_oracle_reduction_label_layout(self._raw_offsets(), self.save_path)
        print(f"Saved label layout to {path}")

    def reset(self, _event=None):
        for city, offset in self.initial_offsets.items():
            self.label_artists[city]["annotation"].set_position(offset)
            self.current_offsets[city] = offset
        self.fig.canvas.draw_idle()
        print("Reset label positions to the loaded layout")

    def quit(self, _event=None):
        plt.close(self.fig)


def main():
    _bootstrap_paths()
    from utils import load_config
    from visualization.analysis import (
        ORACLE_REDUCTION_LABEL_LAYOUT_PATH,
        load_oracle_reduction_label_layout,
        plot_regions_map_oracle_reduction_rank,
    )

    parser = argparse.ArgumentParser(description="Interactively tune paper map labels")
    parser.add_argument("--env", type=str, default=None)
    parser.add_argument(
        "--figure",
        type=str,
        default="regions_map_oracle_reduction_rank_map_only",
        choices=["regions_map_oracle_reduction_rank_map_only"],
    )
    parser.add_argument("--layout", type=str, default=ORACLE_REDUCTION_LABEL_LAYOUT_PATH)
    args = parser.parse_args()

    project_root = os.path.normpath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")
    )
    config = load_config(os.path.join(project_root, "experiment.toml"), env=args.env)
    stats_path = os.path.join(config["data_dir"], "all_region_stats.csv")
    if not os.path.exists(stats_path):
        raise FileNotFoundError(f"Missing {stats_path}; run consolidate_stats first")

    df = pd.read_csv(stats_path)
    label_offsets = load_oracle_reduction_label_layout(args.layout)
    result = plot_regions_map_oracle_reduction_rank(
        df_stats=df,
        stats_csv_path=stats_path,
        save_path=None,
        figsize=(8.5, 7),
        for_latex=False,
        plot_title=False,
        show_rank_inset=False,
        figure_name=args.figure,
        label_leaders_from_bubble_edge=True,
        draw_south_china_inset_bubbles=True,
        color_scale_style="legend",
        label_offsets=label_offsets,
        return_label_artists=True,
    )
    if result is None:
        raise RuntimeError("Could not render interactive map label tuner")

    _LabelDragController(
        result["fig"],
        result["ax_map"],
        result["label_artists"],
        args.layout,
    )
    print("Drag labels with the mouse, then click 'Save Layout'.")
    print("Use 'Reset' to restore loaded positions or 'Quit' to close.")
    plt.show()


if __name__ == "__main__":
    main()
