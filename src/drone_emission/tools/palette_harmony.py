"""Synthetic colour design proof; run outside the sandbox, not a paper renderer."""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.patches import Rectangle
import numpy as np
from colorspacious import cspace_convert


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_suffix(".json"))
    parser.add_argument("--compare-config", type=Path)
    args = parser.parse_args()
    config_path = args.config
    config = json.loads(config_path.read_text())
    args.output_dir.mkdir(parents=True, exist_ok=False)
    ramps, audit = {}, {}
    t = np.linspace(0, 1, 257)
    for name in ("sequential", "diverging"):
        anchors = np.array([to_rgb(config["base"][key]) for key in config[name]])
        jab = cspace_convert(anchors, "sRGB1", "CAM02-UCS")
        path = np.column_stack([np.interp(t, [0, .5, 1], jab[:, i]) for i in range(3)])
        rgb = cspace_convert(path, "CAM02-UCS", "sRGB1")
        if np.any((rgb < -1e-8) | (rgb > 1 + 1e-8)):
            raise ValueError(f"{name}: interpolation leaves sRGB gamut")
        rgb = np.clip(rgb, 0, 1)
        rgb[[0, 128, 256]] = anchors
        ramps[name] = rgb
        light = cspace_convert(rgb, "sRGB1", "CAM02-UCS")[:, 0]
        monotonic = (np.all(np.diff(light) < 0) if name == "sequential" else
                     np.all(np.diff(light[:129]) > 0) and np.all(np.diff(light[128:]) < 0))
        if not monotonic:
            raise ValueError(f"{name}: lightness reverses")
        audit[name] = {"control_hex": [config["base"][key] for key in config[name]],
                       "control_lightness": light[[0, 128, 256]].tolist(),
                       "lightness_span": float(np.ptp(light)), "monotonic_per_arm": bool(monotonic)}
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12,
                         "axes.grid": False, "axes.spines.top": False,
                         "axes.spines.right": False, "pdf.fonttype": 42})
    fig = plt.figure(figsize=(14, 12), layout="constrained")
    grid = fig.add_gridspec(5, 2, height_ratios=[1, 1.1, 1.7, 1.7, .25])
    fig.suptitle(config.get("title", "Deep sea & terracotta | palette design preview"), fontsize=22, weight="bold")
    ax = fig.add_subplot(grid[0, :])
    for i, (name, value) in enumerate(config["base"].items()):
        ax.add_patch(Rectangle((i, .4), .9, .5, facecolor=value, edgecolor="#BBBBBB"))
        ax.text(i + .45, .23, name, ha="center", fontsize=11)
        ax.text(i + .45, .02, value, ha="center", fontsize=10)
    ax.set(xlim=(-.1, 9), ylim=(-.1, 1)); ax.axis("off")
    rng = np.random.default_rng(config["seed"])
    x, y = rng.uniform(.02, .98, (2, 200))
    for column, name in enumerate(("sequential", "diverging")):
        rgb = ramps[name]
        ax = fig.add_subplot(grid[1, column])
        ax.set_title(name.capitalize() + " | continuous and 7 steps", loc="left", weight="bold")
        ax.imshow(rgb[None], extent=(0, 1, .55, 1), aspect="auto")
        ax.imshow(rgb[np.linspace(0, 256, 7).astype(int)][None], extent=(0, 1, 0, .4), aspect="auto")
        ax.set(ylim=(0, 1), yticks=[], xticks=[0, .5, 1], xticklabels=["Low", "Middle", "High"])
        ax = fig.add_subplot(grid[2, column])
        ax.set_title("White-background scatter | opacity 0.8", loc="left")
        ax.scatter(x, y, c=rgb[(x * 256).astype(int)], s=45, alpha=.8, linewidths=0)
        ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="Normalized colour position", yticks=[])
        ax = fig.add_subplot(grid[3, column])
        if column == 0:
            ax.set_title("Sequential bars | low to high", loc="left")
            ax.bar(np.arange(7), np.linspace(.3, 1, 7), color=rgb[np.linspace(0, 256, 7).astype(int)])
            ax.set(xticks=np.arange(7), xticklabels=[str(i) for i in range(1, 8)], ylabel="Synthetic value")
        else:
            ax.set_title("Fixed category colours | equal emphasis", loc="left")
            ax.bar(np.arange(6), np.ones(6), color=list(config["base"].values())[:6])
            ax.set(xticks=np.arange(6), xticklabels=list(config["base"])[:6], yticks=[])
    ax = fig.add_subplot(grid[4, :]); ax.axis("off")
    domain_note = "shared limits 0.05-0.80" if config.get("gef_range") == [.05, .8] else "shared limits pending domain audit"
    ax.text(0, .5, f"Synthetic design sample. GEF centre = 0.3; {domain_note}. No colour-vision constraint.", fontsize=11)
    for suffix in ("png", "pdf"):
        fig.savefig(args.output_dir / f"palette_proof.{suffix}", dpi=180, facecolor="white")
    plt.close(fig)
    if args.compare_config:
        previous = json.loads(args.compare_config.read_text())
        fig, axes = plt.subplots(2, 1, figsize=(13, 4.8), layout="constrained")
        fig.suptitle("Fixed palette comparison | previous and vivid candidate", fontsize=19, weight="bold")
        for ax, palette, title in zip(axes, [previous, config], ["Previous: deep sea & terracotta", "New candidate: vivid cool & warm"]):
            ax.set_title(title, loc="left", fontsize=13)
            for i, (name, value) in enumerate(list(palette["base"].items())[:6]):
                ax.add_patch(Rectangle((i, .35), .94, .6, facecolor=value))
                ax.text(i + .47, .18, name, ha="center", fontsize=11)
                ax.text(i + .47, -.03, value, ha="center", fontsize=11)
            ax.set(xlim=(-.05, 6), ylim=(-.15, 1)); ax.axis("off")
        fig.savefig(args.output_dir / "palette_comparison.png", dpi=180, facecolor="white")
        plt.close(fig)
        (args.output_dir / "comparison_config_used.json").write_text(args.compare_config.read_text())
    np.savez(args.output_dir / "ramps.npz", **ramps)
    (args.output_dir / "config_used.json").write_text(config_path.read_text())
    (args.output_dir / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    (args.output_dir / "provenance.json").write_text(json.dumps({
        "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "interpolation": "CAM02-UCS; exact fixed-table anchors", "research_data_used": False,
        "production_palette_changed": False, "seed": config["seed"]}, indent=2) + "\n")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
