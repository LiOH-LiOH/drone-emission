"""Standalone design proof, not a formal paper renderer or research dataset.

Run outside the sandbox. Requires numpy, matplotlib, and colorspacious.
Only writes the explicitly supplied, new review directory.
"""

from pathlib import Path
import argparse
import hashlib
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb, to_hex
from matplotlib.patches import Rectangle
import numpy as np
from colorspacious import cspace_convert


CONFIG = Path(__file__).with_suffix(".json")
VISION = {"Normal": None, "Protanopia": "protanomaly",
          "Deuteranopia": "deuteranomaly", "Tritanopia": "tritanomaly"}


def simulate(rgb, kind):
    if kind is None:
        return rgb.copy(), 0.0
    result = cspace_convert(rgb, {"name": "sRGB1+CVD", "cvd_type": kind,
                                 "severity": 100}, "sRGB1")
    clipped = float(np.mean(np.any((result < 0) | (result > 1), axis=-1)))
    return np.clip(result, 0, 1), clipped


def make_ramp(config, name):
    spec = config[name]
    controls = cspace_convert([to_rgb(config["base"][p]) for p in spec["parents"]],
                             "sRGB1", "CAM02-UCS")
    controls[:, 0] = spec["lightness"]
    controls[:, 1:] *= (np.array(spec["chroma"]) /
                        np.linalg.norm(controls[:, 1:], axis=1))[:, None]
    t = np.linspace(0, 1, 257)  # Includes the exact diverging midpoint.
    # Compress chroma uniformly if needed; do not clip the scientific ramp.
    for factor in np.linspace(1, 0, 1001):
        adjusted = controls.copy()
        adjusted[:, 1:] *= factor
        path = np.column_stack([np.interp(t, spec["positions"], adjusted[:, i])
                                for i in range(3)])
        rgb = cspace_convert(path, "CAM02-UCS", "sRGB1")
        if np.all((rgb >= 0) & (rgb <= 1)):
            break
    else:
        raise ValueError("No in-gamut path")
    control_rgb = cspace_convert(adjusted, "CAM02-UCS", "sRGB1")
    return rgb, {"space": "CAM02-UCS", "gamut_chroma_factor": float(factor),
                 "controls": [{"position": pos, "parent": parent, "hex": to_hex(c),
                               "Jab": jab.tolist()}
                              for pos, parent, c, jab in zip(spec["positions"],
                                  spec["parents"], control_rgb, adjusted)]}


def contrast_white(rgb):
    linear = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    luminance = linear @ np.array([0.2126, 0.7152, 0.0722])
    return 1.05 / (luminance + 0.05)


def audit(rgb, name):
    report = {}
    for label, kind in VISION.items():
        viewed, clipped = simulate(rgb, kind)
        jab = cspace_convert(viewed, "sRGB1", "CAM02-UCS")
        light = jab[:, 0]
        ordered = (np.all(np.diff(light) < 0) if name == "sequential" else
                   np.all(np.diff(light[:129]) > 0) and np.all(np.diff(light[128:]) < 0))
        steps = np.linalg.norm(np.diff(jab, axis=0), axis=1)
        discrete = jab[np.linspace(0, 256, 7).astype(int)]
        report[label] = {"monotonic_lightness": bool(ordered),
                         "J_range": [float(light.min()), float(light.max())],
                         "endpoint_J_difference": float(abs(light[0] - light[-1])),
                         "min_adjacent_7_bin_distance": float(np.linalg.norm(np.diff(discrete, axis=0), axis=1).min()),
                         "step_distance_min_max": [float(steps.min()), float(steps.max())],
                         "simulated_gamut_clipped_fraction": clipped}
    report["white_background"] = {
        "minimum_contrast_opaque": float(contrast_white(rgb).min()),
        "minimum_contrast_alpha_0_8": float(contrast_white(0.8 * rgb + 0.2).min()),
        "note": "Diagnostic only; no universal scatter visibility threshold is asserted."}
    return report


def strip(axes, rgb, y0, y1):
    axes.imshow(rgb[None, :, :], extent=(0, 1, y0, y1), aspect="auto",
                interpolation="nearest", origin="lower")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    config = json.loads(CONFIG.read_text())
    ramps, registry, audits = {}, {}, {}
    for name in ("sequential", "diverging"):
        ramps[name], registry[name] = make_ramp(config, name)
        audits[name] = audit(ramps[name], name)
        assert all(audits[name][view]["monotonic_lightness"] for view in VISION), (name, audits[name])
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.grid": False, "axes.spines.top": False,
                         "axes.spines.right": False, "pdf.fonttype": 42})
    fig = plt.figure(figsize=(13, 15), layout="constrained")
    grid = fig.add_gridspec(6, 2, height_ratios=[1.0, 1.0, 1.25, 1.7, 1.35, 0.45])
    fig.suptitle("Palette proof  |  fixed base + two controlled ramps", fontsize=20, weight="bold")
    base_ax = fig.add_subplot(grid[0, :])
    base_ax.set_title("01  Fixed base colours — Paul Tol Bright + charcoal / white", loc="left", weight="bold")
    for i, (name, value) in enumerate(config["base"].items()):
        base_ax.add_patch(Rectangle((i, 0.4), 0.88, 0.5, facecolor=value,
                                    edgecolor=config["base"]["gray"], linewidth=0.7))
        base_ax.text(i + 0.44, 0.27, name, ha="center")
        base_ax.text(i + 0.44, 0.06, value, ha="center", fontsize=10)
    base_ax.set(xlim=(-0.05, 9), ylim=(-0.05, 1)); base_ax.axis("off")
    rng = np.random.default_rng(20260911)
    x = rng.uniform(0.02, 0.98, 180)
    y = rng.uniform(0.25, 0.9, 180)
    for column, name in enumerate(("sequential", "diverging")):
        rgb = ramps[name]
        axes = fig.add_subplot(grid[1, column])
        axes.set_title("02  " + ("Sequential: cyan to blue" if column == 0 else
                                "Diverging: blue / warm tint / coral"), loc="left", weight="bold")
        strip(axes, rgb, 0.52, 0.95)
        strip(axes, rgb[np.linspace(0, 256, 7).astype(int)], 0.08, 0.40)
        positions = registry[name]["controls"]
        for control in positions:
            axes.text(control["position"], -0.08, control["hex"], ha=("left" if control["position"] == 0 else
                      "right" if control["position"] == 1 else "center"), fontsize=10)
        axes.set(xlim=(0, 1), ylim=(-0.28, 1)); axes.axis("off")
        dots = fig.add_subplot(grid[2, column])
        dots.set_title("03  White-background points (synthetic; alpha = 0.8)", loc="left", fontsize=11)
        dots.scatter(x, y, c=rgb[(x * 256).astype(int)], s=32, alpha=0.8, linewidths=0)
        positions = np.linspace(0.05, 0.95, 7)
        dots.scatter(positions, np.full(7, 0.1), c=rgb[np.linspace(0, 256, 7).astype(int)], s=100, linewidths=0)
        dots.set(xlim=(0, 1), ylim=(0, 1), yticks=[], xticks=[0, 0.5, 1],
                 xticklabels=["low", "middle", "high"] if column == 0 else ["negative", "zero", "positive"])
        light_ax = fig.add_subplot(grid[3, column])
        light_ax.set_title("04  Perceptual lightness (CAM02-UCS J')", loc="left", fontsize=11)
        styles = ["-", "--", ":", "-."]
        for (label, kind), linestyle in zip(VISION.items(), styles):
            viewed, _ = simulate(rgb, kind)
            light = cspace_convert(viewed, "sRGB1", "CAM02-UCS")[:, 0]
            light_ax.plot(np.linspace(0, 1, 257), light, color=config["base"]["charcoal"],
                          linestyle=linestyle, label=label, linewidth=1.5)
        light_ax.set(xlabel="Normalized value", ylabel="Lightness", ylim=(20, 90))
        light_ax.legend(fontsize=9, ncol=2, loc="lower center", frameon=False)
        cvd_ax = fig.add_subplot(grid[4, column])
        cvd_ax.set_title("05  Colour-vision simulation (severity 100)", loc="left", fontsize=11)
        for row, (label, kind) in enumerate(VISION.items()):
            viewed, _ = simulate(rgb, kind)
            ypos = 3 - row
            strip(cvd_ax, viewed, ypos + 0.10, ypos + 0.73)
            cvd_ax.text(-0.025, ypos + 0.42, label, ha="right", va="center", fontsize=10)
        cvd_ax.set(xlim=(-0.27, 1), ylim=(0, 4)); cvd_ax.axis("off")
    footer = fig.add_subplot(grid[5, :]); footer.axis("off")
    footer.text(0, 0.85, "Candidate only. Scatter positions are synthetic; no paper data or production palette changed.", fontsize=10)
    footer.text(0, 0.35, "CVD: Machado et al. (2009), via colorspacious. Simulations are approximate; gray-scale cannot encode the sign.", fontsize=9)
    for extension in ("png", "pdf"):
        fig.savefig(args.output_dir / ("palette_proof." + extension), dpi=220, facecolor="white")
    plt.close(fig)
    # Save both continuous ramps and discrete bins for exact reproducibility.
    np.savez(args.output_dir / "ramps.npz", **ramps)
    (args.output_dir / "palette_registry.json").write_text(json.dumps({
        "base": config["base"], "derived": registry,
        "bins7": {name: [to_hex(rgb[i]) for i in np.linspace(0, 256, 7).astype(int)]
                  for name, rgb in ramps.items()}, "status": config["status"]}, indent=2) + "\n")
    (args.output_dir / "audit.json").write_text(json.dumps(audits, indent=2) + "\n")
    (args.output_dir / "config_used.json").write_text(CONFIG.read_text())
    (args.output_dir / "provenance.json").write_text(json.dumps({
        "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "config_sha256": hashlib.sha256(CONFIG.read_bytes()).hexdigest(),
        "seed": 20260911, "research_data_used": False,
        "sources": ["https://sronpersonalpages.nl/~pault/data/colourschemes.pdf",
                    "https://colorspacious.readthedocs.io/en/latest/tutorial.html",
                    "https://matplotlib.org/stable/users/explain/colors/colormaps.html"]}, indent=2) + "\n")
    print(json.dumps({"output": str(args.output_dir), "controls": registry,
                      "checks": audits}, indent=2))


if __name__ == "__main__":
    main()
