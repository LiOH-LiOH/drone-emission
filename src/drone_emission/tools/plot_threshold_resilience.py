"""Render exploratory fixed-threshold Monte Carlo resilience figures.

Examples:
    python -m tools.plot_threshold_resilience --env remote
    python -m tools.plot_threshold_resilience --env remote --only national,drivers
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import pandas as pd  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src" / "drone_emission"))

from utils import load_config  # noqa: E402
from visualization.sensitivity import (  # noqa: E402
    THRESHOLD_RESILIENCE_TYPICAL_CITIES,
    plot_threshold_resilience_driver_forest,
    plot_threshold_resilience_national_overview,
    plot_threshold_resilience_typical_city_distributions,
    validate_threshold_resilience_samples,
    validate_threshold_resilience_summary,
)

FIGURE_NAMES = {
    "national": "threshold_resilience_national_overview.png",
    "cities": "threshold_resilience_typical_city_distributions.png",
    "drivers": "threshold_resilience_driver_forest.png",
}


def _parse_only(value: str | None) -> set[str]:
    if not value:
        return set(FIGURE_NAMES)
    selected = {item.strip() for item in value.split(",") if item.strip()}
    unknown = sorted(selected - set(FIGURE_NAMES))
    if unknown:
        raise ValueError(
            f"unknown figure names {unknown}; choose from {sorted(FIGURE_NAMES)}"
        )
    return selected


def _sample_paths(data_dir: Path) -> dict[str, Path]:
    return {
        city: data_dir
        / "city"
        / city
        / "analysis"
        / "threshold_resilience"
        / "samples.csv.gz"
        for city in THRESHOLD_RESILIENCE_TYPICAL_CITIES
    }


def render_threshold_resilience_figures(
    summary_path: Path,
    drivers_path: Path,
    data_dir: Path,
    output_dir: Path,
    *,
    only: set[str],
) -> list[Path]:
    """Load validated runtime outputs and render selected exploratory figures."""
    if not summary_path.exists():
        raise FileNotFoundError(f"resilience summary missing: {summary_path}")
    if not drivers_path.exists():
        raise FileNotFoundError(f"resilience drivers missing: {drivers_path}")
    summary = pd.read_csv(summary_path)
    drivers = pd.read_csv(drivers_path)
    validate_threshold_resilience_summary(summary, expected_cities=314)
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = []

    if "national" in only:
        output = output_dir / FIGURE_NAMES["national"]
        plot_threshold_resilience_national_overview(summary, save_path=str(output))
        outputs.append(output)

    if "cities" in only:
        paths = _sample_paths(data_dir)
        missing = [str(path) for path in paths.values() if not path.exists()]
        if missing:
            raise FileNotFoundError(
                "representative-city sample files missing: " + ", ".join(missing)
            )
        samples = {city: pd.read_csv(path) for city, path in paths.items()}
        validate_threshold_resilience_samples(samples)
        output = output_dir / FIGURE_NAMES["cities"]
        plot_threshold_resilience_typical_city_distributions(
            summary, samples, save_path=str(output)
        )
        outputs.append(output)

    if "drivers" in only:
        output = output_dir / FIGURE_NAMES["drivers"]
        plot_threshold_resilience_driver_forest(drivers, save_path=str(output))
        outputs.append(output)

    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render fixed-threshold Monte Carlo resilience figures"
    )
    parser.add_argument("--env", default=None)
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--drivers", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--only",
        help="Comma-separated subset: national,cities,drivers",
    )
    args = parser.parse_args()

    config = load_config(PROJECT_ROOT / "experiment.toml", env=args.env)
    data_dir = Path(config["data_dir"])
    figures_dir = Path(config["figures_dir"])
    summary_path = args.summary or data_dir / "all_threshold_resilience.csv"
    drivers_path = args.drivers or data_dir / "all_threshold_resilience_drivers.csv"
    output_dir = (
        args.output_dir
        or figures_dir / "analysis" / "threshold_resilience"
    )
    outputs = render_threshold_resilience_figures(
        summary_path,
        drivers_path,
        data_dir,
        output_dir,
        only=_parse_only(args.only),
    )
    for output in outputs:
        print(f"Verified output: {output} ({output.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
