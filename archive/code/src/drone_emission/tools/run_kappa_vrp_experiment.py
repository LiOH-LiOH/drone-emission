"""Run the city-derived VRP experiment for the limiting e-bike kappa.

Examples:
    cd src/drone_emission
    ../../.venv/bin/python -m tools.run_kappa_vrp_experiment --env remote --smoke
    ../../.venv/bin/python -m tools.run_kappa_vrp_experiment --env remote
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from analysis.kappa_vrp import run_experiment
from utils import load_config


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Estimate the limiting same-restaurant e-bike bundling factor "
            "with directed closed-route VRPs."
        )
    )
    parser.add_argument(
        "--env",
        default="remote",
        help="experiment.toml environment; remote is required for the full dataset",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run one replication and capacities 1-4 without sensitivities.",
    )
    parser.add_argument(
        "--skip-sensitivities",
        action="store_true",
        help="Run the three headline curves only.",
    )
    args = parser.parse_args()

    config = load_config(str(PROJECT_ROOT / "experiment.toml"), env=args.env)
    result = run_experiment(
        exp_config=config,
        project_root=PROJECT_ROOT,
        smoke=args.smoke,
        skip_sensitivities=args.skip_sensitivities or args.smoke,
    )
    print(
        json.dumps(
            {
                "output_dir": result["output_dir"],
                "figure_png": result["figure_png"],
                "figure_pdf": result["figure_pdf"],
                "plateau": result["plateau"],
                "elapsed_seconds": result["manifest"]["elapsed_seconds"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
