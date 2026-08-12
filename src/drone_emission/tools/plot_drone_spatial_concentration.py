"""Render drone spatial-concentration figures from the persisted city cache.

This command never computes MPCS. Generate the cache explicitly with
``analysis.drone_spatial_concentration`` before rendering figures.

Usage:
    cd src/drone_emission
    ../../.venv/bin/python -m tools.plot_drone_spatial_concentration \
        --env remote --for-latex --only all
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import pandas as pd  # noqa: E402

from analysis.drone_spatial_concentration import (  # noqa: E402
    ABSOLUTE_REDUCTION_COLUMN,
    RELATIVE_REDUCTION_COLUMN,
    load_representative_payloads,
)
from utils import load_config  # noqa: E402
from visualization.analysis import (  # noqa: E402
    plot_city_reduction_concentration_scatter,
    plot_representative_city_drone_heatmaps,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
CACHE_FILENAME = "city_drone_spatial_concentration.csv"
REDUCTION_SCATTER_FILENAME = "city_reduction_potential_scatter.png"
CONCENTRATION_DISTRIBUTION_FILENAME = "city_concentration_distribution.png"
HEATMAP_FILENAME = "representative_city_drone_heatmaps.png"
COMMON_REQUIRED_COLUMNS = {
    "region_key",
    "region_name_en",
    "region_level",
    "status",
    "pilot_concentration_score_pct",
    "concentration_class",
    "is_representative",
}
SCATTER_REQUIRED_COLUMNS = {
    ABSOLUTE_REDUCTION_COLUMN,
    RELATIVE_REDUCTION_COLUMN,
}
HEATMAP_REQUIRED_COLUMNS = {
    "eligible_annual_orders",
    "eligible_order_share_pct",
    "pilot_typology",
    "representative_typology",
    "capture_3km_pct",
    "capture_3km_center_x_m",
    "capture_3km_center_y_m",
}


def _load_cache(path: Path, only: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Concentration cache not found: {path}. Run "
            "analysis.drone_spatial_concentration first."
        )
    results = pd.read_csv(path)
    required = set(COMMON_REQUIRED_COLUMNS)
    if only in {"scatter", "all"}:
        required.update(SCATTER_REQUIRED_COLUMNS)
    if only in {"heatmaps", "all"}:
        required.update(HEATMAP_REQUIRED_COLUMNS)
    missing = sorted(required.difference(results.columns))
    if missing:
        raise ValueError(f"Concentration cache is missing columns: {missing}")
    if results["region_key"].isna().any():
        raise ValueError("Concentration cache contains missing region_key values")
    if results["region_key"].duplicated().any():
        raise ValueError("Concentration cache contains duplicate region_key values")
    return results


def render_cached_concentration_figures(
    cache_path: Path,
    data_dir: Path,
    output_dir: Path,
    *,
    only: str = "all",
    for_latex: bool = False,
    plot_title: bool = False,
) -> list[Path]:
    """Render selected figures without invoking the MPCS solver."""
    results = _load_cache(cache_path, only)
    successful = results[results["status"].eq("ok")].copy()
    if successful.empty:
        raise ValueError("Concentration cache has no successful city results")

    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    if only in {"scatter", "all"}:
        scatter_path = output_dir / REDUCTION_SCATTER_FILENAME
        distribution_path = output_dir / CONCENTRATION_DISTRIBUTION_FILENAME
        rendered_paths = plot_city_reduction_concentration_scatter(
            successful,
            scatter_save_path=str(scatter_path),
            distribution_save_path=str(distribution_path),
            for_latex=for_latex,
        )
        outputs.extend(Path(path) for path in rendered_paths if path)

    if only in {"heatmaps", "all"}:
        representative_payloads = load_representative_payloads(results, data_dir)
        if not representative_payloads:
            raise ValueError("Concentration cache has no representative cities")
        heatmap_path = output_dir / HEATMAP_FILENAME
        rendered = plot_representative_city_drone_heatmaps(
            representative_payloads,
            save_path=str(heatmap_path),
            for_latex=for_latex,
            plot_title=plot_title,
        )
        outputs.append(Path(rendered))

    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render drone concentration figures from the persisted cache"
    )
    parser.add_argument("--env", default=None, help="Configuration environment")
    parser.add_argument("--cache", type=Path, default=None, help="Cached city CSV")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--only",
        choices=("scatter", "heatmaps", "all"),
        default="all",
    )
    parser.add_argument("--for-latex", action="store_true", help="Save PDF figures")
    parser.add_argument("--plot-title", action="store_true", help="Show figure titles")
    args = parser.parse_args()

    config = load_config(PROJECT_ROOT / "experiment.toml", env=args.env)
    data_dir = Path(config["data_dir"])
    figures_dir = Path(config["figures_dir"])
    cache_path = (
        args.cache
        or data_dir / "analysis" / "drone_spatial_concentration" / CACHE_FILENAME
    )
    output_dir = (
        args.output_dir
        or figures_dir / "analysis" / "drone_spatial_concentration"
    )
    outputs = render_cached_concentration_figures(
        cache_path,
        data_dir,
        output_dir,
        only=args.only,
        for_latex=args.for_latex,
        plot_title=args.plot_title,
    )
    for output in outputs:
        print(f"Verified cached figure: {output} ({output.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
