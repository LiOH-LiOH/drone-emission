"""Candidate-only comparison of delivery-mode lifecycle emissions.

This script is intentionally not registered in the formal paper manifest.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib.pyplot as plt

from drone_emission.analysis.weighted_quantile import weighted_csv_quantile
from drone_emission.emission import DELIVERY_MODES, lifecycle_emission_per_order
from drone_emission.utils.config import configure_cache, load_config


BEIJING_GRID_EF_KG_PER_KWH = 0.558
# DESNZ 2026 mineral petrol, Net CV:
# (0.25464 direct + 0.06582 WTT) kg CO2e/kWh / 3.6 MJ/kWh.
PETROL_WTW_EF_KG_PER_MJ = 0.08901666666666667

MODE_LABELS = {
    "walking": "Walking",
    "pedal_bicycle": "Pedal bicycle",
    "pedelec_ebike": "Pedelec e-bike",
    "electric_moped": "Electric moped",
    "fuel_moped": "Fuel moped",
    "battery_electric_car": "Battery-electric car",
    "fuel_car": "Fuel car",
    "drone": "Drone",
}

MODE_COLORS = {
    "walking": "#8f8f8f",
    "pedal_bicycle": "#4f8f62",
    "pedelec_ebike": "#76a96d",
    "electric_moped": "#5f9fbd",
    "fuel_moped": "#c58a52",
    "battery_electric_car": "#4678a8",
    "fuel_car": "#a95f43",
    "drone": "#7b6aa8",
}


def _next_review_directory(project_root: Path) -> Path:
    date = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d")
    stem = project_root / "figures" / "review" / f"mode-lifecycle-comparison_{date}"
    candidate = stem
    suffix = 2
    while candidate.exists():
        candidate = Path(f"{stem}_{suffix:02d}")
        suffix += 1
    return candidate


def build_comparison(
    *,
    od_csv: str | Path,
    exp_config,
    grid_ef_kg_per_kwh: float = BEIJING_GRID_EF_KG_PER_KWH,
    fuel_wtw_ef_kg_per_mj: float = PETROL_WTW_EF_KG_PER_MJ,
    quantile_cache_dir: str | Path | None = None,
):
    """Calculate all modes at Beijing's sales-weighted road-distance P75."""
    quantile = weighted_csv_quantile(
        od_csv,
        value_column="road_distance_m",
        weight_column="annual_sales",
        quantile=0.75,
        cache_dir=quantile_cache_dir,
    )
    distance_km = quantile.value / 1000.0
    rows = []
    for mode in DELIVERY_MODES:
        emissions_g = lifecycle_emission_per_order(
            mode,
            distance_km,
            exp_config=exp_config,
            grid_ef_kg_per_kwh=grid_ef_kg_per_kwh,
            fuel_wtw_ef_kg_per_mj=fuel_wtw_ef_kg_per_mj,
        )
        rows.append((mode, float(emissions_g)))
    rows.sort(key=lambda item: (item[1], DELIVERY_MODES.index(item[0])))
    return rows, distance_km, quantile


def plot_comparison(
    rows,
    *,
    distance_km: float,
    save_path: str | Path,
    plot_title: bool = False,
    figsize=(180.0 / 25.4, 92.0 / 25.4),
):
    """Plot one ascending horizontal bar per delivery mode."""
    save_path = Path(save_path)
    labels = [MODE_LABELS[mode] for mode, _ in rows]
    values = [value for _, value in rows]
    colors = [MODE_COLORS[mode] for mode, _ in rows]

    fig, ax = plt.subplots(figsize=figsize, facecolor="white")
    bars = ax.barh(labels, values, color=colors, height=0.68)
    max_value = max(values) if values else 0.0
    label_offset = max(max_value * 0.012, 0.8)
    for bar, value in zip(bars, values):
        ax.text(
            value + label_offset,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.1f}",
            va="center",
            ha="left",
            fontsize=7.5,
            color="0.2",
        )

    ax.set_xlim(0, max_value * 1.15 + label_offset)
    ax.set_xlabel("Lifecycle emissions (g CO$_2$e/order)", fontsize=8.5)
    ax.tick_params(axis="both", labelsize=7.5, length=2.5)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.grid(axis="x", color="0.9", linewidth=0.5)
    ax.set_axisbelow(True)
    if plot_title:
        ax.set_title(
            f"Delivery modes at the same {distance_km:.2f}-km one-way distance",
            fontsize=9,
            loc="left",
        )
        save_path = save_path.with_name(
            f"{save_path.stem}_t{save_path.suffix}"
        )
    fig.tight_layout(pad=0.8)
    save_path.parent.mkdir(parents=True, exist_ok=False)
    fig.savefig(save_path, dpi=300, facecolor="white")
    plt.close(fig)
    return save_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--od-csv",
        default="data/city/beijing/beijing_od_pairs.csv",
        help="Beijing OD CSV with road_distance_m and annual_sales",
    )
    parser.add_argument("--config", default="experiment.toml")
    parser.add_argument("--env", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--plot-title", action="store_true")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[2]
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else _next_review_directory(project_root)
    )
    config = load_config(args.config, env=args.env)
    configure_cache(config)
    rows, distance_km, quantile = build_comparison(
        od_csv=args.od_csv,
        exp_config=config,
    )
    figure_path = output_dir / "delivery_mode_lifecycle_comparison.pdf"
    figure_path = plot_comparison(
        rows,
        distance_km=distance_km,
        save_path=figure_path,
        plot_title=args.plot_title,
    )

    report = {
        "schema_version": 1,
        "figure_name": "delivery_mode_lifecycle_comparison_candidate",
        "figure_path": str(figure_path),
        "formal_manifest_registered": False,
        "functional_unit": "g CO2e per unbundled order",
        "distance": {
            "definition": "common one-way travelled distance; round trip is 2d",
            "source_column": "road_distance_m",
            "weight_column": "annual_sales",
            "quantile": 0.75,
            "method": "smallest value with cumulative weight >= q * total weight",
            "value_km": distance_km,
            "valid_rows": quantile.valid_rows,
            "total_annual_sales": quantile.total_weight,
        },
        "scenario": {
            "grid_ef_kg_per_kwh": BEIJING_GRID_EF_KG_PER_KWH,
            "fuel_wtw_ef_kg_per_mj": PETROL_WTW_EF_KG_PER_MJ,
            "orders_per_trip": 1,
        },
        "boundary": {
            "included": [
                "vehicle and battery manufacturing",
                "assembly and disposal",
                "source-included fluids",
                "vehicle delivery at point of purchase",
                "energy use and upstream supply",
            ],
            "excluded": [
                "road and charging infrastructure",
                "platform and warehouse operations",
                "shared-vehicle operations and rebalancing",
                "human metabolism",
            ],
        },
        "sources": {
            "ground_modes": (
                "https://www.itf-oecd.org/good-to-go-environmental-performance-new-mobility"
            ),
            "fuel_factor": (
                "https://www.gov.uk/government/publications/"
                "greenhouse-gas-reporting-conversion-factors-2026"
            ),
        },
        "emissions_g_per_order": [
            {"mode": mode, "value": value} for mode, value in rows
        ],
    }
    report_path = output_dir / "delivery_mode_lifecycle_comparison_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    print(f"Candidate figure saved to {figure_path}")
    print(f"Candidate report saved to {report_path}")


if __name__ == "__main__":
    main()
