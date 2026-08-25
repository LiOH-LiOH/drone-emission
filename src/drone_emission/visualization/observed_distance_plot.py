import os

import contextily as ctx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import curve_fit

from drone_emission.utils.coordinates import gcj02_to_wgs84
from drone_emission.visualization._utils import (
    _add_title_suffix,
    configure_contextily_cache,
    finalize_plot,
    setup_figure,
    write_figure_report,
)


def plot_observed_delivery_distance_distribution(
    distances,
    save_path=None,
    figsize=(5.0, 3.4),
    for_latex=False,
    plot_title=False,
    sample_metadata=None,
    distance_summary=None,
):
    """Plot the observed delivery-distance distribution and 5 km coverage.

    Args:
        distances: Array of delivery distances in km.
        save_path: Path to save the figure. Shows plot if None.
        figsize: Figure size tuple.
        for_latex: If True, save as PDF.
        plot_title: If True, show title.
        sample_metadata: Optional sample-audit metadata from the data loader.
        distance_summary: Optional audited summary retained when ``distances``
            are reconstructed from the locked histogram.
    """
    bin_width = 0.1
    threshold_km = 5.0

    plot_distances = np.asarray(distances, dtype=float)
    plot_distances = plot_distances[
        np.isfinite(plot_distances) & (plot_distances > 0)
    ]
    if plot_distances.size == 0:
        raise ValueError("No finite positive delivery distances to plot")
    display_max_km = max(1.0, float(np.ceil(plot_distances.max())))

    bins = np.arange(0, display_max_km + bin_width, bin_width)
    counts, bin_edges = np.histogram(plot_distances, bins=bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    below_threshold_count = int(
        np.count_nonzero(plot_distances <= threshold_km)
    )
    above_threshold_count = int(
        plot_distances.size - below_threshold_count
    )
    coverage_pct = 100 * below_threshold_count / plot_distances.size

    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(
        bin_centers,
        counts,
        width=bin_width * 0.9,
        color="#4C72B0",
        edgecolor="white",
        linewidth=0.25,
        alpha=0.86,
    )
    ax.axvline(
        threshold_km,
        color="#4D4D4D",
        ls="--",
        lw=0.9,
    )
    ax.text(
        threshold_km - 0.15,
        0.95,
        f"{coverage_pct:.2f}% of orders\n$\\leq$ 5 km",
        transform=ax.get_xaxis_transform(),
        color="#4D4D4D",
        fontsize=8.5,
        ha="right",
        va="top",
    )

    ax.set_xlim(0, display_max_km)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Delivery distance (km)", fontsize=10, labelpad=3)
    ax.set_ylabel("Order count", fontsize=10, labelpad=3)
    ax.set_xticks(np.arange(0, display_max_km + 1, 1))
    ax.tick_params(axis="both", labelsize=9, length=3, pad=2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for spine in ax.spines.values():
        spine.set_linewidth(0.6)
    if plot_title:
        ax.set_title("Observed Delivery Distance Distribution", fontsize=11)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        if for_latex and not save_path.endswith(".pdf"):
            save_path = save_path.rsplit(".", 1)[0] + ".pdf"
        if plot_title:
            save_path = _add_title_suffix(save_path)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        report_path = save_path.rsplit(".", 1)[0] + "_report.json"
        report = {
            "schema_version": 1,
            "figure_name": "observed_delivery_distance_distribution",
            "figure_path": save_path,
            "core_conclusion": (
                "In the observed Beijing origin--destination sample, "
                f"{coverage_pct:.2f}% of valid positive-distance orders "
                "were no greater than 5 km; this supports the 5-km "
                "candidate-destination radius within this sample."
            ),
            "claim_boundary": (
                "The coverage statistic describes the observed Beijing "
                "sample and is not a universal delivery-distance bound."
            ),
            "sample_metadata": dict(sample_metadata or {}),
            "distance_summary_km": distance_summary or {
                "count": int(plot_distances.size),
                "min": float(np.min(plot_distances)),
                "q1": float(np.quantile(plot_distances, 0.25)),
                "median": float(np.median(plot_distances)),
                "mean": float(np.mean(plot_distances)),
                "q3": float(np.quantile(plot_distances, 0.75)),
                "max": float(np.max(plot_distances)),
            },
            "five_km_coverage": {
                "threshold_km": threshold_km,
                "at_or_below_count": below_threshold_count,
                "above_count": above_threshold_count,
                "total_count": int(plot_distances.size),
                "percentage": float(coverage_pct),
            },
            "histogram": {
                "bin_width_km": bin_width,
                "bins": [
                    {
                        "left_km": float(left),
                        "right_km": float(right),
                        "center_km": float(center),
                        "count": int(count),
                    }
                    for left, right, center, count in zip(
                        bin_edges[:-1],
                        bin_edges[1:],
                        bin_centers,
                        counts,
                    )
                ],
            },
            "axis": {
                "x_min_km": 0.0,
                "x_max_km": display_max_km,
                "y_min_orders": 0,
            },
            "annotation": {
                "threshold_km": threshold_km,
                "text": f"{coverage_pct:.2f}% of orders <= 5 km",
                "placement": "left of the 5-km threshold line",
            },
        }
        write_figure_report(report_path, report)
        print(
            "Observed delivery distance distribution report saved to "
            f"{report_path}"
        )
        print(f"Observed delivery distance distribution saved to {save_path}")
    else:
        plt.show()
    plt.close(fig)
