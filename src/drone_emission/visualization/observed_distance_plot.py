"""Formal Beijing delivery-distance distribution plotting."""

import os

import matplotlib.pyplot as plt
import numpy as np

from drone_emission.visualization._utils import (
    _add_title_suffix,
    write_figure_report,
)


def plot_beijing_order_distance_distribution(
    observed_histogram,
    model_distribution,
    save_path=None,
    figsize=(180.0 / 25.4, 52.0 / 25.4),
    for_latex=False,
    plot_title=False,
    calibration_run_hash=None,
):
    """Plot observed Beijing orders and the accepted gravity-model fit."""
    observed_required = {"center_km", "count"}
    model_required = {"series", "distance_bin_km", "order_weight"}
    if not observed_required.issubset(observed_histogram.columns):
        missing = sorted(observed_required - set(observed_histogram.columns))
        raise ValueError(
            "Observed distance histogram is missing: " + ", ".join(missing)
        )
    if not model_required.issubset(model_distribution.columns):
        missing = sorted(model_required - set(model_distribution.columns))
        raise ValueError(
            "Gravity-model distance source is missing: " + ", ".join(missing)
        )

    observed_distance = observed_histogram["center_km"].to_numpy(dtype=float)
    observed_weight = observed_histogram["count"].to_numpy(dtype=float)
    observed_valid = (
        np.isfinite(observed_distance)
        & np.isfinite(observed_weight)
        & (observed_distance > 0)
        & (observed_weight >= 0)
    )
    observed_distance = observed_distance[observed_valid]
    observed_weight = observed_weight[observed_valid]
    if observed_weight.sum() <= 0:
        raise ValueError("Observed distance histogram contains no positive order count")

    predicted = model_distribution.loc[
        model_distribution["series"].eq("joint_three_parameter")
    ]
    model_distance = predicted["distance_bin_km"].to_numpy(dtype=float)
    model_weight = predicted["order_weight"].to_numpy(dtype=float)
    model_valid = (
        np.isfinite(model_distance)
        & np.isfinite(model_weight)
        & (model_distance >= 0)
        & (model_distance <= 5.0)
        & (model_weight >= 0)
    )
    model_distance = model_distance[model_valid]
    model_weight = model_weight[model_valid]
    if model_weight.sum() <= 0:
        raise ValueError("Accepted gravity-model source contains no prediction weights")

    threshold_km = 5.0
    display_max_km = 7.0
    bin_width = 0.2
    display_edges = np.arange(0.0, display_max_km + bin_width / 2, bin_width)
    model_edges = np.arange(0.0, threshold_km + bin_width / 2, bin_width)
    display_centers = (display_edges[:-1] + display_edges[1:]) / 2
    model_centers = (model_edges[:-1] + model_edges[1:]) / 2
    observed_counts, _ = np.histogram(
        observed_distance,
        bins=display_edges,
        weights=observed_weight,
    )
    predicted_counts, _ = np.histogram(
        model_distance,
        bins=model_edges,
        weights=model_weight,
    )

    total_count = int(round(observed_weight.sum()))
    within_five_count = int(
        round(observed_weight[observed_distance <= threshold_km].sum())
    )
    above_five_count = total_count - within_five_count
    coverage_pct = 100.0 * within_five_count / total_count
    model_scale = within_five_count / predicted_counts.sum()
    predicted_counts = predicted_counts * model_scale

    fig = plt.figure(figsize=figsize, facecolor="white")
    ax = fig.add_axes([0.10, 0.22, 0.88, 0.72])
    bar_width = 0.085
    ax.bar(
        display_centers - bar_width / 2,
        observed_counts,
        width=bar_width,
        color="0.2",
        label="Observed",
    )
    ax.bar(
        model_centers + bar_width / 2,
        predicted_counts,
        width=bar_width,
        color="#c9785d",
        label="Joint gravity model",
    )
    ax.axvline(threshold_km, color="0.35", linestyle="--", linewidth=0.9)
    ax.text(
        threshold_km - 0.12,
        0.96,
        f"{coverage_pct:.2f}% of observed orders\n$\\leq$ 5 km",
        transform=ax.get_xaxis_transform(),
        color="0.25",
        fontsize=7.5,
        ha="right",
        va="top",
    )
    ax.set_xlim(0, display_max_km)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Delivery distance (km)", fontsize=8.5, labelpad=2)
    ax.set_ylabel("Orders per 0.2 km bin", fontsize=8.5, labelpad=3)
    ax.set_xticks(np.arange(0, display_max_km + 1, 1))
    ax.tick_params(axis="both", labelsize=7.5, length=2.5, pad=1.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="0.9", linewidth=0.5)
    ax.legend(loc="upper right", frameon=False, fontsize=7.5, ncol=1)
    if plot_title:
        ax.set_title("Beijing delivery-distance distribution", fontsize=9, loc="left")

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        if for_latex and not save_path.endswith(".pdf"):
            save_path = save_path.rsplit(".", 1)[0] + ".pdf"
        if plot_title:
            save_path = _add_title_suffix(save_path)
        fig.savefig(save_path, dpi=300, facecolor="white")
        report_path = save_path.rsplit(".", 1)[0] + "_report.json"
        report = {
            "schema_version": 1,
            "figure_name": "beijing_order_distance_distribution",
            "figure_path": save_path,
            "core_conclusion": (
                "The accepted joint gravity model reproduces the observed "
                "Beijing delivery-distance distribution within its 5-km "
                f"candidate radius; {coverage_pct:.2f}% of all valid observed "
                "orders were no greater than 5 km."
            ),
            "claim_boundary": (
                "The 5-km coverage statistic uses the full observed Beijing "
                "histogram, whereas the model distribution is conditional on "
                "the calibrated 5-km candidate set."
            ),
            "calibration": {
                "run_hash": calibration_run_hash,
                "model": "joint_three_parameter",
                "prediction_domain_km": [0.0, threshold_km],
                "prediction_weight_before_scaling": float(model_weight.sum()),
                "prediction_scale_factor": float(model_scale),
                "prediction_weight_after_scaling": float(predicted_counts.sum()),
            },
            "observed_sample": {
                "total_count": total_count,
                "at_or_below_five_km_count": within_five_count,
                "above_five_km_count": above_five_count,
                "at_or_below_five_km_percentage": float(coverage_pct),
            },
            "histogram": {
                "bin_width_km": bin_width,
                "observed_display_range_km": [0.0, display_max_km],
                "model_display_range_km": [0.0, threshold_km],
                "observed_bin_count": int(len(observed_counts)),
                "model_bin_count": int(len(predicted_counts)),
            },
            "layout": {
                "native_size_mm": [
                    float(figsize[0] * 25.4),
                    float(figsize[1] * 25.4),
                ],
                "single_panel": True,
                "panel_label": None,
                "tight_crop": False,
            },
            "axis": {
                "x_min_km": 0.0,
                "x_max_km": display_max_km,
                "y_min_orders": 0.0,
                "ylabel": "Orders per 0.2 km bin",
            },
            "annotation": {
                "threshold_km": threshold_km,
                "text": f"{coverage_pct:.2f}% of observed orders <= 5 km",
            },
        }
        write_figure_report(report_path, report)
        print(f"Beijing order distance distribution report saved to {report_path}")
        print(f"Beijing order distance distribution saved to {save_path}")
    else:
        plt.show()
    plt.close(fig)
