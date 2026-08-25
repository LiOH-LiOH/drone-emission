"""Standalone audit plots for the Beijing gravity calibration."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


COLORS = {
    "current_parameters": "#6f8fa6",
    "uniform_buildings": "#b5a58a",
    "building_volume_only": "#7fa88c",
    "tanner_distance_only": "#a98aa3",
    "joint_three_parameter": "#c9785d",
}
LABELS = {
    "current_parameters": "Current parameters",
    "uniform_buildings": "Uniform",
    "building_volume_only": "Volume only",
    "tanner_distance_only": "Tanner distance only",
    "joint_three_parameter": "Joint three-parameter",
}
SPLIT_LABELS = {
    "test_unseen_shops_all_29_days": "Unseen shops",
    "test_seen_shops_last_5_days": "Seen shops, last 5 days",
}


def _save(fig, pdf: Path, png: Path) -> None:
    pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    fig.savefig(png, dpi=300, bbox_inches="tight", facecolor="white")


def _plot_data_estimation(output_dir: Path, result: dict, cache_dir: Path) -> None:
    import matplotlib.pyplot as plt

    flow = pd.read_csv(output_dir / "data_flow.csv")
    matches = pd.read_csv(cache_dir / "matched_orders.csv.gz")
    parameters = pd.read_csv(output_dir / "parameter_estimates.csv")
    source_rows = []
    source_rows.extend(
        {"panel": "flow", "item": row.stage, "value": row.count}
        for row in flow.itertuples(index=False)
    )
    bins = np.linspace(0, 50, 26)
    nearest = matches.loc[
        matches["match_method"] == "nearest_within_50m", "match_distance_m"
    ].to_numpy(float)
    counts, edges = np.histogram(nearest, bins=bins)
    source_rows.extend(
        {
            "panel": "match_distance",
            "item": f"{left:g}-{right:g} m",
            "value": int(count),
        }
        for left, right, count in zip(edges[:-1], edges[1:], counts)
    )
    for row in parameters.itertuples(index=False):
        source_rows.append(
            {
                "panel": "parameter",
                "item": row.parameter,
                "value": row.estimate,
                "ci95_low": row.ci95_low,
                "ci95_high": row.ci95_high,
            }
        )
    pd.DataFrame(source_rows).to_csv(
        output_dir / "data_estimation_audit_source.csv", index=False
    )

    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.2))
    retained = int(result["prepared"]["exclusion_flow"]["retained_valid_orders"])
    stages = [
        ("Corrected\norders", result["prepared"]["exclusion_flow"]["raw_corrected_orders"]),
        ("Valid\ncoordinates", result["prepared"]["exclusion_flow"]["raw_corrected_orders"] - result["prepared"]["exclusion_flow"]["excluded_invalid_coordinates"]),
        ("Building\nmatched", result["prepared"]["exclusion_flow"]["orders_matched_within_50m"]),
        ("Final valid\nsample", retained),
    ]
    axes[0].bar(range(len(stages)), [value for _, value in stages], color="#7fa88c")
    axes[0].set_xticks(range(len(stages)), [label for label, _ in stages])
    axes[0].set_ylabel("Orders")
    axes[0].set_title("a  Sample retention", loc="left")
    axes[0].tick_params(axis="x", labelsize=8)
    if len(nearest):
        axes[1].hist(nearest, bins=bins, color="#6f8fa6", edgecolor="white", linewidth=0.4)
    axes[1].set_xlabel("Point-to-building boundary distance (m)")
    axes[1].set_ylabel("Orders")
    axes[1].set_title("b  Nearest-building matches", loc="left")
    y = np.arange(len(parameters))
    axes[2].errorbar(
        parameters["estimate"], y,
        xerr=np.vstack((
            parameters["estimate"] - parameters["ci95_low"],
            parameters["ci95_high"] - parameters["estimate"],
        )),
        fmt="o", color="#c9785d", ecolor="#7f7f7f", capsize=3,
    )
    axes[2].set_yticks(y, [r"$\alpha$", r"$\beta$", r"$\gamma$"])
    axes[2].axvline(0, color="0.8", linewidth=0.8)
    axes[2].set_xlabel("Estimate (95% clustered CI)")
    axes[2].set_title("c  Joint estimates", loc="left")
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="0.9", linewidth=0.6)
    fig.tight_layout()
    try:
        _save(
            fig,
            output_dir / "data_linkage_and_joint_estimation.pdf",
            output_dir / "data_linkage_and_joint_estimation.png",
        )
    finally:
        plt.close(fig)


def _cdf(frame: pd.DataFrame, weight_column: str) -> tuple[np.ndarray, np.ndarray]:
    frame = frame.sort_values("distance_bin_km", kind="mergesort")
    weights = frame[weight_column].to_numpy(float)
    return frame["distance_bin_km"].to_numpy(float), np.cumsum(weights) / weights.sum()


def _plot_prediction(output_dir: Path) -> None:
    import matplotlib.pyplot as plt

    metrics = pd.read_csv(output_dir / "model_metrics.csv")
    distributions = pd.read_csv(output_dir / "distance_distribution_source.csv.gz")
    source = metrics.copy()
    source.to_csv(output_dir / "out_of_sample_prediction_metrics_source.csv", index=False)
    order = list(COLORS)
    fig = plt.figure(figsize=(12.5, 7.2))
    grid = fig.add_gridspec(2, 2, hspace=0.38, wspace=0.28)
    nll_ax = fig.add_subplot(grid[0, 0])
    hit_ax = fig.add_subplot(grid[0, 1])
    cdf_axes = [fig.add_subplot(grid[1, column]) for column in range(2)]
    x = np.arange(2)
    width = 0.15
    for offset, model in enumerate(order):
        subset = metrics.loc[metrics["model"] == model].set_index("split")
        values = [subset.loc[split, "nll_per_order"] for split in SPLIT_LABELS]
        nll_ax.bar(x + (offset - 2) * width, values, width, color=COLORS[model], label=LABELS[model])
    nll_ax.set_xticks(x, SPLIT_LABELS.values())
    nll_ax.set_ylabel("Negative log-likelihood per order")
    nll_ax.set_title("a  Predictive likelihood", loc="left")
    joint = metrics.loc[metrics["model"] == "joint_three_parameter"].set_index("split")
    hit_names = ("top_1", "top_10", "top_1_percent")
    hit_labels = ("Top 1", "Top 10", "Top 1%")
    for offset, split in enumerate(SPLIT_LABELS):
        hit_ax.bar(
            np.arange(3) + (offset - 0.5) * 0.34,
            [joint.loc[split, name] for name in hit_names],
            0.34,
            color=("#6f8fa6", "#c9785d")[offset],
            label=SPLIT_LABELS[split],
        )
    hit_ax.set_xticks(np.arange(3), hit_labels)
    hit_ax.set_ylabel("Observed-building hit rate")
    hit_ax.set_ylim(0, 1)
    hit_ax.set_title("b  Joint-model ranks", loc="left")
    hit_ax.legend(frameon=False, fontsize=8)
    for ax, split in zip(cdf_axes, SPLIT_LABELS):
        subset = distributions.loc[distributions["split"] == split]
        observed = subset.loc[subset["model"] == "joint_three_parameter"]
        distance, cumulative = _cdf(observed, "observed_order_weight")
        ax.step(distance, cumulative, where="post", color="0.2", linewidth=2, label="Observed")
        for model in order:
            model_frame = subset.loc[subset["model"] == model]
            distance, cumulative = _cdf(model_frame, "predicted_order_weight")
            ax.step(distance, cumulative, where="post", color=COLORS[model], linewidth=1.3, label=LABELS[model])
        ax.set_xlabel("Shop-to-building centroid distance (km)")
        ax.set_ylabel("Cumulative probability")
        ax.set_xlim(0, 5)
        ax.set_ylim(0, 1)
        ax.set_title(
            ("c  " if ax is cdf_axes[0] else "d  ") + SPLIT_LABELS[split], loc="left"
        )
    handles, labels = nll_ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, frameon=False, fontsize=8)
    for ax in (nll_ax, hit_ax, *cdf_axes):
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="0.9", linewidth=0.6)
    fig.subplots_adjust(bottom=0.12)
    try:
        _save(
            fig,
            output_dir / "out_of_sample_prediction.pdf",
            output_dir / "out_of_sample_prediction.png",
        )
    finally:
        plt.close(fig)


def _plot_full_order_distance_distribution(output_dir: Path) -> None:
    import matplotlib.pyplot as plt

    source = pd.read_csv(output_dir / "full_order_distance_source.csv.gz")
    observed = source.loc[source["series"] == "observed"]
    predicted = source.loc[source["series"] == "joint_three_parameter"]
    bins = np.arange(0.0, 5.0001, 0.1)
    fig, axes = plt.subplots(2, 1, figsize=(8.8, 7.0), sharex=True)
    axes[0].hist(
        observed["distance_bin_km"], bins=bins,
        weights=observed["order_weight"], histtype="step", linewidth=2.0,
        color="0.2", label="Observed",
    )
    axes[0].hist(
        predicted["distance_bin_km"], bins=bins,
        weights=predicted["order_weight"], histtype="step", linewidth=1.8,
        color=COLORS["joint_three_parameter"], label="Joint gravity model",
    )
    axes[0].set_ylabel("Orders per 100 m bin")
    axes[0].set_title("a  Full-sample delivery-distance distribution", loc="left")
    axes[0].legend(frameon=False)
    for frame, color, label in (
        (observed, "0.2", "Observed"),
        (predicted, COLORS["joint_three_parameter"], "Joint gravity model"),
    ):
        grouped = frame.groupby("distance_bin_km", as_index=False)["order_weight"].sum()
        distance = grouped["distance_bin_km"].to_numpy(float)
        cumulative = np.cumsum(grouped["order_weight"].to_numpy(float))
        cumulative /= cumulative[-1]
        axes[1].step(distance, cumulative, where="post", color=color, linewidth=1.8, label=label)
    axes[1].set_xlabel("Shop-to-building centroid distance (km)")
    axes[1].set_ylabel("Cumulative probability")
    axes[1].set_title("b  Cumulative order distribution", loc="left")
    axes[1].set_xlim(0, 5)
    axes[1].set_ylim(0, 1)
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", color="0.9", linewidth=0.6)
    fig.tight_layout()
    try:
        _save(
            fig,
            output_dir / "beijing_order_distance_distribution.pdf",
            output_dir / "beijing_order_distance_distribution.png",
        )
    finally:
        plt.close(fig)


def _plot_endpoint_grid_heatmap(output_dir: Path, grid_size_m: float = 1000.0) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from matplotlib.ticker import FuncFormatter

    source = pd.read_csv(output_dir / "endpoint_building_order_source.csv.gz")
    source = source.dropna(subset=["centroid_x", "centroid_y"])
    if source.empty:
        raise ValueError("Endpoint grid source contains no finite building centroids")
    x_min = np.floor(source["centroid_x"].min() / grid_size_m) * grid_size_m
    y_min = np.floor(source["centroid_y"].min() / grid_size_m) * grid_size_m
    source["grid_x"] = np.floor((source["centroid_x"] - x_min) / grid_size_m).astype(int)
    source["grid_y"] = np.floor((source["centroid_y"] - y_min) / grid_size_m).astype(int)
    grid = source.groupby(["grid_y", "grid_x"], as_index=False)[
        ["observed_orders", "expected_orders"]
    ].sum()
    max_x = int(grid["grid_x"].max()) + 1
    max_y = int(grid["grid_y"].max()) + 1
    observed = np.zeros((max_y, max_x), dtype=float)
    expected = np.zeros_like(observed)
    observed[grid["grid_y"], grid["grid_x"]] = grid["observed_orders"]
    expected[grid["grid_y"], grid["grid_x"]] = grid["expected_orders"]
    plot_source = grid.copy()
    plot_source["x_min_m"] = x_min + plot_source["grid_x"] * grid_size_m
    plot_source["y_min_m"] = y_min + plot_source["grid_y"] * grid_size_m
    plot_source.to_csv(output_dir / "beijing_endpoint_grid_source.csv", index=False)

    transformed = np.log1p(np.maximum(np.stack([observed, expected]), 0.0))
    norm = Normalize(vmin=0.0, vmax=max(1.0, float(transformed.max())))
    extent = (x_min, x_min + max_x * grid_size_m, y_min, y_min + max_y * grid_size_m)
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 5.2), sharex=True, sharey=True)
    images = []
    for axis, values, title in (
        (axes[0], observed, "a  Observed endpoint orders"),
        (axes[1], expected, "b  Joint gravity-model endpoint orders"),
    ):
        image = axis.imshow(np.log1p(values), origin="lower", extent=extent,
                            cmap="magma", norm=norm, interpolation="none", aspect="equal")
        images.append(image)
        axis.set_title(title, loc="left")
        axis.set_xlabel("Easting (m, EPSG:32650)")
        axis.text(0.03, 0.04, f"Total: {values.sum():,.1f} orders",
                  transform=axis.transAxes, color="white", fontsize=9,
                  bbox={"facecolor": "black", "alpha": 0.35, "pad": 3, "edgecolor": "none"})
    axes[0].set_ylabel("Northing (m, EPSG:32650)")
    colorbar = fig.colorbar(images[1], ax=axes, pad=0.02, fraction=0.035)
    colorbar.set_label("Orders per 1 km² (log1p scale)")
    colorbar.ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{np.expm1(value):g}"))
    fig.suptitle("Beijing endpoint order concentration on aligned 1 km grids", y=0.99)
    fig.tight_layout()
    try:
        _save(
            fig,
            output_dir / "beijing_endpoint_grid_heatmap.pdf",
            output_dir / "beijing_endpoint_grid_heatmap.png",
        )
    finally:
        plt.close(fig)


def _plot_spatial_fit(output_dir: Path, result: dict) -> None:
    """Plot intuitive full-sample merchant and building predictive fit."""
    import matplotlib.pyplot as plt

    merchants = pd.read_csv(output_dir / "merchant_spatial_fit_source.csv.gz")
    merchants = merchants.loc[merchants["scope"] == "full_valid_sample"]
    buildings = pd.read_csv(output_dir / "building_order_fit_source.csv.gz")
    merchant_summary = next(
        row for row in result["spatial_fit"]["merchant_metrics"]
        if row["scope"] == "full_valid_sample"
    )
    building_summary = result["spatial_fit"]["building_metrics"]

    fig, axes = plt.subplots(2, 2, figsize=(11.4, 9.0))
    mean_ax, wasserstein_ax, raw_ax, log_ax = axes.flat
    observed_mean = merchants["observed_mean_distance_km"].to_numpy(float)
    predicted_mean = merchants["predicted_mean_distance_km"].to_numpy(float)
    mean_hex = mean_ax.hexbin(
        observed_mean,
        predicted_mean,
        gridsize=50,
        mincnt=1,
        bins="log",
        cmap="Blues",
    )
    mean_limit = float(max(observed_mean.max(), predicted_mean.max()))
    mean_ax.plot([0, mean_limit], [0, mean_limit], color="#c9785d", linewidth=1.3)
    mean_ax.set_xlim(0, mean_limit)
    mean_ax.set_ylim(0, mean_limit)
    mean_ax.set_aspect("equal", adjustable="box")
    mean_ax.set_xlabel("Observed merchant mean distance (km)")
    mean_ax.set_ylabel("Predicted merchant mean distance (km)")
    mean_ax.set_title("a  Merchant mean delivery distance", loc="left")
    mean_ax.text(
        0.04,
        0.95,
        rf"Predictive $R^2$ = {merchant_summary['merchant_mean_distance_r2_equal_weighted']:.3f}",
        transform=mean_ax.transAxes,
        va="top",
    )
    fig.colorbar(mean_hex, ax=mean_ax, label="Merchants per hexagon")

    wasserstein = merchants["wasserstein_m"].to_numpy(float)
    wasserstein_ax.hist(
        wasserstein,
        bins=40,
        color="#7fa88c",
        edgecolor="white",
        linewidth=0.35,
    )
    for value, label, color in (
        (merchant_summary["wasserstein_m_median"], "Median", "#c9785d"),
        (merchant_summary["wasserstein_m_p90"], "P90", "#6f8fa6"),
    ):
        wasserstein_ax.axvline(
            value,
            color=color,
            linewidth=1.5,
            label=f"{label}: {value:.0f} m",
        )
    wasserstein_ax.set_xlabel("Merchant distance-distribution error, W1 (m)")
    wasserstein_ax.set_ylabel("Merchants")
    wasserstein_ax.set_title("b  Distributional error by merchant", loc="left")
    wasserstein_ax.legend(frameon=False)

    observed_orders = buildings["observed_orders"].to_numpy(float)
    expected_orders = buildings["expected_orders"].to_numpy(float)
    raw_hex = raw_ax.hexbin(
        observed_orders,
        expected_orders,
        gridsize=65,
        mincnt=1,
        bins="log",
        cmap="Blues",
    )
    raw_limit = float(max(observed_orders.max(), expected_orders.max()))
    raw_ax.plot([0, raw_limit], [0, raw_limit], color="#c9785d", linewidth=1.3)
    raw_ax.set_xlim(0, raw_limit)
    raw_ax.set_ylim(0, raw_limit)
    raw_ax.set_aspect("equal", adjustable="box")
    raw_ax.set_xlabel("Observed building orders")
    raw_ax.set_ylabel("Expected building orders")
    raw_ax.set_title("c  Building order counts (raw)", loc="left")
    raw_ax.text(
        0.04,
        0.95,
        rf"Predictive $R^2$ = {building_summary['building_order_count_r2_raw']:.3f}",
        transform=raw_ax.transAxes,
        va="top",
    )
    fig.colorbar(raw_hex, ax=raw_ax, label="Buildings per hexagon")

    observed_log = np.log1p(observed_orders)
    expected_log = np.log1p(expected_orders)
    log_hex = log_ax.hexbin(
        observed_log,
        expected_log,
        gridsize=65,
        mincnt=1,
        bins="log",
        cmap="Blues",
    )
    log_limit = float(max(observed_log.max(), expected_log.max()))
    log_ax.plot([0, log_limit], [0, log_limit], color="#c9785d", linewidth=1.3)
    log_ax.set_xlim(0, log_limit)
    log_ax.set_ylim(0, log_limit)
    log_ax.set_aspect("equal", adjustable="box")
    log_ax.set_xlabel("log(1 + observed building orders)")
    log_ax.set_ylabel("log(1 + expected building orders)")
    log_ax.set_title("d  Building order counts (log scale)", loc="left")
    log_ax.text(
        0.04,
        0.95,
        rf"Predictive $R^2$ = {building_summary['building_order_count_r2_log1p']:.3f}",
        transform=log_ax.transAxes,
        va="top",
    )
    fig.colorbar(log_hex, ax=log_ax, label="Buildings per hexagon")

    for ax in axes.flat:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(color="0.92", linewidth=0.6)
    fig.suptitle("Full-sample spatial fit of the joint gravity model", y=0.995)
    fig.tight_layout()
    try:
        _save(
            fig,
            output_dir / "merchant_and_building_spatial_fit.pdf",
            output_dir / "merchant_and_building_spatial_fit.png",
        )
    finally:
        plt.close(fig)


def render_spatial_fit_audit(output_dir: Path, result: dict) -> None:
    """Render only the full-sample merchant/building spatial-fit audit."""
    _plot_spatial_fit(Path(output_dir), result)


def render_boxcox_challenger_audit(output_dir: Path, result: dict) -> None:
    """Render the standalone Box-Cox selection and test-distribution audit."""
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)
    grid = pd.read_csv(output_dir / "boxcox_cv_grid_metrics.csv")
    test_metrics = pd.read_csv(output_dir / "boxcox_test_metrics.csv")
    test_distributions = pd.read_csv(
        output_dir / "boxcox_test_distance_source.csv.gz"
    )
    full_metrics = pd.read_csv(output_dir / "boxcox_full_sample_metrics.csv")
    full_distributions = pd.read_csv(
        output_dir / "boxcox_full_sample_distance_source.csv.gz"
    )
    selected = float(result["selection"]["selected_lambda"])
    fig, axes = plt.subplots(2, 2, figsize=(10.4, 7.4))
    axes = axes.ravel()

    axes[0].plot(
        grid["boxcox_lambda"],
        grid["wasserstein_1_km"] * 1000.0,
        marker="o",
        color="#6f8fa6",
        linewidth=1.5,
    )
    selected_row = grid.loc[grid["selected"]].iloc[0]
    axes[0].scatter(
        [selected],
        [selected_row["wasserstein_1_km"] * 1000.0],
        color="#c9785d",
        s=48,
        zorder=3,
    )
    axes[0].axvline(selected, color="#c9785d", linewidth=1.0, linestyle="--")
    axes[0].set_xlabel(r"Box-Cox shape $\lambda$")
    axes[0].set_ylabel("Pooled CV Wasserstein distance (m)")
    axes[0].set_title("a  Grouped cross-validation", loc="left")
    axes[0].text(
        0.04,
        0.95,
        rf"Selected $\lambda$ = {selected:g}",
        transform=axes[0].transAxes,
        va="top",
    )

    scope_labels = {
        "full_valid_sample": "All valid linked orders",
        "test_unseen_shops_all_29_days": "Unseen shops",
        "test_seen_shops_last_5_days": "Seen shops, last 5 days",
    }
    test_scopes = (
        "test_unseen_shops_all_29_days",
        "test_seen_shops_last_5_days",
    )
    colors = {"tanner": "#7f8c8d", "boxcox": "#c9785d"}
    labels = {"tanner": "Tanner", "boxcox": "Box-Cox"}
    panels = (
        (
            axes[1],
            "full_valid_sample",
            full_distributions,
            full_metrics,
            "scope",
        ),
        *(
            (axes[index], split, test_distributions, test_metrics, "split")
            for index, split in enumerate(test_scopes, start=2)
        ),
    )
    for panel, (axis, scope, distributions, metrics, scope_column) in enumerate(
        panels, start=1
    ):
        subset = distributions.loc[distributions[scope_column] == scope]
        observed = subset.loc[subset["model"] == "boxcox"]
        distance, cumulative = _cdf(observed, "observed_order_weight")
        axis.step(
            distance,
            cumulative,
            where="post",
            color="0.15",
            linewidth=2.0,
            label="Observed",
        )
        for model in ("tanner", "boxcox"):
            model_frame = subset.loc[subset["model"] == model]
            distance, cumulative = _cdf(model_frame, "predicted_order_weight")
            axis.step(
                distance,
                cumulative,
                where="post",
                color=colors[model],
                linewidth=1.5,
                label=labels[model],
            )
        boxcox_metric = metrics.loc[
            (metrics[scope_column] == scope) & (metrics["model"] == "boxcox")
        ].iloc[0]
        axis.text(
            0.96,
            0.05,
            f"Box-Cox W1: {boxcox_metric['wasserstein_1_km'] * 1000:.0f} m\n"
            f"Mean: {boxcox_metric['predicted_mean_km']:.2f} km\n"
            f"P95: {boxcox_metric['predicted_p95_km']:.2f} km",
            transform=axis.transAxes,
            ha="right",
            va="bottom",
            fontsize=8,
        )
        axis.set_xlim(0, 5)
        axis.set_ylim(0, 1)
        axis.set_xlabel("Shop-to-building centroid distance (km)")
        axis.set_ylabel("Cumulative probability")
        axis.set_title(
            f"{chr(ord('a') + panel)}  {scope_labels[scope]}", loc="left"
        )
    axes[3].legend(frameon=False, loc="lower right", bbox_to_anchor=(1.0, 0.22))
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", color="0.9", linewidth=0.6)
    eligible = result["acceptance"]["eligible_for_acceptance"]
    fig.suptitle(
        "Box-Cox gravity challenger: "
        + ("acceptance gate passed" if eligible else "acceptance gate not passed"),
        y=1.01,
    )
    fig.tight_layout()
    try:
        _save(
            fig,
            output_dir / "boxcox_gravity_challenger.pdf",
            output_dir / "boxcox_gravity_challenger.png",
        )
    finally:
        plt.close(fig)


def render_calibration_audits(output_dir: Path, result: dict, cache_dir: Path) -> None:
    """Render calibration audit figures and their source tables."""
    output_dir, cache_dir = Path(output_dir), Path(cache_dir)
    _plot_data_estimation(output_dir, result, cache_dir)
    _plot_prediction(output_dir)
    _plot_full_order_distance_distribution(output_dir)
    _plot_endpoint_grid_heatmap(output_dir)
    render_spatial_fit_audit(output_dir, result)
