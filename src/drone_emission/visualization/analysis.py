"""Visualization functions for cross-region correlation and emission analysis."""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from visualization._utils import _add_title_suffix

# Import from constants module (no circular dependency)
from analysis.constants import VARIABLE_LABELS


def plot_correlation_heatmap(
    corr_df: pd.DataFrame,
    save_path: str = None,
    figsize: tuple = (10, 8),
    for_latex: bool = False,
    plot_title: bool = False,
    title: str = "Feature-Result Correlations",
):
    """Plot correlation heatmap.

    Args:
        corr_df: Correlation DataFrame (features x results).
        save_path: Path to save figure. Shows plot if None.
        figsize: Figure size tuple.
        for_latex: If True, save as PDF.
        plot_title: If True, show title.
        title: Plot title.
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Get feature labels
    row_labels = [VARIABLE_LABELS.get(f, f) for f in corr_df.index]
    col_labels = [VARIABLE_LABELS.get(c, c) for c in corr_df.columns]

    # Create heatmap
    im = ax.imshow(corr_df.values, cmap="RdBu_r", aspect="auto", vmin=-1, vmax=1)

    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax, shrink=0.8)
    cbar.ax.set_ylabel(
        "Correlation Coefficient", rotation=-90, va="bottom", fontsize=11
    )

    # Set ticks and labels
    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_xticklabels(col_labels, fontsize=10)
    ax.set_yticklabels(row_labels, fontsize=10)

    # Rotate x-axis labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Add correlation values as text
    for i in range(len(row_labels)):
        for j in range(len(col_labels)):
            val = corr_df.iloc[i, j]
            if not np.isnan(val):
                text_color = "white" if abs(val) > 0.5 else "black"
                ax.text(
                    j,
                    i,
                    f"{val:.2f}",
                    ha="center",
                    va="center",
                    color=text_color,
                    fontsize=9,
                )

    if plot_title:
        ax.set_title(title, fontsize=14, fontweight="bold", pad=15)

    plt.tight_layout()

    if save_path:
        if for_latex and not save_path.endswith(".pdf"):
            save_path = save_path.rsplit(".", 1)[0] + ".pdf"
        if plot_title:
            save_path = _add_title_suffix(save_path)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Heatmap saved to {save_path}")
    else:
        plt.show()

    plt.close(fig)


def plot_feature_scatter(
    df: pd.DataFrame,
    feature_var: str,
    result_var: str,
    save_path: str = None,
    figsize: tuple = (8, 6),
    for_latex: bool = False,
    plot_title: bool = False,
):
    """Plot scatter plot of feature vs result with regression line and 95% CI.

    Scales to large datasets: small markers, no per-point labels,
    annotates only top/bottom 5 outliers by residual.

    Args:
        df: DataFrame with region statistics.
        feature_var: Feature variable name.
        result_var: Result variable name.
        save_path: Path to save figure. Shows plot if None.
        figsize: Figure size tuple.
        for_latex: If True, save as PDF.
        plot_title: If True, show title.
    """
    # Filter valid data
    valid_mask = df[[feature_var, result_var]].notna().all(axis=1)
    x = df.loc[valid_mask, feature_var].values
    y = df.loc[valid_mask, result_var].values
    labels = df.loc[valid_mask, "region_name"].values

    if len(x) < 2:
        print(f"Not enough data points for {feature_var} vs {result_var}")
        return

    fig, ax = plt.subplots(figsize=figsize)

    # Scatter plot — smaller markers for large datasets
    marker_size = 100 if len(x) <= 20 else 30
    ax.scatter(
        x, y, s=marker_size, alpha=0.6, edgecolors="darkblue", linewidths=0.8, zorder=10
    )

    # Add regression line with 95% CI band
    if len(x) >= 3:
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        x_line = np.linspace(x.min(), x.max(), 100)
        y_line = slope * x_line + intercept
        ax.plot(
            x_line,
            y_line,
            "r--",
            alpha=0.8,
            linewidth=2,
            label=f"r = {r_value:.3f}, p = {p_value:.1e}",
        )

        # 95% confidence interval
        n = len(x)
        x_mean = np.mean(x)
        se = np.sqrt(
            np.sum((y - (slope * x + intercept)) ** 2) / (n - 2)
        )
        t_val = stats.t.ppf(0.975, n - 2)
        ci = t_val * se * np.sqrt(
            1.0 / n + (x_line - x_mean) ** 2 / np.sum((x - x_mean) ** 2)
        )
        ax.fill_between(
            x_line, y_line - ci, y_line + ci, alpha=0.15, color="red", zorder=5
        )

        ax.legend(loc="best", fontsize=10)

        # Annotate top/bottom 5 outliers by residual (only when many points)
        if len(x) > 20:
            residuals = y - (slope * x + intercept)
            n_outliers = min(5, len(x) // 10)
            if n_outliers > 0:
                outlier_idx = np.argsort(np.abs(residuals))[-n_outliers:]
                for idx in outlier_idx:
                    ax.annotate(
                        labels[idx],
                        (x[idx], y[idx]),
                        xytext=(5, 5),
                        textcoords="offset points",
                        fontsize=7,
                        alpha=0.7,
                    )
    elif len(x) == 2:
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        x_line = np.linspace(x.min(), x.max(), 100)
        y_line = slope * x_line + intercept
        ax.plot(
            x_line, y_line, "r--", alpha=0.8, linewidth=2, label=f"r = {r_value:.3f}"
        )
        ax.legend(loc="best", fontsize=10)

    # Add per-point labels only for small datasets
    if len(x) <= 20:
        for i, label in enumerate(labels):
            ax.annotate(
                label,
                (x[i], y[i]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=9,
                alpha=0.8,
            )

    # Labels
    ax.set_xlabel(VARIABLE_LABELS.get(feature_var, feature_var), fontsize=12)
    ax.set_ylabel(VARIABLE_LABELS.get(result_var, result_var), fontsize=12)

    if plot_title:
        title = f"{VARIABLE_LABELS.get(feature_var, feature_var)} vs {VARIABLE_LABELS.get(result_var, result_var)}"
        ax.set_title(title, fontsize=12, fontweight="bold")

    ax.grid(True, alpha=0.3, linestyle="--")
    plt.tight_layout()

    if save_path:
        if for_latex and not save_path.endswith(".pdf"):
            save_path = save_path.rsplit(".", 1)[0] + ".pdf"
        if plot_title:
            save_path = _add_title_suffix(save_path)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Scatter plot saved to {save_path}")
    else:
        plt.show()

    plt.close(fig)


def plot_top_correlations_bar(
    corr_df: pd.DataFrame,
    result_var: str,
    top_n: int = 5,
    save_path: str = None,
    figsize: tuple = (10, 6),
    for_latex: bool = False,
    plot_title: bool = False,
):
    """Plot bar chart of top correlated features for a result variable.

    Args:
        corr_df: Correlation DataFrame (features x results).
        result_var: Result variable to analyze.
        top_n: Number of top correlations to show.
        save_path: Path to save figure.
        figsize: Figure size tuple.
        for_latex: If True, save as PDF.
        plot_title: If True, show title.
    """
    if result_var not in corr_df.columns:
        print(f"Result variable '{result_var}' not in correlation DataFrame")
        return

    # Get correlations for this result, sorted by absolute value
    correlations = corr_df[result_var].dropna()
    sorted_corr = correlations.reindex(
        correlations.abs().sort_values(ascending=False).index
    )
    top_corr = sorted_corr.head(top_n)

    fig, ax = plt.subplots(figsize=figsize)

    # Create bar colors based on positive/negative
    colors = ["#2ecc71" if v > 0 else "#e74c3c" for v in top_corr.values]

    # Get labels
    feature_labels = [VARIABLE_LABELS.get(f, f) for f in top_corr.index]

    # Bar plot
    bars = ax.barh(range(len(top_corr)), top_corr.values, color=colors, alpha=0.8)

    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, top_corr.values)):
        x_pos = val + 0.02 if val > 0 else val - 0.02
        ha = "left" if val > 0 else "right"
        ax.text(x_pos, i, f"{val:.3f}", va="center", ha=ha, fontsize=10)

    ax.set_yticks(range(len(top_corr)))
    ax.set_yticklabels(feature_labels, fontsize=10)
    ax.set_xlabel("Correlation Coefficient", fontsize=12)
    ax.set_xlim(-1.1, 1.1)
    ax.axvline(x=0, color="gray", linestyle="-", linewidth=0.8)

    if plot_title:
        result_label = VARIABLE_LABELS.get(result_var, result_var)
        ax.set_title(
            f"Top Correlations with {result_label}", fontsize=12, fontweight="bold"
        )

    ax.grid(True, alpha=0.3, linestyle="--", axis="x")
    plt.tight_layout()

    if save_path:
        if for_latex and not save_path.endswith(".pdf"):
            save_path = save_path.rsplit(".", 1)[0] + ".pdf"
        if plot_title:
            save_path = _add_title_suffix(save_path)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Bar chart saved to {save_path}")
    else:
        plt.show()

    plt.close(fig)


def plot_annual_emissions_comparison(
    df: pd.DataFrame = None,
    csv_path: str = None,
    save_path: str = None,
    figsize: tuple = (14, 6),
    for_latex: bool = False,
    plot_title: bool = False,
    top_n: int = None,
):
    """Plot grouped bar chart comparing annual emissions across regions.

    For large datasets, requires top_n to show only top-N and bottom-N cities
    by emission reduction percentage.

    Args:
        df: Pre-loaded emissions DataFrame. If None, loads from csv_path.
        csv_path: Path to all_region_stats.csv (used if df is None).
        save_path: Path to save figure.
        figsize: Figure size tuple.
        for_latex: If True, save as PDF.
        plot_title: If True, show title.
        top_n: Show only top-N and bottom-N cities by reduction. Required if >20 cities.
    """
    if df is None:
        from analysis.correlation import load_annual_emissions
        df = load_annual_emissions(csv_path)

    if df.empty:
        print("No emission data available")
        return

    # Guard: skip if too many cities without top_n
    if len(df) > 20 and top_n is None:
        print(
            f"Warning: {len(df)} cities too many for bar chart. "
            f"Use top_n parameter to show subset. Skipping."
        )
        return

    # Filter to top/bottom N if specified
    if top_n is not None and len(df) > top_n * 2:
        df_sorted = df.sort_values("reduction_pct", ascending=False)
        top = df_sorted.head(top_n)
        bottom = df_sorted.tail(top_n)
        df = pd.concat([top, bottom]).drop_duplicates()
        # Re-sort for display
        df = df.sort_values("reduction_pct", ascending=False)

    # Check if using actual data
    has_drone_data = (
        "all_drone_tonnes" in df.columns and df["all_drone_tonnes"].notna().any()
    )

    fig, ax = plt.subplots(figsize=figsize)

    x = np.arange(len(df))

    if has_drone_data:
        # Three bars: E-bike, Drone, Optimal Mix
        width = 0.25
        ebike_color = "#e74c3c"
        drone_color = "#3498db"
        mix_color = "#2ecc71"

        ax.bar(
            x - width,
            df["all_ebike_tonnes"],
            width,
            label="All E-bike",
            color=ebike_color,
            alpha=0.8,
        )
        ax.bar(
            x,
            df["all_drone_tonnes"],
            width,
            label="All Drone",
            color=drone_color,
            alpha=0.8,
        )
        bars3 = ax.bar(
            x + width,
            df["optimal_mix_tonnes"],
            width,
            label="Optimal Mix",
            color=mix_color,
            alpha=0.8,
        )

        # Add reduction percentage labels on optimal mix bars
        for bar, pct in zip(bars3, df["reduction_pct"]):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"-{pct:.1f}%",
                ha="center",
                va="bottom",
                fontsize=7,
            )

        ylabel = "Annual Lifecycle Emissions (tonnes CO₂e)"
    else:
        # Two bars: E-bike vs Optimal Mix (fallback)
        width = 0.35

        ax.bar(
            x - width / 2,
            df["all_ebike_tonnes"],
            width,
            label="All E-bike",
            color="#e74c3c",
            alpha=0.8,
        )
        bars2 = ax.bar(
            x + width / 2,
            df["optimal_mix_tonnes"],
            width,
            label="Optimal Mix",
            color="#2ecc71",
            alpha=0.8,
        )

        for bar, pct in zip(bars2, df["reduction_pct"]):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1,
                f"-{pct:.1f}%",
                ha="center",
                va="bottom",
                fontsize=8,
                fontweight="bold",
            )

        ylabel = "Annual Lifecycle Emissions (tonnes CO₂e)"

    ax.set_xlabel("Region", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(df["region_name"], fontsize=8, rotation=45, ha="right")
    ax.legend(loc="upper right", fontsize=10)

    if plot_title:
        ax.set_title(
            "Annual Food Delivery Lifecycle Emissions by Scenario",
            fontsize=14,
            fontweight="bold",
        )

    ax.grid(True, alpha=0.3, linestyle="--", axis="y")
    ax.set_ylim(0, df["all_ebike_tonnes"].max() * 1.25)

    plt.tight_layout()

    if save_path:
        if for_latex and not save_path.endswith(".pdf"):
            save_path = save_path.rsplit(".", 1)[0] + ".pdf"
        if plot_title:
            save_path = _add_title_suffix(save_path)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Annual emissions comparison saved to {save_path}")
    else:
        plt.show()

    plt.close(fig)

    return df


def plot_emission_distribution(
    df: pd.DataFrame,
    save_path: str = None,
    figsize: tuple = (14, 6),
    for_latex: bool = False,
    plot_title: bool = False,
):
    """Plot distribution of emission reduction and optimal drone ratio across all cities.

    Two subplots: histogram+KDE of max_lifecycle_reduction_pct, and histogram of
    optimal_drone_ratio_pct. Annotated with n, mean, median, std.

    Args:
        df: DataFrame with region statistics (from load_region_stats).
        save_path: Path to save figure.
        figsize: Figure size tuple.
        for_latex: If True, save as PDF.
        plot_title: If True, show title.
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Subplot 1: max_lifecycle_reduction_pct
    ax1 = axes[0]
    col1 = "max_lifecycle_reduction_pct"
    if col1 in df.columns:
        data1 = df[col1].dropna()
        ax1.hist(data1, bins=30, alpha=0.7, color="#3498db", edgecolor="white", density=True)
        # KDE
        if len(data1) > 5:
            from scipy.stats import gaussian_kde
            kde = gaussian_kde(data1)
            x_kde = np.linspace(data1.min(), data1.max(), 200)
            ax1.plot(x_kde, kde(x_kde), color="darkblue", linewidth=2)
        # Vertical lines
        ax1.axvline(data1.mean(), color="red", linestyle="--", linewidth=1.5, label=f"Mean: {data1.mean():.1f}%")
        ax1.axvline(data1.median(), color="green", linestyle="--", linewidth=1.5, label=f"Median: {data1.median():.1f}%")
        ax1.legend(fontsize=9)
        # Text annotation
        ax1.text(
            0.97, 0.95,
            f"n={len(data1)}\nstd={data1.std():.1f}%",
            transform=ax1.transAxes, ha="right", va="top", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.5),
        )
    ax1.set_xlabel(VARIABLE_LABELS.get(col1, col1), fontsize=11)
    ax1.set_ylabel("Density", fontsize=11)
    ax1.grid(True, alpha=0.3, linestyle="--")

    # Subplot 2: optimal_drone_ratio_pct
    ax2 = axes[1]
    col2 = "optimal_drone_ratio_pct"
    if col2 in df.columns:
        data2 = df[col2].dropna()
        ax2.hist(data2, bins=30, alpha=0.7, color="#2ecc71", edgecolor="white", density=True)
        if len(data2) > 5:
            from scipy.stats import gaussian_kde
            kde2 = gaussian_kde(data2)
            x_kde2 = np.linspace(data2.min(), data2.max(), 200)
            ax2.plot(x_kde2, kde2(x_kde2), color="darkgreen", linewidth=2)
        ax2.axvline(data2.mean(), color="red", linestyle="--", linewidth=1.5, label=f"Mean: {data2.mean():.1f}%")
        ax2.axvline(data2.median(), color="blue", linestyle="--", linewidth=1.5, label=f"Median: {data2.median():.1f}%")
        ax2.legend(fontsize=9)
        ax2.text(
            0.97, 0.95,
            f"n={len(data2)}\nstd={data2.std():.1f}%",
            transform=ax2.transAxes, ha="right", va="top", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.5),
        )
    ax2.set_xlabel(VARIABLE_LABELS.get(col2, col2), fontsize=11)
    ax2.set_ylabel("Density", fontsize=11)
    ax2.grid(True, alpha=0.3, linestyle="--")

    if plot_title:
        fig.suptitle("Distribution of Emission Results Across Cities", fontsize=14, fontweight="bold")

    plt.tight_layout()

    if save_path:
        if for_latex and not save_path.endswith(".pdf"):
            save_path = save_path.rsplit(".", 1)[0] + ".pdf"
        if plot_title:
            save_path = _add_title_suffix(save_path)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Distribution plot saved to {save_path}")
    else:
        plt.show()

    plt.close(fig)


def plot_province_boxplot(
    df: pd.DataFrame,
    variable: str,
    save_path: str = None,
    figsize: tuple = (10, 8),
    for_latex: bool = False,
    plot_title: bool = False,
    min_cities: int = 3,
):
    """Plot horizontal boxplot grouped by province, sorted by median.

    Provinces with fewer than min_cities are grouped as "Other".

    Args:
        df: DataFrame with region statistics.
        variable: Variable to plot (e.g., 'max_lifecycle_reduction_pct').
        save_path: Path to save figure.
        figsize: Figure size tuple.
        for_latex: If True, save as PDF.
        plot_title: If True, show title.
        min_cities: Minimum cities required per province (rest grouped as "Other").
    """
    if "province" not in df.columns or variable not in df.columns:
        print(f"Required columns not found for province boxplot")
        return

    plot_df = df[["province", variable]].dropna().copy()
    if plot_df.empty:
        print("No data for province boxplot")
        return

    # Group small provinces as "Other"
    province_counts = plot_df["province"].value_counts()
    small_provinces = province_counts[province_counts < min_cities].index
    plot_df.loc[plot_df["province"].isin(small_provinces), "province"] = "Other"

    # Sort by median
    medians = plot_df.groupby("province")[variable].median().sort_values()
    ordered_provinces = medians.index.tolist()

    # Prepare data for boxplot
    data = [plot_df[plot_df["province"] == p][variable].values for p in ordered_provinces]

    fig, ax = plt.subplots(figsize=figsize)

    bp = ax.boxplot(
        data,
        vert=False,
        patch_artist=True,
        labels=ordered_provinces,
    )

    # Color boxes by median value using RdYlGn colormap
    cmap = plt.cm.RdYlGn
    norm = plt.Normalize(medians.min(), medians.max())
    for patch, province in zip(bp["boxes"], ordered_provinces):
        color = cmap(norm(medians[province]))
        patch.set_facecolor(color)
        patch.set_alpha(0.8)

    ax.set_xlabel(VARIABLE_LABELS.get(variable, variable), fontsize=12)
    ax.set_ylabel("Province", fontsize=12)
    ax.tick_params(axis="y", labelsize=9)
    ax.grid(True, alpha=0.3, linestyle="--", axis="x")

    if plot_title:
        ax.set_title(
            f"{VARIABLE_LABELS.get(variable, variable)} by Province",
            fontsize=14,
            fontweight="bold",
        )

    plt.tight_layout()

    if save_path:
        if for_latex and not save_path.endswith(".pdf"):
            save_path = save_path.rsplit(".", 1)[0] + ".pdf"
        if plot_title:
            save_path = _add_title_suffix(save_path)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Province boxplot saved to {save_path}")
    else:
        plt.show()

    plt.close(fig)


def plot_regression_coefficients(
    results,
    result_var: str,
    save_path: str = None,
    figsize: tuple = (10, 6),
    for_latex: bool = False,
    plot_title: bool = False,
):
    """Plot horizontal bar chart of standardized regression coefficients with 95% CI.

    Args:
        results: statsmodels OLS RegressionResults.
        result_var: Name of dependent variable (for title).
        save_path: Path to save figure.
        figsize: Figure size tuple.
        for_latex: If True, save as PDF.
        plot_title: If True, show title.
    """
    # Exclude intercept
    params = results.params.drop("const", errors="ignore")
    conf = results.conf_int().drop("const", errors="ignore")
    pvalues = results.pvalues.drop("const", errors="ignore")

    if params.empty:
        print("No coefficients to plot")
        return

    # Sort by absolute value
    order = params.abs().sort_values(ascending=True).index
    params = params[order]
    conf = conf.loc[order]
    pvalues = pvalues[order]

    labels = [VARIABLE_LABELS.get(v, v) for v in params.index]
    y_pos = np.arange(len(params))

    # Error bars (distance from coefficient to CI bounds)
    ci_lower = params.values - conf[0].values
    ci_upper = conf[1].values - params.values
    errors = np.array([ci_lower, ci_upper])

    # Color by significance
    colors = []
    for p in pvalues.values:
        if p < 0.01:
            colors.append("#2ecc71")  # Green — highly significant
        elif p < 0.05:
            colors.append("#3498db")  # Blue — significant
        elif p < 0.1:
            colors.append("#f39c12")  # Orange — marginal
        else:
            colors.append("#95a5a6")  # Gray — not significant

    fig, ax = plt.subplots(figsize=figsize)

    ax.barh(y_pos, params.values, xerr=errors, color=colors, alpha=0.8,
            ecolor="black", capsize=3, height=0.6)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel("Standardized Coefficient (β)", fontsize=12)
    ax.axvline(x=0, color="black", linestyle="-", linewidth=0.8)
    ax.grid(True, alpha=0.3, linestyle="--", axis="x")

    # Add R² and N annotation
    ax.text(
        0.97, 0.03,
        f"N={int(results.nobs)}, R²={results.rsquared:.3f}, Adj R²={results.rsquared_adj:.3f}",
        transform=ax.transAxes, ha="right", va="bottom", fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.5),
    )

    # Legend for significance colors
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#2ecc71", label="p < 0.01"),
        Patch(facecolor="#3498db", label="p < 0.05"),
        Patch(facecolor="#f39c12", label="p < 0.1"),
        Patch(facecolor="#95a5a6", label="n.s."),
    ]
    ax.legend(handles=legend_elements, loc="lower left", fontsize=8, framealpha=0.8)

    if plot_title:
        result_label = VARIABLE_LABELS.get(result_var, result_var)
        ax.set_title(
            f"OLS Coefficients: {result_label}",
            fontsize=12,
            fontweight="bold",
        )

    plt.tight_layout()

    if save_path:
        if for_latex and not save_path.endswith(".pdf"):
            save_path = save_path.rsplit(".", 1)[0] + ".pdf"
        if plot_title:
            save_path = _add_title_suffix(save_path)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Regression coefficient plot saved to {save_path}")
    else:
        plt.show()

    plt.close(fig)


def plot_geographic_pattern(
    df: pd.DataFrame,
    save_path: str = None,
    figsize: tuple = (10, 6),
    for_latex: bool = False,
    plot_title: bool = False,
):
    """Scatter plot of latitude vs max_lifecycle_reduction_pct with regression.

    Args:
        df: DataFrame with region statistics (must have min_lat/max_lat columns).
        save_path: Path to save figure.
        figsize: Figure size tuple.
        for_latex: If True, save as PDF.
        plot_title: If True, show title.
    """
    result_col = "max_lifecycle_reduction_pct"

    # Compute center latitude from boundary columns
    if "min_lat" in df.columns and "max_lat" in df.columns:
        lat = (df["min_lat"] + df["max_lat"]) / 2
    else:
        print("No latitude columns found for geographic pattern plot")
        return

    valid = lat.notna() & df[result_col].notna()
    lat_vals = lat[valid].values
    result_vals = df.loc[valid, result_col].values

    if len(lat_vals) < 3:
        print("Not enough data for geographic pattern plot")
        return

    fig, ax = plt.subplots(figsize=figsize)

    ax.scatter(lat_vals, result_vals, s=30, alpha=0.6, edgecolors="darkblue", linewidths=0.5)

    # Regression line
    slope, intercept, r_value, p_value, std_err = stats.linregress(lat_vals, result_vals)
    x_line = np.linspace(lat_vals.min(), lat_vals.max(), 100)
    y_line = slope * x_line + intercept
    ax.plot(
        x_line, y_line, "r--", linewidth=2, alpha=0.8,
        label=f"r = {r_value:.3f}, p = {p_value:.1e}",
    )

    # 95% CI band
    n = len(lat_vals)
    x_mean = np.mean(lat_vals)
    se = np.sqrt(np.sum((result_vals - (slope * lat_vals + intercept)) ** 2) / (n - 2))
    t_val = stats.t.ppf(0.975, n - 2)
    ci = t_val * se * np.sqrt(
        1.0 / n + (x_line - x_mean) ** 2 / np.sum((lat_vals - x_mean) ** 2)
    )
    ax.fill_between(x_line, y_line - ci, y_line + ci, alpha=0.15, color="red")

    ax.set_xlabel("Latitude (°N)", fontsize=12)
    ax.set_ylabel(VARIABLE_LABELS.get(result_col, result_col), fontsize=12)
    ax.legend(loc="best", fontsize=10)
    ax.grid(True, alpha=0.3, linestyle="--")

    if plot_title:
        ax.set_title("Geographic Pattern: Latitude vs Emission Reduction", fontsize=14, fontweight="bold")

    plt.tight_layout()

    if save_path:
        if for_latex and not save_path.endswith(".pdf"):
            save_path = save_path.rsplit(".", 1)[0] + ".pdf"
        if plot_title:
            save_path = _add_title_suffix(save_path)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Geographic pattern plot saved to {save_path}")
    else:
        plt.show()

    plt.close(fig)
