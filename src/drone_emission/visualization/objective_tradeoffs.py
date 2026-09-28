"""Independent candidate panels for the climate/operations comparison."""
from pathlib import Path
import json

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, TwoSlopeNorm
import numpy as np
import pandas as pd

from .palette import color, colormap
from .prepared_bundle import prepared_plot
from .regions import plot_regions_on_china_map
from drone_emission.analysis.objective_tradeoffs import STRATEGIES

FOUR = ("emission_optimal", "cost_conservative", "cost_optimistic", "time_optimal")
LABELS = {"all_ebike": "All e-bike", "emission_optimal": "Climate optimum",
          "cost_conservative": "Cost optimum\n(conservative)", "cost_optimistic": "Cost optimum\n(optimistic)",
          "time_optimal": "Time optimum", "all_drone": "All drone"}
COLOR_KEYS = dict(zip(STRATEGIES, ("base.gray", "base.green", "base.coral", "base.blue", "base.purple", "base.cyan")))
PANELS = (
    *[f"share_map_{s}" for s in FOUR], "share_distribution",
    *[f"share_pair_{s}" for s in FOUR[1:]], "cohort_emissions",
    *[f"reduction_map_{s}" for s in FOUR], "reduction_distribution", "retention_distribution",
    "cost_premium_map_conservative", "cost_premium_map_optimistic", "cost_premium_distribution",
    "time_penalty_map", "time_penalty_distribution",
    "cost_retention_conservative", "cost_retention_optimistic", "time_retention",
    "cost_components_conservative", "cost_components_optimistic",
    "performance_conservative", "performance_optimistic",
)


def render(context):
    from drone_emission.paper_cohort import cohort_hash, locked_region_keys
    root = context.work_root / "analysis/objective_tradeoffs"
    for name in ("city_strategies.csv", "cohort_strategies.csv", "report.json"):
        if not (root / name).is_file():
            raise FileNotFoundError(f"Missing {root / name}; run analyze --only objective-tradeoffs")
    report = json.loads((root / "report.json").read_text())
    cities = pd.read_csv(root / "city_strategies.csv")
    if report["cohort_sha256"] != cohort_hash() or set(cities.region_key) != set(locked_region_keys()):
        raise ValueError("Tradeoff figures require the locked full cohort")
    stats = pd.read_csv(context.require_file("all_region_stats.csv", stage="aggregate"))
    names = stats[["region_key", "region_name_cn"]].drop_duplicates()
    plot_objective_tradeoffs(cities, pd.read_csv(root / "cohort_strategies.csv"), names,
                            save_path=str(context.output("objective_tradeoffs.pdf")), for_latex=True)


@prepared_plot()
def plot_objective_tradeoffs(cities, cohort, names, save_path=None, figsize=(5.4, 4.0),
                            for_latex=False, plot_title=False):
    """Export the candidate set without composing or publishing manuscript figures."""
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                         "pdf.fonttype": 42, "svg.fonttype": "none"})
    root = Path(save_path).parent
    base = cities.loc[cities.scenario.eq("conservative")]
    by_strategy = {s: base.loc[base.strategy.eq(s)].set_index("region_key") for s in STRATEGIES}
    colors = {s: color(COLOR_KEYS[s]) for s in STRATEGIES}
    files = []

    def finish(fig, name, description):
        if plot_title:
            fig.suptitle(description)
        for ax in fig.axes:
            if ax.axison:
                ax.spines[["top", "right"]].set_visible(False)
        stem = root / f"objective_{name}"
        for ext in ("pdf", "svg", "png"):
            fig.savefig(stem.with_suffix(f".{ext}"), bbox_inches="tight", dpi=300)
        plt.close(fig)
        files.append({"id": name, "question": description, "preview": stem.name + ".png"})

    def distribution(values, labels, y_label, name, reference=None):
        fig, ax = plt.subplots(figsize=figsize, layout="constrained")
        finite = [np.asarray(v)[np.isfinite(v)] for v in values]
        box = ax.boxplot(finite, patch_artist=True, widths=.5, showfliers=False)
        for i, (vals, patch) in enumerate(zip(finite, box["boxes"])):
            patch.set_facecolor(color("base.cyan"))
            # Deterministic horizontal spread; every city remains visible.
            jitter = .17 * np.sin(np.arange(len(vals)) * 2.399963)
            ax.scatter(i + 1 + jitter, vals, s=7, alpha=.4, color=color("base.blue"), linewidths=0)
        ax.set_xticks(range(1, len(labels) + 1), labels, fontsize=8)
        ax.set_ylabel(y_label)
        if reference is not None:
            for value in reference:
                ax.axhline(value, color=color("neutral.secondary"), lw=.8, ls="--")
        finish(fig, name, y_label)

    def map_panel(frame, metric, name, label, norm):
        fig, ax = plt.subplots(figsize=(5.4, 4.4))
        data = names.merge(frame.reset_index()[["region_key", metric]], on="region_key", how="inner")
        cmap = colormap("objective.diverging" if isinstance(norm, TwoSlopeNorm) else "objective.sequential")
        plot_regions_on_china_map(df=data, color_by=metric, ax=ax, norm=norm, cmap=cmap,
                                  draw_colorbar=False, draw_legend=True, plot_title=False)
        ax.set_axis_off()
        fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax,
                     orientation="horizontal", fraction=.045, pad=.035, label=label)
        finish(fig, name, label)

    reduction_values = base.loc[base.strategy.isin(FOUR), "emission_reduction_pct"]
    reduction_norm = TwoSlopeNorm(0, vmin=min(-1., reduction_values.min()), vmax=max(1., reduction_values.max()))
    for s in FOUR:
        map_panel(by_strategy[s], "drone_replacement_rate_pct", f"share_map_{s}",
                  LABELS[s].replace("\n", " ") + "\nDrone order share (%)", Normalize(0, 100))
        map_panel(by_strategy[s], "emission_reduction_pct", f"reduction_map_{s}",
                  LABELS[s].replace("\n", " ") + "\nEmission reduction vs all e-bike (%)", reduction_norm)
    distribution([by_strategy[s].drone_replacement_rate_pct for s in FOUR], [LABELS[s] for s in FOUR],
                 "Drone order share (%)", "share_distribution")
    distribution([by_strategy[s].emission_reduction_pct for s in FOUR], [LABELS[s] for s in FOUR],
                 "Emission reduction vs all e-bike (%)", "reduction_distribution", [0])
    distribution([by_strategy[s].mitigation_retained_pct for s in FOUR[1:]], [LABELS[s] for s in FOUR[1:]],
                 "Climate-optimal mitigation retained (%)", "retention_distribution", [0, 100])
    for s in FOUR[1:]:
        joined = by_strategy["emission_optimal"][["drone_replacement_rate_pct"]].join(
            by_strategy[s][["drone_replacement_rate_pct"]], lsuffix="_climate", rsuffix="_other")
        fig, ax = plt.subplots(figsize=figsize, layout="constrained")
        ax.scatter(joined.iloc[:, 0], joined.iloc[:, 1], s=14, alpha=.6, color=colors[s], linewidths=0)
        ax.plot([0, 100], [0, 100], "--", color=color("neutral.secondary"), lw=1)
        ax.set(xlim=(-2, 102), ylim=(-2, 102), xlabel="Climate-optimal drone share (%)",
               ylabel=LABELS[s].replace("\n", " ") + " drone share (%)")
        finish(fig, f"share_pair_{s}", "How do optimal drone shares differ?")
    whole = cohort.loc[cohort.scenario.eq("conservative")].set_index("strategy").loc[list(STRATEGIES)]
    fig, ax = plt.subplots(figsize=(6.2, 4), layout="constrained")
    bars = ax.bar(range(6), whole.annual_lifecycle_kg / 1e6, color=[colors[s] for s in STRATEGIES])
    ax.bar_label(bars, labels=[f"{v:.1f}%" for v in whole.emission_reduction_pct], padding=4, fontsize=8)
    ax.set_xticks(range(6), [LABELS[s] for s in STRATEGIES], rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("Annual transport lifecycle emissions (kt CO$_2$e)")
    ax.margins(y=.15)
    finish(fig, "cohort_emissions", "Labels: emission reduction relative to all e-bike")

    climates = {s: cities.loc[cities.scenario.eq(s) & cities.strategy.eq("emission_optimal")].set_index("region_key")
                for s in ("conservative", "optimistic")}
    premium_max = max(1e-6, *(f.cost_premium_cny.max() for f in climates.values()))
    for scenario, frame in climates.items():
        map_panel(frame, "cost_premium_cny", f"cost_premium_map_{scenario}",
                  f"{scenario.capitalize()} conditions\nExtra cost of climate optimum (CNY/order)", Normalize(0, premium_max))
        economic = by_strategy[f"cost_{scenario}"]
        joined = frame[["cost_premium_cny"]].join(economic[["mitigation_retained_pct"]])
        fig, ax = plt.subplots(figsize=figsize, layout="constrained")
        ax.scatter(joined.cost_premium_cny, joined.mitigation_retained_pct, s=14, alpha=.6,
                   color=colors[f"cost_{scenario}"], linewidths=0)
        ax.axhline(100, color=color("neutral.secondary"), lw=.8, ls="--")
        ax.axhline(0, color=color("neutral.secondary"), lw=.8, ls=":")
        ax.set(xlabel=f"Extra cost of climate optimum (CNY/order)\n{scenario.capitalize()} conditions",
               ylabel="Cost-optimal mitigation retained (%)")
        finish(fig, f"cost_retention_{scenario}", "Cost of climate priority and mitigation retained by cost priority")
    distribution([f.cost_premium_cny for f in climates.values()], ["Conservative", "Optimistic"],
                 "Extra cost of climate optimum (CNY/order)", "cost_premium_distribution", [0])
    climate = climates["conservative"]
    map_panel(climate, "time_penalty_min", "time_penalty_map", "Extra transport time of climate optimum (min/order)",
              Normalize(0, max(1e-6, climate.time_penalty_min.max())))
    distribution([climate.time_penalty_min], ["Climate optimum vs time optimum"],
                 "Extra transport time (min/order)", "time_penalty_distribution", [0])
    joined = climate[["time_penalty_min"]].join(by_strategy["time_optimal"][["mitigation_retained_pct"]])
    fig, ax = plt.subplots(figsize=figsize, layout="constrained")
    ax.scatter(joined.time_penalty_min, joined.mitigation_retained_pct, s=14, alpha=.6,
               color=colors["time_optimal"], linewidths=0)
    ax.axhline(100, color=color("neutral.secondary"), ls="--", lw=.8)
    ax.axhline(0, color=color("neutral.secondary"), ls=":", lw=.8)
    ax.set(xlabel="Extra time of climate optimum (min/order)", ylabel="Time-optimal mitigation retained (%)")
    finish(fig, "time_retention", "Time cost of climate priority and mitigation retained by speed priority")

    components = {"annual_vehicle_cny": "Vehicles", "annual_battery_cny": "Batteries",
                  "annual_cabinet_capital_cny": "Cabinet capital", "annual_cabinet_electricity_cny": "Cabinet electricity",
                  "annual_staff_cny": "Ground staff", "annual_pilot_cny": "Pilots", "annual_variable_cny": "Rider pay + transport energy"}
    component_colors = [color(k) for k in ("base.blue", "base.cyan", "base.green", "base.yellow", "base.coral", "base.purple", "base.gray")]
    for scenario in climates:
        frame = cohort.loc[cohort.scenario.eq(scenario)].set_index("strategy").loc[list(STRATEGIES)]
        fig, ax = plt.subplots(figsize=(6.5, 4.8), layout="constrained")
        bottom = np.zeros(6)
        for (column, label), c in zip(components.items(), component_colors):
            values = (frame[column] / frame.annual_orders).to_numpy()
            ax.barh(range(6), values, left=bottom, color=c, label=label)
            bottom += values
        ax.set_yticks(range(6), [LABELS[s] for s in STRATEGIES], fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel(f"Total cost under {scenario} conditions (CNY/order)")
        ax.legend(loc="upper center", bbox_to_anchor=(.5, -.16), ncol=2, fontsize=8, frameon=False)
        finish(fig, f"cost_components_{scenario}", "Sources of economic differences")
        columns = ["drone_replacement_rate_pct", "emission_reduction_pct", "cost_change_vs_ebike_pct", "time_change_vs_ebike_pct"]
        values = frame[columns].to_numpy()
        fig, ax = plt.subplots(figsize=(7.2, 4.2), layout="constrained")
        ax.set(xlim=(-.5, 3.5), ylim=(5.5, -.5))
        ax.set_xticks(range(4), ["Drone share (%)", "Emission reduction (%)",
                               f"Cost change (%)\n{scenario.capitalize()} conditions", "Time change (%)"], fontsize=8)
        ax.xaxis.tick_top()
        ax.set_yticks(range(6), [LABELS[s] for s in STRATEGIES], fontsize=8)
        raw = [frame.drone_replacement_rate_pct, frame.annual_lifecycle_kg / frame.annual_orders * 1000,
               frame.mean_total_cost_cny, frame.mean_transport_time_min]
        units = ["% orders", "g CO$_2$e/order", "CNY/order", "min/order"]
        for i in range(6):
            for j in range(4):
                ax.text(j, i, f"{values[i,j]:.1f}%\n{raw[j].iloc[i]:.2f} {units[j]}",
                        ha="center", va="center", fontsize=8, color=color("ink.text"))
        for y in np.arange(-.5, 6, 1):
            ax.axhline(y, color=color("ink.grid"), lw=.6)
        ax.tick_params(length=0)
        finish(fig, f"performance_{scenario}", "Different objectives evaluated on common metrics; no composite score")
    (root / "objective_candidates.json").write_text(json.dumps(files, indent=2), encoding="utf-8")
    print(f"Saved {len(files)} objective-tradeoff candidate panels to {root}")
