"""Cross-evaluate climate, economic and transport-time allocations."""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .full_cost_pilot import ReferenceSystem, build_profiles, development_stages, solve_assignment
from drone_emission.paper_cohort import cohort_hash, locked_region_keys
from drone_emission.utils.atomic import atomic_open

STRATEGIES = ("all_ebike", "emission_optimal", "cost_conservative", "cost_optimistic", "time_optimal", "all_drone")
VERSION = 1


def derive_metrics(frame):
    """Use totals for both city and cohort metrics; never average city ratios."""
    result = []
    for scenario, group in frame.groupby("scenario", sort=False):
        group = group.copy().set_index("strategy")
        orders = group.annual_orders
        group["drone_replacement_rate_pct"] = group.drone_annual_orders / orders * 100
        group["mean_total_cost_cny"] = group.annual_total_cny / orders
        group["mean_transport_time_min"] = group.annual_transport_minutes / orders
        base = group.loc["all_ebike"]
        climate = group.loc["emission_optimal"]
        economic = group.loc[f"cost_{scenario}"]
        fastest = group.loc["time_optimal"]
        group["annual_saved_kg"] = base.annual_lifecycle_kg - group.annual_lifecycle_kg
        group["emission_reduction_pct"] = group.annual_saved_kg / base.annual_lifecycle_kg * 100
        maximum = base.annual_lifecycle_kg - climate.annual_lifecycle_kg
        group["mitigation_retained_pct"] = group.annual_saved_kg / maximum * 100 if maximum > 1e-9 else np.nan
        group["cost_change_vs_ebike_pct"] = (group.mean_total_cost_cny / base.mean_total_cost_cny - 1) * 100
        group["time_change_vs_ebike_pct"] = (group.mean_transport_time_min / base.mean_transport_time_min - 1) * 100
        group["cost_premium_cny"] = group.mean_total_cost_cny - economic.mean_total_cost_cny
        group["cost_premium_pct"] = group.cost_premium_cny / economic.mean_total_cost_cny * 100
        group["time_penalty_min"] = group.mean_transport_time_min - fastest.mean_transport_time_min
        group["time_penalty_pct"] = group.time_penalty_min / fastest.mean_transport_time_min * 100
        result.append(group.reset_index())
    return pd.concat(result, ignore_index=True)


def evaluate_city(profile, config, system):
    annual_orders = float(profile.orders.sum())
    share = profile.share.to_numpy()
    allocations = {
        "all_ebike": np.zeros(len(profile)), "all_drone": share,
        "emission_optimal": share * profile.emission_choice.to_numpy(),
        "time_optimal": share * profile.time_choice.to_numpy(),
    }
    stages = list(development_stages(config, system))
    optima = {}
    for stage, stage_config, stage_system in stages:
        row, allocation = solve_assignment(profile, annual_orders, stage_config, stage_system, return_assignment=True)
        allocations[f"cost_{stage['stage']}"] = allocation
        optima[stage["stage"]] = row
    rows = []
    for stage, stage_config, stage_system in stages:
        scenario = stage["stage"]
        for strategy in STRATEGIES:
            x = allocations[strategy]
            row = solve_assignment(profile, annual_orders, stage_config, stage_system, fixed_assignment=x)
            row.update(
                scenario=scenario, strategy=strategy,
                drone_hours_daily=stage_system.drone_hours_daily,
                drone_annual_orders=float(x.sum() * annual_orders),
                annual_lifecycle_kg=float((np.dot(share - x, profile.ebike_lifecycle_kg)
                                          + np.dot(x, profile.drone_lifecycle_kg)) * annual_orders),
                annual_transport_minutes=row["mean_transport_time_min"] * annual_orders,
            )
            tolerance = max(1., row["annual_total_cny"] * 2e-6)
            if optima[scenario]["annual_total_cny"] > row["annual_total_cny"] + tolerance:
                raise RuntimeError(f"Economic optimum worse than {strategy} in {scenario}")
            if strategy == f"cost_{scenario}":
                if abs(row["annual_total_cny"] - optima[scenario]["annual_total_cny"]) > tolerance:
                    raise RuntimeError("Fixed-allocation cost differs from optimized cost")
                row["solver_relative_gap"] = optima[scenario]["solver_relative_gap"]
            rows.append(row)
    result = derive_metrics(pd.DataFrame(rows))
    for _, group in result.groupby("scenario"):
        climate = group.loc[group.strategy.eq("emission_optimal"), "annual_lifecycle_kg"].iloc[0]
        fastest = group.loc[group.strategy.eq("time_optimal"), "mean_transport_time_min"].iloc[0]
        if (group.annual_lifecycle_kg < climate - 1e-6).any() or (group.mean_transport_time_min < fastest - 1e-9).any():
            raise RuntimeError("Cross-evaluation violates emission/time minima")
    return result


def _write_json(path, payload):
    with atomic_open(path, "w") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def run_objective_tradeoffs(config, *, force=False, region_key=None):
    keys = locked_region_keys()
    if region_key is not None:
        if region_key not in keys:
            raise ValueError("City must belong to the locked paper cohort")
        keys = [region_key]
    root = Path(config["work_dir"]) / "analysis/objective_tradeoffs"
    if region_key:
        root = root / "pilot" / region_key
    data = Path(config["data_dir"])
    stats = pd.read_csv(data / "all_region_stats.csv")
    stats = stats.loc[stats.region_key.isin(keys)].set_index("region_key", drop=False)
    if len(stats) != len(keys) or set(stats.index) != set(keys):
        raise ValueError("Objective tradeoffs inputs differ from locked cohort")
    system = ReferenceSystem(**config.get("full_cost_pilot", {}).get("system", {}))
    frames, audits = [], []
    for index, key in enumerate(keys, 1):
        city = stats.loc[key]
        directory = str(city.region_name_en).lower()
        path = data / "city" / directory / f"{directory}_od_pairs.csv"
        signature = {
            "version": VERSION, "cohort": cohort_hash(), "path": str(path.resolve()),
            "size": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns,
            "parameters": {k: config.get(k, {}) for k in ("full_cost_pilot", "transport_cost", "emission_params")},
        }
        digest = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()
        checkpoint = root / "cities" / f"{key}.csv"
        metadata = checkpoint.with_suffix(".json")
        if not force and checkpoint.is_file() and metadata.is_file() and json.loads(metadata.read_text()).get("signature") == digest:
            frame = pd.read_csv(checkpoint)
            audit = json.loads(metadata.read_text())
        else:
            profile, audit = build_profiles(path, config, system, include_objectives=True)
            if not np.isclose(audit["city_annual_orders"], city.total_odpair_annual_sales, rtol=1e-10):
                raise ValueError(f"{key}: OD demand differs from aggregate")
            frame = evaluate_city(profile, config, system)
            frame.insert(0, "region_key", key)
            frame.insert(1, "region_name_en", city.region_name_en)
            ref = frame.loc[frame.scenario.eq("conservative")].set_index("strategy")
            for strategy, metric, target in (
                ("all_ebike", "annual_lifecycle_kg", "total_ebike_lifecycle"),
                ("emission_optimal", "annual_lifecycle_kg", "oracle_min_lifecycle_kg"),
                ("emission_optimal", "drone_replacement_rate_pct", "oracle_drone_order_ratio_pct"),
            ):
                if not np.isclose(ref.loc[strategy, metric], city[target], rtol=1e-9, atol=1e-6):
                    raise ValueError(f"{key}: {target} differs from main experiment")
            audit.update(signature=digest, region_key=key, input=signature)
            with atomic_open(checkpoint, "w", newline="") as handle:
                frame.to_csv(handle, index=False)
            _write_json(metadata, audit)
        frames.append(frame)
        audits.append(audit)
        print(f"Objective tradeoffs {index}/{len(keys)}: {key}", flush=True)
    cities = pd.concat(frames, ignore_index=True)
    total_columns = [c for c in cities if c.startswith("annual_") or c in (
        "drone_annual_orders", "ebike_count", "drone_count", "pilot_positions", "ground_staff",
        "distribution_centers", "network_modules", "common_dc_cabinets",
        "ordinary_destination_cabinets", "drone_destination_cabinets",
    )]
    cohort = derive_metrics(cities.groupby(["scenario", "strategy"], sort=False)[total_columns].sum().reset_index())
    paths = tuple(root / n for n in ("city_strategies.csv", "cohort_strategies.csv", "report.json", "summary.md"))
    for path, frame in zip(paths[:2], (cities, cohort)):
        with atomic_open(path, "w", newline="") as handle:
            frame.to_csv(handle, index=False)
    report = {
        "analysis_version": VERSION, "cohort_sha256": cohort_hash(), "city_count": len(keys),
        "scope": "single_city_pilot" if region_key else "locked_paper_cohort",
        "system": asdict(system), "scenarios": [s for s, _, _ in development_stages(config, system)],
        "source_dois": ["10.1016/j.scs.2025.106134", "10.1016/j.trd.2022.103296"],
        "boundaries": "Original transport lifecycle emissions; full economic costs; one-way transport time without waiting or pickup.",
        "allocation": "100 m road/air groups split by exact original emission/time choices; fractional economic assignment within groups.",
        "wage_basis": "200 CNY/person/day local proxy; full configured shifts paid. 8/12 h are project assumptions.",
        "aggregation": "Sum annual totals before computing cohort ratios; city distributions are unweighted.",
        "cities": audits,
    }
    _write_json(paths[2], report)
    write_summary(cities, cohort, paths[3])
    return paths


def write_summary(cities, cohort, path):
    lines = ["# 目标比较结果总结", "", f"研究城市：{cities.region_key.nunique()}；仅汇总研究样本，不作全国外推。", "",
             "|经济情景|配置策略|无人机替代率 (%)|减排率 (%)|每单成本 (元)|运输时间 (分钟)|", "|---|---|---:|---:|---:|---:|"]
    for row in cohort.itertuples():
        lines.append(f"|{row.scenario}|{row.strategy}|{row.drone_replacement_rate_pct:.1f}|{row.emission_reduction_pct:.1f}|{row.mean_total_cost_cny:.3f}|{row.mean_transport_time_min:.2f}|")
    lines += ["", "## 研究样本整体权衡（按订单加权）", ""]
    for scenario in ("conservative", "optimistic"):
        rows = cohort.loc[cohort.scenario.eq(scenario)].set_index("strategy")
        economic, climate = rows.loc[f"cost_{scenario}"], rows.loc["emission_optimal"]
        lines.append(f"- {scenario}：经济最优保留碳最优减排潜力的 {economic.mitigation_retained_pct:.1f}%；选择碳最优需额外 {climate.cost_premium_cny:.3f} 元/单（{climate.cost_premium_pct:.1f}%）。碳最优相对全电动车的成本变化为 {climate.cost_change_vs_ebike_pct:.1f}%。")
    rows = cohort.loc[cohort.scenario.eq("conservative")].set_index("strategy")
    climate, fastest = rows.loc["emission_optimal"], rows.loc["time_optimal"]
    lines += [f"- 时间最优保留碳最优减排潜力的 {fastest.mitigation_retained_pct:.1f}%；选择碳最优需额外 {climate.time_penalty_min:.3f} 分钟/单（{climate.time_penalty_pct:.1f}%）。碳最优相对全电动车的运输时间变化为 {climate.time_change_vs_ebike_pct:.1f}%。", "", "## 城市差异", ""]
    for scenario in ("conservative", "optimistic"):
        frame = cities.loc[cities.scenario.eq(scenario)]
        climate = frame.loc[frame.strategy.eq("emission_optimal")]
        economic = frame.loc[frame.strategy.eq(f"cost_{scenario}")]
        lines.append(f"- {scenario}：经济最优增排城市 {(economic.emission_reduction_pct < -1e-8).sum()} 个；减排保留比例中位数 {economic.mitigation_retained_pct.median():.1f}%；碳最优额外成本中位数 {climate.cost_premium_cny.median():.3f} 元/单（{climate.cost_premium_pct.median():.1f}%）。")
    time = cities.loc[cities.scenario.eq("conservative") & cities.strategy.eq("time_optimal")]
    climate = cities.loc[cities.scenario.eq("conservative") & cities.strategy.eq("emission_optimal")]
    lines += [f"- 时间最优增排城市 {(time.emission_reduction_pct < -1e-8).sum()} 个；减排保留比例中位数 {time.mitigation_retained_pct.median():.1f}%；碳最优额外运输时间中位数 {climate.time_penalty_min.median():.2f} 分钟/单。", "",
              "排放采用原运输生命周期口径；成本包含车队、设施和人工。经济情景共同改变设备价格、人工与运营时长。成本最优基于100台整数模块及距离分组近似。", ""]
    with atomic_open(path, "w") as handle:
        handle.write("\n".join(lines))
