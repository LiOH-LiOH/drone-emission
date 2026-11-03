"""Audit a strict cohort subtraction without rerunning city experiments.

Only existing city tables, attribution fits, and sampled pathway curves are
used. Source roots are read-only; an interrupted destination is never reused.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

from drone_emission.analysis_workflow import (
    AUTHORITATIVE_AGGREGATES, _canonical_sha256, _sha256,
)
from drone_emission.paper_cohort import (
    cohort_hash, experiment_excluded_region_keys, load_cohort_contract,
    locked_region_keys, region_keys_sha256,
)
from drone_emission.paper_reuse import (
    GLOBAL_DATA_ARTIFACTS, REUSE_SCHEMA_VERSION, WORK_ARTIFACTS, _artifact_record,
)

SOURCE_COHORT_SHA256 = "869f23b4180521d1678200648c089d4c4eb7ed98fc425f9cb71b7971af6b515f"
EXCLUDED_KEYS = {"aomentebiexingzhengqu", "xianggangtebiexingzhengqu"}
MIGRATED_MODULES = (
    "named-scenarios", "pathway-scenarios", "distance-threshold",
    "threshold-resilience", "city-attribution", "contribution-decomposition",
)
AFFECTED_REQUESTS = (
    "province_total_ebike_lifecycle", "regions_map_oracle_reduction_rank_map_only",
    "city_reduction_potential_scatter", "city_reduction_rank_all_macro_region",
    "city_gef_distribution", "city_gef_lifecycle_components",
    "city_distance_drone_share_emission_fit", "city_detour_reduction_by_gef",
    "contribution_decomposition_city_b", "regions_map_pathway_scenarios",
    "threshold_resilience_national_overview", "r2_population_reduction_contrast",
    "break_even_contribution_decomposition_marching_cubes_subfigures",
    "distance_threshold_policy_comparison", "regions_map_pathway_scenario_maps",
)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def subtract_rows(frame: pd.DataFrame, expected: set[str]) -> pd.DataFrame:
    """Subtract only the two reviewed exclusions; reject other sample changes."""
    if "region_key" not in frame:
        raise ValueError("cohort table has no region_key column")
    actual = set(frame.region_key.dropna().astype(str))
    if actual != expected | EXCLUDED_KEYS:
        raise ValueError(f"source cohort mismatch: missing={expected - actual}, extra={actual - expected - EXCLUDED_KEYS}")
    result = frame.loc[~frame.region_key.astype(str).isin(EXCLUDED_KEYS)].copy()
    if set(result.region_key.astype(str)) != expected:
        raise ValueError("subtracted table does not cover the locked cohort")
    return result


def original_city_rows(frame: pd.DataFrame, expected: set[str]) -> pd.DataFrame:
    """Select the original lock without resetting its sampling row indices."""
    return frame.loc[
        frame.region_level.eq("city")
        & frame.region_key.astype(str).isin(expected | EXCLUDED_KEYS)
    ].copy()


def audit_source(work_root: Path) -> tuple[dict, dict, Path]:
    status = json.loads((work_root / "analysis/status.json").read_text())
    if status.get("paper_cohort_sha256") != SOURCE_COHORT_SHA256:
        raise ValueError("source status is not the reviewed 313-city cohort")
    manifest = json.loads(Path(status["input_manifest"]).read_text())
    body = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    if (_canonical_sha256(body) != manifest["manifest_sha256"]
            or status["input_manifest_sha256"] != manifest["manifest_sha256"]):
        raise ValueError("source input manifest digest mismatch")
    if manifest["cohort_sha256"] != SOURCE_COHORT_SHA256 or manifest["city_count"] != 313:
        raise ValueError("source input manifest has a different cohort")
    data = Path(status["analysis_data_dir"])
    for module in MIGRATED_MODULES:
        record = status["modules"][module]
        if record["status"] != "complete":
            raise ValueError(f"source module incomplete: {module}")
        for output in record["outputs"]:
            path = Path(output["path"])
            if path.stat().st_size != output["size_bytes"] or _sha256(path) != output["sha256"]:
                raise ValueError(f"source module output changed: {path}")
    for record in manifest["sha256_inputs"]:
        declaration = record["path"]
        if declaration.startswith("city:"):
            path = data / "city" / declaration.removeprefix("city:")
        elif declaration.startswith("data:") and declaration[5:] in AUTHORITATIVE_AGGREGATES:
            path = data / declaration[5:]
        elif declaration == "data:gef":
            path = data / "gef_china.csv"
        else:
            continue
        if _sha256(path) != record["sha256"]:
            raise ValueError(f"source authoritative input changed: {path}")
    for record in manifest["stat_inputs"]:
        if record["path"] == "data:combined_data.csv":
            path = data / "combined_data.csv"
        elif record["path"].startswith("city:"):
            path = data / "city" / record["path"][5:]
        else:
            continue
        stat = path.stat()
        if stat.st_size != record["size_bytes"] or stat.st_mtime_ns != record["mtime_ns"]:
            raise ValueError(f"source OD checkpoint changed: {path}")
    return status, manifest, data


def sample_city_od(path: Path, count: int, seed: int, *, chunk_rows=250_000):
    """Reproduce pandas.sample using valid-row positions in one CSV scan."""
    from drone_emission.analysis.pathway_scenarios import OD_COLUMNS, PATHWAY_CURVE_MAX_ROWS_PER_CITY

    if count <= 0:
        raise ValueError(f"missing valid OD count: {path}")
    positions = (
        np.random.RandomState(seed).choice(count, PATHWAY_CURVE_MAX_ROWS_PER_CITY, replace=False)
        if count > PATHWAY_CURVE_MAX_ROWS_PER_CITY else np.arange(count)
    )
    sorted_positions = np.sort(positions)
    chunks, offset, total = [], 0, 0.0
    for chunk in pd.read_csv(path, usecols=list(OD_COLUMNS), chunksize=chunk_rows):
        chunk = chunk.loc[:, list(OD_COLUMNS)].apply(pd.to_numeric, errors="coerce")
        valid = chunk.loc[
            (chunk.road_distance_m >= 0) & (chunk.distance_km > 0)
            & (chunk.annual_sales > 0)
        ].dropna().copy()
        valid.index = np.arange(offset, offset + len(valid))
        total += float(valid.annual_sales.sum())
        lower, upper = np.searchsorted(sorted_positions, [offset, offset + len(valid)])
        selected = sorted_positions[lower:upper]
        if len(selected):
            chunks.append(valid.loc[selected])
        offset += len(valid)
    if offset != count:
        raise ValueError(f"valid OD count changed for {path}: expected={count}, actual={offset}")
    sample = pd.concat(chunks).loc[positions].reset_index(drop=True)
    return sample, total / float(sample.annual_sales.sum())


def rebuild_curves(source: Path, target: Path, config: dict, expected: set[str]) -> dict:
    from drone_emission.analysis.pathway_scenarios import (
        PATHWAY_CURVE_N_POINTS, PATHWAY_CURVE_RANDOM_STATE, PATHWAY_KAPPA_END,
        _baseline_kappa, _evaluate_curve_point,
    )
    from drone_emission.emission.cal_single_delivery import create_drone_params, create_ebike_params
    from drone_emission.utils.grid_emission_factor import fetch_factor

    original_report = json.loads((source / "all_pathway_scenarios_report.json").read_text())
    signature = original_report["signature"]
    if signature["emission_params"] != config.get("emission_params", {}):
        raise ValueError("emission parameters changed since source pathway analysis")
    if signature["curve"] != {"n_points": PATHWAY_CURVE_N_POINTS,
                               "max_rows_per_city": 1500,
                               "random_state": PATHWAY_CURVE_RANDOM_STATE,
                               "kappa_end": PATHWAY_KAPPA_END}:
        raise ValueError("pathway sampling contract changed")
    original = pd.read_csv(source / "all_region_stats.csv")
    # The authoritative aggregate can include the 15 pre-existing exclusions.
    # Keep its original (possibly sparse) indices: they are the sampling seeds.
    original = original_city_rows(original, expected)
    named = pd.read_csv(source / "all_sensitivity_named.csv")
    baseline = _baseline_kappa(named, original)
    new_baseline = _baseline_kappa(pd.read_csv(target / "all_sensitivity_named.csv"), pd.read_csv(target / "all_region_stats.csv"))
    if baseline != new_baseline or baseline != original_report["baseline_kappa"]:
        raise ValueError("cohort subtraction changed pathway parameter grid")
    grids = {"kappa": np.linspace(baseline, PATHWAY_KAPPA_END, PATHWAY_CURVE_N_POINTS),
             "gef": np.linspace(1.0, 0.5, PATHWAY_CURVE_N_POINTS)}
    city_points = []
    ebike, drone = create_ebike_params(config), create_drone_params(config)
    emission = config.get("emission_params", {})
    sample_audit = []
    for source_index, city in original.iterrows():
        key = str(city.region_key)
        path = source / "city" / key / f"{key}_od_pairs.csv"
        seed = PATHWAY_CURVE_RANDOM_STATE + int(source_index)
        sample, scale = sample_city_od(path, int(city.valid_path_count), seed)
        gef = float(fetch_factor(city.province, config["gef_path"]))
        # fetch_factor's national fallback is unchanged by the cohort reduction.
        sample_audit.append({"region_key": key, "source_index": int(source_index),
                             "seed": seed, "sample_rows": len(sample), "scale": scale})
        for pathway, grid in grids.items():
            for x in grid:
                point = _evaluate_curve_point(
                    sample, scale, ebike, drone, gef,
                    kappa=float(x) if pathway == "kappa" else baseline,
                    gef_mult=float(x) if pathway == "gef" else 1.0,
                    baseline_kappa=baseline,
                    rider_weight=float(emission.get("rider_weight", 70.0)),
                    payload_kg=float(emission.get("payload_kg", 1.0)),
                )
                city_points.append({"region_key": key, "pathway": pathway,
                                    "x": float(x), **point})
        # Persist only the small sample, never a copy of the city OD checkpoint.
        sample_dir = (
            target / "analysis/pathway_samples" if key in expected
            else target.parent / "source_curve_samples"
        )
        sample_dir.mkdir(parents=True, exist_ok=True)
        sample.to_csv(sample_dir / f"{key}.csv", index=False)
        print(f"Curve sample {len(sample_audit)}/313: {key}", flush=True)
    points = pd.DataFrame(city_points)

    def aggregate(frame):
        rows = []
        for pathway, grid in grids.items():
            for x in grid:
                selected = frame.loc[frame.pathway.eq(pathway) & frame.x.eq(x)]
                rows.append({"pathway": pathway, "x": float(x),
                             "saved_kt": float(selected.saved_kg.sum()) / 1e6,
                             "drone_share_pct": float(np.nanmedian(selected.drone_share_pct))})
        return pd.DataFrame(rows)

    old_curves = pd.read_csv(source / "all_pathway_curve_points.csv")
    reconstructed = aggregate(points)
    pd.testing.assert_frame_equal(old_curves[["pathway"]], reconstructed[["pathway"]])
    np.testing.assert_allclose(old_curves[["x", "saved_kt", "drone_share_pct"]],
                               reconstructed[["x", "saved_kt", "drone_share_pct"]],
                               rtol=1e-10, atol=1e-9)
    retained = points.loc[points.region_key.isin(expected)].copy()
    aggregate(retained).to_csv(target / "all_pathway_curve_points.csv", index=False)
    retained.to_csv(target / "analysis/pathway_city_curve_points.csv", index=False)
    write_json(target / "analysis/pathway_sample_audit.json", {
        "source_cohort_sha256": SOURCE_COHORT_SHA256,
        "cohort_sha256": cohort_hash(), "original_curve_reproduced": True,
        "cities": sample_audit,
    })
    report = dict(original_report, paper_cohort_sha256=cohort_hash(), city_count=len(expected))
    # The old signature is evidence, not a claim that new data were calculated by
    # the full-OD pathway runner. Renderers validate the new cohort and version.
    report.pop("signature", None)
    report["migration"] = {"source_signature": signature,
                           "method": "retain exact city scenarios; recompute sampled curves"}
    write_json(target / "all_pathway_scenarios_report.json", report)
    return {"original_curve_reproduced": True, "city_samples": len(sample_audit),
            "retained_city_samples": len(expected), "points_per_city": 2 * PATHWAY_CURVE_N_POINTS}


def migrate(config: dict, source_work_root: Path, output: Path) -> dict:
    expected = set(locked_region_keys())
    if len(expected) != 311 or experiment_excluded_region_keys() != EXCLUDED_KEYS:
        raise ValueError("migration requires the reviewed 311-city exclusion contract")
    status, manifest, source = audit_source(source_work_root)
    source_aggregate = pd.read_csv(source / "all_region_stats.csv")
    raw_keys = set(source_aggregate.region_key.astype(str))
    old_keys = expected | EXCLUDED_KEYS
    previous_excluded = set(load_cohort_contract()["excluded_regions"]) - EXCLUDED_KEYS
    if (region_keys_sha256(old_keys) != SOURCE_COHORT_SHA256
            or not old_keys <= raw_keys
            or raw_keys - old_keys - previous_excluded):
        raise ValueError("migration must remove exactly Hong Kong and Macau")
    source_stats = source_aggregate.loc[source_aggregate.region_key.astype(str).isin(old_keys)].copy()
    if output.exists():
        raise FileExistsError(f"refusing to mix with an existing migration batch: {output}")
    output.mkdir(parents=True)
    target = output / "data"
    target.mkdir()
    write_json(output / "source_status.json", status)
    write_json(output / "source_input_manifest.json", manifest)
    write_json(output / "target_cohort.json", load_cohort_contract())
    table_audit = {}
    for name in GLOBAL_DATA_ARTIFACTS:
        if name in {"all_pathway_curve_points.csv", "all_pathway_scenarios_report.json"}:
            continue
        if name.endswith(".json"):
            payload = json.loads((source / name).read_text())
            if set(payload) != raw_keys:
                raise ValueError(f"unexpected aggregate JSON city set: {name}")
            write_json(target / name, {k: v for k, v in payload.items() if k in expected})
        else:
            frame = pd.read_csv(source / name)
            source_row_count = len(frame)
            if name in AUTHORITATIVE_AGGREGATES:
                if set(frame.region_key.astype(str)) != raw_keys:
                    raise ValueError(f"authoritative aggregates disagree: {name}")
                frame = frame.loc[frame.region_key.astype(str).isin(old_keys)].copy()
            filtered = subtract_rows(frame, expected)
            rebound = []
            for column in ("paper_cohort_sha256", "cohort_sha256"):
                if column in filtered:
                    if set(filtered[column].dropna().astype(str)) != {SOURCE_COHORT_SHA256}:
                        raise ValueError(f"unexpected source cohort metadata: {name}:{column}")
                    filtered[column] = cohort_hash()
                    rebound.append(column)
            filtered.to_csv(target / name, index=False)
            pd.testing.assert_frame_equal(filtered.reset_index(drop=True), pd.read_csv(target / name),
                                          check_dtype=False, rtol=1e-13, atol=1e-10)
            table_audit[name] = {"old_rows": len(frame), "new_rows": len(filtered),
                                 "source_aggregate_rows": source_row_count,
                                 "source_sha256": _sha256(source / name),
                                 "rebound_metadata_columns": rebound,
                                 "scientific_values_unchanged": True}
    # Only retained city directories are exposed to downstream readers.
    (target / "city").mkdir()
    for key in sorted(expected):
        (target / "city" / key).symlink_to((source / "city" / key).resolve(), target_is_directory=True)
    for item in source.iterdir():
        if item.is_file() and item.name not in GLOBAL_DATA_ARTIFACTS and not item.name.startswith("all_") and item.name != "paper_cohort_view.json":
            (target / item.name).symlink_to(item.resolve())
    view_config = dict(config, data_dir=str(target), gef_path=str(target / "gef_china.csv"),
                       sales_path=str(target / "combined_data.csv"), work_dir=str(output / "work"))
    curve_audit = rebuild_curves(source, target, view_config, expected)
    from drone_emission.analysis.city_attribution import run_r2_city_attribution, write_r2_city_attribution_report

    old_attribution = source_work_root / "analysis/city_attribution/r2_city_attribution_data.csv"
    attribution = subtract_rows(pd.read_csv(old_attribution), expected)
    attribution_root = output / "work/analysis/city_attribution"
    attribution_root.mkdir(parents=True)
    attribution.to_csv(attribution_root / old_attribution.name, index=False)
    attribution_report = run_r2_city_attribution(attribution.reset_index(drop=True), [])
    attribution_report["source"] = {"migration_source": str(old_attribution),
                                     "source_sha256": _sha256(old_attribution),
                                     "cohort_sha256": cohort_hash()}
    write_r2_city_attribution_report(attribution_report, attribution_root / "r2_city_attribution_analysis_report.json")
    reuse = output / "reuse"
    records = []
    for name in GLOBAL_DATA_ARTIFACTS:
        dest = reuse / "payload/data" / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(target / name, dest)
        records.append(_artifact_record(dest, f"data:{name}"))
    for name in WORK_ARTIFACTS:
        prepared = output / "work" / name
        if not prepared.is_file():
            continue
        dest = reuse / "payload/work" / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(prepared, dest)
        records.append(_artifact_record(dest, f"work:{name}"))
    write_json(reuse / "manifest.json", {
        "schema_version": REUSE_SCHEMA_VERSION, "cohort_sha256": cohort_hash(),
        "city_count": len(expected),
        "grid_emission_factor_fallback": load_cohort_contract()["grid_emission_factor_fallback"],
        "artifacts": records,
    })
    additive = ["total_ebike_lifecycle", "total_drone_lifecycle", "oracle_min_lifecycle_kg", "total_odpair_annual_sales"]
    new_stats = pd.read_csv(target / "all_region_stats.csv")
    totals = {}
    for name in additive:
        old = float(pd.to_numeric(source_stats[name]).sum())
        removed = float(pd.to_numeric(source_stats.loc[source_stats.region_key.isin(EXCLUDED_KEYS), name]).sum())
        new = float(pd.to_numeric(new_stats[name]).sum())
        np.testing.assert_allclose(new, old - removed, rtol=1e-12, atol=1e-8)
        totals[name] = {"old": old, "removed": removed, "new": new}
    result = {"source_work_root": str(source_work_root), "source_cohort_sha256": SOURCE_COHORT_SHA256,
              "cohort_sha256": cohort_hash(), "old_city_count": 313, "new_city_count": 311,
              "removed_region_keys": sorted(EXCLUDED_KEYS), "unchanged_city_count": 311,
              "tables": table_audit, "curves": curve_audit, "additive_totals": totals,
              "reuse_directory": str(reuse), "analysis_modules": list(MIGRATED_MODULES),
              "affected_requests": list(AFFECTED_REQUESTS)}
    write_json(output / "migration_report.json", result)
    return result


def stage_review(seed: Path, output: Path) -> dict:
    """Copy accepted unaffected PDFs and cohort-independent SVG templates."""
    from drone_emission.paper_workflow import load_manifest

    manifest = load_manifest()
    figures = [item for item in manifest["figures"] if item["mode"] == "inkscape"]
    requests = {item["id"]: item for item in manifest["requests"]}
    unchanged = {key: item for key, item in requests.items()
                 if key not in AFFECTED_REQUESTS and key != "objective_tradeoffs"}
    if len(figures) != 12 or len(unchanged) != 16:
        raise ValueError("review manifest scope changed")
    # Keep the actual fixed-city inputs alongside the reused figure, so its
    # source report does not point at absent files in the new work root.
    source_status = json.loads((output / "source_status.json").read_text())
    fixed = source_status["modules"]["four-city-delivery-modes"]
    if fixed["status"] != "complete":
        raise ValueError("fixed-city source analysis is incomplete")
    fixed_inputs = []
    for record in fixed["outputs"]:
        source = Path(record["path"])
        if source.stat().st_size != record["size_bytes"] or _sha256(source) != record["sha256"]:
            raise ValueError(f"fixed-city source analysis changed: {source}")
        target = output / "work/analysis/four_city_delivery_modes" / source.name
        target.parent.mkdir(parents=True, exist_ok=True)
        if source.suffix == ".csv":
            keys = set(pd.read_csv(source, usecols=["region_key"]).region_key)
            if keys != {"chengdu", "guangzhou", "beijing", "shanghai"}:
                raise ValueError("fixed-city analysis scope changed")
            shutil.copy2(source, target)
        else:
            report = json.loads(source.read_text())
            if report["paper_cohort_sha256"] != SOURCE_COHORT_SHA256:
                raise ValueError("fixed-city report has an unexpected source cohort")
            report["paper_cohort_sha256"] = cohort_hash()
            report["cohort_migration"] = {
                "source_report": str(source), "source_sha256": record["sha256"],
                "source_cohort_sha256": SOURCE_COHORT_SHA256,
                "source_aggregate_input": report.get("aggregate_input"),
                "method": "reuse four unchanged cities; no numerical recomputation",
            }
            aggregate = output / "data/all_region_stats.csv"
            report["aggregate_input"] = {
                "path": "all_region_stats.csv",
                "size_bytes": aggregate.stat().st_size,
                "sha256": _sha256(aggregate),
            }
            write_json(target, report)
        fixed_inputs.append({"source": str(source), "source_sha256": record["sha256"],
                             "target": str(target), "sha256": _sha256(target)})
    records = []
    for key, request in unchanged.items():
        prepared = json.loads((seed / "bundle" / key / "prepared.json").read_text())
        if (prepared["cohort_sha256"] != SOURCE_COHORT_SHA256
                or prepared["request_version"] != request["cache_version"]
                or prepared["outputs"] != request["outputs"]):
            raise ValueError(f"accepted PDF request contract differs: {key}")
        # These sixteen requests have no national sample dependency. The four
        # fixed-city mode requests retain precisely the same four cities.
        if any(name.startswith("all_") for name in request["required_inputs"]):
            raise ValueError(f"cannot reuse a nationwide request: {key}")
        for name in request["outputs"]:
            source = seed / "figures/paper/subfigure" / name
            if not source.read_bytes().startswith(b"%PDF-"):
                raise ValueError(f"invalid accepted PDF: {source}")
            target = output / "figures/paper/subfigure" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                raise FileExistsError(f"refusing to replace a rendered panel: {target}")
            shutil.copy2(source, target)
            records.append({"request_id": key, "file": name, "sha256": _sha256(target),
                            "method": "reuse accepted city-independent or fixed-city PDF",
                            "source_prepared_sha256": _sha256(seed / "bundle" / key / "prepared.json")})
    for figure in figures:
        if set(figure["render_requests"]) & set(AFFECTED_REQUESTS):
            continue
        source = seed / "figures/paper" / Path(figure["staging"]).name
        target = output / "figures/paper" / source.name
        shutil.copy2(source, target)
        records.append({"figure_id": figure["id"], "file": source.name,
                        "sha256": _sha256(target), "method": "reuse unchanged composite"})
    bundle = output / "cache/visualization/prepared_paper"
    bundle.mkdir(parents=True, exist_ok=True)
    old_index = json.loads((seed / "bundle/bundle.json").read_text())
    if old_index["cohort_sha256"] != SOURCE_COHORT_SHA256:
        raise ValueError("composition seed is not the accepted old cohort")
    index_path = bundle / "bundle.json"
    index = json.loads(index_path.read_text()) if index_path.exists() else {"schema_version": 1}
    index.update(cohort_sha256=cohort_hash(), composition=old_index["composition"])
    index["composition_migration"] = {"source_sha256": _sha256(seed / "bundle/bundle.json"),
                                       "reason": "unchanged layout; all scientific panels accounted for separately"}
    for figure in figures:
        spec = index["composition"][figure["id"]]
        source = (seed / "bundle" / spec["file"]).resolve()
        if (seed / "bundle").resolve() not in source.parents or _sha256(source) != spec["sha256"]:
            raise ValueError(f"composition template changed: {source}")
        dest = bundle / spec["file"]
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, dest)
    write_json(index_path, index)
    write_json(output / "figures/paper/report/drawing_bundle.json", {"path": str(bundle)})
    report = {"source_seed": str(seed), "cohort_sha256": cohort_hash(),
              "reused_requests": list(unchanged), "outputs": records,
              "fixed_city_inputs": fixed_inputs}
    write_json(output / "figure_reuse_report.json", report)
    return report


def reuse_break_even_cache(config: dict, source_cache: Path, output: Path) -> dict:
    """Retain the unchanged observed sample; never rescan 311 city OD files."""
    import pickle
    from drone_emission.analysis_workflow import analysis_data_dir
    from drone_emission.visualization.contribution_renderers import (
        BREAK_EVEN_PHASE_CACHE_VERSION, _file_cache_signature, _get_ebike_kappa,
    )
    from drone_emission.visualization.contribution_break_even import (
        CONTRIBUTION_SAMPLE_LIMIT, CONTRIBUTION_SAMPLE_SEED,
        _break_even_mesh_coefficients, _build_break_even_scenarios,
    )

    data = analysis_data_dir(config)
    with source_cache.open("rb") as handle:
        cache = pickle.load(handle)
    meta, prepared = cache["meta"], cache["prepared"]
    expected_meta = {"version": BREAK_EVEN_PHASE_CACHE_VERSION,
                     "figure_name": "break_even_phase_diagram",
                     "kappa": _get_ebike_kappa(config),
                     "max_sample_rows": CONTRIBUTION_SAMPLE_LIMIT,
                     "sample_seed": CONTRIBUTION_SAMPLE_SEED}
    for key, value in expected_meta.items():
        if meta[key] != value:
            raise ValueError(f"break-even cache contract changed: {key}")
    gef = data / "gef_china.csv"
    if (_sha256(Path(meta["gef_file"]["path"])) != _sha256(gef)
            or gef.stat().st_size != meta["gef_file"]["size"]
            or gef.stat().st_mtime_ns != meta["gef_file"]["mtime_ns"]):
        raise ValueError("break-even GEF input changed")
    expected = set(locked_region_keys())
    fallback = set(load_cohort_contract()["grid_emission_factor_fallback"]["affected_region_keys"])
    observed = prepared["panel_c"]
    keys = {item["region_key"] for item in observed["city_reports"]}
    if keys != expected - fallback or set(observed["samples"].region_key) != keys:
        raise ValueError("break-even sample differs from the existing non-missing-GEF sample")
    stats = pd.read_csv(data / "all_region_stats.csv").set_index("region_key")
    enriched = pd.read_csv(data / "all_region_stats_enriched.csv").set_index("region_key")
    for record in observed["city_reports"]:
        key = record["region_key"]
        od = data / "city" / key / f"{key}_od_pairs.csv"
        if od.stat().st_mtime_ns > source_cache.stat().st_mtime_ns:
            raise ValueError(f"OD file is newer than the observed cache: {key}")
        if record["sample"]["valid_rows"] != int(stats.loc[key, "valid_path_count"]):
            raise ValueError(f"break-even valid OD count changed: {key}")
        np.testing.assert_allclose(record["gef"], enriched.loc[key, "grid_emission_factor"], rtol=0, atol=1e-12)
    # The analytic model must also reproduce the accepted mesh before reuse.
    ebike, drone, ebike_scenario, drone_scenario = _build_break_even_scenarios(config)
    zero, slope = _break_even_mesh_coefficients(
        prepared["air_distance_grid_km"], prepared["detour_ratios"],
        ebike, drone, ebike_scenario, drone_scenario,
    )
    np.testing.assert_allclose(zero, prepared["panel_a"]["delta_zero"], rtol=1e-12, atol=1e-10)
    np.testing.assert_allclose(slope, prepared["panel_a"]["delta_slope"], rtol=1e-12, atol=1e-10)
    expected_meta.update(stats_file=_file_cache_signature(str(data / "all_region_stats.csv")),
                         gef_file=_file_cache_signature(str(gef)))
    destination = Path(config["cache_dir"]) / "visualization/paper/break_even_phase_diagram_prepared.pkl"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(destination)
    report = {"source": str(source_cache), "source_sha256": _sha256(source_cache),
              "source_meta": meta, "target_meta": expected_meta,
              "observed_city_count": len(keys), "observed_sample_rows": len(observed["samples"]),
              "cohort_city_count": len(expected), "omitted_missing_gef_cities": sorted(fallback),
              "reason": "existing renderer omits missing-GEF cities in both cohorts; observed samples and analytic mesh are unchanged"}
    with destination.open("wb") as handle:
        pickle.dump({"meta": expected_meta, "prepared": prepared, "migration": report}, handle,
                    protocol=pickle.HIGHEST_PROTOCOL)
    write_json(output / "break_even_cache_reuse.json", report)
    return report


def finalize_review(config: dict, output: Path) -> dict:
    from drone_emission.analysis_workflow import analysis_data_dir, require_current_analysis
    from drone_emission.paper_workflow import _write_source_report, figure_required_outputs, load_manifest

    status = require_current_analysis(config, required_modules=set(MIGRATED_MODULES))
    view = dict(config, data_dir=str(analysis_data_dir(config)),
                analysis_input_manifest_sha256=status["input_manifest_sha256"])
    manifest = load_manifest()
    figures = [item for item in manifest["figures"] if item["mode"] == "inkscape"]
    reused = json.loads((output / "figure_reuse_report.json").read_text())
    for record in reused["outputs"]:
        folder = "subfigure" if "request_id" in record else ""
        path = Path(config["figures_dir"]) / "paper" / folder / record["file"]
        if _sha256(path) != record["sha256"]:
            raise ValueError(f"reused output was unexpectedly redrawn: {path}")
    records = []
    for figure in figures:
        paths = [Path(config["figures_dir"]) / "paper/subfigure" / name
                 for name in figure_required_outputs(figure, manifest)]
        paths.append(Path(config["figures_dir"]) / "paper" / Path(figure["staging"]).name)
        for path in paths:
            if not path.read_bytes().startswith(b"%PDF-"):
                raise ValueError(f"missing or invalid review PDF: {path}")
        report_path = _write_source_report(figure, view, paths, manifest)
        report = json.loads(report_path.read_text())
        report["cohort_migration"] = str(output / "migration_report.json")
        report["reused_request_ids"] = [key for key in figure["render_requests"] if key in reused["reused_requests"]]
        write_json(report_path, report)
        records.append({"figure_id": figure["id"], "pdf": str(paths[-1]), "sha256": _sha256(paths[-1])})
    result = {"cohort_sha256": cohort_hash(), "city_count": 311,
              "reused_outputs_verified": True, "figures": records}
    write_json(output / "review_verification.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-work-root", type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument("--stage-review", type=Path, metavar="SEED_ROOT")
    actions.add_argument("--finalize-review", action="store_true")
    actions.add_argument("--reuse-break-even-cache", type=Path)
    parser.add_argument("--env", default="remote")
    args = parser.parse_args()
    from drone_emission.cli import _config

    output = args.output_root.resolve()
    if args.stage_review:
        result = stage_review(args.stage_review.resolve(), output)
    elif args.finalize_review:
        result = finalize_review(_config(args.env), output)
    elif args.reuse_break_even_cache:
        result = reuse_break_even_cache(_config(args.env), args.reuse_break_even_cache.resolve(), output)
    elif args.source_work_root:
        result = migrate(_config(args.env), args.source_work_root.resolve(), output)
    else:
        parser.error("migration requires --source-work-root")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
