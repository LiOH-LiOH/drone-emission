"""Stage remote subfigures for local-only composition, without city data."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

from drone_emission.analysis_workflow import _canonical_sha256, _sha256
from drone_emission.paper_cohort import cohort_hash, load_cohort_contract
from drone_emission.paper_workflow import (
    _write_source_report, figure_required_outputs, load_manifest,
)
from drone_emission.tools.migrate_cohort_exclusion import (
    AFFECTED_REQUESTS, EXCLUDED_KEYS, MIGRATED_MODULES, SOURCE_COHORT_SHA256,
    write_json,
)


def verify_remote_inputs(root: Path) -> tuple[dict, dict, list[dict]]:
    reports = root / "reports"
    migration = json.loads((reports / "migration_report.json").read_text())
    if (migration["cohort_sha256"] != cohort_hash()
            or migration["new_city_count"] != 311
            or set(migration["removed_region_keys"]) != EXCLUDED_KEYS
            or not migration["curves"]["original_curve_reproduced"]):
        raise ValueError("downloaded migration provenance differs from reviewed subtraction")
    status = json.loads((reports / "analysis_status.json").read_text())
    manifest = json.loads((reports / "input_manifest.json").read_text())
    body = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    if (status["paper_cohort_sha256"] != cohort_hash()
            or manifest["cohort_sha256"] != cohort_hash()
            or manifest["city_count"] != 311
            or _canonical_sha256(body) != manifest["manifest_sha256"]
            or status["input_manifest_sha256"] != manifest["manifest_sha256"]):
        raise ValueError("downloaded analysis manifest is inconsistent")
    if any(status["modules"].get(key, {}).get("status") != "complete" for key in MIGRATED_MODULES):
        raise ValueError("required remote analysis modules are incomplete")
    records = json.loads((reports / "remote_subfigures.json").read_text())
    if isinstance(records, dict):
        records = records["files"]
    requests = load_manifest()["requests"]
    expected = {name for request in requests if request["id"] in AFFECTED_REQUESTS
                for name in request["outputs"]}
    if {item["filename"] for item in records} != expected or len(records) != len(expected):
        raise ValueError("downloaded subfigure list differs from the 15 selected requests")
    for record in records:
        path = root / "figures/paper/subfigure" / record["filename"]
        if (path.stat().st_size != record["size_bytes"]
                or _sha256(path) != record["sha256"]
                or not path.read_bytes().startswith(b"%PDF-")):
            raise ValueError(f"downloaded PDF failed verification: {path}")
    return status, manifest, records


def stage(root: Path, accepted: Path) -> dict:
    _, manifest, incoming = verify_remote_inputs(root)
    spec = load_manifest()
    figures = [item for item in spec["figures"] if item["mode"] == "inkscape"]
    unchanged = [item for item in spec["requests"]
                 if item["id"] not in AFFECTED_REQUESTS and item["id"] != "objective_tradeoffs"]
    if len(figures) != 12 or len(unchanged) != 16:
        raise ValueError("local review figure scope changed")
    provenance = root / "reports/local_composition_provenance.json"
    if provenance.exists():
        raise FileExistsError("local composition has already been staged")
    reused = []

    def retain(source: Path, relative: Path, **metadata):
        target = root / "figures" / relative
        if target.exists():
            raise FileExistsError(f"refusing to overwrite a staged artifact: {target}")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        reused.append({"source": str(source), "relative": str(relative),
                       "sha256": _sha256(target), **metadata})

    for request in unchanged:
        for name in request["outputs"]:
            retain(accepted / "paper/subfigure" / name, Path("paper/subfigure") / name,
                   request_id=request["id"], kind="unchanged_subfigure")
            svg = Path(name).with_suffix(".svg").name
            retain(accepted / "inkscape/linked-subfigures" / svg,
                   Path("inkscape/linked-subfigures") / svg,
                   request_id=request["id"], kind="unchanged_linked_svg")
    for figure in figures:
        if set(figure["render_requests"]) & set(AFFECTED_REQUESTS):
            continue
        name = Path(figure["staging"]).name
        retain(accepted / "paper" / name, Path("paper") / name,
               figure_id=figure["id"], kind="unchanged_composite")
    old_bundle = Path(json.loads((accepted / "paper/report/drawing_bundle.json").read_text())["path"])
    old_index = json.loads((old_bundle / "bundle.json").read_text())
    if old_index["cohort_sha256"] != SOURCE_COHORT_SHA256:
        raise ValueError("accepted local layout bundle has a different source cohort")
    bundle = root / "composition_bundle"
    for figure in figures:
        item = old_index["composition"][figure["id"]]
        source = (old_bundle / item["file"]).resolve()
        if old_bundle.resolve() not in source.parents or _sha256(source) != item["sha256"]:
            raise ValueError(f"accepted SVG template hash mismatch: {source}")
        target = bundle / item["file"]
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    write_json(bundle / "bundle.json", {
        "schema_version": 1, "cohort_sha256": cohort_hash(),
        "composition": old_index["composition"],
        "composition_migration": {"source": str(old_bundle),
                                   "source_sha256": _sha256(old_bundle / "bundle.json"),
                                   "method": "reuse unchanged local SVG layouts only"},
    })
    write_json(root / "figures/paper/report/drawing_bundle.json", {"path": str(bundle)})
    result = {"assembly_host": "local", "render_host": "hku-sto",
              "cohort_sha256": cohort_hash(), "city_count": 311,
              "analysis_input_manifest_sha256": manifest["manifest_sha256"],
              "downloaded_subfigures": incoming, "reused_artifacts": reused,
              "affected_requests": list(AFFECTED_REQUESTS),
              "affected_figures": [item["id"] for item in figures
                                   if set(item["render_requests"]) & set(AFFECTED_REQUESTS)],
              "all_figures": [item["id"] for item in figures]}
    write_json(provenance, result)
    return result


def verify(root: Path) -> dict:
    status, manifest, _ = verify_remote_inputs(root)
    provenance = json.loads((root / "reports/local_composition_provenance.json").read_text())
    for record in provenance["reused_artifacts"]:
        if _sha256(root / "figures" / record["relative"]) != record["sha256"]:
            raise ValueError(f"unchanged artifact was modified: {record['relative']}")
    spec = load_manifest()
    remote_data = Path(status["analysis_data_dir"])
    remote_work = remote_data.parents[2]
    config = {"data_dir": str(remote_data), "work_dir": str(remote_work),
              "cache_dir": str(remote_work.parent / "cache"),
              "figures_dir": str(root / "figures"),
              "analysis_input_manifest_sha256": manifest["manifest_sha256"]}
    results = []
    for figure in spec["figures"]:
        if figure["id"] not in provenance["all_figures"]:
            continue
        outputs = [root / "figures/paper/subfigure" / name
                   for name in figure_required_outputs(figure, spec)]
        outputs.append(root / "figures/paper" / Path(figure["staging"]).name)
        if any(not path.read_bytes().startswith(b"%PDF-") for path in outputs):
            raise ValueError(f"missing or invalid composed PDF: {figure['id']}")
        path = _write_source_report(figure, config, outputs, spec)
        report = json.loads(path.read_text())
        report["assembly_host"] = "local"
        report["composition_provenance"] = str(root / "reports/local_composition_provenance.json")
        report["migration_report"] = str(root / "reports/migration_report.json")
        write_json(path, report)
        results.append({"figure_id": figure["id"], "pdf": str(outputs[-1]),
                        "sha256": _sha256(outputs[-1])})
    result = {"assembly_host": "local", "cohort_sha256": cohort_hash(),
              "city_count": load_cohort_contract()["expected_city_count"],
              "downloaded_pdfs_verified": True, "reused_artifacts_unchanged": True,
              "figures": results}
    write_json(root / "reports/review_verification.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--accepted-figures-root", type=Path)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    root = args.output_root.resolve()
    if args.verify:
        result = verify(root)
    elif args.accepted_figures_root:
        result = stage(root, args.accepted_figures_root.resolve())
    else:
        parser.error("staging requires --accepted-figures-root")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
