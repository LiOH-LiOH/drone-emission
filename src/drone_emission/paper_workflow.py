"""Manifest-driven rendering and publication of the locked paper figures."""

from __future__ import annotations

import json
import multiprocessing
import shutil
import time
import traceback
from importlib.resources import files
from pathlib import Path

from .paper_cohort import cohort_hash
from .paths import PROJECT_ROOT

EXPECTED_FIGURE_COUNT = 11
EXPECTED_RENDER_REQUEST_COUNT = 24


def load_manifest() -> dict:
    """Load and structurally validate the locked paper-figure manifest."""
    path = files("drone_emission").joinpath("paper_figures.json")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    figures = manifest.get("figures", [])
    requests = manifest.get("requests", [])
    figure_ids = [figure.get("id") for figure in figures]
    request_ids = [request.get("id") for request in requests]
    if manifest.get("schema_version") != 2:
        raise ValueError("Unsupported paper figure manifest schema")
    if manifest.get("figure_count") != EXPECTED_FIGURE_COUNT:
        raise ValueError(f"Manifest figure_count must be exactly {EXPECTED_FIGURE_COUNT}")
    if manifest.get("render_request_count") != EXPECTED_RENDER_REQUEST_COUNT:
        raise ValueError(
            "Manifest render_request_count must be exactly "
            f"{EXPECTED_RENDER_REQUEST_COUNT}"
        )
    if len(figures) != EXPECTED_FIGURE_COUNT or len(set(figure_ids)) != len(figure_ids):
        raise ValueError(
            f"Manifest must contain {EXPECTED_FIGURE_COUNT} uniquely identified figures"
        )
    if len(requests) != EXPECTED_RENDER_REQUEST_COUNT or len(set(request_ids)) != len(request_ids):
        raise ValueError(
            "Manifest must contain "
            f"{EXPECTED_RENDER_REQUEST_COUNT} uniquely identified render requests"
        )

    declared = set(request_ids)
    referenced: list[str] = []
    for request in requests:
        if not isinstance(request.get("cache_version"), int):
            raise ValueError(f"Render request has no integer cache_version: {request.get('id')}")
        if not isinstance(request.get("required_inputs"), list):
            raise ValueError(f"Render request has invalid inputs: {request.get('id')}")
        outputs = request.get("outputs", [])
        if not outputs or any(
            Path(name).name != name or not name.endswith(".pdf")
            for name in outputs
        ):
            raise ValueError(f"Render request has invalid PDF outputs: {request.get('id')}")

    for figure in figures:
        if figure.get("mode") not in {"direct", "inkscape"}:
            raise ValueError(f"Invalid render mode for {figure.get('id')}")
        expected_name = f"{figure['id']}.pdf"
        if Path(figure["staging"]).name != expected_name:
            raise ValueError(f"Staging filename mismatch for {figure['id']}")
        if Path(figure["publish"]).name != expected_name:
            raise ValueError(f"Publish filename mismatch for {figure['id']}")
        figure_requests = figure.get("render_requests", [])
        if not figure_requests or not set(figure_requests) <= declared:
            raise ValueError(f"Figure request contract is incomplete for {figure['id']}")
        referenced.extend(figure_requests)
        reports = figure.get("report_inputs", [])
        if len(reports) != 1 or not reports[0].endswith(".sources.json"):
            raise ValueError(f"Figure source-report mapping is invalid for {figure['id']}")
        if figure["mode"] == "direct" and not figure.get("source_output"):
            raise ValueError(f"Direct figure has no source_output: {figure['id']}")
        if figure["mode"] == "inkscape" and not figure.get("inkscape_source"):
            raise ValueError(f"Inkscape figure has no source: {figure['id']}")

    if len(referenced) != len(set(referenced)) or set(referenced) != declared:
        raise ValueError("Each manifest render request must be referenced exactly once")
    from .visualization.formal_registry import REQUEST_RENDERERS

    registered = set(REQUEST_RENDERERS)
    if declared != registered:
        raise ValueError(
            "Manifest/renderer registry mismatch: "
            f"missing={sorted(declared - registered)}, "
            f"unused={sorted(registered - declared)}"
        )
    return manifest


def figure_index(manifest: dict | None = None) -> dict[str, dict]:
    manifest = load_manifest() if manifest is None else manifest
    return {figure["id"]: figure for figure in manifest["figures"]}


def request_index(manifest: dict | None = None) -> dict[str, dict]:
    manifest = load_manifest() if manifest is None else manifest
    return {request["id"]: request for request in manifest["requests"]}


def select_figures(only: list[str] | None) -> list[dict]:
    index = figure_index()
    if not only:
        return list(index.values())
    unknown = sorted(set(only).difference(index))
    if unknown:
        raise ValueError(
            "Unknown formal figure ID(s): " + ", ".join(unknown)
            + ". render accepts manifest IDs only."
        )
    return [index[figure_id] for figure_id in dict.fromkeys(only)]


def select_requests(requests: list[str] | None) -> list[dict]:
    index = request_index()
    if not requests:
        return list(index.values())
    unknown = sorted(set(requests).difference(index))
    if unknown:
        raise ValueError("Unknown formal render request(s): " + ", ".join(unknown))
    return [index[request_id] for request_id in dict.fromkeys(requests)]


def figure_requests(figure: dict, manifest: dict | None = None) -> list[dict]:
    index = request_index(manifest)
    return [index[request_id] for request_id in figure["render_requests"]]


def figure_required_inputs(figure: dict, manifest: dict | None = None) -> list[str]:
    return list(
        dict.fromkeys(
            declaration
            for request in figure_requests(figure, manifest)
            for declaration in request["required_inputs"]
        )
    )


def figure_required_outputs(figure: dict, manifest: dict | None = None) -> list[str]:
    return [
        output
        for request in figure_requests(figure, manifest)
        for output in request["outputs"]
    ]


def _valid_pdf(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size < 8:
        return False
    with path.open("rb") as handle:
        return handle.read(5) == b"%PDF-"


def _declaration_root(config: dict, declaration: str) -> tuple[Path, str]:
    prefix, separator, relative = declaration.partition(":")
    roots = {
        "cache": Path(config["cache_dir"]),
        "work": Path(config["work_dir"]),
        "package": PROJECT_ROOT / "src" / "drone_emission",
        "project": PROJECT_ROOT,
    }
    if separator and prefix in roots:
        return roots[prefix], relative
    return Path(config["data_dir"]), declaration


def _resolve_request_inputs(request: dict, config: dict) -> list[Path]:
    matches: list[Path] = []
    missing: list[str] = []
    for declaration in request["required_inputs"]:
        root, pattern = _declaration_root(config, declaration)
        resolved = sorted(path for path in root.glob(pattern) if path.is_file())
        if not resolved:
            missing.append(str(root / pattern))
        matches.extend(resolved)
    if missing:
        raise FileNotFoundError(
            f"{request['id']} missing render input(s):\n- "
            + "\n- ".join(missing)
            + "\nRun the declared prerequisite stage first."
        )
    return sorted(set(path.resolve() for path in matches), key=str)


def _file_signature(path: Path) -> dict:
    stat = path.stat()
    return {
        "path": str(path),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def _request_signature(request: dict, config: dict) -> dict:
    return {
        "cache_version": request["cache_version"],
        "paper_cohort_sha256": cohort_hash(),
        "inputs": [_file_signature(path) for path in _resolve_request_inputs(request, config)],
    }


def _request_outputs(request: dict, config: dict) -> list[Path]:
    root = Path(config["figures_dir"]) / "paper" / "subfigure"
    return [root / name for name in request["outputs"]]


def _receipt_path(request: dict, config: dict) -> Path:
    return (
        Path(config["cache_dir"])
        / "visualization"
        / "paper"
        / "render_requests"
        / f"{request['id']}.json"
    )


def _request_is_fresh(request: dict, config: dict, signature: dict | None = None) -> bool:
    receipt_path = _receipt_path(request, config)
    if not receipt_path.is_file():
        return False
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        expected_signature = signature or _request_signature(request, config)
        outputs = _request_outputs(request, config)
        if receipt.get("signature") != expected_signature:
            return False
        if any(not _valid_pdf(path) for path in outputs):
            return False
        return receipt.get("outputs") == [_file_signature(path.resolve()) for path in outputs]
    except (OSError, ValueError, json.JSONDecodeError):
        return False


def _write_receipt(request: dict, config: dict, signature: dict, elapsed: float) -> None:
    path = _receipt_path(request, config)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "request_id": request["id"],
        "signature": signature,
        "outputs": [
            _file_signature(output.resolve()) for output in _request_outputs(request, config)
        ],
        "elapsed_seconds": elapsed,
    }
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _render_request_worker(config: dict, request_id: str) -> None:
    """Run exactly one renderer in a short-lived process."""
    try:
        from drone_emission.utils.config import configure_cache
        from drone_emission.visualization.formal_registry import render_request
        from drone_emission.visualization.render_context import RenderContext

        configure_cache(config)
        render_request(RenderContext.from_config(config), request_id)
    except Exception:
        traceback.print_exc()
        raise


def _execute_request(request: dict, config: dict, *, force: bool) -> dict:
    started = time.monotonic()
    request_id = request["id"]
    try:
        signature = _request_signature(request, config)
        if not force and _request_is_fresh(request, config, signature):
            print(f"[render:{request_id}] cache-hit", flush=True)
            return {"id": request_id, "status": "cached", "elapsed": 0.0}
        receipt = _receipt_path(request, config)
        receipt.unlink(missing_ok=True)
        print(f"[render:{request_id}] start", flush=True)
        process = multiprocessing.get_context("spawn").Process(
            target=_render_request_worker,
            args=(config, request_id),
        )
        process.start()
        process.join()
        if process.exitcode != 0:
            raise RuntimeError(f"renderer process exited with status {process.exitcode}")
        invalid = [str(path) for path in _request_outputs(request, config) if not _valid_pdf(path)]
        if invalid:
            raise RuntimeError("renderer did not produce valid PDF(s): " + ", ".join(invalid))
        elapsed = time.monotonic() - started
        _write_receipt(request, config, signature, elapsed)
        print(f"[render:{request_id}] success {elapsed:.1f}s", flush=True)
        return {"id": request_id, "status": "success", "elapsed": elapsed}
    except Exception as exc:
        elapsed = time.monotonic() - started
        _receipt_path(request, config).unlink(missing_ok=True)
        print(
            f"[render:{request_id}] failure {elapsed:.1f}s: "
            f"{type(exc).__name__}: {exc}",
            flush=True,
        )
        return {
            "id": request_id,
            "status": "failed",
            "elapsed": elapsed,
            "error": f"{type(exc).__name__}: {exc}",
        }


def _write_source_report(
    figure: dict,
    config: dict,
    outputs: list[Path],
    manifest: dict,
) -> Path:
    figures_root = Path(config["figures_dir"])
    relative = Path(figure["report_inputs"][0])
    if relative.parts[:1] == ("figures",):
        relative = Path(*relative.parts[1:])
    report_path = figures_root / relative
    report_path.parent.mkdir(parents=True, exist_ok=True)
    from .paper_cohort import load_cohort_contract

    cohort = load_cohort_contract()
    payload = {
        "schema_version": 2,
        "figure_id": figure["id"],
        "mode": figure["mode"],
        "data_root": str(Path(config["data_dir"]).resolve()),
        "cache_root": str(Path(config["cache_dir"]).resolve()),
        "work_root": str(Path(config["work_dir"]).resolve()),
        "required_inputs": figure_required_inputs(figure, manifest),
        "render_requests": figure["render_requests"],
        "outputs": [str(path.resolve()) for path in outputs],
        "inkscape_source": figure.get("inkscape_source"),
        "publish": figure["publish"],
        "paper_cohort_id": cohort["cohort_id"],
        "paper_cohort_sha256": cohort["included_region_keys_sha256"],
        "paper_city_count": cohort["expected_city_count"],
        "grid_emission_factor_fallback": cohort["grid_emission_factor_fallback"],
    }
    temporary = report_path.with_suffix(report_path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(report_path)
    return report_path


def render(
    config: dict,
    only: list[str] | None = None,
    requests: list[str] | None = None,
    *,
    force: bool = False,
) -> list[Path]:
    """Render selected request bundles and stage completed direct figures."""
    if only and requests:
        raise ValueError("render --only and --request are mutually exclusive")
    manifest = load_manifest()
    if requests:
        selected_requests = select_requests(requests)
        selected_ids = {request["id"] for request in selected_requests}
        selected_figures = [
            figure
            for figure in manifest["figures"]
            if selected_ids.intersection(figure["render_requests"])
        ]
    else:
        selected_figures = select_figures(only)
        selected_ids = {
            request_id
            for figure in selected_figures
            for request_id in figure["render_requests"]
        }
        selected_requests = [
            request for request in manifest["requests"] if request["id"] in selected_ids
        ]

    figures_root = Path(config["figures_dir"])
    paper_dir = figures_root / "paper"
    subfigure_dir = paper_dir / "subfigure"
    directories = (
        paper_dir,
        subfigure_dir,
        Path(config["work_dir"]),
        Path(config["cache_dir"]),
    )
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)

    results = {
        result["id"]: result
        for result in (
            _execute_request(request, config, force=force)
            for request in selected_requests
        )
    }
    staged: list[Path] = []
    request_specs = request_index(manifest)
    for figure in selected_figures:
        contracts = [request_specs[name] for name in figure["render_requests"]]
        all_fresh = all(_request_is_fresh(contract, config) for contract in contracts)
        if not all_fresh:
            continue
        figure_outputs = [
            subfigure_dir / name
            for name in figure_required_outputs(figure, manifest)
        ]
        if figure["mode"] == "direct":
            source = subfigure_dir / figure["source_output"]
            target = paper_dir / Path(figure["staging"]).name
            shutil.copy2(source, target)
            staged.append(target)
            figure_outputs.append(target)
        _write_source_report(figure, config, figure_outputs, manifest)

    failed = [result for result in results.values() if result["status"] == "failed"]
    if failed:
        details = "\n".join(f"- {item['id']}: {item['error']}" for item in failed)
        raise RuntimeError(f"Formal render request failure(s):\n{details}")
    return staged


def promote(
    config: dict, *, yes: bool, dry_run: bool = False
) -> list[tuple[Path, Path]]:
    """Validate and copy all staged PDFs into the paper repository."""
    if not dry_run and not yes:
        raise PermissionError("promote requires explicit --yes confirmation")
    manifest = load_manifest()
    operations: list[tuple[Path, Path]] = []
    problems: list[str] = []
    paper_dir = Path(config["figures_dir"]) / "paper"
    for figure in manifest["figures"]:
        source = paper_dir / Path(figure["staging"]).name
        target = PROJECT_ROOT / figure["publish"]
        if not _valid_pdf(source):
            problems.append(f"missing or invalid staging PDF: {source}")
        if source.name != target.name or source.name != f"{figure['id']}.pdf":
            problems.append(f"filename mapping mismatch: {figure['id']}")
        operations.append((source, target))
    if problems:
        raise ValueError("Promotion validation failed:\n- " + "\n- ".join(problems))
    if not dry_run:
        for source, target in operations:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
    return operations
