"""Manifest-driven rendering, composition, and publication of paper figures."""

from __future__ import annotations

import json
import multiprocessing
import re
import shutil
import subprocess
import time
import traceback
import tempfile
import hashlib
import os
from dataclasses import replace
from importlib.resources import files
from pathlib import Path
from urllib.parse import unquote, urlparse
from xml.etree import ElementTree

from .paper_cohort import cohort_hash
from .paths import PROJECT_ROOT

EXPECTED_FIGURE_COUNT = 12
EXPECTED_RENDER_REQUEST_COUNT = 31
LINKED_SUBFIGURE_DIR = (
    PROJECT_ROOT / "figures" / "inkscape" / "linked-subfigures"
)
_LOCAL_URL_REFERENCE = re.compile(r"url\(#([^)]+)\)")


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


def select_inkscape_figures(only: list[str] | None) -> list[dict]:
    """Select formal figures whose final layout is an Inkscape SVG."""
    selected = select_figures(only)
    direct = [
        figure["id"] for figure in selected if figure["mode"] != "inkscape"
    ]
    if only and direct:
        raise ValueError(
            "compose accepts Inkscape figure IDs only: " + ", ".join(direct)
        )
    return [figure for figure in selected if figure["mode"] == "inkscape"]


def select_requests(requests: list[str] | None) -> list[dict]:
    index = request_index()
    if not requests:
        return list(index.values())
    unknown = sorted(set(requests).difference(index))
    if unknown:
        raise ValueError("Unknown formal render request(s): " + ", ".join(unknown))
    return [index[request_id] for request_id in dict.fromkeys(requests)]


def _analysis_modules_for_requests(
    requests: list[dict],
    config: dict,
) -> set[str]:
    from .analysis_workflow import _module_outputs

    output_modules = {
        path.name: module
        for module, paths in _module_outputs(config).items()
        for path in paths
    }
    selected: set[str] = set()
    for request in requests:
        for declaration in request["required_inputs"]:
            _, separator, relative = declaration.partition(":")
            name = Path(relative if separator else declaration).name
            if name in output_modules:
                selected.add(output_modules[name])
    return selected


def _bind_render_analysis(
    config: dict,
    requests: list[dict],
) -> dict:
    modules = _analysis_modules_for_requests(requests, config)
    if not modules:
        return config
    from .analysis_workflow import analysis_data_dir, require_current_analysis

    status = require_current_analysis(config, required_modules=modules)
    return dict(
        config,
        data_dir=str(analysis_data_dir(config)),
        analysis_input_manifest_sha256=status["input_manifest_sha256"],
    )


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


def _tool_path(name: str) -> Path:
    resolved = shutil.which(name)
    if resolved:
        return Path(resolved)
    if name == "inkscape":
        application = Path("/Applications/Inkscape.app/Contents/MacOS/inkscape")
        if application.is_file():
            return application
    raise FileNotFoundError(f"Required local command is unavailable: {name}")


def _is_newer(source: Path, target: Path) -> bool:
    return not target.is_file() or source.stat().st_mtime_ns > target.stat().st_mtime_ns


def _linked_subfigure_names(svg_path: Path) -> set[str]:
    """Return external SVG filenames linked from an Inkscape document."""
    root = ElementTree.fromstring(svg_path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for element in root.iter():
        for attribute, value in element.attrib.items():
            if attribute != "href" and not attribute.endswith("}href"):
                continue
            if value.startswith("data:"):
                continue
            path = unquote(urlparse(value).path)
            if path.endswith(".svg"):
                names.add(Path(path).name)
    return names


def _linked_subfigure_prefix(name: str) -> str:
    stem = Path(name).stem
    safe_stem = re.sub(r"[^A-Za-z0-9_.-]+", "-", stem).strip("-.")
    return f"linked-subfigure-{safe_stem or 'panel'}"


def _rewrite_local_svg_references(value: str, id_map: dict[str, str]) -> str:
    if value.startswith("#") and value[1:] in id_map:
        value = f"#{id_map[value[1:]]}"
    return _LOCAL_URL_REFERENCE.sub(
        lambda match: f"url(#{id_map.get(match.group(1), match.group(1))})",
        value,
    )


def _namespace_linked_svg_ids(tree: ElementTree.ElementTree, name: str) -> str:
    """Make imported IDs unique before several SVGs share one document."""
    root = tree.getroot()
    prefix = _linked_subfigure_prefix(name)
    root_id = f"{prefix}-root"
    id_map: dict[str, str] = {}
    for element in root.iter():
        old_id = element.get("id")
        if old_id:
            id_map[old_id] = (
                root_id if element is root else f"{prefix}-{old_id}"
            )

    for element in root.iter():
        old_id = element.get("id")
        if old_id:
            element.set("id", id_map[old_id])
        for attribute, value in list(element.attrib.items()):
            if attribute == "id":
                continue
            element.set(
                attribute,
                _rewrite_local_svg_references(value, id_map),
            )
        if element.text and "url(#" in element.text:
            element.text = _rewrite_local_svg_references(
                element.text,
                id_map,
            )
    root.set("id", root_id)
    return root_id


def _convert_linked_subfigure(
    source: Path,
    target: Path,
    *,
    inkscape: Path,
) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".tmp.svg")
    subprocess.run(
        [
            str(inkscape),
            str(source),
            "--pdf-font-strategy=keep",
            "--export-plain-svg",
            f"--export-filename={temporary}",
        ],
        check=True,
    )
    tree = ElementTree.parse(temporary)
    _namespace_linked_svg_ids(tree, target.name)
    tree.write(temporary, encoding="utf-8", xml_declaration=True)
    temporary.replace(target)


def _vector_export_source(source: Path, managed_names: set[str], *, linked_dir: Path | None = None) -> Path:
    """Inline linked SVG images in a temporary vector export document."""
    tree = ElementTree.parse(source)
    from .paper_svg_style import style_master
    master_report = style_master(tree.getroot(), linked_dir or LINKED_SUBFIGURE_DIR, managed_names, output=source.name)
    report_path = source.parent / "typography" / (source.stem + ".json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(master_report, ensure_ascii=False, indent=2) + "\n")
    converted: set[str] = set()
    for element in tree.getroot().iter():
        if element.tag.rsplit("}", 1)[-1] != "image":
            continue
        for attribute, value in list(element.attrib.items()):
            if attribute != "href" and not attribute.endswith("}href"):
                continue
            if value.startswith("data:"):
                continue
            name = Path(unquote(urlparse(value).path)).name
            if name not in managed_names:
                continue
            linked_root = ElementTree.parse(
                (linked_dir or LINKED_SUBFIGURE_DIR) / name
            ).getroot()
            namespace = element.tag.rsplit("}", 1)[0]
            svg_tag = (
                f"{namespace}}}svg" if element.tag.startswith("{") else "svg"
            )
            nested_svg = ElementTree.Element(svg_tag)
            for geometry_attribute in (
                "x",
                "y",
                "width",
                "height",
                "preserveAspectRatio",
            ):
                if geometry_attribute in element.attrib:
                    nested_svg.set(
                        geometry_attribute,
                        element.attrib.pop(geometry_attribute),
                    )
            element.tag = (
                f"{namespace}}}g" if element.tag.startswith("{") else "g"
            )
            for href_attribute in list(element.attrib):
                if href_attribute == "href" or href_attribute.endswith("}href"):
                    del element.attrib[href_attribute]
            view_box = linked_root.get("viewBox")
            if view_box:
                nested_svg.set("viewBox", view_box)
            nested_svg.text = linked_root.text
            nested_svg.extend(list(linked_root))
            element.append(nested_svg)
            converted.add(name)
            break
    missing = sorted(managed_names - converted)
    if missing:
        raise ValueError(
            f"{source} is missing managed linked image(s): "
            + ", ".join(missing)
        )
    temporary = source.with_name(f".{source.stem}.compose.tmp.svg")
    tree.write(temporary, encoding="utf-8", xml_declaration=True)
    return temporary


def _export_inkscape_figure(
    source: Path,
    target: Path,
    *,
    inkscape: Path,
) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".tmp.pdf")
    subprocess.run(
        [
            str(inkscape),
            str(source),
            "--export-type=pdf",
            "--export-area-drawing",
            "--export-margin=1",
            f"--export-filename={temporary}",
        ],
        check=True,
    )
    if not _valid_pdf(temporary):
        raise RuntimeError(f"Inkscape did not produce a valid PDF: {temporary}")
    temporary.replace(target)


def compose(
    config: dict,
    only: list[str] | None = None,
    *,
    prepare_links: bool = False,
    force: bool = False,
) -> list[Path]:
    """Refresh linked panels and export the selected Inkscape figures."""
    manifest = load_manifest()
    selected = select_inkscape_figures(only)
    figures_root = Path(config["figures_dir"])
    from .paper_preview import invalidate_preview, refresh_previews
    if not prepare_links:
        invalidate_preview(figures_root, "Composition has started; a full successful compose refreshes the overview.")
    subfigure_dir = figures_root / "paper" / "subfigure"
    paper_dir = figures_root / "paper"
    bundle_reference = paper_dir / "report" / "drawing_bundle.json"
    bundle = None
    bundle_index = None
    linked_dir = LINKED_SUBFIGURE_DIR
    if bundle_reference.is_file():
        from .visualization.prepared_bundle import file_hash
        reference = json.loads(bundle_reference.read_text())
        bundle = Path(reference["path"])
        bundle_index = json.loads((bundle / "bundle.json").read_text())
        if bundle_index["cohort_sha256"] != cohort_hash():
            raise ValueError("Composition bundle belongs to a different cohort")
        linked_dir = figures_root / "inkscape" / "linked-subfigures"
    inkscape = _tool_path("inkscape")
    converted: dict[str, Path] = {}

    for figure in selected:
        for output in figure_required_outputs(figure, manifest):
            source = subfigure_dir / output
            if not _valid_pdf(source):
                raise FileNotFoundError(
                    f"{figure['id']} missing valid subfigure PDF: {source}. "
                    "Run render first."
                )
            target = linked_dir / Path(output).with_suffix(".svg").name
            if force or _is_newer(source, target):
                _convert_linked_subfigure(source, target, inkscape=inkscape)
            converted[output] = target

    if prepare_links:
        return list(dict.fromkeys(converted.values()))

    staged: list[Path] = []
    for figure in selected:
        source = PROJECT_ROOT / figure["inkscape_source"]
        if bundle is not None:
            specification = bundle_index["composition"][figure["id"]]
            bundled_source = bundle / specification["file"]
            if file_hash(bundled_source) != specification["sha256"]:
                raise ValueError(f"Corrupt composition template: {bundled_source}")
            source = figures_root / "inkscape" / Path(figure["inkscape_source"]).name
            source.parent.mkdir(parents=True, exist_ok=True)
            if not source.exists() or file_hash(source) != specification["sha256"]:
                shutil.copy2(bundled_source, source)
        if not source.is_file():
            raise FileNotFoundError(f"Missing Inkscape source: {source}")
        required_names = {
            Path(output).with_suffix(".svg").name
            for output in figure_required_outputs(figure, manifest)
        }
        linked_names = _linked_subfigure_names(source)
        missing_links = sorted(required_names - linked_names)
        if missing_links:
            raise ValueError(
                f"{figure['id']} source SVG is missing linked subfigure(s): "
                + ", ".join(missing_links)
                + ". Import them from figures/inkscape/linked-subfigures "
                "using Link mode."
            )
        extra_names = {
            name
            for name in linked_names - required_names
            if _valid_pdf(subfigure_dir / Path(name).with_suffix(".pdf"))
            and (LINKED_SUBFIGURE_DIR / name).is_file()
        }
        for name in extra_names:
            panel_pdf = subfigure_dir / Path(name).with_suffix(".pdf")
            linked_svg = linked_dir / name
            if force or _is_newer(panel_pdf, linked_svg):
                _convert_linked_subfigure(
                    panel_pdf,
                    linked_svg,
                    inkscape=inkscape,
                )
        managed_names = required_names | extra_names
        target = paper_dir / Path(figure["staging"]).name
        dependencies = [source] + [
            converted[name]
            for name in figure_required_outputs(figure, manifest)
        ] + [linked_dir / name for name in sorted(extra_names)]
        from .visualization.typography import DEFAULT_TYPOGRAPHY
        dependencies.extend([DEFAULT_TYPOGRAPHY, Path(__file__), Path(__file__).with_name("paper_svg_style.py")])
        if force or not _valid_pdf(target) or any(
            _is_newer(dependency, target) for dependency in dependencies
        ):
            export_source = _vector_export_source(source, managed_names, linked_dir=linked_dir)
            try:
                _export_inkscape_figure(export_source, target, inkscape=inkscape)
            finally:
                export_source.unlink(missing_ok=True)
            print(f"Composed formal figure: {target}")
            staged.append(target)
        else:
            print(f"Composition is fresh: {target}")
    if {figure["id"] for figure in selected} == {figure["id"] for figure in manifest["figures"]}:
        refresh_previews(figures_root, manifest["figures"], rasterizer=_tool_path("pdftoppm"))
    else:
        print("Overview is stale after partial composition; run full compose to refresh it.")
    from .paper_typography_report import write_typography_summary
    write_typography_summary(figures_root)
    return staged


def _prepare_composition_bundle(root: Path, figures: list[dict]) -> None:
    from .visualization.prepared_bundle import file_hash, write_document

    index_path = root / "bundle.json"
    index = {"schema_version": 1, "cohort_sha256": cohort_hash(), "composition": {}}
    if index_path.is_file():
        previous = json.loads(index_path.read_text())
        if previous.get("cohort_sha256") == cohort_hash():
            index = previous
    for figure in figures:
        source = PROJECT_ROOT / figure["inkscape_source"]
        if not source.is_file():
            raise FileNotFoundError(f"Missing formal composition template: {source}")
        expected = {Path(name).with_suffix(".svg").name for name in figure_required_outputs(figure)}
        external = _linked_subfigure_names(source)
        if external != expected:
            raise ValueError(f"Composition template {source.name} has undeclared or missing panel links: {external ^ expected}")
        digest = file_hash(source)
        target = root / "composition" / f"{digest}.svg"
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            shutil.copy2(source, target)
        index["composition"][figure["id"]] = {
            "file": str(target.relative_to(root)), "sha256": digest,
        }
    write_document(index_path, index)


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
    from .visualization.prepared_bundle import drawing_code_hash, file_hash
    from .visualization.palette import load_palette
    from .visualization.typography import typography_signature

    usage = None
    if config.get("bundle_dir"):
        prepared_path = Path(config["bundle_dir"]) / request["id"] / "prepared.json"
        if prepared_path.is_file():
            usage = json.loads(prepared_path.read_text()).get("palette_usage")
    styling = {
        "typography_sha256": typography_signature(),
        "palette_sha256": load_palette(config.get("palette_path")).signature_for(usage),
        "drawing_code_sha256": drawing_code_hash(),
    }
    if config.get("from_bundle"):
        path = Path(config["from_bundle"]) / request["id"] / "prepared.json"
        return dict(styling, prepared_sha256=file_hash(path))
    return {
        **styling,
        "cache_version": request["cache_version"],
        "paper_cohort_sha256": cohort_hash(),
        "analysis_input_manifest_sha256": config.get(
            "analysis_input_manifest_sha256"
        ),
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
        "schema_version": 2,
        "request_id": request["id"],
        "analysis_input_manifest_sha256": config.get(
            "analysis_input_manifest_sha256"
        ),
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
    from .visualization.typography import paper_typography

    report_dir = Path(config["figures_dir"]) / "paper" / "report" / "typography"
    with paper_typography(report_dir):
        _render_request_worker_impl(config, request_id)


def _render_request_worker_impl(config: dict, request_id: str) -> None:
    """Run exactly one renderer in a short-lived process."""
    try:
        from drone_emission.utils.config import configure_cache
        from drone_emission.visualization.render_context import RenderContext
        from drone_emission.visualization.palette import palette_usage
        from drone_emission.visualization.prepared_bundle import (
            SCHEMA_VERSION, Codec, capture, drawing_code_hash, file_hash,
            replay, validate_bundle, write_document,
        )

        configure_cache(config)
        context = RenderContext.from_config(config)
        # Import module-level colour tables only after selecting the palette.
        from drone_emission.visualization.formal_registry import render_request

        request = request_index()[request_id]
        bundle_root = Path(config["bundle_dir"])
        roots = dict(data=context.data_root, figures=context.figures_root,
                     cache=context.cache_root, work=context.work_root,
                     project=PROJECT_ROOT, package=Path(__file__).parent)
        if config.get("from_bundle"):
            document = validate_bundle(bundle_root, request, cohort_hash())
            replay(bundle_root / request_id, document, roots)
            return
        source_signature = _request_signature(request, config)
        source_signature.pop("palette_sha256")
        source_signature.pop("typography_sha256")
        source_signature = Codec(bundle_root / request_id, roots).encode(source_signature)
        try:
            document = validate_bundle(bundle_root, request, cohort_hash())
        except (OSError, ValueError, KeyError):
            document = None
        if document and document.get("source_signature") == source_signature:
            if not config.get("prepare_only"):
                replay(bundle_root / request_id, document, roots)
            return
        bundle_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=f".{request_id}-", dir=bundle_root) as temporary:
            temporary = Path(temporary)
            capture_roots = roots
            render_context = context
            if config.get("prepare_only"):
                render_context = replace(context, figures_root=temporary / "preparation")
                render_context.subfigure_dir.mkdir(parents=True)
                capture_roots = dict(roots, figures=render_context.figures_root)
            from matplotlib.figure import Figure
            original_savefig = Figure.savefig
            saved = set()

            def prepare_savefig(figure, filename, *args, **kwargs):
                saved.add(Path(filename).name)
                if not config.get("prepare_only"):
                    return original_savefig(figure, filename, *args, **kwargs)

            Figure.savefig = prepare_savefig
            try:
                with capture(temporary, capture_roots) as session:
                    render_request(render_context, request_id)
            finally:
                Figure.savefig = original_savefig
            if not session.entries or not set(request["outputs"]).issubset(saved):
                raise RuntimeError(f"Incomplete prepared request {request_id}: saved={sorted(saved)}")
            document = {
                "schema_version": SCHEMA_VERSION,
                "request_id": request_id,
                "request_version": request["cache_version"],
                "cohort_sha256": cohort_hash(),
                "drawing_code_sha256": drawing_code_hash(),
                "analysis_input_manifest_sha256": config.get("analysis_input_manifest_sha256"),
                "source_signature": source_signature,
                "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True).strip(),
                "outputs": request["outputs"],
                "entries": session.entries,
                "palette_usage": palette_usage(),
                "files": {str(p.relative_to(temporary)): file_hash(p) for p in sorted((temporary / "assets").glob("*"))},
            }
            document["payload_sha256"] = hashlib.sha256(
                json.dumps(document, ensure_ascii=False, sort_keys=True, allow_nan=False).encode()
            ).hexdigest()
            # Immutable content-addressed assets first; atomically publish the
            # document last so an interrupted refresh preserves the old bundle.
            destination = bundle_root / request_id
            destination.mkdir(exist_ok=True)
            if (temporary / "assets").exists():
                (destination / "assets").mkdir(exist_ok=True)
                shared_root = bundle_root / "shared_assets"
                shared_root.mkdir(exist_ok=True)
                for source in (temporary / "assets").glob("*"):
                    target = destination / "assets" / source.name
                    shared = shared_root / source.name
                    if not shared.exists() or file_hash(shared) != file_hash(source):
                        staged = shared.with_suffix(".tmp")
                        shutil.copy2(source, staged)
                        staged.replace(shared)
                    link = target.with_suffix(target.suffix + ".link.tmp")
                    link.unlink(missing_ok=True)
                    os.symlink(f"../../shared_assets/{source.name}", link)
                    link.replace(target)
            write_document(destination / "prepared.json", document)
    except Exception:
        traceback.print_exc()
        raise


def _execute_request(request: dict, config: dict, *, force: bool) -> dict:
    started = time.monotonic()
    request_id = request["id"]
    try:
        signature = _request_signature(request, config)
        if not config.get("prepare_only") and not force and _request_is_fresh(request, config, signature):
            print(f"[render:{request_id}] cache-hit", flush=True)
            return {"id": request_id, "status": "cached", "elapsed": 0.0}
        receipt = _receipt_path(request, config)
        if not config.get("prepare_only"):
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
        if config.get("prepare_only"):
            elapsed = time.monotonic() - started
            print(f"[prepare:{request_id}] success {elapsed:.1f}s", flush=True)
            return {"id": request_id, "status": "prepared", "elapsed": elapsed}
        invalid = [str(path) for path in _request_outputs(request, config) if not _valid_pdf(path)]
        if invalid:
            raise RuntimeError("renderer did not produce valid PDF(s): " + ", ".join(invalid))
        elapsed = time.monotonic() - started
        _write_receipt(request, config, _request_signature(request, config), elapsed)
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
        "analysis_input_manifest_sha256": config.get(
            "analysis_input_manifest_sha256"
        ),
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
    prepare_only: bool = False,
    bundle_dir: Path | None = None,
    from_bundle: Path | None = None,
    palette: Path | None = None,
) -> list[Path]:
    """Render selected request bundles for later formal composition."""
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
    if prepare_only and from_bundle:
        raise ValueError("--prepare-only and --from-bundle are mutually exclusive")
    if from_bundle and bundle_dir:
        raise ValueError("--bundle-dir cannot be combined with --from-bundle")
    from .visualization.palette import configure_palette, DEFAULT_PALETTE
    from .visualization.prepared_bundle import validate_bundle, write_document

    chosen_palette = configure_palette(palette)
    config = dict(config, palette_path=str(Path(palette or DEFAULT_PALETTE).resolve()),
                  palette_sha256=chosen_palette.digest, prepare_only=prepare_only,
                  bundle_dir=str(Path(from_bundle or bundle_dir or (Path(config["cache_dir"]) / "visualization" / "prepared_paper")).resolve()))
    if from_bundle:
        config["from_bundle"] = config["bundle_dir"]
        documents = [validate_bundle(Path(config["bundle_dir"]), request, cohort_hash()) for request in selected_requests]
        hashes = {doc["analysis_input_manifest_sha256"] for doc in documents if doc.get("analysis_input_manifest_sha256")}
        if len(hashes) > 1:
            raise ValueError("Prepared requests belong to different analysis inputs")
        config["analysis_input_manifest_sha256"] = next(iter(hashes), None)
        config["data_dir"] = str(Path(config["bundle_dir"]) / ".unavailable-source-data")
    else:
        config = _bind_render_analysis(config, selected_requests)

    figures_root = Path(config["figures_dir"])
    paper_dir = figures_root / "paper"
    subfigure_dir = paper_dir / "subfigure"
    if not prepare_only:
        from .paper_preview import invalidate_preview
        invalidate_preview(figures_root, "Rendering started; run full compose after rendering to refresh previews.")
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
    write_document(paper_dir / "report" / "render_timings.json", {
        "from_bundle": bool(from_bundle), "prepare_only": prepare_only,
        "palette_sha256": chosen_palette.digest, "requests": list(results.values()),
    })
    usage_report = {}
    for request in selected_requests:
        prepared_path = Path(config["bundle_dir"]) / request["id"] / "prepared.json"
        if prepared_path.is_file():
            usage = json.loads(prepared_path.read_text()).get("palette_usage", {})
            usage_report[request["id"]] = {
                "outputs": request["outputs"],
                "colors": {key: str(chosen_palette.colour(key)) for key in usage.get("colors", [])},
                "colormaps": {key: chosen_palette.colormaps[key] for key in usage.get("colormaps", [])},
                "alpha": {key: chosen_palette.opacities[key] for key in usage.get("alpha", [])},
            }
    write_document(paper_dir / "report" / "palette_usage.json", usage_report)
    if prepare_only:
        failures = [result for result in results.values() if result["status"] == "failed"]
        if failures:
            raise RuntimeError(f"Prepared input failures: {failures}")
        _prepare_composition_bundle(Path(config["bundle_dir"]), selected_figures)
        return []
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
    from .paper_typography_report import write_typography_summary
    write_typography_summary(figures_root)
    if failed:
        details = "\n".join(f"- {item['id']}: {item['error']}" for item in failed)
        raise RuntimeError(f"Formal render request failure(s):\n{details}")
    if from_bundle:
        write_document(paper_dir / "report" / "drawing_bundle.json", {"path": config["bundle_dir"]})
    else:
        _prepare_composition_bundle(Path(config["bundle_dir"]), selected_figures)
        write_document(paper_dir / "report" / "drawing_bundle.json", {"path": config["bundle_dir"]})
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
