"""Small fixture-free contract tests intended for remote validation."""

import ast
import json
from pathlib import Path

from drone_emission.paper_workflow import load_manifest, select_figures


def test_manifest_has_exact_locked_set():
    manifest = load_manifest()
    assert manifest["figure_count"] == 12
    assert len(manifest["figures"]) == 12
    assert {figure["mode"] for figure in manifest["figures"]} == {
        "direct",
        "inkscape",
    }
    assert sum(figure["mode"] == "direct" for figure in manifest["figures"]) == 3
    assert all(figure["required_inputs"] for figure in manifest["figures"])
    assert all(len(figure["report_inputs"]) == 1 for figure in manifest["figures"])


def test_manifest_paths_are_one_to_one():
    figures = load_manifest()["figures"]
    staging = [Path(figure["staging"]).name for figure in figures]
    publish = [Path(figure["publish"]).name for figure in figures]
    assert len(set(staging)) == 12
    assert staging == publish
    assert staging == [f"{figure['id']}.pdf" for figure in figures]


def test_render_selection_rejects_internal_plot_names():
    try:
        select_figures(["city_gef_context_detour_summary"])
    except ValueError as error:
        assert "manifest IDs only" in str(error)
    else:
        raise AssertionError("internal plot name was accepted")


def test_city_pipeline_has_no_visualization_dependency():
    package = Path(__file__).parents[1] / "src" / "drone_emission"
    paths = [package / "main.py"]
    for name in ("preprocessing", "buildings", "demand", "roadnet", "emission"):
        paths.extend((package / name).glob("*.py"))
    violations = []
    for path in paths:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            module = getattr(node, "module", "") or ""
            if module.startswith("drone_emission.visualization"):
                violations.append(f"{path}:{node.lineno}")
    assert violations == []


def test_active_package_does_not_import_archive_or_inject_paths():
    package = Path(__file__).parents[1] / "src" / "drone_emission"
    violations = []
    for path in package.rglob("*.py"):
        if "archive" in path.parts:
            continue
        source = path.read_text(encoding="utf-8")
        if "drone_emission.archive" in source or "sys.path.insert" in source:
            violations.append(str(path))
    assert violations == []


def test_shell_declares_single_global_memory_contract():
    root = Path(__file__).parents[1]
    source = (root / "entrypoint.sh").read_text(encoding="utf-8")
    assert 'DRONE_EMISSION_MEMORY_MAX_GB:-64' in source
    assert 'DRONE_EMISSION_SYSTEM_RESERVE_GB:-8' in source
    assert 'DRONE_EMISSION_CITY_RESERVATION_GB:-12' in source
    assert source.count("systemd-run --user --scope") == 1
    assert 'MemoryMax=${MEMORY_MAX_GB}G' in source
    assert 'MemorySwapMax=0' in source


def test_docs_paper_has_exact_manifest_publication_set():
    root = Path(__file__).parents[1]
    manifest = json.loads(
        (root / "src" / "drone_emission" / "paper_figures.json").read_text()
    )
    declared = {Path(item["publish"]).name for item in manifest["figures"]}
    paper_dir = root / "docs" / "paper" / "figures"
    if paper_dir.is_dir():
        existing = {path.name for path in paper_dir.glob("*.pdf")}
        assert declared == existing
    else:
        assert all(
            item["publish"].startswith("docs/paper/figures/")
            for item in manifest["figures"]
        )
