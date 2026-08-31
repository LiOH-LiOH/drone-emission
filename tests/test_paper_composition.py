import os
from pathlib import Path
from xml.etree import ElementTree

import pytest

from drone_emission import paper_workflow


def _manifest() -> dict:
    return {
        "requests": [
            {
                "id": "panel_request",
                "outputs": ["panel.pdf"],
            }
        ],
        "figures": [
            {
                "id": "fig_test",
                "mode": "inkscape",
                "render_requests": ["panel_request"],
                "inkscape_source": "figures/inkscape/master.svg",
                "staging": "figures/paper/fig_test.pdf",
            }
        ],
    }


def _valid_pdf(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"%PDF-1.4\nfixture")


def test_default_compose_selection_contains_only_inkscape_figures():
    selected = paper_workflow.select_inkscape_figures(None)
    assert len(selected) == 8
    assert all(figure["mode"] == "inkscape" for figure in selected)

    with pytest.raises(ValueError, match="Inkscape figure IDs only"):
        paper_workflow.select_inkscape_figures(
            ["fig4_gef_distance_detour"]
        )


def test_linked_subfigure_names_reads_external_links_only(tmp_path):
    source = tmp_path / "master.svg"
    source.write_text(
        '<svg xmlns="http://www.w3.org/2000/svg" '
        'xmlns:xlink="http://www.w3.org/1999/xlink">'
        '<image xlink:href="linked-subfigures/panel.svg"/>'
        '<image href="data:image/png;base64,AAAA"/>'
        "</svg>",
        encoding="utf-8",
    )
    assert paper_workflow._linked_subfigure_names(source) == {"panel.svg"}


def test_namespace_linked_svg_ids_rewrites_local_references():
    tree = ElementTree.ElementTree(
        ElementTree.fromstring(
            '<svg id="surface1">'
            '<defs><path id="glyph-0-0"/><clipPath id="clip-0"/></defs>'
            '<use href="#glyph-0-0" style="clip-path:url(#clip-0)"/>'
            "</svg>"
        )
    )

    root_id = paper_workflow._namespace_linked_svg_ids(tree, "panel.svg")
    rendered = ElementTree.tostring(tree.getroot(), encoding="unicode")

    assert root_id == "linked-subfigure-panel-root"
    assert 'id="linked-subfigure-panel-root"' in rendered
    assert 'id="linked-subfigure-panel-glyph-0-0"' in rendered
    assert 'id="linked-subfigure-panel-clip-0"' in rendered
    assert 'href="#linked-subfigure-panel-glyph-0-0"' in rendered
    assert "url(#linked-subfigure-panel-clip-0)" in rendered
    assert "url(#clip-0)" not in rendered


def test_prepare_links_converts_subfigure_without_export(tmp_path, monkeypatch):
    project = tmp_path / "project"
    figures = tmp_path / "runtime-figures"
    panel_pdf = figures / "paper" / "subfigure" / "panel.pdf"
    _valid_pdf(panel_pdf)
    linked_dir = project / "figures" / "inkscape" / "linked-subfigures"
    calls = []

    monkeypatch.setattr(paper_workflow, "load_manifest", _manifest)
    monkeypatch.setattr(paper_workflow, "PROJECT_ROOT", project)
    monkeypatch.setattr(paper_workflow, "LINKED_SUBFIGURE_DIR", linked_dir)
    monkeypatch.setattr(paper_workflow, "_tool_path", lambda name: Path(name))

    def convert(source, target, *, pdftocairo):
        calls.append((source, target, pdftocairo))
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("<svg/>", encoding="utf-8")

    monkeypatch.setattr(paper_workflow, "_convert_linked_subfigure", convert)
    outputs = paper_workflow.compose(
        {"figures_dir": str(figures)},
        prepare_links=True,
    )

    assert outputs == [linked_dir / "panel.svg"]
    assert calls == [(panel_pdf, linked_dir / "panel.svg", Path("pdftocairo"))]


def test_compose_exports_when_master_is_newer(tmp_path, monkeypatch):
    project = tmp_path / "project"
    figures = tmp_path / "runtime-figures"
    panel_pdf = figures / "paper" / "subfigure" / "panel.pdf"
    extra_pdf = figures / "paper" / "subfigure" / "extra.pdf"
    linked_svg = (
        project
        / "figures"
        / "inkscape"
        / "linked-subfigures"
        / "panel.svg"
    )
    extra_svg = linked_svg.with_name("extra.svg")
    master = project / "figures" / "inkscape" / "master.svg"
    _valid_pdf(panel_pdf)
    _valid_pdf(extra_pdf)
    linked_svg.parent.mkdir(parents=True, exist_ok=True)
    linked_svg.write_text(
        '<svg viewBox="0 0 10 10"><path id="panel-mark"/></svg>',
        encoding="utf-8",
    )
    extra_svg.write_text(
        '<svg viewBox="0 0 20 20"><path id="extra-mark"/></svg>',
        encoding="utf-8",
    )
    master.write_text(
        '<svg xmlns="http://www.w3.org/2000/svg">'
        '<image href="linked-subfigures/panel.svg" '
        'x="5" y="6" width="10" height="11" '
        'transform="translate(3,4)"/>'
        '<image href="linked-subfigures/extra.svg"/>'
        "</svg>",
        encoding="utf-8",
    )
    old = master.stat().st_mtime - 10
    os.utime(panel_pdf, (old, old))
    os.utime(extra_pdf, (old, old))
    os.utime(linked_svg, (old, old))
    os.utime(extra_svg, (old, old))
    target = figures / "paper" / "fig_test.pdf"
    _valid_pdf(target)
    os.utime(target, (old, old))
    exported = []

    monkeypatch.setattr(paper_workflow, "load_manifest", _manifest)
    monkeypatch.setattr(paper_workflow, "PROJECT_ROOT", project)
    monkeypatch.setattr(
        paper_workflow,
        "LINKED_SUBFIGURE_DIR",
        linked_svg.parent,
    )
    monkeypatch.setattr(paper_workflow, "_tool_path", lambda name: Path(name))

    def export(source, target, *, inkscape):
        exported.append((source.read_text(encoding="utf-8"), target, inkscape))
        _valid_pdf(target)

    monkeypatch.setattr(paper_workflow, "_export_inkscape_figure", export)
    outputs = paper_workflow.compose({"figures_dir": str(figures)})

    assert outputs == [target]
    assert len(exported) == 1
    exported_svg, exported_target, exported_inkscape = exported[0]
    assert "panel-mark" in exported_svg
    assert "extra-mark" in exported_svg
    assert "linked-subfigures/panel.svg" not in exported_svg
    assert "linked-subfigures/extra.svg" not in exported_svg
    assert 'transform="translate(3,4)"' in exported_svg
    assert 'x="5" y="6" width="10" height="11"' in exported_svg
    assert exported_target == target
    assert exported_inkscape == Path("inkscape")
    assert paper_workflow.compose({"figures_dir": str(figures)}) == []
    assert len(exported) == 1


def test_compose_rejects_missing_required_link(tmp_path, monkeypatch):
    project = tmp_path / "project"
    figures = tmp_path / "runtime-figures"
    panel_pdf = figures / "paper" / "subfigure" / "panel.pdf"
    linked_svg = (
        project
        / "figures"
        / "inkscape"
        / "linked-subfigures"
        / "panel.svg"
    )
    master = project / "figures" / "inkscape" / "master.svg"
    _valid_pdf(panel_pdf)
    linked_svg.parent.mkdir(parents=True, exist_ok=True)
    linked_svg.write_text("<svg/>", encoding="utf-8")
    master.write_text(
        '<svg xmlns="http://www.w3.org/2000/svg"/>',
        encoding="utf-8",
    )

    monkeypatch.setattr(paper_workflow, "load_manifest", _manifest)
    monkeypatch.setattr(paper_workflow, "PROJECT_ROOT", project)
    monkeypatch.setattr(
        paper_workflow,
        "LINKED_SUBFIGURE_DIR",
        linked_svg.parent,
    )
    monkeypatch.setattr(paper_workflow, "_tool_path", lambda name: Path(name))

    with pytest.raises(ValueError, match="panel.svg"):
        paper_workflow.compose({"figures_dir": str(figures)})
