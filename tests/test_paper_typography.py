import copy
import json
from xml.etree import ElementTree

import matplotlib
matplotlib.use("Agg")
from matplotlib.figure import Figure
import pytest

from drone_emission.visualization import typography
from drone_emission.paper_svg_style import style_master
from drone_emission.paper_preview import invalidate_preview, refresh_previews


def test_roles_cover_explicit_fonts_legends_and_insets():
    fig = Figure()
    ax = fig.subplots()
    ax.plot([0, 1], [1, 0], label="Drone")
    ax.set_xlabel("Distance", fontsize=4)
    ax.tick_params(labelsize=5)
    legend = ax.legend(fontsize=4)
    inset = fig.add_axes([.6, .6, .2, .2])
    inset.set_ylabel("Inset", fontsize=5)
    annotation = ax.text(.2, .3, "Important", fontsize=6)
    report = typography.apply_typography(fig, "test.pdf")
    assert ax.xaxis.label.get_fontsize() == 9
    assert inset.yaxis.label.get_fontsize() == 9
    assert legend.get_texts()[0].get_fontsize() == 8
    assert annotation.get_fontsize() == 7
    assert all(record["size_pt"] >= 7 for record in report["texts"])
    assert list(ax.lines[0].get_ydata()) == [1, 0]


def test_dense_exception_is_explicit_and_reported():
    fig = Figure()
    ax = fig.subplots()
    ax.text(0, 0, "Dense province", fontsize=5)
    config = copy.deepcopy(typography.load_typography())
    config["exceptions"] = [{"output": "dense.pdf", "role": "secondary", "text_regex": "^Dense province$", "size": 6.5, "reason": "Retain all province labels in the fixed inset."}]
    report = typography.apply_typography(fig, "dense.pdf", config)
    record = next(row for row in report["texts"] if row["text"] == "Dense province")
    assert record["size_pt"] == 6.5 and record["exception_reason"]


def test_svg_uses_physical_font_sizes_and_unscaled_panels(tmp_path):
    (tmp_path / "panel.svg").write_text('<svg width="72pt" height="36pt" viewBox="0 0 72 36"/>')
    root = ElementTree.fromstring('<svg width="100mm" height="100mm" viewBox="0 0 100 100"><g transform="translate(2,3)"><text style="font-size:5px"><tspan>a</tspan></text><image href="panel.svg" x="0" y="0" width="10" height="10"/></g></svg>')
    report = style_master(root, tmp_path, {"panel.svg"})
    image = root.find(".//image")
    assert float(image.get("width")) == pytest.approx(25.4)
    assert float(image.get("height")) == pytest.approx(12.7)
    assert report["texts"][0]["size_pt"] == 11
    assert report["panels"][0]["export_scale"] == 1


def test_preview_failure_keeps_old_overview_marked_stale(tmp_path):
    preview = tmp_path / "preview"
    preview.mkdir()
    (preview / "palette_overview.png").write_bytes(b"old")
    invalidate_preview(tmp_path, "partial compose")
    assert not (preview / "palette_overview.png").exists()
    assert (preview / "palette_overview.stale.png").read_bytes() == b"old"
    with pytest.raises(FileNotFoundError):
        refresh_previews(tmp_path, [{"id": "missing", "staging": "missing.pdf"}], rasterizer="pdftoppm")
    assert json.loads((preview / "status.json").read_text())["status"] == "stale"


def test_typography_configuration_is_not_part_of_data_code_hash(tmp_path, monkeypatch):
    from drone_emission.visualization.prepared_bundle import drawing_code_hash
    before = drawing_code_hash()
    config = tmp_path / "typography.toml"
    config.write_bytes(typography.DEFAULT_TYPOGRAPHY.read_bytes())
    monkeypatch.setattr(typography, "DEFAULT_TYPOGRAPHY", config)
    first = typography.typography_signature()
    config.write_text(config.read_text().replace("tick = 8.0", "tick = 8.5"))
    assert typography.typography_signature() != first
    assert drawing_code_hash() == before
