"""Small fixture-free contract tests intended for remote validation."""

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
