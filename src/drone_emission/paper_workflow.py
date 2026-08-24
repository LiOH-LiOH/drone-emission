"""Manifest-driven rendering and publication of the locked paper figures."""

from __future__ import annotations

import json
import shutil
from importlib.resources import files
from pathlib import Path

from .paths import PROJECT_ROOT

EXPECTED_FIGURE_COUNT = 12


def load_manifest() -> dict:
    """Load and structurally validate the locked paper-figure manifest."""
    path = files("drone_emission").joinpath("paper_figures.json")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    figures = manifest.get("figures", [])
    ids = [figure.get("id") for figure in figures]
    if manifest.get("schema_version") != 1:
        raise ValueError("Unsupported paper figure manifest schema")
    if manifest.get("figure_count") != EXPECTED_FIGURE_COUNT:
        raise ValueError("Manifest figure_count must be exactly 12")
    if len(figures) != EXPECTED_FIGURE_COUNT or len(set(ids)) != len(ids):
        raise ValueError("Manifest must contain 12 uniquely identified figures")
    for figure in figures:
        if figure.get("mode") not in {"direct", "inkscape"}:
            raise ValueError(f"Invalid render mode for {figure.get('id')}")
        expected_name = f"{figure['id']}.pdf"
        if Path(figure["staging"]).name != expected_name:
            raise ValueError(f"Staging filename mismatch for {figure['id']}")
        if Path(figure["publish"]).name != expected_name:
            raise ValueError(f"Publish filename mismatch for {figure['id']}")
    return manifest


def figure_index() -> dict[str, dict]:
    return {figure["id"]: figure for figure in load_manifest()["figures"]}


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
    return [index[figure_id] for figure_id in only]


def _valid_pdf(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size < 8:
        return False
    with path.open("rb") as handle:
        return handle.read(5) == b"%PDF-"


def _render_distance_threshold(config: dict) -> None:
    """Render the two locked distance-policy panels from compact tables."""
    from tools.plot_distance_threshold_analysis import (
        _load_oracle_results,
        _load_summary,
        _resolve_summary_path,
    )
    from visualization.analysis import (
        plot_distance_threshold_city_landscape,
        plot_distance_threshold_oracle_performance,
    )

    data_dir = Path(config["data_dir"])
    out = Path(config["figures_dir"]) / "paper" / "subfigure"
    out.mkdir(parents=True, exist_ok=True)
    summary_path = _resolve_summary_path(data_dir, None)
    summary = _load_summary(summary_path, data_dir)
    oracle = _load_oracle_results(data_dir)
    plot_distance_threshold_city_landscape(
        summary,
        oracle,
        save_path=str(out / "distance_threshold_city_landscape.pdf"),
        for_latex=True,
        plot_title=False,
    )
    plot_distance_threshold_oracle_performance(
        summary,
        oracle,
        save_path=str(out / "distance_threshold_oracle_performance.pdf"),
        for_latex=True,
        plot_title=False,
    )


def render(config: dict, only: list[str] | None = None) -> list[Path]:
    """Render required sources and stage direct figures without publishing."""
    selected = select_figures(only)
    figures_root = Path(config["figures_dir"])
    paper_dir = figures_root / "paper"
    subfigure_dir = paper_dir / "subfigure"
    work_dir = Path(config["work_dir"])
    cache_dir = Path(config["cache_dir"])
    for directory in (paper_dir, subfigure_dir, work_dir, cache_dir):
        directory.mkdir(parents=True, exist_ok=True)

    requests = {
        request
        for figure in selected
        for request in figure.get("render_requests", [])
        if request != "distance_threshold_policy_comparison"
    }
    if requests:
        from utils.config import configure_cache
        from visualization.paper import render_all

        configure_cache(config)
        render_all(
            config["data_dir"],
            config["figures_dir"],
            for_latex=True,
            plot_title=False,
            only=requests,
            exp_config=config,
        )
    if any(
        "distance_threshold_policy_comparison" in figure.get("render_requests", [])
        for figure in selected
    ):
        _render_distance_threshold(config)

    missing: list[str] = []
    staged: list[Path] = []
    for figure in selected:
        for name in figure.get("required_subfigures", []):
            source = subfigure_dir / name
            if not _valid_pdf(source):
                missing.append(str(source))
        if figure["mode"] == "direct":
            source = subfigure_dir / figure["source_output"]
            target = PROJECT_ROOT / figure["staging"]
            # Honor an overridden figures root instead of the manifest's
            # repository-relative prefix.
            target = paper_dir / target.name
            if _valid_pdf(source):
                shutil.copy2(source, target)
                staged.append(target)
    if missing:
        raise FileNotFoundError(
            "Rendering did not produce required PDF source(s):\n- "
            + "\n- ".join(sorted(missing))
        )
    return staged


def promote(
    config: dict, *, yes: bool, dry_run: bool = False
) -> list[tuple[Path, Path]]:
    """Validate and copy all 12 staged PDFs into the paper repository."""
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
