---
name: viz-conventions
description: Use when adding, revising, reviewing, or registering plot functions in this drone-emission project, especially files under src/drone_emission/visualization/ or paper figure generation.
---

# Visualization Plot Function Conventions

Formal manuscript figures are manifest-driven. The only public entry point is:

```bash
python -m drone_emission render [--only MANIFEST_ID]
```

## File layout

Place renderers by evidence theme: configuration/city inputs, city priorities,
GEF and detour, contribution and break-even, national scenarios, threshold
resilience, city attribution, distance thresholds, or observed delivery
distance. Shared helpers may have a small adjacent support module. Do not
recreate generic giant modules or a compatibility facade.

`paper_workflow.py` is the sole formal orchestration layer. Every request in
`paper_figures.json` must occur exactly once in `REQUEST_RENDERERS`; the
registry must contain no undeclared request. Standalone plotting CLIs and
candidate figures belong in `archive/code/`, not the active package.

## Renderer contract

Each registry value accepts one immutable `RenderContext`. Inputs are read from
its configured data/cache/work roots. Outputs may be written only below its
figures/cache/work roots. A renderer must not call `analyze`, rewrite city data,
publish to `docs/paper/figures/`, silently skip a missing required input, or
create a placeholder. Missing inputs must raise an error naming the prerequisite
stage.

## Plot functions

Use data inputs first and the established signature where practical:

```python
def plot_xxx(
    df: pd.DataFrame,
    save_path: str = None,
    figsize: tuple = (10, 6),
    for_latex: bool = False,
    plot_title: bool = False,
):
```

Do not add a `dpi` or `verbose` parameter. Paper output uses 300 dpi for raster
exports. Follow the save-path convention already used by the thematic module.

Titles and decorative subtitles must be gated by `plot_title`; axis labels,
legends, data labels, and essential annotations remain visible. When titled
output is supported, add `_t` before the extension with `_add_title_suffix`.

Use function-specific save messages and concise docstrings.

## Registration checklist

1. Confirm the request is declared once in `paper_figures.json`.
2. Register it once in `formal_registry.REQUEST_RENDERERS`.
3. Keep outputs equal to the manifest's required subfigures.
4. Add input checks with a prerequisite-stage hint.
5. Verify active code has no archive or retired-module import.
6. Do not run local Python or rendering unless the user explicitly authorizes it.
