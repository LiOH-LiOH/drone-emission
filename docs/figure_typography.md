# Paper typography and palette overview

The formal scope remains 31 requests, 46 subfigure PDFs and 12 composed PDFs.
Edit `src/drone_emission/visualization/typography.toml` to tune physical font
sizes. Colours remain in the separate `palette.toml`.

## Typography contract

Arial is the common family. Axis labels use 9 pt; tick labels, legends and
primary annotations use 8 pt; secondary annotations use 7 pt; panel letters
and enabled titles use 11 pt. A final artist pass applies these roles even to
insets, explicit font sizes and hand-built legends. The pass is scoped to formal
render workers and is repeatable for PDF/SVG exports of the same figure.

Dense labels may use smaller sizes without deleting labels. Every exception in
`[[exceptions]]` must name `output`, `role`, `text_regex`, `size` and `reason`.
Use anchored expressions for exact labels. `[[layouts]]` can set a named
output's axes `position` by zero-based `axes` index and optional `x_labelpad`
or `y_labelpad`. `[[text_adjustments]]` can adjust matching text positions,
rotation or horizontal alignment. These changes belong in the same config so
visual tuning does not invalidate scientific inputs.

`paper/report/typography/` records each exported subfigure's text, semantic
role, original size, final size and exception reason. `typography_summary.json`
and `typography_summary.md` summarize sizes and exceptions; stale records are
identified rather than counted as current. SVG master text and linked-panel
physical scales are recorded under `inkscape/typography/` in the review batch.
Mathematical superscripts/subscripts naturally have smaller glyphs than their
parent label; the configured size refers to the parent label.

## Composition and previews

Formal composition preserves each linked subfigure's physical PDF dimensions,
including when tight bounding boxes change after typography adjustments. It
normalizes SVG master text in the export copy; original masters are preserved.

Every successful full `compose` refreshes `preview/<figure_id>.png` and
`preview/palette_overview.png`. The overview uses manifest order, three columns,
four rows and a 6000 × 8000 pixel white canvas. All panels remain complete and
keep their aspect ratio. Thumbnail sizes have no effect on formal PDF sizes.
Preview rasterization requires `pdftoppm`; Pillow assembles the overview.

Rendering, partial composition and failed composition invalidate the overview:
`preview/status.json` records `stale`, and the previous overview is renamed
`palette_overview.stale.png`. A complete successful compose publishes a current
overview and records hashes of all source PDFs. `compose --prepare-links` alone
does not claim to refresh previews.

## Fast local iteration

Use the updated independent drawing bundle with the ordinary manifest CLI:

```bash
PYTHONPATH=src DRONE_EMISSION_FIGURES_ROOT=figures/review/typography-refresh_20260907 \
  .venv/bin/python -m drone_emission --env local render \
  --from-bundle figures/review/typography-refresh_20260907/drawing-bundle
PYTHONPATH=src DRONE_EMISSION_FIGURES_ROOT=figures/review/typography-refresh_20260907 \
  .venv/bin/python -m drone_emission --env local compose
```

Run local Matplotlib outside the sandbox. Typography configuration participates
in output cache signatures but not in prepared scientific-input signatures.
Changing Python drawing code still requires preparation with matching code;
the code/cohort/asset checks are never bypassed for a cosmetic change.
