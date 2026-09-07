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
`wrapped_text` may insert whitespace and line breaks but must preserve all
non-whitespace characters; it cannot change or shorten the scientific label.

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
Export bounds cover the original master and linked PDF viewports plus a 1 mm
margin, avoiding clipping when an updated linked PDF extends beyond the old page.
Invisible imported objects do not contribute to page bounds.
Subfigure contents are never scaled to make them fit the original page.
`src/drone_emission/paper_composition.json` stores small shifts in millimetres
for composition only. The future-scenarios master gains space between columns
and beside the shared colour bar; panel letters move with their panels. These
adjustments trigger composition without invalidating subfigure render caches.

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
PYTHONPATH=src DRONE_EMISSION_FIGURES_ROOT=figures/review/typography-final_20260907 \
  .venv/bin/python -m drone_emission --env local render \
  --from-bundle figures/review/typography-final_20260907/drawing-bundle
PYTHONPATH=src DRONE_EMISSION_FIGURES_ROOT=figures/review/typography-final_20260907 \
  .venv/bin/python -m drone_emission --env local compose
```

Run local Matplotlib outside the sandbox. Typography configuration participates
in output cache signatures but not in prepared scientific-input signatures.
Changing Python drawing code still requires preparation with matching code;
the code/cohort/asset checks are never bypassed for a cosmetic change.

Basemap place names baked into stored raster tiles are image pixels, not text
artists. They retain their original appearance; the typography contract covers
editable scientific labels, legends and annotations, including map attribution.

## Validated review batch (2026-09-07)

The delivered batch is `figures/review/typography-final_20260907/`. It contains
46 subfigure PDFs, 12 composed PDFs, 12 individual previews, the complete
overview and the approximately 194 MB independent `drawing-bundle/`.

All 31 local replay requests succeeded in 83.40 seconds across requests.
All 31 scientific-asset manifests match the previous palette batch. A real
8 → 8.25 pt tick-size probe triggered a redraw; the next identical run hit the
cache. All 887 physical bundle files remained byte-for-byte unchanged and the
configuration was restored to 8 pt. All 25 regression tests passed. The suite covers bundle integrity,
typography roles, repeat exports, whitespace-only wrapping, preview invalidation,
physical SVG sizes, invisible-object bounds and coordinated panel shifts.

Five subfigures retain below-7-pt labels: the four scenario insets each use
6 pt for all 33 province names, and the narrow city-priority scatter uses 6.5 pt
for five population-bin labels and three priority-legend entries. All 140
labels are itemized with reasons in `paper/report/typography_summary.md`.
The province lifecycle units render without missing-glyph warnings.

The reports `typography_validation.json` and `typography_cache_probe.json`
record the artifact, scale, data-integrity and cache checks. Inkscape still
emits PDF font-import warnings; final raster previews are checked in addition
to checking successful process exits and page dimensions.
