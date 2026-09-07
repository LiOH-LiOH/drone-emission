# Paper colours and portable drawing inputs

The formal manifest contains 31 requests and 46 subfigure PDFs, assembled into
12 paper figures. The current palette is
`src/drone_emission/visualization/palette.toml`. Edit this file to try colours;
do not edit numeric bins, normalizations, figure layouts, line styles or markers.

## Colour categories

| Meaning | Configuration | Initial choice |
| --- | --- | --- |
| Delivery mode and corresponding routes | `mode.*` | Drone blue, e-bike orange; shared aliases in all mode figures |
| China macro-region | `region.*` | Seven fixed category colours |
| Other unordered groups | `group.*`, thematic keys | Restrained supplementary colours |
| Lifecycle components and contributions | `contribution_break_even.*`, `city_input_support.*` | Orange road operation, blue flight components, purple embodied balance |
| Continuous GEF | `scale.gef.*` | Light to dark purple |
| Continuous magnitude, height, density | `scale.size.*` | Light to dark blue |
| Reduction and positive-reduction probability | `scale.reduction.*` | Light to dark teal-green |
| Existing signed scales | `scale.diverging.*` | Orange, pale neutral, blue; existing zero and bounds retained |
| Observed/modelled data | `observed_distance_plot.*` | Dark grey observation, blue model |
| Backgrounds, grids, text, missing values | `ink.*`, thematic neutral keys | White and neutral greys |

`@mode.drone` means an alias, not an independent copy of a colour. The thematic
keys retain the old hex suffix as a stable migration identifier; the value is
the current colour and need not equal that suffix. Colormap choices are in the
same file under `[colormaps]`. `@scale.gef` selects the three named GEF stops.
When changing a mode's hue, adjust its `.light` and `.dark` entries together;
these are explicit colour choices rather than automatically generated shades.
Discrete ramps use `scale.gef.bin*` and `scale.reduction.bin*`; these change
colours only, while the existing numerical bin boundaries stay in the renderer.
Transparency is under `[alpha]`; initial values preserve the existing figures.
`paper/report/palette_usage.json` lists each request's colours, colormaps and
opacity settings, so experiments can target the affected figures.
External basemap pixels are preserved unchanged. Colour-vision accessibility is
a preference, not a hard acceptance criterion; no redundant encodings are added.

The bundle also contains checksummed copies of the formal Inkscape templates,
which are not tracked by Git. After a bundle render, `compose` automatically
uses these copies and places refreshed linked SVG panels in the current output
root. It preserves the original templates and their layout.

## Prepare once, redraw locally

Run preparation where the audited city and analysis inputs exist:

```bash
PYTHONPATH=src .venv/bin/python -m drone_emission --env remote render \
  --prepare-only --bundle-dir cache/paper-drawing-inputs
```

Copy that directory to the local machine, then run Matplotlib **outside the
sandbox**:

```bash
PYTHONPATH=src .venv/bin/python -m drone_emission --env local render \
  --from-bundle cache/paper-drawing-inputs
PYTHONPATH=src .venv/bin/python -m drone_emission --env local compose
```

Use `--only FIGURE_ID` or `--request REQUEST_ID` for a subset. `--palette PATH`
is optional; the tracked current palette is the default. Set
`DRONE_EMISSION_FIGURES_ROOT` to keep a review batch apart from current outputs.
`--force` with `--from-bundle` only redraws. Missing/incompatible inputs fail;
replay never starts analysis or falls back to raw city files or the network.

Ordinary `render` also prepares and reuses these inputs, defaulting to
`cache/visualization/prepared_paper`. `--prepare-only` performs the initial
preparation traversal but suppresses figure exports; it may construct temporary
axes to obtain layout-dependent geometry and basemap bounds. `compose` remains
the sole Inkscape assembly route and does not publish to the manuscript folder.

## Data contract

Each request has `prepared.json` and content-addressed assets. Plot entries keep
their explicit arguments; named `prepared_data` boundaries keep expensive
computed results (KDEs, fits, bins, box summaries, projected/local geometry,
paths and three-dimensional meshes). Large graph/router/OD inputs are replaced
with explicit unavailable-source markers after the final display data are saved.
The remaining drawing code performs small array/axis operations only.

Assets shared by several requests live once in `shared_assets`; request folders
refer to them through relative symlinks. Transfer the entire bundle with
`rsync -a` to preserve this deduplication and portability.

Arrays use NPZ without pickle; tables have exact typed array storage plus readable
CSV; map geometry uses GeoJSON or WKB. Basemap arrays and extents are included.
Colour-bearing input fields store semantic keys rather than rendered RGBA values.
No Matplotlib Figure object is serialized.

Documents include cohort and analysis identity, request/schema/code versions,
source commit, source signatures, output names and checksums. Paths relocate
through named roots. Immutable assets are written before the document is
atomically replaced. A source/schema change requires a fresh preparation;
palette edits invalidate PDFs without invalidating the saved scientific inputs.
The shared four-city bin cache also ignores display ordering when comparing
city sets, while retaining source-file, cohort-related input and model checks.
This prevents two panels with different city orders from rebuilding the same
expensive intermediate results.

`paper/report/render_timings.json` records preparation or replay timings and
request failures. Rendering geometry, PDF export and Inkscape composition still
have their own costs; no fixed timing threshold is promised.

## Validated local review batch (2026-09-07)

The review batch is `figures/review/palette-refresh_20260907/`:

- `drawing-bundle/`: the independent, checksummed bundle, about 194 MB including
  12 composition templates. Unused individual detour observations are excluded
  from the city-distance figure; its saved data are bin summaries and audit
  metadata. The earlier transfer staging folder is `drawing-inputs/`.
- `paper/subfigure/`: all 46 manifest PDFs; `paper/`: the 12 assembled PDFs.
- `preview/`: raster review copies of the assembled PDFs.
- `paper/report/`: per-request colour usage, timing and validation records.

To try the next palette locally, edit `palette.toml`, then run outside the sandbox:

```bash
PYTHONPATH=src DRONE_EMISSION_FIGURES_ROOT=figures/review/palette-refresh_20260907 \
  .venv/bin/python -m drone_emission --env local render \
  --from-bundle figures/review/palette-refresh_20260907/drawing-bundle
PYTHONPATH=src DRONE_EMISSION_FIGURES_ROOT=figures/review/palette-refresh_20260907 \
  .venv/bin/python -m drone_emission --env local compose
```

Validation used the audited 313-city cohort. All 31 remote preparation requests
and all 31 local replay requests passed; local replay produced 46 PDFs in 75.84
seconds across requests (1.48–13.08 seconds per request). A request may produce
several PDFs. Eighteen focused bundle/composition regression tests passed.
The palette-switch smoke check preserved all 25 input files byte-for-byte and
the unchanged repeat hit the render cache. The city-distance report matched
before and after removing unused observations, excluding colour/path fields.

Fifteen first-batch reports matched the numerical fields of existing local
reports. The old local city-reduction scatter report used different inputs;
all 313 new scatter points instead match the audited prepared snapshot.
Existing Arial warnings for Unicode subscript two and superscript minus remain
in the province lifecycle figure. Fonts and layout were kept within the agreed
colour-only scope. Inkscape also reports PDF font-import warnings;
the assembled figures were rasterized and visually checked.
