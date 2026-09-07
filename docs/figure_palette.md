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
External basemap pixels are preserved unchanged. Colour-vision accessibility is
a preference, not a hard acceptance criterion; no redundant encodings are added.

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

Arrays use NPZ without pickle; tables have exact typed array storage plus readable
CSV; map geometry uses GeoJSON or WKB. Basemap arrays and extents are included.
Colour-bearing input fields store semantic keys rather than rendered RGBA values.
No Matplotlib Figure object is serialized.

Documents include cohort and analysis identity, request/schema/code versions,
source commit, source signatures, output names and checksums. Paths relocate
through named roots. Immutable assets are written before the document is
atomically replaced. A source/schema change requires a fresh preparation;
palette edits invalidate PDFs without invalidating the saved scientific inputs.

`paper/report/render_timings.json` records preparation or replay timings and
request failures. Rendering geometry, PDF export and Inkscape composition still
have their own costs; no fixed timing threshold is promised.
