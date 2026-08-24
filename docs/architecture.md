# Reproduction architecture

The active workflow has four layers:

1. **Explicit inputs** — `regions.json`, `experiment.toml`, merchant sales,
   grid factors, and optional raw/fetchable building and road-network inputs.
2. **City experiments** — preprocessing, buildings, demand, road routing, and
   lifecycle emissions. These modules never import plotting code.
3. **Nationwide aggregation and paper analysis** — `aggregate` writes compatible
   `all_region_stats.{json,csv}` and enriched statistics. `analyze` builds named
   scenarios, city attribution, spatial concentration, distance thresholds, and
   threshold resilience in cache/work. A work-root data view symlinks immutable
   inputs but copies every replaceable derived file, so legacy-compatible
   analysis functions cannot write through to city data.
4. **Final figures** — the machine-readable
   `src/drone_emission/paper_figures.json` declares exactly 12 assets. Rendering
   creates only declared sources/staging files. Publication is a separate,
   explicitly confirmed operation.

`render` accepts formal manifest IDs, not historical plot-function names. The
nine `inkscape` entries generate required panels and reports only. The three
`direct` entries additionally copy their renderer output to the formal staging
filename. Missing derived inputs are produced by `analyze` or reported;
rendering never rewrites city OD files or per-region statistics. The manifest
records input contracts, render requests, required outputs, source reports,
Inkscape sources, staging paths, and publication paths.

`promote` validates all 12 staging filenames, manifest mappings, and PDF magic
headers before copying. Neither `render` nor `reproduce` writes to the ignored
Overleaf working tree under `docs/paper/`. `supplementary.tex` remains outside
the active reproduction, validation, and publication scope.

The remote environment resolves `/mnt/minor_storage/$USER/drone-emission`
dynamically. All roots can be overridden independently through the environment
variables documented in the README.

Formal multi-city execution is shell-orchestrated. One parent systemd scope
enforces the default 64GB limit over the complete process tree. With an 8GB
reserve and a 12GB scheduling reservation per city, no more than four workers
start concurrently. Later stages remain inside the same scope. Direct Python
invocations do not claim this hard-limit guarantee.

The active package uses fully qualified `drone_emission.*` imports. Pipeline
stages cannot import visualization modules, active code cannot import archive
modules, and package initialization performs no path injection.
