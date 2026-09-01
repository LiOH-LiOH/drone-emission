# Reproduction architecture

The active workflow has four layers:

1. **Explicit inputs** — `regions.json`, `experiment.toml`, merchant sales,
   grid factors, and optional raw/fetchable building and road-network inputs.
2. **City experiments** — preprocessing, buildings, demand, road routing, and
   lifecycle emissions. These modules never import plotting code.
3. **Nationwide aggregation, cohort lock, and paper analysis** — `aggregate`
   writes compatible `all_region_stats.{json,csv}` and enriched statistics.
   `cohort` audits the configured cities against the tracked 313-city contract;
   `analyze` builds named scenarios, city attribution, distance thresholds,
   threshold resilience, and the full-OD contribution decomposition in
   cache/work. A work-root data view symlinks immutable
   inputs but copies every replaceable derived file, so legacy-compatible
   analysis functions cannot write through to city data. The view is versioned
   by the cohort SHA-256 and contains only the locked cities.
4. **Final figures** — the machine-readable
   `src/drone_emission/paper_figures.json` declares exactly 10 assets. Rendering
   creates only declared sources/staging files. Publication is a separate,
   explicitly confirmed operation.

`paper_workflow.py` is the only formal scheduler. Before rendering it verifies
that the 27 unique manifest requests exactly equal the explicit
`formal_registry.REQUEST_RENDERERS` keys. Every renderer receives one immutable
context containing the configured data, figures, cache, and work roots.
Requests that consume analysis products receive a matching cohort-versioned
view and validate only their required analysis modules. Aggregate-only requests
read their declared inputs directly; nationwide renderers apply the locked
cohort selection themselves.

`render` accepts formal manifest IDs, not internal plot-function names. The
10 `inkscape` entries generate required panels and reports; the local-only
`compose` command converts those panel PDFs to external SVG links and exports
the tracked Inkscape layouts to formal staging filenames. It never rewrites the
master SVGs: linked image elements are inlined as nested SVG elements only in a
temporary export copy. Missing inputs fail with a
prerequisite-stage hint; render never invokes `analyze`, creates placeholders,
or silently skips them.
Derived inputs are produced only by an explicit prior `analyze` command. Formal
rendering never rewrites city OD files or per-region statistics. The manifest
records input contracts, render requests, required outputs, source reports,
Inkscape sources, staging paths, and publication paths.

Contribution decomposition is a paper-analysis stage rather than a renderer
side effect. It hashes and streams the three required OD columns, persists one
atomic summary per city, and writes a nationwide final table only when all 313
locked cities pass schema, fingerprint, and ledger-identity checks. Interrupted
or partial runs retain successful city summaries but cannot satisfy the Figure
5 manifest input.

The observed Beijing delivery-distance panel renders from the tracked audited
histogram by default. Raw Beijing OD is required only to rebuild and relock that
calibration input, not for an ordinary paper render.

The separate `calibrate-coordinates` command estimates the historical global
translation of Beijing Eleme points against restaurant POIs. It is an audit
utility below `calibration/`, writes to work/cache, and is deliberately absent
from `reproduce`, the paper analysis scheduler, and the formal figure manifest.

The `calibrate` command is likewise outside `reproduce`. It consumes only the
fingerprinted corrected-order export, the audited transform, and one
preprocessed Beijing GBA GeoPackage. Building matching and the 5 km candidate
CSR are cached beneath the requested run directory. Its two English audit
figures are calibration evidence, not formal manuscript figures, and therefore
are not declared in `paper_figures.json` or the renderer registry. Acceptance is
a separate explicit operation that validates convergence and all live input
fingerprints before changing the three `od_params` values.

`promote` validates all 10 staging filenames, manifest mappings, and PDF magic
headers before copying. Neither `render` nor `reproduce` writes to the ignored
Overleaf working tree under `docs/paper/`. `supplementary.tex` remains outside
the active reproduction, validation, and publication scope.

The remote environment resolves `/mnt/minor_storage/$USER/drone-emission`
dynamically. All roots can be overridden independently through the environment
variables documented in the README.

`paper_cohort.json` is the source-controlled research-sample contract. The
current lock excludes Linzhi for insufficient OD support and uses a minimum of
30 valid weighted OD pairs. The five retained cities without a provincial GEF
use the paper-selected 2022 national mean of 0.5366 kg CO2e/kWh. Candidate
cohort audits are runtime files under work; accepting a changed cohort is an
explicit source change.

An optional `analyze --reuse-from DIR` restores a single audited reuse
directory. Its manifest binds the file hashes, schemas, cohort hash, city
coverage, and GEF fallback. There is no latest-bundle lookup or legacy-path
discovery. A normal analyze without this option follows the compute path.

Formal multi-city execution is shell-orchestrated. One parent systemd scope
enforces the default 64GB limit over the complete process tree. With an 8GB
reserve and a 12GB scheduling reservation per city, no more than four workers
start concurrently. Later stages remain inside the same scope. Direct Python
invocations do not claim this hard-limit guarantee.

Audited multi-city runs pass `--cohort locked --run-dir DIR` through the shell
launcher to the package city scheduler. The scheduler uses one parent event
loop to own all status files and at most the requested number of city
subprocesses; city workers write only their dedicated logs. No shared worker
lock is used. Batch manifests bind explicit resume to the source commit,
cohort, environment, concurrency, and city flags. A failed city is recorded
without retry while independent cities continue, and any failure makes the
batch exit nonzero before downstream stages.

The active visualization package is split by evidence theme. The pre-split
generic modules and candidate CLIs exist only under `archive/code/`; no facade
preserves their imports. Pipeline stages cannot import visualization modules,
active code cannot import archive modules, and package initialization performs
no path injection or filesystem writes.
