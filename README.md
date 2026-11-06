# Drone-delivery emission paper reproduction

This repository contains the city experiments, nationwide analysis, and figure
workflow supporting a study of selective drone substitution for e-bike food
delivery across 311 Chinese cities. The manuscript publication set comprises
12 figures (seven main figures and five Extended Data figures). The current
code manifest also includes one objective-tradeoffs candidate: 13 figure IDs
and 32 render requests in total. Candidate output is outside the 12-figure
publication set.

Commands below run from the repository root through the installed
`drone_emission` package. A fresh clone contains code and selected calibration
summaries, but does not contain all inputs or composition assets needed to
reproduce the manuscript figures.

## Review-release availability

| Material | Availability in this Git repository |
|---|---|
| Model code, `experiment.toml`, cohort contract and figure manifests | Tracked. |
| Selected calibration summaries, distance distributions and endpoint grid | Tracked under `src/drone_emission/calibration/`; these are derived products, not the complete raw calibration inputs. |
| `regions.json` and `regions.template.json` | Ignored and absent from the current tracked tree. Supply an audited static region registry before running setup or the workflow. The cohort contract records exclusions and a hash, not the complete registry. |
| Merchant sales, Beijing observed orders, building inputs and grid-factor table | Not bundled. Input locations are configurable; access and redistribution terms must be established with each data source. |
| Processed nationwide outputs and figure/table source-data bundle | Not bundled; this README currently provides no deposited dataset DOI or downloadable reproduction bundle. |
| Inkscape master layouts and linked panels under `figures/` | Ignored and absent from the tracked tree; required for final composition. |
| Working documentation, manuscript, Supplementary Information and accepted figure evidence under `docs/` | Local only and ignored by this GitHub repository. `docs/paper/` has its own Overleaf Git repository. |

The manuscript's planned GitHub/Zenodo availability statements are publication
commitments, not evidence that the inputs have already been released. This
checkout alone is not a complete reproduction package. The Supplementary
Information and supplementary tables are outside the CLI reproduction contract.
No software `LICENSE` or `CITATION.cff` is currently included; a release license,
preferred citation and archival identifier still need to be supplied.

## Setup and mutable roots

Python 3.10+ is required. First obtain the audited `regions.json` for the study
or an audited `regions.template.json`. The setup script copies the latter only
when `regions.json` is absent; with neither file present it stops at that copy.
Dependencies are listed in `src/drone_emission/requirements.txt` and are not
fully version-pinned, so this is not yet a locked software environment.

```bash
bash scripts/setup.sh
source .venv/bin/activate
```

Runtime locations are independent and may be overridden without editing code:

- `DRONE_EMISSION_DATA_ROOT`: raw inputs, compatible city results, and national aggregates;
- `DRONE_EMISSION_FIGURES_ROOT`: staging PDFs, subfigures, and figure reports;
- `DRONE_EMISSION_CACHE_ROOT`: reusable computational caches;
- `DRONE_EMISSION_WORK_ROOT`: paper-analysis compatibility view and derived results;
- `DRONE_EMISSION_GBA_ROOT`, `DRONE_EMISSION_BEIJING_OD_ROOT`,
  `DRONE_EMISSION_SALES_PATH`, and `DRONE_EMISSION_GEF_PATH`: optional explicit inputs.

The existing `data/{level}/{region}/` layout and national filenames are not
migrated. Remote defaults resolve under `/mnt/minor_storage/$USER/drone-emission`;
the root resolver uses `$USER`. Historical code and provenance records may
still contain machine-specific paths.

## Workflow stages

| Command | Function and writes |
|---|---|
| `python -m drone_emission inputs` | Validates the experiment, region registry, merchant-sales table, grid factors, and configured roots. It does not mutate data. |
| `python -m drone_emission calibrate` | Jointly estimates Beijing building-volume and Tanner-distance parameters from corrected orders and preprocessed GBA buildings. Results are candidates unless `--accept-result` is explicit. |
| `python -m drone_emission calibrate-coordinates` | Estimates the audited global Beijing coordinate translation, or applies an existing transform with `--apply-transform` without loading POIs or regenerating audit figures. |
| `python -m drone_emission cities` | Runs merchant filtering, building preparation, OD construction, road routing, lifecycle emissions, and per-city stats. It never imports plotting code. Completed cities are restored unless `--force` is supplied. |
| `python -m drone_emission aggregate` | Consolidates compatible `all_region_stats.{json,csv}` and `all_region_stats_enriched.csv`. |
| `python -m drone_emission cohort` | Audits post-aggregate city results against the tracked 311-city paper cohort. The minimum is 30 valid weighted OD pairs; the candidate audit is written below work and a changed set fails until explicitly reviewed and accepted. |
| `python -m drone_emission analyze` | Restores or computes named scenarios, distance thresholds, threshold resilience, city attribution, and the full-OD contribution decomposition for the locked cohort. Use `--only contribution-decomposition` for that analysis alone and `--contribution-chunk-rows` to tune its bounded-memory reader. `--reuse-from DIR` restores one explicit SHA-256-audited directory without automatically discovering legacy files. |
| `python -m drone_emission render` | Accepts 13 manifest IDs through `--only` or 32 individual render requests through `--request` (including the objective-tradeoffs candidate). It writes panels to `paper/subfigure/` and source reports to `paper/report/`. Only selected requests that consume analysis products require the corresponding completed analysis modules. |
| `python -m drone_emission --env local compose` | Converts rendered panel PDFs to linked SVGs and exports the 12 manuscript Inkscape layouts to their formal staging PDFs. Use `--prepare-links` before linking panels in Inkscape and `--only FIGURE_ID` for one composition. |
| `python -m drone_emission promote` | Validates all 13 manifest staging PDFs, copies the 12 manuscript assets to `docs/paper/figures/`, and copies the candidate to its separate review path. It requires the candidate staging PDF as well as the manuscript set. Use `--dry-run` or explicitly confirm with `--yes`. |
| `python -m drone_emission status` | Reports staging and required-subfigure completeness without writing. |
| `python -m drone_emission reproduce` | Runs inputs → cities → aggregate → cohort → analyze → render. Existing compatible results are reused; `--force` deliberately recomputes them. It never promotes. |

For an audited cleanup of legacy routed OD data, run the locked city batch with
`--start-at road-clean`. It scans every locked city, stages only cities containing
non-finite or non-positive road distances (or inconsistent OD statistics), and
recomputes emissions without loading the road network. Bind `city-promote` to the
resulting `--run-dir` to publish only the cleaned OD and region statistics.

The 12 manuscript entries declare Inkscape sources; the additional candidate
uses a direct PDF output. Their rendered panel PDFs are
converted into external SVG links by the local `compose` command. Inkscape's
internal PDF importer keeps labels as editable SVG text instead of converting
font glyphs to paths. The separately supplied
master SVGs remain the layout authority: `compose` reads but never rewrites
them. For PDF export, linked image elements are inlined as nested SVG elements
in a temporary copy so the panel content remains vector; the temporary copy is
then removed. `supplementary.tex` is retained but is outside this workflow and
its publication contract.

For the initial migration, run `compose --prepare-links`, then import the files
from `figures/inkscape/linked-subfigures/` into each master SVG using Inkscape's
Link mode. Linked panels may be moved, scaled, or clipped, and annotations can
be drawn above them. Converting a linked panel to editable paths deliberately
ends automatic refresh for that panel. After saving the master SVG, run
`compose`; unchanged compositions are skipped by modification time and
`--force` refreshes all selected links and staging PDFs.

Rendering is read-only with respect to configured data and city results. It
does not run `analyze`, generate placeholders, or silently omit panels. Outputs
are bounded to the configured figures/cache/work roots; only `promote` may
write the manuscript figure directory.

The contribution decomposition is reduced serially, one city and one CSV
chunk at a time, into a cohort-complete 311-row analysis table. Per-city
results are content-fingerprinted and resumable. The contribution-decomposition figure (Main Figure 4 in the manuscript,
`fig5_contribution_decomposition` in the code) reads only that
compact table; it never scans OD files or performs lifecycle analysis while
rendering.

The paper cohort is declared in `src/drone_emission/paper_cohort.json`. It
contains 311 cities, with cohort SHA-256
`598e5f1adf61737101560eaa3d1a02271aec7a72345787029ce1ca4dc3a445e8`.
The previous 313-city set excluded Linzhi for insufficient valid weighted OD
support; the current set additionally excludes Hong Kong and Macau because
merchant geography is inconsistent. The exclusion reasons and fallback cities are recorded in the cohort contract.
The three retained cities without a provincial GEF match (Changdou, Lasa and
Shannan) use the paper-locked 2022 national mean of 0.5366 kg CO2e/kWh. Aggregate files may still describe every configured city;
analysis and rendering consume a cohort-versioned, filtered work view.

For low-cost regression, `analyze --reuse-from DIR` consumes one explicit
SHA-256-audited reuse directory matching the current cohort and schema.
Historical 313-city migration instructions and bundles must not be treated as
the current 311-city reproduction inputs.

## Formal remote launcher and memory limit

`entrypoint.sh` is the only launcher that promises a process-tree memory cap.
It does not initiate SSH and requires an explicit stage:

```bash
./entrypoint.sh --jobs 4 cities
./entrypoint.sh --jobs 4 cities --cohort locked --run-dir logs/paper-cities
./entrypoint.sh --jobs 4 --tmux drone-reproduce reproduce
./entrypoint.sh cities --region chengdu --force --no-fetch
```

The launcher creates one parent user-systemd scope with a default hard
`MemoryMax=96G` and `MemorySwapMax=0`; every stage and worker inherits it. The
city scheduler reserves 12GB for the surrounding process tree and 21GB per city,
so the default maximum is four workers. `--jobs` is an upper bound and values
above the budget are rejected. Special runs may explicitly override
`DRONE_EMISSION_MEMORY_MAX_GB`, `DRONE_EMISSION_SYSTEM_RESERVE_GB`, or
`DRONE_EMISSION_CITY_RESERVATION_GB`. A city reservation is a scheduling value,
not a per-city cgroup. OOM failures are reported and are never retried. If
systemd/cgroup support is absent, the launcher stops instead of running without
the cap. Tmux is opt-in through `--tmux NAME`; foreground execution is default.

Direct `python -m drone_emission ...` commands remain useful for development
and single-stage work but do not carry the formal 96GB guarantee.

For an audited paper-cohort batch, provide `--cohort locked` and a new
`--run-dir`. The package scheduler records every city's start, end, exit code,
and dedicated log while keeping all shared status writes in the parent process.
Failures are collected without automatic retries and make the batch exit
nonzero. An explicit `--resume` skips only cities already recorded with exit
code zero, and only when the source commit, cohort, environment, worker count,
and city flags match the original batch manifest.

An interrupted city that already wrote valid OD pairs can resume without
repeating sales, building, or demand preparation:

```bash
python -m drone_emission --env remote cities \
  --region chongqing --start-at roadnet --no-fetch
```

Explicit `--no-fetch` is strict: missing building inputs and missing,
unreadable, or undersized road-network inputs fail instead of initiating a
download.

## Historical material and retired commands

`archive/code/` and `archive/scripts/` preserve retired implementations for
historical inspection and may describe superseded methods or commands. They
are not imported by the active package. Old proposal, poster, slide and internal
operations documents are kept in a local ignored archive and are not part of
this review release. Use the current package CLI, cohort contract and accepted
manuscript evidence for reproduction.

## Retired command map

| Retired entry point | Replacement |
|---|---|
| `cd src/drone_emission && python main.py --region CITY` | `python -m drone_emission cities --region CITY` |
| old positional `entrypoint.sh 4 --all` | `./entrypoint.sh --jobs 4 cities` |
| `postprocess.sh` | `python -m drone_emission aggregate` then `analyze` |
| OAT/Sobol/Shapley/equity/profit/Kappa and maintenance shells | `archive/code/` or `archive/scripts/` |

See [archive/MANIFEST.md](archive/MANIFEST.md) for retired-code recovery guidance.
Working documents under `docs/` are maintained locally and are not shipped in
this repository; the workflow and availability information above describe the
public checkout.

## Calibration and data handling

`calibrate` and `calibrate-coordinates` are optional, separate audit workflows;
`reproduce` does not rerun or accept calibration. The accepted demand parameters
are in `experiment.toml` and `calibration/gravity_calibration_summary.json`.
A new calibration remains a candidate until explicitly accepted with
`--accept-result`. The coordinate correction is a global-translation assumption;
spatial alignment is an audit check, not held-out model validation.

Raw and corrected observed orders, customer destinations, point-level audit
exports, caches and logs must stay outside version control. The tracked
calibration products include a coordinate-transform baseline and spatially
referenced aggregate endpoint counts; these are not raw order records, but
their sharing permissions still need to be checked against the original data
terms. Do not infer redistribution rights from a file being downloadable or
from its presence in this repository.
