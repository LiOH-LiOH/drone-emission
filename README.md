# Drone-delivery emission paper reproduction

This repository reproduces the city experiments, nationwide analysis, and the
12 locked figures used by the active manuscript. Commands are run from the
repository root through the installed `drone_emission` package.

## Setup and mutable roots

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
no username is embedded in source code.

## Workflow stages

| Command | Function and writes |
|---|---|
| `python -m drone_emission inputs` | Validates the experiment, region registry, merchant-sales table, grid factors, and configured roots. It does not mutate data. |
| `python -m drone_emission calibrate` | Jointly estimates Beijing building-volume and Tanner-distance parameters from corrected orders and preprocessed GBA buildings. Results are candidates unless `--accept-result` is explicit. |
| `python -m drone_emission calibrate-coordinates` | Estimates the audited global Beijing coordinate translation, or applies an existing transform with `--apply-transform` without loading POIs or regenerating audit figures. |
| `python -m drone_emission cities` | Runs merchant filtering, building preparation, OD construction, road routing, lifecycle emissions, and per-city stats. It never imports plotting code. Completed cities are restored unless `--force` is supplied. |
| `python -m drone_emission aggregate` | Consolidates compatible `all_region_stats.{json,csv}` and `all_region_stats_enriched.csv`. |
| `python -m drone_emission cohort` | Audits post-aggregate city results against the tracked 313-city paper cohort. The minimum is 30 valid weighted OD pairs; the candidate audit is written below work and a changed set fails until explicitly reviewed and accepted. |
| `python -m drone_emission analyze` | Restores or computes named scenarios, distance thresholds, threshold resilience, city attribution, and the full-OD contribution decomposition for the locked cohort. Use `--only contribution-decomposition` for that analysis alone and `--contribution-chunk-rows` to tune its bounded-memory reader. `--reuse-from DIR` restores one explicit SHA-256-audited directory without automatically discovering legacy files. |
| `python -m drone_emission render` | Accepts 12 manifest IDs through `--only` or 28 individual render requests through `--request`. It writes panels to `paper/subfigure/` and source reports to `paper/report/`. Only selected requests that consume analysis products require the corresponding completed analysis modules. |
| `python -m drone_emission --env local compose` | Converts rendered panel PDFs to linked SVGs and exports the 12 manifest-backed Inkscape layouts to their formal staging PDFs. Use `--prepare-links` before linking panels in Inkscape and `--only FIGURE_ID` for one composition. |
| `python -m drone_emission promote` | Validates all 12 staging PDFs and is the only command allowed to write `docs/paper/figures/`. Use `--dry-run` or explicitly confirm with `--yes`. |
| `python -m drone_emission status` | Reports staging and required-subfigure completeness without writing. |
| `python -m drone_emission reproduce` | Runs inputs → cities → aggregate → cohort → analyze → render. Existing compatible results are reused; `--force` deliberately recomputes them. It never promotes. |

All 12 manifest entries declare Inkscape sources. Their rendered panel PDFs are
converted into external SVG links by the local `compose` command. The tracked
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
chunk at a time, into a cohort-complete 313-row analysis table. Per-city
results are content-fingerprinted and resumable. Figure 5 reads only that
compact table; it never scans OD files or performs lifecycle analysis while
rendering.

The paper cohort is declared in `src/drone_emission/paper_cohort.json`. It
contains 313 cities: the previously retained 314-city set with Linzhi excluded
because only one valid weighted OD pair is available. Cities without a table
GEF remain in scope and use the paper-locked 2022 national mean of
0.5366 kg CO2e/kWh. Aggregate files may still describe every configured city;
analysis and rendering consume a cohort-versioned, filtered work view.

For low-cost remote regression, `prepare-reuse` builds one explicit reuse
directory from existing city data plus a previously validated spatial cache.
It refreshes lifecycle-dependent results only for Macau, Changdu, Lhasa,
Shannan, and Hong Kong, then freezes the derived inputs and their hashes. This
is a migration utility, not a bundle registry or automatic cache search.

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
`MemoryMax=64G` and `MemorySwapMax=0`; every stage and worker inherits it. The
city scheduler reserves 8GB for the surrounding process tree and 12GB per city,
so the default maximum is four workers. `--jobs` is an upper bound and values
above the budget are rejected. Special runs may explicitly override
`DRONE_EMISSION_MEMORY_MAX_GB`, `DRONE_EMISSION_SYSTEM_RESERVE_GB`, or
`DRONE_EMISSION_CITY_RESERVATION_GB`. A city reservation is a scheduling value,
not a per-city cgroup. OOM failures are reported and are never retried. If
systemd/cgroup support is absent, the launcher stops instead of running without
the cap. Tmux is opt-in through `--tmux NAME`; foreground execution is default.

Direct `python -m drone_emission ...` commands remain useful for development
and single-stage work but do not carry the formal 64GB guarantee.

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

## Retired command map

| Retired entry point | Replacement |
|---|---|
| `cd src/drone_emission && python main.py --region CITY` | `python -m drone_emission cities --region CITY` |
| old positional `entrypoint.sh 4 --all` | `./entrypoint.sh --jobs 4 cities` |
| `postprocess.sh` | `python -m drone_emission aggregate` then `analyze` |
| OAT/Sobol/Shapley/equity/profit/Kappa and maintenance shells | `archive/code/` or `archive/scripts/` |

See [docs/architecture.md](docs/architecture.md) for the dependency and write
boundaries and [archive/MANIFEST.md](archive/MANIFEST.md) for recovery guidance.
The evidence boundary, privacy rules, outputs, and interpretation of the
Beijing coordinate audit are documented in
[docs/beijing_coordinate_restoration.md](docs/beijing_coordinate_restoration.md).
The joint conditional building-choice likelihood, fixed evaluation split, cache
contract, result acceptance, and remote memory boundary are documented in
[docs/beijing_gravity_calibration.md](docs/beijing_gravity_calibration.md).
