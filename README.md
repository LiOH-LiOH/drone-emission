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
| `python -m drone_emission calibrate` | Rebuilds the Tanner fit only when raw Beijing OD text files are available. The locked TOML, calibration CSV, and audited Beijing distance histogram remain the normal inputs; rebuilt output goes to work. |
| `python -m drone_emission cities` | Runs merchant filtering, building preparation, OD construction, road routing, lifecycle emissions, and per-city stats. It never imports plotting code. Completed cities are restored unless `--force` is supplied. |
| `python -m drone_emission aggregate` | Consolidates compatible `all_region_stats.{json,csv}` and `all_region_stats_enriched.csv`. |
| `python -m drone_emission analyze` | Restores or computes named scenarios, distance thresholds, threshold resilience, spatial concentration, and city attribution. Derived files are written below cache/work through a compatibility data view; city data is read-only. |
| `python -m drone_emission render` | Accepts only the 12 manifest IDs through `--only`. It writes necessary panels to `paper/subfigure/`, reports to `paper/report/`, and the three direct figures to `paper/`. |
| `python -m drone_emission promote` | Validates all 12 staging PDFs and is the only command allowed to write `docs/paper/figures/`. Use `--dry-run` or explicitly confirm with `--yes`. |
| `python -m drone_emission status` | Reports staging and required-subfigure completeness without writing. |
| `python -m drone_emission reproduce` | Runs inputs → cities → aggregate → analyze → render. Existing results are reused; `--force` deliberately recomputes them. It never promotes. |

Nine manifest entries declare Inkscape sources and generate only the panels
needed for manual composition. Figure 4, Figure 5, and Extended Data Figure 6
are staged directly. `supplementary.tex` is retained but is outside this
workflow and its publication contract.

## Formal remote launcher and memory limit

`entrypoint.sh` is the only launcher that promises a process-tree memory cap.
It does not initiate SSH and requires an explicit stage:

```bash
./entrypoint.sh --jobs 4 cities
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

## Retired command map

| Retired entry point | Replacement |
|---|---|
| `cd src/drone_emission && python main.py --region CITY` | `python -m drone_emission cities --region CITY` |
| old positional `entrypoint.sh 4 --all` | `./entrypoint.sh --jobs 4 cities` |
| `postprocess.sh` | `python -m drone_emission aggregate` then `analyze` |
| `python -m visualization.paper` | `python -m drone_emission render` |
| OAT/Sobol/Shapley/equity/profit/Kappa and maintenance shells | `archive/code/` or `archive/scripts/` |

See [docs/architecture.md](docs/architecture.md) for the dependency and write
boundaries and [archive/MANIFEST.md](archive/MANIFEST.md) for recovery guidance.
