# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a research project comparing carbon emissions between drone delivery and e-bike delivery systems. It processes geographic data (buildings, road networks, sales data) to generate origin-destination (OD) pairs and calculate life cycle emissions for delivery scenarios.

## Running the Analysis

The main entry point is `src/drone_emission/main.py`. Run from the `src/drone_emission` directory:

```bash
cd src/drone_emission
python main.py --region nanshan
```

The `--region` flag selects which region to analyze (defined in `regions.json`).

### Shell scripts layout

Top-level scripts at the repo root run the pipeline or one of its stages:

- `entrypoint.sh` — main pipeline (preprocessing → buildings → demand → roadnet → emission → equity)
- `run_sensitivity.sh` — sensitivity-analysis stage (depends on emission-stage OD CSVs)
- `run_network_metrics.sh` — OSMnx network metrics (post-hoc, reads existing GraphML files)
- `postprocess.sh` — consolidate per-region stats + cross-region analysis + report

Everything else lives under `scripts/` (utilities, surgical reruns, one-time setup): `scripts/setup.sh`, `scripts/sync_data.sh`, `scripts/test_run.sh`, `scripts/log_summary.sh`, `scripts/rerun_affected_{clean,run}.sh`, `scripts/download_figures.sh`, `scripts/rsync_retry.sh`. Run them as `bash scripts/<name>.sh ...`.

### Canonical rerun sequence

To rerun the whole analysis end-to-end:

```bash
# 1. Main pipeline
./entrypoint.sh --all                  # or: ./entrypoint.sh 4 --all (4 parallel)

# 2. Sensitivity (skips regions whose 3 CSVs already exist)
./run_sensitivity.sh                   # 4 parallel, remote (default)

# 3. (Optional) Network metrics — only if you need updated OSMnx topology stats
./run_network_metrics.sh

# 4. Post-process (consolidate stats + cross-region analysis + report + snapshot)
./postprocess.sh

# 5. (Optional) Cross-region sensitivity tables
cd src/drone_emission && ../../.venv/bin/python -m tools.consolidate_sensitivity
```

### entrypoint.sh

Wraps execution in tmux with resource limits, logs to `./logs/`:
```bash
./entrypoint.sh                # Sequential, remote env (default)
./entrypoint.sh 4              # 4 parallel jobs, remote env
./entrypoint.sh 1 local        # Sequential, local env
```
Add `--all` to process every region in `regions.json` instead of the hardcoded list:
```bash
./entrypoint.sh --all          # Sequential, all regions
./entrypoint.sh 4 --all        # 4 parallel jobs, all regions (each region in its own process)
```

### Monitoring

Summarize pipeline log (progress, failures, active workers):
```bash
bash scripts/log_summary.sh    # latest log in ./logs/
```

### Standalone tools

Consolidate per-region stats into a single summary:
```bash
cd src/drone_emission
python -m tools.consolidate_stats            # merge all per-region stats
python -m tools.consolidate_stats --migrate  # one-time migration from old regions.json
```

Run emission model validation and sensitivity analysis directly:
```bash
python -m emission.cal_single_delivery
```

## Configuration Files

- **`experiment.toml`**: Experiment parameters (data directories, delivery radius, OD generation params)
- **`regions.json`**: **Read-only** static region definitions (boundaries, names, levels). Never written during pipeline runs.
- **`data/{level}/{city}/region_stats.json`**: Per-region runtime stats and cache. Each process writes only its own region's file, enabling safe parallel execution.

## Architecture

### Pipeline Stages (in order)

1. **preprocessing** - Prepare sales data from raw sources
2. **buildings** - Fetch and process building footprints from external APIs
3. **demand** - Generate OD pairs using gravity model (store → building matching)
4. **roadnet** - Fetch OSM road network, compute road distances for OD pairs
5. **emission** - Calculate and compare drone vs e-bike emissions
6. **equity** - Zone-level equity analysis of emission reductions

All stages above are active in `main.py:run_region_analysis()`. The `profit` stage exists but is currently commented out.

### Key Modules

- **`emission/cal_single_delivery.py`**: Core emission models with `EbikeEnergyModel`, `DroneEnergyModel`, and `LifeCycleEmissions` classes. Contains detailed physics-based power calculations.
- **`roadnet/`**: Road network loading and routing. Includes `sparse_graph.py` and `tile_router.py` for efficient shortest path computation with KD-tree optimization and batch processing.
- **`demand/api.py`**: Gravity model implementation for generating delivery demand between stores and buildings.
- **`equity/`**: Zone-level equity analysis of emission reductions (grid-based zones, feature extraction, emission comparison).
- **`profit/`**: Profit/cost analysis comparing drone vs e-bike delivery economics (currently disabled in pipeline).
- **`analysis/`**: Cross-region analysis and correlation studies. Not part of the main pipeline; used for post-hoc research.
- **`calibration/`**: Model calibration scripts (Eleme order data, mean distance, milk-run). Not part of the main pipeline.
- **`tools/`**: Standalone CLI utilities (run via `python -m tools.<name>`):
  - `consolidate_stats.py`: Merges all per-region stats into `data/all_region_stats.json` and `.csv`. Use `--migrate` for one-time migration of stats from `regions.json` to per-region files.
  - `generate_city_configs.py`: Auto-discovers cities from sales data and generates `regions.json` entries with boundaries fetched from Aliyun DataV API.
- **`visualization/`**: Plotting functions for all stages (buildings, demand, emission, equity, profit, roadnet, sales, cross-region analysis).
- **`utils/region_config.py`**: `RegionRegistry` singleton pattern for managing region configurations. Loads static config from `regions.json` and per-region stats from `data/{level}/{city}/region_stats.json`. Use `registry.save_region(key)` to persist a single region's stats.

### Data Flow

```
regions.json (static config) + experiment.toml
        ↓
    RegionInfo + ExpConfig
        ↓
    [Stage APIs receive (region_info, exp_config)]
        ↓
    data/{level}/{region}/*.csv, *.geojson, *.graphml
    data/{level}/{region}/region_stats.json  ← per-region stats
        ↓
    figures/{level}/{region}/*.png
        ↓
    tools/consolidate_stats.py → data/all_region_stats.{json,csv}
```

### Coordinate Systems

- Building data: WGS84 (EPSG:4326)
- Store data: GCJ02 (Chinese coordinate system) - converted to WGS84 using `utils/coordinates.py`
- Road network: WGS84 via OSMnx

## Dependencies

Key libraries: osmnx, networkx, geopandas, haversine, scikit-learn (BallTree for spatial indexing)

Install with:
```bash
pip install -r src/drone_emission/requirements.txt
```

## LaTeX Documents

LaTeX source files are in `docs/`. Compilation is configured via VS Code LaTeX Workshop.

### Build Output
- Each document's **auxiliary files** (.aux, .log, .bbl, etc.) go to a `build/` subdirectory relative to its `main.tex` (e.g. `docs/slides/build/`). Use `-output-directory=build` when compiling from the document's directory.
- **PDF files** → `docs/pdf/`

### Recipes
- **"xelatex -> biber -> xelatex*2 -> copy"**: For documents with bibliography (slides, proposal)
- **"xelatex -> copy"**: For simple documents without bibliography (poster)

### Documents
- `docs/slides/main.tex`: Beamer presentation for PhD confirmation seminar
- `docs/proposal/main.tex`: Research proposal document
- `docs/poster/main.tex`: Conference poster
- `docs/paper/main.tex`: Nature Sustainability paper (separate git repo, synced with Overleaf)

### Paper (`docs/paper/`)
- **Target:** Nature Sustainability (≤5,000 words main text, ≤6 display items, ≤50 refs)
- **Sync:** Separate git repo cloned from Overleaf. `git push` from `docs/paper/` to sync. Listed in main repo's `.gitignore`.
- **Template:** Springer Nature `sn-jnl.cls` with `sn-nature` style, compiled with `pdflatex`
- **Build:** `cd docs/paper && pdflatex main && bibtex main && pdflatex main && pdflatex main`
- **Structure:** Introduction → Results & Discussion → Conclusion → Methods (after refs)
- **Style:** Nature family — results-first, no separate lit review, policy-oriented framing, number-rich, multi-panel figures. See `docs/paper/reviews/summary_*.md` for reference paper summaries.
- **Sections:** `sections/{intro,results,method,conclusion}.tex` — currently contain structured TODO outlines
- **Bibliography:** `sn-bibliography.bib` (copied from `docs/shared/delivery_emission.bib`)
- **Figures:** `figures/fig1-fig6` (6 main figures, copied from `figures/analysis/` and `figures/lifecycle/`)
