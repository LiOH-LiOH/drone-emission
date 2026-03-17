# Drone vs E-bike Delivery Emissions

This project provides a country-scale comparison of life-cycle carbon emissions between drone and e-bike food delivery in China. As drone delivery infrastructure expands rapidly — the market is projected to grow from $3.47B (2025) to $20.98B (2034) — there remains a lack of predictive models for evaluating whether drones truly offer a sustainable advantage over existing last-mile delivery methods.

Most prior studies compare drones against trucks or vans using aggregate statistics. This work instead compares drones directly against e-bikes (the dominant mode for food delivery in China) using real origin-destination data. The pipeline generates door-to-door delivery demand from Meituan merchant sales data and building footprints via a Tanner impedance gravity model calibrated on ~278K Eleme orders, computes both air distances (drone) and road network shortest paths (e-bike) from OpenStreetMap, and performs physics-based life-cycle emission assessment covering use-phase energy, manufacturing, and end-of-life.

Key findings across studied regions show that an all-drone scenario can reduce emissions by 27-33% versus all-e-bike, while an optimal drone-e-bike mix achieves 30-36% reduction. The emission advantage is strongly correlated with road network complexity and detour ratio.

## Setup

Requires Python 3.10+.

```bash
bash setup.sh              # create venv and install dependencies
bash setup.sh python3.11   # specify python version
```

## Quick Start

Run the analysis pipeline for a single region:

```bash
cd src/drone_emission
python main.py --region nanshan
```

The `--region` flag selects which region to analyze (defined in `regions.json`).

## Usage

### Full pipeline (all regions)

`entrypoint.sh` wraps execution in a tmux session with resource limits:

```bash
./entrypoint.sh                # 4 parallel jobs, remote env (default)
./entrypoint.sh 8              # 8 parallel jobs
./entrypoint.sh 1 local        # sequential, local env
./entrypoint.sh 4 --all        # all regions from regions.json
```

### Post-processing

Consolidate per-region stats and run cross-region analysis after the pipeline completes:

```bash
./postprocess.sh               # remote env (default)
./postprocess.sh local         # local env
```

## Pipeline Stages

The pipeline runs the following stages in order for each region:

| Stage | Description |
|-------|-------------|
| **preprocessing** | Prepare sales data from raw sources |
| **buildings** | Fetch and process building footprints from external APIs |
| **demand** | Generate OD pairs using a gravity model (store-to-building matching) |
| **roadnet** | Fetch OSM road network, compute road distances for OD pairs |
| **emission** | Calculate and compare drone vs e-bike life-cycle emissions |
| **equity** | Zone-level equity analysis of emission reductions |

## Project Structure

```
src/drone_emission/
├── main.py                  ← Pipeline entry point
│
├── preprocessing/           ┐
├── buildings/               │
├── demand/                  │ Pipeline stages (per-region)
├── roadnet/                 │
├── emission/                │
├── equity/                  │
├── profit/                  ┘
│
├── analysis/                ┐ Non-pipeline analysis
├── calibration/             ┘
│
├── tools/                   ← Standalone CLI utilities
│   ├── consolidate_stats.py
│   └── generate_city_configs.py
│
├── utils/                   ┐ Shared
└── visualization/           ┘
```

## Configuration

- **`experiment.toml`** — Experiment parameters: delivery radius, OD generation params, emission model physics (e-bike, drone, e-car, gasoline car), equity grid settings, and per-environment overrides.
- **`regions.json`** — Read-only static region definitions (boundaries, names, levels). Never written during pipeline runs.
- **`data/{level}/{city}/region_stats.json`** — Per-region runtime stats and cache. Each process writes only its own file, enabling safe parallel execution.

## Data Flow

```
regions.json + experiment.toml
        ↓
    RegionInfo + ExpConfig
        ↓
    Pipeline stages (per region)
        ↓
    data/{level}/{region}/          ← CSVs, GeoJSON, GraphML
    figures/{level}/{region}/       ← plots
        ↓
    tools/consolidate_stats.py
        ↓
    data/all_region_stats.{json,csv}
```

## Key Dependencies

osmnx, networkx, geopandas, haversine, scikit-learn, shapely, matplotlib, pandas, numpy
