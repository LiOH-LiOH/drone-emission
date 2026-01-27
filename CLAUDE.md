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

To run emission model validation and sensitivity analysis directly:
```bash
python -m emission.cal_single_delivery
```

## Configuration Files

- **`experiment.toml`**: Experiment parameters (data directories, delivery radius, OD generation params)
- **`regions.json`**: Region definitions with boundaries, stats, and cache. Modified at runtime to persist computed statistics.

## Architecture

### Pipeline Stages (in order)

1. **preprocessing** - Prepare sales data from raw sources
2. **buildings** - Fetch and process building footprints from external APIs
3. **demand** - Generate OD pairs using gravity model (store → building matching)
4. **roadnet** - Fetch OSM road network, compute road distances for OD pairs
5. **emission** - Calculate and compare drone vs e-bike emissions
6. **urban_factors** - Analyze urban features (population density, terrain, delivery distance) vs emission reduction potential

Each stage is controlled by uncommenting the relevant function call in `main.py:run_region_analysis()`.

### Key Modules

- **`emission/cal_single_delivery.py`**: Core emission models with `EbikeEnergyModel`, `DroneEnergyModel`, and `LifeCycleEmissions` classes. Contains detailed physics-based power calculations.
- **`roadnet/router.py`**: `RoadNetRouter` class for efficient shortest path computation with KD-tree optimization and batch processing.
- **`demand/api.py`**: Gravity model implementation for generating delivery demand between stores and buildings.
- **`urban_factors/`**: Analysis of urban features (population density, terrain, delivery distance distribution) and their relationship to emission reduction potential.
- **`utils/region_config.py`**: `RegionRegistry` singleton pattern for managing region configurations with stats/cache persistence.

### Data Flow

```
regions.json + experiment.toml
        ↓
    RegionInfo + ExpConfig
        ↓
    [Stage APIs receive (region_info, exp_config)]
        ↓
    data/{level}/{region}/*.csv, *.geojson, *.graphml
        ↓
    figures/{level}/{region}/*.png
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
