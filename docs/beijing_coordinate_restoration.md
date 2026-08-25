# Beijing order coordinate restoration

## Evidence boundary

The manuscript and Supplementary Information show that the Beijing Eleme
orders were spatially aligned with real administrative districts, commercial
areas, Amap POI/AOI, buildings, and OpenStreetMap roads, but they do not report
the coordinate system, geomasking operation, or numerical offset. Relevant
visual and textual anchors are the manuscript's Fig. 2 (pp. 3–4), Fig. 6
(p. 7), Data availability statement (p. 11), and Supplementary Information
(p. 13). They include Beijing-wide order patterns, Sanlitun, Qinghe–Shangdi,
Miyun Airport, district distributions, a 1.3 km mean straight-line distance,
and 96.64% of orders within 3 km. Those observations are validation anchors;
they cannot identify a unique coordinate transformation by themselves.

The numerical correction instead comes from repository history. The retired
calibration code records a match of all 11,633 unique shops against 2022
Beijing restaurant POIs, with longitude offset `-5.1192°`, latitude offset
`+0.7810°`, 11,267 shops (about 97.0%) within 100 m, and a mean nearest-POI
distance of about 28.8 m. The source-controlled
`calibration/beijing_coordinate_transform.json` preserves that reviewed
baseline and fingerprints the inputs available when it was recorded. Metrics
that were not present in history are deliberately `null`, not reconstructed.

## Model and command

The calibration model is one global constant translation:

```text
corrected_gcj02 = original_coordinate + (longitude_offset, latitude_offset)
```

This is an explicit assumption, not a claim that all geomasking uses
translation. Geomasking can also use aggregation, affine transformations, or
random perturbation. In this dataset, preservation of pickup–delivery relative
distances and the high match rate from one offset support the simpler global
translation model. See the [geomasking methods
review](https://pmc.ncbi.nlm.nih.gov/articles/PMC4590956/) for the wider method
space.

Run the independent audit from the repository root:

```bash
python -m drone_emission --env local calibrate-coordinates
```

Optional arguments are `--orders`, `--poi`, `--output`,
`--poi-datum {gcj02,wgs84}`, `--match-radius-m`, `--write-orders`,
`--no-basemap`, and `--heatmap-gridsize`. Defaults select the round-1 Beijing
training orders and the 2022 Beijing restaurant POI shapefile. POIs are treated
as GCJ-02 by default even though the supplied `.prj` declares WGS 84; this
declared-versus-adopted mismatch is recorded in the audit JSON.

The estimator performs a coarse-to-fine KD-tree search around the historical
offset. Candidate offsets are compared in Beijing UTM zone 50N: first maximize
the number of shops within the selected match radius (100 m by default), then
minimize the mean nearest-POI distance over all shops. All 11,633 unique shops
participate; there is no train/holdout interpretation. The 2020 orders and 2022
POIs are temporally mismatched, but are used only to estimate a city-level
translation.

## Outputs and interpretation

The default output is below the configured work root at
`calibration/coordinate_restore/` and contains:

- `coordinate_transform.json`, including offsets, search settings, input
  SHA-256 fingerprints, datum decisions, counts, threshold match rates, and
  nearest-neighbour error summaries;
- `corrected_shops.csv.gz`, retaining original coordinates and appending
  corrected GCJ-02/WGS84 coordinates and nearest-POI audit fields;
- `coordinate_alignment_audit.{pdf,png}` with its search/local-link source CSV;
- `beijing_supply_demand_heatmap.{pdf,png}` with its compressed source CSV;
- `corrected_orders.csv.gz` only when `--write-orders` is explicitly supplied.

The first audit figure includes the search objective, the Sanlitun and
Qinghe–Shangdi anchors, and one data-selected typical central and peripheral
area. Every local view uses a fixed 2 km radius and shows shop-to-nearest-POI
links. These are descriptive alignment checks, not held-out validation. The
second figure compares order-weighted pickup supply and delivery demand using
identical Beijing extents, hexagon resolution, and a shared logarithmic color
scale.

Basemaps use the configured contextily cache and may fetch missing OSM tiles.
A basemap failure is an error; `--no-basemap` is the only way to authorize an
unmapped version. These outputs are calibration evidence and are not registered
in the formal paper figure manifest.

## Privacy and reuse

Runtime outputs are ignored and may contain sensitive customer destinations.
Do not commit corrected orders, heatmap source points, shops, figures, caches,
or logs. The tracked baseline contains hashes and aggregate statistics only,
never customer coordinates. Before sharing an audit bundle, review whether the
point-level shop/link and demand source CSVs require aggregation or access
controls.
