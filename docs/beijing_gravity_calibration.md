# Beijing joint gravity calibration

The active calibration estimates the conditional building-choice model

\[
P(j\mid i) \propto V_j^\alpha d_{ij}^\beta \exp(-\gamma d_{ij}),
\]

where distance is the WGS84 Haversine centroid distance in kilometres,
`alpha >= 0`, `beta >= 0`, and `gamma > 0`. The likelihood is conditioned on
each shop's complete set of finite-positive-volume buildings within 5 km; the
normalizing constant is therefore shop-specific and no global constant is
estimated. `min_sales` remains an OD-generation setting and is never applied to
calibration orders.

## Inputs and privacy boundary

First materialize the corrected 29-file order set by applying the locked,
per-file-fingerprinted transform. This mode does not read POIs, search for a new
offset, or create coordinate figures:

```bash
python -m drone_emission --env remote calibrate-coordinates \
  --orders /path/to/raw/order \
  --apply-transform \
  --transform src/drone_emission/calibration/beijing_coordinate_transform.json \
  --output /path/to/corrected_orders.csv.gz
```

Run the calibration with corrected orders and the preprocessed Beijing GBA
GeoPackage:

```bash
python -m drone_emission --env remote calibrate \
  --orders-corrected /path/to/corrected_orders.csv.gz \
  --buildings /path/to/beijing_buildings_preprocessed.gpkg \
  --transform src/drone_emission/calibration/beijing_coordinate_transform.json \
  --output /path/to/run_directory \
  --memory-limit-gb 64
```

To add or refresh only the spatial-fit metrics and third audit figure for an
existing converged run, reuse its memory-mapped cache without refitting:

```bash
python -m drone_emission --env remote calibrate-spatial-fit \
  --result /path/to/run/gravity_calibration.json
```

Raw customer coordinates, the order-to-building match table, and candidate CSR
arrays are ignored runtime data. They remain in the remote run cache and must
not be committed or copied back. The selected JSON, CSV, PDF, PNG, and task log
may be retrieved.

## Fixed preparation and evaluation contract

Deliveries are associated with a containing building first. Otherwise, the
nearest building boundary within 50 m is selected. Ties use centroid distance
and then the stable building ID. Invalid coordinates, non-positive or invalid
raw volume, missing matches, true buildings outside the 5 km shop candidate
set, and invalid centroid distances are reported separately.

SHA-256 of `shop_id` assigns one bucket in five to the unseen-shop test. The
remaining shops use their first 24 global order dates for training and their
last five dates for the temporal test. The tests are mutually exclusive. After
out-of-sample model checks, the final parameters are re-estimated from every
retained order.
Standard errors and 95% Wald intervals use shop-clustered sandwich covariance.

The primary, intuitive spatial-fit assessment also uses every retained order.
For each merchant, it compares the observed mean delivery distance with the
model's exact probability-weighted mean over that merchant's candidate
buildings. The reported merchant predictive R-squared is

\[
R^2 = 1 - \frac{\sum_i (\bar d_i-\widehat{\bar d}_i)^2}
                  {\sum_i (\bar d_i-\bar d)^2},
\]

with one equal-weighted observation per merchant; an order-weighted value is
supplementary. This is a direct prediction score, not squared correlation or a
post-hoc regression, and can be negative. Median and P90 first-Wasserstein
errors in metres show how closely each merchant's full delivery-distance
distribution is reproduced.

Building-level fit compares matched order counts with exact expected counts
`sum_i n_i P(j|i)`. It includes every building in at least one evaluated
merchant's 5 km candidate set, including buildings with zero observed orders.
Raw-count predictive R-squared is primary and log1p-count predictive R-squared
is supplementary. The two held-out merchant sets repeat merchant metrics using
training-only parameters as small validation checks; they are not the headline
result.

The cache stores candidate attributes as memory-mapped NumPy CSR arrays and
orders as shop-building counts. `--rebuild-prepared` explicitly discards only
the fingerprint-named preparation cache. The code estimates a conservative
peak and refuses to start when it is not below `--memory-limit-gb`; the remote
launcher must independently enforce `MemoryHigh=60G`, `MemoryMax=64G`, and no
additional swap.

## Results and acceptance

`gravity_calibration.json` records input fingerprints, preparation rules,
exclusions, split sizes, optimization diagnostics, clustered inference, and all
fit metrics. `spatial_fit_metrics.csv` provides the concise full-sample and
held-out merchant summaries. `merchant_spatial_fit_source.csv.gz` and
`building_order_fit_source.csv.gz` preserve the exact source rows behind the
spatial-fit audit. CSV tables also contain parameters, model metrics, data flow,
and the other plotted sources. The three audit figures are emitted as PDF and
300 dpi PNG but are intentionally absent from the paper figure manifest. The
third figure, `merchant_and_building_spatial_fit`, contains merchant mean
distance, merchant Wasserstein error, and raw/log1p building order-count panels.

The default result is a candidate and does not modify tracked configuration.
Passing `--accept-result` validates the live input hashes, schema, convergence,
and parameter constraints before updating `alpha`, `tanner_a`, and `tanner_b`
in `experiment.toml` and writing the tracked calibration summary. Better
out-of-sample performance than the current parameters is reported but is not an
acceptance gate.

The standard Tanner calibration also writes two full-sample Beijing endpoint
audits: `beijing_order_distance_distribution.pdf` compares observed and
joint-model delivery-distance orders, while `beijing_endpoint_grid_heatmap.pdf`
compares observed and joint-model expected endpoint orders on aligned 1 km
metric grids. Both use the same valid-linked-order sample and are standalone
calibration outputs rather than manifest figures.
