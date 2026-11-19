# New food-delivery data exploration

Standalone descriptive figures; this directory is not imported by the active
package or registered in the paper workflow. Run on the remote host only.

```sh
.venv/bin/python archive/code/food_delivery_exploration/explore.py all \
  --input /mnt/minor_storage/jingxian/drone-emission/data/raw/food_delivery_organized_20261119 \
  --cache /mnt/minor_storage/jingxian/drone-emission/cache/food_delivery_exploration \
  --output figures/exploration/food-delivery/BATCH
```

`prepare` builds a cached descriptive view. `render` reuses it; `all` does both.
Use a new output directory for each review. Input files remain unchanged.

The unit is a unique platform/account/receipt key, not a verified independent
order. Valid city-enriched coordinates take precedence over original coordinates;
New York zone precedes city. Remaining ties follow sorted filenames and row order.
Positive order timestamps can be filled from another record of the same receipt.
All source-region memberships survive deduplication. Regional maps preserve source
coverage (the San Francisco file includes the Bay Area).

The 2022 comparison selects local **receipt year**, not order year. Hourly and
weekly rhythms use actual `orderTime` only. Monthly receipt and order timestamps
are plotted separately. Counts describe the sample, without market expansion.
The distance is endpoint geodesic distance, not a route distance.

PNG previews and PDF/SVG exports are produced remotely; default review transfers
include PNGs and lightweight CSV/JSON/Markdown provenance only. Detailed records,
prepared inputs and downloaded map assets remain remote. Base maps are CARTO
Positron (OpenStreetMap/CARTO attribution) and Natural Earth 1:110m country outlines.
