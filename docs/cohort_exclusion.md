# Excluding Hong Kong and Macau

The active experiment contains exactly the previous 313 locked cities minus
`xianggangtebiexingzhengqu` and `aomentebiexingzhengqu`: 311 cities, hash
`598e5f1adf61737101560eaa3d1a02271aec7a72345787029ce1ca4dc3a445e8`.
The original ten merchant records have mainland addresses conflicting with
their province/city labels. They remain raw evidence and are not reassigned.

`paper_cohort.json` records these as `invalid_merchant_geography` exclusions.
Unlike exclusions for incomplete computation, these remain ineligible even
when historical city outputs are present. City CLI requests reject them before
cache restoration; batch city processing and aggregate omit them. Cohort
acceptance cannot reintroduce them. The national fallback GEF stays 0.5366;
the remaining fallback cities are Changdou, Lasa and Shannan.

## Audited subtraction

`drone_emission.tools.migrate_cohort_exclusion` requires an explicit source work
root and a new output root. It verifies the reviewed 313-city manifest, city
stats hashes, OD metadata and the six needed analysis modules. It never
searches other roots or runs a missing prerequisite. The old schema-1 reuse
bundle is not used as a schema-6 bundle.

The authoritative source aggregates can contain all 328 configured cities.
Select the original locked 313 before subtraction; keep original aggregate row
indices for curve sampling rather than resetting the selected frame's index.
All independent city scientific results are preserved. Cohort metadata columns
are rebound with the old source hash retained in the migration report. City
attribution is refitted from the existing city feature table. Full-OD exact
scenario results are filtered; only pathway curves are re-evaluated on the
existing 1,500-row-per-city sampling rule. A streaming three-column OD scan
reconstructs the original random-row sample and annual-sales scale. Original
city indices/seeds are preserved. The old national curve must be reproduced
before the 311-city curve is accepted. Small samples and per-city curve points
are persisted to support future subtraction without rescanning OD files.

Example remote migration (new batch path required):

```sh
.venv/bin/python -m drone_emission.tools.migrate_cohort_exclusion \
  --source-work-root /mnt/minor_storage/jingxian/drone-emission/work \
  --output-root /mnt/minor_storage/jingxian/drone-emission/review/NEW_BATCH
```

Set the data/work/cache/figures root environment variables to this batch, then
restore only the six recorded modules using the package CLI:

```sh
.venv/bin/python -m drone_emission --env remote analyze \
  --reuse-from "$BATCH/reuse" \
  --only named-scenarios,pathway-scenarios,distance-threshold,threshold-resilience,city-attribution,contribution-decomposition
```

## Review figures

The existing break-even observed-context renderer uses only cities with an
explicit provincial GEF. Its 308-city, 616,000-OD sample does not include either
Hong Kong or Macau, so this observed sample is unchanged after subtraction.
`--reuse-break-even-cache SOURCE_PKL` validates the cache's sampling contract,
retained city keys, per-city counts, GEFs, input age, and analytic mesh, then
stages it under the new input signature. It avoids a second national OD scan.
The three remaining fallback cities are explicitly reported as omitted from
this observed-context panel, preserving the previous scientific definition.

Fifteen affected render requests cover ten existing composites. Sixteen
requests and two wholly unchanged composites are reused from the accepted
local figures. The new objective-tradeoffs candidate is outside this migration.
Use `render --request` for affected requests only; do not run a full render or
reproduce. No colors, typography, layout or geographic base layers change.

**Composition is local.** After the remote render, download only the 30 affected
subfigure PDFs plus lightweight provenance into a new local review batch.
Do not install Inkscape or compose figures on the remote server. Reuse the
existing local 16 subfigure PDFs, their linked SVGs, the two unchanged final
PDFs, and the accepted local composition templates.

`drone_emission.tools.compose_cohort_review` stages this local assembly. It
requires the new analysis manifest and a remote SHA-256/size inventory of the
downloaded PDFs, copies only the unchanged local artifacts, and verifies all
template hashes. It creates a separate composition-only bundle with the new
cohort reference; the accepted prepared scientific bundle is never rewritten.

```sh
.venv/bin/python -m drone_emission.tools.compose_cohort_review \
  --output-root "$LOCAL_BATCH" --accepted-figures-root figures
```

Set `DRONE_EMISSION_FIGURES_ROOT` to `$LOCAL_BATCH/figures`, then use the
manifest-backed local `compose --only` for the ten affected composites. Run
local composition outside the sandbox. The copied linked SVGs for the other
sixteen requests need no reconversion. The two unchanged finals are not
recomposed. Use `compose_cohort_review --verify` to check the remote PDF hashes,
retained artifacts and the twelve final PDFs and write current source reports.

`compose --preview-only --only FIGURE_IDS` then refreshes the twelve PNG
previews and overview locally, without rerendering or recomposing. The unrelated
objective-tradeoffs candidate is never started.

The migration creates an audited schema-6 reuse manifest, numeric migration
report, source input/status snapshots, figure reuse report and final review
verification. The user explicitly requests remote subfigure PDF transfer for
local assembly; this supersedes the default remote-preview-only transfer.
Formal data, caches and manuscript PDFs are not replaced until the
review batch is accepted. Historical reports and manuscript prose are not
silently rewritten by this operation.
