# Audited 313-city rerun

The audited runner only accepts the locked paper cohort, strict offline inputs,
an empty staging data root, and a clean committed worktree. Run all remote
commands in one persistent `hku-sto` SSH TTY.

```bash
export DRONE_EMISSION_MEMORY_MAX_GB=96
./entrypoint.sh --jobs 4 cities \
  --cohort locked \
  --run-dir /mnt/minor_storage/$USER/drone-emission/runs/313-<version> \
  --source-data-root /mnt/minor_storage/$USER/drone-emission/data \
  --staging-data-root /mnt/minor_storage/$USER/drone-emission/staging/313-<version> \
  --force --no-fetch
```

Preflight validates all 313 building and GraphML inputs, GraphML bbox coverage,
and free disk space. It links only raw buildings and GraphML files into staging,
then freezes a light/medium/heavy resource plan. Anqing, Beijing, and Chongqing
run serially as canaries before the remaining cities are scheduled against the
84 GiB budget. A failure is recorded and is never retried during that invocation.

Repair a recorded failure through the original manifest. Starting at roadnet is
accepted only when the staged OD checkpoint passes the integer-demand and schema
audit.

```bash
./entrypoint.sh --jobs 4 cities \
  --cohort locked --run-dir <same-run-dir> --resume --repair <region-key> \
  --start-at roadnet --no-fetch
```

After all cities pass, inspect `city_status.tsv`, `state.json`, and the staging
root's `acceptance_report.json`. Publishing is a separate, explicitly confirmed
transaction. It retains backups until every replacement succeeds, rolls back a
partial failure, and removes backups after success.

```bash
./entrypoint.sh city-promote \
  --source-data-root <formal-data-root> \
  --staging-data-root <staging-data-root> --yes
```

Then point `DRONE_EMISSION_DATA_ROOT` at the formal data root and run `aggregate
--force`, `cohort`, `analyze --force`, and `render --force` in order. Figure
promotion to `docs/paper` remains a separate manual action.
