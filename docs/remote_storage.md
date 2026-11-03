# Remote storage organization

The remote storage root is `/mnt/minor_storage/$USER/drone-emission`.
The code checkout remains `/home/jingxian/code/drone-emission`; its local
`data`, `cache`, `figures`, and `logs` directories are separate directories,
not aliases of the storage root. In particular, the default remote figure
root is still the code checkout's `figures` directory. Do not merge these
trees based on directory names alone.

| Directory | Purpose and retention |
| --- | --- |
| `data/` | Authoritative inputs, city results, and aggregates. Keep paths unchanged. |
| `gba_data/` | Building inputs and conversion state. Preserve lock files. |
| `cache/` | Computational caches and prepared figure bundles. Prepared bundles are retained inputs, not disposable temporary files. |
| `work/` | Cohort audits and fingerprinted paper analyses. Preserve paths and links. |
| `paper-reuse/` | Explicit audited reuse sources. Keep until their consumers are retired. |
| `runs/` | City-run manifests, status, and recovery records. |
| `staging/` | Unaccepted city results, paired with their run records. |
| `figures/` | Explicitly directed storage-side figure outputs; not the default figure root. |
| `logs/` | Existing pipeline logs. New isolated batches may keep logs inside the batch. |
| `review/` | New figure-review batches. |
| `validation/` | New isolated validation batches. |
| `experiments/` | New exploratory analyses. |
| `archive/figure-validation/` | Audited historical figure and validation batches. |
| `archive/experiments/` | Audited historical exploratory batches. |
| `archive/backups/` | Reserved for backups whose dependencies have been audited. |
| `inventory/` | Organization audit batches, references, migration journals, and rollback scripts. |

Create new batches below the appropriate parent using
`purpose_YYYYMMDD_commit` (Asia/Shanghai date), with a random or numeric suffix
when needed. Never overwrite an existing batch. The isolated validation
script creates `validation/isolated_YYYYMMDD_commit.XXXXXX`.

## Archiving existing batches

Archive whole batches while preserving their names and internal structure.
Before moving one, check active jobs, inbound symbolic links, references from
other batches and the code checkout, and links leaving the batch. Retain
directories with unresolved dependencies and record the reason in the audit.
Do not infer that a directory is disposable from its name or age.

Each organization operation must record its exact reference-scan scope and
limitations. The initial operation scans all symbolic links, without following
them, and searches configuration, source, report, log, SVG/XML, and TSV files
in the storage and code trees. It excludes Git internals, virtual environments,
and Python bytecode. Large scientific CSV tables and binary data are not
searched for embedded text paths. This is a static dependency audit, not a
pipeline execution test.

Use same-filesystem renames, refuse destination collisions, and compare
relative file metadata and symbolic-link targets before and after each move.
Record completed moves in a journal with a rollback script that refuses
overwriting a recreated original path. Do not delete data, rewrite historical
provenance, add compatibility links, or clean caches as part of organization.

`data/backups`, `data/_backup`, dated backups, old postprocessing outputs,
`runs`, and `staging` require a separate dependency and acceptance audit.
Some legacy data files are still linked from `work/analysis`; leave them in
place. Keep prepared figure bundles together with their `shared_assets`.

Run remote commands through one persistent SSH TTY, as required by
`AGENTS.md`. No Python execution or pipeline rerun is needed for this
filesystem-only operation.
