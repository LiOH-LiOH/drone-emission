---
name: remote-figure-test
description: Generate and fetch drone-emission paper figures on the hku-sto remote after local code has been committed and pushed. Use when the user asks to test image or figure generation on Remote, hku-sto, remote HKU storage, or says after commitandpush/commit and push that a paper figure should be generated remotely and fetched locally.
---

# Remote Figure Test

Validate formal manifest rendering on `hku-sto` without modifying official data,
figure staging, or paper outputs.

## Defaults

- Local repo: `/Users/linjingxian/Code/drone-emission`
- Remote host: `hku-sto`
- Remote repo: `/home/jingxian/code/drone-emission`
- Public command: `python -m drone_emission render`

## Required workflow

1. Check local Git status. Commit and push source before remote execution.
   Preserve unrelated local files.
2. When sub-agents are available, use one `gpt-5.4` validation worker. Tell it
   not to edit or revert source, not to run local Python, and to preserve all
   unrelated local and remote changes.
3. Open one persistent interactive SSH TTY. Reuse it for remote status, pull,
   checks, rendering, and verification. Never reconnect for individual
   commands, clean untracked files, or launch duplicate jobs. Stop after three
   connection-loss failures.
4. Inspect remote Git state, pull the pushed commit, and record path/size/mtime
   snapshots for official data, city results, figures, and paper artifacts.
5. Use `mktemp -d` for isolated figures, cache, and work roots. Export the three
   corresponding `DRONE_EMISSION_*_ROOT` variables. Reuse the existing data root
   and Chengdu results; do not recompute Chengdu or run the city pipeline.
6. Run package import, CLI, cohort/manifest/registry equality,
   dependency-graph, and retained fixture tests. Restore retained nationwide
   analyses only from the explicit audited reuse directory; never search
   legacy roots. Render only through `python -m drone_emission --env remote
   render`. Never invoke `analyze` implicitly from render.
7. For a full test, require exactly 39 manifest subfigure PDFs, three direct
   staging PDFs, and twelve source reports. Check filenames, `%PDF-` headers,
   page counts/sizes, report inputs, and requested numeric invariants. No
   undeclared figure may appear in isolated output.
8. Run `promote --dry-run` with the isolated staging root. Compare official
   snapshots and confirm no path size/mtime changed.
9. Exit the same SSH session only after verification. Fetch requested artifacts
   with `scp` into a new `figures/review/<batch>/` directory. Never overwrite an
   earlier batch. Retry connection-loss fetches up to three times, using a
   temporary destination on retries.
10. Report commit, commands, isolated roots, artifact counts, fetched paths and
    sizes, invariants, warnings, and final Git status.

Do not commit generated figures unless explicitly requested.
