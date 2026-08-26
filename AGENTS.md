# Repository Guidelines

## Project Structure & Module Organization
Source code lives in `src/drone_emission/`. The package CLI is the active entry point, while stage modules are split into `preprocessing/`, `buildings/`, `demand/`, `roadnet/`, `emission/`, `analysis/`, and `calibration/`. Shared helpers belong in `utils/`; standalone support tools belong in `tools/`; formal plotting code belongs in `visualization/`; the manifest scheduler is `paper_workflow.py`. Archived candidate code and retired plotting entry points live under `archive/code/` and are not imported by active code. Generated runtime outputs go to ignored directories such as `data/`, `figures/`, `cache/`, and `logs/`. Documentation assets are under `docs/`.

## Build, Test, and Development Commands
- `bash scripts/setup.sh`: create `.venv`, copy `regions.json` from the template when missing, and install `src/drone_emission/requirements.txt`.
- `source .venv/bin/activate`: activate the local environment before running Python commands.
- `python -m drone_emission --env local cities --region nanshan`: run one city locally through the package CLI.
- `python -m drone_emission --env local aggregate`: consolidate compatible city results into national stats.
- `python -m drone_emission --env local cohort`: audit the tracked paper-city sample after aggregation.
- `python -m drone_emission --env local analyze`: build derived nationwide analysis inputs required by formal figures.
- `python -m drone_emission --env local render [--only FIGURE_ID]`: render only manifest-backed paper figures.
- `./entrypoint.sh --jobs 4 cities`: run the multi-city pipeline wrapper with four workers under the shared memory contract.
- `bash scripts/log_summary.sh [logs/file.log]`: summarize a pipeline log.

## Coding Style & Naming Conventions
Use Python 3.10+ and follow PEP 8 conventions: four-space indentation, `snake_case` for modules, functions, and variables, and `PascalCase` for classes. Keep stage APIs small and explicit; prefer passing `region_info` and `exp_config` rather than reading globals in new code. Use `pathlib.Path` for project-root paths when practical, and keep comments focused on non-obvious pipeline behavior.

## Testing Guidelines
There is no single default end-to-end test command for every change. Validate with the narrowest runnable workflow that matches the touched stage, and prefer static inspection when the user does not authorize Python execution. For pipeline work, use the package CLI such as `python -m drone_emission --env local cities --region <region> --no-fetch`, `aggregate`, `analyze`, or the manifest-backed `render`. Record the region, environment, and command used in PR notes when runtime validation is performed.

Agent preference: do not run Python code, Python modules, tests, or local figure rendering unless the user explicitly asks for that validation. Use only the manifest-backed `python -m drone_emission render` command for formal figures. When preparing to run Python for checks or figure generation, ask first or skip and report that validation/rendering was not run.

Paper analyses must use the tracked cohort contract. Do not derive the paper
sample directly from every city in `regions.json`, silently accept a changed
candidate set, or search legacy directories for missing analysis files. Use an
explicit audited `--reuse-from` directory when old derived data are part of a
remote regression test.

## Paper Citation Quality
For `docs/paper/` work, use strong academic sources by default: Nature-family journals, Transportation Research series journals, high-quality field journals, official data sources, and mandatory method/data references. Avoid weak or low-quality venues such as MDPI journals, generic sustainability journals, proceedings-like papers, market reports, and weak reviews for scholarly claims. If a weaker source is the only available fit for a necessary claim, explicitly warn the user before relying on it and record the reason in the paper citation notes.

## Commit & Pull Request Guidelines
Recent commits use short imperative summaries such as `Fix circuity issue` and `Update network features`; keep the first line concise and action-oriented. Pull requests should describe the affected pipeline stage, list validation commands, note data/config assumptions, and include figures or screenshots when visualization output changes. Link related issues or experiment notes when available.

## Security & Configuration Tips
Do not commit generated data, logs, caches, figures, virtual environments, or local `regions.json`. Keep experiment parameters in `experiment.toml`, and treat region stats under `data/{level}/{region}/region_stats.json` as runtime state.

## Persistent Remote Sessions
For all work on `hku-sto`, never run a sequence of separate commands such as `ssh hku-sto '...'` that reconnect for each step. Open one background interactive SSH TTY, keep it alive, and send all remote status, pull, test, render, and verification commands through that same session. Poll and reuse the existing process until the workflow is complete; do not start duplicate SSH sessions or duplicate render jobs. File transfer may use a separate connection after remote verification, but remote command execution must stay in the persistent TTY.

Prefer `rsync` over a separate connection for remote file transfers, with
`scp` only as a fallback. Never use `rsync --delete` for review artifacts. Save
retrieved figures to a new `figures/review/<purpose>_YYYYMMDD/` batch by default,
using the local `Asia/Shanghai` date and a short English kebab-case purpose. Do
not overwrite or mix with an existing batch; append `_02`, `_03`, and so on.
