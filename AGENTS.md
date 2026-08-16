# Repository Guidelines

## Project Structure & Module Organization
Source code lives in `src/drone_emission/`. `main.py` is the pipeline entry point, while stage modules are split into `preprocessing/`, `buildings/`, `demand/`, `roadnet/`, `emission/`, `equity/`, and `profit/`. Shared helpers belong in `utils/`; standalone scripts and CLIs belong in `tools/`; plotting code belongs in `visualization/`; broader analysis and calibration logic live in `analysis/` and `calibration/`. Top-level shell scripts (`entrypoint.sh`, `run_sensitivity.sh`, `run_network_metrics.sh`, `postprocess.sh`) run the pipeline or its stages; supporting utilities (setup, smoke test, log summary, surgical reruns, data sync) live under `scripts/`. Generated runtime outputs go to ignored directories such as `data/`, `figures/`, `cache/`, and `logs/`. Documentation assets are under `docs/`.

## Build, Test, and Development Commands
- `bash scripts/setup.sh`: create `.venv`, copy `regions.json` from the template when missing, and install `src/drone_emission/requirements.txt`.
- `source .venv/bin/activate`: activate the local environment before running Python commands.
- `cd src/drone_emission && python main.py --region nanshan --env local`: run one region locally.
- `bash scripts/test_run.sh [region] [env]`: smoke-test the full single-region pipeline, defaulting to `changsha remote`.
- `./entrypoint.sh 4 --all`: run the multi-region pipeline wrapper with four workers.
- `./run_sensitivity.sh`: run the sensitivity stage across regions (after the emission stage).
- `./postprocess.sh local`: consolidate per-region stats and run cross-region analysis.
- `bash scripts/log_summary.sh [logs/file.log]`: summarize a pipeline log.

## Coding Style & Naming Conventions
Use Python 3.10+ and follow PEP 8 conventions: four-space indentation, `snake_case` for modules, functions, and variables, and `PascalCase` for classes. Keep stage APIs small and explicit; prefer passing `region_info` and `exp_config` rather than reading globals in new code. Use `pathlib.Path` for project-root paths when practical, and keep comments focused on non-obvious pipeline behavior.

## Testing Guidelines
There is no formal pytest suite in the current tree. Validate changes with the narrowest runnable workflow: use `bash scripts/test_run.sh <region> local` for end-to-end changes, or run the affected module/tool directly from `src/drone_emission`. For data-fetching work, use existing cached files when possible with `python main.py --region <region> --env local --no-fetch`. Record the region, environment, and command used in PR notes.

Agent preference: do not run Python code, Python modules, or test commands in this project unless the user explicitly asks for that validation. This includes local figure rendering commands such as `python -m visualization.paper`, ad hoc plotting scripts, and any command that imports project Python code to generate or inspect outputs. When preparing to run Python for checks or figure generation, ask first or skip and report that validation/rendering was not run.

## Paper Citation Quality
For `docs/paper/` work, use strong academic sources by default: Nature-family journals, Transportation Research series journals, high-quality field journals, official data sources, and mandatory method/data references. Avoid weak or low-quality venues such as MDPI journals, generic sustainability journals, proceedings-like papers, market reports, and weak reviews for scholarly claims. If a weaker source is the only available fit for a necessary claim, explicitly warn the user before relying on it and record the reason in the paper citation notes.

## Commit & Pull Request Guidelines
Recent commits use short imperative summaries such as `Fix circuity issue` and `Update network features`; keep the first line concise and action-oriented. Pull requests should describe the affected pipeline stage, list validation commands, note data/config assumptions, and include figures or screenshots when visualization output changes. Link related issues or experiment notes when available.

## Security & Configuration Tips
Do not commit generated data, logs, caches, figures, virtual environments, or local `regions.json`. Keep experiment parameters in `experiment.toml`, and treat region stats under `data/{level}/{region}/region_stats.json` as runtime state.

## Persistent Remote Sessions
For all work on `hku-sto`, never run a sequence of separate commands such as `ssh hku-sto '...'` that reconnect for each step. Open one background interactive SSH TTY, keep it alive, and send all remote status, pull, test, render, and verification commands through that same session. Poll and reuse the existing process until the workflow is complete; do not start duplicate SSH sessions or duplicate render jobs. File transfer with `scp` may use a separate connection after remote verification, but remote command execution must stay in the persistent TTY.
