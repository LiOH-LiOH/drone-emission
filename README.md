# Drone-delivery emission paper reproduction

This repository reproduces the city experiments, nationwide analysis, and the
12 locked figures used by the active manuscript. The public workflow is a
standard Python package invoked from the repository root.

## Setup

```bash
bash scripts/setup.sh
source .venv/bin/activate
```

## Reproduce

```bash
python -m drone_emission inputs
python -m drone_emission cities
python -m drone_emission aggregate
python -m drone_emission analyze
python -m drone_emission render
python -m drone_emission status
```

`python -m drone_emission reproduce` runs those stages in order. Existing city
and nationwide results are restored by default; pass `--force` only when a
recalculation is intentional. Calibration is locked in `experiment.toml` and
`src/drone_emission/calibration/alpha_calibration_5km.csv`; use
`python -m drone_emission calibrate` only when the Beijing raw OD files exist.

The root locations are configurable without editing source code:

- `DRONE_EMISSION_DATA_ROOT`
- `DRONE_EMISSION_FIGURES_ROOT`
- `DRONE_EMISSION_CACHE_ROOT`
- `DRONE_EMISSION_WORK_ROOT`
- `DRONE_EMISSION_GBA_ROOT`
- `DRONE_EMISSION_BEIJING_OD_ROOT`
- `DRONE_EMISSION_SALES_PATH`
- `DRONE_EMISSION_GEF_PATH`

The existing city layout and nationwide filenames remain unchanged under the
configured data root. Rendering writes subfigures to
`figures/paper/subfigure/` and complete direct figures to `figures/paper/`.
Nine composites are assembled manually from the declared Inkscape sources.
Only `python -m drone_emission promote --yes` can copy the complete set into
`docs/paper/figures/`; use `--dry-run` to validate without writing.

## Old-to-new command map

| Retired command | Replacement |
|---|---|
| `python main.py --region CITY` | `python -m drone_emission cities --region CITY` |
| `entrypoint.sh ...` analysis logic | `python -m drone_emission cities` |
| `postprocess.sh` | `python -m drone_emission aggregate` then `analyze` |
| `python -m visualization.paper` | `python -m drone_emission render` |
| sensitivity and maintenance shells | retained named scenarios or archived code |

`entrypoint.sh` remains only as a remote tmux/systemd parallel wrapper; it does
not initiate SSH. See [docs/architecture.md](docs/architecture.md) and
[archive/MANIFEST.md](archive/MANIFEST.md) for boundaries and recovery details.
