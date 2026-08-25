# Archive manifest

Archived source files retain their former relative path below `archive/code/`.
Retired shell wrappers remain below `archive/scripts/`, and historical generated
outputs stay below `archive/figures/`. Nothing in `archive/` is imported by
active code.

| Former path/category | Archived because | Active replacement | Restore path/method |
|---|---|---|---|
| `src/drone_emission/visualization/analysis.py` | Pre-split giant module mixed formal requests with dormant correlation, regression, and candidate figure helpers | Thematic renderers plus `src/drone_emission/paper_workflow.py` and `src/drone_emission/visualization/formal_registry.py` | Copy `archive/code/src/drone_emission/visualization/analysis.py` back to the same relative path, then re-register requests deliberately if any output becomes formal |
| `src/drone_emission/visualization/paper.py` | Legacy plotting scheduler bypassed the locked manifest and bundled candidate render flows | `python -m drone_emission render` via `paper_workflow.py` | Copy `archive/code/src/drone_emission/visualization/paper.py` back only for historical inspection; do not recreate a compatibility facade |
| `src/drone_emission/visualization/emission.py` | Pre-split figure logic mixed formal lifecycle panels with retired candidate plots | `config_city_inputs.py`, `city_gef_detour.py`, `contribution_renderers.py`, and related thematic modules | Copy `archive/code/src/drone_emission/visualization/emission.py` back to its former path if historical code review is needed |
| `src/drone_emission/visualization/{sensitivity,concepts,calibration}.py` | Mixed retained helpers with OAT, Sobol, Shapley, MC, typology, driver, and other non-formal plots | `threshold_resilience.py`, `route_concepts.py`, `observed_distance.py`, and `lifecycle_panels.py` | Restore the archived file to its original path and manually separate any desired helper before reuse |
| `src/drone_emission/analysis/{correlation,cross_regions,demand_distance,report}.py` | Dormant analysis/report entry points depended on candidate plotting or superseded report flows | Explicit `aggregate` and `analyze` stages; active city-attribution helpers remain in `analysis/` | Copy the archived file from `archive/code/src/drone_emission/analysis/` back to the original relative path |
| `src/drone_emission/analysis/kappa_vrp.py` and its runner/test | Kappa VRP experiment retired from the locked evidence chain | Locked bundling parameter and named-scenario analysis | Restore the archived module/test pair only if reviving the experiment intentionally |
| `src/drone_emission/sensitivity/{oat,shapley,sobol_mc,baseline_retention,adjusted_resistance}.py` | Candidate sensitivity workflows and threshold variants are no longer formal outputs | `named_scenarios.py` and `threshold_resilience.py` | Copy the archived module from `archive/code/src/drone_emission/sensitivity/` back to its original path |
| `src/drone_emission/tools/{plot_distance_threshold_analysis,plot_drone_spatial_concentration,plot_threshold_resilience,paper_figure_reports,run_sensitivity_all,summarize_sensitivity,run_threshold_baseline_retention,run_threshold_adjusted_resistance,run_kappa_vrp_experiment}.py` | Standalone plotting/reporting CLIs bypassed the locked manifest or targeted retired candidates | `python -m drone_emission render`, `aggregate`, and `analyze` | Recover the exact pre-cleanup tool from `archive/code/src/drone_emission/tools/` or commit `a520754` |
| matching candidate visualization tests under `archive/code/tests/` | They asserted retired outputs such as 5 km scatter, Kappa VRP, threshold variants, and candidate sensitivity plots | `tests/test_reproduction_contract.py` and retained formal-analysis tests | Copy the archived test back to its former relative path only if that candidate workflow is intentionally restored |
| `src/drone_emission/calibration/{cali_elm,cali_mean_distance}.py` | The histogram fit and separate alpha sweep did not estimate the conditional building-choice model jointly | `calibration/gravity_calibration.py` and `calibration/gravity_model.py` | Copy either file from `archive/code/calibration/` only for historical inspection; active code must not import it |
| `src/drone_emission/equity/` | Outside the active manuscript evidence chain | None | Copy the archived directory back to `src/drone_emission/equity/` to reactivate |
| `src/drone_emission/profit/` | Outside the active manuscript evidence chain | None in the active render workflow | Copy the archived directory back to `src/drone_emission/profit/` to reactivate |
| retired shell wrappers in `archive/scripts/` | Multiple competing entry points for aggregate, sensitivity, and maintenance runs | `python -m drone_emission ...` and `./entrypoint.sh --jobs ...` | Move the desired script back to the repository root or `scripts/` and update its targets deliberately |
| unrelated local generated figures in `archive/figures/` | Runtime history, not locked manuscript inputs | `figures/paper/`, `figures/paper/subfigure/`, and `figures/paper/report/` | Move the archived artifact out of `archive/figures/` when manual comparison is needed |

If an archived plot is reactivated as a formal figure, restore the source file
to its former path, then deliberately add a manifest request and exactly one
`REQUEST_RENDERERS` entry. Do not restore a generic compatibility facade. To
restore generated figures, move the desired artifact from `archive/figures/`;
these files remain ignored and are not part of commits.
