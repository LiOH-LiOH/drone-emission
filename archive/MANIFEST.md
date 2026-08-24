# Archive manifest

Archived files retain their former relative path below `archive/code/` or are
grouped below `archive/scripts/` and `archive/figures/`. Nothing here is imported
by active code.

| Former path/category | Reason | Active replacement |
|---|---|---|
| `src/drone_emission/equity/` | Outside the active manuscript evidence chain | None; restore the directory to reactivate |
| `src/drone_emission/profit/` | Outside the active manuscript evidence chain | Retained alternative ground-vehicle lifecycle model |
| `analysis/kappa_vrp.py` and runner/test | Kappa VRP experiment retired | Locked observed bundling parameter and named scenarios |
| sensitivity OAT, Sobol, Shapley | Candidate sensitivity workflows retired | Named scenarios and threshold resilience |
| baseline retention and adjusted resistance | Candidate threshold metrics retired | Distance-threshold and threshold-resilience analyses |
| matching visualization modules/tests | No locked figure dependency | Manifest-driven `render` |
| outer/maintenance shell scripts | Multiple competing entry points | `python -m drone_emission ...` |
| unrelated local generated figures | Runtime history, not manuscript inputs | `figures/paper/`, required subfigures/reports, and composites 5–9/11–14 |
| historical plot CLIs and emission sensitivity runners removed during package cleanup | They mixed computation with candidate plotting and bypassed the manifest | Recover the exact pre-refactor implementation from commit `a520754`; formal plots use `python -m drone_emission render` |

To restore code, move the archived file back to the same path relative to the
repository root and update the active package and manifest deliberately. To
restore generated figures, move the desired artifact from `archive/figures/`;
these files remain ignored and are not part of commits.
