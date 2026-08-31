"""Read-only inputs and bounded output roots for formal figure rendering."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path


@dataclass(frozen=True)
class RenderContext:
    """Resolved roots shared by every manifest renderer."""

    data_root: Path
    figures_root: Path
    cache_root: Path
    work_root: Path
    experiment: dict

    @classmethod
    def from_config(cls, config: dict) -> "RenderContext":
        context = cls(
            data_root=Path(config["data_dir"]).resolve(),
            figures_root=Path(config["figures_dir"]).resolve(),
            cache_root=Path(config["cache_dir"]).resolve(),
            work_root=Path(config["work_dir"]).resolve(),
            experiment=dict(config),
        )
        analysis_hash = config.get("analysis_input_manifest_sha256")
        if analysis_hash:
            from drone_emission.paper_cohort import cohort_hash, locked_region_keys

            cohort_view = context.data_root / "paper_cohort_view.json"
            if not cohort_view.is_file():
                raise FileNotFoundError(
                    f"Missing cohort-filtered render view: {cohort_view}. "
                    "Run `python -m drone_emission analyze` first."
                )
            metadata = json.loads(cohort_view.read_text(encoding="utf-8"))
            if metadata.get("input_manifest_sha256") != analysis_hash:
                raise RuntimeError("render data view has the wrong input manifest")
            if metadata.get("cohort_sha256") != cohort_hash():
                raise RuntimeError(
                    "render data view belongs to a different paper cohort"
                )
            if int(metadata.get("city_count", -1)) != len(locked_region_keys()):
                raise RuntimeError("render data view has the wrong paper city count")
        for root in (context.figures_root, context.cache_root, context.work_root):
            root.mkdir(parents=True, exist_ok=True)
        matplotlib_cache = context.cache_root / "matplotlib"
        matplotlib_cache.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_cache))
        os.environ.setdefault("XDG_CACHE_HOME", str(context.cache_root))
        context.subfigure_dir.mkdir(parents=True, exist_ok=True)
        return context

    @property
    def paper_dir(self) -> Path:
        return self.figures_root / "paper"

    @property
    def subfigure_dir(self) -> Path:
        return self.paper_dir / "subfigure"

    def output(self, name: str) -> Path:
        """Return a PDF output path proven to stay below the figures root."""
        path = (self.subfigure_dir / name).resolve()
        if self.figures_root not in path.parents:
            raise ValueError(f"Renderer output escapes figures root: {path}")
        return path

    def require_file(self, relative: str, *, stage: str) -> Path:
        path = self.data_root / relative
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing render input: {path}. Run `python -m drone_emission "
                f"{stage}` first, or restore the declared input."
            )
        return path
