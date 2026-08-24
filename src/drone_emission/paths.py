"""Resolve all mutable roots without machine-specific assumptions."""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def apply_path_overrides(config: dict, env: str | None = None) -> dict:
    """Return a config copy with environment-variable path overrides applied."""
    result = dict(config)
    user_name = os.environ.get("USER", "user")
    remote_root = Path("/mnt/minor_storage") / user_name / "drone-emission"
    default_data = remote_root / "data" if env == "remote" else PROJECT_ROOT / "data"
    default_figures = PROJECT_ROOT / "figures"
    default_cache = remote_root / "cache" if env == "remote" else PROJECT_ROOT / "cache"
    default_work = remote_root / "work" if env == "remote" else PROJECT_ROOT / "outputs" / "work"
    data_root = Path(os.environ.get("DRONE_EMISSION_DATA_ROOT", default_data))
    figures_root = Path(
        os.environ.get("DRONE_EMISSION_FIGURES_ROOT", default_figures)
    )
    cache_root = Path(
        os.environ.get(
            "DRONE_EMISSION_CACHE_ROOT", default_cache
        )
    )
    work_root = Path(
        os.environ.get("DRONE_EMISSION_WORK_ROOT", default_work)
    )
    result.update(
        data_dir=str(data_root),
        figures_dir=str(figures_root),
        cache_dir=str(cache_root),
        work_dir=str(work_root),
        sales_path=os.environ.get(
            "DRONE_EMISSION_SALES_PATH", str(data_root / "combined_data.csv")
        ),
        gef_path=os.environ.get(
            "DRONE_EMISSION_GEF_PATH", str(data_root / "gef_china.csv")
        ),
    )
    default_gba = remote_root / "gba_data" if env == "remote" else PROJECT_ROOT / "gba_data"
    result["gba_data_dir"] = os.environ.get("DRONE_EMISSION_GBA_ROOT", str(default_gba))
    return result
