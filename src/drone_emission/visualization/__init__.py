import os
from pathlib import Path


def _configure_matplotlib_cache():
    """Use project-local writable caches before importing matplotlib."""
    project_root = Path(__file__).resolve().parents[3]
    cache_root = project_root / "cache"
    matplotlib_cache = cache_root / "matplotlib"
    fontconfig_cache = cache_root / "fontconfig"
    try:
        matplotlib_cache.mkdir(parents=True, exist_ok=True)
        fontconfig_cache.mkdir(parents=True, exist_ok=True)
    except OSError:
        return
    os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_cache))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_root))


_configure_matplotlib_cache()

import matplotlib.pyplot as plt

plt.rcParams["font.family"] = ["Arial", "DejaVu Sans", "sans-serif"]
plt.rcParams["mathtext.fontset"] = "custom"
plt.rcParams["mathtext.rm"] = "Arial"
plt.rcParams["mathtext.it"] = "Arial:italic"
plt.rcParams["mathtext.bf"] = "Arial:bold"
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
