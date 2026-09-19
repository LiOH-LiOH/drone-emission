"""Named, portable colours for the formal paper renderers."""

from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from types import MappingProxyType
import json

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib

DEFAULT_PALETTE = Path(__file__).with_name("palette.toml")


class Colour(str):
    """A matplotlib-compatible colour retaining its semantic bundle key."""

    def __new__(cls, value: str, key: str):
        instance = super().__new__(cls, value)
        instance.key = key
        return instance


class Opacity(float):
    def __new__(cls, value: float, key: str):
        instance = super().__new__(cls, value)
        instance.key = key
        return instance


@dataclass(frozen=True)
class Palette:
    values: MappingProxyType
    colormaps: MappingProxyType
    opacities: MappingProxyType
    digest: str
    ramps: MappingProxyType
    normalizations: MappingProxyType

    def colour(self, key: str) -> Colour:
        seen = set()
        value = self.values[key]
        while value.startswith("@"):
            if value in seen:
                raise ValueError(f"Circular palette alias: {key}")
            seen.add(value)
            value = self.values[value[1:]]
        return Colour(value, key)

    def signature_for(self, usage: dict | None) -> str:
        if not usage:
            return self.digest
        selected = set(usage["colors"])
        for key in usage["colormaps"]:
            name = self.colormaps[key]
            if name.startswith("@scale."):
                selected.update(f"{name[1:]}.{stop}" for stop in ("low", "mid", "high"))
        pending = list(selected)
        while pending:
            value = self.values[pending.pop()]
            if value.startswith("@") and value[1:] not in selected:
                selected.add(value[1:])
                pending.append(value[1:])
        payload = {
            "normalizations": dict(self.normalizations),
            "ramps": dict(self.ramps),
            "colors": {key: self.values[key] for key in sorted(selected)},
            "colormaps": {key: self.colormaps[key] for key in usage["colormaps"]},
            "alpha": {key: self.opacities[key] for key in usage.get("alpha", [])},
        }
        return sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def load_palette(path: str | Path | None = None) -> Palette:
    document = tomllib.loads(Path(path or DEFAULT_PALETTE).read_text())
    values = document["colors"]
    if not all(isinstance(v, str) for v in values.values()):
        raise ValueError("Palette colors must be strings or @key aliases")
    palette = Palette(
        MappingProxyType(values),
        MappingProxyType(document.get("colormaps", {})),
        MappingProxyType(document.get("alpha", {})),
        sha256(json.dumps(document, sort_keys=True).encode()).hexdigest(),
        MappingProxyType(document.get("ramps", {})),
        MappingProxyType(document.get("normalizations", {})),
    )
    from matplotlib.colors import is_color_like

    for key in values:
        if not is_color_like(palette.colour(key)):
            raise ValueError(f"Invalid palette colour: {key}")
    for key, value in palette.opacities.items():
        if not isinstance(value, (int, float)) or not 0 <= value <= 1:
            raise ValueError(f"Palette opacity must lie in [0, 1]: {key}")
    for key, ramp in palette.ramps.items():
        if len(ramp) < 2 or any(len(rgb) != 3 or any(
            not isinstance(v, (int, float)) or not 0 <= v <= 1 for v in rgb
        ) for rgb in ramp):
            raise ValueError(f"Invalid RGB lookup table: {key}")
    for key, limits in palette.normalizations.items():
        if len(limits) != 3 or not limits[0] < limits[1] < limits[2]:
            raise ValueError(f"Normalization must have ordered low, centre, high: {key}")
    return palette


_CURRENT: ContextVar[Palette | None] = ContextVar("paper_palette", default=None)
_USED_COLORS: ContextVar[set | None] = ContextVar("paper_color_usage", default=None)
_USED_CMAPS: ContextVar[set | None] = ContextVar("paper_colormap_usage", default=None)
_USED_ALPHAS: ContextVar[set | None] = ContextVar("paper_opacity_usage", default=None)


def configure_palette(path: str | Path | None = None) -> Palette:
    palette = load_palette(path)
    _CURRENT.set(palette)
    _USED_COLORS.set(set())
    _USED_CMAPS.set(set())
    _USED_ALPHAS.set(set())
    import matplotlib as mpl
    from cycler import cycler

    defaults = {
        "text.color": "ink.text", "axes.labelcolor": "ink.text",
        "xtick.color": "ink.text", "ytick.color": "ink.text",
        "axes.edgecolor": "ink.secondary", "grid.color": "ink.grid",
        "axes.facecolor": "ink.background", "figure.facecolor": "ink.background",
        "savefig.facecolor": "ink.background",
    }
    mpl.rcParams.update({key: color(value) for key, value in defaults.items()})
    mpl.rcParams["axes.prop_cycle"] = cycler(color=colors(*(f"group.{i}" for i in range(7))))
    return palette


def color(key: str) -> Colour:
    palette = _CURRENT.get()
    if palette is None:
        palette = configure_palette()
    _USED_COLORS.get().add(key)
    return palette.colour(key)


def colors(*keys: str) -> list[Colour]:
    return [color(key) for key in keys]


def sequential_cmap(kind: str = "size"):
    from matplotlib.colors import LinearSegmentedColormap, ListedColormap

    palette = _CURRENT.get() or configure_palette()
    if kind in palette.ramps:
        return ListedColormap(palette.ramps[kind], name=f"paper_{kind}")

    return LinearSegmentedColormap.from_list(
        f"paper_{kind}",
        colors(f"scale.{kind}.low", f"scale.{kind}.mid", f"scale.{kind}.high"),
    )


def colormap(key: str, lut: int | None = None):
    palette = _CURRENT.get() or configure_palette()
    _USED_CMAPS.get().add(key)
    name = palette.colormaps[key]
    if name.startswith("@scale."):
        result = sequential_cmap(name.removeprefix("@scale."))
    else:
        from matplotlib import colormaps
        result = colormaps[name]
    return result.resampled(lut) if lut is not None else result


def gef_norm(sequential: bool = False):
    """Shared absolute GEF scale, including the model domain and fixed reference."""
    from matplotlib.colors import Normalize, TwoSlopeNorm

    palette = _CURRENT.get() or configure_palette()
    low, center, high = palette.normalizations["gef"]
    if sequential:
        return Normalize(vmin=low, vmax=high)
    return TwoSlopeNorm(vmin=low, vcenter=center, vmax=high)


def linearize_colorbar(colorbar):
    """Keep distances proportional to values, independently of colour allocation."""
    ticks = colorbar.get_ticks()
    if colorbar.orientation == "horizontal":
        colorbar.ax.set_xscale("linear")
    else:
        colorbar.ax.set_yscale("linear")
    colorbar.set_ticks(ticks)


def gef_fit_color(values):
    """Darken the group's median GEF colour using the registered charcoal blend."""
    import numpy as np
    from matplotlib.colors import to_rgb

    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return tuple(to_rgb(color("ink.text")))
    rgb = np.asarray(colormap("city_gef_support.gef")(gef_norm()(np.median(values)))[:3])
    amount = opacity("city_gef_support.fit_darkening")
    return tuple((1 - amount) * rgb + amount * np.asarray(to_rgb(color("ink.text"))))


def palette_usage() -> dict:
    return {"colors": sorted(_USED_COLORS.get() or []),
            "colormaps": sorted(_USED_CMAPS.get() or []),
            "alpha": sorted(_USED_ALPHAS.get() or [])}


def opacity(key: str) -> Opacity:
    palette = _CURRENT.get() or configure_palette()
    _USED_ALPHAS.get().add(key)
    return Opacity(palette.opacities[key], key)
