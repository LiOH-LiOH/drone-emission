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


@dataclass(frozen=True)
class Palette:
    values: MappingProxyType
    colormaps: MappingProxyType
    digest: str

    def colour(self, key: str) -> Colour:
        seen = set()
        value = self.values[key]
        while value.startswith("@"):
            if value in seen:
                raise ValueError(f"Circular palette alias: {key}")
            seen.add(value)
            value = self.values[value[1:]]
        return Colour(value, key)


def load_palette(path: str | Path | None = None) -> Palette:
    document = tomllib.loads(Path(path or DEFAULT_PALETTE).read_text())
    values = document["colors"]
    if not all(isinstance(v, str) for v in values.values()):
        raise ValueError("Palette colors must be strings or @key aliases")
    palette = Palette(
        MappingProxyType(values),
        MappingProxyType(document.get("colormaps", {})),
        sha256(json.dumps(document, sort_keys=True).encode()).hexdigest(),
    )
    from matplotlib.colors import is_color_like

    for key in values:
        if not is_color_like(palette.colour(key)):
            raise ValueError(f"Invalid palette colour: {key}")
    return palette


_CURRENT: ContextVar[Palette | None] = ContextVar("paper_palette", default=None)


def configure_palette(path: str | Path | None = None) -> Palette:
    palette = load_palette(path)
    _CURRENT.set(palette)
    return palette


def color(key: str) -> Colour:
    palette = _CURRENT.get()
    if palette is None:
        palette = configure_palette()
    return palette.colour(key)


def colors(*keys: str) -> list[Colour]:
    return [color(key) for key in keys]


def sequential_cmap(kind: str = "size"):
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list(
        f"paper_{kind}",
        colors(f"scale.{kind}.low", f"scale.{kind}.mid", f"scale.{kind}.high"),
    )


def colormap(key: str, lut: int | None = None):
    palette = _CURRENT.get() or configure_palette()
    name = palette.colormaps[key]
    if name.startswith("@scale."):
        result = sequential_cmap(name.removeprefix("@scale."))
    else:
        from matplotlib import colormaps
        result = colormaps[name]
    return result.resampled(lut) if lut is not None else result
