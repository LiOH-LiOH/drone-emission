"""Portable plot inputs and named preparation results, never Figure pickles.

``prepared_plot`` records the public drawing inputs. ``prepared_data`` marks
expensive or external-data boundaries inside a drawing function. During replay
those boundaries return their saved numerical result without executing source
code. The ordered, named records are tied to a source/schema/cohort signature.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from functools import wraps
import hashlib
import importlib
import inspect
import json
from pathlib import Path
import sys
from dataclasses import asdict, is_dataclass

import numpy as np
import pandas as pd

from .palette import Colour, color

SCHEMA_VERSION = 1
_SESSION = ContextVar("prepared_plot_session", default=None)
PLOT_FUNCTIONS = {}


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def drawing_code_hash() -> str:
    root = Path(__file__).parent
    files = sorted(root.glob("*.py")) + sorted((root / "layouts").glob("*.json"))
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


class SourceOnly:
    """A deliberately absent large source, only usable inside preparation."""

    def __init__(self, name):
        self.name = name

    def __getattr__(self, name):
        raise RuntimeError(
            f"Replay attempted to use raw source {self.name}.{name}; "
            "move this operation into a prepared_data boundary"
        )


class Codec:
    def __init__(self, root: Path, roots: dict):
        self.root = root
        self.roots = {name: Path(value).resolve() for name, value in roots.items()}
        self.assets = root / "assets"

    def text_asset(self, text, suffix):
        self.assets.mkdir(parents=True, exist_ok=True)
        name = hashlib.sha256(text.encode()).hexdigest() + suffix
        path = self.assets / name
        if not path.exists():
            path.write_text(text)
        return f"assets/{name}"

    def encode(self, value):
        if isinstance(value, Colour):
            return {"type": "colour", "key": value.key}
        if isinstance(value, SourceOnly):
            return {"type": "source_only", "name": value.name}
        if is_dataclass(value) and type(value).__name__ == "WeightedQuantileResult":
            return {"type": "weighted_quantile", "fields": self.encode(asdict(value))}
        if value is None or isinstance(value, (bool, int)):
            return value
        if isinstance(value, (float, np.floating)):
            return float(value) if np.isfinite(value) else {"type": "float", "value": str(value)}
        if isinstance(value, np.integer):
            return int(value)
        if isinstance(value, np.bool_):
            return bool(value)
        if isinstance(value, Path):
            return {"type": "path", "value": self.encode(str(value))}
        if isinstance(value, str):
            candidate = Path(value).resolve() if value.startswith("/") else None
            for name, root in sorted(self.roots.items(), key=lambda item: len(item[1].parts), reverse=True):
                if candidate is not None and (candidate == root or root in candidate.parents):
                    return {"type": "root_path", "root": name, "relative": str(candidate.relative_to(root))}
            return value
        if isinstance(value, np.ndarray):
            if value.dtype.hasobject:
                return {"type": "object_array", "shape": list(value.shape), "items": self.encode(value.tolist())}
            self.assets.mkdir(parents=True, exist_ok=True)
            key = hashlib.sha256(value.dtype.str.encode() + str(value.shape).encode() + value.tobytes()).hexdigest()
            path = self.assets / f"{key}.npz"
            if not path.exists():
                np.savez_compressed(path, values=value)
            return {"type": "array", "file": f"assets/{path.name}"}
        if isinstance(value, pd.DataFrame):
            geometry = getattr(value, "geometry", None)
            is_geo = geometry is not None and hasattr(value, "crs")
            columns = list(value.columns)
            result = {
                "type": "frame", "columns": self.encode(columns),
                "index": self.encode(list(value.index)),
                "index_name": self.encode(value.index.name),
                "data": [self.encode(value[column].to_numpy()) for column in columns],
                "dtypes": [str(value[column].dtype) for column in columns],
            }
            if is_geo:
                result.update(geometry=geometry.name, crs=str(value.crs))
            readable = pd.DataFrame(value.copy())
            for column in readable.columns:
                if readable[column].dtype == object or (is_geo and column == geometry.name):
                    readable[column] = readable[column].map(
                        lambda item: "@" + item.key if isinstance(item, Colour)
                        else item.wkb_hex if hasattr(item, "wkb_hex") else item
                    )
            result["csv"] = self.text_asset(readable.to_csv(index=True), ".csv")
            return result
        if isinstance(value, pd.Series):
            return {"type": "series", "frame": self.encode(value.to_frame(name="value")), "name": self.encode(value.name)}
        if isinstance(value, dict):
            if value.get("type") in {"FeatureCollection", "Feature", "Polygon", "MultiPolygon", "LineString", "MultiLineString"}:
                return {"type": "geojson", "file": self.text_asset(json.dumps(value, ensure_ascii=False, separators=(",", ":")), ".geojson")}
            return {"type": "dict", "items": [[self.encode(k), self.encode(v)] for k, v in value.items()]}
        if isinstance(value, (tuple, list, set)):
            return {"type": type(value).__name__, "items": [self.encode(v) for v in value]}
        if hasattr(value, "wkb_hex"):
            return {"type": "geometry", "wkb": value.wkb_hex}
        if value is pd.NA or value is pd.NaT:
            return {"type": "missing"}
        if isinstance(value, pd.Timestamp):
            return {"type": "timestamp", "value": value.isoformat()}
        raise TypeError(f"Non-portable prepared value: {type(value).__name__}")

    def decode(self, value):
        if not isinstance(value, dict):
            return value
        kind = value["type"]
        if kind == "colour":
            return color(value["key"])
        if kind == "source_only":
            return SourceOnly(value["name"])
        if kind == "weighted_quantile":
            from drone_emission.analysis.weighted_quantile import WeightedQuantileResult
            return WeightedQuantileResult(**self.decode(value["fields"]))
        if kind == "float":
            return float(value["value"])
        if kind == "path":
            return Path(self.decode(value["value"]))
        if kind == "root_path":
            root = self.roots[value["root"]]
            path = (root / value["relative"]).resolve()
            if path != root and root not in path.parents:
                raise ValueError("Prepared path escapes its declared root")
            return str(path)
        if kind == "array":
            path = (self.root / value["file"]).resolve()
            if self.root.resolve() not in path.parents:
                raise ValueError("Prepared array escapes bundle")
            with np.load(path, allow_pickle=False) as archive:
                return archive["values"]
        if kind == "geojson":
            path = (self.root / value["file"]).resolve()
            if self.root.resolve() not in path.parents:
                raise ValueError("Prepared geometry escapes bundle")
            return json.loads(path.read_text())
        if kind == "object_array":
            return np.array(self.decode(value["items"]), dtype=object).reshape(value["shape"])
        if kind == "frame":
            columns = self.decode(value["columns"])
            result = pd.DataFrame({k: self.decode(v) for k, v in zip(columns, value["data"])}, index=self.decode(value["index"]))
            result.index.name = self.decode(value["index_name"])
            for column, dtype in zip(columns, value["dtypes"]):
                if dtype not in {"geometry", "object"}:
                    result[column] = result[column].astype(dtype)
            if "geometry" in value:
                import geopandas as gpd
                result = gpd.GeoDataFrame(result, geometry=value["geometry"], crs=value["crs"])
            return result
        if kind == "series":
            result = self.decode(value["frame"])["value"]
            result.name = self.decode(value["name"])
            return result
        if kind == "dict":
            return {self.decode(k): self.decode(v) for k, v in value["items"]}
        if kind in {"list", "tuple", "set"}:
            items = [self.decode(v) for v in value["items"]]
            return {"list": list, "tuple": tuple, "set": set}[kind](items)
        if kind == "geometry":
            from shapely import wkb
            return wkb.loads(value["wkb"], hex=True)
        if kind == "missing":
            return pd.NA
        if kind == "timestamp":
            return pd.Timestamp(value["value"])
        raise ValueError(f"Unknown prepared value type: {kind}")


@dataclass
class Session:
    codec: Codec
    mode: str
    entries: list = field(default_factory=list)
    current: dict | None = None
    data_position: int = 0
    data_depth: int = 0


def prepared_plot(*, omit=()):
    """Register a drawing entry and its source-only argument boundaries."""
    def decorate(function):
        name = f"{function.__module__}:{function.__name__}"
        signature = inspect.signature(function)

        @wraps(function)
        def wrapped(*args, **kwargs):
            session = _SESSION.get()
            if session is None or session.current is not None:
                return function(*args, **kwargs)
            bound = signature.bind(*args, **kwargs)
            saved = dict(bound.arguments)
            if isinstance(saved.get("exp_config"), dict):
                saved["exp_config"] = {
                    key: value for key, value in saved["exp_config"].items()
                    if key not in {"palette_path", "palette_sha256", "prepare_only", "from_bundle", "bundle_dir"}
                }
            for key in omit:
                if key in saved and saved[key] is not None:
                    saved[key] = SourceOnly(key)
            entry = {"function": name, "arguments": session.codec.encode(saved), "data": []}
            session.entries.append(entry)
            session.current = entry
            try:
                return function(*args, **kwargs)
            finally:
                session.current = None

        PLOT_FUNCTIONS[name] = wrapped
        return wrapped
    return decorate


def prepared_data(function):
    """Persist a colour-independent computation at a named data boundary."""
    name = f"{function.__module__}:{function.__qualname__}"

    @wraps(function)
    def wrapped(*args, **kwargs):
        session = _SESSION.get()
        if session is None or session.current is None or session.data_depth:
            return function(*args, **kwargs)
        if session.mode == "replay":
            records = session.current["data"]
            position = session.data_position
            if position >= len(records) or records[position]["function"] != name:
                raise RuntimeError(f"Prepared data call order mismatch at {name}; prepare this bundle again")
            session.data_position += 1
            return session.codec.decode(records[position]["value"])
        session.data_depth += 1
        try:
            value = function(*args, **kwargs)
        finally:
            session.data_depth -= 1
        session.current["data"].append({"function": name, "value": session.codec.encode(value)})
        return value
    return wrapped


@contextmanager
def capture(root: Path, roots: dict):
    session = Session(Codec(root, roots), "capture")
    token = _SESSION.set(session)
    try:
        yield session
    finally:
        _SESSION.reset(token)


def replay(root: Path, document: dict, roots: dict):
    session = Session(Codec(root, roots), "replay")
    token = _SESSION.set(session)
    denied = []
    guard_active = True
    allowed = root.resolve()
    blocked_roots = [Path(roots["data"]).resolve(), Path(roots["work"]).resolve(),
                     Path(roots["project"]).resolve() / "data"]

    def audit(event, args):
        if not guard_active:
            return
        if event == "socket.connect":
            denied.append("network connection")
            raise RuntimeError("Prepared replay may not access the network")
        if event == "open" and isinstance(args[0], (str, bytes)):
            mode = args[1]
            if isinstance(mode, str) and any(flag in mode for flag in "wax+"):
                return
            path = Path(args[0]).resolve()
            if path == allowed or allowed in path.parents:
                return
            if any(path == base or base in path.parents for base in blocked_roots):
                denied.append(str(path))
                raise RuntimeError(f"Prepared replay may not read original analysis data: {path}")

    sys.addaudithook(audit)
    try:
        for entry in document["entries"]:
            module, _ = entry["function"].split(":")
            if not module.startswith("drone_emission.visualization."):
                raise ValueError("Bundle entry is outside the formal visualization package")
            importlib.import_module(module)
            if entry["function"] not in PLOT_FUNCTIONS:
                raise ValueError(f"Unregistered prepared plot: {entry['function']}")
            session.current = entry
            session.data_position = 0
            PLOT_FUNCTIONS[entry["function"]](**session.codec.decode(entry["arguments"]))
            if session.data_position != len(entry["data"]):
                raise RuntimeError("Unused prepared data records; drawing code and bundle disagree")
    finally:
        guard_active = False
        _SESSION.reset(token)
    if denied:
        raise RuntimeError(f"Prepared replay attempted forbidden input access: {denied}")


def write_document(path: Path, value: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def validate_bundle(root: Path, request: dict, cohort_hash: str) -> dict:
    path = root / request["id"] / "prepared.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing prepared input {path}; run render --prepare-only first")
    document = json.loads(path.read_text())
    payload_hash = document.pop("payload_sha256", None)
    actual_hash = hashlib.sha256(json.dumps(document, ensure_ascii=False, sort_keys=True, allow_nan=False).encode()).hexdigest()
    if payload_hash != actual_hash:
        raise ValueError(f"Corrupt prepared document: {path}")
    expected = {
        "schema_version": SCHEMA_VERSION,
        "request_id": request["id"],
        "request_version": request["cache_version"],
        "cohort_sha256": cohort_hash,
        "drawing_code_sha256": drawing_code_hash(),
        "outputs": request["outputs"],
    }
    for key, value in expected.items():
        if document.get(key) != value:
            raise ValueError(f"Prepared input {request['id']} has incompatible {key}; prepare again")
    for relative, digest in document["files"].items():
        asset = (path.parent / relative).resolve()
        if path.parent.resolve() not in asset.parents:
            raise ValueError("Prepared asset escapes request directory")
        if not asset.is_file() or file_hash(asset) != digest:
            raise ValueError(f"Missing or corrupt prepared asset: {asset}")
    return document


@prepared_data
def local_road_edges(graph, boundaries, **kwargs):
    return graph.get_local_edges(boundaries, **kwargs)


@prepared_data
def shortest_path(router, origin, destination):
    return router.find_shortest_path(origin, destination)


@prepared_data
def evaluate_kde(values, points, **kwargs):
    from scipy.stats import gaussian_kde
    return gaussian_kde(values, **kwargs)(points)


@prepared_data
def fit_polynomial(x, y, deg=1):
    return np.polyfit(x, y, deg)


@prepared_data
def histogram(values, **kwargs):
    return np.histogram(values, **kwargs)


@prepared_data
def box_statistics(values):
    from matplotlib.cbook import boxplot_stats
    return boxplot_stats(values)


@prepared_data
def linear_fit_interval(x, y, lower, upper):
    from scipy import stats
    slope, intercept, _, _, _ = stats.linregress(x, y)
    x_line = np.linspace(lower, upper, 100)
    y_line = slope * x_line + intercept
    n = len(x)
    se = np.sqrt(np.sum((y - (slope * x + intercept)) ** 2) / (n - 2))
    ci = stats.t.ppf(0.975, n - 2) * se * np.sqrt(
        1.0 / n + (x_line - np.mean(x)) ** 2 / np.sum((x - np.mean(x)) ** 2)
    )
    return x_line, y_line, ci


@prepared_data
def read_plot_csv(path, **kwargs):
    return pd.read_csv(path, **kwargs)


@prepared_data
def source_file_metadata(path):
    stat = Path(path).stat()
    return {"size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}


@prepared_data
def kde_with_factor(values, points, **kwargs):
    from scipy.stats import gaussian_kde
    estimator = gaussian_kde(values, **kwargs)
    return estimator(points), float(estimator.factor)


@prepared_data
def _basemap_image(bounds, cache_dir, crs):
    # Reuse the established retrieval/warp to preserve precisely the same map.
    import matplotlib.pyplot as plt
    from drone_emission.calibration.coordinate_restore import _add_basemap
    figure, axis = plt.subplots()
    try:
        axis.set_xlim(bounds[:2])
        axis.set_ylim(bounds[2:])
        _add_basemap(axis, Path(cache_dir), crs=crs)
        if len(axis.images) != 1:
            raise RuntimeError("Expected one prepared basemap image")
        artist = axis.images[0]
        return {"pixels": np.asarray(artist.get_array()),
                "extent": tuple(artist.get_extent()),
                "origin": artist.origin, "interpolation": artist.get_interpolation()}
    finally:
        plt.close(figure)


def add_prepared_basemap(axis, cache_dir, crs):
    bounds = tuple(axis.get_xlim()) + tuple(axis.get_ylim())
    image = _basemap_image(bounds, cache_dir, crs)
    axis.imshow(image["pixels"], extent=image["extent"], origin=image["origin"],
                interpolation=image["interpolation"], zorder=0)
    axis.set_xlim(bounds[:2])
    axis.set_ylim(bounds[2:])
