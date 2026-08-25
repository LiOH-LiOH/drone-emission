"""Audit and restore the globally translated Beijing Eleme coordinates.

This is an independent calibration workflow.  It is intentionally not part of
the city pipeline or the paper-figure manifest.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from pyproj import Transformer
from scipy.spatial import cKDTree

from drone_emission.utils.coordinates import gcj02_to_wgs84, wgs84_to_gcj02


HISTORICAL_OFFSET = (-5.1192, 0.7810)
BEIJING_METRIC_CRS = "EPSG:32650"
_TO_METRIC = Transformer.from_crs("EPSG:4326", BEIJING_METRIC_CRS, always_xy=True)
_FROM_METRIC = Transformer.from_crs(BEIJING_METRIC_CRS, "EPSG:4326", always_xy=True)
METRIC_THRESHOLDS_M = (25.0, 50.0, 100.0, 200.0)
DEFAULT_SEARCH_LEVELS = (
    {"radius_degrees": 0.0200, "step_degrees": 0.0020},
    {"radius_degrees": 0.0020, "step_degrees": 0.0002},
    {"radius_degrees": 0.0002, "step_degrees": 0.00002},
)
EXACT_RERANK_CANDIDATES = 25
ORDER_COORDINATE_COLUMNS = (
    "pick_lng",
    "pick_lat",
    "deliver_lng",
    "deliver_lat",
)
POI_NAME_COLUMNS = ("name", "NAME", "名称", "poi_name", "POI_NAME")
POI_CATEGORY_COLUMNS = (
    "type",
    "TYPE",
    "类别",
    "category",
    "poi_category",
    "tag",
)


@dataclass(frozen=True)
class SearchResult:
    """Estimated translation and the complete coarse-to-fine search trace."""

    longitude_degrees: float
    latitude_degrees: float
    distances_m: np.ndarray
    nearest_indices: np.ndarray
    surface: pd.DataFrame


def _input_files(path: Path, suffixes: set[str] | None = None) -> list[Path]:
    if path.is_file():
        if path.suffix.lower() == ".shp":
            files = sorted(
                sibling
                for sibling in path.parent.glob(path.stem + ".*")
                if sibling.is_file()
            )
            return files
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(f"Input does not exist: {path}")
    allowed = suffixes or {".txt", ".csv", ".csv.gz"}
    files = sorted(
        item
        for item in path.rglob("*")
        if item.is_file()
        and any(str(item).lower().endswith(suffix) for suffix in allowed)
    )
    if not files:
        raise FileNotFoundError(f"No supported input files found below: {path}")
    return files


def fingerprint_files(files: Sequence[Path], root: Path | None = None) -> dict:
    """Return a stable SHA-256 fingerprint over names and file content hashes."""
    if not files:
        raise ValueError("Cannot fingerprint an empty input set")
    digest = hashlib.sha256()
    entries = []
    for path in sorted(Path(item) for item in files):
        file_hash = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                file_hash.update(chunk)
        relative = path.relative_to(root).as_posix() if root else path.name
        value = file_hash.hexdigest()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(value.encode("ascii"))
        digest.update(b"\n")
        entries.append({"path": relative, "sha256": value})
    return {"sha256": digest.hexdigest(), "file_count": len(files), "files": entries}


def load_orders(path: Path) -> tuple[pd.DataFrame, list[Path]]:
    """Load one or many order CSV/TXT files and retain their provenance."""
    path = Path(path)
    files = _input_files(path, {".txt", ".csv", ".csv.gz"})
    frames = []
    for file_path in files:
        frame = pd.read_csv(file_path)
        missing = [name for name in ORDER_COORDINATE_COLUMNS if name not in frame]
        if "shop_id" not in frame:
            missing.append("shop_id")
        if missing:
            raise ValueError(
                f"Order file {file_path} is missing required column(s): "
                + ", ".join(missing)
            )
        if "source_file" in frame:
            raise ValueError(
                f"Order file {file_path} already has reserved provenance column source_file"
            )
        frame = frame.copy()
        frame["source_file"] = (
            file_path.relative_to(path).as_posix() if path.is_dir() else file_path.name
        )
        frames.append(frame)
    orders = pd.concat(frames, ignore_index=True, sort=False)
    for column in ORDER_COORDINATE_COLUMNS:
        orders[column] = pd.to_numeric(orders[column], errors="coerce")
    finite = np.isfinite(orders[list(ORDER_COORDINATE_COLUMNS)].to_numpy()).all(axis=1)
    if not finite.all():
        count = int((~finite).sum())
        raise ValueError(f"Order inputs contain {count} row(s) with invalid coordinates")
    return orders, files


def unique_shops(orders: pd.DataFrame) -> pd.DataFrame:
    """Return one original pickup coordinate per shop without mutating orders."""
    shops = (
        orders.loc[:, ["shop_id", "pick_lng", "pick_lat"]]
        .drop_duplicates(subset="shop_id", keep="first")
        .reset_index(drop=True)
        .rename(
            columns={
                "pick_lng": "original_pick_lng",
                "pick_lat": "original_pick_lat",
            }
        )
    )
    if shops.empty:
        raise ValueError("Order inputs contain no shops")
    return shops


def _first_column(columns: Iterable[str], candidates: Sequence[str]) -> str | None:
    lookup = {str(column).casefold(): str(column) for column in columns}
    for candidate in candidates:
        if candidate.casefold() in lookup:
            return lookup[candidate.casefold()]
    return None


def load_poi(path: Path, datum: str) -> tuple[pd.DataFrame, list[Path], str | None]:
    """Load point POIs and normalize coordinates to both GCJ-02 and WGS84."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"POI input does not exist or is not a file: {path}")
    declared_crs = None
    if path.suffix.lower() in {".shp", ".gpkg", ".geojson"}:
        import geopandas as gpd

        source = gpd.read_file(path)
        if path.suffix.lower() == ".shp" and path.with_suffix(".prj").is_file():
            declared_crs = path.with_suffix(".prj").read_text(
                encoding="utf-8", errors="replace"
            ).strip()
        else:
            declared_crs = None if source.crs is None else source.crs.to_string()
        if source.empty or source.geometry.isna().all():
            raise ValueError(f"POI input contains no point geometry: {path}")
        if not source.geometry.geom_type.eq("Point").all():
            raise ValueError(f"POI input must contain only point geometry: {path}")
        if source.crs is not None and not source.crs.is_geographic:
            source = source.to_crs("EPSG:4326")
        frame = pd.DataFrame(source.drop(columns="geometry"))
        frame["_source_lng"] = source.geometry.x.to_numpy()
        frame["_source_lat"] = source.geometry.y.to_numpy()
        files = _input_files(path)
    else:
        frame = pd.read_csv(path)
        files = [path]
        lng_column = _first_column(
            frame.columns, ("lng", "lon", "longitude", "longitude_gcj02", "x")
        )
        lat_column = _first_column(
            frame.columns, ("lat", "latitude", "latitude_gcj02", "y")
        )
        if lng_column is None or lat_column is None:
            raise ValueError(
                f"POI CSV {path} needs longitude/lng and latitude/lat columns"
            )
        frame["_source_lng"] = pd.to_numeric(frame[lng_column], errors="coerce")
        frame["_source_lat"] = pd.to_numeric(frame[lat_column], errors="coerce")

    name_column = _first_column(frame.columns, POI_NAME_COLUMNS)
    if name_column is None:
        raise ValueError(
            f"POI input {path} is missing a name field; accepted names: "
            + ", ".join(POI_NAME_COLUMNS)
        )
    category_column = _first_column(frame.columns, POI_CATEGORY_COLUMNS)
    finite = np.isfinite(frame[["_source_lng", "_source_lat"]].to_numpy()).all(axis=1)
    frame = frame.loc[finite].reset_index(drop=True)
    if frame.empty:
        raise ValueError(f"POI input contains no finite coordinates: {path}")
    source_lng = frame["_source_lng"].to_numpy(float)
    source_lat = frame["_source_lat"].to_numpy(float)
    if datum == "gcj02":
        gcj_lng, gcj_lat = source_lng, source_lat
        wgs_lng, wgs_lat = _convert_pairs(gcj_lng, gcj_lat, gcj02_to_wgs84)
    elif datum == "wgs84":
        wgs_lng, wgs_lat = source_lng, source_lat
        gcj_lng, gcj_lat = _convert_pairs(wgs_lng, wgs_lat, wgs84_to_gcj02)
    else:
        raise ValueError(f"Unsupported POI datum: {datum}")
    normalized = pd.DataFrame(
        {
            "poi_name": frame[name_column].astype("string"),
            "poi_category": (
                frame[category_column].astype("string")
                if category_column
                else pd.Series(pd.NA, index=frame.index, dtype="string")
            ),
            "poi_lng_gcj02": gcj_lng,
            "poi_lat_gcj02": gcj_lat,
            "poi_lng_wgs84": wgs_lng,
            "poi_lat_wgs84": wgs_lat,
        }
    )
    return normalized, files, declared_crs


def _convert_pairs(lng: np.ndarray, lat: np.ndarray, converter) -> tuple[np.ndarray, np.ndarray]:
    converted = np.asarray(
        [converter(float(x), float(y)) for x, y in zip(lng, lat)], dtype=float
    )
    return converted[:, 0], converted[:, 1]


def _metric_xy(lng: np.ndarray, lat: np.ndarray) -> np.ndarray:
    x, y = _TO_METRIC.transform(lng, lat)
    return np.column_stack((x, y))


def _axis_values(center: float, radius: float, step: float) -> np.ndarray:
    steps = int(round(radius / step))
    return center + np.arange(-steps, steps + 1, dtype=float) * step


def estimate_global_translation(
    shops: pd.DataFrame,
    poi: pd.DataFrame,
    match_radius_m: float = 100.0,
    initial_offset: tuple[float, float] = HISTORICAL_OFFSET,
    search_levels: Sequence[dict] = DEFAULT_SEARCH_LEVELS,
) -> SearchResult:
    """Estimate one translation by lexicographic nearest-POI scoring.

    Candidate offsets first maximize the shop count within ``match_radius_m``;
    ties then minimize the mean nearest-neighbour distance over every shop.
    """
    if match_radius_m <= 0:
        raise ValueError("match_radius_m must be positive")
    shop_lng = shops["original_pick_lng"].to_numpy(float)
    shop_lat = shops["original_pick_lat"].to_numpy(float)
    poi_tree = cKDTree(
        _metric_xy(
            poi["poi_lng_wgs84"].to_numpy(float),
            poi["poi_lat_wgs84"].to_numpy(float),
        )
    )
    center_lng, center_lat = map(float, initial_offset)
    reference_lng, reference_lat = center_lng, center_lat
    reference_wgs_lng, reference_wgs_lat = _convert_pairs(
        shop_lng + reference_lng,
        shop_lat + reference_lat,
        gcj02_to_wgs84,
    )
    trace = []
    best_distances = None
    best_indices = None
    for level_number, level in enumerate(search_levels, start=1):
        radius = float(level["radius_degrees"])
        step = float(level["step_degrees"])
        if radius <= 0 or step <= 0 or step > radius:
            raise ValueError(f"Invalid search level: {level}")
        level_candidates = []
        for offset_lng in _axis_values(center_lng, radius, step):
            for offset_lat in _axis_values(center_lat, radius, step):
                # GCJ-02 correction varies smoothly over this small search window.
                # Reuse the exact datum correction at the historical center while
                # exploring candidates, then recompute the selected candidate
                # exactly below.
                query_xy = _metric_xy(
                    reference_wgs_lng + offset_lng - reference_lng,
                    reference_wgs_lat + offset_lat - reference_lat,
                )
                distances, _ = poi_tree.query(query_xy, k=1)
                matched_count = int(np.count_nonzero(distances <= match_radius_m))
                mean_distance = float(np.mean(distances))
                row = {
                    "level": level_number,
                    "offset_lng_degrees": float(offset_lng),
                    "offset_lat_degrees": float(offset_lat),
                    "matched_count": matched_count,
                    "match_rate": matched_count / len(shop_lng),
                    "mean_nearest_distance_m": mean_distance,
                    "exact_rerank": False,
                }
                trace.append(row)
                level_candidates.append(row)
        shortlist = sorted(
            level_candidates,
            key=lambda row: (
                -row["matched_count"],
                row["mean_nearest_distance_m"],
                row["offset_lng_degrees"],
                row["offset_lat_degrees"],
            ),
        )[:EXACT_RERANK_CANDIDATES]
        best_key = None
        level_best = None
        for row in shortlist:
            exact_wgs_lng, exact_wgs_lat = _convert_pairs(
                shop_lng + row["offset_lng_degrees"],
                shop_lat + row["offset_lat_degrees"],
                gcj02_to_wgs84,
            )
            exact_distances, exact_indices = poi_tree.query(
                _metric_xy(exact_wgs_lng, exact_wgs_lat), k=1
            )
            exact_count = int(np.count_nonzero(exact_distances <= match_radius_m))
            exact_mean = float(np.mean(exact_distances))
            row["exact_rerank"] = True
            row["exact_matched_count"] = exact_count
            row["exact_match_rate"] = exact_count / len(shop_lng)
            row["exact_mean_nearest_distance_m"] = exact_mean
            key = (
                -exact_count,
                exact_mean,
                row["offset_lng_degrees"],
                row["offset_lat_degrees"],
            )
            if best_key is None or key < best_key:
                best_key = key
                level_best = (
                    row["offset_lng_degrees"],
                    row["offset_lat_degrees"],
                    exact_distances,
                    exact_indices,
                )
        if level_best is None:
            raise RuntimeError("Coordinate search produced no candidates")
        center_lng, center_lat, best_distances, best_indices = level_best
    final_wgs_lng, final_wgs_lat = _convert_pairs(
        shop_lng + center_lng,
        shop_lat + center_lat,
        gcj02_to_wgs84,
    )
    best_distances, best_indices = poi_tree.query(
        _metric_xy(final_wgs_lng, final_wgs_lat), k=1
    )
    return SearchResult(
        center_lng,
        center_lat,
        np.asarray(best_distances),
        np.asarray(best_indices, dtype=int),
        pd.DataFrame(trace),
    )


def append_corrected_shop_coordinates(
    shops: pd.DataFrame, poi: pd.DataFrame, result: SearchResult
) -> pd.DataFrame:
    """Append restored coordinates and nearest-POI audit fields to shops."""
    corrected = shops.copy(deep=True)
    corrected["pick_lng_gcj02"] = (
        corrected["original_pick_lng"] + result.longitude_degrees
    )
    corrected["pick_lat_gcj02"] = (
        corrected["original_pick_lat"] + result.latitude_degrees
    )
    wgs_lng, wgs_lat = _convert_pairs(
        corrected["pick_lng_gcj02"].to_numpy(),
        corrected["pick_lat_gcj02"].to_numpy(),
        gcj02_to_wgs84,
    )
    corrected["pick_lng_wgs84"] = wgs_lng
    corrected["pick_lat_wgs84"] = wgs_lat
    nearest = poi.iloc[result.nearest_indices].reset_index(drop=True)
    corrected["nearest_poi_name"] = nearest["poi_name"]
    corrected["nearest_poi_category"] = nearest["poi_category"]
    corrected["nearest_poi_lng_wgs84"] = nearest["poi_lng_wgs84"]
    corrected["nearest_poi_lat_wgs84"] = nearest["poi_lat_wgs84"]
    corrected["nearest_poi_distance_m"] = result.distances_m
    return corrected


def append_corrected_order_coordinates(
    orders: pd.DataFrame, offset_lng: float, offset_lat: float
) -> pd.DataFrame:
    """Append corrected pickup/delivery coordinate pairs without overwriting input."""
    corrected = orders.copy(deep=True)
    for prefix in ("pick", "deliver"):
        gcj_lng = corrected[f"{prefix}_lng"].to_numpy(float) + offset_lng
        gcj_lat = corrected[f"{prefix}_lat"].to_numpy(float) + offset_lat
        wgs_lng, wgs_lat = _convert_pairs(gcj_lng, gcj_lat, gcj02_to_wgs84)
        corrected[f"{prefix}_lng_gcj02"] = gcj_lng
        corrected[f"{prefix}_lat_gcj02"] = gcj_lat
        corrected[f"{prefix}_lng_wgs84"] = wgs_lng
        corrected[f"{prefix}_lat_wgs84"] = wgs_lat
    return corrected


def match_metrics(distances_m: np.ndarray) -> dict:
    values = np.asarray(distances_m, dtype=float)
    return {
        "match_rates": {
            f"within_{int(threshold)}m": float(np.mean(values <= threshold))
            for threshold in METRIC_THRESHOLDS_M
        },
        "mean_nearest_distance_m": float(np.mean(values)),
        "median_nearest_distance_m": float(np.median(values)),
        "p95_nearest_distance_m": float(np.percentile(values, 95)),
    }


def build_transform_record(
    result: SearchResult,
    orders: pd.DataFrame,
    shops: pd.DataFrame,
    poi: pd.DataFrame,
    order_fingerprint: dict,
    poi_fingerprint: dict,
    poi_datum: str,
    poi_declared_crs: str | None,
    match_radius_m: float,
) -> dict:
    return {
        "schema_version": 1,
        "status": "estimated",
        "model": "global_constant_translation",
        "offset": {
            "longitude_degrees": result.longitude_degrees,
            "latitude_degrees": result.latitude_degrees,
            "operation": "corrected = original + offset",
        },
        "coordinate_reference": {
            "orders_original": "translated_gcj02",
            "orders_corrected_native": "gcj02",
            "orders_corrected_export": "wgs84",
            "poi_adopted_datum": poi_datum,
            "poi_declared_crs": poi_declared_crs,
            "declared_adopted_datum_mismatch": bool(
                poi_declared_crs
                and poi_datum == "gcj02"
                and ("4326" in poi_declared_crs or "WGS" in poi_declared_crs.upper())
            ),
            "metric_crs": BEIJING_METRIC_CRS,
        },
        "search": {
            "initial_offset": {
                "longitude_degrees": HISTORICAL_OFFSET[0],
                "latitude_degrees": HISTORICAL_OFFSET[1],
            },
            "levels": list(DEFAULT_SEARCH_LEVELS),
            "objective": [
                f"maximize shops within {match_radius_m:g} m",
                "minimize all-shop mean nearest-POI distance",
            ],
            "match_radius_m": match_radius_m,
            "candidate_count": int(len(result.surface)),
            "exact_rerank_candidates_per_level": EXACT_RERANK_CANDIDATES,
        },
        "inputs": {"orders": order_fingerprint, "poi": poi_fingerprint},
        "counts": {
            "orders": int(len(orders)),
            "unique_shops": int(len(shops)),
            "poi": int(len(poi)),
        },
        "metrics": match_metrics(result.distances_m),
    }


def _add_basemap(ax, cache_dir: Path) -> None:
    try:
        import contextily as ctx

        tile_cache = Path(cache_dir) / "tiles"
        tile_cache.mkdir(parents=True, exist_ok=True)
        ctx.set_cache_dir(str(tile_cache))
        ctx.add_basemap(
            ax,
            crs="EPSG:4326",
            # The canonical Mapnik endpoint may return a 403 image while still
            # reporting a successful response to contextily.  HOT is an OSM
            # tile style with an endpoint that supports this audit workflow.
            source=ctx.providers.OpenStreetMap.HOT,
            attribution=False,
            zorder=0,
        )
    except Exception as error:
        raise RuntimeError(
            "OpenStreetMap basemap retrieval failed. Cached tiles are used first; "
            "rerun with network access or pass --no-basemap explicitly."
        ) from error


def _typical_centers(corrected_shops: pd.DataFrame) -> list[tuple[str, float, float]]:
    xy = _metric_xy(
        corrected_shops["pick_lng_wgs84"].to_numpy(),
        corrected_shops["pick_lat_wgs84"].to_numpy(),
    )
    center_xy = _metric_xy(np.array([116.3975]), np.array([39.9087]))[0]
    work = corrected_shops.copy()
    work["x"] = xy[:, 0]
    work["y"] = xy[:, 1]
    work["center_distance_m"] = np.linalg.norm(xy - center_xy, axis=1)
    work["zone_x"] = np.floor(work["x"] / 4000).astype(int)
    work["zone_y"] = np.floor(work["y"] / 4000).astype(int)
    selections = []
    for label, mask in (
        ("Typical central area", work["center_distance_m"] <= 20000),
        ("Typical peripheral area", work["center_distance_m"] > 20000),
    ):
        subset = work.loc[mask]
        grouped = (
            subset.groupby(["zone_x", "zone_y"], as_index=False)
            .agg(
                local_shop_count=("shop_id", "size"),
                median_error_m=("nearest_poi_distance_m", "median"),
                center_x=("x", "median"),
                center_y=("y", "median"),
            )
        )
        eligible = grouped[grouped["local_shop_count"] >= 20].copy()
        if eligible.empty:
            eligible = grouped.copy()
        if eligible.empty:
            continue
        target = float(subset["nearest_poi_distance_m"].median())
        eligible["error_gap"] = (eligible["median_error_m"] - target).abs()
        chosen = eligible.sort_values(
            ["error_gap", "local_shop_count"], ascending=[True, False]
        ).iloc[0]
        lng, lat = _FROM_METRIC.transform(chosen["center_x"], chosen["center_y"])
        selections.append((label, float(lng), float(lat)))
    return selections


def plot_alignment_audit(
    corrected_shops: pd.DataFrame,
    poi: pd.DataFrame,
    result: SearchResult,
    output_pdf: Path,
    output_png: Path,
    source_csv: Path,
    no_basemap: bool,
    cache_dir: Path,
) -> None:
    """Plot search evidence and four fixed-radius local alignment audits."""
    import matplotlib.pyplot as plt

    anchor_gcj02 = (
        ("Sanlitun paper anchor", 116.4551, 39.9368),
        ("Qinghe–Shangdi paper anchor", 116.3190, 40.0350),
    )
    anchors = [
        (label, *gcj02_to_wgs84(lng, lat))
        for label, lng, lat in anchor_gcj02
    ] + _typical_centers(corrected_shops)
    if len(anchors) < 4:
        raise ValueError("Insufficient spatial coverage to choose four local audit areas")
    shops_xy = _metric_xy(
        corrected_shops["pick_lng_wgs84"].to_numpy(),
        corrected_shops["pick_lat_wgs84"].to_numpy(),
    )
    poi_xy = _metric_xy(
        poi["poi_lng_wgs84"].to_numpy(), poi["poi_lat_wgs84"].to_numpy()
    )
    rows = []
    fig = plt.figure(figsize=(13, 8.5))
    grid = fig.add_gridspec(2, 3, width_ratios=(1.05, 1, 1))
    objective_ax = fig.add_subplot(grid[:, 0])
    final_level = result.surface[result.surface["level"] == result.surface["level"].max()]
    scatter = objective_ax.scatter(
        final_level["offset_lng_degrees"],
        final_level["offset_lat_degrees"],
        c=final_level["match_rate"],
        s=34,
        cmap="viridis",
    )
    objective_ax.scatter(
        [result.longitude_degrees], [result.latitude_degrees], marker="*", s=170,
        color="#d62728", edgecolor="white", linewidth=0.8, label="Selected offset",
    )
    objective_ax.set_xlabel("Longitude offset (degrees)")
    objective_ax.set_ylabel("Latitude offset (degrees)")
    objective_ax.ticklabel_format(useOffset=False)
    objective_ax.legend(loc="lower right", frameon=False)
    fig.colorbar(scatter, ax=objective_ax, orientation="horizontal", pad=0.08,
                 label="Share within match radius")
    metrics = match_metrics(result.distances_m)
    objective_ax.text(
        0.03, 0.97,
        f"Offset: ({result.longitude_degrees:+.5f}°, {result.latitude_degrees:+.5f}°)\n"
        f"Within 100 m: {metrics['match_rates']['within_100m']:.1%}\n"
        f"Mean / median / p95: {metrics['mean_nearest_distance_m']:.1f} / "
        f"{metrics['median_nearest_distance_m']:.1f} / "
        f"{metrics['p95_nearest_distance_m']:.1f} m",
        transform=objective_ax.transAxes, va="top", fontsize=9,
        bbox={"facecolor": "white", "edgecolor": "0.8", "alpha": 0.9},
    )

    local_axes = [fig.add_subplot(grid[row, column]) for row in range(2) for column in (1, 2)]
    for ax, (label, center_lng, center_lat) in zip(local_axes, anchors[:4]):
        center_xy = _metric_xy(np.array([center_lng]), np.array([center_lat]))[0]
        shop_mask = np.linalg.norm(shops_xy - center_xy, axis=1) <= 2000
        poi_mask = np.linalg.norm(poi_xy - center_xy, axis=1) <= 2000
        local_shops = corrected_shops.loc[shop_mask]
        local_poi = poi.loc[poi_mask]
        radius_lng = 2000 / (111320 * math.cos(math.radians(center_lat)))
        radius_lat = 2000 / 110540
        ax.set_xlim(center_lng - radius_lng, center_lng + radius_lng)
        ax.set_ylim(center_lat - radius_lat, center_lat + radius_lat)
        if not no_basemap:
            try:
                _add_basemap(ax, cache_dir)
            except Exception:
                plt.close(fig)
                raise
        ax.scatter(
            local_poi["poi_lng_wgs84"], local_poi["poi_lat_wgs84"], s=6,
            color="#4c78a8", alpha=0.42, edgecolors="none", label="Food POI", zorder=3,
        )
        ax.scatter(
            local_shops["pick_lng_wgs84"], local_shops["pick_lat_wgs84"], s=13,
            color="#e45756", alpha=0.85, edgecolors="white", linewidth=0.2,
            label="Corrected shop", zorder=5,
        )
        for _, shop in local_shops.iterrows():
            ax.plot(
                [shop["pick_lng_wgs84"], shop["nearest_poi_lng_wgs84"]],
                [shop["pick_lat_wgs84"], shop["nearest_poi_lat_wgs84"]],
                color="0.25", alpha=0.25, linewidth=0.35, zorder=4,
            )
            rows.append(
                {
                    "audit_area": label,
                    "center_lng_wgs84": center_lng,
                    "center_lat_wgs84": center_lat,
                    "shop_id": shop["shop_id"],
                    "shop_lng_wgs84": shop["pick_lng_wgs84"],
                    "shop_lat_wgs84": shop["pick_lat_wgs84"],
                    "nearest_poi_lng_wgs84": shop["nearest_poi_lng_wgs84"],
                    "nearest_poi_lat_wgs84": shop["nearest_poi_lat_wgs84"],
                    "nearest_poi_distance_m": shop["nearest_poi_distance_m"],
                }
            )
        ax.set_title(f"{label}\n2 km radius; n={len(local_shops)} shops", fontsize=9)
        ax.set_aspect(1 / math.cos(math.radians(center_lat)))
        ax.ticklabel_format(useOffset=False, style="plain")
        ax.tick_params(labelsize=7)
    handles, labels = local_axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(source_csv, index=False)
    try:
        fig.savefig(output_pdf, dpi=300, bbox_inches="tight")
        fig.savefig(output_png, dpi=300, bbox_inches="tight")
    finally:
        plt.close(fig)


def plot_supply_demand_heatmap(
    corrected_orders: pd.DataFrame,
    output_pdf: Path,
    output_png: Path,
    source_csv: Path,
    gridsize: int,
    no_basemap: bool,
    cache_dir: Path,
) -> None:
    """Plot Beijing-wide pickup supply and delivery demand on shared scales."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    if gridsize <= 1:
        raise ValueError("heatmap_gridsize must be greater than one")
    supply = (
        corrected_orders.groupby(["pick_lng_wgs84", "pick_lat_wgs84"], as_index=False)
        .size()
        .rename(columns={"size": "weight"})
    )
    demand = corrected_orders.loc[:, ["deliver_lng_wgs84", "deliver_lat_wgs84"]].copy()
    demand["weight"] = 1
    source = pd.concat(
        [
            supply.rename(
                columns={
                    "pick_lng_wgs84": "lng_wgs84",
                    "pick_lat_wgs84": "lat_wgs84",
                }
            ).assign(layer="pickup_supply"),
            demand.rename(
                columns={
                    "deliver_lng_wgs84": "lng_wgs84",
                    "deliver_lat_wgs84": "lat_wgs84",
                }
            ).assign(layer="delivery_demand"),
        ],
        ignore_index=True,
    )
    source.to_csv(source_csv, index=False, compression="gzip")
    x_min, x_max = source["lng_wgs84"].min(), source["lng_wgs84"].max()
    y_min, y_max = source["lat_wgs84"].min(), source["lat_wgs84"].max()
    pad_x, pad_y = (x_max - x_min) * 0.03, (y_max - y_min) * 0.03
    extent = (x_min - pad_x, x_max + pad_x, y_min - pad_y, y_max + pad_y)
    fig = plt.figure(figsize=(13, 6))
    grid = fig.add_gridspec(1, 3, width_ratios=(1, 1, 0.045), wspace=0.08)
    axes = [fig.add_subplot(grid[0, 0])]
    axes.append(fig.add_subplot(grid[0, 1], sharex=axes[0], sharey=axes[0]))
    colorbar_ax = fig.add_subplot(grid[0, 2])
    collections = []
    for ax, frame, x, y, title in (
        (axes[0], supply, "pick_lng_wgs84", "pick_lat_wgs84", "Pickup supply (order-weighted)"),
        (axes[1], demand, "deliver_lng_wgs84", "deliver_lat_wgs84", "Delivery demand"),
    ):
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
        if not no_basemap:
            try:
                _add_basemap(ax, cache_dir)
            except Exception:
                plt.close(fig)
                raise
        collection = ax.hexbin(
            frame[x], frame[y], C=frame["weight"], reduce_C_function=np.sum,
            gridsize=gridsize, extent=extent, mincnt=1, cmap="magma",
            norm=LogNorm(vmin=1), alpha=0.82, linewidths=0, zorder=3,
        )
        collections.append(collection)
        ax.set_title(title)
        ax.set_xlabel("Longitude (WGS84)")
        ax.set_aspect(1 / math.cos(math.radians(float((y_min + y_max) / 2))))
    axes[0].set_ylabel("Latitude (WGS84)")
    common_max = max(float(np.max(item.get_array())) for item in collections)
    for collection in collections:
        collection.set_clim(1, common_max)
    fig.colorbar(
        collections[0], cax=colorbar_ax, label="Orders per hexagon (log scale)"
    )
    fig.subplots_adjust(left=0.07, right=0.95, bottom=0.1, top=0.92)
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    try:
        fig.savefig(output_pdf, dpi=300, bbox_inches="tight")
        fig.savefig(output_png, dpi=300, bbox_inches="tight")
    finally:
        plt.close(fig)


def run_coordinate_restore(
    orders_path: Path,
    poi_path: Path,
    output_dir: Path,
    poi_datum: str = "gcj02",
    match_radius_m: float = 100.0,
    write_orders: bool = False,
    no_basemap: bool = False,
    heatmap_gridsize: int = 100,
    cache_dir: Path | None = None,
) -> dict:
    """Run coordinate estimation, exports, and the two audit figures."""
    orders, order_files = load_orders(Path(orders_path))
    shops = unique_shops(orders)
    poi, poi_files, declared_crs = load_poi(Path(poi_path), poi_datum)
    result = estimate_global_translation(shops, poi, match_radius_m=match_radius_m)
    corrected_shops = append_corrected_shop_coordinates(shops, poi, result)
    corrected_orders = append_corrected_order_coordinates(
        orders, result.longitude_degrees, result.latitude_degrees
    )
    order_root = Path(orders_path) if Path(orders_path).is_dir() else None
    poi_root = Path(poi_path).parent if Path(poi_path).suffix.lower() == ".shp" else None
    order_fingerprint = fingerprint_files(order_files, order_root)
    poi_fingerprint = fingerprint_files(poi_files, poi_root)
    record = build_transform_record(
        result, orders, shops, poi, order_fingerprint, poi_fingerprint,
        poi_datum, declared_crs, match_radius_m,
    )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "coordinate_transform.json").write_text(
        json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    corrected_shops.to_csv(output_dir / "corrected_shops.csv.gz", index=False, compression="gzip")
    result.surface.to_csv(output_dir / "coordinate_search_surface.csv", index=False)
    if write_orders:
        corrected_orders.to_csv(
            output_dir / "corrected_orders.csv.gz", index=False, compression="gzip"
        )
    plot_alignment_audit(
        corrected_shops, poi, result,
        output_dir / "coordinate_alignment_audit.pdf",
        output_dir / "coordinate_alignment_audit.png",
        output_dir / "coordinate_alignment_audit_source.csv",
        no_basemap, cache_dir or output_dir / "basemap_cache",
    )
    plot_supply_demand_heatmap(
        corrected_orders,
        output_dir / "beijing_supply_demand_heatmap.pdf",
        output_dir / "beijing_supply_demand_heatmap.png",
        output_dir / "beijing_supply_demand_heatmap_source.csv.gz",
        heatmap_gridsize, no_basemap, cache_dir or output_dir / "basemap_cache",
    )
    return record


def apply_coordinate_transform(
    orders_path: Path,
    transform_path: Path,
    output_path: Path,
) -> dict:
    """Materialize corrected orders from one audited transform only.

    This deliberately does not load restaurant POIs, search for a new offset, or
    generate coordinate-audit figures.  Every raw order file hash must match the
    transform record before any output is written.
    """
    orders_path = Path(orders_path)
    transform_path = Path(transform_path)
    output_path = Path(output_path)
    transform = json.loads(transform_path.read_text(encoding="utf-8"))
    if transform.get("model") != "global_constant_translation":
        raise ValueError("Transform is not a global_constant_translation record")
    expected = transform.get("inputs", {}).get("orders")
    if not isinstance(expected, dict) or not expected.get("files"):
        raise ValueError("Transform does not contain per-file order fingerprints")
    order_root = orders_path if orders_path.is_dir() else None
    order_files = _input_files(orders_path, {".txt", ".csv", ".csv.gz"})
    actual = fingerprint_files(order_files, order_root)
    expected_core = {
        key: expected[key] for key in ("sha256", "file_count", "files")
        if key in expected
    }
    if actual != expected_core:
        raise ValueError(
            "Raw order fingerprint does not match the audited coordinate transform"
        )
    if actual["file_count"] != 29:
        raise ValueError(
            f"Expected the audited 29-day order set, found {actual['file_count']} files"
        )
    orders, _ = load_orders(orders_path)
    offset = transform.get("offset", {})
    corrected = append_corrected_order_coordinates(
        orders,
        float(offset["longitude_degrees"]),
        float(offset["latitude_degrees"]),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    corrected.to_csv(output_path, index=False, compression="gzip")
    output_fingerprint = fingerprint_files([output_path])
    return {
        "schema_version": 1,
        "mode": "apply_existing_transform",
        "transform": str(transform_path),
        "orders": actual,
        "corrected_orders": output_fingerprint,
        "counts": {
            "orders": int(len(corrected)),
            "shops": int(corrected["shop_id"].nunique()),
        },
    }
