import json
import os

import cpca
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from matplotlib.colors import LinearSegmentedColormap, Normalize, TwoSlopeNorm
from matplotlib.patches import Patch

WEIGHTED_REDUCTION_COLUMNS = (
    "oracle_min_reduction_vs_ebike_pct",
    "max_lifecycle_reduction_pct",
    "lifecycle_all_drone_reduction_pct",
)

# Project root directory
_PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)


def _batch_city_to_adcode(city_names: list) -> dict:
    """Resolve Chinese city names to 6-digit adcodes using cpca (batched)."""
    adcode_map = {}
    try:
        df = cpca.transform(list(city_names))
        for city_name, adcode in zip(city_names, df["adcode"]):
            if not pd.isna(adcode):
                adcode_map[city_name] = str(int(adcode))[:6]
    except Exception as e:
        print(f"Error in batch adcode resolution: {e}")
    return adcode_map


def _fetch_city_boundaries(adcode: str):
    """Fetch city boundary GeoJSON from cache or DataV API."""
    cache_dir = os.path.join(_PROJECT_ROOT, "cache", "boundaries")
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f"{adcode}.json")

    if os.path.exists(cache_file):
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass

    url = f"https://geo.datav.aliyun.com/areas_v3/bound/{adcode}_full.json"
    try:
        print(f"Downloading boundaries for adcode {adcode}...")
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        data = response.json()
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
        return data
    except Exception as e:
        print(f"Failed to fetch boundaries for adcode {adcode}: {e}")
    return None


def _resolve_color_by(df: pd.DataFrame, color_by: str) -> str:
    """Prefer weighted reduction columns when legacy reduction output is absent."""
    if color_by in df.columns:
        return color_by
    if color_by == "max_lifecycle_reduction_pct":
        for candidate in WEIGHTED_REDUCTION_COLUMNS:
            if candidate in df.columns:
                return candidate
    return color_by


def _fetch_china_boundaries():
    """Fetch China provincial boundaries from DataV API."""
    # China adcode is 100000
    adcode = "100000"
    cache_dir = os.path.join(_PROJECT_ROOT, "cache", "boundaries")
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f"{adcode}.json")

    # Try to load from cache
    if os.path.exists(cache_file):
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading cache: {e}")

    # Download if not in cache
    url = f"https://geo.datav.aliyun.com/areas_v3/bound/{adcode}_full.json"
    try:
        print("Downloading China boundaries...")
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        data = response.json()

        # Save to cache
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)

        return data
    except Exception as e:
        print(f"Failed to fetch boundaries: {e}")
        return None


def _plot_china_boundaries(ax, geojson_data):
    """Plot China boundaries on the axes."""
    if not geojson_data:
        return

    for feature in geojson_data.get("features", []):
        geometry = feature.get("geometry")
        if not geometry:
            continue

        geom_type = geometry.get("type")
        coords = geometry.get("coordinates")

        def plot_polygon(rings):
            for ring in rings:
                if isinstance(ring[0][0], list):
                    # Nested polygon
                    plot_polygon(ring)
                else:
                    xs, ys = zip(*[(pt[0], pt[1]) for pt in ring])
                    ax.plot(xs, ys, color="gray", linewidth=0.5, alpha=0.8, zorder=1)

        if geom_type == "Polygon":
            plot_polygon(coords)
        elif geom_type == "MultiPolygon":
            for polygon in coords:
                plot_polygon(polygon)


_MUNICIPALITY_PREFIXES = frozenset({"11", "12", "31", "50", "81", "82"})
_PROVINCE_SUFFIXES = (
    "省",
    "市",
    "壮族自治区",
    "回族自治区",
    "维吾尔自治区",
    "自治区",
    "特别行政区",
)
_PROVINCE_ALIASES = {
    "内蒙古自治区": "内蒙古",
    "广西壮族自治区": "广西",
    "宁夏回族自治区": "宁夏",
    "新疆维吾尔自治区": "新疆",
    "西藏自治区": "西藏",
    "香港特别行政区": "香港",
    "澳门特别行政区": "澳门",
}


def _fill_geometry(ax, geometry, facecolor):
    """Fill a single GeoJSON geometry dict on the axes."""
    if not geometry:
        return

    geom_type = geometry.get("type")
    coords = geometry.get("coordinates")

    def fill_rings(rings):
        if isinstance(rings[0][0], (list, tuple)) and isinstance(
            rings[0][0][0], (list, tuple)
        ):
            for subrings in rings:
                fill_rings(subrings)
        else:
            outer = rings[0]
            xs, ys = zip(*[(pt[0], pt[1]) for pt in outer])
            ax.fill(
                xs, ys,
                facecolor=facecolor,
                edgecolor="none",
                linewidth=0,
                zorder=2,
            )

    if geom_type == "Polygon":
        fill_rings(coords)
    elif geom_type == "MultiPolygon":
        for polygon in coords:
            fill_rings(polygon)


def _normalize_province_feature_name(name: str) -> str:
    """Normalize DataV province names to match the GEF table."""
    name = str(name).strip()
    if name in _PROVINCE_ALIASES:
        return _PROVINCE_ALIASES[name]
    for suffix in _PROVINCE_SUFFIXES:
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def _plot_geometry_outline(ax, geometry, color="#666666", linewidth=0.5, alpha=1.0, zorder=4):
    """Plot GeoJSON polygon rings as outlines."""
    if not geometry:
        return

    geom_type = geometry.get("type")
    coords = geometry.get("coordinates")

    def plot_rings(rings):
        if not rings:
            return
        if isinstance(rings[0][0], (list, tuple)) and isinstance(
            rings[0][0][0], (list, tuple)
        ):
            for subrings in rings:
                plot_rings(subrings)
        else:
            for ring in rings:
                xs, ys = zip(*[(pt[0], pt[1]) for pt in ring])
                ax.plot(
                    xs,
                    ys,
                    color=color,
                    linewidth=linewidth,
                    alpha=alpha,
                    zorder=zorder,
                )

    if geom_type == "Polygon":
        plot_rings(coords)
    elif geom_type == "MultiPolygon":
        for polygon in coords:
            plot_rings(polygon)


def _province_gef_report(
    save_path,
    gef_path,
    gef_by_province,
    colored_provinces,
    missing_provinces,
    jd_drawn,
    south_china_inset_extent,
):
    """Write a JSON audit report for the province GEF China map."""
    report_path = save_path.rsplit(".", 1)[0] + "_report.json"
    values = [float(v) for v in gef_by_province.values()]
    payload = {
        "figure_name": "province_gef_china_map",
        "figure_path": save_path,
        "source_gef_path": gef_path,
        "source_encoding": "gbk",
        "value_column": "因子",
        "unit": "kg CO2e/kWh",
        "colored_province_count": len(colored_provinces),
        "colored_provinces": colored_provinces,
        "missing_provinces": missing_provinces,
        "gef_range": {
            "min": min(values) if values else None,
            "max": max(values) if values else None,
        },
        "color_scale": "low GEF green to high GEF red",
        "south_china_sea_inset": {
            "drawn": True,
            "extent": list(south_china_inset_extent),
            "nine_dash_feature_drawn": bool(jd_drawn),
        },
    }
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, allow_nan=False)
    print(f"Province GEF China map report saved to {report_path}")


def plot_province_gef_china_map(
    gef_path: str,
    save_path: str = None,
    figsize: tuple = (3.4, 3.0),
    for_latex: bool = False,
    plot_title: bool = False,
    south_china_inset_bounds: tuple = (0.72, 0.03, 0.25, 0.32),
    south_china_inset_extent: tuple = (107, 123.5, 3, 25),
):
    """Plot province-level China GEF heatmap with South China Sea inset.

    Args:
        gef_path: Path to province-level GEF CSV with columns `城市` and `因子`.
        save_path: Path to save figure. Shows plot if None.
        figsize: Figure size tuple.
        for_latex: If True, save as PDF.
        plot_title: If True, show title.
        south_china_inset_bounds: Axes-fraction bounds for the South China Sea inset.
        south_china_inset_extent: Data extent for the inset as
            (min_lng, max_lng, min_lat, max_lat).
    """
    if not gef_path or not os.path.exists(gef_path):
        print(f"Province GEF China map skipped — missing {gef_path}")
        return

    gef_df = pd.read_csv(gef_path, encoding="gbk")
    needed = {"城市", "因子"}
    missing_cols = needed - set(gef_df.columns)
    if missing_cols:
        print(f"Province GEF China map skipped — missing columns {sorted(missing_cols)}")
        return

    gef_df = gef_df.copy()
    gef_df["province"] = gef_df["城市"].map(_normalize_province_feature_name)
    gef_df["gef"] = pd.to_numeric(gef_df["因子"], errors="coerce")
    gef_by_province = (
        gef_df.dropna(subset=["province", "gef"])
        .drop_duplicates("province")
        .set_index("province")["gef"]
        .to_dict()
    )
    if not gef_by_province:
        print("Province GEF China map skipped — no valid GEF values")
        return

    china_geojson = _fetch_china_boundaries()
    if not china_geojson:
        print("Province GEF China map skipped — China boundaries unavailable")
        return

    provinces = []
    nine_dash_feature = None
    for feature in china_geojson.get("features", []):
        props = feature.get("properties", {})
        adcode = str(props.get("adcode", ""))
        if adcode == "100000_JD" or props.get("adchar") == "JD":
            nine_dash_feature = feature
            continue
        if not adcode.isdigit():
            continue
        name = props.get("name", "")
        province = _normalize_province_feature_name(name)
        geometry = feature.get("geometry")
        if geometry:
            provinces.append((province, name, geometry))

    values = np.array(list(gef_by_province.values()), dtype=float)
    cmap = LinearSegmentedColormap.from_list(
        "gef_low_green_high_red",
        ["#2ca25f", "#f7fcb9", "#fdae61", "#d73027"],
    )
    norm = Normalize(vmin=float(np.nanmin(values)), vmax=float(np.nanmax(values)))
    no_data_color = "#d9d9d9"
    outline_color = "#6f6f6f"

    fig, ax = plt.subplots(figsize=figsize)

    colored_provinces = []
    missing_provinces = []

    def draw_panel(target_ax, draw_missing_legend=False, draw_nine_dash=False):
        for province, display_name, geometry in provinces:
            gef = gef_by_province.get(province)
            if gef is None or np.isnan(gef):
                facecolor = no_data_color
                if province not in missing_provinces:
                    missing_provinces.append(province)
            else:
                facecolor = cmap(norm(float(gef)))
                if province not in colored_provinces:
                    colored_provinces.append(province)
            _fill_geometry(target_ax, geometry, facecolor)
            _plot_geometry_outline(
                target_ax,
                geometry,
                color=outline_color,
                linewidth=0.35,
                alpha=0.85,
                zorder=3,
            )

        jd_drawn = False
        if draw_nine_dash and nine_dash_feature is not None:
            _fill_geometry(target_ax, nine_dash_feature.get("geometry"), "#f7f7f7")
            _plot_geometry_outline(
                target_ax,
                nine_dash_feature.get("geometry"),
                color="#4d4d4d",
                linewidth=0.55,
                alpha=0.95,
                zorder=5,
            )
            jd_drawn = True

        if draw_missing_legend:
            target_ax.legend(
                handles=[
                    Patch(
                        facecolor=no_data_color,
                        edgecolor=outline_color,
                        linewidth=0.4,
                        label="No GEF data",
                    )
                ],
                loc="lower left",
                fontsize=6.5,
                frameon=False,
                handlelength=1.0,
                borderaxespad=0.2,
            )
        return jd_drawn

    draw_panel(ax, draw_missing_legend=True)

    china_lat_range = (18, 54)
    china_lng_range = (73, 135)
    ax.set_xlim(china_lng_range)
    ax.set_ylim(china_lat_range)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    avg_lat = np.mean(china_lat_range)
    ax.set_aspect(1.0 / np.cos(np.radians(avg_lat)))
    for spine in ax.spines.values():
        spine.set_visible(False)

    inset_ax = ax.inset_axes(south_china_inset_bounds, zorder=10)
    jd_drawn = draw_panel(inset_ax, draw_nine_dash=True)
    inset_min_lng, inset_max_lng, inset_min_lat, inset_max_lat = south_china_inset_extent
    inset_ax.set_xlim(inset_min_lng, inset_max_lng)
    inset_ax.set_ylim(inset_min_lat, inset_max_lat)
    inset_ax.set_xticks([])
    inset_ax.set_yticks([])
    inset_ax.set_facecolor("white")
    for spine in inset_ax.spines.values():
        spine.set_linestyle("--")
        spine.set_linewidth(0.7)
        spine.set_edgecolor("#777777")

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.66, pad=0.01, fraction=0.045)
    cbar.set_label("GEF (kg CO2e/kWh)", fontsize=7)
    cbar.ax.tick_params(labelsize=6.5, length=2)
    cbar.outline.set_linewidth(0.4)

    if plot_title:
        ax.set_title("Province Grid Emission Factor", fontsize=9, fontweight="bold")

    fig.tight_layout(pad=0.2)

    if save_path:
        if for_latex and not save_path.endswith(".pdf"):
            save_path = save_path.rsplit(".", 1)[0] + ".pdf"
        if plot_title:
            from ._utils import _add_title_suffix
            save_path = _add_title_suffix(save_path)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Province GEF China map saved to {save_path}")
        _province_gef_report(
            save_path,
            gef_path,
            gef_by_province,
            sorted(colored_provinces),
            sorted(missing_provinces),
            jd_drawn,
            south_china_inset_extent,
        )
    else:
        plt.show()

    plt.close(fig)
    return fig, ax


def _ensure_all_china_city_areas():
    """Return list of (adcode, name, geometry_dict) for every city-level area in China.

    For municipalities and SARs the province-level geometry (from the China
    boundary) is used directly. For regular provinces the boundary is fetched
    and its city/prefecture features are returned.
    """
    china = _fetch_china_boundaries()
    if not china:
        return []

    areas = []
    missing_provinces = 0

    for feature in china.get("features", []):
        props = feature.get("properties", {})
        prov_adcode = str(props.get("adcode", ""))
        prov_name = props.get("name", "")
        # Skip pseudo-provinces like 九段线 (100000_JD)
        if not prov_adcode.isdigit():
            continue
        prov_prefix = prov_adcode[:2] if len(prov_adcode) >= 2 else ""

        if prov_prefix in _MUNICIPALITY_PREFIXES:
            # Municipality or SAR: use province-level geometry directly
            geometry = feature.get("geometry")
            if geometry:
                areas.append((prov_adcode, prov_name, geometry))
        else:
            # Regular province: fetch full boundary to get city features
            prov_geojson = _fetch_city_boundaries(prov_adcode)
            if not prov_geojson:
                # Fallback: use province-level geometry from China boundary
                geometry = feature.get("geometry")
                if geometry:
                    areas.append((prov_adcode, prov_name, geometry))
                else:
                    missing_provinces += 1
                continue

            for city_feat in prov_geojson.get("features", []):
                city_props = city_feat.get("properties", {})
                city_adcode = str(city_props.get("adcode", ""))
                city_name = city_props.get("name", "")
                geometry = city_feat.get("geometry")
                if geometry:
                    areas.append((city_adcode, city_name, geometry))

    print(
        f"Built {len(areas)} city-level areas across China "
        f"(missing {missing_provinces} provinces)"
    )
    return areas


def plot_regions_on_china_map(
    region_keys: list = None,
    regions_json_path: str = None,
    csv_path: str = None,
    save_path: str = None,
    figsize: tuple = (12, 10),
    for_latex: bool = False,
    plot_title: bool = False,
    color_by: str = "max_lifecycle_reduction_pct",
    ax: "plt.Axes" = None,
    df: "pd.DataFrame" = None,
    colorbar_label: str = None,
    norm: "TwoSlopeNorm" = None,
    cmap: "LinearSegmentedColormap" = None,
    draw_colorbar: bool = True,
    draw_legend: bool = True,
    south_china_inset_bounds: tuple = (0.86, 0.0, 0.14, 0.28),
    south_china_inset_extent: tuple = (107, 121, 3, 24),
):
    """Plot a full city-boundary choropleth map of China.

    Every city-level area is filled: study cities with valid reduction data use
    a diverging green colour scale centred at 0; all other areas are light gray.
    China province outlines are drawn on top for geographic context.

    Args:
        region_keys: List of region keys to plot. If None, plots all regions.
        regions_json_path: Path to regions.json (legacy mode).
        csv_path: Path to all_region_stats.csv (preferred).
        save_path: Path to save the figure. If None, displays the plot.
        figsize: Figure size tuple (width, height).
        for_latex: If True, skip title for LaTeX export and save as PDF.
        plot_title: If True, show title.
        color_by: Column name to color-code regions by.
        ax: External axes to draw into. When provided, skips figure creation and save.
        df: Optional pre-built DataFrame (takes precedence over csv_path). Must contain
            `region_name_cn` and the `color_by` column.
        colorbar_label: Override label for the colorbar / legend (defaults to
            VARIABLE_LABELS[color_by]).
        norm: Pre-built TwoSlopeNorm to share scale across multi-panel callers.
            If provided, skips per-axes auto-normalization.
        draw_colorbar: If False, skip drawing the per-axes colorbar (caller draws a shared one).
        draw_legend: If False, skip the "Unstudied regions" legend patch.
        south_china_inset_bounds: Axes-fraction bounds for the South China Sea inset.
        south_china_inset_extent: Data extent for the inset as
            (min_lng, max_lng, min_lat, max_lat).
    """
    # --- Load study-city data ---------------------------------------------------
    if df is not None:
        if region_keys is not None:
            df = df[df["region_key"].isin(region_keys)]
        color_by = _resolve_color_by(df, color_by)
        region_names_cn = df["region_name_cn"].values
        color_values = df[color_by].values if color_by in df.columns else None
    elif csv_path is not None:
        df = pd.read_csv(csv_path)
        if region_keys is not None:
            df = df[df["region_key"].isin(region_keys)]
        color_by = _resolve_color_by(df, color_by)

        region_names_cn = df["region_name_cn"].values
        color_values = df[color_by].values if color_by in df.columns else None
    else:
        if regions_json_path is None:
            regions_json_path = os.path.join(_PROJECT_ROOT, "regions.json")
        with open(regions_json_path, "r", encoding="utf-8") as f:
            regions_data = json.load(f)

        names_cn = []
        for rk, ri in regions_data.items():
            if region_keys is not None and rk not in region_keys:
                continue
            names_cn.append(ri.get("region_name_cn", rk))
        region_names_cn = np.array(names_cn)
        color_values = None

    # Study city name → adcode
    unique_study_names = list(set(region_names_cn))
    study_name_to_adcode = _batch_city_to_adcode(unique_study_names)

    # Study city adcode → reduction value (float or NaN)
    study_adcode_to_val = {}
    if color_values is not None:
        color_vals = color_values.astype(float)
        for i, name_cn in enumerate(region_names_cn):
            adcode = study_name_to_adcode.get(name_cn)
            if adcode:
                study_adcode_to_val[adcode] = color_vals[i]

    # --- Build full-China city coverage -----------------------------------------
    all_areas = _ensure_all_china_city_areas()

    # --- Colour mapping ---------------------------------------------------------
    from analysis.constants import VARIABLE_LABELS

    if cmap is None:
        cmap = LinearSegmentedColormap.from_list("green_diverging", [
            "#d7191c",
            "#fdae61",
            "#ffffbf",
            "#a6d96a",
            "#1a9641",
        ])

    valid_values = [v for v in study_adcode_to_val.values() if not np.isnan(v)]
    if norm is None:
        if valid_values:
            max_abs = max(abs(min(valid_values)), abs(max(valid_values)))
            norm = TwoSlopeNorm(vmin=-max_abs, vcenter=0, vmax=max_abs)

    GRAY_FILL = "#cccccc"

    # --- Plot -------------------------------------------------------------------
    _external_ax = ax is not None
    if _external_ax:
        fig = ax.get_figure()
    else:
        fig, ax = plt.subplots(figsize=figsize)

    china_lat_range = (18, 54)
    china_lng_range = (73, 135)

    china_geojson = _fetch_china_boundaries()

    def _draw_map(target_ax, areas_list, study_vals, cmap_, norm_):
        for adcode, name, geometry in areas_list:
            val = study_vals.get(adcode)
            if val is not None and norm_ is not None and not np.isnan(val):
                facecolor = cmap_(norm_(val))
            else:
                facecolor = GRAY_FILL
            _fill_geometry(target_ax, geometry, facecolor)
        _plot_china_boundaries(target_ax, china_geojson)

    # 1. Main map
    plotted_total = 0
    plotted_colored = 0
    for adcode, name, geometry in all_areas:
        val = study_adcode_to_val.get(adcode)
        if val is not None and norm is not None and not np.isnan(val):
            facecolor = cmap(norm(val))
            plotted_colored += 1
        else:
            facecolor = GRAY_FILL
        _fill_geometry(ax, geometry, facecolor)
        plotted_total += 1
    _plot_china_boundaries(ax, china_geojson)

    # 2. Colorbar
    if draw_colorbar and norm is not None:
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, shrink=0.7, pad=0.02)
        cbar.set_label(
            colorbar_label if colorbar_label is not None else VARIABLE_LABELS.get(color_by, color_by),
            fontsize=10,
        )

    # 3. Gray legend patch for unstudied regions
    if draw_legend:
        from matplotlib.patches import Patch
        ax.legend(
            handles=[Patch(facecolor=GRAY_FILL, edgecolor="none", label="Unstudied regions")],
            loc="lower left",
            fontsize=12,
            framealpha=0.8,
            handlelength=1.5,
            handleheight=1.2,
        )

    print(
        f"Plotted {plotted_total} areas ({plotted_colored} colored by reduction, "
        f"{plotted_total - plotted_colored} gray)"
    )

    # --- Axes setup -------------------------------------------------------------
    ax.set_xlim(china_lng_range)
    ax.set_ylim(china_lat_range)
    ax.set_xlabel("Longitude", fontsize=12)
    ax.set_ylabel("Latitude", fontsize=12)
    ax.grid(True, alpha=0.3, linestyle="--")

    avg_lat = np.mean(china_lat_range)
    ax.set_aspect(1.0 / np.cos(np.radians(avg_lat)))

    if plot_title:
        ax.set_title("Study Regions in China", fontsize=14, fontweight="bold")

    # 4. South China Sea inset
    inset_ax = ax.inset_axes(south_china_inset_bounds)
    _draw_map(inset_ax, all_areas, study_adcode_to_val, cmap, norm)
    inset_min_lng, inset_max_lng, inset_min_lat, inset_max_lat = south_china_inset_extent
    inset_ax.set_xlim(inset_min_lng, inset_max_lng)
    inset_ax.set_ylim(inset_min_lat, inset_max_lat)
    inset_ax.set_xticks([])
    inset_ax.set_yticks([])
    for spine in inset_ax.spines.values():
        spine.set_linestyle("--")
        spine.set_linewidth(1.0)
        spine.set_edgecolor("gray")

    if not _external_ax:
        plt.tight_layout()

    # --- Save / show ------------------------------------------------------------
    if _external_ax:
        return
    if save_path:
        if for_latex and not save_path.endswith(".pdf"):
            save_path = save_path.rsplit(".", 1)[0] + ".pdf"
        if plot_title:
            from ._utils import _add_title_suffix
            save_path = _add_title_suffix(save_path)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Plot saved to {save_path}")
    else:
        plt.show()

    plt.close(fig)
    return fig, ax
