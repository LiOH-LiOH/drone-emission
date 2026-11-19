"""Independent new-delivery sample atlas, without paper-workflow integration."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
from datetime import datetime, timezone

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.patches import Rectangle
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd
from pyproj import Geod, Transformer

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
from drone_emission.visualization.palette import (  # noqa: E402
    color, configure_palette, sequential_cmap,
)

REGIONS = {
    "ny": ("New York region", "America/New_York", 32618),
    "la": ("Los Angeles region", "America/Los_Angeles", 32611),
    "sf": ("San Francisco Bay Area", "America/Los_Angeles", 32610),
    "hk": ("Hong Kong", "Asia/Hong_Kong", 32650),
}
PLATFORMS = ["grubhub", "ubereats", "postmates", "doordash", "deliveroo", "foodpanda"]
LABELS = ["Grubhub", "Uber Eats", "Postmates", "DoorDash", "Deliveroo", "Foodpanda"]
FIELDS = ["accountEmailId", "receiptId", "orderId", "receivedDate", "orderTime",
          "pickupLat", "pickupLon", "dropoffLat", "dropoffLon", "countryCode", "plantform"]
COORDS = ["pickupLat", "pickupLon", "dropoffLat", "dropoffLon"]
VERSION = 1
MAP_USER_AGENT = "DroneEmissionDeliveryAtlas/1.0 (+https://github.com/LiOH-LiOH/drone-emission)"


def configure_maps(cache):
    import contextily as ctx
    tile_dir = cache / "tiles_esri_gray_identified"
    tile_dir.mkdir(parents=True, exist_ok=True)
    ctx.set_cache_dir(str(tile_dir))
    ctx.tile.USER_AGENT = MAP_USER_AGENT
    return ctx


def valid_od(frame):
    a = frame[COORDS].to_numpy(dtype=float)
    return (np.isfinite(a).all(axis=1) & (np.abs(a[:, 0]) <= 90)
            & (np.abs(a[:, 2]) <= 90) & (np.abs(a[:, 1]) <= 180)
            & (np.abs(a[:, 3]) <= 180)
            & ~((a[:, 0] == 0) & (a[:, 1] == 0))
            & ~((a[:, 2] == 0) & (a[:, 3] == 0)))


def timestamps(series):
    seconds = pd.to_numeric(series, errors="coerce")
    seconds = seconds.where(seconds > 0)
    seconds = seconds.where(seconds < 1e12, seconds / 1000)
    return pd.to_datetime(seconds, unit="s", errors="coerce", utc=True)


def order_wall_time(series):
    """Decode the source local-clock encoding, without a second timezone shift.

    Grubhub's supplied dictionary explicitly marks orderTime as Local Time.
    Foodpanda has the same encoding: UTC email time precedes encoded order time
    by about 7.4 hours, or follows the Hong Kong local order by about 0.6 hours.
    The UTC tag applied during numeric parsing is a storage convenience only.
    """
    return series.dt.tz_localize(None)


def signature(input_dir):
    payload = [str(VERSION)]
    for name in ("manifest.json", "profile.json", "usable_od_summary.json"):
        payload.append(hashlib.sha256((input_dir / name).read_bytes()).hexdigest())
    return hashlib.sha256("|".join(payload).encode()).hexdigest()


def prepare(input_dir, cache):
    cache.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((input_dir / "manifest.json").read_text())
    entries = {e["canonical"]: e for e in manifest["entries"] if not e["duplicate"]
               and e["canonical"].endswith(".csv")}
    chunks, quality = [], []
    for rel in sorted(entries):
        source = input_dir / rel
        if hashlib.sha256(source.read_bytes()).hexdigest() != entries[rel]["sha256"]:
            raise ValueError(f"Source checksum mismatch: {rel}")
        df = pd.read_csv(source, usecols=lambda c: c in FIELDS, dtype=str)
        for c in COORDS:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        city = "/city_extracts/" in rel
        df["platform"] = df["plantform"].str.lower() if city else rel.split("/")[1]
        df["receipt_time"] = timestamps(df["receivedDate"])
        df["order_time"] = timestamps(df["orderTime"])
        df["od"] = valid_od(df)
        df["ny"] = "New York" in rel
        df["la"] = "Los Angeles" in rel
        df["sf"] = "San Francisco" in rel
        df["hk"] = df["countryCode"].eq("HK")
        df["priority"] = (1 if "New York_city" in rel else 0) if city else 2
        df["source"] = rel
        df["coordinate_source"] = "City-enriched" if city else "Original"
        for c in ("accountEmailId", "receiptId"):
            if df[c].isna().any():
                raise ValueError(f"Missing receipt key in {rel}")
        df["key"] = (df["platform"] + "\x1f" + df["accountEmailId"]
                     + "\x1f" + df["receiptId"])
        quality.append({"source": str(Path(rel).parent), "rows": len(df),
                        "od": int(df.od.sum()),
                        "od_time": int((df.od & df.order_time.notna()).sum())})
        keep = ["key", "platform", "countryCode", *COORDS, "receipt_time", "order_time",
                "od", *REGIONS, "priority", "source", "coordinate_source"]
        chunks.append(df[keep])
        print(f"READ {rel}: {len(df):,}", flush=True)
    all_rows = pd.concat(chunks, ignore_index=True)
    # Stable source order and row order settle ties without a conflict workflow.
    ordered = all_rows.sort_values(["od", "priority"], ascending=[False, True], kind="stable")
    view = ordered.drop_duplicates("key").set_index("key").copy()
    memberships = all_rows.groupby("key", sort=False)[list(REGIONS)].max()
    view[list(REGIONS)] = memberships.reindex(view.index)
    for col in ("order_time", "receipt_time"):
        available = ordered.loc[ordered[col].notna()].drop_duplicates("key").set_index("key")[col]
        view[col] = view[col].fillna(available)
    view["distance_km"] = np.nan
    good = view.loc[view.od]
    _, _, distances = Geod(ellps="WGS84").inv(
        good.pickupLon.to_numpy(), good.pickupLat.to_numpy(),
        good.dropoffLon.to_numpy(), good.dropoffLat.to_numpy())
    view.loc[view.od, "distance_km"] = np.asarray(distances) / 1000
    # Private identifiers are not needed by plotting or its source-data tables.
    view.reset_index(drop=True).to_pickle(cache / "sample_view.pkl")
    pd.DataFrame(quality).groupby("source", sort=False).sum().to_csv(cache / "source_quality.csv")
    counts = {"raw_rows": len(all_rows), "unique_receipts": len(view),
              "valid_od": int(view.od.sum()),
              "valid_od_time": int((view.od & view.order_time.notna()).sum())}
    expected = json.loads((input_dir / "usable_od_summary.json").read_text())
    q = pd.read_csv(cache / "source_quality.csv")
    for row in q.itertuples():
        old = expected["groups"][row.source]
        assert (row.rows, row.od, row.od_time) == (old["rows"], old["od"], old["odTime"])
    assert counts["unique_receipts"] == expected["unionCounts"]["all_unique_receipt_keys"]["all"]
    assert counts["valid_od"] == expected["unionCounts"]["all_unique_receipt_keys"]["od"]
    payload = {"schema_version": VERSION, "input_signature": signature(input_dir), "counts": counts,
               "created_utc": datetime.now(timezone.utc).isoformat(),
               "selection": "Unique platform/account/receipt; valid enriched OD first, zone before city; stable source/row ties; positive timestamps filled within receipt",
               "year_2022": "Local receivedDate year; actual orderTime only for hourly/weekly rhythm",
               "regions": "Source-file memberships; no administrative boundary clipping"}
    (cache / "prepared.json").write_text(json.dumps(payload, indent=2))
    print(json.dumps(counts), flush=True)


def setup():
    configure_palette()
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.titlesize": 12, "axes.labelsize": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "svg.fonttype": "none", "pdf.fonttype": 42,
                         "savefig.facecolor": "white"})


def compact(x, _=None):
    return f"{x / 1e6:g}m" if abs(x) >= 1e6 else f"{x / 1e3:g}k" if abs(x) >= 1000 else f"{x:g}"


class Atlas:
    def __init__(self, view, cache, output):
        self.df, self.cache, self.output = view, cache, output
        self.files, self.tables, self.notes = [], [], []
        self.cmap = sequential_cmap("sequential")
        self.blue, self.teal = color("base.blue"), color("base.green")
        self.yellow, self.gray = color("base.yellow"), color("base.gray")
        self.platform_colors = [color(f"group.{i}") for i in range(6)]
        self.summaries = []
        for r in REGIONS:
            for year in (None, 2022):
                d = self.region(r, year)
                self.summaries.append({"region": REGIONS[r][0], "period": "Full history" if year is None else "2022 receipts",
                                       "receipts": len(d), "valid_od": int(d.od.sum()),
                                       "valid_od_time": int((d.od & d.order_time.notna()).sum())})

    def region(self, r, year=None, od=False):
        d = self.df.loc[self.df[r]]
        if year is not None:
            local = d.receipt_time.dt.tz_convert(REGIONS[r][1])
            d = d.loc[local.dt.year.eq(year)]
        return d.loc[d.od] if od else d

    def table(self, name, df):
        filename = f"{name}.csv"
        df.to_csv(self.output / filename, index=False)
        self.tables.append(filename)

    def save(self, fig, name, note=""):
        # Reserve a real footer band; figure text is not handled by constrained layout.
        fig.get_layout_engine().set(rect=(0, .045, 1, .95))
        fig.text(.01, .008, "Sample receipts; no market expansion. " + note,
                 ha="left", va="bottom", fontsize=8, color=color("ink.secondary"))
        for suffix in ("png", "pdf", "svg"):
            fig.savefig(self.output / f"{name}.{suffix}", dpi=300, facecolor="white")
        self.files.append(f"{name}.png")
        plt.close(fig)
        print(f"SAVED {name}", flush=True)

    def global_map(self):
        import geopandas as gpd
        import requests
        ctx = configure_maps(self.cache)
        target = self.cache / "maps" / "ne_110m_admin_0_countries.zip"
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            response = requests.get("https://naturalearth.s3.amazonaws.com/110m_cultural/ne_110m_admin_0_countries.zip", timeout=90)
            response.raise_for_status()
            temporary = target.with_suffix(".partial")
            temporary.write_bytes(response.content)
            temporary.replace(target)
        world = gpd.read_file(f"zip://{target}")
        fig = plt.figure(figsize=(14, 8.5), layout="constrained")
        grid = fig.add_gridspec(2, 2, height_ratios=[1.2, 1])
        axes = [fig.add_subplot(grid[0, :]), fig.add_subplot(grid[1, 0]), fig.add_subplot(grid[1, 1])]
        d = self.df.loc[self.df.od]
        for ax in axes:
            world.plot(ax=ax, facecolor=color("neutral.light"), edgecolor="white", linewidth=.25)
            ax.scatter(d.dropoffLon, d.dropoffLat, s=.5, color=self.blue, alpha=.16,
                       linewidths=0, rasterized=True)
            ax.set_aspect("equal"); ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
        axes[0].set(xlim=(-180, 180), ylim=(-60, 85), title=f"Global sample coverage | {len(d):,} valid OD receipts")
        us = d.loc[d.countryCode.eq("US")]
        hk = d.loc[d.hk]
        for ax, subset, label in zip(axes[1:], (us, hk), ("United States", "Hong Kong")):
            bounds = [subset.dropoffLon.min(), subset.dropoffLat.min(), subset.dropoffLon.max(), subset.dropoffLat.max()]
            pad_x = max((bounds[2] - bounds[0]) * .08, .03)
            pad_y = max((bounds[3] - bounds[1]) * .08, .03)
            ax.set(xlim=(bounds[0]-pad_x, bounds[2]+pad_x), ylim=(bounds[1]-pad_y, bounds[3]+pad_y),
                   title=f"{label} | n={len(subset):,}")
            if label == "Hong Kong":
                ax.clear()
                ax.set(xlim=(bounds[0]-pad_x, bounds[2]+pad_x), ylim=(bounds[1]-pad_y, bounds[3]+pad_y),
                       title=f"{label} | n={len(subset):,}", xlabel="Longitude", ylabel="Latitude")
                ctx.add_basemap(ax, crs="EPSG:4326", source=ctx.providers.Esri.WorldGrayCanvas,
                                zoom=10, attribution=False, alpha=.65)
                ax.scatter(subset.dropoffLon, subset.dropoffLat, s=.5, color=self.blue,
                           alpha=.18, linewidths=0, rasterized=True)
        for r, (name, _, _) in REGIONS.items():
            q = self.region(r, od=True)
            x, y = q.dropoffLon.median(), q.dropoffLat.median()
            ax = axes[2] if r == "hk" else axes[1]
            ax.annotate(name, (x, y), xytext=(5, 8), textcoords="offset points", fontsize=8,
                        bbox={"facecolor": "white", "alpha": .85, "edgecolor": "none"})
        self.save(fig, "01_global_coverage", "Valid OD destinations. Natural Earth; " + ctx.providers.Esri.WorldGrayCanvas.attribution)

    def quality(self):
        fig = plt.figure(figsize=(14, 10), layout="constrained")
        gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.55])
        colors = [self.gray, self.yellow, self.teal]
        for i, (title, d) in enumerate((("All samples", self.df), ("United States", self.df[self.df.countryCode.eq("US")]),
                                      ("Hong Kong", self.df[self.df.hk]))):
            ax = fig.add_subplot(gs[0, i])
            vals = [int((~d.od).sum()), int((d.od & d.order_time.isna()).sum()), int((d.od & d.order_time.notna()).sum())]
            ax.pie(vals, colors=colors, startangle=90, wedgeprops={"width": .32, "edgecolor": "white"},
                   autopct=lambda pct: f"{pct:.1f}%" if pct >= 3 else "", pctdistance=1.15,
                   textprops={"fontsize": 10})
            ax.text(0, 0, f"{d.od.sum():,}\nvalid OD", ha="center", va="center", fontsize=12)
            ax.set_title(f"{title}\nn={len(d):,}", pad=14)
        q = pd.read_csv(self.cache / "source_quality.csv")
        vals = np.array([q.rows-q.od, q.od-q.od_time, q.od_time]).T
        ax = fig.add_subplot(gs[1, :]); left = np.zeros(len(q))
        labels = ["No valid OD", "Valid OD, no order time", "Valid OD + order time"]
        for j, (label, c) in enumerate(zip(labels, colors)):
            widths = vals[:, j] / q.rows * 100
            ax.barh(np.arange(len(q)), widths, left=left, label=label, color=c)
            for k, w in enumerate(widths):
                if w >= 7:
                    ax.text(left[k]+w/2, k, f"{w:.1f}%", ha="center", va="center", fontsize=9)
            left += widths
        names = ["City-enriched (4 files)" if "city_extracts" in s else " / ".join(s.split("/")[1:]) for s in q.source]
        ax.set_yticks(np.arange(len(q)), [f"{s}   n={n:,}" for s, n in zip(names, q.rows)])
        ax.invert_yaxis(); ax.set(xlim=(0, 100), xlabel="Share of source records (%)")
        handles, legend_labels = ax.get_legend_handles_labels()
        fig.legend(handles, legend_labels, ncol=3, loc="outside upper center", frameon=False)
        ax.set_title("Source completeness (source records; overlaps retained here)")
        self.table("source_quality", q)
        self.save(fig, "02_validity", "Donuts: unique merged receipts. Bars: each source's own denominator.")

    def grids(self):
        info, max_count = {}, 1
        for r, (_, _, epsg) in REGIONS.items():
            transform = Transformer.from_crs(4326, epsg, always_xy=True)
            for year in (None, 2022):
                d = self.region(r, year, od=True)
                for end in ("pickup", "dropoff"):
                    x, y = transform.transform(d[end+"Lon"].to_numpy(), d[end+"Lat"].to_numpy())
                    xy = np.column_stack((np.floor(np.array(x)/1000).astype(int), np.floor(np.array(y)/1000).astype(int)))
                    bins, counts = np.unique(xy, axis=0, return_counts=True)
                    assert counts.sum() == len(d)
                    max_count = max(max_count, int(counts.max()) if len(counts) else 1)
                    info[r, year, end] = (bins, counts)
        return info, LogNorm(vmin=1, vmax=max(2, max_count))

    def maps(self):
        ctx = configure_maps(self.cache)
        from matplotlib.collections import PolyCollection
        grids, norm = self.grids()
        for year in (None, 2022):
            fig, axes = plt.subplots(4, 2, figsize=(14, 18), layout="constrained")
            for row, (r, (name, _, epsg)) in enumerate(REGIONS.items()):
                full = self.region(r, od=True)
                t = Transformer.from_crs(4326, epsg, always_xy=True)
                to_map = Transformer.from_crs(epsg, 3857, always_xy=True)
                coords = []
                for end in ("pickup", "dropoff"):
                    x, y = t.transform(full[end+"Lon"].to_numpy(), full[end+"Lat"].to_numpy())
                    coords.append(np.column_stack((x, y)))
                points = np.concatenate(coords)
                full_lo, full_hi = points.min(axis=0), points.max(axis=0)
                lo, hi = np.quantile(points, [.005, .995], axis=0)
                pad = np.maximum((hi-lo)*.05, 1500)
                xmin, ymin = to_map.transform(*(lo-pad)); xmax, ymax = to_map.transform(*(hi+pad))
                extent = (xmin, xmax, ymin, ymax)
                raster = self.cache / "maps" / f"{r}_esri_gray_focus.tif"
                if not raster.exists():
                    zoom = 11
                    while zoom > 4 and ctx.howmany(xmin, ymin, xmax, ymax, zoom, verbose=False) > 256:
                        zoom -= 1
                    ctx.bounds2raster(xmin, ymin, xmax, ymax, str(raster), zoom=zoom,
                                      source=ctx.providers.Esri.WorldGrayCanvas, n_connections=1,
                                      wait=1, max_retries=2)
                for col, end in enumerate(("pickup", "dropoff")):
                    ax = axes[row, col]
                    ax.set(xlim=extent[:2], ylim=extent[2:])
                    ctx.add_basemap(ax, source=str(raster), crs="EPSG:3857", attribution=False)
                    bins, counts = grids[r, year, end]
                    vertices = []
                    for bx, by in bins:
                        x = np.array([bx, bx+1, bx+1, bx])*1000
                        y = np.array([by, by, by+1, by+1])*1000
                        vx, vy = to_map.transform(x, y)
                        vertices.append(np.column_stack((vx, vy)))
                    collection = PolyCollection(vertices, array=counts.astype(float), cmap=self.cmap,
                                                norm=norm, linewidths=0, alpha=.85, rasterized=True)
                    ax.add_collection(collection)
                    ax.set(xlim=extent[:2], ylim=extent[2:])
                    period_frame = self.region(r, year, od=True)
                    n = len(period_frame)
                    px, py = t.transform(period_frame[end+"Lon"].to_numpy(),
                                         period_frame[end+"Lat"].to_numpy())
                    endpoint = np.column_stack((px, py))
                    in_view = ((endpoint >= lo-pad) & (endpoint <= hi+pad)).all(axis=1)
                    assert int(in_view.sum()) <= n
                    ax.set_title(f"{name} | {'Origins' if col == 0 else 'Destinations'} | n={n:,}")
                    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
                    locator = ax.inset_axes([.77, .71, .22, .25])
                    locator.scatter(endpoint[:, 0], endpoint[:, 1], s=.15, color=self.blue,
                                    alpha=.15, linewidths=0, rasterized=True)
                    locator.add_patch(Rectangle(lo-pad, *(hi-lo+2*pad), fill=False,
                                                edgecolor=color("base.coral"), linewidth=1))
                    locator.set(xlim=(full_lo[0]-pad[0], full_hi[0]+pad[0]),
                                ylim=(full_lo[1]-pad[1], full_hi[1]+pad[1]), xticks=[], yticks=[])
                    locator.set_title("Full extent", fontsize=6, pad=2)
                    ax.text(.01, .96, f"Main view: {in_view.sum():,}/{n:,}", transform=ax.transAxes,
                            fontsize=8, va="top", bbox={"facecolor":"white", "alpha":.8, "edgecolor":"none"})
                    # Use local projected distances for a calibrated scale bar.
                    length = max(1000, 10**math.floor(math.log10(max(hi-lo)/5)))
                    sx, sy = lo[0], lo[1]
                    a, b = to_map.transform([sx, sx+length], [sy, sy])
                    ax.plot(a, b, color=color("ink.text"), lw=2)
                    ax.text(a[0], b[0], f" {length/1000:g} km", va="bottom", fontsize=8)
                    ax.text(.01, .01, "Tiles © Esri", transform=ax.transAxes,
                            fontsize=6, bbox={"facecolor":"white", "alpha":.8, "edgecolor":"none"})
                    self.table(f"grid_{r}_{year or 'full'}_{end}", pd.DataFrame({
                        "utm_x_min_m": bins[:, 0]*1000, "utm_y_min_m": bins[:, 1]*1000,
                        "samples_per_km2": counts, "epsg": epsg}))
            cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=self.cmap), ax=axes.ravel().tolist(),
                              location="bottom", shrink=.6, aspect=45, pad=.02)
            cb.set_label("Receipt samples per 1 km² cell (logarithmic color scale)")
            fig.suptitle("Endpoint sample density | " + ("Full history" if year is None else "2022 receipt-year cohort"), fontsize=17)
            self.save(fig, "03_density_full" if year is None else "04_density_2022",
                      "Central 99% + padding; full-extent insets. " + ctx.providers.Esri.WorldGrayCanvas.attribution)

    def monthly(self):
        fig, axes = plt.subplots(4, 2, figsize=(14, 13), layout="constrained")
        records = []
        for i, (r, (name, tz, _)) in enumerate(REGIONS.items()):
            d = self.region(r, od=True)
            for j, (field, label) in enumerate((("receipt_time", "Email receipt month"), ("order_time", "Actual order month"))):
                ax = axes[i, j]
                series = d[field].dropna()
                local = (order_wall_time(series) if field == "order_time"
                         else series.dt.tz_convert(tz).dt.tz_localize(None))
                if local.empty:
                    ax.text(.5, .5, "No actual order timestamps", ha="center", transform=ax.transAxes)
                else:
                    counts = local.dt.to_period("M").value_counts().sort_index()
                    months = pd.period_range(counts.index.min(), counts.index.max(), freq="M")
                    counts = counts.reindex(months, fill_value=0)
                    dates = months.to_timestamp()
                    bars = ax.bar(dates, counts, width=24, color=self.blue if j == 0 else self.teal)
                    bars[0].set_hatch("//"); bars[-1].set_hatch("//")
                    for month, n in counts.items():
                        records.append({"region":name, "time_basis":label, "month":str(month), "count":n})
                    ax.set_xlim(pd.Timestamp("2018-01-01"), pd.Timestamp("2024-05-01"))
                    ax.yaxis.set_major_formatter(FuncFormatter(compact))
                ax.set_title(f"{name} | {label} | n={len(local):,}")
                ax.set_ylabel("Valid OD samples"); ax.tick_params(axis="x", rotation=30)
        self.table("monthly_counts", pd.DataFrame(records))
        self.save(fig, "05_monthly_coverage", "Local time. First/last observed months hatched (partial coverage possible).")

    def rhythms(self):
        for year in (None, 2022):
            fig, axes = plt.subplots(4, 2, figsize=(14, 13), layout="constrained")
            records = []
            for i, (r, (name, tz, _)) in enumerate(REGIONS.items()):
                d = self.region(r, year, od=True)
                local = order_wall_time(d.order_time.dropna())
                hour = local.dt.hour.value_counts().reindex(range(24), fill_value=0)
                pct = hour.to_numpy()/max(len(local), 1)*100
                axes[i, 0].bar(range(24), pct, color=self.teal, width=.9)
                axes[i, 0].set(xlim=(-.5, 23.5), xticks=[0, 6, 12, 18, 23], xlabel="Local order hour",
                               ylabel="Share of timed OD samples (%)", title=f"{name} | n={len(local):,}")
                heat = np.zeros((7, 24), dtype=int)
                np.add.at(heat, (local.dt.dayofweek.to_numpy(), local.dt.hour.to_numpy()), 1)
                assert heat.sum() == len(local)
                image = axes[i, 1].imshow(heat/max(len(local), 1)*100, aspect="auto", cmap=self.cmap, vmin=0)
                axes[i, 1].set(yticks=range(7), yticklabels=["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
                               xticks=[0, 6, 12, 18, 23], xlabel="Local order hour", title="Weekday × hour")
                fig.colorbar(image, ax=axes[i, 1], label="Share (%)", shrink=.8)
                for weekday in range(7):
                    for h in range(24):
                        records.append({"region":name, "weekday":weekday, "hour":h, "count":heat[weekday, h]})
            name = "06_rhythms_full" if year is None else "06_rhythms_2022"
            fig.suptitle("Actual order-time distribution | " + ("Full history" if year is None else "2022 receipt cohort"), fontsize=16)
            self.table(name, pd.DataFrame(records))
            self.save(fig, name, "Positive orderTime only. Source local clock; no second timezone conversion or email substitution.")

    def distances(self):
        fig, axes = plt.subplots(2, 2, figsize=(13, 9), layout="constrained")
        records = []
        for ax, (r, (name, _, _)) in zip(axes.flat, REGIONS.items()):
            for year, c, label in ((None, self.blue, "Full history"), (2022, self.yellow, "2022 receipts")):
                values = np.sort(self.region(r, year, od=True).distance_km.to_numpy())
                if not len(values):
                    continue
                y = np.arange(1, len(values)+1)/len(values)*100
                median, p90 = np.quantile(values, [.5, .9])
                ax.plot(values, y, color=c, lw=1.8, label=f"{label}: n={len(values):,}; median={median:.2f}; P90={p90:.2f} km")
                records.append({"region":name, "period":label, "n":len(values), "median_km":median, "p90_km":p90,
                                "min_km":values[0], "max_km":values[-1]})
            ax.set(xscale="symlog", xlim=(0, None), ylim=(0, 101), title=name,
                   xlabel="Endpoint geodesic distance (km; symlog axis)", ylabel="Cumulative share (%)")
            ax.legend(loc="lower right", fontsize=8, frameon=False)
            ax.grid(alpha=.2)
        self.table("distance_summary", pd.DataFrame(records))
        self.save(fig, "07_distances", "All valid OD distances retained, including zero and long-distance tails. No road routing.")

    def platforms(self):
        fig, axes = plt.subplots(2, 2, figsize=(14, 10), layout="constrained")
        records = []
        for row, year in enumerate((None, 2022)):
            table = np.zeros((4, 6), dtype=int)
            for i, r in enumerate(REGIONS):
                d = self.region(r, year, od=True)
                table[i] = d.platform.value_counts().reindex(PLATFORMS, fill_value=0)
                for p, n in zip(PLATFORMS, table[i]):
                    records.append({"region":REGIONS[r][0], "period":year or "Full", "platform":p, "valid_od":n})
            for j in range(2):
                ax = axes[row, j]; left = np.zeros(4)
                for k, (label, c) in enumerate(zip(LABELS, self.platform_colors)):
                    values = table[:, k] if j == 0 else table[:, k]/np.maximum(table.sum(axis=1), 1)*100
                    ax.barh(range(4), values, left=left, color=c, label=label)
                    left += values
                ax.set_yticks(range(4), [x[0] for x in REGIONS.values()]); ax.invert_yaxis()
                ax.set_title(("Full history" if year is None else "2022 receipt cohort") + (" | count" if j == 0 else " | composition"))
                ax.set_xlabel("Valid OD samples" if j == 0 else "Share (%)")
                if j == 0:
                    ax.xaxis.set_major_formatter(FuncFormatter(compact))
                else:
                    ax.set_xlim(0, 100)
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="outside upper center", ncol=6, frameon=False)
        self.table("platform_counts", pd.DataFrame(records))
        self.save(fig, "08_platforms", "Composition of valid OD samples, not platform market shares.")

    def overview(self):
        fig, axes = plt.subplots(3, 3, figsize=(18, 17), layout="constrained")
        for ax, name in zip(axes.flat, self.files):
            ax.imshow(plt.imread(self.output / name)); ax.axis("off")
            ax.set_title(name.removesuffix(".png").replace("_", " "), fontsize=10)
        fig.suptitle("New food-delivery sample atlas", fontsize=20)
        fig.savefig(self.output / "00_overview.png", dpi=180, facecolor="white")
        plt.close(fig)

    def finish(self, prepared):
        self.table("region_counts", pd.DataFrame(self.summaries))
        report = {"created_utc": datetime.now(timezone.utc).isoformat(), "prepared":prepared,
                  "time_semantics": "receivedDate: UTC epoch converted to region timezone. orderTime: source local-clock epoch encoding, decoded without offset conversion. Grubhub dictionary says Local Time; Foodpanda interpretation supported by ~-7.4h email-minus-encoded-order difference.",
                  "pngs":["00_overview.png", *self.files], "source_tables":self.tables,
                  "palette_sha256":hashlib.sha256((ROOT / "src/drone_emission/visualization/palette.toml").read_bytes()).hexdigest(),
                  "notes":self.notes}
        (self.output / "report.json").write_text(json.dumps(report, indent=2))
        readme = """# New food-delivery sample atlas

Nine thematic figures and one contact sheet. PNG previews, PDF/SVG exports and
aggregate source-data tables are generated together. Counts refer to unique
platform/account/receipt keys, not verified independent orders. Maps use valid OD
pairs only. Original and city-enriched sources are merged with valid enriched
coordinates preferred; New York zone precedes city; remaining ties are stable.
Source bars in figure 02 retain overlapping source records intentionally.

Regional membership follows supplied city files (including their surrounding
zones) and Hong Kong country flags. No administrative clipping. 2022 selects the
local receipt year. Monthly receipt-time coverage and source order-time coverage
are separate. Hour/weekday plots never substitute email timestamps for orders.
Source orderTime encodes local wall-clock values as epoch numbers: Grubhub's
dictionary explicitly says Local Time. Foodpanda shows the same convention
(email minus encoded order time median -7.43 hours; accounting for HK UTC+8 gives
about 34 minutes from order to email). These values are decoded without a second
timezone conversion. Receipt timestamps use normal UTC-to-local conversion.
Coordinates are treated as WGS84; validity is numeric/range/non-(0,0), not a claim
of positional accuracy. Grid cells are 1 km² in local UTM projections, with shared
log color normalization across endpoint maps. Distances are geodesic endpoint
distances, not roads or courier trajectories. No market expansion is applied.

Basemaps: Esri World Gray Canvas (Tiles © Esri and its named data contributors).
World outlines: Natural Earth. Full provider attribution appears on map figures.
Main map windows use central 99% of full-history endpoints plus padding; locator
insets show every valid endpoint across the full source extent. No rows are
removed from counts, grid tables, distance curves or sample statistics.
First/last observed months are hatched because coverage may be incomplete.
Merged counts can differ from rowwise source summaries because timestamps are
filled from another record of the same receipt. Missing order times remain missing.
"""
        (self.output / "README.md").write_text(readme)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "render", "all"])
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.action in ("prepare", "all"):
        prepare(args.input, args.cache)
    if args.action in ("render", "all"):
        prepared = json.loads((args.cache / "prepared.json").read_text())
        if prepared["input_signature"] != signature(args.input):
            raise ValueError("Prepared view is stale; run prepare or all")
        args.output.mkdir(parents=True, exist_ok=False)
        setup()
        atlas = Atlas(pd.read_pickle(args.cache / "sample_view.pkl"), args.cache, args.output)
        atlas.global_map()
        atlas.quality()
        atlas.maps()
        atlas.monthly()
        atlas.rhythms()
        atlas.distances()
        atlas.platforms()
        atlas.overview()
        atlas.finish(prepared)


if __name__ == "__main__":
    main()
