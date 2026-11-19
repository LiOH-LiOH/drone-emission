"""Add an at-a-glance coverage and complete-OD summary to the sample atlas."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import explore
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    df = pd.read_pickle(args.cache / "sample_view.pkl")
    entries = [(label, df.loc[df[key]]) for key, (label, _, _) in explore.REGIONS.items()]
    entries.append(("Other US locations", df.loc[df.countryCode.eq("US") & ~df[["ny", "la", "sf"]].any(axis=1)]))
    entries.append(("All data (deduplicated)", df))
    rows = []
    for label, frame in entries:
        dates = frame.receipt_time.dropna()
        rows.append({"region": label, "receipt_start_utc": dates.min().strftime("%Y-%m-%d"),
                     "receipt_end_utc": dates.max().strftime("%Y-%m-%d"),
                     "total_receipts": len(frame), "complete_od": int(frame.od.sum()),
                     "complete_od_pct": 100 * frame.od.mean()})
    result = pd.DataFrame(rows)
    assert result.iloc[:-1].total_receipts.sum() == len(df)
    assert result.iloc[:-1].complete_od.sum() == int(df.od.sum())
    args.output.mkdir(parents=True, exist_ok=False)
    result.to_csv(args.output / "coverage_summary.csv", index=False)

    # The raw city text gives the named coverage beyond the four focus regions.
    # Use the original US tables once, not their overlapping city-enriched copies.
    manifest = json.loads((args.input / "manifest.json").read_text())
    names = []
    for entry in manifest["entries"]:
        rel = entry["canonical"]
        if entry["duplicate"] or not rel.startswith("polyu_") or not rel.endswith(".csv"):
            continue
        city = pd.read_csv(args.input / rel, usecols=["city"], dtype=str)["city"]
        names.append(city.fillna("").str.strip().str.replace(r"\s+", " ", regex=True))
    city = pd.concat(names, ignore_index=True)
    inventory = city[city.ne("")].value_counts().rename_axis("source_city_label").reset_index(name="raw_us_receipts")
    inventory.to_csv(args.output / "reported_us_city_labels.csv", index=False)
    explore.setup()
    blue, teal, gray = (explore.color(c) for c in ("base.blue", "base.green", "base.gray"))
    fig = plt.figure(figsize=(15, 10), layout="constrained")
    gs = fig.add_gridspec(2, 2, height_ratios=[1.1, 1])
    ax = fig.add_subplot(gs[0, :]); ax.axis("off")
    table = ax.table(cellText=[[
        row.region, f"{row.receipt_start_utc} to {row.receipt_end_utc}",
        f"{row.total_receipts:,}", f"{row.complete_od:,}", f"{row.complete_od_pct:.2f}%"
    ] for row in result.itertuples()],
        colLabels=["Coverage", "Receipt date span (UTC)", "Total receipts", "Complete OD", "OD share"],
        colWidths=[.24, .32, .15, .15, .14], cellLoc="center", loc="center", bbox=[0, .05, 1, .8])
    table.auto_set_font_size(False); table.set_fontsize(11)
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor("white")
        if r == 0:
            cell.set_facecolor(blue); cell.set_text_props(color="white", weight="bold")
        elif r == len(result):
            cell.set_facecolor(explore.color("derived.blue_light")); cell.set_text_props(weight="bold")
        else:
            cell.set_facecolor(explore.color("neutral.light") if r % 2 else "white")
    ax.set_title("Coverage, date span and usable origin–destination pairs", fontsize=19, pad=18)
    left = fig.add_subplot(gs[1, 0])
    q = result.iloc[:-1]
    left.barh(range(len(q)), q.complete_od, color=teal, label="Complete OD")
    left.barh(range(len(q)), q.total_receipts-q.complete_od, left=q.complete_od, color=gray, label="Other records")
    left.set_yticks(range(len(q)), q.region); left.invert_yaxis()
    left.xaxis.set_major_formatter(explore.FuncFormatter(explore.compact))
    left.set(xlabel="Unique receipt samples", title="Complete coordinate pairs within each coverage group")
    left.legend(frameon=False, loc="lower right")
    right = fig.add_subplot(gs[1, 1])
    top = inventory.head(12).iloc[::-1]
    right.barh(top.source_city_label, top.raw_us_receipts, color=blue)
    right.xaxis.set_major_formatter(explore.FuncFormatter(explore.compact))
    right.set(xlabel="Original US receipt samples (before city enrichment)",
              title="Named US coverage: top 12 source city labels")
    fig.text(.015, .035, "Four focus regions are not the entire geographic coverage. Full raw city-label inventory accompanies this figure.", fontsize=9)
    fig.text(.015, .015, "Complete OD: four valid numeric coordinates, neither endpoint (0,0). Dates are email receipt dates, not order-time coverage.", fontsize=9)
    fig.get_layout_engine().set(rect=(0, .07, 1, .92))
    for extension in ("png", "pdf", "svg"):
        fig.savefig(args.output / f"09_coverage_summary.{extension}", dpi=300, facecolor="white")
    plt.close(fig)
    (args.output / "README.md").write_text(
        "# Coverage summary\n\n[Summary figure](09_coverage_summary.png)\n\n"
        "[Exact region counts and UTC receipt-date spans](coverage_summary.csv)\n\n"
        "[All reported US city labels](reported_us_city_labels.csv)\n\n"
        "The original city field is a raw text label, not a reconciled administrative city inventory. "
        "New York and Los Angeles groups retain the supplied region coverage; San Francisco includes the Bay Area. "
        "Counts are unique platform/account/receipt keys; Hong Kong includes distinct receipt and placement events.\n")
    print(result.to_string(index=False), flush=True)
    print(f"Raw US city labels: {len(inventory):,}; unspecified: {city.eq('').sum():,}", flush=True)


if __name__ == "__main__":
    main()
