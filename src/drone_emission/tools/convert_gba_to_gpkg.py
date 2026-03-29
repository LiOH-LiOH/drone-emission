"""Convert GBA GeoJSON tiles to GeoPackage for efficient spatial queries.

GeoJSON files must be fully parsed as text, making bbox-filtered reads of
multi-GB tiles extremely slow and I/O-heavy.  GeoPackage stores a spatial
index (R-tree) so that ``gpd.read_file(path, bbox=...)`` only touches the
relevant rows — reducing I/O from gigabytes to megabytes.

Usage (run from src/drone_emission/):
    python -m tools.convert_gba_to_gpkg /path/to/gba_data/LoD1_GeoJSON

Converts every .geojson under the given directory to a .gpkg sitting next
to it.  Already-converted tiles (where the .gpkg exists and is newer than
the .geojson) are skipped.
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path


def convert_one(geojson_path: Path) -> bool:
    """Convert a single GeoJSON to GeoPackage. Returns True on success."""
    gpkg_path = geojson_path.with_suffix(".gpkg")

    if gpkg_path.exists() and gpkg_path.stat().st_mtime > geojson_path.stat().st_mtime:
        print(f"  Skip (already converted): {geojson_path.name}")
        return True

    size_gb = geojson_path.stat().st_size / (1024**3)
    print(f"  Converting {geojson_path.name} ({size_gb:.1f} GB) ...")

    tmp_path = gpkg_path.with_suffix(".gpkg.tmp")
    t0 = time.time()

    # Stream features via fiona so we never hold the entire file in memory.
    import fiona

    with fiona.open(str(geojson_path)) as src:
        schema = src.schema.copy()
        crs = src.crs
        n = 0
        with fiona.open(
            str(tmp_path), "w", driver="GPKG", schema=schema, crs=crs
        ) as dst:
            for feat in src:
                dst.write(feat)
                n += 1
                if n % 500_000 == 0:
                    print(f"    ... {n:,} features")

    elapsed = time.time() - t0

    # Atomic rename
    tmp_path.rename(gpkg_path)
    gpkg_gb = gpkg_path.stat().st_size / (1024**3)
    print(f"    Done: {gpkg_gb:.1f} GB in {elapsed:.0f}s")
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("gba_dir", help="Root of LoD1_GeoJSON directory")
    parser.add_argument(
        "--file", help="Convert only this specific .geojson file (basename)"
    )
    args = parser.parse_args()

    gba_dir = Path(args.gba_dir)
    if not gba_dir.is_dir():
        print(f"Error: {gba_dir} is not a directory", file=sys.stderr)
        sys.exit(1)

    if args.file:
        targets = list(gba_dir.rglob(args.file))
        if not targets:
            print(f"Error: {args.file} not found under {gba_dir}", file=sys.stderr)
            sys.exit(1)
    else:
        targets = sorted(gba_dir.rglob("*.geojson"))

    print(f"Found {len(targets)} GeoJSON file(s) to convert")

    ok = 0
    fail = 0
    for path in targets:
        try:
            if convert_one(path):
                ok += 1
        except Exception as e:
            print(f"    FAILED: {e}", file=sys.stderr)
            fail += 1

    print(f"\nDone: {ok} converted, {fail} failed")


if __name__ == "__main__":
    main()
