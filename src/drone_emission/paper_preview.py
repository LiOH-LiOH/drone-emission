"""Manifest-ordered review previews, produced only by formal composition."""

import hashlib
import json
from pathlib import Path
import subprocess
import tempfile


def invalidate_preview(figures_root, reason):
    directory = Path(figures_root) / "preview"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "status.json").write_text(json.dumps({"status": "stale", "reason": reason}, indent=2) + "\n")
    overview = directory / "palette_overview.png"
    if overview.exists():
        overview.replace(directory / "palette_overview.stale.png")


def refresh_previews(figures_root, figures, *, rasterizer):
    from PIL import Image, ImageDraw, ImageFont, ImageOps

    root = Path(figures_root)
    directory = root / "preview"
    directory.mkdir(parents=True, exist_ok=True)
    source_pdfs = [root / "paper" / Path(item["staging"]).name for item in figures]
    for path in source_pdfs:
        if not path.is_file() or not path.read_bytes().startswith(b"%PDF-"):
            raise FileNotFoundError(f"Cannot refresh complete overview: {path}")
    font = None
    for name in ("/System/Library/Fonts/Supplemental/Arial.ttf", "Arial.ttf", "DejaVuSans.ttf"):
        try:
            font = ImageFont.truetype(name, 30)
            break
        except OSError:
            pass
    if font is None:
        font = ImageFont.load_default()
    columns, tile = 3, 2000
    rows = (len(figures) + columns - 1) // columns
    overview = Image.new("RGB", (columns * tile, rows * tile), "white")
    draw = ImageDraw.Draw(overview)
    sources = []
    with tempfile.TemporaryDirectory(prefix=".preview-", dir=directory) as temporary:
        temporary = Path(temporary)
        for i, (figure, pdf) in enumerate(zip(figures, source_pdfs)):
            prefix = temporary / figure["id"]
            subprocess.run([str(rasterizer), "-singlefile", "-f", "1", "-scale-to", "2400", "-png", str(pdf), str(prefix)], check=True)
            x, y = (i % columns) * tile, (i // columns) * tile
            with Image.open(prefix.with_suffix(".png")) as preview:
                thumbnail = ImageOps.contain(preview.convert("RGB"), (1900, 1830), Image.Resampling.LANCZOS)
                overview.paste(thumbnail, (x + (tile - thumbnail.width) // 2, y + 110 + (1830 - thumbnail.height) // 2))
            draw.text((x + 50, y + 40), f"{i + 1:02d}  {figure['id']}", fill="#333333", font=font)
            draw.rectangle((x + 20, y + 20, x + tile - 20, y + tile - 20), outline="#dddddd", width=2)
            sources.append({"id": figure["id"], "pdf": str(pdf), "sha256": hashlib.sha256(pdf.read_bytes()).hexdigest()})
        overview.save(temporary / "palette_overview.png", dpi=(300, 300))
        for path in temporary.glob("*.png"):
            path.replace(directory / path.name)
    (directory / "status.json").write_text(json.dumps({"status": "current", "sources": sources}, indent=2) + "\n")
    print(f"Updated {len(figures)} previews and palette overview: {directory / 'palette_overview.png'}")
