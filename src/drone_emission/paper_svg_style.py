"""Physical typography and unscaled linked panels in composition copies."""

from pathlib import Path
import json
import re
from urllib.parse import unquote, urlparse
from xml.etree import ElementTree

from .visualization.typography import load_typography

COMPOSITION_STYLE = Path(__file__).with_name("paper_composition.json")


def length_pt(value):
    match = re.fullmatch(r"\s*([\d.eE+\-]+)\s*(mm|cm|in|pt|px)?\s*", value)
    if not match:
        raise ValueError(f"Unsupported SVG physical length: {value}")
    return float(match[1]) * {"mm": 72 / 25.4, "cm": 72 / 2.54, "in": 72, "pt": 1, "px": .75, None: .75}[match[2]]


def style_master(root, linked_dir, names, *, output=""):
    if not root.get("viewBox") or not root.get("width"):
        return {"texts": [], "panels": []}  # Minimal test fixtures.
    config = load_typography()
    composition = json.loads(COMPOSITION_STYLE.read_text())
    box = [float(v) for v in root.get("viewBox").replace(",", " ").split()]
    unit_pt = length_pt(root.get("width")) / box[2]
    report = {"texts": [], "panels": []}
    rectangles = [(box[0], box[1], box[0] + box[2], box[1] + box[3])]

    def visit(element, scale=1.0, tx=0.0, ty=0.0):
        for operation, args in re.findall(r"([a-zA-Z]+)\(([^)]*)\)", element.get("transform", "")):
            values = [float(v) for v in re.split(r"[,\s]+", args.strip())]
            if operation == "scale":
                if len(values) > 1 and abs(values[0] - values[1]) > 1e-6:
                    raise ValueError("Nonuniform scaling is not allowed for paper text/panels")
                scale *= abs(values[0])
            elif operation == "translate":
                tx += values[0] * scale
                ty += (values[1] if len(values) > 1 else 0) * scale
            else:
                raise ValueError(f"Unsupported master transform: {operation}")
        kind = element.tag.rsplit("}", 1)[-1]
        if kind == "text":
            label = "".join(element.itertext()).strip()
            if label:
                role = "panel" if re.fullmatch(r"\(?[a-z]\)?", label) else "annotation"
                size = config["sizes"][role]
                reason = None
                for rule in config.get("exceptions", []):
                    if rule["output"] == output and rule["role"] == role and re.search(rule["text_regex"], label):
                        size, reason = rule["size"], rule["reason"]
                        break
                for adjustment in composition.get("texts", []):
                    if adjustment["output"] == output and re.search(adjustment["text_regex"], label):
                        dx = adjustment.get("dx_mm", 0) * 72 / 25.4 / (unit_pt * scale)
                        for item in element.iter():
                            if item.get("x"):
                                item.set("x", " ".join(str(float(x) + dx) for x in re.split(r"[,\s]+", item.get("x").strip())))
                for item in element.iter():
                    if item.tag.rsplit("}", 1)[-1] not in ("text", "tspan"):
                        continue
                    style = dict(part.split(":", 1) for part in item.get("style", "").split(";") if ":" in part)
                    style.update({"font-size": f"{size / (unit_pt * scale):.8f}px", "font-family": config["family"]})
                    item.set("style", ";".join(f"{key}:{value}" for key, value in style.items()))
                report["texts"].append({"text": label, "role": role, "size_pt": size, "exception_reason": reason})
        if kind == "image":
            href = next((v for k, v in element.attrib.items() if k == "href" or k.endswith("}href")), "")
            name = Path(unquote(urlparse(href).path)).name
            if name in names:
                for adjustment in composition.get("panels", []):
                    if adjustment["output"] == output and adjustment["panel"] == name:
                        dx = adjustment.get("dx_mm", 0) * 72 / 25.4 / (unit_pt * scale)
                        element.set("x", str(float(element.get("x", "0")) + dx))
                panel = ElementTree.parse(Path(linked_dir) / name).getroot()
                if panel.get("width") and panel.get("height"):
                    width, height = length_pt(panel.get("width")), length_pt(panel.get("height"))
                    old_width, old_height = float(element.get("width")), float(element.get("height"))
                    element.set("width", str(width / (unit_pt * scale)))
                    element.set("height", str(height / (unit_pt * scale)))
                    element.set("preserveAspectRatio", "xMinYMin meet")
                    x = float(element.get("x", "0")) * scale + tx
                    y = float(element.get("y", "0")) * scale + ty
                    rectangles.append((x, y, x + width / unit_pt, y + height / unit_pt))
                    report["panels"].append({"panel": name, "source_width_pt": width, "source_height_pt": height,
                                              "previous_scale_x": old_width * unit_pt * scale / width,
                                              "previous_scale_y": old_height * unit_pt * scale / height,
                                              "export_scale": 1.0})
        for child in element:
            visit(child, scale, tx, ty)

    visit(root)
    # Inkscape's drawing bounds include invisible/out-of-clip imported objects.
    # Grow the actual page using the linked PDF viewports, never those objects.
    padding = (72 / 25.4) / unit_pt
    left = min(r[0] for r in rectangles) - padding
    top = min(r[1] for r in rectangles) - padding
    right = max(r[2] for r in rectangles) + padding
    bottom = max(r[3] for r in rectangles) + padding
    root.set("viewBox", f"{left} {top} {right - left} {bottom - top}")
    root.set("width", f"{(right - left) * unit_pt}pt")
    root.set("height", f"{(bottom - top) * unit_pt}pt")
    report["page_size_pt"] = [(right - left) * unit_pt, (bottom - top) * unit_pt]
    return report
