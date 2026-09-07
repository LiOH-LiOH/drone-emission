"""Apply the paper typography contract to the complete figure at export.

The final artist pass covers explicit font sizes as well as rcParams, including
inset axes and legends. Scientific inputs never depend on this configuration.
"""

from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import re

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

DEFAULT_TYPOGRAPHY = Path(__file__).with_name("typography.toml")


def load_typography():
    document = tomllib.loads(DEFAULT_TYPOGRAPHY.read_text())
    for role in ("axis_label", "tick", "legend", "annotation", "secondary", "panel", "title"):
        value = document["sizes"][role]
        if not isinstance(value, (int, float)) or not 0 < value <= 72:
            raise ValueError(f"Invalid typography size: {role}")
    for rule in document.get("exceptions", []):
        if not rule.get("reason") or not 0 < rule["size"] <= 72:
            raise ValueError("Typography exceptions require a size and reason")
        re.compile(rule["text_regex"])
    return document


def typography_signature():
    return hashlib.sha256(DEFAULT_TYPOGRAPHY.read_bytes()).hexdigest()


def _roles(figure):
    roles = {}
    for axes in figure.axes:
        for axis in (axes.xaxis, axes.yaxis, getattr(axes, "zaxis", None)):
            if axis is None:
                continue
            roles[axis.label] = "axis_label"
            roles[axis.get_offset_text()] = "tick"
            for tick in axis.get_major_ticks() + axis.get_minor_ticks():
                roles[tick.label1] = roles[tick.label2] = "tick"
        for title in (axes.title, axes._left_title, axes._right_title):
            roles[title] = "title"
    from matplotlib.legend import Legend
    for legend in figure.findobj(Legend):
        roles[legend.get_title()] = "legend"
        roles.update({text: "legend" for text in legend.get_texts()})
    return roles


def apply_typography(figure, output, config=None):
    from matplotlib.text import Text

    config = config or load_typography()
    for layout in config.get("layouts", []):
        if layout["output"] == Path(output).name:
            axes = figure.axes[layout["axes"]]
            axes.set_position(layout["position"])
            for name in ("x", "y"):
                if name + "_labelpad" in layout:
                    getattr(axes, name + "axis").labelpad = layout[name + "_labelpad"]
    roles = _roles(figure)
    records = []
    for text in figure.findobj(Text):
        label = text.get_text()
        if not text.get_visible() or not label.strip():
            continue
        original = getattr(text, "_paper_original_size", text.get_fontsize())
        role = roles.get(text, "annotation")
        if text not in roles:
            if re.fullmatch(r"\(?[a-z]\)?", label) and text.get_fontweight() in ("bold", "heavy", 700):
                role = "panel"
            elif original < 7 or "©" in label or "OpenStreetMap" in label:
                role = "secondary"
        text._paper_original_size = original
        size = config["sizes"][role]
        reason = None
        for rule in config.get("exceptions", []):
            if rule["output"] == Path(output).name and rule["role"] == role and re.search(rule["text_regex"], label):
                size, reason = rule["size"], rule["reason"]
                break
        text.set_fontfamily(config["family"])
        text.set_fontsize(size)
        for adjustment in config.get("text_adjustments", []):
            if adjustment["output"] == Path(output).name and re.search(adjustment["text_regex"], label):
                if "position" in adjustment:
                    text.set_position(adjustment["position"])
                if "rotation" in adjustment:
                    text.set_rotation(adjustment["rotation"])
                if "ha" in adjustment:
                    text.set_ha(adjustment["ha"])
        records.append({"text": label, "role": role, "size_pt": size,
                        "original_size_pt": original, "exception_reason": reason})
    return {"output": Path(output).name, "family": config["family"],
            "typography_sha256": typography_signature(), "texts": records}


@contextmanager
def paper_typography(report_dir):
    """Scope export normalization to a formal render worker, never global users."""
    import matplotlib as mpl
    from matplotlib.figure import Figure

    config = load_typography()
    original = Figure.savefig
    report_dir = Path(report_dir)

    def savefig(figure, filename, *args, **kwargs):
        figure.canvas.draw()
        report = apply_typography(figure, filename, config)
        figure.canvas.draw()
        report["figure_size_inches"] = figure.get_size_inches().tolist()
        report["axes_positions"] = [list(axes.get_position().bounds) for axes in figure.axes]
        result = original(figure, filename, *args, **kwargs)
        report_dir.mkdir(parents=True, exist_ok=True)
        target = report_dir / (Path(filename).stem + ".json")
        target.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        return result

    with mpl.rc_context({"font.family": config["family"], "mathtext.fontset": "custom",
                         "mathtext.rm": config["family"], "mathtext.it": config["family"] + ":italic",
                         "mathtext.bf": config["family"] + ":bold"}):
        Figure.savefig = savefig
        try:
            yield
        finally:
            Figure.savefig = original
