"""Display-only export rules, independent of prepared scientific inputs."""

from contextlib import contextmanager
from hashlib import sha256
from pathlib import Path
import json


def display_signature():
    return sha256(Path(__file__).read_bytes()).hexdigest()


def apply_display_style(figure, output):
    from matplotlib.axes import Axes
    from matplotlib.container import BarContainer

    changes = []
    for axes in figure.findobj(Axes):
        if axes.name == "3d":
            continue
        axes.grid(False, which="both", axis="both")
        if Path(output).name == "province_total_ebike_lifecycle.pdf":
            if axes.yaxis.get_label_position() == "right":
                axes.set_ylim(bottom=0)
                changes.append({"rule": "right_axis_zero", "ylim": list(axes.get_ylim())})
        if Path(output).name == "city_distance_drone_share_emission_fit.pdf":
            bars = [patch for container in axes.containers
                    if isinstance(container, BarContainer) for patch in container.patches]
            if bars:
                left, right = axes.get_xlim()
                left = min(left, min(bar.get_x() for bar in bars))
                right = max(right, max(bar.get_x() + bar.get_width() for bar in bars))
                # Keep the edge stroke inside the viewport as well as the bar fill.
                pad = (right - left) * 0.003
                axes.set_xlim(left - pad, right + pad)
                changes.append({"rule": "complete_bar_width", "xlim": list(axes.get_xlim())})
    return {"output": Path(output).name, "two_dimensional_grid": False,
            "display_sha256": display_signature(), "adjustments": changes}


@contextmanager
def paper_display_style(report_dir):
    from matplotlib.figure import Figure

    original = Figure.savefig
    report_dir = Path(report_dir)

    def savefig(figure, filename, *args, **kwargs):
        # Idempotent when a renderer exports the same Figure more than once.
        if not getattr(figure, "_paper_display_applied", False):
            figure._paper_display_report = apply_display_style(figure, filename)
            figure._paper_display_applied = True
        result = original(figure, filename, *args, **kwargs)
        report_dir.mkdir(parents=True, exist_ok=True)
        (report_dir / (Path(filename).stem + ".json")).write_text(
            json.dumps(figure._paper_display_report, indent=2) + "\n")
        return result

    Figure.savefig = savefig
    try:
        yield
    finally:
        Figure.savefig = original
