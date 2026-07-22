"""Build compact, writing-oriented reports for the five manuscript figures.

Run from ``src/drone_emission`` after all registered paper subfigures have
been rendered::

    ../../.venv/bin/python -m tools.paper_figure_reports

The command is intentionally strict: every registered PDF and raw report must
exist, the raw report must not predate its PDF, and every final report must fit
within the 128 KiB writing-context budget.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import statistics
import tempfile
from typing import Any


MAX_REPORT_BYTES = 128 * 1024
MAX_INLINE_LIST_ITEMS = 48
PROJECT_ROOT = Path(__file__).resolve().parents[3]
REGISTRY_PATH = (
    Path(__file__).resolve().parents[1]
    / "visualization"
    / "paper_report_registry.json"
)
FIGURES_DIR = PROJECT_ROOT / "figures"
REPORT_DIR = FIGURES_DIR / "paper" / "report"

_DROP_EXACT = {
    "axis",
    "axes",
    "coordinate_convention",
    "connector_lines",
    "export",
    "figure_size_inches",
    "layout",
    "layout_mode",
    "native_size_inches",
    "palette",
    "save_padding_inches",
}
_DROP_SUBSTRINGS = (
    "bbox_",
    "_figure_fraction",
    "label_offset",
    "pixel_",
)


class ReportBuildError(RuntimeError):
    """Raised when the strict five-figure report build cannot complete."""


def _project_path(path_text: str) -> Path:
    """Resolve a registry path relative to the repository root."""
    return PROJECT_ROOT / Path(path_text)


def _relative_path(path: Path) -> str:
    """Return a stable repository-relative path."""
    return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()


def _safe_number(value: Any) -> float | int | None:
    """Return a finite JSON number, or None for non-finite values."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    if not math.isfinite(number):
        return None
    return int(number) if number.is_integer() else number


def _quantile(sorted_values: list[float], fraction: float) -> float:
    """Return a linearly interpolated quantile for sorted values."""
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = (len(sorted_values) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    weight = position - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def _numeric_summary(values: list[Any]) -> dict[str, Any]:
    """Summarize a long numeric sequence without retaining every point."""
    clean = [float(value) for value in values if _safe_number(value) is not None]
    clean.sort()
    if not clean:
        return {"count": len(values), "finite_count": 0}
    return {
        "count": len(values),
        "finite_count": len(clean),
        "min": _safe_number(clean[0]),
        "q1": _safe_number(_quantile(clean, 0.25)),
        "median": _safe_number(statistics.median(clean)),
        "mean": _safe_number(statistics.fmean(clean)),
        "q3": _safe_number(_quantile(clean, 0.75)),
        "max": _safe_number(clean[-1]),
    }


def _record_list_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize a long record list by numeric and low-cardinality fields."""
    field_names = sorted({key for record in records for key in record})
    numeric_fields: dict[str, Any] = {}
    categorical_fields: dict[str, Any] = {}
    for key in field_names:
        values = [record.get(key) for record in records if record.get(key) is not None]
        if values and all(_safe_number(value) is not None for value in values):
            numeric_fields[key] = _numeric_summary(values)
            continue
        scalar_values = [
            value for value in values if isinstance(value, (str, bool, int, float))
        ]
        if len(scalar_values) != len(values):
            continue
        unique = sorted({str(value) for value in scalar_values})
        categorical_fields[key] = (
            {"unique_count": len(unique), "values": unique}
            if len(unique) <= 20
            else {"unique_count": len(unique)}
        )
    result: dict[str, Any] = {"record_count": len(records)}
    if numeric_fields:
        result["numeric_field_summaries"] = numeric_fields
    if categorical_fields:
        result["categorical_field_summaries"] = categorical_fields
    return result


def _drop_key(key: str) -> bool:
    """Return whether a raw-report field is irrelevant to paper writing."""
    lowered = key.lower()
    if lowered in _DROP_EXACT:
        return True
    if lowered.endswith("_axis") or lowered.startswith("axis_"):
        return True
    return any(token in lowered for token in _DROP_SUBSTRINGS)


def _normalise_string(value: str) -> str:
    """Remove machine-private prefixes while retaining useful source paths."""
    if not os.path.isabs(value):
        return value
    marker = "/drone-emission/"
    if marker in value:
        return value.split(marker, 1)[1]
    return Path(value).name


def _compact(value: Any) -> Any:
    """Recursively retain structured facts and remove layout-heavy detail."""
    if isinstance(value, dict):
        compacted = {
            key: _compact(item)
            for key, item in value.items()
            if not _drop_key(key)
        }
        return {key: item for key, item in compacted.items() if item is not None}
    if isinstance(value, list):
        if len(value) <= MAX_INLINE_LIST_ITEMS:
            return [_compact(item) for item in value]
        if all(_safe_number(item) is not None for item in value):
            return _numeric_summary(value)
        if all(isinstance(item, dict) for item in value):
            return _compact(_record_list_summary(value))
        scalar_values = [
            item for item in value if isinstance(item, (str, bool, int, float))
        ]
        unique = sorted({str(item) for item in scalar_values})
        result: dict[str, Any] = {"count": len(value)}
        if len(scalar_values) == len(value):
            result["unique_count"] = len(unique)
            if len(unique) <= 20:
                result["values"] = unique
        return result
    if isinstance(value, str):
        return _normalise_string(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _select_report(payload: dict[str, Any], selector: list[str] | None) -> Any:
    """Select a registered sub-tree from a shared raw report."""
    selected: Any = payload
    for key in selector or []:
        if not isinstance(selected, dict) or key not in selected:
            raise ReportBuildError(
                f"Report selector {'/'.join(selector or [])} is missing"
            )
        selected = selected[key]
    return selected


def _validate_raw_report(
    source_path: Path,
    report_path: Path,
    payload: dict[str, Any],
) -> None:
    """Validate existence, freshness, and declared output identity."""
    if not source_path.is_file() or source_path.stat().st_size == 0:
        raise ReportBuildError(f"Missing or empty source PDF: {_relative_path(source_path)}")
    if not report_path.is_file() or report_path.stat().st_size == 0:
        raise ReportBuildError(f"Missing or empty raw report: {_relative_path(report_path)}")
    if report_path.stat().st_mtime < source_path.stat().st_mtime:
        raise ReportBuildError(
            "Raw report predates source PDF: "
            f"{_relative_path(report_path)} < {_relative_path(source_path)}"
        )

    declared = payload.get("figure_path")
    if isinstance(declared, str) and Path(declared).name != source_path.name:
        raise ReportBuildError(
            f"Raw report {report_path.name} declares a different figure: {declared}"
        )
    declared_many = payload.get("figure_paths")
    if isinstance(declared_many, dict):
        declared_names = {
            Path(value).name for value in declared_many.values() if isinstance(value, str)
        }
        if source_path.name not in declared_names:
            raise ReportBuildError(
                f"Shared raw report {report_path.name} does not declare {source_path.name}"
            )


def _read_component(component: dict[str, Any]) -> tuple[dict[str, Any], str]:
    """Read, validate, select, and compact one component report."""
    source_path = _project_path(component["source_path"])
    report_path = _project_path(component["report_path"])
    try:
        payload = json.loads(report_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ReportBuildError(
            f"Missing raw report: {_relative_path(report_path)}"
        ) from exc
    except json.JSONDecodeError as exc:
        raise ReportBuildError(
            f"Invalid JSON raw report: {_relative_path(report_path)}: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise ReportBuildError(f"Raw report must be an object: {report_path}")
    _validate_raw_report(source_path, report_path, payload)
    selected = _select_report(payload, component.get("report_selector"))
    result = {
        "role": component["role"],
        "source_path": component["source_path"],
        "raw_report_path": component["report_path"],
        "structured_facts": _compact(selected),
    }
    return result, component["report_path"]


def _source_manifest(figure: dict[str, Any]) -> dict[str, Any]:
    """Return the report-directory source manifest derived from the registry."""
    def clean_component(component: dict[str, Any]) -> dict[str, Any]:
        return {
            "role": component["role"],
            "source_path": component["source_path"],
        }

    manifest = {
        "schema_version": 1,
        "figure_id": figure["figure_id"],
        "figure_path": figure["figure_path"],
        "composition_type": figure["composition_type"],
        "assembly_tool": figure["assembly_tool"],
        "assembly_notes": figure["assembly_notes"],
        "generator": figure.get("generator"),
        "panels": [
            {
                "panel_id": panel["panel_id"],
                "title": panel["title"],
                "components": [
                    clean_component(component) for component in panel["components"]
                ],
            }
            for panel in figure.get("panels", [])
        ],
        "components": [
            clean_component(component) for component in figure.get("components", [])
        ],
    }
    return manifest


def _build_figure_report(figure: dict[str, Any]) -> dict[str, Any]:
    """Build one complete writing-oriented figure report."""
    raw_reports: set[str] = set()
    panels = []
    covered_sources = 0
    for panel in figure.get("panels", []):
        components = []
        for component in panel["components"]:
            compact_component, raw_path = _read_component(component)
            components.append(compact_component)
            raw_reports.add(raw_path)
            covered_sources += 1
        panels.append(
            {
                "panel_id": panel["panel_id"],
                "title": panel["title"],
                "components": components,
            }
        )

    components = []
    for component in figure.get("components", []):
        compact_component, raw_path = _read_component(component)
        components.append(compact_component)
        raw_reports.add(raw_path)
        covered_sources += 1

    standalone_facts = None
    if figure["composition_type"] == "standalone":
        component = {
            "role": "Complete standalone figure",
            "source_path": figure["figure_path"],
            "report_path": figure["standalone_report_path"],
        }
        compact_component, raw_path = _read_component(component)
        standalone_facts = compact_component["structured_facts"]
        raw_reports.add(raw_path)
        covered_sources = 1

    expected_sources = sum(
        len(panel["components"]) for panel in figure.get("panels", [])
    ) + len(figure.get("components", []))
    if figure["composition_type"] == "standalone":
        expected_sources = 1

    return {
        "schema_version": 1,
        "figure_id": figure["figure_id"],
        "figure_path": figure["figure_path"],
        "composition_type": figure["composition_type"],
        "report_purpose": "Structured factual evidence for English manuscript writing",
        "coverage": {
            "expected_sources": expected_sources,
            "covered_sources": covered_sources,
            "complete": expected_sources == covered_sources,
        },
        "panels": panels,
        "components": components,
        "standalone_facts": standalone_facts,
        "provenance": {
            "registry_path": _relative_path(REGISTRY_PATH),
            "raw_report_paths": sorted(raw_reports),
            "content_policy": (
                "Structured facts only; layout coordinates, styling detail, and "
                "long point arrays are omitted or summarized."
            ),
        },
    }


def _json_bytes(payload: dict[str, Any]) -> bytes:
    """Serialize deterministic, readable strict JSON."""
    return (
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=False,
        )
        + "\n"
    ).encode("utf-8")


def _atomic_batch_write(outputs: dict[Path, bytes]) -> None:
    """Prepare every output before replacing any existing report."""
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    temporary_paths: dict[Path, Path] = {}
    try:
        for destination, data in outputs.items():
            with tempfile.NamedTemporaryFile(
                mode="wb",
                dir=destination.parent,
                prefix=f".{destination.name}.",
                suffix=".tmp",
                delete=False,
            ) as temporary:
                temporary.write(data)
                temporary.flush()
                os.fsync(temporary.fileno())
                temporary_paths[destination] = Path(temporary.name)
        for destination, temporary_path in temporary_paths.items():
            os.replace(temporary_path, destination)
    finally:
        for temporary_path in temporary_paths.values():
            if temporary_path.exists():
                temporary_path.unlink()


def main() -> None:
    """Validate all sources and atomically write five reports and manifests."""
    try:
        registry = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"Cannot read paper report registry: {exc}") from exc

    figures = registry.get("figures")
    if not isinstance(figures, list) or len(figures) != 5:
        raise SystemExit("Paper report registry must contain exactly five figures")

    outputs: dict[Path, bytes] = {}
    try:
        for figure in figures:
            report_payload = _build_figure_report(figure)
            if not report_payload["coverage"]["complete"]:
                raise ReportBuildError(
                    f"Incomplete coverage for {figure['figure_id']}"
                )
            report_bytes = _json_bytes(report_payload)
            if len(report_bytes) > MAX_REPORT_BYTES:
                raise ReportBuildError(
                    f"{figure['figure_id']}.report.json is {len(report_bytes):,} bytes; "
                    f"limit is {MAX_REPORT_BYTES:,} bytes"
                )
            outputs[REPORT_DIR / f"{figure['figure_id']}.report.json"] = report_bytes
            outputs[REPORT_DIR / f"{figure['figure_id']}.sources.json"] = _json_bytes(
                _source_manifest(figure)
            )
    except ReportBuildError as exc:
        raise SystemExit(f"Paper figure report build failed: {exc}") from exc

    _atomic_batch_write(outputs)
    for path in sorted(outputs):
        print(f"Paper figure metadata saved to {_relative_path(path)} ({len(outputs[path]):,} bytes)")


if __name__ == "__main__":
    main()
