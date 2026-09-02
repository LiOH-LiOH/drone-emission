"""Bounded-memory weighted quantiles for large CSV inputs."""

from __future__ import annotations

import hashlib
import heapq
import json
import math
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from drone_emission.utils.atomic import atomic_open
from drone_emission.utils.config import get_cache_dir


_CACHE_VERSION = 1


@dataclass(frozen=True)
class WeightedQuantileResult:
    value: float
    valid_rows: int
    total_weight: float
    quantile: float


def _cache_path(
    csv_path: Path,
    *,
    value_column: str,
    weight_column: str,
    quantile: float,
    cache_dir: str | Path | None,
) -> tuple[Path, dict]:
    source = str(csv_path.resolve())
    stat = csv_path.stat()
    source_key = hashlib.sha256(source.encode("utf-8")).hexdigest()[:16]
    query = {
        "value_column": value_column,
        "weight_column": weight_column,
        "quantile": quantile,
        "algorithm_version": _CACHE_VERSION,
    }
    query_key = hashlib.sha256(
        json.dumps(query, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:16]
    root = (
        Path(cache_dir)
        if cache_dir is not None
        else Path(get_cache_dir("analysis", "weighted_quantiles"))
    )
    signature = {
        "source": source,
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }
    return root / source_key / f"{query_key}.json", {
        "source": signature,
        "query": query,
    }


def _read_cached_result(cache_path: Path, expected: dict):
    try:
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
        if payload.get("source") != expected["source"]:
            return None
        if payload.get("query") != expected["query"]:
            return None
        result = payload["result"]
        return WeightedQuantileResult(
            value=float(result["value"]),
            valid_rows=int(result["valid_rows"]),
            total_weight=float(result["total_weight"]),
            quantile=float(result["quantile"]),
        )
    except (
        AttributeError,
        FileNotFoundError,
        json.JSONDecodeError,
        KeyError,
        TypeError,
        ValueError,
    ):
        return None


def weighted_csv_quantile(
    csv_path: str | Path,
    *,
    value_column: str,
    weight_column: str,
    quantile: float,
    chunk_rows: int = 250_000,
    cache_dir: str | Path | None = None,
) -> WeightedQuantileResult:
    """Return and cache an exact empirical weighted quantile."""
    if not math.isfinite(quantile) or not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must be between zero and one")
    if chunk_rows <= 0:
        raise ValueError("chunk_rows must be positive")

    csv_path = Path(csv_path)
    cache_path, cache_identity = _cache_path(
        csv_path,
        value_column=value_column,
        weight_column=weight_column,
        quantile=quantile,
        cache_dir=cache_dir,
    )
    cached = _read_cached_result(cache_path, cache_identity)
    if cached is not None:
        return cached

    header = pd.read_csv(csv_path, nrows=0).columns
    missing = sorted({value_column, weight_column} - set(header))
    if missing:
        raise ValueError(f"{csv_path} is missing columns: {missing}")

    total_weight = 0.0
    valid_rows = 0
    with tempfile.TemporaryDirectory(prefix="weighted-quantile-") as temp_dir:
        run_paths: list[Path] = []
        for run_index, chunk in enumerate(
            pd.read_csv(
                csv_path,
                usecols=[value_column, weight_column],
                chunksize=chunk_rows,
            )
        ):
            values = pd.to_numeric(chunk[value_column], errors="coerce").to_numpy(
                dtype=float
            )
            weights = pd.to_numeric(chunk[weight_column], errors="coerce").to_numpy(
                dtype=float
            )
            valid = (
                np.isfinite(values)
                & np.isfinite(weights)
                & (values > 0)
                & (weights > 0)
            )
            values = values[valid]
            weights = weights[valid]
            if not len(values):
                continue
            order = np.argsort(values, kind="stable")
            run = np.column_stack((values[order], weights[order]))
            run_path = Path(temp_dir) / f"run_{run_index:06d}.npy"
            np.save(run_path, run, allow_pickle=False)
            run_paths.append(run_path)
            total_weight += float(weights.sum())
            valid_rows += int(len(values))

        if not run_paths or total_weight <= 0:
            raise ValueError(f"{csv_path} contains no positive weighted values")

        runs = [np.load(path, mmap_mode="r") for path in run_paths]
        heap = [(float(run[0, 0]), index, 0) for index, run in enumerate(runs)]
        heapq.heapify(heap)
        target = quantile * total_weight
        cumulative = 0.0
        selected = float(heap[0][0])

        while heap:
            selected = float(heap[0][0])
            equal_value_weight = 0.0
            while heap and heap[0][0] == selected:
                _, run_index, row_index = heapq.heappop(heap)
                run = runs[run_index]
                equal_value_weight += float(run[row_index, 1])
                next_index = row_index + 1
                if next_index < len(run):
                    heapq.heappush(
                        heap,
                        (float(run[next_index, 0]), run_index, next_index),
                    )
            cumulative += equal_value_weight
            if cumulative >= target:
                break

    result = WeightedQuantileResult(
        value=selected,
        valid_rows=valid_rows,
        total_weight=total_weight,
        quantile=quantile,
    )
    payload = {
        **cache_identity,
        "result": {
            "value": result.value,
            "valid_rows": result.valid_rows,
            "total_weight": result.total_weight,
            "quantile": result.quantile,
        },
    }
    with atomic_open(cache_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    return result
