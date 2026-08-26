"""Audited, bounded-concurrency execution for the locked city cohort."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Callable, Sequence

from .paths import PROJECT_ROOT


STATUS_FIELDS = (
    "region_key",
    "status",
    "started_at",
    "ended_at",
    "exit_code",
    "attempt",
    "log_path",
)


@dataclass
class ActiveCity:
    """One running city subprocess and its open log handle."""

    process: subprocess.Popen
    log_handle: object


CommandFactory = Callable[[str], Sequence[str]]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _source_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _fingerprint(payload: dict) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _atomic_text(path: Path, content: str) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def _atomic_json(path: Path, payload: dict) -> None:
    _atomic_text(
        path,
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )


def _write_status(path: Path, rows: dict[str, dict[str, str]]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=STATUS_FIELDS, delimiter="\t")
        writer.writeheader()
        for region_key in sorted(rows):
            writer.writerow(rows[region_key])
    os.replace(temporary, path)


def _load_status(path: Path, region_keys: list[str]) -> dict[str, dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = {
            row["region_key"]: row
            for row in csv.DictReader(handle, delimiter="\t")
        }
    if set(rows) != set(region_keys):
        raise ValueError("batch status city set does not match the manifest")
    return rows


def _new_status(region_keys: list[str], log_dir: Path) -> dict[str, dict[str, str]]:
    return {
        region_key: {
            "region_key": region_key,
            "status": "pending",
            "started_at": "",
            "ended_at": "",
            "exit_code": "",
            "attempt": "0",
            "log_path": str(log_dir / f"{region_key}.log"),
        }
        for region_key in region_keys
    }


def _counts(rows: dict[str, dict[str, str]]) -> dict[str, int]:
    return {
        status: sum(row["status"] == status for row in rows.values())
        for status in ("pending", "running", "success", "failed")
    }


def _default_command(
    environment: str,
    force: bool,
    no_fetch: bool,
) -> CommandFactory:
    def command(region_key: str) -> list[str]:
        result = [
            sys.executable,
            "-m",
            "drone_emission",
            "--env",
            environment,
            "cities",
            "--region",
            region_key,
        ]
        if force:
            result.append("--force")
        if no_fetch:
            result.append("--no-fetch")
        return result

    return command


def run_city_batch(
    *,
    region_keys: list[str],
    environment: str,
    jobs: int,
    run_dir: Path,
    force: bool,
    no_fetch: bool,
    cohort_hash: str,
    resume: bool = False,
    source_commit: str | None = None,
    command_factory: CommandFactory | None = None,
    poll_interval: float = 1.0,
    summary_interval: float = 60.0,
) -> int:
    """Run cities with bounded subprocess concurrency and one status writer."""
    if jobs < 1:
        raise ValueError("jobs must be positive")
    if not region_keys or len(region_keys) != len(set(region_keys)):
        raise ValueError("region_keys must be non-empty and unique")

    run_dir = run_dir.resolve()
    city_log_dir = run_dir / "cities"
    run_dir.mkdir(parents=True, exist_ok=True)
    city_log_dir.mkdir(parents=True, exist_ok=True)
    lock_handle = (run_dir / ".batch.lock").open("a+", encoding="utf-8")
    try:
        fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        lock_handle.close()
        raise RuntimeError(f"batch runner already owns {run_dir}") from error

    commit = source_commit or _source_commit()
    contract = {
        "schema_version": 1,
        "source_commit": commit,
        "environment": environment,
        "cohort_hash": cohort_hash,
        "region_keys": sorted(region_keys),
        "jobs": jobs,
        "force": force,
        "no_fetch": no_fetch,
    }
    fingerprint = _fingerprint(contract)
    manifest_path = run_dir / "batch_manifest.json"
    status_path = run_dir / "city_status.tsv"
    failed_path = run_dir / "failed_cities.txt"
    state_path = run_dir / "state.json"
    run_log_path = run_dir / "run.log"

    def log(message: str) -> None:
        with run_log_path.open("a", encoding="utf-8") as handle:
            handle.write(f"{_now()} {message}\n")
            handle.flush()

    try:
        if manifest_path.exists():
            if not resume:
                raise FileExistsError(
                    f"batch manifest already exists; use --resume: {manifest_path}"
                )
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("fingerprint") != fingerprint:
                raise ValueError("resume fingerprint does not match batch manifest")
            if not status_path.is_file():
                raise FileNotFoundError(f"missing resume status: {status_path}")
            rows = _load_status(status_path, region_keys)
            for row in rows.values():
                if row["status"] != "success" or row["exit_code"] != "0":
                    row.update(
                        status="pending",
                        started_at="",
                        ended_at="",
                        exit_code="",
                    )
            log("batch:resume fingerprint=matched")
        else:
            if resume:
                raise FileNotFoundError(
                    f"cannot resume without batch manifest: {manifest_path}"
                )
            manifest = {
                **contract,
                "fingerprint": fingerprint,
                "created_at": _now(),
            }
            _atomic_json(manifest_path, manifest)
            rows = _new_status(region_keys, city_log_dir)

        _write_status(status_path, rows)
        _atomic_text(failed_path, "")
        pending = [
            region_key
            for region_key in sorted(region_keys)
            if rows[region_key]["status"] != "success"
        ]
        active: dict[str, ActiveCity] = {}
        command_for = command_factory or _default_command(
            environment, force, no_fetch
        )
        last_summary = 0.0

        def persist(stage: str) -> None:
            failed = sorted(
                key for key, row in rows.items() if row["status"] == "failed"
            )
            _write_status(status_path, rows)
            _atomic_text(
                failed_path, "".join(f"{region_key}\n" for region_key in failed)
            )
            _atomic_json(
                state_path,
                {
                    "schema_version": 1,
                    "stage": stage,
                    "updated_at": _now(),
                    "source_commit": commit,
                    "cohort_hash": cohort_hash,
                    "fingerprint": fingerprint,
                    "counts": _counts(rows),
                    "active": {
                        key: item.process.pid for key, item in sorted(active.items())
                    },
                },
            )

        persist("city_batch_running")
        log(
            "batch:start "
            f"cities={len(region_keys)} pending={len(pending)} jobs={jobs}"
        )

        while pending or active:
            while pending and len(active) < jobs:
                region_key = pending.pop(0)
                row = rows[region_key]
                attempt = int(row.get("attempt") or 0) + 1
                log_path = Path(row["log_path"])
                log_handle = log_path.open("a", encoding="utf-8", buffering=1)
                log_handle.write(
                    f"{_now()} city:start region={region_key} attempt={attempt}\n"
                )
                environment_vars = dict(os.environ, PYTHONUNBUFFERED="1")
                try:
                    process = subprocess.Popen(
                        list(command_for(region_key)),
                        cwd=PROJECT_ROOT,
                        stdout=log_handle,
                        stderr=subprocess.STDOUT,
                        env=environment_vars,
                    )
                except Exception as error:
                    log_handle.write(
                        f"{_now()} city:launch_failed {type(error).__name__}: {error}\n"
                    )
                    log_handle.close()
                    row.update(
                        status="failed",
                        started_at=_now(),
                        ended_at=_now(),
                        exit_code="-1",
                        attempt=str(attempt),
                    )
                    log(f"city:end region={region_key} exit=-1 launch_failed=true")
                    persist("city_batch_running")
                    continue
                row.update(
                    status="running",
                    started_at=_now(),
                    ended_at="",
                    exit_code="",
                    attempt=str(attempt),
                )
                active[region_key] = ActiveCity(process, log_handle)
                log(f"city:start region={region_key} pid={process.pid} attempt={attempt}")
                persist("city_batch_running")

            completed = []
            for region_key, item in active.items():
                exit_code = item.process.poll()
                if exit_code is None:
                    continue
                item.log_handle.write(
                    f"{_now()} city:end region={region_key} exit={exit_code}\n"
                )
                item.log_handle.close()
                row = rows[region_key]
                row.update(
                    status="success" if exit_code == 0 else "failed",
                    ended_at=_now(),
                    exit_code=str(exit_code),
                )
                completed.append(region_key)
                log(f"city:end region={region_key} exit={exit_code}")
            for region_key in completed:
                del active[region_key]
            if completed:
                persist("city_batch_running")

            monotonic = time.monotonic()
            if monotonic - last_summary >= summary_interval:
                counts = _counts(rows)
                log(
                    "batch:progress "
                    + " ".join(f"{key}={value}" for key, value in counts.items())
                )
                persist("city_batch_running")
                last_summary = monotonic
            if pending or active:
                time.sleep(poll_interval)

        counts = _counts(rows)
        stage = "city_batch_complete" if not counts["failed"] else "city_batch_failed"
        persist(stage)
        log(
            "batch:end "
            + " ".join(f"{key}={value}" for key, value in counts.items())
        )
        return 0 if not counts["failed"] else 1
    finally:
        fcntl.flock(lock_handle, fcntl.LOCK_UN)
        lock_handle.close()
