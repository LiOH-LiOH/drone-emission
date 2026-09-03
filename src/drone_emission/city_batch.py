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
    "stage",
    "started_at",
    "ended_at",
    "exit_code",
    "attempt",
    "log_path",
    "resource_class",
    "reservation_gb",
    "current_rss_gb",
    "peak_rss_gb",
    "duration_s",
    "exit_signal",
    "termination_reason",
    "building_bytes",
    "graphml_bytes",
    "old_od_bytes",
    "cgroup_oom_kill_delta",
)


@dataclass
class ActiveCity:
    """One running city subprocess and its open log handle."""

    process: subprocess.Popen
    log_handle: object
    started_monotonic: float
    reservation_gb: int
    peak_rss_bytes: int = 0
    cgroup_events: dict[str, int] | None = None


CommandFactory = Callable[[str], Sequence[str]]
ResultValidator = Callable[[str], None]


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
    from drone_emission.utils.atomic import atomic_open

    with atomic_open(path, "w", encoding="utf-8") as handle:
        handle.write(content)


def _atomic_json(path: Path, payload: dict) -> None:
    _atomic_text(
        path,
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )


def _write_status(path: Path, rows: dict[str, dict[str, str]]) -> None:
    from drone_emission.utils.atomic import atomic_open

    with atomic_open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=STATUS_FIELDS, delimiter="\t")
        writer.writeheader()
        for region_key in sorted(rows):
            writer.writerow(rows[region_key])


def _load_status(path: Path, region_keys: list[str]) -> dict[str, dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = {
            row["region_key"]: row
            for row in csv.DictReader(handle, delimiter="\t")
        }
    if set(rows) != set(region_keys):
        raise ValueError("batch status city set does not match the manifest")
    return rows


def _new_status(
    region_keys: list[str], log_dir: Path, resource_plan: dict[str, dict]
) -> dict[str, dict[str, str]]:
    return {
        region_key: {
            "region_key": region_key,
            "status": "pending",
            "stage": "pending",
            "started_at": "",
            "ended_at": "",
            "exit_code": "",
            "attempt": "0",
            "log_path": str(log_dir / f"{region_key}.log"),
            "resource_class": str(resource_plan[region_key]["resource_class"]),
            "reservation_gb": str(resource_plan[region_key]["reservation_gb"]),
            "current_rss_gb": "0",
            "peak_rss_gb": "0",
            "duration_s": "",
            "exit_signal": "",
            "termination_reason": "",
            "building_bytes": str(resource_plan[region_key].get("building_bytes", 0)),
            "graphml_bytes": str(resource_plan[region_key].get("graphml_bytes", 0)),
            "old_od_bytes": str(resource_plan[region_key].get("old_od_bytes", 0)),
            "cgroup_oom_kill_delta": "0",
        }
        for region_key in region_keys
    }


def _cgroup_memory_events() -> dict[str, int]:
    try:
        cgroup = next(
            line.split("::", 1)[1]
            for line in Path("/proc/self/cgroup").read_text().splitlines()
            if line.startswith("0::")
        )
        path = Path("/sys/fs/cgroup") / cgroup.lstrip("/") / "memory.events"
        return {
            key: int(value)
            for key, value in (line.split() for line in path.read_text().splitlines())
        }
    except (OSError, ValueError, StopIteration):
        return {}


def _process_tree_rss(pid: int) -> int:
    """Read Linux procfs without adding a psutil runtime dependency."""
    pending = [pid]
    seen = set()
    total = 0
    while pending:
        current = pending.pop()
        if current in seen:
            continue
        seen.add(current)
        try:
            for line in Path(f"/proc/{current}/status").read_text().splitlines():
                if line.startswith("VmRSS:"):
                    total += int(line.split()[1]) * 1024
                    break
            children = Path(
                f"/proc/{current}/task/{current}/children"
            ).read_text().split()
            pending.extend(int(child) for child in children)
        except (OSError, ValueError):
            continue
    return total


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
    resource_plan: list[dict] | None = None,
    canary_keys: list[str] | None = None,
    memory_budget_gb: int = 84,
    environment_overrides: dict[str, str] | None = None,
    repair_keys: list[str] | None = None,
    result_validator: ResultValidator | None = None,
    batch_mode: str = "full",
) -> int:
    """Run cities with bounded subprocess concurrency and one status writer."""
    if jobs < 1:
        raise ValueError("jobs must be positive")
    if jobs > 4:
        raise ValueError("audited city batch supports at most four workers")
    if not region_keys or len(region_keys) != len(set(region_keys)):
        raise ValueError("region_keys must be non-empty and unique")

    plan_by_key = {item["region_key"]: item for item in (resource_plan or [])}
    for key in region_keys:
        plan_by_key.setdefault(
            key,
            {"region_key": key, "resource_class": "light", "reservation_gb": 21},
        )
    if set(plan_by_key) != set(region_keys):
        raise ValueError("resource plan city set does not match region_keys")
    if any(
        int(item["reservation_gb"]) > memory_budget_gb
        for item in plan_by_key.values()
    ):
        raise ValueError("resource reservation exceeds scheduling budget")
    canary_keys = canary_keys or []
    if any(key not in region_keys for key in canary_keys):
        raise ValueError("canary city is outside the locked cohort")
    if repair_keys is not None and any(key not in region_keys for key in repair_keys):
        raise ValueError("repair city is outside the batch manifest")

    commit = source_commit or _source_commit()
    contract = {
        "schema_version": 1,
        "batch_mode": batch_mode,
        "source_commit": commit,
        "environment": environment,
        "cohort_hash": cohort_hash,
        "region_keys": sorted(region_keys),
        "jobs": jobs,
        "force": force,
        "no_fetch": no_fetch,
        "memory_budget_gb": memory_budget_gb,
        "canary_keys": canary_keys,
        "resource_plan": [plan_by_key[key] for key in sorted(region_keys)],
        "environment_overrides": dict(sorted((environment_overrides or {}).items())),
    }
    fingerprint = _fingerprint(contract)

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
            for key, row in rows.items():
                if row["status"] != "success" or row["exit_code"] != "0":
                    if repair_keys is not None and key not in repair_keys:
                        continue
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
            rows = _new_status(region_keys, city_log_dir, plan_by_key)

        _write_status(status_path, rows)
        _atomic_text(failed_path, "")
        remaining = [
            region_key
            for region_key in sorted(region_keys)
            if rows[region_key]["status"] == "pending"
        ]
        pending_canaries = [key for key in canary_keys if key in remaining]
        pending = pending_canaries + [key for key in remaining if key not in canary_keys]
        canary_gate_open = not pending_canaries
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
                if not canary_gate_open and active:
                    break
                active_reserved = sum(item.reservation_gb for item in active.values())
                eligible = [
                    key for key in pending
                    if active_reserved + int(plan_by_key[key]["reservation_gb"])
                    <= memory_budget_gb
                    and (canary_gate_open or key == pending_canaries[0])
                ]
                if not eligible:
                    break
                region_key = eligible[0]
                pending.remove(region_key)
                row = rows[region_key]
                attempt = int(row.get("attempt") or 0) + 1
                log_path = Path(row["log_path"])
                log_handle = log_path.open("a", encoding="utf-8", buffering=1)
                log_handle.write(
                    f"{_now()} city:start region={region_key} attempt={attempt}\n"
                )
                environment_vars = dict(os.environ, PYTHONUNBUFFERED="1")
                environment_vars.update(environment_overrides or {})
                environment_vars["DRONE_EMISSION_STAGE_FILE"] = str(
                    run_dir / "cities" / f"{region_key}.stage"
                )
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
                    if region_key in pending_canaries:
                        pending_canaries.remove(region_key)
                        pending.clear()
                    persist("city_batch_running")
                    continue
                row.update(
                    status="running",
                    stage="starting",
                    started_at=_now(),
                    ended_at="",
                    exit_code="",
                    attempt=str(attempt),
                )
                active[region_key] = ActiveCity(
                    process,
                    log_handle,
                    time.monotonic(),
                    int(plan_by_key[region_key]["reservation_gb"]),
                    cgroup_events=_cgroup_memory_events(),
                )
                log(f"city:start region={region_key} pid={process.pid} attempt={attempt}")
                persist("city_batch_running")

            completed = []
            for region_key, item in active.items():
                stage_path = run_dir / "cities" / f"{region_key}.stage"
                try:
                    rows[region_key]["stage"] = stage_path.read_text().strip()
                except OSError:
                    pass
                rss = _process_tree_rss(item.process.pid)
                item.peak_rss_bytes = max(item.peak_rss_bytes, rss)
                rows[region_key]["current_rss_gb"] = f"{rss / 1024**3:.3f}"
                rows[region_key]["peak_rss_gb"] = (
                    f"{item.peak_rss_bytes / 1024**3:.3f}"
                )
                exit_code = item.process.poll()
                if exit_code is None:
                    continue
                item.log_handle.write(
                    f"{_now()} city:end region={region_key} exit={exit_code}\n"
                )
                item.log_handle.close()
                row = rows[region_key]
                events_after = _cgroup_memory_events()
                oom_delta = max(
                    0,
                    events_after.get("oom_kill", 0)
                    - (item.cgroup_events or {}).get("oom_kill", 0),
                )
                signal = -exit_code if exit_code < 0 else 0
                reason = "oom" if signal == 9 and oom_delta else (
                    "external_signal" if signal else ("exit_code" if exit_code else "success")
                )
                exceeded_canary_limit = (
                    region_key in canary_keys
                    and item.peak_rss_bytes > memory_budget_gb * 1024**3
                )
                validation_error = None
                if exit_code == 0 and not exceeded_canary_limit and result_validator:
                    try:
                        result_validator(region_key)
                    except Exception as error:
                        validation_error = f"{type(error).__name__}: {error}"
                        log(
                            f"city:validation_failed region={region_key} "
                            f"error={validation_error}"
                        )
                row.update(
                    status=(
                        "success"
                        if exit_code == 0
                        and not exceeded_canary_limit
                        and validation_error is None
                        else "failed"
                    ),
                    ended_at=_now(),
                    exit_code=str(exit_code),
                    duration_s=f"{time.monotonic() - item.started_monotonic:.3f}",
                    exit_signal=str(signal) if signal else "",
                    termination_reason=(
                        "canary_memory_limit" if exceeded_canary_limit else (
                            "artifact_validation" if validation_error else reason
                        )
                    ),
                    cgroup_oom_kill_delta=str(oom_delta),
                    stage=(
                        "complete"
                        if exit_code == 0
                        and not exceeded_canary_limit
                        and validation_error is None
                        else row["stage"]
                    ),
                )
                completed.append(region_key)
                log(f"city:end region={region_key} exit={exit_code}")
            for region_key in completed:
                del active[region_key]
            if completed:
                for key in completed:
                    if key in pending_canaries:
                        pending_canaries.remove(key)
                if any(rows[key]["status"] == "failed" for key in canary_keys):
                    pending = []
                    persist("canary_failed")
                elif not pending_canaries:
                    canary_gate_open = True
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
