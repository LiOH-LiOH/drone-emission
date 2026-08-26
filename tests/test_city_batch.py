"""Fixture-free tests for the audited city subprocess scheduler."""

from __future__ import annotations

import csv
import fcntl
import json
from pathlib import Path
import sys

import pytest

from drone_emission.city_batch import run_city_batch
from drone_emission.cli import build_parser


def _command_factory(failing: set[str] | None = None):
    failing = failing or set()

    def command(region_key: str) -> list[str]:
        exit_code = 7 if region_key in failing else 0
        code = (
            "import sys, time; "
            "print('worker', sys.argv[1], flush=True); "
            "time.sleep(0.03); "
            f"raise SystemExit({exit_code})"
        )
        return [sys.executable, "-c", code, region_key]

    return command


def _statuses(run_dir: Path) -> dict[str, dict[str, str]]:
    with (run_dir / "city_status.tsv").open(encoding="utf-8", newline="") as handle:
        return {
            row["region_key"]: row
            for row in csv.DictReader(handle, delimiter="\t")
        }


def _run(run_dir: Path, **overrides) -> int:
    arguments = {
        "region_keys": ["alpha", "beta", "gamma", "delta"],
        "environment": "test",
        "jobs": 4,
        "run_dir": run_dir,
        "force": True,
        "no_fetch": True,
        "cohort_hash": "cohort-test",
        "source_commit": "commit-test",
        "command_factory": _command_factory(),
        "poll_interval": 0.005,
        "summary_interval": 0.01,
    }
    arguments.update(overrides)
    return run_city_batch(**arguments)


def test_four_city_batch_finishes_without_finalize_deadlock(tmp_path):
    assert _run(tmp_path) == 0

    rows = _statuses(tmp_path)
    assert {row["status"] for row in rows.values()} == {"success"}
    assert {row["exit_code"] for row in rows.values()} == {"0"}
    assert all(row["started_at"] and row["ended_at"] for row in rows.values())
    state = json.loads((tmp_path / "state.json").read_text(encoding="utf-8"))
    assert state["stage"] == "city_batch_complete"
    assert state["counts"] == {
        "failed": 0,
        "pending": 0,
        "running": 0,
        "success": 4,
    }


def test_failure_is_recorded_while_other_cities_continue(tmp_path):
    assert _run(
        tmp_path,
        command_factory=_command_factory({"beta"}),
    ) == 1

    rows = _statuses(tmp_path)
    assert rows["beta"]["status"] == "failed"
    assert rows["beta"]["exit_code"] == "7"
    assert all(rows[key]["status"] == "success" for key in ("alpha", "gamma", "delta"))
    assert (tmp_path / "failed_cities.txt").read_text(encoding="utf-8") == "beta\n"


def test_strict_resume_skips_only_recorded_successes(tmp_path):
    assert _run(
        tmp_path,
        region_keys=["alpha", "beta"],
        jobs=2,
        command_factory=_command_factory({"beta"}),
    ) == 1

    assert _run(
        tmp_path,
        region_keys=["alpha", "beta"],
        jobs=2,
        resume=True,
        command_factory=_command_factory(),
    ) == 0

    rows = _statuses(tmp_path)
    assert rows["alpha"]["attempt"] == "1"
    assert rows["beta"]["attempt"] == "2"
    alpha_log = (tmp_path / "cities" / "alpha.log").read_text(encoding="utf-8")
    beta_log = (tmp_path / "cities" / "beta.log").read_text(encoding="utf-8")
    assert alpha_log.count("city:start") == 1
    assert beta_log.count("city:start") == 2


def test_resume_rejects_a_changed_fingerprint(tmp_path):
    assert _run(tmp_path) == 0
    with pytest.raises(ValueError, match="fingerprint"):
        _run(tmp_path, resume=True, source_commit="different-commit")


def test_second_runner_cannot_own_the_same_run_directory(tmp_path):
    lock_path = tmp_path / ".batch.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(RuntimeError, match="already owns"):
            _run(tmp_path)


def test_cli_exposes_audited_batch_options():
    args = build_parser().parse_args(
        [
            "--env",
            "remote",
            "cities",
            "--jobs",
            "4",
            "--cohort",
            "locked",
            "--run-dir",
            "/tmp/paper-cities",
            "--force",
            "--no-fetch",
        ]
    )
    assert args.jobs == 4
    assert args.cohort == "locked"
    assert args.run_dir == Path("/tmp/paper-cities")
