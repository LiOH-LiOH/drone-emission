"""Durable atomic writers used by mutable city-pipeline artifacts."""

from __future__ import annotations

from contextlib import contextmanager
import os
from pathlib import Path
from typing import BinaryIO, Iterator, TextIO


def fsync_directory(path: Path) -> None:
    """Persist a directory entry update when the platform supports it."""
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_replace(temporary: Path, target: Path) -> None:
    """Flush ``temporary`` and atomically publish it as ``target``."""
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    os.replace(temporary, target)
    fsync_directory(target.parent)


@contextmanager
def atomic_open(
    target: str | Path,
    mode: str = "w",
    *,
    encoding: str | None = "utf-8",
    newline: str | None = None,
) -> Iterator[TextIO | BinaryIO]:
    """Yield a same-directory temporary file and publish only on success."""
    path = Path(target)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    kwargs = {}
    if "b" not in mode:
        kwargs.update(encoding=encoding, newline=newline)
    try:
        with temporary.open(mode, **kwargs) as handle:
            yield handle
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        fsync_directory(path.parent)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
