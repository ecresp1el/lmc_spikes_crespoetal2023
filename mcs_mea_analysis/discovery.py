from __future__ import annotations

"""
Discovery utilities to locate MCS MEA .h5 files on Manny2TB.

We do not modify or delete source files. We only read and report.
"""

from pathlib import Path
from typing import Iterable, List

from .config import CONFIG


def iter_h5_files(
    base_dirs: Iterable[Path] | None = None,
    h5_subdir: str | None = None,
    recursive: bool = False,
) -> Iterable[Path]:
    """
    Iterate all .h5 files under each `<base_dir>/<h5_subdir>` directory.

    - If `<base_dir>/<h5_subdir>` is missing, skip quietly.
    - `recursive=True` walks subdirectories; otherwise, only the top level is listed.
    """
    base_dirs = tuple(base_dirs) if base_dirs is not None else CONFIG.base_dirs
    h5_subdir = h5_subdir or CONFIG.h5_subdir

    for base in base_dirs:
        h5_dir = base / h5_subdir
        if not h5_dir.exists() or not h5_dir.is_dir():
            continue
        if recursive:
            for p in sorted(h5_dir.rglob("*.h5")):
                if p.is_file():
                    yield p
        else:
            for p in sorted(h5_dir.iterdir()):
                if p.suffix.lower() == ".h5" and p.is_file():
                    yield p


def list_h5_files(
    base_dirs: Iterable[Path] | None = None,
    h5_subdir: str | None = None,
    recursive: bool = False,
) -> List[Path]:
    """Return a list of discovered .h5 files for convenience."""
    return list(
        iter_h5_files(base_dirs=base_dirs, h5_subdir=h5_subdir, recursive=recursive)
    )

