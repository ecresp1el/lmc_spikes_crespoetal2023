from __future__ import annotations

"""
End-to-end probing pipeline:
- Discover .h5 files using configured Manny2TB paths
- Probe each file using MCS reader
- Write a JSONL summary and a CSV summary to the external drive

This is intentionally lightweight: it ensures access and a clear separation for MCS MEA data.
"""

import csv
import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Iterable

from .config import CONFIG
from .discovery import iter_h5_files
from .mcs_reader import ProbeResult, probe_mcs_h5


def run_probe(
    files: Iterable[Path] | None = None,
    output_root: Path | None = None,
) -> list[ProbeResult]:
    files = list(files) if files is not None else list(iter_h5_files())
    out_root = output_root or CONFIG.output_root

    # Prepare output dirs on external drive
    logs_dir = out_root / "logs"
    probe_dir = out_root / "probe"
    csv_dir = out_root / "summaries"
    for d in (logs_dir, probe_dir, csv_dir):
        d.mkdir(parents=True, exist_ok=True)

    # Probe
    results: list[ProbeResult] = []
    for p in files:
        results.append(probe_mcs_h5(p))

    # Timestamped outputs
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    jsonl_path = probe_dir / f"probe_{ts}.jsonl"
    csv_path = csv_dir / f"probe_{ts}.csv"

    # Write JSONL (one record per line)
    with jsonl_path.open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(asdict(r), default=str) + "\n")

    # Write CSV
    fieldnames = [
        "path",
        "exists",
        "is_hdf5_signature",
        "mcs_available",
        "mcs_loaded",
        "loader",
        "error",
        "metadata",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in results:
            row = asdict(r)
            row["path"] = str(row["path"])  # stringify
            if row.get("metadata") is not None:
                row["metadata"] = json.dumps(row["metadata"], ensure_ascii=False)
            w.writerow(row)

    # Log list of discovered files
    log_path = logs_dir / f"discovered_{ts}.log"
    with log_path.open("w", encoding="utf-8") as f:
        for p in files:
            f.write(str(p) + "\n")

    return results

