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
from .metadata import GroupLabeler, MetadataExtractor


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

    # Build aggregated file index with inferred group label and basic info
    index_items: list[dict[str, object]] = []
    for r in results:
        gi = GroupLabeler.infer_from_path(r.path)
        bi = MetadataExtractor.extract_basic(r.path)
        index_items.append(
            {
                "file_name": r.path.name,
                "path": str(r.path),
                "round": gi.round_name,
                "plate": gi.plate,
                "group_label": gi.label,
                "is_test": gi.is_test,
                "timestamp": gi.timestamp,
                "sampling_rate_hz": bi.sampling_rate_hz,
                "n_channels": bi.n_channels,
                "duration_seconds": bi.duration_seconds,
                "mcs_available": r.mcs_available,
                "mcs_loaded": r.mcs_loaded,
                "loader": r.loader,
                "error": r.error,
            }
        )

    index_path = probe_dir / f"file_index_{ts}.json"
    with index_path.open("w", encoding="utf-8") as f:
        json.dump({"files": index_items}, f, indent=2)

    # Also write a human-friendly CSV of the file index
    index_csv_path = csv_dir / f"file_index_{ts}.csv"
    index_fields = [
        "file_name",
        "path",
        "round",
        "plate",
        "group_label",
        "is_test",
        "timestamp",
        "sampling_rate_hz",
        "n_channels",
        "duration_seconds",
        "mcs_available",
        "mcs_loaded",
        "loader",
        "error",
    ]
    with index_csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=index_fields)
        w.writeheader()
        for item in index_items:
            w.writerow(item)

    # Log list of discovered files
    log_path = logs_dir / f"discovered_{ts}.log"
    with log_path.open("w", encoding="utf-8") as f:
        for p in files:
            f.write(str(p) + "\n")

    return results
