#!/usr/bin/env python3
from __future__ import annotations

"""
Quick metadata peek for a few discovered MCS .h5 files.
Prints sampling rate, channel count, and duration if available.
"""

import sys
from pathlib import Path

# Ensure repo root is importable when running from scripts/
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mcs_mea_analysis.discovery import iter_h5_files
from mcs_mea_analysis.metadata import GroupLabeler, MetadataExtractor


def main(argv: list[str]) -> int:
    limit = 5
    if len(argv) > 1:
        try:
            limit = int(argv[1])
        except Exception:
            pass

    count = 0
    for p in iter_h5_files():
        gi = GroupLabeler.infer_from_path(p)
        bi = MetadataExtractor.extract_basic(p)
        print(
            f"{p.name}\n"
            f"  group={gi.label}, round={gi.round_name}, plate={gi.plate}, ts={gi.timestamp}, test={gi.is_test}\n"
            f"  sampling_rate_hz={bi.sampling_rate_hz}, n_channels={bi.n_channels}, duration_s={bi.duration_seconds}\n"
        )
        count += 1
        if count >= limit:
            break
    if count == 0:
        print("No .h5 files discovered. Check config paths and drive mount.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))

