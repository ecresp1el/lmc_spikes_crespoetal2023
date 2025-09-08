#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

# Ensure repo root import
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mcs_mea_analysis.config import CONFIG
from mcs_mea_analysis.plotting import PlotConfig, RawMEAPlotter


def latest_index(probe_dir: Path) -> Path | None:
    files = sorted(probe_dir.glob("file_index_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    return files[0] if files else None


def main(argv: list[str]) -> int:
    # Usage: python scripts/plot_mcs_raw.py [index_json|latest] [seconds]
    out_root = CONFIG.output_root
    probe_dir = out_root / "probe"
    idx_arg = argv[1] if len(argv) > 1 else "latest"
    # seconds: float or 'full' for entire recording
    if len(argv) > 2:
        seconds_arg = argv[2].strip().lower()
        seconds = None if seconds_arg == "full" else float(seconds_arg)
    else:
        seconds = None

    if idx_arg == "latest":
        idx = latest_index(probe_dir)
        if idx is None:
            print("No index JSON found. Run scripts/run_mcs_scan.py first.")
            return 1
    else:
        idx = Path(idx_arg)
        if not idx.exists():
            print(f"Index JSON not found: {idx}")
            return 1

    cfg = PlotConfig(output_root=out_root, time_seconds=seconds)
    plotter = RawMEAPlotter(cfg)
    plotter.plot_from_index(idx, only_eligible=True)
    print(f"Plotted raw PDFs for eligible files using index: {idx}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
