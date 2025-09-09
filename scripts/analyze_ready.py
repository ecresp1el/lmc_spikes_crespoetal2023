from __future__ import annotations

"""
CLI: Analyze the ready set (chem + NPZ) in a single, standardized pipeline.

Steps (GUI-independent):
1) Build the readiness index (chem+NPZ by default; configurable flags)
2) Filter rows to ready==True (optional round/group/plate filters)
3) For each row:
   - Process NPZ to per-recording IFR summary (unless --skip-processing)
   - Plot all channels' IFR grid from NPZ (unless --skip-plotting)

Usage examples:
  # Default output_root and readiness policy (chem+NPZ)
  python scripts/analyze_ready.py

  # Use explicit output root
  python scripts/analyze_ready.py /Volumes/Manny2TB/mcs_mea_outputs

  # Rebuild readiness with tighter rules and analyze only CTZ in round 5 plates 1 and 6
  python scripts/analyze_ready.py --require-opto --eligible --group CTZ --round mea_blade_round5 --plate 1 --plate 6

Flags:
  --require-opto / --no-require-opto (default: no)
  --eligible / --no-eligible (default: no)
  --require-fr (default: no)
  --ignore-ignored (default: yes; pass to include ignored)
  --skip-processing (skip NPZ -> CSV metrics)
  --skip-plotting  (skip plotting grid from NPZ)
  --force-plots    (overwrite existing plots)
  --group <label>  (repeatable)
  --round <name>   (repeatable)
  --plate <int>    (repeatable)
"""

import sys
from pathlib import Path
from typing import List
import csv

from mcs_mea_analysis.ready import ReadinessConfig, build_ready_index
from mcs_mea_analysis.ifr_processing import process_ifr_npz
from mcs_mea_analysis.ifr_analysis import plot_ifr_grid_from_npz, IFRPlotConfig


def _read_ready_csv(csv_path: Path) -> List[dict]:
    rows: List[dict] = []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        rdr = csv.DictReader(f)
        for r in rdr:
            rows.append(r)
    return rows


def main() -> None:
    args = [a for a in sys.argv[1:] if a]
    out_root: Path | None = None
    # Readiness flags
    require_opto = False
    require_eligible = False
    require_fr = False
    require_not_ignored = True
    # Pipeline flags
    skip_processing = False
    skip_plotting = False
    force_plots = False
    # Filters
    want_groups: List[str] = []
    want_rounds: List[str] = []
    want_plates: List[int] = []

    # Parse args
    i = 0
    while i < len(args):
        a = args[i]
        if a == "--require-opto":
            require_opto = True
        elif a == "--no-require-opto":
            require_opto = False
        elif a == "--eligible":
            require_eligible = True
        elif a == "--no-eligible":
            require_eligible = False
        elif a == "--require-fr":
            require_fr = True
        elif a == "--ignore-ignored":
            require_not_ignored = False
        elif a == "--skip-processing":
            skip_processing = True
        elif a == "--skip-plotting":
            skip_plotting = True
        elif a == "--force-plots":
            force_plots = True
        elif a == "--group" and i + 1 < len(args):
            want_groups.append(args[i + 1])
            i += 1
        elif a == "--round" and i + 1 < len(args):
            want_rounds.append(args[i + 1])
            i += 1
        elif a == "--plate" and i + 1 < len(args):
            try:
                want_plates.append(int(args[i + 1]))
            except Exception:
                pass
            i += 1
        elif a.startswith("-"):
            print(f"[analyze-ready] unknown flag: {a}")
        else:
            out_root = Path(a)
        i += 1

    # Build readiness index with current policy
    cfg = ReadinessConfig(
        output_root=out_root or ReadinessConfig().output_root,
        require_opto=require_opto,
        require_not_ignored=require_not_ignored,
        require_eligible=require_eligible,
        require_ifr_npz=True,
        require_fr_summary=require_fr,
    )
    ready_csv, _, rows = build_ready_index(cfg)

    # Filter to ready rows
    rows = [r for r in rows if str(r.get("ready")) == "True"]
    if want_groups:
        rows = [r for r in rows if str(r.get("group_label", "")) in want_groups]
    if want_rounds:
        rows = [r for r in rows if str(r.get("round", "")) in want_rounds]
    if want_plates:
        rows = [r for r in rows if str(r.get("path", "")).find("plate_") != -1 and any(f"plate_{p}" in str(r.get("path", "")) for p in want_plates)]

    print(f"[analyze-ready] ready rows: {len(rows)} (from {ready_csv})")

    # Run processing/plotting
    plot_cfg = IFRPlotConfig(output_root=cfg.output_root)
    for r in rows:
        npz_path = Path(str(r.get("npz_path", "")))
        if not npz_path.exists():
            print(f"[analyze-ready] skip (missing NPZ): {npz_path}")
            continue
        # per-recording processing (NPZ -> CSV)
        if not skip_processing:
            try:
                out_csv = process_ifr_npz(npz_path)
                if out_csv:
                    print(f"[analyze-ready] processed -> {out_csv}")
            except Exception as e:
                print(f"[analyze-ready] processing error: {npz_path} -> {e}")
        # plotting
        if not skip_plotting:
            try:
                grid_pdf = npz_path.parent / f"{npz_path.stem.replace('_ifr_per_channel_1ms','')}_ifr_channels_grid.pdf"
                if force_plots or (not grid_pdf.exists()):
                    plot_ifr_grid_from_npz(npz_path, plot_cfg)
                else:
                    print(f"[analyze-ready] skip plot (exists): {grid_pdf}")
            except Exception as e:
                print(f"[analyze-ready] plot error: {npz_path} -> {e}")


if __name__ == "__main__":
    main()

