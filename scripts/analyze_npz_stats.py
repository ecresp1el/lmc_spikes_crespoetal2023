from __future__ import annotations

"""
CLI: NPZ-based analysis with configurable smoothing and windows.

Reads the readiness index (chem + NPZ by default) and runs statistical analysis
per recording from NPZ, writing per-recording CSVs. Plotting is separate.

Usage examples:
  python -m scripts.analyze_npz_stats
  python -m scripts.analyze_npz_stats --smooth 1,1,1,1,1 --pre 120 --post 120 --alpha 0.01 --effect 0.8
  python -m scripts.analyze_npz_stats --group CTZ --round mea_blade_round5

Args:
  --smooth a,b,c,...   Smoothing kernel for IFR (1 ms bins)
  --pre N              Pre-chem window in seconds (default 60)
  --post N             Post-chem window in seconds (default 60)
  --decimate-ms N      Decimation for stats in ms (default 100)
  --alpha P            Significance level (default 0.05)
  --effect D           Cohen's d threshold (default 0.5)
  --require-both       Require both p<alpha and |d|>=thr (default)
  --either             Use either p<alpha or |d|>=thr
  --force              Overwrite existing *_npz_stats.csv files
"""

import sys
from pathlib import Path
from typing import List
import csv

from mcs_mea_analysis.ready import ReadinessConfig, build_ready_index
from mcs_mea_analysis.analysis_config import NPZAnalysisConfig
from mcs_mea_analysis.npz_stats import analyze_ready_rows


def _read_ready_csv(csv_path: Path) -> List[dict]:
    rows: List[dict] = []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        rdr = csv.DictReader(f)
        for r in rdr:
            rows.append(r)
    return rows


def main() -> None:
    args = [a for a in sys.argv[1:] if a]
    if any(a in ("-h", "--help") for a in args):
        print(__doc__)
        sys.exit(0)
    out_root: Path | None = None
    # ready policy
    require_opto = False
    require_eligible = False
    require_fr = False
    require_not_ignored = True
    # filters
    want_groups: List[str] = []
    want_rounds: List[str] = []
    # analysis cfg
    smooth = [1.0, 1.0, 1.0]
    pre_s = 60.0
    post_s = 60.0
    decimate_ms = 100.0
    alpha = 0.05
    effect_thr = 0.5
    require_both = True
    force = False

    i = 0
    while i < len(args):
        a = args[i]
        if a == "--require-opto":
            require_opto = True
        elif a == "--eligible":
            require_eligible = True
        elif a == "--require-fr":
            require_fr = True
        elif a == "--ignore-ignored":
            require_not_ignored = False
        elif a == "--group" and i + 1 < len(args):
            want_groups.append(args[i + 1])
            i += 1
        elif a == "--round" and i + 1 < len(args):
            want_rounds.append(args[i + 1])
            i += 1
        elif a == "--smooth" and i + 1 < len(args):
            smooth = [float(x) for x in args[i + 1].split(",") if x]
            i += 1
        elif a == "--pre" and i + 1 < len(args):
            pre_s = float(args[i + 1]); i += 1
        elif a == "--post" and i + 1 < len(args):
            post_s = float(args[i + 1]); i += 1
        elif a == "--decimate-ms" and i + 1 < len(args):
            decimate_ms = float(args[i + 1]); i += 1
        elif a == "--alpha" and i + 1 < len(args):
            alpha = float(args[i + 1]); i += 1
        elif a == "--effect" and i + 1 < len(args):
            effect_thr = float(args[i + 1]); i += 1
        elif a == "--require-both":
            require_both = True
        elif a == "--either":
            require_both = False
        elif a == "--force":
            force = True
        elif a.startswith("-"):
            print(f"[npz-stats-cli] unknown flag: {a}")
        else:
            out_root = Path(a)
        i += 1

    # Build readiness
    ready_cfg = ReadinessConfig(
        output_root=out_root or ReadinessConfig().output_root,
        require_opto=require_opto,
        require_not_ignored=require_not_ignored,
        require_eligible=require_eligible,
        require_ifr_npz=True,
        require_fr_summary=require_fr,
    )
    ready_csv, _, rows = build_ready_index(ready_cfg)
    rows = [r for r in rows if str(r.get("ready")) == "True"]
    if want_groups:
        rows = [r for r in rows if str(r.get("group_label", "")) in want_groups]
    if want_rounds:
        rows = [r for r in rows if str(r.get("round", "")) in want_rounds]
    print(f"[npz-stats-cli] ready rows: {len(rows)} (from {ready_csv})")

    # Analyze
    cfg = NPZAnalysisConfig(
        output_root=ready_cfg.output_root,
        annotations_root=ready_cfg.annotations_root,
        smooth_kernel=smooth,
        pre_span_s=pre_s,
        post_span_s=post_s,
        decimate_ms=decimate_ms,
        alpha=alpha,
        effect_size_thr=effect_thr,
        require_both=require_both,
    )
    out_csvs = analyze_ready_rows(rows, cfg, force=force)
    print(f"[npz-stats-cli] wrote {len(out_csvs)} per-recording CSV(s)")


if __name__ == "__main__":
    main()
