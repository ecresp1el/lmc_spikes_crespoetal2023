from __future__ import annotations

"""
Pair Viewer (CLI)
=================

Launch the CTZ–VEH pair viewer GUI from the command line.

Two modes
---------
1) Index-based: pick a plate and pair index (derived from readiness)
   python -m scripts.pair_viewer --plate 1 --idx 0 [--ch 0]

2) Direct-file: specify NPZs (and optionally H5 + chem timestamps)
   python -m scripts.pair_viewer \
       --ctz-npz /...ctz..._ifr_per_channel_1ms.npz \
       --veh-npz /...veh..._ifr_per_channel_1ms.npz \
       [--ctz-h5 /path/ctz.h5] [--veh-h5 /path/veh.h5] \
       [--chem-ctz 181.27] [--chem-veh 183.33] [--plate 10] [--round mea_blade_round4] [--ch 0]

Flags
-----
--plate N         Plate number (required in index-based mode)
--idx K           Pair index within that plate (0-based, required in index mode)
--ch N            Initial channel to display (default 0)
--ctz-npz PATH    Path to CTZ IFR NPZ (direct-file mode)
--veh-npz PATH    Path to VEH IFR NPZ (direct-file mode)
--ctz-h5 PATH     Optional raw CTZ H5 to enable analog plotting
--veh-h5 PATH     Optional raw VEH H5 to enable analog plotting
--chem-ctz S      Optional chem time (s) for CTZ; otherwise tries annotations
--chem-veh S      Optional chem time (s) for VEH; otherwise tries annotations
--round NAME      Optional round label for display and selection naming

Notes
-----
- Selections are auto-saved under `<output_root>/selections/` and reloaded on
  next launch.
- If running from Jupyter, prefer the inline launcher to avoid module path
  issues; otherwise set the working directory to the repo root.
"""

import sys
from pathlib import Path
from typing import List, Optional

# Ensure repo root
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mcs_mea_analysis.ready import ReadinessConfig, build_ready_index
from mcs_mea_analysis.pairings import PairingIndex
from mcs_mea_analysis.pair_viewer_gui import PairInputs, launch_pair_viewer


def _arg_value(args: List[str], flag: str, n: int = 1) -> Optional[str]:
    if flag in args:
        i = args.index(flag)
        if i + n < len(args):
            return args[i + n]
    return None


def main() -> None:
    args = [a for a in sys.argv[1:] if a]
    if any(a in ("-h", "--help") for a in args):
        print(__doc__)
        sys.exit(0)

    # Direct mode
    ctz_npz = _arg_value(args, "--ctz-npz")
    veh_npz = _arg_value(args, "--veh-npz")
    ctz_h5 = _arg_value(args, "--ctz-h5")
    veh_h5 = _arg_value(args, "--veh-h5")
    chem_ctz = _arg_value(args, "--chem-ctz")
    chem_veh = _arg_value(args, "--chem-veh")
    plate = _arg_value(args, "--plate")
    round_name = _arg_value(args, "--round")
    ch_s = _arg_value(args, "--ch")
    ch = int(ch_s) if ch_s is not None else 0

    if ctz_npz and veh_npz:
        pin = PairInputs(
            plate=int(plate) if plate else None,
            round=round_name,
            ctz_npz=Path(ctz_npz),
            veh_npz=Path(veh_npz),
            ctz_h5=Path(ctz_h5) if ctz_h5 else None,
            veh_h5=Path(veh_h5) if veh_h5 else None,
            chem_ctz_s=float(chem_ctz) if chem_ctz else None,
            chem_veh_s=float(chem_veh) if chem_veh else None,
            initial_channel=ch,
        )
        launch_pair_viewer(pin)
        return

    # Index-based mode: --plate N --idx K
    plate_s = _arg_value(args, "--plate")
    idx_s = _arg_value(args, "--idx")
    if not plate_s or not idx_s:
        print("Provide either direct file args (--ctz-npz/--veh-npz) or --plate and --idx.")
        sys.exit(2)
    plate_i = int(plate_s)
    pair_idx = int(idx_s)

    # Build readiness and pairs
    ready_cfg = ReadinessConfig()
    _, _, rows = build_ready_index(ready_cfg)
    px = PairingIndex.from_ready_rows(rows, group_by_round=True)
    pairs_df = __import__("pandas").DataFrame(px.pairs_dataframe())
    pairs_df = pairs_df.query("pair_status=='ready_pair'").copy()
    rows_here = pairs_df[pairs_df["plate"] == plate_i].reset_index(drop=True)
    if rows_here.empty:
        print(f"No ready pairs for plate {plate_i}")
        sys.exit(3)
    if not (0 <= pair_idx < len(rows_here)):
        print(f"Pair idx out of range [0..{len(rows_here)-1}] for plate {plate_i}")
        sys.exit(4)
    r = rows_here.iloc[pair_idx]

    # Find H5 paths and chem stamps from readiness rows
    import pandas as pd
    df = pd.DataFrame(rows)
    def _row_for_stem(stem: str) -> pd.Series:
        m = df[df["recording_stem"] == stem]
        return m.iloc[0] if not m.empty else pd.Series()
    rc = _row_for_stem(r["ctz_stem"]) ; rv = _row_for_stem(r["veh_stem"]) 
    pin = PairInputs(
        plate=int(r["plate"]) if pd.notna(r["plate"]) else None,
        round=str(r["round"]) if pd.notna(r["round"]) else None,
        ctz_npz=Path(r["ctz_npz_path"]),
        veh_npz=Path(r["veh_npz_path"]),
        ctz_h5=Path(rc.get("path")) if isinstance(rc.get("path"), str) and rc.get("path") else None,
        veh_h5=Path(rv.get("path")) if isinstance(rv.get("path"), str) and rv.get("path") else None,
        chem_ctz_s=float(rc.get("chem_timestamp")) if pd.notna(rc.get("chem_timestamp")) else None,
        chem_veh_s=float(rv.get("chem_timestamp")) if pd.notna(rv.get("chem_timestamp")) else None,
        initial_channel=ch,
    )
    launch_pair_viewer(pin)


if __name__ == "__main__":
    main()
