#!/usr/bin/env python3
from __future__ import annotations

"""
Test reading MCS .h5 files using McsPyDataTools and print stream metadata.

Run in your MCS env:
  python scripts/test_mcs_streams.py [N]
  # N = number of files to preview (default 3)
"""

import sys
from pathlib import Path
from typing import Any, Iterable

# Ensure repo root is importable
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mcs_mea_analysis.discovery import iter_h5_files


def as_float(val: Any, unit: str | None = None):
    try:
        if val is None:
            return None
        if unit and hasattr(val, "to"):
            try:
                v2 = val.to(unit)
                if hasattr(v2, "magnitude"):
                    return float(v2.magnitude)
                if hasattr(v2, "m"):
                    return float(v2.m)
            except Exception:
                pass
        if hasattr(val, "magnitude"):
            return float(val.magnitude)
        if hasattr(val, "m"):
            return float(val.m)
        return float(val)
    except Exception:
        return None


def import_mcs():
    for name in ("McsPyDataTools", "mcspydatatools", "McsPy"):
        try:
            return __import__(name), name
        except Exception:
            continue
    raise RuntimeError("McsPyDataTools not importable. Activate env and install it.")


def streams_from_recording(rec: Any) -> list[Any]:
    for attr in ("analog_streams", "AnalogStream", "analogStreams"):
        st = getattr(rec, attr, None)
        if st is None:
            continue
        try:
            return [v for _, v in st.items()] if hasattr(st, "items") else list(st)
        except Exception:
            pass
    return []


def summarize_file(path: Path) -> None:
    mcs, mname = import_mcs()
    McsRecording = getattr(mcs, "McsRecording", None)
    if McsRecording is not None:
        rec = McsRecording(path.as_posix())
    else:
        # Legacy API
        import McsPy.McsData as McsData  # type: ignore
        raw = McsData.RawData(path.as_posix())
        recs = raw.recordings
        if not recs:
            print("  No recordings found")
            return
        rec = next(iter(recs.values()))
    
    print(f"\nFile: {path.name}")
    print(f"  Loader: {mname}")
    sts = streams_from_recording(rec)
    if not sts:
        print("  No analog streams found")
        return

    for si, st in enumerate(sts[:3]):
        # Prefer legacy McsPy: derive from channel_infos
        nchan = None
        sr = None
        nsmpl = None
        ci = getattr(st, "channel_infos", None)
        if ci:
            try:
                nchan = len(ci)
                any_chan = next(iter(ci.values()))
                sr = getattr(any_chan, "sampling_frequency", None)
            except Exception:
                pass
        # Samples from ChannelData
        try:
            nsmpl = int(getattr(st, "channel_data").shape[1])
        except Exception:
            nsmpl = None
        # Duration is per-recording in legacy API
        dur = getattr(rec, "duration_time", None)
        sr_f = as_float(sr, "Hz")
        dur_f = as_float(dur, "s")
        if dur_f is None and (nsmpl is not None) and (sr_f is not None and sr_f > 0):
            try:
                dur_f = float(nsmpl) / float(sr_f)
            except Exception:
                pass
        print(f"  Stream[{si}]: sr_hz={sr_f}, n_channels={nchan}, n_samples={nsmpl}, duration_s={dur_f}")


def main(argv: list[str]) -> int:
    limit = 3
    if len(argv) > 1:
        try:
            limit = int(argv[1])
        except Exception:
            pass

    cnt = 0
    for p in iter_h5_files():
        summarize_file(p)
        cnt += 1
        if cnt >= limit:
            break
    if cnt == 0:
        print("No .h5 files discovered. Check config paths and drive mount.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
