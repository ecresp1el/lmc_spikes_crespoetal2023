from __future__ import annotations

"""
Instantaneous firing-rate (IFR) NPZ processing — GUI‑independent.

Goal
- Provide a standardized, reusable way to process the 1 ms per‑channel IFR NPZs
  that the FR pipeline writes under: <output_root>/plots/<round>/<group>/<stem>/fr/
- Keep processing (metrics, catalogs) separate from plotting.

Outputs (per recording)
- <stem>_ifr_npz_summary.csv — per‑channel pre/post FR (from IFR) and modulation class

Global outputs
- <output_root>/ifr_npz_catalog/ifr_npz_catalog.{csv,jsonl}
- <output_root>/ifr_npz_catalog/ifr_npz_status.csv (counts per recording)

Backwards‑compatibility
- Uses the same directory layout and annotation files as the GUI.
- Does not require spike streams or the GUI to be running.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
import csv
import json
import numpy as np

from .config import CONFIG
from .metadata import GroupLabeler


@dataclass(frozen=True)
class IFRProcessorConfig:
    output_root: Path = CONFIG.output_root
    annotations_root: Path = CONFIG.annotations_root
    abs_thr_hz: float = 0.2  # absolute delta threshold in Hz
    rel_thr_frac: float = 0.2  # relative change threshold (20%)


def find_ifr_npz(output_root: Path | None = None) -> List[Path]:
    root = output_root or CONFIG.output_root
    base = root / "plots"
    if not base.exists():
        return []
    # Glob all *_ifr_per_channel_1ms.npz under plots/*/*/*/fr/
    return sorted(base.glob("**/fr/*_ifr_per_channel_1ms.npz"))


def _chem_time_for_stem(stem: str, annotations_root: Path) -> Optional[float]:
    # Prefer JSON; fallback CSV
    for ext in (".json", ".csv"):
        p = annotations_root / f"{stem}{ext}"
        if not p.exists():
            continue
        try:
            if p.suffix.lower() == ".json":
                data = json.loads(p.read_text())
            else:
                with p.open("r", newline="") as fh:
                    data = list(csv.DictReader(fh))
            for row in data:
                if str(row.get("category", "manual")) == "chemical":
                    return float(row.get("timestamp", 0.0))
        except Exception:
            continue
    return None


def _modulation_label(fr_pre: float, fr_post: float, abs_thr: float, rel_thr: float) -> str:
    d = fr_post - fr_pre
    if fr_pre > 0:
        rel = abs(d) / max(fr_pre, 1e-12)
    else:
        rel = float("inf") if abs(d) > 0 else 0.0
    if (abs(d) >= abs_thr) or (rel >= rel_thr):
        return "positive" if d > 0 else "negative"
    return "nochange"


def process_ifr_npz(
    npz_path: Path,
    cfg: IFRProcessorConfig | None = None,
) -> Optional[Path]:
    """Process a single per‑channel IFR NPZ and write a per‑recording summary CSV.

    Returns the path to the summary CSV or None if skipped (e.g., no chem stamp).
    """
    cfg = cfg or IFRProcessorConfig()
    npz_path = Path(npz_path)
    # Parse metadata from directory structure
    # .../plots/<round>/<group>/<stem>/fr/<stem>_ifr_per_channel_1ms.npz
    stem = npz_path.stem.replace("_ifr_per_channel_1ms", "")
    fr_dir = npz_path.parent
    rec_dir = fr_dir.parent
    group_dir = rec_dir.parent
    round_dir = group_dir.parent
    round_name = round_dir.name
    group_label = group_dir.name

    # Load IFR arrays
    try:
        d = np.load(npz_path)
        time_s = np.asarray(d["time_s"], dtype=float)  # (n_bins,)
        ifr_hz = np.asarray(d["ifr_hz"], dtype=float)  # (n_ch, n_bins)
        ifr_hz_s = np.asarray(d.get("ifr_hz_smooth", ifr_hz), dtype=float)
    except Exception as e:
        print(f"[ifr] load failed: {npz_path} -> {e}")
        return None
    if time_s.size == 0 or ifr_hz_s.size == 0:
        print(f"[ifr] empty arrays: {npz_path}")
        return None

    # Find chem time
    chem_ts = _chem_time_for_stem(stem, cfg.annotations_root)
    if chem_ts is None:
        print(f"[ifr] skip (no chem): {stem}")
        return None

    # Compute per-channel FR from smoothed IFR
    mask_pre = time_s < chem_ts
    mask_post = time_s >= chem_ts
    if not np.any(mask_pre) or not np.any(mask_post):
        print(f"[ifr] skip (invalid chem window): {stem}")
        return None

    fr_pre = np.nanmean(ifr_hz_s[:, mask_pre], axis=1)
    fr_post = np.nanmean(ifr_hz_s[:, mask_post], axis=1)
    fr_full = np.nanmean(ifr_hz_s, axis=1)

    # Write per-recording summary CSV next to NPZ
    out_csv = fr_dir / f"{stem}_ifr_npz_summary.csv"
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "round",
                "group_label",
                "recording_stem",
                "channel",
                "fr_full",
                "fr_pre",
                "fr_post",
                "modulation",
            ],
        )
        w.writeheader()
        n_ch = ifr_hz_s.shape[0]
        for ch in range(n_ch):
            pre_v = float(fr_pre[ch])
            post_v = float(fr_post[ch])
            full_v = float(fr_full[ch])
            mod = _modulation_label(pre_v, post_v, cfg.abs_thr_hz, cfg.rel_thr_frac)
            w.writerow(
                {
                    "round": round_name,
                    "group_label": group_label,
                    "recording_stem": stem,
                    "channel": ch,
                    "fr_full": full_v,
                    "fr_pre": pre_v,
                    "fr_post": post_v,
                    "modulation": mod,
                }
            )
    print(f"[ifr] wrote summary: {out_csv}")
    return out_csv


def process_all_ifr_npz(
    npz_files: Iterable[Path] | None = None,
    cfg: IFRProcessorConfig | None = None,
) -> Tuple[List[Path], Path, Path]:
    """Process many IFR NPZs and rebuild a global catalog/status.

    Returns (per_recording_csvs, catalog_csv, status_csv)
    """
    cfg = cfg or IFRProcessorConfig()
    files = list(npz_files) if npz_files is not None else find_ifr_npz(cfg.output_root)
    out_csvs: List[Path] = []
    for p in files:
        try:
            res = process_ifr_npz(Path(p), cfg)
            if res is not None:
                out_csvs.append(res)
        except Exception as e:
            print(f"[ifr] process error: {p} -> {e}")

    # Rebuild catalog
    cat_dir = cfg.output_root / "ifr_npz_catalog"
    cat_dir.mkdir(parents=True, exist_ok=True)
    cat_csv = cat_dir / "ifr_npz_catalog.csv"
    cat_jsonl = cat_dir / "ifr_npz_catalog.jsonl"
    status_csv = cat_dir / "ifr_npz_status.csv"

    rows: List[Dict[str, object]] = []
    for rec_csv in out_csvs:
        try:
            with rec_csv.open("r", encoding="utf-8", newline="") as f:
                rdr = csv.DictReader(f)
                for r in rdr:
                    rows.append(r)
        except Exception:
            continue

    # Write catalog
    with cat_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "round",
                "group_label",
                "recording_stem",
                "channel",
                "fr_full",
                "fr_pre",
                "fr_post",
                "modulation",
            ],
        )
        w.writeheader()
        for r in rows:
            w.writerow(r)
    with cat_jsonl.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    # Status aggregation
    agg: Dict[str, Dict[str, object]] = {}
    for r in rows:
        stem = str(r.get("recording_stem", ""))
        if not stem:
            continue
        d = agg.setdefault(
            stem,
            {
                "round": r.get("round", ""),
                "group_label": r.get("group_label", ""),
                "n_channels": 0,
                "n_pos": 0,
                "n_neg": 0,
                "n_nochange": 0,
            },
        )
        d["n_channels"] = int(d.get("n_channels", 0)) + 1
        m = str(r.get("modulation", ""))
        if m == "positive":
            d["n_pos"] = int(d.get("n_pos", 0)) + 1
        elif m == "negative":
            d["n_neg"] = int(d.get("n_neg", 0)) + 1
        else:
            d["n_nochange"] = int(d.get("n_nochange", 0)) + 1

    with status_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "recording_stem",
                "round",
                "group_label",
                "n_channels",
                "n_pos",
                "n_neg",
                "n_nochange",
            ],
        )
        w.writeheader()
        for stem, d in sorted(agg.items()):
            w.writerow({"recording_stem": stem, **d})

    print(f"[ifr] catalog: rows={len(rows)} -> {cat_csv} and {cat_jsonl}; status -> {status_csv}")
    return out_csvs, cat_csv, status_csv

