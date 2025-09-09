from __future__ import annotations

"""
NPZ-based statistical analysis (headless).

Given a per-channel 1 ms IFR NPZ and a chemical timestamp, compute per-channel
pre/post statistics with configurable smoothing, windows, and tests. Produces a
per-recording CSV and can be aggregated into a global catalog.

This module is compute-only; plotting lives in `ifr_analysis.py`.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
import csv
import json
import numpy as np
from scipy import stats  # type: ignore

from .analysis_config import NPZAnalysisConfig
from .config import CONFIG


def _chem_time_for_stem(stem: str, annotations_root: Path) -> Optional[float]:
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
                if str(row.get("category", "manual")).lower() == "chemical":
                    return float(row.get("timestamp", 0.0))
        except Exception:
            continue
    return None


def _label_significance(pre: float, post: float, d: float, p: float, cfg: NPZAnalysisConfig) -> str:
    if cfg.require_both:
        if (p < cfg.alpha) and (abs(d) >= cfg.effect_size_thr):
            return "pos" if (post - pre) > 0 else "neg"
        return "ns"
    else:
        if p < cfg.alpha:
            return "pos" if (post - pre) > 0 else "neg"
        if abs(d) >= cfg.effect_size_thr:
            return "pos" if (post - pre) > 0 else "neg"
        return "ns"


def analyze_npz(npz_path: Path, chem_ts: float, cfg: NPZAnalysisConfig | None = None) -> Optional[Path]:
    """Analyze one NPZ file and write per-recording stats CSV.

    Returns the path to the CSV or None on failure.
    """
    cfg = cfg or NPZAnalysisConfig()
    npz_path = Path(npz_path)
    try:
        d = np.load(npz_path)
        time_s = np.asarray(d["time_s"], dtype=float)  # (n_bins,)
        ifr = np.asarray(d["ifr_hz"], dtype=float)     # (n_ch, n_bins)
    except Exception as e:
        print(f"[npz-stats] load failed: {npz_path} -> {e}")
        return None
    if time_s.size == 0 or ifr.size == 0:
        print(f"[npz-stats] empty arrays: {npz_path}")
        return None

    # Smooth with provided kernel
    k = cfg.kernel_np
    ifr_s = np.apply_along_axis(lambda x: np.convolve(x, k, mode='same'), 1, ifr)

    # Windows
    t0_pre = max(0.0, chem_ts - cfg.pre_span_s)
    t1_pre = chem_ts
    t0_post = chem_ts
    t1_post = min(float(time_s[-1]), chem_ts + cfg.post_span_s)
    pre_mask = (time_s >= t0_pre) & (time_s < t1_pre)
    post_mask = (time_s >= t0_post) & (time_s < t1_post)
    if not np.any(pre_mask) or not np.any(post_mask):
        print(f"[npz-stats] invalid windows for {npz_path.name}: pre({t0_pre},{t1_pre}) post({t0_post},{t1_post})")
        return None

    step = cfg.decimate_steps

    # Stats per channel
    stem = npz_path.stem.replace("_ifr_per_channel_1ms", "")
    out_csv = npz_path.parent / f"{stem}_npz_stats.csv"
    fields = [
        "channel", "fr_pre", "fr_post", "delta", "rel_change", "cohen_d", "p_value", "sig_label",
        "t0_pre", "t1_pre", "t0_post", "t1_post", "kernel", "alpha", "effect_thr", "require_both",
    ]
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        n_ch = ifr_s.shape[0]
        for ch in range(n_ch):
            pre_vals = ifr_s[ch, pre_mask][::step]
            post_vals = ifr_s[ch, post_mask][::step]
            pre_mean = float(np.nanmean(pre_vals))
            post_mean = float(np.nanmean(post_vals))
            delta = post_mean - pre_mean
            rel = (delta / pre_mean) if pre_mean > 0 else (np.inf if delta > 0 else (-np.inf if delta < 0 else 0.0))
            # Cohen's d (pooled std)
            s1 = float(np.nanstd(pre_vals, ddof=1))
            s2 = float(np.nanstd(post_vals, ddof=1))
            n1 = max(1, pre_vals.size)
            n2 = max(1, post_vals.size)
            s_pooled = np.sqrt(((n1 - 1) * s1 ** 2 + (n2 - 1) * s2 ** 2) / max(1, (n1 + n2 - 2)))
            d_val = (delta / s_pooled) if s_pooled > 0 else (np.inf if delta > 0 else (-np.inf if delta < 0 else 0.0))
            # Nonparametric test on decimated samples
            try:
                u_stat, p_val = stats.mannwhitneyu(pre_vals, post_vals, alternative='two-sided')
                p_val = float(p_val)
            except Exception:
                p_val = 1.0
            label = _label_significance(pre_mean, post_mean, d_val, p_val, cfg)
            w.writerow(
                {
                    "channel": ch,
                    "fr_pre": pre_mean,
                    "fr_post": post_mean,
                    "delta": delta,
                    "rel_change": rel,
                    "cohen_d": d_val,
                    "p_value": p_val,
                    "sig_label": label,
                    "t0_pre": t0_pre,
                    "t1_pre": t1_pre,
                    "t0_post": t0_post,
                    "t1_post": t1_post,
                    "kernel": ",".join(map(str, cfg.smooth_kernel)),
                    "alpha": cfg.alpha,
                    "effect_thr": cfg.effect_size_thr,
                    "require_both": cfg.require_both,
                }
            )
    print(f"[npz-stats] wrote -> {out_csv}")
    return out_csv


def analyze_ready_rows(rows: List[Dict[str, object]], cfg: NPZAnalysisConfig | None = None, force: bool = False) -> List[Path]:
    cfg = cfg or NPZAnalysisConfig()
    out_csvs: List[Path] = []
    for r in rows:
        npz_path = Path(str(r.get("npz_path", "")))
        if not npz_path.exists():
            print(f"[npz-stats] skip (missing NPZ): {npz_path}")
            continue
        stem = str(r.get("recording_stem", ""))
        chem_ts = r.get("chem_timestamp")
        try:
            chem = float(chem_ts) if chem_ts is not None else _chem_time_for_stem(stem, cfg.annotations_root)
        except Exception:
            chem = None
        if chem is None:
            print(f"[npz-stats] skip (no chem): {stem}")
            continue
        out_csv = npz_path.parent / f"{stem}_npz_stats.csv"
        if out_csv.exists() and not force:
            print(f"[npz-stats] skip (exists): {out_csv}")
            out_csvs.append(out_csv)
            continue
        try:
            res = analyze_npz(npz_path, chem, cfg)
            if res:
                out_csvs.append(res)
        except Exception as e:
            print(f"[npz-stats] error: {npz_path} -> {e}")
    return out_csvs

