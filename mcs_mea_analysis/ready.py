from __future__ import annotations

"""
Readiness index for downstream analysis (GUI‑independent).

Purpose
- Provide a single, standardized way to decide which recordings are ready to
  analyze based on curated annotations and computed NPZ outputs.
- Expandable/configurable via ReadinessConfig flags.

Default readiness
- chemical stamp saved (annotations) AND
- IFR NPZ present (per‑channel 1 ms: *_ifr_per_channel_1ms.npz)

Optional constraints (configurable)
- require_opto: also require opto stamps
- require_eligible: only use index‑eligible (10 kHz, >= 300 s)
- require_fr_summary: also require *_fr_summary.csv
- require_not_ignored: exclude ignored recordings

Outputs
- <output_root>/analysis_ready/ready_index.{csv,jsonl}
  Each row includes: path, recording_stem, round, group_label, chem_timestamp,
  npz_path, annotations_path, and useful index metadata.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import csv
import json

from .config import CONFIG
from .metadata import GroupLabeler


@dataclass(frozen=True)
class ReadinessConfig:
    output_root: Path = CONFIG.output_root
    annotations_root: Path = CONFIG.annotations_root
    require_chemical: bool = True
    require_opto: bool = False
    require_not_ignored: bool = True
    require_eligible: bool = False
    require_ifr_npz: bool = True
    require_fr_summary: bool = False


def _latest_index_json(output_root: Path) -> Optional[Path]:
    p = output_root / "probe"
    if not p.exists():
        return None
    files = sorted(p.glob("file_index_*.json"))
    return files[-1] if files else None


def _load_index(index_json: Path) -> List[Dict[str, object]]:
    try:
        data = json.loads(index_json.read_text())
        return [it for it in data.get("files", []) if isinstance(it, dict)]
    except Exception:
        return []


def _overrides(annotations_root: Path) -> Dict[str, Dict[str, object]]:
    p = annotations_root / "annotations_overrides.json"
    try:
        return json.loads(p.read_text()) if p.exists() else {}
    except Exception:
        return {}


def _cats_for(stem: str, annotations_root: Path) -> Tuple[bool, bool, bool, Optional[float], Optional[Path]]:
    has_chem = False
    has_opto = False
    has_manual = False
    chem_ts: Optional[float] = None
    ann_used: Optional[Path] = None
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
                cat = str(row.get("category", "manual")).lower()
                if cat == "chemical":
                    has_chem = True
                    if chem_ts is None:
                        try:
                            chem_ts = float(row.get("timestamp", 0.0))
                        except Exception:
                            chem_ts = None
                elif cat == "opto":
                    has_opto = True
                else:
                    has_manual = True
            if has_chem or has_opto or has_manual:
                ann_used = p
        except Exception:
            continue
    return has_chem, has_opto, has_manual, chem_ts, ann_used


def _fr_paths(output_root: Path, rec_path: Path) -> Tuple[Path, Path]:
    gi = GroupLabeler.infer_from_path(rec_path)
    round_name = gi.round_name or "unknown_round"
    group_label = gi.label or "UNKNOWN"
    fr_dir = output_root / "plots" / round_name / group_label / rec_path.stem / "fr"
    return (
        fr_dir / f"{rec_path.stem}_ifr_per_channel_1ms.npz",
        fr_dir / f"{rec_path.stem}_fr_summary.csv",
    )


def build_ready_index(cfg: ReadinessConfig | None = None) -> Tuple[Path, Path, List[Dict[str, object]]]:
    cfg = cfg or ReadinessConfig()
    idx = _latest_index_json(cfg.output_root)
    if idx is None:
        raise FileNotFoundError("No file_index_*.json found under output_root/probe")
    items = _load_index(idx)
    ov = _overrides(cfg.annotations_root)

    rows: List[Dict[str, object]] = []
    for it in items:
        rec_path = Path(str(it.get("path", "")))
        stem = rec_path.stem
        cats = _cats_for(stem, cfg.annotations_root)
        has_chem, has_opto, has_manual, chem_ts, ann_path = cats
        eligible = bool(it.get("eligible_10khz_ge300s", False))
        ignored = bool((ov.get(str(rec_path)) or {}).get("ignored", False))
        npz_path, fr_csv = _fr_paths(cfg.output_root, rec_path)
        npz_ok = npz_path.exists() if cfg.require_ifr_npz else True
        fr_ok = fr_csv.exists() if cfg.require_fr_summary else True

        ready = True
        if cfg.require_chemical and not has_chem:
            ready = False
        if cfg.require_opto and not has_opto:
            ready = False
        if cfg.require_not_ignored and ignored:
            ready = False
        if cfg.require_eligible and not eligible:
            ready = False
        if not npz_ok or not fr_ok:
            ready = False

        rows.append(
            {
                "path": str(rec_path),
                "recording_stem": stem,
                "round": it.get("round"),
                "group_label": it.get("group_label"),
                "eligible": eligible,
                "ignored": ignored,
                "has_chemical": has_chem,
                "has_opto": has_opto,
                "has_manual": has_manual,
                "chem_timestamp": chem_ts,
                "npz_path": str(npz_path) if npz_path.exists() else "",
                "fr_summary": str(fr_csv) if fr_csv.exists() else "",
                "annotations_path": str(ann_path) if ann_path else "",
                "ready": ready,
            }
        )

    out_dir = cfg.output_root / "analysis_ready"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "ready_index.csv"
    jsonl_path = out_dir / "ready_index.jsonl"
    fields = [
        "path",
        "recording_stem",
        "round",
        "group_label",
        "eligible",
        "ignored",
        "has_chemical",
        "has_opto",
        "has_manual",
        "chem_timestamp",
        "npz_path",
        "fr_summary",
        "annotations_path",
        "ready",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    with jsonl_path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"[ready] wrote -> {csv_path} and {jsonl_path}")
    return csv_path, jsonl_path, rows

