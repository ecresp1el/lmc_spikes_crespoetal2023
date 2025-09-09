from __future__ import annotations

"""
Unified analysis manifest builder.

Produces a single, canonical file that tracks what can and cannot be analyzed,
and why. It merges information from:

- Latest file index JSON (discovery/probe): path, eligibility, basic info
- Annotations: presence of chemical/opto/manual per recording, ignored flag
- FR outputs: presence of per-recording FR results and IFR NPZ

Outputs
- <output_root>/analysis_manifest.csv
- <output_root>/analysis_manifest.jsonl

This module is GUI-independent and can be run standalone.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import csv
import json

from .config import CONFIG
from .metadata import GroupLabeler


def _latest_index_json(output_root: Path) -> Optional[Path]:
    probe_dir = output_root / "probe"
    if not probe_dir.exists():
        return None
    files = sorted(probe_dir.glob("file_index_*.json"))
    return files[-1] if files else None


def _load_index_items(index_json: Path) -> List[Dict[str, object]]:
    try:
        data = json.loads(index_json.read_text())
        return [it for it in data.get("files", []) if isinstance(it, dict)]
    except Exception:
        return []


def _annotations_categories_for(stem: str, annotations_root: Path) -> Tuple[bool, bool, bool]:
    has_opto = False
    has_chem = False
    has_manual = False
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
                elif cat == "opto":
                    has_opto = True
                else:
                    has_manual = True
        except Exception:
            continue
    return has_opto, has_chem, has_manual


def _load_overrides(annotations_root: Path) -> Dict[str, Dict[str, object]]:
    p = annotations_root / "annotations_overrides.json"
    try:
        return json.loads(p.read_text()) if p.exists() else {}
    except Exception:
        return {}


def _fr_outputs_present(output_root: Path, rec_path: Path) -> Tuple[bool, bool]:
    gi = GroupLabeler.infer_from_path(rec_path)
    round_name = gi.round_name or "unknown_round"
    group_label = gi.label or "UNKNOWN"
    fr_dir = output_root / "plots" / round_name / group_label / rec_path.stem / "fr"
    fr_summary = fr_dir / f"{rec_path.stem}_fr_summary.csv"
    ifr_npz = fr_dir / f"{rec_path.stem}_ifr_per_channel_1ms.npz"
    return fr_summary.exists(), ifr_npz.exists()


def build_manifest(
    output_root: Path | None = None,
    require_opto: bool = True,
) -> Tuple[Path, Path]:
    out_root = output_root or CONFIG.output_root
    annotations_root = CONFIG.annotations_root
    idx = _latest_index_json(out_root)
    if idx is None:
        raise FileNotFoundError("No index JSON found under output_root/probe/")
    items = _load_index_items(idx)
    overrides = _load_overrides(annotations_root)

    rows: List[Dict[str, object]] = []
    for it in items:
        rec_path = Path(str(it.get("path", "")))
        stem = rec_path.stem
        has_opto, has_chem, has_manual = _annotations_categories_for(stem, annotations_root)
        ign = bool((overrides.get(str(rec_path)) or {}).get("ignored", False))
        fr_present, ifr_npz_present = _fr_outputs_present(out_root, rec_path)
        eligible = bool(it.get("eligible_10khz_ge300s", False))
        ready = bool(has_chem and ((has_opto or (not require_opto))) and (not ign) and eligible and fr_present and ifr_npz_present)

        rows.append(
            {
                "path": str(rec_path),
                "recording_stem": stem,
                "round": it.get("round"),
                "group_label": it.get("group_label"),
                "plate": it.get("plate"),
                "timestamp": it.get("timestamp"),
                "eligible": eligible,
                "sampling_rate_hz": it.get("sampling_rate_hz"),
                "n_channels": it.get("n_channels"),
                "duration_seconds": it.get("duration_seconds"),
                "has_opto": has_opto,
                "has_chemical": has_chem,
                "has_manual": has_manual,
                "ignored": ign,
                "fr_present": fr_present,
                "ifr_npz_present": ifr_npz_present,
                "ready_for_processing": ready,
            }
        )

    man_csv = out_root / "analysis_manifest.csv"
    man_jsonl = out_root / "analysis_manifest.jsonl"

    fields = [
        "path",
        "recording_stem",
        "round",
        "group_label",
        "plate",
        "timestamp",
        "eligible",
        "sampling_rate_hz",
        "n_channels",
        "duration_seconds",
        "has_opto",
        "has_chemical",
        "has_manual",
        "ignored",
        "fr_present",
        "ifr_npz_present",
        "ready_for_processing",
    ]
    with man_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    with man_jsonl.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    print(f"[manifest] wrote -> {man_csv} and {man_jsonl}")
    return man_csv, man_jsonl

