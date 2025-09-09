from __future__ import annotations

"""
CLI: Compute FR outputs (including IFR NPZ) without the GUI.

Uses annotations to find chemical timestamps and the latest index to discover
recordings. Skips recordings that already have FR outputs, unless --force.

Usage:
  python scripts/run_fr_batch.py                     # process all chem-stamped
  python scripts/run_fr_batch.py --force             # force recompute all
  python scripts/run_fr_batch.py <recording.h5>      # single file
  python scripts/run_fr_batch.py <out_root> [--force]
"""

import sys
from pathlib import Path
import json
import csv

from mcs_mea_analysis.config import CONFIG
from mcs_mea_analysis.fr_plots import compute_and_save_fr
from mcs_mea_analysis.metadata import GroupLabeler
from mcs_mea_analysis.manifest import build_manifest


def latest_index_json(output_root: Path) -> Path | None:
    probe_dir = output_root / "probe"
    if not probe_dir.exists():
        return None
    files = sorted(probe_dir.glob("file_index_*.json"))
    return files[-1] if files else None


def chem_time_for_path(rec_path: Path, annotations_root: Path) -> float | None:
    stem = rec_path.stem
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


def has_fr_outputs(rec_path: Path, output_root: Path) -> bool:
    gi = GroupLabeler.infer_from_path(rec_path)
    round_name = gi.round_name or "unknown_round"
    group_label = gi.label or "UNKNOWN"
    fr_dir = output_root / "plots" / round_name / group_label / rec_path.stem / "fr"
    csv_p = fr_dir / f"{rec_path.stem}_fr_summary.csv"
    pdf_p = fr_dir / f"{rec_path.stem}_fr_overview.pdf"
    return csv_p.exists() and pdf_p.exists()


def main() -> None:
    args = [a for a in sys.argv[1:] if a]
    force = False
    out_root = CONFIG.output_root
    single: Path | None = None

    # Parse args
    for a in list(args):
        if a == "--force":
            force = True
            args.remove(a)
    if args:
        p = Path(args[0])
        if p.suffix.lower() == ".h5":
            single = p
        else:
            out_root = p

    ann_root = CONFIG.annotations_root

    # Single file path
    if single is not None:
        if not single.exists():
            print(f"[fr-cli] missing: {single}")
            sys.exit(1)
        chem = chem_time_for_path(single, ann_root)
        if chem is None:
            print(f"[fr-cli] no chem stamp: {single}")
            sys.exit(2)
        if (not force) and has_fr_outputs(single, out_root):
            print(f"[fr-cli] skip (already has outputs): {single}")
            sys.exit(0)
        print(f"[fr-cli] start: {single} chem={chem:.6f}s force={force}")
        res = compute_and_save_fr(single, chem, out_root)
        if res:
            print(f"[fr-cli] done -> {res.out_dir}")
            build_manifest(out_root)
        sys.exit(0)

    # Batch from latest index
    idx = latest_index_json(out_root)
    if idx is None:
        print(f"[fr-cli] no index JSON found under {out_root}/probe")
        sys.exit(3)
    try:
        data = json.loads(idx.read_text())
        items = [it for it in data.get("files", []) if isinstance(it, dict)]
    except Exception as e:
        print(f"[fr-cli] index load failed: {idx} -> {e}")
        sys.exit(4)

    todo: list[Path] = []
    for it in items:
        p = Path(str(it.get("path", "")))
        if not p.exists():
            print(f"[fr-cli] skip (missing): {p}")
            continue
        chem = chem_time_for_path(p, ann_root)
        if chem is None:
            print(f"[fr-cli] skip (no chem): {p}")
            continue
        if (not force) and has_fr_outputs(p, out_root):
            print(f"[fr-cli] skip (has outputs): {p}")
            continue
        todo.append(p)

    print(f"[fr-cli] queued: {len(todo)} recording(s)")
    for p in todo:
        chem = chem_time_for_path(p, ann_root)
        print(f"[fr-cli] start: {p} chem={chem:.6f}s force={force}")
        try:
            res = compute_and_save_fr(p, chem, out_root)
            if res:
                print(f"[fr-cli] done -> {res.out_dir}")
        except Exception as e:
            print(f"[fr-cli] error: {p} -> {e}")

    # Rebuild manifest at end
    try:
        build_manifest(out_root)
    except Exception as e:
        print(f"[fr-cli] manifest rebuild failed -> {e}")


if __name__ == "__main__":
    main()

