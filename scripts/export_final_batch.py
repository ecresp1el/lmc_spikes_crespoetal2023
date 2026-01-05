#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import List


DEFAULT_INSET_ARGS = [
    "--eyfp-from-report",
    "--inset-count",
    "1",
    "--inset-size",
    "125",
    "--inset-target",
    "eyfp",
    "el222",
    "--sigma",
    "0.8",
    "--include-dapi",
    "--illumination-sigma",
    "50",
    "--illumination-channels",
    "DAPI",
    "EYFP",
    "EL222",
    "--bg-pct",
    "25",
    "--bg-channels",
    "DAPI",
    "EYFP",
    "EL222",
    "--clip-below-pct",
    "18",
    "--clip-channels",
    "DAPI",
    "EYFP",
    "EL222",
    "--eyfp-bg-pct",
    "75",
    "--eyfp-clip-pct",
    "60",
    "--el222-bg-pct",
    "90",
    "--el222-clip-pct",
    "90",
]


def _pairs_from_args(args: List[str]) -> Dict[str, str | bool]:
    pairs: Dict[str, str | bool] = {}
    idx = 0
    while idx < len(args):
        arg = args[idx]
        if arg.startswith("--"):
            if idx + 1 < len(args) and not args[idx + 1].startswith("--"):
                pairs[arg] = args[idx + 1]
                idx += 2
                continue
            pairs[arg] = True
        idx += 1
    return pairs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="One-shot export: build pooled grid report, then batch insets + combined TIFF.",
    )
    parser.add_argument("--roi-root", type=Path, required=True, help="ROI root folder.")
    parser.add_argument("--nd2-dir", type=Path, required=True, help="ND2 root folder.")
    parser.add_argument(
        "--groups",
        nargs="+",
        default=["ctznmda", "ctz", "nmda"],
        help="Groups to process (default: ctznmda ctz nmda).",
    )
    parser.add_argument(
        "--eyfp-ref-groups",
        nargs="+",
        default=["ctz", "nmda"],
        help="Groups to pool for EYFP bounds (default: ctz nmda).",
    )
    parser.add_argument(
        "--tdtom-ref-groups",
        nargs="+",
        default=["ctznmda", "ctz", "nmda"],
        help="Groups to pool for tdTom bounds (default: ctznmda ctz nmda).",
    )
    parser.add_argument(
        "--grid-tag",
        type=str,
        default="pooled_eyfp_tdtom",
        help="Output tag for the group grid report folder.",
    )
    parser.add_argument(
        "--padding",
        type=int,
        default=20,
        help="Padding between combined rows (pixels).",
    )
    parser.add_argument(
        "--pick-per-group",
        action="store_true",
        help="Prompt for a ROI folder per group.",
    )
    return parser.parse_known_args()


def main() -> None:
    args, extra_args = parse_args()
    extra_args = [arg for arg in extra_args if arg != "--"]
    inset_args = extra_args if extra_args else DEFAULT_INSET_ARGS
    if "--include-dapi" not in inset_args:
        inset_args = inset_args + ["--include-dapi"]

    root = args.roi_root.expanduser()
    nd2_dir = args.nd2_dir.expanduser()
    grid_tag = args.grid_tag.strip()
    group_args = [item for item in args.groups if item.strip()]

    group_script = Path(__file__).with_name("export_group_eyfp_tdtom_grid.py")
    inset_script = Path(__file__).with_name("export_inset_batch.py")

    grid_cmd: List[str] = [
        sys.executable,
        str(group_script),
        "--roi-root",
        str(root),
        "--nd2-dir",
        str(nd2_dir),
        "--groups",
        *group_args,
        "--select",
        "DAPI",
        "EYFP",
        "tdTom",
        "EL222",
        "--use-last",
        "--eyfp-ref-groups",
        *args.eyfp_ref_groups,
        "--tdtom-ref-groups",
        *args.tdtom_ref_groups,
        "--grid-only",
        "--out-tag",
        grid_tag,
    ]
    if args.pick_per_group:
        grid_cmd.append("--pick-per-group")

    subprocess.run(grid_cmd, check=True)

    report_path = root / "nd2_export" / f"group_grid_export_{grid_tag}" / "export_report.json"

    inset_cmd: List[str] = [
        sys.executable,
        str(inset_script),
        "--roi-root",
        str(root),
        "--groups",
        *group_args,
        "--nd2-dir",
        str(nd2_dir),
        "--report-path",
        str(report_path),
        "--padding",
        str(args.padding),
    ]
    inset_cmd.extend(inset_args)

    subprocess.run(inset_cmd, check=True)

    default_inset = DEFAULT_INSET_ARGS + ["--include-dapi"]
    default_inset_map = _pairs_from_args(default_inset)
    inset_map = _pairs_from_args(inset_args)
    inset_overrides = {
        key: value
        for key, value in inset_map.items()
        if key not in default_inset_map or default_inset_map.get(key) != value
    }
    grid_defaults = {
        "groups": ["ctznmda", "ctz", "nmda"],
        "eyfp_ref_groups": ["ctz", "nmda"],
        "tdtom_ref_groups": ["ctznmda", "ctz", "nmda"],
        "grid_tag": "pooled_eyfp_tdtom",
        "padding": 20,
    }
    grid_actual = {
        "groups": group_args,
        "eyfp_ref_groups": args.eyfp_ref_groups,
        "tdtom_ref_groups": args.tdtom_ref_groups,
        "grid_tag": grid_tag,
        "padding": args.padding,
    }
    grid_overrides = {
        key: value for key, value in grid_actual.items() if grid_defaults.get(key) != value
    }

    summary = {
        "roi_root": str(root),
        "nd2_dir": str(nd2_dir),
        "groups": group_args,
        "eyfp_ref_groups": args.eyfp_ref_groups,
        "tdtom_ref_groups": args.tdtom_ref_groups,
        "grid_tag": grid_tag,
        "grid_defaults": grid_defaults,
        "grid_overrides": grid_overrides,
        "grid_command": grid_cmd,
        "inset_command": inset_cmd,
        "inset_defaults": default_inset_map,
        "inset_overrides": inset_overrides,
        "report_path": str(report_path),
    }
    summary_path = root / "nd2_export" / "final_export_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"Saved final export summary to {summary_path}")
    print("Defaults used for grid:", grid_defaults)
    print("Grid overrides:", grid_overrides or "none")
    print("Defaults used for insets:", default_inset_map)
    print("Inset overrides:", inset_overrides or "none")


if __name__ == "__main__":
    main()
