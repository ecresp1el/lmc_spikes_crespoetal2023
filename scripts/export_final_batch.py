#!/usr/bin/env python3
from __future__ import annotations

import argparse
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


if __name__ == "__main__":
    main()
