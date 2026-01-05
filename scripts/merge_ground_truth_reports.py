#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge raw pixel stats with OME metadata.")
    parser.add_argument(
        "--ground-truth-csv",
        type=Path,
        required=True,
        help="nd2_ground_truth_channels.csv path.",
    )
    parser.add_argument(
        "--ome-csv",
        type=Path,
        required=True,
        help="ome_metadata_channels.csv path.",
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        required=True,
        help="Output merged CSV path.",
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        required=True,
        help="Output merged JSON path.",
    )
    return parser.parse_args()


def normalize_name(name: str) -> str:
    return name.strip().lower() if name else ""


def load_csv(path: Path) -> List[Dict[str, str]]:
    with path.open() as f:
        reader = csv.DictReader(f)
        return list(reader)


def main() -> None:
    args = parse_args()
    gt_rows = load_csv(args.ground_truth_csv)
    ome_rows = load_csv(args.ome_csv)

    ome_map: Dict[Tuple[str, str, str], Dict[str, str]] = {}
    for row in ome_rows:
        key = (row.get("file", ""), row.get("channel_index", ""), normalize_name(row.get("channel_name", "")))
        ome_map[key] = row

    merged_rows: List[Dict[str, str]] = []
    for row in gt_rows:
        key = (row.get("file", ""), row.get("channel_index", ""), normalize_name(row.get("channel_name", "")))
        merged = dict(row)
        ome = ome_map.get(key)
        if ome:
            for k, v in ome.items():
                if k in ("file", "channel_index", "channel_name"):
                    continue
                merged[f"ome_{k}"] = v
        merged_rows.append(merged)

    headers = sorted({key for row in merged_rows for key in row.keys()})
    with args.out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(merged_rows)

    args.out_json.write_text(json.dumps(merged_rows, indent=2))
    print(f"Saved merged CSV to {args.out_csv}")
    print(f"Saved merged JSON to {args.out_json}")


if __name__ == "__main__":
    main()
