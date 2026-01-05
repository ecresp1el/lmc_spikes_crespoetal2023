#!/usr/bin/env python3
from __future__ import annotations

import argparse
import dataclasses
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np

try:
    from nd2 import ND2File
except ImportError:  # pragma: no cover - optional ND2 support
    ND2File = None

try:
    from scipy.ndimage import rotate
except ImportError:  # pragma: no cover - optional rotation
    rotate = None


def _normalize_axes(axes) -> List[str]:
    normalized = []
    for axis in axes:
        axis_name = axis.name if hasattr(axis, "name") else str(axis)
        if axis_name.startswith("Axis."):
            axis_name = axis_name.split(".")[-1]
        normalized.append(axis_name.upper())
    return normalized


def _nd2_axes(nd2_file: ND2File) -> List[str]:
    axes = getattr(nd2_file, "axes", None)
    if axes:
        return _normalize_axes(axes)

    sizes = getattr(nd2_file, "sizes", None)
    if sizes is None:
        metadata = getattr(nd2_file, "metadata", None)
        sizes = getattr(metadata, "sizes", None) if metadata else None
    if sizes is None:
        raise AttributeError("ND2File has no axes metadata.")

    if isinstance(sizes, dict):
        return _normalize_axes(sizes.keys())
    if hasattr(sizes, "_fields"):
        return _normalize_axes(sizes._fields)
    if hasattr(sizes, "axes"):
        size_axes = sizes.axes
        if isinstance(size_axes, str):
            return _normalize_axes(size_axes)
        return _normalize_axes(size_axes)
    try:
        size_list = list(sizes)
    except TypeError as exc:
        raise AttributeError("Unsupported ND2 sizes metadata.") from exc
    if size_list and isinstance(size_list[0], (tuple, list)) and len(size_list[0]) == 2:
        return _normalize_axes([axis for axis, _ in size_list])
    return _normalize_axes(size_list)


def load_nd2_channels(path: Path, channel_indices: List[int] | None, z_project: str) -> np.ndarray:
    if ND2File is None:
        raise RuntimeError("nd2 is not installed. Install with: pip install nd2")
    with ND2File(str(path)) as nd2_file:
        data = nd2_file.asarray()
        axes = _nd2_axes(nd2_file)

    data = np.asarray(data)

    def take_axis(axis: str, reducer=None) -> None:
        nonlocal data, axes
        if axis not in axes:
            return
        axis_index = axes.index(axis)
        if reducer is None:
            data = np.take(data, 0, axis=axis_index)
        else:
            data = reducer(data, axis=axis_index)
        axes.pop(axis_index)

    take_axis("T")
    if z_project == "max":
        take_axis("Z", reducer=np.max)
    else:
        take_axis("Z")

    if "C" not in axes or "Y" not in axes or "X" not in axes:
        raise ValueError(f"Unexpected ND2 axes: {axes}")

    c_idx, y_idx, x_idx = axes.index("C"), axes.index("Y"), axes.index("X")
    data = np.moveaxis(data, [c_idx, y_idx, x_idx], [0, 1, 2])

    if data.ndim != 3:
        raise ValueError(f"Unexpected ND2 shape after squeeze: {data.shape}")

    if channel_indices:
        data = data[channel_indices]

    return data.astype(np.float32)


def extract_rotated_roi(image: np.ndarray, center: Tuple[float, float], width: float, height: float, angle: float) -> np.ndarray:
    if image.ndim == 3:
        return np.stack(
            [extract_rotated_roi(image[idx], center, width, height, angle) for idx in range(image.shape[0])],
            axis=0,
        )

    img_h, img_w = image.shape
    cx, cy = center
    radius = int(np.ceil(0.5 * np.hypot(width, height)))
    x0, x1 = int(cx - radius), int(cx + radius)
    y0, y1 = int(cy - radius), int(cy + radius)
    x0 = max(0, x0)
    y0 = max(0, y0)
    x1 = min(img_w, x1)
    y1 = min(img_h, y1)
    if x1 <= x0 or y1 <= y0:
        return np.zeros((1, 1), dtype=image.dtype)

    cropped = image[y0:y1, x0:x1]
    if angle and rotate is not None:
        rotated = rotate(cropped, angle=angle, reshape=True, order=1, mode="nearest")
    else:
        rotated = cropped

    rot_h, rot_w = rotated.shape
    cx_local = rot_w / 2.0
    cy_local = rot_h / 2.0
    half_w = width / 2.0
    half_h = height / 2.0
    crop_x0 = int(round(cx_local - half_w))
    crop_x1 = int(round(cx_local + half_w))
    crop_y0 = int(round(cy_local - half_h))
    crop_y1 = int(round(cy_local + half_h))

    crop_x0 = max(0, crop_x0)
    crop_y0 = max(0, crop_y0)
    crop_x1 = min(rotated.shape[1], crop_x1)
    crop_y1 = min(rotated.shape[0], crop_y1)
    if crop_x1 <= crop_x0 or crop_y1 <= crop_y0:
        return np.zeros((1, 1), dtype=rotated.dtype)
    return rotated[crop_y0:crop_y1, crop_x0:crop_x1]


def find_group_roi_dir(group_dir: Path, roi_name: str | None) -> Path:
    if roi_name:
        candidate = group_dir / roi_name
        if (candidate / "roi_state.json").exists():
            return candidate
        raise FileNotFoundError(f"ROI folder {roi_name} not found under {group_dir}.")
    if (group_dir / "roi_state.json").exists():
        return group_dir
    matches = list(group_dir.rglob("roi_state.json"))
    if len(matches) == 1:
        return matches[0].parent
    if not matches:
        raise FileNotFoundError(f"No roi_state.json found under {group_dir}.")
    sorted_matches = sorted(matches, key=lambda p: p.stat().st_mtime, reverse=True)
    return sorted_matches[0].parent


def resolve_nd2_path(roi_dir: Path, nd2_dir: Path | None) -> Path:
    roi_name = roi_dir.name
    stem = roi_name[:-4] if roi_name.endswith("_roi") else roi_name
    search_root = nd2_dir
    if search_root is None:
        try:
            search_root = roi_dir.parents[2]
        except IndexError:
            search_root = roi_dir.parent
    candidate = search_root / f"{stem}.nd2"
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"ND2 file not found for ROI stem {stem} in {search_root}")


def to_jsonable(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return to_jsonable(dataclasses.asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        if value.size <= 1000:
            return value.tolist()
        return {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "min": float(np.min(value)),
            "max": float(np.max(value)),
        }
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if hasattr(value, "_asdict"):
        return to_jsonable(value._asdict())
    if hasattr(value, "__dict__"):
        return to_jsonable(value.__dict__)
    return str(value)


def summarize_channel(channel: np.ndarray, percentiles: Iterable[float]) -> Dict[str, float]:
    stats = {
        "min": float(np.min(channel)),
        "max": float(np.max(channel)),
        "mean": float(np.mean(channel)),
        "median": float(np.median(channel)),
    }
    pct_values = np.percentile(channel, list(percentiles))
    for pct, val in zip(percentiles, pct_values):
        stats[f"p{pct:g}"] = float(val)
    return stats


def extract_channel_metadata(meta: Any) -> List[Dict[str, Any]]:
    channels = getattr(meta, "channels", None)
    if not channels:
        return []
    out = []
    for ch in channels:
        info: Dict[str, Any] = {}
        for key in (
            "name",
            "channel",
            "index",
            "excitation_wavelength",
            "emission_wavelength",
            "exposure_ms",
            "exposure",
            "exposure_time_ms",
            "laser_power",
            "power",
        ):
            val = getattr(ch, key, None)
            if val is not None:
                info[key] = to_jsonable(val)
        if not info and hasattr(ch, "__dict__"):
            info = to_jsonable(ch.__dict__)
        out.append(info)
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export ND2 ground-truth stats (raw pixel values + metadata) across groups.",
    )
    parser.add_argument("--nd2-dir", type=Path, required=True, help="Directory containing ND2 files.")
    parser.add_argument("--roi-root", type=Path, default=None, help="ROI root folder (optional).")
    parser.add_argument(
        "--groups",
        nargs="+",
        default=["ctznmda", "ctz", "nmda"],
        help="Group names (default: ctznmda ctz nmda).",
    )
    parser.add_argument("--roi-name", type=str, default=None, help="Specific ROI folder name.")
    parser.add_argument(
        "--channels",
        type=int,
        nargs="+",
        default=None,
        help="Channel indices to load (default: all).",
    )
    parser.add_argument(
        "--channel-names",
        type=str,
        nargs="+",
        default=None,
        help="Channel names for indices (default: from ND2 metadata if available).",
    )
    parser.add_argument("--z-project", choices=["max", "first"], default="max", help="Z projection mode.")
    parser.add_argument(
        "--percentiles",
        type=float,
        nargs="+",
        default=[0.5, 1, 5, 50, 95, 99, 99.5],
        help="Percentiles to report (default: 0.5 1 5 50 95 99 99.5).",
    )
    parser.add_argument("--out-dir", type=Path, default=None, help="Output directory.")
    parser.add_argument(
        "--bioformats",
        action="store_true",
        help="Attempt Bio-Formats metadata extraction via showinf (writes OME-XML).",
    )
    parser.add_argument(
        "--bioformats-tool",
        type=Path,
        default=None,
        help="Path to Bio-Formats showinf tool (default: search PATH).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    nd2_dir = args.nd2_dir.expanduser()
    if ND2File is None:
        raise SystemExit("nd2 is not installed. Install with: pip install nd2")

    roi_root = args.roi_root.expanduser() if args.roi_root else None
    out_dir = args.out_dir
    if out_dir is None:
        base = roi_root if roi_root else nd2_dir
        out_dir = base / "nd2_export" / "ground_truth"
    out_dir.mkdir(parents=True, exist_ok=True)
    bio_out_dir = out_dir / "bioformats"
    if args.bioformats:
        bio_out_dir.mkdir(parents=True, exist_ok=True)

    bio_tool = None
    if args.bioformats:
        if args.bioformats_tool:
            bio_tool = str(args.bioformats_tool)
        else:
            bio_tool = shutil.which("showinf")
        if not bio_tool:
            print("Bio-Formats showinf not found in PATH; skipping Bio-Formats extraction.")
            args.bioformats = False

    targets: List[Dict[str, Any]] = []
    if roi_root:
        for group in args.groups:
            group_dir = roi_root / f"group_{group}"
            roi_dir = find_group_roi_dir(group_dir, args.roi_name)
            nd2_path = resolve_nd2_path(roi_dir, nd2_dir)
            targets.append(
                {
                    "group": group,
                    "roi_dir": roi_dir,
                    "nd2_path": nd2_path,
                }
            )
    else:
        for nd2_path in sorted(nd2_dir.rglob("*.nd2")):
            targets.append(
                {
                    "group": None,
                    "roi_dir": None,
                    "nd2_path": nd2_path,
                }
            )

    summary: Dict[str, Any] = {
        "nd2_dir": str(nd2_dir),
        "roi_root": str(roi_root) if roi_root else None,
        "z_project": args.z_project,
        "percentiles": args.percentiles,
        "files": [],
    }

    csv_rows: List[Dict[str, Any]] = []

    for target in targets:
        nd2_path = Path(target["nd2_path"])
        with ND2File(str(nd2_path)) as nd2_file:
            axes = _nd2_axes(nd2_file)
            sizes = getattr(nd2_file, "sizes", None)
            meta = getattr(nd2_file, "metadata", None)
            meta_summary = {
                "axes": axes,
                "sizes": to_jsonable(sizes) if sizes is not None else None,
                "channels": extract_channel_metadata(meta) if meta else [],
            }
            meta_raw = to_jsonable(meta) if meta else None

        channels = load_nd2_channels(nd2_path, args.channels, args.z_project)
        channel_count = channels.shape[0]
        channel_names = args.channel_names
        if channel_names is None or len(channel_names) != channel_count:
            if meta and getattr(meta, "channels", None) and len(meta.channels) == channel_count:
                channel_names = [getattr(ch, "name", f"C{idx}") for idx, ch in enumerate(meta.channels)]
            else:
                channel_names = [f"C{idx}" for idx in range(channel_count)]

        file_entry: Dict[str, Any] = {
            "path": str(nd2_path),
            "group": target.get("group"),
            "roi_dir": str(target["roi_dir"]) if target.get("roi_dir") else None,
            "metadata_summary": meta_summary,
            "metadata_raw": meta_raw,
            "channels": [],
        }

        if args.bioformats and bio_tool:
            cmd = [bio_tool, "-omexml", str(nd2_path)]
            result = subprocess.run(cmd, capture_output=True, text=True)
            ome_path = None
            if result.stdout:
                ome_path = bio_out_dir / f"{nd2_path.stem}_ome.xml"
                ome_path.write_text(result.stdout)
            file_entry["bioformats"] = {
                "command": cmd,
                "returncode": result.returncode,
                "ome_xml": str(ome_path) if ome_path else None,
                "stderr": result.stderr.strip() if result.stderr else None,
            }

        for idx, name in enumerate(channel_names):
            stats_full = summarize_channel(channels[idx], args.percentiles)
            file_entry["channels"].append(
                {
                    "index": idx,
                    "name": name,
                    "stats_full": stats_full,
                }
            )
            row = {
                "file": str(nd2_path),
                "group": target.get("group"),
                "scope": "full",
                "channel_index": idx,
                "channel_name": name,
            }
            row.update(stats_full)
            csv_rows.append(row)

        if target.get("roi_dir"):
            roi_state_path = Path(target["roi_dir"]) / "roi_state.json"
            if roi_state_path.exists():
                roi_state = json.loads(roi_state_path.read_text())
                center = tuple(roi_state["center"])
                width = float(roi_state["width"])
                height = float(roi_state["height"])
                angle = float(roi_state.get("angle", 0.0))
                if angle and rotate is None:
                    angle = 0.0
                roi_crop = extract_rotated_roi(channels, center, width, height, angle)
                for idx, name in enumerate(channel_names):
                    stats_roi = summarize_channel(roi_crop[idx], args.percentiles)
                    file_entry["channels"][idx]["stats_roi"] = stats_roi
                    row = {
                        "file": str(nd2_path),
                        "group": target.get("group"),
                        "scope": "roi",
                        "channel_index": idx,
                        "channel_name": name,
                    }
                    row.update(stats_roi)
                    csv_rows.append(row)

        summary["files"].append(file_entry)

    json_path = out_dir / "nd2_ground_truth_summary.json"
    json_path.write_text(json.dumps(summary, indent=2))

    csv_path = out_dir / "nd2_ground_truth_channels.csv"
    if csv_rows:
        headers = sorted({key for row in csv_rows for key in row.keys()})
        lines = [",".join(headers)]
        for row in csv_rows:
            line = ",".join(str(row.get(col, "")) for col in headers)
            lines.append(line)
        csv_path.write_text("\n".join(lines))

    print(f"Saved ground-truth JSON to {json_path}")
    print(f"Saved channel stats CSV to {csv_path}")


if __name__ == "__main__":
    main()
