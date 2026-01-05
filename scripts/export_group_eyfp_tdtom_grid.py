#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

try:
    from nd2 import ND2File
except ImportError:  # pragma: no cover - optional ND2 support
    ND2File = None

try:
    from scipy.ndimage import rotate, gaussian_filter
except ImportError:  # pragma: no cover - optional smoothing/rotation
    rotate = None
    gaussian_filter = None

try:
    import tifffile
except ImportError:  # pragma: no cover - optional TIFF writing
    tifffile = None


CHANNEL_COLORS = {
    "dapi": (0.0, 0.0, 1.0),
    "gfp": (0.0, 1.0, 0.0),
    "tdtom": (1.0, 0.0, 0.0),
    "el222": (1.0, 0.0, 1.0),
}


def pick_directory(initial_dir: Path, prompt: str) -> Path | None:
    if sys.platform == "darwin":
        script = (
            f'POSIX path of (choose folder with prompt "{prompt}" '
            f'default location POSIX file "{initial_dir}")'
        )
        result = subprocess.run(
            ["osascript", "-e", script],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            chosen = result.stdout.strip()
            if chosen:
                return Path(chosen)

    try:
        from tkinter import Tk, filedialog
    except Exception as exc:
        print(f"Tkinter is not available for folder selection: {exc}")
        return None

    root = Tk()
    root.withdraw()
    path = filedialog.askdirectory(
        title=prompt,
        initialdir=str(initial_dir),
        parent=root,
    )
    root.destroy()
    return Path(path) if path else None


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


def load_nd2_channels(path: Path, channel_indices: List[int], z_project: str) -> np.ndarray:
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


def channel_key(name: str) -> str:
    lower = "".join(ch for ch in name.lower() if ch.isalnum())
    if "dapi" in lower:
        return "dapi"
    if "eyfp" in lower or "yfp" in lower or "gfp" in lower:
        return "gfp"
    if "tdtom" in lower or "tdtomato" in lower:
        return "tdtom"
    if "el222" in lower:
        return "el222"
    return lower


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

    pad_left = max(0, -x0)
    pad_right = max(0, x1 - img_w)
    pad_top = max(0, -y0)
    pad_bottom = max(0, y1 - img_h)

    if any(v > 0 for v in (pad_left, pad_right, pad_top, pad_bottom)):
        image = np.pad(image, ((pad_top, pad_bottom), (pad_left, pad_right)), mode="edge")
        x0 += pad_left
        x1 += pad_left
        y0 += pad_top
        y1 += pad_top

    window = image[y0:y1, x0:x1]
    if angle and rotate is not None:
        rotated = rotate(window, angle, reshape=False, order=1, mode="nearest")
    else:
        rotated = window

    cx_local = rotated.shape[1] / 2.0
    cy_local = rotated.shape[0] / 2.0
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


def scale_channel(channel: np.ndarray, low_pct: float, high_pct: float) -> Tuple[np.ndarray, Tuple[float, float]]:
    vmin, vmax = np.percentile(channel, [low_pct, high_pct])
    if vmax <= vmin:
        vmax = vmin + 1.0
    scaled = np.clip((channel - vmin) / (vmax - vmin), 0.0, 1.0)
    return scaled, (float(vmin), float(vmax))


def apply_bounds(channel: np.ndarray, bounds: Tuple[float, float]) -> np.ndarray:
    vmin, vmax = bounds
    return np.clip((channel - vmin) / (vmax - vmin), 0.0, 1.0)


def auto_bounds(
    channel: np.ndarray,
    low_pct: float,
    base_high: float,
    sat_target: float,
    candidates: List[float] | None = None,
) -> Tuple[Tuple[float, float], float, float]:
    if candidates is None:
        candidates = [base_high, 99.5, 99.7, 99.9, 99.95, 99.99]
    candidates = sorted({pct for pct in candidates if pct >= base_high})
    low_val = float(np.percentile(channel, low_pct))
    chosen = None
    chosen_high = base_high
    chosen_sat = 1.0
    for high_pct in candidates:
        vmin, vmax = np.percentile(channel, [low_pct, high_pct])
        if vmax <= vmin:
            vmax = vmin + 1.0
        sat_ratio = float(np.mean(channel >= vmax))
        chosen = (float(vmin), float(vmax))
        chosen_high = float(high_pct)
        chosen_sat = sat_ratio
        if sat_ratio <= sat_target:
            break
    if chosen is None:
        chosen = (low_val, low_val + 1.0)
    return chosen, chosen_high, chosen_sat


def normalize_to_uint16(image: np.ndarray) -> np.ndarray:
    image = np.clip(image, 0.0, 1.0)
    return (image * 65535.0).round().astype(np.uint16)


def colorize_channel(channel: np.ndarray, color: Tuple[float, float, float]) -> np.ndarray:
    rgb = np.zeros((*channel.shape, 3), dtype=np.float32)
    for idx in range(3):
        rgb[..., idx] = channel * color[idx]
    return np.clip(rgb, 0.0, 1.0)


def match_histogram(source: np.ndarray, template: np.ndarray, bins: int) -> np.ndarray:
    src = source.astype(np.float32, copy=False).ravel()
    tmpl = template.astype(np.float32, copy=False).ravel()
    src_min, src_max = float(np.min(src)), float(np.max(src))
    tmpl_min, tmpl_max = float(np.min(tmpl)), float(np.max(tmpl))
    if src_max <= src_min or tmpl_max <= tmpl_min:
        return source
    src_hist, src_bins = np.histogram(src, bins=bins, range=(src_min, src_max), density=True)
    tmpl_hist, tmpl_bins = np.histogram(tmpl, bins=bins, range=(tmpl_min, tmpl_max), density=True)
    src_cdf = np.cumsum(src_hist)
    tmpl_cdf = np.cumsum(tmpl_hist)
    if src_cdf[-1] == 0 or tmpl_cdf[-1] == 0:
        return source
    src_cdf /= src_cdf[-1]
    tmpl_cdf /= tmpl_cdf[-1]
    src_centers = (src_bins[:-1] + src_bins[1:]) / 2.0
    tmpl_centers = (tmpl_bins[:-1] + tmpl_bins[1:]) / 2.0
    src_values = np.interp(src, src_centers, src_cdf)
    matched = np.interp(src_values, tmpl_cdf, tmpl_centers)
    return matched.reshape(source.shape)


def save_tiff(path: Path, image: np.ndarray) -> None:
    if tifffile is None:
        raise RuntimeError("tifffile is required to save TIFF. Install with: pip install tifffile")
    if image.ndim == 2:
        tifffile.imwrite(path, image, photometric="minisblack")
    else:
        tifffile.imwrite(path, image, photometric="rgb")


def find_roi_state(roi_dir: Path) -> Path:
    direct = roi_dir / "roi_state.json"
    if direct.exists():
        return direct
    matches = list(roi_dir.rglob("roi_state.json"))
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise FileNotFoundError("roi_state.json not found in the selected folder.")
    names = ", ".join(str(path.parent.name) for path in matches)
    raise FileNotFoundError(f"Multiple roi_state.json files found ({names}). Select a specific ROI folder.")


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


def resolve_nd2_path(roi_dir: Path, nd2_dir: Path | None, nd2_path: Path | None) -> Path:
    if nd2_path is not None and nd2_path.exists():
        return nd2_path

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


def crop_to_min_shape(images: Dict[str, Dict[str, np.ndarray]]) -> Dict[str, Dict[str, np.ndarray]]:
    heights = []
    widths = []
    for group_data in images.values():
        for arr in group_data.values():
            heights.append(arr.shape[0])
            widths.append(arr.shape[1])
    if not heights or not widths:
        return images
    min_h = min(heights)
    min_w = min(widths)
    cropped = {}
    for group, group_data in images.items():
        cropped[group] = {key: arr[:min_h, :min_w] for key, arr in group_data.items()}
    return cropped


def make_grid(rows: List[str], cols: List[str], images: Dict[str, Dict[str, np.ndarray]], padding: int) -> np.ndarray:
    images = crop_to_min_shape(images)
    sample = next(iter(images.values()))
    first = next(iter(sample.values()))
    height, width = first.shape[:2]
    pad = max(0, int(padding))

    row_blocks = []
    for group in rows:
        col_blocks = []
        for key in cols:
            img = images[group][key]
            col_blocks.append(img)
        row = col_blocks[0]
        for block in col_blocks[1:]:
            if pad:
                row = np.concatenate([row, np.zeros((height, pad, 3), dtype=row.dtype), block], axis=1)
            else:
                row = np.concatenate([row, block], axis=1)
        row_blocks.append(row)
    grid = row_blocks[0]
    for block in row_blocks[1:]:
        if pad:
            grid = np.concatenate([grid, np.zeros((pad, grid.shape[1], 3), dtype=grid.dtype), block], axis=0)
        else:
            grid = np.concatenate([grid, block], axis=0)
    return grid


def resolve_report_path(roi_root: Path, out_dir: Path | None, explicit: Path | None) -> Path | None:
    if explicit is not None:
        return explicit if explicit.exists() else None
    if out_dir is not None:
        candidate = out_dir / "group_grid_export" / "export_report.json"
        if candidate.exists():
            return candidate
    candidate = roi_root / "nd2_export" / "group_grid_export" / "export_report.json"
    if candidate.exists():
        return candidate
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export EYFP + tdTom ROI grids across groups, scaled to ctznmda.",
    )
    parser.add_argument("--roi-root", type=Path, default=None, help="Root folder with group_* ROI folders.")
    parser.add_argument("--roi-name", type=str, default=None, help="Specific ROI folder name (e.g., 3_roi).")
    parser.add_argument("--nd2-dir", type=Path, default=None, help="Directory to search for ND2 files.")
    parser.add_argument("--out-dir", type=Path, default=None, help="Output directory for TIFFs.")
    parser.add_argument(
        "--report-path",
        type=Path,
        default=None,
        help="Explicit export_report.json to reuse last selections.",
    )
    parser.add_argument(
        "--groups",
        nargs="+",
        default=["ctznmda", "ctz", "nmda"],
        help="Group names to include (default: ctznmda ctz nmda).",
    )
    parser.add_argument(
        "--pick-per-group",
        action="store_true",
        help="Prompt for a ROI folder per group instead of auto-resolving under roi-root.",
    )
    parser.add_argument(
        "--use-last",
        action="store_true",
        help="Reuse roi_state.json selections from the last export_report.json if found.",
    )
    parser.add_argument(
        "--reuse-eyfp-bounds",
        action="store_true",
        help="Reuse last EYFP bounds from export_report.json to keep EYFP unchanged.",
    )
    parser.add_argument(
        "--channels",
        type=int,
        nargs="+",
        default=[0, 1, 2, 3],
        help="Channel indices to load from ND2 (default: 0 1 2 3).",
    )
    parser.add_argument(
        "--channel-names",
        type=str,
        nargs="+",
        default=["DAPI", "EYFP", "tdTom", "EL222"],
        help="Channel names for ND2 indices.",
    )
    parser.add_argument(
        "--select",
        nargs="+",
        default=["EYFP", "tdTom"],
        help="Channel names to export (default: EYFP tdTom).",
    )
    parser.add_argument(
        "--eyfp-per-group",
        dest="eyfp_per_group",
        action="store_true",
        help="Scale EYFP per-group instead of using reference group bounds.",
    )
    parser.add_argument(
        "--eyfp-ref",
        dest="eyfp_per_group",
        action="store_false",
        help="Scale EYFP using reference group bounds (matches tdTom).",
    )
    parser.add_argument("--z-project", choices=["max", "first"], default="max", help="Z projection mode.")
    parser.add_argument("--low-pct", type=float, default=1.0, help="Low percentile for autoscale.")
    parser.add_argument("--high-pct", type=float, default=99.0, help="High percentile for autoscale.")
    parser.add_argument("--tdtom-low", type=float, default=None, help="Override tdTom low percentile.")
    parser.add_argument("--tdtom-high", type=float, default=None, help="Override tdTom high percentile.")
    parser.add_argument(
        "--tdtom-hist-match",
        action="store_true",
        help="Histogram-match tdTom to the reference group (ctznmda).",
    )
    parser.add_argument(
        "--hist-bins",
        type=int,
        default=1024,
        help="Number of bins for histogram matching (default: 1024).",
    )
    parser.add_argument("--sigma", type=float, default=0.6, help="Gaussian sigma for mild smoothing.")
    parser.add_argument("--padding", type=int, default=10, help="Grid padding in pixels.")
    parser.add_argument(
        "--auto-bounds",
        dest="auto_bounds",
        action="store_true",
        help="Automatically raise high percentile to avoid saturation (uses reference group).",
    )
    parser.add_argument(
        "--no-auto-bounds",
        dest="auto_bounds",
        action="store_false",
        help="Disable auto-bounds even when pick-per-group is enabled.",
    )
    parser.add_argument(
        "--sat-target",
        type=float,
        default=0.005,
        help="Allowed saturation fraction when auto-bounds is enabled (default: 0.5%).",
    )
    parser.set_defaults(auto_bounds=None, eyfp_per_group=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if len(args.channel_names) != len(args.channels):
        raise SystemExit("channel-names length must match channels length.")

    roi_root = args.roi_root
    if roi_root is None:
        start = args.nd2_dir if args.nd2_dir else Path.cwd()
        roi_root = pick_directory(start, "Select ROI root folder")
    if roi_root is None:
        raise SystemExit("No ROI root selected.")

    roi_root = roi_root.expanduser()
    groups = [item.strip() for item in args.groups if item.strip()]
    if not groups:
        raise SystemExit("No groups provided.")
    ref_group = groups[0]
    if args.auto_bounds is None:
        auto_bounds_enabled = args.pick_per_group
    else:
        auto_bounds_enabled = args.auto_bounds

    report_path = resolve_report_path(roi_root, args.out_dir, args.report_path)
    report_data = None
    if args.use_last and report_path is not None:
        try:
            report_data = json.loads(report_path.read_text())
        except Exception as exc:
            print(f"Failed to read report at {report_path}: {exc}")
            report_data = None

    tdtom_low = args.tdtom_low
    tdtom_high = args.tdtom_high
    if report_data and tdtom_low is None and tdtom_high is None:
        overrides = report_data.get("tdtom_override")
        if isinstance(overrides, dict):
            tdtom_low = overrides.get("low_pct")
            tdtom_high = overrides.get("high_pct")

    reuse_eyfp_bounds = args.reuse_eyfp_bounds and report_data and "bounds" in report_data
    last_bounds = report_data.get("bounds", {}) if reuse_eyfp_bounds else {}
    eyfp_per_group = bool(args.eyfp_per_group)

    select_set = {name.lower() for name in args.select}
    channel_map = {name: idx for idx, name in enumerate(args.channel_names)}
    selected_names = [name for name in args.channel_names if name.lower() in select_set]
    if not selected_names:
        raise SystemExit("No selected channels match channel-names.")

    smooth_sigma = max(0.0, float(args.sigma))
    if smooth_sigma > 0 and gaussian_filter is None:
        print("Gaussian smoothing unavailable (install scipy). Proceeding without smoothing.")
        smooth_sigma = 0.0

    roi_state_by_group = {}
    nd2_by_group = {}
    raw_by_group: Dict[str, Dict[str, np.ndarray]] = {}

    for group in groups:
        if args.pick_per_group:
            prompt = f"Select ROI folder for {group}"
            roi_dir = pick_directory(roi_root, prompt)
            if roi_dir is None:
                raise SystemExit(f"No ROI folder selected for {group}.")
        else:
            if report_data and "roi_states" in report_data and group in report_data["roi_states"]:
                roi_dir = Path(report_data["roi_states"][group]["roi_dir"])
            else:
                group_dir = roi_root / f"group_{group}"
                if not group_dir.exists():
                    raise FileNotFoundError(f"Missing group folder: {group_dir}")
                roi_dir = find_group_roi_dir(group_dir, args.roi_name)
        roi_state_path = find_roi_state(roi_dir)
        roi_state = json.loads(roi_state_path.read_text())
        center = tuple(roi_state["center"])
        width = float(roi_state["width"])
        height = float(roi_state["height"])
        angle = float(roi_state.get("angle", 0.0))
        if angle and rotate is None:
            print(f"[{group}] Rotation requested but scipy missing; using angle=0.")
            angle = 0.0

        if report_data and "nd2_paths" in report_data and group in report_data["nd2_paths"]:
            nd2_path = Path(report_data["nd2_paths"][group])
        else:
            nd2_path = resolve_nd2_path(roi_dir, args.nd2_dir, None)
        channels = load_nd2_channels(nd2_path, args.channels, args.z_project)
        roi_crop = extract_rotated_roi(channels, center, width, height, angle)

        channel_data = {}
        for name in selected_names:
            idx = channel_map[name]
            channel = roi_crop[idx]
            if smooth_sigma > 0 and gaussian_filter is not None:
                channel = gaussian_filter(channel, sigma=smooth_sigma)
            channel_data[name] = channel
        raw_by_group[group] = channel_data
        roi_state_by_group[group] = {
            "roi_dir": str(roi_dir),
            "center": center,
            "width": width,
            "height": height,
            "angle": angle,
        }
        nd2_by_group[group] = str(nd2_path)

    if ref_group not in raw_by_group:
        raise SystemExit(f"Reference group {ref_group} not loaded.")

    if args.tdtom_hist_match:
        ref_tdtom = None
        for name, channel in raw_by_group[ref_group].items():
            if channel_key(name) == "tdtom":
                ref_tdtom = channel
                break
        if ref_tdtom is None:
            raise SystemExit("tdTom channel not found in reference group for histogram match.")
        for group in groups:
            if group == ref_group:
                continue
            for name, channel in raw_by_group[group].items():
                if channel_key(name) == "tdtom":
                    raw_by_group[group][name] = match_histogram(channel, ref_tdtom, args.hist_bins)
                    break

    bounds_by_channel: Dict[str, Tuple[float, float]] = {}
    eyfp_bounds_by_group: Dict[str, Tuple[float, float]] = {}
    auto_meta: Dict[str, Dict[str, float]] = {}
    for name, channel in raw_by_group[ref_group].items():
        if reuse_eyfp_bounds and name in last_bounds and channel_key(name) == "gfp":
            bounds_by_channel[name] = tuple(last_bounds[name])
        elif name.lower() == "tdtom" and tdtom_low is not None and tdtom_high is not None:
            _, bounds = scale_channel(channel, float(tdtom_low), float(tdtom_high))
            bounds_by_channel[name] = bounds
        elif auto_bounds_enabled:
            bounds, chosen_high, sat_ratio = auto_bounds(
                channel,
                args.low_pct,
                args.high_pct,
                args.sat_target,
            )
            bounds_by_channel[name] = bounds
            auto_meta[name] = {
                "high_pct": chosen_high,
                "sat_ratio": sat_ratio,
            }
        else:
            _, bounds = scale_channel(channel, args.low_pct, args.high_pct)
            bounds_by_channel[name] = bounds

    if eyfp_per_group:
        for group, channels in raw_by_group.items():
            for name, channel in channels.items():
                if channel_key(name) != "gfp":
                    continue
                _, bounds = scale_channel(channel, args.low_pct, args.high_pct)
                eyfp_bounds_by_group[group] = bounds

    scaled_by_group: Dict[str, Dict[str, np.ndarray]] = {}
    pseudo_by_group: Dict[str, Dict[str, np.ndarray]] = {}
    for group in groups:
        scaled_by_group[group] = {}
        pseudo_by_group[group] = {}
        for name, channel in raw_by_group[group].items():
            if eyfp_per_group and channel_key(name) == "gfp":
                bounds = eyfp_bounds_by_group.get(group, bounds_by_channel[name])
            else:
                bounds = bounds_by_channel[name]
            scaled = apply_bounds(channel, bounds)
            scaled_by_group[group][name] = scaled
            key = channel_key(name)
            color = CHANNEL_COLORS.get(key, (1.0, 1.0, 1.0))
            pseudo_by_group[group][name] = colorize_channel(scaled, color)

    output_root = args.out_dir if args.out_dir else roi_root / "nd2_export"
    output_root.mkdir(parents=True, exist_ok=True)
    export_dir = output_root / "group_grid_export"
    export_dir.mkdir(parents=True, exist_ok=True)

    for group in groups:
        for name in selected_names:
            safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in name)
            scaled = scaled_by_group[group][name]
            pseudo = pseudo_by_group[group][name]
            save_tiff(export_dir / f"{group}_{safe}_autoscale.tif", normalize_to_uint16(scaled))
            save_tiff(export_dir / f"{group}_{safe}_pseudo.tif", normalize_to_uint16(pseudo))

    cols = [name for name in selected_names]
    grid = make_grid(groups, cols, pseudo_by_group, args.padding)
    grid_name = "grid_" + "_".join(groups) + "_pseudo.tif"
    save_tiff(export_dir / grid_name, normalize_to_uint16(grid))

    report = {
        "roi_root": str(roi_root),
        "nd2_dir": str(args.nd2_dir) if args.nd2_dir else None,
        "groups": groups,
        "reference_group": ref_group,
        "selected_channels": selected_names,
        "report_used": str(report_path) if report_path else None,
        "tdtom_override": {
            "low_pct": tdtom_low,
            "high_pct": tdtom_high,
        },
        "reuse_eyfp_bounds": bool(reuse_eyfp_bounds),
        "eyfp_per_group": eyfp_per_group,
        "tdtom_hist_match": bool(args.tdtom_hist_match),
        "hist_bins": int(args.hist_bins),
        "bounds": bounds_by_channel,
        "auto_bounds": auto_bounds_enabled,
        "auto_meta": auto_meta,
        "sat_target": args.sat_target,
        "sigma": smooth_sigma,
        "low_pct": args.low_pct,
        "high_pct": args.high_pct,
        "roi_states": roi_state_by_group,
        "nd2_paths": nd2_by_group,
        "output_dir": str(export_dir),
    }
    (export_dir / "export_report.json").write_text(json.dumps(report, indent=2))
    print(f"Saved group TIFFs + grid to {export_dir}")


if __name__ == "__main__":
    main()
