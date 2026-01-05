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


def pick_directory(initial_dir: Path) -> Path | None:
    if sys.platform == "darwin":
        script = (
            'POSIX path of (choose folder with prompt "Select ROI folder" '
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
        title="Select ROI folder",
        initialdir=str(initial_dir),
        parent=root,
    )
    root.destroy()
    return Path(path) if path else None


def pick_nd2_path(initial_dir: Path) -> Path | None:
    if sys.platform == "darwin":
        script = (
            'POSIX path of (choose file with prompt "Select ND2 file" '
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
        print(f"Tkinter is not available for file selection: {exc}")
        return None

    root = Tk()
    root.withdraw()
    path = filedialog.askopenfilename(
        title="Select ND2 file",
        initialdir=str(initial_dir),
        filetypes=[("ND2 files", "*.nd2"), ("All files", "*.*")],
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


def normalize_to_uint16(image: np.ndarray) -> np.ndarray:
    image = np.clip(image, 0.0, 1.0)
    return (image * 65535.0).round().astype(np.uint16)


def compose_merge(norm_channels: Dict[str, np.ndarray]) -> np.ndarray:
    ref = next(iter(norm_channels.values()))
    merged = np.zeros((*ref.shape, 3), dtype=np.float32)
    for key, channel in norm_channels.items():
        color = CHANNEL_COLORS.get(key, (1.0, 1.0, 1.0))
        for idx in range(3):
            merged[..., idx] += channel * color[idx]
    return np.clip(merged, 0.0, 1.0)


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


def colorize_channel(channel: np.ndarray, color: Tuple[float, float, float]) -> np.ndarray:
    rgb = np.zeros((*channel.shape, 3), dtype=np.float32)
    for idx in range(3):
        rgb[..., idx] = channel * color[idx]
    return np.clip(rgb, 0.0, 1.0)


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

    picked = pick_nd2_path(search_root)
    if picked is None:
        raise FileNotFoundError("No ND2 file selected.")
    return picked


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
    raise FileNotFoundError("Multiple roi_state.json files found. Select a specific ROI folder.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export high-resolution ROI crops from ND2 using saved roi_state.json.",
    )
    parser.add_argument("--roi-dir", type=Path, default=None, help="ROI folder containing roi_state.json.")
    parser.add_argument("--nd2-dir", type=Path, default=None, help="Directory to search for ND2 files.")
    parser.add_argument("--nd2-path", type=Path, default=None, help="Explicit ND2 file path.")
    parser.add_argument("--out-dir", type=Path, default=None, help="Output directory for TIFFs.")
    parser.add_argument(
        "--channels",
        type=int,
        nargs="+",
        default=[0, 1, 2, 3],
        help="Channel indices to load (default: 0 1 2 3).",
    )
    parser.add_argument(
        "--channel-names",
        type=str,
        nargs="+",
        default=["DAPI", "EYFP", "tdTom", "EL222"],
        help="Channel names for output filenames.",
    )
    parser.add_argument("--z-project", choices=["max", "first"], default="max", help="Z projection mode.")
    parser.add_argument("--low-pct", type=float, default=1.0, help="Low percentile for autoscale.")
    parser.add_argument("--high-pct", type=float, default=99.0, help="High percentile for autoscale.")
    parser.add_argument("--sigma", type=float, default=0.6, help="Gaussian sigma for mild smoothing.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    roi_dir = args.roi_dir
    if roi_dir is None:
        start = args.nd2_dir if args.nd2_dir else Path.cwd()
        roi_dir = pick_directory(start)
    if roi_dir is None:
        raise SystemExit("No ROI folder selected.")

    roi_dir = roi_dir.expanduser()
    roi_state_path = find_roi_state(roi_dir)
    roi_state = json.loads(roi_state_path.read_text())

    center = tuple(roi_state["center"])
    width = float(roi_state["width"])
    height = float(roi_state["height"])
    angle = float(roi_state.get("angle", 0.0))

    if angle and rotate is None:
        print("Rotation requested but scipy is not installed; using angle=0.")
        angle = 0.0

    nd2_path = resolve_nd2_path(roi_dir, args.nd2_dir, args.nd2_path)
    channels = load_nd2_channels(nd2_path, args.channels, args.z_project)

    if len(args.channel_names) != channels.shape[0]:
        raise SystemExit(
            f"Expected {channels.shape[0]} channel names, got {len(args.channel_names)}"
        )

    raw_roi = extract_rotated_roi(channels, center, width, height, angle)
    smooth_sigma = max(0.0, float(args.sigma))
    if smooth_sigma > 0 and gaussian_filter is None:
        print("Gaussian smoothing unavailable (install scipy). Proceeding without smoothing.")
        smooth_sigma = 0.0

    scaled_channels: Dict[str, np.ndarray] = {}
    scale_bounds: Dict[str, Tuple[float, float]] = {}
    per_name_scaled: Dict[str, np.ndarray] = {}
    per_name_bounds: Dict[str, Tuple[float, float]] = {}
    for idx, name in enumerate(args.channel_names):
        key = channel_key(name)
        channel = raw_roi[idx]
        if smooth_sigma > 0 and gaussian_filter is not None:
            channel = gaussian_filter(channel, sigma=smooth_sigma)
        scaled, bounds = scale_channel(channel, args.low_pct, args.high_pct)
        scaled_channels[key] = scaled
        scale_bounds[key] = bounds
        per_name_scaled[name] = scaled
        per_name_bounds[name] = bounds

    output_root = args.out_dir if args.out_dir else roi_dir / "nd2_export"
    output_root.mkdir(parents=True, exist_ok=True)
    export_dir = output_root / roi_dir.name
    export_dir.mkdir(parents=True, exist_ok=True)

    for idx, name in enumerate(args.channel_names):
        key = channel_key(name)
        safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in name)
        scaled = per_name_scaled[name]
        save_tiff(export_dir / f"{safe}_autoscale.tif", normalize_to_uint16(scaled))
        color = CHANNEL_COLORS.get(key, (1.0, 1.0, 1.0))
        pseudo = colorize_channel(scaled, color)
        save_tiff(export_dir / f"{safe}_pseudo.tif", normalize_to_uint16(pseudo))

    if scaled_channels:
        merge = compose_merge(scaled_channels)
        save_tiff(export_dir / "merged_autoscale.tif", normalize_to_uint16(merge))

    report = {
        "roi_dir": str(roi_dir),
        "nd2_path": str(nd2_path),
        "center": center,
        "width": width,
        "height": height,
        "angle": angle,
        "low_pct": args.low_pct,
        "high_pct": args.high_pct,
        "sigma": smooth_sigma,
        "bounds": per_name_bounds,
        "output_dir": str(export_dir),
    }
    (export_dir / "export_report.json").write_text(json.dumps(report, indent=2))
    print(f"Saved high-res TIFFs to {export_dir}")


if __name__ == "__main__":
    main()
