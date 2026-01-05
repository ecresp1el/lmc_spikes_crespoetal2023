#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
import re
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button, CheckButtons, Slider
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

try:
    from scipy.ndimage import gaussian_filter
except ImportError:  # pragma: no cover - optional smoothing
    gaussian_filter = None
try:
    from nd2 import ND2File
except ImportError:  # pragma: no cover - optional ND2 support
    ND2File = None
try:
    import tifffile
except ImportError:  # pragma: no cover - optional image support
    tifffile = None

plt.rcParams["font.family"] = "Arial"
plt.rcParams["svg.fonttype"] = "none"

DEFAULT_GROUPS = ["ctznmda", "ctz", "nmda"]
CHANNEL_ORDER = ["dapi", "gfp", "tdtom", "el222"]
CHANNEL_LABELS = {
    "dapi": "DAPI",
    "gfp": "GFP",
    "tdtom": "tdTom",
    "el222": "EL222",
}
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


def pick_save_path(initial_dir: Path, default_name: str) -> Path | None:
    if sys.platform == "darwin":
        script = (
            'POSIX path of (choose file name with prompt "Save SVG" '
            f'default name "{default_name}" default location POSIX file "{initial_dir}")'
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
    path = filedialog.asksaveasfilename(
        title="Save SVG",
        initialdir=str(initial_dir),
        defaultextension=".svg",
        filetypes=[("SVG", "*.svg")],
        parent=root,
    )
    root.destroy()
    return Path(path) if path else None


def pick_image_path(initial_dir: Path) -> Path | None:
    if sys.platform == "darwin":
        script = (
            'POSIX path of (choose file with prompt "Select image file" '
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
        print(f"Tkinter is not available for image selection: {exc}")
        return None

    root = Tk()
    root.withdraw()
    path = filedialog.askopenfilename(
        title="Select image file",
        initialdir=str(initial_dir),
        filetypes=[
            ("Image files", "*.png *.jpg *.jpeg *.tif *.tiff *.nd2"),
            ("All files", "*.*"),
        ],
        parent=root,
    )
    root.destroy()
    return Path(path) if path else None


def load_image(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".nd2":
        return load_nd2_preview(path)
    if path.suffix.lower() in {".tif", ".tiff"} and tifffile is not None:
        img = tifffile.imread(path)
    else:
        img = plt.imread(path)
    img = np.asarray(img)
    if img.ndim == 3 and img.shape[-1] == 4:
        img = img[..., :3]
    return img


def _normalize_axes(axes) -> List[str]:
    normalized = []
    for axis in axes:
        if hasattr(axis, "name"):
            axis_name = axis.name
        else:
            axis_name = str(axis)
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


def load_nd2_preview(path: Path) -> np.ndarray:
    if ND2File is None:
        raise RuntimeError("nd2 is not installed. Install with: pip install nd2")
    with ND2File(str(path)) as nd2_file:
        data = np.asarray(nd2_file.asarray())
        axes = _nd2_axes(nd2_file)

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
    take_axis("Z", reducer=np.max)

    if "C" in axes and "Y" in axes and "X" in axes:
        c_idx, y_idx, x_idx = axes.index("C"), axes.index("Y"), axes.index("X")
        data = np.moveaxis(data, [c_idx, y_idx, x_idx], [0, 1, 2])
        channels = data.astype(np.float32)
        num = min(channels.shape[0], len(CHANNEL_ORDER))
        norm = []
        for idx in range(num):
            channel = channels[idx]
            vmin, vmax = np.percentile(channel, [1.0, 99.0])
            if vmax <= vmin:
                vmax = vmin + 1.0
            norm.append(np.clip((channel - vmin) / (vmax - vmin), 0.0, 1.0))
        norm_map = {CHANNEL_ORDER[idx]: norm[idx] for idx in range(num)}
        return compose_merge(norm_map)

    if data.ndim >= 2:
        return np.asarray(data).astype(np.float32)
    raise ValueError("Unexpected ND2 data shape for preview.")


def channel_key(name: str) -> str:
    lower = name.lower()
    compact = re.sub(r"[^a-z0-9]+", "", lower)
    if "dapi" in compact:
        return "dapi"
    if "eyfp" in compact or "yfp" in compact or "gfp" in compact:
        return "gfp"
    if "tdtom" in compact or "tdtomato" in compact:
        return "tdtom"
    if "el222" in compact:
        return "el222"
    return compact


def load_roi_folder(folder: Path) -> Tuple[Dict[str, np.ndarray], Dict[str, str]]:
    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder}")

    arrays: Dict[str, np.ndarray] = {}
    labels: Dict[str, str] = {}
    raw_files = sorted(folder.glob("roi_raw_ch*_*.npy"))
    if not raw_files:
        candidates = [p for p in folder.iterdir() if p.is_dir()]
        subfolders = []
        for sub in candidates:
            if list(sub.glob("roi_raw_ch*_*.npy")):
                subfolders.append(sub)
        if len(subfolders) == 1:
            raw_files = sorted(subfolders[0].glob("roi_raw_ch*_*.npy"))
        elif len(subfolders) > 1:
            names = ", ".join(sub.name for sub in subfolders)
            raise ValueError(f"Select a specific ROI folder (found: {names})")

    for path in raw_files:
        parts = path.stem.split("_", 3)
        if len(parts) < 4:
            continue
        name = parts[3]
        key = channel_key(name)
        arrays[key] = np.load(path)
        labels[key] = name

    if not arrays:
        raise ValueError(f"No roi_raw_ch*.npy files found in {folder}")

    return arrays, labels


def percentile_bounds(values: List[np.ndarray], low: float, high: float) -> Tuple[float, float]:
    stacked = np.concatenate([val.ravel() for val in values])
    vmin, vmax = np.percentile(stacked, [low, high])
    if vmax <= vmin:
        vmax = vmin + 1.0
    return float(vmin), float(vmax)


def normalize_channel(channel: np.ndarray, bounds: Tuple[float, float]) -> np.ndarray:
    vmin, vmax = bounds
    return np.clip((channel - vmin) / (vmax - vmin), 0.0, 1.0)


def normalize_display(image: np.ndarray, low_pct: float, high_pct: float) -> np.ndarray:
    if image.ndim == 2:
        vmin, vmax = np.percentile(image, [low_pct, high_pct])
        if vmax <= vmin:
            vmax = vmin + 1.0
        return np.clip((image - vmin) / (vmax - vmin), 0.0, 1.0)
    image = image.astype(np.float32, copy=False)
    max_val = float(np.nanmax(image)) if image.size else 1.0
    if max_val <= 0:
        max_val = 1.0
    return np.clip(image / max_val, 0.0, 1.0)


def compose_merge(norm_channels: Dict[str, np.ndarray]) -> np.ndarray:
    ref = next(iter(norm_channels.values()))
    merged = np.zeros((*ref.shape, 3), dtype=np.float32)
    for key, channel in norm_channels.items():
        color = CHANNEL_COLORS.get(key, (1.0, 1.0, 1.0))
        for idx in range(3):
            merged[..., idx] += channel * color[idx]
    return np.clip(merged, 0.0, 1.0)


def channel_cmap(key: str) -> LinearSegmentedColormap:
    color = CHANNEL_COLORS.get(key, (1.0, 1.0, 1.0))
    return LinearSegmentedColormap.from_list(f"{key}_map", [(0, 0, 0), color])


def colorize_channel(channel: np.ndarray, color: Tuple[float, float, float]) -> np.ndarray:
    rgb = np.zeros((*channel.shape, 3), dtype=np.float32)
    for idx in range(3):
        rgb[..., idx] = channel * color[idx]
    return np.clip(rgb, 0.0, 1.0)


def phase_correlation_shift(ref: np.ndarray, img: np.ndarray) -> Tuple[int, int]:
    ref_f = np.fft.fft2(ref)
    img_f = np.fft.fft2(img)
    cross_power = ref_f * img_f.conj()
    denom = np.maximum(np.abs(cross_power), 1e-8)
    cross_power /= denom
    corr = np.fft.ifft2(cross_power)
    max_idx = np.unravel_index(np.argmax(np.abs(corr)), corr.shape)
    shifts = np.array(max_idx, dtype=int)
    shape = np.array(ref.shape)
    shifts[shifts > shape // 2] -= shape[shifts > shape // 2]
    return -int(shifts[0]), -int(shifts[1])


def shift_image(image: np.ndarray, dy: int, dx: int) -> np.ndarray:
    height, width = image.shape
    result = np.zeros_like(image)

    src_y0 = max(0, -dy)
    src_y1 = min(height, height - dy)
    dst_y0 = max(0, dy)
    dst_y1 = min(height, height + dy)

    src_x0 = max(0, -dx)
    src_x1 = min(width, width - dx)
    dst_x0 = max(0, dx)
    dst_x1 = min(width, width + dx)

    if src_y1 <= src_y0 or src_x1 <= src_x0:
        return result

    result[dst_y0:dst_y1, dst_x0:dst_x1] = image[src_y0:src_y1, src_x0:src_x1]
    return result


def apply_smoothing(channel: np.ndarray, enabled: bool, sigma: float) -> np.ndarray:
    if not enabled:
        return channel
    if gaussian_filter is None:
        return channel
    return gaussian_filter(channel, sigma=float(sigma))


def subtract_background(channel: np.ndarray, pct: float) -> np.ndarray:
    baseline = np.percentile(channel, pct)
    return np.clip(channel - baseline, 0.0, None)


def prepare_group_channels(
    group_arrays: Dict[str, np.ndarray],
    smooth_enabled: bool,
    sigma: float,
    bg_enabled: bool,
    bg_pct: float,
    bg_channels: set[str],
) -> Dict[str, np.ndarray]:
    output = {}
    for key, arr in group_arrays.items():
        channel = arr
        if bg_enabled and key in bg_channels:
            channel = subtract_background(channel, bg_pct)
        channel = apply_smoothing(channel, smooth_enabled, sigma)
        output[key] = channel
    return output


def align_groups(
    group_channels: Dict[str, Dict[str, np.ndarray]],
    ref_group: str,
) -> Dict[str, Dict[str, np.ndarray]]:
    if ref_group not in group_channels or "dapi" not in group_channels[ref_group]:
        return group_channels

    ref_dapi = group_channels[ref_group]["dapi"]
    aligned = {}
    for group, channels in group_channels.items():
        if group == ref_group or "dapi" not in channels:
            aligned[group] = channels
            continue
        shift = phase_correlation_shift(ref_dapi, channels["dapi"])
        aligned_channels = {}
        for key, arr in channels.items():
            aligned_channels[key] = shift_image(arr, shift[0], shift[1])
        aligned[group] = aligned_channels
    return aligned


def crop_to_min_shape(group_channels: Dict[str, Dict[str, np.ndarray]]) -> Dict[str, Dict[str, np.ndarray]]:
    heights = []
    widths = []
    for channels in group_channels.values():
        for arr in channels.values():
            heights.append(arr.shape[0])
            widths.append(arr.shape[1])
    if not heights or not widths:
        return group_channels
    min_h = min(heights)
    min_w = min(widths)

    cropped = {}
    for group, channels in group_channels.items():
        cropped[group] = {key: arr[:min_h, :min_w] for key, arr in channels.items()}
    return cropped


def resolve_bounds(
    prepared: Dict[str, Dict[str, np.ndarray]],
    available_groups: List[str],
    low_pct: float,
    high_pct: float,
    el222_override: bool,
    el222_low: float,
    el222_high: float,
) -> Tuple[Dict[str, Tuple[float, float]], List[str]]:
    bounds: Dict[str, Tuple[float, float]] = {}
    warnings: List[str] = []

    dapi_vals = [prepared[g]["dapi"] for g in available_groups if "dapi" in prepared[g]]
    if dapi_vals:
        bounds["dapi"] = percentile_bounds(dapi_vals, low_pct, high_pct)
    else:
        warnings.append("No DAPI channel found.")

    gfp_vals = [prepared[g]["gfp"] for g in available_groups if "gfp" in prepared[g]]
    if gfp_vals:
        bounds["gfp"] = percentile_bounds(gfp_vals, low_pct, high_pct)
    else:
        warnings.append("No GFP/EYFP channel found.")

    if el222_override:
        tdtom_vals = [prepared[g]["tdtom"] for g in available_groups if "tdtom" in prepared[g]]
        el_vals = [prepared[g]["el222"] for g in available_groups if "el222" in prepared[g]]
        if tdtom_vals:
            bounds["tdtom"] = percentile_bounds(tdtom_vals, low_pct, high_pct)
        else:
            warnings.append("No tdTom channel found.")
        if el_vals:
            bounds["el222"] = percentile_bounds(el_vals, el222_low, el222_high)
        else:
            warnings.append("No EL222 channel found.")
    else:
        shared_vals = []
        for key in ("tdtom", "el222"):
            for group in available_groups:
                if key in prepared[group]:
                    shared_vals.append(prepared[group][key])
        if shared_vals:
            shared_bounds = percentile_bounds(shared_vals, low_pct, high_pct)
            bounds["tdtom"] = shared_bounds
            bounds["el222"] = shared_bounds
        else:
            warnings.append("No tdTom/EL222 channels found.")

    for key in CHANNEL_ORDER:
        if key in bounds:
            continue
        vals = [prepared[g][key] for g in available_groups if key in prepared[g]]
        if vals:
            bounds[key] = percentile_bounds(vals, low_pct, high_pct)
            warnings.append(f"Using per-group bounds for {CHANNEL_LABELS.get(key, key)}.")

    return bounds, warnings


def per_image_bounds(
    prepared: Dict[str, Dict[str, np.ndarray]],
    group: str,
    key: str,
    low_pct: float,
    high_pct: float,
    el_override: bool,
    el_low: float,
    el_high: float,
) -> Tuple[float, float] | None:
    if group not in prepared or key not in prepared[group]:
        return None
    if key == "el222" and el_override:
        return percentile_bounds([prepared[group][key]], el_low, el_high)
    return percentile_bounds([prepared[group][key]], low_pct, high_pct)


def build_gui(initial_dir: Path, out_dir: Path | None, roi_root: Path | None, groups: List[str]) -> None:
    fig = plt.figure(figsize=(18, 9))
    grid = fig.add_gridspec(
        1,
        2,
        width_ratios=[3.5, 1.2],
        wspace=0.06,
    )
    grid_ax = fig.add_subplot(grid[0, 0])
    grid_ax.axis("off")
    grid_box = grid_ax.get_position()

    control_ax = fig.add_subplot(grid[0, 1])
    control_ax.axis("off")
    control_box = control_ax.get_position()

    def add_grid_axes(x0: float, y0: float, width: float, height: float) -> plt.Axes:
        return fig.add_axes(
            [
                grid_box.x0 + x0 * grid_box.width,
                grid_box.y0 + y0 * grid_box.height,
                width * grid_box.width,
                height * grid_box.height,
            ]
        )

    def add_control_axes(x0: float, y0: float, width: float, height: float) -> plt.Axes:
        return fig.add_axes(
            [
                control_box.x0 + x0 * control_box.width,
                control_box.y0 + y0 * control_box.height,
                width * control_box.width,
                height * control_box.height,
            ]
        )

    group_paths: Dict[str, Path | None] = {group: None for group in groups}
    group_arrays: Dict[str, Dict[str, np.ndarray]] = {}
    group_labels: Dict[str, Dict[str, str]] = {}
    original_paths: Dict[str, Path | None] = {group: None for group in groups}
    original_images: Dict[str, np.ndarray] = {}

    status_message = {"text": "Select ROI folders for each group."}
    auto_scale_state = {"enabled": True}
    zoom_state = {"xlim": None, "ylim": None, "syncing": False}

    help_ax = add_control_axes(0.05, 0.76, 0.90, 0.22)
    help_ax.axis("off")
    help_text = (
        "Grid builder:\n"
        "- Auto-loads group_* folders when available.\n"
        "- Select ROI folder for each group.\n"
        "- Select original image for each group.\n"
        "- Toggle scale bars and pseudocolor if needed.\n"
        "- Adjust smoothing if desired.\n"
        "- Background subtract per channel if needed.\n"
        "- Adjust percentiles for global normalization.\n"
        "- Use EL222 override to set its scale separately.\n"
        "- Auto-scale per image overrides global scaling.\n"
        "- Sync zoom keeps the view matched across groups.\n"
        "- Save SVG when satisfied.\n"
    )
    help_ax.text(0.0, 1.0, help_text, va="top", ha="left", fontsize=9)
    status_artist = help_ax.text(
        0.0,
        0.05,
        f"Status: {status_message['text']}",
        va="bottom",
        ha="left",
        fontsize=9,
    )

    def set_status(message: str) -> None:
        status_message["text"] = message
        status_artist.set_text(f"Status: {message}")
        fig.canvas.draw_idle()

    def channel_summary(channels: Dict[str, np.ndarray]) -> str:
        present = [CHANNEL_LABELS.get(key, key) for key in CHANNEL_ORDER if key in channels]
        missing = [CHANNEL_LABELS.get(key, key) for key in CHANNEL_ORDER if key not in channels]
        summary = f"Channels: {', '.join(present)}"
        if missing:
            summary += f" | Missing: {', '.join(missing)}"
        return summary

    def short_name(value: Path | None, limit: int = 14) -> str:
        if value is None:
            return "None"
        name = value.name
        if len(name) <= limit:
            return name
        return f"{name[:6]}...{name[-6:]}"

    def update_label(group_name: str) -> None:
        roi_name = short_name(group_paths.get(group_name))
        img_name = short_name(original_paths.get(group_name))
        group_labels_text[group_name].set_text(f"{roi_name} | {img_name}")

    def log_channel_stats(group_name: str, channels: Dict[str, np.ndarray]) -> None:
        print(f"[{group_name}] Loaded channels:")
        for key in CHANNEL_ORDER:
            if key not in channels:
                print(f"  - {CHANNEL_LABELS.get(key, key)}: MISSING")
                continue
            arr = channels[key]
            if arr.size == 0:
                print(f"  - {CHANNEL_LABELS.get(key, key)}: EMPTY array")
                continue
            arr_min = float(np.nanmin(arr))
            arr_max = float(np.nanmax(arr))
            arr_mean = float(np.nanmean(arr))
            print(
                f"  - {CHANNEL_LABELS.get(key, key)}: shape={arr.shape} "
                f"min={arr_min:.3f} max={arr_max:.3f} mean={arr_mean:.3f}"
            )

    def log_norm_stats(group_name: str, norm: Dict[str, np.ndarray]) -> None:
        print(f"[{group_name}] Normalized ranges:")
        for key in CHANNEL_ORDER:
            if key not in norm:
                print(f"  - {CHANNEL_LABELS.get(key, key)}: MISSING")
                continue
            arr = norm[key]
            arr_min = float(np.nanmin(arr))
            arr_max = float(np.nanmax(arr))
            arr_mean = float(np.nanmean(arr))
            print(
                f"  - {CHANNEL_LABELS.get(key, key)}: min={arr_min:.3f} "
                f"max={arr_max:.3f} mean={arr_mean:.3f}"
            )

    def log_bounds(group_name: str, bounds_map: Dict[str, Tuple[float, float]]) -> None:
        print(f"[{group_name}] Display bounds:")
        for key in CHANNEL_ORDER:
            if key not in bounds_map:
                print(f"  - {CHANNEL_LABELS.get(key, key)}: MISSING")
                continue
            vmin, vmax = bounds_map[key]
            print(
                f"  - {CHANNEL_LABELS.get(key, key)}: vmin={vmin:.3f} vmax={vmax:.3f}"
            )

    group_buttons = {}
    original_buttons = {}
    group_labels_text = {}
    start_y = 0.68
    for idx, group in enumerate(groups):
        y0 = start_y - idx * 0.07
        ax = add_control_axes(0.05, y0, 0.32, 0.05)
        btn = Button(ax, f"ROI {group}")
        group_buttons[group] = btn

        orig_ax = add_control_axes(0.39, y0, 0.26, 0.05)
        orig_btn = Button(orig_ax, f"Image {group}")
        original_buttons[group] = orig_btn

        label_ax = add_control_axes(0.67, y0, 0.28, 0.05)
        label_ax.axis("off")
        label_text = label_ax.text(0.0, 0.5, "None | None", va="center", ha="left", fontsize=7)
        group_labels_text[group] = label_text

        def make_select(group_name: str):
            def _select(_event):
                path = pick_directory(initial_dir)
                if not path:
                    return
                if not path.name.lower().startswith("group_") and not list(path.glob("roi_raw_ch*_*.npy")):
                    set_status("Select a specific ROI folder that contains roi_raw_ch*.npy files.")
                    return
                try:
                    arrays, labels = load_roi_folder(path)
                except Exception as exc:
                    set_status(f"Failed to load {path.name}: {exc}")
                    return
                group_paths[group_name] = path
                group_arrays[group_name] = arrays
                group_labels[group_name] = labels
                update_label(group_name)
                log_channel_stats(group_name, arrays)
                set_status(f"Loaded {path.name} for {group_name}. {channel_summary(arrays)}")
                update_grid()
            return _select

        btn.on_clicked(make_select(group))

        def make_select_image(group_name: str):
            def _select(_event):
                path = pick_image_path(initial_dir)
                if not path:
                    return
                try:
                    image = load_image(path)
                except Exception as exc:
                    set_status(f"Failed to load image {path.name}: {exc}")
                    return
                original_paths[group_name] = path
                original_images[group_name] = image
                update_label(group_name)
                set_status(f"Loaded image {path.name} for {group_name}.")
                update_grid()
            return _select

        orig_btn.on_clicked(make_select_image(group))

    smooth_ax = add_control_axes(0.05, 0.46, 0.40, 0.06)
    smooth_default = gaussian_filter is not None
    smooth_checks = CheckButtons(smooth_ax, ["Gaussian"], [smooth_default])

    view_ax = add_control_axes(0.52, 0.46, 0.40, 0.06)
    view_checks = CheckButtons(
        view_ax,
        ["Pseudo channels", "Scale bars", "Sync zoom"],
        [False, True, True],
    )

    sigma_ax = add_control_axes(0.05, 0.42, 0.87, 0.03)
    sigma_slider = Slider(sigma_ax, "Sigma", 0.5, 5.0, valinit=1.2, valstep=0.1)

    bg_ax = add_control_axes(0.05, 0.36, 0.40, 0.05)
    bg_checks = CheckButtons(bg_ax, ["Background"], [True])

    bg_chan_ax = add_control_axes(0.52, 0.33, 0.40, 0.10)
    bg_chan_checks = CheckButtons(
        bg_chan_ax,
        ["DAPI", "GFP", "tdTom", "EL222"],
        [False, True, False, True],
    )

    bg_pct_ax = add_control_axes(0.05, 0.29, 0.87, 0.03)
    bg_pct_slider = Slider(bg_pct_ax, "BG %", 0, 20, valinit=5.0, valstep=0.5)

    el_override_ax = add_control_axes(0.05, 0.24, 0.40, 0.05)
    el_override_checks = CheckButtons(el_override_ax, ["EL222 own scale"], [False])

    el_low_ax = add_control_axes(0.05, 0.20, 0.87, 0.03)
    el_high_ax = add_control_axes(0.05, 0.16, 0.87, 0.03)
    el_low_slider = Slider(el_low_ax, "EL Low %", 0, 20, valinit=1.0, valstep=0.5)
    el_high_slider = Slider(el_high_ax, "EL High %", 80, 100, valinit=99.0, valstep=0.5)

    low_ax = add_control_axes(0.05, 0.12, 0.87, 0.03)
    high_ax = add_control_axes(0.05, 0.08, 0.87, 0.03)
    low_slider = Slider(low_ax, "Low %", 0, 20, valinit=1.0, valstep=0.5)
    high_slider = Slider(high_ax, "High %", 80, 100, valinit=99.0, valstep=0.5)

    auto_ax = add_control_axes(0.52, 0.24, 0.40, 0.05)
    auto_button = Button(auto_ax, "Auto-scale per image: ON")

    reset_zoom_ax = add_control_axes(0.52, 0.19, 0.40, 0.05)
    reset_zoom_button = Button(reset_zoom_ax, "Reset zoom")

    save_ax = add_control_axes(0.05, 0.01, 0.87, 0.05)
    save_button = Button(save_ax, "Save SVG")

    grid_images = {}
    row_labels = {}
    grid_cbar_axes: Dict[Tuple[str, int], plt.Axes] = {}
    grid_cbar_objs: Dict[Tuple[str, int], object] = {}

    def build_grid_axes(num_cols: int) -> None:
        grid_images.clear()
        row_labels.clear()
        grid_cbar_axes.clear()
        grid_cbar_objs.clear()
        for group_idx, group in enumerate(groups):
            for col_idx in range(num_cols):
                x0 = col_idx / num_cols
                y0 = 1.0 - (group_idx + 1) / len(groups)
                ax = add_grid_axes(
                    x0 + 0.02,
                    y0 + 0.05,
                    1 / num_cols - 0.03,
                    1 / len(groups) - 0.07,
                )
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_title("")
                for spine in ax.spines.values():
                    spine.set_visible(True)
                    spine.set_linewidth(0.4)
                    spine.set_edgecolor("0.3")
                grid_images[(group, col_idx)] = ax.imshow(
                    np.zeros((10, 10)),
                    cmap="gray",
                    interpolation="nearest",
                )
                if 0 < col_idx <= len(CHANNEL_ORDER):
                    cax = inset_axes(
                        ax,
                        width="4%",
                        height="80%",
                        loc="lower left",
                        bbox_to_anchor=(1.02, 0.1, 1, 1),
                        bbox_transform=ax.transAxes,
                        borderpad=0,
                    )
                    cax.set_visible(False)
                    grid_cbar_axes[(group, col_idx)] = cax

    def all_grid_axes() -> List[plt.Axes]:
        axes = []
        seen = set()
        for img in grid_images.values():
            ax = img.axes
            if ax not in seen:
                axes.append(ax)
                seen.add(ax)
        return axes

    def apply_zoom(xlim: Tuple[float, float], ylim: Tuple[float, float]) -> None:
        zoom_state["syncing"] = True
        for ax in all_grid_axes():
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
        zoom_state["xlim"] = xlim
        zoom_state["ylim"] = ylim
        zoom_state["syncing"] = False

    def update_grid() -> None:
        available_groups = [group for group in groups if group in group_arrays]
        if len(available_groups) == 0:
            set_status("Select ROI folders for each group.")
            return

        smooth_enabled = smooth_checks.get_status()[0]
        if smooth_enabled and gaussian_filter is None:
            set_status("Gaussian smoothing unavailable (install scipy).")
            smooth_enabled = False
        sigma = sigma_slider.val
        view_flags = view_checks.get_status()
        pseudo_enabled = view_flags[0]
        scale_enabled = view_flags[1]
        sync_zoom = view_flags[2]
        bg_enabled = bg_checks.get_status()[0]
        bg_pct = bg_pct_slider.val
        bg_flags = bg_chan_checks.get_status()
        bg_channels = {
            key
            for key, flag in zip(CHANNEL_ORDER, bg_flags)
            if flag
        }
        el_override = el_override_checks.get_status()[0]
        el_low_pct = el_low_slider.val
        el_high_pct = max(el_low_pct + 0.5, el_high_slider.val)
        auto_enabled = auto_scale_state["enabled"]
        prepared = {
            group: prepare_group_channels(
                group_arrays[group],
                smooth_enabled,
                sigma,
                bg_enabled,
                bg_pct,
                bg_channels,
            )
            for group in available_groups
        }

        prepared = crop_to_min_shape(prepared)

        low_pct = low_slider.val
        high_pct = max(low_pct + 0.5, high_slider.val)

        bounds, warnings = resolve_bounds(
            prepared,
            available_groups,
            low_pct,
            high_pct,
            el_override,
            el_low_pct,
            el_high_pct,
        )

        num_cols = len(CHANNEL_ORDER) + 2
        if not grid_images:
            build_grid_axes(num_cols)

        for group in groups:
            for col_idx in range(num_cols):
                img = grid_images[(group, col_idx)]
                ax = img.axes
                ax.set_title("")

        missing_notes = []
        empty_notes = []
        for group_idx, group in enumerate(groups):
            if group not in prepared:
                for col_idx in range(num_cols):
                    img = grid_images[(group, col_idx)]
                    img.set_data(np.zeros((10, 10)))
                missing_notes.append(f"{group}: no data")
                continue

            group_norm = {}
            bounds_map = {}
            for key in CHANNEL_ORDER:
                if key in prepared[group]:
                    if auto_enabled:
                        per_bounds = per_image_bounds(
                            prepared,
                            group,
                            key,
                            low_pct,
                            high_pct,
                            el_override,
                            el_low_pct,
                            el_high_pct,
                        )
                        if per_bounds is not None:
                            bounds_map[key] = per_bounds
                            group_norm[key] = normalize_channel(prepared[group][key], per_bounds)
                    else:
                        if key in bounds:
                            bounds_map[key] = bounds[key]
                            group_norm[key] = normalize_channel(prepared[group][key], bounds[key])
            log_bounds(group, bounds_map)
            log_norm_stats(group, group_norm)

            orig_img = original_images.get(group)
            orig_panel = grid_images[(group, 0)]
            if orig_img is None:
                orig_panel.set_data(np.zeros((10, 10)))
                missing_notes.append(f"{group}: Original")
            else:
                display = normalize_display(orig_img, low_pct, high_pct)
                orig_panel.set_data(display)
                if display.ndim == 2:
                    orig_panel.set_cmap("gray")
                else:
                    orig_panel.set_cmap(None)
                orig_panel.set_clim(0.0, 1.0)
            if group_idx == 0:
                orig_panel.axes.set_title("ORIGINAL")

            for col_idx, key in enumerate(CHANNEL_ORDER):
                img = grid_images[(group, col_idx + 1)]
                ax = img.axes
                if key not in group_norm:
                    img.set_data(np.zeros((10, 10)))
                    missing_notes.append(f"{group}: {CHANNEL_LABELS.get(key, key)}")
                else:
                    if pseudo_enabled:
                        color = CHANNEL_COLORS.get(key, (1.0, 1.0, 1.0))
                        img.set_data(colorize_channel(group_norm[key], color))
                        img.set_cmap(None)
                    else:
                        img.set_data(group_norm[key])
                        img.set_cmap("gray")
                    img.set_clim(0.0, 1.0)
                    if np.nanmax(group_norm[key]) <= 0:
                        empty_notes.append(f"{group}: {CHANNEL_LABELS.get(key, key)} flat")
                cax = grid_cbar_axes.get((group, col_idx + 1))
                if cax is not None:
                    if scale_enabled and key in bounds_map and key in group_norm:
                        cax.set_visible(True)
                        cmap = channel_cmap(key) if pseudo_enabled else plt.get_cmap("gray")
                        norm = Normalize(vmin=bounds_map[key][0], vmax=bounds_map[key][1])
                        cbar = grid_cbar_objs.get((group, col_idx + 1))
                        sm = ScalarMappable(norm=norm, cmap=cmap)
                        if cbar is None:
                            cbar = fig.colorbar(sm, cax=cax)
                            cbar.ax.tick_params(labelsize=6, length=2)
                            grid_cbar_objs[(group, col_idx + 1)] = cbar
                        else:
                            cbar.mappable.set_norm(norm)
                            cbar.mappable.set_cmap(cmap)
                            cbar.update_normal(cbar.mappable)
                        cax.yaxis.set_ticks_position("right")
                    else:
                        cax.set_visible(False)
                ax.set_title(f"{CHANNEL_LABELS.get(key, key).upper()}" if group_idx == 0 else "")

            merge = compose_merge(group_norm) if group_norm else np.zeros((10, 10, 3))
            merge_img = grid_images[(group, num_cols - 1)]
            merge_img.set_data(merge)
            merge_img.set_cmap(None)
            merge_img.axes.set_title("MERGED" if group_idx == 0 else "")

        for group in groups:
            row_ax = grid_images[(group, 0)].axes
            if group not in row_labels:
                row_labels[group] = row_ax.text(
                    -0.05,
                    0.5,
                    group.upper(),
                    transform=row_ax.transAxes,
                    ha="right",
                    va="center",
                    fontsize=10,
                )
            else:
                row_labels[group].set_text(group.upper())

        if sync_zoom and zoom_state["xlim"] is not None and zoom_state["ylim"] is not None:
            apply_zoom(zoom_state["xlim"], zoom_state["ylim"])

        fig.canvas.draw_idle()
        notes = []
        if warnings:
            notes.extend(warnings)
        if missing_notes:
            notes.append("Missing: " + "; ".join(sorted(set(missing_notes))))
        if empty_notes:
            notes.append("Low contrast: " + "; ".join(sorted(set(empty_notes))))
        if bg_enabled:
            notes.append(f"BG%={bg_pct:.1f} on {', '.join(sorted(bg_channels))}")
        if el_override:
            notes.append(f"EL222%={el_low_pct:.1f}-{el_high_pct:.1f}")
        notes.append("Auto-scale on" if auto_enabled else "Auto-scale off")
        if scale_enabled:
            notes.append("Scale bars on")
        if notes:
            set_status("Preview updated. " + " ".join(notes))
        else:
            set_status("Preview updated.")

    def reset_zoom(_event) -> None:
        zoom_state["xlim"] = None
        zoom_state["ylim"] = None
        for img in grid_images.values():
            ax = img.axes
            arr = img.get_array()
            if arr is None:
                continue
            shape = arr.shape
            if len(shape) >= 2:
                height, width = shape[0], shape[1]
                ax.set_xlim(-0.5, width - 0.5)
                ax.set_ylim(height - 0.5, -0.5)
        fig.canvas.draw_idle()

    def on_release(event) -> None:
        if zoom_state["syncing"]:
            return
        view_flags = view_checks.get_status()
        if len(view_flags) < 3 or not view_flags[2]:
            return
        if not grid_images or event.inaxes is None:
            return
        axes_set = {img.axes for img in grid_images.values()}
        if event.inaxes not in axes_set:
            return
        apply_zoom(event.inaxes.get_xlim(), event.inaxes.get_ylim())
        fig.canvas.draw_idle()

    def auto_load_from_root(root: Path) -> None:
        if not root.exists():
            return
        loaded = []
        for group in groups:
            candidate = root / f"group_{group}"
            if not candidate.exists():
                continue
            try:
                arrays, labels = load_roi_folder(candidate)
            except Exception as exc:
                set_status(f"Failed to load {candidate.name}: {exc}")
                continue
            group_paths[group] = candidate
            group_arrays[group] = arrays
            group_labels[group] = labels
            update_label(group)
            log_channel_stats(group, arrays)
            loaded.append(group)
        if loaded:
            set_status(f"Auto-loaded groups: {', '.join(loaded)}.")
            update_grid()

    def save_svg(_event) -> None:
        available_groups = [group for group in groups if group in group_arrays]
        if len(available_groups) != len(groups):
            set_status("Select ROI folders for all selected groups before saving.")
            return
        if out_dir is None:
            initial = initial_dir
        else:
            initial = out_dir
        path = pick_save_path(initial, "roi_group_grid.svg")
        if not path:
            set_status("Save canceled.")
            return
        if path.suffix.lower() != ".svg":
            path = path.with_suffix(".svg")

        smooth_enabled = smooth_checks.get_status()[0]
        sigma = sigma_slider.val
        view_flags = view_checks.get_status()
        pseudo_enabled = view_flags[0]
        scale_enabled = view_flags[1]
        bg_enabled = bg_checks.get_status()[0]
        bg_pct = bg_pct_slider.val
        bg_flags = bg_chan_checks.get_status()
        bg_channels = {
            key
            for key, flag in zip(CHANNEL_ORDER, bg_flags)
            if flag
        }
        el_override = el_override_checks.get_status()[0]
        el_low_pct = el_low_slider.val
        el_high_pct = max(el_low_pct + 0.5, el_high_slider.val)
        auto_enabled = auto_scale_state["enabled"]
        low_pct = low_slider.val
        high_pct = max(low_pct + 0.5, high_slider.val)

        prepared = {
            group: prepare_group_channels(
                group_arrays[group],
                smooth_enabled,
                sigma,
                bg_enabled,
                bg_pct,
                bg_channels,
            )
            for group in available_groups
        }
        prepared = crop_to_min_shape(prepared)

        bounds, warnings = resolve_bounds(
            prepared,
            available_groups,
            low_pct,
            high_pct,
            el_override,
            el_low_pct,
            el_high_pct,
        )

        num_cols = len(CHANNEL_ORDER) + 2
        save_fig = plt.figure(figsize=(num_cols * 2.2, len(groups) * 2.4))
        save_grid = save_fig.add_gridspec(len(groups), num_cols, wspace=0.02, hspace=0.02)

        for row_idx, group in enumerate(groups):
            group_norm = {}
            bounds_map = {}
            if group in prepared:
                for key in CHANNEL_ORDER:
                    if key in prepared[group]:
                        if auto_enabled:
                            per_bounds = per_image_bounds(
                                prepared,
                                group,
                                key,
                                low_pct,
                                high_pct,
                                el_override,
                                el_low_pct,
                                el_high_pct,
                            )
                            if per_bounds is not None:
                                bounds_map[key] = per_bounds
                                group_norm[key] = normalize_channel(prepared[group][key], per_bounds)
                        else:
                            if key in bounds:
                                bounds_map[key] = bounds[key]
                                group_norm[key] = normalize_channel(prepared[group][key], bounds[key])

            ax = save_fig.add_subplot(save_grid[row_idx, 0])
            ax.set_xticks([])
            ax.set_yticks([])
            orig_img = original_images.get(group)
            if orig_img is None:
                ax.imshow(np.zeros((10, 10)), cmap="gray", interpolation="nearest")
            else:
                display = normalize_display(orig_img, low_pct, high_pct)
                if display.ndim == 2:
                    ax.imshow(display, cmap="gray", interpolation="nearest")
                else:
                    ax.imshow(display, interpolation="nearest")
            if row_idx == 0:
                ax.set_title("ORIGINAL", fontsize=10)

            for col_idx, key in enumerate(CHANNEL_ORDER):
                ax = save_fig.add_subplot(save_grid[row_idx, col_idx + 1])
                ax.set_xticks([])
                ax.set_yticks([])
                if key in group_norm:
                    if pseudo_enabled:
                        color = CHANNEL_COLORS.get(key, (1.0, 1.0, 1.0))
                        ax.imshow(colorize_channel(group_norm[key], color), interpolation="nearest")
                    else:
                        ax.imshow(group_norm[key], cmap="gray", interpolation="nearest")
                else:
                    ax.imshow(np.zeros((10, 10)), cmap="gray", interpolation="nearest")
                if row_idx == 0:
                    ax.set_title(CHANNEL_LABELS.get(key, key).upper(), fontsize=10)
                if col_idx == 0:
                    ax.set_ylabel(group.upper(), rotation=0, labelpad=30, fontsize=10, va="center")
                if scale_enabled and key in group_norm:
                    cax = inset_axes(
                        ax,
                        width="4%",
                        height="80%",
                        loc="lower left",
                        bbox_to_anchor=(1.02, 0.1, 1, 1),
                        bbox_transform=ax.transAxes,
                        borderpad=0,
                    )
                    cmap = channel_cmap(key) if pseudo_enabled else plt.get_cmap("gray")
                    if key in bounds_map:
                        sm = ScalarMappable(
                            norm=Normalize(vmin=bounds_map[key][0], vmax=bounds_map[key][1]),
                            cmap=cmap,
                        )
                    else:
                        sm = ScalarMappable(
                            norm=Normalize(vmin=bounds[key][0], vmax=bounds[key][1]),
                            cmap=cmap,
                        )
                    cbar = save_fig.colorbar(sm, cax=cax)
                    cbar.ax.tick_params(labelsize=6, length=2)

            ax = save_fig.add_subplot(save_grid[row_idx, num_cols - 1])
            ax.set_xticks([])
            ax.set_yticks([])
            merge = compose_merge(group_norm) if group_norm else np.zeros((10, 10, 3))
            ax.imshow(merge, interpolation="nearest")
            if row_idx == 0:
                ax.set_title("MERGED", fontsize=10)

        save_fig.savefig(path, format="svg")
        plt.close(save_fig)
        if warnings:
            set_status(f"Saved SVG to {path}. " + " ".join(warnings))
        else:
            set_status(f"Saved SVG to {path}")

    save_button.on_clicked(save_svg)
    reset_zoom_button.on_clicked(reset_zoom)
    fig.canvas.mpl_connect("button_release_event", on_release)

    def toggle_auto_scale(_event) -> None:
        auto_scale_state["enabled"] = not auto_scale_state["enabled"]
        label = "Auto-scale per image: ON" if auto_scale_state["enabled"] else "Auto-scale per image: OFF"
        auto_button.label.set_text(label)
        set_status(label)
        update_grid()

    auto_button.on_clicked(toggle_auto_scale)

    def on_slider_change(_):
        update_grid()

    low_slider.on_changed(on_slider_change)
    high_slider.on_changed(on_slider_change)
    sigma_slider.on_changed(on_slider_change)
    smooth_checks.on_clicked(lambda _: update_grid())
    view_checks.on_clicked(lambda _: update_grid())
    bg_checks.on_clicked(lambda _: update_grid())
    bg_chan_checks.on_clicked(lambda _: update_grid())
    bg_pct_slider.on_changed(on_slider_change)
    el_override_checks.on_clicked(lambda _: update_grid())
    el_low_slider.on_changed(on_slider_change)
    el_high_slider.on_changed(on_slider_change)

    auto_root = roi_root if roi_root is not None else (out_dir if out_dir is not None else initial_dir)
    auto_load_from_root(auto_root)

    plt.show()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a cross-group ROI grid from saved ROI output folders.",
    )
    parser.add_argument(
        "--nd2-dir",
        type=Path,
        default=Path.cwd(),
        help="Starting directory for folder pickers.",
    )
    parser.add_argument(
        "--roi-root",
        type=Path,
        default=None,
        help="Root folder containing group_* ROI folders to auto-load.",
    )
    parser.add_argument(
        "--groups",
        nargs="+",
        default=None,
        help="Group names to include (e.g., ctznmda nmda).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Default output directory for saving the SVG.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_groups = args.groups if args.groups else DEFAULT_GROUPS
    parsed_groups: List[str] = []
    for item in raw_groups:
        for part in str(item).split(","):
            part = part.strip()
            if part:
                parsed_groups.append(part)
    if not parsed_groups:
        parsed_groups = list(DEFAULT_GROUPS)
    seen = set()
    groups = []
    for group in parsed_groups:
        if group not in seen:
            groups.append(group)
            seen.add(group)
    build_gui(args.nd2_dir, args.out_dir, args.roi_root, groups)


if __name__ == "__main__":
    main()
