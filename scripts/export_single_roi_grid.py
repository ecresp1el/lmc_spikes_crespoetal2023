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
except ImportError:  # pragma: no cover
    ND2File = None

try:
    from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
    from matplotlib.figure import Figure
except ImportError:  # pragma: no cover
    Figure = None
    FigureCanvas = None

try:
    from scipy.ndimage import rotate, gaussian_filter, median_filter
except ImportError:  # pragma: no cover
    rotate = None
    gaussian_filter = None
    median_filter = None

try:
    import tifffile
except ImportError:  # pragma: no cover
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


def normalize_inset_targets(targets: List[str]) -> List[str]:
    normalized = []
    for target in targets:
        key = target.strip().lower()
        if not key:
            continue
        if key == "merged":
            normalized.append("merged")
        else:
            normalized.append(channel_key(key))
    return normalized


def combine_grayscale(images: List[np.ndarray], mode: str) -> np.ndarray:
    if not images:
        raise ValueError("No images to combine.")
    if len(images) == 1:
        return images[0]
    stack = np.stack(images, axis=0)
    mode = mode.lower()
    if mode == "max":
        return np.max(stack, axis=0)
    if mode == "min":
        return np.min(stack, axis=0)
    if mode == "multiply":
        return np.prod(stack, axis=0)
    return np.mean(stack, axis=0)


def diffmap_view(gfp: np.ndarray, el222: np.ndarray, threshold: float) -> np.ndarray:
    g = np.clip(gfp, 0.0, 1.0)
    e = np.clip(el222, 0.0, 1.0)
    if threshold > 0:
        g_mask = g >= threshold
        e_mask = e >= threshold
        g_only = g_mask & ~e_mask
        e_only = e_mask & ~g_mask
        both = g_mask & e_mask
        rgb = np.zeros((*g.shape, 3), dtype=np.float32)
        rgb[..., 0] = e_only.astype(np.float32) + both.astype(np.float32)
        rgb[..., 1] = g_only.astype(np.float32) + both.astype(np.float32)
        return np.clip(rgb, 0.0, 1.0)
    g_only = g * (1.0 - e)
    e_only = e * (1.0 - g)
    both = np.minimum(g, e)
    rgb = np.zeros((*g.shape, 3), dtype=np.float32)
    rgb[..., 0] = e_only + both
    rgb[..., 1] = g_only + both
    return np.clip(rgb, 0.0, 1.0)


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


def normalize_to_uint16(image: np.ndarray) -> np.ndarray:
    image = np.clip(image, 0.0, 1.0)
    return (image * 65535.0).round().astype(np.uint16)


def colorize_channel(channel: np.ndarray, color: Tuple[float, float, float]) -> np.ndarray:
    rgb = np.zeros((*channel.shape, 3), dtype=np.float32)
    for idx in range(3):
        rgb[..., idx] = channel * color[idx]
    return np.clip(rgb, 0.0, 1.0)


def render_colorbar(
    height: int,
    width: int,
    color: Tuple[float, float, float],
    vmin: float,
    vmax: float,
    text_size: int,
) -> np.ndarray:
    bar = np.ones((height, width, 3), dtype=np.float32)
    gradient = np.linspace(1.0, 0.0, height).reshape(height, 1)
    grad_w = max(4, int(width * 0.4))
    for idx in range(3):
        bar[:, :grad_w, idx] = gradient * color[idx]

    if Figure is None or FigureCanvas is None:
        return bar

    dpi = 100
    fig = Figure(figsize=(width / dpi, height / dpi), dpi=dpi, facecolor="white")
    FigureCanvas(fig)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()
    ax.imshow(bar, interpolation="nearest")
    ax.text(
        1,
        1,
        f"{vmax:.1f}",
        ha="left",
        va="top",
        fontsize=text_size,
        color="black",
        bbox={"facecolor": "white", "edgecolor": "none", "pad": 1},
    )
    ax.text(
        1,
        height - 2,
        f"{vmin:.1f}",
        ha="left",
        va="bottom",
        fontsize=text_size,
        color="black",
        bbox={"facecolor": "white", "edgecolor": "none", "pad": 1},
    )
    fig.canvas.draw()
    buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img = buf.reshape(height, width, 3)
    return (img / 255.0).astype(np.float32)


def compose_merge(norm_channels: Dict[str, np.ndarray]) -> np.ndarray:
    ref = next(iter(norm_channels.values()))
    merged = np.zeros((*ref.shape, 3), dtype=np.float32)
    for key, channel in norm_channels.items():
        color = CHANNEL_COLORS.get(key, (1.0, 1.0, 1.0))
        for idx in range(3):
            merged[..., idx] += channel * color[idx]
    return np.clip(merged, 0.0, 1.0)


def correct_illumination(channel: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0 or gaussian_filter is None:
        return channel
    background = gaussian_filter(channel, sigma=float(sigma))
    return np.clip(channel - background, 0.0, None)


def subtract_background_pct(channel: np.ndarray, pct: float) -> np.ndarray:
    if pct <= 0:
        return channel
    baseline = np.percentile(channel, pct)
    return np.clip(channel - baseline, 0.0, None)


def clip_below_percentile(channel: np.ndarray, pct: float) -> np.ndarray:
    if pct <= 0:
        return channel
    threshold = np.percentile(channel, pct)
    return np.where(channel < threshold, 0.0, channel)


def save_tiff(path: Path, image: np.ndarray) -> None:
    if tifffile is None:
        raise RuntimeError("tifffile is required to save TIFF. Install with: pip install tifffile")
    if image.ndim == 2:
        tifffile.imwrite(path, image, photometric="minisblack")
    else:
        tifffile.imwrite(path, image, photometric="rgb")


def clamp_box(
    x: float,
    y: float,
    w: float,
    h: float,
    shape: Tuple[int, int],
) -> Tuple[int, int, int, int]:
    height, width = int(shape[0]), int(shape[1])
    w = max(1, int(round(w)))
    h = max(1, int(round(h)))
    x = int(round(x))
    y = int(round(y))
    x = max(0, min(x, width - 1))
    y = max(0, min(y, height - 1))
    w = min(w, width - x)
    h = min(h, height - y)
    return x, y, w, h


def resize_nearest(image: np.ndarray, height: int, width: int) -> np.ndarray:
    src_h, src_w = image.shape[:2]
    if src_h == height and src_w == width:
        return image
    y_idx = np.linspace(0, src_h - 1, height)
    x_idx = np.linspace(0, src_w - 1, width)
    yy, xx = np.meshgrid(y_idx, x_idx, indexing="ij")
    y0 = np.floor(yy).astype(int)
    x0 = np.floor(xx).astype(int)
    return image[y0, x0]


def pad_to_height(image: np.ndarray, target_h: int) -> np.ndarray:
    height, width = image.shape[:2]
    if height == target_h:
        return image
    pad_total = max(0, target_h - height)
    pad_top = pad_total // 2
    pad_bottom = pad_total - pad_top
    pad_width = ((pad_top, pad_bottom), (0, 0), (0, 0))
    return np.pad(image, pad_width, mode="constant")


def match_height(image: np.ndarray, target_h: int, allow_resize: bool) -> np.ndarray:
    height, width = image.shape[:2]
    if height == target_h:
        return image
    if allow_resize:
        return resize_nearest(image, target_h, width)
    if height > target_h:
        start = max(0, (height - target_h) // 2)
        return image[start : start + target_h]
    return pad_to_height(image, target_h)


def square_box(x: int, y: int, w: int, h: int, size: int | None) -> Tuple[int, int, int, int]:
    if size is None:
        side = min(w, h)
    else:
        side = int(size)
    cx = x + w / 2.0
    cy = y + h / 2.0
    x0 = int(round(cx - side / 2.0))
    y0 = int(round(cy - side / 2.0))
    return x0, y0, side, side


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


def make_grid(images: List[np.ndarray], padding: int, resize_height: int | None = None) -> np.ndarray:
    pad = max(0, int(padding))
    heights = [img.shape[0] for img in images]
    target_h = resize_height if resize_height is not None else min(heights)

    resized = []
    for img in images:
        h, w = img.shape[:2]
        if h == target_h:
            resized.append(img)
            continue
        scale = target_h / float(h)
        new_w = max(1, int(round(w * scale)))
        y_idx = np.linspace(0, h - 1, target_h)
        x_idx = np.linspace(0, w - 1, new_w)
        yy, xx = np.meshgrid(y_idx, x_idx, indexing="ij")
        y0 = np.floor(yy).astype(int)
        x0 = np.floor(xx).astype(int)
        resized.append(img[y0, x0])

    row = resized[0]
    for block in resized[1:]:
        if pad:
            row = np.concatenate([row, np.zeros((row.shape[0], pad, 3), dtype=row.dtype), block], axis=1)
        else:
            row = np.concatenate([row, block], axis=1)
    return row


def pick_insets(image: np.ndarray, count: int, size: int) -> List[Tuple[int, int, int, int]]:
    try:
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
    except Exception as exc:
        print(f"Matplotlib is required for inset picking: {exc}")
        return []

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(image, interpolation="nearest")
    ax.set_title("Drag squares to position. Press Enter to accept.")
    ax.set_xticks([])
    ax.set_yticks([])

    size = max(10, int(size))
    height, width = image.shape[:2]
    pad = max(5, int(size * 0.1))
    rects: List[Rectangle] = []
    for idx in range(count):
        x = pad + idx * (size + pad)
        y = pad
        x = min(x, max(0, width - size))
        y = min(y, max(0, height - size))
        rect = Rectangle((x, y), size, size, fill=False, edgecolor="yellow", linewidth=1.5)
        ax.add_patch(rect)
        rects.append(rect)

    drag = {"rect": None, "offset": (0.0, 0.0)}

    def on_press(event) -> None:
        if event.inaxes != ax or event.button != 1:
            return
        if event.xdata is None or event.ydata is None:
            return
        for rect in rects:
            contains, _ = rect.contains(event)
            if contains:
                drag["rect"] = rect
                drag["offset"] = (event.xdata - rect.get_x(), event.ydata - rect.get_y())
                return

    def on_motion(event) -> None:
        rect = drag["rect"]
        if rect is None:
            return
        if event.inaxes != ax or event.xdata is None or event.ydata is None:
            return
        off_x, off_y = drag["offset"]
        new_x = event.xdata - off_x
        new_y = event.ydata - off_y
        new_x = min(max(0, new_x), max(0, width - size))
        new_y = min(max(0, new_y), max(0, height - size))
        rect.set_xy((new_x, new_y))
        fig.canvas.draw_idle()

    def on_release(_event) -> None:
        drag["rect"] = None

    def on_key(event) -> None:
        if event.key == "enter":
            plt.close(fig)
        if event.key == "escape":
            rects.clear()
            plt.close(fig)

    fig.canvas.mpl_connect("button_press_event", on_press)
    fig.canvas.mpl_connect("motion_notify_event", on_motion)
    fig.canvas.mpl_connect("button_release_event", on_release)
    fig.canvas.mpl_connect("key_press_event", on_key)
    plt.show()

    boxes: List[Tuple[int, int, int, int]] = []
    for rect in rects:
        x, y = rect.get_x(), rect.get_y()
        boxes.append((int(round(x)), int(round(y)), size, size))

    try:
        if not boxes:
            raw = input("No inset selected. Enter x y w h or press Enter to abort: ").strip()
        else:
            raw = ""
    except EOFError:
        raw = ""
    if raw:
        parts = raw.replace(",", " ").split()
        if len(parts) == 4 and all(part.lstrip("-").isdigit() for part in parts):
            x, y, w, h = (int(part) for part in parts)
            boxes.append((x, y, w, h))
    return boxes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a single-condition ROI grid (all channels + merged) at full resolution.",
    )
    parser.add_argument("--roi-dir", type=Path, default=None, help="ROI folder containing roi_state.json.")
    parser.add_argument("--nd2-dir", type=Path, default=None, help="Directory to search for ND2 files.")
    parser.add_argument("--out-dir", type=Path, default=None, help="Output directory for TIFF grid.")
    parser.add_argument(
        "--report-path",
        type=Path,
        default=None,
        help="export_report.json to reuse tdTom bounds from the group comparison.",
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
        "--include-dapi",
        action="store_true",
        help="Include DAPI in the grid (default: off).",
    )
    parser.add_argument(
        "--inset",
        type=int,
        nargs=4,
        action="append",
        default=None,
        metavar=("X", "Y", "W", "H"),
        help="Inset box (x y w h) in ROI crop pixel coordinates. Repeat for multiple.",
    )
    parser.add_argument(
        "--pick-inset",
        action="store_true",
        help="Open a window to click-drag the inset box.",
    )
    parser.add_argument(
        "--inset-target",
        nargs="+",
        default=["el222"],
        help="Inset source(s): merged, eyfp, tdtom, el222, or dapi.",
    )
    parser.add_argument(
        "--inset-composite",
        type=str,
        default="avg",
        help="Composite mode for multiple inset targets: avg, max, min, multiply, diffmap.",
    )
    parser.add_argument(
        "--diff-threshold",
        type=float,
        default=0.2,
        help="Threshold for diffmap overlap (0 uses continuous mode).",
    )
    parser.add_argument(
        "--inset-count",
        type=int,
        default=2,
        help="Number of inset boxes to pick when --pick-inset is used.",
    )
    parser.add_argument(
        "--inset-size",
        type=int,
        default=200,
        help="Square size (pixels) for inset crops (default: 200).",
    )
    parser.add_argument(
        "--reuse-insets",
        action="store_true",
        help="Reuse and reposition inset boxes from the last grid_export_report.json.",
    )
    parser.add_argument(
        "--no-inset-resize",
        action="store_true",
        help="Do not resize inset to match grid height.",
    )
    parser.add_argument(
        "--illumination-sigma",
        type=float,
        default=0.0,
        help="Sigma for background correction (0 disables).",
    )
    parser.add_argument(
        "--illumination-channels",
        nargs="+",
        default=["EL222"],
        help="Channels to apply illumination correction (default: EL222).",
    )
    parser.add_argument(
        "--bg-pct",
        type=float,
        default=0.0,
        help="Percentile to subtract as background (0 disables).",
    )
    parser.add_argument(
        "--bg-channels",
        nargs="+",
        default=["EL222"],
        help="Channels to apply background percentile subtraction (default: EL222).",
    )
    parser.add_argument(
        "--eyfp-bg-pct",
        type=float,
        default=None,
        help="Override background percentile for EYFP only.",
    )
    parser.add_argument(
        "--tdtom-bg-pct",
        type=float,
        default=None,
        help="Override background percentile for tdTom only.",
    )
    parser.add_argument(
        "--el222-bg-pct",
        type=float,
        default=None,
        help="Override background percentile for EL222 only.",
    )
    parser.add_argument(
        "--clip-below-pct",
        type=float,
        default=0.0,
        help="Set values below this percentile to black (0 disables).",
    )
    parser.add_argument(
        "--clip-channels",
        nargs="+",
        default=["EL222"],
        help="Channels to apply clip-below percentile (default: EL222).",
    )
    parser.add_argument(
        "--eyfp-clip-pct",
        type=float,
        default=None,
        help="Override clip-below percentile for EYFP only.",
    )
    parser.add_argument(
        "--tdtom-clip-pct",
        type=float,
        default=None,
        help="Override clip-below percentile for tdTom only.",
    )
    parser.add_argument(
        "--el222-clip-pct",
        type=float,
        default=None,
        help="Override clip-below percentile for EL222 only.",
    )
    parser.add_argument(
        "--despeckle-size",
        type=int,
        default=0,
        help="Median filter size for despeckle (0 disables).",
    )
    parser.add_argument(
        "--despeckle-channels",
        nargs="+",
        default=["EL222"],
        help="Channels to apply despeckle (default: EL222).",
    )
    parser.add_argument(
        "--colorbar",
        action="store_true",
        help="Append a per-channel colorbar to major panels (not merged or insets).",
    )
    parser.add_argument(
        "--colorbar-width",
        type=int,
        default=40,
        help="Width of the colorbar in pixels (default: 40).",
    )
    parser.add_argument(
        "--colorbar-text-size",
        type=int,
        default=8,
        help="Font size for colorbar labels (default: 8).",
    )
    parser.add_argument(
        "--colorbar-height-frac",
        type=float,
        default=0.25,
        help="Height fraction of the panel used for colorbar (default: 0.25).",
    )
    parser.add_argument(
        "--print-ranges",
        action="store_true",
        help="Print vmin/vmax ranges used for each channel.",
    )
    parser.add_argument(
        "--eyfp-from-report",
        action="store_true",
        help="Use EYFP bounds from report-path when available.",
    )
    parser.add_argument(
        "--el222-gain",
        type=float,
        default=1.4,
        help="Gain multiplier for EL222 after scaling (default: 1.4).",
    )
    parser.add_argument(
        "--tdtom-gain",
        type=float,
        default=1.0,
        help="Gain multiplier for tdTom after scaling (default: 1.0).",
    )
    parser.add_argument(
        "--dapi-gain",
        type=float,
        default=1.0,
        help="Gain multiplier for DAPI after scaling (default: 1.0).",
    )
    parser.add_argument("--z-project", choices=["max", "first"], default="max", help="Z projection mode.")
    parser.add_argument("--low-pct", type=float, default=0.5, help="Low percentile for autoscale.")
    parser.add_argument("--high-pct", type=float, default=99.5, help="High percentile for autoscale.")
    parser.add_argument(
        "--auto-scale-only",
        action="store_true",
        help="Ignore bg/illumination/gain overrides and only auto-scale per channel.",
    )
    parser.add_argument("--sigma", type=float, default=0.0, help="Gaussian sigma for smoothing.")
    parser.add_argument("--padding", type=int, default=10, help="Grid padding in pixels.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if len(args.channel_names) != len(args.channels):
        raise SystemExit("channel-names length must match channels length.")

    roi_dir = args.roi_dir
    if roi_dir is None:
        start = args.nd2_dir if args.nd2_dir else Path.cwd()
        roi_dir = pick_directory(start, "Select ROI folder")
    if roi_dir is None:
        raise SystemExit("No ROI folder selected.")
    roi_dir = roi_dir.expanduser()

    roi_state_path = roi_dir / "roi_state.json"
    if not roi_state_path.exists():
        raise FileNotFoundError(f"roi_state.json not found in {roi_dir}")
    roi_state = json.loads(roi_state_path.read_text())

    center = tuple(roi_state["center"])
    width = float(roi_state["width"])
    height = float(roi_state["height"])
    angle = float(roi_state.get("angle", 0.0))

    if angle and rotate is None:
        print("Rotation requested but scipy is not installed; using angle=0.")
        angle = 0.0

    nd2_path = resolve_nd2_path(roi_dir, args.nd2_dir)
    channels = load_nd2_channels(nd2_path, args.channels, args.z_project)
    roi_crop = extract_rotated_roi(channels, center, width, height, angle)

    smooth_sigma = max(0.0, float(args.sigma))
    if smooth_sigma > 0 and gaussian_filter is None:
        print("Gaussian smoothing unavailable (install scipy). Proceeding without smoothing.")
        smooth_sigma = 0.0
    illum_sigma = max(0.0, float(args.illumination_sigma))
    if illum_sigma > 0 and gaussian_filter is None:
        print("Illumination correction unavailable (install scipy). Proceeding without correction.")
        illum_sigma = 0.0
    illum_targets = {channel_key(name) for name in args.illumination_channels}
    despeckle_size = max(0, int(args.despeckle_size))
    if despeckle_size > 0 and median_filter is None:
        print("Median filter unavailable (install scipy). Proceeding without despeckle.")
        despeckle_size = 0
    despeckle_targets = {channel_key(name) for name in args.despeckle_channels}
    bg_pct = max(0.0, float(args.bg_pct))
    bg_targets = {channel_key(name) for name in args.bg_channels}
    clip_pct = max(0.0, float(args.clip_below_pct))
    clip_targets = {channel_key(name) for name in args.clip_channels}
    eyfp_bg_pct = None if args.eyfp_bg_pct is None else max(0.0, float(args.eyfp_bg_pct))
    eyfp_clip_pct = None if args.eyfp_clip_pct is None else max(0.0, float(args.eyfp_clip_pct))
    tdtom_bg_pct = None if args.tdtom_bg_pct is None else max(0.0, float(args.tdtom_bg_pct))
    tdtom_clip_pct = None if args.tdtom_clip_pct is None else max(0.0, float(args.tdtom_clip_pct))
    el222_bg_pct = None if args.el222_bg_pct is None else max(0.0, float(args.el222_bg_pct))
    el222_clip_pct = None if args.el222_clip_pct is None else max(0.0, float(args.el222_clip_pct))
    auto_only = bool(args.auto_scale_only)

    tdtom_bounds = None
    eyfp_bounds = None
    if args.report_path and args.report_path.exists():
        report = json.loads(args.report_path.read_text())
        bounds = report.get("bounds", {})
        report_per_group = bool(report.get("per_group_all"))
        if not report_per_group:
            tdtom_bounds = bounds.get("tdTom")
        if args.eyfp_from_report:
            pooled = report.get("eyfp_bounds_pooled")
            if pooled:
                eyfp_bounds = tuple(float(item) for item in pooled)
            else:
                for name, value in bounds.items():
                    if channel_key(name) == "gfp":
                        eyfp_bounds = tuple(float(item) for item in value)
                        break

    scaled_channels: Dict[str, np.ndarray] = {}
    pseudo_channels: Dict[str, np.ndarray] = {}
    bounds_used: Dict[str, Tuple[float, float]] = {}

    for idx, name in enumerate(args.channel_names):
        key = channel_key(name)
        channel = roi_crop[idx]
        key = channel_key(name)
        if not auto_only:
            if illum_sigma > 0 and key in illum_targets:
                channel = correct_illumination(channel, illum_sigma)
        if key in bg_targets:
            if key == "gfp" and eyfp_bg_pct is not None:
                pct = eyfp_bg_pct
            elif key == "tdtom" and tdtom_bg_pct is not None:
                pct = tdtom_bg_pct
            elif key == "el222" and el222_bg_pct is not None:
                pct = el222_bg_pct
            else:
                pct = bg_pct
            if pct > 0:
                channel = subtract_background_pct(channel, pct)
        if key in clip_targets:
            if key == "gfp" and eyfp_clip_pct is not None:
                pct = eyfp_clip_pct
            elif key == "tdtom" and tdtom_clip_pct is not None:
                pct = tdtom_clip_pct
            elif key == "el222" and el222_clip_pct is not None:
                pct = el222_clip_pct
            else:
                pct = clip_pct
            if pct > 0:
                channel = clip_below_percentile(channel, pct)
        if despeckle_size > 0 and key in despeckle_targets:
            channel = median_filter(channel, size=despeckle_size)
        if smooth_sigma > 0 and gaussian_filter is not None:
            channel = gaussian_filter(channel, sigma=smooth_sigma)
        bounds = None
        if not auto_only and key == "gfp" and eyfp_bounds is not None:
            bounds = tuple(eyfp_bounds)
            scaled = apply_bounds(channel, bounds)
        elif not auto_only and key == "tdtom" and tdtom_bounds is not None:
            bounds = tuple(tdtom_bounds)
            scaled = apply_bounds(channel, bounds)
        else:
            scaled, bounds = scale_channel(channel, args.low_pct, args.high_pct)
        if bounds is not None:
            bounds_used[key] = bounds
        if key == "tdtom":
            gain = float(args.tdtom_gain) if not auto_only else 1.0
            scaled = np.clip(scaled * gain, 0.0, 1.0)
        if key == "el222":
            gain = float(args.el222_gain) if not auto_only else 1.0
            scaled = np.clip(scaled * gain, 0.0, 1.0)
        if key == "dapi":
            scaled = np.clip(scaled * float(args.dapi_gain), 0.0, 1.0)
        scaled_channels[key] = scaled
        color = CHANNEL_COLORS.get(key, (1.0, 1.0, 1.0))
        pseudo_channels[key] = colorize_channel(scaled, color)

    merge = compose_merge(scaled_channels)
    pseudo_merge = merge

    grid_order = ["gfp", "tdtom", "el222"]
    if args.include_dapi:
        grid_order = ["dapi"] + grid_order
    base_images = []
    for key in grid_order:
        if key not in pseudo_channels:
            continue
        panel = pseudo_channels[key]
        if args.colorbar and key in bounds_used:
            color = CHANNEL_COLORS.get(key, (1.0, 1.0, 1.0))
            vmin, vmax = bounds_used[key]
            bar_h = max(10, int(panel.shape[0] * float(args.colorbar_height_frac)))
            bar_w = max(10, int(args.colorbar_width))
            bar_w = min(bar_w, panel.shape[1])
            bar = render_colorbar(
                bar_h,
                bar_w,
                color,
                vmin,
                vmax,
                int(args.colorbar_text_size),
            )
            panel = panel.copy()
            y0 = panel.shape[0] - bar_h
            y1 = panel.shape[0]
            x0 = 0
            x1 = bar_w
            panel[y0:y1, x0:x1] = bar
        base_images.append(panel)
    base_images.append(pseudo_merge)
    base_row = make_grid(base_images, args.padding)

    if args.print_ranges and bounds_used:
        group = None
        try:
            group = roi_dir.parent.name
        except Exception:
            group = None
        label = f"[{group}]" if group else "[roi]"
        for key, bounds in bounds_used.items():
            vmin, vmax = bounds
            print(f"{label} {key} vmin={vmin:.2f} vmax={vmax:.2f} ({args.low_pct:g}-{args.high_pct:g})")

    last_report_path = roi_dir / "nd2_export" / "grid_export_report.json"
    last_report = None
    if last_report_path.exists():
        try:
            last_report = json.loads(last_report_path.read_text())
        except Exception as exc:
            print(f"Failed to read {last_report_path}: {exc}")

    inset_boxes: List[Tuple[int, int, int, int]] = []
    if args.inset:
        inset_boxes.extend([tuple(item) for item in args.inset])
    if args.reuse_insets and last_report and "insets" in last_report:
        previous = last_report.get("insets")
        if isinstance(previous, list):
            inset_boxes.extend([tuple(item) for item in previous if isinstance(item, list) and len(item) == 4])
    if args.pick_inset:
        targets = normalize_inset_targets(args.inset_target)
        inset_source = None
        if "merged" in targets:
            inset_source = pseudo_merge
        elif args.inset_composite.lower() == "diffmap":
            gfp = scaled_channels.get("gfp")
            el222 = scaled_channels.get("el222")
            if gfp is not None and el222 is not None:
                inset_source = diffmap_view(gfp, el222, float(args.diff_threshold))
        else:
            grayscale = []
            for target in targets:
                if target in scaled_channels:
                    grayscale.append(scaled_channels[target])
                elif target in pseudo_channels:
                    grayscale.append(pseudo_channels[target][..., 0])
            if grayscale:
                inset_source = combine_grayscale(grayscale, args.inset_composite)
        if inset_source is None:
            print(f"Inset target(s) {targets} not found; skipping inset picker.")
        else:
            if args.reuse_insets and inset_boxes:
                picked = pick_insets(inset_source, len(inset_boxes), int(args.inset_size))
                inset_boxes = picked
            else:
                picked = pick_insets(inset_source, max(1, int(args.inset_count)), int(args.inset_size))
                if not picked:
                    raise SystemExit("No inset selected. Use drag selection (not the zoom tool).")
                inset_boxes.extend(picked)

    if inset_boxes:
        source_map = {"merged": pseudo_merge, **pseudo_channels}
        inset_columns = []
        for inset_box in inset_boxes:
            inset_panels = []
            for key in grid_order:
                src = source_map.get(key)
                if src is None:
                    continue
                sq_x, sq_y, sq_w, sq_h = square_box(
                    inset_box[0],
                    inset_box[1],
                    inset_box[2],
                    inset_box[3],
                    args.inset_size,
                )
                x, y, w, h = clamp_box(sq_x, sq_y, sq_w, sq_h, src.shape[:2])
                crop = src[y : y + h, x : x + w]
                inset_panels.append(resize_nearest(crop, args.inset_size, args.inset_size))
            src = source_map.get("merged")
            if src is not None:
                sq_x, sq_y, sq_w, sq_h = square_box(
                    inset_box[0],
                    inset_box[1],
                    inset_box[2],
                    inset_box[3],
                    args.inset_size,
                )
                x, y, w, h = clamp_box(sq_x, sq_y, sq_w, sq_h, src.shape[:2])
                crop = src[y : y + h, x : x + w]
                inset_panels.append(resize_nearest(crop, args.inset_size, args.inset_size))
            column = inset_panels[0]
            for panel in inset_panels[1:]:
                if args.padding:
                    column = np.concatenate(
                        [column, np.zeros((args.padding, column.shape[1], 3), dtype=column.dtype), panel],
                        axis=0,
                    )
                else:
                    column = np.concatenate([column, panel], axis=0)
            inset_columns.append(match_height(column, base_row.shape[0], not args.no_inset_resize))

        grid = base_row
        for column in inset_columns:
            if args.padding:
                grid = np.concatenate(
                    [grid, np.zeros((grid.shape[0], args.padding, 3), dtype=grid.dtype), column],
                    axis=1,
                )
            else:
                grid = np.concatenate([grid, column], axis=1)
    else:
        grid = base_row

    output_root = args.out_dir if args.out_dir else roi_dir / "nd2_export"
    output_root.mkdir(parents=True, exist_ok=True)
    grid_path = output_root / "grid_all_channels_pseudo.tif"
    save_tiff(grid_path, normalize_to_uint16(grid))

    report = {
        "roi_dir": str(roi_dir),
        "nd2_path": str(nd2_path),
        "tdtom_bounds": tdtom_bounds,
        "eyfp_from_report": bool(args.eyfp_from_report),
        "eyfp_bounds": list(eyfp_bounds) if eyfp_bounds is not None else None,
        "low_pct": args.low_pct,
        "high_pct": args.high_pct,
        "sigma": smooth_sigma,
        "auto_scale_only": auto_only,
        "illumination_sigma": illum_sigma,
        "illumination_channels": sorted(illum_targets),
        "despeckle_size": despeckle_size,
        "despeckle_channels": sorted(despeckle_targets),
        "bg_pct": bg_pct,
        "bg_channels": sorted(bg_targets),
        "eyfp_bg_pct": eyfp_bg_pct,
        "tdtom_bg_pct": tdtom_bg_pct,
        "el222_bg_pct": el222_bg_pct,
        "clip_below_pct": clip_pct,
        "clip_channels": sorted(clip_targets),
        "eyfp_clip_pct": eyfp_clip_pct,
        "tdtom_clip_pct": tdtom_clip_pct,
        "el222_clip_pct": el222_clip_pct,
        "include_dapi": bool(args.include_dapi),
        "insets": inset_boxes,
        "inset_target": normalize_inset_targets(args.inset_target),
        "inset_count": args.inset_count,
        "inset_size": int(args.inset_size),
        "inset_composite": args.inset_composite,
        "diff_threshold": float(args.diff_threshold),
        "el222_gain": float(args.el222_gain),
        "tdtom_gain": float(args.tdtom_gain),
        "dapi_gain": float(args.dapi_gain),
        "grid_path": str(grid_path),
    }
    (output_root / "grid_export_report.json").write_text(json.dumps(report, indent=2))
    print(f"Saved grid TIFF to {grid_path}")


if __name__ == "__main__":
    main()
