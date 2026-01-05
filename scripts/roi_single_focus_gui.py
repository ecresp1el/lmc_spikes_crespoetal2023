#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.patches import Rectangle
from matplotlib.figure import Figure
from matplotlib.widgets import Button, RectangleSelector, Slider, CheckButtons

try:
    from scipy.ndimage import gaussian_filter
except ImportError:  # pragma: no cover - optional smoothing
    gaussian_filter = None

plt.rcParams["font.family"] = "Arial"
plt.rcParams["svg.fonttype"] = "none"

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

SVG_DPI = 300


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


def load_roi_folder(folder: Path) -> Dict[str, np.ndarray]:
    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder}")
    raw_files = sorted(folder.glob("roi_raw_ch*_*.npy"))
    if not raw_files:
        raise ValueError(f"No roi_raw_ch*.npy files found in {folder}")
    arrays: Dict[str, np.ndarray] = {}
    for path in raw_files:
        parts = path.stem.split("_", 3)
        if len(parts) < 4:
            continue
        name = parts[3]
        key = channel_key(name)
        arrays[key] = np.load(path)
    if not arrays:
        raise ValueError(f"No channels parsed in {folder}")
    return arrays


def percentile_bounds(channel: np.ndarray, low_pct: float, high_pct: float) -> Tuple[float, float]:
    vmin, vmax = np.percentile(channel, [low_pct, high_pct])
    if vmax <= vmin:
        vmax = vmin + 1.0
    return float(vmin), float(vmax)


def normalize_channel(channel: np.ndarray, bounds: Tuple[float, float], gamma: float) -> np.ndarray:
    vmin, vmax = bounds
    norm = np.clip((channel - vmin) / (vmax - vmin), 0.0, 1.0)
    if gamma != 1.0:
        norm = np.power(norm, 1.0 / gamma)
    return norm


def subtract_background(channel: np.ndarray, pct: float) -> np.ndarray:
    baseline = np.percentile(channel, pct)
    return np.clip(channel - baseline, 0.0, None)


def compose_merge(norm_channels: Dict[str, np.ndarray]) -> np.ndarray:
    ref = next(iter(norm_channels.values()))
    merged = np.zeros((*ref.shape, 3), dtype=np.float32)
    for key, channel in norm_channels.items():
        color = CHANNEL_COLORS.get(key, (1.0, 1.0, 1.0))
        for idx in range(3):
            merged[..., idx] += channel * color[idx]
    return np.clip(merged, 0.0, 1.0)


def colorize_channel(channel: np.ndarray, color: Tuple[float, float, float]) -> np.ndarray:
    rgb = np.zeros((*channel.shape, 3), dtype=np.float32)
    for idx in range(3):
        rgb[..., idx] = channel * color[idx]
    return np.clip(rgb, 0.0, 1.0)


def extract_roi(image: np.ndarray, roi: Tuple[int, int, int, int]) -> np.ndarray:
    x0, y0, w, h = roi
    return image[y0 : y0 + h, x0 : x0 + w]


def build_gui(initial_dir: Path, roi_dir: Path | None, out_dir: Path | None) -> None:
    fig = plt.figure(figsize=(18, 10))
    grid = fig.add_gridspec(2, 2, width_ratios=[3.5, 1.2], height_ratios=[1, 1], wspace=0.08, hspace=0.10)
    full_ax = fig.add_subplot(grid[:, 0])
    full_ax.axis("off")
    control_ax = fig.add_subplot(grid[:, 1])
    control_ax.axis("off")
    control_box = control_ax.get_position()

    def add_control_axes(x0: float, y0: float, width: float, height: float) -> plt.Axes:
        return fig.add_axes(
            [
                control_box.x0 + x0 * control_box.width,
                control_box.y0 + y0 * control_box.height,
                width * control_box.width,
                height * control_box.height,
            ]
        )

    status = {"text": "Select a ROI folder."}
    channel_data: Dict[str, np.ndarray] = {}
    current_folder: Path | None = None
    roi_state = {"x": 0, "y": 0, "w": 50, "h": 50}

    def set_status(text: str) -> None:
        status["text"] = text
        status_artist.set_text(f"Status: {text}")
        fig.canvas.draw_idle()

    help_ax = add_control_axes(0.05, 0.80, 0.90, 0.18)
    help_ax.axis("off")
    help_text = (
        "Single ROI focus:\n"
        "- Load a ROI folder from previous outputs.\n"
        "- Drag on merged view to set ROI size.\n"
        "- Drag inside ROI to move it (size stays fixed).\n"
        "- Adjust ROI sliders if needed.\n"
        "- Apply smoothing if needed.\n"
        "- Save an Illustrator-ready SVG of the full + ROI panels.\n"
    )
    help_ax.text(0.0, 1.0, help_text, va="top", ha="left", fontsize=9)
    status_artist = help_ax.text(0.0, 0.05, f"Status: {status['text']}", va="bottom", ha="left", fontsize=9)

    load_ax = add_control_axes(0.05, 0.72, 0.90, 0.05)
    load_button = Button(load_ax, "Load ROI Folder")

    pseudo_ax = add_control_axes(0.05, 0.66, 0.40, 0.05)
    pseudo_checks = CheckButtons(pseudo_ax, ["Pseudo"], [False])
    smooth_ax = add_control_axes(0.52, 0.66, 0.40, 0.05)
    smooth_default = gaussian_filter is not None
    smooth_checks = CheckButtons(smooth_ax, ["Gaussian"], [smooth_default])

    view_ax = add_control_axes(0.05, 0.60, 0.90, 0.05)
    view_checks = CheckButtons(view_ax, ["Auto-scale", "Background"], [False, False])

    low_ax = add_control_axes(0.05, 0.54, 0.87, 0.03)
    high_ax = add_control_axes(0.05, 0.50, 0.87, 0.03)
    gamma_ax = add_control_axes(0.05, 0.46, 0.87, 0.03)
    sigma_ax = add_control_axes(0.05, 0.42, 0.87, 0.03)
    low_slider = Slider(low_ax, "Low %", 0, 20, valinit=1.0, valstep=0.5)
    high_slider = Slider(high_ax, "High %", 80, 100, valinit=99.0, valstep=0.5)
    gamma_slider = Slider(gamma_ax, "Gamma", 0.2, 3.0, valinit=1.0, valstep=0.05)
    sigma_slider = Slider(sigma_ax, "Sigma", 0.5, 5.0, valinit=1.2, valstep=0.1)

    roi_x_ax = add_control_axes(0.05, 0.38, 0.87, 0.03)
    roi_y_ax = add_control_axes(0.05, 0.34, 0.87, 0.03)
    roi_w_ax = add_control_axes(0.05, 0.30, 0.87, 0.03)
    roi_h_ax = add_control_axes(0.05, 0.26, 0.87, 0.03)
    roi_x_slider = Slider(roi_x_ax, "ROI X", 0, 1, valinit=0, valstep=1)
    roi_y_slider = Slider(roi_y_ax, "ROI Y", 0, 1, valinit=0, valstep=1)
    roi_w_slider = Slider(roi_w_ax, "ROI W", 5, 50, valinit=50, valstep=1)
    roi_h_slider = Slider(roi_h_ax, "ROI H", 5, 50, valinit=50, valstep=1)

    bg_ax = add_control_axes(0.05, 0.22, 0.87, 0.03)
    bg_slider = Slider(bg_ax, "BG %", 0, 20, valinit=5.0, valstep=0.5)

    save_svg_ax = add_control_axes(0.05, 0.14, 0.90, 0.05)
    save_svg_button = Button(save_svg_ax, "Save Figure (SVG)")

    full_axes = []
    crop_axes = []
    full_fig = None
    crop_fig = None

    def init_panels(num_channels: int) -> None:
        nonlocal full_axes, crop_axes, full_fig, crop_fig
        full_fig = fig.add_gridspec(2, num_channels + 1, left=0.02, right=0.62, top=0.95, bottom=0.05, hspace=0.25, wspace=0.15)
        full_axes = [fig.add_subplot(full_fig[0, i]) for i in range(num_channels)]
        full_axes.append(fig.add_subplot(full_fig[0, num_channels]))
        crop_axes = [fig.add_subplot(full_fig[1, i]) for i in range(num_channels)]
        crop_axes.append(fig.add_subplot(full_fig[1, num_channels]))
        for ax in full_axes + crop_axes:
            ax.set_xticks([])
            ax.set_yticks([])

    roi_patch = Rectangle((0, 0), 1, 1, fill=False, edgecolor="yellow", linewidth=2)

    def update_sliders_bounds(height: int, width: int) -> None:
        roi_x_slider.valmin = 0
        roi_x_slider.valmax = max(0, width - 1)
        roi_x_slider.ax.set_xlim(roi_x_slider.valmin, roi_x_slider.valmax)
        roi_y_slider.valmin = 0
        roi_y_slider.valmax = max(0, height - 1)
        roi_y_slider.ax.set_xlim(roi_y_slider.valmin, roi_y_slider.valmax)
        roi_w_slider.valmax = width
        roi_h_slider.valmax = height
        roi_w_slider.ax.set_xlim(roi_w_slider.valmin, roi_w_slider.valmax)
        roi_h_slider.ax.set_xlim(roi_h_slider.valmin, roi_h_slider.valmax)

    def set_roi(x0: int, y0: int, w: int, h: int) -> None:
        roi_state["x"] = int(max(0, x0))
        roi_state["y"] = int(max(0, y0))
        roi_state["w"] = int(max(5, w))
        roi_state["h"] = int(max(5, h))
        roi_x_slider.set_val(roi_state["x"])
        roi_y_slider.set_val(roi_state["y"])
        roi_w_slider.set_val(roi_state["w"])
        roi_h_slider.set_val(roi_state["h"])

    def compute_norm_channels() -> Dict[str, np.ndarray]:
        low_pct = low_slider.val
        high_pct = max(low_pct + 0.5, high_slider.val)
        gamma = gamma_slider.val
        auto_enabled = view_checks.get_status()[0]
        bg_enabled = view_checks.get_status()[1]
        bg_pct = bg_slider.val
        smooth_enabled = smooth_checks.get_status()[0]
        if smooth_enabled and gaussian_filter is None:
            smooth_enabled = False
            set_status("Gaussian smoothing unavailable (install scipy).")
        sigma = sigma_slider.val
        norm_channels: Dict[str, np.ndarray] = {}
        for key, channel in channel_data.items():
            source = channel
            if bg_enabled:
                source = subtract_background(source, bg_pct)
            if smooth_enabled and gaussian_filter is not None:
                source = gaussian_filter(source, sigma=float(sigma))
            if auto_enabled:
                vmin = float(np.nanmin(source))
                vmax = float(np.nanmax(source))
                if vmax <= vmin:
                    vmax = vmin + 1.0
                bounds = (vmin, vmax)
            else:
                bounds = percentile_bounds(source, low_pct, high_pct)
            norm_channels[key] = normalize_channel(source, bounds, gamma)
        return norm_channels

    def update_display() -> None:
        if not channel_data:
            return
        pseudo_enabled = pseudo_checks.get_status()[0]
        auto_enabled = view_checks.get_status()[0]
        bg_enabled = view_checks.get_status()[1]
        bg_pct = bg_slider.val
        smooth_enabled = smooth_checks.get_status()[0] and gaussian_filter is not None
        sigma = sigma_slider.val

        norm_channels = compute_norm_channels()

        merge = compose_merge(norm_channels)
        roi = (roi_state["x"], roi_state["y"], roi_state["w"], roi_state["h"])

        for idx, key in enumerate(CHANNEL_ORDER):
            if key not in norm_channels:
                continue
            full_axes[idx].set_title(CHANNEL_LABELS.get(key, key))
            crop_axes[idx].set_title(f"ROI {CHANNEL_LABELS.get(key, key)}")
            if pseudo_enabled:
                full_axes[idx].imshow(
                    colorize_channel(norm_channels[key], CHANNEL_COLORS[key]),
                    interpolation="nearest",
                )
                crop_axes[idx].imshow(
                    colorize_channel(extract_roi(norm_channels[key], roi), CHANNEL_COLORS[key]),
                    interpolation="nearest",
                )
            else:
                full_axes[idx].imshow(norm_channels[key], cmap="gray", interpolation="nearest")
                crop_axes[idx].imshow(
                    extract_roi(norm_channels[key], roi),
                    cmap="gray",
                    interpolation="nearest",
                )

        full_axes[-1].imshow(merge, interpolation="nearest")
        full_axes[-1].set_title("Merged")
        crop_axes[-1].imshow(extract_roi(merge, roi), interpolation="nearest")
        crop_axes[-1].set_title("ROI Merged")

        full_axes[-1].add_patch(roi_patch)
        roi_patch.set_xy((roi_state["x"], roi_state["y"]))
        roi_patch.set_width(roi_state["w"])
        roi_patch.set_height(roi_state["h"])

        fig.canvas.draw_idle()
        notes = []
        if auto_enabled:
            notes.append("Auto-scale")
        if bg_enabled:
            notes.append(f"BG {bg_pct:.1f}%")
        if smooth_enabled:
            notes.append(f"Smooth {sigma:.1f}")
        if notes:
            set_status("Preview updated (" + ", ".join(notes) + ").")

    def on_select(click, release) -> None:
        if click.xdata is None or release.xdata is None:
            return
        x0, y0 = int(min(click.xdata, release.xdata)), int(min(click.ydata, release.ydata))
        x1, y1 = int(max(click.xdata, release.xdata)), int(max(click.ydata, release.ydata))
        set_roi(x0, y0, max(5, x1 - x0), max(5, y1 - y0))
        update_display()

    selector = RectangleSelector(
        full_axes[-1] if full_axes else full_ax,
        on_select,
        useblit=True,
        button=[1],
        minspanx=5,
        minspany=5,
        spancoords="pixels",
        interactive=False,
    )
    move_state = {"active": False, "offset": (0.0, 0.0)}

    def inside_roi(x: float, y: float) -> bool:
        return (
            roi_state["x"] <= x <= roi_state["x"] + roi_state["w"]
            and roi_state["y"] <= y <= roi_state["y"] + roi_state["h"]
        )

    def on_press(event) -> None:
        if event.inaxes is None or event.button != 1:
            return
        if event.inaxes != selector.ax:
            return
        if event.xdata is None or event.ydata is None:
            return
        if inside_roi(event.xdata, event.ydata):
            move_state["active"] = True
            move_state["offset"] = (
                event.xdata - roi_state["x"],
                event.ydata - roi_state["y"],
            )
            selector.set_active(False)

    def on_motion(event) -> None:
        if not move_state["active"]:
            return
        if event.inaxes != selector.ax:
            return
        if event.xdata is None or event.ydata is None:
            return
        offset_x, offset_y = move_state["offset"]
        new_x = int(round(event.xdata - offset_x))
        new_y = int(round(event.ydata - offset_y))
        set_roi(new_x, new_y, roi_state["w"], roi_state["h"])

    def on_release(event) -> None:
        if not move_state["active"]:
            return
        move_state["active"] = False
        selector.set_active(True)
        update_display()

    def load_folder(_event) -> None:
        path = pick_directory(initial_dir)
        if not path:
            return
        try:
            arrays = load_roi_folder(path)
        except Exception as exc:
            set_status(f"Failed to load {path.name}: {exc}")
            return
        nonlocal channel_data
        channel_data = arrays
        nonlocal current_folder
        current_folder = path
        num_channels = len([key for key in CHANNEL_ORDER if key in channel_data])
        if num_channels == 0:
            set_status("No recognized channels in folder.")
            return
        init_panels(num_channels)
        height, width = next(iter(channel_data.values())).shape
        update_sliders_bounds(height, width)
        set_roi(width // 4, height // 4, width // 2, height // 2)
        selector.ax = full_axes[-1]
        update_display()
        set_status(f"Loaded {path.name}. Draw a smaller ROI.")

    def save_svg(_event) -> None:
        if not channel_data:
            set_status("Load a ROI folder first.")
            return
        target_dir = out_dir if out_dir else Path.cwd()
        target_dir.mkdir(parents=True, exist_ok=True)
        base = current_folder.name if current_folder else "roi_sub"
        svg_path = target_dir / f"{base}_subroi.svg"

        norm_channels = compute_norm_channels()
        roi = (roi_state["x"], roi_state["y"], roi_state["w"], roi_state["h"])
        pseudo_enabled = pseudo_checks.get_status()[0]

        keys = [key for key in CHANNEL_ORDER if key in norm_channels]
        num_channels = len(keys)
        if num_channels == 0:
            set_status("No channels available for SVG export.")
            return

        out_fig = Figure(figsize=(2.6 * (num_channels + 1), 5.2), dpi=SVG_DPI)
        FigureCanvas(out_fig)
        out_grid = out_fig.add_gridspec(2, num_channels + 1, wspace=0.05, hspace=0.1)

        for idx, key in enumerate(keys):
            ax_full = out_fig.add_subplot(out_grid[0, idx])
            ax_crop = out_fig.add_subplot(out_grid[1, idx])
            ax_full.set_xticks([])
            ax_full.set_yticks([])
            ax_crop.set_xticks([])
            ax_crop.set_yticks([])
            ax_full.set_title(CHANNEL_LABELS.get(key, key))
            ax_crop.set_title(f"ROI {CHANNEL_LABELS.get(key, key)}")
            if pseudo_enabled:
                ax_full.imshow(
                    colorize_channel(norm_channels[key], CHANNEL_COLORS[key]),
                    interpolation="nearest",
                )
                ax_crop.imshow(
                    colorize_channel(extract_roi(norm_channels[key], roi), CHANNEL_COLORS[key]),
                    interpolation="nearest",
                )
            else:
                ax_full.imshow(norm_channels[key], cmap="gray", interpolation="nearest")
                ax_crop.imshow(
                    extract_roi(norm_channels[key], roi),
                    cmap="gray",
                    interpolation="nearest",
                )

        merge = compose_merge(norm_channels)
        ax_full = out_fig.add_subplot(out_grid[0, num_channels])
        ax_crop = out_fig.add_subplot(out_grid[1, num_channels])
        for ax in (ax_full, ax_crop):
            ax.set_xticks([])
            ax.set_yticks([])
        ax_full.set_title("Merged")
        ax_crop.set_title("ROI Merged")
        ax_full.imshow(merge, interpolation="nearest")
        ax_crop.imshow(extract_roi(merge, roi), interpolation="nearest")

        out_fig.savefig(svg_path, format="svg", dpi=SVG_DPI, facecolor="white")
        set_status(f"Saved SVG to {svg_path}")

    load_button.on_clicked(load_folder)
    save_svg_button.on_clicked(save_svg)
    low_slider.on_changed(lambda _=None: update_display())
    high_slider.on_changed(lambda _=None: update_display())
    gamma_slider.on_changed(lambda _=None: update_display())
    sigma_slider.on_changed(lambda _=None: update_display())
    bg_slider.on_changed(lambda _=None: update_display())
    roi_x_slider.on_changed(lambda _=None: update_display())
    roi_y_slider.on_changed(lambda _=None: update_display())
    roi_w_slider.on_changed(lambda _=None: update_display())
    roi_h_slider.on_changed(lambda _=None: update_display())
    pseudo_checks.on_clicked(lambda _=None: update_display())
    smooth_checks.on_clicked(lambda _=None: update_display())
    view_checks.on_clicked(lambda _=None: update_display())

    fig.canvas.mpl_connect("button_press_event", on_press)
    fig.canvas.mpl_connect("motion_notify_event", on_motion)
    fig.canvas.mpl_connect("button_release_event", on_release)

    if roi_dir:
        try:
            channel_data = load_roi_folder(roi_dir)
            num_channels = len([key for key in CHANNEL_ORDER if key in channel_data])
            init_panels(num_channels)
            height, width = next(iter(channel_data.values())).shape
            update_sliders_bounds(height, width)
            set_roi(width // 4, height // 4, width // 2, height // 2)
            selector.ax = full_axes[-1]
            update_display()
            set_status(f"Loaded {roi_dir.name}. Draw a smaller ROI.")
        except Exception as exc:
            set_status(f"Failed to load {roi_dir.name}: {exc}")

    plt.show()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Single ROI focus GUI for choosing a smaller representative ROI.",
    )
    parser.add_argument(
        "--roi-dir",
        type=Path,
        default=None,
        help="ROI output folder to load at startup.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output folder for sub-ROI exports.",
    )
    parser.add_argument(
        "--nd2-dir",
        type=Path,
        default=Path.cwd(),
        help="Starting directory for the folder picker.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_gui(args.nd2_dir, args.roi_dir, args.out_dir)


if __name__ == "__main__":
    main()
