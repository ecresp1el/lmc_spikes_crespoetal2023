#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from matplotlib.widgets import Button, RectangleSelector, Slider
from nd2 import ND2File
from scipy.ndimage import rotate

DEFAULT_CHANNEL_NAMES = ["DAPI", "EYFP", "tdTom", "EL222"]
DEFAULT_CHANNEL_COLORS = [
    (0.0, 0.0, 1.0),  # DAPI -> blue
    (0.0, 1.0, 0.0),  # EYFP -> green
    (1.0, 0.0, 0.0),  # tdTom -> red
    (1.0, 0.0, 1.0),  # EL222 -> magenta
]


@dataclass
class RoiState:
    center: Tuple[float, float]
    width: float
    height: float
    angle: float


@dataclass
class DisplayState:
    low_pct: float
    high_pct: float
    gamma: float
    gains: List[float]


@dataclass
class ImageState:
    channels: np.ndarray
    names: List[str]
    colors: List[Tuple[float, float, float]]


@dataclass
class UiState:
    roi: RoiState
    display: DisplayState
    use_roi_scale: bool


def load_nd2_channels(path: Path, channel_indices: List[int], z_project: str) -> np.ndarray:
    with ND2File(str(path)) as nd2_file:
        data = nd2_file.asarray()
        axes = list(nd2_file.axes)

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


def percentile_bounds(channel: np.ndarray, low_pct: float, high_pct: float) -> Tuple[float, float]:
    vmin, vmax = np.percentile(channel, [low_pct, high_pct])
    if vmax <= vmin:
        vmax = vmin + 1.0
    return float(vmin), float(vmax)


def normalize_channel_with_bounds(
    channel: np.ndarray, bounds: Tuple[float, float], gamma: float
) -> np.ndarray:
    vmin, vmax = bounds
    norm = np.clip((channel - vmin) / (vmax - vmin), 0.0, 1.0)
    if gamma != 1.0:
        norm = np.power(norm, 1.0 / gamma)
    return norm


def compose_merge(
    norm_channels: List[np.ndarray],
    colors: List[Tuple[float, float, float]],
    gains: List[float],
) -> np.ndarray:
    merge = np.zeros((*norm_channels[0].shape, 3), dtype=np.float32)
    for channel, color, gain in zip(norm_channels, colors, gains):
        for idx in range(3):
            merge[..., idx] += channel * color[idx] * gain
    return np.clip(merge, 0.0, 1.0)


def extract_rotated_roi(image: np.ndarray, roi: RoiState) -> np.ndarray:
    if image.ndim == 3:
        return np.stack(
            [extract_rotated_roi(image[..., idx], roi) for idx in range(image.shape[2])],
            axis=-1,
        )

    height, width = image.shape
    cx, cy = roi.center
    radius = int(np.ceil(0.5 * np.hypot(roi.width, roi.height)))
    x0, x1 = int(cx - radius), int(cx + radius)
    y0, y1 = int(cy - radius), int(cy + radius)

    pad_left = max(0, -x0)
    pad_right = max(0, x1 - width)
    pad_top = max(0, -y0)
    pad_bottom = max(0, y1 - height)

    if any(v > 0 for v in (pad_left, pad_right, pad_top, pad_bottom)):
        image = np.pad(
            image,
            ((pad_top, pad_bottom), (pad_left, pad_right)),
            mode="edge",
        )
        x0 += pad_left
        x1 += pad_left
        y0 += pad_top
        y1 += pad_top

    window = image[y0:y1, x0:x1]
    rotated = rotate(window, roi.angle, reshape=False, order=1, mode="nearest")

    cx_local = rotated.shape[1] / 2.0
    cy_local = rotated.shape[0] / 2.0
    half_w = roi.width / 2.0
    half_h = roi.height / 2.0

    crop_x0 = int(round(cx_local - half_w))
    crop_x1 = int(round(cx_local + half_w))
    crop_y0 = int(round(cy_local - half_h))
    crop_y1 = int(round(cy_local + half_h))

    return rotated[crop_y0:crop_y1, crop_x0:crop_x1]


def build_gui(image_state: ImageState, ui_state: UiState, save_path: Path | None) -> None:
    channels = image_state.channels
    num_channels = channels.shape[0]

    fig = plt.figure(figsize=(20, 9))
    grid = fig.add_gridspec(
        2,
        num_channels + 1,
        height_ratios=[2.5, 1.8],
        hspace=0.22,
        wspace=0.12,
    )

    full_axes = [fig.add_subplot(grid[0, idx]) for idx in range(num_channels)]
    full_merge_ax = fig.add_subplot(grid[0, num_channels])
    crop_axes = [fig.add_subplot(grid[1, idx]) for idx in range(num_channels)]
    crop_merge_ax = fig.add_subplot(grid[1, num_channels])

    for ax in full_axes + crop_axes + [full_merge_ax, crop_merge_ax]:
        ax.set_xticks([])
        ax.set_yticks([])

    roi_patch = Rectangle((0, 0), 1, 1, fill=False, edgecolor="yellow", linewidth=2)
    roi_patch.set_angle(ui_state.roi.angle)
    full_merge_ax.add_patch(roi_patch)

    def update_roi_patch() -> None:
        roi_patch.set_width(ui_state.roi.width)
        roi_patch.set_height(ui_state.roi.height)
        roi_patch.set_xy(
            (
                ui_state.roi.center[0] - ui_state.roi.width / 2,
                ui_state.roi.center[1] - ui_state.roi.height / 2,
            )
        )
        roi_patch.set_angle(ui_state.roi.angle)

    def update_display(_=None) -> None:
        norm_channels = []
        for channel in channels:
            if ui_state.use_roi_scale:
                roi_channel = extract_rotated_roi(channel, ui_state.roi)
                bounds = percentile_bounds(
                    roi_channel, ui_state.display.low_pct, ui_state.display.high_pct
                )
            else:
                bounds = percentile_bounds(
                    channel, ui_state.display.low_pct, ui_state.display.high_pct
                )
            full_norm = normalize_channel_with_bounds(
                channel, bounds, ui_state.display.gamma
            )
            norm_channels.append(full_norm)

        merge = compose_merge(norm_channels, image_state.colors, ui_state.display.gains)

        for ax, channel, name, color in zip(
            full_axes, norm_channels, image_state.names, image_state.colors
        ):
            ax.imshow(channel, cmap="gray")
            ax.set_title(name)
            for spine in ax.spines.values():
                spine.set_edgecolor(color)
                spine.set_linewidth(2)

        full_merge_ax.imshow(merge)
        full_merge_ax.set_title("Merged")

        update_roi_patch()

        crop_norms = [extract_rotated_roi(channel, ui_state.roi) for channel in norm_channels]
        crop_merge = extract_rotated_roi(merge, ui_state.roi)

        for ax, channel, name in zip(crop_axes, crop_norms, image_state.names):
            ax.imshow(channel, cmap="gray")
            ax.set_title(f"ROI {name}")

        crop_merge_ax.imshow(crop_merge)
        crop_merge_ax.set_title("ROI Merged")

        fig.canvas.draw_idle()

    def on_select(click, release) -> None:
        x0, y0 = click.xdata, click.ydata
        x1, y1 = release.xdata, release.ydata
        if x0 is None or x1 is None:
            return
        ui_state.roi.center = ((x0 + x1) / 2.0, (y0 + y1) / 2.0)
        ui_state.roi.width = max(5.0, abs(x1 - x0))
        ui_state.roi.height = max(5.0, abs(y1 - y0))
        center_x_slider.set_val(ui_state.roi.center[0])
        center_y_slider.set_val(ui_state.roi.center[1])
        width_slider.set_val(ui_state.roi.width)
        height_slider.set_val(ui_state.roi.height)
        update_display()

    RectangleSelector(
        full_merge_ax,
        on_select,
        useblit=True,
        button=[1],
        minspanx=5,
        minspany=5,
        spancoords="pixels",
        interactive=False,
    )

    fig.add_axes([0.05, 0.02, 0.35, 0.12]).axis("off")

    slider_axes = {
        "low": fig.add_axes([0.48, 0.08, 0.22, 0.025]),
        "high": fig.add_axes([0.48, 0.05, 0.22, 0.025]),
        "gamma": fig.add_axes([0.48, 0.02, 0.22, 0.025]),
    }

    roi_slider_axes = {
        "cx": fig.add_axes([0.07, 0.11, 0.30, 0.02]),
        "cy": fig.add_axes([0.07, 0.08, 0.30, 0.02]),
        "w": fig.add_axes([0.07, 0.05, 0.30, 0.02]),
        "h": fig.add_axes([0.07, 0.02, 0.30, 0.02]),
    }

    angle_ax = fig.add_axes([0.05, 0.155, 0.35, 0.02])

    low_slider = Slider(slider_axes["low"], "Low %", 0, 20, valinit=ui_state.display.low_pct, valstep=0.5)
    high_slider = Slider(slider_axes["high"], "High %", 80, 100, valinit=ui_state.display.high_pct, valstep=0.5)
    gamma_slider = Slider(slider_axes["gamma"], "Gamma", 0.2, 3.0, valinit=ui_state.display.gamma, valstep=0.05)

    center_x_slider = Slider(
        roi_slider_axes["cx"],
        "ROI X",
        0,
        channels.shape[2],
        valinit=ui_state.roi.center[0],
        valstep=1,
    )
    center_y_slider = Slider(
        roi_slider_axes["cy"],
        "ROI Y",
        0,
        channels.shape[1],
        valinit=ui_state.roi.center[1],
        valstep=1,
    )
    width_slider = Slider(
        roi_slider_axes["w"],
        "ROI W",
        5,
        channels.shape[2],
        valinit=ui_state.roi.width,
        valstep=1,
    )
    height_slider = Slider(
        roi_slider_axes["h"],
        "ROI H",
        5,
        channels.shape[1],
        valinit=ui_state.roi.height,
        valstep=1,
    )

    angle_slider = Slider(angle_ax, "Angle", -180, 180, valinit=ui_state.roi.angle, valstep=1)

    gain_sliders = []
    for idx in range(num_channels):
        ax = fig.add_axes([0.73, 0.08 - idx * 0.035, 0.23, 0.025])
        slider = Slider(
            ax,
            f"Gain {image_state.names[idx]}",
            0.0,
            3.0,
            valinit=ui_state.display.gains[idx],
            valstep=0.05,
        )
        gain_sliders.append(slider)

    def slider_update(_):
        ui_state.display.low_pct = low_slider.val
        ui_state.display.high_pct = max(low_slider.val + 0.5, high_slider.val)
        ui_state.display.gamma = gamma_slider.val
        ui_state.display.gains = [slider.val for slider in gain_sliders]
        ui_state.roi.center = (center_x_slider.val, center_y_slider.val)
        ui_state.roi.width = width_slider.val
        ui_state.roi.height = height_slider.val
        ui_state.roi.angle = angle_slider.val
        update_display()

    low_slider.on_changed(slider_update)
    high_slider.on_changed(slider_update)
    gamma_slider.on_changed(slider_update)
    angle_slider.on_changed(slider_update)
    center_x_slider.on_changed(slider_update)
    center_y_slider.on_changed(slider_update)
    width_slider.on_changed(slider_update)
    height_slider.on_changed(slider_update)
    for slider in gain_sliders:
        slider.on_changed(slider_update)

    toggle_ax = fig.add_axes([0.72, 0.02, 0.10, 0.035])
    toggle_button = Button(toggle_ax, "ROI Scale")

    def toggle_roi_scale(_):
        ui_state.use_roi_scale = not ui_state.use_roi_scale
        toggle_button.label.set_text("ROI Scale" if ui_state.use_roi_scale else "Full Scale")
        update_display()

    toggle_button.on_clicked(toggle_roi_scale)

    reset_ax = fig.add_axes([0.84, 0.02, 0.12, 0.035])
    reset_button = Button(reset_ax, "Reset ROI")

    def reset_roi(_):
        height, width = channels.shape[1], channels.shape[2]
        ui_state.roi.center = (width / 2.0, height / 2.0)
        ui_state.roi.width = width * 0.3
        ui_state.roi.height = height * 0.3
        ui_state.roi.angle = 0.0
        center_x_slider.set_val(ui_state.roi.center[0])
        center_y_slider.set_val(ui_state.roi.center[1])
        width_slider.set_val(ui_state.roi.width)
        height_slider.set_val(ui_state.roi.height)
        angle_slider.set_val(ui_state.roi.angle)
        update_display()

    reset_button.on_clicked(reset_roi)

    if save_path:
        save_ax = fig.add_axes([0.72, 0.155, 0.24, 0.03])
        save_button = Button(save_ax, "Save ROI JSON")

        def save_roi(_):
            payload = {
                "center": [float(ui_state.roi.center[0]), float(ui_state.roi.center[1])],
                "width": float(ui_state.roi.width),
                "height": float(ui_state.roi.height),
                "angle": float(ui_state.roi.angle),
                "use_roi_scale": bool(ui_state.use_roi_scale),
                "display": {
                    "low_pct": float(ui_state.display.low_pct),
                    "high_pct": float(ui_state.display.high_pct),
                    "gamma": float(ui_state.display.gamma),
                    "gains": [float(gain) for gain in ui_state.display.gains],
                },
            }
            save_path.write_text(json.dumps(payload, indent=2))

        save_button.on_clicked(save_roi)

    update_display()
    plt.show()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ND2 ROI GUI for 4-channel EL222 visualization with rotated ROI controls.",
    )
    parser.add_argument("path", type=Path, help="Path to .nd2 file")
    parser.add_argument(
        "--channels",
        type=int,
        nargs="+",
        default=[0, 1, 2, 3],
        help="Channel indices to load (default: 0 1 2 3)",
    )
    parser.add_argument(
        "--channel-names",
        type=str,
        nargs="+",
        default=DEFAULT_CHANNEL_NAMES,
        help="Channel names (default: DAPI EYFP tdTom EL222)",
    )
    parser.add_argument(
        "--z-project",
        choices=["max", "first"],
        default="max",
        help="How to handle Z stacks (default: max)",
    )
    parser.add_argument(
        "--save-roi",
        type=Path,
        default=None,
        help="Path to save ROI JSON when clicking Save ROI JSON button.",
    )
    parser.add_argument(
        "--low-pct",
        type=float,
        default=1.0,
        help="Low percentile for intensity scaling (default: 1)",
    )
    parser.add_argument(
        "--high-pct",
        type=float,
        default=99.0,
        help="High percentile for intensity scaling (default: 99)",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=1.0,
        help="Gamma for intensity scaling (default: 1.0)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    channels = load_nd2_channels(args.path, args.channels, args.z_project)

    if len(args.channel_names) != channels.shape[0]:
        raise SystemExit(
            f"Expected {channels.shape[0]} channel names, got {len(args.channel_names)}"
        )

    image_state = ImageState(
        channels=channels,
        names=args.channel_names,
        colors=DEFAULT_CHANNEL_COLORS[: channels.shape[0]],
    )

    height, width = channels.shape[1], channels.shape[2]
    ui_state = UiState(
        roi=RoiState(center=(width / 2, height / 2), width=width * 0.3, height=height * 0.3, angle=0.0),
        display=DisplayState(
            low_pct=args.low_pct,
            high_pct=args.high_pct,
            gamma=args.gamma,
            gains=[1.0] * channels.shape[0],
        ),
        use_roi_scale=False,
    )

    print(
        "ROI controls:\n"
        "  drag left mouse on merged panel to set ROI\n"
        "  sliders control ROI center, size, and rotation\n"
        "  ROI Scale toggles percentiles based on the ROI\n"
    )

    build_gui(image_state, ui_state, args.save_roi)


if __name__ == "__main__":
    main()
