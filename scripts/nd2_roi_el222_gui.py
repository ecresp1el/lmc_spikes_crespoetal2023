#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import transforms
from matplotlib.patches import Rectangle
from matplotlib.widgets import Button, RectangleSelector, Slider
try:
    from nd2 import ND2File
except ImportError as exc:  # pragma: no cover - runtime import guard
    raise SystemExit(
        "Missing dependency 'nd2'. Install with: pip install nd2"
    ) from exc

try:
    from scipy.ndimage import rotate
except ImportError as exc:  # pragma: no cover - runtime import guard
    raise SystemExit(
        "Missing dependency 'scipy'. Install with: pip install scipy"
    ) from exc

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
        raise AttributeError("ND2File has no axes metadata; update nd2 or expose axes order.")

    if hasattr(sizes, "keys"):
        try:
            return _normalize_axes(list(sizes.keys()))
        except TypeError:
            pass

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
        raise AttributeError("Unsupported ND2 sizes metadata; update nd2.") from exc

    if size_list and isinstance(size_list[0], (tuple, list)) and len(size_list[0]) == 2:
        return _normalize_axes([axis for axis, _ in size_list])
    return _normalize_axes(size_list)


def load_nd2_channels(path: Path, channel_indices: List[int], z_project: str) -> np.ndarray:
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


def _safe_name(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "_", name.strip())
    return cleaned or "channel"


def build_gui(
    image_state: ImageState,
    ui_state: UiState,
    save_path: Path | None,
    save_dir: Path | None,
    load_config: dict,
) -> None:
    channels = image_state.channels
    num_channels = channels.shape[0]

    fig = plt.figure(figsize=(20, 10))
    main_grid = fig.add_gridspec(
        2,
        2,
        width_ratios=[3.6, 1.4],
        height_ratios=[2.5, 1.8],
        hspace=0.18,
        wspace=0.06,
    )
    image_grid = main_grid[:, 0].subgridspec(
        2,
        num_channels + 1,
        height_ratios=[2.5, 1.8],
        hspace=0.22,
        wspace=0.12,
    )

    full_axes = [fig.add_subplot(image_grid[0, idx]) for idx in range(num_channels)]
    full_merge_ax = fig.add_subplot(image_grid[0, num_channels])
    crop_axes = [fig.add_subplot(image_grid[1, idx]) for idx in range(num_channels)]
    crop_merge_ax = fig.add_subplot(image_grid[1, num_channels])

    control_ax = fig.add_subplot(main_grid[:, 1])
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

    for ax in full_axes + crop_axes + [full_merge_ax, crop_merge_ax]:
        ax.set_xticks([])
        ax.set_yticks([])

    roi_patch = Rectangle((0, 0), 1, 1, fill=False, edgecolor="yellow", linewidth=2)
    full_merge_ax.add_patch(roi_patch)

    drag_state = {"active": False, "start": (0.0, 0.0), "center": (0.0, 0.0)}

    def compute_norm_channels() -> List[np.ndarray]:
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
        return norm_channels

    def apply_pseudocolor(
        channel: np.ndarray, color: Tuple[float, float, float], gain: float
    ) -> np.ndarray:
        rgb = np.zeros((*channel.shape, 3), dtype=np.float32)
        for idx in range(3):
            rgb[..., idx] = channel * color[idx] * gain
        return np.clip(rgb, 0.0, 1.0)

    def update_roi_patch() -> None:
        roi_patch.set_width(ui_state.roi.width)
        roi_patch.set_height(ui_state.roi.height)
        roi_patch.set_xy(
            (
                ui_state.roi.center[0] - ui_state.roi.width / 2,
                ui_state.roi.center[1] - ui_state.roi.height / 2,
            )
        )
        rot = transforms.Affine2D().rotate_deg_around(
            ui_state.roi.center[0],
            ui_state.roi.center[1],
            ui_state.roi.angle,
        )
        roi_patch.set_transform(rot + full_merge_ax.transData)

    def recompute_crop_norms(norm_channels: List[np.ndarray]) -> List[np.ndarray]:
        return [extract_rotated_roi(channel, ui_state.roi) for channel in norm_channels]

    preview_state = {"enabled": False}
    preview_button = None
    group_state = {"label": load_config.get("group", "ctznmda")}
    save_state = {
        "ok": False,
        "message": "Not saved yet",
        "snapshot": None,
        "ever_saved": False,
    }

    def current_state() -> dict:
        return {
            "low": ui_state.display.low_pct,
            "high": ui_state.display.high_pct,
            "gamma": ui_state.display.gamma,
            "gains": tuple(ui_state.display.gains),
            "roi": (
                ui_state.roi.center,
                ui_state.roi.width,
                ui_state.roi.height,
                ui_state.roi.angle,
            ),
            "use_roi_scale": ui_state.use_roi_scale,
            "preview": preview_state["enabled"],
        }

    def snapshot_state() -> dict:
        return {
            "low": ui_state.display.low_pct,
            "high": ui_state.display.high_pct,
            "gamma": ui_state.display.gamma,
            "gains": tuple(ui_state.display.gains),
            "roi": (
                ui_state.roi.center,
                ui_state.roi.width,
                ui_state.roi.height,
                ui_state.roi.angle,
            ),
            "use_roi_scale": ui_state.use_roi_scale,
            "group": group_state["label"],
            "path": str(current_nd2_path) if current_nd2_path else None,
        }

    cache_norm_channels = compute_norm_channels()
    cache_crop_norms = recompute_crop_norms(cache_norm_channels)
    cache_merge = compose_merge(
        cache_norm_channels, image_state.colors, ui_state.display.gains
    )
    cache_crop_merge = compose_merge(
        cache_crop_norms, image_state.colors, ui_state.display.gains
    )
    prev_state = current_state()

    full_images = []
    for ax, channel, name, color in zip(
        full_axes, cache_norm_channels, image_state.names, image_state.colors
    ):
        im = ax.imshow(channel, cmap="gray", interpolation="nearest")
        ax.set_title(name)
        for spine in ax.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(2)
        full_images.append(im)

    full_merge_im = full_merge_ax.imshow(cache_merge, interpolation="nearest")
    full_merge_ax.set_title("Merged")

    crop_images = []
    for ax, channel, name in zip(crop_axes, cache_crop_norms, image_state.names):
        im = ax.imshow(channel, cmap="gray", interpolation="nearest")
        ax.set_title(f"ROI {name}")
        crop_images.append(im)

    crop_merge_im = crop_merge_ax.imshow(cache_crop_merge, interpolation="nearest")
    crop_merge_ax.set_title("ROI Merged")

    update_roi_patch()

    def update_display(_=None) -> None:
        nonlocal cache_norm_channels, cache_crop_norms, cache_merge, cache_crop_merge, prev_state
        state = current_state()
        roi_changed = state["roi"] != prev_state["roi"]
        scale_changed = (
            state["low"] != prev_state["low"]
            or state["high"] != prev_state["high"]
            or state["gamma"] != prev_state["gamma"]
            or state["use_roi_scale"] != prev_state["use_roi_scale"]
        )
        gains_changed = state["gains"] != prev_state["gains"]
        preview_changed = state["preview"] != prev_state["preview"]

        if scale_changed or (roi_changed and state["use_roi_scale"]):
            cache_norm_channels = compute_norm_channels()
            for im, channel in zip(full_images, cache_norm_channels):
                im.set_data(channel)

            cache_merge = compose_merge(
                cache_norm_channels, image_state.colors, ui_state.display.gains
            )
            full_merge_im.set_data(cache_merge)

            cache_crop_norms = recompute_crop_norms(cache_norm_channels)
            update_crop_images(cache_crop_norms)

            cache_crop_merge = compose_merge(
                cache_crop_norms, image_state.colors, ui_state.display.gains
            )
            crop_merge_im.set_data(cache_crop_merge)
        else:
            if roi_changed:
                cache_crop_norms = recompute_crop_norms(cache_norm_channels)
                update_crop_images(cache_crop_norms)
                cache_crop_merge = compose_merge(
                    cache_crop_norms, image_state.colors, ui_state.display.gains
                )
                crop_merge_im.set_data(cache_crop_merge)

            if gains_changed:
                cache_merge = compose_merge(
                    cache_norm_channels, image_state.colors, ui_state.display.gains
                )
                full_merge_im.set_data(cache_merge)
                cache_crop_merge = compose_merge(
                    cache_crop_norms, image_state.colors, ui_state.display.gains
                )
                crop_merge_im.set_data(cache_crop_merge)
                if preview_state["enabled"]:
                    update_crop_images(cache_crop_norms)

        if preview_changed and not scale_changed:
            update_crop_images(cache_crop_norms)

        if roi_changed:
            update_roi_patch()

        if save_state["ok"] and save_state["snapshot"] is not None:
            if snapshot_state() != save_state["snapshot"]:
                save_state["ok"] = False
                save_state["snapshot"] = None
                save_state["message"] = "Unsaved changes"
                update_help_text()

        prev_state = state
        fig.canvas.draw_idle()

    update_timer = fig.canvas.new_timer(interval=60)
    update_timer.single_shot = True

    def request_update() -> None:
        update_timer.stop()
        update_timer.start()

    update_timer.add_callback(update_display)

    def on_select(click, release) -> None:
        if drag_state["active"]:
            return
        x0, y0 = click.xdata, click.ydata
        x1, y1 = release.xdata, release.ydata
        if x0 is None or x1 is None:
            return
        ui_state.roi.center = ((x0 + x1) / 2.0, (y0 + y1) / 2.0)
        ui_state.roi.width = max(5.0, abs(x1 - x0))
        ui_state.roi.height = max(5.0, abs(y1 - y0))
        set_slider_val(center_x_slider, ui_state.roi.center[0])
        set_slider_val(center_y_slider, ui_state.roi.center[1])
        set_slider_val(width_slider, ui_state.roi.width)
        set_slider_val(height_slider, ui_state.roi.height)
        request_update()

    rect_selector = RectangleSelector(
        full_merge_ax,
        on_select,
        useblit=True,
        button=[1],
        minspanx=5,
        minspany=5,
        spancoords="pixels",
        interactive=False,
    )

    help_ax = add_control_axes(0.05, 0.79, 0.90, 0.19)
    help_ax.axis("off")

    def render_help_text() -> str:
        return (
            "ROI how-to:\n"
            "- Drag on merged image to draw ROI.\n"
            "- Drag inside yellow ROI to move it.\n"
            "- Use ROI X/Y/W/H sliders to move/resize.\n"
            "- Use Angle slider to rotate the ROI.\n"
            "- ROI Scale toggles percentile scaling.\n"
            "- Preview toggles pseudocolor in ROI panels (press P).\n"
            "- Save ROI + Preview writes outputs + enables preview.\n"
            "- Save ROI Crops writes PNG + NPY outputs.\n"
            "- Group button cycles ctznmda/ctz/nmda.\n"
            "- Load ND2 opens a file picker.\n"
            f"Group: {group_state['label']} | Status: {save_state['message']}"
        )

    help_artist = help_ax.text(0.0, 1.0, render_help_text(), va="top", ha="left", fontsize=9)

    def update_help_text() -> None:
        help_artist.set_text(render_help_text())
        fig.canvas.draw_idle()

    slider_axes = {
        "low": add_control_axes(0.53, 0.46, 0.42, 0.04),
        "high": add_control_axes(0.53, 0.41, 0.42, 0.04),
        "gamma": add_control_axes(0.53, 0.36, 0.42, 0.04),
    }

    roi_slider_axes = {
        "cx": add_control_axes(0.05, 0.46, 0.42, 0.04),
        "cy": add_control_axes(0.05, 0.41, 0.42, 0.04),
        "w": add_control_axes(0.05, 0.36, 0.42, 0.04),
        "h": add_control_axes(0.05, 0.31, 0.42, 0.04),
    }

    angle_ax = add_control_axes(0.05, 0.25, 0.42, 0.04)

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
    gain_gap = 0.008
    gain_start = 0.24
    min_gain_y = 0.05
    if num_channels > 0:
        max_total = max(0.0, gain_start - min_gain_y)
        gain_height = min(
            0.035, max(0.02, (max_total - gain_gap * (num_channels - 1)) / num_channels)
        )
    else:
        gain_height = 0.035
    for idx in range(num_channels):
        y0 = gain_start - idx * (gain_height + gain_gap)
        ax = add_control_axes(0.53, y0, 0.42, gain_height)
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
        request_update()

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

    def set_slider_val(slider: Slider, value: float) -> None:
        slider.eventson = False
        slider.set_val(value)
        slider.eventson = True

    def clamp_center(x: float, y: float) -> Tuple[float, float]:
        x = max(0.0, min(float(x), channels.shape[2]))
        y = max(0.0, min(float(y), channels.shape[1]))
        return x, y

    def on_press(event) -> None:
        if event.inaxes != full_merge_ax or event.button != 1:
            return
        if event.xdata is None or event.ydata is None:
            return
        contains, _ = roi_patch.contains(event)
        if not contains:
            return
        drag_state["active"] = True
        drag_state["start"] = (event.xdata, event.ydata)
        drag_state["center"] = ui_state.roi.center
        rect_selector.set_active(False)

    def on_motion(event) -> None:
        if not drag_state["active"]:
            return
        if event.xdata is None or event.ydata is None:
            return
        dx = event.xdata - drag_state["start"][0]
        dy = event.ydata - drag_state["start"][1]
        new_center = clamp_center(drag_state["center"][0] + dx, drag_state["center"][1] + dy)
        ui_state.roi.center = new_center
        set_slider_val(center_x_slider, new_center[0])
        set_slider_val(center_y_slider, new_center[1])
        request_update()

    def on_release(_event) -> None:
        if not drag_state["active"]:
            return
        drag_state["active"] = False
        rect_selector.set_active(True)

    fig.canvas.mpl_connect("button_press_event", on_press)
    fig.canvas.mpl_connect("motion_notify_event", on_motion)
    fig.canvas.mpl_connect("button_release_event", on_release)

    current_nd2_path = load_config.get("current_path")

    group_ax = add_control_axes(0.05, 0.72, 0.42, 0.05)
    group_button = Button(group_ax, f"Group {group_state['label']}")

    def cycle_group(_event) -> None:
        groups = ["ctznmda", "ctz", "nmda"]
        current = group_state["label"]
        idx = groups.index(current) if current in groups else 0
        next_label = groups[(idx + 1) % len(groups)]
        group_state["label"] = next_label
        group_button.label.set_text(f"Group {next_label}")
        save_state["ok"] = False
        save_state["snapshot"] = None
        save_state["message"] = f"Group set to {next_label}. Save before loading next ND2."
        update_help_text()

    group_button.on_clicked(cycle_group)

    load_ax = add_control_axes(0.53, 0.72, 0.42, 0.05)
    load_button = Button(load_ax, "Load ND2")

    def update_slider_limits(width: float, height: float) -> None:
        center_x_slider.valmin = 0
        center_x_slider.valmax = width
        center_x_slider.ax.set_xlim(center_x_slider.valmin, center_x_slider.valmax)
        center_y_slider.valmin = 0
        center_y_slider.valmax = height
        center_y_slider.ax.set_xlim(center_y_slider.valmin, center_y_slider.valmax)
        width_slider.valmin = 5
        width_slider.valmax = width
        width_slider.ax.set_xlim(width_slider.valmin, width_slider.valmax)
        height_slider.valmin = 5
        height_slider.valmax = height
        height_slider.ax.set_xlim(height_slider.valmin, height_slider.valmax)

    def update_image_limits(width: float, height: float) -> None:
        for ax, im in zip(full_axes, full_images):
            ax.set_xlim(0, width)
            ax.set_ylim(height, 0)
            im.set_extent((0, width, height, 0))
        full_merge_ax.set_xlim(0, width)
        full_merge_ax.set_ylim(height, 0)
        full_merge_im.set_extent((0, width, height, 0))

    def load_nd2_from_path(path: Path) -> None:
        nonlocal channels, cache_norm_channels, cache_crop_norms, cache_merge, cache_crop_merge, prev_state, current_nd2_path
        new_channels = load_nd2_channels(
            path,
            load_config["channel_indices"],
            load_config["z_project"],
        )
        if new_channels.shape[0] != num_channels:
            print(
                f"ND2 has {new_channels.shape[0]} channels, expected {num_channels}; "
                "restart the GUI for a different channel count."
            )
            return

        current_nd2_path = path
        load_config["current_path"] = path
        load_config["initial_dir"] = str(path.parent)

        channels = new_channels
        image_state.channels = new_channels

        height, width = channels.shape[1], channels.shape[2]
        ui_state.roi.width = min(ui_state.roi.width, width)
        ui_state.roi.height = min(ui_state.roi.height, height)
        ui_state.roi.center = clamp_center(ui_state.roi.center[0], ui_state.roi.center[1])
        if ui_state.roi.center[0] <= 0 or ui_state.roi.center[1] <= 0:
            ui_state.roi.center = (width / 2.0, height / 2.0)

        update_slider_limits(width, height)
        set_slider_val(center_x_slider, ui_state.roi.center[0])
        set_slider_val(center_y_slider, ui_state.roi.center[1])
        set_slider_val(width_slider, ui_state.roi.width)
        set_slider_val(height_slider, ui_state.roi.height)

        cache_norm_channels = compute_norm_channels()
        cache_crop_norms = recompute_crop_norms(cache_norm_channels)
        cache_merge = compose_merge(
            cache_norm_channels, image_state.colors, ui_state.display.gains
        )
        cache_crop_merge = compose_merge(
            cache_crop_norms, image_state.colors, ui_state.display.gains
        )
        for im, channel in zip(full_images, cache_norm_channels):
            im.set_data(channel)
        for im, channel in zip(crop_images, cache_crop_norms):
            im.set_data(channel)
        full_merge_im.set_data(cache_merge)
        crop_merge_im.set_data(cache_crop_merge)
        update_image_limits(width, height)

        prev_state = current_state()
        update_display()
        save_state["ok"] = False
        save_state["snapshot"] = None
        save_state["message"] = f"Loaded {path.name}. Save ROI before loading next ND2."
        update_help_text()

    def pick_nd2_path() -> Path | None:
        initial_dir = Path(load_config.get("initial_dir", Path.cwd()))

        if sys.platform == "darwin":
            default_loc = str(initial_dir)
            script = (
                'POSIX path of (choose file with prompt "Select ND2 file" '
                f'default location POSIX file "{default_loc}")'
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
            from tkinter import Tk, TclError, filedialog
        except Exception as exc:
            print(f"Tkinter is not available for file selection: {exc}")
            return None

        root = Tk()
        root.withdraw()
        try:
            root.attributes("-topmost", True)
            root.lift()
            root.update()
        except TclError:
            pass
        path = filedialog.askopenfilename(
            title="Select ND2 file",
            initialdir=str(initial_dir),
            filetypes=[("ND2 files", "*.nd2"), ("All files", "*.*")],
            parent=root,
        )
        root.destroy()

        return Path(path) if path else None

    def load_new_nd2(_event) -> None:
        if save_state["ever_saved"] and current_nd2_path and not save_state["ok"]:
            save_state["message"] = "Save ROI crops before loading the next ND2."
            update_help_text()
            print("Save ROI crops before loading the next ND2.")
            return
        path = pick_nd2_path()
        if not path:
            print("No ND2 selected.")
            return
        load_nd2_from_path(path)

    load_button.on_clicked(load_new_nd2)

    toggle_ax = add_control_axes(0.05, 0.48, 0.42, 0.05)
    toggle_button = Button(toggle_ax, "ROI Scale")

    def toggle_roi_scale(_):
        ui_state.use_roi_scale = not ui_state.use_roi_scale
        toggle_button.label.set_text("ROI Scale" if ui_state.use_roi_scale else "Full Scale")
        request_update()

    toggle_button.on_clicked(toggle_roi_scale)

    reset_ax = add_control_axes(0.53, 0.48, 0.42, 0.05)
    reset_button = Button(reset_ax, "Reset ROI")

    def reset_roi(_):
        height, width = channels.shape[1], channels.shape[2]
        ui_state.roi.center = (width / 2.0, height / 2.0)
        ui_state.roi.width = width * 0.3
        ui_state.roi.height = height * 0.3
        ui_state.roi.angle = 0.0
        set_slider_val(center_x_slider, ui_state.roi.center[0])
        set_slider_val(center_y_slider, ui_state.roi.center[1])
        set_slider_val(width_slider, ui_state.roi.width)
        set_slider_val(height_slider, ui_state.roi.height)
        set_slider_val(angle_slider, ui_state.roi.angle)
        request_update()

    reset_button.on_clicked(reset_roi)

    def resolve_save_dir() -> Path | None:
        group_dir = f"group_{group_state['label'].lower()}"
        if save_dir:
            if current_nd2_path:
                return save_dir / group_dir / f"{current_nd2_path.stem}_roi"
            return save_dir / group_dir
        if save_path:
            return save_path.parent / group_dir / f"{save_path.stem}_roi"
        if current_nd2_path:
            return current_nd2_path.parent / group_dir / f"{current_nd2_path.stem}_roi"
        return None

    def update_crop_images(crop_norms: List[np.ndarray]) -> None:
        if preview_state["enabled"]:
            for im, channel, color, gain in zip(
                crop_images, crop_norms, image_state.colors, ui_state.display.gains
            ):
                im.set_data(apply_pseudocolor(channel, color, gain))
                im.set_extent((0, channel.shape[1], channel.shape[0], 0))
        else:
            for im, channel in zip(crop_images, crop_norms):
                im.set_data(channel)
                im.set_extent((0, channel.shape[1], channel.shape[0], 0))

    def set_preview_enabled(enabled: bool) -> None:
        preview_state["enabled"] = enabled
        if preview_button is not None:
            preview_button.label.set_text("Hide Preview" if enabled else "Preview ROI")
        request_update()

    def save_roi_outputs(target_dir: Path) -> None:
        target_dir.mkdir(parents=True, exist_ok=True)
        raw_crops = [extract_rotated_roi(channel, ui_state.roi) for channel in channels]
        norm_channels = compute_norm_channels()
        roi_norms = [extract_rotated_roi(channel, ui_state.roi) for channel in norm_channels]

        for idx, (raw, norm, name, color, gain) in enumerate(
            zip(raw_crops, roi_norms, image_state.names, image_state.colors, ui_state.display.gains)
        ):
            safe_name = _safe_name(name)
            np.save(target_dir / f"roi_raw_ch{idx}_{safe_name}.npy", raw.astype(np.float32))
            plt.imsave(
                target_dir / f"roi_gray_ch{idx}_{safe_name}.png",
                norm,
                cmap="gray",
            )
            plt.imsave(
                target_dir / f"roi_pseudo_ch{idx}_{safe_name}.png",
                apply_pseudocolor(norm, color, gain),
            )

        roi_merge = compose_merge(roi_norms, image_state.colors, ui_state.display.gains)
        plt.imsave(target_dir / "roi_merge.png", roi_merge)

        payload = {
            "center": [float(ui_state.roi.center[0]), float(ui_state.roi.center[1])],
            "width": float(ui_state.roi.width),
            "height": float(ui_state.roi.height),
            "angle": float(ui_state.roi.angle),
            "use_roi_scale": bool(ui_state.use_roi_scale),
            "group": group_state["label"],
            "display": {
                "low_pct": float(ui_state.display.low_pct),
                "high_pct": float(ui_state.display.high_pct),
                "gamma": float(ui_state.display.gamma),
                "gains": [float(gain) for gain in ui_state.display.gains],
            },
        }
        (target_dir / "roi_state.json").write_text(json.dumps(payload, indent=2))

    def toggle_preview() -> None:
        set_preview_enabled(not preview_state["enabled"])

    def on_key(event) -> None:
        if event.key and event.key.lower() == "p":
            toggle_preview()

    fig.canvas.mpl_connect("key_press_event", on_key)

    save_ax = add_control_axes(0.05, 0.66, 0.90, 0.05)
    save_button = Button(save_ax, "Save ROI + Preview")

    def save_roi(_):
        payload = {
            "center": [float(ui_state.roi.center[0]), float(ui_state.roi.center[1])],
            "width": float(ui_state.roi.width),
            "height": float(ui_state.roi.height),
            "angle": float(ui_state.roi.angle),
            "use_roi_scale": bool(ui_state.use_roi_scale),
            "group": group_state["label"],
            "display": {
                "low_pct": float(ui_state.display.low_pct),
                "high_pct": float(ui_state.display.high_pct),
                "gamma": float(ui_state.display.gamma),
                "gains": [float(gain) for gain in ui_state.display.gains],
            },
        }
        target_dir = resolve_save_dir()
        if target_dir is None:
            save_state["message"] = "Set --save-dir or load an ND2 to save outputs."
            update_help_text()
            print(save_state["message"])
            return
        try:
            if save_path:
                save_path.write_text(json.dumps(payload, indent=2))
            save_roi_outputs(target_dir)
        except Exception as exc:
            save_state["ok"] = False
            save_state["snapshot"] = None
            save_state["message"] = f"Save failed: {exc}"
            update_help_text()
            print(save_state["message"])
            return

        save_state["ok"] = True
        save_state["ever_saved"] = True
        save_state["snapshot"] = snapshot_state()
        save_state["message"] = f"Saved to {target_dir}"
        update_help_text()
        print(save_state["message"])
        set_preview_enabled(True)

    save_button.on_clicked(save_roi)

    preview_ax = add_control_axes(0.05, 0.60, 0.90, 0.05)
    preview_button = Button(preview_ax, "Preview ROI")
    preview_button.on_clicked(lambda _: toggle_preview())

    save_crops_ax = add_control_axes(0.05, 0.54, 0.90, 0.05)
    save_crops_button = Button(save_crops_ax, "Save ROI Crops")

    def save_crops(_):
        target_dir = resolve_save_dir()
        if target_dir is None:
            save_state["message"] = "Set --save-dir or load an ND2 to save outputs."
            update_help_text()
            print(save_state["message"])
            return

        try:
            save_roi_outputs(target_dir)
        except Exception as exc:
            save_state["ok"] = False
            save_state["snapshot"] = None
            save_state["message"] = f"Save failed: {exc}"
            update_help_text()
            print(save_state["message"])
            return

        save_state["ok"] = True
        save_state["ever_saved"] = True
        save_state["snapshot"] = snapshot_state()
        save_state["message"] = f"Saved to {target_dir}"
        update_help_text()
        print(save_state["message"])

    save_crops_button.on_clicked(save_crops)

    update_display()
    plt.show()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ND2 ROI GUI for 4-channel EL222 visualization with rotated ROI controls.",
    )
    parser.add_argument(
        "path",
        type=Path,
        nargs="?",
        default=None,
        help="Optional path to .nd2 file (omit to use the Load ND2 button).",
    )
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
        "--save-dir",
        type=Path,
        default=None,
        help="Directory to save ROI crops (PNG + NPY). Defaults next to --save-roi if set.",
    )
    parser.add_argument(
        "--nd2-dir",
        type=Path,
        default=None,
        help="Starting directory for the ND2 file picker.",
    )
    parser.add_argument(
        "--group",
        choices=["ctznmda", "ctz", "nmda"],
        default="ctznmda",
        help="Initial experimental group label (ctznmda, ctz, or nmda).",
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
    if args.path is not None:
        channels = load_nd2_channels(args.path, args.channels, args.z_project)
    else:
        num_channels = len(args.channels)
        channels = np.zeros((num_channels, 256, 256), dtype=np.float32)

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
        "  drag inside yellow ROI to move it\n"
        "  use ROI X/Y/W/H sliders to move/resize\n"
        "  Angle slider rotates the ROI\n"
        "  ROI Scale toggles percentiles based on the ROI\n"
        "  Preview toggles pseudocolor in ROI panels (press P)\n"
        "  Save ROI + Preview writes outputs and enables preview\n"
        "  Save ROI Crops writes PNG + NPY outputs\n"
        "  Group button cycles ctznmda/ctz/nmda\n"
        "  Load ND2 opens a file picker (requires a save)\n"
    )

    if args.path is not None:
        initial_dir = args.nd2_dir if args.nd2_dir else args.path.parent
    else:
        initial_dir = args.nd2_dir if args.nd2_dir else Path.cwd()
    load_config = {
        "channel_indices": args.channels,
        "z_project": args.z_project,
        "initial_dir": str(initial_dir),
        "current_path": args.path,
        "group": args.group,
    }
    build_gui(image_state, ui_state, args.save_roi, args.save_dir, load_config)


if __name__ == "__main__":
    main()
