#!/usr/bin/env python3
from __future__ import annotations

"""
In‑vivo Expression Grid (per condition × channels + RGB overlay)
================================================================

What this does
--------------
- Scans user‑provided condition folders for a specific animal/file ID and three channels
  (defaults: C1=DAPI, C2=EYFP, C3=tdTom) stored as TIFFs.
- Loads one grayscale image per channel for each condition, computes per‑channel global
  display limits (e.g., 1–99th percentiles) across conditions, and renders a grid:
  rows = conditions, cols = channels + an RGB overlay (R=tdTom/C3, G=EYFP/C2, B=DAPI/C1).
- Saves SVG and PDF next to the output base path.

Requirements
------------
- Python packages: tifffile, numpy, matplotlib
  Install: `pip install tifffile matplotlib numpy`

Usage examples
--------------
Basic (three conditions, three channels; uses the first matching ID in each folder):
  python -m scripts.plot_invivo_expression_grid \
    --condition "Luciferin+NMDA=/path/to/ctz_nmda" \
    --condition "Luciferin=/path/to/ctz_only" \
    --condition "NMDA=/path/to/no_ctz_veh_nmda" \
    --ids 1961 1971 19121 \
    --out LMC4f_invivo_expression_grid

If each condition should use a specific ID, pass an explicit map:
  python -m scripts.plot_invivo_expression_grid \
    --condition "Luciferin+NMDA=/path/ctz_nmda" \
    --condition "Luciferin=/path/ctz_only" \
    --condition "NMDA=/path/no_ctz_veh_nmda" \
    --id-map "Luciferin+NMDA:1961" --id-map "Luciferin:1971" --id-map "NMDA:19121" \
    --out LMC4f_invivo_expression_grid

Flags
-----
- --condition NAME=DIR     Add a condition (can be passed multiple times)
- --ids ID [ID ...]       Candidate IDs to search per condition (first match wins)
- --id-map NAME:ID        Force a specific ID per condition (overrides --ids)
- --low --high            Percentiles for global display limits per channel (default 1, 99)
- --out PATH              Output path base (writes .svg and .pdf)
- --chan-tag C:NAME       Human tag per channel key (defaults: C1:DAPI C2:EYFP C3:tdTom)
- --chan-cmap C:CMAP      Colormap name per channel (defaults: C1:Blues C2:Greens C3:Reds)
- --name-contains SUBSTR  Only use files whose names include these substrings
                          (repeatable; case-insensitive). Default filters: "zoomed", "step2".

Troubleshooting
---------------
- "No file found for …": ensure filenames include both the ID and the channel tag like `_C1`, `_C2`, `_C3`.
- Mixed image sizes: this tool assumes all images for a given condition have identical shape.
  If not, inspect inputs and resample externally before plotting to avoid misalignment in the RGB.
"""

import argparse
import glob
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.ticker import FuncFormatter
from matplotlib import patches
import tifffile


def _rational_to_float(val) -> float:
    try:
        # tifffile may return Rational, Fraction, tuple, or int
        if hasattr(val, 'numerator') and hasattr(val, 'denominator'):
            return float(val.numerator) / float(val.denominator)
        if isinstance(val, (tuple, list)) and len(val) == 2:
            return float(val[0]) / float(val[1])
        return float(val)
    except Exception:
        return float('nan')


def try_physical_size_um(path: str, shape: Tuple[int, int]) -> Tuple[Optional[float], Optional[float]]:
    """Best-effort physical width/height in micrometers from TIFF tags.

    Returns (width_um, height_um) or (None, None) if unavailable.
    """
    h, w = shape[:2]
    try:
        with tifffile.TiffFile(path) as tf:
            page = tf.pages[0]
            tags = page.tags
            try:
                unit_code = tags['ResolutionUnit'].value  # 1=None, 2=inch, 3=cm
            except KeyError:
                unit_code = None
            try:
                xres = _rational_to_float(tags['XResolution'].value)  # pixels per unit
                yres = _rational_to_float(tags['YResolution'].value)
            except KeyError:
                xres = yres = float('nan')

            if unit_code == 2:  # inch
                unit_um = 25400.0
            elif unit_code == 3:  # centimeter
                unit_um = 10000.0
            else:
                unit_um = None

            if unit_um is None or not np.isfinite(xres) or not np.isfinite(yres) or xres <= 0 or yres <= 0:
                return (None, None)

            px_size_x_um = unit_um / xres
            px_size_y_um = unit_um / yres
            width_um = w * px_size_x_um
            height_um = h * px_size_y_um
            return (width_um, height_um)
    except Exception:
        return (None, None)


def read_tiff_resolution(path: str) -> Tuple[Optional[float], Optional[float], Optional[str]]:
    try:
        with tifffile.TiffFile(path) as tf:
            page = tf.pages[0]
            tags = page.tags
            try:
                unit_code = tags['ResolutionUnit'].value
            except KeyError:
                unit_code = None
            try:
                xres = _rational_to_float(tags['XResolution'].value)
                yres = _rational_to_float(tags['YResolution'].value)
            except KeyError:
                xres = yres = None
            if unit_code == 2:
                unit = 'INCH'
            elif unit_code == 3:
                unit = 'CENTIMETER'
            else:
                unit = None
            return xres, yres, unit
    except Exception:
        return (None, None, None)


def write_tiff_preserve_resolution(path: Path, arr: np.ndarray, res: Tuple[Optional[float], Optional[float], Optional[str]]) -> None:
    xres, yres, unit = res
    kwargs = {}
    if xres and yres and unit:
        kwargs['resolution'] = (xres, yres)
        kwargs['resolutionunit'] = unit
    if arr.ndim == 3 and arr.shape[-1] in (3, 4):
        kwargs['photometric'] = 'rgb'
    else:
        kwargs['photometric'] = 'minisblack'
    kwargs['metadata'] = None
    tifffile.imwrite(path, arr, **kwargs)
def try_pixel_size_um(path: str) -> Tuple[Optional[float], Optional[float]]:
    """Return (px_um_x, px_um_y) from TIFF tags if available; else (None, None)."""
    try:
        with tifffile.TiffFile(path) as tf:
            page = tf.pages[0]
            tags = page.tags
            try:
                unit_code = tags['ResolutionUnit'].value  # 1=None, 2=inch, 3=cm
            except KeyError:
                unit_code = None
            try:
                xres = _rational_to_float(tags['XResolution'].value)  # pixels per unit
                yres = _rational_to_float(tags['YResolution'].value)
            except KeyError:
                return (None, None)

            if unit_code == 2:  # inch
                unit_um = 25400.0
            elif unit_code == 3:  # centimeter
                unit_um = 10000.0
            else:
                return (None, None)

            if not np.isfinite(xres) or not np.isfinite(yres) or xres <= 0 or yres <= 0:
                return (None, None)
            return (unit_um / xres, unit_um / yres)
    except Exception:
        return (None, None)


def parse_kv(items: List[str], sep: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for it in items or []:
        if sep not in it:
            raise ValueError(f"Expected key{sep}value, got: {it}")
        k, v = it.split(sep, 1)
        k = k.strip(); v = v.strip()
        if not k or not v:
            raise ValueError(f"Empty key or value in: {it}")
        out[k] = v
    return out


def load_matching_tiff(folder: Path, file_id: str, channel_tag: str, contains: Optional[List[str]] = None) -> Tuple[np.ndarray, str]:
    pattern = str(folder / f"*{file_id}*_{channel_tag}*.tif*")
    matches = glob.glob(pattern)
    # Optional filename substring filtering (case-insensitive)
    if contains:
        tokens = [t.lower() for t in contains if t]
        matches = [m for m in matches if all(tok in os.path.basename(m).lower() for tok in tokens)]
    if not matches:
        raise FileNotFoundError(
            f"No file found for id={file_id} chan={channel_tag} in {folder}"
            + (f" with substrings {contains}" if contains else "")
        )
    # Prefer shortest filename or most recent? Keep first for now
    chosen = matches[0]
    img = tifffile.imread(chosen)
    # Ensure 2D
    if img.ndim > 2:
        # Take first plane if z-stack or multi-page
        img = np.asarray(img)[0]
    return np.asarray(img), chosen


def compute_global_percentiles(imgs: List[np.ndarray], low: float, high: float) -> Tuple[float, float]:
    flat = np.concatenate([np.ravel(im) for im in imgs]).astype(float)
    return float(np.percentile(flat, low)), float(np.percentile(flat, high))


def _argparse() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot condition × channels grid with RGB overlay")
    p.add_argument('--condition', action='append', default=[
        'Luciferin+NMDA=/Users/ecrespo/Desktop/lmc_invivo_images_el222/data-Manny-1/LMC4f coded raw images/Coded Images/lmc4_invivo_fiji_outputinages/ctz_nmda',
        'Luciferin=/Users/ecrespo/Desktop/lmc_invivo_images_el222/data-Manny-1/LMC4f coded raw images/Coded Images/lmc4_invivo_fiji_outputinages/ctz_only',
        'NMDA=/Users/ecrespo/Desktop/lmc_invivo_images_el222/data-Manny-1/LMC4f coded raw images/Coded Images/lmc4_invivo_fiji_outputinages/no_ctz_veh_nmda',
    ], help='NAME=DIR per condition (repeat)')
    p.add_argument('--ids', nargs='*', default=['1961','1971','19121'], help='Candidate IDs to search for (first match wins)')
    p.add_argument('--id-map', action='append', default=[], help='NAME:ID — force ID per condition (repeat)')
    p.add_argument('--low', type=float, default=1.0, help='Lower percentile for display (default 1)')
    p.add_argument('--high', type=float, default=99.0, help='Upper percentile for display (default 99)')
    p.add_argument('--out', type=Path, default=Path('LMC4f_invivo_expression_grid'), help='Output base path (no suffix)')
    p.add_argument('--chan-tag', action='append', default=['C1:DAPI','C2:EYFP','C3:tdTom'], help='C:TAG labels (repeat)')
    p.add_argument('--chan-cmap', action='append', default=['C1:Blues','C2:Greens','C3:Reds'], help='C:CMAP colormap names (repeat)')
    p.add_argument('--name-contains', action='append', default=None, help='Only use files whose names include these substrings (repeatable; case-insensitive). Default: zoomed, step2')
    p.add_argument('--align-mismatch', choices=['none','crop'], default='crop', help='If channel shapes differ within a condition: center-crop to smallest or do nothing (may error). Default: crop')
    p.add_argument('--save-merged', type=Path, default=None, help='Directory to save per-condition merged RGB TIFFs (8-bit). If not set, saves under out-root/merged')
    p.add_argument('--norm-mode', choices=['global','per_condition','match_ref'], default='match_ref', help='Normalization across conditions: global percentiles, per-condition percentiles, or match to a reference condition by quantile scaling')
    p.add_argument('--norm-quantile', type=float, default=99.0, help='Quantile (0-100) used for matching when norm-mode=match_ref (default 99)')
    p.add_argument('--ref-condition', type=str, default='Luciferin+NMDA', help='Reference condition name for norm-mode=match_ref; if not present, falls back to first condition')
    p.add_argument('--gamma', type=float, default=1.0, help='Gamma adjustment applied after normalization (>=0). 1.0 means no change')
    p.add_argument('--colorbar-mode', choices=['normalized','raw','none'], default='normalized', help='Show colorbars in normalized a.u., raw intensity labels, or hide them')
    p.add_argument('--scalebar-um', type=float, default=100.0, help='Scalebar length in µm on RGB panel. 0 means auto-pick; negative disables')
    p.add_argument('--scalebar-pos', choices=['lower left','lower right','upper left','upper right'], default='lower right', help='Scalebar anchor position on RGB panel')
    p.add_argument('--scalebar-color', type=str, default='white', help='Scalebar and label color')
    p.add_argument('--out-root', type=Path, default=None, help='Root output directory where all exports are collated. Default: common parent of condition dirs + /<out base>')
    a = p.parse_args()
    # Default to requiring both 'zoomed' and 'step2' unless user provided filters
    if a.name_contains is None:
        a.name_contains = ['zoomed', 'step2']
    return a


def main() -> int:
    args = _argparse()

    cond_map = parse_kv(args.condition, '=')
    if not cond_map:
        raise SystemExit('Provide at least one --condition NAME=DIR')
    id_map = parse_kv(args.id_map, ':')

    chan_tags = parse_kv(args.chan_tag, ':')
    chan_cmaps = parse_kv(args.chan_cmap, ':')
    # Validate channels
    channels = [c for c in ('C1','C2','C3') if c in chan_tags]
    if len(channels) != 3:
        # Proceed with whatever was specified but warn in console
        print('[warn] Expected channels C1,C2,C3; proceeding with:', channels)

    # Resolve folder paths
    cond_dirs: Dict[str, Path] = {}
    for name, d in cond_map.items():
        p = Path(d)
        if not p.exists():
            raise SystemExit(f'Condition folder not found: {name} -> {p}')
        cond_dirs[name] = p

    # Derive a common output root if not provided
    if args.out_root is None:
        common_parent = Path(os.path.commonpath([str(p) for p in cond_dirs.values()]))
        out_root = common_parent / args.out.name
    else:
        out_root = args.out_root
    figures_dir = out_root / 'figures'
    channels_raw_dir = out_root / 'channels' / 'raw'
    channels_norm_dir = out_root / 'channels' / 'norm'
    channels_pseudo_dir = out_root / 'channels' / 'pseudocolor'
    merged_dir = args.save_merged if args.save_merged is not None else (out_root / 'merged')
    for d in (figures_dir, channels_raw_dir, channels_norm_dir, channels_pseudo_dir, merged_dir):
        d.mkdir(parents=True, exist_ok=True)

    # Step 1 — load and group images by condition
    grouped: Dict[str, Dict[str, np.ndarray]] = {}
    chosen_files: Dict[str, Dict[str, str]] = {}
    cond_px_um: Dict[str, Tuple[Optional[float], Optional[float]]] = {}
    cond_res: Dict[str, Tuple[Optional[float], Optional[float], Optional[str]]] = {}
    chosen_id: Dict[str, str] = {}
    for name, folder in cond_dirs.items():
        # pick ID: explicit map first, else first candidate that matches any channel
        file_id = id_map.get(name)
        if file_id is None:
            # try given ids in order, honoring optional name filters
            tokens = [t.lower() for t in (args.name_contains or []) if t]
            for cand in args.ids:
                pattern = str(folder / f"*{cand}*_C1*.tif*")
                matches = glob.glob(pattern)
                if tokens:
                    matches = [m for m in matches if all(tok in os.path.basename(m).lower() for tok in tokens)]
                if matches:
                    file_id = cand; break
        if file_id is None:
            raise SystemExit(f'Could not determine an ID for condition {name}. Use --id-map {name}:ID or provide matching --ids.')
        # Load per channel
        grouped[name] = {}
        chosen_files[name] = {}
        chosen_id[name] = file_id
        shapes = []
        print(f"[pick] Condition={name}  folder={folder}  ID={file_id}")
        for ch in channels:
            img, fpath = load_matching_tiff(folder, file_id, ch, contains=args.name_contains)
            grouped[name][ch] = img
            chosen_files[name][ch] = fpath
            shapes.append(img.shape)
            w_um, h_um = try_physical_size_um(fpath, img.shape)
            if w_um is not None and h_um is not None:
                print(f"  - {ch}: {fpath}  shape={img.shape}  size≈{w_um:.2f}×{h_um:.2f} µm")
            else:
                print(f"  - {ch}: {fpath}  shape={img.shape}")
        # Pixel size (µm/px) per condition from C1 if available
        ref_path = chosen_files[name].get('C1') or next(iter(chosen_files[name].values()))
        cond_px_um[name] = try_pixel_size_um(ref_path)
        cond_res[name] = read_tiff_resolution(ref_path)
        if len({s for s in shapes}) != 1:
            print(f"[warn] Images for {name} have different shapes {shapes}.")
            if args.align_mismatch == 'crop':
                # Center-crop all channels to the smallest HxW
                min_h = min(s[0] for s in shapes)
                min_w = min(s[1] for s in shapes)
                def center_crop(a: np.ndarray, th: int, tw: int) -> np.ndarray:
                    h, w = a.shape[:2]
                    rs = max((h - th) // 2, 0)
                    cs = max((w - tw) // 2, 0)
                    return a[rs:rs+th, cs:cs+tw]
                for ch in channels:
                    grouped[name][ch] = center_crop(grouped[name][ch], min_h, min_w)
                print(f"[info] Aligned by center-cropping to {min_h}x{min_w} for condition {name}.")
            else:
                print(f"[warn] align-mismatch=none; proceeding without alignment. RGB stack may fail.")

    # Step 2 — build display arrays using selected normalization mode (maps to 0..1)
    cond_names = list(grouped.keys())
    disp: Dict[str, Dict[str, np.ndarray]] = {name: {} for name in cond_names}
    # Limits for raw units (used for labeling if requested)
    raw_limits_global: Dict[str, Tuple[float,float]] = {}
    raw_limits_percond: Dict[str, Dict[str, Tuple[float,float]]] = {name: {} for name in cond_names}

    def apply_gamma(x: np.ndarray) -> np.ndarray:
        g = float(args.gamma)
        if g <= 0:
            return x
        return np.power(x, 1.0 / g)

    if args.norm_mode == 'global':
        # global limits per channel across all conditions
        for ch in channels:
            imgs = [grouped[name][ch] for name in cond_names]
            vmin, vmax = compute_global_percentiles(imgs, low=float(args.low), high=float(args.high))
            raw_limits_global[ch] = (vmin, vmax)
            for name in cond_names:
                arr = grouped[name][ch].astype(float)
                nrm = np.clip((arr - vmin) / (vmax - vmin + 1e-12), 0.0, 1.0)
                disp[name][ch] = apply_gamma(nrm)
                raw_limits_percond[name][ch] = (vmin, vmax)
    elif args.norm_mode == 'per_condition':
        for ch in channels:
            for name in cond_names:
                arr = grouped[name][ch].astype(float)
                vmin, vmax = compute_global_percentiles([arr], low=float(args.low), high=float(args.high))
                raw_limits_percond[name][ch] = (vmin, vmax)
                nrm = np.clip((arr - vmin) / (vmax - vmin + 1e-12), 0.0, 1.0)
                disp[name][ch] = apply_gamma(nrm)
    else:  # match_ref
        ref = args.ref_condition or cond_names[0]
        if ref not in grouped:
            print(f"[warn] Reference condition '{ref}' not found. Falling back to '{cond_names[0]}'")
            ref = cond_names[0]
        q = float(args.norm_quantile)
        # scale each condition to match the reference quantile per channel
        scaled: Dict[str, Dict[str, np.ndarray]] = {name: {} for name in cond_names}
        for ch in channels:
            ref_arr = grouped[ref][ch].astype(float)
            ref_q = float(np.percentile(ref_arr, q))
            for name in cond_names:
                arr = grouped[name][ch].astype(float)
                cond_q = float(np.percentile(arr, q))
                scale = ref_q / (cond_q + 1e-12)
                scaled[name][ch] = arr * scale
        # now compute global limits across scaled arrays per channel, then normalize
        for ch in channels:
            imgs = [scaled[name][ch] for name in cond_names]
            vmin, vmax = compute_global_percentiles(imgs, low=float(args.low), high=float(args.high))
            raw_limits_global[ch] = (vmin, vmax)
            for name in cond_names:
                arr = scaled[name][ch]
                nrm = np.clip((arr - vmin) / (vmax - vmin + 1e-12), 0.0, 1.0)
                disp[name][ch] = apply_gamma(nrm)
                raw_limits_percond[name][ch] = (vmin, vmax)

    # Step 3 — plot grid
    nrows = len(grouped)
    ncols = len(channels) + 1  # channels + RGB
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(4*ncols, 3.6*nrows))
    if nrows == 1:
        axes = np.array([axes])

    for i, cond in enumerate(cond_names):
        aligned_shape = None
        for j, ch in enumerate(channels):
            ax = axes[i, j]
            img_disp = disp[cond][ch]
            im = ax.imshow(img_disp, cmap=chan_cmaps.get(ch, 'gray'), vmin=0.0, vmax=1.0)
            ax.set_title(f"{cond} — {chan_tags.get(ch, ch)}", fontsize=10)
            ax.axis('off')
            if args.colorbar_mode != 'none':
                cbar = fig.colorbar(im, ax=ax, shrink=0.65)
                cbar.ax.tick_params(labelsize=8)
                if args.colorbar_mode == 'normalized':
                    cbar.set_label('Normalized (a.u.)', fontsize=8)
                else:  # raw labels on a normalized bar
                    vmin_raw, vmax_raw = raw_limits_percond[cond][ch]
                    span = (vmax_raw - vmin_raw)
                    def to_raw(x):
                        return vmin_raw + x * span
                    cbar.formatter = FuncFormatter(lambda x, pos: f"{to_raw(x):.0f}")
                    cbar.set_label('Raw intensity', fontsize=8)
                    cbar.update_ticks()
        # RGB overlay in last column
        ax = axes[i, ncols-1]
        def _norm(chkey: str) -> np.ndarray:
            return disp[cond][chkey]
        # R=tdTom (C3), G=EYFP (C2), B=DAPI (C1)
        r = _norm('C3') if 'C3' in channels else np.zeros_like(grouped[cond][channels[0]])
        g = _norm('C2') if 'C2' in channels else np.zeros_like(r)
        b = _norm('C1') if 'C1' in channels else np.zeros_like(r)
        try:
            rgb = np.stack([r, g, b], axis=-1)
        except ValueError as e:
            print(f"[error] Cannot create RGB for condition '{cond}' due to shape mismatch: R{r.shape} G{g.shape} B{b.shape}.")
            print("        Selected files:")
            for ch in channels:
                fpath = chosen_files.get(cond, {}).get(ch)
                if fpath is not None:
                    print(f"         {ch}: {fpath}  shape={grouped[cond][ch].shape}")
            raise
        ax.imshow(rgb)
        ax.set_title(f"{cond} — RGB", fontsize=10)
        ax.axis('off')

        # Draw scalebar on RGB panel if pixel size is known
        if args.scalebar_um < 0:
            pass  # disabled
        else:
            px_um_x, px_um_y = cond_px_um.get(cond, (None, None))
            if px_um_x is not None and px_um_y is not None:
                H, W = rgb.shape[:2]
                # auto-pick length ≈ 1/5 of width, snapped to nice values
                if args.scalebar_um == 0.0:
                    width_um = W * px_um_x
                    candidates = [10, 20, 50, 100, 200, 500, 1000]
                    target = width_um * 0.2
                    bar_um = max([c for c in candidates if c <= target] or [candidates[0]])
                else:
                    bar_um = float(args.scalebar_um)
                bar_px = max(int(round(bar_um / px_um_x)), 1)
                # placement
                pad = max(int(round(min(H, W) * 0.02)), 6)
                thick = max(int(round(min(H, W) * 0.006)), 2)
                if args.scalebar_pos == 'lower right':
                    x0 = W - pad - bar_px; y0 = H - pad - thick
                elif args.scalebar_pos == 'lower left':
                    x0 = pad; y0 = H - pad - thick
                elif args.scalebar_pos == 'upper left':
                    x0 = pad; y0 = pad
                else:  # upper right
                    x0 = W - pad - bar_px; y0 = pad
                rect = patches.Rectangle((x0, y0), bar_px, thick, linewidth=0, edgecolor=None, facecolor=args.scalebar_color)
                ax.add_patch(rect)
                # label
                label = f"{int(bar_um) if abs(bar_um - int(bar_um)) < 1e-6 else bar_um:g} µm"
                va = 'bottom' if 'lower' in args.scalebar_pos else 'top'
                ax.text(x0 + bar_px/2, y0 - 4 if 'lower' in args.scalebar_pos else y0 + thick + 4,
                        label, color=args.scalebar_color, ha='center', va=va, fontsize=9, weight='bold')

        # Optionally save per-condition merged RGB TIFF (8-bit)
        # Save merged RGB with preserved resolution
        rgb8 = (np.clip(rgb, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
        out_tif = merged_dir / f"{cond}_{chosen_id.get(cond, 'unknown')}_merged_rgb.tif"
        try:
            write_tiff_preserve_resolution(out_tif, rgb8, cond_res[cond])
            print(f"[save] Wrote merged RGB -> {out_tif}")
        except Exception as ex:
            print(f"[warn] Failed to write merged RGB TIFF for {cond}: {ex}")

        # Save raw, normalized grayscale, and pseudocolor per-channel images
        for ch in channels:
            raw_arr = grouped[cond][ch]
            raw_path = channels_raw_dir / f"{cond}_{chosen_id.get(cond, 'unknown')}_{ch}_raw.tif"
            try:
                write_tiff_preserve_resolution(raw_path, raw_arr, cond_res[cond])
            except Exception as ex:
                print(f"[warn] Failed to write raw channel {ch} for {cond}: {ex}")

            # Save grayscale normalized
            norm_arr = (np.clip(disp[cond][ch], 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
            norm_path = channels_norm_dir / f"{cond}_{chosen_id.get(cond, 'unknown')}_{ch}_norm.tif"
            try:
                write_tiff_preserve_resolution(norm_path, norm_arr, cond_res[cond])
            except Exception as ex:
                print(f"[warn] Failed to write grayscale normalized channel {ch} for {cond}: {ex}")

        # Verify shape and resolution preserved for merged vs display arrays
        try:
            with tifffile.TiffFile(str(out_tif)) as tf:
                sh = tf.pages[0].shape
            xres, yres, unit = read_tiff_resolution(str(out_tif))
            px_um_x2, px_um_y2 = try_pixel_size_um(str(out_tif))
            ok_shape = sh[:2] == rgb.shape[:2]
            ok_res = (cond_px_um[cond] == (None, None)) or (
                px_um_x2 is not None and px_um_y2 is not None and
                abs(px_um_x2 - (cond_px_um[cond][0] or px_um_x2)) < 1e-6 and
                abs(px_um_y2 - (cond_px_um[cond][1] or px_um_y2)) < 1e-6
            )
            if ok_shape and ok_res:
                print(f"[check] {cond}: shape preserved {sh[:2]}, pixel size preserved {px_um_x2}×{px_um_y2} µm")
            else:
                print(f"[ERROR] {cond}: shape/resolution changed! shape_saved={sh[:2]} vs rgb={rgb.shape[:2]}, px_saved={px_um_x2, px_um_y2} vs base={cond_px_um[cond]}")
        except Exception as ex:
            print(f"[warn] Could not verify saved merged file shape/resolution for {cond}: {ex}")

            # Save RGB pseudocolor normalized
            pseudo_path = channels_pseudo_dir / f"{cond}_{chosen_id.get(cond, 'unknown')}_{ch}_pseudo.tif"
            rgb_tints = {
                "C1": (0.2, 0.4, 1.0),  # soft blue for DAPI
                "C2": (0.3, 1.0, 0.3),  # soft green for EYFP
                "C3": (1.0, 0.3, 0.3),  # soft red for tdTom
            }
            norm_rgb = np.zeros((*disp[cond][ch].shape, 3), dtype=np.uint8)
            tint = rgb_tints.get(ch, (1.0, 1.0, 1.0))
            for i in range(3):
                norm_rgb[..., i] = (np.clip(disp[cond][ch], 0.0, 1.0) * 255.0 * tint[i] + 0.5).astype(np.uint8)
            try:
                tifffile.imwrite(pseudo_path, norm_rgb)
            except Exception as ex:
                print(f"[warn] Failed to write pseudocolor normalized channel {ch} for {cond}: {ex}")

    fig.tight_layout()
    out_svg = figures_dir / f"{args.out.name}.svg"
    out_pdf = figures_dir / f"{args.out.name}.pdf"
    fig.savefig(out_svg)
    fig.savefig(out_pdf)
    print(f"Wrote -> {out_svg} and {out_pdf}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
