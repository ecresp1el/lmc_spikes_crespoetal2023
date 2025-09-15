#!/usr/bin/env python3
from __future__ import annotations

"""
Ephys Merged Grid (R 1 (..)_Merged.tif)
=======================================

What this does
--------------
- Searches a base directory (recursively) for specific roots like "R 1 (2)" ending with
  "*_Merged.tif". For each root, loads the image and extracts three channels in the order:
  DAPI (blue), EYFP (green), tdTom (red).
- Normalizes channels for display with configurable modes and gamma.
- Plots a grid: rows = roots, cols = DAPI | EYFP | tdTom | RGB overlay.
- Saves raw-extracted channels, normalized channels, merged RGB TIFFs, and grid SVG/PDF.

Assumptions
-----------
- "*_Merged.tif" may be an RGB(A) image. In this case, channels are extracted as
  DAPI=blue, EYFP=green, tdTom=red from the color overlay.
- If the TIFF is multi-channel or multi-page (e.g., shape (3,H,W)), the first channel is DAPI,
  second EYFP, third tdTom.

Defaults tuned for visualization (per-image normalization).
"""

import argparse
import glob
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib import cm as mplcm
from matplotlib.colors import Normalize
from matplotlib.ticker import FuncFormatter
from matplotlib import patches
import tifffile


def _rational_to_float(val) -> float:
    try:
        if hasattr(val, 'numerator') and hasattr(val, 'denominator'):
            return float(val.numerator) / float(val.denominator)
        if isinstance(val, (tuple, list)) and len(val) == 2:
            return float(val[0]) / float(val[1])
        return float(val)
    except Exception:
        return float('nan')


def try_pixel_size_um(path: str) -> Tuple[Optional[float], Optional[float]]:
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

            if unit_code == 2:
                unit_um = 25400.0
            elif unit_code == 3:
                unit_um = 10000.0
            else:
                return (None, None)

            if not np.isfinite(xres) or not np.isfinite(yres) or xres <= 0 or yres <= 0:
                return (None, None)
            return (unit_um / xres, unit_um / yres)
    except Exception:
        return (None, None)


def compute_global_percentiles(imgs: List[np.ndarray], low: float, high: float) -> Tuple[float, float]:
    flat = np.concatenate([np.ravel(im) for im in imgs]).astype(float)
    return float(np.percentile(flat, low)), float(np.percentile(flat, high))


def extract_channels_from_image(arr: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (DAPI, EYFP, tdTom) as grayscale arrays from a merged image array.

    Strategy:
    - If arr is HxWx3 or HxWx4, interpret as RGB(A) and take B,G,R respectively.
    - If arr is 3xHxW or 4xHxW, take first three planes as DAPI, EYFP, tdTom.
    - If arr is HxW (single plane), replicate as all channels.
    - Otherwise, try to reduce to first page or first 3 channels.
    """
    a = np.asarray(arr)
    if a.ndim == 3 and a.shape[-1] in (3, 4):
        # color last
        B = a[..., 2]
        G = a[..., 1]
        R = a[..., 0]
        return B, G, R
    if a.ndim == 3 and a.shape[0] in (3, 4):
        # channels first
        dapi = a[0]
        eyfp = a[1]
        tdtom = a[2]
        return dapi, eyfp, tdtom
    if a.ndim == 2:
        return a, a, a
    # Fallback: if more than 3 dims, try to squeeze
    a = np.squeeze(a)
    if a.ndim == 3:
        # retry
        return extract_channels_from_image(a)
    # Last resort: replicate zeros
    z = np.zeros(a.shape[-2:]) if a.ndim >= 2 else np.zeros((1, 1))
    return z, z, z


def _argparse() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot ephys merged images grid: DAPI, EYFP, tdTom + RGB")
    p.add_argument('--base-dir', type=Path, default=Path('/Users/ecrespo/Desktop/lmc_invivo_images_el222/data-Manny-1/LMC4f coded raw images/in_ephys_images'), help='Base directory to scan recursively')
    p.add_argument('--roots', nargs='*', default=[
        'R 1 (2)', 'R 1 (3)', 'R 1 (8)', 'R 1 (9)', 'R 1 (19)'
    ], help='Image roots to match, e.g., "R 1 (2)"')
    p.add_argument('--suffix', type=str, default='_Merged.tif', help='Filename suffix to match after root (default: _Merged.tif)')
    p.add_argument('--low', type=float, default=1.0, help='Lower percentile for display (default 1)')
    p.add_argument('--high', type=float, default=99.0, help='Upper percentile for display (default 99)')
    p.add_argument('--norm-mode', choices=['global','per_image','match_ref'], default='per_image', help='Normalization across images: global, per image, or match to a reference root by quantile scaling')
    p.add_argument('--norm-quantile', type=float, default=99.0, help='Quantile (0-100) used for matching when norm-mode=match_ref (default 99)')
    p.add_argument('--ref-root', type=str, default='R 1 (2)', help='Reference root name for norm-mode=match_ref')
    p.add_argument('--gamma', type=float, default=1.0, help='Gamma adjustment applied after normalization (>=0). 1.0 means no change')
    p.add_argument('--colorbar-mode', choices=['normalized','raw','none'], default='normalized', help='Show colorbars in normalized a.u., raw intensity labels, or hide them')
    p.add_argument('--scalebar-um', type=float, default=100.0, help='Scalebar length in µm on RGB panel. 0 auto; negative disables')
    p.add_argument('--scalebar-pos', choices=['lower left','lower right','upper left','upper right'], default='lower right', help='Scalebar anchor position on RGB panel')
    p.add_argument('--scalebar-color', type=str, default='white', help='Scalebar and label color')
    p.add_argument('--out', type=Path, default=Path('LMC4f_ephys_merged_grid'), help='Output base name')
    p.add_argument('--out-root', type=Path, default=None, help='Root output directory for all exports. Default: <base-dir>/<out>')
    return p.parse_args()


def main() -> int:
    args = _argparse()

    base = args.base_dir
    if not base.exists():
        raise SystemExit(f"Base directory not found: {base}")

    # Resolve output root
    out_root = args.out_root or (base / args.out.name)
    figures_dir = out_root / 'figures'
    channels_raw_dir = out_root / 'channels' / 'raw'
    channels_norm_dir = out_root / 'channels' / 'norm'
    channels_pseudo_dir = out_root / 'channels' / 'pseudo'
    merged_dir = out_root / 'merged'
    for d in (figures_dir, channels_raw_dir, channels_norm_dir, channels_pseudo_dir, merged_dir):
        d.mkdir(parents=True, exist_ok=True)

    # Locate files per root
    files: Dict[str, Path] = {}
    for root in args.roots:
        pattern = os.path.join(str(base), '**', f"*{root}*{args.suffix}")
        matches = glob.glob(pattern, recursive=True)
        if not matches:
            print(f"[warn] No file for root='{root}' with suffix='{args.suffix}' in {base}")
            continue
        # pick first match deterministically by shortest path then lexicographic
        matches.sort(key=lambda p: (len(p), p))
        files[root] = Path(matches[0])
        print(f"[pick] {root} -> {files[root]}")

    if not files:
        raise SystemExit('No matching files found. Check --base-dir, --roots, and --suffix.')

    # Load and extract channels
    raw: Dict[str, Dict[str, np.ndarray]] = {}
    px_um_map: Dict[str, Tuple[Optional[float], Optional[float]]] = {}
    for root, fpath in files.items():
        arr = tifffile.imread(str(fpath))
        dapi, eyfp, tdtom = extract_channels_from_image(arr)
        raw[root] = {'DAPI': np.asarray(dapi), 'EYFP': np.asarray(eyfp), 'tdTom': np.asarray(tdtom)}
        px_um_map[root] = try_pixel_size_um(str(fpath))
        shp = dapi.shape
        print(f"  {root}: shape={shp}  px_size_um={px_um_map[root]}")

    # Build display arrays 0..1 per chosen normalization
    roots = list(raw.keys())
    channels = ['DAPI','EYFP','tdTom']
    disp: Dict[str, Dict[str, np.ndarray]] = {r: {} for r in roots}
    raw_limits_perroot: Dict[str, Dict[str, Tuple[float,float]]] = {r: {} for r in roots}

    def apply_gamma(x: np.ndarray) -> np.ndarray:
        g = float(args.gamma)
        if g <= 0:
            return x
        return np.power(x, 1.0 / g)

    if args.norm_mode == 'global':
        for ch in channels:
            imgs = [raw[r][ch] for r in roots]
            vmin, vmax = compute_global_percentiles(imgs, low=float(args.low), high=float(args.high))
            for r in roots:
                arr = raw[r][ch].astype(float)
                nrm = np.clip((arr - vmin) / (vmax - vmin + 1e-12), 0.0, 1.0)
                disp[r][ch] = apply_gamma(nrm)
                raw_limits_perroot[r][ch] = (vmin, vmax)
    elif args.norm_mode == 'per_image':
        for ch in channels:
            for r in roots:
                arr = raw[r][ch].astype(float)
                vmin, vmax = compute_global_percentiles([arr], low=float(args.low), high=float(args.high))
                nrm = np.clip((arr - vmin) / (vmax - vmin + 1e-12), 0.0, 1.0)
                disp[r][ch] = apply_gamma(nrm)
                raw_limits_perroot[r][ch] = (vmin, vmax)
    else:  # match_ref
        ref = args.ref_root or roots[0]
        if ref not in raw:
            print(f"[warn] Reference root '{ref}' not found. Falling back to '{roots[0]}'")
            ref = roots[0]
        q = float(args.norm_quantile)
        scaled: Dict[str, Dict[str, np.ndarray]] = {r: {} for r in roots}
        for ch in channels:
            ref_arr = raw[ref][ch].astype(float)
            ref_q = float(np.percentile(ref_arr, q))
            for r in roots:
                arr = raw[r][ch].astype(float)
                cond_q = float(np.percentile(arr, q))
                scale = ref_q / (cond_q + 1e-12)
                scaled[r][ch] = arr * scale
        for ch in channels:
            imgs = [scaled[r][ch] for r in roots]
            vmin, vmax = compute_global_percentiles(imgs, low=float(args.low), high=float(args.high))
            for r in roots:
                arr = scaled[r][ch]
                nrm = np.clip((arr - vmin) / (vmax - vmin + 1e-12), 0.0, 1.0)
                disp[r][ch] = apply_gamma(nrm)
                raw_limits_perroot[r][ch] = (vmin, vmax)

    # Plot grid
    nrows = len(roots)
    ncols = 4
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(4*ncols, 3.6*nrows))
    if nrows == 1:
        axes = np.array([axes])

    cmaps = {'DAPI': 'Blues', 'EYFP': 'Greens', 'tdTom': 'Reds'}
    for i, r in enumerate(roots):
        for j, ch in enumerate(channels):
            ax = axes[i, j]
            im = ax.imshow(disp[r][ch], cmap=cmaps[ch], vmin=0.0, vmax=1.0)
            ax.set_title(f"{r} — {ch}", fontsize=10)
            ax.axis('off')
            if args.colorbar_mode != 'none':
                cbar = fig.colorbar(im, ax=ax, shrink=0.65)
                cbar.ax.tick_params(labelsize=8)
                if args.colorbar_mode == 'normalized':
                    cbar.set_label('Normalized (a.u.)', fontsize=8)
                else:
                    vmin_raw, vmax_raw = raw_limits_perroot[r][ch]
                    span = (vmax_raw - vmin_raw)
                    cbar.formatter = FuncFormatter(lambda x, pos: f"{(vmin_raw + x*span):.0f}")
                    cbar.set_label('Raw intensity', fontsize=8)
                    cbar.update_ticks()

        # RGB overlay
        ax = axes[i, ncols-1]
        rch = disp[r]['tdTom']; gch = disp[r]['EYFP']; bch = disp[r]['DAPI']
        rgb = np.stack([rch, gch, bch], axis=-1)
        ax.imshow(rgb)
        ax.set_title(f"{r} — RGB", fontsize=10)
        ax.axis('off')

        # Scalebar
        if args.scalebar_um >= 0:
            px_um_x, px_um_y = px_um_map.get(r, (None, None))
            if px_um_x is not None and px_um_y is not None:
                H, W = rgb.shape[:2]
                if args.scalebar_um == 0:
                    width_um = W * px_um_x
                    candidates = [10, 20, 50, 100, 200, 500, 1000]
                    target = width_um * 0.2
                    bar_um = max([c for c in candidates if c <= target] or [candidates[0]])
                else:
                    bar_um = float(args.scalebar_um)
                bar_px = max(int(round(bar_um / px_um_x)), 1)
                pad = max(int(round(min(H, W) * 0.02)), 6)
                thick = max(int(round(min(H, W) * 0.006)), 2)
                if args.scalebar_pos == 'lower right':
                    x0 = W - pad - bar_px; y0 = H - pad - thick
                elif args.scalebar_pos == 'lower left':
                    x0 = pad; y0 = H - pad - thick
                elif args.scalebar_pos == 'upper left':
                    x0 = pad; y0 = pad
                else:
                    x0 = W - pad - bar_px; y0 = pad
                rect = patches.Rectangle((x0, y0), bar_px, thick, linewidth=0, edgecolor=None, facecolor=args.scalebar_color)
                ax.add_patch(rect)
                label = f"{int(bar_um) if abs(bar_um - int(bar_um)) < 1e-6 else bar_um:g} µm"
                va = 'bottom' if 'lower' in args.scalebar_pos else 'top'
                ax.text(x0 + bar_px/2, y0 - 4 if 'lower' in args.scalebar_pos else y0 + thick + 4,
                        label, color=args.scalebar_color, ha='center', va=va, fontsize=9, weight='bold')

        # Save images
        # Raw extracted (as read from merged): keep original dtype
        for ch in channels:
            raw_arr = raw[r][ch]
            raw_path = channels_raw_dir / f"{r.replace('/', '_')}_{ch}_raw.tif"
            try:
                tifffile.imwrite(raw_path, raw_arr)
            except Exception as ex:
                print(f"[warn] Failed to write raw channel {ch} for {r}: {ex}")
        # Normalized 8-bit
        for ch in channels:
            norm_arr = (np.clip(disp[r][ch], 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
            norm_path = channels_norm_dir / f"{r.replace('/', '_')}_{ch}_norm.tif"
            try:
                tifffile.imwrite(norm_path, norm_arr)
            except Exception as ex:
                print(f"[warn] Failed to write normalized channel {ch} for {r}: {ex}")
        # Pseudocolor RGB 8-bit per channel (Blues/Greens/Reds)
        for ch in channels:
            cmap = mplcm.get_cmap(cmaps[ch])
            rgba = cmap(np.clip(disp[r][ch], 0.0, 1.0))  # HxWx4 floats 0..1
            rgb8 = (rgba[..., :3] * 255.0 + 0.5).astype(np.uint8)
            pseudo_path = channels_pseudo_dir / f"{r.replace('/', '_')}_{ch}_pseudo.tif"
            try:
                tifffile.imwrite(pseudo_path, rgb8)
            except Exception as ex:
                print(f"[warn] Failed to write pseudocolor channel {ch} for {r}: {ex}")

        # Merged
        rgb8 = (np.clip(rgb, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
        out_tif = merged_dir / f"{r.replace('/', '_')}_merged_rgb.tif"
        try:
            tifffile.imwrite(out_tif, rgb8)
            print(f"[save] Wrote merged RGB -> {out_tif}")
        except Exception as ex:
            print(f"[warn] Failed to write merged RGB TIFF for {r}: {ex}")

    fig.tight_layout()
    out_svg = figures_dir / f"{args.out.name}.svg"
    out_pdf = figures_dir / f"{args.out.name}.pdf"
    fig.savefig(out_svg)
    fig.savefig(out_pdf)
    print(f"Wrote -> {out_svg} and {out_pdf}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
