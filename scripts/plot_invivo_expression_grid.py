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
    p.add_argument('--condition', action='append', default=[], help='NAME=DIR per condition (repeat)')
    p.add_argument('--ids', nargs='*', default=[], help='Candidate IDs to search for (first match wins)')
    p.add_argument('--id-map', action='append', default=[], help='NAME:ID — force ID per condition (repeat)')
    p.add_argument('--low', type=float, default=1.0, help='Lower percentile for display (default 1)')
    p.add_argument('--high', type=float, default=99.0, help='Upper percentile for display (default 99)')
    p.add_argument('--out', type=Path, default=Path('LMC4f_invivo_expression_grid'), help='Output base path (no suffix)')
    p.add_argument('--chan-tag', action='append', default=['C1:DAPI','C2:EYFP','C3:tdTom'], help='C:TAG labels (repeat)')
    p.add_argument('--chan-cmap', action='append', default=['C1:Blues','C2:Greens','C3:Reds'], help='C:CMAP colormap names (repeat)')
    p.add_argument('--name-contains', action='append', default=[], help='Only use files whose names include these substrings (repeatable; case-insensitive)')
    a = p.parse_args()
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

    # Step 1 — load and group images by condition
    grouped: Dict[str, Dict[str, np.ndarray]] = {}
    chosen_files: Dict[str, Dict[str, str]] = {}
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
        if len({s for s in shapes}) != 1:
            print(f'[warn] Images for {name} have different shapes {shapes}. RGB overlay may appear misaligned.')

    # Step 2 — compute per‑channel global percentiles across conditions
    global_limits: Dict[str, Tuple[float,float]] = {}
    for ch in channels:
        imgs = [grouped[cond][ch] for cond in grouped]
        global_limits[ch] = compute_global_percentiles(imgs, low=float(args.low), high=float(args.high))

    # Step 3 — plot grid
    nrows = len(grouped)
    ncols = len(channels) + 1  # channels + RGB
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(4*ncols, 3.6*nrows))
    if nrows == 1:
        axes = np.array([axes])

    cond_names = list(grouped.keys())
    for i, cond in enumerate(cond_names):
        for j, ch in enumerate(channels):
            ax = axes[i, j]
            img = grouped[cond][ch]
            vmin, vmax = global_limits[ch]
            im = ax.imshow(img, cmap=chan_cmaps.get(ch, 'gray'), vmin=vmin, vmax=vmax)
            ax.set_title(f"{cond} — {chan_tags.get(ch, ch)}", fontsize=10)
            ax.axis('off')
            cbar = fig.colorbar(ScalarMappable(norm=Normalize(vmin, vmax), cmap=chan_cmaps.get(ch, 'gray')), ax=ax, shrink=0.65)
            cbar.ax.tick_params(labelsize=8)
        # RGB overlay in last column
        ax = axes[i, ncols-1]
        def _norm(chkey: str) -> np.ndarray:
            vmin, vmax = global_limits[chkey]
            arr = grouped[cond][chkey].astype(float)
            arr = np.clip((arr - vmin) / (vmax - vmin + 1e-12), 0.0, 1.0)
            return arr
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

    fig.tight_layout()
    out_svg = args.out.with_suffix('.svg')
    out_pdf = args.out.with_suffix('.pdf')
    fig.savefig(out_svg)
    fig.savefig(out_pdf)
    print(f"Wrote -> {out_svg} and {out_pdf}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
