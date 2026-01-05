#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List

import numpy as np

try:
    import tifffile
except ImportError:  # pragma: no cover - optional TIFF writing
    tifffile = None


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


def ensure_reuse_insets(args: List[str]) -> List[str]:
    cleaned = [arg for arg in args if arg != "--"]
    if "--pick-inset" in cleaned or "--reuse-insets" in cleaned or "--inset" in cleaned:
        return cleaned
    return ["--reuse-insets"] + cleaned


def has_flag(args: Iterable[str], flag: str) -> bool:
    return flag in args


def run_single(
    script_path: Path,
    roi_dir: Path,
    extra_args: List[str],
    nd2_dir: Path | None,
    report_path: Path | None,
) -> None:
    cmd = [sys.executable, str(script_path), "--roi-dir", str(roi_dir)]
    if nd2_dir and not has_flag(extra_args, "--nd2-dir"):
        cmd.extend(["--nd2-dir", str(nd2_dir)])
    if report_path and not has_flag(extra_args, "--report-path"):
        cmd.extend(["--report-path", str(report_path)])
    cmd.extend(extra_args)
    subprocess.run(cmd, check=True)


def pad_to_width(image: np.ndarray, target_w: int) -> np.ndarray:
    height, width = image.shape[:2]
    if width == target_w:
        return image
    pad_right = max(0, target_w - width)
    pad_width = ((0, 0), (0, pad_right), (0, 0))
    return np.pad(image, pad_width, mode="constant")


def combine_rows(images: List[np.ndarray], padding: int) -> np.ndarray:
    if not images:
        raise ValueError("No images to combine.")
    max_w = max(img.shape[1] for img in images)
    padded = [pad_to_width(img, max_w) for img in images]
    if padding <= 0:
        return np.concatenate(padded, axis=0)
    gap = np.zeros((padding, max_w, 3), dtype=padded[0].dtype)
    rows: List[np.ndarray] = []
    for img in padded:
        if rows:
            rows.append(gap)
        rows.append(img)
    return np.concatenate(rows, axis=0)


def open_path(path: Path) -> None:
    if sys.platform == "darwin":
        subprocess.run(["open", str(path)], check=False)
    elif sys.platform.startswith("linux"):
        subprocess.run(["xdg-open", str(path)], check=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch export ROI inset grids for multiple groups and combine into one TIFF.",
    )
    parser.add_argument("--roi-root", type=Path, default=None, help="ROI root folder.")
    parser.add_argument(
        "--groups",
        nargs="+",
        default=["ctznmda", "ctz", "nmda"],
        help="Group names to process (default: ctznmda ctz nmda).",
    )
    parser.add_argument(
        "--roi-dirs",
        nargs="+",
        default=None,
        help="Explicit ROI directories (overrides roi-root/groups).",
    )
    parser.add_argument("--roi-name", type=str, default=None, help="Specific ROI folder name.")
    parser.add_argument("--nd2-dir", type=Path, default=None, help="ND2 root folder.")
    parser.add_argument("--report-path", type=Path, default=None, help="export_report.json to reuse.")
    parser.add_argument(
        "--combined-path",
        type=Path,
        default=None,
        help="Output path for combined TIFF.",
    )
    parser.add_argument(
        "--padding",
        type=int,
        default=20,
        help="Padding between combined rows (pixels).",
    )
    parser.add_argument(
        "--open-combined",
        dest="open_combined",
        action="store_true",
        help="Open the combined TIFF after export.",
    )
    parser.add_argument(
        "--no-open-combined",
        dest="open_combined",
        action="store_false",
        help="Do not open the combined TIFF.",
    )
    parser.set_defaults(open_combined=True)
    return parser.parse_known_args()


def main() -> None:
    args, extra_args = parse_args()
    extra_args = ensure_reuse_insets(list(extra_args))

    script_path = Path(__file__).with_name("export_single_roi_grid.py")
    if not script_path.exists():
        raise FileNotFoundError(f"export_single_roi_grid.py not found at {script_path}")

    if args.roi_dirs:
        roi_dirs = [Path(item).expanduser() for item in args.roi_dirs]
    else:
        if args.roi_root is None:
            raise SystemExit("Provide --roi-dirs or --roi-root.")
        roi_root = args.roi_root.expanduser()
        roi_dirs = []
        for group in args.groups:
            group_dir = roi_root / f"group_{group}"
            roi_dirs.append(find_group_roi_dir(group_dir, args.roi_name))

    grid_paths: List[Path] = []
    for roi_dir in roi_dirs:
        run_single(script_path, roi_dir, extra_args, args.nd2_dir, args.report_path)
        grid_paths.append(roi_dir / "nd2_export" / "grid_all_channels_pseudo.tif")

    if tifffile is None:
        raise RuntimeError("tifffile is required to combine TIFFs. Install with: pip install tifffile")

    grids = [tifffile.imread(str(path)) for path in grid_paths]
    combined = combine_rows(grids, int(args.padding))

    if args.combined_path:
        combined_path = args.combined_path.expanduser()
    else:
        base_root = args.roi_root.expanduser() if args.roi_root else roi_dirs[0].parent.parent
        combined_path = base_root / "nd2_export" / "combined_inset_grid.tif"
    combined_path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(str(combined_path), combined, photometric="rgb")
    print(f"Saved combined TIFF to {combined_path}")
    if args.open_combined:
        open_path(combined_path)


if __name__ == "__main__":
    main()
