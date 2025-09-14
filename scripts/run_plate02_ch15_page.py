#!/usr/bin/env python3
from __future__ import annotations

"""
Convenience launcher for a fixed pair (plate 02) and three channels of interest.

Runs the CTZ|VEH multi‑channel page (filtered overlay + aligned spike raster)
using scripts.plot_pair_channel_page with preset arguments.

Edit CHS below to change which channels are shown.

Usage:
  python -m scripts.run_plate02_ch15_page
"""

import subprocess
import sys


def main() -> None:
    # Prespecified channels of interest (0-based)
    CHS = [15, 22, 33]
    cmd = [
        sys.executable, "-m", "scripts.plot_pair_channel_page",
        "--ctz-h5", "/Volumes/Manny2TB/mea_blade_round5_led_ctz/h5_files/plate_02_led_ctz_2023-12-05T09-10-45.h5",
        "--veh-h5", "/Volumes/Manny2TB/mea_blade_round5_led_ctz/h5_files/plate_02_led_veh_2023-12-05T16-16-23.h5",
        "--chem-ctz", "180.1597",
        "--chem-veh", "181.6293",
        "--plate", "2",
        "--round", "mea_blade_round5",
        "--chs",
        *[str(c) for c in CHS],
        "--pre", "0.5",
        "--post", "0.5",
        "--hp", "300",
        "--order", "4",
    ]
    print("Running:")
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()

