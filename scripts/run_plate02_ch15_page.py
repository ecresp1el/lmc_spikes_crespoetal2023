#!/usr/bin/env python3
from __future__ import annotations

"""
Convenience launcher for a fixed pair (plate 02) and channel 15.

Runs the 2×3 CTZ|VEH page (raw, filtered, filtered+spikes) using
scripts.plot_pair_channel_page with preset arguments.

Usage:
  python -m scripts.run_plate02_ch15_page
"""

import subprocess
import sys


def main() -> None:
    cmd = [
        sys.executable, "-m", "scripts.plot_pair_channel_page",
        "--ctz-h5", "/Volumes/Manny2TB/mea_blade_round5_led_ctz/h5_files/plate_02_led_ctz_2023-12-05T09-10-45.h5",
        "--veh-h5", "/Volumes/Manny2TB/mea_blade_round5_led_ctz/h5_files/plate_02_led_veh_2023-12-05T16-16-23.h5",
        "--chem-ctz", "180.1597",
        "--chem-veh", "181.6293",
        "--plate", "2",
        "--round", "mea_blade_round5",
        "--ch", "15",
        "--pre", "0.5",
        "--post", "0.5",
        "--hp", "300",
        "--order", "4",
        # Let the plotting script choose the default output path next to CTZ H5.
        # To override, add: "--out", "/desired/path/plate02_ch15_2x3.png",
    ]
    print("Running:")
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()

