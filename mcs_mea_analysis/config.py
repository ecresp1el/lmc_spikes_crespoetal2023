from __future__ import annotations

"""
Central configuration for MCS MEA file discovery and outputs.

All reading happens from the external Manny2TB drive using hardcoded paths,
and all outputs are written back to Manny2TB. Analysis code remains in this repo.
"""

from pathlib import Path
from dataclasses import dataclass


@dataclass(frozen=True)
class MCSConfig:
    # Hardcoded base directories on the external drive (Manny2TB)
    base_dirs: tuple[Path, ...] = (
        Path("/Volumes/Manny2TB/mea_blade_round3_led_ctz"),
        Path("/Volumes/Manny2TB/mea_blade_round4_led_ctz"),
        Path("/Volumes/Manny2TB/mea_blade_round5_led_ctz"),
    )

    # Subdir containing the .h5 files as organized previously
    h5_subdir: str = "h5_files"

    # Where to write outputs/logs on the external drive
    output_root: Path = Path("/Volumes/Manny2TB/mcs_mea_outputs")


CONFIG = MCSConfig()

