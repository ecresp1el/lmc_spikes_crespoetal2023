from __future__ import annotations

"""
CLI: Process all IFR NPZ files (GUI-independent).

Usage:
  python -m scripts.process_ifr_npz             # uses default CONFIG paths
  python -m scripts.process_ifr_npz <output_root>
"""

import sys
from pathlib import Path
from mcs_mea_analysis.ifr_processing import (
    IFRProcessorConfig,
    find_ifr_npz,
    process_all_ifr_npz,
)


def main() -> None:
    if any(a in ("-h", "--help") for a in sys.argv[1:]):
        print(__doc__)
        sys.exit(0)
    out_root = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    cfg = IFRProcessorConfig(output_root=out_root or IFRProcessorConfig().output_root)
    files = find_ifr_npz(cfg.output_root)
    print(f"[cli] found NPZ: {len(files)}")
    process_all_ifr_npz(files, cfg)


if __name__ == "__main__":
    main()
