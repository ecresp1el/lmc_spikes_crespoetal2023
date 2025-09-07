#!/usr/bin/env python3
from __future__ import annotations

"""
CLI runner to discover MCS MEA `.h5` files on the external drive and probe them.
Outputs are written to `/Volumes/Manny2TB/mcs_mea_outputs/`.
"""

import sys
from pathlib import Path

from mcs_mea_analysis.pipeline import run_probe


def main(argv: list[str]) -> int:
    # Optional: allow passing a custom output root via CLI
    output_root: Path | None = None
    if len(argv) > 1:
        output_root = Path(argv[1])

    results = run_probe(output_root=output_root)
    ok = any(r.exists and (r.mcs_loaded or r.is_hdf5_signature) for r in results)
    print(f"Probed {len(results)} file(s). Any OK: {ok}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))

