from __future__ import annotations

"""
CLI: Build unified analysis manifest.

Usage:
  python -m scripts.build_manifest             # uses default CONFIG paths
  python -m scripts.build_manifest <output_root> [--no-require-opto]
"""

import sys
from pathlib import Path
from mcs_mea_analysis.manifest import build_manifest


def main() -> None:
    if any(a in ("-h", "--help") for a in sys.argv[1:]):
        print(__doc__)
        sys.exit(0)
    out_root: Path | None = None
    require_opto = True
    args = [a for a in sys.argv[1:] if a]
    if args:
        if args[0].startswith("--"):
            if args[0] == "--no-require-opto":
                require_opto = False
        else:
            out_root = Path(args[0])
            if len(args) > 1 and args[1] == "--no-require-opto":
                require_opto = False
    build_manifest(out_root, require_opto=require_opto)


if __name__ == "__main__":
    main()
