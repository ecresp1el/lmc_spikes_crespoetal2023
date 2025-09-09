from __future__ import annotations

"""
CLI: Build the standardized readiness index (chem+NPZ by default).

Usage:
  python -m scripts.build_ready                  # default CONFIG + defaults
  python -m scripts.build_ready --require-opto   # require opto as well
  python -m scripts.build_ready --no-eligible    # ignore eligibility filter
  python -m scripts.build_ready <output_root> [flags]

Flags:
  --require-opto / --no-require-opto (default: no)
  --eligible / --no-eligible (default: no)
  --require-fr (default: no)
  --ignore-ignored (default: yes)
"""

import sys
from pathlib import Path
from mcs_mea_analysis.ready import ReadinessConfig, build_ready_index


def main() -> None:
    args = [a for a in sys.argv[1:] if a]
    if any(a in ("-h", "--help") for a in args):
        print(__doc__)
        sys.exit(0)
    out_root: Path | None = None
    require_opto = False
    require_eligible = False
    require_fr = False
    require_not_ignored = True

    # Parse
    i = 0
    while i < len(args):
        a = args[i]
        if a == "--require-opto":
            require_opto = True
        elif a == "--no-require-opto":
            require_opto = False
        elif a == "--eligible":
            require_eligible = True
        elif a == "--no-eligible":
            require_eligible = False
        elif a == "--require-fr":
            require_fr = True
        elif a == "--ignore-ignored":
            require_not_ignored = False
        elif a.startswith("-"):
            print(f"[ready-cli] unknown flag: {a}")
        else:
            out_root = Path(a)
        i += 1

    cfg = ReadinessConfig(
        output_root=out_root or ReadinessConfig().output_root,
        require_opto=require_opto,
        require_not_ignored=require_not_ignored,
        require_eligible=require_eligible,
        require_ifr_npz=True,
        require_fr_summary=require_fr,
    )
    build_ready_index(cfg)


if __name__ == "__main__":
    main()
