#!/usr/bin/env python3
from __future__ import annotations

"""
Three Pairs — Percent Change PSTH (5×2 layout)
==============================================

Thin wrapper around scripts.plot_psth_three_pairs_grid that enables
percent-change renormalization by default. It preserves the same 5×2 layout
and annotations (early/late shading, chem line, global scale bars), but plots
each trace as percent change relative to its early-window baseline per side:

  y_pct = 100 * (y - baseline) / baseline

The early window remains shaded and aligns to ~0% so changes outside that
window are more visually obvious (enhancements > 0%, decreases < 0%).

Notes
-----
- If you pass --no-renorm, percent-change is disabled (raw normalized units).
- When not specifying --out, the output base appends "__pct" to avoid
  overwriting the non-percent figure.
- The vertical scale label defaults to "%" when percent mode is on.

Usage
-----
python -m scripts.plot_psth_three_pairs_grid_percent \
  --plates 2 4 5 \
  --x-min -0.2 --x-max 1.0 \
  --out /tmp/psth_three_pairs_pct
"""

import sys
from typing import List

from scripts import plot_psth_three_pairs_grid as grid


def main(argv: List[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    # Enable percent-renorm by default unless user explicitly opts out or already set it
    if '--percent-renorm' not in args and '--no-renorm' not in args:
        args = ['--percent-renorm', *args]
    return grid.main(args)


if __name__ == '__main__':
    raise SystemExit(main())

