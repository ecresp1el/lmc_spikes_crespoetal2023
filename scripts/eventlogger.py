from __future__ import annotations

"""
Convenience launcher for the MCS MEA Event Logging GUI.

Usage:
    python scripts/eventlogger.py [<recording.h5>] [--index <file_index.json>]
"""

from mcs_mea_analysis.mcs_mea_eventlogging_gui import main


if __name__ == "__main__":
    main()

