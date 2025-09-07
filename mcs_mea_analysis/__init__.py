"""
MCS MEA analysis utilities (object-based).

Code in this package reads MEA .h5 files from the external drive paths
and writes outputs back to the external drive. Nothing in this repo is deleted.

Dependencies:
- Optional: McsPyDataTools (preferred for MCS .h5)
- Optional: h5py (fallback for basic HDF5 probing)
"""

__all__ = [
    "config",
    "discovery",
    "mcs_reader",
    "pipeline",
]

