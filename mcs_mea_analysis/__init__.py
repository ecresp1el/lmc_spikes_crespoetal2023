"""
MCS MEA analysis utilities (object-based).

Code in this package reads MEA .h5 files from the external drive paths
and writes outputs back to the external drive. Nothing in this repo is deleted.

Access policy:
- Only McsPyDataTools is used to access/read MCS data.
"""

__all__ = [
    "config",
    "discovery",
    "mcs_reader",
    "pipeline",
]
