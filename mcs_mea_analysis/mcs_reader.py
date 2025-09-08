from __future__ import annotations

"""
Reader utilities for MCS MEA .h5 files.

Policy: Only McsPyDataTools is used to access/read data. If the package is not
available or cannot open the file, we do not fall back to other readers.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Tuple


@dataclass
class ProbeResult:
    path: Path
    exists: bool
    is_hdf5_signature: bool
    mcs_available: bool
    mcs_loaded: bool
    loader: Optional[str] = None
    error: Optional[str] = None
    # Optional light-weight metadata if available
    metadata: Optional[dict[str, Any]] = None


def probe_mcs_h5(path: Path) -> ProbeResult:
    """
    Attempt a lightweight open of an MCS .h5 file.

    Strategy (in order):
    1) McsPyDataTools (if installed) — preferred
    2) h5py (if installed) — basic HDF5 open/inspect
    3) Raw HDF5 signature check — ensures file is reachable
    """
    exists = path.exists() and path.is_file()
    if not exists:
        return ProbeResult(
            path=path,
            exists=False,
            is_hdf5_signature=False,
            mcs_available=_mcs_available(),
            mcs_loaded=False,
            loader=None,
            error="File not found",
            metadata=None,
        )

    # Default: no metadata yet
    metadata: dict[str, Any] | None = None

    # Try McsPyDataTools (only allowed access path)
    if _mcs_available():
        try:
            md, meta = _try_mcs_load(path)
            return ProbeResult(
                path=path,
                exists=True,
                is_hdf5_signature=True,  # opened via MCS implies valid HDF5
                mcs_available=True,
                mcs_loaded=True,
                loader=md,
                error=None,
                metadata=meta,
            )
        except Exception as e:  # noqa: BLE001
            # Fall through to next options
            mcs_err = str(e)
    else:
        mcs_err = None

    # No other access is allowed. Report existence and MCS status only.
    return ProbeResult(
        path=path,
        exists=True,
        is_hdf5_signature=False,
        mcs_available=_mcs_available(),
        mcs_loaded=False,
        loader=None,
        error=mcs_err,
        metadata=None,
    )


def _import_mcspy() -> Tuple[Optional[Any], Optional[str]]:
    """Attempt to import McsPyDataTools under common module names.
    Returns (module, import_name) or (None, None)."""
    for name in ("McsPyDataTools", "mcspydatatools", "McsPy"):
        try:
            mod = __import__(name)
            return mod, name
        except Exception:
            continue
    return None, None


def _mcs_available() -> bool:
    mod, _ = _import_mcspy()
    return mod is not None


def _try_mcs_load(path: Path) -> tuple[str, dict[str, Any]]:
    """
    Attempt to load via McsPyDataTools.

    McsPyDataTools API differs between versions; try a few known entry points.
    We only query very light metadata to keep this step fast and robust.
    """
    # Strategy 1: new-style `McsRawData`/`McsRecording` (if present)
    mod, import_name = _import_mcspy()
    if mod is None:
        raise RuntimeError("McsPyDataTools not importable")

    # Try attribute lookup on the imported module
    McsRecording = getattr(mod, "McsRecording", None)
    if McsRecording is not None:
        rec = McsRecording(path.as_posix())  # type: ignore[call-arg]
        meta = {
            "loader": f"{import_name}.McsRecording",
            "recording_name": getattr(rec, "name", None),
            "num_segments": getattr(rec, "channel_segments", None),
        }
        return "McsRecording", meta
    # Fallback within same package: look for a generic open
    open_file = getattr(mod, "open_file", None)
    if callable(open_file):
        h = open_file(path.as_posix())  # type: ignore[misc]
        meta = {"loader": f"{import_name}.open_file", "repr": repr(h)}
        return "open_file", meta

    # Strategy 2: legacy McsPy.McsData API
    try:
        import McsPy.McsData as McsData  # type: ignore

        raw = McsData.RawData(path.as_posix())
        meta = {
            "loader": "McsPy.McsData.RawData",
            "n_recordings": len(getattr(raw, "recordings", {}) or {}),
            "program_name": getattr(raw, "program_name", None),
            "mea_layout": getattr(raw, "mea_layout", None),
        }
        return "McsPy.McsData.RawData", meta
    except Exception:
        pass

    # If we reached here, an MCS package is installed but API didn't match.
    raise RuntimeError(
        "MCS Python package is installed, but no supported loader entry point matched."
    )
