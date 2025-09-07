from __future__ import annotations

"""
Metadata helpers: infer group labels from filenames/paths and extract basic
recording info (sampling rate, channel count, duration) when possible.

This module keeps object-based, composable pieces for later scaling.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Any, Iterator
import re


@dataclass(frozen=True)
class GroupInfo:
    label: str  # e.g., "CTZ", "VEH", "SPONT", "UNKNOWN"
    round_name: Optional[str]
    plate: Optional[int]
    timestamp: Optional[str]
    is_test: bool


@dataclass(frozen=True)
class BasicInfo:
    sampling_rate_hz: Optional[float]
    n_channels: Optional[int]
    duration_seconds: Optional[float]


class GroupLabeler:
    CTZ_PAT = re.compile(r"\bctz\b|led_ctz", re.IGNORECASE)
    VEH_PAT = re.compile(r"\bveh\b|led_veh", re.IGNORECASE)
    SPONT_PAT = re.compile(r"spont|spontaneous|no[_-]?led|no\s*led|nolight|baseline|dark", re.IGNORECASE)
    ROUND_PAT = re.compile(r"mea_blade_round\d+", re.IGNORECASE)
    PLATE_PAT = re.compile(r"plate_(\d+)", re.IGNORECASE)
    TS_PAT = re.compile(r"(\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2})")

    @staticmethod
    def infer_from_path(p: Path) -> GroupInfo:
        s = p.name
        ps = str(p)

        if GroupLabeler.CTZ_PAT.search(s):
            label = "CTZ"
        elif GroupLabeler.VEH_PAT.search(s):
            label = "VEH"
        elif GroupLabeler.SPONT_PAT.search(s):
            label = "SPONT"
        else:
            label = "UNKNOWN"

        round_name = None
        for part in p.parts:
            m = GroupLabeler.ROUND_PAT.search(part)
            if m:
                round_name = m.group(0)
                break

        plate = None
        mp = GroupLabeler.PLATE_PAT.search(s)
        if mp:
            try:
                plate = int(mp.group(1))
            except Exception:
                plate = None

        ts = None
        mt = GroupLabeler.TS_PAT.search(s)
        if mt:
            ts = mt.group(1)

        is_test = "test" in s.lower()

        return GroupInfo(
            label=label,
            round_name=round_name,
            plate=plate,
            timestamp=ts,
            is_test=is_test,
        )


def _walk_h5_datasets(h5obj: Any) -> Iterator[Any]:
    """Yield all h5py.Dataset objects under an HDF5 file/group."""
    try:
        import h5py  # type: ignore
    except Exception:  # pragma: no cover - optional dep
        return

    def _recur(obj: Any) -> Iterator[Any]:
        if isinstance(obj, h5py.Dataset):
            yield obj
        elif isinstance(obj, h5py.Group):
            for k in obj.keys():
                try:
                    yield from _recur(obj[k])
                except Exception:
                    continue
    yield from _recur(h5obj)


class MetadataExtractor:
    @staticmethod
    def extract_basic(path: Path) -> BasicInfo:
        """
        Extract sampling rate, channel count, and duration strictly via
        McsPyDataTools/McsPy. No h5py fallback is used.
        """
        # McsPyDataTools / McsPy
        try:
            # Try new-style McsPyDataTools
            from McsPyDataTools import McsRecording  # type: ignore

            rec = McsRecording(path.as_posix())
            # Heuristic introspection: look for an analog stream-like object with shape info
            # Because API can differ, we try a few common attributes.
            sr = None
            nchan = None
            dur = None

            # Probe attributes by introspection
            for attr_name in (
                "analog_streams",
                "AnalogStream",
                "analogStreams",
            ):
                streams = getattr(rec, attr_name, None)
                if streams is None:
                    continue
                # Streams could be dict-like
                try:
                    items = streams.items() if hasattr(streams, "items") else enumerate(streams)
                except Exception:
                    items = []
                for _, st in items:
                    # Common attribute guesses
                    sr = sr or getattr(st, "sample_rate", None) or getattr(st, "sampling_rate", None)
                    # Try nested channel info
                    nchan = nchan or getattr(st, "channel_count", None) or getattr(st, "num_channels", None)
                    # Try duration or convert from sample count
                    dur = dur or getattr(st, "duration", None)
                    samples = getattr(st, "sample_count", None)
                    if dur is None and samples is not None and sr:
                        try:
                            dur = float(samples) / float(sr)
                        except Exception:
                            pass
                    if sr or nchan or dur:
                        break
                if sr or nchan or dur:
                    break

            return BasicInfo(
                sampling_rate_hz=_to_float(sr),
                n_channels=_to_int(nchan),
                duration_seconds=_to_float(dur),
            )
        except Exception:
            pass

        # No info available without McsPyDataTools
        return BasicInfo(sampling_rate_hz=None, n_channels=None, duration_seconds=None)


def _to_float(x: Any) -> Optional[float]:
    try:
        return float(x)
    except Exception:
        return None


def _to_int(x: Any) -> Optional[int]:
    try:
        return int(x)
    except Exception:
        return None


def _find_attr_number(h5obj: Any, keys: list[str]) -> Optional[float]:
    """Search numeric attribute by a list of candidate names (case-insensitive)."""
    try:
        attrs = getattr(h5obj, "attrs", {})
        for k in attrs.keys():
            for want in keys:
                if str(k).lower() == want.lower():
                    try:
                        v = attrs[k]
                        # If 0-dim array/scalar
                        if hasattr(v, "shape") and getattr(v, "shape", ()) == ():
                            v = v[()]
                        return float(v)
                    except Exception:
                        continue
        return None
    except Exception:
        return None
