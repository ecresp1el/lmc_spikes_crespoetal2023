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
    n_samples: Optional[int]
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


def _as_float(val: Any, preferred_unit: str | None = None) -> Optional[float]:
    """Try to coerce a numeric or Pint-like quantity to float.
    If preferred_unit is given, attempt unit conversion via `.to(preferred_unit)`.
    Supports attributes like `.magnitude`, `.m`.
    """
    try:
        if val is None:
            return None
        # Attempt Pint quantity handling
        if preferred_unit and hasattr(val, "to"):
            try:
                v2 = val.to(preferred_unit)
                if hasattr(v2, "magnitude"):
                    return float(v2.magnitude)
                if hasattr(v2, "m"):
                    return float(v2.m)
            except Exception:
                pass
        # Direct magnitude
        if hasattr(val, "magnitude"):
            return float(val.magnitude)
        if hasattr(val, "m"):
            return float(val.m)
        # Plain number
        return float(val)
    except Exception:
        return None


class MetadataExtractor:
    @staticmethod
    def extract_basic(path: Path) -> BasicInfo:
        """
        Extract sampling rate, channel count, and duration strictly via
        McsPyDataTools/McsPy. No h5py fallback is used.
        """
        # McsPyDataTools / McsPy
        try:
            # Try McsPyDataTools under common names
            try:
                from McsPyDataTools import McsRecording  # type: ignore
            except Exception:
                from mcspydatatools import McsRecording  # type: ignore

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
                    sr = sr or getattr(st, "sample_rate", None) or getattr(st, "sampling_rate", None) or getattr(st, "sampling_frequency", None)
                    # Try nested channel info
                    nchan = nchan or getattr(st, "channel_count", None) or getattr(st, "num_channels", None)
                    if nchan is None:
                        for cname in ("channel_infos", "channels", "channel_ids", "channel_labels"):
                            ci = getattr(st, cname, None)
                            try:
                                nchan = len(ci) if ci is not None else nchan
                            except Exception:
                                pass
                    # Try duration or convert from sample count
                    dur = dur or getattr(st, "duration", None) or getattr(st, "time_extent", None)
                    samples = getattr(st, "sample_count", None)
                    if samples is None:
                        samples = getattr(st, "num_samples", None)
                    if dur is None and samples is not None and sr:
                        try:
                            dur = float(samples) / float(_as_float(sr, "Hz") or float(sr))
                        except Exception:
                            pass
                    if sr or nchan or dur:
                        break
                if sr or nchan or dur:
                    break

            # Derive samples if possible from recording/streams (API-specific; skip here)
            return BasicInfo(
                sampling_rate_hz=_as_float(sr, "Hz"),
                n_channels=_to_int(nchan),
                n_samples=None,
                duration_seconds=_as_float(dur, "s"),
            )
        except Exception:
            pass

        # Legacy McsPy.McsData API
        try:
            import McsPy.McsData as McsData  # type: ignore

            raw = McsData.RawData(path.as_posix())
            recs = getattr(raw, "recordings", {}) or {}
            # choose first recording
            first_rec = next(iter(recs.values())) if recs else None
            sr = None
            nchan = None
            dur = None
            if first_rec is not None:
                # duration as Pint quantity
                dur = getattr(first_rec, "duration_time", None)
                # pick first analog stream
                analogs = getattr(first_rec, "analog_streams", {}) or {}
                if analogs:
                    st = next(iter(analogs.values()))
                    # channel count
                    ci = getattr(st, "channel_infos", None)
                    try:
                        nchan = len(ci) if ci is not None else None
                    except Exception:
                        nchan = None
                    # sampling frequency
                    # use any channel's sampling_frequency
                    if ci:
                        try:
                            any_chan = next(iter(ci.values()))
                            sr = getattr(any_chan, "sampling_frequency", None)
                        except Exception:
                            sr = None
                    # sample count from ChannelData shape
                    try:
                        nsmpl = int(getattr(st, "channel_data").shape[1])  # channels x samples
                    except Exception:
                        nsmpl = None

            sr_f = _as_float(sr, "Hz")
            dur_f = _as_float(dur, "s")
            if dur_f is None and (nsmpl is not None) and (sr_f is not None and sr_f > 0):
                try:
                    dur_f = float(nsmpl) / float(sr_f)
                except Exception:
                    pass

            return BasicInfo(
                sampling_rate_hz=sr_f,
                n_channels=_to_int(nchan),
                n_samples=nsmpl,
                duration_seconds=dur_f,
            )
        except Exception:
            pass

        # No info available without MCS packages
        return BasicInfo(sampling_rate_hz=None, n_channels=None, n_samples=None, duration_seconds=None)


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
