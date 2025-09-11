from __future__ import annotations

"""
Pairings Utilities (OOP)
========================

Goal
----
Provide a small, GUI-independent object model to:
- Group readiness rows by plate (optionally by round)
- Split CTZ vs VEH, keeping "others" separate
- Pair CTZ/VEH recordings in timestamp order
- Report NPZ availability and per-pair readiness

Usage
-----
from mcs_mea_analysis.pairings import PairingIndex

px = PairingIndex.from_ready_rows(rows, group_by_round=True)
summary = px.summary_rows()            # counts per plate(+round)
pairs_rows = px.pairs_dataframe()      # expanded pairs + unpaired entries

Design Notes
------------
- We treat "pairing" as a light-weight task using only metadata in the ready
  index. We do not read any NPZ/H5 here.
- Ordering inside a plate(+round) group is by ascending recording timestamp.
"""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
import itertools

from .metadata import GroupLabeler


@dataclass(frozen=True)
class PairItem:
    """Normalized per-recording info derived from readiness rows.

    Attributes
    ----------
    path : Path
        Recording file path from the index
    recording_stem : str
        Stem (basename without extension)
    round : str | None
        Round name if available
    plate : int | None
        Plate number if inferable
    group_label : str
        CTZ, VEH, SPONT, UNKNOWN, etc.
    npz_path : Path | None
        Path to *_ifr_per_channel_1ms.npz if present
    has_npz : bool
        Whether NPZ exists at `npz_path`
    ready : bool
        Whether the readiness policy marked this row as ready
    timestamp : str | None
        Timestamp parsed from stem/path (e.g., 2023-11-30T08-17-28)
    chem_timestamp : float | None
        Chem timestamp (seconds) if present in annotations
    """
    path: Path
    recording_stem: str
    round: Optional[str]
    plate: Optional[int]
    group_label: str  # expected: 'CTZ' or 'VEH' (others allowed but not paired)
    npz_path: Optional[Path]
    has_npz: bool
    ready: bool
    timestamp: Optional[str]  # as captured in index, e.g., 2024-07-01T12-30-00
    chem_timestamp: Optional[float]

    @property
    def timestamp_dt(self) -> Optional[datetime]:
        ts = self.timestamp
        if not ts:
            return None
        # Expected format: YYYY-MM-DDTHH-MM-SS
        try:
            return datetime.strptime(ts, "%Y-%m-%dT%H-%M-%S")
        except Exception:
            return None


@dataclass
class PlatePairs:
    """Container of CTZ/VEH items for a grouping key.

    Methods provide small aggregations (counts, pairing in timestamp order).
    """
    key: Tuple[Optional[int], Optional[str]]  # (plate, round) if round grouping used
    ctz: List[PairItem]
    veh: List[PairItem]
    others: List[PairItem]

    def summary(self) -> Dict[str, int]:
        return {
            "plate": self.key[0] if self.key else None,
            "round": self.key[1] if self.key else None,
            "n_ctz": len(self.ctz),
            "n_ctz_ready": sum(1 for x in self.ctz if x.ready and x.has_npz),
            "n_veh": len(self.veh),
            "n_veh_ready": sum(1 for x in self.veh if x.ready and x.has_npz),
        }

    def pair_by_timestamp(self) -> Tuple[List[Tuple[PairItem, PairItem]], List[PairItem], List[PairItem]]:
        """Return (pairs, unpaired_ctz, unpaired_veh) using ascending timestamp order.

        If timestamps are missing, falls back to recording_stem sort.
        Only CTZ/VEH are paired; 'others' are ignored for pairing.
        """
        def sort_key(x: PairItem):
            return (x.timestamp_dt or datetime.min, x.recording_stem)

        A = sorted(self.ctz, key=sort_key)
        B = sorted(self.veh, key=sort_key)
        n = min(len(A), len(B))
        pairs = list(zip(A[:n], B[:n]))
        return pairs, A[n:], B[n:]


class PairingIndex:
    """Build and query CTZ vs VEH pairings from readiness rows.

    Grouping
    --------
    - Key is (plate, round) by default; set `group_by_round=False` to collapse
      across rounds.

    Guarantees
    ----------
    - The method `pairs_dataframe()` returns both paired rows (with status
      `ready_pair` if both sides are ready and have NPZ) and unpaired rows for
      visibility (`ctz_unpaired`, `veh_unpaired`).
    """

    def __init__(self, groups: Dict[Tuple[Optional[int], Optional[str]], PlatePairs]):
        self.groups = groups

    @staticmethod
    def from_ready_rows(
        rows: Iterable[dict],
        group_by_round: bool = True,
    ) -> "PairingIndex":
        """Construct a pairing index from readiness rows.

        Parameters
        ----------
        rows : Iterable[dict]
            Dicts as produced by `build_ready_index` (CSV/JSONL rows)
        group_by_round : bool
            Whether to keep separate groups per round (default True)
        """
        groups: Dict[Tuple[Optional[int], Optional[str]], PlatePairs] = {}
        for r in rows:
            path = Path(str(r.get("path", "")))
            stem = str(r.get("recording_stem", path.stem))
            # Prefer explicit index fields; fallback to inference
            plate = r.get("plate")
            round_name = r.get("round")
            if plate is None or round_name is None:
                gi = GroupLabeler.infer_from_path(path)
                plate = gi.plate if plate is None else plate
                round_name = gi.round_name if round_name is None else round_name
            try:
                plate = int(plate) if plate is not None else None
            except Exception:
                plate = None

            group_label = (r.get("group_label") or "").upper() or GroupLabeler.infer_from_path(path).label
            npz_str = r.get("npz_path") or ""
            npz_path = Path(npz_str) if npz_str else None
            has_npz = bool(npz_path and npz_path.exists())
            ready = str(r.get("ready", "False")) == "True" or bool(r.get("ready"))
            chem_ts = r.get("chem_timestamp")
            try:
                chem_ts = float(chem_ts) if chem_ts is not None else None
            except Exception:
                chem_ts = None
            ts_str = r.get("timestamp")
            if not ts_str:
                # Fallback inference
                ts_str = GroupLabeler.infer_from_path(path).timestamp

            key = (plate, (round_name if group_by_round else None))
            grp = groups.get(key)
            item = PairItem(
                path=path,
                recording_stem=stem,
                round=round_name,
                plate=plate,
                group_label=group_label,
                npz_path=npz_path,
                has_npz=has_npz,
                ready=ready,
                timestamp=ts_str,
                chem_timestamp=chem_ts,
            )
            if grp is None:
                grp = PlatePairs(key=key, ctz=[], veh=[], others=[])
                groups[key] = grp
            if group_label == "CTZ":
                grp.ctz.append(item)
            elif group_label == "VEH":
                grp.veh.append(item)
            else:
                grp.others.append(item)
        return PairingIndex(groups)

    def summary_rows(self) -> List[dict]:
        """Return per-group summary counts and estimated number of pairs."""
        rows: List[dict] = []
        for key, grp in sorted(self.groups.items(), key=lambda kv: (kv[0][0] if kv[0][0] is not None else -1, str(kv[0][1]))):
            s = grp.summary()
            pairs, un_ctz, un_veh = grp.pair_by_timestamp()
            s.update(
                {
                    "n_pairs_est": len(pairs),
                    "n_unpaired_ctz": len(un_ctz),
                    "n_unpaired_veh": len(un_veh),
                }
            )
            rows.append(s)
        return rows

    def pairs_dataframe(self) -> List[dict]:
        """Expanded list of candidate pairs with NPZ/ready flags for QA tables.

        Includes both fully paired rows and unpaired leftovers so that a QA
        table can show missing pairs, recording stems, and NPZ presence.
        """
        out: List[dict] = []
        for key, grp in self.groups.items():
            pairs, un_ctz, un_veh = grp.pair_by_timestamp()
            for a, b in pairs:
                out.append(
                    {
                        "plate": key[0],
                        "round": key[1],
                        "ctz_stem": a.recording_stem,
                        "veh_stem": b.recording_stem,
                        "ctz_ready_npz": a.ready and a.has_npz,
                        "veh_ready_npz": b.ready and b.has_npz,
                        "ctz_npz_path": str(a.npz_path or ""),
                        "veh_npz_path": str(b.npz_path or ""),
                        "ctz_time": a.timestamp,
                        "veh_time": b.timestamp,
                        "pair_status": "ready_pair" if (a.ready and a.has_npz and b.ready and b.has_npz) else "incomplete",
                    }
                )
            # Also list unpaired for visibility
            for a in un_ctz:
                out.append(
                    {
                        "plate": key[0],
                        "round": key[1],
                        "ctz_stem": a.recording_stem,
                        "veh_stem": "",
                        "ctz_ready_npz": a.ready and a.has_npz,
                        "veh_ready_npz": False,
                        "ctz_npz_path": str(a.npz_path or ""),
                        "veh_npz_path": "",
                        "ctz_time": a.timestamp,
                        "veh_time": "",
                        "pair_status": "ctz_unpaired",
                    }
                )
            for b in un_veh:
                out.append(
                    {
                        "plate": key[0],
                        "round": key[1],
                        "ctz_stem": "",
                        "veh_stem": b.recording_stem,
                        "ctz_ready_npz": False,
                        "veh_ready_npz": b.ready and b.has_npz,
                        "ctz_npz_path": "",
                        "veh_npz_path": str(b.npz_path or ""),
                        "ctz_time": "",
                        "veh_time": b.timestamp,
                        "pair_status": "veh_unpaired",
                    }
                )
        return out

    @staticmethod
    def load_npz_arrays(item: PairItem):
        """Load IFR arrays for a given recording.

        Returns
        -------
        (time_s, ifr_hz, ifr_hz_smooth): tuple of np.ndarray
            IFR arrays as stored in *_ifr_per_channel_1ms.npz

        Raises
        ------
        FileNotFoundError, ValueError
            If the NPZ is missing or cannot be read.
        """
        import numpy as np
        if not item.npz_path or not item.npz_path.exists():
            raise FileNotFoundError(f"NPZ missing for {item.recording_stem}: {item.npz_path}")
        d = np.load(item.npz_path)
        time_s = np.asarray(d["time_s"], dtype=float)
        ifr = np.asarray(d["ifr_hz"], dtype=float)
        ifr_s = np.asarray(d.get("ifr_hz_smooth", ifr), dtype=float)
        return time_s, ifr, ifr_s
