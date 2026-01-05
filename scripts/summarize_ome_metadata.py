#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import xml.etree.ElementTree as ET


def parse_float(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _extract_xml(text: str) -> str:
    start = text.find("<OME")
    if start == -1:
        start = text.find("<?xml")
    if start == -1:
        raise ValueError("OME-XML start tag not found.")
    end = text.rfind("</OME>")
    if end != -1:
        end += len("</OME>")
        return text[start:end]
    return text[start:]


def parse_ome(path: Path) -> Dict[str, Any]:
    raw = path.read_text(errors="replace")
    xml_text = _extract_xml(raw)
    root = ET.fromstring(xml_text)
    ns_uri = root.tag.split("}")[0].strip("{") if "}" in root.tag else ""

    def q(tag: str) -> str:
        return f"{{{ns_uri}}}{tag}" if ns_uri else tag

    instruments = root.findall(q("Instrument"))
    instrument = instruments[0] if instruments else None

    instrument_info: Dict[str, Any] = {}
    if instrument is not None:
        objective = instrument.find(q("Objective"))
        if objective is not None:
            instrument_info["objective"] = {
                "manufacturer": objective.get("Manufacturer"),
                "model": objective.get("Model"),
                "nominal_magnification": parse_float(objective.get("NominalMagnification")),
                "lens_na": parse_float(objective.get("LensNA")),
                "immersion": objective.get("Immersion"),
            }
        detector = instrument.find(q("Detector"))
        if detector is not None:
            instrument_info["detector"] = {
                "manufacturer": detector.get("Manufacturer"),
                "model": detector.get("Model"),
                "type": detector.get("Type"),
            }
        laser = instrument.find(q("Laser"))
        if laser is not None:
            instrument_info["laser"] = {
                "type": laser.get("Type"),
                "wavelength": parse_float(laser.get("Wavelength")),
                "wavelength_unit": laser.get("WavelengthUnit"),
                "power": parse_float(laser.get("Power")),
                "power_unit": laser.get("PowerUnit"),
            }

    images = root.findall(q("Image"))
    if not images:
        return {
            "path": str(path),
            "instrument": instrument_info,
            "images": [],
        }

    image_entries: List[Dict[str, Any]] = []
    for image in images:
        pixels = image.find(q("Pixels"))
        if pixels is None:
            continue
        pixels_info = {
            "size_x": parse_float(pixels.get("SizeX")),
            "size_y": parse_float(pixels.get("SizeY")),
            "size_z": parse_float(pixels.get("SizeZ")),
            "size_c": parse_float(pixels.get("SizeC")),
            "size_t": parse_float(pixels.get("SizeT")),
            "physical_size_x": parse_float(pixels.get("PhysicalSizeX")),
            "physical_size_y": parse_float(pixels.get("PhysicalSizeY")),
            "physical_size_x_unit": pixels.get("PhysicalSizeXUnit"),
            "physical_size_y_unit": pixels.get("PhysicalSizeYUnit"),
        }

        channels = []
        for ch in pixels.findall(q("Channel")):
            channels.append(
                {
                    "id": ch.get("ID"),
                    "name": ch.get("Name"),
                    "excitation_wavelength": parse_float(ch.get("ExcitationWavelength")),
                    "excitation_wavelength_unit": ch.get("ExcitationWavelengthUnit"),
                    "emission_wavelength": parse_float(ch.get("EmissionWavelength")),
                    "emission_wavelength_unit": ch.get("EmissionWavelengthUnit"),
                }
            )

        exposure_unit = None
        exposure_by_c: Dict[int, List[float]] = {}
        for plane in pixels.findall(q("Plane")):
            the_c = plane.get("TheC")
            if the_c is None:
                continue
            c_idx = int(the_c)
            exposure = parse_float(plane.get("ExposureTime"))
            if exposure is None:
                continue
            exposure_unit = plane.get("ExposureTimeUnit") or exposure_unit
            exposure_by_c.setdefault(c_idx, []).append(exposure)

        exposure_summary = {}
        for c_idx, values in exposure_by_c.items():
            exposure_summary[c_idx] = {
                "mean": float(sum(values) / len(values)),
                "median": float(sorted(values)[len(values) // 2]),
                "min": float(min(values)),
                "max": float(max(values)),
            }

        image_entries.append(
            {
                "name": image.get("Name"),
                "pixels": pixels_info,
                "channels": channels,
                "exposure_unit": exposure_unit,
                "exposure_by_channel": exposure_summary,
            }
        )

    return {
        "path": str(path),
        "instrument": instrument_info,
        "images": image_entries,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize OME-XML metadata into JSON/CSV.")
    parser.add_argument(
        "--bio-dir",
        type=Path,
        required=True,
        help="Directory containing OME-XML files (from export_nd2_ground_truth.py --bioformats).",
    )
    parser.add_argument("--out-dir", type=Path, default=None, help="Output directory.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bio_dir = args.bio_dir.expanduser()
    out_dir = args.out_dir.expanduser() if args.out_dir else bio_dir.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    summaries = []
    for xml_path in sorted(bio_dir.glob("*.xml")):
        try:
            summaries.append(parse_ome(xml_path))
        except Exception as exc:
            print(f"Skipping {xml_path.name}: {exc}")

    json_path = out_dir / "ome_metadata_summary.json"
    json_path.write_text(json.dumps(summaries, indent=2))

    channel_rows: List[Dict[str, Any]] = []
    for entry in summaries:
        for image in entry.get("images", []):
            exposure_unit = image.get("exposure_unit")
            for idx, ch in enumerate(image.get("channels", [])):
                row = {
                    "file": entry.get("path"),
                    "image_name": image.get("name"),
                    "channel_index": idx,
                    "channel_name": ch.get("name"),
                    "excitation_wavelength": ch.get("excitation_wavelength"),
                    "excitation_wavelength_unit": ch.get("excitation_wavelength_unit"),
                    "emission_wavelength": ch.get("emission_wavelength"),
                    "emission_wavelength_unit": ch.get("emission_wavelength_unit"),
                    "exposure_unit": exposure_unit,
                }
                exposure = image.get("exposure_by_channel", {}).get(idx)
                if exposure:
                    row.update({f"exposure_{k}": v for k, v in exposure.items()})
                channel_rows.append(row)

    if channel_rows:
        csv_path = out_dir / "ome_metadata_channels.csv"
        headers = sorted({key for row in channel_rows for key in row.keys()})
        with csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            writer.writerows(channel_rows)
    print(f"Saved OME JSON to {json_path}")
    if channel_rows:
        print(f"Saved OME channel CSV to {out_dir / 'ome_metadata_channels.csv'}")


if __name__ == "__main__":
    main()
