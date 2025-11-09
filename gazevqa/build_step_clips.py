#!/usr/bin/env python3
"""
Slice ELAN-labelled multi-view videos into fixed-length action clips.

Expected directory layout (example for S031):

<dataset_root>/
  S031/
    firstview/
      assembly.mp4
    firstviewraw/
      assembly.mp4
    secondview/
      assembly.mp4
    thirdview/
      assembly.mp4

<elan_root>/
  S031/
    *.eaf

For each ELAN file we read the `Subject_Action` tier, cut out every single
action interval, and save one multi-view clip per step. The resulting videos
are written to `--output-root/<label>/<view>/<phase>_stepXX.mp4`, and metadata
is written to `annotation.json`.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
from pympi.Elan import Eaf


DEFAULT_VIEWS = ("firstview", "firstviewraw", "secondview", "thirdview")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Crop ELAN-labelled videos into action clips.")
    parser.add_argument("--elan-root", type=Path, required=True, help="Folder containing ELAN/<label>/*.eaf")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help="Folder containing <label>/<view>/<video_file> structure.",
    )
    parser.add_argument("--output-root", type=Path, default=Path("output"), help="Destination root for cropped videos.")
    parser.add_argument("--views", nargs="+", default=list(DEFAULT_VIEWS), help="Ordered list of view subfolders.")
    parser.add_argument("--assembly-filename", default="assembly.mp4", help="Assembly-phase MP4 filename inside each view.")
    parser.add_argument("--disassembly-filename", default="disassembly.mp4", help="Disassembly-phase MP4 filename inside each view.")
    parser.add_argument("--codec", choices=["copy", "h264", "h265"], default="h264", help="FFmpeg video codec.")
    parser.add_argument("--bitrate", default="12M", help="Bitrate used when re-encoding video.")
    parser.add_argument("--skip-ffmpeg", action="store_true", help="Only emit metadata JSON without trimming videos.")
    return parser.parse_args()


# ------------------------------------------------------------------------- ELAN --
def load_annotations(elan_path: Path) -> tuple[np.ndarray, np.ndarray]:
    elan = Eaf(str(elan_path))

    def collect_rows(tier_name: str) -> List[List[float | str]]:
        if tier_name not in elan.tiers:
            return []
        rows: List[List[float | str]] = []
        for event in elan.tiers[tier_name][0].values():
            start = elan.timeslots[event[0]] / 1000.0
            end = elan.timeslots[event[1]] / 1000.0
            rows.append([event[2], start, end])
        return rows

    action_rows = collect_rows("Subject_Action")
    instruction_rows: List[List[float | str]] = []
    for candidate in ("Instruction", "instruction", "Instructions"):
        instruction_rows.extend(collect_rows(candidate))

    actions = np.array(action_rows, dtype=object) if action_rows else np.empty((0, 3), dtype=object)
    instructions = np.array(instruction_rows, dtype=object) if instruction_rows else np.empty((0, 3), dtype=object)
    return actions, instructions


# ---------------------------------------------------------------------- Helpers --
def find_label_dir(dataset_root: Path, label: str, views: Sequence[str]) -> Path | None:
    """
    Locate the directory containing the required view folders for a label.
    Handles layouts like dataset_root/S031 and dataset_root/S031/S031.
    """
    candidates: Iterable[Path] = [
        dataset_root / label,
        dataset_root / label / label,
    ]
    for candidate in candidates:
        if not candidate.is_dir():
            continue
        existing = [(candidate / view).is_dir() for view in views]
        if any(existing):
            return candidate
    return None


def view_sources(
    label_dir: Path,
    views: Sequence[str],
    assembly_filename: str,
    disassembly_filename: str,
) -> Dict[str, Dict[str, Path]]:
    sources: Dict[str, Dict[str, Path]] = {}
    for view in views:
        assembly_path = label_dir / view / assembly_filename
        disassembly_path = label_dir / view / disassembly_filename
        if not assembly_path.is_file() and not disassembly_path.is_file():
            print(f"[warn] skipped {label_dir.name}/{view}: missing both assembly and disassembly video")
            continue
        sources[view] = {}
        if assembly_path.is_file():
            sources[view]["assembly"] = assembly_path
        if disassembly_path.is_file():
            sources[view]["disassembly"] = disassembly_path
    if not sources:
        raise FileNotFoundError(f"No usable views found under {label_dir}")
    return sources


def format_clip_name(step_idx: int) -> str:
    return f"step{step_idx:02d}.mp4"


def extract_phase_bounds(*annotation_sets: np.ndarray) -> Dict[str, float]:
    bounds: Dict[str, float] = {}
    for annotations in annotation_sets:
        if annotations.size == 0:
            continue
        for text, start, end in annotations:
            normalized = " ".join(str(text).strip().lower().split())
            if "assembly completed" in normalized:
                bounds.setdefault("assembly_end", float(start))
            elif "start disassembly" in normalized:
                marker_time = float(end)
                bounds.setdefault("start_disassembly", marker_time)
            elif "disassembly completed" in normalized:
                bounds.setdefault("disassembly_end", float(end))
    if "start_disassembly" not in bounds:
        print("[warn] No 'start disassembly' marker found in Subject_Action/Instruction tiers, defaulting to assembly-only.")
    return bounds


def determine_phase(start: float, end: float, bounds: Dict[str, float]) -> str:
    disassembly_start = bounds.get("start_disassembly")
    if disassembly_start is None:
        return "assembly"
    if start >= disassembly_start:
        return "disassembly"
    if end <= disassembly_start:
        return "assembly"
    # overlap -> default to disassembly for safety
    return "disassembly"



def cut_video(source: Path, start: float, end: float, out_path: Path, codec: str, bitrate: str) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-y", "-ss", f"{start:.3f}", "-to", f"{end:.3f}", "-i", str(source)]
    if codec == "copy":
        cmd += ["-vcodec", "copy", "-acodec", "copy"]
    elif codec == "h264":
        cmd += ["-vcodec", "libx264", "-b:v", bitrate]
    else:
        cmd += ["-vcodec", "libx265", "-b:v", bitrate]
    cmd.append(str(out_path))
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


# ------------------------------------------------------------------------- Main --
def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    manifest: Dict[str, List[Dict[str, float | str]]] = {}

    elan_files = sorted(
        p
        for p in args.elan_root.rglob("*.eaf")
        if any("seg3" in part.lower() for part in p.parts)
    )
    if not elan_files:
        raise FileNotFoundError(f"No ELAN files found under {args.elan_root}")

    for elan_path in elan_files:
        label = elan_path.relative_to(args.elan_root).parts[0]
        label_dir = find_label_dir(args.dataset_root, label, args.views)
        if not label_dir:
            print(f"[warn] skipped {label}: cannot find view folders under {args.dataset_root}")
            continue

        action_rows, instruction_rows = load_annotations(elan_path)
        if len(action_rows) == 0:
            print(f"[warn] skipped {label}: no Subject_Action annotations found")
            continue

        sources = view_sources(label_dir, args.views, args.assembly_filename, args.disassembly_filename)
        phase_bounds = extract_phase_bounds(action_rows, instruction_rows)
        print(
            f"[debug] {label}: phase_bounds={phase_bounds}, "
            f"views={list(sources.keys())}, instructions_loaded={len(instruction_rows)}"
        )
        clips: List[Dict[str, float | str]] = []

        assembly_offset = float(action_rows[:, 1].min())
        phase_counts = {"assembly": 0, "disassembly": 0}
        global_step_idx = 0
        for _, start_time, end_time in action_rows:
            start_time = float(start_time)
            end_time = float(end_time)
            phase = determine_phase(start_time, end_time, phase_bounds)
            phase_counts[phase] += 1
            global_step_idx += 1
            clip_name = format_clip_name(global_step_idx)
            if phase == "disassembly":
                offset = phase_bounds.get("start_disassembly", assembly_offset)
            else:
                offset = assembly_offset
            adj_start = max(0.0, start_time - offset)
            adj_end = max(adj_start, end_time - offset)
            print(
                f"[debug] {label} {clip_name}: phase={phase}, "
                f"orig=({start_time:.3f}-{end_time:.3f}), offset={offset:.3f}, "
                f"adjusted=({adj_start:.3f}-{adj_end:.3f})"
            )

            clip_meta: Dict[str, float | str] = {
                "object": label,
                "clip": clip_name,
                "phase": phase,
                "start_time": start_time,
                "end_time": end_time,
                "relative_start": adj_start,
                "relative_end": adj_end,
                "step_index": global_step_idx,
                "phase_step_index": phase_counts[phase],
            }

            if not args.skip_ffmpeg:
                for view, source_bundle in sources.items():
                    source = source_bundle.get(phase)
                    if source is None:
                        print(f"[warn] {label} {clip_name}: view {view} missing '{phase}' video, skipping view")
                        continue
                    out_path = args.output_root / label / view / clip_name
                    cut_video(source, adj_start, adj_end, out_path, args.codec, args.bitrate)
                    clip_meta[f"{view}_path"] = str(out_path)

            clips.append(clip_meta)

        if clips:
            manifest[label] = clips

    manifest_path = args.output_root / "annotation.json"
    with manifest_path.open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"[videocrop] wrote metadata for {len(manifest)} labels to {manifest_path}")


if __name__ == "__main__":
    main()
