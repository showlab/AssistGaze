#!/usr/bin/env python3
"""
Build unified CLIP video features directly from raw multi-view folders.

Each instance folder (e.g., S035) should contain subdirectories such as
`firstview`, `secondview`, etc., each filled with mp4 files following the
pattern: `{mode}_action{start}_{end}.mp4`, where `mode` is `assembly` or
`disassembly`.

For every (mode, start, end) combination we gather up to four views, encode
them with CLIP by sampling a fixed number of frames, and store the results in
one `.pt` file that only contains the four view keys (`first`, `second`,
`third`, `firstraw`), each holding all sampled frame embeddings
(`frames_per_view x 512`). Downstream we flatten the views (4) and frames (8)
into a `(32, 512)` tensor to remain consistent with the legacy layout.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, Tuple

import torch

from .encoders import VideoClipEncoder

VIDEO_PATTERN = re.compile(
    r"^(?P<mode>assembly|disassembly)_action(?P<start>\d+)_(?P<end>\d+)\.mp4$",
    re.IGNORECASE,
)
DEFAULT_VIEWS = ("firstview", "firstviewraw", "secondview", "thirdview")


class MultiViewVideoBuilder:
    def __init__(
        self,
        raw_root: Path,
        output_dir: Path,
        views: Tuple[str, ...] = DEFAULT_VIEWS,
        num_frames: int = 8,
        device: str = "cuda:0",
        clip_model: str = "ViT-B/32",
    ) -> None:
        self.raw_root = raw_root
        self.output_dir = output_dir
        self.views = tuple(views)
        self.encoder = VideoClipEncoder(
            model_name=clip_model,
            device=device,
            num_frames=num_frames,
            batch_size=4,
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.alias_map = self._build_alias_map()

    # ------------------------------------------------------------------ public API
    def run(self) -> None:
        instance_dirs = sorted([d for d in self.raw_root.iterdir() if d.is_dir()])
        for instance_dir in instance_dirs:
            instance_id = instance_dir.name
            print(f"[video-builder] {instance_id}")
            meta = self._collect_instance(instance_dir)
            self._encode_instance(instance_id, meta)

    # ----------------------------------------------------------------- internals
    def _collect_instance(self, instance_dir: Path) -> Dict[str, Dict[str, Dict]]:
        instance_meta: Dict[str, Dict[str, Dict]] = {}
        for view_name in self.views:
            view_path = instance_dir / view_name
            if not view_path.is_dir():
                continue
            for video_file in sorted(view_path.glob("*.mp4")):
                parsed = self._parse_video_name(video_file.name)
                if not parsed:
                    continue
                mode = parsed["mode"]
                step_key = f"{parsed['start']}_{parsed['end']}"
                block = instance_meta.setdefault(mode, {}).setdefault(
                    step_key,
                    {
                        "start_step": int(parsed["start"]),
                        "end_step": int(parsed["end"]),
                        "views": {},
                    },
                )
                block["views"][view_name] = video_file
        return instance_meta

    def _encode_instance(self, instance_id: str, meta: Dict[str, Dict[str, Dict]]) -> None:
        instance_out = self.output_dir / instance_id
        instance_out.mkdir(parents=True, exist_ok=True)

        for mode, steps in meta.items():
            for step_key, info in steps.items():
                view_payload: Dict[str, torch.Tensor] = {}
                for view_name in self.views:
                    video_path = info["views"].get(view_name)
                    if not video_path:
                        continue
                    try:
                        encoded = self.encoder.encode(str(video_path), aggregate="stack").cpu()
                    except Exception as exc:  # graceful fallback
                        print(f"[video-builder][warn] failed to encode {video_path}: {exc}")
                        continue
                    view_payload[view_name] = encoded

                if not view_payload:
                    continue

                payload = {}
                for idx, view_name in enumerate(self.views):
                    alias = self._alias_view(view_name, idx)
                    payload[alias] = view_payload.get(view_name)

                out_path = instance_out / f"{mode}_action{step_key}.pt"
                torch.save(payload, out_path)

    @staticmethod
    def _parse_video_name(name: str):
        match = VIDEO_PATTERN.match(name)
        if not match:
            return None
        return {
            "mode": match.group("mode").lower(),
            "start": match.group("start"),
            "end": match.group("end"),
        }

    def _build_alias_map(self) -> Dict[str, str]:
        aliases = {
            "firstview": "first",
            "secondview": "second",
            "thirdview": "third",
            "firstviewraw": "firstraw",
        }
        return {view: aliases.get(view, view) for view in self.views}

    def _alias_view(self, view_name: str, idx: int) -> str:
        return self.alias_map.get(view_name, f"view{idx}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Encode multi-view videos into CLIP features.")
    parser.add_argument("--raw-root", required=True, help="Folder containing Sxxx instances.")
    parser.add_argument("--output-dir", default="video_features_pt", help="Where to save per-instance tensors.")
    parser.add_argument("--views", nargs="+", default=DEFAULT_VIEWS, help="Ordered list of view subfolder names.")
    parser.add_argument("--frames-per-video", type=int, default=8, help="CLIP frames sampled per video.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--clip-model", default="ViT-B/32")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    builder = MultiViewVideoBuilder(
        raw_root=Path(args.raw_root),
        output_dir=Path(args.output_dir),
        views=tuple(args.views),
        num_frames=args.frames_per_video,
        device=args.device,
        clip_model=args.clip_model,
    )
    builder.run()


if __name__ == "__main__":
    main()
