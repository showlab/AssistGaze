#!/usr/bin/env python3
"""
Build unified CLIP video features directly from raw multi-view folders.

Each instance folder (e.g., S035) should contain subdirectories such as
`firstview`, `secondview`, etc., each filled with mp4 files following the
pattern: `stepXX.mp4`, where `XX` is the sequential step index defined in the
QALIST spreadsheets (assembly first, then disassembly continues counting).

For every step we gather up to four views, encode them with CLIP by sampling a
fixed number of frames, and store the results in one `.pt` file that only
contains the four view keys (`first`, `second`, `third`, `firstraw`), each
holding all sampled frame embeddings (`frames_per_view x 512`). Downstream we
flatten the views (4) and frames (8) into a `(32, 512)` tensor to remain
consistent with the legacy layout.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, Tuple

import torch

from .encoders import VideoClipEncoder

STEP_PATTERN = re.compile(r"^step(?P<index>\d+)\.mp4$", re.IGNORECASE)
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
        if not self.raw_root.exists():
            raise FileNotFoundError(f"[video-builder] raw_root not found: {self.raw_root}")
        instance_dirs = sorted([d for d in self.raw_root.iterdir() if d.is_dir()])
        if not instance_dirs:
            print(f"[video-builder] no instance folders under {self.raw_root}")
            return
        for instance_dir in instance_dirs:
            instance_id = instance_dir.name
            print(f"[video-builder] {instance_id}")
            meta = self._collect_instance(instance_dir)
            self._encode_instance(instance_id, meta)

    # ----------------------------------------------------------------- internals
    def _collect_instance(self, instance_dir: Path) -> Dict[int, Dict]:
        instance_meta: Dict[int, Dict] = {}
        missing_views = []
        for view_name in self.views:
            view_path = instance_dir / view_name
            if not view_path.is_dir():
                missing_views.append(view_name)
                continue
            for video_file in sorted(view_path.glob("*.mp4")):
                parsed = self._parse_video_name(video_file.name)
                if not parsed:
                    continue
                step_index = parsed["step_index"]
                block = instance_meta.setdefault(step_index, {"views": {}})
                block["views"][view_name] = video_file
        if missing_views:
            print(f"[video-builder][warn] {instance_dir.name}: missing view folders {missing_views}")
        if not instance_meta:
            print(f"[video-builder][warn] {instance_dir.name}: no mp4 files detected")
        return instance_meta

    def _encode_instance(self, instance_id: str, meta: Dict[int, Dict]) -> None:
        instance_out = self.output_dir / instance_id
        instance_out.mkdir(parents=True, exist_ok=True)

        if not meta:
            print(f"[video-builder][warn] {instance_id}: metadata empty, skip encoding")
            return

        for step_index in sorted(meta.keys()):
            info = meta[step_index]
            view_payload: Dict[str, torch.Tensor] = {}
            for view_name in self.views:
                video_path = info["views"].get(view_name)
                if not video_path:
                    continue
                try:
                    encoded = self.encoder.encode(str(video_path), aggregate="stack").cpu()
                except Exception as exc:  # graceful fallback
                    print(f"[video-builder][warn] {instance_id} step{step_index:02d} view={view_name} encode failed: {exc}")
                    continue
                view_payload[view_name] = encoded

            if not view_payload:
                print(f"[video-builder][warn] {instance_id} step{step_index:02d}: no views encoded, skipping output")
                continue

            payload = {
                "step_index": step_index,
                "frames_per_view": self.encoder.num_frames,
            }
            for idx, view_name in enumerate(self.views):
                alias = self._alias_view(view_name, idx)
                payload[alias] = view_payload.get(view_name)

            out_path = instance_out / f"step{step_index:02d}.pt"
            torch.save(payload, out_path)
            print(f"[video-builder] {instance_id} -> {out_path.relative_to(self.output_dir)}")

    @staticmethod
    def _parse_video_name(name: str):
        match = STEP_PATTERN.match(name)
        if not match:
            return None
        return {"step_index": int(match.group("index"))}

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
