#!/usr/bin/env python3
"""
Build ROI features for SAM-labelled images.

Directory layout that the builder expects:

<sam_root>/
  S005/
    assembly/
      1/
        firstviewraw/*.jpg
        secondviewraw/*.jpg        (optional)
        ...
        label_rectangle/*.json     (Labelme rectangles with same basename)
      2/
        ...
    disassembly/
      ...

Each <step>/<view> image is paired with a rectangle-json file describing one or
more bounding boxes. For every image we encode ROIs via Faster R-CNN and save
the resulting tensors to `<output_root>/<instance>/<mode>/<step>.pt`.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Sequence

import torch

from .encoders import RoiFeatureExtractor

IMAGE_INDEX_PATTERN = re.compile(r"(\d+)")


class SamImageFeatureBuilder:
    def __init__(
        self,
        sam_root: Path,
        output_root: Path,
        device: str = "cuda:0",
        views_suffix: str = "viewraw",
        image_exts: Sequence[str] = (".jpg", ".png", ".jpeg"),
        max_boxes: int | None = None,
    ) -> None:
        self.sam_root = sam_root
        self.output_root = output_root
        self.views_suffix = views_suffix
        self.image_exts = tuple(image_exts)
        self.max_boxes = max_boxes
        self.extractor = RoiFeatureExtractor(device=device)
        self.output_root.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ public API
    def run(self) -> None:
        instances = sorted([p for p in self.sam_root.iterdir() if p.is_dir()])
        for instance_dir in instances:
            instance_id = instance_dir.name
            print(f"[image-builder] {instance_id}")
            for mode_dir in sorted([d for d in instance_dir.iterdir() if d.is_dir()]):
                mode = mode_dir.name
                self._process_mode(instance_id, mode, mode_dir)

    # ---------------------------------------------------------------- internally
    def _process_mode(self, instance_id: str, mode: str, mode_dir: Path) -> None:
        step_dirs = sorted([d for d in mode_dir.iterdir() if d.is_dir()])
        for step_dir in step_dirs:
            step_id = step_dir.name
            entries = self._collect_step_entries(instance_id, mode, step_id, step_dir)
            if not entries:
                continue
            out_dir = self.output_root / instance_id / mode
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"{step_id}.pt"
            torch.save(entries, out_path)

    def _collect_step_entries(
        self,
        instance_id: str,
        mode: str,
        step_id: str,
        step_dir: Path,
    ) -> List[Dict]:
        entries: List[Dict] = []
        label_dir = step_dir / "label_rectangle"
        if not label_dir.is_dir():
            return entries

        view_dirs = [d for d in step_dir.iterdir() if d.is_dir() and d.name.endswith(self.views_suffix)]
        for view_dir in view_dirs:
            view_name = view_dir.name
            for image_path in self._iter_images(view_dir):
                image_index = self._extract_image_index(image_path)
                boxes = self._load_boxes(label_dir / f"{image_path.stem}.json")
                if not boxes:
                    continue
                try:
                    features = self.extractor.encode(str(image_path), boxes)
                except Exception as exc:
                    print(f"[image-builder][warn] skip {image_path}: {exc}")
                    continue
                entries.append(
                    {
                        "instance_id": instance_id,
                        "mode": mode,
                        "step_id": step_id,
                        "view": view_name,
                        "image_index": image_index,
                        "image_filename": image_path.name,
                        "image_path": str(image_path),
                        "boxes_xyxy": boxes,
                        "features": features.cpu(),
                    }
                )
        return entries

    def _iter_images(self, view_dir: Path):
        files = []
        for ext in self.image_exts:
            files.extend(view_dir.glob(f"*{ext}"))
        files.sort(key=self._image_sort_key)
        for path in files:
            yield path

    @staticmethod
    def _image_sort_key(path: Path):
        match = IMAGE_INDEX_PATTERN.search(path.stem)
        if match:
            return (int(match.group(1)), path.name.lower())
        return (float("inf"), path.name.lower())

    @staticmethod
    def _extract_image_index(path: Path):
        match = IMAGE_INDEX_PATTERN.search(path.stem)
        if match:
            return int(match.group(1))
        return None

    def _load_boxes(self, label_path: Path) -> List[List[float]]:
        if not label_path.exists():
            return []
        data = json.loads(label_path.read_text())
        shapes = data.get("shapes", [])
        boxes: List[List[float]] = []
        for shape in shapes:
            if shape.get("shape_type") != "rectangle":
                continue
            points = shape.get("points", [])
            if len(points) != 2:
                continue
            (x1, y1), (x2, y2) = points
            xmin, xmax = sorted([float(x1), float(x2)])
            ymin, ymax = sorted([float(y1), float(y2)])
            boxes.append([xmin, ymin, xmax, ymax])
            if self.max_boxes and len(boxes) >= self.max_boxes:
                break
        return boxes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Encode SAM image crops with Faster R-CNN.")
    parser.add_argument("--sam-root", required=True, help="Path to SAMv2 directory (containing Sxxx folders).")
    parser.add_argument("--output-root", default="image_features_pt", help="Where to store encoded tensors.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-boxes", type=int, default=None, help="Optionally limit boxes per image.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    builder = SamImageFeatureBuilder(
        sam_root=Path(args.sam_root),
        output_root=Path(args.output_root),
        device=args.device,
        max_boxes=args.max_boxes,
    )
    builder.run()


if __name__ == "__main__":
    main()
