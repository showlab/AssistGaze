#!/usr/bin/env python3
"""
Expand combined multimodal QA records into per-choice training/eval instances.

Given the `.pt` outputs from feature_combiner (one file per object directory),
this script generates every (video, text, image) answer combination per
question, labels the combination as positive only when all required modalities
pick the correct option, and splits questions into train/test sets before
exploding them to avoid leakage.

Example:
    python -m gazevqa.pipelines.build_instance_splits \
        --input-root QALIST2106 \
        --output-dir qa_instances \
        --train-ratio 0.9 \
        --seed 42
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch

DEFAULT_VIDEO_SHAPE = torch.Size((32, 512))

TEXT_LABEL_TEMPLATE = "Text_Answer{idx}_Label "
VIDEO_LABEL_TEMPLATE = "Video_Answer{idx}_Label"
IMAGE_LABEL_TEMPLATE = "Image_Answer{idx}_Label"


class InstanceBuilder:
    def __init__(
        self,
        input_root: Path,
        output_dir: Path,
        train_ratio: float = 0.9,
        seed: int = 42,
    ) -> None:
        self.input_root = input_root
        self.output_dir = output_dir
        self.train_ratio = train_ratio
        self.rng = random.Random(seed)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run(self) -> None:
        files = self._gather_combined_files()
        if not files:
            raise FileNotFoundError(f"No combined .pt files found under {self.input_root}")

        train_records: List[Dict] = []
        test_records: List[Dict] = []
        question_counts = {"train": 0, "test": 0}

        for pt_path in files:
            entries = torch.load(pt_path, map_location="cpu")
            for question_idx, record in enumerate(entries):
                split = "train" if self.rng.random() < self.train_ratio else "test"
                combos = self._expand_record(pt_path, question_idx, record)
                if split == "train":
                    train_records.extend(combos)
                else:
                    test_records.extend(combos)
                question_counts[split] += 1

        train_path = self.output_dir / "train.pt"
        test_path = self.output_dir / "test.pt"
        torch.save(train_records, train_path)
        torch.save(test_records, test_path)

        print(f"[instance-builder] train questions: {question_counts['train']} ({len(train_records)} combos)")
        print(f"[instance-builder] test questions : {question_counts['test']} ({len(test_records)} combos)")
        print(f"[instance-builder] wrote {train_path} and {test_path}")

    # ------------------------------------------------------------------ helpers
    def _gather_combined_files(self) -> List[Path]:
        if self.input_root.is_file():
            return [self.input_root]
        pattern = "*/*.pt"  # matches S035/S035.pt
        return sorted([p for p in self.input_root.glob(pattern) if p.is_file()])

    def _expand_record(
        self,
        source_path: Path,
        question_idx: int,
        record: Dict,
    ) -> List[Dict]:
        text_opts, text_required = self._gather_text_options(record)
        video_opts, video_required = self._gather_video_options(record)
        image_opts, image_required = self._gather_image_options(record)

        combos: List[Dict] = []
        question_tensor = torch.as_tensor(record.get("QUESTIONS")).clone().half()
        video_slots = self._pad_video_slots(video_opts)
        default_video_feature = video_slots[0]["feature"]
        default_text_feature = text_opts[0]["feature"]
        default_image_feature = image_opts[0]["feature"]
        question_number = self._infer_question_number(record, question_idx)

        for vid_idx, vid in enumerate(video_opts):
            for txt_idx, txt in enumerate(text_opts):
                for img_idx, img in enumerate(image_opts):
                    label = self._is_positive_combo(
                        vid["label"], txt["label"], img["label"],
                        video_required, text_required, image_required,
                    )
                    entry = {
                        "Question": question_tensor.clone(),
                        "Text_Answer": txt["feature"].clone() if txt["feature"] is not None else default_text_feature.clone(),
                        "Image_Answer": img["feature"].clone() if img["feature"] is not None else default_image_feature.clone(),
                        "Video_Answer": vid["feature"].clone() if vid["feature"] is not None else default_video_feature.clone(),
                        "Label": 1 if label else 0,
                        "question_number": question_number,
                    }
                    for slot_idx, slot in enumerate(video_slots, start=1):
                        entry[f"Video_Feature{slot_idx}"] = slot["feature"].clone() if slot["feature"] is not None else default_video_feature.clone()
                    combos.append(entry)
        return combos

    def _gather_text_options(self, record: Dict) -> Tuple[List[Dict], bool]:
        options: List[Dict] = []
        for idx in range(1, 5):
            feature = record.get(f"Text_Answer{idx}")
            if feature is None:
                continue
            tensor = torch.as_tensor(feature).clone().half()
            label_key = TEXT_LABEL_TEMPLATE.format(idx=idx)
            label = int(record.get(label_key, 0))
            options.append({"feature": tensor, "label": label})

        required = any(opt["label"] == 1 for opt in options)
        if not options:
            options.append({"feature": torch.zeros((1, 512), dtype=torch.float16), "label": 1})
            required = False
        return options, required

    def _gather_video_options(self, record: Dict) -> Tuple[List[Dict], bool]:
        options: List[Dict] = []
        for idx in range(1, 6):
            feature = record.get(f"Video_Answer{idx}")
            if feature is None:
                continue
            tensor = torch.as_tensor(feature).clone().half()
            label = int(record.get(VIDEO_LABEL_TEMPLATE.format(idx=idx), 0))
            options.append({"feature": tensor, "label": label})
        required = any(opt["label"] == 1 for opt in options)
        if not options:
            options.append({"feature": torch.zeros(DEFAULT_VIDEO_SHAPE, dtype=torch.float16), "label": 1})
            required = False
        return options, required

    def _gather_image_options(self, record: Dict) -> Tuple[List[Dict], bool]:
        options: List[Dict] = []
        for idx in range(1, 4):
            feature = record.get(f"Image_Answer{idx}")
            if feature is None:
                continue
            tensor = torch.as_tensor(feature).clone()
            label = int(record.get(IMAGE_LABEL_TEMPLATE.format(idx=idx), 0))
            options.append({"feature": tensor, "label": label})
        required = any(opt["label"] == 1 for opt in options)
        if not options:
            options.append({"feature": torch.zeros((1, 1024)), "label": 1})
            required = False
        return options, required

    @staticmethod
    def _is_positive_combo(
        video_label: int,
        text_label: int,
        image_label: int,
        video_required: bool,
        text_required: bool,
        image_required: bool,
    ) -> bool:
        video_ok = True if not video_required else video_label == 1
        text_ok = True if not text_required else text_label == 1
        image_ok = True if not image_required else image_label == 1
        return video_ok and text_ok and image_ok

    def _pad_video_slots(self, video_opts: List[Dict], target: int = 5) -> List[Dict]:
        slots = video_opts[:target]
        embed_shape = slots[0]["feature"].shape if slots else DEFAULT_VIDEO_SHAPE
        while len(slots) < target:
            slots.append({"feature": torch.zeros(embed_shape, dtype=torch.float16)})
        return slots

    @staticmethod
    def _infer_question_number(record: Dict, default_idx: int) -> int:
        for key in ("question_number", "Global Step ID", "IMAGE QUESTION ID", "local_step_id"):
            value = record.get(key)
            try:
                if value is not None:
                    return int(value)
            except (TypeError, ValueError):
                continue
        return int(default_idx)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create train/test splits of multimodal QA instances.")
    parser.add_argument("--input-root", required=True, help="Folder containing combined .pt files (e.g., QALIST2106).")
    parser.add_argument("--output-dir", default="qa_instances", help="Where to write train/test .pt files.")
    parser.add_argument("--train-ratio", type=float, default=0.9, help="Fraction of questions assigned to train.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for question-level split.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    builder = InstanceBuilder(
        input_root=Path(args.input_root),
        output_dir=Path(args.output_dir),
        train_ratio=args.train_ratio,
        seed=args.seed,
    )
    builder.run()


if __name__ == "__main__":
    main()
