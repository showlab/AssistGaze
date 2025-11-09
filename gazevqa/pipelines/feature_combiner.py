#!/usr/bin/env python3
"""
Cleaner feature-combining pipeline for the QALIST dataset.

This script keeps the original combineallfeatures1406.py behavior but organizes it into
small, testable units. It pulls CLIP text embeddings, pre-computed video features, and
SAM/Faster R-CNN ROI tensors into a single Python list per object folder which is then
saved as <object_id>.pt next to the Excel source.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import math
import re
import random
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

from .encoders import TextEncoder


class FeatureCombiner:
    """Orchestrates reading metadata, encoding text, and attaching precomputed features."""

    def __init__(
        self,
        root_dir: Path,
        clip_feature_dir: Path,
        sam_feature_dir: Path,
        clip_model: str = "ViT-B/32",
        device: str = "cuda:0",
        shuffle_answers: bool = True,
        shuffle_seed: int = 42,
    ) -> None:
        self.root_dir = root_dir
        self.clip_feature_dir = clip_feature_dir
        self.sam_feature_dir = sam_feature_dir
        self.text_encoder = TextEncoder(model_name=clip_model, device=device)
        self.view_keys: Sequence[str] = ("first", "second", "third", "firstraw")
        self.video_embedding_dim: int = 512
        self.shuffle_answers = shuffle_answers
        self.rng = random.Random(shuffle_seed)

    # --------------------------------------------------------------------- public API
    def run(self) -> None:
        """Iterate over each object directory and generate a *.pt output beside it."""
        for object_dir in sorted(self.root_dir.iterdir()):
            if not object_dir.is_dir():
                continue

            object_id = object_dir.name
            print(f"[FeatureCombiner] processing {object_id}")
            video_lookup, video_shape = self._load_video_lookup(object_id)
            combined_rows = self._process_object_dir(object_dir, object_id, video_lookup, video_shape)

            out_path = object_dir / f"{object_id}.pt"
            torch.save(combined_rows, out_path)

    # ---------------------------------------------------------------- internal helpers
    def _process_object_dir(
        self,
        object_dir: Path,
        object_id: str,
        video_lookup: Dict[str, torch.Tensor],
        video_shape: torch.Size,
    ) -> List[Dict[str, torch.Tensor]]:
        results: List[Dict[str, torch.Tensor]] = []
        excel_files = sorted(object_dir.glob("*.xlsx"))

        for excel_path in excel_files:
            type_name = excel_path.stem.split("_")[-2].lower()
            for sheet_idx, sheet_df in self._iter_excel_sheets(excel_path):
                sam_path = self.sam_feature_dir / object_id / type_name / f"{sheet_idx}.pt"
                image_features = torch.load(sam_path)

                for _, row in sheet_df.iterrows():
                    results.append(self._assemble_row(row, image_features, video_lookup, video_shape))

                del image_features  # release RAM ASAP

        return results

    def _assemble_row(
        self,
        row: pd.Series,
        image_features: Sequence[dict],
        video_lookup: Dict[str, torch.Tensor],
        video_shape: torch.Size,
    ) -> Dict[str, torch.Tensor]:
        record: Dict[str, torch.Tensor] = {}

        record["QUESTIONS"] = self.text_encoder(row["QUESTIONS"])
        for idx in range(1, 5):
            record[f"Text_Answer{idx}"] = self.text_encoder(row[f"Text_Answer{idx}"])
            label_key = f"Text_Answer{idx}_Label "
            record[label_key] = row[label_key]

        record["Object ID"] = row["Object ID"]
        record["Global Step ID"] = row["Global Step ID"]
        record["IMAGE QUESTION ID"] = row["IMAGE QUESTION ID"]
        record["local_step_id"] = row["local_step_id"]

        video_labels = [
            row["Video_Answer1_Label"],
            row["Video_Answer2_Label"],
            row["Video_Answer3_Label"],
            row["Video_Answer4_Label"],
            row["Video_Answer5_Label"],
        ]
        record.update(
            self._collect_video_answers(
                object_id=str(row["Object ID"]),
                steps=[row[f"Video_Answer{i}"] for i in range(1, 6)],
                labels=video_labels,
                video_lookup=video_lookup,
                video_shape=video_shape,
            )
        )
        for idx, label in enumerate(video_labels, start=1):
            record[f"Video_Answer{idx}_Label"] = label

        image_labels = [
            row["Image_Answer1_Label"],
            row["Image_Answer2_Label"],
            row["Image_Answer3_Label"],
        ]
        image_entries = self._build_image_entries(
            question_id=row["IMAGE QUESTION ID"],
            labels=image_labels,
            image_features=image_features,
        )
        image_entries = self._shuffle_entries(image_entries)
        for idx, entry in enumerate(image_entries, start=1):
            record[f"Image_Answer{idx}"] = entry["feature"]
            record[f"Image_Answer{idx}_Label"] = entry["label"]

        return record

    def _build_image_entries(
        self,
        question_id: str | int,
        labels: Sequence[int],
        image_features: Sequence[dict],
    ) -> List[Dict[str, torch.Tensor]]:
        placeholder = torch.zeros((1, 1024))
        base_answers = [
            {"feature": placeholder.clone(), "label": labels[0]},
            {"feature": placeholder.clone(), "label": labels[1]},
            {"feature": placeholder.clone(), "label": labels[2]},
        ]

        if all(int(label) == 1 for label in labels):
            return base_answers

        if question_id is None:
            return base_answers
        if isinstance(question_id, str):
            question_id = question_id.strip()
            if not question_id:
                return base_answers
        if isinstance(question_id, float) and math.isnan(question_id):
            return base_answers

        try:
            target_qid = int(question_id)
        except (TypeError, ValueError):
            return base_answers

        has_legacy_format = any(
            isinstance(feature, dict) and "question id" in feature for feature in image_features
        )

        if has_legacy_format:
            for feature in image_features:
                if int(feature.get("question id", -1)) != target_qid:
                    continue
                return self._assign_rotating_bboxes(feature, placeholder, labels)
            return base_answers

        matched_entry = self._find_image_entry_by_index(image_features, target_qid)
        if matched_entry is None:
            return base_answers
        return self._assign_roi_features(matched_entry, placeholder, labels)

    @staticmethod
    def _assign_rotating_bboxes(
        feature: dict,
        placeholder: torch.Tensor,
        labels: Sequence[int],
    ) -> List[Dict[str, torch.Tensor]]:
        answers = [
            {"feature": placeholder.clone(), "label": labels[0]},
            {"feature": placeholder.clone(), "label": labels[1]},
            {"feature": placeholder.clone(), "label": labels[2]},
        ]
        bbox_keys = [
            key for key in feature.keys() if isinstance(key, str) and "bbox" in key
        ]
        if not bbox_keys:
            return answers

        small_list = ["1", "2", "3"]
        for key in bbox_keys:
            parts = key.split("_")
            suffix = parts[-1] if parts else ""
            middle = parts[-2] if len(parts) >= 2 else ""
            tensor = torch.as_tensor(feature[key]).clone()

            if suffix == "1":
                answers[0]["feature"] = tensor
                if middle in small_list:
                    small_list.remove(middle)
                continue

            if small_list and middle == small_list[0]:
                answers[1]["feature"] = tensor
            else:
                answers[2]["feature"] = tensor

        return answers

    def _find_image_entry_by_index(
        self, image_features: Sequence[dict], target_index: int
    ) -> Optional[dict]:
        for feature in image_features:
            if not isinstance(feature, dict):
                continue
            idx = feature.get("image_index")
            if idx is None:
                idx = self._infer_image_index(feature.get("image_filename"))
            if idx is None:
                idx = self._infer_image_index(feature.get("image_path"))
            if idx is None:
                continue
            if int(idx) == target_index:
                return feature
        return None

    @staticmethod
    def _infer_image_index(name: Optional[str]) -> Optional[int]:
        if not name:
            return None
        stem = Path(name).stem
        match = re.search(r"(\d+)", stem)
        if not match:
            return None
        return int(match.group(1))

    def _assign_roi_features(
        self, feature_entry: dict, placeholder: torch.Tensor, labels: Sequence[int]
    ) -> List[Dict[str, torch.Tensor]]:
        answers = [
            {"feature": placeholder.clone(), "label": labels[0]},
            {"feature": placeholder.clone(), "label": labels[1]},
            {"feature": placeholder.clone(), "label": labels[2]},
        ]
        tensor = feature_entry.get("features")
        if tensor is None:
            return answers
        tens = torch.as_tensor(tensor).float()
        if tens.ndim == 1:
            tens = tens.unsqueeze(0)
        for idx in range(min(3, tens.shape[0])):
            answers[idx]["feature"] = tens[idx : idx + 1]
        return answers

    def _load_video_lookup(self, object_id: str) -> Tuple[Dict[str, torch.Tensor], torch.Size]:
        npy_path = self.clip_feature_dir / f"saved_features_{object_id}.npy"
        if npy_path.exists():
            return self._load_legacy_video_lookup(npy_path)

        pt_dir = self.clip_feature_dir / object_id
        if pt_dir.is_dir():
            return self._load_pt_video_lookup(object_id, pt_dir)

        default_shape = torch.Size((len(self.view_keys), self.video_embedding_dim))
        return {}, default_shape

    def _load_legacy_video_lookup(self, npy_path: Path) -> Tuple[Dict[str, torch.Tensor], torch.Size]:
        entries = np.load(npy_path, allow_pickle=True)
        lookup: Dict[str, torch.Tensor] = {}
        feature_shape: Optional[torch.Size] = None
        for entry in entries:
            feature = torch.as_tensor(entry["features"]).clone()
            feature_shape = feature.shape
            key = f"{entry['object_id']}:{str(entry['step_id'])}"
            lookup[key] = feature

        if feature_shape is None:
            feature_shape = torch.Size((len(self.view_keys), self.video_embedding_dim))
        return lookup, feature_shape

    def _load_pt_video_lookup(self, object_id: str, object_dir: Path) -> Tuple[Dict[str, torch.Tensor], torch.Size]:
        lookup: Dict[str, torch.Tensor] = {}
        feature_shape = torch.Size((len(self.view_keys), self.video_embedding_dim))

        for pt_path in sorted(object_dir.glob("*.pt")):
            entry = torch.load(pt_path, map_location="cpu")
            stacked = self._stack_view_tensors(entry)
            feature_shape = stacked.shape
            step_id = pt_path.stem
            key = f"{object_id}:{step_id}"
            lookup[key] = stacked

        return lookup, feature_shape

    def _stack_view_tensors(self, entry: Dict) -> torch.Tensor:
        rows: List[torch.Tensor] = []
        embed_dim = self.video_embedding_dim

        frames_per_view = int(entry.get("frames_per_view", 1))

        for key in self.view_keys:
            tensor = entry.get(key)
            if tensor is None:
                continue
            tens = torch.as_tensor(tensor).float()
            if tens.ndim == 1:
                tens = tens.unsqueeze(0)
            embed_dim = tens.shape[-1]
            break

        for key in self.view_keys:
            tensor = entry.get(key)
            if tensor is None:
                rows.append(torch.zeros((frames_per_view, embed_dim)))
                continue
            tens = torch.as_tensor(tensor).float()
            if tens.ndim == 1:
                tens = tens.unsqueeze(0)
            rows.append(tens)

        if not rows:
            rows = [torch.zeros((frames_per_view, embed_dim)) for _ in self.view_keys]
        self.video_embedding_dim = embed_dim
        return torch.cat(rows, dim=0)

    def _shuffle_entries(self, entries: List[Dict]) -> List[Dict]:
        if not self.shuffle_answers:
            return entries
        shuffled = entries[:]
        self.rng.shuffle(shuffled)
        return shuffled

    @staticmethod
    def _iter_excel_sheets(excel_path: Path) -> Iterable[tuple[int, pd.DataFrame]]:
        with pd.ExcelFile(excel_path) as workbook:
            for idx, sheet_name in enumerate(workbook.sheet_names, start=1):
                yield idx, workbook.parse(sheet_name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Combine QALIST multimodal features.")
    parser.add_argument("--root-dir", default="QALIST2106", help="Folder with object subdirs.")
    parser.add_argument(
        "--clip-feature-dir",
        default="clipfeatures0610",
        help="Folder containing saved_features_*.npy or per-object .pt files from build_video_features.py.",
    )
    parser.add_argument(
        "--sam-feature-dir", default="samfeatures", help="Folder containing SAM/Faster R-CNN tensors."
    )
    parser.add_argument("--clip-model", default="ViT-B/32")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--shuffle-images", action="store_true", default=True, help="Randomize image answer order per question.")
    parser.add_argument("--shuffle-seed", type=int, default=42, help="Seed used when shuffling image answers.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    combiner = FeatureCombiner(
        root_dir=Path(args.root_dir),
        clip_feature_dir=Path(args.clip_feature_dir),
        sam_feature_dir=Path(args.sam_feature_dir),
        clip_model=args.clip_model,
        device=args.device,
        shuffle_answers=args.shuffle_images,
        shuffle_seed=args.shuffle_seed,
    )
    combiner.run()
    print("Completed!")


if __name__ == "__main__":
    main()
