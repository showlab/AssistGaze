"""
Utilities for extracting CLIP embeddings from video clips.
"""

from __future__ import annotations

from typing import List

import clip
import numpy as np
from decord import VideoReader, cpu as decord_cpu
from PIL import Image
import torch
import torch.nn.functional as F


class VideoClipEncoder:
    """
    Encodes uniformly sampled video frames with a CLIP image tower.

    Parameters
    ----------
    model_name: str
        CLIP checkpoint identifier.
    device: str
        Torch device string such as "cuda:0".
    num_frames: int
        Number of frames to sample uniformly across the clip.
    batch_size: int
        Frame batch size for CLIP inference (helps when VRAM is tight).
    """

    def __init__(
        self,
        model_name: str = "ViT-B/32",
        device: str = "cuda:0",
        num_frames: int = 8,
        batch_size: int = 4,
    ) -> None:
        self.device = torch.device(device)
        self.model, self.preprocess = clip.load(model_name, device=self.device, jit=False)
        self.num_frames = num_frames
        self.batch_size = batch_size
        visual = getattr(self.model, "visual", None)
        self.embedding_dim = getattr(visual, "output_dim", 512)

    @torch.no_grad()
    def encode(self, video_path: str, aggregate: str = "mean") -> torch.Tensor:
        """
        Return CLIP embeddings for `video_path`.

        aggregate: "mean" (default) returns a single 512-d vector.
                   "stack" returns all frame embeddings with shape (num_frames, 512).
        """
        aggregate = aggregate or "mean"
        vr = VideoReader(video_path, ctx=decord_cpu(0))
        total_frames = len(vr)
        if total_frames == 0:
            if aggregate == "stack":
                return torch.zeros((self.num_frames, self.embedding_dim))
            return torch.zeros((self.embedding_dim,))

        frame_indices = np.linspace(0, max(total_frames - 1, 0), self.num_frames).astype(np.int64)

        frames: List[Image.Image] = [Image.fromarray(vr[int(i)].asnumpy()) for i in frame_indices]
        embeddings: List[torch.Tensor] = []
        for start in range(0, len(frames), self.batch_size):
            batch = frames[start : start + self.batch_size]
            tensor = torch.stack([self.preprocess(frame) for frame in batch]).to(self.device)
            encoded = self.model.encode_image(tensor)
            embeddings.append(F.normalize(encoded, dim=-1))

        stacked = torch.cat(embeddings, dim=0)
        if aggregate == "stack":
            return stacked.cpu()
        return stacked.mean(dim=0).cpu()

    def __call__(self, video_path: str) -> torch.Tensor:
        return self.encode(video_path)
