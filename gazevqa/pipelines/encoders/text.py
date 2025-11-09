"""
CLIP-based text encoder with simple caching to avoid redundant GPU work.
"""

from __future__ import annotations

from typing import Dict, Optional

import clip
import torch


class TextEncoder:
    """
    Thin wrapper around OpenAI CLIP for encoding natural-language strings.

    Parameters
    ----------
    model_name: str
        CLIP checkpoint identifier (e.g., "ViT-B/32").
    device: str
        Torch device string such as "cuda:0" or "cpu".
    empty_token: str
        Sentinel value treated as a missing answer.
    template: Optional[str]
        Optional prompt template. Use `{text}` as placeholder for the raw string.
    """

    def __init__(
        self,
        model_name: str = "ViT-B/32",
        device: str = "cuda:0",
        empty_token: str = "X",
        template: Optional[str] = "Question: {text}",
    ) -> None:
        self.device = torch.device(device)
        self.model, _ = clip.load(model_name, device=self.device, jit=False)
        self.empty_token = empty_token.upper()
        self.template = template
        self.cache: Dict[str, torch.Tensor] = {}

    def encode(self, text: str) -> torch.Tensor:
        """Return a CPU tensor containing the CLIP embedding for `text`."""
        if not isinstance(text, str) or text.strip().upper() == self.empty_token:
            return torch.zeros((1, 512))

        cached = self.cache.get(text)
        if cached is not None:
            return cached

        prompt = self.template.format(text=text) if self.template else text
        tokens = clip.tokenize([prompt]).to(self.device)
        with torch.no_grad():
            embedding = self.model.encode_text(tokens).cpu()
        self.cache[text] = embedding
        return embedding

    def __call__(self, text: str) -> torch.Tensor:
        return self.encode(text)
