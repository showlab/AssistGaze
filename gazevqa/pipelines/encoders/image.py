"""
Region-of-interest (ROI) feature extractor built on Faster R-CNN.
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.ops import MultiScaleRoIAlign
import torchvision.transforms as T


class RoiFeatureExtractor:
    """
    Uses a pretrained Faster R-CNN backbone/head to embed user-specified boxes.
    """

    def __init__(self, device: str = "cuda:0", min_size: int = 800) -> None:
        self.device = torch.device(device)
        self.detector = fasterrcnn_resnet50_fpn(weights="DEFAULT").to(self.device).eval()
        self.transform = T.Compose([T.ToTensor(), T.ConvertImageDtype(torch.float32)])
        self.roi_pool: MultiScaleRoIAlign = self.detector.roi_heads.box_roi_pool
        self.box_head = self.detector.roi_heads.box_head
        self.min_size = min_size

    @torch.no_grad()
    def encode(self, image_path: str, boxes_xyxy: Sequence[Sequence[float]]) -> torch.Tensor:
        """Extract pooled ROI descriptors for the provided bounding boxes."""
        if len(boxes_xyxy) == 0:
            return torch.zeros((0, 1024))

        image = Image.open(image_path).convert("RGB")
        tensor = self.transform(image).to(self.device)

        _, h, w = tensor.shape
        scale = self.min_size / min(h, w)
        if scale < 1:
            size = (int(h * scale), int(w * scale))
            tensor = F.interpolate(tensor.unsqueeze(0), size=size, mode="bilinear", align_corners=False)[0]

        features = self.detector.backbone(tensor.unsqueeze(0))
        if isinstance(features, torch.Tensor):
            features = {"0": features}

        boxes = torch.as_tensor(boxes_xyxy, dtype=torch.float32, device=self.device)
        roi_features = self.roi_pool(
            features,
            [boxes],
            [(tensor.shape[1], tensor.shape[2])],
        )
        roi_features = self.box_head(roi_features)
        return F.normalize(roi_features, dim=-1).cpu()

    def __call__(self, image_path: str, boxes_xyxy):
        return self.encode(image_path, boxes_xyxy)
