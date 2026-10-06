import logging

import pyiqa
import torch

from .base import ImageMetric

logger = logging.getLogger(__name__)


class LPIPSMetric(ImageMetric):
    """LPIPS (Learned Perceptual Image Patch Similarity) metric using pyiqa implementation."""

    def __init__(self, net="lpips", device="cuda"):
        self.device = torch.device(device)
        # pyiqa supports 'lpips', 'lpips+', 'lpips-vgg', etc.
        self.metric = pyiqa.create_metric(net, device=self.device)

    def compute(self, img_gen: torch.Tensor, img_gt: torch.Tensor = None) -> float:
        """
        Compute LPIPS.
        """
        if img_gt is None:
            raise ValueError("LPIPS requires a ground truth image.")

        # pyiqa metrics generally expect [0, 1] range and handle internal normalization
        with torch.no_grad():
            score = self.metric(img_gen, img_gt)

        return score.item()
