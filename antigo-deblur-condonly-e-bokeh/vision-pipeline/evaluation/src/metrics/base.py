from abc import ABC, abstractmethod

import torch


class BaseMetric(ABC):
    """Base abstract class for all metrics."""

    pass


class ImageMetric(BaseMetric):
    """Abstract class for metrics that evaluate images individually."""

    @abstractmethod
    def compute(self, img_gen: torch.Tensor, img_gt: torch.Tensor = None) -> float:
        """
        Compute the metric for a generated image.

        Args:
            img_gen: Generated image tensor (B, C, H, W).
            img_gt: Ground truth image tensor (B, C, H, W), optional for no-ref metrics.
        """
        pass


class DatasetMetric(BaseMetric):
    """Abstract class for metrics that evaluate image distributions."""

    @abstractmethod
    def compute(self, dir_gen: str, dir_gt: str) -> float:
        """
        Compute the metric for two directories of images.

        Args:
            dir_gen: Path to directory with generated images.
            dir_gt: Path to directory with ground truth images.
        """
        pass
