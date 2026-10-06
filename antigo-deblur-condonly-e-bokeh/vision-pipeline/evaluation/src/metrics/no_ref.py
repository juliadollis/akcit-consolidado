import pyiqa
import torch

from .base import ImageMetric


class PyIQAMetric(ImageMetric):
    """Generic class for pyiqa-based metrics (no-reference)."""

    def __init__(self, metric_name: str, device="cuda"):
        """
        Args:
            metric_name: One of 'clipiqa', 'maniqa', 'musiq', etc.
            device: Device to run the metric on.
        """
        self.device = torch.device(device)
        self.metric = pyiqa.create_metric(metric_name, device=self.device)

    def compute(self, img_gen: torch.Tensor, img_gt: torch.Tensor = None) -> float:
        """
        Compute the pyiqa metric.

        Args:
            img_gen: Tensor in [0, 1].
            img_gt: Ignored (no-reference metric).

        Returns:
            Metric score.
        """
        with torch.no_grad():
            score = self.metric(img_gen)

        return score.item()
