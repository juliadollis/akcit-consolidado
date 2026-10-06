import logging
from concurrent.futures import ThreadPoolExecutor

import torch
import torch.nn.functional as F
from torchvision import transforms
from tqdm import tqdm

from ..metrics.base import DatasetMetric, ImageMetric
from ..utils.image_io import get_pil_image

# Configure logger
logger = logging.getLogger(__name__)


class PipelineEvaluator:
    """Orchestrator for the image quality evaluation pipeline with parallel execution."""

    def __init__(self, device="cuda", num_threads=4):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.image_metrics = {}
        self.dataset_metrics = {}
        self.num_threads = num_threads
        logger.info(
            f"Initialized PipelineEvaluator on {self.device} with {num_threads} parallel threads."
        )

    def register_metric(self, name: str, metric_instance):
        """Registers a metric instance."""
        if isinstance(metric_instance, ImageMetric):
            self.image_metrics[name] = metric_instance
            logger.info(f"Registered ImageMetric: {name}")
        elif isinstance(metric_instance, DatasetMetric):
            self.dataset_metrics[name] = metric_instance
            logger.info(f"Registered DatasetMetric: {name}")
        else:
            raise TypeError(
                "Metric must be an instance of ImageMetric or DatasetMetric"
            )

    def _compute_single_metric(self, name, metric, img_gen, img_gt):
        """Helper for parallel execution of a single metric on a single pair."""
        try:
            return name, metric.compute(img_gen, img_gt)
        except Exception as e:
            logger.exception(f"Error in metric {name}: {str(e)}")
            return name, 0.0

    def evaluate_hf_dataset(self, dataset, dataset_display_name=None):
        """
        Executes the evaluation pipeline directly on a HuggingFace Dataset.
        Expected columns: 'image_generated' and 'image_focus' containing PIL Images.
        """
        results = {}
        display_name = dataset_display_name or "HF_Dataset"
        logger.info(f">>> Starting evaluation for: {display_name}")

        accumulated_scores = dict.fromkeys(self.image_metrics, 0.0)
        num_pairs = len(dataset)

        if num_pairs == 0:
            logger.error(f"Dataset {display_name} is empty.")
            return results

        transform = transforms.Compose([transforms.ToTensor()])

        for row in tqdm(dataset, desc=f"Evaluating {display_name}"):
            try:
                # Get PIL images robustly
                img_gen_pil = get_pil_image(row["image_generated"]).convert("RGB")
                img_gt_pil = get_pil_image(row["image_focus"]).convert("RGB")

                # Convert to tensors
                img_gen = transform(img_gen_pil).unsqueeze(0).to(self.device)
                img_gt = transform(img_gt_pil).unsqueeze(0).to(self.device)

                # Resize if shapes don't match
                if img_gen.shape != img_gt.shape:
                    gen_size = img_gen.shape[2] * img_gen.shape[3]
                    gt_size = img_gt.shape[2] * img_gt.shape[3]
                    if gt_size > gen_size:
                        img_gt = F.interpolate(
                            img_gt,
                            size=img_gen.shape[2:],
                            mode="bicubic",
                            align_corners=False,
                        )
                        img_gt = torch.clamp(img_gt, 0, 1)
                    else:
                        img_gen = F.interpolate(
                            img_gen,
                            size=img_gt.shape[2:],
                            mode="bicubic",
                            align_corners=False,
                        )
                        img_gen = torch.clamp(img_gen, 0, 1)

                with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
                    futures = [
                        executor.submit(
                            self._compute_single_metric, name, metric, img_gen, img_gt
                        )
                        for name, metric in self.image_metrics.items()
                    ]
                    for future in futures:
                        name, score = future.result()
                        accumulated_scores[name] += score

            except Exception as e:
                # Use file_name_base if available for better logging
                item_id = row.get("file_name_base", "unknown")
                logger.exception(f"Failed pair {item_id}: {str(e)}")
                continue

        for name, total_score in accumulated_scores.items():
            results[name] = total_score / num_pairs
            logger.info(f"{name}: {results[name]:.4f}")

        return results
