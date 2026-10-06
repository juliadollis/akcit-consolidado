import logging
import sys
from datetime import datetime

import pandas as pd
import torch
from datasets import load_dataset
from huggingface_hub import HfApi, hf_hub_download

from .core.evaluator import PipelineEvaluator
from .metrics.full_ref import LPIPSMetric
from .metrics.no_ref import PyIQAMetric


def setup_logging(verbose=False):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


class CloudDeblurEvaluator:
    """
    A reusable interface for running image quality evaluations from HuggingFace
    datasets and pushing the results back to a HuggingFace dataset repository.
    """

    def __init__(self, device="cuda", threads=2, verbose=False):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.threads = threads
        self.logger = logging.getLogger(self.__class__.__name__)

        self.evaluator = PipelineEvaluator(device=self.device, num_threads=self.threads)
        self._register_default_metrics()

    def _register_default_metrics(self):
        """Registers the default IQA metrics."""
        self.evaluator.register_metric(
            "LPIPS", LPIPSMetric(device=self.device, net="lpips+")
        )
        self.evaluator.register_metric(
            "DISTS", LPIPSMetric(device=self.device, net="dists")
        )
        self.evaluator.register_metric(
            "CLIP-IQA", PyIQAMetric("clipiqa+", device=self.device)
        )
        self.evaluator.register_metric(
            "MANIQA", PyIQAMetric("maniqa-kadid", device=self.device)
        )
        self.evaluator.register_metric(
            "MUSIQ", PyIQAMetric("musiq", device=self.device)
        )

    def evaluate(
        self,
        hf_dataset: str,
        hf_split: str = "test",
        experiment_name: str = None,
        hf_token: str = None,
    ) -> dict:
        """
        Downloads a dataset from HuggingFace and evaluates its images.

        When ``experiment_name`` is given, only that experiment's subfolder
        (``{experiment_name}/data/*.parquet``) is loaded, so multiple
        experiments can coexist in the same repo without mixing.
        """
        self.logger.info(
            f"Processing dataset: {hf_dataset} "
            f"(split: {hf_split}, experiment: {experiment_name})"
        )
        if experiment_name:
            dataset = load_dataset(
                hf_dataset,
                data_files={hf_split: f"{experiment_name}/data/*.parquet"},
                split=hf_split,
                token=hf_token,
            )
        else:
            dataset = load_dataset(hf_dataset, split=hf_split, token=hf_token)

        with torch.no_grad():
            results = self.evaluator.evaluate_hf_dataset(
                dataset, dataset_display_name=hf_dataset
            )
        return results

    RESULTS_FILENAME = "results.csv"

    def save_results_to_hub(
        self,
        results: dict,
        dataset_name: str,
        output_repo_id: str,
        model_name: str = "Unknown",
        experiment_name: str = "Unknown",
        hf_token: str = None,
    ):
        """Append one result row to a single CSV on the Hub (``results.csv``),
        creating it if absent. Never overwrites past rows."""
        fieldnames = ["LPIPS", "DISTS", "CLIP-IQA", "MANIQA", "MUSIQ"]

        row = {
            "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Experiment": experiment_name,
            "Model": model_name,
            "Dataset": dataset_name,
        }
        for metric in fieldnames:
            row[metric] = float(f"{results.get(metric, 0.0):.4f}")

        new_row_df = pd.DataFrame([row])
        api = HfApi(token=hf_token)

        # Missing file -> create new; download error on an existing file
        # propagates so history is never overwritten with a partial read.
        existing_df = None
        if api.file_exists(
            repo_id=output_repo_id,
            filename=self.RESULTS_FILENAME,
            repo_type="dataset",
        ):
            self.logger.info(
                f"Fetching existing {self.RESULTS_FILENAME} from {output_repo_id}..."
            )
            local_path = hf_hub_download(
                repo_id=output_repo_id,
                filename=self.RESULTS_FILENAME,
                repo_type="dataset",
                token=hf_token,
                force_download=True,
            )
            existing_df = pd.read_csv(local_path)
        else:
            self.logger.info(
                f"No existing {self.RESULTS_FILENAME} at {output_repo_id}. "
                "Creating a new file."
            )

        if existing_df is not None:
            combined_df = pd.concat([existing_df, new_row_df], ignore_index=True)
            self.logger.info(
                f"Existing file found ({len(existing_df)} rows). Appending 1 new row."
            )
        else:
            combined_df = new_row_df

        # Upload from memory (no temp file, no path collision).
        buffer = combined_df.to_csv(index=False).encode("utf-8")

        api.create_repo(repo_id=output_repo_id, repo_type="dataset", exist_ok=True)
        self.logger.info(
            f"Uploading {len(combined_df)} total rows to "
            f"{output_repo_id}/{self.RESULTS_FILENAME}..."
        )
        api.upload_file(
            path_or_fileobj=buffer,
            path_in_repo=self.RESULTS_FILENAME,
            repo_id=output_repo_id,
            repo_type="dataset",
            commit_message=f"Append results: {model_name} on {dataset_name}",
        )
        self.logger.info("Successfully pushed results to Hugging Face Hub.")

    def run_pipeline(
        self,
        hf_datasets,
        output_repo_id: str,
        model_name: str = "Unknown",
        experiment_name: str = "Unknown",
        hf_split: str = "test",
        hf_token: str = None,
    ) -> list:
        """
        Executes the full pipeline: Evaluates one or more datasets and pushes results to the hub.
        """
        if isinstance(hf_datasets, str):
            hf_datasets = [hf_datasets]

        all_results = []
        for dataset_name in hf_datasets:
            try:
                results = self.evaluate(
                    dataset_name,
                    hf_split=hf_split,
                    experiment_name=experiment_name,
                    hf_token=hf_token,
                )
                self.save_results_to_hub(
                    results,
                    dataset_name,
                    output_repo_id,
                    model_name=model_name,
                    experiment_name=experiment_name,
                    hf_token=hf_token,
                )
                self.logger.info(f"Pipeline execution completed for {dataset_name}.")
                all_results.append(
                    {"model": model_name, "dataset": dataset_name, "results": results}
                )
            except Exception as e:
                self.logger.error(
                    f"Failed pipeline for dataset {dataset_name}: {str(e)}"
                )

        return all_results
