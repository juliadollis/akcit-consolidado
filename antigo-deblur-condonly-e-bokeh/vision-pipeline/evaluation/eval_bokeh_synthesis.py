import argparse
import io
import logging
import sys

import clip
import cv2
import numpy as np
import pandas as pd
import pyiqa
import torch
from datasets import Dataset, concatenate_datasets, load_dataset
from PIL import Image
from scipy.stats import pearsonr
from skimage.metrics import structural_similarity as ssim
from torchvision import transforms
from tqdm import tqdm


def setup_logging(verbose=False):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


CLIP_IMAGE_SIZE = 224
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
# CORRECAO (2026-08-26): era [0, 5, 10, 15], mas as imagens do sweep sao
# k01/k05/k10/k15 — o primeiro ponto e K=1, nao K=0. O eixo x da correlacao
# estava deslocado em relacao as imagens que ele correlaciona.
K_VALUES = [1, 5, 10, 15]
K_IMAGE_FIELDS = ["image_k01", "image_k05", "image_k10", "image_k15"]
# LEITURA DO SINAL: LVCorr = Pearson(K, variancia do Laplaciano). K maior pede
# MAIS desfoque, e mais desfoque REDUZ a variancia do Laplaciano. Entao um
# modelo que obedece o K tem LVCorr NEGATIVA. Valor perto de zero = o modelo
# ignora o K.


# CLIP preprocessing configuration
clip_preprocess = transforms.Compose(
    [
        transforms.Resize((CLIP_IMAGE_SIZE, CLIP_IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(CLIP_MEAN, CLIP_STD),
    ]
)


def get_pil_image(data):
    if hasattr(data, "convert"):
        return data
    elif isinstance(data, bytes):
        return Image.open(io.BytesIO(data))
    elif isinstance(data, dict):
        if "bytes" in data and data["bytes"] is not None:
            return Image.open(io.BytesIO(data["bytes"]))
        elif "path" in data and data["path"] is not None:
            return Image.open(data["path"])
    raise ValueError(f"Unrecognized image data format: {type(data)}")


def img_to_gray_cv2(pil_img):
    """Convert a PIL image to a grayscale OpenCV ndarray."""
    img_cv = cv2.cvtColor(np.array(pil_img.convert("RGB")), cv2.COLOR_RGB2BGR)
    return cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)


def calculate_laplacian_variance(pil_img):
    """Calculate Laplacian variance to estimate blur level."""
    gray_img = img_to_gray_cv2(pil_img)
    return cv2.Laplacian(gray_img, cv2.CV_64F).var()


# VARIANTE DE METRICA (2026-08-26): trocado `lpips` por `lpips+`.
# O avaliador de DEBLUR do mesmo repo (evaluation/src/deblur_evaluator.py) usa
# lpips+ / clipiqa+ / maniqa-kadid / musiq, e com essas variantes o peso OFICIAL
# do paper reproduziu os numeros publicados da Tab. 2 quase exatamente
# (LPIPS 0.2385 medido vs 0.2408 publicado). Ou seja: lpips+ e a variante que
# casa com o paper. O avaliador de BOKEH usava `lpips` simples, que tem escala
# diferente e tornava os numeros incomparaveis.
class CloudBokehEvaluator:
    """
    A reusable interface for running bokeh quality evaluations from HuggingFace
    datasets and pushing the results back to a HuggingFace dataset repository.
    """

    def __init__(self, device="cuda", verbose=False):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.logger = logging.getLogger(self.__class__.__name__)

        self.logger.info(f"Loading metric models on {self.device}...")
        self.lpips = pyiqa.create_metric("lpips+", device=self.device)
        self.dists = pyiqa.create_metric("dists", device=self.device)

        self.logger.info("Loading CLIP model...")
        self.clip_model, _ = clip.load("ViT-B/32", device=self.device)
        self.clip_model.eval()

        # To tensor transform for PyIQA
        self.to_tensor = transforms.ToTensor()

    def get_clip_embedding(self, pil_image):
        """Extract the visual embedding of an image with CLIP."""
        image_input = (
            clip_preprocess(pil_image.convert("RGB")).unsqueeze(0).to(self.device)
        )
        with torch.no_grad():
            image_features = self.clip_model.encode_image(image_input)
            image_features /= image_features.norm(dim=-1, keepdim=True)
        return image_features

    def evaluate(
        self, hf_dataset: str, hf_split: str = "test", hf_token: str = None
    ) -> dict:
        self.logger.info(f"Processing dataset: {hf_dataset} (split: {hf_split})")
        dataset = load_dataset(hf_dataset, split=hf_split, token=hf_token)

        num_items = len(dataset)
        if num_items == 0:
            self.logger.error(f"Dataset {hf_dataset} is empty.")
            return {}

        results = {
            "SSIM": 0.0,
            "LPIPS": 0.0,
            "DISTS": 0.0,
            "CLIP-I": 0.0,
            "LVCorr": 0.0,
        }

        valid_items = 0

        for row in tqdm(dataset, desc=f"Evaluating {hf_dataset}"):
            try:
                real_bokeh_image = get_pil_image(row["image_real_bokeh"]).convert("RGB")
                best_generated_image = get_pil_image(row["image_best_k"]).convert("RGB")

                real_bokeh_gray = img_to_gray_cv2(real_bokeh_image)
                best_generated_gray = img_to_gray_cv2(best_generated_image)

                if best_generated_gray.shape != real_bokeh_gray.shape:
                    best_generated_gray = cv2.resize(
                        best_generated_gray,
                        (real_bokeh_gray.shape[1], real_bokeh_gray.shape[0]),
                    )

                ssim_val = ssim(real_bokeh_gray, best_generated_gray)

                real_tensor = self.to_tensor(real_bokeh_image).unsqueeze(0).to(self.device)
                best_tensor = self.to_tensor(best_generated_image).unsqueeze(0).to(
                    self.device
                )

                if best_tensor.shape != real_tensor.shape:
                    import torch.nn.functional as F

                    best_tensor = F.interpolate(
                        best_tensor,
                        size=real_tensor.shape[2:],
                        mode="bicubic",
                        align_corners=False,
                    )
                    best_tensor = torch.clamp(best_tensor, 0, 1)

                with torch.no_grad():
                    lpips_val = self.lpips(best_tensor, real_tensor).item()
                    dists_val = self.dists(best_tensor, real_tensor).item()

                    best_embedding = self.get_clip_embedding(best_generated_image)
                    real_embedding = self.get_clip_embedding(real_bokeh_image)
                    clip_i_val = torch.cosine_similarity(
                        best_embedding, real_embedding
                    ).item()

                k_images = [get_pil_image(row[field]) for field in K_IMAGE_FIELDS]

                laplacian_variances = [
                    calculate_laplacian_variance(img) for img in k_images
                ]

                if np.std(laplacian_variances) == 0 or np.std(K_VALUES) == 0:
                    lv_corr = 0.0
                else:
                    lv_corr, _ = pearsonr(K_VALUES, laplacian_variances)

                results["SSIM"] += ssim_val
                results["LPIPS"] += lpips_val
                results["DISTS"] += dists_val
                results["CLIP-I"] += clip_i_val
                results["LVCorr"] += lv_corr

                valid_items += 1

            except Exception as e:
                item_id = row.get("image_id", "unknown")
                self.logger.error(f"Failed to process item {item_id}: {str(e)}")
                continue

        if valid_items > 0:
            for k in results.keys():
                results[k] /= valid_items
            self.logger.info(
                f"Final results: SSIM={results['SSIM']:.4f}, LPIPS={results['LPIPS']:.4f}, DISTS={results['DISTS']:.4f}, CLIP-I={results['CLIP-I']:.4f}, LVCorr={results['LVCorr']:.4f}"
            )
        else:
            self.logger.error("No valid items processed.")

        return results

    def save_results_to_hub(
        self,
        results: dict,
        dataset_name: str,
        output_repo_id: str,
        model_name: str = "Unknown",
        hf_token: str = None,
    ):
        if not results:
            self.logger.warning("No results to save.")
            return

        row = {"Model": model_name, "Dataset": dataset_name}
        for metric, val in results.items():
            row[metric] = float(f"{val:.4f}")

        new_ds = Dataset.from_pandas(pd.DataFrame([row]))

        try:
            self.logger.info(
                f"Attempting to load historical dataset from {output_repo_id}..."
            )
            existing_ds = load_dataset(
                output_repo_id,
                split="train",
                token=hf_token,
                download_mode="force_redownload",
            )
            combined_ds = concatenate_datasets([existing_ds, new_ds])
            self.logger.info("Historical dataset found. Appending new results.")
        except Exception as e:
            self.logger.info(
                f"Historical dataset not found or empty: {e}. Creating a new dataset."
            )
            combined_ds = new_ds

        self.logger.info(f"Pushing to {output_repo_id}...")
        combined_ds.push_to_hub(output_repo_id, token=hf_token)
        self.logger.info("Metrics updated successfully on Hugging Face Hub.")

    def run_pipeline(
        self,
        hf_datasets,
        output_repo_id: str,
        model_name: str = "Unknown",
        hf_split: str = "test",
        hf_token: str = None,
    ) -> list:
        if isinstance(hf_datasets, str):
            hf_datasets = [hf_datasets]

        all_results = []
        for dataset_name in hf_datasets:
            try:
                results = self.evaluate(
                    dataset_name, hf_split=hf_split, hf_token=hf_token
                )
                self.save_results_to_hub(
                    results,
                    dataset_name,
                    output_repo_id,
                    model_name=model_name,
                    hf_token=hf_token,
                )
                self.logger.info(f"Pipeline completed for {dataset_name}.")
                all_results.append(
                    {"model": model_name, "dataset": dataset_name, "results": results}
                )
            except Exception as e:
                self.logger.error(
                    f"Pipeline failed for dataset {dataset_name}: {str(e)}"
                )

        return all_results


def main():
    parser = argparse.ArgumentParser(
        description="Cloud evaluation for Bokeh Synthesis (Generative Refocusing)"
    )
    parser.add_argument(
        "--hf_dataset",
        type=str,
        nargs="+",
        required=True,
        help="Name of the Hugging Face dataset(s)",
    )
    parser.add_argument(
        "--model_name", type=str, required=True, help="Name of the evaluated model"
    )
    parser.add_argument("--hf_split", type=str, default="test", help="Dataset split")
    parser.add_argument(
        "--device", type=str, default="cuda", help="Execution device (cuda/cpu)"
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Enable verbose logs"
    )
    parser.add_argument(
        "--output_hf_repo",
        type=str,
        required=True,
        help="Destination HF repo for metrics",
    )
    parser.add_argument(
        "--hf_token", type=str, default=None, help="HF token for private datasets"
    )
    args = parser.parse_args()

    setup_logging(args.verbose)

    evaluator = CloudBokehEvaluator(device=args.device, verbose=args.verbose)

    evaluator.run_pipeline(
        hf_datasets=args.hf_dataset,
        output_repo_id=args.output_hf_repo,
        model_name=args.model_name,
        hf_split=args.hf_split,
        hf_token=args.hf_token,
    )


if __name__ == "__main__":
    main()
