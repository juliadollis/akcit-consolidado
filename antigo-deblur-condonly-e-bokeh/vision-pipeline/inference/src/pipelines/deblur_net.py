import argparse
import io
import os

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from datasets import load_dataset
from diffusers import FluxPipeline
from dotenv import load_dotenv
from Genfocus.pipeline.flux import Condition, generate, seed_everything
from huggingface_hub import HfApi, list_repo_files
from PIL import Image
from tqdm import tqdm

from evaluation import CloudDeblurEvaluator
from src.utils.image import load_image_from_row, resize_and_pad_image

MODEL_ID = "black-forest-labs/FLUX.1-dev"
OFFICIAL_WEIGHTS_DIR = "pesos_oficiais"
DEFAULT_LORA_PATH = f"{OFFICIAL_WEIGHTS_DIR}/deblurNet.safetensors"
DATASET_REPO_TYPE = "dataset"
VALIDATION_SPLIT = "validation"
HF_DATASET_PREFIX = "hf://datasets"
TEMP_PROCESSING_PARQUET = "temp_batch_processing.parquet"
DEFAULT_BATCH_SIZE = 50
DEFAULT_STEPS = 28
DEFAULT_LONG_SIDE = 0
MIN_TILED_DENOISE_SIDE = 512
SEED_VALUE = 42
DATASET_INPUT_REPO_KEY = "input_repo"
DATASET_OUTPUT_REPO_KEY = "output_repo"
DATASET_VALIDATION_PATTERN_KEY = "validation_pattern"
EVALUATION_OUTPUT_REPO_ENV_KEY = "DEBLUR_EVAL_OUTPUT_REPO"
MODEL_NAME_ENV_KEY = "DEBLUR_MODEL_NAME"
DEFAULT_MODEL_NAME = "DeblurNet"
DEFAULT_EVALUATION_THREADS = 2

BLUR_IMAGE_COLUMN = "image_blur"
FOCUS_IMAGE_COLUMN = "image_focus"
BASE_NAME_COLUMN = "file_name_base"
GENERATED_IMAGE_COLUMN = "image_generated"

HF_PARQUET_METADATA = {
    b"huggingface": b'{"info": {"features": {"file_name_base": {"dtype": "string", "_type": "Value"}, "image_generated": {"_type": "Image"}, "image_focus": {"_type": "Image"}}}}'
}


def convert_pil_to_bytes(img: Image.Image) -> bytes:
    """
    Convert a PIL image into PNG bytes for Parquet storage.
    """
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return buffer.getvalue()


# main_adapter: o `generate` do Genfocus monta adapters = [main_adapter]*2 +
# c_adapters, e o DEFAULT e None (= LoRA so nas condicoes, "cond-only"). Isso e
# o certo para o peso OFICIAL do paper. Mas o NOSSO DeblurNet foi treinado na
# variante main+cond: com main_adapter=None a saida sai LAVADA (LPIPS ~0.85,
# ver secao 6 do handoff). Sem este parametro, a comparacao nosso-vs-oficial
# seria invalida — foi exatamente o tipo de erro que estragou a avaliacao de
# bokeh. Passe MAIN_ADAPTER=deblurring para o nosso peso.
def run_hf_batch_inference(
    input_repo_id: str,
    validation_pattern: str,
    output_repo_id: str,
    experiment_name: str,
    batch_size: int = DEFAULT_BATCH_SIZE,
    lora_path: str = DEFAULT_LORA_PATH,
    main_adapter: str | None = None,
    hf_token: str = None,
) -> bool:
    """
    Run FLUX deblurring on a Hugging Face validation split and upload Parquet batches.
    """
    steps = DEFAULT_STEPS
    long_side = DEFAULT_LONG_SIDE

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    print(f"Detected device: {device} | Primary dtype: {dtype}")

    if not os.path.exists(lora_path):
        print(f"[ERROR] LoRA weights file not found at: {lora_path}")
        return False

    lora_dir = os.path.dirname(lora_path)
    lora_weight_name = os.path.basename(lora_path)

    print("Checking existing files in destination repository...")
    api = HfApi()
    try:
        api.create_repo(
            repo_id=output_repo_id,
            repo_type=DATASET_REPO_TYPE,
            exist_ok=True,
            token=hf_token,
        )
        repo_files_on_hub = list_repo_files(
            repo_id=output_repo_id,
            repo_type=DATASET_REPO_TYPE,
            token=hf_token,
        )
    except Exception as exc:
        print(f"[WARNING] Could not inspect existing files: {exc}")
        repo_files_on_hub = []

    print("Loading full FLUX pipeline...")
    pipe_flux = FluxPipeline.from_pretrained(
        MODEL_ID, torch_dtype=dtype, token=hf_token
    )

    if device == "cuda":
        pipe_flux.to("cuda")
        if hasattr(pipe_flux, "vae"):
            pipe_flux.vae.enable_tiling()

    print("Loading Deblur LoRA weights...")
    try:
        pipe_flux.load_lora_weights(
            lora_dir, weight_name=lora_weight_name, adapter_name="deblurring"
        )
        pipe_flux.set_adapters(["deblurring"])
    except Exception as exc:
        print(f"[ERROR] Failed to load Deblur LoRA: {exc}")
        return False

    print(f"Loading validation split from source repository: {input_repo_id}...")
    try:
        direct_path = f"{HF_DATASET_PREFIX}/{input_repo_id}/{validation_pattern}"
        hf_dataset = load_dataset(
            "parquet",
            data_files={VALIDATION_SPLIT: direct_path},
            token=hf_token,
        )
        validation_rows = hf_dataset[VALIDATION_SPLIT]
    except Exception as exc:
        print(f"[ERROR] Failed to load Hugging Face dataset: {exc}")
        return False

    total_images = len(validation_rows)
    print(f"Total images found in source dataset: {total_images}")

    output_rows = []

    for index, row in enumerate(
        tqdm(validation_rows, desc="Evaluating and Generating Batches")
    ):
        current_batch_number = (index // batch_size) + 1
        batch_parquet_name = (
            f"{experiment_name}/data/validation_part_{current_batch_number:03d}.parquet"
        )

        if batch_parquet_name in repo_files_on_hub:
            continue

        file_name = row.get(BASE_NAME_COLUMN) or f"val_{index:05d}.png"

        try:
            blur_image = load_image_from_row(row[BLUR_IMAGE_COLUMN]).convert("RGB")
            processed_input = resize_and_pad_image(blur_image, long_side)
            width, height = processed_input.size

            focus_image = load_image_from_row(row[FOCUS_IMAGE_COLUMN]).convert("RGB")
            focus_image = resize_and_pad_image(focus_image, long_side)

            condition = Condition(processed_input, "deblurring", [0, 0], 1.0)
            seed_everything(SEED_VALUE)

            with torch.no_grad():
                pipeline_output = generate(
                    pipeline=pipe_flux,
                    height=height,
                    width=width,
                    prompt="a sharp photo with everything in focus",
                    num_inference_steps=steps,
                    conditions=[condition],
                    main_adapter=main_adapter,
                    NO_TILED_DENOISE=min(width, height) < MIN_TILED_DENOISE_SIDE,
                )
                generated_image = pipeline_output.images[0]

            generated_image_bytes = convert_pil_to_bytes(generated_image)
            focus_image_bytes = convert_pil_to_bytes(focus_image)

            output_rows.append(
                {
                    BASE_NAME_COLUMN: str(file_name),
                    GENERATED_IMAGE_COLUMN: generated_image_bytes,
                    FOCUS_IMAGE_COLUMN: focus_image_bytes,
                }
            )

            is_batch_full = len(output_rows) == batch_size
            is_last_row = (index + 1) == total_images
            if is_batch_full or is_last_row:
                print(
                    f"\n[Batch {current_batch_number}] Building and uploading Parquet to the Hub..."
                )

                output_df = pd.DataFrame(output_rows)
                schema = pa.schema(
                    [
                        pa.field(BASE_NAME_COLUMN, pa.string()),
                        pa.field(GENERATED_IMAGE_COLUMN, pa.binary()),
                        pa.field(FOCUS_IMAGE_COLUMN, pa.binary()),
                    ]
                )
                arrow_table = pa.Table.from_pandas(
                    output_df,
                    schema=schema.with_metadata(HF_PARQUET_METADATA),
                    preserve_index=False,
                )

                pq.write_table(arrow_table, TEMP_PROCESSING_PARQUET)

                api.upload_file(
                    path_or_fileobj=TEMP_PROCESSING_PARQUET,
                    path_in_repo=batch_parquet_name,
                    repo_id=output_repo_id,
                    repo_type=DATASET_REPO_TYPE,
                    token=hf_token,
                )
                print(
                    f"Batch {current_batch_number} saved successfully to the remote Hub."
                )

                output_rows = []
                if os.path.exists(TEMP_PROCESSING_PARQUET):
                    os.remove(TEMP_PROCESSING_PARQUET)

        except Exception as exc:
            print(f"\n[ERROR] Critical failure at index {index} ({file_name}): {exc}")
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    print(
        "\nEntire validation dataset processed and stored successfully on Hugging Face Hub."
    )
    del pipe_flux
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return True


def run_hf_deblur_evaluation(
    generated_repo_id: str,
    evaluation_repo_id: str,
    model_name: str,
    experiment_name: str,
    hf_token: str = None,
) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    evaluator = CloudDeblurEvaluator(
        device=device,
        threads=DEFAULT_EVALUATION_THREADS,
    )
    evaluator.run_pipeline(
        hf_datasets=[generated_repo_id],
        output_repo_id=evaluation_repo_id,
        model_name=model_name,
        experiment_name=experiment_name,
        hf_split=VALIDATION_SPLIT,
        hf_token=hf_token,
    )
    del evaluator
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main():
    load_dotenv()

    parser = argparse.ArgumentParser(description="DeblurNet inference + evaluation")
    parser.add_argument(
        "--experiment_name",
        type=str,
        required=True,
        help=(
            "Name of the experiment. Used as the subfolder for generated images "
            "in the output repo and recorded in the results.csv row."
        ),
    )
    args = parser.parse_args()

    evaluation_repo_id = os.getenv(EVALUATION_OUTPUT_REPO_ENV_KEY)
    if not evaluation_repo_id:
        print(
            f"[ERROR] Missing {EVALUATION_OUTPUT_REPO_ENV_KEY}. Deblur evaluation must run after inference."
        )
        return

    model_name = os.getenv(MODEL_NAME_ENV_KEY, DEFAULT_MODEL_NAME)

    realdof = {
        DATASET_INPUT_REPO_KEY: "akcit-pixel/RealDOF",
        DATASET_OUTPUT_REPO_KEY: "AkcitPixel2/REALDOF_INFER",
        DATASET_VALIDATION_PATTERN_KEY: "data/validation-*.parquet",
    }
    ddpd = {
        DATASET_INPUT_REPO_KEY: "akcit-pixel/DDPD",
        DATASET_OUTPUT_REPO_KEY: "AkcitPixel2/DDPD_INFER",
        DATASET_VALIDATION_PATTERN_KEY: "data/validation-*.parquet",
    }
    datasets = [realdof, ddpd]

    hf_token = os.getenv("HF_TOKEN")

    for dataset_config in datasets:
        print(
            f"\nStarting dataset processing: {dataset_config[DATASET_INPUT_REPO_KEY]}"
        )
        inference_succeeded = run_hf_batch_inference(
            input_repo_id=dataset_config[DATASET_INPUT_REPO_KEY],
            validation_pattern=dataset_config[DATASET_VALIDATION_PATTERN_KEY],
            output_repo_id=dataset_config[DATASET_OUTPUT_REPO_KEY],
            experiment_name=args.experiment_name,
            batch_size=DEFAULT_BATCH_SIZE,
            lora_path=DEFAULT_LORA_PATH,
            hf_token=hf_token,
        )

        if not inference_succeeded:
            print(
                f"[WARNING] Skipping evaluation for {dataset_config[DATASET_OUTPUT_REPO_KEY]} because inference failed."
            )
            continue

        print(
            f"\nStarting evaluation for generated dataset: {dataset_config[DATASET_OUTPUT_REPO_KEY]}"
        )
        run_hf_deblur_evaluation(
            generated_repo_id=dataset_config[DATASET_OUTPUT_REPO_KEY],
            evaluation_repo_id=evaluation_repo_id,
            model_name=model_name,
            experiment_name=args.experiment_name,
            hf_token=hf_token,
        )


if __name__ == "__main__":
    main()
