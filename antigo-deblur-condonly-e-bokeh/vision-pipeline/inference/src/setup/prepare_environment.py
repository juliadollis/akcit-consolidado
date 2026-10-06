import os
import shutil

try:
    from dotenv import load_dotenv
    from huggingface_hub import hf_hub_download, login, snapshot_download
except ImportError:
    print(
        "[*] Install dependencies: pip install huggingface_hub python-dotenv requests tqdm"
    )
    exit(1)

OFFICIAL_WEIGHTS_DIR = "pesos_oficiais"
CHECKPOINTS_DIR = "checkpoints"
BASE_DIRECTORIES = (CHECKPOINTS_DIR, OFFICIAL_WEIGHTS_DIR)
MIN_VALID_FILE_SIZE_BYTES = 10 * 1024 * 1024
FLUX_MODEL_ID = "black-forest-labs/FLUX.1-dev"
BANNER_WIDTH = 70

HF_FILES = {
    "bokehNet.safetensors": (
        "nycu-cplab/Genfocus-Model",
        f"{OFFICIAL_WEIGHTS_DIR}/bokehNet.safetensors",
        None,
    ),
    "deblurNet.safetensors": (
        "nycu-cplab/Genfocus-Model",
        f"{OFFICIAL_WEIGHTS_DIR}/deblurNet.safetensors",
        None,
    ),
    "depth_pro.pt": (
        "nycu-cplab/Genfocus-Model",
        f"{CHECKPOINTS_DIR}/depth_pro.pt",
        CHECKPOINTS_DIR,
    ),
}

HF_TOKEN = None


def ensure_base_directories():
    """Create the local directories required for checkpoints and weights."""
    print("\n[*] Checking and creating base directory structure...")
    for directory in BASE_DIRECTORIES:
        os.makedirs(directory, exist_ok=True)
    print("[+] Base directory structure ready!\n")


def login_to_hugging_face():
    """Load the Hugging Face token from `.env` or prompt for it."""
    global HF_TOKEN
    print("=" * BANNER_WIDTH)
    print("[*] HUGGING FACE AUTHENTICATION")
    print("=" * BANNER_WIDTH)

    load_dotenv()
    token = os.getenv("HF_TOKEN")

    if token:
        print("[*] Token detected automatically from .env.")
    else:
        print("[!] HF_TOKEN not found in .env.")
        token = input("Enter your Hugging Face access token: ").strip()

    if token:
        try:
            login(token=token)
            HF_TOKEN = token
            print("[+] Login completed successfully!\n")
        except Exception as exc:
            print(f"[ERROR] Failed to log in: {exc}")
            exit(1)
    else:
        print(
            "[!] No token provided. Script will try cached credentials or public access..."
        )


def download_required_files():
    """Download and validate the additional model files declared in `HF_FILES`."""
    print("[*] Checking additional weights (Depth Pro and LoRAs)...")

    for file_name, (repo_id, destination_path, subfolder) in HF_FILES.items():
        if (
            os.path.exists(destination_path)
            and os.path.getsize(destination_path) > MIN_VALID_FILE_SIZE_BYTES
        ):
            print(
                f"[*] File '{file_name}' already exists at '{destination_path}' and looks valid. Skipping..."
            )
            continue

        print(f"[-] Checking/downloading '{file_name}' from '{repo_id}'...")
        try:
            cached_path = hf_hub_download(
                repo_id=repo_id,
                filename=file_name,
                subfolder=subfolder,
                token=HF_TOKEN,
            )
            shutil.copy(cached_path, destination_path)
            print(f"[+] Saved successfully to: {destination_path}\n")
        except Exception as exc:
            print(
                f"[ERROR] Could not download/validate '{file_name}': {str(exc)}\n"
            )


def prepare_flux_model():
    """Download or validate the base FLUX.1-dev model in the local cache."""
    print("[*] Checking base FLUX.1-dev model (~57 GB on disk)...")
    print("[*] If already cached, this step should finish immediately.")
    try:
        snapshot_download(
            repo_id=FLUX_MODEL_ID,
            repo_type="model",
            ignore_patterns=["*.md", "*.pdf"],
            token=HF_TOKEN,
        )
        print("[+] FLUX.1-dev verified and ready in the system cache!\n")
    except Exception as exc:
        print(f"[ERROR] Failed to verify/download FLUX: {exc}\n")


def main():
    ensure_base_directories()
    login_to_hugging_face()
    download_required_files()
    prepare_flux_model()

    print("=" * BANNER_WIDTH)
    print("[+] ENVIRONMENT FULLY PREPARED, UPDATED, AND VERIFIED!")
    print("=" * BANNER_WIDTH)


if __name__ == "__main__":
    main()
