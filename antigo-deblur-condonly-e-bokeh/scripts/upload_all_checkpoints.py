"""Sobe checkpoints que JÁ existem no disco pro HF, um repo POR-STEP.

Carrega o FLUX UMA vez e, pra cada step_*.pt, injeta o LoRA e sobe pra
<repo-base>-step<N>. Assim você guarda snapshots de vários steps pra depois
avaliar quanto o modelo melhora com mais steps.

Uso:
  python -m scripts.upload_all_checkpoints \
    --config configs/train_base_4gpu.yaml \
    --repo-base juliadollis/deblur-condlora \
    [--every 1] [--only 5000,10000,13500] [--ckpts-dir ...]
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch

from genfocus_train.backbone import create_backbone
from genfocus_train.config import load_config
from genfocus_train.trainer import (
    _checkpoint_dir,
    _sorted_checkpoints,
    load_lora_checkpoint_into_backbone,
    upload_lora_snapshot_to_hf,
)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, help="o mesmo config do treino (define o modelo/output_dir)")
    p.add_argument("--repo-base", required=True, help="ex.: juliadollis/deblur-condlora  -> vira -step<N>")
    p.add_argument("--ckpts-dir", default=None, help="pasta dos checkpoints (default: do config)")
    p.add_argument("--every", type=int, default=1, help="sobe 1 a cada N checkpoints (pra não subir todos)")
    p.add_argument("--only", default=None, help="csv de steps específicos, ex.: 5000,10000,13500")
    args = p.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA não disponível (o export precisa carregar o FLUX na GPU).")
    torch.cuda.set_device(0)
    device = torch.device("cuda")
    dtype = torch.bfloat16

    cfg = load_config(args.config)
    print("[upload-all] carregando FLUX + LoRA (1 vez)...")
    backbone = create_backbone(cfg.model, stage="deblur")
    backbone.load(dtype=dtype, device=device)

    ckdir = Path(args.ckpts_dir) if args.ckpts_dir else _checkpoint_dir(Path(cfg.runtime.output_dir), "deblur")
    ckpts = _sorted_checkpoints(ckdir)
    if not ckpts:
        raise RuntimeError(f"Nenhum step_*.pt em {ckdir}")

    if args.only:
        want = {int(x) for x in args.only.split(",")}
        ckpts = [c for c in ckpts if int(c.stem.split("_")[-1]) in want]
    elif args.every > 1:
        ckpts = ckpts[:: args.every]

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    print(f"[upload-all] {len(ckpts)} checkpoints -> {args.repo_base}-step<N>")
    for c in ckpts:
        step = load_lora_checkpoint_into_backbone(backbone, c)
        upload_lora_snapshot_to_hf(backbone, "deblur", args.repo_base, step, token=token)

    print("[upload-all] pronto.")


if __name__ == "__main__":
    main()
