"""CLI de treino da DeblurNet (GenRefocus, Stage 1).

Comandos:
  smoke   — 3 steps em FLUX real, valida o pipeline antes do treino real.
  deblur  — treina a DeblurNet (Stage 1).
  export  — exporta um checkpoint (step_*.pt) para .safetensors usável na inferência.

Uso (a partir da raiz do projeto, com Genfocus no PYTHONPATH):
  python -m genfocus_train.train smoke  --config configs/train_smoke.yaml
  python -m genfocus_train.train deblur --config configs/train_base.yaml
  python -m genfocus_train.train export --config configs/train_ddpd.yaml \
      --output deblurNet.safetensors        # pega o ÚLTIMO checkpoint
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch

from .backbone import create_backbone
from .config import TrainConfig, load_config
from .models import DeblurNet
from .trainer import (
    _checkpoint_dir,
    _sorted_checkpoints,
    export_lora_safetensors,
    load_lora_checkpoint_into_backbone,
    run_deblur_stage,
    run_smoke_test,
)


# DeblurNet: condicionamento por imagem (token concat). FLUX é text-to-image,
# então usamos um prompt fixo. [UNSPECIFIED no paper] — confronte com o prompt
# da inferência oficial (Inference_deblurNet.py) antes do treino final.
DEBLUR_PROMPT = "a sharp photo with everything in focus"


def _build_backbone_and_deblur(
    config: TrainConfig, dtype: torch.dtype, device: torch.device
):
    """Cria backbone, carrega FLUX, encoda o prompt fixo, retorna (backbone, model)."""
    backbone = create_backbone(config.model, stage="deblur")
    backbone.load(dtype=dtype, device=device)

    text_embeddings = backbone.precompute_text(DEBLUR_PROMPT, device=device, dtype=dtype)
    backbone.release_text_encoders()

    model = DeblurNet(backbone=backbone, text_embeddings=text_embeddings)
    return backbone, model


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GenRefocus DeblurNet training CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    def add_common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--config", required=True)
        p.add_argument("--output-dir", default=None)

    add_common(sub.add_parser("smoke", help="Smoke test (3 steps, FLUX real, 1 batch)"))
    add_common(sub.add_parser("deblur", help="Treino da DeblurNet (Stage 1)"))

    p_export = sub.add_parser(
        "export", help="Exporta um checkpoint para .safetensors (inferência)"
    )
    add_common(p_export)
    p_export.add_argument(
        "--checkpoint",
        default=None,
        help="Caminho de um step_*.pt específico. Default: o último do output_dir.",
    )
    p_export.add_argument(
        "--output",
        default=None,
        help="Caminho do .safetensors de saída. Default: <output_dir>/deblur/deblur_step<N>.safetensors",
    )

    return parser.parse_args()


def _output_dir(config: TrainConfig, override: str | None) -> Path:
    return Path(override) if override else Path(config.runtime.output_dir)


def _resolve_dtype_device(config: TrainConfig) -> tuple[torch.dtype, torch.device]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA não disponível. O treino requer GPU.")
    # Multi-GPU (accelerate launch): cada processo carrega o modelo na SUA GPU.
    # LOCAL_RANK é setado pelo launcher; em single-GPU fica 0 (= comportamento atual).
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    mp = config.runtime.mixed_precision.lower()
    if mp == "bf16":
        dtype = torch.bfloat16
    elif mp in {"fp16", "16"}:
        dtype = torch.float16
    elif mp in {"no", "fp32"}:
        dtype = torch.float32
    else:
        raise ValueError(f"mixed_precision inválido: {mp}")
    torch.backends.cuda.matmul.allow_tf32 = True
    return dtype, device


def _run_export(config: TrainConfig, dtype, device, args) -> None:
    """Exporta o LoRA de um checkpoint para .safetensors (sem text encoders)."""
    output_dir = _output_dir(config, args.output_dir)

    # Só o backbone (com LoRA injetado) — export não precisa de prompt/DeblurNet.
    backbone = create_backbone(config.model, stage="deblur")
    backbone.load(dtype=dtype, device=device)

    if args.checkpoint:
        ckpt_path = Path(args.checkpoint)
        if not ckpt_path.is_file():
            raise FileNotFoundError(f"Checkpoint não encontrado: {ckpt_path}")
    else:
        ckpts = _sorted_checkpoints(_checkpoint_dir(output_dir, "deblur"))
        if not ckpts:
            raise RuntimeError(
                f"Nenhum step_*.pt em {output_dir}/deblur/checkpoints. "
                "O treino já salvou algum checkpoint? (save_every_steps)"
            )
        ckpt_path = ckpts[-1]

    step = load_lora_checkpoint_into_backbone(backbone, ckpt_path)

    if args.output:
        out_path = Path(args.output)
    else:
        out_path = output_dir / "deblur" / f"deblur_step{step}.safetensors"

    export_lora_safetensors(backbone, out_path, "deblur")
    print(f"[main] Exportado checkpoint step={step} -> {out_path}")


def main() -> None:
    args = _parse_args()
    config = load_config(args.config)
    output_dir = _output_dir(config, args.output_dir)
    dtype, device = _resolve_dtype_device(config)

    if args.command == "export":
        _run_export(config, dtype, device, args)
        return

    backbone, model = _build_backbone_and_deblur(config, dtype, device)

    if args.command == "smoke":
        run_smoke_test(config, backbone, model, output_dir)
        print("[main] Smoke test passou.")
        return

    if args.command == "deblur":
        ckpt = run_deblur_stage(config, backbone, model, output_dir)
        print(f"[main] DeblurNet finalizado. Checkpoint: {ckpt}")
        return

    raise ValueError(f"Comando desconhecido: {args.command}")


if __name__ == "__main__":
    main()
