"""CLI de treino do GenRefocus (DeblurNet Stage 1 + BokehNet Stage 2).

Comandos:
  smoke   — 3 steps em FLUX real, valida o pipeline antes do treino real.
  deblur  — treina a DeblurNet (Stage 1). LoRA rank 128.
  bokeh   — treina a BokehNet  (Stage 2). LoRA rank 64, 2 condições (AIF + D_def).
  export  — exporta um checkpoint (step_*.pt) para .safetensors usável na inferência.

Uso (a partir da raiz do projeto, com Genfocus no PYTHONPATH):
  python -m genfocus_train.train smoke  --config configs/train_smoke.yaml
  python -m genfocus_train.train deblur --config configs/train_base.yaml
  python -m genfocus_train.train bokeh  --config configs/train_bokeh.yaml
  python -m genfocus_train.train export --config configs/train_ddpd.yaml \
      --stage deblur --output deblurNet.safetensors   # pega o ÚLTIMO checkpoint
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch

from .backbone import create_backbone
from .config import TrainConfig, load_config
from .models import BokehNet, DeblurNet
from .trainer import (
    _checkpoint_dir,
    _sorted_checkpoints,
    export_lora_safetensors,
    load_lora_checkpoint_into_backbone,
    run_bokeh_stage,
    run_deblur_stage,
    run_smoke_test,
)


# Prompts fixos: FLUX é text-to-image, o condicionamento real vem por token concat.
# Ambos conferidos contra a inferência OFICIAL (o treino tem que usar o MESMO prompt
# que a inferência, senão o modelo aprende num condicionamento de texto e roda noutro):
#   Inference_deblurNet.py: prompt="a sharp photo with everything in focus"
#   Inference_bokehNet.py:  prompt="an excellent photo with a large aperture"
DEBLUR_PROMPT = "a sharp photo with everything in focus"
BOKEH_PROMPT = "an excellent photo with a large aperture"

STAGE_PROMPTS = {"deblur": DEBLUR_PROMPT, "bokeh": BOKEH_PROMPT}


def _build_backbone_and_model(
    config: TrainConfig, stage: str, dtype: torch.dtype, device: torch.device
):
    """Cria backbone (rank do stage), carrega FLUX, encoda o prompt, retorna (backbone, model)."""
    backbone = create_backbone(config.model, stage=stage)
    backbone.load(dtype=dtype, device=device)

    text_embeddings = backbone.precompute_text(STAGE_PROMPTS[stage], device=device, dtype=dtype)
    backbone.release_text_encoders()

    if stage == "deblur":
        model = DeblurNet(backbone=backbone, text_embeddings=text_embeddings)
    elif stage == "bokeh":
        model = BokehNet(
            backbone=backbone,
            text_embeddings=text_embeddings,
            max_coc=config.model.max_coc,   # antes o max_coc do config era ignorado
        )
    else:
        raise ValueError(f"stage inválido: {stage}")
    return backbone, model


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GenRefocus DeblurNet training CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    def add_common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--config", required=True)
        p.add_argument("--output-dir", default=None)

    p_smoke = sub.add_parser("smoke", help="Smoke test (3 steps, FLUX real, 1 batch)")
    add_common(p_smoke)
    p_smoke.add_argument(
        "--stage",
        default="deblur",
        choices=["deblur", "bokeh"],
        help=(
            "Qual stage exercitar no smoke. 'deblur' (1 condição) ou 'bokeh' "
            "(2 condições: AIF + defocus map). Para bokeh, use um config com "
            "bloco data.bokeh (ex.: configs/train_bokeh_smoke.yaml)."
        ),
    )
    add_common(sub.add_parser("deblur", help="Treino da DeblurNet (Stage 1)"))
    p_bokeh = sub.add_parser("bokeh", help="Treino da BokehNet (Stage 2)")
    add_common(p_bokeh)
    p_bokeh.add_argument(
        "--init-lora",
        default=None,
        help=(
            "Fase 2 do currículo do paper (real). Caminho de um step_*.pt OU do "
            "output_dir da FASE 1 (sintético) — inicia o LoRA daí, com "
            "optimizer/scheduler frescos. Só aplica quando o run começa do zero."
        ),
    )

    p_export = sub.add_parser(
        "export", help="Exporta um checkpoint para .safetensors (inferência)"
    )
    add_common(p_export)
    p_export.add_argument(
        "--stage",
        default="deblur",
        choices=["deblur", "bokeh"],
        help="Qual stage exportar (define o rank do LoRA e a pasta do checkpoint).",
    )
    p_export.add_argument(
        "--checkpoint",
        default=None,
        help="Caminho de um step_*.pt específico. Default: o último do output_dir.",
    )
    p_export.add_argument(
        "--output",
        default=None,
        help="Caminho do .safetensors de saída. Default: <output_dir>/<stage>/<stage>_step<N>.safetensors",
    )

    return parser.parse_args()


def _output_dir(config: TrainConfig, override: str | None) -> Path:
    return Path(override) if override else Path(config.runtime.output_dir)


def _resolve_init_lora(path_str: str) -> Path:
    """Resolve o --init-lora para um step_*.pt concreto.

    Aceita: um step_*.pt direto; um dir de checkpoints; OU o output_dir da fase 1
    (nesse caso procura em <dir>/bokeh/checkpoints e pega o último step).
    """
    p = Path(path_str)
    if p.is_file():
        return p
    if p.is_dir():
        # tenta <dir>/bokeh/checkpoints (output_dir da fase 1); senão o próprio dir
        cand = _checkpoint_dir(p, "bokeh")
        ckdir = cand if cand.is_dir() else p
        ckpts = _sorted_checkpoints(ckdir)
        if not ckpts:
            raise FileNotFoundError(f"Nenhum step_*.pt em {ckdir} (--init-lora={path_str}).")
        return ckpts[-1]
    raise FileNotFoundError(f"--init-lora não encontrado: {path_str}")


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
    stage = args.stage

    # Só o backbone (com LoRA injetado) — export não precisa de prompt/modelo.
    backbone = create_backbone(config.model, stage=stage)
    backbone.load(dtype=dtype, device=device)

    if args.checkpoint:
        ckpt_path = Path(args.checkpoint)
        if not ckpt_path.is_file():
            raise FileNotFoundError(f"Checkpoint não encontrado: {ckpt_path}")
    else:
        ckpts = _sorted_checkpoints(_checkpoint_dir(output_dir, stage))
        if not ckpts:
            raise RuntimeError(
                f"Nenhum step_*.pt em {output_dir}/{stage}/checkpoints. "
                "O treino já salvou algum checkpoint? (save_every_steps)"
            )
        ckpt_path = ckpts[-1]

    step = load_lora_checkpoint_into_backbone(backbone, ckpt_path)

    if args.output:
        out_path = Path(args.output)
    else:
        out_path = output_dir / stage / f"{stage}_step{step}.safetensors"

    export_lora_safetensors(backbone, out_path, stage)
    print(f"[main] Exportado checkpoint step={step} ({stage}) -> {out_path}")


def main() -> None:
    args = _parse_args()
    config = load_config(args.config)
    output_dir = _output_dir(config, args.output_dir)
    dtype, device = _resolve_dtype_device(config)

    if args.command == "export":
        _run_export(config, dtype, device, args)
        return

    if args.command == "smoke":
        # Smoke agora é stage-aware: --stage deblur (default) ou --stage bokeh.
        stage = args.stage
        backbone, model = _build_backbone_and_model(config, stage, dtype, device)
        run_smoke_test(config, backbone, model, output_dir, stage=stage)
        print(f"[main] Smoke test ({stage}) passou.")
        return

    stage = "bokeh" if args.command == "bokeh" else "deblur"
    backbone, model = _build_backbone_and_model(config, stage, dtype, device)

    if args.command == "deblur":
        ckpt = run_deblur_stage(config, backbone, model, output_dir)
        print(f"[main] DeblurNet finalizado. Checkpoint: {ckpt}")
        return

    if args.command == "bokeh":
        init_lora = _resolve_init_lora(args.init_lora) if args.init_lora else None
        ckpt = run_bokeh_stage(config, backbone, model, output_dir, init_lora_path=init_lora)
        print(f"[main] BokehNet finalizado. Checkpoint: {ckpt}")
        return

    raise ValueError(f"Comando desconhecido: {args.command}")


if __name__ == "__main__":
    main()
