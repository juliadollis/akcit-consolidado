"""CLI de treino do GenRefocus (DeblurNet Stage 1 + BokehNet Stage 2).

Comandos:
  check   — valida o config SEM GPU e SEM treinar. Rode ANTES de submeter.
  smoke   — 3 steps em FLUX real, valida o pipeline antes do treino real.
  deblur  — treina a DeblurNet (Stage 1). LoRA rank 128.
  bokeh   — treina a BokehNet  (Stage 2). LoRA rank 64, 2 condições (AIF + D_def).
  export  — exporta um checkpoint (step_*.pt) para .safetensors + sidecar .json.

Uso (a partir da raiz do projeto, com Genfocus no PYTHONPATH):
  python -m genfocus_train.train check  --config configs/train_base_4gpu.yaml
  python -m genfocus_train.train smoke  --config configs/train_smoke.yaml
  python -m genfocus_train.train deblur --config configs/train_base_4gpu.yaml
  python -m genfocus_train.train bokeh  --config configs/train_bokeh.yaml
  python -m genfocus_train.train export --config configs/train_base_4gpu.yaml \
      --stage deblur --output deblurNet.safetensors   # pega o ÚLTIMO checkpoint

ÁRVORE NOVA (retreinar-deblur/): o treino antigo continua intacto em
`genrefocus_deblurnet_paper/` e `genrefocus_deblurnet/`. As diferenças estão
registradas em MUDANCAS_CODIGO.md.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch

from .backbone import create_backbone
from .config import (
    TrainConfig,
    lora_rank_for,
    load_config,
    stage_config,
    train_guidance_for,
)
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
    # create_backbone recebe o TrainConfig INTEIRO (não só o model): ele precisa
    # do StageConfig para saber o `sigma_mu_source` e do ModelConfig para saber
    # o guidance de treino do estágio e o `lora_on_main`. Ver C2/C4/C8.
    backbone = create_backbone(config, stage=stage)
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

    p_check = sub.add_parser(
        "check",
        help=(
            "Valida o config SEM treinar e SEM GPU: imprime os eixos do "
            "experimento, o batch efetivo e o que falta de credencial. "
            "Rode isto antes de submeter ao SLURM."
        ),
    )
    add_common(p_check)
    p_check.add_argument("--stage", default="deblur", choices=["deblur", "bokeh"])

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


def _run_check(config: TrainConfig, args) -> int:
    """Dry-run do config. NÃO carrega FLUX, NÃO exige GPU, NÃO treina.

    Existe porque todo defeito que este comando pega custaria, no cluster, uma
    fila inteira até estourar: eixo de experimento com valor errado, credencial
    ausente descoberta só na hora do upload, batch efetivo diferente do que se
    pensou, `steps` menor que o warmup.

    Retorna o nº de problemas encontrados (0 = pronto para submeter).
    """
    stage = args.stage
    cfg = stage_config(config, stage)
    rt, lg = config.runtime, config.logging
    problemas: list[str] = []
    avisos: list[str] = []

    n_gpus_env = os.environ.get("SLURM_GPUS_ON_NODE") or os.environ.get("WORLD_SIZE")
    n_gpus = int(n_gpus_env) if (n_gpus_env or "").isdigit() else 1
    batch_efetivo = cfg.batch_size * rt.gradient_accumulation_steps * n_gpus

    print("=" * 72)
    print(f"CHECK  stage={stage}  config={args.config}")
    print("=" * 72)

    print("\n-- Eixos que definem o modelo (tudo isto vai no metadata) --")
    print(f"  lora_on_main      : {config.model.lora_on_main}"
          f"   ({'main+cond -> inferência EXIGE main_adapter' if config.model.lora_on_main else 'cond-only -> inferência oficial'})")
    print(f"  lora_rank         : {lora_rank_for(config, stage)}")
    print(f"  train_guidance    : {train_guidance_for(config, stage)}"
          f"   (condição: {config.model.cond_train_guidance})")
    print(f"  scale_mode        : {cfg.scale_mode}")
    print(f"  sigma_mu_source   : {cfg.sigma_mu_source}")
    print(f"  top_k_mode        : {cfg.top_k_mode}  (scene_key={cfg.scene_key})")
    print(f"  image_size        : {cfg.image_size}")

    print("\n-- Escala do treino --")
    print(f"  steps             : {cfg.steps}")
    print(f"  batch_size        : {cfg.batch_size}")
    print(f"  grad_accum        : {rt.gradient_accumulation_steps}")
    print(f"  GPUs detectadas   : {n_gpus}  (via SLURM_GPUS_ON_NODE/WORLD_SIZE)")
    print(f"  batch EFETIVO     : {batch_efetivo}   (paper §4.1: 1 × 8 × 4 = 32)")
    print(f"  warmup            : {config.scheduler.warmup_steps}")
    print(f"  lr                : {config.optimizer.lr}")

    print("\n-- Dados --")
    for s in cfg.datasets:
        extra = f"  top_k_sharpest={s.top_k_sharpest} ({cfg.top_k_mode})" if s.top_k_sharpest else ""
        print(f"  treino : {s.name}:{s.split}{extra}")
    if cfg.val_datasets:
        for s in cfg.val_datasets:
            print(f"  val    : {s.name}:{s.split}")
    else:
        print("  val    : (nenhum)")

    print("\n-- Persistência --")
    print(f"  output_dir        : {rt.output_dir}")
    print(f"  save_every_steps  : {rt.save_every_steps}")
    print(f"  keep_last_n       : {rt.keep_last_n_checkpoints}")
    print(f"  save_best         : {rt.save_best}")
    print(f"  eval_every_steps  : {rt.eval_every_steps}")
    print(f"  resume            : {rt.resume}")
    print(f"  wandb             : {lg.use_wandb}  projeto={lg.wandb_project} resume={lg.wandb_resume}")
    print(f"  HF upload         : a cada {lg.upload_every_steps} -> {lg.upload_hf_repo_base}"
          f"  (final={lg.upload_final} best={lg.upload_best} privado={lg.upload_private})")

    # ── validações que valem uma fila de cluster ────────────────────────────
    if cfg.steps <= config.scheduler.warmup_steps:
        problemas.append(
            f"steps={cfg.steps} <= warmup_steps={config.scheduler.warmup_steps}: "
            "o treino acabaria ainda no warmup, nunca atingindo o lr base."
        )
    if batch_efetivo != 32:
        avisos.append(
            f"batch efetivo {batch_efetivo} != 32 do paper (§4.1). Intencional? "
            "Se as GPUs não foram detectadas aqui, o número muda no cluster."
        )
    if rt.save_every_steps <= 0 or rt.save_every_steps > cfg.steps:
        problemas.append(
            f"save_every_steps={rt.save_every_steps} com steps={cfg.steps}: "
            "o treino pode terminar (ou morrer) sem nunca salvar um checkpoint."
        )
    if rt.eval_every_steps > 0 and not cfg.val_datasets:
        avisos.append(
            f"eval_every_steps={rt.eval_every_steps} mas val_datasets está vazio: "
            "a validação será pulada e save_best não terá métrica."
        )
    if rt.save_best and rt.eval_every_steps <= 0:
        avisos.append(
            "save_best=True com eval_every_steps=0: sem validação, o 'melhor' "
            "cairia na loss de treino de um único micro-batch, que não é sinal."
        )
    if lg.upload_hf_repo_base and not (
        os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    ):
        problemas.append(
            "upload_hf_repo_base configurado mas HF_TOKEN/HUGGINGFACE_HUB_TOKEN "
            "ausente no ambiente: os snapshots não subiriam."
        )
    if (lg.upload_final or lg.upload_best or lg.upload_every_steps) and not lg.upload_hf_repo_base:
        avisos.append(
            "upload ligado mas upload_hf_repo_base=None: nada será enviado ao HF."
        )
    if lg.use_wandb and not os.environ.get("WANDB_API_KEY"):
        avisos.append(
            "use_wandb=True mas WANDB_API_KEY ausente: o wandb cai em modo "
            "offline/anônimo e o log fica só no JSONL local."
        )
    if not os.environ.get("HF_TOKEN") and not os.environ.get("HUGGINGFACE_HUB_TOKEN"):
        problemas.append(
            "HF_TOKEN ausente: o dataloader usa `get_required_env('HF_TOKEN')` "
            "para baixar os datasets e falharia no primeiro batch."
        )
    if config.model.lora_on_main:
        avisos.append(
            "lora_on_main=True: este checkpoint EXIGE "
            "generate(..., main_adapter=...) na inferência. Com main_adapter=None "
            "a saída sai lavada."
        )

    print("\n" + "=" * 72)
    for a in avisos:
        print(f"  [aviso]    {a}")
    for p in problemas:
        print(f"  [PROBLEMA] {p}")
    if not problemas and not avisos:
        print("  Nada a apontar.")
    print("=" * 72)
    print(f"CHECK: {len(problemas)} problema(s), {len(avisos)} aviso(s).")
    return len(problemas)


def _run_export(config: TrainConfig, dtype, device, args) -> None:
    """Exporta o LoRA de um checkpoint para .safetensors (sem text encoders)."""
    output_dir = _output_dir(config, args.output_dir)
    stage = args.stage

    # Só o backbone (com LoRA injetado) — export não precisa de prompt/modelo.
    backbone = create_backbone(config, stage=stage)
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

    # Sidecar com a procedência. O trainer grava isto a cada save; o export
    # precisa gravar também, senão um .safetensors exportado de um checkpoint
    # antigo sai sem dizer QUAL variante é — e foi exatamente essa ambiguidade
    # (cond-only vs main+cond) que produziu o bug da saída lavada. Ver C8.
    sidecar = out_path.with_suffix(".json")
    info = dict(backbone.lora_info())
    info.update({
        "stage": stage,
        "global_step": step,
        "checkpoint": str(ckpt_path),
        "exported_from": "genfocus_train.train export",
    })
    sidecar.write_text(
        json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[main] Exportado checkpoint step={step} ({stage}) -> {out_path}")
    print(f"[main] Procedência -> {sidecar}")


def main() -> None:
    args = _parse_args()
    config = load_config(args.config)
    output_dir = _output_dir(config, args.output_dir)

    # `check` roda ANTES do _resolve_dtype_device de propósito: ele existe para
    # ser executado no login node, sem GPU, antes de submeter o job.
    if args.command == "check":
        raise SystemExit(1 if _run_check(config, args) else 0)

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
