"""
Training runtime para a DeblurNet (GenRefocus, Stage 1).

Escopo: apenas DeblurNet. BokehNet (Stage 2) ainda não foi integrado ao
dataloader HF (o dataset HF não fornece depth/defocus/K), então não há
stage de bokeh aqui — `models.BokehNet` permanece como scaffold para o futuro.

Características:
  - Sem _FallbackAccelerator: se accelerate não está instalado, falha explícita.
  - Loss = flow_matching_loss (de models.py).
  - Checkpoint salva SÓ LoRA (~50–200 MB, não 24 GB).
  - scheduler.step / contagem de step gated por accelerator.sync_gradients.
  - load_checkpoint com validação de shape estrita.
  - Dados vêm do Hugging Face via `data.build_dataset` / `data.build_dataloader`.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Literal

import torch
from torch import nn

from .config import StageConfig, TrainConfig, config_hash, write_effective_config
from .data import DatasetRuntimeConfig, build_dataloader, build_dataset
from .models import DeblurNet, flow_matching_loss
from .backbone import FluxBackbone


StageName = Literal["deblur", "bokeh"]


# =============================================================================
# Accelerator — sem fallback. Se não tiver, falha.
# =============================================================================

def _build_accelerator(config: TrainConfig):
    """Constrói o Accelerator ou falha com mensagem clara."""
    try:
        from accelerate import Accelerator
    except ImportError as exc:
        raise ImportError(
            "Pacote `accelerate` é obrigatório. Instale com: pip install accelerate"
        ) from exc

    # Timeout do process group ELÁSTICO: no multi-GPU, cada rank carrega o FLUX
    # e roda o filtro do RealBokeh (~20 min) ANTES do primeiro coletivo. Se os
    # ranks dessincronizam (ex.: contenção de disco lendo o dataset em paralelo),
    # o barrier padrão do NCCL (30 min) estoura e o job CRASHA no startup. 90 min
    # dá folga. (Em single-GPU isso é inócuo.)
    kwargs_handlers = []
    try:
        from datetime import timedelta

        from accelerate.utils import InitProcessGroupKwargs

        kwargs_handlers.append(
            InitProcessGroupKwargs(timeout=timedelta(minutes=90))
        )
    except Exception:  # pragma: no cover - versões antigas do accelerate
        pass

    return Accelerator(
        gradient_accumulation_steps=config.runtime.gradient_accumulation_steps,
        mixed_precision=config.runtime.mixed_precision,
        kwargs_handlers=kwargs_handlers,
    )


def _autocast_dtype(config: TrainConfig) -> torch.dtype | None:
    """Dtype para torch.autocast em loops que não usam Accelerator."""
    mp = config.runtime.mixed_precision.lower()
    if mp == "bf16":
        return torch.bfloat16
    if mp in {"fp16", "16"}:
        return torch.float16
    return None  # fp32/no autocast


# =============================================================================
# Logger
# =============================================================================

class ExperimentLogger:
    """Log para JSONL local + wandb (opcional)."""

    def __init__(self, config: TrainConfig, output_dir: Path, stage: StageName) -> None:
        self.output_dir = output_dir
        self.stage = stage
        self.metrics_path = output_dir / stage / "metrics.jsonl"
        self.metrics_path.parent.mkdir(parents=True, exist_ok=True)

        cfg_hash_short = config_hash(config)[:8]

        self._wandb_run = None
        if config.logging.use_wandb:
            try:
                import wandb
                self._wandb_run = wandb.init(
                    project=config.logging.wandb_project,
                    entity=config.logging.wandb_entity,
                    name=config.logging.run_name or f"{stage}-{cfg_hash_short}",
                    config={"stage": stage, "config_hash": cfg_hash_short},
                )
            except ImportError:
                print("[logger] wandb não instalado; log somente em JSONL local.")
                self._wandb_run = None
            except Exception as exc:
                print(f"[logger] wandb falhou ({exc}); log somente local.")
                self._wandb_run = None

    def log(self, payload: dict[str, float | int], step: int) -> None:
        row = {"step": step, **payload}
        with self.metrics_path.open("a", encoding="utf-8") as h:
            h.write(json.dumps(row, ensure_ascii=True) + "\n")
        if self._wandb_run is not None:
            self._wandb_run.log(payload, step=step)

    def finish(self) -> None:
        if self._wandb_run is not None:
            self._wandb_run.finish()


# =============================================================================
# Estado e otimizador
# =============================================================================

@dataclass
class TrainState:
    global_step: int = 0
    best_loss: float = math.inf


def _make_optimizer(config: TrainConfig, params: list[torch.nn.Parameter]) -> torch.optim.Optimizer:
    if config.optimizer.name.lower() != "adamw":
        raise ValueError(f"Optimizer não suportado: {config.optimizer.name}")
    return torch.optim.AdamW(
        params=params,
        lr=config.optimizer.lr,
        betas=tuple(config.optimizer.betas),
        eps=config.optimizer.eps,
        weight_decay=config.optimizer.weight_decay,
    )


class WarmupCosineScheduler:
    """Warmup linear + decaimento cosseno."""

    def __init__(self, optimizer, total_steps: int, warmup_steps: int, min_lr_ratio: float):
        self.optimizer = optimizer
        self.total_steps = max(total_steps, 1)
        self.warmup_steps = max(warmup_steps, 0)
        self.min_lr_ratio = min_lr_ratio
        self.base_lrs = [g["lr"] for g in optimizer.param_groups]
        self.last_step = 0

    def step(self, global_step: int) -> None:
        self.last_step = global_step
        for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups, strict=True):
            group["lr"] = self._lr_at(base_lr, global_step)

    def _lr_at(self, base_lr: float, step: int) -> float:
        if step < self.warmup_steps:
            return base_lr * float(step + 1) / float(max(1, self.warmup_steps))
        progress = (step - self.warmup_steps) / float(max(1, self.total_steps - self.warmup_steps))
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return base_lr * (self.min_lr_ratio + (1.0 - self.min_lr_ratio) * cosine)

    def state_dict(self) -> dict[str, Any]:
        return {
            "last_step": self.last_step,
            "base_lrs": self.base_lrs,
            "total_steps": self.total_steps,
            "warmup_steps": self.warmup_steps,
            "min_lr_ratio": self.min_lr_ratio,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.last_step = int(state["last_step"])
        self.base_lrs = list(state["base_lrs"])
        self.total_steps = int(state["total_steps"])
        self.warmup_steps = int(state["warmup_steps"])
        self.min_lr_ratio = float(state["min_lr_ratio"])


def _make_scheduler(config: TrainConfig, optimizer, total_steps: int) -> WarmupCosineScheduler:
    if config.scheduler.name.lower() != "cosine":
        raise ValueError(f"Scheduler não suportado: {config.scheduler.name}")
    return WarmupCosineScheduler(
        optimizer=optimizer,
        total_steps=total_steps,
        warmup_steps=config.scheduler.warmup_steps,
        min_lr_ratio=config.scheduler.min_lr_ratio,
    )


# =============================================================================
# Checkpoint
# =============================================================================

def _get_git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


def _checkpoint_dir(output_dir: Path, stage: StageName) -> Path:
    path = output_dir / stage / "checkpoints"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _sorted_checkpoints(path: Path) -> list[Path]:
    return sorted(path.glob("step_*.pt"), key=lambda p: int(p.stem.split("_")[-1]))


def save_checkpoint(
    *,
    output_dir: Path,
    stage: StageName,
    backbone: FluxBackbone,
    optimizer: torch.optim.Optimizer,
    scheduler: WarmupCosineScheduler,
    state: TrainState,
    cfg_hash: str,
    metrics_snapshot: dict[str, float],
    keep_last_n: int,
) -> Path:
    """
    Salva checkpoint contendo:
      - state_dict dos parâmetros TREINÁVEIS do transformer (LoRA)
      - state_dict do optimizer e scheduler
      - metadata (step, hash, git, métricas)
    """
    ckpt_dir = _checkpoint_dir(output_dir, stage)
    ckpt_path = ckpt_dir / f"step_{state.global_step}.pt"

    lora_state = {
        name: param.detach().cpu()
        for name, param in backbone.transformer.named_parameters()
        if param.requires_grad
    }

    payload = {
        "lora_state": lora_state,
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "train_state": {"global_step": state.global_step, "best_loss": state.best_loss},
        "metadata": {
            "stage": stage,
            "global_step": state.global_step,
            "config_hash": cfg_hash,
            "metrics_snapshot": metrics_snapshot,
            "git_commit": _get_git_commit(),
        },
    }
    torch.save(payload, ckpt_path)

    checkpoints = _sorted_checkpoints(ckpt_dir)
    if keep_last_n > 0 and len(checkpoints) > keep_last_n:
        for old in checkpoints[:-keep_last_n]:
            old.unlink(missing_ok=True)

    latest = ckpt_dir / "latest.json"
    latest.write_text(
        json.dumps({"checkpoint": str(ckpt_path)}, ensure_ascii=True),
        encoding="utf-8",
    )
    return ckpt_path


def export_lora_safetensors(
    backbone: FluxBackbone,
    output_path: Path,
    stage: StageName,
) -> Path:
    """
    Exporta os pesos LoRA em .safetensors compatível com
    Inference_deblurNet.py (`pipe.load_lora_weights(...)`).
    """
    from safetensors.torch import save_file
    from peft.utils import get_peft_model_state_dict

    output_path.parent.mkdir(parents=True, exist_ok=True)
    raw = get_peft_model_state_dict(backbone.transformer, adapter_name="default")

    # Normaliza as chaves para o formato diffusers que `load_lora_weights`
    # espera e que o checkpoint oficial usa:
    #   transformer.<modulo>.lora_A.weight  /  ...lora_B.weight
    # add_adapter expõe named_parameters SEM o prefixo `transformer.` e às vezes
    # com o infixo do adapter (`.default.`); aqui removemos/ajustamos.
    lora_state: dict[str, Any] = {}
    for key, value in raw.items():
        norm = key.replace(".default.", ".")
        if not norm.startswith("transformer."):
            norm = "transformer." + norm
        lora_state[norm] = value

    save_file(lora_state, str(output_path))
    sample = list(lora_state.keys())[:3]
    print(f"[export] {len(lora_state)} tensores LoRA salvos. ex: {sample}")
    return output_path


def upload_lora_snapshot_to_hf(
    backbone: FluxBackbone,
    stage: StageName,
    repo_base: str,
    step: int,
    output_dir: Path,
    token: str | None = None,
) -> None:
    """Exporta o LoRA ATUAL e sobe pro HF como <stage>_step<N>.safetensors.

    UM repo só (`repo_base`), um ARQUIVO por step — assim o histórico do treino
    fica todo no mesmo lugar (antes era um repo novo por step, o que gerava ~8
    repos por fase e complicava comparar).

    O .safetensors temporário é escrito no output_dir, NÃO em /tmp: dentro do
    container o /tmp é tmpfs (RAM), e cada snapshot tem centenas de MB.

    NÃO-FATAL: se falhar (rede/token), só avisa e o treino segue.
    """
    token = token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    tmp = output_dir / stage / f".upload_{stage}_step{step}.safetensors"
    try:
        from huggingface_hub import HfApi

        repo_id = repo_base
        if "/" not in repo_id and token:
            from huggingface_hub import whoami

            repo_id = f"{whoami(token=token)['name']}/{repo_id}"

        export_lora_safetensors(backbone, tmp, stage)

        api = HfApi(token=token)
        api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
        api.upload_file(
            path_or_fileobj=str(tmp), path_in_repo=f"{stage}_step{step}.safetensors",
            repo_id=repo_id, repo_type="model",
        )
        print(
            f"[upload] step {step} -> https://huggingface.co/{repo_id}"
            f"/blob/main/{stage}_step{step}.safetensors"
        )
    except Exception as exc:  # noqa: BLE001 - upload nunca deve quebrar o treino
        print(f"[upload] WARN: falha subindo o step {step}: {exc}")
    finally:
        tmp.unlink(missing_ok=True)


def load_lora_checkpoint_into_backbone(
    backbone: FluxBackbone,
    ckpt_path: Path,
) -> int:
    """
    Carrega os pesos LoRA de um checkpoint de treino (step_*.pt) no transformer
    do backbone (strict shape check). NÃO toca optimizer/scheduler — uso fora do
    loop de treino (ex.: exportação). Retorna o global_step do checkpoint.
    """
    payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    lora_state = payload["lora_state"]
    transformer_params = dict(backbone.transformer.named_parameters())

    copied = 0
    for name, tensor in lora_state.items():
        target = transformer_params.get(name)
        if target is None:
            continue
        if target.shape != tensor.shape:
            raise RuntimeError(
                f"Shape mismatch ao carregar LoRA: {name} "
                f"ckpt={tuple(tensor.shape)} model={tuple(target.shape)}"
            )
        target.data.copy_(tensor.to(target.device, target.dtype))
        copied += 1

    missing = [
        n for n, p in transformer_params.items()
        if p.requires_grad and n not in lora_state
    ]
    if missing:
        raise RuntimeError(
            f"Checkpoint não tem todos os LoRA params: missing={missing[:5]}..."
        )
    print(f"[export] {copied} tensores LoRA carregados de {ckpt_path}")
    return int(payload.get("train_state", {}).get("global_step", 0))


def maybe_resume_checkpoint(
    *,
    output_dir: Path,
    stage: StageName,
    backbone: FluxBackbone,
    optimizer: torch.optim.Optimizer,
    scheduler: WarmupCosineScheduler,
) -> TrainState:
    """Carrega último checkpoint, se existir. Strict shape check."""
    ckpt_dir = _checkpoint_dir(output_dir, stage)
    checkpoints = _sorted_checkpoints(ckpt_dir)
    if not checkpoints:
        return TrainState()

    latest = checkpoints[-1]
    print(f"[resume] Carregando checkpoint: {latest}")
    payload = torch.load(latest, map_location="cpu", weights_only=False)

    lora_state = payload["lora_state"]
    transformer_params = dict(backbone.transformer.named_parameters())
    missing, unexpected = [], []
    for name, tensor in lora_state.items():
        if name not in transformer_params:
            unexpected.append(name)
            continue
        target = transformer_params[name]
        if target.shape != tensor.shape:
            raise RuntimeError(
                f"Shape mismatch ao carregar LoRA: {name} "
                f"ckpt={tuple(tensor.shape)} model={tuple(target.shape)}"
            )
        target.data.copy_(tensor.to(target.device, target.dtype))

    for name in transformer_params:
        if transformer_params[name].requires_grad and name not in lora_state:
            missing.append(name)

    if missing:
        raise RuntimeError(
            f"Checkpoint não tem todos os LoRA params: missing={missing[:5]}..."
        )
    if unexpected:
        print(f"[resume] WARN: keys inesperadas no checkpoint (ignoradas): {unexpected[:5]}...")

    optimizer.load_state_dict(payload["optimizer"])
    scheduler.load_state_dict(payload["scheduler"])
    s = payload.get("train_state", {})
    return TrainState(
        global_step=int(s.get("global_step", 0)),
        best_loss=float(s.get("best_loss", math.inf)),
    )


# =============================================================================
# Loop de treino
# =============================================================================

def _set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    import numpy as np
    np.random.seed(seed)


def _cycle(loader: Iterable[dict[str, Any]]) -> Iterator[dict[str, Any]]:
    while True:
        for batch in loader:
            yield batch


def _stage_step_count(stage: StageName, config: TrainConfig) -> int:
    if stage == "deblur":
        return config.data.deblur.steps
    if stage == "bokeh":
        if config.data.bokeh is None:
            raise ValueError("config.data.bokeh não definido (falta o bloco `data.bokeh:` no YAML).")
        return config.data.bokeh.steps
    raise NotImplementedError(f"Stage {stage} não implementado.")


def _prepare_runtime_artifacts(config: TrainConfig, output_dir: Path) -> str:
    output_dir.mkdir(parents=True, exist_ok=True)
    cfg_hash = config_hash(config)
    write_effective_config(config, output_dir / "effective_config.yaml")
    metadata = {
        "config_hash": cfg_hash,
        "git_commit": _get_git_commit(),
        "seed": config.runtime.seed,
    }
    (output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=True, indent=2),
        encoding="utf-8",
    )
    return cfg_hash


def _make_train_batch(
    model: nn.Module,
    stage: StageName,
    batch: dict[str, Any],
    occlusion_lambda: float = 0.0,
    occlusion_pool: str = "max",
    occlusion_theta: float = 0.0,
    geo_branches: bool = False,
):
    """Despacha para o método de treino do wrapper (DeblurNet ou BokehNet)."""
    if stage == "deblur":
        return model.make_train_batch(
            blurry_image=batch["blurry_image"],
            sharp_image=batch["aif_image"],   # par deblur: image_focus = sharp/AIF
        )
    if stage == "bokeh":
        # Contrato do batch de bokeh (o dataloader entrega estas chaves):
        #   aif_image    (B, 3, H, W) em [-1, 1]  — entrada, all-in-focus
        #   bokeh_image  (B, 3, H, W) em [-1, 1]  — alvo (x_0 do flow)
        #   defocus_map  (B, 3, H, W) em [ 0, 1]  — PRÉ-COMPUTADO (coluna defocus_map/65535)
        #   geo_map      (B, 6, H, W) em [ 0, 1]  — OPCIONAL, condicionamento
        #                geométrico. Ausente = comportamento anterior, idêntico.
        return model.make_train_batch(
            aif_image=batch["aif_image"],
            target_image=batch["bokeh_image"],
            defocus_map=batch["defocus_map"],
            geo_map=batch.get("geo_map"),
            geo_branches=geo_branches,
            occlusion_lambda=occlusion_lambda,
            occlusion_pool=occlusion_pool,
            occlusion_theta=occlusion_theta,
        )
    raise NotImplementedError(f"Stage {stage} não implementado.")


def _train_loop(
    *,
    stage: StageName,
    config: TrainConfig,
    model: nn.Module,           # DeblurNet ou BokehNet
    backbone: FluxBackbone,     # acesso direto ao backbone
    dataloader: Iterable[dict[str, Any]],
    output_dir: Path,
    init_lora_path: Path | None = None,  # Fase 2 do bokeh: inicia o LoRA da fase 1
) -> Path:
    """
    Loop de treino. Separação de responsabilidades:
      - model.make_train_batch(batch) → TrainBatchOutputs(prediction, target)
      - flow_matching_loss(pred, target) → scalar
      - accelerator.accumulate(transformer) para gradient accum + DDP sync.
    """
    _set_seed(config.runtime.seed)

    accelerator = _build_accelerator(config)

    optimizer = _make_optimizer(config, params=backbone.trainable_parameters())
    scheduler = _make_scheduler(
        config, optimizer=optimizer, total_steps=_stage_step_count(stage, config)
    )

    # IMPORTANTE — multi-GPU NÃO usa DDP-wrap. O forward dos autores
    # (transformer_forward) acessa submódulos direto (self.x_embedder, etc.), o
    # que o wrapper DistributedDataParallel não expõe (AttributeError). Além
    # disso, como esse forward não passa pelo .forward() do módulo, o DDP nem
    # sincronizaria os gradientes. Então:
    #   - single-GPU: prepara o transformer (no-op/identidade) + optimizer (igual já validado).
    #   - multi-GPU: mantém o transformer CRU e sincroniza os grads MANUALMENTE
    #     (all-reduce AVG nos trainable params, no step de sync — ver loop abaixo).
    if accelerator.num_processes > 1:
        optimizer = accelerator.prepare(optimizer)
        transformer_prepared = backbone.transformer  # cru: transformer_forward acessa self.x_embedder
    else:
        transformer_prepared, optimizer = accelerator.prepare(backbone.transformer, optimizer)
        backbone.transformer = transformer_prepared

    # Só o rank principal loga (senão cada GPU abre um run no wandb).
    logger = (
        ExperimentLogger(config=config, output_dir=output_dir, stage=stage)
        if accelerator.is_main_process
        else None
    )
    cfg_hash = _prepare_runtime_artifacts(config, output_dir)

    state = TrainState()
    if config.runtime.resume:
        state = maybe_resume_checkpoint(
            output_dir=output_dir,
            stage=stage,
            backbone=backbone,
            optimizer=optimizer,
            scheduler=scheduler,
        )

    # Fase 2 do BokehNet (currículo do paper: sintético -> real). SÓ quando o run
    # começa do ZERO (sem checkpoint próprio pra resumir): carrega os pesos LoRA
    # da fase 1 no backbone, mantendo optimizer/scheduler/step FRESCOS (nova fase,
    # novo warmup+cosine sobre os steps da fase 2, 60K real por §4.1). Num re-`sbatch` da fase 2, o resume
    # acima já pegou o checkpoint da fase 2 (state.global_step>0) e aqui é pulado.
    if init_lora_path is not None and state.global_step == 0:
        loaded_step = load_lora_checkpoint_into_backbone(backbone, init_lora_path)
        print(f"[train] fase 2: LoRA inicializado da fase 1 ({init_lora_path}, step {loaded_step}).")

    # Multi-GPU sem DDP: replicamos MANUALMENTE o que o DDP faria —
    #   (1) broadcast dos pesos treináveis do rank 0 -> todos (todos começam
    #       IDÊNTICOS; o LoRA é init gaussiano ALEATÓRIO, então sem isso cada
    #       GPU começaria com pesos diferentes e os modelos divergiriam);
    #   (2) all-reduce dos gradientes no backward (ver loop abaixo).
    # (no resume, os ranks já carregam o mesmo checkpoint; o broadcast é
    #  redundante mas inofensivo e garante consistência.)
    if accelerator.num_processes > 1:
        for p in backbone.trainable_parameters():
            torch.distributed.broadcast(p.data, src=0)
        accelerator.wait_for_everyone()

    # Multi-GPU: shardeia o dataloader entre os processos (cada GPU vê um pedaço
    # diferente dos dados via DistributedSampler). Em single-GPU mantemos o
    # loader cru — caminho idêntico ao já validado.
    if accelerator.num_processes > 1:
        dataloader = accelerator.prepare(dataloader)

    iterator = _cycle(dataloader)
    backbone.transformer.train()

    # EMA da loss: em flow matching a loss por step é MUITO ruidosa (sigma e ruído
    # sorteados a cada step), então a loss bruta sobe/desce mesmo quando o modelo
    # está aprendendo. A EMA suaviza isso e deixa clara a tendência de queda —
    # essencial para validar o overfit no Wandb.
    ema_loss: float | None = None
    ema_beta = 0.98

    total_steps = _stage_step_count(stage, config)
    batch_efetivo = config.runtime.gradient_accumulation_steps * accelerator.num_processes
    print(
        f"[train] stage={stage} steps={total_steps} start={state.global_step} "
        f"num_gpus={accelerator.num_processes} grad_accum={config.runtime.gradient_accumulation_steps} "
        f"batch_efetivo={batch_efetivo}"
    )

    while state.global_step < total_steps:
        batch = next(iterator)

        with accelerator.accumulate(transformer_prepared):
            device = backbone.device
            batch_on_device = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }

            # autocast explícito: o transformer_forward do Genfocus é uma função
            # custom que NÃO passa pelo forward() do módulo prepared, então o
            # autocast que o accelerate injeta no módulo é IGNORADO. Sem este
            # wrapper, tensores float32 batem em pesos bf16 (mesmo problema que
            # o smoke evita com torch.autocast manual). accelerator.autocast()
            # aplica o mixed_precision configurado.
            with accelerator.autocast():
                outputs = _make_train_batch(
                    model=model, stage=stage, batch=batch_on_device,
                    occlusion_lambda=config.runtime.occlusion_lambda,
                    occlusion_pool=config.runtime.occlusion_pool,
                    occlusion_theta=config.runtime.occlusion_theta,
                    geo_branches=config.runtime.geo_branches,
                )
                loss = flow_matching_loss(
                    outputs.prediction, outputs.target, outputs.weight
                )

            accelerator.backward(loss)

            if accelerator.sync_gradients:
                # Multi-GPU sem DDP: sincroniza os grads acumulados entre as GPUs
                # manualmente (média). Cada processo acumulou sobre o SEU shard;
                # o all-reduce AVG dá o gradiente data-parallel correto.
                if accelerator.num_processes > 1:
                    for p in backbone.trainable_parameters():
                        if p.grad is not None:
                            torch.distributed.all_reduce(
                                p.grad, op=torch.distributed.ReduceOp.AVG
                            )

                accelerator.clip_grad_norm_(
                    backbone.trainable_parameters(),
                    max_norm=config.runtime.max_grad_norm,
                )

            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        # scheduler.step e contagem de step só nas iterações onde houve sync.
        if accelerator.sync_gradients:
            state.global_step += 1
            scheduler.step(state.global_step)

            loss_val = float(loss.detach().item())
            if loss_val < state.best_loss:
                state.best_loss = loss_val

            ema_loss = loss_val if ema_loss is None else ema_beta * ema_loss + (1.0 - ema_beta) * loss_val

            if accelerator.is_main_process and state.global_step % config.runtime.log_every_steps == 0:
                # Pico de VRAM (GB) — pra saber quão perto de 80GB estamos
                # (importante com gradient_checkpointing off). max_memory_allocated
                # é o pico desde o início; resetamos pra ver o pico por janela.
                peak_gb = torch.cuda.max_memory_allocated() / 1e9
                torch.cuda.reset_peak_memory_stats()
                lr_now = float(optimizer.param_groups[0]["lr"])
                logger.log(
                    {
                        f"{stage}/loss": loss_val,
                        f"{stage}/loss_ema": ema_loss,
                        f"{stage}/lr": lr_now,
                        f"{stage}/best_loss": state.best_loss,
                        f"{stage}/peak_vram_gb": peak_gb,
                    },
                    step=state.global_step,
                )
                # Print no stdout (.out) pra acompanhar no terminal ao vivo.
                print(
                    f"[train] step={state.global_step}/{total_steps} "
                    f"loss={loss_val:.4f} ema={ema_loss:.4f} lr={lr_now:.2e} "
                    f"vram_pico={peak_gb:.1f}GB/80GB",
                    flush=True,
                )

            if accelerator.is_main_process and state.global_step % config.runtime.save_every_steps == 0:
                save_checkpoint(
                    output_dir=output_dir,
                    stage=stage,
                    backbone=backbone,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    state=state,
                    cfg_hash=cfg_hash,
                    metrics_snapshot={"loss": loss_val, "best_loss": state.best_loss},
                    keep_last_n=config.runtime.keep_last_n_checkpoints,
                )

            # Upload periódico de SNAPSHOT pro HF (repo por-step), pra comparar
            # depois quanto melhora com mais steps. Não-fatal.
            if (
                accelerator.is_main_process
                and config.logging.upload_every_steps
                and config.logging.upload_hf_repo_base
                and state.global_step % config.logging.upload_every_steps == 0
            ):
                upload_lora_snapshot_to_hf(
                    backbone, stage,
                    config.logging.upload_hf_repo_base, state.global_step,
                    output_dir=output_dir,
                )

    # Save final SÓ no rank principal — em multi-GPU, todos os ranks têm pesos
    # idênticos (broadcast+all-reduce), mas escrever o mesmo arquivo de 2+
    # processos ao mesmo tempo corrompe. Os outros ranks esperam no
    # wait_for_everyone abaixo até o principal terminar de salvar.
    last_ckpt = None
    if accelerator.is_main_process:
        last_ckpt = save_checkpoint(
            output_dir=output_dir,
            stage=stage,
            backbone=backbone,
            optimizer=optimizer,
            scheduler=scheduler,
            state=state,
            cfg_hash=cfg_hash,
            metrics_snapshot={"loss": state.best_loss, "best_loss": state.best_loss},
            keep_last_n=config.runtime.keep_last_n_checkpoints,
        )
        sf_path = output_dir / stage / f"{stage}.safetensors"
        export_lora_safetensors(backbone, sf_path, stage)
        print(f"[train] Final checkpoint: {last_ckpt}")
        print(f"[train] Final safetensors: {sf_path}")
        if logger is not None:
            logger.finish()

    accelerator.wait_for_everyone()
    return last_ckpt


# =============================================================================
# Entrypoints públicos
# =============================================================================

def _run_stage(
    stage: StageName,
    stage_cfg: StageConfig,
    config: TrainConfig,
    backbone: FluxBackbone,
    model: nn.Module,
    output_dir: Path,
    init_lora_path: Path | None = None,
) -> Path:
    """Monta dataset + loader e roda o loop. Comum a deblur e bokeh."""
    dataset = build_dataset(
        stage=stage,
        stage_config=stage_cfg,
        # augment=False (ex. overfit) → crop central determinístico, sem flip.
        runtime=DatasetRuntimeConfig(
            image_size=stage_cfg.image_size,
            train=stage_cfg.augment,
            defocus_source=stage_cfg.defocus_source,
            max_coc=config.model.max_coc,
            min_calibration_ssim=stage_cfg.min_calibration_ssim,
            kfix_repo=stage_cfg.kfix_repo,
            # Estes DOIS sites sao duplicados (treino e smoke). Esquecer um faz o
            # smoke passar e o treino divergir, ou o contrario. Ha teste.
            geo_condition=stage_cfg.geo_condition,
            geo_escalares=stage_cfg.geo_escalares,
            geo_constantes=stage_cfg.geo_constantes,
            geo_field=stage_cfg.geo_field,
            geo_sem_escalares=stage_cfg.geo_sem_escalares,
            geo_ruido_controle=stage_cfg.geo_ruido_controle,
        ),
    )
    loader = build_dataloader(
        dataset,
        batch_size=stage_cfg.batch_size,
        num_workers=config.runtime.num_workers,
        shuffle=stage_cfg.shuffle,
        pin_memory=config.runtime.pin_memory,
    )
    return _train_loop(
        stage=stage,
        config=config,
        model=model,
        backbone=backbone,
        dataloader=loader,
        output_dir=output_dir,
        init_lora_path=init_lora_path,
    )


def run_deblur_stage(
    config: TrainConfig,
    backbone: FluxBackbone,
    model: DeblurNet,
    output_dir: Path,
) -> Path:
    """Treina DeblurNet (Stage 1)."""
    return _run_stage("deblur", config.data.deblur, config, backbone, model, output_dir)


def run_bokeh_stage(
    config: TrainConfig,
    backbone: FluxBackbone,
    model: nn.Module,     # BokehNet
    output_dir: Path,
    init_lora_path: Path | None = None,
) -> Path:
    """Treina BokehNet (Stage 2).

    init_lora_path: caminho de um step_*.pt da FASE 1 (sintético). Quando dado e o
    run começa do zero, inicializa o LoRA com esses pesos (currículo do paper).
    """
    if config.data.bokeh is None:
        raise ValueError("config.data.bokeh não definido (falta o bloco `data.bokeh:` no YAML).")
    return _run_stage(
        "bokeh", config.data.bokeh, config, backbone, model, output_dir,
        init_lora_path=init_lora_path,
    )


# =============================================================================
# Smoke test (caminho REAL, com FLUX, com 3 steps)
# =============================================================================

def run_smoke_test(
    config: TrainConfig,
    backbone: FluxBackbone,
    model: nn.Module,          # DeblurNet ou BokehNet
    output_dir: Path,
    stage: StageName = "deblur",
) -> None:
    """
    Smoke test: 3 steps no caminho real (FLUX + LoRA + transformer_forward).

    Stage-aware:
      - "deblur": 1 condição (blurry). Caminho já validado do Stage 1.
      - "bokeh":  2 condições (AIF + defocus map). Exercita o caminho de 2
        condições ANTES de gastar 40K/60K steps de cluster — pega bugs de
        shape do group_mask (diag entre condições), do dataloader de bokeh
        (defocus [0,1], 3 canais) e do forward de 2 conds cedo.

    Critérios de sucesso:
      1. make_train_batch retorna tensores com shapes esperados.
      2. Loss é numérica e finita.
      3. backward() roda sem erro.
      4. optimizer.step() atualiza pesos LoRA (max-diff > 1e-8).
      5. Salva LoRA .safetensors sem erro.

    Roda SEM accelerate (autocast manual) para isolar o caminho de FLUX puro.
    """
    print("=" * 70)
    print(f"SMOKE TEST ({stage}) — 3 steps no caminho real (FLUX + LoRA)")
    print("=" * 70)

    _set_seed(config.runtime.seed)

    if stage == "deblur":
        stage_cfg = config.data.deblur
    elif stage == "bokeh":
        if config.data.bokeh is None:
            raise ValueError(
                "config.data.bokeh não definido (falta o bloco `data.bokeh:` no "
                "YAML). Para smoke de bokeh use configs/train_bokeh_smoke.yaml."
            )
        stage_cfg = config.data.bokeh
    else:
        raise NotImplementedError(f"Stage {stage} não suportado no smoke.")

    dataset = build_dataset(
        stage=stage,
        stage_config=stage_cfg,
        # train=False → crop central determinístico (smoke reprodutível)
        runtime=DatasetRuntimeConfig(
            image_size=stage_cfg.image_size,
            train=False,
            defocus_source=stage_cfg.defocus_source,
            max_coc=config.model.max_coc,
            min_calibration_ssim=stage_cfg.min_calibration_ssim,
            kfix_repo=stage_cfg.kfix_repo,
            # Estes DOIS sites sao duplicados (treino e smoke). Esquecer um faz o
            # smoke passar e o treino divergir, ou o contrario. Ha teste.
            geo_condition=stage_cfg.geo_condition,
            geo_escalares=stage_cfg.geo_escalares,
            geo_constantes=stage_cfg.geo_constantes,
            geo_field=stage_cfg.geo_field,
            geo_sem_escalares=stage_cfg.geo_sem_escalares,
            geo_ruido_controle=stage_cfg.geo_ruido_controle,
        ),
    )
    if len(dataset) == 0:
        raise RuntimeError(
            f"Dataset vazio para os sources: {[s.name + ':' + s.split for s in stage_cfg.datasets]}"
        )

    loader = build_dataloader(
        dataset,
        batch_size=1,
        num_workers=0,
        shuffle=False,
        pin_memory=False,
    )

    optimizer = _make_optimizer(config, params=backbone.trainable_parameters())

    backbone.transformer.train()
    iterator = iter(loader)
    initial_param = backbone.trainable_parameters()[0].detach().clone()

    autocast_dtype = _autocast_dtype(config)
    autocast_ctx = (
        torch.autocast(device_type="cuda", dtype=autocast_dtype)
        if autocast_dtype is not None
        else torch.autocast(device_type="cuda", enabled=False)
    )

    for step in range(3):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)

        device = backbone.device
        batch_on_device = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }

        with autocast_ctx:
            # despacha pro make_train_batch certo (deblur=1 cond, bokeh=2 conds).
            outputs = _make_train_batch(model=model, stage=stage, batch=batch_on_device)
            loss = flow_matching_loss(
                outputs.prediction, outputs.target, outputs.weight
            )

        if not torch.isfinite(loss):
            raise RuntimeError(f"Smoke step {step}: loss não-finita: {loss.item()}")

        loss.backward()
        torch.nn.utils.clip_grad_norm_(backbone.trainable_parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()

        print(f"[smoke] step={step + 1}/3 loss={loss.item():.6f}")

    final_param = backbone.trainable_parameters()[0].detach()
    diff = (final_param - initial_param).abs().max().item()
    if diff < 1e-8:
        raise RuntimeError(
            f"Smoke test: parâmetros LoRA NÃO mudaram após 3 steps "
            f"(max diff={diff}). Algo no pipeline de gradiente está errado."
        )

    # Exporta safetensors do smoke também (valida que o caminho de export funciona)
    sf_path = output_dir / "smoke.safetensors"
    sf_path.parent.mkdir(parents=True, exist_ok=True)
    export_lora_safetensors(backbone, sf_path, stage=stage)

    print(f"[smoke] PASSOU. Param diff após 3 steps: {diff:.2e}")
    print(f"[smoke] Safetensors exportado: {sf_path}")
    print("=" * 70)
