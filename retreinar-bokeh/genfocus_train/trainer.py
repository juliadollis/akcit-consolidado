"""
Training runtime da DeblurNet / BokehNet (GenRefocus).

ÁRVORE NOVA (retreinar-deblur/). O treino antigo segue intacto em
`genrefocus_deblurnet_paper/` e `genrefocus_deblurnet/`.

O que este arquivo garante (e por quê):
  - **Nada se perde**: checkpoint a cada `save_every_steps`, `best.pt` que NUNCA
    entra na poda, `latest.json`, salvamento final, e upload pro Hugging Face
    periódico + melhor + final. Falha de upload nunca derruba o treino.
  - **A curva se lê**: wandb com id estável, então re-`sbatch` CONTINUA o mesmo
    run em vez de abrir um novo e picar o gráfico.
  - **O checkpoint diz o que ele é**: o metadata carrega tudo que muda o modelo
    (variante do LoRA, guidance de treino, eixos do experimento, prompt), e um
    `.json` irmão acompanha todo `.safetensors` exportado.
  - **A validação é sinal, não ruído**: sigma e ruído determinísticos, então
    `val_loss` é comparável entre steps.

Correções desta revisão, com a referência do plano:
  C3  warmup aplicado ANTES do primeiro update, e índice 0-based corrigido.
  C7  seed por rank (senão as N GPUs sorteiam o mesmo sigma e o mesmo ruído).
  C8  metadata completo no checkpoint + sidecar .json + resume que RECUSA
      checkpoint de arquitetura pré-C1 em vez de descartar peso em silêncio.
  C10 validação determinística e seleção de checkpoint por ela.

Preservado do original porque está certo:
  - multi-GPU SEM DDP-wrap (o `transformer_forward` dos autores acessa
    submódulos direto), com broadcast inicial + all-reduce manual dos grads;
  - `accelerator.autocast()` explícito (o forward custom não passa pelo
    `.forward()` do módulo preparado);
  - contagem de step e `scheduler.step` guardados por `sync_gradients`.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import os
import random
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Literal

import numpy as np
import torch
from torch import nn

from .backbone import FluxBackbone
from . import control as _control
from .config import (
    StageConfig,
    TrainConfig,
    config_hash,
    stage_config,
    write_effective_config,
)
from .data import DatasetRuntimeConfig, build_dataloader, build_dataset
from .models import flow_matching_loss

StageName = Literal["deblur", "bokeh", "bokeh_shape"]


# =============================================================================
# Accelerator — sem fallback. Se não tiver, falha.
# =============================================================================

class ParadaPorControlabilidade(RuntimeError):
    """O probe do T11 disparou o critério de parada declarado no plano.

    NÃO é falha de infraestrutura: é o experimento dizendo que o modelo está
    perdendo controle de K. O checkpoint mais recente está no disco; o run para
    para que a causa seja investigada em vez de gastar o resto do orçamento.
    """


def _plugin_de_acumulacao(config: TrainConfig):
    """`sync_with_dataloader=False` — o que fecha a sobra de fim de época.

    DEFEITO CORRIGIDO (revisão externa). Por default o Accelerate força um
    `sync_gradients` quando o dataloader ACABA, mesmo no meio de uma janela de
    acumulação. Como `_cycle` reitera o loader a cada época, toda volta
    terminava com um `optimizer.step()` de um grupo INCOMPLETO — normalizado
    como se fosse cheio.

    Efeito: um step a cada época com batch efetivo menor que 32 e gradiente
    com peso errado. Não é ruído: é um viés sistemático, uma vez por época, e
    com ~26K amostras e batch 32 são centenas de steps assim num run de 60K.

    Com `sync_with_dataloader=False` a janela atravessa a fronteira de época e
    TODO step usa exatamente `gradient_accumulation_steps` microbatches — que é
    o que o §4.1 descreve ("gradient accumulation over 8 steps").


    NÃO VERIFICADO EM EXECUÇÃO: `accelerate` não estava instalado no ambiente em
    que isto foi escrito. Por isso a construção é defensiva — se a versão
    instalada não aceitar `sync_with_dataloader`, o treino NÃO quebra: cai no
    caminho antigo e GRITA, para que a sobra de época seja uma escolha visível
    em vez de um silêncio.
    """
    try:
        from accelerate.utils import GradientAccumulationPlugin
    except ImportError:
        return None

    accum = int(config.runtime.gradient_accumulation_steps)
    try:
        return GradientAccumulationPlugin(
            num_steps=accum,
            sync_with_dataloader=False,
            adjust_scheduler=False,   # o scheduler é nosso, por step global
        )
    except TypeError as exc:  # versão sem um dos parâmetros
        print(
            f"[train] AVISO GRAVE: GradientAccumulationPlugin não aceitou "
            f"`sync_with_dataloader` nesta versão do accelerate ({exc}). O treino "
            "vai rodar com a SOBRA DE ÉPOCA: um step por época com batch efetivo "
            "menor que o configurado. Atualize o accelerate ou aceite o desvio.",
            flush=True,
        )
        return None


def _conferir_memoria_compartilhada(config: TrainConfig) -> None:
    """`num_workers > 0` sem `/dev/shm` grande mata o treino com `Bus error`.

    Os workers do DataLoader passam tensores por memória compartilhada. O
    default do Docker é **64 MB**, e com `pin_memory` e imagens de 512 px isso
    estoura. A falha é das piores possíveis:

      RuntimeError: DataLoader worker (pid ...) is killed by signal: Bus error.

    — e ela só aparece DEPOIS de carregar o FLUX (minutos), vem de um processo
    filho, e o pai pode terminar com status que parece limpo. Medido duas vezes
    em 2026-09-25, contra a mesma árvore de código que tinha acabado de rodar
    40.000 steps sem um arranhão: o container da fase 1 subiu com
    `--shm-size=64g --ipc=host` e os de verificação, não.

    O diagnóstico custa dez minutos de GPU; a checagem custa um `statvfs`.
    """
    import shutil

    workers = int(config.runtime.num_workers)
    if workers <= 0:
        return
    shm = Path("/dev/shm")
    if not shm.is_dir():
        return
    total_gb = shutil.disk_usage(shm).total / 1e9
    if total_gb >= 1.0:
        return
    raise RuntimeError(
        f"/dev/shm tem só {total_gb:.2f} GB e `num_workers={workers}`. Os "
        "workers do DataLoader trocam tensores por memória compartilhada: com "
        "o default de 64 MB do Docker eles morrem com\n"
        "  RuntimeError: DataLoader worker ... killed by signal: Bus error\n"
        "depois de o FLUX já ter carregado, num processo filho, às vezes com "
        "status de saída limpo.\n\n"
        "No `docker run`, acrescente:  --shm-size=64g --ipc=host\n"
        "(foi assim que a fase 1 de 40K steps rodou).\n"
        "Sem Docker e sem como aumentar, use `num_workers: 0` — mais lento, "
        "mas não morre."
    )


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

        kwargs_handlers.append(InitProcessGroupKwargs(timeout=timedelta(minutes=90)))
    except Exception:  # pragma: no cover - versões antigas do accelerate
        pass

    plugin = _plugin_de_acumulacao(config)
    if plugin is not None:
        return Accelerator(
            # substitui o `gradient_accumulation_steps` e acrescenta
            # `sync_with_dataloader=False` — ver `_plugin_de_acumulacao`.
            gradient_accumulation_plugin=plugin,
            mixed_precision=config.runtime.mixed_precision,
            kwargs_handlers=kwargs_handlers,
        )
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


def _stage_prompt(stage: str) -> str | None:
    """Prompt fixo do estágio, para registrar no metadata.

    Import TARDIO de propósito: `train.py` importa este módulo, então um import
    no topo criaria ciclo. A fonte da verdade continua sendo uma só
    (`train.STAGE_PROMPTS`) — duplicar a string aqui é como treino e inferência
    divergem sem ninguém perceber.
    """
    try:
        from .train import STAGE_PROMPTS

        return STAGE_PROMPTS.get(stage)
    except Exception:  # pragma: no cover - só afeta o metadata, não o treino
        return None


# =============================================================================
# Logger
# =============================================================================

def _wandb_run_id(output_dir: Path, stage: StageName, config: TrainConfig) -> str:
    """Id ESTÁVEL do run do wandb.

    Sem isto, cada re-`sbatch` abre um run novo e a curva de loss do treino fica
    picada em N pedaços que não se comparam. O id é derivado do destino do
    treino (output_dir + stage), então o mesmo treino sempre reencontra o mesmo
    run, e é persistido em disco para sobreviver a mudanças de caminho relativo.
    """
    if config.logging.wandb_id:
        return str(config.logging.wandb_id)

    id_path = output_dir / stage / "wandb_id.txt"
    if id_path.is_file():
        gravado = id_path.read_text(encoding="utf-8").strip()
        if gravado:
            return gravado

    semente = f"{output_dir.resolve()}::{stage}"
    novo = hashlib.sha1(semente.encode("utf-8")).hexdigest()[:8]
    try:
        id_path.parent.mkdir(parents=True, exist_ok=True)
        id_path.write_text(novo, encoding="utf-8")
    except Exception as exc:  # pragma: no cover
        print(f"[logger] não consegui persistir o wandb id ({exc}); seguindo assim mesmo.")
    return novo


class ExperimentLogger:
    """Log para JSONL local + wandb (opcional)."""

    def __init__(
        self,
        config: TrainConfig,
        output_dir: Path,
        stage: StageName,
        extra_config: dict[str, Any] | None = None,
    ) -> None:
        self.output_dir = output_dir
        self.stage = stage
        self.metrics_path = output_dir / stage / "metrics.jsonl"
        self.metrics_path.parent.mkdir(parents=True, exist_ok=True)

        cfg_hash_short = config_hash(config)[:8]

        self._wandb_run = None
        if config.logging.use_wandb:
            try:
                import wandb

                # O `config` do run carrega os EIXOS DO EXPERIMENTO. É o que
                # permite comparar os braços do fatorial (scale_mode ×
                # sigma_mu_source, guidance, top_k_mode) direto na interface do
                # wandb, em vez de ter que abrir o YAML de cada job.
                run_config = {"stage": stage, "config_hash": cfg_hash_short}
                run_config.update(extra_config or {})

                init_kwargs: dict[str, Any] = dict(
                    project=config.logging.wandb_project,
                    entity=config.logging.wandb_entity,
                    name=config.logging.run_name or f"{stage}-{cfg_hash_short}",
                    config=run_config,
                )
                if config.logging.wandb_resume:
                    init_kwargs["id"] = _wandb_run_id(output_dir, stage, config)
                    init_kwargs["resume"] = "allow"

                self._wandb_run = wandb.init(**init_kwargs)
                print(
                    f"[logger] wandb: project={config.logging.wandb_project} "
                    f"id={init_kwargs.get('id', '(auto)')} resume={config.logging.wandb_resume}"
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
            try:
                self._wandb_run.log(payload, step=step)
            except Exception as exc:  # pragma: no cover - log nunca derruba treino
                print(f"[logger] wandb.log falhou ({exc}); seguindo com JSONL.")

    def finish(self) -> None:
        if self._wandb_run is not None:
            try:
                self._wandb_run.finish()
            except Exception:  # pragma: no cover
                pass


# =============================================================================
# Estado e otimizador
# =============================================================================

@dataclass
class TrainState:
    global_step: int = 0
    best_loss: float = math.inf
    # Época do sampler, para a retomada não recomeçar da permutação 0.
    epoca: int = 0


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
    """Warmup linear + decaimento cosseno.

    C3 — `_lr_at` recebe `step` 0-BASED, contando updates JÁ ocorridos.
    A versão anterior usava `float(step + 1)`, o que, somado ao fato de o
    scheduler só ser aplicado DEPOIS do primeiro `optimizer.step()`, fazia o
    primeiro update rodar com o `lr` base cheio (1e-4) em vez do primeiro passo
    do warmup (2e-7 com warmup=500), e deixava o warmup inteiro um step
    adiantado. Ver `_train_loop`, onde o `scheduler.step` inicial foi
    acrescentado antes do laço.
    """

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
            # `step` = updates JÁ concluídos; o LR calculado aqui é o do PRÓXIMO
            # update, que é o de número `step + 1` — daí o `+1`, que também faz o
            # último passo da rampa valer exatamente `base_lr` e emendar sem
            # degrau no ramo cosseno.
            #
            # CONFERIDO numericamente: com o `scheduler.step()` inicial de
            # `_train_loop`, os updates saem 2e-7, 4e-7, 6e-7, 8e-7... Trocar por
            # `max(step, 1)` REPETIRIA o primeiro valor (2e-7, 2e-7, 4e-7, ...) e
            # deixaria a rampa com 501 updates. A fórmula nunca esteve errada: o
            # bug era só ela não ser aplicada antes do primeiro update.
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
# Metadata do run (C8)
# =============================================================================

def _get_git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


def _versoes_do_ambiente() -> dict[str, str]:
    """Versões das bibliotecas que MUDAM o modelo. Falha nunca derruba o treino."""
    saida: dict[str, str] = {}
    for nome in ("torch", "diffusers", "peft", "accelerate", "transformers",
                 "datasets", "numpy"):
        try:
            saida[nome] = str(__import__(nome).__version__)
        except Exception:  # noqa: BLE001
            saida[nome] = "?"
    try:
        import torch as _t

        saida["cuda"] = str(_t.version.cuda)
    except Exception:  # noqa: BLE001
        pass
    return saida


def _commit_do_forward() -> str:
    """Commit do `third_party/Genfocus` — o `transformer_forward` é DELE.

    O forward dos autores não é nosso código e não aparece no `git_commit` do
    projeto. Se ele mudar, o modelo muda, e nada no artefato diria.
    """
    try:
        import Genfocus.pipeline.flux as _flux

        raiz = Path(_flux.__file__).resolve().parents[2]
        out = subprocess.run(
            ["git", "-C", str(raiz), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5, check=False,
        )
        if out.returncode == 0 and out.stdout.strip():
            return out.stdout.strip()
        # sem .git (o normal: a pasta vem junto no rsync) -> hash do arquivo
        import hashlib

        return "sha256:" + hashlib.sha256(
            Path(_flux.__file__).read_bytes()
        ).hexdigest()[:16]
    except Exception as exc:  # noqa: BLE001
        return f"?({type(exc).__name__})"


def build_run_metadata(
    *,
    stage: StageName,
    config: TrainConfig,
    stage_cfg: StageConfig,
    backbone: FluxBackbone,
    num_processes: int = 1,
) -> dict[str, Any]:
    """Tudo que MUDA o modelo, num dicionário só (C8).

    Existem duas variantes de LoRA neste projeto — cond-only e main+cond — e
    elas exigem chamadas de inferência DIFERENTES: a main+cond precisa de
    `main_adapter="deblurring"`, e rodá-la com o default oficial
    (`main_adapter=None`) produz saída lavada. O repo HF `...-paper-4gpu` é
    main+cond apesar do nome. Sem este registro dentro do artefato, descobrir
    qual é qual depende de memória humana — que já falhou uma vez.
    """
    info: dict[str, Any] = {}
    try:
        info = dict(backbone.lora_info())
    except Exception as exc:  # pragma: no cover - metadata nunca derruba treino
        info = {"lora_info_erro": repr(exc)}

    accum = int(config.runtime.gradient_accumulation_steps)
    return {
        "stage": stage,
        # ── PROCEDÊNCIA (#16) ────────────────────────────────────────────────
        # Sem isto, "o checkpoint X foi treinado com que versão?" depende de
        # memória humana. E a resposta muda o modelo: o comportamento do
        # `add_adapter` do PEFT quanto a adapters ativos já mudou entre versões
        # (foi o defeito A1), e `calculate_shift` é do diffusers.
        "versoes": _versoes_do_ambiente(),
        # gravado para o guard de resume (#9) poder comparar convenções
        "defocus_source": getattr(stage_cfg, "defocus_source", None),
        "fontes_fingerprint": _fontes_fingerprint(config, stage),
        "control_version": _control.CONTROL_VERSION,
        "genfocus_commit": _commit_do_forward(),
        "git_commit": _get_git_commit(),
        "config_hash": config_hash(config),
        # ── o que define o modelo ────────────────────────────────────────────
        **info,
        # O treino NUNCA põe LoRA no branch de texto. Registrado explicitamente
        # porque `generate(main_adapter=...)` do pipeline oficial liga o adapter
        # no texto junto com o main (o índice 0 de `adapters` é o texto), e
        # reproduzir este treino exige desligar isso na inferência.
        "text_adapter": None,
        # ── eixos do experimento ─────────────────────────────────────────────
        "image_size": int(stage_cfg.image_size),
        "scale_mode": stage_cfg.scale_mode,
        "sigma_mu_source": stage_cfg.sigma_mu_source,
        "top_k_mode": stage_cfg.top_k_mode,
        "scene_key": stage_cfg.scene_key,
        "augment": bool(stage_cfg.augment),
        "prompt": _stage_prompt(stage),
        # ── escala do treino ─────────────────────────────────────────────────
        "steps_total": int(stage_cfg.steps),
        "batch_size": int(stage_cfg.batch_size),
        "gradient_accumulation_steps": accum,
        "num_gpus": int(num_processes),
        "batch_efetivo": int(stage_cfg.batch_size) * accum * int(num_processes),
        "lr": float(config.optimizer.lr),
        "warmup_steps": int(config.scheduler.warmup_steps),
        "mixed_precision": config.runtime.mixed_precision,
        "datasets": [f"{s.name}:{s.split}" for s in stage_cfg.datasets],
        "val_datasets": [f"{s.name}:{s.split}" for s in stage_cfg.val_datasets],
    }


# =============================================================================
# Checkpoint
# =============================================================================

def _checkpoint_dir(output_dir: Path, stage: StageName) -> Path:
    path = output_dir / stage / "checkpoints"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _sorted_checkpoints(path: Path) -> list[Path]:
    """Checkpoints periódicos, em ordem de step.

    Casa só `step_*.pt` de propósito: `best.pt` fica FORA desta lista e por
    isso nunca entra na poda de `keep_last_n_checkpoints`.
    """
    return sorted(path.glob("step_*.pt"), key=lambda p: int(p.stem.split("_")[-1]))


def _lora_state_dict(backbone: FluxBackbone) -> dict[str, torch.Tensor]:
    """Pesos a gravar no checkpoint.

    Regra: TODO parâmetro de LoRA, treinável ou não — e não só os treináveis.

    Antes era `if param.requires_grad`, e no §3.3 isso excluía o adapter BASE,
    que é congelado por definição. O checkpoint da fase de forma não conteria o
    modelo que faz bokeh, e retomar dele produziria forma sobre ruído gaussiano.
    Fora do §3.3 não muda nada: lá todo parâmetro de LoRA é treinável.
    """
    return {
        name: param.detach().cpu()
        for name, param in backbone.transformer.named_parameters()
        if param.requires_grad or ".lora_" in name
    }


def _params_do_adapter(backbone: FluxBackbone, adapter: str) -> list[str]:
    """Nomes dos parâmetros de UM adapter, treináveis ou não."""
    return [
        n for n, _ in backbone.transformer.named_parameters()
        if f".{adapter}." in n and ".lora_" in n
    ]


def _estado_dos_rngs() -> dict[str, Any]:
    """Estados de RNG que o treino consome. Ver `save_checkpoint`."""
    estado: dict[str, Any] = {
        "torch_cpu": torch.get_rng_state(),
        "numpy": np.random.get_state(),
        "python": random.getstate(),
    }
    if torch.cuda.is_available():
        estado["torch_cuda"] = torch.cuda.get_rng_state_all()
    return estado


def _restaurar_rngs(estado: dict[str, Any] | None) -> bool:
    """Restaura o que der; devolve se conseguiu. Nunca derruba a retomada."""
    if not estado:
        return False
    try:
        torch.set_rng_state(estado["torch_cpu"])
        np.random.set_state(estado["numpy"])
        random.setstate(estado["python"])
        if torch.cuda.is_available() and estado.get("torch_cuda") is not None:
            guardado = estado["torch_cuda"]
            if len(guardado) == torch.cuda.device_count():
                torch.cuda.set_rng_state_all(guardado)
            else:
                print(
                    f"[resume] AVISO: checkpoint tem {len(guardado)} estados de GPU "
                    f"e agora há {torch.cuda.device_count()}. RNG da GPU NÃO "
                    "restaurado — a retomada não é bit a bit."
                )
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"[resume] AVISO: não consegui restaurar o RNG ({exc}); seguindo.")
        return False


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
    run_metadata: dict[str, Any] | None = None,
    filename: str | None = None,
) -> Path:
    """Salva LoRA + optimizer + scheduler + metadata.

    `filename` fixo (ex. "best.pt") escreve fora da série `step_*.pt` e escapa
    da poda — é como o melhor checkpoint sobrevive a um treino longo.
    """
    ckpt_dir = _checkpoint_dir(output_dir, stage)
    ckpt_path = ckpt_dir / (filename or f"step_{state.global_step}.pt")

    payload = {
        "lora_state": _lora_state_dict(backbone),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "train_state": {"global_step": state.global_step, "best_loss": state.best_loss},
        # ── RETOMADA EXATA (revisão externa) ─────────────────────────────────
        # Sem isto, retomar reiniciava a sequência aleatória: o mesmo sigma e o
        # mesmo ruído do começo, e a mesma ordem de dados da época 0. Um run
        # que caiu 5 vezes via 5 vezes o mesmo começo.
        #
        # O que É restaurado: RNG do torch (CPU e todas as GPUs), do numpy, e a
        # ÉPOCA do sampler.
        # O que NÃO é, e está declarado: a posição EXATA dentro da época. Os
        # workers do DataLoader têm RNG próprio derivado no fork, e o
        # `WeightedRandomSampler` é com reposição — restaurar o índice exato
        # exigiria serializar o estado de cada worker. A época + os RNGs
        # recuperam a DISTRIBUIÇÃO, não a sequência byte a byte.
        "rng": _estado_dos_rngs(),
        "epoca": int(getattr(state, "epoca", 0)),
        "metadata": {
            **(run_metadata or {}),
            "stage": stage,
            "global_step": state.global_step,
            "config_hash": cfg_hash,
            "metrics_snapshot": metrics_snapshot,
        },
    }
    # Escrita ATÔMICA: um job morto no meio do torch.save deixaria um .pt
    # truncado que só falharia no próximo resume, horas depois.
    tmp_path = ckpt_path.with_suffix(".pt.tmp")
    torch.save(payload, tmp_path)
    os.replace(tmp_path, ckpt_path)

    if filename is None:
        checkpoints = _sorted_checkpoints(ckpt_dir)
        if keep_last_n > 0 and len(checkpoints) > keep_last_n:
            for old in checkpoints[:-keep_last_n]:
                old.unlink(missing_ok=True)

        latest = ckpt_dir / "latest.json"
        latest.write_text(
            json.dumps({"checkpoint": str(ckpt_path), "step": state.global_step}, ensure_ascii=True),
            encoding="utf-8",
        )
    return ckpt_path


def export_lora_safetensors(
    backbone: FluxBackbone,
    output_path: Path,
    stage: StageName,
    run_metadata: dict[str, Any] | None = None,
) -> Path:
    """Exporta o LoRA em .safetensors compatível com `pipe.load_lora_weights`.

    Escreve também um `.json` IRMÃO com o metadata: o artefato que sobe pro HF
    tem que dizer sozinho qual variante de LoRA ele é e com que guidance foi
    treinado, senão a informação só existe na cabeça de quem rodou.
    """
    from peft.utils import get_peft_model_state_dict
    from safetensors.torch import save_file

    output_path.parent.mkdir(parents=True, exist_ok=True)
    # §3.3: no estágio de forma o artefato é o LoRA de FORMA — o base já foi
    # exportado no fim da fase 2 e não mudou (está congelado). Exportar o base
    # aqui republicaria um arquivo idêntico com nome novo e esconderia o que
    # realmente foi treinado.
    from .backbone import ADAPTER_NAME, SHAPE_ADAPTER_NAME

    adapter = SHAPE_ADAPTER_NAME if stage == "bokeh_shape" else ADAPTER_NAME
    raw = get_peft_model_state_dict(backbone.transformer, adapter_name=adapter)

    # Normaliza para o formato diffusers que `load_lora_weights` espera e que o
    # checkpoint oficial usa: transformer.<modulo>.lora_A.weight
    lora_state: dict[str, Any] = {}
    for key, value in raw.items():
        norm = key.replace(f".{adapter}.", ".").replace(".default.", ".")
        if not norm.startswith("transformer."):
            norm = "transformer." + norm
        lora_state[norm] = value

    save_file(lora_state, str(output_path))

    sidecar = output_path.with_suffix(".json")
    try:
        sidecar.write_text(
            json.dumps(
                {**(run_metadata or {}), "n_tensores_lora": len(lora_state)},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
    except Exception as exc:  # pragma: no cover
        print(f"[export] AVISO: não consegui escrever o sidecar {sidecar}: {exc}")

    sample = list(lora_state.keys())[:3]
    print(f"[export] {len(lora_state)} tensores LoRA -> {output_path}. ex: {sample}")
    return output_path


# =============================================================================
# Hugging Face
# =============================================================================

_UPLOAD_DESLIGADO_AVISADO = False


def upload_lora_snapshot_to_hf(
    backbone: FluxBackbone,
    stage: StageName,
    repo_base: str | None,
    step: int,
    output_dir: Path,
    *,
    token: str | None = None,
    run_metadata: dict[str, Any] | None = None,
    private: bool = True,
    retries: int = 3,
    rotulo: str | None = None,
) -> bool:
    """Sobe o LoRA atual pro HF como `<stage>_<rotulo|step<N>>.safetensors`.

    UM repo, um ARQUIVO por step: o histórico do treino fica todo no mesmo
    lugar, e dá para comparar steps sem caçar repositórios.

    NUNCA derruba o treino. Rede e token falham; 60 mil steps de GPU não podem
    morrer por causa disso. Retorna True/False só para o chamador logar.
    """
    global _UPLOAD_DESLIGADO_AVISADO

    if not repo_base:
        if not _UPLOAD_DESLIGADO_AVISADO:
            print(
                "[upload] `logging.upload_hf_repo_base` não definido — upload pro HF "
                "DESLIGADO. Os checkpoints locais continuam sendo salvos."
            )
            _UPLOAD_DESLIGADO_AVISADO = True
        return False

    token = token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    nome = rotulo or f"step{step}"
    # O .safetensors temporário vai no output_dir, NÃO em /tmp: dentro do
    # container o /tmp é tmpfs (RAM) e cada snapshot tem centenas de MB.
    tmp = output_dir / stage / f".upload_{stage}_{nome}.safetensors"
    tmp_json = tmp.with_suffix(".json")

    try:
        export_lora_safetensors(backbone, tmp, stage, run_metadata=run_metadata)
    except Exception as exc:  # noqa: BLE001
        print(f"[upload] WARN: falha exportando o snapshot {nome}: {exc}")
        tmp.unlink(missing_ok=True)
        tmp_json.unlink(missing_ok=True)
        return False

    try:
        from huggingface_hub import HfApi

        repo_id = repo_base
        if "/" not in repo_id and token:
            from huggingface_hub import whoami

            repo_id = f"{whoami(token=token)['name']}/{repo_id}"

        api = HfApi(token=token)
        api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True, private=private)

        for tentativa in range(1, max(1, retries) + 1):
            try:
                api.upload_file(
                    path_or_fileobj=str(tmp),
                    path_in_repo=f"{stage}_{nome}.safetensors",
                    repo_id=repo_id,
                    repo_type="model",
                )
                if tmp_json.is_file():
                    api.upload_file(
                        path_or_fileobj=str(tmp_json),
                        path_in_repo=f"{stage}_{nome}.json",
                        repo_id=repo_id,
                        repo_type="model",
                    )
                print(
                    f"[upload] {nome} -> https://huggingface.co/{repo_id}"
                    f"/blob/main/{stage}_{nome}.safetensors"
                )
                return True
            except Exception as exc:  # noqa: BLE001
                if tentativa >= max(1, retries):
                    raise
                espera = 5 * (2 ** (tentativa - 1))  # 5s, 10s, 20s...
                print(
                    f"[upload] tentativa {tentativa}/{retries} falhou ({exc}); "
                    f"nova tentativa em {espera}s."
                )
                time.sleep(espera)
    except Exception as exc:  # noqa: BLE001 - upload nunca deve quebrar o treino
        print(f"[upload] WARN: desisti do upload de {nome}: {exc}")
        return False
    finally:
        tmp.unlink(missing_ok=True)
        tmp_json.unlink(missing_ok=True)
    return False


# =============================================================================
# Resume
# =============================================================================

def _aplicar_lora_state(
    backbone: FluxBackbone, lora_state: dict[str, torch.Tensor]
) -> tuple[list[str], list[str]]:
    """Copia os pesos LoRA in-place. Retorna (missing, unexpected)."""
    transformer_params = dict(backbone.transformer.named_parameters())
    missing, unexpected = [], []
    for name, tensor in lora_state.items():
        target = transformer_params.get(name)
        if target is None:
            unexpected.append(name)
            continue
        if target.shape != tensor.shape:
            raise RuntimeError(
                f"Shape mismatch ao carregar LoRA: {name} "
                f"ckpt={tuple(tensor.shape)} model={tuple(target.shape)}"
            )
        target.data.copy_(tensor.to(target.device, target.dtype))
    for name, param in transformer_params.items():
        if param.requires_grad and name not in lora_state:
            missing.append(name)
    return missing, unexpected


def _recusar_checkpoint_pre_c1(unexpected: list[str]) -> None:
    """C8/C1 — um checkpoint da arquitetura antiga NÃO pode passar em silêncio.

    Antes do C1, a string solta "proj_out" em `LORA_TARGET_MODULES` capturava
    também a projeção final `transformer.proj_out` (módulo de TOPO, sem ponto
    no nome), injetando 344 módulos onde o checkpoint oficial tem 343. Depois
    da correção esse peso vira `unexpected` e a versão anterior deste código
    apenas imprimia um WARN — ou seja, descartaria uma camada treinada sem
    avisar, mudando o modelo silenciosamente.
    """
    topo = [n for n in unexpected if "." not in n.split(".lora_")[0]]
    if topo:
        raise RuntimeError(
            f"Checkpoint anterior ao C1 (LoRA em módulo de topo: {topo}). "
            "Incompatível com a arquitetura corrigida — treine do zero."
        )


def load_lora_checkpoint_into_backbone(
    backbone: FluxBackbone, ckpt_path: Path, *, exigir_completo: bool = True
) -> int:
    """Carrega só os pesos LoRA (export, init de fase 2). Retorna o global_step.

    `exigir_completo=False` é o caso do §3.3: o checkpoint da fase 2 traz os
    pesos do adapter BASE (`.default.`), enquanto os treináveis do estágio de
    forma são os do adapter `shape`. A checagem de `missing` — que compara com
    "todo parâmetro treinável" — acusaria o adapter de forma inteiro como
    ausente, o que é o esperado e não é erro.

    O que se confere no lugar, e é mais forte: que TODO peso do checkpoint
    encontrou destino (`unexpected` vazio) e que o adapter base ficou de fato
    coberto. Um checkpoint que não cobre o base deixaria o "congelado" do §3.3
    congelando ruído gaussiano.
    """
    payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    lora_state = payload["lora_state"]
    missing, unexpected = _aplicar_lora_state(backbone, lora_state)
    _recusar_checkpoint_pre_c1(unexpected)
    if unexpected:
        raise RuntimeError(
            f"Checkpoint tem {len(unexpected)} pesos sem destino no modelo: "
            f"{unexpected[:5]}. Carregar parcialmente muda o modelo em silêncio."
        )
    if exigir_completo:
        if missing:
            raise RuntimeError(
                f"Checkpoint não tem todos os LoRA params: missing={missing[:5]}..."
            )
    else:
        from .backbone import ADAPTER_NAME

        # DEFEITO CORRIGIDO (revisão externa, reproduzido): antes isto filtrava
        # `missing`, que por construção só contém parâmetros TREINÁVEIS — e o
        # base do §3.3 é congelado. A lista vinha sempre vazia e um checkpoint
        # VAZIO era aceito. A checagem agora é contra os parâmetros do base que
        # existem no MODELO, não contra `missing`.
        do_base = _params_do_adapter(backbone, ADAPTER_NAME)
        if not do_base:
            raise RuntimeError(
                "Nenhum parâmetro do adapter base encontrado no modelo — o §3.3 "
                "precisa dele para congelar."
            )
        base_faltando = [n for n in do_base if n not in lora_state]
        if base_faltando:
            raise RuntimeError(
                f"O checkpoint cobre {len(do_base) - len(base_faltando)} de "
                f"{len(do_base)} parâmetros do adapter base. Faltam, por exemplo: "
                f"{base_faltando[:3]}.\n"
                "No §3.3 o base é o que fica CONGELADO E ATIVO — um base "
                "incompleto treinaria forma sobre um modelo que não faz bokeh, "
                "que é exatamente o que 'we freeze all original LoRA weights' "
                "existe para impedir."
            )
    print(f"[ckpt] {len(lora_state)} tensores LoRA carregados de {ckpt_path}")
    return int(payload.get("train_state", {}).get("global_step", 0))


def _fontes_fingerprint(config: TrainConfig, stage: StageName) -> str | None:
    """Impressão digital da COMPOSIÇÃO DE DADOS de um estágio.

    Existe porque o `stage` não distingue as fases do currículo: a fase 1
    (sintética, rota A) e a fase 2 (real, rotas B+C) são as duas `bokeh`. O que
    muda entre elas são as FONTES — e é isso que não pode ser atravessado por
    um `resume`.

    Inclui as fontes e os filtros que mudam QUAIS amostras entram. NÃO inclui
    `steps`, `lr`, upload nem wandb: mudar esses num re-`sbatch` é legítimo e
    não deve bloquear a retomada.
    """
    try:
        cfg = stage_config(config, stage)
    except ValueError:
        return None
    partes = [f"{s.name}:{s.split}" for s in cfg.datasets]
    partes.append(f"defocus={cfg.defocus_source}")
    partes.append(f"ssim={cfg.min_calibration_ssim}")
    partes.append(f"teto={cfg.max_levels_per_scene}")
    partes.append(f"replay={cfg.synthetic_replay_fraction}")
    return "|".join(partes)


def _verificar_compatibilidade_do_checkpoint(
    payload: dict, *, stage: StageName, config: TrainConfig, caminho: Path
) -> None:
    """Recusa retomar de um checkpoint que não é deste treino (#9).

    O `_recusar_checkpoint_pre_c1` cobria só a ARQUITETURA (LoRA em módulo de
    topo). Faltavam três perguntas que mudam o modelo tanto quanto ela, e cujas
    respostas já estão gravadas no metadata desde o C8 — só ninguém as lia:

      fase            retomar a fase 2 de um checkpoint da fase 1 pularia o
                      `--init-lora` e herdaria optimizer e scheduler da fase
                      errada, quando o §3.2(a) chama a sintética de "pretrain"
                      (pré-treino SEGUIDO de treino, não continuação);
      convenção       um checkpoint treinado em `recompute`/`kfix` retomado sob
                      `metric_disparity` mistura duas convenções do mapa no
                      MESMO run — que é literalmente o defeito do `kfix`;
      configuração    `config_hash` diferente significa que algum eixo mudou;
                      não é erro por si (um re-`sbatch` legítimo pode mudar
                      `upload_every_steps`), então isso AVISA em vez de abortar.

    Divergência de fase ou de convenção ABORTA. O resume é o momento em que um
    engano custa dias de GPU antes de aparecer.
    """
    meta = payload.get("metadata") or {}

    # A "fase" do currículo NÃO é o `stage`: as fases 1 (sintética) e 2 (real)
    # são as DUAS `bokeh`. Comparar `stage` não distinguia uma da outra — a
    # revisão externa pegou isso. O que distingue é a COMPOSIÇÃO DE DADOS, e é
    # ela que é comparada logo abaixo por `fontes_fingerprint`.
    fase_ckpt = meta.get("stage")
    if fase_ckpt is not None and str(fase_ckpt) != str(stage):
        raise RuntimeError(
            f"{caminho.name} foi gravado no stage {fase_ckpt!r} e o run atual é "
            f"{stage!r}. Retomar cruzaria as fases do currículo do §4.1. Para "
            "INICIAR uma fase a partir de outra, use `--init-lora` — que carrega "
            "os pesos e mantém optimizer/scheduler FRESCOS, que é o que o paper "
            "descreve."
        )

    conv_ckpt = meta.get("defocus_source")
    conv_atual = stage_config(config, stage).defocus_source if stage != "deblur" else None
    if conv_ckpt is not None and conv_atual is not None and conv_ckpt != conv_atual:
        raise RuntimeError(
            f"{caminho.name} foi treinado com defocus_source={conv_ckpt!r} e o run "
            f"atual usa {conv_atual!r}. Duas convenções do mapa de defocus no "
            "mesmo run é o defeito do experimento `kfix` — ele subiu o LF-Bokeh e "
            "derrubou RealBokeh e RealDOF no mesmo lote. Comece um run novo."
        )

    fontes_ckpt = meta.get("fontes_fingerprint")
    fontes_agora = _fontes_fingerprint(config, stage)
    if fontes_ckpt is not None and fontes_agora is not None and fontes_ckpt != fontes_agora:
        raise RuntimeError(
            f"{caminho.name} foi treinado com outra COMPOSIÇÃO DE DADOS.\n"
            f"  checkpoint: {fontes_ckpt}\n"
            f"  agora     : {fontes_agora}\n"
            "As fases 1 (sintética) e 2 (real) do §4.1 são ambas `stage=bokeh`, "
            "então o `stage` não as distingue — as fontes sim. Retomar através "
            "da fronteira herdaria optimizer e scheduler da fase errada, quando "
            "o §3.2(a) chama a sintética de 'pretrain': pré-treino SEGUIDO de "
            "treino, não continuação. Para iniciar uma fase a partir de outra, "
            "use `--init-lora`."
        )

    hash_ckpt = meta.get("config_hash")
    hash_atual = config_hash(config)
    if hash_ckpt and hash_ckpt != hash_atual:
        print(
            f"[resume] AVISO: config_hash diverge ({hash_ckpt} no checkpoint, "
            f"{hash_atual} agora). Se a mudança foi só de logging/upload, siga; "
            "se algum eixo do modelo mudou, o run deixou de ser comparável.",
            flush=True,
        )


def maybe_resume_checkpoint(
    *,
    output_dir: Path,
    stage: StageName,
    config: TrainConfig,
    backbone: FluxBackbone,
    optimizer: torch.optim.Optimizer,
    scheduler: WarmupCosineScheduler,
) -> TrainState:
    """Carrega o último checkpoint periódico, se existir."""
    ckpt_dir = _checkpoint_dir(output_dir, stage)
    checkpoints = _sorted_checkpoints(ckpt_dir)
    if not checkpoints:
        return TrainState()

    latest = checkpoints[-1]
    print(f"[resume] Carregando checkpoint: {latest}")
    payload = torch.load(latest, map_location="cpu", weights_only=False)
    _verificar_compatibilidade_do_checkpoint(
        payload, stage=stage, config=config, caminho=latest
    )

    missing, unexpected = _aplicar_lora_state(backbone, payload["lora_state"])
    _recusar_checkpoint_pre_c1(unexpected)
    if missing:
        raise RuntimeError(f"Checkpoint não tem todos os LoRA params: missing={missing[:5]}...")
    if unexpected:
        print(f"[resume] WARN: keys inesperadas (ignoradas): {unexpected[:5]}...")

    optimizer.load_state_dict(payload["optimizer"])
    scheduler.load_state_dict(payload["scheduler"])
    if _restaurar_rngs(payload.get("rng")):
        print("[resume] RNG restaurado (torch CPU/GPU, numpy, random).")
    else:
        print(
            "[resume] AVISO: checkpoint SEM estado de RNG (gravado por versão "
            "anterior). A retomada repete a sequência aleatória do começo."
        )

    s = payload.get("train_state", {})
    estado = TrainState(
        global_step=int(s.get("global_step", 0)),
        best_loss=float(s.get("best_loss", math.inf)),
    )
    estado.epoca = int(payload.get("epoca", 0))
    return estado


# =============================================================================
# Dataset / dataloader
# =============================================================================

def _set_seed(seed: int) -> None:
    import numpy as np

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def _cycle(
    loader: Iterable[dict[str, Any]],
    epoca_inicial: int = 0,
    estado: Any = None,
) -> Iterator[dict[str, Any]]:
    """Itera para sempre, avisando o sampler distribuído a cada volta.

    T18 — sem `set_epoch`, o `DistributedSampler` usa a MESMA permutação em todas
    as épocas e o MESMO particionamento entre GPUs. Com ~26K amostras reais e 60K
    steps × 32 = 1,92M amostras vistas, são ~74 épocas repetindo a mesma ordem:
    o batch acumulado vê sempre as mesmas vizinhanças, o que reduz a diversidade
    efetiva de combinação sem que nada avise.

    `getattr` em vez de `isinstance`: o sampler pode estar embrulhado
    (`BatchSampler`, `SkipBatchSampler` do accelerate) e o que importa é expor o
    método, não a classe.
    """
    epoca = int(epoca_inicial)
    while True:
        sampler = getattr(loader, "sampler", None)
        if sampler is None:
            sampler = getattr(loader, "batch_sampler", None)
        set_epoch = getattr(sampler, "set_epoch", None)
        if callable(set_epoch):
            set_epoch(epoca)
        for batch in loader:
            yield batch
        epoca += 1
        # DEFEITO CORRIGIDO (revisão adversarial): `epoca` era LOCAL e nunca
        # voltava para o `TrainState`, então o checkpoint gravava sempre 0 e a
        # "retomada na época em que parou" não acontecia — a correção era
        # inerte e o docstring do `save_checkpoint` afirmava algo falso.
        if estado is not None:
            estado.epoca = epoca


def _dataset_runtime(
    stage_cfg: StageConfig, config: TrainConfig, *, train: bool
) -> DatasetRuntimeConfig:
    """Monta o DatasetRuntimeConfig num ÚNICO lugar.

    Antes existiam dois sites duplicados (treino e smoke). Esquecer um fazia o
    smoke passar e o treino divergir — ou o contrário. Agora é um só.

    Campos que o `DatasetRuntimeConfig` ainda não conhecer são descartados com
    aviso, em vez de estourar `TypeError`: os arquivos desta revisão são
    editados em paralelo e um desencontro temporário não pode travar o treino
    inteiro sem dizer o que faltou.
    """
    desejado: dict[str, Any] = {
        "image_size": stage_cfg.image_size,
        "train": train and stage_cfg.augment,
        # C4 — eixos do experimento de escala/sigma
        "scale_mode": stage_cfg.scale_mode,
        "sigma_mu_source": stage_cfg.sigma_mu_source,
        # C5 — granularidade do filtro de nitidez
        "top_k_mode": stage_cfg.top_k_mode,
        "scene_key": stage_cfg.scene_key,
        # BokehNet (herdado)
        "defocus_source": stage_cfg.defocus_source,
        "max_coc": config.model.max_coc,
        "min_calibration_ssim": stage_cfg.min_calibration_ssim,
        "kfix_repo": stage_cfg.kfix_repo,
        "geo_condition": stage_cfg.geo_condition,
        "geo_escalares": stage_cfg.geo_escalares,
        "geo_constantes": stage_cfg.geo_constantes,
        "geo_field": stage_cfg.geo_field,
        "geo_sem_escalares": stage_cfg.geo_sem_escalares,
        "geo_ruido_controle": stage_cfg.geo_ruido_controle,
        "excluir_cenas_de_avaliacao": stage_cfg.excluir_cenas_de_avaliacao,
        # ── contrato `metric_disparity_official_v1` ─────────────────────────
        "require_valid_for_control": stage_cfg.require_valid_for_control,
        "exclude_censored_k": stage_cfg.exclude_censored_k,
        "max_levels_per_scene": stage_cfg.max_levels_per_scene,
        "route_weights": stage_cfg.route_weights,
        "exclude_refined_focus": stage_cfg.exclude_refined_focus,
        "scene_split_manifest": stage_cfg.scene_split_manifest,
        "scene_split_partition": stage_cfg.scene_split_partition,
        "allow_retired_defocus_sources": stage_cfg.allow_retired_defocus_sources,
        # ── L1: forma do release e resolução de referências (release.py) ────
        "release_format": stage_cfg.release_format,
        "mirror_roots": stage_cfg.mirror_roots,
        "permitir_download_do_release": stage_cfg.permitir_download_do_release,
        "verificar_sha256_da_origem": stage_cfg.verificar_sha256_da_origem,
    }
    validos = {f.name for f in dataclasses.fields(DatasetRuntimeConfig)}
    ignorados = sorted(set(desejado) - validos)
    if ignorados:
        # ERRO, não aviso. Antes isto imprimia e seguia, "para um desencontro
        # temporário entre arquivos não travar o treino". Mas o resultado de
        # seguir é um run de 40K steps rodando com uma configuração DIFERENTE da
        # que o YAML declara, e o aviso rola no log de 110 mil linhas de quatro
        # ranks. É a mesma forma do defeito `--env GRAD_ACCUM`: uma chave que o
        # operador escreve e que nada lê. Um `TypeError` no segundo 3 custa um
        # restart; o silêncio custa a rodada inteira.
        raise ValueError(
            f"DatasetRuntimeConfig não tem os campos {ignorados}, que o trainer "
            "tenta passar. Isso significa que `config.py`/`trainer.py` e "
            "`data.py` estão dessincronizados: o treino rodaria com o DEFAULT do "
            "dataloader, não com o que o YAML pede. Acerte o campo em "
            "`data.DatasetRuntimeConfig` (ou remova-o de `_dataset_runtime`) "
            "antes de subir o treino."
        )
    return DatasetRuntimeConfig(**{k: v for k, v in desejado.items() if k in validos})


def _build_val_loader(
    stage: StageName, stage_cfg: StageConfig, config: TrainConfig
) -> Any | None:
    """Loader de validação: determinístico, sem shuffle, sem augment.

    SÓ NO RANK PRINCIPAL. Antes, os N ranks construíam o dataset de validação em
    paralelo — cada um varrendo o manifesto inteiro e resolvendo pixels do
    espelho. Com 4 ranks sobre 69.700 amostras isso é 4x o trabalho, com
    contenção de disco, e foi onde o treino multi-GPU TRAVOU: um rank ficava
    para trás, os outros esperavam no primeiro coletivo, e o watchdog do NCCL
    abortava o processo (medido duas vezes; subir o timeout de 480 para 3600 s
    não resolveu, porque não era lentidão e sim dessincronia).

    A validação é um DIAGNÓSTICO: o número serve para escolher o `best.pt`, que
    só o rank principal grava. Não há coletivo dentro de `run_validation`, então
    os outros ranks receberem `None` é seguro — eles simplesmente não validam.
    """
    if not stage_cfg.val_datasets:
        return None
    # `RANK`/`LOCAL_RANK` são postos pelo `accelerate launch`; ausentes = 1 processo.
    rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")) or 0)
    if rank != 0:
        return None
    val_cfg = dataclasses.replace(
        stage_cfg,
        datasets=list(stage_cfg.val_datasets),
        max_samples=int(config.runtime.eval_max_samples),
        augment=False,
        shuffle=False,
        # T6/T9: na validação o teto por cena SAI (queremos todos os níveis da
        # cena de validação, é justamente onde a variação de K se mede) e a
        # partição do manifesto muda. Split por LINHA vazaria a cena inteira: a
        # mesma cena aparece com 2 a 21 aberturas.
        max_levels_per_scene=None,
        scene_split_partition="val",
        # ── A5 — o replay sintético NÃO entra na validação ───────────────────
        # `synthetic_replay_datasets` aponta para a rota A com `split: train`:
        # é literalmente a fonte de TREINO do braço. Sem zerar aqui, o
        # `BokehConcatDataset` montava metade do loader de validação com ela, e
        # o `val_loss` que escolhe o `best.pt` media 50% em dado visto.
        #
        # E o braço afetado seria justamente o do replay — o único que existe
        # para TESTAR uma hipótese nossa (esquecimento do controle de K). Uma
        # métrica contaminada ali não erra um número: invalida o experimento.
        synthetic_replay_fraction=0.0,
        synthetic_replay_datasets=[],
    )
    dataset = build_dataset(
        stage=stage, stage_config=val_cfg, runtime=_dataset_runtime(val_cfg, config, train=False)
    )
    if len(dataset) == 0:
        print("[eval] AVISO: dataset de validação vazio; validação desligada.")
        return None
    return build_dataloader(
        dataset,
        batch_size=1,
        num_workers=0,
        shuffle=False,
        pin_memory=False,
    )


# =============================================================================
# Forward de treino
# =============================================================================

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
    # C4 — `full_seq_len` são os tokens da IMAGEM DE ORIGEM inteira, usados
    # quando `sigma_mu_source="full_image"`. Pode não existir no batch (modo
    # "crop"); nesse caso o backbone cai no seq_len do próprio crop.
    full_seq_len = batch.get("full_seq_len")

    if stage == "deblur":
        return model.make_train_batch(
            blurry_image=batch["blurry_image"],
            sharp_image=batch["aif_image"],
            full_seq_len=full_seq_len,
        )
    if stage in ("bokeh", "bokeh_shape"):
        # §3.3: a imagem do kernel de abertura só existe no estágio de forma. No
        # estágio "bokeh" a chave não vem no batch e `shape_image=None` deixa o
        # caminho bit a bit igual ao anterior.
        return model.make_train_batch(
            aif_image=batch["aif_image"],
            target_image=batch["bokeh_image"],
            defocus_map=batch["defocus_map"],
            shape_image=batch.get("shape_image"),
            geo_map=batch.get("geo_map"),
            geo_branches=geo_branches,
            occlusion_lambda=occlusion_lambda,
            occlusion_pool=occlusion_pool,
            occlusion_theta=occlusion_theta,
            full_seq_len=full_seq_len,
        )
    raise NotImplementedError(f"Stage {stage} não implementado.")


# =============================================================================
# Validação determinística (C10)
# =============================================================================

class _SigmaFixo:
    """Força `backbone.sample_sigma` a devolver um valor fixo, temporariamente.

    Por que monkeypatch e não um parâmetro: `forward_train_step` sorteia o sigma
    por dentro, e a validação precisa do MESMO sigma em todo step para a métrica
    ser comparável. Trocar a função pública por uma constante é contido,
    reversível e não exige mudar a assinatura do backbone (que é editado por
    outro caminho). A assinatura é absorvida com *args/**kwargs de propósito,
    para sobreviver a mudanças em `sample_sigma`.
    """

    def __init__(self, backbone: FluxBackbone, valor: float, batch_size: int):
        self.backbone = backbone
        self.valor = float(valor)
        self.batch_size = int(batch_size)
        self._original = None

    def __enter__(self):
        self._original = self.backbone.sample_sigma
        device = self.backbone.device
        valor, bs = self.valor, self.batch_size

        def _fixo(*args: Any, **kwargs: Any) -> torch.Tensor:
            dtype = kwargs.get("dtype")
            if dtype is None:
                dtype = next(
                    (a for a in args if isinstance(a, torch.dtype)), torch.float32
                )
            return torch.full((bs,), valor, device=device, dtype=dtype)

        self.backbone.sample_sigma = _fixo  # type: ignore[method-assign]
        return self

    def __exit__(self, *exc: Any) -> None:
        self.backbone.sample_sigma = self._original  # type: ignore[method-assign]


@torch.no_grad()
def run_validation(
    *,
    stage: StageName,
    config: TrainConfig,
    model: nn.Module,
    backbone: FluxBackbone,
    val_loader: Any,
    accelerator: Any,
) -> float | None:
    """Loss de validação DETERMINÍSTICA.

    O `best_loss` anterior era a loss de treino de UM micro-batch, com sigma e
    ruído sorteados — sobe e desce sozinha e não serve para escolher
    checkpoint. Aqui:
      - sigma vem de uma grade FIXA `linspace(0.1, 0.9, eval_num_sigmas)`,
        atribuída às amostras em rodízio (amostra i recebe grade[i % n]).
        Rodízio, e não produto cartesiano, porque n_sigmas × n_amostras
        forwards custaria minutos a cada validação; assim são n_amostras
        forwards cobrindo a grade inteira.
      - o ruído sai de um RNG semeado com `eval_seed`, e o estado global do
        RNG é salvo e restaurado, para a validação não deslocar a sequência
        aleatória do treino.
    Resultado: o mesmo número em dois steps significa o mesmo desempenho.
    """
    if val_loader is None:
        return None

    sigmas = torch.linspace(0.1, 0.9, int(config.runtime.eval_num_sigmas)).tolist()

    estado_cpu = torch.get_rng_state()
    estado_cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    # B4 — `_set_seed` abaixo semeia numpy também; sem salvar isto, VALIDAR
    # deslocava a sequência do treino, que é o que o docstring promete não fazer.
    estado_np = np.random.get_state()
    estado_py = random.getstate()

    backbone.transformer.eval()
    total, n = 0.0, 0
    try:
        _set_seed(int(config.runtime.eval_seed))
        device = backbone.device
        for i, batch in enumerate(val_loader):
            batch_on_device = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()
            }
            bs = int(batch_on_device["aif_image"].shape[0])
            sigma = sigmas[i % len(sigmas)]
            with _SigmaFixo(backbone, sigma, bs):
                with accelerator.autocast():
                    # T20 — os pesos de oclusão TÊM que ser os mesmos do treino:
                    # é esta loss que escolhe o `best.pt`. Sem repassar, a
                    # validação media uma perda diferente da treinada.
                    outputs = _make_train_batch(
                        model=model,
                        stage=stage,
                        batch=batch_on_device,
                        occlusion_lambda=config.runtime.occlusion_lambda,
                        occlusion_pool=config.runtime.occlusion_pool,
                        occlusion_theta=config.runtime.occlusion_theta,
                        geo_branches=config.runtime.geo_branches,
                    )
                    loss = flow_matching_loss(outputs.prediction, outputs.target, outputs.weight)
            valor = float(loss.detach().item())
            if math.isfinite(valor):
                total += valor
                n += 1
    except Exception as exc:  # noqa: BLE001 - validação nunca derruba o treino
        print(f"[eval] WARN: validação falhou ({exc}); seguindo sem val_loss neste step.")
        return None
    finally:
        backbone.transformer.train()
        torch.set_rng_state(estado_cpu)
        if estado_cuda is not None:
            torch.cuda.set_rng_state_all(estado_cuda)
        np.random.set_state(estado_np)
        random.setstate(estado_py)

    if n == 0:
        return None
    return total / n


# =============================================================================
# Loop de treino
# =============================================================================

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
        json.dumps(metadata, ensure_ascii=True, indent=2), encoding="utf-8"
    )
    return cfg_hash


def _metricas_de_controle(stage: StageName, batch: dict[str, Any]) -> dict[str, float]:
    """Escalares do mapa de defocus, para o log (T12).

    Vêm prontos do `BokehMetricDataset` (`prepare_aligned_bokeh_metric` já os
    calcula ao montar o mapa), então aqui é só reduzir o batch. Ausentes no
    estágio deblur e nos caminhos aposentados — devolve `{}` e o log segue igual.

    O que cada um denuncia:
      `defocus_mean/max`  — mapa achatado ou saturado
      `frac_saturado`     — `max_coc` pequeno demais para a fonte (foi 17,3% da
                            rota B no experimento kfix)
      `k_efetivo`         — o K DEPOIS da reescala do crop (T2). Se a média não
                            variar entre batches, o K virou constante de novo.
    """
    saida: dict[str, float] = {}
    for chave in ("defocus_mean", "defocus_max", "defocus_frac_saturado", "k_efetivo"):
        valor = batch.get(chave)
        if valor is None:
            continue
        tensor = valor if isinstance(valor, torch.Tensor) else torch.as_tensor(valor)
        saida[f"{stage}/{chave}"] = float(tensor.float().mean().item())
        if chave == "k_efetivo" and tensor.numel() > 1:
            # Desvio dentro do batch: com batch efetivo 32 e K constante, isto é
            # zero — e K constante é exatamente o defeito da rota B antiga.
            saida[f"{stage}/k_efetivo_std"] = float(tensor.float().std().item())
    return saida


def _train_loop(
    *,
    stage: StageName,
    config: TrainConfig,
    model: nn.Module,
    backbone: FluxBackbone,
    dataloader: Iterable[dict[str, Any]],
    val_loader: Any,
    output_dir: Path,
    init_lora_path: Path | None = None,
) -> Path | None:
    accelerator = _build_accelerator(config)
    _conferir_memoria_compartilhada(config)

    # C7 — seed POR RANK. Com a mesma seed em todos os processos, as N GPUs
    # sorteavam o MESMO sigma e o MESMO ruído a cada micro-step: o batch efetivo
    # 32 (4 GPUs × accum 8) tinha só 8 sigmas distintos, inflando a variância do
    # gradiente sem ganho nenhum. É seguro diferenciar porque os pesos LoRA são
    # sincronizados por broadcast explícito do rank 0, logo abaixo.
    seed = config.runtime.seed + (
        accelerator.process_index if config.runtime.seed_per_rank else 0
    )
    _set_seed(seed)
    # Publica a seed para o dataloader: com `num_workers=0` não há
    # `worker_info` e o RNG da augmentação cairia na entropia do SO.
    from .data import definir_seed_do_processo

    definir_seed_do_processo(seed)

    stage_cfg = stage_config(config, stage)
    total_steps = int(stage_cfg.steps)

    optimizer = _make_optimizer(config, params=backbone.trainable_parameters())
    scheduler = _make_scheduler(config, optimizer=optimizer, total_steps=total_steps)

    # Multi-GPU NÃO usa DDP-wrap: o `transformer_forward` dos autores acessa
    # submódulos direto (self.x_embedder, ...), que o wrapper não expõe, e como
    # esse forward não passa pelo `.forward()` do módulo o DDP nem sincronizaria
    # os gradientes. Então: single-GPU prepara normalmente; multi-GPU mantém o
    # transformer cru e sincroniza os grads manualmente (all-reduce AVG).
    if accelerator.num_processes > 1:
        optimizer = accelerator.prepare(optimizer)
        transformer_prepared = backbone.transformer
    else:
        transformer_prepared, optimizer = accelerator.prepare(backbone.transformer, optimizer)
        backbone.transformer = transformer_prepared

    run_metadata = build_run_metadata(
        stage=stage,
        config=config,
        stage_cfg=stage_cfg,
        backbone=backbone,
        num_processes=accelerator.num_processes,
    )

    logger = (
        ExperimentLogger(
            config=config, output_dir=output_dir, stage=stage, extra_config=run_metadata
        )
        if accelerator.is_main_process
        else None
    )
    cfg_hash = _prepare_runtime_artifacts(config, output_dir)

    state = TrainState()
    if config.runtime.resume:
        state = maybe_resume_checkpoint(
            output_dir=output_dir,
            stage=stage,
            config=config,
            backbone=backbone,
            optimizer=optimizer,
            scheduler=scheduler,
        )

    # Fase 2 do BokehNet (currículo do paper: sintético -> real): só quando o run
    # começa do ZERO, carrega o LoRA da fase 1 e mantém optimizer/scheduler/step
    # frescos (nova fase, novo warmup + cosseno sobre os steps da fase 2).
    if stage == "bokeh_shape" and state.global_step > 0:
        # DEFEITO CORRIGIDO (revisão externa): ao RETOMAR a fase de forma, o
        # `init_lora` era pulado (a condição é `global_step == 0`) e o base vinha
        # só do checkpoint. Com o `_lora_state_dict` antigo — que gravava apenas
        # treináveis — o base ficava com a inicialização gaussiana e o
        # fine-tune de forma rodava sobre um modelo que não faz bokeh, em
        # silêncio. Agora o `_lora_state_dict` grava o base, e aqui a gente
        # CONFERE que ele veio mesmo.
        from .backbone import ADAPTER_NAME

        base = _params_do_adapter(backbone, ADAPTER_NAME)
        nulos = [n for n in base if not torch.any(
            dict(backbone.transformer.named_parameters())[n].detach()
        )]
        if len(nulos) == len(base):
            raise RuntimeError(
                "Retomada da fase de forma com o adapter base TODO ZERADO: o "
                "checkpoint de onde se retomou foi gravado pela versão antiga, "
                "que só salvava parâmetros treináveis. Recomece a fase 3 do "
                "checkpoint da fase 2 (`--init-lora`), sem `resume`."
            )

    if init_lora_path is not None and state.global_step == 0:
        loaded_step = load_lora_checkpoint_into_backbone(
            backbone, init_lora_path, exigir_completo=(stage != "bokeh_shape")
        )
        print(f"[train] fase 2: LoRA inicializado da fase 1 ({init_lora_path}, step {loaded_step}).")

    # C3 — aplica o LR ANTES do primeiro update. Sem isto, o `param_group` ainda
    # carrega o `lr` base e o primeiro `optimizer.step()` roda com 1e-4 em vez do
    # primeiro passo do warmup (2e-7 com warmup=500). No resume, reposiciona o LR
    # no ponto certo da curva.
    scheduler.step(state.global_step)

    # Multi-GPU sem DDP: replica manualmente o que o DDP faria — (1) broadcast dos
    # pesos treináveis do rank 0 (o LoRA é init gaussiano ALEATÓRIO; sem isto cada
    # GPU começaria diferente e os modelos divergiriam), (2) all-reduce dos grads
    # no passo de sync (ver o laço).
    if accelerator.num_processes > 1:
        for p in backbone.trainable_parameters():
            torch.distributed.broadcast(p.data, src=0)
        accelerator.wait_for_everyone()
        dataloader = accelerator.prepare(dataloader)

    # retoma na época em que parou: sem isto, um run que caiu 5 vezes veria 5
    # vezes a MESMA permutação inicial dos dados.
    iterator = _cycle(
        dataloader, epoca_inicial=getattr(state, "epoca", 0), estado=state
    )
    backbone.transformer.train()

    # EMA da loss: em flow matching a loss por step é MUITO ruidosa (sigma e ruído
    # sorteados a cada step), então a loss bruta oscila mesmo com o modelo
    # aprendendo. A EMA deixa a tendência legível.
    ema_loss: float | None = None
    ema_beta = 0.98
    # #13 — acumuladores da janela de acumulação de gradiente
    soma_loss, n_micro = 0.0, 0

    # ── T11: probe de controlabilidade ──────────────────────────────────────
    probe_every = int(getattr(config.runtime, "probe_every_steps", 0) or 0)
    probe_conjunto = None
    probe_melhor: float | None = None
    alarmes_seguidos = 0
    if stage in ("bokeh", "bokeh_shape") and probe_every and accelerator.num_processes > 1:
        # RECUSA, em vez de degradar. O probe leva ~5 min no rank 0 gerando 32
        # imagens de 28 passos. Nesse tempo os outros ranks avançam sozinhos até
        # o `all_reduce` dos gradientes e ficam presos lá — que é EXATAMENTE o
        # cenário do watchdog do NCCL que matou este treino duas vezes (subir o
        # timeout de 480 para 3600 s não resolveu, porque não era lentidão e sim
        # dessincronia). Pior: a `ParadaPorControlabilidade` é levantada só no
        # rank 0, que vai para o `finally` e para no `wait_for_everyone()`
        # enquanto os outros esperam num `all_reduce` — coletivos diferentes,
        # deadlock até o SIGABRT.
        #
        # E o princípio declarado antes do run é que o instrumento é OBSERVADOR:
        # medição não muda a trajetória do treino. O lugar dela é fora.
        raise ValueError(
            f"`probe_every_steps={probe_every}` com {accelerator.num_processes} "
            "processos. O probe interno roda só no rank 0 e trava os demais no "
            "próximo coletivo — é a dessincronia que abortou este treino duas "
            "vezes pelo watchdog do NCCL.\n"
            "Ponha `probe_every_steps: 0` no YAML e meça por fora, com\n"
            "  scripts/monitor_externo.py --output-dir <output_dir> ...\n"
            "que lê os `step_*.pt` já gravados, roda em GPU e container "
            "separados, e não toca em RNG, optimizer, pesos nem coletivo."
        )
    if stage in ("bokeh", "bokeh_shape") and probe_every:
        probe_dir = getattr(config.runtime, "probe_set_dir", None)
        if not probe_dir:
            print(
                "[probe] ================================================================\n"
                "[probe] AVISO GRAVE: `probe_set_dir` não configurado. O treino da\n"
                "[probe] BokehNet vai rodar SEM o único instrumento sensível ao seu modo\n"
                "[probe] de falha conhecido. Na fase 2 anterior o LVCorr caiu de +0,9059\n"
                "[probe] para +0,4365, degradando MONOTONICAMENTE, com a loss caindo o\n"
                "[probe] tempo todo. Gere o conjunto com scripts/build_probe_set.py.\n"
                "[probe] ================================================================",
                flush=True,
            )
            probe_every = 0
        else:
            try:
                from .probe import carregar_conjunto_do_probe

                probe_conjunto = carregar_conjunto_do_probe(probe_dir)
                print(
                    f"[probe] conjunto com {len(probe_conjunto)} imagens; "
                    f"LVCorr a cada {probe_every} steps.",
                    flush=True,
                )
            except Exception as exc:  # noqa: BLE001
                print(f"[probe] AVISO: não consegui carregar o conjunto ({exc}); probe DESLIGADO.")
                probe_every = 0

    eval_every = int(config.runtime.eval_every_steps)
    # O AVISO só faz sentido no rank principal. Nos demais o `val_loader` é None
    # POR PROJETO (`_build_val_loader` devolve None fora do rank 0, para não
    # repetir a construção do dataset em N processos — foi o que travou o
    # multi-GPU no watchdog do NCCL). Imprimir "validação DESLIGADA" ali é
    # FALSO, e custou caro: a fase 1 de 40K steps validou normalmente, mas os
    # três avisos dos ranks 1-3 no log fizeram o README registrar por dois dias
    # que a validação não tinha rodado — e que o `best.pt` era lixo. Não era:
    # `best.pt` saiu do step 40.000 com `val_loss=0,0602` medido na partição de
    # validação. Uma mensagem errada num log é um defeito, não um detalhe.
    if eval_every and val_loader is None and accelerator.is_main_process:
        print(
            "[eval] AVISO: eval_every_steps > 0 mas não há `val_datasets` no YAML — "
            "validação DESLIGADA e `best_loss` seguirá a loss de treino."
        )

    batch_efetivo = run_metadata["batch_efetivo"]
    print(
        f"[train] stage={stage} steps={total_steps} start={state.global_step} "
        f"num_gpus={accelerator.num_processes} "
        f"grad_accum={config.runtime.gradient_accumulation_steps} "
        f"batch_efetivo={batch_efetivo} seed={seed} "
        f"lora={run_metadata.get('adapter_variant')} "
        f"guidance={run_metadata.get('train_guidance')} "
        f"scale_mode={stage_cfg.scale_mode} sigma_mu={stage_cfg.sigma_mu_source}",
        flush=True,
    )

    last_ckpt: Path | None = None
    try:
        while state.global_step < total_steps:
            batch = next(iterator)

            with accelerator.accumulate(transformer_prepared):
                device = backbone.device
                batch_on_device = {
                    k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }

                # autocast explícito: `transformer_forward` é uma função custom que
                # NÃO passa pelo `.forward()` do módulo preparado, então o autocast
                # que o accelerate injeta no módulo é ignorado.
                with accelerator.autocast():
                    outputs = _make_train_batch(
                        model=model,
                        stage=stage,
                        batch=batch_on_device,
                        occlusion_lambda=config.runtime.occlusion_lambda,
                        occlusion_pool=config.runtime.occlusion_pool,
                        occlusion_theta=config.runtime.occlusion_theta,
                        geo_branches=config.runtime.geo_branches,
                    )
                    loss = flow_matching_loss(
                        outputs.prediction, outputs.target, outputs.weight
                    )

                accelerator.backward(loss)
                # #13 — acumula a loss da JANELA de acumulação. A loss de um
                # microbatch é uma amostra de σ e de ruído; com accum 8 em 4
                # GPUs, a que ia para o log era 1/32 do que o step de fato
                # otimizou.
                soma_loss += float(loss.detach().item())
                n_micro += 1

                if accelerator.sync_gradients:
                    # Multi-GPU sem DDP: cada processo acumulou sobre o SEU shard;
                    # o all-reduce AVG dá o gradiente data-parallel correto.
                    if accelerator.num_processes > 1:
                        for p in backbone.trainable_parameters():
                            if p.grad is not None:
                                torch.distributed.all_reduce(
                                    p.grad, op=torch.distributed.ReduceOp.AVG
                                )
                    accelerator.clip_grad_norm_(
                        backbone.trainable_parameters(), max_norm=config.runtime.max_grad_norm
                    )

                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            if not accelerator.sync_gradients:
                continue

            state.global_step += 1
            scheduler.step(state.global_step)

            # média sobre os microbatches desta janela...
            loss_val = soma_loss / max(1, n_micro)
            soma_loss, n_micro = 0.0, 0
            # ...e sobre as GPUs, para o gráfico ser o do batch EFETIVO 32 e
            # não o do shard do rank 0.
            if accelerator.num_processes > 1:
                t = torch.tensor([loss_val], device=backbone.device)
                torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.AVG)
                loss_val = float(t.item())
            ema_loss = (
                loss_val if ema_loss is None else ema_beta * ema_loss + (1.0 - ema_beta) * loss_val
            )

            # ── validação ────────────────────────────────────────────────────
            val_loss = None
            if eval_every and val_loader is not None and state.global_step % eval_every == 0:
                val_loss = run_validation(
                    stage=stage,
                    config=config,
                    model=model,
                    backbone=backbone,
                    val_loader=val_loader,
                    accelerator=accelerator,
                )

            # ── T11: LVCorr ──────────────────────────────────────────────────
            if (
                probe_every
                and probe_conjunto
                and accelerator.is_main_process
                and state.global_step % probe_every == 0
            ):
                from .probe import lvcorr_agregado, run_lvcorr_probe

                por_imagem = run_lvcorr_probe(
                    backbone=backbone,
                    text_embeddings=model.text_embeddings,
                    conjunto=probe_conjunto,
                    num_inference_steps=int(config.runtime.probe_num_inference_steps),
                    seed=int(config.runtime.probe_seed),
                )
                if por_imagem:
                    valor = lvcorr_agregado(por_imagem)
                    print(
                        f"[probe] step={state.global_step} LVCorr={valor:+.4f} "
                        f"({len(por_imagem)} imagens)",
                        flush=True,
                    )
                    if logger is not None:
                        logger.log({f"{stage}/lvcorr": valor}, step=state.global_step)
                    if math.isfinite(valor):
                        if probe_melhor is None or valor > probe_melhor:
                            probe_melhor = valor
                        queda = probe_melhor - valor
                        if queda > float(config.runtime.probe_alerta_queda):
                            alarmes_seguidos += 1
                            print(
                                f"[probe] ALARME: LVCorr caiu {queda:+.4f} em relação ao "
                                f"melhor do run ({probe_melhor:+.4f}). É a assinatura do "
                                "esquecimento do controle de K medido na fase 2 anterior. "
                                "O critério declarado é PARAR e investigar em vez de "
                                "deixar rodar até o fim.",
                                flush=True,
                            )
                            # O PLANO_RETREINO diz, literalmente: "em dois
                            # probes consecutivos com alarme, o run para". O
                            # código só imprimia — e um critério de parada que
                            # não para é um critério que não existe.
                            if alarmes_seguidos >= int(
                                config.runtime.probe_alarmes_para_parar
                            ):
                                raise ParadaPorControlabilidade(
                                    f"LVCorr caiu {queda:+.4f} abaixo do melhor do "
                                    f"run em {alarmes_seguidos} probes seguidos "
                                    f"(melhor {probe_melhor:+.4f}, agora {valor:+.4f}). "
                                    "É a assinatura do esquecimento do controle de K "
                                    "medido na fase 2 anterior, que degradou "
                                    "MONOTONICAMENTE com a loss caindo. O critério "
                                    "declarado ANTES do run é parar e investigar. "
                                    "O último checkpoint está salvo."
                                )
                        else:
                            alarmes_seguidos = 0

            # `best_loss` segue a validação quando ela existe; só cai para a loss
            # de treino quando não há split de validação configurado.
            melhorou = False
            referencia = val_loss if val_loss is not None else (
                loss_val if val_loader is None else None
            )
            if referencia is not None and referencia < state.best_loss:
                state.best_loss = referencia
                melhorou = True

            if accelerator.is_main_process and state.global_step % config.runtime.log_every_steps == 0:
                # Pico de VRAM por janela de log (resetado a cada leitura), para
                # saber quão perto do teto da GPU o treino está rodando.
                peak_gb = 0.0
                if torch.cuda.is_available():
                    peak_gb = torch.cuda.max_memory_allocated() / 1e9
                    torch.cuda.reset_peak_memory_stats()
                lr_now = float(optimizer.param_groups[0]["lr"])
                payload: dict[str, float | int] = {
                    f"{stage}/loss": loss_val,
                    f"{stage}/loss_ema": ema_loss,
                    f"{stage}/lr": lr_now,
                    f"{stage}/best_loss": state.best_loss,
                    f"{stage}/peak_vram_gb": peak_gb,
                }
                # ── T12: o SINAL DE CONTROLE, medido ────────────────────────
                # A loss de flow matching é cega para o defeito que custou a
                # fase 2 inteira. O defeito D1 (`max(defocus) == 1,0` em TODAS
                # as amostras, com k de 33 a 195) teria aparecido AQUI, no
                # primeiro step. Custo: quatro escalares já calculados no
                # dataloader.
                payload.update(_metricas_de_controle(stage, batch_on_device))
                if val_loss is not None:
                    payload[f"{stage}/val_loss"] = val_loss
                logger.log(payload, step=state.global_step)
                print(
                    f"[train] step={state.global_step}/{total_steps} "
                    f"loss={loss_val:.4f} ema={ema_loss:.4f} lr={lr_now:.2e} "
                    + (f"val={val_loss:.4f} " if val_loss is not None else "")
                    + f"vram_pico={peak_gb:.1f}GB",
                    flush=True,
                )
            elif accelerator.is_main_process and val_loss is not None:
                logger.log({f"{stage}/val_loss": val_loss}, step=state.global_step)

            # ── checkpoints ──────────────────────────────────────────────────
            if accelerator.is_main_process:
                if state.global_step % config.runtime.save_every_steps == 0:
                    last_ckpt = save_checkpoint(
                        output_dir=output_dir, stage=stage, backbone=backbone,
                        optimizer=optimizer, scheduler=scheduler, state=state,
                        cfg_hash=cfg_hash,
                        metrics_snapshot={"loss": loss_val, "best_loss": state.best_loss},
                        keep_last_n=config.runtime.keep_last_n_checkpoints,
                        run_metadata=run_metadata,
                    )

                # `best.pt` fica FORA da série `step_*.pt`, então nunca é podado.
                if melhorou and config.runtime.save_best and val_loss is not None:
                    save_checkpoint(
                        output_dir=output_dir, stage=stage, backbone=backbone,
                        optimizer=optimizer, scheduler=scheduler, state=state,
                        cfg_hash=cfg_hash,
                        metrics_snapshot={"val_loss": val_loss, "best_loss": state.best_loss},
                        keep_last_n=0, run_metadata=run_metadata, filename="best.pt",
                    )
                    if config.logging.upload_best:
                        upload_lora_snapshot_to_hf(
                            backbone, stage, config.logging.upload_hf_repo_base,
                            state.global_step, output_dir=output_dir,
                            run_metadata={**run_metadata, "val_loss": val_loss,
                                          "global_step": state.global_step},
                            private=config.logging.upload_private,
                            retries=config.logging.upload_retries,
                            rotulo="best",
                        )

                if (
                    config.logging.upload_every_steps
                    and state.global_step % config.logging.upload_every_steps == 0
                ):
                    upload_lora_snapshot_to_hf(
                        backbone, stage, config.logging.upload_hf_repo_base,
                        state.global_step, output_dir=output_dir,
                        run_metadata={**run_metadata, "global_step": state.global_step},
                        private=config.logging.upload_private,
                        retries=config.logging.upload_retries,
                    )
    finally:
        # Salvamento final SÓ no rank principal: em multi-GPU todos os ranks têm
        # pesos idênticos, mas dois processos escrevendo o mesmo arquivo corrompem.
        #
        # Este bloco roda TAMBÉM quando o laço morreu por exceção — é assim que um
        # crash no step 41.000 não joga fora 41.000 steps de GPU. Por isso vai
        # inteiro em try/except: um erro AQUI não pode mascarar a exceção original.
        if accelerator.is_main_process:
            try:
                last_ckpt = save_checkpoint(
                    output_dir=output_dir, stage=stage, backbone=backbone,
                    optimizer=optimizer, scheduler=scheduler, state=state,
                    cfg_hash=cfg_hash,
                    metrics_snapshot={"best_loss": state.best_loss},
                    keep_last_n=config.runtime.keep_last_n_checkpoints,
                    run_metadata=run_metadata,
                )
                sf_path = output_dir / stage / f"{stage}.safetensors"
                export_lora_safetensors(backbone, sf_path, stage, run_metadata=run_metadata)
                print(f"[train] Final checkpoint: {last_ckpt}")
                print(f"[train] Final safetensors: {sf_path}")

                if config.logging.upload_final:
                    upload_lora_snapshot_to_hf(
                        backbone, stage, config.logging.upload_hf_repo_base,
                        state.global_step, output_dir=output_dir,
                        run_metadata={**run_metadata, "global_step": state.global_step},
                        private=config.logging.upload_private,
                        retries=config.logging.upload_retries,
                        rotulo=f"final_step{state.global_step}",
                    )
            except Exception as exc:  # noqa: BLE001
                print(f"[train] WARN: falha no salvamento final: {exc}")
            finally:
                if logger is not None:
                    logger.finish()

        accelerator.wait_for_everyone()

    return last_ckpt


# =============================================================================
# Entrypoints públicos
# =============================================================================

def _run_stage(
    stage: StageName,
    config: TrainConfig,
    backbone: FluxBackbone,
    model: nn.Module,
    output_dir: Path,
    init_lora_path: Path | None = None,
) -> Path | None:
    """Monta dataset + loaders e roda o laço. Comum a deblur e bokeh."""
    stage_cfg = stage_config(config, stage)

    dataset = build_dataset(
        stage=stage,
        stage_config=stage_cfg,
        runtime=_dataset_runtime(stage_cfg, config, train=True),
    )
    loader = build_dataloader(
        dataset,
        batch_size=stage_cfg.batch_size,
        num_workers=config.runtime.num_workers,
        shuffle=stage_cfg.shuffle,
        pin_memory=config.runtime.pin_memory,
    )
    val_loader = (
        _build_val_loader(stage, stage_cfg, config)
        if config.runtime.eval_every_steps
        else None
    )
    return _train_loop(
        stage=stage,
        config=config,
        model=model,
        backbone=backbone,
        dataloader=loader,
        val_loader=val_loader,
        output_dir=output_dir,
        init_lora_path=init_lora_path,
    )


def run_deblur_stage(
    config: TrainConfig, backbone: FluxBackbone, model: nn.Module, output_dir: Path
) -> Path | None:
    """Treina DeblurNet (Stage 1)."""
    return _run_stage("deblur", config, backbone, model, output_dir)


def run_bokeh_stage(
    config: TrainConfig,
    backbone: FluxBackbone,
    model: nn.Module,
    output_dir: Path,
    init_lora_path: Path | None = None,
) -> Path | None:
    """Treina BokehNet (Stage 2). `init_lora_path` = checkpoint da fase 1."""
    return _run_stage(
        "bokeh", config, backbone, model, output_dir, init_lora_path=init_lora_path
    )


def run_bokeh_shape_stage(
    config: TrainConfig,
    backbone: FluxBackbone,
    model: nn.Module,
    output_dir: Path,
    init_lora_path: Path | None = None,
) -> Path | None:
    """§3.3 — fine-tune do LoRA de FORMA, com o LoRA base CONGELADO.

    `init_lora_path` aqui é o checkpoint da FASE 2 (a BokehNet pronta): ele é
    carregado no adapter base, que `backbone._inject_shape_lora` já congelou.
    Sem ele, o "congelado" seria um LoRA de inicialização gaussiana, e o
    fine-tune de forma partiria de um modelo que não sintetiza bokeh.
    """
    if init_lora_path is None:
        raise ValueError(
            "run_bokeh_shape_stage exige `--init-lora` apontando para o checkpoint "
            "da FASE 2. O §3.3 congela o LoRA já treinado; congelar um LoRA "
            "aleatório treinaria forma sobre um modelo que não faz bokeh."
        )
    return _run_stage(
        "bokeh_shape", config, backbone, model, output_dir,
        init_lora_path=init_lora_path,
    )


# =============================================================================
# Smoke test (caminho REAL, com FLUX, 3 steps)
# =============================================================================

def run_smoke_test(
    config: TrainConfig,
    backbone: FluxBackbone,
    model: nn.Module,
    output_dir: Path,
    stage: StageName = "deblur",
) -> None:
    """3 steps no caminho real (FLUX + LoRA + transformer_forward).

    Critérios: shapes esperados, loss finita, backward sem erro, pesos LoRA
    mudam depois do optimizer.step, export do safetensors funciona.

    Roda SEM accelerate (autocast manual) para isolar o caminho de FLUX puro,
    e com a seed PURA (single-process; tem que ser reprodutível).
    """
    print("=" * 70)
    print(f"SMOKE TEST ({stage}) — 3 steps no caminho real (FLUX + LoRA)")
    print("=" * 70)

    _set_seed(config.runtime.seed)
    stage_cfg = stage_config(config, stage)

    # train=False -> crop determinístico, sem flip: smoke reprodutível.
    dataset = build_dataset(
        stage=stage,
        stage_config=stage_cfg,
        runtime=_dataset_runtime(stage_cfg, config, train=False),
    )
    if len(dataset) == 0:
        raise RuntimeError(
            f"Dataset vazio para os sources: "
            f"{[s.name + ':' + s.split for s in stage_cfg.datasets]}"
        )

    loader = build_dataloader(
        dataset, batch_size=1, num_workers=0, shuffle=False, pin_memory=False
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
            k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()
        }

        with autocast_ctx:
            outputs = _make_train_batch(model=model, stage=stage, batch=batch_on_device)
            loss = flow_matching_loss(outputs.prediction, outputs.target, outputs.weight)

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
            f"Smoke test: parâmetros LoRA NÃO mudaram após 3 steps (max diff={diff}). "
            "Algo no pipeline de gradiente está errado."
        )

    run_metadata = build_run_metadata(
        stage=stage, config=config, stage_cfg=stage_cfg, backbone=backbone, num_processes=1
    )
    sf_path = output_dir / "smoke.safetensors"
    sf_path.parent.mkdir(parents=True, exist_ok=True)
    export_lora_safetensors(backbone, sf_path, stage=stage, run_metadata=run_metadata)

    print(f"[smoke] PASSOU. Param diff após 3 steps: {diff:.2e}")
    print(f"[smoke] Safetensors exportado: {sf_path}")
    print("=" * 70)
