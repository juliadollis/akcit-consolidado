"""Schema de configuração do treino da DeblurNet (GenRefocus, Stage 1).

Mescla a config do código de treino (FLUX real, rectified flow) com a
descrição dos datasets do Hugging Face (do dataloader do colega):

  - `StageConfig.datasets`: lista de `{name, split}` (repos HF), em vez de um
    manifest JSONL local.
  - `ModelConfig` sem `backend`/`diffusion_target`: é sempre FLUX real +
    rectified flow velocity (única coisa correta para FLUX). Hardcoded em
    backbone.py.

Defaults alinhados ao paper (§4.1):
  - LoRA rank 128 (Deblur).
  - 60K steps.
  - batch efetivo 32 → 1 GPU × gradient_accumulation_steps=32.
  - datasets: DPDD (`akcit-pixel/DDPD`) + RealBokeh (`akcit-pixel/RealBokeh`).

[UNSPECIFIED no paper] optimizer/lr/scheduler/warmup/resolução/precisão:
  usamos defaults sensatos de FLUX LoRA (AdamW lr=1e-4, cosine warmup 500,
  bf16, 512²). Documentado no README.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from hashlib import sha256
from pathlib import Path
from typing import Any

import yaml


@dataclass
class OptimizerConfig:
    name: str = "adamw"
    lr: float = 1.0e-4
    weight_decay: float = 1.0e-4   # FLUX LoRA training oficial usa 1e-4
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1.0e-8


@dataclass
class SchedulerConfig:
    name: str = "cosine"
    warmup_steps: int = 500        # FLUX LoRA: 500 é típico (paper não especifica)
    min_lr_ratio: float = 0.1


@dataclass
class ModelConfig:
    pretrained_model_name_or_path: str = "black-forest-labs/FLUX.1-dev"
    vae_subfolder: str = "vae"
    transformer_subfolder: str = "transformer"
    max_coc: float = 100.0
    deblur_lora_rank: int = 128    # paper §4.1
    bokeh_lora_rank: int = 64      # paper §4.1 (Stage 2, futuro)
    gradient_checkpointing: bool = True


@dataclass
class RuntimeConfig:
    output_dir: str = "outputs"
    seed: int = 42
    num_workers: int = 4
    pin_memory: bool = True
    mixed_precision: str = "bf16"
    gradient_accumulation_steps: int = 32  # 1 GPU H100 → batch efetivo 32 (paper)
    max_grad_norm: float = 1.0
    log_every_steps: int = 10
    save_every_steps: int = 1000
    keep_last_n_checkpoints: int = 3
    resume: bool = True


@dataclass
class DatasetSourceConfig:
    name: str   # repo HF, ex. "akcit-pixel/DDPD"
    split: str  # ex. "train"
    # Filtro do paper (§4.1) p/ o RealBokeh: "compute the Laplacian variance of
    # each image as a focus measure ... retain the top 3000 images with the
    # highest Laplacian variance". top_k_sharpest=3000 mantém só as 3000 mais
    # nítidas DESTE source. None = usa o source inteiro (ex.: DPDD).
    top_k_sharpest: int | None = None
    sharpness_column: str = "image_focus"   # coluna usada como medida de foco


@dataclass
class StageConfig:
    datasets: list[DatasetSourceConfig]
    steps: int
    batch_size: int = 1
    image_size: int = 512
    shuffle: bool = True
    # Para testes de overfit: limita o dataset às N primeiras amostras (None = tudo).
    max_samples: int | None = None
    # Augmentation (random crop + hflip). Desligue (False) para overfit determinístico.
    augment: bool = True


def _paper_deblur_sources() -> list[DatasetSourceConfig]:
    """Composição do paper para a DeblurNet: DPDD + subset RealBokeh."""
    return [
        DatasetSourceConfig(name="akcit-pixel/DDPD", split="train"),
        DatasetSourceConfig(name="akcit-pixel/RealBokeh", split="train"),
    ]


@dataclass
class DataConfig:
    deblur: StageConfig = field(
        default_factory=lambda: StageConfig(
            datasets=_paper_deblur_sources(), steps=60000
        )
    )


@dataclass
class LoggingConfig:
    use_wandb: bool = True
    wandb_project: str = "genrefocus-train"
    wandb_entity: str | None = None
    run_name: str | None = None


@dataclass
class TrainConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    data: DataConfig = field(default_factory=DataConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)


def _as_stage_config(payload: dict[str, Any]) -> StageConfig:
    datasets = [
        DatasetSourceConfig(
            name=str(item["name"]),
            split=str(item["split"]),
            top_k_sharpest=(
                None if item.get("top_k_sharpest") is None else int(item["top_k_sharpest"])
            ),
            sharpness_column=str(item.get("sharpness_column", "image_focus")),
        )
        for item in payload["datasets"]
    ]
    if not datasets:
        raise ValueError("StageConfig.datasets não pode ser vazio.")
    max_samples = payload.get("max_samples")
    return StageConfig(
        datasets=datasets,
        steps=int(payload["steps"]),
        batch_size=int(payload.get("batch_size", 1)),
        image_size=int(payload.get("image_size", 512)),
        shuffle=bool(payload.get("shuffle", True)),
        max_samples=None if max_samples is None else int(max_samples),
        augment=bool(payload.get("augment", True)),
    )


def _coerce_config_dict(payload: dict[str, Any]) -> TrainConfig:
    model = ModelConfig(**payload.get("model", {}))

    runtime_payload = dict(payload.get("runtime", {}))
    mp = runtime_payload.get("mixed_precision")
    if isinstance(mp, bool):
        runtime_payload["mixed_precision"] = "bf16" if mp else "no"
    elif isinstance(mp, str):
        runtime_payload["mixed_precision"] = mp.lower()
    runtime = RuntimeConfig(**runtime_payload)

    optimizer = OptimizerConfig(**payload.get("optimizer", {}))
    scheduler = SchedulerConfig(**payload.get("scheduler", {}))

    data_block = payload.get("data", {})
    deblur_payload = data_block.get("deblur")
    if deblur_payload is None:
        deblur = StageConfig(datasets=_paper_deblur_sources(), steps=60000)
    else:
        deblur = _as_stage_config(deblur_payload)
    data = DataConfig(deblur=deblur)

    logging = LoggingConfig(**payload.get("logging", {}))
    return TrainConfig(
        model=model,
        runtime=runtime,
        optimizer=optimizer,
        scheduler=scheduler,
        data=data,
        logging=logging,
    )


def load_config(path: str | Path) -> TrainConfig:
    path = Path(path)
    with path.open("r", encoding="utf-8") as h:
        payload = yaml.safe_load(h) or {}
    return _coerce_config_dict(payload)


def to_dict(config: TrainConfig) -> dict[str, Any]:
    return asdict(config)


def write_effective_config(config: TrainConfig, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as h:
        yaml.safe_dump(to_dict(config), h, sort_keys=False)


def config_hash(config: TrainConfig) -> str:
    payload = yaml.safe_dump(to_dict(config), sort_keys=True)
    return sha256(payload.encode("utf-8")).hexdigest()[:16]
