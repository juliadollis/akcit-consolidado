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
    # ── Perda ponderada por oclusão (plano §5.3, em espaço de token) ────────
    # w = 1 + occlusion_lambda * O(x), com O reduzido 16× e reempacotado na
    # ordem do _pack_latents. 0.0 = perda anterior, BIT A BIT (há teste).
    # O documento sugere entre 1 e 3, calibrado pela fração da perda que vem da
    # região de borda (logada como `loss_frac_borda`).
    occlusion_lambda: float = 0.0
    # "max" (default) ou "avg". Max porque uma borda de oclusão tem 1 a 2 px e a
    # média a 16× dilui a amplitude por ~1/16: um lambda de 3 viraria 0,19.
    # "avg" existe como variante de ablação. Ver geo_cond/loss_weight.py.
    occlusion_pool: str = "max"
    # Limiar sobre o O JA REDUZIDO A TOKEN: w = 1 + lambda * [O_token > theta].
    # 0.0 = peso continuo (o que a condicao A' usou).
    #
    # Existe por medicao: O_token e fortemente assimetrico (p50 = 0,019 mas
    # p90 = 0,639), entao com peso continuo a normalizacao pela media rebaixa
    # ~75% dos tokens a ~0,84 para financiar o topo. Foi o que fez o A' piorar
    # FORA da borda (E_fora +31%). Ver REGISTRO_GEO_COND.md 2026-09-07.
    occlusion_theta: float = 0.0
    # Alimenta os 2 branches geométricos ao transformer. False com
    # geo_condition=True é a condição A' do plano: o canal O ainda é calculado
    # (a perda ponderada precisa dele) mas a arquitetura não muda. É o que
    # separa "a perda ajudou" de "os canais ajudaram".
    geo_branches: bool = False


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
    # BokehNet: origem do mapa de defocus. Ver DEFOCUS_SOURCES em data.py.
    #   "recompute" (default, CORRETO): monta clip(k*|depth-s1|/max_coc, 0, 1) a
    #       partir das colunas depth+k+s1, replicando a inferência oficial.
    #   "column": usa a coluna `defocus_map` do df, que está normalizada POR
    #       IMAGEM e portanto NÃO carrega o K. Só para comparação com o
    #       comportamento antigo — não treina controle de bokeh.
    defocus_source: str = "recompute"
    # FILTRO DE QUALIDADE DA CALIBRAÇÃO DO K (paper §3.2(c), literal):
    #   "The selected K* is then used as the pseudo-bokeh-level label for
    #    training, provided that its corresponding SSIM exceeds a predefined
    #    threshold to ensure reliable supervision."
    # O K da rota c vem de um sweep que maximiza SSIM contra o bokeh real
    # (Eq. 5); a coluna `calibration_ssim` guarda o SSIM alcançado. Amostras
    # abaixo do limiar tiveram calibração RUIM e o K delas é ruído.
    #
    # Aplica-se SÓ a fontes que TÊM a coluna (rota c). A rota b deriva o K da
    # EXIF pela Eq. 3, não por sweep, então não tem SSIM e passa inteira — que
    # é o comportamento do paper, onde o limiar é descrito apenas em (c).
    #
    # None = sem filtro (comportamento anterior). O paper não publica o valor do
    # limiar ("predefined threshold"), então o nosso é escolha documentada.
    min_calibration_ssim: float | None = None
    # Repo HF com a tabela de K corrigido pela Eq. 3 (defocus_source="kfix").
    kfix_repo: str | None = None
    # ── Condicionamento geométrico (geo_cond) ───────────────────────────────
    # False = comportamento anterior, sem custo e sem chave nova no batch.
    geo_condition: bool = False
    # Tabela por `stem` com z_min_m, z_max_m_bruto, z_focus_m, focallength_px,
    # largura_px, altura_px. Caminho .jsonl local ou repo HF. Gerada por
    # geo_cond/jobs/f0b_escala_metrica.py.
    geo_escalares: str | None = None
    # As 8 constantes de geo_cond/constants.py. MEDIDAS no conjunto de treino,
    # sem default: um número plausível silencioso reintroduz a classe de defeito
    # que a auditoria encontrou.
    geo_constantes: dict | None = None
    # "inverse" (u=1/Z, default por física) ou "depth" (Z).
    geo_field: str = "inverse"
    # Amostra sem escalares: "erro" aborta, "pula" descarta. NUNCA inventa.
    geo_sem_escalares: str = "erro"
    # CONTROLE do plano: substitui G por ruído de MESMA ESTATISTICA (média e
    # desvio por canal, medidos no próprio batch). Se o ganho persistir, ele veio
    # do aumento de capacidade e não da informação geométrica.
    geo_ruido_controle: bool = False


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
    # Stage 2. Paper §4.1: "BokehNet is trained in two stages: (i) 40K steps
    # on synthetic data, and (ii) 60K steps on real data." Currículo de 2
    # fases implementado via YAMLs separados (train_bokeh_synth_*.yaml,
    # train_bokeh_real_*.yaml) + --init-lora. None = stage de bokeh não
    # configurado neste YAML (default só usado por quem não passa `data.bokeh`).
    bokeh: StageConfig | None = None


@dataclass
class LoggingConfig:
    use_wandb: bool = True
    wandb_project: str = "genrefocus-train"
    wandb_entity: str | None = None
    run_name: str | None = None
    # Upload periódico do LoRA pro HF durante o treino. 0 = desligado.
    # A cada `upload_every_steps`, sobe pra um repo POR-STEP: <base>-step<N>
    # (guarda snapshots de vários steps pra comparar depois).
    upload_every_steps: int = 0
    upload_hf_repo_base: str | None = None


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
    min_ssim = payload.get("min_calibration_ssim")
    return StageConfig(
        datasets=datasets,
        steps=int(payload["steps"]),
        batch_size=int(payload.get("batch_size", 1)),
        image_size=int(payload.get("image_size", 512)),
        shuffle=bool(payload.get("shuffle", True)),
        max_samples=None if max_samples is None else int(max_samples),
        augment=bool(payload.get("augment", True)),
        defocus_source=str(payload.get("defocus_source", "recompute")),
        min_calibration_ssim=None if min_ssim is None else float(min_ssim),
        kfix_repo=payload.get("kfix_repo"),
        # ATENCAO: StageConfig e montado campo a campo AQUI. Adicionar o campo na
        # dataclass sem adicionar a linha abaixo faz a chave do YAML ser
        # SILENCIOSAMENTE IGNORADA. Ha teste que cobre exatamente isso.
        geo_condition=bool(payload.get("geo_condition", False)),
        geo_escalares=payload.get("geo_escalares"),
        geo_constantes=payload.get("geo_constantes"),
        geo_field=str(payload.get("geo_field", "inverse")),
        geo_sem_escalares=str(payload.get("geo_sem_escalares", "erro")),
        geo_ruido_controle=bool(payload.get("geo_ruido_controle", False)),
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

    bokeh_payload = data_block.get("bokeh")
    bokeh = None if bokeh_payload is None else _as_stage_config(bokeh_payload)

    data = DataConfig(deblur=deblur, bokeh=bokeh)

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
