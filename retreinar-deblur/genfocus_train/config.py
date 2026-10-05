"""Schema de configuração do treino da DeblurNet (GenRefocus, Stage 1).

ÁRVORE NOVA (retreinar-deblur/). O treino antigo continua intacto em
`genrefocus_deblurnet_paper/` e `genrefocus_deblurnet/` — nada aqui o afeta.

Esta config é o CONTRATO entre backbone.py, data.py, trainer.py e os YAMLs.
Todo campo novo desta revisão existe por um motivo registrado em
`MUDANCAS_CODIGO.md` e `PLANO_CORRECOES_DEBLURNET.md`; a referência (C1..C11)
está no comentário de cada um.

PRINCÍPIO: nada que muda o modelo pode ficar hardcodado. Um experimento
fatorial não se roda com valor fixo no código, e um checkpoint sem o registro
do que o gerou não é reproduzível.

ATENÇÃO — armadilha conhecida: `StageConfig` é montado campo a campo em
`_as_stage_config`. Adicionar um campo na dataclass SEM adicionar a linha
correspondente lá faz a chave do YAML ser SILENCIOSAMENTE IGNORADA.
Há teste que cobre exatamente isso (tests/test_config_roundtrip.py).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from hashlib import sha256
from pathlib import Path
from typing import Any

import yaml


# =============================================================================
# Vocabulários fechados (validados no load — um typo no YAML tem que explodir,
# não virar comportamento silencioso)
# =============================================================================

SCALE_MODES = ("short_side", "native", "long_side")
SIGMA_MU_SOURCES = ("crop", "full_image")
TOP_K_MODES = ("row", "scene")
DEFOCUS_SOURCES = ("recompute", "column", "kfix")


def _check(value: str, allowed: tuple[str, ...], campo: str) -> str:
    if value not in allowed:
        raise ValueError(
            f"{campo}={value!r} inválido. Valores aceitos: {list(allowed)}."
        )
    return value


@dataclass
class OptimizerConfig:
    name: str = "adamw"
    lr: float = 1.0e-4
    weight_decay: float = 1.0e-4
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1.0e-8


@dataclass
class SchedulerConfig:
    name: str = "cosine"
    warmup_steps: int = 500
    min_lr_ratio: float = 0.1


@dataclass
class ModelConfig:
    pretrained_model_name_or_path: str = "black-forest-labs/FLUX.1-dev"
    vae_subfolder: str = "vae"
    transformer_subfolder: str = "transformer"
    max_coc: float = 100.0
    deblur_lora_rank: int = 128    # paper §4.1 (o artefato oficial tem 64; ver plano)
    bokeh_lora_rank: int = 64      # paper §4.1
    gradient_checkpointing: bool = True

    # ── C1 ──────────────────────────────────────────────────────────────────
    # Nº de módulos LoRA que o `add_adapter` DEVE produzir. O checkpoint
    # oficial (bokehNet.safetensors, 686 tensores) tem 343 módulos:
    #   19 duplos × 6 + 38 single × 6 + x_embedder = 343.
    # Antes desta revisão injetávamos 344, porque a string solta "proj_out"
    # capturava também a projeção final `transformer.proj_out`, que roda FORA
    # do controle por branch do `specify_lora`. A trava aborta o treino se a
    # contagem mudar. 0 = desliga a trava (não recomendado).
    expected_lora_modules: int = 343

    # ── C2 ──────────────────────────────────────────────────────────────────
    # Guidance embutido no `temb` de cada branch. NÃO é CFG: o FLUX.1-dev é
    # guidance-distilled e recebe esse escalar como ENTRADA do modelo.
    # FATO conferido: a inferência oficial usa valores diferentes por estágio —
    #   deblur: Inference_deblurNet.py e demo.py chamam generate() SEM
    #           guidance_scale, e o default de `generate` é 3.5.
    #   bokeh:  Inference_bokehNet.py e demo.py passam guidance_scale=1.0.
    #   condição: 1.0 nos dois casos (c_guidances = torch.ones).
    # HIPÓTESE não medida: que treinar com o mesmo valor da inferência é melhor.
    # O default 3.5 no deblur é a única escolha que roda no script oficial sem
    # parâmetro extra — A CONFIRMAR pelo experimento 2×2 do plano (C2).
    deblur_train_guidance: float = 3.5
    bokeh_train_guidance: float = 1.0
    cond_train_guidance: float = 1.0

    # ── C8 ──────────────────────────────────────────────────────────────────
    # Em qual branch a LoRA age. A inferência oficial usa main_adapter=None,
    # que dá [texto=None, main=None, cond=LoRA] = cond-only.
    # `lora_on_main=True` reproduz a variante main+cond (o modelo de 60k já
    # avaliado) e EXIGE `main_adapter="deblurring"` na inferência.
    # `lora_on_text` existe só para deixar explícito que o treino NUNCA põe
    # LoRA no texto — é o descasamento do C6. Manter False.
    lora_on_main: bool = False
    lora_on_text: bool = False


@dataclass
class RuntimeConfig:
    output_dir: str = "outputs"
    seed: int = 42
    num_workers: int = 4
    pin_memory: bool = True
    mixed_precision: str = "bf16"
    gradient_accumulation_steps: int = 8
    max_grad_norm: float = 1.0
    log_every_steps: int = 10
    save_every_steps: int = 250
    keep_last_n_checkpoints: int = 5
    resume: bool = True

    # ── C7 ──────────────────────────────────────────────────────────────────
    # Seed diferente por rank. Os pesos LoRA são sincronizados por broadcast
    # explícito, então RNG distinto entre ranks é o que queremos: com a mesma
    # seed em todos, as 4 GPUs sorteavam o MESMO sigma e o MESMO ruído, e o
    # batch efetivo 32 tinha só 8 sigmas distintos.
    seed_per_rank: bool = True

    # ── C10 ─────────────────────────────────────────────────────────────────
    # Validação com sigma e ruído FIXOS (determinística), para a métrica ser
    # comparável entre steps. 0 = desligada.
    eval_every_steps: int = 500
    eval_max_samples: int = 32
    eval_num_sigmas: int = 9       # grade fixa em (0,1)
    eval_seed: int = 1234
    save_best: bool = True         # guarda o melhor por val_loss, além dos N últimos

    # ── perda ponderada por oclusão (BokehNet; herdado, sem mudança) ────────
    occlusion_lambda: float = 0.0
    occlusion_pool: str = "max"
    occlusion_theta: float = 0.0
    geo_branches: bool = False


@dataclass
class DatasetSourceConfig:
    name: str
    split: str
    # paper §B.1: manter as N imagens mais nítidas por variância do Laplaciano.
    top_k_sharpest: int | None = None
    sharpness_column: str = "image_focus"


@dataclass
class StageConfig:
    datasets: list[DatasetSourceConfig]
    steps: int
    batch_size: int = 1
    image_size: int = 512
    shuffle: bool = True
    max_samples: int | None = None
    augment: bool = True

    # ── C10 ─────────────────────────────────────────────────────────────────
    # Fontes de validação. Vazio = sem validação (e o eval_every_steps é
    # ignorado, com aviso no log).
    val_datasets: list[DatasetSourceConfig] = field(default_factory=list)

    # ── C4, eixo 1: escala espacial ─────────────────────────────────────────
    # FATO: o treino e a avaliação hoje descasam em escala, e descasam em
    # sentidos OPOSTOS conforme o long_side usado na inferência.
    #   "short_side" — lado MENOR vai para image_size, depois crop image_size².
    #                  É o comportamento anterior a esta revisão.
    #   "native"     — SEM resize; crop image_size² no pixel nativo (só faz
    #                  upscale se algum lado for menor que image_size). É o
    #                  mesmo tile que a inferência com tiling vê.
    #   "long_side"  — lado MAIOR vai para image_size, imagem INTEIRA (sem
    #                  crop). Produz aspecto variável: exige batch_size=1 ou
    #                  bucketing. Só para o run de referência do plano.
    scale_mode: str = "short_side"

    # ── C4, eixo 2: de onde sai o mu do sigma ───────────────────────────────
    #   "crop"       — mu de calculate_shift(tokens do crop). Comportamento
    #                  anterior a esta revisão.
    #   "full_image" — mu de calculate_shift(tokens da imagem de origem
    #                  inteira). É o que a inferência faz: em flux.py o
    #                  `image_seq_len` sai do latente da imagem COMPLETA,
    #                  antes do tiling, e todo tile herda esse cronograma.
    # HIPÓTESE não demonstrada: que casar isso melhora. O modelo é condicionado
    # em sigma e o treino cobre (0,1) inteiro — é desbalanceamento de DENSIDADE,
    # não fora-de-domínio. Por isso é eixo de experimento, não correção.
    sigma_mu_source: str = "crop"

    # ── C5: granularidade do filtro de nitidez ──────────────────────────────
    #   "row"   — ranqueia LINHAS (comportamento anterior). Como o score é
    #             medido em `image_focus`, que é a mesma imagem em todas as
    #             linhas de uma cena, os empates são exatos e o top-k puxa
    #             cenas inteiras.
    #   "scene" — ranqueia CENAS e sorteia a linha dentro da cena a cada
    #             __getitem__. ALTERAÇÃO DE PROTOCOLO: a fidelidade ao paper
    #             não está provada ("top 3,000 sharpest images" é ambíguo num
    #             df que é cena × abertura). Tem que ser medida em A/B.
    top_k_mode: str = "row"
    # Como identificar a cena quando top_k_mode="scene".
    #   "auto"  — tenta, nesta ordem: coluna `scene`/`stem`; prefixo de
    #             `file_name_base`; hash dos bytes de `image_focus`.
    # Ou o nome literal de uma coluna, ou "image_hash", ou "file_name_prefix".
    scene_key: str = "auto"

    # ── BokehNet (herdado, sem mudança nesta revisão) ────────────────────────
    defocus_source: str = "recompute"
    min_calibration_ssim: float | None = None
    kfix_repo: str | None = None
    geo_condition: bool = False
    geo_escalares: str | None = None
    geo_constantes: dict | None = None
    geo_field: str = "inverse"
    geo_sem_escalares: str = "erro"
    geo_ruido_controle: bool = False

    def __post_init__(self) -> None:
        _check(self.scale_mode, SCALE_MODES, "scale_mode")
        _check(self.sigma_mu_source, SIGMA_MU_SOURCES, "sigma_mu_source")
        _check(self.top_k_mode, TOP_K_MODES, "top_k_mode")
        _check(self.defocus_source, DEFOCUS_SOURCES, "defocus_source")
        if self.image_size % 16 != 0:
            raise ValueError(
                f"image_size={self.image_size} tem que ser múltiplo de 16 "
                "(VAE 8× + _pack_latents 2× do FLUX)."
            )
        if self.scale_mode == "long_side" and self.batch_size > 1:
            raise ValueError(
                "scale_mode='long_side' produz aspecto variável por amostra; "
                "o collate exige shapes iguais. Use batch_size=1."
            )


def _paper_deblur_sources() -> list[DatasetSourceConfig]:
    """Composição do paper §B.1 para a DeblurNet: DPDD completo + subset RealBokeh."""
    return [
        DatasetSourceConfig(name="akcit-pixel/DDPD", split="train"),
        DatasetSourceConfig(
            name="akcit-pixel/RealBokeh", split="train",
            top_k_sharpest=3000, sharpness_column="image_focus",
        ),
    ]


def _paper_deblur_val_sources() -> list[DatasetSourceConfig]:
    return [DatasetSourceConfig(name="akcit-pixel/DDPD", split="validation")]


@dataclass
class DataConfig:
    deblur: StageConfig = field(
        default_factory=lambda: StageConfig(
            datasets=_paper_deblur_sources(),
            val_datasets=_paper_deblur_val_sources(),
            steps=60000,
        )
    )
    bokeh: StageConfig | None = None


@dataclass
class LoggingConfig:
    # ── wandb ───────────────────────────────────────────────────────────────
    use_wandb: bool = True
    wandb_project: str = "genrefocus-deblurnet"
    wandb_entity: str | None = None
    run_name: str | None = None
    # Re-`sbatch` tem que CONTINUAR o mesmo run, não abrir um novo — senão o
    # gráfico da loss fica picado em N runs e a curva não se lê. O id é
    # derivado do output_dir e persistido em <output_dir>/<stage>/wandb_id.txt.
    wandb_resume: bool = True
    wandb_id: str | None = None       # None = deriva/persiste automaticamente

    # ── Hugging Face ────────────────────────────────────────────────────────
    # Snapshot periódico do LoRA. UM repo, um ARQUIVO por step, para o
    # histórico do treino ficar todo no mesmo lugar.
    upload_every_steps: int = 1000
    upload_hf_repo_base: str | None = None
    upload_final: bool = True         # sempre sobe o último, mesmo sem periódico
    upload_best: bool = True          # sobe também o melhor por val_loss
    upload_private: bool = True       # repo privado por padrão
    # Upload NUNCA derruba o treino: falha de rede/token só emite aviso.
    upload_retries: int = 3


@dataclass
class TrainConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    data: DataConfig = field(default_factory=DataConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)


def _as_sources(payload: list[dict[str, Any]] | None) -> list[DatasetSourceConfig]:
    if not payload:
        return []
    return [
        DatasetSourceConfig(
            name=str(item["name"]),
            split=str(item["split"]),
            top_k_sharpest=(
                None if item.get("top_k_sharpest") is None else int(item["top_k_sharpest"])
            ),
            sharpness_column=str(item.get("sharpness_column", "image_focus")),
        )
        for item in payload
    ]


def _as_stage_config(payload: dict[str, Any]) -> StageConfig:
    datasets = _as_sources(payload.get("datasets"))
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
        val_datasets=_as_sources(payload.get("val_datasets")),
        # ── campos desta revisão — ver a ARMADILHA no topo do módulo ────────
        scale_mode=str(payload.get("scale_mode", "short_side")),
        sigma_mu_source=str(payload.get("sigma_mu_source", "crop")),
        top_k_mode=str(payload.get("top_k_mode", "row")),
        scene_key=str(payload.get("scene_key", "auto")),
        # ── BokehNet (herdado) ──────────────────────────────────────────────
        defocus_source=str(payload.get("defocus_source", "recompute")),
        min_calibration_ssim=None if min_ssim is None else float(min_ssim),
        kfix_repo=payload.get("kfix_repo"),
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
        deblur = StageConfig(
            datasets=_paper_deblur_sources(),
            val_datasets=_paper_deblur_val_sources(),
            steps=60000,
        )
    else:
        deblur = _as_stage_config(deblur_payload)

    bokeh_payload = data_block.get("bokeh")
    bokeh = None if bokeh_payload is None else _as_stage_config(bokeh_payload)

    logging = LoggingConfig(**payload.get("logging", {}))
    return TrainConfig(
        model=model, runtime=runtime, optimizer=optimizer,
        scheduler=scheduler, data=DataConfig(deblur=deblur, bokeh=bokeh),
        logging=logging,
    )


def load_config(path: str | Path) -> TrainConfig:
    path = Path(path)
    with path.open("r", encoding="utf-8") as h:
        payload = yaml.safe_load(h) or {}
    return _coerce_config_dict(payload)


def stage_config(config: TrainConfig, stage: str) -> StageConfig:
    """StageConfig do estágio, com erro claro se faltar o bloco no YAML."""
    if stage == "deblur":
        return config.data.deblur
    if stage == "bokeh":
        if config.data.bokeh is None:
            raise ValueError(
                "config.data.bokeh não definido (falta o bloco `data.bokeh:` no YAML)."
            )
        return config.data.bokeh
    raise ValueError(f"stage inválido: {stage!r} (esperado 'deblur' ou 'bokeh')")


def train_guidance_for(config: TrainConfig, stage: str) -> float:
    """C2 — guidance do branch principal/texto no treino, por estágio."""
    if stage == "deblur":
        return float(config.model.deblur_train_guidance)
    if stage == "bokeh":
        return float(config.model.bokeh_train_guidance)
    raise ValueError(f"stage inválido: {stage!r}")


def lora_rank_for(config: TrainConfig, stage: str) -> int:
    if stage == "deblur":
        return int(config.model.deblur_lora_rank)
    if stage == "bokeh":
        return int(config.model.bokeh_lora_rank)
    raise ValueError(f"stage inválido: {stage!r}")


def to_dict(config: TrainConfig) -> dict[str, Any]:
    return asdict(config)


def write_effective_config(config: TrainConfig, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as h:
        yaml.safe_dump(to_dict(config), h, sort_keys=False, allow_unicode=True)


def config_hash(config: TrainConfig) -> str:
    payload = yaml.safe_dump(to_dict(config), sort_keys=True, allow_unicode=True)
    return sha256(payload.encode("utf-8")).hexdigest()[:16]
