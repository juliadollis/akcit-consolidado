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
# "metric_disparity" é o contrato do release novo (ver genfocus_train/control.py).
# As outras três são APOSENTADAS e exigem `allow_retired_defocus_sources: true`.
DEFOCUS_SOURCES = ("metric_disparity", "recompute", "column", "kfix")
RETIRED_DEFOCUS_SOURCES = ("recompute", "column", "kfix")
# Forma do release (L1). "arvore" é o que o `bokehnet-regen` publica;
# "tabela" é o caminho antigo, com os pixels nas colunas. Ver release.py.
RELEASE_FORMATS = ("auto", "arvore", "tabela")


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
    # ATENÇÃO (T4 da AUDITORIA_TREINO_BOKEHNET.md): no caminho
    # `defocus_source="metric_disparity"` este campo NÃO normaliza nada — o
    # normalizador é `control.MAX_COC`, constante de módulo, sem setter. O campo
    # sobrevive como ASSERÇÃO: se alguém escrever outro valor no YAML, o load
    # falha em vez de o mapa ser normalizado por um valor enquanto o metadado
    # diz outro. Foi assim que o experimento `kfix` rodou duas convenções no
    # mesmo lote (max_coc=10,5107 na rota B, 100,0 na C).
    max_coc: float = 100.0
    deblur_lora_rank: int = 128    # paper §4.1 (o artefato oficial tem 64; ver plano)
    bokeh_lora_rank: int = 64      # paper §4.1
    # §3.3: "we freeze all original LoRA weights and introduce a new, trainable
    # LoRA module". O paper NÃO publica o rank desse LoRA de forma; 64 é o mesmo
    # da BokehNet, escolha nossa declarada.
    bokeh_shape_lora_rank: int = 64
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
    bokeh_shape_train_guidance: float = 1.0   # §3.3 é fine-tune da BokehNet
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

    # ── T11 — PROBE DE CONTROLABILIDADE (LVCorr) ────────────────────────────
    # O instrumento que faltava. A loss de flow matching é cega para o modo de
    # falha que custou a fase 2 (LVCorr +0,9059 -> +0,4365, degradação MONÓTONA
    # no tempo de treino, com a loss caindo o tempo todo). Ver genfocus_train/probe.py.
    # 0 = desligado.
    probe_every_steps: int = 5000
    # Diretório com o conjunto FIXO do probe (.npz por imagem). None = probe
    # DESLIGADO, com aviso GRITADO no início do run: um treino de bokeh sem este
    # número não tem instrumento sensível ao seu modo de falha conhecido.
    probe_set_dir: str | None = None
    probe_num_inference_steps: int = 28   # igual à inferência oficial
    probe_seed: int = 1234
    # Piso de alerta: queda desta magnitude em relação ao MELHOR LVCorr já visto
    # no run emite alarme no log. Critério declarado ANTES de rodar, para não
    # virar racionalização depois.
    probe_alerta_queda: float = 0.10
    # Quantos probes SEGUIDOS com alarme param o run. O PLANO_RETREINO declara
    # "dois probes consecutivos" ANTES do run, de propósito: critério fixado
    # depois de ver o resultado é racionalização. 0 = só avisa, não para.
    probe_alarmes_para_parar: int = 2

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

    # ── BokehNet — contrato `metric_disparity_official_v1` ───────────────────
    # Default trocado nesta árvore. Ver T1/T5 da AUDITORIA_TREINO_BOKEHNET.md.
    defocus_source: str = "metric_disparity"
    # §3.2(c): "provided that its corresponding SSIM exceeds a predefined
    # threshold". O paper NÃO publica o valor. O 0,6 do treino anterior foi
    # calibrado sobre a distribuição ANTIGA, gerada com um renderer que nunca era
    # o BokehMe e com o K na convenção errada — por isso o default aqui é None
    # (falha explícita) em vez de herdar 0,6 por inércia. Ver T8.
    min_calibration_ssim: float | None = None
    kfix_repo: str | None = None
    # Filtros do release (T7). `is_valid_for_control` já embute
    # `not is_k_censored`; os dois existem separados para o histograma de
    # descarte dizer QUAL condição pegou.
    require_valid_for_control: bool = True
    exclude_censored_k: bool = True
    # DECISÃO NOSSA (A8), não literal do paper: o supplement B.2 descreve os 13K
    # que os AUTORES coletaram como séries de "2 to 4 images per set", o que é a
    # composição do dado deles e não um filtro para as outras rotas. Medido:
    # 25% das amostras vinham de 6,2% das cenas. None = sem teto. Ver T9.
    max_levels_per_scene: int | None = 4
    # Proporção entre rotas, por NOME DE FONTE. None = proporcional ao tamanho
    # dos shards, que é acidente e não decisão. Ver T10.
    route_weights: dict | None = None
    # Excluir amostras cuja região de foco veio do refinamento automático (o
    # substituto do passo manual do §3.2(c)). Existe para MEDIR o efeito do
    # refinamento, que é requisito explícito do bokehnet-regen.
    exclude_refined_focus: bool = False
    # Split por CENA, lido de manifesto — nunca sorteado no dataloader. Gate nº 20
    # do PLANO_REGERACAO: "nenhum scene_id aparece em treino e validação". Ver T6.
    scene_split_manifest: str | None = None
    scene_split_partition: str = "train"
    # Porta das convenções aposentadas. Ver T5.
    allow_retired_defocus_sources: bool = False
    # ── L1: a FORMA do release. Ver `genfocus_train/release.py` ─────────────
    # O `bokehnet-regen` publica uma ÁRVORE DE ARQUIVOS (`manifest.jsonl`,
    # `depth/<id>.png`, `meta/<id>.json`, `split.json`), não uma tabela com os
    # pixels nas colunas. "auto" decide pelo que dá para ver SEM REDE (pasta
    # local com manifesto, ou snapshot já em disco); "arvore" força o leitor
    # novo, que é o que um repo do Hub do release exige.
    release_format: str = "auto"
    # `source_dataset` -> diretório do snapshot local do espelho de origem. Só
    # é preciso quando o release NÃO é autocontido: por decisão de cota (365 GB
    # contra 115 GB livres) a rota C referencia AIF e bokeh, a B referencia a
    # bokeh e a A referencia a AIF. Ausência não vira default — a amostra
    # levanta nomeando a fonte que faltou.
    mirror_roots: dict | None = None
    # Baixar release/espelho do Hub. Default False, e é decisão declarada: uma
    # auditoria deste projeto mediu 552 GB de egress inexplicado vindo de
    # download automático. Ler do disco é o default; baixar se pede.
    permitir_download_do_release: bool = False
    # Conferir o sha256 dos bytes de origem contra `source_images.jsonl`. É o
    # que faz do join com o espelho uma evidência em vez de uma afirmação.
    verificar_sha256_da_origem: bool = True
    # ── Fase 2: repetição de sintético (desvio DECLARADO, ver T16) ───────────
    # O §4.1 é literal: "(ii) 60K steps on real data". 0.0 reproduz isso e é o
    # braço primário. >0 mistura essa fração de batches da rota A na fase 2, para
    # testar a hipótese de esquecimento catastrófico do controle de K
    # (LVCorr +0,9059 na fase 1 -> +0,4365 na fase 2, degradação MONÓTONA).
    synthetic_replay_fraction: float = 0.0
    synthetic_replay_datasets: list[DatasetSourceConfig] = field(default_factory=list)
    # ── §3.3: condicionamento por forma de abertura ─────────────────────────
    # Coluna do release com a imagem do kernel de abertura. Quando presente, a
    # BokehNet recebe uma TERCEIRA condição e o estágio passa a ser o da §3.3.
    shape_column: str | None = None
    geo_condition: bool = False
    geo_escalares: str | None = None
    geo_constantes: dict | None = None
    geo_field: str = "inverse"
    geo_sem_escalares: str = "erro"
    geo_ruido_controle: bool = False
    excluir_cenas_de_avaliacao: str | None = None

    def __post_init__(self) -> None:
        _check(self.scale_mode, SCALE_MODES, "scale_mode")
        _check(self.sigma_mu_source, SIGMA_MU_SOURCES, "sigma_mu_source")
        _check(self.top_k_mode, TOP_K_MODES, "top_k_mode")
        _check(self.defocus_source, DEFOCUS_SOURCES, "defocus_source")
        _check(self.release_format, RELEASE_FORMATS, "release_format")
        if self.mirror_roots is not None:
            if not isinstance(self.mirror_roots, dict) or not self.mirror_roots:
                raise ValueError(
                    "mirror_roots tem que ser um mapa não vazio "
                    "`{source_dataset: caminho_do_snapshot}`. Vazio seria o mesmo "
                    "que ausente, e ausente já tem significado: resolver pixel só "
                    "pelo release."
                )
            for nome, caminho in self.mirror_roots.items():
                if not str(caminho).strip():
                    raise ValueError(
                        f"mirror_roots[{nome!r}] vazio. Caminho em branco viraria "
                        "o diretório corrente, e aí o índice do espelho sairia de "
                        "onde ninguém pediu."
                    )
        if self.image_size % 16 != 0:
            raise ValueError(
                f"image_size={self.image_size} tem que ser múltiplo de 16 "
                "(VAE 8× + _pack_latents 2× do FLUX)."
            )
        if self.defocus_source in RETIRED_DEFOCUS_SOURCES and not self.allow_retired_defocus_sources:
            raise ValueError(
                f"defocus_source={self.defocus_source!r} é convenção APOSENTADA. "
                "Para reproduzir um run histórico, declare "
                "`allow_retired_defocus_sources: true` no YAML."
            )
        if not (0.0 <= float(self.synthetic_replay_fraction) < 1.0):
            raise ValueError(
                f"synthetic_replay_fraction={self.synthetic_replay_fraction} fora de [0,1)."
            )
        if self.synthetic_replay_fraction > 0 and not self.synthetic_replay_datasets:
            raise ValueError(
                "synthetic_replay_fraction > 0 exige `synthetic_replay_datasets` "
                "(a rota sintética de onde repetir). Sem fonte, a fração seria "
                "silenciosamente ignorada."
            )
        if self.max_levels_per_scene is not None and int(self.max_levels_per_scene) < 1:
            raise ValueError(
                f"max_levels_per_scene={self.max_levels_per_scene} < 1 zeraria o dataset."
            )
        if self.route_weights is not None:
            for nome, peso in self.route_weights.items():
                if float(peso) < 0:
                    raise ValueError(f"route_weights[{nome!r}]={peso} negativo.")
            if sum(float(v) for v in self.route_weights.values()) <= 0:
                raise ValueError("route_weights soma zero: o dataset ficaria vazio.")
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
        # ── BokehNet — contrato novo. ATENÇÃO à ARMADILHA no topo do módulo:
        # adicionar o campo na dataclass SEM adicionar a linha aqui faz a chave
        # do YAML ser SILENCIOSAMENTE IGNORADA. Há teste que cobre isso.
        defocus_source=str(payload.get("defocus_source", "metric_disparity")),
        min_calibration_ssim=None if min_ssim is None else float(min_ssim),
        kfix_repo=payload.get("kfix_repo"),
        require_valid_for_control=bool(payload.get("require_valid_for_control", True)),
        exclude_censored_k=bool(payload.get("exclude_censored_k", True)),
        max_levels_per_scene=(
            None if payload.get("max_levels_per_scene", 4) is None
            else int(payload.get("max_levels_per_scene", 4))
        ),
        route_weights=payload.get("route_weights"),
        exclude_refined_focus=bool(payload.get("exclude_refined_focus", False)),
        scene_split_manifest=payload.get("scene_split_manifest"),
        scene_split_partition=str(payload.get("scene_split_partition", "train")),
        allow_retired_defocus_sources=bool(
            payload.get("allow_retired_defocus_sources", False)
        ),
        # ── L1 — ARMADILHA do topo do módulo vale aqui também ───────────────
        release_format=str(payload.get("release_format", "auto")),
        mirror_roots=payload.get("mirror_roots"),
        permitir_download_do_release=bool(
            payload.get("permitir_download_do_release", False)
        ),
        verificar_sha256_da_origem=bool(
            payload.get("verificar_sha256_da_origem", True)
        ),
        synthetic_replay_fraction=float(payload.get("synthetic_replay_fraction", 0.0)),
        synthetic_replay_datasets=_as_sources(payload.get("synthetic_replay_datasets")),
        shape_column=payload.get("shape_column"),
        geo_condition=bool(payload.get("geo_condition", False)),
        geo_escalares=payload.get("geo_escalares"),
        geo_constantes=payload.get("geo_constantes"),
        geo_field=str(payload.get("geo_field", "inverse")),
        geo_sem_escalares=str(payload.get("geo_sem_escalares", "erro")),
        geo_ruido_controle=bool(payload.get("geo_ruido_controle", False)),
        excluir_cenas_de_avaliacao=payload.get("excluir_cenas_de_avaliacao"),
    )


def _coerce_config_dict(payload: dict[str, Any]) -> TrainConfig:
    model = ModelConfig(**payload.get("model", {}))
    # T4 — `max_coc` é ASSERÇÃO, não parâmetro. O normalizador real vive em
    # `genfocus_train/control.py:MAX_COC` e não tem setter. Um YAML que escreva
    # outro valor está pedindo normalização por fonte, que é o defeito D1 com
    # granularidade grossa — o `kfix` com max_coc=10,5107 subiu o LF-Bokeh e
    # derrubou RealBokeh e RealDOF no mesmo lote.
    from . import control as _control
    if abs(float(model.max_coc) - _control.MAX_COC) > 1e-9:
        raise ValueError(
            f"model.max_coc={model.max_coc} != {_control.MAX_COC}. Este campo é "
            "uma asserção do contrato, não um ajuste. Para mudar o normalizador "
            "seria preciso mudar `control.MAX_COC`, subir o CONTROL_VERSION e "
            "REGERAR os dados."
        )

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
    if stage in ("bokeh", "bokeh_shape"):
        if config.data.bokeh is None:
            raise ValueError(
                "config.data.bokeh não definido (falta o bloco `data.bokeh:` no YAML)."
            )
        if stage == "bokeh_shape" and not config.data.bokeh.shape_column:
            raise ValueError(
                "stage 'bokeh_shape' exige `data.bokeh.shape_column` no YAML — a "
                "coluna do release com a imagem do kernel de abertura (§3.3)."
            )
        return config.data.bokeh
    raise ValueError(
        f"stage inválido: {stage!r} (esperado 'deblur', 'bokeh' ou 'bokeh_shape')"
    )


def train_guidance_for(config: TrainConfig, stage: str) -> float:
    """C2 — guidance do branch principal/texto no treino, por estágio."""
    if stage == "deblur":
        return float(config.model.deblur_train_guidance)
    if stage == "bokeh":
        return float(config.model.bokeh_train_guidance)
    if stage == "bokeh_shape":
        return float(config.model.bokeh_shape_train_guidance)
    raise ValueError(f"stage inválido: {stage!r}")


def lora_rank_for(config: TrainConfig, stage: str) -> int:
    if stage == "deblur":
        return int(config.model.deblur_lora_rank)
    if stage == "bokeh":
        return int(config.model.bokeh_lora_rank)
    if stage == "bokeh_shape":
        # §3.3: o LoRA base fica CONGELADO e este é o novo, treinável.
        return int(config.model.bokeh_shape_lora_rank)
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
