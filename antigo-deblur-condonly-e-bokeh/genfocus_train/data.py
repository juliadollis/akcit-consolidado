"""Dataset e dataloader da DeblurNet, baseados em Hugging Face `datasets`.

Adaptado do dataloader do colega (`branch-hf-dataloader/src/data.py`) com duas
mudanças necessárias para casar com o backbone FLUX do código de treino:

  1. Normalização para [-1, 1] (e não [0, 1]).
     O `FluxBackbone.encode_image_to_tokens` (backbone.py) espera imagens em
     [-1, 1] — é o range que o VAE do FLUX foi treinado para receber. O loader
     original entregava [0, 1] porque o backbone *mock* dele fazia `*2-1` por
     dentro; aqui o backbone real NÃO faz isso, então normalizamos aqui.

  2. Pré-processamento fiel ao paper (sem distorcer aspect ratio).
     O loader original fazia `resize((S, S))` quadrado, distorcendo a geometria.
     Aqui: resize do lado-menor para `image_size` + crop `S×S` alinhado de forma
     idêntica entre blurry e AIF (random no treino, central na validação),
     com flip horizontal sincronizado opcional no treino.

Contrato de saída (DeblurNet / Stage 1):
    {
      "id": str,
      "file_name_base": str,
      "blurry_image": tensor (3, S, S) float32 em [-1, 1],   # image_blur
      "aif_image":    tensor (3, S, S) float32 em [-1, 1],   # image_focus
    }
"""

from __future__ import annotations

import json
import os
import struct
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import torch
from datasets import Dataset as HFDataset
from datasets import concatenate_datasets, load_dataset
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from .config import DatasetSourceConfig, StageConfig
from .env import get_required_env


@dataclass(frozen=True)
class DatasetRuntimeConfig:
    image_size: int = 512
    train: bool = True   # True → random crop + hflip; False → center crop determinístico
    # Origem do mapa de defocus do BokehNet. Ver DEFOCUS_SOURCES abaixo.
    defocus_source: str = "recompute"
    max_coc: float = 100.0
    # Limiar de SSIM da calibração do K (paper §3.2(c)). None = sem filtro.
    # Ver _filter_by_calibration_ssim e StageConfig.min_calibration_ssim.
    min_calibration_ssim: float | None = None
    # Repo HF com a tabela de K corrigido (Eq. 3). Usado por defocus_source="kfix".
    kfix_repo: str | None = None
    # ── Condicionamento geométrico (geo_cond) ───────────────────────────────
    # Desligado por default: com geo_condition=False o dataloader se comporta
    # EXATAMENTE como antes, sem custo e sem chave nova no batch.
    geo_condition: bool = False
    # Repo HF (ou caminho local .jsonl) com os escalares por `stem` que os canais
    # métricos exigem: z_min_m, z_max_m_bruto, z_focus_m, focallength_px,
    # largura_px, altura_px. Gerado por geo_cond/jobs/f0b_escala_metrica.py.
    geo_escalares: str | None = None
    # Constantes FIXAS de normalização, as 8 de geo_cond/constants.py.
    geo_constantes: dict | None = None
    # "inverse" (u=1/Z, default por física) ou "depth" (Z). Ver geo_cond/README.
    geo_field: str = "inverse"
    # Amostra sem entrada na tabela de escalares: "erro" aborta, "pula" descarta.
    # NUNCA inventa escalares — um default silencioso aqui é o defeito que a
    # auditoria encontrou no `k=50` da rota b.
    geo_sem_escalares: str = "erro"
    # Controle: G vira ruído de mesma estatística. Ver StageConfig.
    geo_ruido_controle: bool = False


StageDatasetType = Literal["deblur", "bokeh"]

# Mapeamento confirmado pelos autores do dataset (akcit-pixel/*):
#   image_blur       = entrada DESFOCADA          -> blurry_image
#   image_focus      = ground-truth all-in-focus  -> aif_image
#   image_pre_deblur = pré-foco gerado pela DRB-Net (variante opcional do paper
#                      que usa pré-deblur como entrada extra) -> NÃO usado aqui.
DEBLUR_REQUIRED_COLUMNS = frozenset({"image_blur", "image_focus", "file_name_base"})

# BokehNet (Stage 2). Colunas dos repos AKCITPixel3/* (rotas a/b/c).
#   aif         = all-in-focus (entrada)   -> aif_image   [-1,1]
#   bokeh       = alvo (bokeh sintetizado) -> bokeh_image [-1,1]
#   depth, k, s1 = insumos do mapa de defocus (ver DEFOCUS_SOURCES)
#   defocus_map = mapa DERIVADO, pré-computado pelo pipeline de dados
#
# ATENÇÃO — por que NÃO usamos a coluna `defocus_map` por default:
# ela foi salva normalizada POR IMAGEM (`dm / dm.max()`, ver bokehnet_common.py:87
# do repo bokehnet-data-pipeline, dentro do bloco `save_visualizations`). Medido
# empiricamente nas rotas a e b: `max(defocus_map)/65535 == 1.00000` em TODAS as
# amostras, com k variando de 33 a 195, e a coluna bate com `|D-s1|/max|D-s1|`
# a menos de 2e-5 (a quantização do uint16). Ou seja, o `k` (bokeh level) NÃO
# influencia o mapa: a intensidade do blur foi apagada, sobrou só o formato.
# Treinar assim ensina o modelo a IGNORAR o K (mesma entrada, alvos com blur
# diferente → aprende a média) e deixa a inferência oficial, que usa
# `clip(K*|D-D_foco|/MAX_COC, 0, 1)`, fora da distribuição de treino.
#
# NOTA: correlação com `|k(D-s1)|` NÃO detecta esse defeito (correlação é
# invariante a escala). O teste que discrimina é comparar o `max` ABSOLUTO
# entre amostras de k diferente.
DEFOCUS_SOURCES = ("recompute", "column", "kfix")
#   "recompute" (default): monta o mapa a partir de depth+k+s1 com a MESMA
#       fórmula da inferência oficial (Inference_bokehNet.py:138-140):
#           clip(k * |D - s1| / max_coc, 0, 1)
#       Não altera nada nos dfs — `depth`, `k` e `s1` já vêm como colunas
#       próprias, intactas; só ignoramos a coluna derivada quebrada.
#   "column": usa a coluna `defocus_map` como está. Só para reproduzir/comparar
#       o comportamento antigo; NÃO produz um modelo com controle de K.
#   "kfix" (2026-08-19): como "recompute", mas para a ROTA B usa o K recomposto
#       pela Eq. 3 do paper em vez do fallback 50.0, e o max_coc CALIBRADO. Os
#       valores vêm de um df HF pequeno (só escalares) gerado por
#       scripts/rotab_kfix.py, casado por `stem`. Amostras sem entrada no df de
#       correção caem no comportamento "recompute" normal.
#       Ver DECISOES_FASE2.md secao 6.
BOKEH_REQUIRED_COLUMNS = frozenset({"aif", "bokeh", "depth", "k", "s1"})
BOKEH_REQUIRED_COLUMNS_LEGACY = frozenset({"aif", "bokeh", "defocus_map"})
# Só estas colunas são mantidas ao carregar (permite concatenar rotas com schemas
# diferentes — a rota "a" não tem source_path/exif etc.).
BOKEH_KEEP_COLUMNS = ["aif", "bokeh", "defocus_map", "depth", "k", "s1", "stem"]


# =============================================================================
# Transformação de imagem (funções puras — testáveis sem rede/GPU)
# =============================================================================

def _to_pil_rgb(image_like: Any) -> Image.Image:
    if isinstance(image_like, Image.Image):
        return image_like.convert("RGB")
    return Image.fromarray(np.asarray(image_like)).convert("RGB")


def _normalize_to_unit_signed(img: Image.Image) -> torch.Tensor:
    """PIL RGB → tensor (3, H, W) float32 em [-1, 1]."""
    arr = np.asarray(img, dtype=np.float32) / 127.5 - 1.0
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


def prepare_aligned_pair(
    blurry_like: Any,
    aif_like: Any,
    image_size: int,
    train: bool,
    rng: np.random.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Resize (lado-menor → image_size) + crop S×S idêntico nas duas imagens.

    Garante alinhamento pixel-a-pixel reescalando blurry e AIF para o MESMO
    tamanho-alvo (derivado da AIF), e aplicando a MESMA box de crop e o MESMO
    flip. Retorna (blurry, aif) em [-1, 1], shape (3, S, S).
    """
    if image_size % 16 != 0:
        raise ValueError(
            f"image_size deve ser múltiplo de 16 (constraint do VAE/_pack_latents do FLUX); recebido {image_size}."
        )

    blur_pil = _to_pil_rgb(blurry_like)
    aif_pil = _to_pil_rgb(aif_like)

    w, h = aif_pil.size
    scale = image_size / min(w, h)
    new_w = max(image_size, round(w * scale))
    new_h = max(image_size, round(h * scale))

    aif_r = aif_pil.resize((new_w, new_h), Image.BICUBIC)
    blur_r = blur_pil.resize((new_w, new_h), Image.BICUBIC)  # mesmo target → alinhado

    max_x = new_w - image_size
    max_y = new_h - image_size
    if train:
        if rng is None:
            rng = np.random.default_rng()
        x = int(rng.integers(0, max_x + 1)) if max_x > 0 else 0
        y = int(rng.integers(0, max_y + 1)) if max_y > 0 else 0
    else:
        x = max_x // 2
        y = max_y // 2

    box = (x, y, x + image_size, y + image_size)
    aif_c = aif_r.crop(box)
    blur_c = blur_r.crop(box)

    if train and rng is not None and rng.random() < 0.5:
        aif_c = aif_c.transpose(Image.FLIP_LEFT_RIGHT)
        blur_c = blur_c.transpose(Image.FLIP_LEFT_RIGHT)

    return _normalize_to_unit_signed(blur_c), _normalize_to_unit_signed(aif_c)


# =============================================================================
# Transformação de bokeh (triplet AIF + bokeh + defocus_map)
# =============================================================================

DEFOCUS_U16 = 65535.0  # defocus_map é uint16; /65535 → [0,1] (o range da inferência)


def _defocus_to_float_pil(image_like: Any) -> Image.Image:
    """Coluna `defocus_map` (uint16, 1 canal) → PIL modo 'F' em [0,1].

    O df salva o mapa como I;16 (0..65535). Dividimos por 65535 para chegar no
    MESMO [0,1] que a inferência oficial injeta no VAE (No_preprocess=True).
    Usamos PIL 'F' (float) para poder reescalar com BILINEAR sem overshoot.
    """
    arr = np.asarray(image_like)
    if arr.ndim == 3:  # veio como RGB por acaso — usa 1 canal
        arr = arr[..., 0]
    arr = arr.astype(np.float32) / DEFOCUS_U16
    return Image.fromarray(arr, mode="F")


def _defocus_pil_to_3ch(img: Image.Image) -> torch.Tensor:
    """PIL 'F' em [0,1] → tensor (3, H, W) float32 em [0,1] (3 canais idênticos)."""
    arr = np.asarray(img, dtype=np.float32)
    arr = np.clip(arr, 0.0, 1.0)
    t = torch.from_numpy(arr)[None].repeat(3, 1, 1).contiguous()  # (3, H, W)
    return t


def defocus_from_depth(
    depth01: np.ndarray, k: float, s1: float, max_coc: float = 100.0
) -> np.ndarray:
    """Mapa de defocus a partir de depth normalizado, k e s1.

    Réplica exata da fórmula da inferência oficial (Inference_bokehNet.py:138-140):
        defocus_abs = |k * (D - D_foco)| ; cond = clip(defocus_abs / MAX_COC, 0, 1)

    `depth01` é o `depth` do df já em [0,1] (uint16 ÷ 65535). NÃO renormalizamos
    após o crop: `s1` está na escala da imagem INTEIRA, então renormalizar pelo
    min/max do recorte deslocaria o plano de foco.
    """
    if max_coc <= 0:
        raise ValueError(f"max_coc deve ser > 0; recebido {max_coc}.")
    defocus = np.abs(float(k) * (depth01.astype(np.float32) - float(s1)))
    return np.clip(defocus / float(max_coc), 0.0, 1.0)


def prepare_aligned_bokeh(
    aif_like: Any,
    bokeh_like: Any,
    defocus_like: Any = None,
    image_size: int = 512,
    train: bool = True,
    rng: np.random.Generator | None = None,
    *,
    depth_like: Any = None,
    k: float | None = None,
    s1: float | None = None,
    defocus_source: str = "recompute",
    max_coc: float = 100.0,
    kfix_entry: dict[str, float] | None = None,
    retornar_geometria: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
    """Resize + crop S×S IDÊNTICO nas 3 imagens (aif, bokeh, mapa de defocus).

    Geometria derivada da AIF (lado-menor → image_size), mesma box de crop e
    mesmo flip nas três. RGB (aif, bokeh) em [-1,1] via BICUBIC; o mapa em
    [0,1] via BILINEAR (evita overshoot fora de [0,1]).

    defocus_source="recompute" (default): usa `depth_like` + `k` + `s1` e aplica
    `defocus_from_depth` DEPOIS do crop (o crop é uma operação por-pixel, então
    recortar o depth e então aplicar a fórmula == aplicar e então recortar).
    defocus_source="column": usa `defocus_like` como está (comportamento antigo;
    ver a nota em DEFOCUS_SOURCES sobre por que ele não treina controle de K).

    Retorna (aif [-1,1], bokeh [-1,1], defocus [0,1]), todas (3, S, S).

    Com `retornar_geometria=True` devolve um 4o elemento: um dict com o depth
    JÁ REDIMENSIONADO e ainda NÃO recortado, mais a caixa de crop e o flip. É o
    que o condicionamento geométrico precisa para reproduzir exatamente a mesma
    transformação nos canais derivados, sem recomputar o resize nem sortear um
    crop diferente. Ver geo_cond/dataloader.py.
    """
    if image_size % 16 != 0:
        raise ValueError(f"image_size deve ser múltiplo de 16; recebido {image_size}.")
    if defocus_source not in DEFOCUS_SOURCES:
        raise ValueError(
            f"defocus_source inválido: {defocus_source!r}. Esperado um de {DEFOCUS_SOURCES}."
        )

    aif_pil = _to_pil_rgb(aif_like)
    bokeh_pil = _to_pil_rgb(bokeh_like)
    if defocus_source in ("recompute", "kfix"):
        if defocus_source == "kfix" and kfix_entry is None:
            # Sem entrada de correcao para esta amostra (ex.: rota c, que nao
            # passa pela Eq. 3): cai no comportamento normal.
            defocus_source = "recompute"
        if depth_like is None or k is None or s1 is None:
            raise ValueError(
                "defocus_source='recompute' exige depth_like, k e s1 "
                f"(recebido depth={depth_like is not None}, k={k}, s1={s1})."
            )
        # `depth` do df tem a MESMA codificação uint16/65535 do defocus_map.
        def_pil = _defocus_to_float_pil(depth_like)
    else:
        if defocus_like is None:
            raise ValueError("defocus_source='column' exige defocus_like.")
        def_pil = _defocus_to_float_pil(defocus_like)

    w, h = aif_pil.size
    scale = image_size / min(w, h)
    new_w = max(image_size, round(w * scale))
    new_h = max(image_size, round(h * scale))

    aif_r = aif_pil.resize((new_w, new_h), Image.BICUBIC)
    bokeh_r = bokeh_pil.resize((new_w, new_h), Image.BICUBIC)
    def_r = def_pil.resize((new_w, new_h), Image.BILINEAR)
    # `def_r` vira o mapa de defocus mais abaixo; o condicionamento geométrico
    # precisa do DEPTH redimensionado, que é o mesmo array só quando a origem é
    # "recompute"/"kfix". Guardamos aqui, antes de qualquer transformação.
    depth_r = def_r if defocus_source in ("recompute", "kfix") else None

    max_x = new_w - image_size
    max_y = new_h - image_size
    if train:
        if rng is None:
            rng = np.random.default_rng()
        x = int(rng.integers(0, max_x + 1)) if max_x > 0 else 0
        y = int(rng.integers(0, max_y + 1)) if max_y > 0 else 0
    else:
        x = max_x // 2
        y = max_y // 2

    box = (x, y, x + image_size, y + image_size)
    aif_c = aif_r.crop(box)
    bokeh_c = bokeh_r.crop(box)
    def_c = def_r.crop(box)

    flip_aplicado = bool(train and rng is not None and rng.random() < 0.5)
    if flip_aplicado:
        aif_c = aif_c.transpose(Image.FLIP_LEFT_RIGHT)
        bokeh_c = bokeh_c.transpose(Image.FLIP_LEFT_RIGHT)
        def_c = def_c.transpose(Image.FLIP_LEFT_RIGHT)

    if defocus_source == "kfix":
        # def_c e o DEPTH normalizado em [0,1]; a Eq. 3 + disparidade metrica
        # entram agora (ver defocus_kfix e DECISOES_FASE2.md secao 6).
        defocus_arr = defocus_kfix(np.asarray(def_c, dtype=np.float32), kfix_entry).astype(np.float32)
        def_c = Image.fromarray(defocus_arr, mode="F")
    elif defocus_source == "recompute":
        # def_c ainda é o DEPTH em [0,1]; a fórmula do paper entra agora.
        defocus_arr = defocus_from_depth(np.asarray(def_c, dtype=np.float32), k, s1, max_coc)
        def_c = Image.fromarray(defocus_arr, mode="F")

    saida = (
        _normalize_to_unit_signed(aif_c),
        _normalize_to_unit_signed(bokeh_c),
        _defocus_pil_to_3ch(def_c),
    )
    if not retornar_geometria:
        return saida
    return (*saida, {
        # depth em [0,1], redimensionado e AINDA NÃO recortado
        "depth01_redimensionado": np.asarray(depth_r, dtype=np.float32),
        "caixa": box,
        "flip": bool(flip_aplicado),
        "escala": float(scale),
    })


# =============================================================================
# Carregamento dos datasets HF
# =============================================================================

def _laplacian_variance(image_like: Any, probe_size: int = 256) -> float:
    """Variância do Laplaciano (focus measure). Maior = mais nítida.

    Reduz a imagem para `probe_size`² em cinza (rápido e ranking estável),
    aplica o Laplaciano discreto (kernel de 4 vizinhos) via numpy e retorna a
    variância. Sem dependência de cv2/scipy.
    """
    img = _to_pil_rgb(image_like).convert("L").resize((probe_size, probe_size), Image.BILINEAR)
    a = np.asarray(img, dtype=np.float32)
    lap = (
        4.0 * a[1:-1, 1:-1]
        - a[:-2, 1:-1]
        - a[2:, 1:-1]
        - a[1:-1, :-2]
        - a[1:-1, 2:]
    )
    return float(lap.var())


def _filter_cache_path(cache_id: str, column: str, k: int, n: int) -> str | None:
    """Caminho do cache dos índices do filtro. None se não der pra cachear."""
    try:
        base = os.environ.get("HF_HOME") or os.path.expanduser("~/.cache/huggingface")
        cache_dir = os.path.join(base, "genfocus_filter_cache")
        os.makedirs(cache_dir, exist_ok=True)
        safe = cache_id.replace("/", "_")
        # n (tamanho do dataset) na chave → invalida o cache se o dataset mudar.
        return os.path.join(cache_dir, f"{safe}_{column}_top{k}_n{n}.json")
    except Exception:
        return None


def _select_top_k_sharpest(
    dataset: HFDataset, k: int, column: str, cache_id: str | None = None
) -> HFDataset:
    """Mantém as `k` amostras mais nítidas de `dataset` (paper §4.1, RealBokeh).

    Mede a variância do Laplaciano da coluna `column` (default image_focus) de
    cada amostra, ordena desc. e seleciona o top-k. Se k >= len, retorna tudo.

    CACHE: a medição decodifica TODAS as imagens (~lento). Os índices escolhidos
    são salvos num JSON (chave = source+coluna+k+tamanho). Em restarts/resume, lê
    o cache e PULA a medição inteira. Se o cache falhar, recalcula (nunca quebra).
    """
    n = len(dataset)
    if k >= n:
        print(f"[filter] top_k_sharpest={k} >= dataset ({n}); usando o source inteiro.")
        return dataset
    if column not in dataset.column_names:
        raise ValueError(
            f"sharpness_column '{column}' não existe. Colunas: {sorted(dataset.column_names)}"
        )

    cache_path = _filter_cache_path(cache_id, column, k, n) if cache_id else None
    if cache_path and os.path.isfile(cache_path):
        try:
            with open(cache_path, "r", encoding="utf-8") as fh:
                idx = json.load(fh)
            if isinstance(idx, list) and len(idx) == k:
                print(f"[filter] cache HIT → {cache_path} (pulando a medição de nitidez).")
                return dataset.select(idx)
            print(f"[filter] cache inválido ({len(idx)}!={k}); recalculando.")
        except Exception as exc:
            print(f"[filter] falha lendo cache ({exc}); recalculando.")

    print(f"[filter] medindo nitidez (Laplaciano) de {n} imagens em '{column}'...")
    scores = np.empty(n, dtype=np.float64)
    # Itera linha a linha (decodifica preguiçoso, 1 imagem por vez → memória O(1)).
    for i, row in enumerate(dataset.select_columns([column])):
        scores[i] = _laplacian_variance(row[column])
        if (i + 1) % 500 == 0:
            print(f"[filter]   {i + 1}/{n}")

    top_idx = np.argsort(scores)[::-1][:k]
    top_idx_sorted = sorted(int(j) for j in top_idx)
    cutoff = float(scores[top_idx[-1]])
    print(
        f"[filter] mantendo top-{k}/{n} mais nítidas (corte Laplaciano-var >= {cutoff:.2f}); "
        f"descartando {n - k}."
    )

    # Salva o cache de forma ATÔMICA (tmp + rename) — evita corromper se 2
    # processos escreverem ao mesmo tempo na 1ª execução.
    if cache_path:
        try:
            tmp = f"{cache_path}.tmp.{os.getpid()}"
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump(top_idx_sorted, fh)
            os.replace(tmp, cache_path)
            print(f"[filter] cache salvo: {cache_path}")
        except Exception as exc:
            print(f"[filter] falha salvando cache ({exc}); seguindo sem cache.")

    return dataset.select(top_idx_sorted)


def _load_hf_split(source: DatasetSourceConfig) -> HFDataset:
    dataset = load_dataset(
        source.name, split=source.split, token=get_required_env("HF_TOKEN")
    )
    if source.top_k_sharpest is not None:
        dataset = _select_top_k_sharpest(
            dataset,
            k=int(source.top_k_sharpest),
            column=source.sharpness_column,
            cache_id=f"{source.name}_{source.split}",
        )
    return dataset


def _load_hf_sources(sources: list[DatasetSourceConfig]) -> HFDataset:
    if not sources:
        raise ValueError("Pelo menos um source HF precisa estar configurado.")
    loaded = [_load_hf_split(source) for source in sources]
    return loaded[0] if len(loaded) == 1 else concatenate_datasets(loaded)


def carregar_tabela_kfix(repo: str) -> dict[str, dict[str, float]]:
    """Carrega o df de correcao do K (Eq. 3) e indexa por `stem`.

    O df e minusculo (so escalares por amostra): k_eq3, z_focus_m, z_min_m,
    z_max_m, max_coc_calibrado. A profundidade METRICA e recuperavel da coluna
    `depth` que ja existe no dataset original, porque a relacao entre as duas e
    LINEAR (medido: correlacao 1.0000 em 50 amostras, ver Etapa A). Por isso NAO
    precisamos re-armazenar mapa nenhum.
    """
    ds = load_dataset(repo, split="train", token=get_required_env("HF_TOKEN"))
    tab = {}
    for r in ds:
        st = str(r.get("stem", ""))
        if st:
            tab[st] = {
                "k_eq3": float(r["k_eq3"]),
                "z_focus_m": float(r["z_focus_m"]),
                "z_min_m": float(r["z_min_m"]),
                "z_max_m": float(r["z_max_m"]),
                "max_coc": float(r["max_coc_calibrado"]),
            }
    print(f"[data] kfix: {len(tab)} amostras com K corrigido pela Eq. 3 ({repo})")
    return tab


def carregar_escalares_geo(origem: str) -> dict[str, dict[str, float]]:
    """Carrega os escalares por `stem` que o condicionamento geométrico exige.

    `origem` é um caminho local .jsonl (saída de f0b_escala_metrica.py) ou um
    repo HF com o mesmo conteúdo. Campos por amostra:
        z_min_m, z_max_m_bruto, z_focus_m, focallength_px, largura_px, altura_px

    IMPORTANTE, e verificado: a reconstrução usa `z_max_m_bruto`, NÃO o
    `z_max_m` percentilado. O job F0c ajustou `z ~ a*depth01 + b` contra uma
    execução nova do Depth Pro em 40 amostras da rota c e mediu R^2 = 1,00000
    com erro relativo de 0,04% em `a`: a coluna `depth` armazenada foi
    normalizada com o max BRUTO, céu incluído. Usar o percentil aqui
    reconstruiria uma profundidade errada em toda amostra, e o erro seria
    invisível no mapa de defocus, que reescala tudo por `max_coc`.
    """
    obrigatorios = ("z_min_m", "z_max_m_bruto", "z_focus_m",
                    "focallength_px", "largura_px", "altura_px")
    linhas: list[str] = []
    if os.path.exists(origem):
        with open(origem) as f:
            linhas = f.readlines()
    else:
        ds = load_dataset(origem, split="train", token=get_required_env("HF_TOKEN"))
        tab = {}
        for r in ds:
            st = str(r.get("stem", ""))
            if st and all(r.get(c) is not None for c in obrigatorios):
                tab[st] = {c: float(r[c]) for c in obrigatorios}
        print(f"[data] geo: {len(tab)} amostras com escalares ({origem})")
        return tab
    tab = {}
    for ln in linhas:
        ln = ln.strip()
        if not ln:
            continue
        try:
            r = json.loads(ln)
        except json.JSONDecodeError:
            continue
        st = str(r.get("stem", ""))
        if st and all(r.get(c) is not None for c in obrigatorios):
            tab[st] = {c: float(r[c]) for c in obrigatorios}
    if not tab:
        raise ValueError(f"nenhum escalar geométrico utilizável em {origem!r}")
    print(f"[data] geo: {len(tab)} amostras com escalares ({origem})")
    return tab


def defocus_kfix(depth_norm: np.ndarray, e: dict[str, float]) -> np.ndarray:
    """Mapa de defocus com o K da Eq. 3, em disparidade metrica.

        z_m  = z_min + depth_norm * (z_max - z_min)      (relacao LINEAR, Etapa A)
        CoC  = K * |1/z_mm - 1/z_foco_mm|                (px)
        mapa = clip(CoC / max_coc_calibrado, 0, 1)

    max_coc vem CALIBRADO (nao e o 100 herdado): o CoC real destas fotos e de
    1-3 px, entao /100 deixaria o mapa em [0,0.03], fora da distribuicao de
    treino. Ver DECISOES_FASE2.md.
    """
    z_m = e["z_min_m"] + depth_norm * (e["z_max_m"] - e["z_min_m"])
    z_mm = np.clip(z_m, 1e-4, None) * 1000.0
    coc = e["k_eq3"] * np.abs(1.0 / z_mm - 1.0 / (e["z_focus_m"] * 1000.0))
    return np.clip(coc / max(e["max_coc"], 1e-6), 0.0, 1.0)


def _filter_by_calibration_ssim(
    ds: HFDataset, nome: str, limiar: float
) -> HFDataset:
    """Descarta amostras cuja calibração do K não atingiu o limiar de SSIM.

    Isto É o procedimento do paper, §3.2(c), citação literal:
        "The selected K* is then used as the pseudo-bokeh-level label for
         training, provided that its corresponding SSIM exceeds a predefined
         threshold to ensure reliable supervision."

    O K da rota c vem do sweep da Eq. 5 (argmax SSIM entre o render e o bokeh
    real). Quando esse máximo é baixo, nenhum K explicou o alvo, e o valor
    gravado é ruído — inclusive os `k=0` que a auditoria de 2026-08-13 pegou
    (mapa de defocus identicamente zero com alvo visivelmente borrado).

    Fontes NÃO calibradas por sweep passam intactas: a rota b deriva o K da EXIF
    pela Eq. 3, e no paper o limiar aparece só no item (c). ATENÇÃO: a rota b TEM
    a coluna `calibration_ssim`, só que NULA em todas as linhas — checar apenas a
    existência da coluna descartaria as 11.635 amostras dela (foi o que o smoke
    da fase 2 pegou em 2026-08-13). O teste correto é se a coluna está PREENCHIDA.

    Também descartamos `k <= 0`. Isto NÃO é um critério extra inventado: a Eq. 5
    do paper busca `argmax` sobre o intervalo ABERTO `(K_min, K_max)`, então um K
    gravado exatamente no limite inferior não é saída válida do sweep, e sim
    sentinela de calibração que não rodou. Medido em 2026-08-13 na rota c: 51
    amostras (1,7%) com k=0, e o SSIM delas é ALTO (mediana 0,76), ou seja, o
    limiar de SSIM sozinho NÃO as pega — são dois defeitos ortogonais.
    """
    if "calibration_ssim" not in ds.column_names:
        print(f"[data] {nome}: sem coluna calibration_ssim (K não vem de sweep) — sem filtro.")
        return ds

    amostra = ds["calibration_ssim"][: min(len(ds), 2000)]
    preenchidos = sum(1 for v in amostra if v is not None and float(v) > 0.0)
    if preenchidos == 0:
        print(
            f"[data] {nome}: coluna calibration_ssim existe mas está VAZIA "
            "(K não vem do sweep de SSIM; provavelmente da EXIF pela Eq. 3) — "
            "fonte não filtrada, como no paper, onde o limiar é só do item (c)."
        )
        return ds

    antes = len(ds)
    ds = ds.filter(
        lambda ssim, k: (
            ssim is not None and float(ssim) >= limiar and float(k or 0.0) > 0.0
        ),
        input_columns=["calibration_ssim", "k"],
    )
    depois = len(ds)
    cortadas = antes - depois
    pct = (100.0 * cortadas / antes) if antes else 0.0
    print(
        f"[data] {nome}: filtro do paper (SSIM >= {limiar:g} E k > 0) descartou "
        f"{cortadas}/{antes} amostras ({pct:.1f}%) com calibração de K não confiável."
    )
    if depois == 0:
        raise ValueError(
            f"O filtro min_calibration_ssim={limiar} zerou o dataset '{nome}'. "
            "Limiar alto demais para esta rota."
        )
    return ds


def _load_hf_sources_bokeh(
    sources: list[DatasetSourceConfig],
    defocus_source: str = "recompute",
    min_calibration_ssim: float | None = None,
) -> HFDataset:
    """Carrega os dfs de bokeh, mantendo só as colunas comuns antes de concatenar.

    As rotas a/b/c têm schemas diferentes (a não tem source_path/exif/mask...),
    então `concatenate_datasets` quebra sem alinhar colunas. Selecionamos o
    subconjunto de BOKEH_KEEP_COLUMNS presente em TODAS as rotas e concatenamos.

    As colunas exigidas dependem de `defocus_source`: "recompute" precisa de
    depth+k+s1; "column" precisa da coluna derivada `defocus_map`.
    """
    if not sources:
        raise ValueError("Pelo menos um source HF de bokeh precisa estar configurado.")
    required = (
        BOKEH_REQUIRED_COLUMNS if defocus_source == "recompute"
        else BOKEH_REQUIRED_COLUMNS_LEGACY
    )
    loaded = []
    for source in sources:
        ds = load_dataset(
            source.name, split=source.split, token=get_required_env("HF_TOKEN")
        )
        missing = sorted(required - set(ds.column_names))
        if missing:
            raise ValueError(
                f"Dataset de bokeh '{source.name}' não tem as colunas {missing} "
                f"exigidas por defocus_source='{defocus_source}'. "
                f"Presentes: {sorted(ds.column_names)}"
            )
        # Filtro do paper §3.2(c) ANTES do select_columns: `calibration_ssim`
        # não está em BOKEH_KEEP_COLUMNS e seria descartada logo abaixo.
        if min_calibration_ssim is not None:
            ds = _filter_by_calibration_ssim(ds, source.name, min_calibration_ssim)
        loaded.append(ds)

    # Interseção entre TODAS as rotas: concatenate_datasets exige schema idêntico.
    common = set(loaded[0].column_names)
    for ds in loaded[1:]:
        common &= set(ds.column_names)
    keep = [c for c in BOKEH_KEEP_COLUMNS if c in common]
    loaded = [ds.select_columns(keep) for ds in loaded]
    return loaded[0] if len(loaded) == 1 else concatenate_datasets(loaded)


def _validate_columns(dataset: HFDataset, required: frozenset[str]) -> None:
    missing = sorted(required - set(dataset.column_names))
    if missing:
        raise ValueError(
            f"Dataset HF não tem as colunas obrigatórias: {', '.join(missing)}. "
            f"Colunas presentes: {sorted(dataset.column_names)}"
        )


# Erros que um arquivo de imagem corrompido/truncado no disco produz. PIL
# levanta OSError("image file is truncated") no .load()/.convert(), SyntaxError
# quando o cabeçalho do PNG não bate, e struct.error vaza de alguns caminhos do
# PngImagePlugin. `UnidentifiedImageError` já é subclasse de OSError.
#
# ValueError está FORA de propósito: é o que `_validate_image_tensor` usa para
# erro de shape/dtype, que é bug de código e deve continuar quebrando o treino
# em vez de virar um skip silencioso.
_CORRUPT_IMAGE_ERRORS = (OSError, SyntaxError, struct.error)


def _validate_image_tensor(name: str, tensor: torch.Tensor, image_size: int) -> None:
    if tuple(tensor.shape) != (3, image_size, image_size):
        raise ValueError(f"{name} deve ser (3,{image_size},{image_size}), got {tuple(tensor.shape)}")
    if tensor.dtype != torch.float32:
        raise ValueError(f"{name} deve ser float32, got {tensor.dtype}")


# =============================================================================
# Dataset PyTorch
# =============================================================================

class HuggingFaceDeblurDataset(Dataset[dict[str, Any]]):
    """Par (blurry, AIF) para a DeblurNet, vindo dos repos HF `akcit-pixel/*`."""

    def __init__(
        self,
        sources: list[DatasetSourceConfig],
        runtime: DatasetRuntimeConfig,
        max_samples: int | None = None,
    ) -> None:
        self.runtime = runtime
        self.dataset = _load_hf_sources(sources)
        _validate_columns(self.dataset, DEBLUR_REQUIRED_COLUMNS)
        if max_samples is not None:
            n = min(int(max_samples), len(self.dataset))
            self.dataset = self.dataset.select(range(n))

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.dataset[index]
        blurry, aif = prepare_aligned_pair(
            record["image_blur"],
            record["image_focus"],
            image_size=self.runtime.image_size,
            train=self.runtime.train,
        )
        _validate_image_tensor("blurry_image", blurry, self.runtime.image_size)
        _validate_image_tensor("aif_image", aif, self.runtime.image_size)
        return {
            "id": str(record["file_name_base"]),
            "file_name_base": str(record["file_name_base"]),
            "blurry_image": blurry,
            "aif_image": aif,
        }


class _GeoMixin:
    """Condicionamento geométrico, compartilhado pelos dois datasets de bokeh.

    Fica separado porque o `HuggingFaceBokehDataset` e o `LocalBokehFolderDataset`
    precisam do MESMO comportamento, e duplicar seria a forma clássica de os dois
    divergirem em silêncio (foi o que aconteceu com `DatasetRuntimeConfig`, que é
    construído em dois lugares no trainer).

    A conta pesada mora em `geo_cond/`, que é testado sem rede e sem GPU.
    """

    runtime: "DatasetRuntimeConfig"

    def _filtrar_sem_escalares(self) -> None:
        """Descarta as amostras sem escalares geométricos, ANTES do treino.

        Mesmo padrão do `_filter_by_calibration_ssim`: filtrar na carga, contar e
        avisar. A alternativa (pular no worker) levantaria dentro do DataLoader e
        derrubaria o rank inteiro, que foi como isto apareceu na primeira
        tentativa de subir o A'.

        Medido: 11 das 11.635 amostras da rota b (0,076%) não têm `z_focus_m`,
        porque a `foreground_mask` delas é vazia ou tem menos de 64 px. A Eq. 4
        não é definida sem máscara, então elas não têm plano de foco, e o certo é
        não treinar nelas em vez de inventar um.
        """
        if not self.runtime.geo_condition or not hasattr(self, "dataset"):
            return
        tab = self._geo_escalares
        n0 = len(self.dataset)
        stems = self.dataset["stem"] if "stem" in self.dataset.column_names else None
        if stems is None:
            return
        manter = [i for i, s in enumerate(stems) if str(s) in tab]
        if len(manter) == n0:
            return
        if not manter:
            raise ValueError(
                "o filtro de escalares geométricos zerou o dataset: nenhum `stem` "
                f"do treino aparece em {self.runtime.geo_escalares!r}. "
                "Confira se a tabela cobre as rotas configuradas."
            )
        self.dataset = self.dataset.select(manter)
        print(f"[data] geo: {n0 - len(manter)}/{n0} amostras sem escalares "
              f"({100*(n0-len(manter))/n0:.3f}%) descartadas — sem `z_focus_m`, "
              "a Eq. 4 não é definida e o plano de foco seria inventado.")

    def _geo_init(self) -> None:
        self._geo_consts = None
        self._geo_escalares: dict[str, dict[str, float]] = {}
        if not self.runtime.geo_condition:
            return
        from geo_cond.constants import GeoConstants

        if not self.runtime.geo_escalares:
            raise ValueError(
                "geo_condition=True exige geo_escalares (tabela por `stem` com "
                "z_min_m, z_max_m_bruto, z_focus_m, focallength_px, largura_px, "
                "altura_px). Gere com geo_cond/jobs/f0b_escala_metrica.py."
            )
        if not self.runtime.geo_constantes:
            raise ValueError(
                "geo_condition=True exige geo_constantes, as 8 de "
                "geo_cond/constants.py. Elas são MEDIDAS no conjunto de treino, "
                "não têm default: um número plausível silencioso reintroduz a "
                "classe de defeito que a auditoria encontrou."
            )
        self._geo_consts = GeoConstants(**self.runtime.geo_constantes)
        self._geo_escalares = carregar_escalares_geo(self.runtime.geo_escalares)
        self._filtrar_sem_escalares()

    def _geo_item(self, stem: str, geometria: dict) -> dict[str, Any]:
        from geo_cond.dataloader import GeoAmostra, stack_para_amostra

        e = self._geo_escalares.get(stem)
        if e is None:
            if self.runtime.geo_sem_escalares == "pula":
                raise IndexError(f"sem escalares geométricos para {stem!r}")
            raise KeyError(
                f"sem escalares geométricos para {stem!r}. Rode o F0b na rota "
                f"dela, ou use geo_sem_escalares='pula'. NUNCA invente escalares."
            )
        canais, segunda_ok, _ = stack_para_amostra(
            geometria["depth01_redimensionado"],
            geometria["caixa"],
            geometria["flip"],
            GeoAmostra(
                z_min_m=e["z_min_m"],
                # BRUTO, não o percentil: ver carregar_escalares_geo
                z_max_m=e["z_max_m_bruto"],
                focallength_px=e["focallength_px"],
                largura_px=int(e["largura_px"]),
                altura_px=int(e["altura_px"]),
            ),
            self._geo_consts,
            field=self.runtime.geo_field,
        )
        t = torch.from_numpy(canais)
        if self.runtime.geo_ruido_controle:
            # Ruído com a MESMA média e o MESMO desvio por canal desta amostra,
            # recortado a [0,1] como os canais reais. Preserva a estatística de
            # primeira ordem e destrói a estrutura espacial, que é exatamente a
            # separação que o controle precisa fazer.
            m = t.mean(dim=(1, 2), keepdim=True)
            s = t.std(dim=(1, 2), keepdim=True)
            t = (torch.randn_like(t) * s + m).clamp_(0.0, 1.0)
        return {
            "geo_map": t,
            "geo_segunda_ordem_valida": bool(segunda_ok),
        }


class HuggingFaceBokehDataset(_GeoMixin, Dataset[dict[str, Any]]):
    """Tripla (AIF, bokeh, defocus_map) para a BokehNet, dos repos AKCITPixel3/*."""

    def __init__(
        self,
        sources: list[DatasetSourceConfig],
        runtime: DatasetRuntimeConfig,
        max_samples: int | None = None,
    ) -> None:
        self.runtime = runtime
        self.dataset = _load_hf_sources_bokeh(
            sources, runtime.defocus_source, runtime.min_calibration_ssim
        )
        self._kfix = (
            carregar_tabela_kfix(runtime.kfix_repo)
            if runtime.defocus_source == "kfix" and runtime.kfix_repo else {}
        )
        if max_samples is not None:
            n = min(int(max_samples), len(self.dataset))
            self.dataset = self.dataset.select(range(n))
        self._geo_init()

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.dataset[index]
        recompute = self.runtime.defocus_source in ("recompute", "kfix")
        stem = str(record.get("stem", index))
        entrada_kfix = self._kfix.get(stem) if self._kfix else None
        geo_ligado = bool(self.runtime.geo_condition)
        saida = prepare_aligned_bokeh(
            record["aif"],
            record["bokeh"],
            None if recompute else record["defocus_map"],
            image_size=self.runtime.image_size,
            train=self.runtime.train,
            depth_like=record["depth"] if recompute else None,
            k=float(record["k"]) if recompute else None,
            s1=float(record["s1"]) if recompute else None,
            defocus_source=self.runtime.defocus_source,
            max_coc=self.runtime.max_coc,
            kfix_entry=entrada_kfix,
            retornar_geometria=geo_ligado,
        )
        aif, bokeh, defocus = saida[0], saida[1], saida[2]
        _validate_image_tensor("aif_image", aif, self.runtime.image_size)
        _validate_image_tensor("bokeh_image", bokeh, self.runtime.image_size)
        _validate_image_tensor("defocus_map", defocus, self.runtime.image_size)
        item = {
            "id": stem,
            "file_name_base": stem,
            "aif_image": aif,
            "bokeh_image": bokeh,
            "defocus_map": defocus,
        }
        if geo_ligado:
            item.update(self._geo_item(stem, saida[3]))
        return item


class LocalBokehFolderDataset(_GeoMixin, Dataset[dict[str, Any]]):
    """Tripla (AIF, bokeh, depth->defocus) de uma pasta local gerada pelo
    scripts/prefetch_bokeh_local.py (streaming + resize lado-menor 512).

    Layout esperado: <root>/{aif,bokeh,depth}/NNNNNNN_stem.png + metadata.jsonl.
    As imagens JÁ estão no lado-menor `image_size` (o prefetch aplica a MESMA
    fórmula de resize de prepare_aligned_bokeh), então o resize do preparo vira
    no-op e o resultado do treino é idêntico ao do dataset remoto.

    Só suporta defocus_source="recompute": o prefetch não salva a coluna
    `defocus_map` (que está quebrada nos dfs — normalizada por imagem).
    """

    # Quantas amostras seguintes sondar quando o arquivo do índice pedido está
    # corrompido, antes de desistir e deixar a exceção subir. Ver __getitem__.
    _MAX_SKIP = 8

    def __init__(
        self,
        root: str,
        runtime: DatasetRuntimeConfig,
        max_samples: int | None = None,
    ) -> None:
        if runtime.defocus_source != "recompute":
            raise ValueError(
                "LocalBokehFolderDataset só suporta defocus_source='recompute' "
                "(o prefetch não salva a coluna defocus_map; ver prefetch_bokeh_local.py)."
            )
        self.runtime = runtime
        self.root = root
        meta_path = os.path.join(root, "metadata.jsonl")
        if not os.path.isfile(meta_path):
            raise FileNotFoundError(
                f"Dataset local de bokeh não encontrado: {meta_path}. "
                "Rode scripts/prefetch_bokeh_local.py primeiro (ver docstring)."
            )
        # Arquivos que já falharam ao carregar (por worker) — só para não repetir
        # o mesmo aviso a cada época. Ver __getitem__.
        self._corrupt_files: set[str] = set()
        # DEDUP por `idx`: o metadata.jsonl é append-only, então rodar o prefetch
        # duas vezes sobre a mesma pasta (ex.: processo do login node + job do
        # SLURM em paralelo) repete linhas. Mantemos a ÚLTIMA ocorrência de cada
        # idx, senão as linhas repetidas dariam peso amostral desigual.
        #
        # ATENÇÃO: um comentário anterior aqui dizia que as imagens da corrida
        # eram "idênticas (mesmo idx → mesmo nome de arquivo, reescrito por
        # cima)". ISSO ESTAVA ERRADO e custou o job 29267: dois processos
        # escrevendo o MESMO arquivo ao mesmo tempo deixam bytes truncados. A
        # dedup conserta a contagem, não os bytes no disco. Ver __getitem__.
        by_idx: dict[Any, dict[str, Any]] = {}
        n_lines = 0
        with open(meta_path, "r", encoding="utf-8") as fh:
            for line in fh:
                rec = json.loads(line)
                if not rec.get("file"):  # pula linhas de erro do prefetch
                    continue
                n_lines += 1
                by_idx[rec.get("idx", len(by_idx))] = rec
        self.records = [by_idx[k] for k in sorted(by_idx)]
        if not self.records:
            raise ValueError(f"metadata.jsonl vazio em {root} — prefetch ainda não rodou?")
        if n_lines != len(self.records):
            print(
                f"[data] {root}: {n_lines} linhas no metadata.jsonl → "
                f"{len(self.records)} amostras únicas (dedup por idx; "
                "prefetch provavelmente rodou mais de uma vez sobre esta pasta)."
            )
        if max_samples is not None:
            self.records = self.records[: int(max_samples)]
        self._geo_init()

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Carrega a amostra, PULANDO arquivos corrompidos no disco.

        Por que existe o skip: o job 29267 (fase 1, 2026-08-06) morreu no step
        3380 porque UM png da pasta estava truncado — `OSError: image file is
        truncated` no worker do DataLoader derruba o rank inteiro e o
        `accelerate` mata o job. Num treino de 40K steps (7 dias) isso não pode
        acontecer por causa de um arquivo. A origem do arquivo truncado foi a
        corrida do prefetch (processo do login node + job SLURM escrevendo a
        MESMA pasta em paralelo); ver a nota de dedup no __init__.

        O skip é determinístico (sonda index+1, index+2, ...) para não depender
        do estado de RNG do worker, e é LIMITADO: se `_MAX_SKIP` amostras
        seguidas falharem, a exceção sobe. Assim uma falha sistemática (pasta
        sumiu, disco fora do ar) continua quebrando o treino em vez de virar um
        loop silencioso.
        """
        n = len(self.records)
        last_error: Exception | None = None
        for offset in range(self._MAX_SKIP):
            probe = (index + offset) % n
            try:
                return self._load_record(probe)
            except _CORRUPT_IMAGE_ERRORS as exc:
                last_error = exc
                name = self.records[probe].get("file", "?")
                if name not in self._corrupt_files:
                    self._corrupt_files.add(name)
                    print(
                        f"[data] ARQUIVO CORROMPIDO, pulando: {name} ({type(exc).__name__}: {exc}). "
                        f"Total de arquivos ruins vistos por este worker: {len(self._corrupt_files)}.",
                        flush=True,
                    )
        raise RuntimeError(
            f"{self._MAX_SKIP} amostras consecutivas ilegíveis a partir do índice {index} "
            f"em {self.root}. Isso não é um arquivo isolado corrompido — verifique se a pasta "
            f"do prefetch existe e está legível. Último erro: {last_error!r}"
        ) from last_error

    def _load_record(self, index: int) -> dict[str, Any]:
        rec = self.records[index]
        name = rec["file"]
        aif = Image.open(os.path.join(self.root, "aif", name))
        bokeh = Image.open(os.path.join(self.root, "bokeh", name))
        depth = Image.open(os.path.join(self.root, "depth", name))
        geo_ligado = bool(self.runtime.geo_condition)
        saida = prepare_aligned_bokeh(
            aif,
            bokeh,
            None,
            image_size=self.runtime.image_size,
            train=self.runtime.train,
            depth_like=depth,
            k=float(rec["k"]),
            s1=float(rec["s1"]),
            defocus_source="recompute",
            max_coc=self.runtime.max_coc,
            retornar_geometria=geo_ligado,
        )
        aif_t, bokeh_t, defocus = saida[0], saida[1], saida[2]
        _validate_image_tensor("aif_image", aif_t, self.runtime.image_size)
        _validate_image_tensor("bokeh_image", bokeh_t, self.runtime.image_size)
        _validate_image_tensor("defocus_map", defocus, self.runtime.image_size)
        stem = f"{rec.get('idx', index)}_{rec.get('stem', '')}"
        item = {
            "id": stem,
            "file_name_base": stem,
            "aif_image": aif_t,
            "bokeh_image": bokeh_t,
            "defocus_map": defocus,
        }
        if geo_ligado:
            # a pasta local guarda o `stem` original em rec["stem"]; é ele que
            # casa com a tabela de escalares, não o id composto acima.
            item.update(self._geo_item(str(rec.get("stem", "")), saida[3]))
        return item


def _is_local_source(name: str) -> bool:
    """Path absoluto = dataset local (pasta do prefetch); resto = repo HF."""
    return os.path.isabs(name)


def build_dataset(
    stage: StageDatasetType, stage_config: StageConfig, runtime: DatasetRuntimeConfig
) -> Dataset[dict[str, Any]]:
    if stage == "deblur":
        return HuggingFaceDeblurDataset(
            stage_config.datasets, runtime, max_samples=stage_config.max_samples
        )
    if stage == "bokeh":
        local = [s for s in stage_config.datasets if _is_local_source(s.name)]
        hub = [s for s in stage_config.datasets if not _is_local_source(s.name)]
        parts: list[Dataset[dict[str, Any]]] = []
        for src in local:
            parts.append(
                LocalBokehFolderDataset(src.name, runtime, max_samples=stage_config.max_samples)
            )
        if hub:
            parts.append(
                HuggingFaceBokehDataset(hub, runtime, max_samples=stage_config.max_samples)
            )
        if len(parts) == 1:
            return parts[0]
        # NOTA: com múltiplas fontes, max_samples se aplica POR fonte (documentado).
        from torch.utils.data import ConcatDataset

        return ConcatDataset(parts)  # type: ignore[return-value]
    raise NotImplementedError(f"Stage '{stage}' não suportado.")


# =============================================================================
# Collate + DataLoader
# =============================================================================

def collate_strict(batch: list[dict[str, Any]]) -> dict[str, Any]:
    if not batch:
        raise ValueError("Não é possível fazer collate de um batch vazio.")
    keys = list(batch[0].keys())
    keyset = set(keys)
    for sample in batch[1:]:
        if set(sample.keys()) != keyset:
            raise ValueError("Batch com samples heterogêneos; as chaves não batem.")

    output: dict[str, Any] = {}
    for key in keys:
        values = [sample[key] for sample in batch]
        if isinstance(values[0], torch.Tensor):
            output[key] = torch.stack(values, dim=0)
        else:
            output[key] = values
    return output


def build_dataloader(
    dataset: Dataset[dict[str, Any]],
    batch_size: int,
    num_workers: int,
    shuffle: bool,
    pin_memory: bool,
) -> DataLoader[dict[str, Any]]:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_strict,
        drop_last=True,
        persistent_workers=num_workers > 0,
    )
