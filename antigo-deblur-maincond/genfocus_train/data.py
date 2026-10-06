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


StageDatasetType = Literal["deblur"]

# Mapeamento confirmado pelos autores do dataset (akcit-pixel/*):
#   image_blur       = entrada DESFOCADA          -> blurry_image
#   image_focus      = ground-truth all-in-focus  -> aif_image
#   image_pre_deblur = pré-foco gerado pela DRB-Net (variante opcional do paper
#                      que usa pré-deblur como entrada extra) -> NÃO usado aqui.
DEBLUR_REQUIRED_COLUMNS = frozenset({"image_blur", "image_focus", "file_name_base"})


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


def _validate_columns(dataset: HFDataset, required: frozenset[str]) -> None:
    missing = sorted(required - set(dataset.column_names))
    if missing:
        raise ValueError(
            f"Dataset HF não tem as colunas obrigatórias: {', '.join(missing)}. "
            f"Colunas presentes: {sorted(dataset.column_names)}"
        )


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


def build_dataset(
    stage: StageDatasetType, stage_config: StageConfig, runtime: DatasetRuntimeConfig
) -> Dataset[dict[str, Any]]:
    if stage != "deblur":
        raise NotImplementedError(
            f"Stage '{stage}' não suportado por este pacote (foco: DeblurNet/Stage 1). "
            "BokehNet (Stage 2) ainda não está integrado ao dataloader HF."
        )
    return HuggingFaceDeblurDataset(
        stage_config.datasets, runtime, max_samples=stage_config.max_samples
    )


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
