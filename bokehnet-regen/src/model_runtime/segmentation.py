"""BiRefNet — o único segmentador. Mesma regra do Depth Pro.

O pipeline antigo tinha `BiRefNet → RMBG → GrabCut` com `except` nu, e `mask_source`
continuava dizendo `"automatic"` depois de cair no GrabCut. Proveniência que mente é
pior que proveniência ausente: nada denunciava, e a máscara define `focus_disparity`,
que define K.

Aqui existe um segmentador. `MaskSource` é enum fechado e não tem valor `AUTOMATIC` —
se um dia houver um segundo modelo, ele ganha valor próprio, nunca reusa o do BiRefNet.

## O que esta máscara é, e o que ela não é

O paper (§3.2(b), paper.txt:344-348) usa o BiRefNet para obter uma **in-focus mask**.
Mas o BiRefNet é um segmentador de objeto **saliente**, e as duas coisas coincidem só
quando o fotógrafo focou o objeto saliente. Quando ele focou o fundo, a máscara sai no
primeiro plano e o plano de foco inteiro fica errado — e a IoU entre a máscara da AIF
e a da bokeh não pega, porque as duas concordam no objeto errado.

Quem testa a hipótese de verdade é `qc.gates.focus_mask_is_sharpest`, que mede nitidez
dentro contra fora da máscara **na imagem bokeh**. Este módulo produz a máscara; o gate
diz se ela merece confiança.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np

from control.contract import reject

BACKEND_NAME = "birefnet"

#: Resolução de entrada do BiRefNet. Vem do modelo, não é escolha nossa.
_INPUT_SIZE = (1024, 1024)


def _model_files(root: Path) -> list[Path]:
    """Os arquivos que DEFINEM o modelo, em ordem estável.

    Ignora todo caminho com componente oculto. O `snapshot_download` deixa
    `.cache/huggingface/download/*.metadata` dentro do próprio diretório do modelo, e
    esses arquivos carregam etag e horário do download — variam entre máquinas e entre
    downloads do MESMO modelo.

    Se entrassem no hash, dois snapshots byte a byte idênticos do BiRefNet produziriam
    `mask_model_sha256` diferentes, e a proveniência diria "modelo diferente" onde não
    há diferença nenhuma. É o modo de falha oposto ao que o hash existe para evitar, e
    igualmente ruim: um hash que muda sem o modelo mudar não distingue mais nada.
    """
    return sorted(p for p in root.rglob("*")
                  if p.is_file() and not any(parte.startswith(".")
                                             for parte in p.relative_to(root).parts))


def _sha256_dir(path: Path) -> str:
    """Hash estável do conteúdo de um snapshot de modelo.

    Um diretório, não um arquivo: o BiRefNet vem como snapshot do HF. Hasheia o
    **caminho relativo** e o conteúdo de cada arquivo, em ordem, para o valor não
    depender da ordem do filesystem. Caminho relativo, e não só o nome: dois arquivos
    homônimos em subdiretórios diferentes seriam indistinguíveis se trocassem de lugar.
    """
    root = Path(path)
    digest = hashlib.sha256()
    for file in _model_files(root):
        digest.update(str(file.relative_to(root)).encode("utf-8"))
        with file.open("rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                digest.update(block)
    return digest.hexdigest()


def _resize_bilinear(array: np.ndarray, size_hw: tuple[int, int]) -> np.ndarray:
    dst_h, dst_w = size_hw
    src_h, src_w = array.shape[:2]
    if (src_h, src_w) == (dst_h, dst_w):
        return array
    ys = np.linspace(0, src_h - 1, dst_h)
    xs = np.linspace(0, src_w - 1, dst_w)
    y0 = np.floor(ys).astype(np.int64); y1 = np.minimum(y0 + 1, src_h - 1)
    x0 = np.floor(xs).astype(np.int64); x1 = np.minimum(x0 + 1, src_w - 1)
    wy = (ys - y0)[:, None]; wx = (xs - x0)[None, :]
    a = array[np.ix_(y0, x0)]; b = array[np.ix_(y0, x1)]
    c = array[np.ix_(y1, x0)]; d = array[np.ix_(y1, x1)]
    return ((a * (1 - wx) + b * wx) * (1 - wy)
            + (c * (1 - wx) + d * wx) * wy).astype(array.dtype)


@dataclass
class BiRefNetRuntime:
    """Modelo carregado uma vez, reusado por todo o run."""

    model_path: Path
    device: str = "cuda"
    threshold: float = 0.5
    _model: Any = None
    _sha: Optional[str] = None
    calls: int = 0

    def __post_init__(self) -> None:
        self.model_path = Path(self.model_path).expanduser()
        if not self.model_path.is_dir():
            raise FileNotFoundError(
                f"snapshot do BiRefNet não encontrado: {self.model_path}. NÃO existe "
                "segmentador alternativo — a cascata que caía no GrabCut mantinha "
                "`mask_source='automatic'`, e a proveniência mentia."
            )
        self._sha = _sha256_dir(self.model_path)

    def load(self) -> "BiRefNetRuntime":
        if self._model is not None:
            return self
        import torch                                        # noqa: F401
        from transformers import AutoModelForImageSegmentation

        self._model = (
            AutoModelForImageSegmentation
            .from_pretrained(str(self.model_path), trust_remote_code=True)
            .to(self.device).eval()
        )
        return self

    def provenance(self) -> dict:
        return {
            "mask_backend": BACKEND_NAME,
            "mask_model_sha256": self._sha,
            "mask_model_path": str(self.model_path),
            "mask_threshold": self.threshold,
            "device": self.device,
        }

    def infer(self, image_rgb: np.ndarray) -> np.ndarray:
        """Máscara booleana na resolução da imagem recebida.

        Devolve a máscara CRUA do modelo, sem pós-processamento. Erodir, preencher
        buraco ou pegar a maior componente são decisões que mudam `focus_disparity` e
        que, se acontecessem aqui, ficariam invisíveis nos metadados. Quem quiser
        limpar a máscara faz isso explicitamente e registra.
        """
        return self.infer_probabilities(image_rgb) > self.threshold

    def infer_probabilities(self, image_rgb: np.ndarray) -> np.ndarray:
        """O mapa de probabilidade cru, antes do limiar.

        Existe separado de `infer` porque máscara vazia tem duas causas opostas — o
        modelo não achou nada, ou achou e o limiar cortou — e a booleana não distingue
        as duas. `scripts/diagnose_empty_masks.py` usa isto para decidir de quem é o
        problema quando uma cena inteira é descartada.
        """
        import torch
        from PIL import Image as PILImage
        from torchvision import transforms

        if self._model is None:
            self.load()
        image_rgb = np.asarray(image_rgb)
        if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
            reject("resolution_invalid", f"esperado HxWx3, recebido {image_rgb.shape}")
        image_hw = (int(image_rgb.shape[0]), int(image_rgb.shape[1]))

        prepare = transforms.Compose([
            transforms.Resize(_INPUT_SIZE),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        with torch.no_grad():
            tensor = prepare(PILImage.fromarray(image_rgb.astype(np.uint8)))
            logits = self._model(tensor.unsqueeze(0).to(self.device))[-1]
            probability = logits.sigmoid().squeeze().detach().cpu().numpy().astype(np.float32)

        self.calls += 1
        return _resize_bilinear(probability, image_hw)
