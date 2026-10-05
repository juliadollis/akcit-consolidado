"""Depth Pro — o único backend de profundidade. Sem cascata, sem fallback.

O pipeline antigo tinha três `except Exception: pass` em sequência: Depth Pro →
Depth Anything → MiDaS. O Depth Anything devolve **disparidade**, que era normalizada
igual e gravada igual, deixando a amostra **espelhada** — perto virava longe. A
informação mútua do QC é cega a inversão, então nenhum gate pegava. Não há evidência
de que disparou, e também não há como saber.

Aqui existe um backend. Se ele falhar, a amostra é rejeitada com motivo e o run segue
contando. O nome do backend vai em `ControlLabel.depth_backend`, é validado contra
`DEPTH_BACKENDS` e chega ao disco — é a prova de que o D11 não aconteceu.

## Sobre a resolução

O Depth Pro roda numa resolução própria e o resultado é reamostrado para a resolução
da IMAGEM antes de sair daqui. Isso não é cosmético: `k_value` é medido na escala de
pixel da imagem, e `encode_depth` exige `image_hw` justamente para que as duas escalas
nunca sejam inferidas uma da outra. Uma auditoria mediu 2,63× de erro no K quando a
profundidade saía em 1152×1536 sobre uma foto 3024×4032.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np

from control.contract import MetricDepth, reject, validate_metric_depth

BACKEND_NAME = "depth_pro"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _resize_bilinear(array: np.ndarray, size_hw: tuple[int, int]) -> np.ndarray:
    """Bilinear em numpy, para levar a profundidade à resolução da imagem.

    Bilinear e não vizinho aqui, de propósito: este é um UPSAMPLE da saída nativa do
    Depth Pro para a resolução da foto, e a rede já entrega um campo suave. O vizinho
    mais próximo é a escolha certa no DOWNSAMPLE de armazenamento (`dataio.encoding`),
    onde o risco é atravessar descontinuidade e inventar plano intermediário.
    """
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
    top = a * (1 - wx) + b * wx
    bottom = c * (1 - wx) + d * wx
    return (top * (1 - wy) + bottom * wy).astype(array.dtype)


@dataclass
class DepthProRuntime:
    """Modelo carregado uma vez, reusado por todo o run.

    `provenance()` devolve o hash do checkpoint, que vai em toda amostra: é o que
    permite provar depois de qual peso a profundidade daquele rótulo veio.
    """

    checkpoint_path: Path
    device: str = "cuda"
    _model: Any = None
    _transform: Any = None
    _sha: Optional[str] = None
    calls: int = 0

    def __post_init__(self) -> None:
        self.checkpoint_path = Path(self.checkpoint_path).expanduser()
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(
                f"checkpoint do Depth Pro não encontrado: {self.checkpoint_path}. "
                "NÃO existe backend alternativo — a cascata que caía no Depth Anything "
                "é o defeito D11."
            )
        self._sha = _sha256(self.checkpoint_path)

    def load(self) -> "DepthProRuntime":
        """Carrega o modelo. Separado do `__init__` para o hash ser barato de obter."""
        if self._model is not None:
            return self
        import torch                                        # noqa: F401
        from depth_pro.depth_pro import (
            DEFAULT_MONODEPTH_CONFIG_DICT, create_model_and_transforms,
        )
        from dataclasses import replace as _replace

        config = _replace(DEFAULT_MONODEPTH_CONFIG_DICT,
                          checkpoint_uri=str(self.checkpoint_path))
        model, transform = create_model_and_transforms(config=config, device=self.device)
        model.eval()
        self._model, self._transform = model, transform
        return self

    def provenance(self) -> dict:
        return {
            "depth_backend": BACKEND_NAME,
            "depth_model_sha256": self._sha,
            "depth_checkpoint": str(self.checkpoint_path),
            "device": self.device,
        }

    def infer(self, image_rgb: np.ndarray) -> MetricDepth:
        """Profundidade MÉTRICA em metros, na resolução da imagem recebida.

        Rejeita em vez de consertar: profundidade não-finita, não-positiva ou com
        faixa degenerada vira `SampleRejected` com slug, que entra no histograma.
        """
        import torch
        from PIL import Image as PILImage

        if self._model is None:
            self.load()
        image_rgb = np.asarray(image_rgb)
        if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
            reject("resolution_invalid", f"esperado HxWx3, recebido {image_rgb.shape}")
        image_hw = (int(image_rgb.shape[0]), int(image_rgb.shape[1]))

        with torch.no_grad():
            tensor = self._transform(PILImage.fromarray(image_rgb.astype(np.uint8)))
            prediction = self._model.infer(tensor.unsqueeze(0).to(self.device))
            depth = prediction["depth"].squeeze().detach().cpu().numpy().astype(np.float32)

        # A profundidade sai na resolução da imagem, sempre. Deixar a resolução nativa
        # do Depth Pro vazar para o resto do pipeline foi medido como 2,63x de erro no K.
        depth = _resize_bilinear(depth, image_hw)
        self.calls += 1
        return validate_metric_depth(depth, backend=BACKEND_NAME)
