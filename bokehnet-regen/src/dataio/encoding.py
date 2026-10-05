"""Codificação de profundidade em disco — uint16 LINEAR EM DISPARIDADE.

A escolha do espaço de quantização não é detalhe. O dataset antigo gravava
profundidade métrica normalizada min-max, o que gasta quase toda a resolução do
uint16 no fundo distante, que é onde a disparidade praticamente não varia: 24,7% das
amostras da rota B tinham a cena útil em **menos de 256 níveis** de 65535, com mediana
de 23 níveis no subgrupo de `z_max >= 1000 m`.

Quantizar em disparidade resolve **para cena natural** — conteúdo perto e céu longe —
porque é o espaço em que o CoC vive:

    disp   = 1/z
    disp01 = (disp - disp_min) / (disp_max - disp_min)
    u16    = round(disp01 * 65535)

O passo de quantização é uniforme em disparidade, então o erro de CoC também é:
`erro_coc = K * passo`. Para z em [1, 100] m e K = 50, isso dá 7,5e-4 px — quatro
ordens de grandeza abaixo do que importa.

Medido numa cena natural (90% do conteúdo em 1–50 m, 10% de céu a 10.000 m):

    profundidade métrica :   322 níveis de 65535   <- o defeito medido na rota B
    disparidade          : 3.006 níveis            <- 9,3x melhor

**A vantagem não é universal, e o limite está travado por teste.** Numa cena *linear
em z* — densa no fundo distante — a ordem se inverte. Cenas fotográficas não são
assim; a afirmação vale para as nossas e só para elas.

## Sobre a resolução em que a profundidade é gravada

Gravar em resolução cheia não cabe na cota (207,8 GB só de controle contra 115 GB de
folga). A profundidade é gravada com o lado longo limitado, e o `k_value` é gravado
**na escala de pixel da imagem ORIGINAL**, junto de `source_hw` e `depth_hw`.

Quem consome converte com `control.k_at_resolution`. É a mesma função que o dataloader
usa para o crop de treino, então não existe uma segunda interpretação de escala — que
é exatamente o defeito que o contrato existe para impedir.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from control.contract import reject

UINT16_MAX = 65535

#: Lado longo da profundidade gravada. Congelado por release: rotas diferentes
#: escolhendo valores diferentes é a mesma família do `max_coc` por amostra.
#: Orçamento em `dataio.writer.estimate_disk_budget`; 768 cabe nos 115 GB de folga.
DEPTH_LONG_SIDE = 768


@dataclass(frozen=True)
class EncodedDepth:
    """Disparidade quantizada, com os limites que a reconstroem."""

    disparity_u16: np.ndarray
    disparity_min: float          # = 1 / z_max, JÁ na resolução gravada
    disparity_max: float          # = 1 / z_min, idem
    image_hw: tuple[int, int]     # resolução da IMAGEM — a escala em que `k_value` vive
    depth_hw: tuple[int, int]     # resolução em que a profundidade foi gravada

    @property
    def z_min_m(self) -> float:
        return 1.0 / self.disparity_max

    @property
    def z_max_m(self) -> float:
        return 1.0 / self.disparity_min

    def to_metadata(self) -> dict:
        return {
            "disparity_min": float(self.disparity_min),
            "disparity_max": float(self.disparity_max),
            "z_min_m": float(self.z_min_m),
            "z_max_m": float(self.z_max_m),
            "image_h": int(self.image_hw[0]), "image_w": int(self.image_hw[1]),
            "depth_h": int(self.depth_hw[0]), "depth_w": int(self.depth_hw[1]),
            "depth_encoding": "uint16_linear_in_disparity",
        }


def resize_depth_nearest(depth_m: np.ndarray, long_side: int) -> np.ndarray:
    """Reamostra a profundidade para o lado longo pedido, por vizinho mais próximo.

    Vizinho mais próximo de propósito: interpolar profundidade **atravessa
    descontinuidade** e inventa um plano intermediário que não existe na cena. Numa
    borda de objeto, a média entre 1 m e 20 m é 10,5 m — uma superfície fantasma que
    o renderer depois borra como se fosse real.
    """
    height, width = depth_m.shape[:2]
    if max(height, width) <= long_side:
        return depth_m
    scale = long_side / float(max(height, width))
    new_h, new_w = max(1, round(height * scale)), max(1, round(width * scale))
    yi = np.minimum((np.arange(new_h) / scale).astype(np.int64), height - 1)
    xi = np.minimum((np.arange(new_w) / scale).astype(np.int64), width - 1)
    return depth_m[yi][:, xi]


def encode_depth(
    depth_m: np.ndarray, *, image_hw: tuple[int, int], long_side: int = DEPTH_LONG_SIDE,
) -> EncodedDepth:
    """Profundidade métrica -> uint16 linear em disparidade.

    `image_hw` é OBRIGATÓRIO e é o shape da **imagem**, não do array de profundidade.
    Os dois divergem: o Depth Pro pode rodar a 384x512 sobre uma foto 1500x2000. Antes
    este campo se chamava `source_hw` e guardava o shape do array recebido — nome de
    uma coisa, valor de outra —, e `k_at_resolution` calculava o fator errado em
    silêncio, que é justamente o defeito de escala que o contrato existe para impedir.
    """
    depth_m = np.asarray(depth_m, dtype=np.float32)
    if len(image_hw) != 2 or min(image_hw) <= 0:
        raise ValueError(f"image_hw inválido: {image_hw!r}")
    resized = resize_depth_nearest(depth_m, long_side)

    # Os extremos vêm da profundidade CHEIA, não da reduzida.
    #
    # O vizinho mais próximo com decimação 5x amostra 1 pixel em 27, então um objeto
    # pequeno e muito próximo simplesmente some — e com ele o extremo da disparidade.
    # Medido: cena 3024x4032 com um objeto de 3x3 px a 0,30 m dá `disp_max = 3,3333`
    # em resolução cheia e `1,0000` na reduzida. O span é o normalizador de [0,1] que
    # reconstrói o mapa E alimenta `k_for_bokehme`: usar o span reduzido dava um K de
    # renderer **3,4x** menor, uma ordem de grandeza acima do `k_effective_factor`
    # de 0,9873 que já medimos e declaramos. E o efeito depende do CONTEÚDO — some
    # com objeto grande —, que é como ele passaria despercebido.
    #
    # Com o span cheio, os valores reduzidos são um subconjunto de [d_min, d_max] e o
    # `clip` abaixo é no-op. O u16 gravado não alcança 0 nem 65535, o que é honesto:
    # o extremo existe na cena e não sobreviveu à decimação.
    full_disparity = 1.0 / depth_m.astype(np.float64)
    d_min, d_max = float(full_disparity.min()), float(full_disparity.max())
    disparity = 1.0 / resized.astype(np.float64)
    span = d_max - d_min
    if not np.isfinite(span) or span <= 0:
        # `reject`, não `ValueError`: sem slug a falha escapa do histograma de motivos,
        # que é o instrumento que calibra todo limiar e denuncia fallback novo.
        reject("depth_disparity_span_degenerate", f"[{d_min}, {d_max}]")

    normalized = (disparity - d_min) / span
    quantized = np.rint(np.clip(normalized, 0.0, 1.0) * UINT16_MAX).astype(np.uint16)
    return EncodedDepth(quantized, d_min, d_max,
                        (int(image_hw[0]), int(image_hw[1])),
                        (int(resized.shape[0]), int(resized.shape[1])))


def decode_disparity(encoded: EncodedDepth) -> np.ndarray:
    """uint16 -> disparidade em 1/m. É a grandeza que o contrato usa."""
    span = encoded.disparity_max - encoded.disparity_min
    return (encoded.disparity_min
            + encoded.disparity_u16.astype(np.float64) / UINT16_MAX * span).astype(np.float32)


def decode_depth_m(encoded: EncodedDepth) -> np.ndarray:
    """uint16 -> profundidade métrica em metros. Só para quem precisa de `z`."""
    return (1.0 / decode_disparity(encoded)).astype(np.float32)


def quantization_coc_error_px(encoded: EncodedDepth, k_value_at_depth_hw: float) -> float:
    """Erro máximo de CoC da quantização, em pixels **da resolução gravada**.

    `k_value_at_depth_hw` tem que estar na escala de `depth_hw`, não na da imagem.
    Misturar um K medido em `image_hw` com uma quantização que vive em `depth_hw`
    produz um número em "px" sem resolução — a regra 3 do contrato existe justamente
    contra isso. Converta com `control.k_at_resolution` antes de chamar.
    """
    step = (encoded.disparity_max - encoded.disparity_min) / UINT16_MAX
    return float(abs(k_value_at_depth_hw) * step * 0.5)
