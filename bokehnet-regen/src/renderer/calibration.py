"""Eq. 5 — calibração do bokeh level por SSIM, para a rota C.

    K* = argmax_K  SSIM( R(I_aif, D; D_focus, K),  I_real )

Duas mudanças em relação ao pipeline antigo, ambas necessárias:

1. **Custo.** O grid de 24 pontos grossos + 16 finos, cada um subindo um
   `subprocess demo.py` com CUDA e dois checkpoints, dava ~40 processos por amostra —
   semanas de GPU para 20-30K amostras, com QOS de 2 jobs. Aqui: grid grosso curto
   para localizar o máximo, depois **seção áurea** dentro do bracket, que reaproveita
   uma avaliação por iteração. Tipicamente 14 a 18 avaliações, todas in-process.

2. **Censura.** `k == 300` exato em 1.379 de 2.932 amostras da rota C não significa
   "K físico é 300": significa "o ótimo ainda crescia na borda da busca". O teto é
   expandido enquanto o máximo estiver na borda, e o que sobrar encostado sai marcado
   `is_censored` — gravado, fora da loss de controle.

O paper não publica `K_min`, `K_max` nem o limiar de SSIM. Os defaults aqui são `[A]`,
a calibrar pelo piloto e congelar no release.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from control.contract import SampleRejected
from qc.metrics import ssim

#: Faixa inicial de busca, na convenção oficial (disparidade em 1/m). [A] não medido.
#: Âncoras: rota B mediana ~16,6; rota C esperada entre 3,6 e 36; default oficial 15.
#: A faixa é generosa de propósito — quem decide o teto real é a expansão adaptativa.
K_MIN_DEFAULT = 0.5
K_MAX_DEFAULT = 120.0
#: Parada dura da expansão. Encostar aqui é censura, não medida.
K_ABSOLUTE_MAX_DEFAULT = 960.0

#: Razão áurea invertida — cada iteração reaproveita uma avaliação.
_INV_PHI = (np.sqrt(5.0) - 1.0) / 2.0

RenderFn = Callable[[np.ndarray, np.ndarray, float, float], np.ndarray]


@dataclass
class KCalibration:
    """Resultado da Eq. 5, com tudo que é preciso para auditar depois."""

    k_value: float
    calibration_ssim: float
    is_censored: bool
    search: dict = field(default_factory=dict)

    def to_metadata(self) -> dict:
        return {
            "k_value": float(self.k_value),
            "k_source": "eq5_ssim_sweep",
            "calibration_ssim": float(self.calibration_ssim),
            "is_k_censored": bool(self.is_censored),
            "k_search": dict(self.search),
        }


def _resize_nearest(array: np.ndarray, size_hw: tuple[int, int]) -> np.ndarray:
    """Vizinho mais próximo — **só para PROFUNDIDADE**.

    Interpolar profundidade atravessa descontinuidade e inventa plano intermediário:
    numa borda de objeto, a média entre 1 m e 20 m é 10,5 m, uma superfície fantasma
    que o renderer depois borra como se fosse real.

    NÃO use em foto. Ver `_resize_area`.
    """
    h, w = size_hw
    src_h, src_w = array.shape[:2]
    yi = (np.arange(h) * (src_h / h)).astype(np.int32).clip(0, src_h - 1)
    xi = (np.arange(w) * (src_w / w)).astype(np.int32).clip(0, src_w - 1)
    return array[yi][:, xi]


def _resize_area(image: np.ndarray, size_hw: tuple[int, int]) -> np.ndarray:
    """Média por bloco — **para FOTO**, incluindo o alvo do SSIM.

    Vizinho mais próximo numa foto natural produz aliasing. Se o alvo real da Eq. 5
    for reduzido assim, o SSIM passa a comparar um render suave contra um alvo
    serrilhado, e o `argmax` desloca: seria um viés sistemático em **todo** K* da rota
    C. Média por bloco é o análogo do `INTER_AREA`, sem depender de cv2.
    """
    dst_h, dst_w = size_hw
    src_h, src_w = image.shape[:2]
    if (src_h, src_w) == (dst_h, dst_w):
        return image
    src = image.astype(np.float64)
    if src.ndim == 2:
        src = src[:, :, None]
        achatado = True
    else:
        achatado = False

    # Média de área VERDADEIRA: os pixels de borda entram com peso FRACIONÁRIO.
    #
    # A versão anterior usava `floor` no início e `ceil` no fim do bloco, com peso 1
    # para todo pixel tocado. Em 2000 -> 512 isso fazia **496 de 511** blocos
    # compartilharem uma linha com o vizinho, e a largura efetiva oscilava entre 4 e 5
    # linhas em vez das 3,906 corretas — 3,3% de erro absoluto médio contra a média de
    # área de referência (PIL `BOX`), medido.
    #
    # Consequência medida no `argmax` de SSIM(K): **nenhuma** — a AIF e o alvo passam
    # pela mesma redução, então o borrão extra entra dos dois lados e cancela (K* = 15,5
    # nas duas implementações, no mesmo experimento controlado). O conserto vale porque
    # está certo e é barato, não porque desbloqueia algo.
    def _pesos(origem: int, destino: int) -> np.ndarray:
        """Matriz `destino x origem` de pesos que somam 1 em cada linha."""
        bordas = np.linspace(0.0, float(origem), destino + 1)
        idx = np.arange(origem, dtype=np.float64)
        inicio = np.maximum(bordas[:-1, None], idx[None, :])
        fim = np.minimum(bordas[1:, None], idx[None, :] + 1.0)
        w = np.clip(fim - inicio, 0.0, None)
        return w / w.sum(axis=1, keepdims=True)

    out = np.einsum("ij,jkc->ikc", _pesos(src_h, dst_h), src)
    out = np.einsum("ij,kjc->kic", _pesos(src_w, dst_w), out)
    if achatado:
        out = out[:, :, 0]
    if np.issubdtype(image.dtype, np.integer):
        info = np.iinfo(image.dtype)
        out = np.clip(np.rint(out), info.min, info.max)
    return out.astype(image.dtype)


#: Nome público de `_resize_area`, para quem precisa reduzir FOTO fora deste módulo.
#: A rota C reduz a AIF e a bokeh para medir a retenção de detalhe, e tem que usar
#: exatamente a mesma redução do sweep: duas implementações de redimensionamento
#: divergem, e a divergência entraria como viés sistemático sem nada denunciar — a
#: mesma família do "cópias divergem" que produziu quatro interpretações de K.
resize_area_for_photo = _resize_area

#: Idem para PROFUNDIDADE e MÁSCARA — vizinho mais próximo, nunca interpolação.
resize_nearest = _resize_nearest


def calibrate_k(
    render_fn: RenderFn,
    *,
    aif_bgr: np.ndarray,
    target_bgr: np.ndarray,
    depth_m: np.ndarray,
    focus_disparity: float,
    k_min: float = K_MIN_DEFAULT,
    k_max: float = K_MAX_DEFAULT,
    k_absolute_max: float = K_ABSOLUTE_MAX_DEFAULT,
    coarse_points: int = 7,
    tolerance: float = 0.25,
    max_evaluations: int = 40,
    work_long_side: Optional[int] = 512,
) -> KCalibration:
    """Localiza `K*` e devolve o resultado com a marcação de censura.

    A calibração roda numa resolução de trabalho reduzida por custo, e **K é
    convertido nas duas direções**: `K_trabalho = K * escala` para renderizar,
    e o `K*` devolvido volta para a escala da imagem original. CoC em pixel escala
    com a resolução; esquecer isso é o mesmo defeito que o fator do crop de treino.
    """
    if aif_bgr.shape[:2] != target_bgr.shape[:2]:
        raise SampleRejected("resolution_invalid", f"aif {aif_bgr.shape[:2]} != alvo {target_bgr.shape[:2]}")
    if not (0 <= k_min < k_max <= k_absolute_max):
        raise SampleRejected("k_out_of_configured_range", f"faixa inválida: {k_min}, {k_max}, {k_absolute_max}")

    full_h, full_w = aif_bgr.shape[:2]
    scale = 1.0
    aif_w, target_w, depth_w = aif_bgr, target_bgr, depth_m
    if work_long_side and max(full_h, full_w) > work_long_side:
        scale = work_long_side / float(max(full_h, full_w))
        size = (max(1, round(full_h * scale)), max(1, round(full_w * scale)))
        aif_w = _resize_area(aif_bgr, size)         # foto: média por bloco
        target_w = _resize_area(target_bgr, size)   # foto: o alvo do SSIM, idem
        depth_w = _resize_nearest(depth_m, size)    # profundidade: vizinho

    cache: dict[float, float] = {}
    evaluations = 0

    def score(k_full: float) -> float:
        """SSIM para um K expresso na escala da imagem ORIGINAL."""
        nonlocal evaluations
        key = round(float(k_full), 6)
        if key in cache:
            return cache[key]
        if evaluations >= max_evaluations:
            raise SampleRejected("k_out_of_configured_range", f"orçamento de {max_evaluations} avaliações esgotado")
        rendered = render_fn(aif_w, depth_w, float(focus_disparity), float(k_full) * scale)
        value = ssim(rendered, target_w)
        cache[key] = value
        evaluations += 1
        return value

    # -- fase 1: grid grosso, expandindo o teto enquanto o máximo estiver na borda ---
    expansions = 0
    current_max = float(k_max)
    while True:
        grid = np.linspace(k_min, current_max, coarse_points)
        scores = [score(float(k)) for k in grid]
        best = int(np.argmax(scores))
        if best < len(grid) - 1 or current_max >= k_absolute_max:
            break
        current_max = min(current_max * 2.0, k_absolute_max)
        expansions += 1

    # -- fase 2: seção áurea dentro do bracket ------------------------------------
    lo = float(grid[max(0, best - 1)])
    hi = float(grid[min(len(grid) - 1, best + 1)])
    c = hi - _INV_PHI * (hi - lo)
    d = lo + _INV_PHI * (hi - lo)
    fc, fd = score(c), score(d)
    while (hi - lo) > tolerance and evaluations < max_evaluations:
        if fc > fd:
            hi, d, fd = d, c, fc
            c = hi - _INV_PHI * (hi - lo)
            fc = score(c)
        else:
            lo, c, fc = c, d, fd
            d = lo + _INV_PHI * (hi - lo)
            fd = score(d)

    k_star = min(cache, key=lambda k: -cache[k])
    best_ssim = cache[k_star]

    # -- censura -------------------------------------------------------------------
    # Censura é o ótimo ENCOSTAR na borda, não "ficar perto dela". A escala certa é a
    # resolução da seção áurea (`tolerance`), NUNCA o passo do grid grosso, que só
    # serve para bracketear e é ~80x maior.
    #
    # Medido com renderer de disco perfeito e os defaults: usar o passo do grid dava
    # limiar efetivo de k <= 10,46, e K = 3,6 · 5 · 8 · 10 saíam marcados censurados
    # com K* recuperado exatamente e SSIM 1,0000. A âncora da rota C é 3,6 a 36, então
    # a cauda inferior INTEIRA sairia da loss de controle. É o `k == 300` invertido:
    # em vez de gravar censura como medida, gravava medida como censura.
    edge = max(float(tolerance), 1e-9)
    at_upper = bool(best == len(grid) - 1) or bool(k_star >= current_max - edge)
    at_lower = bool(k_star <= k_min + edge)
    censored = bool(at_upper or at_lower)

    return KCalibration(
        k_value=float(k_star),
        calibration_ssim=float(best_ssim),
        is_censored=censored,
        search={
            "k_min": float(k_min),
            "k_max": float(current_max),
            "k_absolute_max": float(k_absolute_max),
            "expansions": int(expansions),
            "evaluations": int(evaluations),
            "coarse_points": int(coarse_points),
            "tolerance": float(tolerance),
            "work_long_side": int(work_long_side) if work_long_side else None,
            "work_scale": float(scale),
            "at_upper_bound": bool(at_upper),
            "at_lower_bound": bool(at_lower),
        },
    )
