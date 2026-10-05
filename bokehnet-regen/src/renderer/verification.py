"""Harness de verificação do renderer — os três testes do `renderer-verifier`.

Existe porque o pipeline antigo NUNCA instalou o BokehMe: `_render_bokehme` levantava
`ImportError` incondicional e `render_bokeh` caía sempre num gaussiano de 16 camadas
com kernel travado em 51 px. Ninguém percebeu porque nada media o borrão produzido.

O harness é separado do renderer de propósito: ele mede a SAÍDA, seja qual for o
renderer. Assim dá para testar o próprio harness contra um borrão sintético de raio
conhecido, aqui, sem GPU — e rodar exatamente o mesmo código contra o BokehMe no
cluster.

Os três testes:
  1. `radial_profile` + `edge_width_ratio`  -> disco, não gaussiana
  2. `measure_blur_radius_px`               -> raio == K * |Delta_disp|
  3. decisão de `highlight`                 -> registrada no config, não medida aqui

IMPORTANTE — a `render_fn` usada aqui NÃO pode quantizar para uint8.

Um ponto de luz de 255 espalhado num disco de raio 12 px dá 0,56 por pixel, que
`astype(np.uint8)` trunca para ZERO: a imagem inteira vira preto e a medição devolve
raio 0. Quantização é preocupação de armazenamento, não de renderização, então o
adaptador do renderer deve expor a saída em float para a verificação e só converter
para uint8 na gravação. Medido: raio 5 -> 3,2 por pixel; raio 12 -> 0,56; raio 25 -> 0,13.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from qc.metrics import to_gray


# --------------------------------------------------------------------------------
# Cena sintética: um ponto de luz sobre fundo preto, num plano de profundidade
# --------------------------------------------------------------------------------

def point_light_scene(
    size: int = 257,
    depth_m: float = 10.0,
    background_value: int = 0,
) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    """Imagem com um único pixel branco no centro, e profundidade constante.

    Profundidade constante de propósito: com `z` uniforme, `CoC = K*(1/z - focus_disp)`
    é o MESMO em todo pixel, então o borrão do ponto é o próprio PSF do renderer, sem
    mistura de camadas. É a condição mais limpa para medir raio e forma.

    Devolve `(imagem_bgr_uint8, profundidade_m_float32, (y, x) do ponto)`.
    """
    if size % 2 == 0:
        raise ValueError("use tamanho ímpar para o ponto cair exatamente no centro")
    image = np.full((size, size, 3), background_value, dtype=np.uint8)
    center = (size // 2, size // 2)
    image[center[0], center[1], :] = 255
    depth = np.full((size, size), float(depth_m), dtype=np.float32)
    return image, depth, center


# --------------------------------------------------------------------------------
# Teste 1 — disco, não gaussiana
# --------------------------------------------------------------------------------

def radial_profile(image: np.ndarray, center: tuple[int, int], max_radius: int | None = None) -> np.ndarray:
    """Intensidade média por raio inteiro, a partir de `center`."""
    gray = to_gray(image)
    h, w = gray.shape
    yy, xx = np.ogrid[:h, :w]
    r = np.sqrt((yy - center[0]) ** 2 + (xx - center[1]) ** 2)
    if max_radius is None:
        max_radius = int(min(center[0], center[1], h - 1 - center[0], w - 1 - center[1]))
    bins = np.clip(r.astype(np.int32), 0, max_radius)
    total = np.bincount(bins.ravel(), weights=gray.ravel(), minlength=max_radius + 1)
    count = np.bincount(bins.ravel(), minlength=max_radius + 1)
    return (total / np.maximum(count, 1))[: max_radius + 1]


def _radius_at_fraction(profile: np.ndarray, fraction: float) -> float:
    """Raio em que o perfil cai para `fraction` do pico, com interpolação linear."""
    peak = float(profile[0])
    if peak <= 0:
        return 0.0
    target = fraction * peak
    below = np.nonzero(profile < target)[0]
    if below.size == 0:
        return float(len(profile) - 1)
    i = int(below[0])
    if i == 0:
        return 0.0
    hi, lo = float(profile[i - 1]), float(profile[i])
    if hi == lo:
        return float(i)
    return (i - 1) + (hi - target) / (hi - lo)


def edge_width_ratio(profile: np.ndarray) -> float:
    """Largura da transição 90%->10% do pico, normalizada pelo raio de meia altura.

    Discriminador entre disco e gaussiana:

      - disco ideal: o perfil é plano até o raio e cai a pique. r90 ~ r10 ~ r,
        então a razão tende a 0.
      - gaussiana `exp(-r^2/2s^2)`: r90 = 0,459s, r50 = 1,177s, r10 = 2,146s, e a
        razão vale (2,146 - 0,459)/1,177 = **1,43**, independente de s.

    Limiar recomendado: **< 0,5 é disco, > 1,0 é gaussiana**. Entre os dois, inspecionar.
    """
    r90 = _radius_at_fraction(profile, 0.9)
    r50 = _radius_at_fraction(profile, 0.5)
    r10 = _radius_at_fraction(profile, 0.1)
    if r50 <= 0:
        return float("inf")
    return (r10 - r90) / r50


#: Acima disto o PSF é gaussiano e o renderer está errado. Ver `edge_width_ratio`.
GAUSSIAN_EDGE_RATIO = 1.43
DISC_EDGE_RATIO_MAX = 0.5


# --------------------------------------------------------------------------------
# Teste 2 — raio == K * |Delta_disp|
# --------------------------------------------------------------------------------

def measure_blur_radius_px(
    image: np.ndarray,
    center: tuple[int, int],
    energy_fraction: float = 0.95,
) -> float:
    """Raio do borrão a partir da energia acumulada.

    Não usa limiar de intensidade: o brilho do ponto se espalha, então um limiar fixo
    mede coisa diferente conforme o raio. Energia acumulada é invariante a isso.

    Para um disco uniforme de raio R, 95% da energia está dentro de `sqrt(0.95)*R`,
    então o valor é corrigido para devolver R.
    """
    profile = radial_profile(image, center)
    radii = np.arange(len(profile), dtype=np.float64)
    energy = profile * (2.0 * np.pi * np.maximum(radii, 0.5))   # peso de anel
    total = energy.sum()
    if total <= 0:
        return 0.0
    cumulative = np.cumsum(energy) / total
    idx = np.nonzero(cumulative >= energy_fraction)[0]
    r_energy = float(idx[0]) if idx.size else float(len(profile) - 1)
    return r_energy / np.sqrt(energy_fraction)


@dataclass(frozen=True)
class RadiusCheck:
    k_value: float
    expected_px: float
    measured_px: float

    @property
    def relative_error(self) -> float:
        return abs(self.measured_px - self.expected_px) / max(self.expected_px, 1e-9)

    def passed(self, tolerance: float = 0.02) -> bool:
        return self.relative_error <= tolerance


def check_radius_matches_contract(
    render_fn: Callable[[np.ndarray, np.ndarray, float, float], np.ndarray],
    *,
    k_value: float,
    scene_depth_m: float = 10.0,
    focus_depth_m: float = 2.0,
    size: int = 257,
) -> RadiusCheck:
    """O teste que amarra o renderer à Eq. 2. Sem ele, todo K de um sweep é número
    sem unidade.

    `render_fn(aif_bgr, depth_m, focus_disparity, k_value) -> bgr_uint8`.

    O esperado vem direto do contrato: `CoC = K * |1/z - 1/z_focus|`, em pixels.
    """
    image, depth, center = point_light_scene(size=size, depth_m=scene_depth_m)
    focus_disparity = 1.0 / float(focus_depth_m)
    expected = abs(float(k_value) * (1.0 / float(scene_depth_m) - focus_disparity))
    rendered = render_fn(image, depth, focus_disparity, float(k_value))
    return RadiusCheck(float(k_value), expected, measure_blur_radius_px(rendered, center))


def check_radius_is_linear_in_k(
    render_fn: Callable[[np.ndarray, np.ndarray, float, float], np.ndarray],
    # Faixa alinhada com `scripts/verify_renderer.py`: com |Δdisp| = 0,4 dá 3,2 a
    # 38,4 px. Abaixo de ~5 px o piso aditivo de ~1 px do medidor domina e reprova
    # até um disco IDEAL — medido: slope 1,0535 com (4,8,16) contra 0,9973 com esta.
    k_values: tuple[float, ...] = (8.0, 16.0, 32.0, 64.0, 96.0),
    **kwargs,
) -> list[RadiusCheck]:
    """Linearidade em K é o que a Eq. 2 promete, e o que o teto de kernel quebra.

    Foi exatamente essa quebra que produziu `k == 300` exato em 1.379 de 2.932
    amostras da rota C: acima de um K o gaussiano de kernel 51 px parava de borrar,
    o SSIM parava de piorar, e o argmax corria para a borda da busca.
    """
    return [check_radius_matches_contract(render_fn, k_value=k, **kwargs) for k in k_values]


# --------------------------------------------------------------------------------
# Resposta de raio: ajuste linear, que é onde a linearidade em K realmente vive
# --------------------------------------------------------------------------------

@dataclass(frozen=True)
class RadiusResponse:
    """Ajuste `medido = slope * esperado + intercept` sobre vários K.

    Julgar cada ponto por erro relativo é a estatística ERRADA, e me levou a um
    veredito falso de "satura". O medidor tem um piso aditivo de ~1 px (fonte
    pontual discretizada, mais o componente neural que suaviza a borda), então em
    raio pequeno o erro relativo estoura mesmo quando a resposta é perfeitamente
    linear.

    Assinaturas, para não confundir de novo:
      - offset aditivo puro   -> resíduo ~0, slope ~1, intercept > 0.
        As razões ao dobrar K sobem para 2,0 vindas de baixo: 1,75 · 1,86 · 1,92.
      - saturação de kernel   -> resíduo CRESCE com K, e a razão CAI abaixo de 2.
      - erro de escala        -> resíduo ~0, intercept ~0, slope != 1.
    """

    k_values: tuple[float, ...]
    expected_px: tuple[float, ...]
    measured_px: tuple[float, ...]
    slope: float
    intercept_px: float
    max_residual_px: float

    @property
    def relative_residual(self) -> float:
        """Resíduo máximo como fração do maior raio testado.

        Absoluto em pixel é a estatística errada: 0,4 px é ruído num raio de 38 px e
        é enorme num raio de 3 px. Medido no BokehMe: `bokeh_classical` dá resíduo
        0,0000 px (reta exata) e o híbrido `bokeh_pred` dá 0,42 px sobre 38,4 px de
        alcance, ou seja 1,1% — a diferença vem do peso do `error_map`, que varia
        com K ao misturar clássico e neural.
        """
        largest = max(self.expected_px) if self.expected_px else 0.0
        return self.max_residual_px / largest if largest > 0 else float("inf")

    def is_linear(self, max_relative_residual: float = 0.02) -> bool:
        """Resposta é uma reta. Não diz nada sobre a escala — isso é o `slope`."""
        return self.relative_residual <= max_relative_residual

    def scale_matches_contract(self, tolerance: float = 0.02) -> bool:
        """`slope == 1` é o contrato: raio renderizado == K * |Delta_disp|."""
        return abs(self.slope - 1.0) <= tolerance

    def to_dict(self) -> dict:
        return {
            "k_values": [float(k) for k in self.k_values],
            "expected_px": [float(v) for v in self.expected_px],
            "measured_px": [float(v) for v in self.measured_px],
            "slope": float(self.slope),
            "intercept_px": float(self.intercept_px),
            "max_residual_px": float(self.max_residual_px),
            "relative_residual": float(self.relative_residual),
        }


def fit_radius_response(checks: list[RadiusCheck]) -> RadiusResponse:
    """Regressão de mínimos quadrados de `medido` contra `esperado`."""
    if len(checks) < 3:
        raise ValueError("são necessários ao menos 3 valores de K para separar escala de offset")
    expected = np.array([c.expected_px for c in checks], dtype=np.float64)
    measured = np.array([c.measured_px for c in checks], dtype=np.float64)
    design = np.vstack([expected, np.ones_like(expected)]).T
    slope, intercept = np.linalg.lstsq(design, measured, rcond=None)[0]
    residual = measured - (slope * expected + intercept)
    return RadiusResponse(
        k_values=tuple(c.k_value for c in checks),
        expected_px=tuple(expected.tolist()),
        measured_px=tuple(measured.tolist()),
        slope=float(slope),
        intercept_px=float(intercept),
        max_residual_px=float(np.abs(residual).max()),
    )
