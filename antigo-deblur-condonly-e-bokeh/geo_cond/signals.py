"""Sinais geométricos derivados da profundidade, para condicionar a BokehNet.

Seis canais, todos em [0,1], prontos para entrar no VAE em [0,1] CRU
(`No_preprocess=True`), a mesma convenção do mapa de defocus:

    G = [ u , O , s , n_x , n_y , K~ ]

Portado de `depth-riemannian/riemann/`, com as correções que a auditoria de
2026-09-03 levantou. Cada uma está marcada com CORRECAO no ponto onde entra.

DUAS CONVENCOES DE COORDENADA, DE PROPOSITO
-------------------------------------------
Elas parecem inconsistentes e não são. Cada uma é a certa para o seu objeto, e
trocar uma pela outra quebra a invariância que ela protege.

  * `O`, `s`, `n_x`, `n_y` descrevem a ANISOTROPIA DO NUCLEO DE DESFOQUE, que é
    um fenômeno do plano da imagem. Vivem em coordenadas de imagem NORMALIZADAS,
    d/d(x/L), com L em pixels passado pelo chamador. Isso os torna invariantes a
    redimensionamento, que importa porque o treino roda a 512 e a inferência do
    paper usa a resolução original com tiling.

  * `K~` descreve a FORMA 3D da superfície. Exige a superfície retroprojetada
    S(u,v) = ((u-cx)Z/fx, (v-cy)Z/fy, Z), em coordenadas de PIXEL com `fx` em
    pixels. K e H são invariantes por reparametrização, então basta que `fx`
    corresponda à resolução do array recebido. A correção do resize entra pelo
    `fx`, não pelo espaçamento.

Ver `AUDITORIA_DADOS_ROTAS_BC.md` seções 3.5 e 3.7, e a seção 3.2 do
`PLANO_CONDICIONAMENTO_GEOMETRICO.md`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .constants import GeoConstants

__all__ = [
    "CANAIS",
    "GeoStack",
    "depth01_to_metric",
    "geometric_stack",
    "gaussian_kernel_1d",
    "smooth",
    "grad_normalized",
    "second_derivs_px",
    "occlusion",
    "area_element_and_normals",
    "gaussian_curvature_backprojected",
    "quantization_levels",
    "niveis_uteis",
]

#: Ordem dos canais na pilha. NAO reordene sem mudar o agrupamento em branches
#: (ver PLANO seção 8.3) e sem retreinar: a ordem é parte do contrato do modelo.
CANAIS = ("u", "O", "s", "n_x", "n_y", "K~")

_EPS = 1e-12


@dataclass(frozen=True)
class GeoStack:
    """Saída de `geometric_stack`."""

    canais: np.ndarray
    """(6, H, W) float32, cada canal em [0,1], na ordem de `CANAIS`."""

    segunda_ordem_valida: bool
    """False quando a quantização da profundidade não sustenta derivada segunda.

    Quando False, o canal `K~` sai no valor NEUTRO (0.5, que é K=0 depois da
    normalização simétrica) em vez de ruído de quantização. O chamador deve
    propagar essa flag para excluir a amostra da supervisão de 2a ordem."""

    niveis_quantizacao: int
    """Quantos níveis uint16 distintos a cena útil ocupa. Diagnóstico."""


# =============================================================================
# Núcleos e derivadas
# =============================================================================

def gaussian_kernel_1d(sigma: float) -> np.ndarray:
    """Gaussiana 1D normalizada, com raio derivado de sigma.

    CORRECAO: `riemann/losses.py:86` fixa `radius = 2` independentemente de
    sigma. Com o sigma=2.0 que `visual_signals.py:146` usa, isso dá 5 taps
    cobrindo +-1 sigma, ou seja uma caixa truncada e não uma gaussiana. Aqui o
    raio é `round(3*sigma)`, como em `riemann/geometry.py:87`.
    """
    if sigma <= 0:
        raise ValueError(f"sigma deve ser > 0; recebido {sigma}")
    r = int(round(3.0 * sigma))
    x = np.arange(-r, r + 1, dtype=np.float64)
    k = np.exp(-(x ** 2) / (2.0 * sigma ** 2))
    return (k / k.sum()).astype(np.float64)


def smooth(a: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussiana separável com padding replicado."""
    k = gaussian_kernel_1d(sigma)
    r = (len(k) - 1) // 2
    out = np.pad(a.astype(np.float64), ((0, 0), (r, r)), mode="edge")
    out = np.apply_along_axis(lambda m: np.convolve(m, k, mode="valid"), 1, out)
    out = np.pad(out, ((r, r), (0, 0)), mode="edge")
    out = np.apply_along_axis(lambda m: np.convolve(m, k, mode="valid"), 0, out)
    return out


def grad_normalized(f: np.ndarray, px_per_unit: float) -> tuple[np.ndarray, np.ndarray]:
    """Gradiente em coordenadas de imagem NORMALIZADAS: d/d(x/L).

    `px_per_unit` é L, quantos pixels correspondem a uma unidade de coordenada
    normalizada. No treino é o lado curto do redimensionamento (512). Na
    inferência o chamador aplica a MESMA regra sobre a imagem que está usando.

    CORRECAO: `riemann/losses.py:48-50` usa `h = 1/max(H,W)` calculado a partir
    do array recebido. Sob crop aleatório isso muda a normalização entre treino
    e inferência, e é o mesmo defeito que a coluna `defocus_map` sofreu. Aqui L
    vem de fora e não depende do recorte.

    Ambos os eixos usam o MESMO L, senão uma imagem não quadrada ganharia
    anisotropia artificial exatamente no canal que mede anisotropia.
    """
    if px_per_unit <= 0:
        raise ValueError(f"px_per_unit deve ser > 0; recebido {px_per_unit}")
    fx = np.zeros_like(f, dtype=np.float64)
    fy = np.zeros_like(f, dtype=np.float64)
    # diferenças centrais no interior, laterais na borda (replicado)
    fx[:, 1:-1] = (f[:, 2:] - f[:, :-2]) * 0.5
    fx[:, 0] = f[:, 1] - f[:, 0]
    fx[:, -1] = f[:, -1] - f[:, -2]
    fy[1:-1, :] = (f[2:, :] - f[:-2, :]) * 0.5
    fy[0, :] = f[1, :] - f[0, :]
    fy[-1, :] = f[-1, :] - f[-2, :]
    return fx * px_per_unit, fy * px_per_unit


def second_derivs_px(f: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """f_uu, f_vv, f_uv em coordenadas de PIXEL (espaçamento 1).

    Pixel, e não normalizada, porque quem consome é a curvatura retroprojetada,
    que casa com `fx` em pixels. O termo misto usa o estêncil de 4 cantos com
    fator 1/4, que a auditoria confirmou correto na implementação original.
    """
    fuu = np.zeros_like(f, dtype=np.float64)
    fvv = np.zeros_like(f, dtype=np.float64)
    fuv = np.zeros_like(f, dtype=np.float64)
    fuu[:, 1:-1] = f[:, 2:] - 2.0 * f[:, 1:-1] + f[:, :-2]
    fvv[1:-1, :] = f[2:, :] - 2.0 * f[1:-1, :] + f[:-2, :]
    fuv[1:-1, 1:-1] = (
        f[2:, 2:] - f[2:, :-2] - f[:-2, 2:] + f[:-2, :-2]
    ) * 0.25
    return fuu, fvv, fuv


# =============================================================================
# Canais
# =============================================================================

def occlusion(
    grad_mag: np.ndarray, tau: float, valid: np.ndarray | None = None
) -> np.ndarray:
    """O = min(||grad f|| / tau, 1), com tau FIXO.

    CORRECAO 1: tau é constante, nunca percentil por imagem. Ver `constants.py`.
    CORRECAO 2: onde `valid` é falso, O sai 0 e a fronteira do inválido é
    erodida, senão ela vira a maior "oclusão" do mapa. É a mesma correção que
    `riemann/metrics.py:114-121` aplicou ao `boundary_fscore` e que o
    `geometry_maps.occlusion_map` nunca recebeu.
    CORRECAO 3: sem `torch.quantile`, que estoura acima de ~16,7M elementos.
    """
    if tau <= 0:
        raise ValueError(f"tau deve ser > 0; recebido {tau}")
    o = np.clip(grad_mag / tau, 0.0, 1.0)
    if valid is not None:
        o = o * _erode3(valid).astype(np.float64)
    return o


def _erode3(mask: np.ndarray) -> np.ndarray:
    """Erosão 3x3 booleana, sem scipy."""
    m = mask.astype(bool)
    out = m.copy()
    out[1:, :] &= m[:-1, :]
    out[:-1, :] &= m[1:, :]
    out[:, 1:] &= m[:, :-1]
    out[:, :-1] &= m[:, 1:]
    return out


def area_element_and_normals(
    fx: np.ndarray, fy: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Elemento de área e normais do grafo de Monge do campo.

    g_ij = delta_ij + d_i f d_j f  =>  sqrt(det g) = sqrt(1 + ||grad f||^2)
    s = log(sqrt(det g)) = 0.5 * log1p(||grad f||^2)   >= 0
    n = (-f_x, -f_y, 1) / sqrt(1 + ||grad f||^2)

    Nota de honestidade que vale para o texto do paper: `s`, `n_x` e `n_y` são
    uma REPARAMETRIZACAO de grad f, não informação complementar. São 2 graus de
    liberdade em 3 canais, em coordenadas polares. A parametrização é limitada e
    bem condicionada, que é o motivo de usá-la, mas dizer que "juntos determinam
    completamente o termo de primeira ordem" sugere complementaridade que não há.
    """
    q = fx * fx + fy * fy
    denom = np.sqrt(1.0 + q)
    s = 0.5 * np.log1p(q)
    n_x = -fx / denom
    n_y = -fy / denom
    return s, n_x, n_y


def gaussian_curvature_backprojected(
    z: np.ndarray, fx_px: float, fy_px: float, cx: float, cy: float, sigma: float
) -> np.ndarray:
    """Curvatura gaussiana da superfície 3D retroprojetada, em 1/m^2.

        S(u,v) = ( (u-cx) Z / fx , (v-cy) Z / fy , Z )

    CORRECAO: usa a formulação retroprojetada de `riemann/geometry.py`, não o
    grafo de Monge de `riemann/geometry_maps.principal_curvatures`. O
    `RETESTE_CURVATURA.md` mediu que a versão de Monge devolve ~10 onde o K
    verdadeiro é 0,25 (esfera de R=2 m a 10 m), e é ela que gerou as figuras do
    documento de proposta.

    `fx_px` tem de corresponder à resolução de `z`. Se `z` foi redimensionado,
    o chamador já aplicou o fator. K e H são invariantes por reparametrização,
    então com o `fx` certo o resultado é invariante à resolução.

    A suavização entra ANTES das segundas derivadas, com raio round(3*sigma).
    """
    if fx_px <= 0 or fy_px <= 0:
        raise ValueError(f"focal deve ser > 0; recebido fx={fx_px} fy={fy_px}")
    zs = smooth(z, sigma)
    H, W = zs.shape
    v_idx, u_idx = np.mgrid[0:H, 0:W]
    a = (u_idx - cx) / fx_px          # X = a * Z
    b = (v_idx - cy) / fy_px          # Y = b * Z

    zu = np.zeros_like(zs); zv = np.zeros_like(zs)
    zu[:, 1:-1] = (zs[:, 2:] - zs[:, :-2]) * 0.5
    zu[:, 0] = zs[:, 1] - zs[:, 0]; zu[:, -1] = zs[:, -1] - zs[:, -2]
    zv[1:-1, :] = (zs[2:, :] - zs[:-2, :]) * 0.5
    zv[0, :] = zs[1, :] - zs[0, :]; zv[-1, :] = zs[-1, :] - zs[-2, :]
    zuu, zvv, zuv = second_derivs_px(zs)

    inv_fx = 1.0 / fx_px
    inv_fy = 1.0 / fy_px

    # S_u = ( (Z + (u-cx) Z_u)/fx , (v-cy) Z_u/fy , Z_u )
    Su = (inv_fx * (zs + (u_idx - cx) * zu), b * zu, zu)
    Sv = (a * zv, inv_fy * (zs + (v_idx - cy) * zv), zv)
    Suu = (inv_fx * (2.0 * zu + (u_idx - cx) * zuu), b * zuu, zuu)
    Suv = (inv_fx * (zv + (u_idx - cx) * zuv), inv_fy * (zu + (v_idx - cy) * zuv), zuv)
    Svv = (a * zvv, inv_fy * (2.0 * zv + (v_idx - cy) * zvv), zvv)

    dot = lambda p, q: p[0] * q[0] + p[1] * q[1] + p[2] * q[2]
    E = dot(Su, Su); F = dot(Su, Sv); G = dot(Sv, Sv)

    nx = Su[1] * Sv[2] - Su[2] * Sv[1]
    ny = Su[2] * Sv[0] - Su[0] * Sv[2]
    nz = Su[0] * Sv[1] - Su[1] * Sv[0]
    nrm = np.sqrt(nx * nx + ny * ny + nz * nz) + _EPS
    N = (nx / nrm, ny / nrm, nz / nrm)

    L = dot(Suu, N); M = dot(Suv, N); Nn = dot(Svv, N)
    den = E * G - F * F
    # CORRECAO: epsilon no denominador. `metric_tensor_loss` levantava NaN no
    # backward em toda região plana por falta dele (P2b da auditoria).
    return (L * Nn - M * M) / np.where(np.abs(den) < _EPS, _EPS, den)


# =============================================================================
# Profundidade métrica e quantização
# =============================================================================

def depth01_to_metric(depth01: np.ndarray, z_min_m: float, z_max_m: float) -> np.ndarray:
    """depth01 em [0,1] -> profundidade em metros.

        z = z_min + depth01 * (z_max - z_min)

    Esta é a relação de `genfocus_train/data.py:483`, e ela foi VERIFICADA em
    pixels: recalcular o `coc_p99_px` gravado na kfix a partir dos pixels de
    `depth` sob esta hipótese dá erro mediano de 0,011% (15/15 amostras abaixo de
    1%), contra 3,008% da hipótese de disparidade. Ver AUDITORIA seção 3.

    `z_max_m` TEM DE SER O VALOR BRUTO, não um percentil. O job F0c ajustou
    `z ~ a*depth01 + b` contra uma execução nova do Depth Pro em 40 amostras da
    rota c: R^2 mediano 1,00000, erro relativo de `a` 0,04% e de `b` 0,01%,
    38/40 abaixo de 5%. Ou seja a coluna `depth` armazenada foi normalizada com o
    max BRUTO, céu incluído. Passar aqui um `z_max` percentilado reconstrói uma
    profundidade ERRADA, e o erro é invisível no mapa de defocus, que reescala
    tudo por `max_coc` de qualquer forma.

    O percentil continua útil, mas para outra coisa: diagnosticar a sentinela e
    alimentar o gate de `niveis_uteis`. Não para reconstruir.
    """
    if not (z_max_m > z_min_m > 0):
        raise ValueError(f"faixa métrica inválida: z_min={z_min_m} z_max={z_max_m}")
    return z_min_m + depth01.astype(np.float64) * (z_max_m - z_min_m)


def quantization_levels(depth01: np.ndarray, u16_levels: int = 65535) -> int:
    """Quantos níveis uint16 distintos o mapa inteiro ocupa.

    ATENCAO: esta contagem inclui o CEU. Numa cena com céu no teto de 10.000 m ela
    devolve número alto e enganoso, porque os níveis estão quase todos no fundo
    irrelevante enquanto a cena útil vive nos primeiros milésimos. Para o gate dos
    canais de segunda ordem use `niveis_uteis`, que é o número que importa.
    """
    q = np.rint(np.clip(depth01, 0.0, 1.0) * u16_levels).astype(np.int64)
    return int(np.unique(q).size)


def niveis_uteis(
    z_min_m: float, z_max_m: float, z_focus_m: float,
    *, corte: float = 20.0, u16_levels: int = 65535,
) -> float:
    """Níveis uint16 que a REGIAO UTIL recebe na codificação armazenada.

    Região útil = `z <= corte * z_focus`, onde o círculo de confusão ainda
    distingue profundidades (`CoC ~ |1/z - 1/z_focus|` satura muito antes disso).

    Medido na rota c (2.932 amostras, job F0b):

    | subconjunto | n | p10 | p50 | abaixo de 256 |
    |---|---|---|---|---|
    | `z_max` não saturado | 2.653 | 65.535 | 65.535 | 0 (0,0%) |
    | `z_max` saturado em 10.000 m | 279 | 133 | 921 | 62 (22,2%) |

    Ou seja o problema é real mas **concentrado**: 62 de 2.932 amostras (2,1%)
    ficam abaixo de 256 níveis úteis. Nelas a derivada segunda é ruído de
    quantização, não geometria, e os canais de 2a ordem devem sair neutros.

    `z_max_m` aqui é o valor **BRUTO**, não o percentil. Ver a nota em
    `depth01_to_metric`.
    """
    faixa = z_max_m - z_min_m
    if faixa <= 0 or z_focus_m <= 0:
        return 0.0
    util = min(corte * z_focus_m, z_max_m) - z_min_m
    return u16_levels * max(util, 0.0) / faixa


# =============================================================================
# Driver
# =============================================================================

def geometric_stack(
    depth01: np.ndarray,
    *,
    z_min_m: float,
    z_max_m: float,
    fx_px: float,
    fy_px: float,
    cx: float,
    cy: float,
    px_per_unit: float,
    consts: GeoConstants,
    field: str = "inverse",
    valid: np.ndarray | None = None,
) -> GeoStack:
    """Monta os 6 canais em [0,1] a partir da profundidade normalizada.

    Parameters
    ----------
    depth01
        (H, W) em [0,1], profundidade MÉTRICA min-max normalizada por imagem
        (convenção verificada em pixels, ver `depth01_to_metric`).
    z_min_m, z_max_m
        Faixa métrica da amostra. Vem da tabela kfix na rota b e do job de Depth
        Pro na rota c. O `z_max_m` já deve chegar CORRIGIDO pelo percentil, ver
        `GeoConstants.z_percentile_max`: a correção é do job de dados, não daqui,
        porque depende do mapa inteiro e não do recorte.
    fx_px, fy_px, cx, cy
        Intrínsecos NA RESOLUCAO de `depth01`. Se a imagem foi redimensionada, o
        chamador já aplicou o fator (medido: o dataloader reescala o lado menor
        para 512, o que muda `fx` por 512/min(W,H), fator que varia por amostra).
    px_per_unit
        Pixels por unidade de coordenada normalizada, para os canais de 1a ordem.
        No treino é o lado curto do redimensionamento. Ver `grad_normalized`.
    field
        "inverse" (default) usa u = 1/Z; "depth" usa Z, a formulação literal do
        documento de proposta.

        O default é "inverse" por física: o raio do círculo de confusão é linear
        em 1/Z, e o termo de anisotropia da expansão é
        eps = gamma*||grad D||/(Z^2 c), com grad(1/Z) = -grad D / Z^2. Logo a
        magnitude que governa a anisotropia é ||grad u||, não ||grad D||. Com
        ||grad D|| o fundo distante domina o sinal, que é justamente onde o
        borrão é mais uniforme: um degrau de 1 m para 20 m dá 19 em D e 0,95 em
        u, e um de 20 m para 40 m dá 20 em D, MAIOR, e 0,025 em u.

        "depth" existe para que a diferença seja ablacionável, não porque as duas
        sejam equivalentes.
    valid
        Máscara booleana opcional de pixels válidos.

    Returns
    -------
    GeoStack
    """
    if field not in ("inverse", "depth"):
        raise ValueError(f"field deve ser 'inverse' ou 'depth'; recebido {field!r}")
    if depth01.ndim != 2:
        raise ValueError(f"depth01 deve ser (H, W); recebido {depth01.shape}")

    z = depth01_to_metric(depth01, z_min_m, z_max_m)
    u = 1.0 / z

    f = u if field == "inverse" else z
    f_x, f_y = grad_normalized(f, px_per_unit)
    grad_mag = np.sqrt(f_x * f_x + f_y * f_y)

    ch_O = occlusion(grad_mag, consts.tau_occlusion, valid)
    s, n_x, n_y = area_element_and_normals(f_x, f_y)

    ch_u = np.clip(u / consts.u_max, 0.0, 1.0)
    ch_s = np.clip(s / consts.s_max, 0.0, 1.0)
    ch_nx = (n_x + 1.0) * 0.5
    ch_ny = (n_y + 1.0) * 0.5

    niveis = quantization_levels(depth01)
    segunda_ok = niveis >= consts.min_quant_levels
    if segunda_ok:
        K = gaussian_curvature_backprojected(
            z, fx_px, fy_px, cx, cy, consts.smooth_sigma
        )
        Kt = np.sign(K) * np.log1p(np.abs(K) / consts.k0_curvature)
        ch_K = np.clip(Kt, -consts.kt_max, consts.kt_max) / (2.0 * consts.kt_max) + 0.5
    else:
        # NEUTRO (0.5 == K=0 depois da normalização simétrica), não ruído.
        ch_K = np.full_like(depth01, 0.5, dtype=np.float64)

    canais = np.stack([ch_u, ch_O, ch_s, ch_nx, ch_ny, ch_K], axis=0)
    if not np.all(np.isfinite(canais)):
        raise FloatingPointError("canal geométrico não finito; investigue antes de treinar")
    canais = np.clip(canais, 0.0, 1.0).astype(np.float32)
    return GeoStack(
        canais=canais, segunda_ordem_valida=bool(segunda_ok), niveis_quantizacao=niveis
    )
