"""Sinais geométricos derivados da DISPARIDADE métrica, para condicionar a BokehNet.

Seis canais, todos em [0,1], prontos para entrar no VAE em [0,1] CRU
(`No_preprocess=True`), a mesma convenção do mapa de defocus:

    G = [ u , O , s , n_x , n_y , K~ ]

A matemática (curvatura retroprojetada, suavização com raio 3σ, gradiente
normalizado) é portada de `referencia_antiga/geo_cond_v1/signals.py`, onde foi
verificada contra esfera / plano / cilindro. O que NÃO é portado é o contrato de
entrada.

O QUE MUDOU NA ENTRADA — LEIA ISTO ANTES DE ESTRANHAR O CANAL `u`
-----------------------------------------------------------------
A versão antiga recebia `depth01` = profundidade MÉTRICA min-max normalizada, e
reconstruía metros por uma tabela externa (`z_min_m`, `z_max_m`). O release
`metric_disparity_official_v1` grava DISPARIDADE: o PNG uint16 dividido por
65535 é linear entre `disparity_min` e `disparity_max`, ambos em 1/m e ambos no
`meta/<cena>/<id>.json` da própria amostra:

    disp_metrica = disparity_min + d01 * (disparity_max - disparity_min)   # 1/m
    z_metrico    = 1.0 / disp_metrica                                      # m

Consequência que surpreende quem leu a versão antiga: **o canal `u` é
`disp_metrica` DIRETO, só dividido por `consts.u_max`. Não há inversão de
profundidade em lugar nenhum do caminho de `u`.** Na versão antiga era
`u = 1/z` com `z` reconstruído; aqui a disparidade JÁ É o dado armazenado, e
`z = 1/disp` é que passou a ser a grandeza derivada. A composição
`1/(1/disp)` seria identidade, e escrevê-la assim só serviria para parecer
familiar e esconder que a ida e volta por `z` perde precisão perto do infinito.

DUAS CONVENÇÕES DE COORDENADA, DE PROPÓSITO
-------------------------------------------
Elas parecem inconsistentes e não são. Cada uma é a certa para o seu objeto, e
trocar uma pela outra quebra a invariância que ela protege.

  * `O`, `s`, `n_x`, `n_y` descrevem a ANISOTROPIA DO NÚCLEO DE DESFOQUE, que é
    um fenômeno do plano da imagem. Vivem em coordenadas de imagem NORMALIZADAS,
    d/d(x/L), com L derivado de `amostra.largura_px`/`altura_px`. Isso os torna
    invariantes a redimensionamento, que importa porque o treino roda a 512² e a
    inferência oficial usa a resolução original com tiling.

  * `K~` descreve a FORMA 3D da superfície. Exige a superfície retroprojetada
    S(u,v) = ((u−cx)Z/fx, (v−cy)Z/fy, Z), em coordenadas de PIXEL com `fx` em
    pixels NA RESOLUÇÃO DO ARRAY. K é invariante por reparametrização, então
    basta que `fx` corresponda à resolução recebida: a correção do resize entra
    pelo `fx`, não pelo espaçamento.

A régua que separa a formulação certa da errada é fechada e não depende de dado:
esfera de raio R tem K = 1/R². O `RETESTE_CURVATURA.md` mediu que a formulação
de Monge (`geometry_maps.principal_curvatures`) devolve ~10 onde o valor
verdadeiro é 0,25 (esfera de R = 2 m a 10 m). Por isso aqui é a retroprojetada.
"""

from __future__ import annotations

import numpy as np

from .constants import GeoConstants
from .contrato import CANAIS, AmostraGeo, PilhaGeo

__all__ = [
    "CANAIS",
    "AmostraGeo",
    "PilhaGeo",
    "pilha_geometrica",
    "disparidade_metrica",
    "profundidade_metrica",
    "nucleo_gaussiano_1d",
    "suavizar",
    "gradiente_normalizado",
    "derivadas_segundas_px",
    "oclusao",
    "elemento_de_area_e_normais",
    "curvatura_gaussiana_retroprojetada",
    "niveis_quantizacao",
    "comprimento_de_normalizacao",
]

_EPS = 1e-12

#: Piso de disparidade, em 1/m, abaixo do qual `z = 1/disp` deixa de ser um
#: número e vira uma sentinela. 1e-6 1/m é z = 1.000 km: nenhuma cena do release
#: tem isso, então o piso só é atingido quando `disparity_min` é literalmente 0
#: (céu a distância infinita). Esses pixels entram na máscara de inválidos em vez
#: de gerarem um `z` de 1e6 m que dominaria qualquer gradiente de `field="depth"`.
_DISP_PISO = 1e-6

#: Níveis do uint16 do release. O PNG é `uint16_linear_in_disparity`, logo
#: `d01 = PNG / 65535`.
NIVEIS_U16 = 65535


# =============================================================================
# Reconstrução métrica — o contrato novo, em duas linhas
# =============================================================================

def disparidade_metrica(disp01: np.ndarray, amostra: AmostraGeo) -> np.ndarray:
    """`d01` em [0,1] -> disparidade métrica em 1/m.

        disp = disparity_min + d01 * (disparity_max - disparity_min)

    Esta é a relação declarada pelo `depth_encoding: uint16_linear_in_disparity`
    do release, com os dois extremos vindo do `meta/` da PRÓPRIA amostra. É a
    diferença de fundo para a campanha antiga, onde a faixa vinha de uma tabela
    externa e por isso podia (e chegou a) divergir entre treino e inferência.
    """
    d_min = float(amostra.disparity_min)
    d_max = float(amostra.disparity_max)
    if not np.isfinite(d_min) or not np.isfinite(d_max):
        raise ValueError(f"faixa de disparidade não finita: [{d_min}, {d_max}]")
    if d_max <= d_min:
        raise ValueError(
            f"faixa de disparidade inválida: disparity_min={d_min} "
            f"disparity_max={d_max}; exige-se max > min"
        )
    if d_min < 0.0:
        raise ValueError(f"disparity_min negativa ({d_min}): disparidade é 1/m ≥ 0")
    return d_min + np.asarray(disp01, dtype=np.float64) * (d_max - d_min)


def profundidade_metrica(disp: np.ndarray) -> np.ndarray:
    """Disparidade métrica (1/m) -> profundidade (m), com piso.

    `z = 1/disp`. O piso `_DISP_PISO` evita divisão por zero quando
    `disparity_min == 0` (céu). Os pixels afetados saem marcados como inválidos
    por `mascara_valida`, então não contaminam a oclusão.
    """
    return 1.0 / np.maximum(np.asarray(disp, dtype=np.float64), _DISP_PISO)


def comprimento_de_normalizacao(amostra: AmostraGeo) -> float:
    """O `L` de d/d(x/L), em pixels.

    DECISÃO, porque o contrato não a fixa: **um único L para os dois eixos**, e
    esse L é `min(largura_px, altura_px)`.

    Único, porque usar `largura_px` em x e `altura_px` em y daria a uma imagem
    16:9 uma anisotropia de fator 1,78 EXATAMENTE nos canais que medem
    anisotropia (`n_x`, `n_y`, e por tabela `O` e `s`). O núcleo de desfoque não
    sabe que a imagem foi recortada num formato; inventar anisotropia de
    enquadramento é criar sinal que não existe na física.

    `min` e não `max` porque é a convenção sob a qual as constantes são
    calibradas: o pipeline reescala o LADO CURTO para a resolução de treino, e o
    lado curto é o único comprimento que o resize preserva entre amostras de
    proporções diferentes. Qualquer escolha consistente daria invariância a
    resolução; a escolha do `min` é o que amarra a escala absoluta de
    `tau_occlusion` e `s_max` ao número que o job de calibração vai medir.
    """
    W = int(amostra.largura_px)
    H = int(amostra.altura_px)
    if W <= 0 or H <= 0:
        raise ValueError(f"dimensões inválidas: largura_px={W} altura_px={H}")
    return float(min(W, H))


# =============================================================================
# Núcleos e derivadas
# =============================================================================

def nucleo_gaussiano_1d(sigma: float) -> np.ndarray:
    """Gaussiana 1D normalizada, com raio DERIVADO de sigma: r = round(3σ).

    CORREÇÃO herdada: `riemann/losses.py:86` fixava `radius = 2` para qualquer
    sigma. Com o sigma = 2,0 que `visual_signals.py:146` usava, isso dá 5 taps
    cobrindo ±1σ — ou seja uma caixa truncada, e não uma gaussiana: o kernel
    perde ~32% da massa e o que sobra é redistribuído pela renormalização, o que
    ALARGA a resposta em alta frequência que a suavização existia para tirar.
    Aqui o raio é `round(3σ)`, como em `riemann/geometry.py:87`.
    """
    if sigma <= 0:
        raise ValueError(f"sigma deve ser > 0; recebido {sigma}")
    r = int(round(3.0 * sigma))
    x = np.arange(-r, r + 1, dtype=np.float64)
    k = np.exp(-(x ** 2) / (2.0 * sigma ** 2))
    return k / k.sum()


def _conv1d_replicado(a: np.ndarray, k: np.ndarray, eixo: int) -> np.ndarray:
    """Convolução 1D ao longo de `eixo`, com padding replicado.

    Soma de cópias deslocadas em vez de `np.apply_along_axis(np.convolve, ...)`:
    o kernel tem no máximo algumas dezenas de taps e a versão vetorizada é ~50x
    mais rápida num mapa 512², o que importa porque isto roda por amostra no
    dataloader. O kernel é simétrico, então a inversão que a convolução faria é
    irrelevante.
    """
    r = (len(k) - 1) // 2
    pad = [(0, 0), (0, 0)]
    pad[eixo] = (r, r)
    ap = np.pad(np.asarray(a, dtype=np.float64), pad, mode="edge")
    n = a.shape[eixo]
    out = np.zeros(a.shape, dtype=np.float64)
    for i, w in enumerate(k):
        sl = [slice(None), slice(None)]
        sl[eixo] = slice(i, i + n)
        out += w * ap[tuple(sl)]
    return out


def suavizar(a: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussiana separável com padding replicado."""
    k = nucleo_gaussiano_1d(sigma)
    return _conv1d_replicado(_conv1d_replicado(a, k, eixo=1), k, eixo=0)


def gradiente_normalizado(f: np.ndarray, L: float) -> tuple[np.ndarray, np.ndarray]:
    """Gradiente em coordenadas de imagem NORMALIZADAS: d/d(x/L).

    `L` é quantos pixels correspondem a uma unidade de coordenada normalizada,
    e vem de FORA (de `comprimento_de_normalizacao`), nunca do `.shape` do array.

    CORREÇÃO herdada: `riemann/losses.py:48-50` usava `h = 1/max(H,W)` calculado
    a partir do array recebido. Sob crop aleatório isso muda a normalização entre
    treino e inferência — o mesmo defeito que a coluna `defocus_map` sofreu e que
    invalidou a campanha anterior (erro relativo residual 0,926, Pearson 0,341).

    Ambos os eixos usam o MESMO L. Ver `comprimento_de_normalizacao`.
    """
    if L <= 0:
        raise ValueError(f"L deve ser > 0; recebido {L}")
    f = np.asarray(f, dtype=np.float64)
    fx = np.zeros_like(f)
    fy = np.zeros_like(f)
    # diferenças centrais no interior, laterais na borda (replicado)
    fx[:, 1:-1] = (f[:, 2:] - f[:, :-2]) * 0.5
    fx[:, 0] = f[:, 1] - f[:, 0]
    fx[:, -1] = f[:, -1] - f[:, -2]
    fy[1:-1, :] = (f[2:, :] - f[:-2, :]) * 0.5
    fy[0, :] = f[1, :] - f[0, :]
    fy[-1, :] = f[-1, :] - f[-2, :]
    return fx * L, fy * L


def derivadas_segundas_px(f: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """f_uu, f_vv, f_uv em coordenadas de PIXEL (espaçamento 1).

    Pixel, e não normalizada, porque quem consome é a curvatura retroprojetada,
    que casa com `fx` em pixels. O termo misto usa o estêncil de 4 cantos com
    fator 1/4, que a auditoria confirmou correto na implementação original.
    """
    f = np.asarray(f, dtype=np.float64)
    fuu = np.zeros_like(f)
    fvv = np.zeros_like(f)
    fuv = np.zeros_like(f)
    fuu[:, 1:-1] = f[:, 2:] - 2.0 * f[:, 1:-1] + f[:, :-2]
    fvv[1:-1, :] = f[2:, :] - 2.0 * f[1:-1, :] + f[:-2, :]
    fuv[1:-1, 1:-1] = (f[2:, 2:] - f[2:, :-2] - f[:-2, 2:] + f[:-2, :-2]) * 0.25
    return fuu, fvv, fuv


# =============================================================================
# Canais de primeira ordem
# =============================================================================

def _erode3(mask: np.ndarray) -> np.ndarray:
    """Erosão 3x3 booleana (vizinhança-4), sem scipy."""
    m = np.asarray(mask, dtype=bool)
    out = m.copy()
    out[1:, :] &= m[:-1, :]
    out[:-1, :] &= m[1:, :]
    out[:, 1:] &= m[:, :-1]
    out[:, :-1] &= m[:, 1:]
    return out


def oclusao(grad_mag: np.ndarray, tau: float, valida: np.ndarray | None = None) -> np.ndarray:
    """O = min(‖∇f‖ / tau, 1), com tau FIXO entre imagens.

    CORREÇÃO 1: tau é constante, nunca percentil por imagem. Ver `constants.py`.

    CORREÇÃO 2: onde `valida` é falso, O sai 0 E a fronteira do inválido é
    ERODIDA. Sem a erosão, a descontinuidade artificial na borda de uma região
    inválida produz o maior ‖∇f‖ do mapa inteiro e vira "a maior oclusão da
    imagem" — um artefato de máscara supervisionando o modelo como se fosse
    geometria. É a mesma correção que `riemann/metrics.py:114-121` aplicou ao
    `boundary_fscore` e que o `geometry_maps.occlusion_map` nunca recebeu.

    CORREÇÃO 3: sem `torch.quantile`, que estoura acima de ~16,7M elementos.
    """
    if tau <= 0:
        raise ValueError(f"tau deve ser > 0; recebido {tau}")
    o = np.clip(np.asarray(grad_mag, dtype=np.float64) / tau, 0.0, 1.0)
    if valida is not None:
        o = o * _erode3(valida).astype(np.float64)
    return o


def elemento_de_area_e_normais(
    fx: np.ndarray, fy: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Elemento de área e normais do grafo de Monge do campo.

        g_ij = δ_ij + ∂_i f ∂_j f   ⇒   sqrt(det g) = sqrt(1 + ‖∇f‖²)
        s = log(sqrt(det g)) = ½·log1p(‖∇f‖²)   ≥ 0
        n = (−f_x, −f_y, 1) / sqrt(1 + ‖∇f‖²)

    Nota de honestidade que vale para o texto do paper: `s`, `n_x` e `n_y` são
    uma REPARAMETRIZAÇÃO de ∇f, não informação complementar. São 2 graus de
    liberdade em 3 canais, em coordenadas polares. A parametrização é limitada e
    bem condicionada, que é o motivo de usá-la, mas dizer que "juntos determinam
    completamente o termo de primeira ordem" sugere complementaridade que não há.
    É também o motivo de `G1_IDX = (s, n_x, n_y)` ficarem JUNTOS no mesmo branch:
    separá-los daria a um branch metade de uma coordenada polar.
    """
    q = fx * fx + fy * fy
    denom = np.sqrt(1.0 + q)
    s = 0.5 * np.log1p(q)
    return s, -fx / denom, -fy / denom


# =============================================================================
# Canal de segunda ordem
# =============================================================================

def curvatura_gaussiana_retroprojetada(
    z: np.ndarray, fx_px: float, fy_px: float, cx: float, cy: float, sigma: float
) -> np.ndarray:
    """Curvatura gaussiana da superfície 3D retroprojetada, em 1/m².

        S(u,v) = ( (u−cx)·Z/fx , (v−cy)·Z/fy , Z )

    CORREÇÃO: usa a formulação retroprojetada de `riemann/geometry.py`, não o
    grafo de Monge de `riemann/geometry_maps.principal_curvatures`. O
    `RETESTE_CURVATURA.md` mediu que a versão de Monge devolve ~10 onde o K
    verdadeiro é 0,25 (esfera de R = 2 m a 10 m) — fator 40, e foi ela que gerou
    as figuras do documento de proposta. `test_signals.py` refaz essa medição.

    `fx_px` tem de corresponder à resolução de `z`. Se `z` foi redimensionado, o
    chamador já aplicou o fator. K é invariante por reparametrização, então com o
    `fx` certo o resultado é invariante à resolução — e com o `fx` errado sai
    errado por mais de 50%, o que `test_signals.py` também exige.

    A suavização entra ANTES das segundas derivadas, com raio round(3σ).
    """
    if not (fx_px > 0) or not (fy_px > 0):
        raise ValueError(f"focal deve ser > 0; recebido fx={fx_px} fy={fy_px}")
    zs = suavizar(z, sigma)
    H, W = zs.shape
    v_idx, u_idx = np.mgrid[0:H, 0:W]
    a = (u_idx - cx) / fx_px          # X = a·Z
    b = (v_idx - cy) / fy_px          # Y = b·Z

    zu = np.zeros_like(zs)
    zv = np.zeros_like(zs)
    zu[:, 1:-1] = (zs[:, 2:] - zs[:, :-2]) * 0.5
    zu[:, 0] = zs[:, 1] - zs[:, 0]
    zu[:, -1] = zs[:, -1] - zs[:, -2]
    zv[1:-1, :] = (zs[2:, :] - zs[:-2, :]) * 0.5
    zv[0, :] = zs[1, :] - zs[0, :]
    zv[-1, :] = zs[-1, :] - zs[-2, :]
    zuu, zvv, zuv = derivadas_segundas_px(zs)

    inv_fx = 1.0 / fx_px
    inv_fy = 1.0 / fy_px

    # S_u = ( (Z + (u−cx)·Z_u)/fx , (v−cy)·Z_u/fy , Z_u )
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

    L2 = dot(Suu, N); M = dot(Suv, N); N2 = dot(Svv, N)
    den = E * G - F * F
    # CORREÇÃO: epsilon no denominador. `metric_tensor_loss` levantava NaN no
    # backward em TODA região plana por falta dele (P2b da auditoria) — e região
    # plana é a maioria dos pixels de um retrato.
    return (L2 * N2 - M * M) / np.where(np.abs(den) < _EPS, _EPS, den)


def niveis_quantizacao(disp01: np.ndarray) -> int:
    """Quantos níveis uint16 distintos o mapa ocupa.

    Na codificação NOVA esta contagem simples já é o número que importa, e isso é
    uma mudança em relação à `geo_cond_v1`, que precisava de um `niveis_uteis`
    separado. Lá a contagem no mapa inteiro era enganosa porque o uint16 era
    linear em Z: com céu a 10.000 m, a cena útil até 60 m ficava com
    65.535·59/9.999 ≈ 387 níveis enquanto a contagem bruta dava dezenas de
    milhares. Aqui o uint16 é linear em DISPARIDADE, e a mesma cena recebe
    65.535·(1 − 1/60)/(1 − 1/10.000) ≈ 64.440 níveis: a resolução caiu onde o CoC
    é insensível (o fundo), que é onde ela devia cair. Logo não há discrepância
    entre a contagem bruta e a contagem útil que justifique dois números.
    """
    q = np.rint(np.clip(np.asarray(disp01, dtype=np.float64), 0.0, 1.0) * NIVEIS_U16)
    return int(np.unique(q.astype(np.int64)).size)


def mascara_valida(disp01: np.ndarray, disp: np.ndarray) -> np.ndarray:
    """Pixels em que a geometria significa alguma coisa.

    Inválido é: `d01` não finito (buraco no mapa) ou disparidade no piso
    `_DISP_PISO` (ponto no infinito, só alcançável quando `disparity_min == 0`).
    Nas amostras do release em que `disparity_min > 0` — a esmagadora maioria,
    porque o Depth Pro devolve profundidade finita — esta máscara é toda
    verdadeira e a erosão de `oclusao` não remove nada, nem sequer na borda.
    """
    d01 = np.asarray(disp01, dtype=np.float64)
    return np.isfinite(d01) & np.isfinite(disp) & (disp > _DISP_PISO)


# =============================================================================
# Driver
# =============================================================================

def pilha_geometrica(
    disp01: np.ndarray,
    amostra: AmostraGeo,
    consts: GeoConstants,
    *,
    field: str = "inverse",
    valida: np.ndarray | None = None,
) -> PilhaGeo:
    """Monta os 6 canais em [0,1] a partir da disparidade normalizada do release.

    Parameters
    ----------
    disp01
        (H, W) float em [0,1] — o PNG uint16 dividido por 65535. NÃO é
        profundidade: ver o cabeçalho do módulo.
    amostra
        Escalares do `meta/<cena>/<id>.json` da amostra. `disparity_min`/`max`
        dão a escala absoluta; `largura_px`/`altura_px` dão o `L` do gradiente
        normalizado; `focallength_px` (que o release NÃO traz) habilita `K~`.
    consts
        Constantes fixas de normalização, calibradas no conjunto de treino.
    field
        "inverse" (default) deriva a DISPARIDADE; "depth" deriva Z.

        O default é "inverse" por física, não por gosto: o raio do círculo de
        confusão é linear na disparidade, e o termo de anisotropia da expansão é
        eps = γ·‖∇D‖/(Z²c) com ∇u = −∇D/Z², logo eps ∝ ‖∇u‖. Com ‖∇D‖ o fundo
        distante domina o sinal, que é justamente onde o borrão é mais uniforme:
        um degrau de 1 m para 20 m dá 19 em D e 0,95 em u; um de 20 m para 40 m
        dá 20 em D, MAIOR, e 0,025 em u — razão 38 numa ordenação e 0,95 na
        outra. "depth" existe para a diferença ser ablacionável, não porque seja
        defensável.
    valida
        Máscara booleana opcional de pixels válidos, combinada (AND) com a
        máscara derivada dos próprios dados. Keyword-only e opcional: o contrato
        fixa só `field`, e o dataloader que não tem máscara não precisa mudar.

    Returns
    -------
    PilhaGeo
    """
    if field not in ("inverse", "depth"):
        raise ValueError(f"field deve ser 'inverse' ou 'depth'; recebido {field!r}")
    disp01 = np.asarray(disp01, dtype=np.float64)
    if disp01.ndim != 2:
        raise ValueError(f"disp01 deve ser (H, W); recebido {disp01.shape}")

    H, W = disp01.shape

    # --- reconstrução métrica: as duas linhas que definem o contrato novo -----
    disp = disparidade_metrica(disp01, amostra)   # 1/m
    z = profundidade_metrica(disp)                # m

    ok = mascara_valida(disp01, disp)
    if valida is not None:
        valida = np.asarray(valida, dtype=bool)
        if valida.shape != disp01.shape:
            raise ValueError(
                f"máscara {valida.shape} não bate com disp01 {disp01.shape}"
            )
        ok = ok & valida

    # --- primeira ordem, em coordenadas de imagem normalizadas ---------------
    L = comprimento_de_normalizacao(amostra)
    f = disp if field == "inverse" else z
    f_x, f_y = gradiente_normalizado(f, L)
    grad_mag = np.hypot(f_x, f_y)

    ch_O = oclusao(grad_mag, consts.tau_occlusion, ok if not ok.all() else None)
    s, n_x, n_y = elemento_de_area_e_normais(f_x, f_y)

    # O canal `u` é a disparidade métrica DIRETO. Sem inversão: o release já
    # grava disparidade, e `1/(1/disp)` só seria uma volta que perde precisão.
    ch_u = np.clip(disp / consts.u_max, 0.0, 1.0)
    ch_s = np.clip(s / consts.s_max, 0.0, 1.0)
    ch_nx = (n_x + 1.0) * 0.5
    ch_ny = (n_y + 1.0) * 0.5

    # --- segunda ordem, na superfície retroprojetada --------------------------
    niveis = niveis_quantizacao(disp01)
    tem_focal = amostra.focallength_px is not None and float(amostra.focallength_px) > 0
    quant_ok = niveis >= int(consts.min_niveis_para_segunda_ordem)
    segunda_ok = bool(tem_focal and quant_ok)

    if segunda_ok:
        fx_px = float(amostra.focallength_px)
        # cx, cy: o `meta/` não traz ponto principal, então assume-se o centro do
        # ARRAY RECEBIDO — ver a nota de decisão no fim do módulo. fy = fx porque
        # o release também não traz pixels não quadrados.
        cx = (W - 1) / 2.0
        cy = (H - 1) / 2.0
        K = curvatura_gaussiana_retroprojetada(z, fx_px, fx_px, cx, cy, consts.smooth_sigma)
        Kt = np.sign(K) * np.log1p(np.abs(K) / consts.k0_curvature)
        ch_K = np.clip(Kt, -consts.kt_max, consts.kt_max) / (2.0 * consts.kt_max) + 0.5
    else:
        # NEUTRO (0,5 == K=0 depois da normalização simétrica), não ruído, e
        # nunca um `fx` inventado: a curvatura retroprojetada é EXTREMAMENTE
        # sensível a fx (fator 2 no fx move K por mais de 50%), então um palpite
        # produziria um canal plausível e errado — a assinatura exata do defeito
        # que invalidou a campanha anterior.
        ch_K = np.full((H, W), 0.5, dtype=np.float64)

    canais = np.stack([ch_u, ch_O, ch_s, ch_nx, ch_ny, ch_K], axis=0)
    if not np.all(np.isfinite(canais)):
        raise FloatingPointError(
            "canal geométrico não finito; investigue antes de treinar "
            f"(canais não finitos: "
            f"{[CANAIS[i] for i in range(6) if not np.all(np.isfinite(canais[i]))]})"
        )
    canais = np.clip(canais, 0.0, 1.0).astype(np.float32)

    diagnostico = {
        "niveis_quantizacao": int(niveis),
        "frac_O_saturado": float(np.mean(ch_O >= 1.0)),
        # Dicionário por NOME, não lista: quem compara máximo absoluto ENTRE
        # amostras (o único teste que detecta normalização por imagem, porque
        # correlação é invariante a escala) tem de saber de qual canal está
        # falando sem contar índices.
        "max_abs_por_canal": {
            nome: float(np.abs(canais[i]).max()) for i, nome in enumerate(CANAIS)
        },
        "field": field,
        "L_normalizacao": L,
        "focallength_px": (None if not tem_focal else float(amostra.focallength_px)),
        "quantizacao_suficiente": bool(quant_ok),
        "frac_invalida": float(np.mean(~ok)),
        # Bate o (H,W) do array contra o que a amostra declara. Não levanta, para
        # não quebrar um chamador que use outra convenção de L sob crop, mas
        # deixa o descasamento auditável em vez de silencioso.
        "shape_bate_com_amostra": bool(
            (W == int(amostra.largura_px)) and (H == int(amostra.altura_px))
        ),
    }
    return PilhaGeo(
        canais=canais,
        segunda_ordem_valida=segunda_ok,
        diagnostico=diagnostico,
    )


# =============================================================================
# DECISÕES QUE O CONTRATO NÃO FIXA (registradas aqui, não num README)
# =============================================================================
#
# 1. `L` do gradiente normalizado = min(largura_px, altura_px), o MESMO para os
#    dois eixos. Ver `comprimento_de_normalizacao`. Alternativa rejeitada: L_x =
#    largura, L_y = altura, que daria a uma imagem 16:9 anisotropia artificial de
#    fator 1,78 nos canais que medem anisotropia.
#
# 2. Ponto principal (cx, cy) = centro do array recebido. `AmostraGeo` não traz
#    cx/cy e o `meta/` do release não os grava. Sob crop o ponto principal
#    verdadeiro sai do centro; o erro resultante em K é de segunda ordem no
#    deslocamento e, sem o dado, a alternativa seria inventá-lo. Quando o
#    `meta/` passar a trazer o ponto principal, é aqui que ele entra.
#
# 3. `fy = fx`. `AmostraGeo` tem um único `focallength_px`. Pixels não quadrados
#    não aparecem no release.
#
# 4. Os níveis de quantização são contados no mapa inteiro, sem a noção de
#    "região útil" da `geo_cond_v1`. Ver `niveis_quantizacao`: na codificação em
#    disparidade as duas contagens convergem (387 → 64.440 níveis na mesma cena),
#    então manter duas seria manter um número que não discrimina mais nada.
