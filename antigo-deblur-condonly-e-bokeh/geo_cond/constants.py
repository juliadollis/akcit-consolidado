"""Constantes de normalização dos canais geométricos.

TODAS as constantes que normalizam um canal vivem aqui, são fixas entre imagens,
e são calibradas UMA vez sobre o conjunto de treino. Nenhuma delas pode ser
computada por imagem.

O motivo é o mesmo que `genfocus_train/data.py:216-219` já registra para o `s1`:

    "NAO renormalizamos após o crop: s1 está na escala da imagem INTEIRA, então
     renormalizar pelo min/max do recorte deslocaria o plano de foco."

Sob crop aleatório um percentil calculado no recorte não bate com o da imagem
inteira, e treino e inferência veriam normalizações diferentes. Pior: numa cena
sem descontinuidade real o percentil é ruído, e o mapa de oclusão vira ruído
saturado de quadro cheio.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict


@dataclass(frozen=True)
class GeoConstants:
    """Constantes fixas de normalização. Nenhuma tem default: são medidas.

    Todos os canais saem em [0, 1], porque a condição entra no VAE em [0,1] CRU
    (`No_preprocess=True`), a mesma convenção do mapa de defocus. Ver o contrato
    em `genfocus_train/models.py:136-152`.
    """

    tau_occlusion: float
    """Divisor do mapa de oclusão: O = min(||grad f|| / tau, 1).
    Unidade: a de ||grad f|| em coordenadas de imagem NORMALIZADAS, d/d(x/W).
    NAO é percentil por imagem."""

    u_max: float
    """Teto da profundidade inversa, em 1/m. u_n = clip(u / u_max, 0, 1)."""

    s_max: float
    """Teto do elemento de área, s = log(sqrt(det g)) >= 0.
    s_n = clip(s / s_max, 0, 1)."""

    k0_curvature: float
    """Escala da compressão logarítmica da curvatura, em 1/m^2:
    K~ = sgn(K) * log(1 + |K| / k0)."""

    kt_max: float
    """Teto simétrico do K~ comprimido, para levar a [0,1]:
    K~_n = clip(K~, -kt_max, +kt_max) / (2*kt_max) + 0.5."""

    z_percentile_max: float
    """Percentil que substitui o max ao reconstruir a faixa métrica.

    Existe porque `z_max_m == 10000.0` exato, o teto do Depth Pro, em 2.988 das
    11.635 amostras da rota b (25,7%). Usar o max cru nessas amostras espalha a
    faixa útil por 4 ordens de grandeza e destrói a quantização.
    Ver AUDITORIA_DADOS_ROTAS_BC.md, seção 5."""

    smooth_sigma: float
    """Sigma da suavização gaussiana aplicada ANTES das segundas derivadas.

    O raio é derivado como round(3*sigma), NAO fixo. O `riemann/losses.py:86`
    fixa radius=2 independentemente de sigma, o que com sigma=2.0 dá um kernel de
    5 taps cobrindo +-1 sigma, ou seja quase uma caixa truncada e não uma
    gaussiana. O `riemann/geometry.py:87` faz certo, e é o que replicamos."""

    min_quant_levels: int
    """Mínimo de níveis uint16 que a cena útil precisa cobrir para que os canais
    de SEGUNDA ORDEM sejam considerados válidos.

    Medido: em 2.870 das 11.635 amostras (24,7%) a cena útil cabe em menos de 256
    níveis, e no subgrupo com z_max >= 1000 m a mediana é 23 níveis. Derivada
    segunda de um campo com 23 degraus é ruído de quantização, não geometria."""

    def as_dict(self) -> dict:
        return asdict(self)


# Marcador explícito de "ainda não calibrado".
#
# NAO substitua por números plausíveis. O ponto do módulo é que estas constantes
# venham de uma calibração sobre o conjunto de treino, sejam gravadas no config e
# fiquem registradas junto do run. Um default silencioso reintroduz exatamente a
# classe de defeito que a auditoria encontrou.
NAO_CALIBRADO: "GeoConstants | None" = None
