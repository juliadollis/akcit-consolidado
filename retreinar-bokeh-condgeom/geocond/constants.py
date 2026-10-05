"""Constantes de normalização dos canais geométricos.

TODAS as constantes que normalizam um canal vivem aqui, são fixas entre imagens,
e são calibradas UMA vez sobre o conjunto de treino. Nenhuma delas pode ser
computada por imagem — é a convenção nº 3 de `contrato.py`.

O motivo é o mesmo que `genfocus_train/data.py` já registra para o `s1`:

    "NAO renormalizamos após o crop: s1 está na escala da imagem INTEIRA, então
     renormalizar pelo min/max do recorte deslocaria o plano de foco."

Sob crop aleatório um percentil calculado no recorte não bate com o da imagem
inteira, e treino e inferência veriam normalizações diferentes. Pior: numa cena
sem descontinuidade real o percentil é ruído, e o mapa de oclusão vira ruído
saturado de quadro cheio.

O QUE SUMIU EM RELAÇÃO À `geo_cond_v1`, E POR QUÊ
-------------------------------------------------
Dois campos da versão antiga NÃO voltam, e a ausência é informação:

* `z_percentile_max` — existia porque `z_max_m == 10000.0` exato (o teto do
  Depth Pro) aparecia em 2.988 de 11.635 amostras (25,7%), e usar o max cru
  espalhava a faixa útil por 4 ordens de grandeza. No release novo a faixa vem
  como `disparity_min`/`disparity_max` POR AMOSTRA no `meta/`, e o céu fica no
  extremo COMPRIMIDO da disparidade em vez de no extremo esticado da
  profundidade. Não há o que percentilar.

* `min_quant_levels` vira `min_niveis_para_segunda_ordem`, com o mesmo papel mas
  outro número esperado. Na codificação antiga (uint16 linear em Z, 1 m a
  10.000 m) a cena útil até 60 m recebia 65.535·59/9.999 ≈ 387 níveis. Na
  codificação nova (uint16 linear em disparidade, 1/10.000 a 1/1) a MESMA cena
  recebe 65.535·(1 − 1/60)/(1 − 1/10.000) ≈ 64.440 níveis. A quantização passou
  a cair onde o CoC é insensível, que é exatamente o que se queria — logo o gate
  de segunda ordem dispara MUITO menos, e um limiar herdado da campanha antiga
  estaria calibrado para o problema errado.

Nenhuma constante tem default. Um default silencioso reintroduz exatamente a
classe de defeito que a auditoria encontrou: um número plausível, nunca medido,
que atravessa o treino inteiro sem levantar exceção.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, fields

__all__ = ["GeoConstants", "NAO_CALIBRADO"]


@dataclass(frozen=True)
class GeoConstants:
    """Constantes fixas de normalização. Nenhuma tem default: são MEDIDAS.

    Todos os canais saem em [0, 1], porque a condição entra no VAE em [0,1] CRU
    (`No_preprocess=True`), a mesma convenção do mapa de defocus. Ver a convenção
    nº 1 de `contrato.py`.
    """

    tau_occlusion: float
    """Divisor do mapa de oclusão: O = min(‖∇f‖ / tau, 1).

    Unidade: a de ‖∇f‖ em coordenadas de imagem NORMALIZADAS, d/d(x/L), com o
    campo `f` em 1/m quando `field="inverse"`. NÃO é percentil por imagem, e o
    valor da campanha antiga (`tau_occlusion = 88,170`) NÃO serve: ele foi
    calibrado sobre ‖∇(1/Z)‖ com Z reconstruído de profundidade normalizada por
    imagem, distribuição que o release novo não tem."""

    u_max: float
    """Teto da disparidade métrica, em 1/m: u = clip(disp / u_max, 0, 1).

    É o teto de 1/Z — ou seja, o inverso da menor profundidade que se admite na
    cena. Um `u_max = 2,0` significa "nada mais perto que 50 cm satura o canal"."""

    s_max: float
    """Teto do elemento de área, s = log(sqrt(det g)) ≥ 0.
    s_n = clip(s / s_max, 0, 1). Como det g = 1 + ‖∇f‖², `s_max` e
    `tau_occlusion` medem a MESMA grandeza em escalas diferentes (log contra
    linear); não são independentes, e calibrá-las em jobs separados sem conferir
    a coerência já foi fonte de confusão uma vez."""

    k0_curvature: float
    """Escala da compressão logarítmica da curvatura, em 1/m²:
    K~ = sgn(K) · log(1 + |K| / k0). Abaixo de `k0` a compressão é praticamente
    linear; acima, logarítmica. É o joelho da curva, e fixa qual raio de
    curvatura ocupa a metade superior do canal: k0 = 1/R² ⇒ R = 1/sqrt(k0)."""

    kt_max: float
    """Teto simétrico do K~ comprimido, para levar a [0,1]:
    K~_n = clip(K~, -kt_max, +kt_max) / (2·kt_max) + 0,5.
    O ponto neutro (K = 0) cai exatamente em 0,5, que é o valor que o canal
    assume quando a segunda ordem é invalidada."""

    smooth_sigma: float
    """Sigma da suavização gaussiana aplicada ANTES das segundas derivadas.

    O raio é derivado como round(3·sigma), NÃO fixo. O `riemann/losses.py:86`
    fixava radius=2 independentemente de sigma, o que com sigma=2,0 dá um kernel
    de 5 taps cobrindo ±1σ — uma caixa truncada, não uma gaussiana. O
    `riemann/geometry.py:87` faz certo, e é o que se replica aqui."""

    min_niveis_para_segunda_ordem: int
    """Mínimo de níveis uint16 distintos que o mapa precisa ocupar para que o
    canal de SEGUNDA ORDEM (`K~`) seja considerado válido.

    Derivada segunda de um campo com poucas dezenas de degraus é ruído de
    quantização, não geometria. Ver a nota no topo do módulo sobre por que o
    limiar da campanha antiga (256, sobre uint16 linear em Z) não transfere: na
    codificação em disparidade a mesma cena que dava ~387 níveis úteis passa a
    dar ~64.440."""

    # -------------------------------------------------------------- ida e volta

    def to_dict(self) -> dict:
        """Dicionário puro, para gravar no YAML do run junto dos pesos.

        A constante tem de viajar COM o checkpoint: um canal normalizado por
        `u_max = 2,0` no treino e por outro valor na inferência é a mesma classe
        de defeito que invalidou a campanha anterior."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "GeoConstants":
        """Reconstrói a partir do YAML, ESTRITO nos dois sentidos.

        Campo faltando é erro, e campo A MAIS também. O segundo caso não é
        pedantismo: um YAML da campanha antiga traz `z_percentile_max` e
        `min_quant_levels`, que aqui não existem mais. Aceitá-lo em silêncio
        (ignorando o extra e usando o que sobra) carregaria constantes
        calibradas na distribuição errada para dentro do treino novo — que é
        exatamente o que o `referencia_antiga/LEIA.md` proíbe.
        """
        esperados = {f.name for f in fields(cls)}
        recebidos = set(d)
        faltando = esperados - recebidos
        sobrando = recebidos - esperados
        if faltando or sobrando:
            raise ValueError(
                "GeoConstants.from_dict: chaves não batem com o contrato. "
                f"faltando={sorted(faltando)} sobrando={sorted(sobrando)}. "
                "Se 'sobrando' menciona z_percentile_max ou min_quant_levels, "
                "este YAML é da campanha geo_cond_v1 e está invalidado "
                "(ver referencia_antiga/LEIA.md)."
            )
        return cls(
            tau_occlusion=float(d["tau_occlusion"]),
            u_max=float(d["u_max"]),
            s_max=float(d["s_max"]),
            k0_curvature=float(d["k0_curvature"]),
            kt_max=float(d["kt_max"]),
            smooth_sigma=float(d["smooth_sigma"]),
            min_niveis_para_segunda_ordem=int(d["min_niveis_para_segunda_ordem"]),
        )


#: Marcador explícito de "ainda não calibrado".
#:
#: NÃO substitua por números plausíveis. O ponto do módulo é que estas
#: constantes venham de uma calibração sobre o conjunto de treino, sejam
#: gravadas no config e fiquem registradas junto do run.
NAO_CALIBRADO: "GeoConstants | None" = None
