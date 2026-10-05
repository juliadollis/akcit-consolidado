"""O contrato do `geocond` — assinaturas e convenções, num lugar só.

Este arquivo existe para que os módulos do pacote e a fiação no `genfocus_train`
sejam escritos contra a MESMA definição. Ele não faz conta: só declara os tipos,
a ordem dos canais e as três convenções que, se divergirem, quebram em silêncio.

    A DIFERENÇA EM RELAÇÃO À CAMPANHA ANTIGA, EM UMA LINHA

    O PNG do release NÃO é profundidade: é DISPARIDADE, `uint16` linear entre
    `disparity_min` e `disparity_max` do `meta/` (`depth_encoding:
    uint16_linear_in_disparity`). A versão antiga recebia profundidade métrica
    normalizada e reconstruía a escala por uma tabela externa. Aqui a escala
    vem na própria amostra, e o canal `u` é o array lido, reescalado — não uma
    inversão.

TRÊS CONVENÇÕES, E O QUE CADA UMA PROTEGE
-----------------------------------------

1. **Saída em [0,1] CRU.** Todo canal sai em [0,1] e entra no VAE sem o passo
   `[0,1] -> [-1,1]`, porque a inferência oficial usa `No_preprocess=True` para
   o mapa de defocus. Treinar num range e inferir noutro foi o defeito nº 1
   deste projeto.

2. **Gradiente em coordenadas de imagem NORMALIZADAS**, `d/d(x/L)`, com `L` em
   pixels passado pelo chamador. Torna `O`, `s`, `n_x`, `n_y` invariantes a
   redimensionamento — o treino roda a 512² e a inferência usa a resolução
   original com tiling.

3. **Constantes de normalização FIXAS entre imagens.** Nenhuma é percentil por
   imagem. Sob crop aleatório o percentil do recorte não bate com o da imagem
   inteira, e treino e inferência veriam normalizações diferentes.

O QUE É FÍSICA, E NÃO ESTILO
----------------------------
A grandeza que governa a anisotropia do núcleo é `‖∇u‖`, não `‖∇D‖`:

    raio do CoC  ∝  |u - u_foco|          (linear na disparidade)
    eps = gamma·‖∇D‖ / (Z²c)  e  ∇u = -∇D/Z²   ⇒   eps ∝ ‖∇u‖

Com `‖∇D‖` o fundo distante domina o sinal, que é justamente onde o borrão é
mais uniforme: um degrau de 1 m para 20 m dá 19 em `D` e 0,95 em `u`; um de
20 m para 40 m dá 20 em `D`, MAIOR, e 0,025 em `u`. Só em `u` a ordenação é a
opticamente correta. `field="depth"` existe para a diferença ser ablacionável,
não porque seja defensável.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "CANAIS",
    "CONTRATO_DE_CONTROLE",
    "AmostraGeo",
    "PilhaGeo",
    "G1_IDX",
    "G2_IDX",
]

#: O `control_version` do release contra o qual este pacote foi escrito. A
#: fiação confere: um release de outra convenção tem que falhar alto, não
#: produzir canais plausíveis.
CONTRATO_DE_CONTROLE = "metric_disparity_official_v1"

#: Ordem dos canais na pilha. É parte do contrato do modelo: reordenar sem
#: retreinar troca o significado de um canal sem levantar exceção.
CANAIS = ("u", "O", "s", "n_x", "n_y", "K~")

#: Agrupamento em dois branches de condição. NÃO é livre: o `group_mask` faz
#: cada condição atender só a si mesma, ao texto e ao branch principal, então
#: os dois branches geométricos são MUTUAMENTE CEGOS. O que precisa ser lido
#: junto fica junto — G1 é o termo de primeira ordem COMPLETO (a decomposição
#: polar de ∇u); G2 é escala, visibilidade e segunda ordem. A combinação entre
#: os dois acontece no branch principal, que enxerga tudo.
G1_IDX = (2, 3, 4)   # s, n_x, n_y
G2_IDX = (0, 1, 5)   # u, O, K~


@dataclass(frozen=True)
class AmostraGeo:
    """Os escalares de UMA amostra. Todos saem do `meta/<cena>/<id>.json`.

    Nenhum deles é inventável. Sem eles não há escala absoluta, e sem escala
    absoluta `s`, `n` e `K~` não significam nada — não são invariantes a
    reescala do eixo z.
    """

    disparity_min: float
    """1/m. `meta["disparity_min"]`. Junto com o max, reconstrói a disparidade
    métrica a partir do PNG: `disp = disparity_min + d01*(max-min)`."""

    disparity_max: float
    """1/m. `meta["disparity_max"]`."""

    largura_px: int
    """Largura do ARRAY de profundidade recebido, em pixels. Depois do resize,
    é a do array redimensionado — o gradiente normalizado usa `L` para voltar a
    d/d(x/L), e usar a largura errada desfaz a invariância que ele protege."""

    altura_px: int
    """Altura do array recebido, em pixels."""

    focallength_px: float | None = None
    """Focal em pixels, NA RESOLUÇÃO DO ARRAY RECEBIDO.

    O `meta/` do release **não traz este campo** — conferido campo a campo no
    release da rota A. Quando é `None`, o canal `K~` sai no valor NEUTRO (0,5,
    que é K=0 depois da normalização simétrica) e `PilhaGeo.segunda_ordem_valida`
    sai `False`. Isso é deliberado: a curvatura retroprojetada exige `fx`, e
    inventar um produziria um canal plausível e errado.
    """


@dataclass(frozen=True)
class PilhaGeo:
    """Saída do cálculo dos canais."""

    canais: "object"
    """(6, H, W) float32, cada canal em [0,1], na ordem de `CANAIS`."""

    segunda_ordem_valida: bool
    """False quando `focallength_px` falta ou a quantização não sustenta
    derivada segunda. Nesses casos `K~` sai neutro (0,5) em vez de ruído.
    O chamador deve propagar a flag, não ignorá-la."""

    diagnostico: dict
    """Números para auditoria, nunca para decisão dentro do treino:
    `niveis_quantizacao`, `frac_O_saturado`, `max_abs_por_canal`.

    O `max_abs_por_canal` existe por um motivo específico: correlação NÃO
    detecta normalização por imagem, porque é invariante a escala — foi assim
    que o mapa de defocus quebrado passou despercebido. O teste que discrimina
    é comparar o MÁXIMO ABSOLUTO ENTRE AMOSTRAS."""
