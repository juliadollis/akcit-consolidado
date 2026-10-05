"""A ponte entre o dataloader do treino e os sinais geométricos.

Mora aqui, e não em `genfocus_train/data.py`, por um motivo operacional: assim
ela é testável sem rede, sem GPU e sem `torch`. O gancho no `data.py` chama
`stack_para_amostra` e mais nada.

O QUE ESTE MÓDULO FAZ, E O QUE ELE DELIBERADAMENTE NÃO FAZ
----------------------------------------------------------
Ele faz três coisas: confere que os escalares da amostra descrevem o ARRAY que
recebeu, chama `signals.geometric_stack`, e aplica a GEOMETRIA do dataloader
(recorte e espelhamento) ao resultado.

Ele NÃO reconstrói escala métrica a partir de tabela externa nenhuma. Essa era a
máquina da campanha antiga (`geo_escalares`, casada por `stem`, com os jobs f0b
e f0e por trás). No contrato `metric_disparity_official_v1` a escala vem NATIVA
em cada `meta/<cena>/<id>.json` — `disparity_min`, `disparity_max` — e carregar a
tabela por inércia só reintroduziria um ponto de divergência silenciosa entre o
que o release diz e o que o treino assume.

A PARTE DELICADA É A GEOMETRIA, NÃO O SINAL
-------------------------------------------
`prepare_aligned_bokeh_metric` faz três transformações, nesta ordem:

  1. RESIZE para a grade (new_w, new_h).     fx' = fx * escala
  2. CROP de S x S na caixa (x0,y0,x1,y1).   cx' = cx*escala - x0
  3. HFLIP com probabilidade 1/2.            cx'' = S-1-cx'  E  n_x troca de SINAL

O item 3 é o que passa despercebido, e é o motivo do teste dedicado:

    ESPELHAR AS NORMAIS NÃO É ESPELHAR O ARRAY.

As normais são um CAMPO VETORIAL. Sob espelhamento horizontal a componente x
aponta para o outro lado; o array espelhado sozinho preserva o valor errado. Como
o canal é gravado em [0,1] por `(n_x + 1)/2`, inverter o sinal de `n_x` é o mesmo
que fazer `1 - canal`. Logo o correto é

    canal_espelhado = 1 - flip_horizontal(canal)     (só para n_x)

e não apenas `flip_horizontal(canal)`. Errar isso não levanta exceção nenhuma:
ensina a rede a associar "inclinação para a direita" com "inclinação para a
esquerda" em metade das amostras, e o efeito é a rede aprender a IGNORAR a
direção — exatamente o canal que G1 existe para carregar.

`n_y` NÃO troca de sinal: o espelhamento é horizontal. `s`, `u`, `O` e `K~` são
escalares invariantes por reflexão (magnitudes e curvatura gaussiana), então para
eles espelhar o array basta.

ESTRATÉGIA: CALCULAR NO MAPA INTEIRO, RECORTAR DEPOIS
-----------------------------------------------------
Escolha declarada, porque as duas ordens NÃO dão o mesmo resultado na borda do
recorte e a escolha tem de ser a mesma no treino e na inferência. Calculamos a
pilha no mapa REDIMENSIONADO E INTEIRO e só então recortamos. Três razões, em
ordem de peso:

  1. **Artefato de borda.** As derivadas (gradiente, e sobretudo a suavização
     gaussiana + segundas derivadas de `K~`) usam padding replicado. Calcular
     DENTRO do recorte poria uma borda falsa nos quatro lados de TODA amostra, a
     cada época num lugar diferente, porque o crop é aleatório. Calculando antes,
     a única borda falsa é a borda real da imagem — e essa a inferência também
     tem.

  2. **Paridade treino/inferência.** A inferência oficial não recorta: processa a
     imagem inteira (o tiling é do VAE, depois). Calcular no mapa inteiro é
     literalmente o que a inferência faz; calcular no recorte seria um regime que
     só existe no treino.

  3. **Contabilidade dos intrínsecos.** Com o cálculo no mapa inteiro, `cx`/`cy`
     são o centro geométrico do array e o `x0,y0` do crop não entra na conta. Na
     ordem inversa o principal point teria de ser transladado pelo crop, uma
     conta a mais para divergir entre o treino e a inferência.

O custo é calcular em ~W'xH' em vez de SxS. É desprezível ao lado do VAE.
"""

from __future__ import annotations

import numpy as np

from .contrato import CANAIS, AmostraGeo, PilhaGeo

__all__ = ["IDX_NX", "espelhar_pilha", "stack_para_amostra"]

#: Índice do canal `n_x` em `CANAIS`. É o ÚNICO canal que troca de sinal sob
#: espelhamento horizontal — ver a nota do módulo. Derivado de `CANAIS` e não
#: escrito como literal de propósito: reordenar `CANAIS` sem mexer aqui seria
#: exatamente a classe de defeito que `contrato.py` existe para impedir.
IDX_NX = CANAIS.index("n_x")


def espelhar_pilha(canais: np.ndarray) -> np.ndarray:
    """Espelha a pilha horizontalmente tratando `n_x` como campo vetorial.

    Todos os canais são espelhados no array; além disso `n_x` tem o SINAL
    invertido, o que na codificação `(n_x+1)/2 ∈ [0,1]` significa `1 - valor`.

    Propriedade que o teste dedicado tranca: aplicar esta função duas vezes é a
    identidade (`1 - (1 - v) = v` e `flip(flip(a)) = a`).
    """
    if canais.ndim != 3 or canais.shape[0] != len(CANAIS):
        raise ValueError(
            f"pilha deve ser ({len(CANAIS)}, H, W) na ordem {CANAIS}; "
            f"recebido {tuple(canais.shape)}"
        )
    out = np.ascontiguousarray(canais[:, :, ::-1])
    # `1 - v` e não `-v`: o canal está na codificação [0,1], não em [-1,1].
    out[IDX_NX] = 1.0 - out[IDX_NX]
    return out


def stack_para_amostra(
    disp01_completo: np.ndarray,
    caixa: tuple[int, int, int, int],
    flip: bool,
    amostra: AmostraGeo,
    consts,
    *,
    field: str = "inverse",
) -> PilhaGeo:
    """Pilha (6, S, S) em [0,1] para UMA amostra, já recortada e espelhada.

    Parameters
    ----------
    disp01_completo
        (H', W') em [0,1], a DISPARIDADE normalizada do PNG do release **depois
        do resize e ANTES do crop** — a mesma grade e a mesma reamostragem
        (NEAREST) que o mapa de defocus usou. A escala métrica sai de
        `amostra.disparity_min/max`; este array é a parte adimensional.

        NÃO é profundidade. No contrato `metric_disparity_official_v1` o PNG é
        `uint16` linear em disparidade (`depth_encoding:
        "uint16_linear_in_disparity"`), e tratar esses bits como profundidade
        inverteria a ordenação de todo o canal `u`.
    caixa
        `(x0, y0, x1, y1)`, a MESMA que `prepare_aligned_bokeh_metric` aplicou.
    flip
        Se o hflip foi aplicado às imagens. Ver a nota do módulo: aqui ele NÃO é
        um `[:, ::-1]`.
    amostra
        Escalares da amostra, lidos do `meta/`. `largura_px`/`altura_px` têm de
        ser os do ARRAY `disp01_completo`, e isso é CONFERIDO abaixo.
    consts
        `geocond.constants.GeoConstants`. Tipado solto de propósito: este módulo
        não precisa importar `constants` para repassar um objeto, e não importar
        mantém a ponte utilizável (e testável) enquanto o pacote é escrito.
    field
        "inverse" (default, por física) ou "depth". Repassado a `signals`.

    Returns
    -------
    PilhaGeo
        `canais` (6, S, S) float32 em [0,1]; `segunda_ordem_valida` e
        `diagnostico` propagados do cálculo no mapa inteiro, com as chaves do
        RECORTE acrescentadas.
    """
    disp01 = np.asarray(disp01_completo)
    if disp01.ndim != 2:
        raise ValueError(
            f"disp01_completo deve ser (H, W); recebido {tuple(disp01.shape)}"
        )
    altura, largura = int(disp01.shape[0]), int(disp01.shape[1])

    # ── A conferência que protege a invariância a resolução ──────────────────
    # `signals` normaliza o gradiente por `px_per_unit`, derivado DAS DIMENSÕES
    # DO ARRAY, e retroprojeta `K~` com uma focal expressa NA RESOLUÇÃO DO ARRAY.
    # Se `amostra.largura_px`/`altura_px` forem as da imagem ORIGINAL (o erro
    # natural, porque é o que o `meta/` traz em `image_w`/`image_h`), a
    # normalização passa a ter um fator de escala por amostra e a invariância a
    # redimensionamento — a razão de ser da convenção 2 do contrato — se desfaz
    # em silêncio. Por isso é erro, e não ajuste.
    if (amostra.altura_px, amostra.largura_px) != (altura, largura):
        raise ValueError(
            "AmostraGeo descreve um array diferente do recebido: "
            f"amostra=({amostra.altura_px}, {amostra.largura_px}) vs "
            f"disp01_completo={(altura, largura)}. `largura_px`/`altura_px` são "
            "as do ARRAY passado a signals (depois do resize), NÃO as de "
            "`image_w`/`image_h` do meta — e a focal, quando existir, tem de "
            "estar na mesma resolução."
        )

    x0, y0, x1, y1 = (int(v) for v in caixa)
    if not (0 <= x0 < x1 <= largura and 0 <= y0 < y1 <= altura):
        raise ValueError(
            f"caixa {(x0, y0, x1, y1)} fora do array {(altura, largura)}"
        )

    from .signals import pilha_geometrica  # import tardio: ver nota de `consts`

    # A `AmostraGeo` viaja INTEIRA para `signals`, que deriva dela os três
    # intrínsecos de que precisa — e essa centralização é deliberada, porque as
    # três convenções são exatamente as que, se divergirem entre os dois
    # módulos, quebram em silêncio:
    #
    #   L  = min(largura_px, altura_px)   o `L` de d/d(x/L) da convenção 2.
    #        `min` e não `max` porque é o lado que o resize preserva, e é sob
    #        essa convenção que `tau_occlusion` e `s_max` serão calibrados.
    #        O MESMO L nos dois eixos: largura em x e altura em y daria a uma
    #        imagem 16:9 anisotropia artificial de fator 1,78 justamente nos
    #        canais que medem anisotropia.
    #   cx,cy = centro geométrico do array. O release não publica principal
    #        point, e o centro é a convenção do próprio Depth Pro, que gerou a
    #        profundidade. Inventar outro seria inventar um intrínseco.
    #   fx = `amostra.focallength_px`, que o `meta/` NÃO traz — conferido campo
    #        a campo no release da rota A. Sendo `None`, `signals` devolve `K~`
    #        NEUTRO (0,5) e `segunda_ordem_valida=False`. Não é fallback
    #        numérico: é a ausência sendo propagada como ausência.
    #
    # Duplicar esses cálculos aqui foi o que a primeira versão deste arquivo
    # fez, e é como duas cópias da mesma regra começam a divergir. Há teste de
    # que `signals.comprimento_de_normalizacao(amostra)` vale `min(H, W)` do
    # array — se a convenção mudar lá, o teste quebra aqui.
    pilha = pilha_geometrica(disp01, amostra, consts, field=field)

    recorte = np.asarray(pilha.canais)[:, y0:y1, x0:x1]
    if flip:
        recorte = espelhar_pilha(recorte)
    recorte = np.ascontiguousarray(recorte, dtype=np.float32)

    # `diagnostico` do cálculo no mapa INTEIRO + o que descreve o RECORTE.
    # Os dois são necessários e não são o mesmo número: `niveis_quantizacao` é
    # propriedade do PNG de origem, enquanto `max_abs_por_canal_recorte` descreve
    # o que o modelo de fato vê. E é o MÁXIMO ABSOLUTO ENTRE AMOSTRAS o teste que
    # discrimina normalização por imagem — correlação não discrimina, porque é
    # invariante a escala, e foi assim que o mapa de defocus quebrado passou.
    diagnostico = dict(pilha.diagnostico or {})
    diagnostico.update(
        {
            "recorte_caixa": (x0, y0, x1, y1),
            "recorte_flip": bool(flip),
            "max_abs_por_canal_recorte": [
                float(np.abs(recorte[i]).max()) for i in range(recorte.shape[0])
            ],
        }
    )
    return PilhaGeo(
        canais=recorte,
        segunda_ordem_valida=bool(pilha.segunda_ordem_valida),
        diagnostico=diagnostico,
    )
