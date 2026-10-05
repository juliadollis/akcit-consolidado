#!/usr/bin/env python3
"""Valida um release e publica no Hugging Face.

    # 1. conferir, sem subir nada (é o default)
    python3 scripts/publish_release.py --release-dir output/c_realbokeh

    # 2. subir, depois de ler o relatório
    python3 scripts/publish_release.py --release-dir output/c_realbokeh \\
        --repo-id akcit-pixel/bokehnet-regen-c --yes

## A regra desta ferramenta

**Valida sempre, sobe só com `--yes`.** O upload é irreversível na prática — um
dataset publicado com rótulo errado vira o dataset que alguém treina em cima seis meses
depois, exatamente o que aconteceu com o release anterior. Então a checagem roda
primeiro, imprime, e só então pergunta.

Repositório **privado por default**, e **nunca sobrescreve** um repo que já tem
arquivos: `--allow-existing` é obrigatório para isso, e mesmo assim a ferramenta lista
o que já está lá antes.

## O que a validação exige

Um release publicável tem que responder três perguntas sem depender de ninguém:

1. **O que cada amostra afirma** — `meta/<id>.json` passa em `validate_metadata`, e
   `control_version` é o mesmo em todas.
2. **De onde vieram os pixels** — toda amostra da rota C tem linha no
   `source_images.jsonl` com sha256 da AIF e da bokeh. Sem isso o release é rótulo
   solto apontando para um espelho privado, e ninguém consegue provar que o K foi
   calibrado contra aqueles bytes.
3. **Onde está a fronteira do split** — `split.json` existe, cobre todas as cenas, e
   nenhuma cena aparece dos dois lados (`check_no_leak`, comparando o split gravado em
   cada linha do manifesto contra o materializado).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dataio.layout import (                                      # noqa: E402
    DEPTH_DIR, DepthLayout, depth_layout_of_run_config, depth_ref, depth_ref_of,
)
from dataio.sample import validate_metadata                      # noqa: E402
from dataio.split import SceneSplit, check_no_leak               # noqa: E402
from qc.focus_region import FocusSource                          # noqa: E402


class Problema(Exception):
    """Impede a publicação. Sempre com o número que a torna verificável."""


# --------------------------------------------------------------------------------
# O que cada rota É — a única tabela da qual o resto deste arquivo depende
# --------------------------------------------------------------------------------
#
# As três rotas do §3.2 não são variações de tema: elas têm **fonte**, **equações**,
# **modelos** e **artefatos** diferentes. Antes desta tabela o card era uma f-string com
# o texto da rota C escrito à mão, e o resultado está medido: o README de
# `output/b_oficial_release/` — um release da rota B — diz `pretty_name: BokehNet regen
# — rota C` e descreve o sweep da Eq. 5, que a rota B não usa
# (`reference/AVALIACAO_DATASETS_BC.md:625-640`).
#
# A regra que esta tabela impõe: **o card só pode afirmar o que a rota do release de
# fato faz**, e o número vem sempre do manifesto ou da proveniência gravada. Campo que
# não existe para a rota **não vira "(não registrada)"** — a seção some.

#: As fontes de foco que são de fato o substituto automático do passo manual do
#: §3.2(c). **`sampled_plane` não está aqui**, e é esse o ponto: na rota A o plano de
#: foco é SORTEADO, não refinado.
#:
#: `FocusRegionRecord.was_refined` (`dataio/sample.py:228`) devolve `True` para tudo que
#: não é `BIREFNET`, então `focus_was_refined` sai `True` nas 4.578 amostras da rota A —
#: e `validate_metadata` EXIGE que ele saia assim (`dataio/sample.py:472-476`), de modo
#: que o metadado não pode ser corrigido sem mudar o contrato. Contar o refinamento pelo
#: booleano fazia o relatório dizer `refined 4853 de 4853` num lote em que nada foi
#: refinado, e o card mandava filtrar por um campo que não seleciona o que promete.
#: Aqui o relatório conta pela FONTE, que é o que descreve o que aconteceu, e denuncia a
#: divergência em vez de herdá-la.
FONTES_REFINADAS = frozenset({FocusSource.BIREFNET_REFINED.value,
                              FocusSource.RETENTION_ONLY.value})

#: Quantos sha256 de pixel gerado são recalculados a partir dos bytes em disco. Conferir
#: 70 mil JPEGs de ~800 KB é ~55 GB de leitura, o que transforma a validação de um
#: release em um job; conferir **zero** é o que o validador fazia até aqui. A amostragem
#: é determinística (ordenada pelo sha256 do próprio `sample_id`, não pela ordem do
#: manifesto) e o tamanho usado vai no relatório — um número de amostragem que não
#: aparece no relatório é indistinguível de nenhuma amostragem.
AMOSTRAGEM_SHA256_PIXEL = 256


@dataclass(frozen=True)
class Rota:
    """Uma rota do §3.2, do ponto de vista de quem valida e de quem publica.

    Os campos são de dois tipos: os de **contrato** (o que a validação exige) e os de
    **prosa** (o que o card pode afirmar). Os dois vivem juntos de propósito — foi a
    separação entre "o que o código confere" e "o que o texto diz" que deixou um release
    da rota B ser publicado com a página da rota C.
    """

    letra: str
    #: Como o card chama a rota, e o que entra no `pretty_name`.
    nome: str
    #: A subseção do paper que a define.
    secao: str

    # -- contrato de proveniência ------------------------------------------------
    #: Em que unidade **esta rota** grava `source_images.jsonl`.
    #:
    #: Não é detalhe de formato: na rota A a AIF de origem é **uma por cena** e as 41
    #: variantes saem dos mesmos bytes, então a linha é por `scene_id`
    #: (`sources/genphoto_ebb.py:565-569`) e gravá-la 41 vezes seria 41x de espaço para
    #: provar a mesma coisa. Na rota C a AIF e a bokeh são as duas de origem e as duas
    #: por amostra (`sources/mirror_images.py:255`, `sources/lfdof_images.py:353`).
    #: Ler as duas com a mesma chave é o `KeyError: 'sample_id'` que derrubava a rota A.
    unidade_do_ledger_de_origem: str
    #: `True` quando a ausência do ledger de origem reprova o release.
    #:
    #: A rota B referencia a foto de origem por `bokeh_ref` (uma linha de
    #: `atfortes/BokehDiffusion`) e o release publicado não tem `source_images.jsonl`;
    #: reprovar retroativamente por um arquivo que a rota não escrevia transformaria um
    #: aviso honesto em bloqueio falso. Fica como **aviso**, visível e contado.
    exige_ledger_de_origem: bool
    #: `(papel, campo_sha256)` do pixel que ESTA rota gera, ou `None`.
    #:
    #: `papel` é o sufixo do arquivo em `generated/<sample_id>_<papel>.jpg`.
    #: A rota A gera a **bokeh**, que é o ALVO do treino (`routes/route_a.py:216,1248`).
    #: A rota B gera a **AIF**, que é a ENTRADA (`routes/route_b.py:187,1036`) — o alvo
    #: dela é a própria fotografia real. A rota C não gera pixel nenhum: ela rotula
    #: pares que já existem (`routes/route_c.py:30-33`).
    #:
    #: Os dois primeiros casos são o que o §3 do docstring desta ferramenta pede e o que
    #: ela não conferia: um release da rota A com 10.000 bokehs faltando passava.
    pixel_gerado: Optional[tuple[str, str]]
    #: `True` quando o pixel gerado acima é o alvo que a rede aprende a produzir.
    pixel_gerado_e_o_alvo: bool
    #: `True` quando a rota passa pelo refinamento da região em foco do §3.2(c).
    refina_a_regiao_de_foco: bool
    #: `True` quando `K` vem de uma varredura com teto — só aí `is_k_censored` mede algo.
    calibra_k_por_varredura: bool
    #: `True` quando existe segundo cálculo independente de `K` (`k_analytic`).
    tem_validador_analitico: bool

    # -- prosa do card -----------------------------------------------------------
    #: O parágrafo de abertura. `{fontes}` é substituído pelas fontes MEDIDAS no
    #: manifesto do release.
    abertura: str
    #: O que a rede aprende a produzir, e onde esse pixel está.
    alvo: str
    #: O bloco do sinal de controle — as equações que ESTA rota aplica, e só elas.
    sinal_de_controle: str
    #: As linhas de "o que mudou em relação ao dataset anterior" desta rota.
    mudancas: tuple[str, ...]
    #: As limitações desta rota. Nenhuma frase sobre modelo que a rota não roda.
    limitacoes: tuple[str, ...]
    #: Seções `(título, corpo)` que só existem nesta rota. `{n}` e `{treino}` são
    #: substituídos pelos números MEDIDOS no release.
    secoes_extras: tuple[tuple[str, str], ...]
    #: Linhas da tabela de proveniência: `(rótulo, chave em provenance_base)`. Uma linha
    #: cujo valor não existe **some**, em vez de imprimir `?`.
    linhas_de_proveniencia: tuple[tuple[str, str], ...]


#: O sinal de controle, comum às três rotas. O que muda é **de onde sai `K`** e **de
#: onde sai `focus_disp`**, e é isso que cada `Rota.sinal_de_controle` acrescenta.
_DEFOCUS = """```python
z          = depth_pro(aif)              # METROS
disp       = 1.0 / z                     # 1/m
defocus    = clip(abs(K * (disp - focus_disp)) / 100.0, 0.0, 1.0)
```"""

_MUDANCA_MAX_COC = (
    "**O normalizador parou de variar.** `max_coc` era uma flag por rota; um "
    "normalizador que muda entre amostras é um segundo rótulo escondido dentro do "
    "primeiro. Agora é **100,0 fixo e global**, e a validação do release recusa "
    "qualquer amostra que diga outra coisa.")
_MUDANCA_SPLIT = (
    "**O split passou a ser por cena.** Com split por imagem, a mesma cena aparecia "
    "nos dois lados e a validação media memorização.")
_MUDANCA_CASCATA = (
    "**Acabaram as cascatas que mentiam na proveniência.** O pipeline antigo caía de "
    "Depth Pro para Depth Anything, que devolve **disparidade** e não profundidade "
    "métrica — a amostra saía espelhada sem deixar rastro. Aqui existe **um** modelo "
    "por papel; se ele falha, a amostra é rejeitada com motivo registrado.")

ROTAS: dict[str, Rota] = {
    "a": Rota(
        letra="a", nome="rota A", secao="§3.2(a)",
        unidade_do_ledger_de_origem="scene_id",
        exige_ledger_de_origem=True,
        pixel_gerado=("bokeh", "bokeh_jpeg_sha256"),
        pixel_gerado_e_o_alvo=True,
        refina_a_regiao_de_foco=False,
        calibra_k_por_varredura=False,
        tem_validador_analitico=False,
        abertura=(
            "pares **sintéticos**. A imagem all-in-focus é uma fotografia nítida real "
            "({fontes}), o plano de foco e o nível de bokeh `K` são **sorteados**, e o "
            "alvo é **renderizado** pelo BokehMe [43].\n\n"
            "O paper diz apenas *\"randomly sample a focus plane D_focus and a target "
            "bokeh level K\"* (`paper.txt:326-328`) e **cala sobre a distribuição**. "
            "Amostrar `K` da distribuição empírica das rotas B e C "
            "(`k_source = \"sampled_from_bc\"`) é **decisão nossa**, declarada, e "
            "existe para manter o sintético na mesma escala física do real."),
        alvo=(
            "**O alvo do treino está neste repositório, em `generated/`.** Ele é o "
            "único pixel desta rota que não é de origem: cada `generated/<id>_bokeh.jpg` "
            "foi produzido por nós, e `generated_images.jsonl` traz o `bokeh_jpeg_sha256` "
            "do **JPEG em disco** — não do array em memória, porque JPEG q95 é lossy e "
            "hashear o array provaria uma coisa enquanto o release contém outra.\n\n"
            "A AIF é **referência**: `source_images.jsonl` traz o sha256 dela por "
            "**cena**, e não por amostra, porque as variantes de uma cena saem dos "
            "mesmos bytes de AIF."),
        sinal_de_controle=(
            "Esta rota aplica a **Eq. 2 e só ela** (`paper.txt:312`). O plano de foco "
            "e o nível de bokeh são sorteados, então não há EXIF a ler (o alvo não foi "
            "fotografado), não há máscara a medir (`paper.txt:329-330`) e não há "
            "fotografia real contra a qual ajustar `K`. Nenhuma outra equação do §3.2 "
            "se aplica aqui.\n\n"
            + _DEFOCUS + "\n\n"
            "`focus_disp` é **sorteado** dentro da faixa de disparidade da própria cena, "
            "e `K` é sorteado da distribuição empírica de B e C por hipercubo latino 2D — "
            "um estrato de `K` e um de plano de foco por variante. `focus_source = "
            "\"sampled_plane\"` em toda amostra, e `provenance.extra.sampling` grava o "
            "estrato, a semente e o esquema da distribuição."),
        mudancas=(
            _MUDANCA_MAX_COC,
            "**`K` deixou de ser escolhido a olho.** Ele é sorteado da distribuição "
            "empírica medida nas rotas B e C, na grandeza `k_per_long_side`, e a "
            "distribuição usada está gravada por sha256 na proveniência do run.",
            _MUDANCA_CASCATA,
            _MUDANCA_SPLIT,
        ),
        limitacoes=(
            "**39,4% das amostras têm `K < 8`, e nelas o alvo é quase a própria "
            "AIF.** Medido sobre as 69.700 do release: K mínimo 0,22, p25 3,13, mediana "
            "12,49, p75 27,53, máximo 903,79. Nessas amostras o CoC renderizado é "
            "sub-pixel, então o par (entrada, alvo) é quase uma identidade e ensina "
            "pouco. É consequência DELIBERADA de gerar com todos os gates em modo "
            "medir (`--seed 0`, nenhum limiar congelado): nada foi descartado para que "
            "o limiar pudesse ser escolhido com o histograma na mão em vez de no "
            "escuro. **Nada se perde** — `k_value` está em cada `meta/`, o filtro é "
            "reversível e não exige regerar. Quem treinar deve decidir o corte "
            "explicitamente; treinar com as 69.700 sem filtrar dedica ~2 de cada 5 "
            "passos a pares quase-idênticos. A `GenerativePhotography` é mais afetada "
            "(46,6% com `K < 8`) que a `EBB!` (32,4%), porque tem metade da resolução "
            "(512x768 contra 1024x1558) e K escala com o lado longo.",
            "**2,2% das amostras têm `K > 96`, acima da faixa em que o renderizador "
            "foi VERIFICADO.** O `renderer_verification.json` mediu o BokehMe em "
            "K ∈ {8, 16, 32, 64, 96}; acima disso a linearidade do raio não foi "
            "conferida, e a p95 da saturação do mapa de defoco nessas amostras é 0,66 "
            "— dois terços dos pixels no teto de `max_coc`. Filtrável pelo mesmo "
            "`k_value`.",
            "**Viés de renderizador.** O alvo é renderizado, não fotografado, e o "
            "próprio paper registra que a rota (a) é limitada por isso "
            "(`paper.txt:333-334`). É a razão de o currículo do §4.1 treinar 40K steps "
            "aqui e 60K em dado real, e não o contrário.",
            "**`focus_agreement`, `focus_precision` e `focus_iou` são 1,0 POR "
            "CONSTRUÇÃO, não por medição.** A região é definida pelo plano de foco "
            "sorteado e comparada com ele mesmo. Lê-los como evidência de qualidade "
            "nesta rota é erro, e por isso a nota vai gravada em "
            "`provenance.extra.focus_region_semantics` de cada amostra.",
            "**`focus_retention_in_region` é `NaN`.** Não é medida ausente: não existe "
            "retenção a medir, porque a rota A tem UMA foto e não um par. `0,0` "
            "significaria \"reteve zero\", que é falso.",
            "**`NaN` aparece literal no JSON**, em `manifest.jsonl` e em todo `meta/`, "
            "por causa do campo acima. `NaN` **não é JSON válido** pela especificação: "
            "o `json` do Python e o `pyarrow` (logo, `datasets`) aceitam; "
            "`JSON.parse` do navegador e qualquer parser estrito recusam. Se o seu "
            "leitor for estrito, troque `NaN` por `null` antes de parsear — o valor "
            "significa \"não existe\", não \"zero\".",
        ),
        secoes_extras=(),
        linhas_de_proveniencia=(
            ("pipeline", "pipeline_commit"),
            ("profundidade", "depth_backend"),
            ("renderer", "renderer"),
            ("fator efetivo de K medido", "k_effective_factor"),
            ("distribuição de K", "k_distribution_sha256"),
            ("origem do split", "split_origin"),
        ),
    ),
    "b": Rota(
        letra="b", nome="rota B", secao="§3.2(b)",
        unidade_do_ledger_de_origem="sample_id",
        exige_ledger_de_origem=False,
        pixel_gerado=("aif", "aif_jpeg_sha256"),
        pixel_gerado_e_o_alvo=False,
        refina_a_regiao_de_foco=False,
        calibra_k_por_varredura=False,
        tem_validador_analitico=False,
        abertura=(
            "fotografia com bokeh **real** ({fontes}). A imagem all-in-focus não "
            "existe na origem: ela é **produzida pela DeblurNet**, o primeiro estágio do "
            "próprio GenRefocus. O `K` vem da **Eq. 3 aplicada à EXIF** da foto, e o "
            "alvo é a **própria fotografia**.\n\n"
            "**Não há renderizador nesta rota.** A tupla de supervisão é "
            "`(I_aif, I_out, D_def)` (`paper.txt:321`), a Fig. 3(b) alimenta a fileira "
            "\"Bokeh images\" com a foto real (`paper.txt:271, 283`), e o renderizador "
            "[43] aparece só nas outras duas subseções. Logo `renderer` é `null` em "
            "toda amostra — e não um dicionário afirmando `is_final_label_renderer`, "
            "que é o que o pipeline antigo gravava."),
        alvo=(
            "**O alvo é a foto de origem, e ela não está aqui.** Cada amostra aponta "
            "para a linha da origem por `source_sample_id` e `bokeh_ref`.\n\n"
            "O que está aqui, em `generated/`, é a **entrada**: `generated/<id>_aif.jpg` "
            "é a all-in-focus que a DeblurNet produziu, com `aif_jpeg_sha256` em "
            "`generated_images.jsonl` — do JPEG em disco, porque é o JPEG que o release "
            "contém. Confundir os dois papéis é o erro que este parágrafo existe para "
            "impedir: o que geramos aqui é a ENTRADA, e o alvo continua sendo a "
            "fotografia da origem."),
        sinal_de_controle=(
            "Esta rota aplica a **Eq. 3** (`K` da EXIF) e a **Eq. 4** (`D_focus` da "
            "máscara do **BiRefNet**), e alimenta a Eq. 2. **Não há varredura de `K`**: "
            "sem renderizador não há o que varrer, então `calibration_ssim` é `null` e "
            "`is_k_censored` é `false` em toda amostra — não porque nada foi censurado, "
            "mas porque não existe teto de busca a censurar.\n\n"
            + _DEFOCUS + "\n\n"
            "```python\n"
            "focus_disp = median(disp[mask])          # Eq. 4, mediana NA disparidade\n"
            "k_eq3      = (f_mm**2 * z_focus_mm) / (2 * F * (z_focus_mm - f_mm)) "
            "* pixel_ratio\n"
            "K          = k_eq3 / 1000.0              # a Eq. 3 é em mm; a inferência "
            "é em 1/m\n"
            "```\n\n"
            "`f`, `F` e `focal_length_35` saem da EXIF da própria foto. Faltou EXIF ou "
            "faltou a largura do sensor, a amostra é **rejeitada** com o motivo "
            "registrado — nunca substituída por constante (regra 4 do contrato). O "
            "`k_effective_factor` de 0,9873, medido no renderizador das outras rotas, "
            "**não** é aplicado aqui: ele descreve um renderizador que esta rota não "
            "usa."),
        mudancas=(
            "**A AIF deixou de sair lavada.** O pipeline antigo chamava `generate()` "
            "sem `main_adapter`, e a DeblurNet devolvia uma imagem sem o LoRA de "
            "deblur. Agora a tripla de pesos é amarrada por `DeblurVariant` e viaja com "
            "sha256 em cada amostra.",
            "**A Eq. 3 parou de errar por 1000x.** `k_eq3` sai em px·mm e a "
            "disparidade em 1/m; o pipeline antigo alimentava um com o outro. "
            "`pixel_ratio` também passou a usar `max(H, W)`, e não a largura "
            "(`paper.txt:1186-1187`).",
            _MUDANCA_MAX_COC,
            _MUDANCA_CASCATA,
            _MUDANCA_SPLIT,
        ),
        limitacoes=(
            "**A AIF é gerada, então ela tem os defeitos da DeblurNet.** Ela é a "
            "ENTRADA da rede de bokeh; um artefato dela vira condicionamento, não "
            "rótulo, mas vira.",
            "**A Eq. 3 depende da EXIF da origem**, que não foi verificada contra "
            "medida independente. O que se pode afirmar é que ela é interna e "
            "consistente: `K` correlaciona +0,781 com a focal, como `K ∝ f²` exige.",
            "**Sem gabarito para a Eq. 4.** Não há distância de foco medida no ITW, "
            "então o caso \"o fotógrafo focou o fundo e o segmentador achou o objeto "
            "saliente\" não é detectável por nenhuma métrica deste release.",
        ),
        secoes_extras=(),
        linhas_de_proveniencia=(
            ("pipeline", "pipeline_commit"),
            ("profundidade", "depth_backend"),
            ("máscara", "mask_backend"),
            ("origem do split", "split_origin"),
        ),
    ),
    "c": Rota(
        letra="c", nome="rota C", secao="§3.2(c)",
        unidade_do_ledger_de_origem="sample_id",
        exige_ledger_de_origem=True,
        pixel_gerado=None,
        pixel_gerado_e_o_alvo=False,
        refina_a_regiao_de_foco=True,
        calibra_k_por_varredura=True,
        tem_validador_analitico=True,
        abertura=(
            "pares **reais** (all-in-focus, bokeh) vindos de {fontes}, com o sinal de "
            "controle calibrado pelo sweep da **Eq. 5**.\n\n"
            "As duas imagens são fotografias da mesma cena: nada aqui é renderizado. O "
            "BokehMe entra **só dentro da varredura**, para achar o `K` cuja "
            "renderização mais se parece com a foto real — e o `K` é que vira rótulo, "
            "não a renderização."),
        alvo=(
            "**O alvo é a fotografia com bokeh da origem.** Esta rota não gera pixel "
            "nenhum: ela rotula pares que já existem. Regravá-los custaria 365 GB "
            "contra 115 GB de cota livre e criaria uma segunda cópia que pode divergir "
            "da primeira."),
        sinal_de_controle=(
            "Esta rota aplica a **Eq. 4** (`D_focus` da região refinada) e a **Eq. 5** "
            "(varredura de `K` por SSIM), e alimenta a Eq. 2. A **Eq. 3 não é o "
            "rótulo** aqui: ela entra como `k_analytic`, segundo cálculo independente, "
            "quando a largura do sensor é conhecida.\n\n"
            + _DEFOCUS + "\n\n"
            "```python\n"
            "focus_disp = median(disp[mask])          # Eq. 4, mediana NA disparidade\n"
            "K          = argmax_K SSIM(R(I_aif, D; focus_disp, K), I_real)   # Eq. 5\n"
            "```\n\n"
            "A profundidade é guardada como **disparidade uint16**, lado longo 768, com "
            "os extremos medidos na resolução cheia — `disparity_min`/`disparity_max` no "
            "metadado desfazem a quantização."),
        mudancas=(
            "**A imagem de referência agora é all-in-focus de verdade.** Antes, a "
            "\"AIF\" era escolhida como o maior f-stop dentro da pasta dos *alvos*: "
            "mediana **f/14**, e em 12,7% das cenas f/5.6 ou mais aberto. Ou seja, numa "
            "cena a cada oito, a profundidade, a máscara e o sweep de K saíam de uma "
            "foto que já tinha bokeh forte. A AIF verdadeira sempre existiu, em "
            "`train/in/<id>_f22.JPG` — f/22 em 100% das cenas, confirmada byte a byte "
            "por sha256.",
            "**O K deixou de bater no teto.** A busca antiga usava um limite escolhido "
            "a priori e produzia `k == 300` exato em **47,0%** das amostras: metade do "
            "dataset tinha como rótulo o teto da busca, não o K da cena. O teto agora "
            "vem medido de um piloto, não de intuição.",
            _MUDANCA_MAX_COC,
            "**Acabaram as cascatas que mentiam na proveniência.** O pipeline antigo "
            "caía de BiRefNet para RMBG para GrabCut com `except` nu, e continuava "
            "gravando `mask_source=\"automatic\"`; e de Depth Pro para Depth Anything, "
            "que devolve **disparidade** e não profundidade métrica. Aqui existe **um** "
            "modelo por papel; se ele falha, a amostra é rejeitada com motivo "
            "registrado.",
            _MUDANCA_SPLIT,
        ),
        limitacoes=(
            "**A escala métrica do Depth Pro é o gargalo atual**, não a máscara. "
            "**33,1%** das amostras do piloto têm o gabarito de foco **fora da faixa de "
            "disparidade da própria cena** — nenhuma máscara chegaria lá, nem a "
            "perfeita. Restringindo às alcançáveis, a concordância sobe de 32,1% para "
            "**45,0%**.",
            "**Sem validador analítico onde a largura do sensor não é publicada**, "
            "então a Eq. 3 não fecha e não há segundo cálculo independente para "
            "conferir o K do sweep. O número de amostras nessa situação está na tabela "
            "acima.",
            "**O limiar de SSIM da Eq. 5 é `[A]`.** O paper afirma aplicá-lo duas vezes "
            "e não publica o valor.",
            "**421 pares do `train` da origem (2,05%)** são anotados por ela como "
            "`misaligned` ou `shift_<X>px`, em 228 cenas. O campo viaja no metadado e "
            "ninguém filtra por ele; o efeito do desalinhamento na calibração não foi "
            "medido.",
        ),
        secoes_extras=(
            ("O teto de aberturas por cena",
             "O suplemento descreve séries *\"containing 2 to 4 images per set\"* "
             "(`paper.txt:1001-1003`) e um total de **13K** imagens novas. Com teto de "
             "4, a enumeração do split `train` dá **13.799** amostras — e a anotação "
             "manual que o paper relata (8 horas a 4-8 s por imagem) dá 3.600 a 7.200 "
             "máscaras, contra as 4.399 cenas da origem. Duas contas independentes "
             "fecham. Este release entrega **{treino}** no `train`: a diferença são as "
             "amostras rejeitadas no run, e cada uma está em `rejections.jsonl` com o "
             "motivo.\n\n"
             "Sem teto seriam 20.495, 58% acima do publicado, e **244 cenas (6,2%) "
             "gerariam 25% das amostras**. O teto não descarta cena nenhuma — só tira o "
             "peso excessivo de umas poucas. Os níveis mantidos são **espaçados "
             "uniformemente** pela faixa de aberturas, preservando a amplitude de K "
             "dentro da cena, que é o sinal que a rede precisa aprender."),
        ),
        linhas_de_proveniencia=(
            ("pipeline", "pipeline_commit"),
            ("profundidade", "depth_backend"),
            ("máscara", "mask_backend"),
            ("renderer", "renderer"),
            ("fator efetivo de K medido", "k_effective_factor"),
            ("origem do split", "split_origin"),
        ),
    ),
}


def rota_do_release(resumo: dict) -> Rota:
    """A rota deste release, ou `Problema` dizendo por que não dá para saber.

    Um card descreve **uma** rota: fonte, equações, modelos e artefatos mudam entre
    elas, e um texto que tentasse cobrir duas afirmaria de cada amostra coisas que só
    valem para metade. Release misto é um release a separar, não um card a escrever.
    """
    rotas = sorted(k for k in resumo["por_rota"] if k is not None)
    if len(rotas) != 1:
        raise Problema(
            f"o manifesto tem {len(rotas)} rotas ({rotas or 'nenhuma'}) e o card "
            "descreve uma. Fonte, equações, modelos e artefatos mudam entre as rotas; "
            "um card para duas afirmaria de cada amostra o que vale só para metade. "
            "Publique um release por rota.")
    if rotas[0] not in ROTAS:
        raise Problema(
            f"rota {rotas[0]!r} desconhecida — as rotas do §3.2 são "
            f"{sorted(ROTAS)}. Sem saber o que a rota faz, o card só poderia inventar.")
    return ROTAS[rotas[0]]


# --------------------------------------------------------------------------------
# Validação
# --------------------------------------------------------------------------------

def _linhas(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def _indexa(path: Path, chave: str, *, rota: "Rota", problemas: list[str]) -> dict:
    """Indexa um ledger pela chave que a **rota** declara escrever nele.

    Não é `if chave in linha: ...`. Uma linha do ledger sem a chave da rota não é um
    caso a pular: é o ledger contradizendo a rota que o escreveu, e um release assim não
    prova o que diz provar. Então a linha é **contada e reportada**, não descartada em
    silêncio — que é a diferença entre corrigir o `KeyError: 'sample_id'` e escondê-lo.
    """
    linhas = _linhas(path)
    com_a_chave = [l for l in linhas if chave in l]
    indice = {l[chave]: l for l in com_a_chave}
    sem_a_chave = len(linhas) - len(com_a_chave)
    if sem_a_chave:
        problemas.append(
            f"{sem_a_chave} de {len(linhas)} linhas de {path.name} sem {chave!r}, que é "
            f"a unidade em que a {rota.nome} escreve esse ledger. Ou o ledger é de "
            "outra rota, ou o release mistura dois runs.")
    return indice


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for bloco in iter(lambda: fh.read(1 << 20), b""):
            h.update(bloco)
    return h.hexdigest()


def _amostragem(ids: list[str], quantos: int) -> list[str]:
    """Um subconjunto **determinístico** de `ids`, e independente da ordem do manifesto.

    Ordenar pelo sha256 do próprio `sample_id` — em vez de fatiar o manifesto — evita
    escolher sempre o mesmo prefixo de cena: os ids são determinísticos e ordenados por
    cena, então `ids[:256]` conferiria 256 amostras de seis cenas e chamaria isso de
    amostra do release.
    """
    return sorted(ids, key=lambda sid: hashlib.sha256(sid.encode()).digest())[:quantos]


def _diretorios_por_amostra(rota: "Rota") -> tuple[str, ...]:
    """Os diretórios que ESTE release tem que ter, um arquivo por amostra em cada.

    `generated/` entra só nas rotas que geram pixel: exigi-lo da rota C, que rotula
    pares que já existem, reprovaria um release correto.

    **`depth/` saiu desta lista.** Ele não é mais "um arquivo por amostra": no layout
    `per_scene` são 1.700 PNG para 69.700 amostras, e comparar a contagem de `depth/`
    com a de `meta/` reprovaria um release CORRETO. O que `depth/` tem que satisfazer é
    outra coisa — toda amostra com profundidade alcançável, e nenhum arquivo de
    profundidade sem dono —, e quem confere isso é `_confere_profundidade`.
    """
    return ("mask", "meta") + (
        ("generated",) if rota.pixel_gerado is not None else ())


def _run_config(release: Path) -> dict:
    caminho = release / "run_config.json"
    if not caminho.is_file():
        return {}
    try:
        return json.loads(caminho.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise Problema(f"{caminho} ilegível: {exc}") from exc


def _layout_do_release(release: Path) -> DepthLayout:
    """O layout de `depth/` que o release DECLARA no `run_config.json`.

    Declarado, nunca inferido. A alternativa — contar arquivos em `depth/` e comparar
    com o manifesto — não distingue um release por cena de um release por amostra que
    perdeu 68.000 arquivos, e essas duas coisas têm desfechos opostos: publicar e não
    publicar.

    Ausente significa `per_sample`: os releases sem o campo são exatamente os anteriores
    a ele (`bokehnet-regen-rota-c`, e o `-rota-a`), e todos são por amostra.
    """
    try:
        return depth_layout_of_run_config(_run_config(release))
    except ValueError as exc:
        raise Problema(
            f"`depth_layout` desconhecido em {release}/run_config.json: {exc}. Os "
            f"valores são {[l.value for l in DepthLayout]}. Um layout que a ferramenta "
            "não conhece não dá para validar nem para publicar.") from exc


def _confere_profundidade(release: Path, manifesto: list[dict], *,
                          layout: DepthLayout, problemas: list[str]) -> dict:
    """`depth/` está certo — e "certo" aqui NÃO é ter uma contagem igual à de `meta/`.

    Com a dedup por cena, `depth/` tem 1.700 arquivos para 69.700 amostras, e a trava
    antiga (`_confere_contagem_por_diretorio`) reprovaria esse release por 68.000
    arquivos "faltando". As duas coisas que de fato importam continuam valendo nos dois
    layouts, e são estas:

    1. **toda amostra tem profundidade ALCANÇÁVEL.** Não "existe um arquivo com o nome
       dela": existe o arquivo que o `depth_ref` dela aponta. É a pergunta certa nos
       dois layouts, e a única que o layout por cena permite fazer.
    2. **nenhum arquivo de profundidade sem dono.** Um PNG em `depth/` que linha nenhuma
       reivindica é resto de outro run, e vai subir junto — a metade que a trava antiga
       pegava, e que a dedup não pode perder. Com 1.700 arquivos reivindicados e 69.700
       no disco, é exatamente isto que denuncia um release meio migrado.

    Mais uma terceira, que só existe porque o layout agora é declarado: **a referência
    bate com o layout declarado**. Um manifesto cheio de `depth/<sample_id>.png` num
    release que diz `per_scene` é um release em que a dedup não aconteceu, e o card
    mandaria quem baixa procurar no caminho errado.
    """
    caminho_depth = release / DEPTH_DIR
    no_disco = ({p.relative_to(release).as_posix()
                 for p in caminho_depth.rglob("*") if p.is_file()}
                if caminho_depth.is_dir() else set())

    reivindicados: set[str] = set()
    sem_arquivo: list[str] = []
    fora_do_layout: list[str] = []
    ilegiveis: list[str] = []
    for linha in manifesto:
        try:
            ref = depth_ref_of(linha)
        except ValueError as exc:
            ilegiveis.append(f"{linha.get('sample_id')}: {exc}")
            continue
        reivindicados.add(ref)
        if ref not in no_disco:
            sem_arquivo.append(linha.get("sample_id"))
        esperado = depth_ref(sample_id=linha["sample_id"],
                             scene_id=linha["scene_id"], layout=layout)
        if ref != esperado:
            fora_do_layout.append(f"{linha['sample_id']}: {ref} != {esperado}")

    if ilegiveis:
        problemas.append(
            f"{len(ilegiveis)} linhas do manifesto sem profundidade resolvível "
            f"(ex.: {ilegiveis[0]}).")
    if sem_arquivo:
        problemas.append(
            f"{len(sem_arquivo)} amostras sem arquivo de profundidade alcançável "
            f"(ex.: {sem_arquivo[:3]}). A referência é o `depth_ref` de cada linha; um "
            "release em que ela não resolve não treina.")
    if fora_do_layout:
        problemas.append(
            f"{len(fora_do_layout)} `depth_ref` fora do layout declarado "
            f"`{layout.value}` (ex.: {fora_do_layout[:3]}). O `run_config.json` afirma "
            "um layout e o manifesto grava outro — quem baixar vai procurar no caminho "
            "que o card descreve, que é o declarado.")

    orfaos = sorted(no_disco - reivindicados)
    if orfaos:
        problemas.append(
            f"{len(orfaos)} arquivos em {DEPTH_DIR}/ que linha nenhuma do manifesto "
            f"reivindica (ex.: {orfaos[:3]}). Arquivo de profundidade sem dono é resto "
            "de outro run — ou de uma migração pela metade — e vai subir junto.")

    return {"arquivos": len(no_disco), "referencias": len(reivindicados),
            "amostras": len(manifesto), "layout": layout.value}


def _confere_contagem_por_diretorio(release: Path, esperado: int, *, rota: "Rota",
                                    problemas: list[str]) -> dict[str, int]:
    """Todos os diretórios por amostra têm a MESMA contagem, e ela é a do manifesto.

    A checagem acima (`faltando`) pergunta, para cada linha do manifesto, se o arquivo
    existe. Ela não compara os diretórios **entre si**, e foi por essa fresta que
    `juliadollis/bokehnet-regen-rota-b` foi publicado com **13.615 arquivos em `mask/` e
    4.688 em `meta/`**: faltam ~8.900 `meta/`, que é onde vivem K, profundidade, plano de
    foco e toda a proveniência. Sem `meta/` o dataset não treina, e nada no validador
    dizia isso.

    Conta recursivamente de propósito: no disco o layout é plano, mas um release
    reagrupado por cena tem `depth/<scene_id>/<id>.png` e um `iterdir()` contaria pastas.
    """
    contagens: dict[str, int] = {}
    for sub in _diretorios_por_amostra(rota):
        caminho = release / sub
        contagens[sub] = (sum(1 for p in caminho.rglob("*") if p.is_file())
                          if caminho.is_dir() else 0)

    for sub, quantos in sorted(contagens.items()):
        if quantos == esperado:
            continue
        if quantos < esperado:
            problemas.append(
                f"{sub}/ tem {quantos} arquivos para {esperado} amostras no manifesto — "
                f"faltam {esperado - quantos}. As outras pastas: "
                f"{ {k: v for k, v in contagens.items() if k != sub} }. Um release em "
                f"que {sub}/ está curto não treina, e a contagem por amostra do "
                "manifesto sozinha não denuncia a diferença entre as pastas.")
        else:
            problemas.append(
                f"{sub}/ tem {quantos} arquivos para {esperado} amostras no manifesto — "
                f"{quantos - esperado} a mais. Arquivo que nenhuma linha do manifesto "
                "reivindica é resto de outro run, e vai subir junto.")
    return contagens


def _confere_pixel_gerado(release: Path, manifesto: list[dict], *, rota: "Rota",
                          problemas: list[str]) -> int:
    """O pixel que ESTA rota gerou está aqui, não está vazio, e é o que o ledger diz.

    Nada conferia isto. Na rota A o `generated/<id>_bokeh.jpg` é **o que a rede aprende
    a produzir**, e a validação percorria só `depth/`, `mask/` e `meta/`
    (`publish_release.py:91`): um release com 10.000 bokehs faltando — ou com 10.000
    arquivos de 0 byte — passava com zero problemas.

    Três checagens, em ordem de custo:

    1. **toda amostra tem linha no `generated_images.jsonl`**, com sha256 não nulo. Sem
       isso o release não prova o pixel que ele mesmo produziu;
    2. **o arquivo existe e não tem 0 byte.** Um JPEG de 0 byte é o desfecho de um
       processo morto no meio da escrita, e é invisível para quem conta arquivos;
    3. **o sha256 gravado bate com os bytes em disco**, numa amostragem determinística
       de `AMOSTRAGEM_SHA256_PIXEL`. Reler 70 mil JPEGs de ~800 KB é ~55 GB por
       validação; reler zero era o que se fazia. O tamanho da amostra volta no resumo e
       é impresso, porque amostragem cujo `n` não aparece não é verificável.

    Devolve quantos sha256 foram de fato recalculados.
    """
    papel, campo_sha = rota.pixel_gerado
    ledger = _indexa(release / "generated_images.jsonl", "sample_id", rota=rota,
                     problemas=problemas)

    sem_linha, sem_sha, ausentes, vazios = [], [], [], []
    for linha in manifesto:
        sid = linha["sample_id"]
        registro = ledger.get(sid)
        if registro is None:
            sem_linha.append(sid)
        elif not registro.get(campo_sha):
            sem_sha.append(sid)
        caminho = release / "generated" / f"{sid}_{papel}.jpg"
        if not caminho.is_file():
            ausentes.append(sid)
        elif caminho.stat().st_size == 0:
            vazios.append(sid)

    o_que_e = "o ALVO do treino" if rota.pixel_gerado_e_o_alvo else "a ENTRADA da rede"
    if sem_linha:
        problemas.append(
            f"{len(sem_linha)} amostras sem linha em generated_images.jsonl "
            f"(ex.: {sem_linha[:3]}). A {rota.nome} GERA o {papel} — ele é {o_que_e} —, "
            "e sem o sha256 o release não prova o pixel que ele mesmo produziu.")
    if sem_sha:
        problemas.append(
            f"{len(sem_sha)} linhas de generated_images.jsonl com {campo_sha} nulo "
            f"(ex.: {sem_sha[:3]}) — o ledger existe e não prova nada.")
    if ausentes:
        problemas.append(
            f"{len(ausentes)} amostras sem arquivo em generated/ (ex.: {ausentes[:3]}). "
            f"Na {rota.nome} esse arquivo é {o_que_e}.")
    if vazios:
        problemas.append(
            f"{len(vazios)} arquivos de 0 byte em generated/ (ex.: {vazios[:3]}) — "
            "escrita interrompida, e a contagem de arquivos não denuncia.")

    conferiveis = [l["sample_id"] for l in manifesto
                   if l["sample_id"] in ledger
                   and ledger[l["sample_id"]].get(campo_sha)
                   and (release / "generated" /
                        f"{l['sample_id']}_{papel}.jpg").is_file()]
    divergentes = []
    escolhidos = _amostragem(conferiveis, AMOSTRAGEM_SHA256_PIXEL)
    for sid in escolhidos:
        caminho = release / "generated" / f"{sid}_{papel}.jpg"
        if _sha256(caminho) != ledger[sid][campo_sha]:
            divergentes.append(sid)
    if divergentes:
        problemas.append(
            f"{len(divergentes)} de {len(escolhidos)} amostras conferidas têm "
            f"{campo_sha} DIFERENTE dos bytes em disco (ex.: {divergentes[:3]}). O "
            "ledger e o release descrevem imagens diferentes.")
    return len(escolhidos)


def valida(release: Path) -> dict:
    problemas: list[str] = []
    avisos: list[str] = []

    manifesto = _linhas(release / "manifest.jsonl")
    if not manifesto:
        raise Problema(f"{release}/manifest.jsonl vazio ou ausente — não há release.")

    # A rota é lida ANTES de qualquer checagem de pixel, porque é ela que diz **o que
    # checar**: qual ledger existe, em que unidade ele é escrito, e se há pixel gerado
    # por nós a conferir. Sem isso a validação só sabia olhar a rota C, e as rotas que
    # geram pixel passavam sem que nada do que elas produzem fosse conferido.
    por_rota = Counter(l.get("route") for l in manifesto)
    rota = rota_do_release({"por_rota": dict(por_rota)})

    split_path = release / "split.json"
    if not split_path.is_file():
        raise Problema(
            f"{split_path} ausente. Split materializado é requisito do release: deixado "
            "para o config do treino, ele diverge entre runs e some no rsync.")
    split = SceneSplit.load(split_path)

    ids = [linha["sample_id"] for linha in manifesto]
    repetidos = [i for i, n in Counter(ids).items() if n > 1]
    if repetidos:
        problemas.append(f"{len(repetidos)} sample_id repetidos no manifesto "
                         f"(ex.: {repetidos[:3]}) — a contagem publicada está inflada.")

    # -- arquivos por amostra --------------------------------------------------
    #
    # `depth/` NÃO entra aqui. Ele deixou de ser um arquivo por amostra: no layout
    # `per_scene` são 1.700 PNG para 69.700 amostras, e perguntar por
    # `depth/<sample_id>.png` reprovaria 68.000 amostras de um release correto. A
    # pergunta certa — "a profundidade desta amostra é alcançável?" — é a de
    # `_confere_profundidade`, que vale nos dois layouts.
    faltando: dict[str, list[str]] = {}
    for linha in manifesto:
        sid = linha["sample_id"]
        for sub, ext in (("mask", ".png"), ("meta", ".json")):
            if not (release / sub / f"{sid}{ext}").is_file():
                faltando.setdefault(sub, []).append(sid)
    for sub, quais in faltando.items():
        problemas.append(f"{len(quais)} amostras sem arquivo em {sub}/ "
                         f"(ex.: {quais[:3]})")

    _confere_contagem_por_diretorio(release, len(set(ids)), rota=rota,
                                    problemas=problemas)
    layout = _layout_do_release(release)
    profundidade = _confere_profundidade(release, manifesto, layout=layout,
                                         problemas=problemas)

    # -- metadados -------------------------------------------------------------
    versoes, backends, invalidos = Counter(), Counter(), []
    # De onde saiu a região que definiu `D_focus` em cada amostra, e quantas passaram
    # pelo refinamento. Vai para o resumo e para o card: quem baixa o release precisa
    # poder montar o treino COM e SEM as amostras refinadas, e para isso precisa saber
    # quantas são antes de baixar o release inteiro.
    #
    # `refinadas` conta a FONTE (`focus_source` em `FONTES_REFINADAS`), não o booleano
    # `focus_was_refined`. Contar pelo booleano fazia o relatório afirmar `refined 4853
    # de 4853` num lote da rota A em que **nada** foi refinado: `was_refined` é
    # `FocusSource(...) is not BIREFNET` (`dataio/sample.py:228`), e o plano de foco
    # sorteado da rota A não é `BIREFNET`. `marcadas_como_refinadas` continua contado ao
    # lado justamente para que a divergência apareça como aviso em vez de virar o
    # número publicado.
    fontes_de_foco: Counter = Counter()
    #: O layout que cada `meta/` DECLARA. Contado porque ele e o `run_config.json` são
    #: duas afirmações sobre a mesma coisa, e duas afirmações que podem divergir é a
    #: forma exata do defeito que este projeto persegue. `AUSENTE` é o release anterior
    #: ao campo, que é legítimo e tem que continuar validando.
    layouts_nos_metas: Counter = Counter()
    refinadas = marcadas_como_refinadas = censuradas = sem_validador = 0
    for linha in manifesto:
        meta_path = release / "meta" / f"{linha['sample_id']}.json"
        if not meta_path.is_file():
            continue
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        try:
            validate_metadata(meta)
        except Exception as exc:
            invalidos.append(f"{linha['sample_id']}: {exc}")
        versoes[meta.get("control_version")] += 1
        backends[meta.get("depth_backend")] += 1
        fontes_de_foco[meta.get("focus_source", "AUSENTE")] += 1
        layouts_nos_metas[meta.get("depth_layout", "AUSENTE")] += 1
        refinadas += meta.get("focus_source") in FONTES_REFINADAS
        marcadas_como_refinadas += bool(meta.get("focus_was_refined"))
        censuradas += bool(meta.get("is_k_censored"))
        sem_validador += meta.get("k_analytic") is None
    if invalidos:
        problemas.append(f"{len(invalidos)} metadados inválidos "
                         f"(ex.: {invalidos[0]})")
    if len(versoes) > 1:
        problemas.append(
            f"control_version divergente no mesmo release: {dict(versoes)}. Foi assim "
            "que duas convenções de normalização entraram no mesmo batch.")
    if len(backends) > 1:
        problemas.append(f"depth_backend divergente: {dict(backends)} — profundidade "
                         "métrica e disparidade normalizada não são a mesma coisa.")

    # O `run_config.json` e os `meta/` afirmam o mesmo layout, ou o release mistura dois
    # runs. `AUSENTE` só é aceitável quando o release inteiro é anterior ao campo — e aí
    # `_layout_do_release` já devolveu `per_sample`, que é o que aqueles releases são.
    declarados = {v for v in layouts_nos_metas if v != "AUSENTE"}
    if len(layouts_nos_metas) > 1 or (declarados and declarados != {layout.value}):
        problemas.append(
            f"layout de profundidade divergente: run_config.json diz "
            f"`{layout.value}` e os meta/ dizem {dict(layouts_nos_metas)}. Um release "
            "com dois layouts é um release remontado de dois runs, e metade dele "
            "aponta para arquivos que o card não descreve.")

    # -- proveniência dos pixels ------------------------------------------------
    #
    # Dois ledgers, e cada um prova uma coisa diferente:
    #
    # * `source_images.jsonl` prova os bytes que **não são nossos** — a AIF e/ou a
    #   bokeh que vieram da origem. A UNIDADE dele é da rota: por `scene_id` na rota A
    #   (41 variantes saem dos mesmos bytes de AIF, `sources/genphoto_ebb.py:565-569`),
    #   por `sample_id` na rota C (as duas imagens são de origem e são por amostra).
    #   Ler os dois com `l["sample_id"]` era o `KeyError: 'sample_id'` que derrubava a
    #   validação da rota A inteira — e que a rota B nunca viu só porque não escrevia o
    #   arquivo, de modo que `_linhas` devolvia `[]` e o laço nunca iterava.
    #
    # * `generated_images.jsonl` prova os bytes que **são nossos** — o pixel que a rota
    #   gerou. Ele não era conferido por nada: um release da rota A com 10.000 bokehs
    #   faltando passava, e a bokeh é exatamente o que a rede aprende a produzir.
    chave = rota.unidade_do_ledger_de_origem
    ledger_de_origem = _indexa(release / "source_images.jsonl", chave, rota=rota,
                               problemas=problemas)
    faltando_origem = sorted({l[chave] for l in manifesto
                              if l[chave] not in ledger_de_origem})
    if faltando_origem:
        recado = (
            f"{len(faltando_origem)} {chave} da {rota.nome} sem linha em "
            f"source_images.jsonl (ex.: {faltando_origem[:3]}). Sem o sha256 dos bytes "
            "de origem, o release não prova contra quais bytes o rótulo foi produzido.")
        (problemas if rota.exige_ledger_de_origem else avisos).append(recado)

    conferidos_sha256 = 0
    if rota.pixel_gerado is not None:
        conferidos_sha256 = _confere_pixel_gerado(
            release, manifesto, rota=rota, problemas=problemas)

    autocontido = (release / "source").is_dir() and any((release / "source").iterdir())
    if not autocontido:
        de_fora = "a AIF de entrada" if rota.pixel_gerado_e_o_alvo else "o ALVO"
        avisos.append(
            f"release NÃO autocontido: {de_fora} das {len(manifesto)} amostras da "
            f"{rota.nome} aponta para a origem. O join é verificável por sha256, mas "
            "quem baixar precisa de acesso a ela. Para embutir os pixels, gere com "
            "--store-source-images.")

    # -- split ------------------------------------------------------------------
    cenas_manifesto = {l["scene_id"] for l in manifesto}
    fora = cenas_manifesto - set(split.assignment)
    if fora:
        problemas.append(f"{len(fora)} cenas no manifesto e fora do split.json "
                         f"(ex.: {sorted(fora)[:3]})")
    vazamento = check_no_leak(manifesto, split)
    if not vazamento.clean:
        problemas.append(f"vazamento de split:{vazamento.summary()}")

    if refinadas / max(len(manifesto), 1) > 0.5:
        avisos.append(
            f"{refinadas} de {len(manifesto)} amostras tiveram a região em "
            f"foco REFINADA ({dict(fontes_de_foco)}). O refinamento é o substituto automático "
            "do passo manual do §3.2(c) e está marcado por amostra — mas um lote em que "
            "ele domina precisa do laudo de scripts/validate_focus_refinement.py antes "
            "de virar treino.")
    # O booleano por onde o dataset se filtra e a fonte que descreve o que aconteceu não
    # podem divergir em silêncio. Divergem na rota A, e por construção: `was_refined` é
    # `FocusSource(...) is not BIREFNET` (`dataio/sample.py:228`), então o plano de foco
    # SORTEADO sai marcado como refinado — e `validate_metadata` exige que saia
    # (`dataio/sample.py:472-476`), de modo que o metadado está internamente coerente e
    # mesmo assim afirma o que não houve. O relatório conta pela fonte e diz o tamanho
    # da divergência; consertar o campo é mudança em `dataio/`, não aqui.
    if marcadas_como_refinadas != refinadas:
        avisos.append(
            f"`focus_was_refined` é true em {marcadas_como_refinadas} amostras, mas só "
            f"{refinadas} têm `focus_source` em {sorted(FONTES_REFINADAS)}. Quem filtrar "
            "por esse booleano NÃO vai selecionar as refinadas: `was_refined` é "
            "`focus_source is not birefnet` (dataio/sample.py:228), e fontes como "
            "`sampled_plane` (plano de foco SORTEADO, sem refinamento nenhum) caem do "
            "lado errado. Filtre por `focus_source`.")

    contagens_split = Counter(l["split"] for l in manifesto)
    if contagens_split.get("val", 0) == 0:
        avisos.append("nenhuma amostra em `val`. Validação vazia é o jeito mais "
                      "silencioso de não ter validação.")

    if censuradas:
        pct = 100 * censuradas / len(manifesto)
        (problemas if pct > 20 else avisos).append(
            f"{censuradas} amostras ({pct:.1f}%) com K censurado no teto da busca. "
            "No release anterior eram 47,0% — acima de 20% o teto está errado, não os "
            "dados.")

    if problemas:
        raise Problema("\n  - ".join(["release reprovado:"] + problemas))

    return {
        "amostras": len(manifesto),
        "cenas": len(cenas_manifesto),
        "por_split": dict(contagens_split),
        "por_rota": dict(por_rota),
        "rota": rota.letra,
        # As fontes de origem saem do MANIFESTO, que grava `source_dataset` em toda
        # linha — não da `provenance_base`, onde a chave existe na rota C e não existe
        # na rota A. Era dali que vinha o `(não registrada)` que o card imprimia como
        # licença dos pixels da rota A.
        "fontes": sorted({l["source_dataset"] for l in manifesto
                          if l.get("source_dataset")}),
        "control_version": next(iter(versoes)),
        "depth_backend": next(iter(backends)),
        # Em CENAS e em AMOSTRAS, como toda contagem deste projeto: `depth/` com 1.700
        # arquivos para 69.700 amostras é a economia funcionando, e imprimir só um dos
        # dois números faria o mesmo release parecer completo ou mutilado.
        "depth_layout": layout.value,
        "depth_arquivos": profundidade["arquivos"],
        "k_censurado": censuradas,
        "sem_validador_analitico": sem_validador,
        "focus_sources": dict(fontes_de_foco),
        "refined": refinadas,
        "marcadas_como_refinadas": marcadas_como_refinadas,
        "pixel_gerado_sha256_conferidos": conferidos_sha256,
        "autocontido": autocontido,
        "avisos": avisos,
    }


# --------------------------------------------------------------------------------
# Card
# --------------------------------------------------------------------------------

def _mil(valor: int) -> str:
    """15423 -> '15.423'. O card é em português; separador trocado engana o leitor."""
    return f"{valor:,}".replace(",", ".")


def _pt(valor: float, casas: int = 1) -> str:
    """4.2 -> '4,2'. Para casar com os números de prosa do card, todos em pt-BR."""
    return f"{valor:.{casas}f}".replace(".", ",")


#: Os artefatos de raiz que o card descreve, e o que cada um é. A lista impressa é a
#: INTERSEÇÃO desta tabela com o que existe no release: um card que promete
#: `generated_images.jsonl` num release que não o tem manda quem baixa procurar um
#: arquivo que não existe, e um card que **omite** `generated/` — como o da rota C fazia
#: ao ser reusado na rota A — esconde exatamente o pixel que a rede aprende a produzir.
_DESCRICAO_DOS_ARTEFATOS: tuple[tuple[str, str], ...] = (
    ("manifest.jsonl", "uma linha por amostra, para varredura rápida"),
    ("split.json", "split POR CENA, materializado no release"),
    ("source_images.jsonl",
     "sha256 dos bytes de ORIGEM — os que não são nossos"),
    ("generated_images.jsonl",
     "sha256 do JPEG em `generated/` — o pixel que ESTA rota gerou"),
    ("rejections.jsonl",
     "UM item processado por linha — `status` `ok` ou `rejected`, e o `reason` quando\n"
     "    recusado. Não é só a lista de recusas: é o desfecho de tudo que entrou, que é\n"
     "    o que permite calcular a taxa sem regerar"),
    ("run_config.json", "os argumentos do run e a proveniência comum a todas as amostras"),
)


def _artefatos(release: Path, rota: Rota, cena_no_caminho: str,
               depth_no_caminho: str = "", por_cena: bool = False) -> str:
    """O bloco `Arquivos` do card, montado do que o release TEM.

    Escrito à mão, ele saiu errado nas duas direções ao mesmo tempo: prometia
    `source_images.jsonl` a uma rota que não o escreve e omitia `generated/` na rota que
    guarda ali o alvo do treino.

    `depth/` tem caminho próprio porque ele pode ser por cena enquanto o resto é por
    amostra, e pode ficar plano enquanto o resto é reagrupado. Um card que dissesse
    `depth/<id>.png` num release deduplicado mandaria quem baixa procurar 69.700
    arquivos que não existem.
    """
    linhas = [
        (f"depth/{depth_no_caminho}<scene_id>.png" if por_cena
         else f"depth/{depth_no_caminho}<id>.png"),
        ("    disparidade uint16, lado longo 768 — UMA por cena, compartilhada pelas\n"
         "    amostras dela. O caminho de cada amostra está em `depth_ref`, no `meta/`\n"
         "    e na linha do manifesto: leia o campo, não remonte o nome"
         if por_cena else "    disparidade uint16, lado longo 768"),
        f"mask/{cena_no_caminho}<id>.png",
        ("    a REGIÃO FINAL de foco — a que definiu `focus_disparity`, já refinada\n"
         "    quando `focus_source` diz isso"
         if rota.refina_a_regiao_de_foco else
         "    a região que definiu `focus_disparity`; `mask_source` diz de onde ela veio"),
        f"meta/{cena_no_caminho}<id>.json",
        "    K, focus_disparity, gates medidos, proveniência completa",
    ]
    if rota.pixel_gerado is not None:
        papel, _ = rota.pixel_gerado
        o_que_e = ("o ALVO do treino" if rota.pixel_gerado_e_o_alvo
                   else "a ENTRADA da rede")
        linhas += [f"generated/{cena_no_caminho}<id>_{papel}.jpg",
                   f"    {o_que_e}, JPEG q95 — gerado por nós, com sha256 no ledger"]
    for nome, descricao in _DESCRICAO_DOS_ARTEFATOS:
        if (release / nome).is_file():
            linhas += [nome, f"    {descricao}"]
    return "\n".join(linhas)


def _tabela_de_proveniencia(rota: Rota, prov: dict) -> str:
    """As linhas da tabela de proveniência cujo valor EXISTE neste release.

    Campo ausente some. Antes ele virava `?` ou `(não registrada)`, e a rota A — que não
    roda segmentador — publicava uma linha `máscara: None · …` afirmando ter usado um
    modelo que ela nunca carregou.
    """
    def _hash_curto(chave: str) -> str:
        valor = prov.get(chave)
        return f" · `{str(valor)[:16]}…`" if valor else ""

    linhas = []
    for rotulo, chave in rota.linhas_de_proveniencia:
        valor = prov.get(chave)
        if chave == "renderer":
            renderer = valor or {}
            # O bloco do renderizador grava `renderer_commit`; `commit` é só o nome
            # curto que algum run antigo usou. Ler apenas um dos dois imprimia `?` no
            # card de um release cujo commit estava gravado.
            commit = str(renderer.get("renderer_commit")
                         or renderer.get("commit") or "")[:12]
            if commit:
                linhas.append(f"| renderer | BokehMe `{commit}` |")
            continue
        if valor in (None, "", []):
            continue
        if chave.endswith("_sha256"):
            linhas.append(f"| {rotulo} | `{str(valor)[:16]}…` |")
        elif chave == "depth_backend":
            linhas.append(f"| {rotulo} | `{valor}`{_hash_curto('depth_model_sha256')} |")
        elif chave == "mask_backend":
            linhas.append(f"| {rotulo} | `{valor}`{_hash_curto('mask_model_sha256')} |")
        else:
            linhas.append(f"| {rotulo} | `{valor}` |")
    return "\n".join(linhas)


def escreve_card(release: Path, resumo: dict, repo_id: str) -> Path:
    """O README que vira a página do dataset. Diz o que o dado É, e o que ele NÃO é.

    **Tudo o que este card afirma sai da rota do release, do manifesto e da proveniência
    gravada.** Nada é escrito à mão aqui, e essa é a correção de um defeito medido: a
    versão anterior era uma f-string com o texto da rota C, e publicou dois releases da
    rota B dizendo `pretty_name: BokehNet regen — rota C`, com seções sobre RealBokeh,
    BiRefNet e a Eq. 5 — nenhum dos três presente naquela rota
    (`reference/AVALIACAO_DATASETS_BC.md:625-640`).
    """
    rota = rota_do_release(resumo)
    cfg_path = release / "run_config.json"
    cfg = json.loads(cfg_path.read_text(encoding="utf-8")) if cfg_path.is_file() else {}
    prov = cfg.get("provenance_base", {})

    # As fontes vêm do MANIFESTO (`source_dataset` em toda linha), não da
    # `provenance_base` — onde a chave existe na rota C e não existe na rota A. Era dali
    # que saía o `(não registrada)` que o card imprimia como licença dos pixels.
    fontes = resumo.get("fontes") or (
        [prov["source_dataset"]] if prov.get("source_dataset") else [])
    fontes_md = ", ".join(f"`{f}`" for f in fontes) or "a origem declarada no manifesto"

    n = resumo["amostras"]
    refinadas = resumo["refined"]
    treino = resumo["por_split"].get("train", 0)
    splits = " · ".join(f"{k} {_mil(v)}" for k, v in sorted(resumo["por_split"].items()))

    # -- números, e só as linhas que significam alguma coisa NESTA rota ---------
    numeros = [f"| amostras | {_mil(n)} |",
               f"| **cenas** | {_mil(resumo['cenas'])} |",
               f"| amostras por split (a fronteira é POR CENA) | {splits} |"]
    if rota.calibra_k_por_varredura:
        pct_cens = _pt(100 * resumo["k_censurado"] / max(n, 1))
        numeros.append(f"| K censurado no teto da busca | "
                       f"{_mil(resumo['k_censurado'])} (**{pct_cens}%**) |")
    if rota.refina_a_regiao_de_foco:
        pct_ref = _pt(100 * refinadas / max(n, 1))
        numeros.append(f"| região em foco refinada | "
                       f"{_mil(refinadas)} (**{pct_ref}%**) |")
    if rota.tem_validador_analitico:
        numeros.append(f"| sem validador analítico | "
                       f"{_mil(resumo['sem_validador_analitico'])} |")
    if resumo.get("pixel_gerado_sha256_conferidos"):
        numeros.append(
            f"| sha256 de `generated/` reconferidos byte a byte na validação | "
            f"{_mil(resumo['pixel_gerado_sha256_conferidos'])} |")

    # -- composição da região em foco, contada do release ----------------------
    composicao = "\n".join(
        f"| `{fonte}` | {_mil(q)} | {_pt(100 * q / max(n, 1))}% |"
        for fonte, q in sorted(resumo["focus_sources"].items(), key=lambda kv: -kv[1]))

    # -- layout publicado ------------------------------------------------------
    reagrupadas = _agrupa_por_cena(release)
    profundidade_por_cena = (
        DepthLayout(resumo.get("depth_layout", DepthLayout.PER_SAMPLE.value))
        is DepthLayout.PER_SCENE)
    # `mask/` é a pasta que representa "os arquivos por amostra" no texto: ela existe em
    # toda rota e é sempre por amostra, então `<scene_id>/` no caminho delas é
    # exatamente a condição de o reagrupamento ter acontecido para o leitor do card.
    cena_no_caminho = "<scene_id>/" if "mask" in reagrupadas else ""
    depth_no_caminho = "<scene_id>/" if DEPTH_DIR in reagrupadas else ""
    agrupamento = ("""
Os arquivos por amostra ficam **dentro da pasta da cena** — `mask/<scene_id>/<id>.png`.
Não é decoração: o Hub recusa o push de qualquer diretório com mais de 10.000 arquivos,
e são {n} amostras. A cena foi o agrupamento escolhido porque `scene_id` já está em
toda linha do `manifest.jsonl` e em todo `meta/`, então montar o caminho é ler um campo:

```python
f"mask/{{linha['scene_id']}}/{{linha['sample_id']}}.png"
```

O nome do arquivo e o conteúdo são os mesmos do release em disco; muda só o nível de
diretório. Foram reagrupadas: {pastas}.
""".format(n=_mil(n), pastas=", ".join(f"`{s}/`" for s in sorted(reagrupadas)))
        if reagrupadas else "")

    dedup = ("""
## A profundidade é por CENA, e o caminho dela está no metadado

`depth/` tem **{arquivos}** PNG para **{amostras}** amostras, e isso é a forma correta
do release, não uma pasta incompleta. A profundidade sai da AIF, e as amostras de uma
cena compartilham a AIF — medido na rota A: 120 de 120 cenas com a profundidade **byte a
byte idêntica** entre as 41 variantes. Gravar uma cópia por amostra eram 31,7 GB onde
0,77 GB bastam.

**Não remonte o caminho a partir do `sample_id`.** `scene_id` não é dedutível de
`sample_id` em rota nenhuma. Cada `meta/<id>.json` e cada linha do `manifest.jsonl`
trazem o campo `depth_ref` com o caminho relativo ao release, e `run_config.json` traz
`depth_layout` dizendo qual layout é este:

```python
import json
linha = json.loads(primeira_linha_do_manifesto)
profundidade = release / linha["depth_ref"]      # "depth/<scene_id>.png"
```

Os limites que reconstroem a escala — `disparity_min` e `disparity_max` — continuam
**por amostra** no `meta/`. O PNG compartilhado carrega a forma; a escala é de cada uma.
""".format(arquivos=_mil(resumo.get("depth_arquivos", 0)), amostras=_mil(n))
        if profundidade_por_cena else "")

    pixels = ("Os pixels de origem estão neste repositório, em `source/`, nos **bytes "
              "originais** — não recomprimidos, então o sha256 do ledger bate com o "
              "arquivo ao lado."
              if resumo["autocontido"] else
              f"Os pixels de origem **não** estão aqui: cada amostra aponta para "
              f"{fontes_md} pelo `source_sample_id`, e `source_images.jsonl` traz o "
              "sha256 para que o join seja verificável byte a byte.")

    # -- seções que só existem em algumas rotas --------------------------------
    secao_refinamento = ""
    if rota.refina_a_regiao_de_foco:
        secao_refinamento = f"""
## O refinamento da região em foco — leia antes de treinar

{_pt(100 * refinadas / max(n, 1))}% das amostras tiveram a região em foco **refinada**, e
isso precisa de explicação porque é um desvio declarado do paper.

Antes de qualquer coisa, o que está no disco: **`mask/<id>.png` é a REGIÃO FINAL de foco**
— a que produziu `focus_disparity` —, **não** a saída crua do BiRefNet. Quando a amostra
foi refinada, o arquivo é a região refinada, e `mask_source` diz isso; a validação do
release reprova metadado em que os dois discordem, justamente para ninguém auditar o
rótulo olhando a máscara errada.

Composição medida **neste** release:

| `focus_source` | amostras | |
|---|---|---|
{composicao}

**Filtre por `focus_source`, não por `focus_was_refined`.** O booleano é
`focus_source is not birefnet` (`dataio/sample.py:228`): descreve o refinamento nesta
rota, e não descreve em toda rota — há rotas em que ele sai `true` sem que nada tenha
sido refinado.

O §3.2(c) usa o BiRefNet para obter a máscara de foco, e reconhece que ela é *"sometimes
unreliable"* nestes datasets. A solução dele é **refinamento manual**, e ele diz
explicitamente que **não descarta** os casos ruins (`paper.txt:359-368`).

Medimos o tamanho do problema num piloto de 162 amostras, sem refino: a máscara do
BiRefNet acerta o plano de foco em apenas **35,2%** das amostras, comparada contra a
distância de foco que a origem publica **medida a ±1 cm**. A razão mediana é 0,579 — o
plano escolhido fica ~1,7x mais longe do que estava. E **20,6%** dos pares eram
descartados por máscara vazia.

A causa está nas imagens: a origem é feita de **cenas**, não de fotos de objeto — um
tronco de árvore num parque, um muro de pedra. O BiRefNet segmenta objeto **saliente**, e
fora desse domínio ele devolve probabilidade exatamente 0 em cerca de um terço dos casos.

Substituímos o passo manual por um automático que usa a definição física de estar em
foco: temos a AIF **e** a bokeh da mesma cena, e no plano de foco a bokeh **preservou** o
detalhe da AIF.

```
retencao(x) = media_local(|laplaciano(bokeh)|) / media_local(|laplaciano(aif)|)
```

O denominador normaliza pela textura da própria cena — é o que faz medida de nitidez
absoluta falhar, porque folhagem desfocada tem mais alta frequência que parede lisa em
foco.

**Cada amostra carrega `focus_source` e `focus_was_refined`**, no metadado e no manifesto,
mais `focus_disparity_from_initial_mask`: o rótulo que ela teria tido **sem** refino. Isso
permite treinar com e sem as refinadas e medir a diferença, em vez de acreditar.

Medido no bloco pareado do piloto com refino (302 amostras) — a mesma amostra com e sem
refino, nas 88 refinadas que tinham linha de base: dentro de ±25% do gabarito, **19,3%
antes contra 23,9% depois**, com 59 amostras melhorando e 29 piorando. Nesse mesmo piloto
o descarte por máscara vazia caiu de 20,6% para **zero**.

Os números deste parágrafo são **do piloto**, não deste lote: a validação pareada não foi
refeita sobre as {_mil(n)} amostras. Os contadores da tabela acima, sim, são contados
deste release.
"""

    extras = "".join(
        f"\n## {titulo}\n\n{corpo.format(n=_mil(n), treino=_mil(treino))}\n"
        for titulo, corpo in rota.secoes_extras)

    mudancas = "\n\n".join(rota.mudancas)
    limitacoes = "\n".join(f"- {item}" for item in rota.limitacoes)
    tabela_prov = _tabela_de_proveniencia(rota, prov)
    numeros_md = "\n".join(numeros)
    arquivos = _artefatos(release, rota, cena_no_caminho, depth_no_caminho,
                          por_cena=profundidade_por_cena)

    card = f"""---
license: other
pretty_name: BokehNet regen — {rota.nome}
tags: [bokeh, depth-of-field, defocus, image-to-image]
---

# {repo_id}

Dados de treino da BokehNet regerados segundo o **GenRefocus** (arXiv:2512.16923v3),
{rota.secao} — a **{rota.nome}**: {rota.abertura.format(fontes=fontes_md)}

Este não é um repack do dataset anterior. É uma regeração do zero, motivada por defeitos
medidos no pipeline original — cada um deles está descrito abaixo com o número que o
comprova, porque é isso que permite decidir se este dado é confiável.

## Números

| | |
|---|---|
{numeros_md}

A contagem em **cenas** aparece junto com a de amostras de propósito: número de amostras
não é número de unidades independentes — e é a cena que define o split.

## O que a rede aprende a produzir

{rota.alvo}

## O que mudou em relação ao dataset anterior

{mudancas}

## O contrato do sinal de controle

`control_version = {resumo['control_version']}`. Toda amostra segue exatamente isto:

{rota.sinal_de_controle}
{secao_refinamento}{extras}
## Arquivos

```
{arquivos}
```
{agrupamento}{dedup}
{pixels}

## O que este dataset NÃO é

- **Não é revisado por humano.** O paper filtra com inspeção manual; aqui isso foi
  substituído por gates automáticos, todos **medidos e gravados** em cada `meta/`.
- **Os gates estão em modo medir.** Nenhum limiar bloqueia: este é o lote **bruto**, com
  toda métrica registrada. O dataset de treino sai daqui por **filtro do manifesto**, sem
  reprocessar, porque o `sample_id` é determinístico. Escolher limiar antes de ver a
  distribuição foi o que produziu os 47% de K censurado no release anterior.
- **`rejections.jsonl` faz parte do release.** O histograma de motivos é o que permite
  recalibrar sem regerar, e é ele que denuncia fallback novo.
- **Amostra com `is_valid_for_control == false` não deve entrar no treino de controle.**
  Ela fica aqui porque escondê-la esconderia a taxa.

## Proveniência

| | |
|---|---|
{tabela_prov}

## Limitações conhecidas

{limitacoes}

## Licença

Os rótulos são deste projeto. **Os pixels seguem a licença da origem ({fontes_md})** —
verifique-a antes de redistribuir.
"""
    destino = release / "README.md"
    destino.write_text(card, encoding="utf-8")
    return destino


# --------------------------------------------------------------------------------
# Upload
# --------------------------------------------------------------------------------

#: Não é release: o HF cria no `create_repo`, e o `.cache/huggingface` é o diário de
#: bordo que o `upload_large_folder` usa para retomar. Contar qualquer um deles como
#: "o repo já tem arquivos" trancava a retentativa depois de um upload interrompido —
#: justamente quando retomar é o que se quer.
_ANDAIME = frozenset({".gitattributes"})

IGNORAR = ["*.tmp", "__pycache__/*", "_mirror_index.json", ".cache/huggingface/*"]

#: Acima disto o commit único não passa. Os dois limites do Hub são opostos, e um
#: release desta rota fica exatamente entre eles — os dois foram batidos de verdade
#: publicando este release, de 46.274 arquivos e 7,6 GB:
#:
#: 1. **um commit com tudo → 413 Payload Too Large.** Os bytes subiram inteiros (8 GB
#:    pelo Xet) e o commit foi recusado no fim, o que é o pior desfecho possível.
#: 2. **um commit por punhado de arquivos → 429.** O `upload_large_folder` fatia em
#:    lotes de ~75 e precisaria de ~600 commits; o Hub permite **256 commits/hora** e o
#:    corta no meio, e a cada 429 ele reduz o lote, o que piora a conta.
#:
#: Então a fatia certa é grossa mas não única: `LOTE_POR_COMMIT` arquivos por commit dá
#: ~10 commits para este release — longe do 413 e longe do 429. Release pequeno
#: continua indo em **um** commit, porque aí ele é atômico: aparece inteiro ou não
#: aparece.
LIMITE_COMMIT_UNICO_ARQUIVOS = 8_000
LIMITE_COMMIT_UNICO_BYTES = 2 * 1024**3
#: 2.000 dá ~24 commits para este release: uma ordem de grandeza abaixo do 429, e cada
#: commit ainda termina dentro de `TIMEOUT_REQUEST_S`. Com 5.000 o servidor levava mais
#: que o timeout para processar o lote.
LOTE_POR_COMMIT = 2_000

#: Teto do Hub para arquivos **grandes** (LFS) num único commit. Medido publicando a
#: rota B: `413 Payload Too Large` com a mensagem literal *"A maximum of 25k Large Files
#: are supported per commit"*. `LOTE_POR_COMMIT` já fica uma ordem de grandeza abaixo,
#: mas a constante existe para que a razão do número esteja escrita, e não só o número.
MAX_ARQUIVOS_GRANDES_POR_COMMIT = 25_000


def _arquivos(release: Path) -> list[Path]:
    """O que vai subir. O `.cache/huggingface` de uma tentativa anterior fica fora: ele
    não é release, e inflaria o número impresso antes do upload."""
    return sorted(p for p in release.rglob("*")
                  if p.is_file()
                  and ".cache" not in p.relative_to(release).parts
                  and "__pycache__" not in p.relative_to(release).parts
                  and p.suffix != ".tmp"
                  and p.name != "_mirror_index.json")


def _tamanho(release: Path) -> tuple[int, int]:
    arquivos = _arquivos(release)
    return len(arquivos), sum(p.stat().st_size for p in arquivos)


#: Pastas por amostra: no disco são planas, no repositório precisam de um nível a mais.
#:
#: `generated` entrou depois, e a ausência dela custou uma publicação inteira. A rota C
#: não gera pixel — ela só rotula pares que já existem —, então `depth`/`mask`/`meta`
#: bastavam. A rota B **gera a AIF**: são 13.615 JPEGs de ~800 KB em `generated/`, que
#: estouram os 10.000 por pasta E contam como "large file" no limite de 25.000 por
#: commit. O upload morreu com `413 Payload Too Large`.
_POR_AMOSTRA = ("depth", "mask", "meta", "generated")

#: O Hub recusa o push com **400 Bad Request** quando um diretório passa de 10.000
#: arquivos — medido publicando este release, em que `depth/`, `mask/` e `meta/` têm
#: 15.423 cada. Não é limite de commit, é do repositório: nenhum fatiamento resolve, o
#: layout é que não cabe.
MAX_ARQUIVOS_POR_PASTA = 10_000


def _agrupa_por_cena(release: Path) -> frozenset[str]:
    """QUAIS pastas precisam ser reagrupadas para caber no Hub. Vazio = nenhuma.

    Antes era um booleano para o release inteiro, e a dedup mostrou por que isso está
    errado: com `depth/` em 1.700 arquivos e `mask/`, `meta/` e `generated/` em 69.700,
    o release inteiro era reagrupado por causa das três — e `depth/` ganhava um nível de
    diretório de que não precisa, transformando `depth/<scene_id>.png` em
    `depth/<scene_id>/<scene_id>.png` e quebrando o `depth_ref` gravado em cada `meta/`.

    Por pasta, cada uma responde pelo seu próprio tamanho: `depth/` fica plano, o resto
    é reagrupado, e a referência do metadado continua valendo no repositório.

    Só reagrupa quem não cabe: manter o layout plano onde ele cabe evita mudar o
    contrato de caminho de um release pequeno sem motivo nenhum.
    """
    return frozenset(
        sub for sub in _POR_AMOSTRA
        if (release / sub).is_dir()
        and sum(1 for _ in (release / sub).iterdir()) > MAX_ARQUIVOS_POR_PASTA)


def _cenas_do_manifesto(release: Path) -> dict[str, str]:
    return {l["sample_id"]: l["scene_id"] for l in _linhas(release / "manifest.jsonl")}


def _caminho_no_repo(relativo: Path, cenas: dict[str, str],
                     agrupadas: frozenset[str] = frozenset(_POR_AMOSTRA),
                     depth_por_cena: dict[str, str] | None = None) -> str:
    """Onde o arquivo vai morar no repositório.

    Agrupa por **cena** (`depth/<scene_id>/<sample_id>.png`) em vez de deixar os 15.423
    arquivos soltos em `depth/`. A cena é a escolha certa entre os agrupamentos
    possíveis por três motivos: é a unidade que o projeto já usa para o split e para as
    contagens; está gravada em toda linha do manifesto e em todo `meta/`, então a regra
    de recuperar o caminho é ler um campo e não rodar um hash; e uma cena tem de 2 a 21
    amostras, então nenhuma pasta chega perto do limite.

    O nome do arquivo não muda, e o conteúdo não muda. Só entra um nível de diretório.

    `agrupadas` diz **quais** pastas estouraram o limite: só elas ganham o nível a mais.
    Num release com a profundidade deduplicada, `depth/` tem 1.700 arquivos e fica
    plano, enquanto `mask/`, `meta/` e `generated/` são reagrupados — e é isso que
    mantém válido o `depth_ref` que cada `meta/` gravou.

    `depth_por_cena` mapeia `<scene_id>.png -> scene_id` para o caso raro em que
    `depth/` está por cena E estoura o limite: ali o arquivo não pertence a um
    `sample_id`, e procurá-lo no índice de amostras devolveria `None`.
    """
    partes = relativo.parts
    if len(partes) == 2 and partes[0] in agrupadas:
        if partes[0] == DEPTH_DIR and depth_por_cena:
            cena = depth_por_cena.get(partes[1])
        else:
            cena = cenas.get(_sample_id_do_arquivo(partes[1], cenas))
        if cena:
            return f"{partes[0]}/{cena}/{partes[1]}"
    return relativo.as_posix()


def _sample_id_do_arquivo(nome: str, cenas: dict[str, str]) -> str:
    """O `sample_id` a que este arquivo pertence.

    Em `depth/`, `mask/` e `meta/` o arquivo é `<sample_id>.<ext>` e o stem basta. Em
    `generated/` **não**: ali o nome é `<sample_id>_<papel>.jpg` — na rota B,
    `b_20002539892_aif.jpg`. Usar o stem cru devolveria `b_20002539892_aif`, que não
    está no manifesto, e o arquivo cairia no `return` de baixo: ficaria solto em
    `generated/`, com 13.615 irmãos, estourando de novo o limite de 10.000 por pasta.

    Foi exatamente esse caminho que deixou a publicação da rota B morrer em `413`.
    """
    stem = Path(nome).stem
    if stem in cenas:
        return stem
    # `<sample_id>_<papel>`: corta sufixos até casar com um id conhecido. O laço é curto
    # (os papéis têm um ou dois segmentos) e para no primeiro acerto.
    while "_" in stem:
        stem = stem.rsplit("_", 1)[0]
        if stem in cenas:
            return stem
    return Path(nome).stem


#: O 429 do Hub é por hora, então a espera útil é da ordem de minutos, não de segundos.
ESPERA_429_S = (300, 600, 900, 1800, 1800)

#: O default do `huggingface_hub` é **10 s** (`constants.DEFAULT_REQUEST_TIMEOUT`), e o
#: POST de commit de milhares de arquivos leva minutos do lado do servidor: com 10 s ele
#: sempre estoura em `httpx.ReadTimeout`, e o release fica pela metade. O cliente httpx
#: é criado na primeira chamada e lê a constante ali, então elevá-la antes do primeiro
#: request é o único jeito de alcançá-lo — não há env var para isto nesta versão.
TIMEOUT_REQUEST_S = 900


def _retentavel(exc: Exception) -> str:
    """Motivo pelo qual vale tentar o mesmo commit de novo, ou "" se não vale.

    Os dois casos são ambíguos-mas-idempotentes: repetir um commit com **os mesmos
    arquivos e os mesmos caminhos** não muda o repositório, então repetir é mais seguro
    que abortar no meio.
    """
    texto = f"{type(exc).__name__}: {exc}"
    if "429" in texto or "Too Many Requests" in texto:
        return "429 (limite de commits/hora)"
    if "Timeout" in texto or "timed out" in texto:
        return "timeout no POST do commit"
    return ""


def _commita_esperando_o_limite(api, repo_id: str, *, operations, mensagem: str) -> None:
    """Um commit, com espera nos erros que o Hub devolve sob carga.

    Sem isto, bater no limite de 256 commits/hora — ou no timeout de um commit grande —
    abortava o upload no meio e deixava o repositório com metade do release, que é pior
    que não publicar, porque parece publicado. Espera e tenta de novo; desiste alto,
    nunca em silêncio.
    """
    for tentativa, espera in enumerate((*ESPERA_429_S, None), 1):
        try:
            api.create_commit(repo_id=repo_id, repo_type="dataset",
                              operations=operations, commit_message=mensagem)
            return
        except Exception as exc:
            motivo = _retentavel(exc)
            if not motivo:
                raise
            if espera is None:
                raise Problema(
                    f"{motivo} ainda depois de {tentativa} tentativas. O release pode "
                    "estar PELA METADE no repositório: rode de novo mais tarde com "
                    "--allow-existing para completar.") from exc
            print(f"[hf] {motivo}. Esperando {espera // 60} min e tentando de novo "
                  f"(tentativa {tentativa}) …", flush=True)
            time.sleep(espera)


def _confere_o_que_subiu(api, repo_id: str, release: Path) -> None:
    """Depois do upload: o repositório tem a mesma contagem por pasta que o disco?

    Não é fluxo de rede novo — é o mesmo `list_repo_files` que `publica` já chama antes
    de decidir se o destino está vazio. O que ele passa a responder é a pergunta que
    ninguém fazia depois: **o que chegou lá é o que saiu daqui?**

    `juliadollis/bokehnet-regen-rota-b` está no Hub com 13.615 arquivos em `mask/` e
    **4.688** em `meta/`. O upload não avisou, a validação local não podia saber, e o
    dataset publicado não treina. Um commit recusado no meio do fatiamento produz
    exatamente essa assinatura: as primeiras pastas completas, uma no meio truncada.

    Reprova alto (`Problema`) porque um release pela metade no Hub é pior que nenhum:
    ele **parece** publicado. Se a listagem não puder ser obtida, diz isso — em nenhum
    caso conclui que está tudo bem sem ter contado.
    """
    try:
        remotos = api.list_repo_files(repo_id, repo_type="dataset")
    except Exception as exc:
        print(f"[hf] AVISO: não deu para listar {repo_id} depois do upload ({exc}). "
              "A contagem por pasta NÃO foi conferida — confira na página do dataset.",
              flush=True)
        return

    locais = [p.relative_to(release).as_posix() for p in _arquivos(release)]
    def _por_prefixo(caminhos) -> Counter:
        return Counter(c.split("/", 1)[0] for c in caminhos if "/" in c)

    esperado, chegou = _por_prefixo(locais), _por_prefixo(remotos)
    curtos = [(sub, chegou.get(sub, 0), quantos)
              for sub, quantos in sorted(esperado.items())
              if chegou.get(sub, 0) != quantos]
    if not curtos:
        print(f"[hf] conferido: {dict(sorted(esperado.items()))} — "
              "cada pasta chegou com a contagem do disco.")
        return
    raise Problema(
        "o repositório NÃO tem o que saiu daqui:\n  - "
        + "\n  - ".join(f"{sub}/: {la} no Hub, {loc} no disco "
                         f"({'faltam ' + str(loc - la) if la < loc else 'sobram ' + str(la - loc)})"
                         for sub, la, loc in curtos)
        + "\n  Rode de novo com --allow-existing para completar. Um release pela "
          "metade no Hub é pior que nenhum: ele parece publicado.")


def publica(release: Path, repo_id: str, *, private: bool, allow_existing: bool) -> None:
    import huggingface_hub
    from huggingface_hub import HfApi

    # Antes de qualquer request: ver TIMEOUT_REQUEST_S.
    constantes = getattr(huggingface_hub, "constants", None)
    if constantes is not None and getattr(
            constantes, "DEFAULT_REQUEST_TIMEOUT", 0) < TIMEOUT_REQUEST_S:
        constantes.DEFAULT_REQUEST_TIMEOUT = TIMEOUT_REQUEST_S

    api = HfApi()                       # token vem do login/env; nunca impresso
    existe = True
    try:
        arquivos = api.list_repo_files(repo_id, repo_type="dataset")
    except Exception:
        existe, arquivos = False, []

    conteudo = [f for f in arquivos
                if f not in _ANDAIME and not f.startswith(".cache/")]
    if existe and conteudo and not allow_existing:
        raise Problema(
            f"{repo_id} já tem {len(conteudo)} arquivos de release "
            f"(ex.: {conteudo[:3]}).\n"
            "Publicar por cima sobrescreveria um release existente. Prefira um destino "
            "novo; se a intenção é mesmo atualizar este, passe --allow-existing.")

    if not existe:
        api.create_repo(repo_id, repo_type="dataset", private=private)
        print(f"[hf] repositório criado: {repo_id} (private={private})")

    n_arquivos, n_bytes = _tamanho(release)
    grande = (n_arquivos > LIMITE_COMMIT_UNICO_ARQUIVOS
              or n_bytes > LIMITE_COMMIT_UNICO_BYTES)
    print(f"[hf] enviando {release} → {repo_id} "
          f"({n_arquivos} arquivos, {n_bytes / 1024**3:.2f} GB, "
          f"{'vários commits' if grande else 'commit único'}) …")

    if grande:
        from huggingface_hub import CommitOperationAdd, CommitOperationDelete

        reagrupou = _agrupa_por_cena(release)
        cenas = _cenas_do_manifesto(release) if reagrupou else {}
        # `<scene_id>.png -> scene_id`, para o caso de `depth/` estar por cena e ainda
        # assim estourar o limite (mais de 10.000 cenas num release). Só quando o
        # release DECLARA `per_scene`: num release por amostra os arquivos de `depth/`
        # se chamam `<sample_id>.png`, e procurá-los neste índice não acharia nada —
        # eles ficariam soltos em `depth/`, que é o 400 que o reagrupamento existe para
        # evitar.
        depth_por_cena = ({f"{c}.png": c for c in set(cenas.values())}
                          if DEPTH_DIR in reagrupou
                          and _layout_do_release(release) is DepthLayout.PER_SCENE
                          else {})
        destinos = [(_caminho_no_repo(p.relative_to(release), cenas, reagrupou,
                                      depth_por_cena), p)
                    for p in _arquivos(release)]

        # Restos de uma tentativa anterior no layout plano ficariam no repositório para
        # sempre, e quem clonasse veria cada amostra duas vezes. Só as pastas que ESTE
        # upload vai reagrupar: apagar `depth/` plano num release que o mantém plano
        # apagaria o release.
        planos = sorted({f.split("/")[0] for f in conteudo
                         if f.split("/")[0] in reagrupou and f.count("/") == 1})
        if planos:
            print(f"[hf] apagando {planos} do layout plano de uma tentativa anterior …",
                  flush=True)
            _commita_esperando_o_limite(
                api, repo_id,
                operations=[CommitOperationDelete(path_in_repo=d, is_folder=True)
                            for d in planos],
                mensagem="remove o layout plano que o Hub recusa (10k arquivos/pasta)")

        lotes = [destinos[i:i + LOTE_POR_COMMIT]
                 for i in range(0, len(destinos), LOTE_POR_COMMIT)]
        for i, lote in enumerate(lotes, 1):
            print(f"[hf] commit {i}/{len(lotes)} ({len(lote)} arquivos) …", flush=True)
            _commita_esperando_o_limite(
                api, repo_id,
                operations=[CommitOperationAdd(path_in_repo=destino,
                                               path_or_fileobj=str(p))
                            for destino, p in lote],
                mensagem=(f"release, contrato metric_disparity_official_v1 "
                          f"({i}/{len(lotes)})"))
    else:
        api.upload_folder(
            folder_path=str(release), repo_id=repo_id, repo_type="dataset",
            commit_message="release, contrato metric_disparity_official_v1",
            ignore_patterns=IGNORAR)
    _confere_o_que_subiu(api, repo_id, release)
    print(f"[hf] pronto: https://huggingface.co/datasets/{repo_id}")


def main() -> int:
    p = argparse.ArgumentParser(description="Valida um release e publica no HF.")
    p.add_argument("--release-dir", required=True)
    p.add_argument("--repo-id", default="", help="ex.: akcit-pixel/bokehnet-regen-c")
    p.add_argument("--yes", action="store_true",
                   help="sobe de verdade. Sem isto, só valida e escreve o card.")
    p.add_argument("--public", action="store_true", help="default é privado")
    p.add_argument("--allow-existing", action="store_true")
    args = p.parse_args()

    release = Path(args.release_dir)
    try:
        resumo = valida(release)
    except Problema as exc:
        print(f"\n{exc}\n")
        return 1

    print(f"\n{'=' * 66}\n  release: {release}")
    for chave in ("amostras", "cenas", "por_split", "por_rota", "rota", "fontes",
                  "control_version", "depth_backend", "depth_layout",
                  "depth_arquivos", "k_censurado",
                  "sem_validador_analitico", "focus_sources", "refined",
                  "marcadas_como_refinadas", "pixel_gerado_sha256_conferidos",
                  "autocontido"):
        print(f"    {chave:<26} {resumo[chave]}")
    for aviso in resumo["avisos"]:
        print(f"  [aviso] {aviso}")
    print("=" * 66)

    if args.repo_id:
        card = escreve_card(release, resumo, args.repo_id)
        print(f"\n  card escrito: {card}")

    if not args.yes:
        print("\n  Nada foi enviado (falta --yes). Leia o card e o resumo acima.\n")
        return 0
    if not args.repo_id:
        print("\n  --yes sem --repo-id: para onde?\n")
        return 1

    try:
        publica(release, args.repo_id, private=not args.public,
                allow_existing=args.allow_existing)
    except Problema as exc:
        print(f"\n{exc}\n")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
