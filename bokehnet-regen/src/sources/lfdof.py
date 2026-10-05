"""Adaptador de fonte do LFDOF — enumera os pares (AIF real, bokeh real) da rota C.

Irmão de `sources/realbokeh.py`, e a outra metade da rota C: o paper exige o LFDOF em
três lugares (`paper.txt:326`, `:359`, `:528-529`) e a Fig. 3(c) o nomeia na legenda
(`paper.txt:296`). Sem ele o que existe é meia rota C
(`reference/ROTA_C_AUDITORIA.md:828-830`).

Uma origem só, e ela basta:

| origem | o que traz | acesso |
|---|---|---|
| `akcit-pixel/LFDOF` (privado) | as IMAGENS já pareadas: `image_focus` (AIF), `image_blur` (alvo), e o `file_name_base` que identifica cena e **nível** | token HF |

Diferença estrutural em relação à RealBokeh, e é ela que governa o módulo inteiro: o
LFDOF **não publica óptica**. Não há `metadata/<cena>.json`, não há f-number, não há
focal, não há distância de foco — porque não há fotografia com abertura: o desfoque é
sintetizado a partir de uma light field. Então este módulo não faz join com ninguém.
Ele parseia um nome e afirma o que o nome afirma.

## A estrutura REAL, medida em 2026-09-10

Medido lendo **só a coluna `file_name_base`** dos 83 shards parquet do repo privado,
por projeção de coluna sobre um file-like seekable em HTTP `Range` — o padrão que
`reference/ACHADOS.md` registra na seção "Como medir sem baixar imagem". Custo: 1,1 MB
por shard de ~500 MB, ~90 MB no total contra os 37 GB do repo.

```
config                                : default (único)
shards                                : 83  = 78 de `train` + 5 de `test`
linhas                                : 11.972 = 11.247 (train) + 725 (test)
row groups por shard                  : 5 em 83/83 shards
linhas por shard                      : 145 em 20 shards, 144 em 63 shards
colunas                               : image_pre_deblur, image_blur, image_focus,
                                        file_name_base
cenas distintas                       : 840 = 790 (train) + 50 (test)
file_name_base repetidos              : 0 / 11.972
(split, cena, nível) repetidos        : 0 / 11.972
```

Os totais 11.247 / 725 **confirmam** o que `reference/ACHADOS.md` já registrava `[M]`.
O que muda em relação ao que estava documentado são quatro coisas, todas medidas, e
duas delas contradizem suposições que a auditoria tinha registrado. Ver abaixo.

## O `scene_id` — de onde sai e por que agrupa certo

O nome é `lfdof_<split>_data_<cena>_level_<N>_<alinhamento>`, casado por regex estrito
em **11.972 de 11.972** linhas. `_data_` é literal e vem do gerador do espelho, do
mesmo jeito que `_f_` na RealBokeh não é f-number. O `<cena>` é o id da light field:
quatro dígitos, puramente numérico em 840 de 840 cenas, sem `_`.

```
scene_id = scene_key(source_split, scene_number) = "train_1275", "test_2464", ...
```

**Por que agrupa certo** — três medições, em ordem crescente de força:

1. Todas as variantes de uma AIF carregam o **mesmo `<cena>`**: os 15 níveis da cena
   1275 são `lfdof_train_data_1275_level_1..15_aligned`.
2. Cada cena vive em **exatamente um split**. Os números de cena de `train` e de `test`
   são **disjuntos**: `train ∩ test = 0` de 840. Logo nenhum agrupamento por cena
   atravessa a fronteira do split de origem.
3. **A prova em pixel**, e é a que fecha: o `image_focus` é **byte a byte idêntico**
   em todas as linhas de uma mesma cena. Medido em 90 linhas de 9 cenas, em 3 row
   groups de 3 shards diferentes (`train-00000` rg0, `train-00040` rg2,
   `test-00000` rg0): 0 cenas com mais de um sha256 de `image_focus`. Exemplo: os 15
   níveis da cena 1275 compartilham `d358a3847630…`.

   Ou seja: a chave `(split, cena)` particiona as linhas **exatamente** pelo conjunto
   que compartilha uma AIF. Não é inferência sobre o nome; é a igualdade dos bytes.

Errar isso seria vazamento garantido: com 15 desfocadas por AIF em 682 das 840 cenas,
um split por imagem poria a mesma AIF nos dois lados em praticamente toda cena, e a
métrica de validação subiria sem que nada denunciasse (`dataio/split.py:1-17`).

### A lição da RealBokeh **não** se repete aqui — e isso foi MEDIDO, não assumido

`reference/ACHADOS.md`, seção "O espelho publica os TRÊS splits", registra que na
RealBokeh a numeração de cena **reinicia em cada split**: as 220 cenas de `test`
reusam os números `1..220` que `train` também usa, e um `scene_id` cru fundiria três
cenas físicas numa só (split furado) e colidiria 2.495 `sample_id`.

No LFDOF a interseção medida é **0**:

```
train ∩ test (números de cena crus)   : 0   (790 e 50 cenas, nenhuma em comum)
(cena, nível) repetido IGNORANDO split: 0 / 11.972
```

**Mesmo assim a chave é qualificada pelo split.** Não é cerimônia:

- Qualificar é correto nos dois mundos. Se a numeração é global (o caso medido), a
  qualificação é injetiva e não perde nada; se ela reiniciasse, a qualificação é o que
  impede a fusão. Não qualificar só é correto num dos dois.
- Custa zero e mantém as duas fontes com a **mesma** definição de chave, então
  `dataio.split` vê um `scene_id` só, com uma regra só.
- O espelho pode ganhar um split `validation` depois — a RealBokeh ganhou, e a medição
  anterior dela ("publica só `train`") estava incompleta justamente por isso.

O que não fica implícito é a própria invariante: `scene_numbers_shared_between_splits`
mede a interseção e `enumeration_summary` a imprime. Se um dia der diferente de zero,
o run diz, em vez de a qualificação esconder que a mesma cena física caiu nos dois
lados.

## O que o LFDOF NÃO tem — e por que devolver `None` aqui é o CORRETO

`f_number`, `aif_f_number`, `focal_length_mm`, `focus_plane_distance_m`: **nenhum
existe**. Não é dado faltando, é grandeza que não se aplica — o desfoque vem de uma
light field, não de um diafragma. `PairSource` já declara os quatro como `Optional`
(`routes/route_c.py:83-90`), e o gate `aif_aperture_is_narrow` ganhou
`applicable=False` exatamente para este caso (`qc/gates.py:50-58`, `:318-324`): sem
isso, o `NaN` reprovava **100% do LFDOF** com o slug `gate_aif_aperture_wide`
("abertura larga"), que afirma algo falso sobre o dado — o defeito C2 de
`reference/ROTA_C_AUDITORIA.md:490-537`.

Por isso os quatro campos são `init=False`: **ninguém pode injetá-los**. Aceitar um
valor por parâmetro seria abrir a porta para `--f-number 2.8` "só para o validador
rodar", que é fallback numérico com outro nome. A ausência é estrutural e é afirmada
como estrutural.

**Consequência assumida**: `_analytic_k` (`routes/route_c.py:301-330`) devolve `None`
para toda amostra do LFDOF — o `if not (focal_length_mm and f_number and
focus_plane_distance_m)` já para na primeira condição. As 11.972 amostras do LFDOF
**não têm validador analítico**; o rótulo é o sweep da Eq. 5 e só ele. Isso é aceitável
e não rejeita nada: o validador audita o rótulo, não o produz. `publish_release.py`
conta e reporta `sem_validador_analitico` sem reprovar.

Em compensação — e é o motivo de o LFDOF valer o trabalho —, aqui a AIF é
**genuinamente all-in-focus**, renderizada da light field. Some o offset sistemático da
RealBokeh, onde a "AIF" é f/22 e o sweep mede só o borrão que FALTA. No LFDOF `K*` é o
K absoluto, comparável direto entre fontes, e é ele que calibra a leitura do K da
RealBokeh (`reference/ROTA_C_AUDITORIA.md:541-545`).

## `image_pre_deblur` NÃO é imagem de origem — é saída de modelo

O schema tem **quatro** colunas, não três. A terceira imagem é
`image_pre_deblur`, e ela é o pré-foco gerado pela **DRB-Net**, a variante opcional
§3.4 da DeblurNet (`../HANDOFF_PROJECT_HISTORY.md:120-128`,
`../genrefocus_deblurnet_paper/README.md:95`). Nunca foi usada no nosso treino.

Este módulo **não a referencia**, e a constante `LFDOF_UNUSED_COLUMN` existe para dizer
isso por escrito em vez de por omissão. Usá-la como AIF seria pôr uma imagem
*desborrada por uma rede* no lugar da all-in-focus real: a profundidade, a máscara e o
sweep de K sairiam todos de saída de modelo, e o rótulo passaria a depender de um
quarto modelo cujo hash não está na proveniência (`CLAUDE.md`, "Proveniência por
amostra"). É a forma exata do defeito D6 da RealBokeh, com outro nome — lá a "AIF" era
a foto de maior f-stop dentro de `gt/`.

Medido para não deixar dúvida sobre serem imagens diferentes: na cena 1275 os sha256 de
`image_pre_deblur` variam **por nível** (`c793c6197d04`, `ab0178dee47f`, …), enquanto o
`image_focus` é constante — o pré-deblur é função da desfocada, não da cena.

## A anotação de alinhamento EXISTE — e isso contradiz a auditoria

`reference/ROTA_C_AUDITORIA.md:546-548` (item 6) afirma: *"Registro geométrico é
perfeito por construção (mesma light field), então a anotação de alinhamento que a
RealBokeh publica não tem análogo — e não deve ser inventada."*

**Medido sobre as 11.972 linhas, o espelho do LFDOF publica a mesma anotação**, e ela
não é sempre `aligned`:

```
aligned          11.528   (96,29%)
misaligned          204   ( 1,70%)
shift_<X.Y>px       240   ( 2,00%)   31 valores distintos, de 2,0 a 5,0 px
                    ----
não-aligned         444   ( 3,71%)
```

São **444 linhas** em que a origem declara que o registro entre `image_focus` e
`image_blur` não fechou — quase o dobro da fração da RealBokeh (2,05% `[M]`). Não é
inventar campo: é ler o que a origem escreveu. O item 6 da auditoria estava `[A]`, e a
medição o derruba.

Distribuição por cena, que é o que importa para decidir o que fazer com isso:

```
cenas com alinhamento MISTO (aligned e não-aligned)  : 195 / 840
cenas 100% não-aligned                               :   4 / 840
```

Como 195 cenas são mistas, um gate de alinhamento derruba linhas, não cenas — o split
não muda de forma. **Consumir ou não é decisão da rota, não deste módulo**, que só
entrega o campo. O deslocamento vem em PIXELS; a resolução em que foi medido é
`LFDOF_IMAGE_HW` — ver a nota `[A]` na constante.

## `level` é 1-based, contíguo em 837 das 840 cenas — e é ORDINAL OPACO

```
menor nível observado em qualquer cena             : 1
cenas com nível 0                                  : 0
cenas com níveis EXATAMENTE 1..n, contíguos        : 837 / 840  (99,64%)
maior nível observado                              : 17  (1 cena)
níveis por cena                                    : 2 (3 cenas) … 15 (682 cenas), 17 (1)
```

As 3 cenas com buraco, e o buraco é **no MEIO**, não na cauda:

```
train 2022 : 14 níveis presentes, falta o 10  (1..9, 11..15)
train 3758 : 14 níveis presentes, falta o  8  (1..7,  9..15)
train 4863 : 14 níveis presentes, falta o  8  (1..7,  9..15)
```

Isto **difere** da RealBokeh, onde os níveis ausentes eram sempre a cauda em 11/11
cenas (`realbokeh.py:96-99`). Duas consequências, e as duas são de projeto:

1. **`scene_level_count` é a contagem OBSERVADA**, nunca `max(level)`. Nas três cenas
   acima `max(level)` daria 15 e a cena tem 14 pares. Um `scene_level_count` derivado
   do máximo mentiria sobre a diversidade efetiva, que é a razão de o campo existir.
2. **Nível ausente NÃO é rejeição.** Na RealBokeh o nível é índice em `target_avs` e um
   nível fora da lista rejeita, porque o clamp gravaria o f-number errado
   (`realbokeh.py:496-501`). Aqui não há lista, não há f-number, e não há nada a
   indexar: o par que existe é válido e o que falta simplesmente não existe no espelho.
   O buraco é **contado e impresso** por `enumeration_summary`, não engolido.

O que o nível **não** é: quantidade física. `level` é o ordinal da variante de desfoque
sintetizada da light field. **Não medimos**, e portanto não afirmamos, que ele seja
monótono no raio do desfoque, nem que seja comparável entre cenas. `[A]` de qualquer um
dos dois seria uma ordenação inventada, e nada aqui depende disso — o rótulo é o sweep
da Eq. 5 sobre os pixels, que não olha o nível.

## Resolução — única, e por isso declarada

```
(H, W) de image_focus e de image_blur : (688, 1008) em 90/90 linhas medidas
aif.shape == bokeh.shape              : 90/90
formato                               : PNG
modo                                  : RGBA nas duas colunas
alpha                                 : min = max = 255 em 90/90 (totalmente opaco)
```

`[M]` nas 90 linhas de 3 shards; `[A]` que valha para as 11.972 — a verificação por
amostra é do carregador (`sources/lfdof_images.py`), que **rejeita** resolução
diferente em vez de redimensionar. K vive em pixel e uma imagem que chega em outra
resolução muda o significado do rótulo sem mudar nada no JSON (`reference/CONTRATO.md`).

O canal alpha merece nota: `PIL.Image.convert("RGB")` sobre RGBA **compõe sobre preto**
em silêncio. Com alpha ≡ 255 isso é exatamente descartar um plano constante, sem perda.
Se alguma linha vier com alpha < 255, compor sobre preto seria inventar pixel — a
definição de fallback numérico. O carregador rejeita com
`source_image_alpha_not_opaque` em vez de compor.

## Sem fallback

Toda ausência levanta `SampleRejected` com slug de vocabulário fechado. Nenhum campo
tem default numérico, nenhum `.get(chave, <numero>)`, nenhum `except: pass`. Os quatro
campos ópticos são `None` **estrutural** e `init=False`, que é o oposto de fallback:
fallback é afirmar um número que não se mediu; isto é afirmar que a grandeza não existe.

## Slugs que este módulo precisa e que o contrato ainda NÃO registra

Dois, os dois do carregador de pixels, e `src/control/contract.py` **não foi editado**
(outra pessoa mantém o arquivo). Ver `NEW_REJECTION_REASONS` abaixo — o `_reject`
daqui já trata os dois casos, e nenhuma chamada muda quando eles forem registrados.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Iterable, Mapping, Optional

from control.contract import REJECTION_REASONS, SampleRejected
from control.contract import reject as _contract_reject
from qc.rejection import RejectionLog

# --------------------------------------------------------------------------------
# Identidade da origem
# --------------------------------------------------------------------------------

#: Espelho privado, já pareado. `image_focus` é a AIF, `image_blur` é o alvo.
LFDOF_DATASET = "akcit-pixel/LFDOF"

#: Fonte original pública do LFDOF, para auditoria e para a questão de licença:
#: `sweb.cityu.edu.hk/miullam/AIFNET/` -> `LFDOF.zip`, ~11 GB `[M]`. **Sem licença
#: explícita na página** `[M]` (`reference/ACHADOS.md`). Isso é bloqueante para
#: `--store-source-images` (release autocontido com os pixels dentro), não para o
#: rótulo. Referência bibliográfica `[52]` do paper: Ruan et al., AIFNet
#: (`paper.txt:873-875`).
UPSTREAM_URL = "https://sweb.cityu.edu.hk/miullam/AIFNET/"

#: Colunas do espelho que a rota C lê. Nomes, não caminhos: quem carrega a imagem
#: (`LoadPair` em `routes/route_c.py`) decide se lê do parquet, de disco ou de um mock.
LFDOF_AIF_COLUMN = "image_focus"
LFDOF_BOKEH_COLUMN = "image_blur"

#: A quarta coluna do schema, que este módulo **não** referencia em lugar nenhum.
#: Pré-foco gerado pela DRB-Net (variante §3.4 da DeblurNet), saída de MODELO e não
#: imagem de origem. Existe como constante para que a decisão de não usá-la esteja
#: escrita, e não apenas ausente — ver o cabeçalho do módulo.
LFDOF_UNUSED_COLUMN = "image_pre_deblur"

#: (H, W) das imagens. **[M]** em 90 linhas de 3 shards distintos; **[A]** para as
#: 11.972. Existe aqui por uma razão só: o contrato manda que toda quantidade em pixel
#: carregue a resolução em que foi medida, e `alignment_shift_px_at_source_hw` é uma
#: quantidade em pixel.
#:
#: **[A]** que o deslocamento anotado no nome tenha sido medido NESTA resolução. A favor:
#: os valores do LFDOF (2,0 a 5,0 px) e os da RealBokeh (2,0 a 4,9 px) caem na mesma
#: faixa, o que é o esperado se a mesma ferramenta de anotação rodou nas duas na
#: resolução nativa de cada uma. Não foi confirmado. Se um dia for confirmada outra
#: escala, o fator entra aqui e em lugar nenhum mais.
LFDOF_IMAGE_HW: tuple[int, int] = (688, 1008)

#: Splits que a origem publica. **São dois**, não três — medido: 78 shards `data/train-*`
#: e 5 shards `data/test-*`, e o split que o nome declara bate com o do shard em
#: 11.972/11.972 linhas. Não há `validation`. A RealBokeh ensinou que essa medição pode
#: ficar desatualizada quando a origem cresce, então `_parse_name` **rejeita** split
#: desconhecido em vez de aceitar qualquer palavra: um `validation` novo aparece como
#: `source_name_unparseable` no histograma, alto e claro, em vez de virar um terceiro
#: `scene_id` que ninguém pediu.
SOURCE_SPLITS = frozenset({"train", "test"})


# --------------------------------------------------------------------------------
# Rejeição
# --------------------------------------------------------------------------------

#: Slugs desta fonte que **já** estão em `control.contract.SOURCE_REJECTION_REASONS`.
#: Nenhum é novo: a camada de fonte do LFDOF falha das mesmas maneiras que a da
#: RealBokeh, menos as três que dependem de `metadata/<cena>.json` — que aqui não
#: existe e por isso não pode faltar.
SOURCE_REJECTION_REASONS = frozenset({
    #: `file_name_base` fora do padrão do espelho, ou com split que a origem não
    #: publica. Não é "nome feio": é um nome de que não dá para extrair cena nem
    #: nível, logo não dá para agrupar por cena, logo a amostra fica fora do split —
    #: que é vazamento com outro nome.
    "source_name_unparseable",
    #: Nível < 1. O nível do espelho é 1-based (medido: menor nível 1, zero cenas com
    #: nível 0, em 840 cenas). Diferente da RealBokeh, aqui o nível **não indexa**
    #: nada, então nível ausente no meio da sequência não é motivo de rejeição — só o
    #: nível impossível é.
    "source_level_out_of_range",
    #: Duas linhas do espelho com o mesmo (cena, nível). Sem este gate elas viram dois
    #: `sample_id` iguais, e o writer/retomada trata a segunda como a primeira.
    #: Medido: 0 em 11.972 hoje. O gate existe para o dia em que deixar de ser 0.
    "source_duplicate_sample",
    #: Os bytes da imagem no shard não decodificam, vêm vazios, ou a imagem chega numa
    #: resolução diferente da declarada. Usado pelo carregador de pixels.
    "source_image_unreadable",
})

#: **SLUGS NOVOS — precisam ser registrados em `control.contract`.**
#:
#: `src/control/contract.py` não foi editado de propósito: outra pessoa mantém o
#: arquivo, e o vocabulário fechado é dela. Enquanto estes dois não estiverem lá,
#: `_reject` levanta `SampleRejected` com o slug direto; `RejectionLog` agrega por
#: `exc.reason`, então nada se perde do histograma — o que falta é o conjunto ser
#: fechado de novo. Assim que forem registrados, `_reject` passa a usar
#: `contract.reject` sozinho, sem mudar nenhuma chamada.
#:
#: Por que são **dois slugs** e não um "erro de imagem" genérico, nem
#: `source_image_unreadable` reaproveitado: o histograma é o instrumento que denuncia
#: fallback, e ele só serve se cada bucket implicar um conserto diferente. "O PNG está
#: corrompido" (conserto: reconstruir o shard), "a coluna está trocada" (conserto: o
#: script que montou o espelho) e "há alpha não-opaco" (conserto: decidir e MEDIR uma
#: política de composição) são três consertos que não se parecem.
#: Os dois slugs que nasceram nesta fonte. **Já registrados** em
#: `control.contract.SOURCE_REJECTION_REASONS` — este conjunto permanece como
#: documentação de origem e é conferido por teste contra o contrato, para que um slug
#: novo aqui não escape do vocabulário fechado.
NEW_REJECTION_REASONS = frozenset({
    #: O `path` da célula contradiz o papel da coluna — p.ex. a coluna `image_focus`
    #: trazendo `..._blur_level_3_aligned.png`. Cada célula do parquet carrega o nome
    #: do arquivo original, e ele nomeia o papel (`focus`, `blur`, `pre-deblur`):
    #: medido, bate em 180/180 células conferidas. É a checagem cruzada que este
    #: módulo tem no lugar da que a RealBokeh faz contra `target_avs` — e o defeito
    #: que ela pega é o pior possível, porque colunas trocadas produzem um dataset
    #: inteiro plausível com AIF e alvo invertidos, e nada mais denuncia.
    "source_image_role_mismatch",
    #: RGBA com alpha < 255. `convert("RGB")` comporia sobre preto em silêncio, o que
    #: é inventar pixel onde a origem declarou transparência — fallback numérico na
    #: entrada da profundidade, da máscara e do sweep de SSIM. Medido: alpha ≡ 255 em
    #: 90/90 linhas, então na prática este slug não deve aparecer; se aparecer, é
    #: descoberta e não ruído.
    "source_image_alpha_not_opaque",
})

#: Tudo que este módulo (mais o carregador) pode levantar.
ALL_REJECTION_REASONS = SOURCE_REJECTION_REASONS | NEW_REJECTION_REASONS


def reject_source(reason: str, detail: str = "") -> None:
    """Único caminho de rejeição desta FONTE — o enumerador e o carregador usam este.

    Público de propósito: `sources/lfdof_images.py` precisa levantar os mesmos slugs, e
    um segundo `_reject` privado lá seria a semente de dois vocabulários divergentes
    para a mesma fonte.

    Delega ao contrato, e só. Os dois slugs próprios desta fonte
    (`source_image_role_mismatch` e `source_image_alpha_not_opaque`) já estão
    registrados em `control.contract.SOURCE_REJECTION_REASONS`, então este módulo não
    mantém vocabulário paralelo: slug inventado levanta `KeyError` lá, e o histograma
    agrega junto com o das rotas e o da RealBokeh.
    """
    _contract_reject(reason, detail)


#: Alias interno, no molde de `realbokeh._reject`.
_reject = reject_source


# --------------------------------------------------------------------------------
# O nome do espelho
# --------------------------------------------------------------------------------

#: `lfdof_<split>_data_<cena>_level_<N>_<alinhamento>`
#:
#: `_data_` é literal e vem do gerador do espelho. O `<cena>` é o id da light field e
#: não contém `_` (medido: 840 de 840 são 4 dígitos puramente numéricos).
#: `<alinhamento>` é `aligned`, `misaligned` ou `shift_<X.Y>px` (medido: 33 formas
#: distintas, 11.528 + 204 + 240 = 11.972 linhas).
#:
#: O split é enumerado na alternância em vez de `[a-z]+` de propósito — ver a nota em
#: `SOURCE_SPLITS`.
_NAME_RE = re.compile(
    r"^lfdof"
    r"_(?P<split>train|test)"
    r"_data_(?P<scene>[0-9]+)"
    r"_level_(?P<level>[0-9]+)"
    r"_(?P<alignment>aligned|misaligned|shift_(?P<shift_px>[0-9]+(?:\.[0-9]+)?)px)$"
)


def scene_key(split: str, scene_number: str | int) -> str:
    """`(split, numero)` -> chave de cena. **Única definição** desta chave no LFDOF.

    Mesma forma que `realbokeh.scene_key`, de propósito: `dataio.split` vê um
    `scene_id` só, com uma regra só. Medido no LFDOF, os números de cena de `train` e
    `test` são disjuntos (interseção 0 de 840), então a qualificação não é o que salva
    o agrupamento aqui — ela é o que o mantém correto se a origem mudar, e o que
    mantém as duas fontes com a mesma chave. Ver o cabeçalho do módulo.
    """
    return f"{split}_{scene_number}"


ALIGNMENT_ALIGNED = "aligned"
ALIGNMENT_MISALIGNED = "misaligned"
ALIGNMENT_SHIFT = "shift"


@dataclass(frozen=True)
class ParsedName:
    """Tudo que o `file_name_base` afirma, e nada além disso."""

    #: **Qualificado pelo split.** Ver `scene_key` e o cabeçalho do módulo.
    scene_id: str
    #: O número cru, como aparece no nome. É o id da light field no LFDOF original.
    scene_number: str
    level: int
    source_split: str
    alignment: str                              # "aligned" | "misaligned" | "shift"
    #: Deslocamento residual anotado pela origem, em PIXELS de `LFDOF_IMAGE_HW`.
    #: `None` quando o nome não anota deslocamento — o que é **diferente de zero**.
    #: `alignment == "misaligned"` é exatamente esse caso: a origem diz que não fechou
    #: e NÃO diz quanto. Gravar 0,0 aqui seria fallback numérico, e ainda por cima o
    #: mais perigoso: 0,0 é o valor de "perfeitamente alinhado".
    alignment_shift_px_at_source_hw: Optional[float]
    raw: str


def _parse_name(name: str) -> ParsedName:
    if not isinstance(name, str) or not name:
        _reject("source_name_unparseable", f"nome vazio ou não-string: {name!r}")
    match = _NAME_RE.match(name)
    if match is None:
        _reject("source_name_unparseable",
                f"{name!r} fora do padrão "
                f"lfdof_<split>_data_<cena>_level_<N>_<alinhamento>, com "
                f"<split> em {sorted(SOURCE_SPLITS)} e <alinhamento> em "
                "aligned|misaligned|shift_<X.Y>px")

    split = match.group("split")
    if split not in SOURCE_SPLITS:
        # Inalcançável com o regex de hoje, e de propósito: é a afirmação de que a
        # lista de splits vive em `SOURCE_SPLITS`, no lugar onde ela é usada. Se
        # alguém relaxar o regex para `[a-z]+`, esta linha é que segura.
        _reject("source_name_unparseable",
                f"{name!r}: split {split!r} fora de {sorted(SOURCE_SPLITS)}")

    level = int(match.group("level"))
    if level < 1:
        # Redundante com o regex de hoje (`[0-9]+` aceita "0"), e de propósito: é a
        # afirmação de que o nível é 1-based, no lugar onde ela é usada.
        _reject("source_level_out_of_range",
                f"{name!r}: nível {level} < 1. O nível do espelho é 1-based "
                "(menor nível observado 1, zero cenas com nível 0, em 840 cenas)")

    shift_text = match.group("shift_px")
    if shift_text is None:
        alignment = match.group("alignment")     # "aligned" ou "misaligned"
        shift_px: Optional[float] = None
    else:
        alignment = ALIGNMENT_SHIFT
        shift_px = float(shift_text)

    return ParsedName(
        scene_id=scene_key(split, match.group("scene")),
        scene_number=match.group("scene"),
        level=level,
        source_split=split,
        alignment=alignment,
        alignment_shift_px_at_source_hw=shift_px,
        raw=name,
    )


def parse_file_name_base(name: str) -> tuple[str, int]:
    """`file_name_base` -> `(scene_id, level)`.

    **Rejeita** nome fora do padrão com `SampleRejected("source_name_unparseable")`.
    Não devolve `None`: um `None` silencioso vira `scene_id=None` no manifesto, e uma
    amostra sem cena é uma amostra fora do split — que é vazamento com outro nome.

    Para o resto do que o nome afirma (split e anotação de alinhamento), use
    `parse_full_name`.
    """
    parsed = _parse_name(name)
    return parsed.scene_id, parsed.level


def parse_full_name(name: str) -> ParsedName:
    """Como `parse_file_name_base`, mas devolve split e alinhamento também."""
    return _parse_name(name)


# --------------------------------------------------------------------------------
# O par
# --------------------------------------------------------------------------------

@dataclass(frozen=True)
class LFDOFPair:
    """Um par (AIF real, bokeh real) do LFDOF, com o que a origem afirma sobre ele.

    Satisfaz o `PairSource` de `routes/route_c.py` — `scene_id`, `sample_id`,
    `source_dataset`, `source_sample_id`, `source_split`, `aif_ref`, `bokeh_ref`,
    `f_number`, `focal_length_mm`, `focus_plane_distance_m`, `aif_f_number`.

    Ao contrário do `RealBokehPair`, em que nenhum campo é `Optional` porque a origem
    publica todos, aqui **quatro campos são estruturalmente `None`**: o LFDOF não tem
    óptica para publicar. Eles são `init=False` para que ninguém possa injetá-los —
    ver o cabeçalho do módulo, seção "O que o LFDOF NÃO tem".

    Não há campo `focus_plane_uncertainty_m` nem `raw_*_path`: a origem não publica
    distância de foco (logo não há barra de erro a carregar) e o repo público upstream
    é um único `LFDOF.zip` sem layout por cena estável no qual apontar. O que existe de
    verificável por linha é o `path` de cada célula do parquet, e ele é conferido pelo
    carregador (`sources/lfdof_images.py`), não gravado aqui.
    """

    scene_id: str
    level: int
    sample_id: str
    source_dataset: str
    source_sample_id: str
    source_split: str

    #: Onde as imagens estão, no espelho. São nomes de COLUNA — quem carrega decide de
    #: onde lê. Ver `LoadPair` em `routes/route_c.py`.
    aif_ref: str
    bokeh_ref: str

    #: Anotação de alinhamento da própria origem, que **existe** no LFDOF (3,71% de
    #: linhas não-`aligned`, medido). Ver o cabeçalho do módulo.
    alignment: str
    alignment_shift_px_at_source_hw: Optional[float]

    #: Quantos pares esta cena tem no espelho — a contagem **OBSERVADA**, nunca
    #: `max(level)`: 3 cenas têm buraco no meio da sequência e o máximo mentiria em 1.
    #: Sem este campo, "11.247 amostras" esconde que elas vêm de 790 cenas, e é essa
    #: razão que diz qual é a diversidade efetiva (`CLAUDE.md`, "Contagens em cenas E
    #: em amostras").
    scene_level_count: int

    # -- o que o LFDOF NÃO publica ---------------------------------------------
    # `init=False`: ausência estrutural, não campo a preencher. Ver o cabeçalho.
    #: f-number do ALVO. O desfoque vem de light field, não de diafragma.
    f_number: Optional[float] = field(default=None, init=False)
    #: f-number da AIF. `None` faz `aif_aperture_is_narrow` devolver
    #: `applicable=False` (`qc/gates.py:318-324`) — inaplicável, não reprovado.
    aif_f_number: Optional[float] = field(default=None, init=False)
    #: Os dois termos da Eq. 3 que a origem não tem. Consequência: `_analytic_k`
    #: devolve `None` e o LFDOF não tem validador analítico do sweep.
    focal_length_mm: Optional[float] = field(default=None, init=False)
    focus_plane_distance_m: Optional[float] = field(default=None, init=False)

    @property
    def is_aligned(self) -> bool:
        """`True` só quando a origem AFIRMA alinhamento. `misaligned` e `shift_*px`
        são `False`, e a diferença entre eles fica em `alignment`."""
        return self.alignment == ALIGNMENT_ALIGNED

    @property
    def has_analytic_validator(self) -> bool:
        """Sempre `False` no LFDOF, e é afirmação, não acidente.

        Existe para que um relatório possa contar "quantas amostras têm validador"
        sem reimplementar a condição de `_analytic_k` — e para que a resposta do LFDOF
        seja legível como escolha da origem, não como bug do pipeline.
        """
        return False


def _sample_id(scene_id: str, level: int) -> str:
    """`c_lfdof_<split>_<cena>_l<nivel>`.

    Não usa o `file_name_base` cru de propósito: ele carrega a anotação de alinhamento,
    então re-anotar um par mudaria o `sample_id`, e a retomada por
    `RejectionLog.completed_ids()` reprocessaria a amostra como se fosse nova. O
    `file_name_base` original fica inteiro em `source_sample_id`.

    O prefixo `c_lfdof_` separa do `c_realbokeh_` da outra fonte da mesma rota: as duas
    escrevem no mesmo release, e id de amostra colidindo entre fontes é a mesma classe
    de defeito que o `source_duplicate_sample` dentro de uma fonte.
    """
    return f"c_lfdof_{scene_id}_l{level}"


def pair_from_name(
    file_name_base: str,
    *,
    scene_level_count: int,
    source_dataset: str = LFDOF_DATASET,
) -> LFDOFPair:
    """Constrói UM par, ou levanta `SampleRejected` com o motivo.

    É o caminho unitário: `enumerate_pairs` é o laço com o histograma em volta.
    Separado para poder ser testado por caso de falha, um a um.

    `scene_level_count` é **obrigatório e sem default**. Ele não sai do nome — sai de
    contar as linhas daquela cena, que é informação de corpus e não de linha. Um
    default (`1`, ou `max(level)`) seria um número inventado por este módulo sobre a
    diversidade do dataset, que é exatamente o formato do fallback que o contrato
    proíbe. Quem tem só um nome na mão não sabe quantos irmãos ele tem, e a assinatura
    diz isso.
    """
    parsed = _parse_name(file_name_base)
    count = int(scene_level_count)
    if count < 1:
        raise ValueError(
            f"scene_level_count = {scene_level_count!r} para {file_name_base!r}: a "
            "cena tem pelo menos este par, então a contagem é >= 1. Valor errado aqui "
            "não é dado faltando da origem, é bug de quem contou — e por isso levanta "
            "ValueError em vez de rejeitar a amostra."
        )
    return LFDOFPair(
        scene_id=parsed.scene_id,
        level=parsed.level,
        sample_id=_sample_id(parsed.scene_id, parsed.level),
        source_dataset=source_dataset,
        source_sample_id=file_name_base,
        source_split=parsed.source_split,
        aif_ref=LFDOF_AIF_COLUMN,
        bokeh_ref=LFDOF_BOKEH_COLUMN,
        alignment=parsed.alignment,
        alignment_shift_px_at_source_hw=parsed.alignment_shift_px_at_source_hw,
        scene_level_count=count,
    )


def enumerate_pairs(
    file_name_bases: Iterable[str],
    *,
    log: RejectionLog,
    source_dataset: str = LFDOF_DATASET,
) -> list[LFDOFPair]:
    """Enumera os pares do LFDOF a partir dos `file_name_base` do espelho.

    Não recebe metadata: não existe metadata a receber. A única entrada é a lista de
    nomes, que é o que `MirrorIndex.names()` devolve já na ordem de leitura sequencial.

    `log` é **obrigatório e sem default**. Um default `None` transformaria a chamada
    curta — que é a que todo mundo escreve — em descarte silencioso, e a regra do
    projeto é que rejeição sem motivo registrado não existe. Um `RejectionLog()` sem
    `path` custa nada e já agrega o histograma em memória.

    **Duas passagens, e a primeira é o motivo.** `scene_level_count` só existe depois
    de ver todos os nomes da cena, então a passagem 1 parseia e conta, e a passagem 2
    constrói. Um nome que não parseia é rejeitado **uma vez só**, na passagem 1, e não
    entra na contagem de nenhuma cena — o que é o certo: uma linha que não se sabe de
    que cena é não pode inflar a diversidade de cena nenhuma.

    A ordem de entrada é preservada na saída, para que
    `mirror_images.order_pairs_for_sequential_read` continue valendo e o carregador
    consiga manter um shard aberto por vez.

    **Chame com o conjunto COMPLETO de nomes.** `scene_level_count` e
    `scenes_with_level_gaps` descrevem o que esta chamada viu: enumerar um subconjunto
    — um shard só, um row group só, uma amostra de piloto — produz contagens do
    subconjunto, não do dataset, e um "14 níveis" que na verdade é 15 mentiria sobre a
    diversidade no manifesto. Medido no smoke: enumerar só o row group 0 do shard 0
    corta a cena 1279 no nível 8 e o diagnóstico reporta um buraco que não existe no
    espelho inteiro. O fluxo certo é o de `run_route_c.py`: enumera tudo a partir de
    `MirrorIndex.names()`, e só DEPOIS sorteia o piloto com `sample_pairs_for_pilot`.
    """
    parsed_ok: list[ParsedName] = []
    for name in file_name_bases:
        try:
            parsed_ok.append(_parse_name(name))
        except SampleRejected as exc:
            # `sample_id` do nome cru: um nome que não parseia não tem `sample_id`
            # nosso, e inventar um esconderia a linha no JSONL.
            log.reject_from(str(name), exc, {"source_dataset": source_dataset})

    per_scene = Counter(p.scene_id for p in parsed_ok)

    pairs: list[LFDOFPair] = []
    seen: dict[str, str] = {}                 # sample_id -> file_name_base que o criou
    for parsed in parsed_ok:
        pair = pair_from_name(parsed.raw,
                              scene_level_count=per_scene[parsed.scene_id],
                              source_dataset=source_dataset)
        if pair.sample_id in seen:
            try:
                _reject("source_duplicate_sample",
                        f"{parsed.raw!r} e {seen[pair.sample_id]!r} produzem o mesmo "
                        f"sample_id {pair.sample_id!r}. Medido: 0 colisões em 11.972 "
                        "linhas — se isto disparou, o espelho mudou.")
            except SampleRejected as exc:
                log.reject_from(str(parsed.raw), exc, {"scene_id": pair.scene_id})
            continue
        seen[pair.sample_id] = parsed.raw
        pairs.append(pair)
        log.accept(pair.sample_id, {"scene_id": pair.scene_id, "level": pair.level,
                                    "source_split": pair.source_split})
    return pairs


# --------------------------------------------------------------------------------
# Split — o que alimenta `dataio.split.split_from_source`
# --------------------------------------------------------------------------------

def scene_source_splits(pairs: Iterable[LFDOFPair]) -> dict[str, str]:
    """`{scene_id: source_split}`, pronto para `dataio.split.split_from_source`.

    O LFDOF já separa `train` / `test` **por cena** (790 e 50 cenas, medido), e reusar
    isso é melhor que sortear: preserva a intenção de quem montou o dataset e mantém
    comparabilidade com quem já publicou número nele.

    Uma cena que aparecer em dois splits é `ValueError`, não "o último ganha": cena nos
    dois lados é vazamento, e o LFDOF entrega de 2 a 17 desfocadas da MESMA AIF — é o
    caso em que um split por imagem infla a validação sem que nada denuncie.

    Nota de uso, medida: aqui os dois lados **existem** no espelho (ao contrário da
    RealBokeh, cujo espelho por muito tempo pareceu publicar só `train`), então
    `split_from_source` produz um `SceneSplit` com validação não vazia sem ninguém
    precisar sortear. As 50 cenas de `test` são 5,95% das 840, perto do
    `--val-fraction 0.05` que `run_route_c.py` usa por default.
    """
    out: dict[str, str] = {}
    for pair in pairs:
        if pair.source_split not in SOURCE_SPLITS:
            raise ValueError(
                f"cena {pair.scene_id!r}: split de origem {pair.source_split!r} fora de "
                f"{sorted(SOURCE_SPLITS)}"
            )
        previous = out.get(pair.scene_id)
        if previous is not None and previous != pair.source_split:
            raise ValueError(
                f"cena {pair.scene_id!r} aparece em dois splits de origem: "
                f"{previous!r} e {pair.source_split!r}. Cena nos dois lados é "
                "vazamento — a unidade do split é a CENA, nunca a amostra."
            )
        out[pair.scene_id] = pair.source_split
    if not out:
        raise ValueError("nenhum par: não há split a montar")
    return out


def scene_numbers_shared_between_splits(
    pairs: Iterable[LFDOFPair],
) -> dict[str, set[str]]:
    """`{numero_de_cena: {splits}}` para os números que aparecem em MAIS de um split.

    Medido hoje: **vazio** (interseção `train ∩ test` = 0 sobre 840 cenas). Esta função
    existe para que essa medição continue sendo medição.

    O `scene_id` é qualificado pelo split, e essa qualificação tem um custo escondido:
    se um dia a origem passar a reusar números entre splits — como a RealBokeh faz —,
    `train_1275` e `test_1275` viram duas cenas distintas em silêncio, e se por trás
    delas houver a MESMA light field, a mesma cena física estará nos dois lados do
    split sem que `scene_source_splits` levante nada. É a única forma de vazamento que
    a qualificação não pega, então ela é medida e impressa em vez de suposta.

    Devolver dicionário em vez de levantar é deliberado: reuso de número **não é
    prova** de cena repetida (na RealBokeh é reuso legítimo, com cenas físicas
    diferentes). O que é prova, se um dia isto der não-vazio, é comparar o sha256 do
    `image_focus` dos dois lados — o carregador já registra esse sha no ledger.
    """
    por_numero: dict[str, set[str]] = defaultdict(set)
    for pair in pairs:
        # `scene_id` é `f"{split}_{numero}"`; o número é o que sobra depois do split.
        numero = pair.scene_id[len(pair.source_split) + 1:]
        por_numero[numero].add(pair.source_split)
    return {n: s for n, s in por_numero.items() if len(s) > 1}


def scenes_with_level_gaps(pairs: Iterable[LFDOFPair]) -> dict[str, list[int]]:
    """`{scene_id: [niveis ausentes]}` para as cenas cuja sequência não é `1..n`.

    Medido: **3** das 840 cenas, e o buraco é no MEIO (train 2022 sem o nível 10,
    train 3758 e train 4863 sem o 8). Não é rejeição — não há nada indexado pelo nível
    no LFDOF, o par que existe é válido e o que falta não existe no espelho. É
    diagnóstico: `enumeration_summary` imprime a contagem, porque um buraco que
    aparecesse em 300 cenas em vez de 3 significaria que o espelho perdeu linhas, e
    isso tem que ser visível sem ninguém ir procurar.
    """
    niveis: dict[str, set[int]] = defaultdict(set)
    for pair in pairs:
        niveis[pair.scene_id].add(pair.level)
    out: dict[str, list[int]] = {}
    for scene_id, levels in niveis.items():
        faltando = sorted(set(range(1, max(levels) + 1)) - levels)
        if faltando:
            out[scene_id] = faltando
    return dict(sorted(out.items()))


# --------------------------------------------------------------------------------
# Contagens — em cenas E em amostras
# --------------------------------------------------------------------------------

def enumeration_summary(pairs: list[LFDOFPair], log: RejectionLog) -> str:
    """O relatório que fecha a enumeração.

    Imprime cena E amostra, porque 11.247 amostras vindas de 790 cenas não são 11.247
    unidades de diversidade — e imprime três coisas que o LFDOF exige e a RealBokeh
    não: a distribuição de alinhamento (que a auditoria supunha não existir aqui), a
    contagem de cenas com buraco de nível, e a interseção de números de cena entre
    splits, que é a invariante em que o `scene_id` qualificado se apoia.

    Termina com o histograma de `log`, que é o que todo run tem que imprimir.
    """
    per_scene = Counter(p.scene_id for p in pairs)
    lines = [
        "",
        "=" * 62,
        f"  fonte     : {LFDOF_DATASET}  (sem metadata: a origem não publica óptica)",
        f"  pares     : {len(pairs)}",
        f"  cenas     : {len(per_scene)}",
    ]
    if per_scene:
        lines.append("  níveis/cena: " + " · ".join(
            f"{k} -> {v}" for k, v in sorted(Counter(per_scene.values()).items())))

    alignment = Counter(p.alignment for p in pairs)
    if alignment:
        nao_alinhados = len(pairs) - alignment.get(ALIGNMENT_ALIGNED, 0)
        lines.append("  alinhamento: " + " · ".join(
            f"{k} -> {v}" for k, v in alignment.most_common()))
        lines.append(f"               não-aligned: {nao_alinhados} "
                     f"({100 * nao_alinhados / max(len(pairs), 1):.2f}%) — a origem "
                     "DECLARA que o registro não fechou nestas linhas")

    splits = Counter(p.source_split for p in pairs)
    lines.append("  split de origem: " + " · ".join(
        f"{k} -> {v}" for k, v in splits.most_common()))
    cenas_por_split = Counter()
    for scene_id, split in {p.scene_id: p.source_split for p in pairs}.items():
        cenas_por_split[split] += 1
    lines.append("  cenas por split: " + " · ".join(
        f"{k} -> {v}" for k, v in cenas_por_split.most_common()))

    gaps = scenes_with_level_gaps(pairs)
    lines.append(f"  cenas com buraco na sequência de níveis: {len(gaps)}"
                 + (f"  (ex.: " + ", ".join(f"{k} sem {v}" for k, v in
                                            list(gaps.items())[:3]) + ")" if gaps else ""))

    compartilhados = scene_numbers_shared_between_splits(pairs)
    lines.append(f"  números de cena em mais de um split: {len(compartilhados)}"
                 f"  (medido no espelho: 0 — o scene_id qualificado se apoia nisto)")

    lines.append("  validador analítico (Eq. 3): AUSENTE em 100% — a origem não "
                 "publica f-number,")
    lines.append("                               focal nem distância de foco. O "
                 "rótulo é o sweep da Eq. 5.")
    lines.append("=" * 62)
    return "\n".join(lines) + log.summary()
