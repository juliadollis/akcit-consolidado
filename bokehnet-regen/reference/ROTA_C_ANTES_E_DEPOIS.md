# Rota C — como era, o que mudamos, e o que falta rodar

Documento de referência da rota C: §3.2(c) do GenRefocus, os pares reais de LFDOF e
RealBokeh, com K calibrado pelo sweep da Eq. 5.

---

## 1. Como a rota C era

O pipeline original (`bokehnet-preprocessing`, gerou o dataset v0). Cada item abaixo é
um defeito medido, com a consequência que ele teve no rótulo.

### A "AIF" não era all-in-focus

A imagem de referência era escolhida como **o maior f-stop dentro de `gt/`** — a pasta
dos alvos. A mediana por cena era **f/14**, e em **12,7%** das cenas era f/5.6 ou mais
aberto; em **3,0%**, f/2.8.

Ou seja: numa cena a cada oito, a profundidade, a máscara e o sweep de K saíam todos de
uma foto que já tinha bokeh forte. A AIF verdadeira existia o tempo todo, em
`train/in/<id>_f22.JPG`, que é f/22 em 100% das cenas — e foi confirmada byte a byte
contra a coluna `image_focus` do espelho.

### K censurado em quase metade das amostras

A busca usava `--k-max 300`, escolhido a priori. Resultado: **`k == 300` exato em 1.379
de 2.932 amostras — 47,0%**. Metade do dataset tinha um rótulo que não era o K da cena,
era o teto da busca. Nada media a distribuição, então ninguém viu.

### `max_coc` variando por amostra

Era flag de CLI, uma por rota, todas independentes. Um normalizador que muda por amostra
é um segundo rótulo escondido dentro do primeiro — e foi assim que o "kfix" acabou
rodando **duas convenções de normalização no mesmo batch**, com RealBokeh subindo
(+0,4832) e RealDOF caindo (−0,4599) ao mesmo tempo.

### Cascata de modelos que mentia na proveniência

`BiRefNet → RMBG → GrabCut` com `except` nu, e `mask_source` continuando a dizer
`"automatic"` depois de cair no GrabCut. O mesmo padrão na profundidade: Depth Pro caindo
para Depth Anything, que devolve **disparidade** e não profundidade métrica, normalizada
igual e gravada igual — a amostra saía espelhada sem deixar rastro.

### Split por imagem

A RealBokeh entrega de 2 a 21 aberturas da **mesma cena**. Com split por imagem, a mesma
cena aparecia nos dois lados: a validação media memorização, não generalização.

### Sem histograma de rejeição, sem proveniência, sem resolução

Não havia registro de por que uma amostra sumia, nem hash dos modelos que produziram o
rótulo, nem a resolução em que K foi medido — e K é um número **em pixel**.

---

## 2. O que fizemos

### O contrato, numa implementação só

```python
z          = depth_pro(aif)              # METROS
disp       = 1.0 / z                     # 1/m
focus_disp = median(disp[mask])          # mediana NA disparidade, não 1/median(z)
K          = k_eq3 / 1000.0              # a Eq. 3 é em mm; a inferência é em 1/m
max_coc    = 100.0                       # CONGELADO, global, não é campo nem parâmetro
defocus    = clip(abs(K*(disp - focus_disp)) / max_coc, 0.0, 1.0)
```

`src/control/contract.py` é a única implementação. `validate_metadata` **rejeita**
metadado com `max_coc != 100.0` — o normalizador escondido não tem por onde entrar.

### A AIF correta, provada

`image_focus` do espelho **é** `train/in/<id>_f22.JPG`, confirmado por sha256. O f-number
da AIF é **lido** do metadado da origem (`source_av`, f/22 em 4.400/4.400 cenas), nunca
assumido, e alimenta um gate dedicado.

### Um modelo por papel, com hash

Depth Pro e BiRefNet, sem cascata. Se falham, a amostra é **rejeitada com slug
registrado**. O sha256 de cada modelo entra na proveniência de cada amostra — e o hash
ignora o cache do HF, que carrega etag e horário de download e faria dois snapshots
idênticos parecerem modelos diferentes.

### O renderizador verificado em GPU

BokehMe (`[43]`), com laudo: raio **linear em K**, resíduo relativo do ajuste ~0. Sem o
laudo o pipeline **recusa rodar**, e ele confere que o laudo é do mesmo checkout por
sha256 de `arnet.pth`, `iunet.pth` e do `pipeline` extraído por AST.

### O refinamento do plano de foco — a mudança de método

Medimos no piloto: a máscara do BiRefNet acerta o plano de foco em **35,2%** das amostras
(contra a distância medida que a RealBokeh publica a ±1 cm), com razão mediana **0,579** —
o plano escolhido fica ~1,7× mais longe do que estava. E **20,6%** das amostras eram
**descartadas** por máscara vazia.

A causa: a RealBokeh é feita de **cenas**, não de fotos de objeto — um tronco de árvore
num parque, um muro de pedra. O BiRefNet segmenta objeto **saliente**, fora do domínio
dele. Em 1/3 das cenas devolve probabilidade **exatamente 0,000**: está declinando.

O paper antecipa isso, nestes mesmos dois datasets, e **rejeita explicitamente descartar**
(`paper.txt:359-368`): ele corrige M à mão, re-selecionando "uma região pequena porém
confiável em foco". Substituímos o passo manual por um automático que usa a definição
física de estar em foco:

```
retencao(x) = media_local(|laplaciano(bokeh)|) / media_local(|laplaciano(aif)|)
```

Perto de 1 no plano de foco, perto de 0 no que borrou. O denominador **normaliza pela
textura da própria cena** — é o que faz medida de nitidez absoluta falhar, porque
folhagem desfocada tem mais energia de alta frequência que parede lisa em foco.

Toda amostra grava `focus_source` e `focus_was_refined`, mais
`focus_disparity_from_initial_mask` — o rótulo que ela teria tido **sem** refino, que é o
que torna a validação **pareada** em vez de confundida.

### O teto de 4 níveis por cena

O paper descreve *"2 to 4 images per set"* (`paper.txt:1001-1003`). Com teto de 4 no split
`train`: **13.799 amostras**, contra os "13K" que ele publica — e a anotação de 8 h a 4–8 s
por imagem dá 3.600 a 7.200 máscaras, contra nossas 4.399 cenas. Duas contas independentes
fecham. Sem teto: 20.495, 58% acima.

Corrige também um desequilíbrio que a contagem escondia: **244 cenas (6,2%) geravam 5.124
amostras (25%)**. Nenhuma cena é descartada — só perdem peso as que tinham demais.

### Split por cena, materializado, herdado da origem

O espelho publica os três splits (`train` 20.495 / `test` 1.257 / `validation` 1.238), e
a numeração de cena **reinicia em cada um** — `train_1`, `test_1` e `validation_1` são
cenas físicas diferentes. Sem qualificar a chave pelo split, 2.495 amostras boas seriam
descartadas pelo gate de duplicata, e metadata de uma cena seria aplicada a outra.

### Tudo que se recusa a passar em silêncio

Trinta e dois slugs de rejeição num vocabulário **fechado** — slug inventado levanta
`KeyError`. Todo run termina imprimindo o histograma. Dez gates de qualidade, **todos em
modo medir** até o piloto congelar os números. Um validador de release com onze
checagens, **cada uma com um teste que a faz reprovar**.

---

## 3. O pipeline, como ele roda

```
par real (AIF, bokeh) — RealBokeh ou LFDOF
        │
        ├─ enumeração: cena, nível, f-number, alinhamento, sha256 dos bytes
        ├─ teto de 4 níveis por cena, espaçados uniformemente
        │
        ├─► Depth Pro (na AIF) ──────────► D, em METROS
        │
        ├─► BiRefNet (na AIF) ───────────► M inicial
        │        │
        │   retenção de detalhe (AIF x bokeh) ──► refinamento
        │        │                                    │
        │        └──────────────► M final ────────────┘   + focus_source gravado
        │                            │
        │                     D_focus = mediana(1/D[M])      (Eq. 4, na disparidade)
        │
        ├─► sweep de K (Eq. 5):
        │       K* = argmax SSIM( BokehMe(AIF, D; D_focus, K), bokeh_real )
        │       seção áurea, com censura detectada e gravada
        │
        ├─► dez gates, todos medindo
        │
        └─► grava: disparidade uint16 a 768 · máscara · metadado · manifesto ·
                   split.json · ledger de sha256 · histograma de rejeição
```

O **alvo é a foto real**. O BokehMe aparece **só dentro do sweep**, como simulador para
comparar — nunca produz o rótulo.

---

## 4. O que falta até o dataset estar pronto

| # | passo | depende de | estado |
|---|---|---|---|
| 1 | **Piloto com refinamento**, ~200 amostras, todos os gates medindo | GPU | *bloqueado: cluster fora* |
| 2 | **`validate_focus_refinement.py`** no bloco pareado, contra a régua de **35,2%** | piloto | — |
| 3 | **Congelar os dez limiares** a partir das distribuições medidas | piloto | — |
| 4 | **Run completo**: RealBokeh ~13,8K + LFDOF | limiares | — |
| 5 | **Validar o release** — 11 checagens | run | — |
| 6 | **Publicar no HF** com o card | validação | — |

O caminho crítico inteiro passa pelo passo 1, e o passo 1 precisa de GPU.

### O número que decide

O passo 2 responde a única pergunta que ainda não tem resposta medida: **o refinamento
melhora os 35,2%?**

Ele decide pelo bloco **pareado** — a mesma amostra com e sem refino, via
`focus_disparity_from_initial_mask` — e não pela comparação entre `focus_source`, que é
confundida: o grupo `birefnet` é, por construção, o subgrupo em que o segmentador já
concordava com a física. O script foi escrito para imprimir `PIOROU` se for o caso, e há
teste que exige essa palavra.

### O que só se sabe rodando

Tudo que envolve dado real em escala: se os 20,6% de descarte de fato viram amostras
aproveitadas, quanto o K muda, qual a taxa de censura com a faixa nova, e quanto tempo
leva. Hoje o que existe é: o pipeline roda ponta a ponta em GPU (job 32224, 162 amostras,
K censurado em **1,2%** contra 47%, `k_value` mediana 16,12), e 648 testes passam.
