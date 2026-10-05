# Registro da revisão do plano de correções da DeblurNet

Diário de bordo desta revisão. Registra **de onde veio cada afirmação** do
`PLANO_CORRECOES_DEBLURNET.md`, o que foi conferido contra fonte, o que era erro
meu, e o que continua sendo hipótese não medida.

Regra que este registro impõe ao plano: toda linha do plano cai em uma de três
categorias, e a categoria tem que estar visível no texto.

| Categoria | Significa |
|---|---|
| **FATO** | conferido contra o código, o paper, ou aritmética reproduzível. Tem fonte citada. |
| **HIPÓTESE** | argumento mecanístico ou expectativa. Não medido. Precisa de experimento. |
| **PROTOCOLO** | alteração deliberada do procedimento, cuja fidelidade ao paper não está provada. Precisa de A/B. |

---

## 1. Linha do tempo

### 1.1 Auditoria inicial
Leitura do `paper.pdf` (29 páginas, texto extraído com `pdftotext -layout`) e do
`third_party/Genfocus/` (inferência oficial dos autores). Comparação linha a
linha com `genfocus_train/`. Produziu 6 achados, publicados como
`PLANO_CORRECOES_DEBLURNET.md`.

### 1.2 Crítica da usuária — rodada 1
Levantou T2 (escala treino ≠ escala avaliação, com a aritmética do `mu`), T3
(`top_k_sharpest` seleciona linhas e não cenas), T4 (refinamento sobre a
severidade do descasamento de guidance por variante), mais três achados de
código: LoRA no `proj_out` de topo, `main_adapter` ligando LoRA no branch de
texto, e o warmup começando errado. **Todos confirmados.** Ver seção 3.

### 1.3 Crítica da usuária — rodada 2 (metodológica)
Apontou que C2, C4, C5 e C11 misturavam fato medido com hipótese, e que três
afirmações minhas estavam erradas. **Aceita integralmente.** Ver seção 4.

### 1.4 Revisão executada
Seis agentes em paralelo: quatro reescrevendo C2, C4, C5 e C11+C3 (cada um em
arquivo de rascunho isolado, sem colisão), dois verificando de forma
independente — um contra o paper, outro contra a inferência oficial e as
referências `arquivo:linha`. Resultados na seção 5.

---

## 2. Fontes usadas

| Fonte | Caminho | Papel |
|---|---|---|
| Paper | `paper.pdf` (arXiv 2512.16923v3) | fonte para §3.1, §3.5, §4.1, §B.1, Tabela 2 |
| Inferência oficial | `genrefocus_deblurnet_paper/third_party/Genfocus/` | contrato que o treino tem que reproduzir |
| Treino cond-only | `genrefocus_deblurnet_paper/genfocus_train/` | alvo das correções |
| Treino main+cond | `genrefocus_deblurnet/genfocus_train/` | variante do modelo de 60k já avaliado |
| Nossa inferência | `genrefocus_deblurnet_paper/vision-pipeline/inference/` | protocolo de avaliação |
| Histórico | `HANDOFF_PROJECT_HISTORY.md`, `historico-ultimo.md` | contagens de dataset, header do .safetensors oficial |

`third_party/Genfocus/` é clone de referência e **não deve ser modificado** — é
o que permite comparar. Qualquer desvio nosso vai na cópia em
`vision-pipeline/inference/Genfocus/`.

---

## 3. Achados confirmados (rodada 1)

### 3.1 LoRA no `proj_out` de topo — FATO
PEFT casa alvo por `key == target or key.endswith("." + target)`. A chave de topo
do `FluxTransformer2DModel` é literalmente `proj_out`, então a string solta na
`LORA_TARGET_MODULES` captura também `transformer.proj_out`, que
`transformer_forward` executa **fora** de qualquer `specify_lora`.

Aritmética contra o header do `bokehNet.safetensors` oficial (686 tensores,
343 módulos, dumpado por Range request em `HANDOFF_PROJECT_HISTORY.md`):
19×6 + 38×6 + 1 = 343 (oficial) versus 344 (nosso). Fecha exato.

Consequência não óbvia registrada no plano: é **consistente** entre treino e
inferência (as duas aplicam), então não é o bug da saída lavada — é desvio de
arquitetura. E invalida a afirmação de que a variante `_paper` é "LoRA só na
condição".

### 3.2 Guidance 1.0 no treino vs 3.5 na inferência de deblur — FATO
Duas fontes oficiais independentes: `Inference_deblurNet.py` e `demo.py` chamam
`generate()` sem `guidance_scale` (default 3.5), enquanto os dois caminhos de
bokeh passam `guidance_scale=1.0` explicitamente. A auditoria anterior do
projeto (`historico-ultimo.md`) validou "guidance 1.0" contra a inferência de
**bokeh** e a conclusão foi carregada para o deblur sem reconferir.

### 3.3 Warmup — FATO
Inconsistência interna: `_lr_at` prevê 2e-7 para o início, o código entrega 1e-4
no primeiro update, e o índice fica deslocado em 1 pelo resto do warmup.
Verificado executando o scheduler isoladamente.

### 3.4 `top_k_sharpest` por linha — FATO (magnitude a medir)
`np.argsort(scores)[::-1][:k]` sobre linhas, com score idêntico dentro da cena
(mesma `image_focus`), logo empates exatos e cenas inteiras selecionadas.

### 3.5 `main_adapter` liga LoRA no branch de texto — FATO
`adapters = [main_adapter]*2 + c_adapters` e o índice 0 é o texto; em
`single_block_forward` o texto passa por `specify_lora` com `adapters[0]`.

### 3.6 Aritmética de escala e `mu` — FATO
Reproduzida em Python com a config do FLUX-dev. Ver a tabela do C4.

---

## 4. Erros meus corrigidos nesta revisão

| # | Onde | O que eu afirmei | O que está certo |
|---|---|---|---|
| E1 | C2 | "fazer o A/B de guidance depois do C1" | Incoerente: o C1 muda o código, não os pesos de 60k. O checkpoint antigo tem a camada extra gravada e mede uma arquitetura diferente da pós-C1. |
| E2 | C2 | A/B de inferência decide o valor de treino | Não decide. Só dois treinos decidem. Virou matriz 2×2 treino×inferência. |
| E3 | C2 | tabela leve/moderado/severo | É argumento mecanístico, não medição. Rotulado como predição. |
| E4 | C4 | casar o `mu` é "a correção" | É heurística com precedente, não requisito derivável. O modelo é condicionado em σ e o treino cobre (0,1); é desbalanceamento de densidade. Eu mesmo disse isso e depois deixei o plano contradizer. |
| E5 | C4 | experimento R1/R2/R3 | Confundia dois eixos. Virou fatorial 2×2 (escala × fonte do mu). |
| E6 | C4 | R3 com seq 1024 / mu 0,6300 | Lado maior → 512 numa 3:2 dá 512×336: seq **672**, mu **0,5704**, exp **1,769**. E o aspecto variável exigiria bucketing. |
| E7 | C5 | "o paper quer 3.000 cenas distintas" | Interpretação de frase ambígua. Virou alteração de protocolo a ser medida. |
| E8 | C5 | `row_i = ... rng.integers(len(grupo))` | Usava posição no grupo como índice global. Correto: `row_i = grupo[pos]`. |
| E9 | C11-1 | resolução de treino torna a Tabela 2 incomparável | Comparabilidade é do protocolo de **avaliação**. E o controle certo já existe: os pesos oficiais rodados pelo nosso pipeline. |
| E10 | C11-2 | crop nativo "elimina" a heterogeneidade entre fontes | Não elimina. Faz treino e avaliação concordarem por fonte; a diferença física entre câmeras continua. O resize por lado menor é que **cria** heterogeneidade adicional. |
| E11 | C11-3 | "o split oficial do DPDD é 350/74/76" | Veio da minha memória, sem fonte conferida. Marcado como a verificar. |

Discordância registrada: a usuária classificou o C3 como "intenção aprovada". Eu
sustento que é bug fechado — a inconsistência entre a fórmula implementada e o
que o código entrega é verificável sem nenhuma suposição sobre o que seria
ótimo. Atenuante registrado no plano: como `lora_B` nasce em zero, o primeiro
update só move B.

---

## 5. Verificação independente

_(preenchido ao fim da rodada de agentes — ver seção 6)_

---

## 6. Estado de cada item

_(preenchido ao fim da rodada de agentes)_
