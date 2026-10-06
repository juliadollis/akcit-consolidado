# Condicionamento geométrico na BokehNet

> Plano de implementação. Como levar os sinais riemannianos derivados da
> profundidade, e a perda ponderada por oclusão, do documento de proposta
> ("Condicionamento Geométrico para Refocusing Generativo", Wallisson Policarpo
> Teodoro) até um treino que roda e uma comparação que se defende.
>
> Base: reprodução completa, fase 2 concluída. Alvo: BokehNet apenas.
> Objetivo: publicação. Criado em 2026-09-03.
>
> Auditoria dos dados que fundamenta as seções 3 e 8: `AUDITORIA_DADOS_ROTAS_BC.md`.
> Ferramentas de medição: `scripts/audit_hf/`.
>
> Versão em página: https://claude.ai/code/artifact/7adff9c8-74a4-4cdd-a2ae-627996070e0f

---

## 0. Sumário


A proposta é boa e o argumento central sobrevive a revisão: os sinais geométricos não
são informação nova, são a forma explícita dos termos que aparecem na expansão do
operador de desfoque. Três coisas precisam ser corrigidas antes de qualquer treino, e
uma quarta decidida.

| | |
|---|---|
| correções antes de treinar | 4 |
| ablação, 3 treinos em paralelo (6 GPUs) | ~6,7 dias |
| VRAM já em uso a 512² | 66 de 80 GB |
| amostras sem escala métrica nem focal | 2.932 (rota c, 20,1%) |

| O que | Situação |
|---|---|
| A perda da seção 5.3 está escrita em espaço de pixel; o treino roda em espaço de token latente | REESCREVER |
| O ponto de injeção proposto (soma no patch embedding) não existe no pipeline; o mecanismo é branch de tokens | TROCAR |
| Escala métrica: rota b OK (11.635), rota c ausente (2.932) | MEDIDO, ~16 min de GPU resolve |
| Focal em pixels: rota b 100%, rota c 0% (`exif` NULA nas 2.932) | MEDIDO, o Depth Pro já devolve |
| Seis canais: DECIDIDO. Dois branches, ~2,5x de atenção, gradient checkpointing obrigatório | MEDIR no smoke |
| Métrica `E_bleed`, que o próprio documento coloca como pré-condição | CONSTRUIR |
| Protocolo estatístico pareado por cena | reuso, já existe |

**A recomendação de maior valor:** medir `E_bleed` nos quatro pesos que já existem
antes de escrever uma linha de condicionamento. Se o melhor deles já estiver perto do
piso, não há espaço para melhora e a campanha inteira morre em dois dias, em vez de num
mês.

---

## 1. Contexto


### 1.1 O que o paper faz

GenRefocus decompõe refocusing de imagem única em dois estágios sobre FLUX.1-dev com
LoRA. A **DeblurNet** vai de imagem borrada para all-in-focus. A **BokehNet** vai de
`(AIF, mapa de defocus)` para bokeh controlável, com

```
D_def = K · |D − D_focus|
```

e `D` vindo do Depth Pro. A contribuição real não é arquitetural: é o esquema de treino
que completa sinais ausentes em dado real. A limitação número 1 que os próprios autores
declaram é que imperfeição na profundidade se propaga para o mapa de defocus e gera
atribuição errada de borrão.

**A tensão a nomear no texto:** a maior fraqueza declarada do paper é confiabilidade da
profundidade, e esta proposta aposta mais fichas em profundidade. Isso não a invalida,
mas define onde o risco mora: os sinais de primeira ordem são robustos, os de segunda
ordem não são.

### 1.2 Estado da reprodução

| Item | Situação |
|---|---|
| DeblurNet, 60K steps | OK, LPIPS 0,1446 na DDPD test |
| BokehNet fase 1, 40K sintético | concluída |
| BokehNet fase 2, 60K real | concluída, 3 variantes |
| Controle de forma de abertura, §3.3 | bloqueado, falta PointLight-1K |

As três variantes de fase 2 (`fase2-real`, `fase2-kfix`, `fase2-rotac-only`) partem do
mesmo LoRA da fase 1 com hiperparâmetros idênticos, então são ablações limpas de dado.

**Correção de premissa, medida na tabela curada em 2026-09-03.** A `kfix` não é a melhor
configuração: é a **pior** entre os modelos treinados, nas três mesas e nas três métricas.

| | LF-Bokeh | RealBokeh v2 | RealDOF |
|---|---|---|---|
| só rota c (a+c) | **0,1952** | **0,1140** | **0,1271** |
| nosso original (a+b+c) | 0,2110 | 0,1282 | 0,2148 |
| kfix | 0,2252 | 0,1450 | 0,2524 |
| linha de identidade | 0,2371 | 0,3587 | 0,3279 |

LPIPS, menor melhor. SSIM e DISTS dão a mesma ordem: nove comparações concordantes, com
gaps de 0,014 a 0,038, ou seja 20 a 50 vezes o piso de ruído de 0,0007.

A `kfix` mudou três coisas de uma vez (K pela Eq. 3, plano de foco de `s1` para
`z_focus_m`, e `max_coc` global), então isso é evidência contra o **pacote kfix**, não
contra um item isolado. O que estabelece com segurança: **a campanha não parte da `kfix`.**

**Composição do dado da campanha: rotas b + c**, a fase 2 do paper, sem alteração. A
questão de qual plano de foco usar (seção 3.2 e teste T1) decide o `defocus_source`.

### 1.3 O que a proposta acrescenta

```
G(x) = [ u , O , s , n_x , n_y , K~ ](x)     ∈ R^6

u  = 1/Z                       profundidade inversa
O  = min(‖∇D‖/τ , 1)           oclusão, τ = percentil de ‖∇D‖
s  = log √(det g)              elemento de área
n_x , n_y                      normais no plano
K~ = sgn(K)·log(1 + |K|/K0)    curvatura gaussiana comprimida

L = Σ_x w(x)·‖Î(x) − I(x)‖_1        w(x) = 1 + λ_o·O(x)
```

A ordem de prioridade do documento está correta e é mantida aqui: métrica primeiro,
depois a perda ponderada, depois o canal de oclusão, depois anisotropia, e só então
curvatura.

---

## 2. Como o pipeline realmente condiciona


Duas premissas da seção 5 do documento não correspondem ao código. Ambas mudam o
desenho, e uma delas muda para melhor.

### 2.1 Não existe patch embedding convolucional

O documento propõe `h0 = PatchEmbed(z_t) + Proj(E_g(G))` e classifica a soma no
embedding como a estratégia de custo baixo. No pipeline:

- `x_embedder` é um `nn.Linear(64 -> 3072)`, não uma convolução
  (`third_party/Genfocus/Genfocus/pipeline/flux.py:383-386`). A patchificação já
  aconteceu antes, no `_pack_latents`.
- O condicionamento não é soma. Cada condição é um **branch de tokens separado**, e o
  `S_t = [X_t ; E(I_in)]` do paper se realiza como concatenação de key e value dentro
  da atenção (`flux.py:236-250`), regulada por `group_mask` (`flux.py:240`).
- O código já é genérico em número de condições: `backbone.py:407-414` monta adapters,
  timesteps, guidances e ids por list comprehension a partir de `n_cond`, e
  `group_mask` se generaliza em `backbone.py:431-434`.

**Consequência favorável.** Um branch de condição extra custa **zero código no
transformer e zero parâmetro novo**: o adapter LoRA `"bokeh"` é compartilhado entre os
branches de condição (`backbone.py:408`), e o checkpoint resultante roda na inferência
oficial sem alteração. É literalmente o mecanismo que o próprio paper usa para forma de
abertura na §3.3: *"we append its tokens directly to the unified sequence"*.

A soma no embedding, ao contrário, exigiria cirurgia no `transformer_forward`
vendorizado, criaria um módulo de inicialização aleatória que precisaria de zero-init
tipo ControlNet para não destruir o comportamento pré-treinado, e quebraria a
compatibilidade com `Genfocus.pipeline.flux.generate`.

**O custo real é comprimento de sequência:**

| Configuração | Tokens | Atenção rel. | Cabe em 80 GB? |
|---|---|---|---|
| Hoje: texto + main + AIF + defocus | 3.584 | 1,00x | 66 GB medidos |
| + 1 branch geométrico (3 canais) | 4.608 | ~1,65x | com grad. checkpointing |
| + 2 branches geométricos (6 canais) | 5.632 | ~2,5x | medir antes de subir |

Os YAMLs da fase 2 estão com `gradient_checkpointing: false`. Qualquer branch extra
exige religar, com cerca de 30% a mais de tempo por step.

**Detalhe que decide o agrupamento dos canais.** O `group_mask` faz cada condição
atender só a si mesma, ao texto e ao branch principal (`backbone.py:431-434`,
verificado empiricamente contra a inferência oficial). Dois branches geométricos
ficariam **mutuamente cegos**. Se forem dois, o agrupamento dos seis canais entre eles
é decisão de projeto, não arbitrária.

### 2.2 A perda não vive em espaço de pixel

O treino real é uma única linha:

```python
# genfocus_train/models.py:176-181
def flow_matching_loss(prediction, target):
    return F.mse_loss(prediction.float(), target.float())
```

sobre velocidade de flow matching em tokens latentes empacotados de forma `(B, N, D)`,
chamada em `trainer.py:639` e `trainer.py:931`. Não existe `Î` nem `I` em pixel em
nenhum ponto do loop, e decodificar o latente a cada step colocaria o decoder do VAE no
caminho do gradiente por 60K steps. O tratamento está na seção 5.

### 2.3 Convenção de range, que é load-bearing

A AIF entra no VAE em `[-1,1]`; o mapa de defocus entra em `[0,1]` **cru**, porque a
inferência oficial usa `No_preprocess=True` e pula o passo `[0,1] -> [-1,1]`. A
assimetria é deliberada. Qualquer mapa novo segue o contrato `[0,1]` cru.

Treinar num range e inferir noutro foi o bug número 1 deste projeto. A paridade de
inferência não é item opcional do plano.

---

## 3. Pré-requisitos de dados

> **Medidos em 2026-09-03 sobre 100% das linhas.** Números, método e ressalvas em
> `AUDITORIA_DADOS_ROTAS_BC.md`. Ferramentas em `scripts/audit_hf/`.

### 3.1 O quadro medido

| | rota b | rota c |
|---|---|---|
| linhas | 11.635 | 2.932 |
| `exif` preenchida | **11.635 (100%)** | **0 (100% NULA)** |
| `calibration_ssim` preenchida | 0 (100% NULA) | 2.932 (100%) |
| cobertura da tabela kfix | 11.635 (bijeção exata) | **0** |
| escala métrica de `z` | sim | **não** |
| `fx` em pixels | sim, 11.635/11.635 | **0/2.932** |

É a armadilha do `calibration_ssim` espelhada: no `/first-rows` o `exif` da rota c
aparece como string vazia, mas é `NULL` nas 2.932. Um teste de existência passaria.

**Com 6 canais, 20,1% do dado da fase 2 não consegue produzir 4 dos 6.**

### 3.2 `depth` é profundidade métrica, e `data.py:483` está correto

O repositório se contradizia. Resolvido lendo pixels, sem usar `s1`: recalcular o
`coc_p99_px` gravado na kfix a partir dos pixels de `depth` sob cada convenção.

| hipótese | erro mediano | máximo | abaixo de 1% |
|---|---|---|---|
| **profundidade métrica min-max** | **0,011%** | 0,16% | **15/15** |
| disparidade min-max | 3,008% | 75,33% | 4/15 |

`z = z_min_m + depth_norm·(z_max_m − z_min_m)` está certo. Os 4 canais métricos
podem ser construídos sobre ele, onde houver `z_min_m`/`z_max_m`.

**Inconsistência que sobra, RESOLVIDA em 2026-09-04 pelo teste T1.** `s1` e
`z_focus_m` são planos de foco diferentes: `|s1_implicado − s1_gravado|` tem
mediana 0,0031, p90 0,0793, máximo 0,4882, e só 18,4% batem a 1e-4.

Medindo a Eq. 4 diretamente nos pixels (mediana da profundidade dentro da máscara
do BiRefNet), em 180 amostras:

| | mediana | p90 |
|---|---|---|
| `\|Eq4 − s1\|` | 0,00321 | 0,06491 |
| `\|Eq4 − z_focus_m normalizado\|` | **0,00001** | **0,00007** |

`z_focus_m` vence em 178/180 (98,9%), e em 138/138 (100%) entre as amostras com
`z_max` não saturado. **O `z_focus_m` É a Eq. 4, a 1e-5. O `s1` não é.**

A confirmação independente por nitidez (T1b, 120 amostras, variância do
laplaciano sobre o bokeh real, sem usar máscara) aponta na mesma direção mas
fracamente: 55% contra 45%, com erros 100x maiores dos dois lados. Serve para
dizer que não há contradição, não para confirmar sozinha.

**Decisão: `z_focus_m`.** Consequência: ele só existe para a rota b, então o job
da rota c tem de calculá-lo. A `foreground_mask` da rota c já existe e está 100%
preenchida, então continua sendo só a passada de Depth Pro, e a Eq. 4 sai da
máscara que já está lá.

### 3.3 Como destravar a rota c: uma passada de Depth Pro

`vision-pipeline/inference/src/pipelines/bokeh_net.py:133` já chama
`depth_model.infer(img_t, f_px=None)`. Com `f_px=None` o Depth Pro **estima a
focal** e devolve `focallength_px` junto da profundidade métrica, no mesmo forward.
`grep -rn "focallength_px"` no repositório retorna **zero usos**: o valor é
calculado e descartado nas três cópias do pipeline.

Logo, uma passada nas 2.932 imagens da rota c resolve escala **e** focal de uma vez.
A rota c não precisa da Eq. 3: ela tem K próprio, do sweep de SSIM.

| | custo | entrega |
|---|---|---|
| Depth Pro nas 2.932 AIF da rota c | ~16 min de GPU + job de escrita | `z_min_m`, `z_max_m`, `z_focus_m`, `focallength_px` por `stem` |

**Validação de graça:** rodar o mesmo job nas 11.635 da rota b, onde a focal da
EXIF já existe, e medir a concordância entre `focallength_px` estimado e
`f_mm · pixel_ratio`. Transforma "o Depth Pro estima bem a focal?" de suposição em
número. Se concordarem, use a estimada nas duas rotas, para proveniência homogênea.

### 3.4 Sentinelas e degenerações que precisam de gate

Nenhum gate existente pega estas, porque todos validam o mapa de defocus, que usa
`1/z`, enquanto os 4 canais métricos usam `z` linear e as suas derivadas.

| achado | contagem | ação |
|---|---|---|
| `z_max_m == 10.000,0` exato (teto do Depth Pro) | **2.988/11.635 (25,7%)** | substituir o max pelo p99,5 do mapa antes de normalizar, e regravar `z_max_m` |
| Cena útil em menos de 256 níveis uint16 | **2.870 (24,7%)**; mediana de 23 níveis no subgrupo `z_max >= 1000 m` | gate de quantização: excluir da supervisão dos canais de 2ª ordem |
| `k == 300` na rota c (teto do sweep, Eq. 5) | **1.379/2.932 (47,0%)** | filtrar pelo mesmo argumento de intervalo aberto que já descarta `k = 0` |
| `coc_p99_px >= max_coc` (mapa satura) | **2.018/11.635 (17,3%)** | `max_coc` é global (10,5107); o `diag_mapa_kfix.py` olha 24 amostras e aprova pela mediana |

Rota c com K estritamente interior **e** `cal_ssim >= 0,6`: **1.483/2.932 (50,6%)**.

### 3.5 A correção de `fx` que o resize exige

`fx_px = f_mm · pixel_ratio = W_px · focal_length_35 / 36`, identidade verificada
com diferença mediana zero. O `pixel_ratio` foi computado sobre a imagem **já
redimensionada** (lado maior ~1024), confirmado em 900/900 linhas em 9 offsets,
então não há fator desconhecido por amostra. O medo estava infundado.

**Mas** o dataloader reescala o lado menor para 512 e recorta, o que muda `fx` por
`512/min(W,H)`, fator que varia por amostra (1024x574 dá 0,892; 624x1024 dá 0,821).
É recuperável em runtime a partir de `img.size`, e **nenhum código hoje aplica essa
correção**. Sem ela a curvatura retroprojetada sai errada por um fator que varia
entre amostras, que é exatamente a classe de defeito que já custou uma rodada aqui.

### 3.6 Constantes de normalização fixas

Com 6 canais são cinco constantes, não três, todas calibradas uma vez sobre o
conjunto de treino e gravadas no config: `τ` do mapa de oclusão, `u_max` da
profundidade inversa, `s_max` do elemento de área, `K0` da compressão da curvatura,
e o percentil de substituição do `z_max` saturado.

`τ` tem que ser constante, não percentil por imagem: sob crop aleatório o percentil
do recorte não bate com o da imagem inteira, e numa cena sem descontinuidade real o
`perc99` é ruído e o mapa vira ruído saturado de quadro cheio. O próprio
`data.py:216-219` já documenta o princípio para o `s1`.

Nota de leitura: `geometry_maps.py:141` usa percentil **97**, não 99 como o
documento de proposta diz. `metrics._depth_edges:63` usa 99.

### 3.7 Resolução

`riemann/losses.py:48-50` usa `h = 1/max(H,W)`, o que multiplica `‖∇D‖` por 512 a
1536 e faz `√det g ≈ ‖∇D‖` quase em todo lugar. O treino roda a 512² e a inferência
do paper usa a resolução original com tiling. Gradiente em coordenadas de imagem
normalizadas, `∂D/∂(x/W)`, resolve. A 512² é no-op; a diferença aparece no tiling.

## 4. A pilha de canais


### 4.1 Duas observações sobre o conjunto proposto

**O canal `u` é redundante.** O modelo já recebe `D_def = clip(k·|D − s1|/max_coc, 0, 1)`,
que *é* o círculo de confusão. O argumento da seção 2.1 do documento, de que alimentar
`u` entrega uma quantidade linear no alvo, já está satisfeito: o pipeline nunca
alimentou `Z`, alimenta o CoC direto. O que `u` acrescenta é escala absoluta, que
`D_def` já modula por `K`.

**`s`, `n_x` e `n_y` são reparametrização, não complemento.** Numa superfície de Monge,
`n = (−D_x, −D_y, 1)/√(1+‖∇D‖²)` e `s = log√(1+‖∇D‖²)`, então os três canais codificam
dois graus de liberdade, que são `∇D`. Não é defeito: é uma parametrização limitada e
bem condicionada. Mas a redação atual ("juntos, determinam completamente o termo de
primeira ordem") sugere informação complementar quando é a mesma informação em
coordenadas polares, e um revisor vai apontar. A ablação C contra D continua válida,
porque magnitude sozinha é invariante a rotação.

**Correção física que vale para todos os canais de gradiente.** Como
`ε = γ‖∇D‖/(Z²c)` e `∇u = −∇D/Z²`, a grandeza relevante para a anisotropia é `‖∇u‖`,
não `‖∇D‖`. O documento usa `u` no canal 1, corretamente, e volta para `D` em toda a
geometria. Com `‖∇D‖`, o fundo distante domina o sinal, que é justamente onde o borrão
é mais uniforme. Um degrau de 1 m para 20 m dá Δ = 19 em `D` e 0,95 em `u`. Um degrau
de 20 m para 40 m dá Δ = 20 em `D`, que é *maior*, e 0,025 em `u`. Só em `u` a
ordenação é a opticamente correta.

### 4.2 A configuração decidida: 6 canais

Decisão tomada. A pilha completa entra, em 2 branches de condição, agrupados pelo
critério da seção 8.3 (o `group_mask` deixa os dois branches mutuamente cegos, então
o que precisa ser lido junto fica junto):

```
G1 = [ s , (n_x+1)/2 , (n_y+1)/2 ]     termo de PRIMEIRA ORDEM completo
G2 = [ u , O , K~_norm ]               escala, visibilidade, segunda ordem

todos em [0,1] cru (No_preprocess=True), como o mapa de defocus
∇ em coordenadas normalizadas, ∂/∂(x/W), invariante a resize
```

O que isso custa, e que não é opcional:

| | consequência |
|---|---|
| Sequência | 3.584 para 5.632 tokens, atenção ~2,47x |
| VRAM | `gradient_checkpointing: true` passa a ser obrigatório, e o pico tem que ser medido no smoke |
| Dado | os 4 canais métricos exigem a seção 3.3 (Depth Pro na rota c) e os gates da seção 3.4 |
| Ablação | B testa os 6 juntos. Qual canal carrega o ganho fica para a segunda rodada |

A alternativa de 3 canais scale-free (`[O, ĝ_x, ĝ_y]`, a decomposição polar do
gradiente, um branch só) fica registrada como plano de contingência: é o que se
roda se o smoke mostrar que 2 branches não cabem em 80 GB, ou se os trabalhos de
dado da fase F0b atrasarem. Ela não depende de escala métrica nem de focal, então
roda com o dado como ele está hoje.

### 4.3 Sobre a curvatura, especificamente

O documento pede expectativa moderada. A evidência interna sugere ir além disso e não
gastar rodada de GPU com ela nesta campanha. Como termo de perda em estimação de
profundidade, sobre profundidade *de verdade* (Spring), a curvatura perdeu para o
controle berHu em todas as métricas:

| Condição | bF ↑ | AbsRel ↓ | RMSE ↓ | δ1 ↑ |
|---|---|---|---|---|
| B0 berHu (controle) | **0,5986** | **0,2460** | **4,2073** | 0,6981 |
| B3 curvatura gaussiana, teto 1000 | 0,5745 | 0,2857 | 4,4548 | 0,6977 |

O consolidador se recusa a chamar a comparação, porque são 3 seeds contra 1, então não
é conclusivo. Mas também não é encorajador, e sobre profundidade estimada será pior: as
segundas derivadas amplificam ruído por `1/h²`.

Como a decisão foi incluir os 6 canais, a curvatura entra. Duas condições para que
ela seja informação e não ruído, ambas medidas na auditoria: usar
`geometry.surface_curvatures` com o `fx` corrigido pelo resize (seção 3.5), e aplicar
o gate de quantização (seção 3.4), porque em 2.870 amostras a cena útil cabe em menos
de 256 níveis uint16 e a derivada segunda ali é ruído de quantização, não geometria.

### 4.4 Correções a aplicar ao portar os sinais

| Onde | O quê |
|---|---|
| `riemann/losses.py:86` | Fixa `radius = 2` independente de σ. Com o σ = 2,0 que `visual_signals.py:146` usa, o kernel de 5 taps cobre ±1σ, ou seja é quase uma caixa truncada, não uma gaussiana. `geometry.py:87` faz certo com `r = round(3σ)`. |
| `geometry_maps.occlusion_map` | Sem máscara de validade: a fronteira de região inválida vira a maior oclusão do mapa. O `boundary_fscore` recebeu a erosão de correção em `metrics.py:114-121`; este não recebeu. |
| `geometry_maps.occlusion_map` | `torch.quantile` estoura acima de ~16,7M elementos. Achata para `(B, H·W)`, então a 1536² com batch ≥ 8 quebra. Usar `numpy.percentile` ou `kthvalue`. |
| curvatura | Usar `geometry.surface_curvatures`, não `geometry_maps.principal_curvatures`. |

---

## 5. A perda ponderada, em espaço de token


É a modificação de melhor relação custo-benefício da proposta inteira: não muda
arquitetura, não adiciona parâmetro, não muda o tempo de inferência. Só precisa ser
reescrita no espaço certo.

### 5.1 A geometria do problema

```
512² pixels
   -> VAE 8x            -> latente 64x64, 16 canais
   -> _pack_latents 2x  -> tokens (B, N=1024, D=64)

1 token = 1 bloco de 16x16 pixels
```

O peso `w`, portanto, vive em `(B, 1024, 1)` e faz broadcast sobre `D`. O mapa `O` em
`(B,1,512,512)` reduz 16x para `(B,1,32,32)` e é empacotado na mesma ordem do
`_pack_latents`.

### 5.2 Três detalhes que decidem se funciona

1. **Redução por max-pool, não média.** Uma borda de oclusão tem 1 a 2 px de largura.
   A média a 16x dilui a amplitude por volta de `1/16`, e um `λ_o = 3` vira um `λ_o`
   efetivo de 0,2. O max-pool preserva a amplitude e estende o peso ao bloco inteiro,
   que é a unidade que o modelo de fato prevê. A média fica como variante de ablação,
   por flag.
2. **Normalizar `w` pela própria média por batch.** Sem isso `λ_o` também multiplica o
   passo efetivo do otimizador, e o controle "perda ponderada sobre a linha de base"
   deixa de separar supervisão concentrada de learning rate maior, que é exatamente o
   que ele existe para separar.
3. **Reusar `self._pipe._pack_latents`** sobre um tensor replicado, em vez de refazer o
   reshape à mão. Garante ordenação idêntica à dos tokens. Um erro silencioso de
   reshape aqui passaria despercebido por 60K steps.

**Ponto de redação para o paper:** reponderar a perda de difusão de forma não uniforme
enviesa o estimador do score. É prática padrão (min-SNR weighting e afins), mas o texto
tem que dizer **reponderação perceptual da supervisão**, não "perda ponderada da
verossimilhança".

### 5.3 Instrumentação

Logar separadamente a perda ponderada, a não ponderada, e a fração da perda total que
vem de `B_θ`. Essa fração é o número que calibra `λ_o`, conforme o próprio documento
propõe. O ponto é `trainer.py:667-696`, que hoje assume um escalar único.

---

## 6. A métrica E_bleed


O documento coloca corretamente esta métrica como pré-condição: sem ela nenhuma das
comparações é conclusiva, porque a região de borda é uma fração pequena dos pixels e
qualquer métrica global dilui o artefato até a terceira casa decimal.

```
E_bleed = (1/|B_θ|) · Σ_{x ∈ B_θ} ‖Î(x) − I(x)‖_1        B_θ = { x : O(x) > θ }
```

- Entra em `deblurnet-eval-pipeline/eval_bokeh_synthesis.py`, ao lado de SSIM, LPIPS,
  DISTS, CLIP-I e LVCorr.
- Reusa a estrutura de `riemann/metrics.py:_depth_edges` (linhas 62-89) e a erosão de
  máscara de `boundary_fscore` (114-121).
- Reportar também sobre a região complementar, para mostrar que uma melhora na borda
  não veio de degradar o resto.

**Os benchmarks não bloqueiam esta métrica**, ao contrário do que a ausência de
coluna `depth` sugere. O pipeline de avaliação já roda o Depth Pro em toda imagem de
benchmark e salva o mapa em `.npy`
(`vision-pipeline/inference/src/pipelines/bokeh_net.py:93-141`). A profundidade já é
calculada, só não é gravada no df. Como é a mesma profundidade em todas as condições
comparadas, o estimador cancela na comparação pareada.

**Fase 0, e é um portão.** Medir nos quatro pesos que já existem: `fase2-real`,
`fase2-kfix`, `fase2-rotac-only` e o oficial. Se o melhor já estiver perto do piso de
identidade, não há espaço e a campanha para aqui. O precedente do raciocínio já está no
projeto: a coluna `margem_LPIPS_sobre_identidade` existe porque os pisos dos três
benchmarks são muito diferentes, e o que se compara é a distância até o piso de cada
um, nunca o valor absoluto.

---

## 7. Implementação


As fases são numeradas porque são de fato uma sequência: cada uma depende da anterior
estar verificada. A fase 0 corre em paralelo com as fases 1 e 2, porque é GPU enquanto
as outras são CPU.

### F0. Métrica E_bleed nos modelos existentes
*~2 h por modelo, 4 modelos, pode rodar nas GPUs de avaliação*

Conforme a seção 6. É o portão da campanha inteira.

### F0b. Trabalhos de dado (bloqueiam os 4 canais métricos)
*~1 dia de trabalho, dos quais ~30 min de GPU*

Nenhum é grande, todos são bloqueantes, e nenhum pode ser feito "durante" o treino.

1. **Depth Pro na rota c** (seção 3.3): 2.932 imagens, `f_px=None`, gravando
   `z_min_m`, `z_max_m`, `z_focus_m` e `focallength_px` por `stem` numa tabela
   `rota-c-kfix`, no mesmo formato da `rota-b-kfix-eq3`.
2. **Validação da focal** (seção 3.3): mesmo job nas 11.635 da rota b, medindo a
   concordância entre `focallength_px` estimado e `f_mm · pixel_ratio` da EXIF.
   Decide a questão 7 com número.
3. **Sentinela de `z_max`** (seção 3.4): regravar `z_max_m` pelo p99,5 do mapa nas
   2.988 amostras em que ele está exatamente em 10.000 m.
4. **Gate de quantização** (seção 3.4): marcar as 2.870 amostras cuja cena útil cabe
   em menos de 256 níveis uint16, para excluí-las da supervisão dos canais de 2ª ordem.
5. **Correção de `fx` pelo resize** (seção 3.5): aplicar `512/min(W,H)` em runtime,
   a partir de `img.size`, no ponto em que a curvatura retroprojetada é calculada.
6. **Decidir `s1` contra `z_focus_m`** pelo teste T1 abaixo, e registrar a escolha.

### Os três testes que fecham as decisões pendentes

**T1. Qual plano de foco implementa a Eq. 4.** ~200 amostras da rota b, no cluster
(exige PIL). A Eq. 4 do paper define `D_focus` como a mediana da profundidade dentro
da máscara do objeto saliente. Então:

```
m = mediana( depth01 [ foreground_mask > 0,5 ] )
a = s1
b = (z_focus_m − z_min_m) / (z_max_m − z_min_m)
quem estiver mais perto de m implementa a Eq. 4
```

O teste é legítimo porque a coluna `depth` e o mapa que o job da kfix usou são **o mesmo
campo**: `z_min_m` e `z_max_m` reconstroem o `coc_p99_px` a partir dos pixels de `depth`
com erro de 0,011%. Sendo o mesmo campo, `s1` e `z_focus_m` deveriam ser a mesma mediana
na mesma máscara. Como não são, ao menos um dos dois não é a mediana na máscara.

Confirmação independente, que não depende de a máscara do BiRefNet estar correta: mapa
de nitidez local (variância do laplaciano em janela) sobre a imagem de **bokeh real** da
rota b, achar a região mais nítida, ler o `depth01` ali. Se T1 e esta confirmação
apontarem o mesmo vencedor, está resolvido. Se discordarem, o achado é que a máscara do
BiRefNet não é o sujeito em foco naquelas imagens, que é uma limitação do método do
próprio paper e vale registro.

**T2. O teto do `k` é censura de faixa ou saturação do renderizador.** ~100 renders.
Refazer o sweep da Eq. 5 nas amostras censuradas com `K_max = 1000`.

Diagnóstico já feito nos escalares: o interior decai de forma monótona
(732, 341, 186, 117, 89, 82 amostras por faixa de 50) e o teto tem **1.385**, cerca de
20 vezes o que a extrapolação daria. É assinatura de **censura à direita**, não de
sentinela como o `k = 0`. As censuradas têm SSIM mediano 0,857 contra 0,886 do interior:
o sweep não achou bom casamento e correu para a borda.

| resultado do teste | leitura | ação |
|---|---|---|
| espalham entre 300 e 1000 | censura de faixa | alargar `K_max` e reetiquetar |
| empilham em 1000 | saturação do `_render_bokeh_simple`, kernel limitado a 51 px | limitar a faixa útil ao que o renderizador expressa, tirar as censuradas da supervisão de controlabilidade, registrar |

**T3. Concordância da focal.** De graça, porque o job da rota c já vai rodar: rodar o
mesmo job nas 11.635 da rota b, onde as duas existem, e medir a distribuição de
`focallength_px / (f_mm · pixel_ratio)`. Regra declarada antes de olhar: erro relativo
mediano abaixo de 5% e p95 abaixo de 15% significa usar Depth Pro nas duas rotas, por
proveniência homogênea. Acima disso, EXIF na b e Depth Pro na c, com a heterogeneidade
declarada como limitação.

### F1. Sinais geométricos no dataloader
*CPU, testável sem GPU*

- Novo módulo `genfocus_train/geo_signals.py`, numpy puro e testável, com
  `geometric_stack(depth01, tau, eps, ...)`. Portado de `depth-riemannian/riemann/`
  com as correções de 4.4 e o gradiente em coordenadas normalizadas.
- Gancho em `prepare_aligned_bokeh` (`data.py:224-323`), no bloco `309-317`, onde
  `def_c` ainda é o depth recortado em `[0,1]` e antes de virar mapa de defocus.
- Calculado no worker do dataloader, sem pré-computar em disco. Economiza cota (371 de
  500 GB hoje) e garante que resize, crop e flip já estejam aplicados, sem risco de
  dessincronizar augmentation.
- Nova chave `geo_map`, tensor `(3, S, S)` em `[0,1]`, nos dois `__getitem__` de bokeh:
  `data.py:692-719` (HF) e `data.py:834-862` (local). `_validate_image_tensor` em
  `data.py:620-624` já aceita esse formato.

### F2. Branch de condição
*~3 linhas em models.py, mais o repasse*

- `models.py:129-169`, `BokehNet.make_train_batch`: novo argumento,
  `encode_image_to_tokens`, append em `condition_tokens_list` e `condition_ids_list`.
  Tudo a jusante já generaliza.
- `trainer.py:498-515`, `_make_train_batch`: repassar a chave.
- Religar `gradient_checkpointing: true` nos YAMLs da campanha.

### F3. Perda ponderada por oclusão
*conforme a seção 5*

- `models.py:54-58`, `TrainBatchOutputs`: campo `weight`, para o peso viajar junto de
  `prediction` e `target`.
- `models.py:176-181`, `flow_matching_loss`: argumento de peso opcional, mantendo o
  `.float()`.
- Os dois call sites: `trainer.py:639` e `trainer.py:931`. São duplicados, esquecer o
  segundo quebra o smoke.

### F4. Config e paridade de inferência
*quatro lugares no config, dois na inferência*

Campos novos: `geo_condition`, `geo_tau`, `geo_channels`, `occlusion_loss_lambda`,
`occlusion_pool`.

**Armadilha conhecida do config.** `StageConfig` é construído campo a campo em
`_as_stage_config`. Adicionar o campo na dataclass (`config.py:87-121`) sem adicionar a
linha em `config.py:170-197` faz a chave do YAML ser **silenciosamente ignorada**.
Precedente: `defocus_source` em `config.py:104` e `194`. E o campo precisa chegar em
`DatasetRuntimeConfig` (`data.py:46-57`) pelos **dois** sites duplicados de construção:
`trainer.py:771-778` (treino) e `trainer.py:880-887` (smoke).

Paridade de inferência em `scripts/infer_bokeh_test.py` e
`vision-pipeline/inference/src/pipelines/bokeh_net.py`, usando
`Condition(..., No_preprocess=True)` para casar o range `[0,1]`.

### F5. Campanha
*~5 dias de wall-clock, 4 GPUs em paralelo*

Conforme a seção 8.

---

## 8. Ablação, custo e protocolo

### 8.1 O modelo de custo, aterrado em medição

`HANDOFF_PROJECT_HISTORY.md:485` registra o ritmo real da fase 1:
**14,8 s/step** (22.932 s para 1.550 steps), em 2 GPUs com `accum 16`, batch
efetivo 32, a 512², `gradient_checkpointing: false`, VRAM 66,3 de 80 GB.

```
custo por micro-batch = 14,8 / 16 = 0,925 s
```

Com 6 canais são 2 branches de condição a mais: a sequência vai de
512 + 3×1024 = 3.584 para 5.632 tokens. As partes lineares crescem 1,57x e a
atenção 2,47x; somando o gradient checkpointing que passa a ser obrigatório
(~1,3x), o multiplicador de step fica em torno de **2,3x a 2,9x**.

> **Este multiplicador é estimativa e precisa ser MEDIDO no smoke antes de subir
> qualquer treino longo.** O smoke tem que reportar s/step e pico de VRAM com os
> 2 branches ligados. Nada de projetar 15K steps sobre um número não medido.

### 8.2 Alocação com 6 GPUs

O batch efetivo 32 do paper tem que ser preservado, e 6 não divide 32. As
combinações exatas já validadas neste projeto são `2 GPUs × accum 16` (fase 1) e
`4 GPUs × accum 8` (fase 2). Logo:

**6 GPUs = 3 treinos simultâneos de 2 GPUs cada, `accum 16`.**

Usa a configuração exata que a fase 1 já rodou sem falha, mantém as 6 GPUs
ocupadas, e não inventa nenhum valor de acumulação novo.

| cond. | o que é | branches | s/step estimado | 15K steps |
|---|---|---|---|---|
| **A** | baseline `kfix`, já treinado | 2 | zero | zero |
| **A'** | A + perda ponderada, sem canais | 2 | 14,8 | ~2,6 dias |
| **B** | A' + 2 branches geométricos (6 canais) | 4 | ~38,5 | ~6,7 dias |
| **B'** | B com ruído de mesma estatística no lugar de G | 4 | ~38,5 | ~6,7 dias |

Os três rodam em paralelo. A' termina no dia 2,6 e libera 2 GPUs, que passam a
rodar a avaliação (`E_bleed` nos modelos existentes, ~2 h por modelo). B e B'
fecham no dia 6,7.

Como o snapshot sobe a cada 1.000 steps, a curva é visível durante todo o percurso.
Regra de parada antecipada, declarada agora: se no step 8.000 nem A' nem B tiverem
melhora de `E_bleed` acima do limiar da seção 8.4, os dois são interrompidos.

`upload_hf_repo_base` **distinto por condição**, e snapshot a cada **1.000 steps**
(a mesma cadência densa que a `kfix` usou, em vez dos 2.500 da fase 2 original). A curva
densa é o que permite dizer "com N steps dava X, com M dá Y" e é o que sustenta a regra
de parada antecipada. Dois treinos que compartilharam o repo corromperam a curva abaixo
do step 14.000 e os checkpoints ficaram sem procedência.

### 8.5 A ablação de dados vem depois, e está 3/4 pronta

A ablação de composição de dados é a **Tabela 6 do paper**: (a) sintético, (b) ITW,
(c) LFDOF e RealBokeh, comparando `a`, `a+b`, `a+c` e `a+b+c`. Ela é posterior à campanha
geométrica e não se mistura com ela.

O que já existe, com métrica medida nas três mesas:

| célula da Tab. 6 | modelo | situação |
|---|---|---|
| `a` | fase 1 (só sintético) | **pronto** |
| `a+c` | só rota c | **pronto** |
| `a+b+c` | nosso original (fase 2) | **pronto** |
| `a+b` | rota b sozinha na fase 2 | **falta**, é o único treino novo |

Ou seja, a Tabela 6 do paper precisa de **uma** rodada, não de quatro. E ela tem valor
próprio independente da geometria: é ela que diz se a rota b, que é 80% do dado, soma ou
subtrai, questão que a tabela curada hoje sugere estar subtraindo mas sem a célula que
isolaria isso.

### 8.3 Agrupamento dos 6 canais nos 2 branches

O `group_mask` faz cada condição atender só a si mesma, ao texto e ao branch
principal (`backbone.py:431-434`), então **os dois branches geométricos são
mutuamente cegos**. O agrupamento tem que juntar as quantidades que precisam ser
lidas em conjunto no mesmo pixel.

```
G1 = [ s , (n_x+1)/2 , (n_y+1)/2 ]     o termo de PRIMEIRA ORDEM completo
                                        (magnitude e direção da inclinação)
G2 = [ u , O , (K~ normalizado) ]      escala, visibilidade e segunda ordem
```

A anisotropia do núcleo é magnitude **e** direção juntas, então `s`, `n_x` e `n_y`
ficam no mesmo branch. A combinação entre G1 e G2 acontece no branch principal, que
enxerga tudo.

Todos os 6 canais entram em `[0,1]` cru (`No_preprocess=True`), como o mapa de
defocus. `n_x` e `n_y` vão por `(n+1)/2`; `u`, `s` e `K~` por clip contra as
constantes fixas da seção 3.6.

### 8.4 Protocolo estatístico e limiar

Não precisa ser escrito. Já existe:

| peça | onde |
|---|---|
| t pareado, Wilcoxon, IC bootstrap do delta, flag de concordância | `riemann/repro.py:174`, `compare_paired` |
| Driver por cena, com aviso abaixo de 8 cenas | `scripts/evaluate_paired.py` |
| Consolidação por seeds, que se recusa a comparar n diferentes | `scripts/consolida_reteste.py` |

**Limiar de relevância prática, declarado ANTES de olhar os resultados.** Base
empírica medida na tabela curada: o piso de ruído entre checkpoints é
ΔLPIPS ≈ 0,0006 a 0,0008, e a menor diferença que o projeto já tratou como real é
ΔLPIPS ≈ 0,0119. Adotamos **ΔLPIPS ≥ 0,01** e o correspondente em `E_bleed`, a ser
fixado quando a métrica tiver a primeira medida na linha de base.

Duas ressalvas que vão para o texto do paper:

1. **O piso acima não é ruído de semente.** É entre dois checkpoints do mesmo
   treino. Não existe nenhuma repetição com semente diferente nas 24 linhas
   curadas, então o ruído entre execuções independentes é desconhecido e este
   limiar o subestima. Mitigação barata antes da campanha: rodar o mesmo checkpoint
   duas vezes com sementes de amostragem diferentes no mesmo benchmark, ~2 h de
   GPU, o que dá o piso de ruído **da medição**, que o efeito tem de superar antes
   de qualquer outra coisa.
2. **O LVCorr não sustenta limiar no estado atual.** Ele troca de sinal entre
   modelos no mesmo benchmark (kfix +0,4599 contra oficial −0,9644 no RealDOF,
   amplitude 1,42). É a métrica de controlabilidade, é justamente a que os canais
   geométricos deveriam melhorar, e precisa ser estabilizada antes de virar
   evidência.

Replicar a condição vencedora em 3 seeds antes de afirmar qualquer coisa. Só a
vencedora vai para 60K completos.

## 9. Verificação


### 9.1 Testes puros, sem GPU

Em `tests/test_data_pipeline.py`, que já é CPU-only com 19 testes:

- plano frontoparalelo dá `O = 0` e direção neutra;
- rampa linear dá `O` constante e direção constante;
- degrau dá `O = 1` exatamente na borda;
- **invariância a resolução**: mesmo depth a 512 e a 1024 dá o mesmo `O` com o
  gradiente normalizado;
- **invariância a escala**: `stack(D)` e `stack(2·D)` batem para os canais que devem
  bater, e documentadamente não batem para os que dependem de escala;
- crop e flip comutam com o cálculo do sinal.

### 9.2 Ordenação dos tokens

Teste que constrói um mapa de peso com um único bloco de 16x16 marcado, empacota, e
confere que o token aceso é o esperado. É o ponto onde um erro silencioso de reshape
passaria despercebido por 60K steps.

### 9.3 Smoke e auditoria no dado real

- `python -m genfocus_train.train smoke` com o config novo: 3 steps, loss finita, e
  **VRAM medida** com o branch extra ligado. Na fase 2 o smoke pegou um bug do filtro
  de SSIM que teria custado 60K steps.
- Auditoria de 300 amostras das rotas b e c, no espírito de
  `scripts/audit_phase2_defocus.py`: distribuição de `O`, fração de pixels acima de θ,
  e checagem explícita de que não há amostra com `O` saturado de quadro cheio.

**O teste que discrimina.** Correlação não detecta normalização por imagem, porque é
invariante a escala. Foi assim que o mapa de defocus quebrado passou despercebido. O
teste que discrimina é comparar o **máximo absoluto entre amostras**. Aplicar o mesmo
teste a cada canal geométrico novo.

### 9.4 Paridade treino e inferência

Gerar a pilha geométrica pelo caminho do dataloader e pelo caminho de inferência para a
mesma imagem, e conferir igualdade numérica. É a versão geométrica do que
`scripts/verify_defocus_encoding.py` já faz para o defocus.

---

## 10. Riscos


| Risco | Avaliação e mitigação |
|---|---|
| **Redundância**: a rede já derivaria os sinais sozinha | Real, e é o que o controle B' testa. O argumento a favor: derivar consome capacidade, e o mapa de defocus já passa pelo VAE a 1/8 da resolução, então o gradiente que a rede veria é o de um mapa reconstruído por um autoencoder treinado em imagem natural. |
| **Desalinhamento treino e inferência** (não nomeado no documento) | Na inferência a BokehNet consome a saída da DeblurNet, então a geometria é estimada sobre imagem gerada, enquanto no treino é estimada sobre AIF real. O mapa de defocus já carrega isso, mas canais a mais amplificam. Mitigação barata: augmentation leve, ruído e desfoque, sobre a pilha geométrica no treino. Vale uma linha no texto de qualquer forma, porque um revisor vai levantar. |
| **Qualidade herdada da profundidade** | A limitação mais séria, e o documento a nomeia. Os sinais de segunda ordem são os mais afetados: as segundas derivadas amplificam ruído por `1/h²`. Mitigação: suavização calibrada antes da derivação, verificação da razão sinal-ruído em superfície que se sabe plana, e a curvatura fora da rodada 1. |
| **VRAM** | 66 de 80 GB já em uso a 512² com checkpointing desligado. Dois branches não cabem sem religar. Medir no smoke, não descobrir no step 3000. |
| **Procedência da rota a** | Ressalva permanente e independente desta proposta: o pipeline gravou só um UUID, sem caminho de origem, então não é possível certificar que um benchmark qualquer está fora do treino da fase 1. Qualquer publicação carrega isso. |
| **Deslocamento de domínio** | As estatísticas dos canais dependem da cena. Constantes calibradas em interior podem não valer em externo. Mitigação: fixar as constantes a partir do conjunto de treino e registrá-las no config, como o documento propõe. |

---

## 11. Decisões: as tomadas e as que faltam

### Tomadas

| # | decisão |
|---|---|
| 1 | **6 canais**, `[u, O, s, n_x, n_y, K~]`, em 2 branches agrupados como na seção 8.3 |
| 2 | **6 GPUs**, como 3 treinos simultâneos de 2 GPUs com `accum 16`, batch efetivo 32 |
| 3 | Alvo: **publicação**, logo os dois controles entram e a vencedora replica em 3 seeds |
| 4 | Limiar de relevância prática: **ΔLPIPS >= 0,01**, declarado antes dos resultados |
| 5 | **Dado da campanha: rotas b + c**, a fase 2 do paper, sem alteração de composição |
| 6 | **A campanha NÃO parte da `kfix`**, que é o pior modelo treinado nas três mesas |
| 7 | Snapshot para o HF a cada **1.000 steps**, repo distinto por condição |
| 8 | A ablação de composição de dados (Tab. 6) vem **depois** e precisa de 1 treino, não 4 |

### Respondidas pela auditoria

| # | pergunta | resposta medida |
|---|---|---|
| 1 | A rota c entra? | Entra, e a passada de Depth Pro da seção 3.3 virou **caminho crítico**, não higiene |
| 2 | De onde vem o `fx`? | Rota b: `f_mm · pixel_ratio`, 11.635/11.635. Rota c: do `focallength_px` que o Depth Pro já devolve e que ninguém usa |
| 4 | Limiar do `E_bleed` | Ancorado em ΔLPIPS >= 0,01, com as duas ressalvas da seção 8.4 |

### Em aberto, cada uma com o teste que a fecha

| # | questão | recomendação | teste |
|---|---|---|---|
| 5 | `s1` ou `z_focus_m`? | ~~`s1`~~ **RESOLVIDO: `z_focus_m`**, ver REGISTRO_GEO_COND.md 2026-09-04 | T1 rodado: 178/180 |
| 6 | Filtrar `k == 300`? | **RESOLVIDO: não filtrar, marcar.** T2 confirmou censura de faixa (alvo 1,92x mais borrado, corr(k,blur)=-0,48) e refutou saturação do renderizador. Corrigir de verdade exige o `_render_bokeh_simple`, que virou caminho crítico | T2 rodado, 300 amostras |
| 7 | Focal do Depth Pro nas duas rotas? | **Sim**, se a concordância passar. Proveniência homogênea vale mais que precisão marginal num canal comprimido em log | **T3**, de graça junto do job da rota c |

## 12. Operação


| Regra | Motivo |
|---|---|
| Nunca excluir job nem apagar nada no cluster sem perguntar. Limite de tempo sempre alto (`--time=7-00:00:00`). | Regra do projeto. |
| dgx-H100-01 é docker sem SLURM. Sempre `--user $(id -u):$(id -g)`, container detached. | Sem o `--user` os arquivos nascem do root: há 54 GB de cache que a usuária não consegue apagar. O detached sobrevive a queda de ssh, que já aconteceu duas vezes. |
| Nunca `docker system prune`, `rmi` ou `rm` de container alheio. | Há containers parados de outras quatro pessoas no host. |
| Nunca sobrescrever `PYTHONUSERBASE`. `PEFT_PIN` está proibido. | diffusers 0.37.1 exige peft >= 0.17; o pin antigo rebaixava transformers e é a origem de quatro jobs falhos. |
| Manter `torch.backends.cuda.enable_cudnn_sdp(False)` em `backbone.py:157-158`. | O branch extra muda a concatenação de seq-len na atenção, que é exatamente o que quebra o kernel cuDNN no backward em H100 com bf16. |
| `rsync` sempre com excludes (`outputs/ third_party/ logs/ wandb/ .git/`). | Um `--delete` sem exclude já apagou checkpoints. |

---

Documento derivado do paper GenRefocus (arXiv 2512.16923v3), da proposta
*Condicionamento Geométrico para Refocusing Generativo* e da leitura direta do código em
`genfocus_train/`, `depth-riemannian/riemann/` e `deblurnet-eval-pipeline/`. Referências
de arquivo e linha conferidas contra a árvore atual. Números de tempo e VRAM vêm de
medições registradas em `DECISOES_FASE2.md`, `PLANO_EXPERIMENTOS.md` e
`historico-ultimo.md`.
