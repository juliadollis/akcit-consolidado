# Condicionamento geométrico na BokehNet

> Consolidado de tudo que foi feito entre 2026-09-03 e 2026-09-09.
>
> Este documento é o ponto de entrada. Os companheiros:
> - `PLANO_CONDICIONAMENTO_GEOMETRICO.md` · o plano, com as decisões e o custo
> - `AUDITORIA_DADOS_ROTAS_BC.md` · a auditoria dos dfs, número a número
> - `REGISTRO_GEO_COND.md` · o log corrido, append-only, com os erros no caminho
> - `geo_cond/README.md` · o código
>
> Página para compartilhar: https://claude.ai/code/artifact/51dfe2bb-7b43-4d94-8abb-aefcb83d3f6b
> Dados gerados: `juliadollis/geocond-auditoria-testes` (HF, privado)

---

## 1. O que é isto

O paper GenRefocus (arXiv 2512.16923) decompõe refocusing em dois estágios sobre
FLUX.1-dev com LoRA: DeblurNet (borrada → all-in-focus) e BokehNet
(AIF + mapa de defocus → bokeh controlável). A reprodução está completa.

O Wallisson propôs alimentar a BokehNet com seis sinais geométricos derivados da
profundidade, mais uma perda que concentra supervisão nas descontinuidades:

```
G(x) = [ u , O , s , n_x , n_y , K~ ](x)

u  = 1/Z                       profundidade inversa
O  = min(||grad D||/tau, 1)    oclusao
s  = log sqrt(det g)           elemento de area
n_x , n_y                      normais no plano
K~ = sgn(K) log(1 + |K|/K0)    curvatura gaussiana comprimida

L = soma_x w(x) ||I^(x) - I(x)||_1 ,   w(x) = 1 + lambda_o O(x)
```

O argumento: os sinais não são informação nova, são a forma explícita dos termos
que aparecem na expansão do operador de desfoque. Fornecê-los prontos tira da
rede o encargo de reconstruir por regressão algo que se calcula em fechado.

**Esse argumento se sustentou.** O que mudou foi a engenharia.

---

## 2. Estado em 2026-09-09

| | |
|---|---|
| Canais implementados | 6, em treino |
| Testes | **106**, rodando no container do cluster |
| Amostras com escala métrica e focal | **14.556** |
| Erro na borda vs. fora (identidade) | **2,23x** |
| Disco | 456G de 500G |

| condição | o que testa | progresso |
|---|---|---|
| **A** (fase2-real, step 15.000) | linha de base pareada | pronto, não precisou treinar |
| **A'** (perda ponderada) | a perda sozinha ajuda? | **concluído, resultado negativo** |
| **A''** (perda com limiar) | a correção do diagnóstico | 2.500/15.000 |
| **B** (perda + 6 canais) | a hipótese central | 11.920/15.000 |
| **B'** (B com ruído no lugar de G) | informação ou capacidade? | 3.860/15.000 |

---

## 3. A auditoria dos dados

Duas rotas, medidas em 100% das linhas por dois métodos independentes. Ferramentas
em `scripts/audit_hf/` (leitor Parquet e decodificador PNG de 16 bits em stdlib,
porque o `datasets-server` está quebrado para esses repos).

### 3.1 O espelho de colunas vazias

| | rota b (11.635) | rota c (2.932) |
|---|---|---|
| `exif` preenchida | **100%** | **0%, tudo NULO** |
| `calibration_ssim` | **0%, tudo NULO** | **100%** |
| tabela kfix (escala métrica) | bijeção exata | zero interseção |

No `/first-rows` o `exif` da rota c aparece como string vazia, mas é `NULL` nas
2.932. Um teste de existência passaria. É a mesma armadilha da
`calibration_ssim`, do outro lado do dataset.

### 3.2 `depth` é profundidade métrica, não disparidade

O repositório se contradizia (`data.py:483` assumia métrica,
`HANDOFF_PROJECT_HISTORY.md:428` afirmava disparidade). Resolvido lendo pixels,
por um teste que não usa `s1`: recalcular o `coc_p99_px` gravado na kfix a partir
dos pixels de `depth` sob cada convenção.

| hipótese | erro mediano | máximo | abaixo de 1% |
|---|---|---|---|
| **profundidade métrica min-max** | **0,011%** | 0,16% | **15/15** |
| disparidade min-max | 3,008% | 75,33% | 4/15 |

Confirmado depois por um segundo caminho: ajuste afim contra uma execução nova do
Depth Pro deu R² = 1,00000 em 40 amostras.

### 3.3 Sentinelas que nenhum gate anterior pegava

Todos os gates existentes validam o mapa de defocus, que usa `1/z`. Os canais
métricos usam `z` linear.

| achado | contagem | consequência |
|---|---|---|
| `z_max == 10.000,0` exato (teto do Depth Pro) | 2.988/11.635 (25,7%) | invisível em `1/z`, fatal para `s`, `n`, `K` |
| cena útil em menos de 256 níveis uint16 | 62/2.932 (2,1%) | derivada segunda ali é ruído de quantização |
| `k == 300` na rota c (teto do sweep) | 1.379/2.932 (47,0%) | ver seção 6.2 |

---

## 4. As correções à proposta

### 4.1 O ponto de injeção da §5.2 não existe no pipeline

A proposta descreve `h0 = PatchEmbed(z_t) + Proj(E_g(G))` e classifica a soma no
embedding como estratégia barata. No código:

- não há PatchEmbed convolucional: `x_embedder` é `nn.Linear(64 -> 3072)`, e a
  patchificação já aconteceu no `_pack_latents`;
- o condicionamento não é soma: cada condição é um **branch de tokens separado**,
  e o `S_t = [X_t ; E(I_in)]` se realiza como concatenação de key/value dentro da
  atenção (`flux.py:236-250`).

**Isso é melhor do que a proposta assumia.** Um branch extra custa zero parâmetro
novo (o adapter LoRA é compartilhado entre os branches de condição) e o
checkpoint roda na inferência oficial sem alteração. É o mesmo mecanismo que o
paper usa para forma de abertura na §3.3.

Custo real: sequência de 3.584 para 5.632 tokens, atenção ~2,5x. Medido:
**33,8 s/step em 2 GPUs, 29,4 GB de VRAM** com gradient checkpointing ligado.

### 4.2 Os 6 canais viram 2 branches, e o agrupamento não é livre

O VAE recebe imagens de 3 canais. E o `group_mask` faz cada condição atender só a
si mesma, ao texto e ao branch principal: **os dois branches geométricos são
mutuamente cegos**. O que precisa ser lido junto fica junto:

```
G1 = [ s , n_x , n_y ]     o termo de PRIMEIRA ORDEM completo
G2 = [ u , O , K~ ]        escala, visibilidade, segunda ordem
```

A combinação entre G1 e G2 acontece no branch principal, que enxerga tudo.

### 4.3 A física pede `grad u`, não `grad D`

Como `eps = gamma ||grad D||/(Z^2 c)` e `grad(1/Z) = -grad D / Z^2`, a magnitude
que governa a anisotropia é `||grad u||`. A proposta usa `u` no canal 1,
corretamente, e volta para `D` em toda a geometria.

Com `||grad D||` o fundo distante domina o sinal, que é onde o borrão é mais
uniforme: um degrau de 1 m para 20 m dá 19 em `D` e 0,95 em `u`; um de 20 m para
40 m dá 20 em `D`, **maior**, e 0,025 em `u`. Implementado `field="inverse"` como
default, com `"depth"` disponível para a diferença ser ablacionável.

### 4.4 Uma observação de redação

Numa superfície de Monge, `n = (-D_x, -D_y, 1)/sqrt(1+||grad D||^2)` e
`s = log sqrt(1+||grad D||^2)`. Então `s`, `n_x` e `n_y` codificam **dois** graus
de liberdade em três canais: são a decomposição polar de `grad D`, não informação
complementar. A parametrização é boa e bem condicionada, mas a frase "juntos,
determinam completamente o termo de primeira ordem" sugere complementaridade que
não há, e um revisor vai apontar.

---

## 5. O que foi implementado

Pacote isolado em `geo_cond/`, 22 arquivos Python, mais a fiação no
`genfocus_train`. Com `geo_condition: false` (o default) o comportamento é
idêntico ao anterior, e `occlusion_lambda: 0.0` reproduz a perda antiga bit a
bit. É isso que torna a condição de controle gratuita.

| arquivo | o que faz |
|---|---|
| `signals.py` | os 6 canais, de `depth01` a `(6,H,W)` em [0,1] |
| `constants.py` | as 8 constantes de normalização, **sem default** |
| `dataloader.py` | ponte com o crop: resize, recorte, espelhamento |
| `loss_weight.py` | oclusão em pixel para peso por token |
| `ebleed.py` | a métrica da §5.4 |
| `jobs/` | 7 jobs: escala métrica, calibração, testes T1-T3, diagnósticos |

Fiação no treino: `data.py` (5 campos, `_GeoMixin`, chave `geo_map`),
`models.py` (2 branches, peso da perda), `trainer.py` (os dois call sites da
perda e os dois de `DatasetRuntimeConfig`), `config.py` (`RuntimeConfig`,
`StageConfig` **e** `_as_stage_config`).

### 5.1 A perda vive em espaço de token, não de pixel

A §5.3 escreve `soma_x w(x)||I^(x) - I(x)||_1`, mas o treino é `F.mse_loss` sobre
velocidade de flow matching em tokens `(B, N, D)`. A 512², `N = 1024` e **um
token cobre 16x16 pixels**. Não existe `I^` em pixel no loop, e decodificar o
latente a cada step poria o VAE no caminho do gradiente por 60K steps.

### 5.2 Espelhar as normais não é espelhar o array

O dataloader aplica hflip aleatório. As normais são um campo vetorial: a
componente `x` aponta para o outro lado. Como o canal é gravado por `(n_x+1)/2`,
o correto é `1 - flip(canal)`. Errar isso ensinaria a rede a associar inclinação
para a direita com inclinação para a esquerda em metade das amostras, sem
levantar exceção nenhuma. Há teste dedicado.

### 5.3 As constantes não têm default, de propósito

Calibradas em 300 amostras, com a tabela de percentis impressa para ser
auditável. Um número plausível silencioso reintroduz a classe de defeito que a
auditoria encontrou.

```
tau_occlusion = 88,170     u_max      = 2,723
s_max         = 4,479      k0_curv    = 14,600
kt_max        = 7,567      z_pct_max  = 99,5
smooth_sigma  = 2,0        min_niveis = 256
```

---

## 6. Os testes que fecharam decisões

### 6.1 T1: o plano de foco é `z_focus_m`, não `s1`

`s1` e `z_focus_m` divergem (mediana 0,0031, p90 0,079, máximo 0,49). Medindo a
Eq. 4 diretamente nos pixels (mediana da profundidade dentro da máscara do
BiRefNet), em 180 amostras:

| | mediana | p90 |
|---|---|---|
| `\|Eq4 - s1\|` | 0,00321 | 0,06491 |
| `\|Eq4 - z_focus_m normalizado\|` | **0,00001** | **0,00007** |

`z_focus_m` vence em 178/180, e em 138/138 entre as não saturadas.
**Eu havia recomendado `s1`; a medição disse o contrário.**

Confirmado por um terceiro caminho: o `z_focus_m` que o nosso job F0b calcula
bate com o da tabela kfix com erro **0,0000 na mediana e no p90**, em 11.624
amostras.

### 6.2 T2: o teto do `k` é censura de faixa, não saturação

Rodado em 300 amostras da rota c:

| | censuradas (k=300) | interiores |
|---|---|---|
| `blur_ratio` mediano (menor = mais borrado) | **0,2255** | 0,4322 |
| `calibration_ssim` mediano | 0,8744 | 0,8999 |

O alvo das censuradas é **1,92x mais borrado**, e no interior
`corr(k, blur_ratio) = -0,483`. O `K` verdadeiro delas é genuinamente maior que
300. A hipótese de saturação do renderizador foi **refutada**: dentro do
interior, o SSIM não degrada de forma monótona acima de 51 px.

Corrigir exige refazer o sweep da Eq. 5. O BokehMe está no cluster mas **não
roda**: falta `cupy`, que o renderizador clássico importa.

### 6.3 T3: a focal do Depth Pro não bate com a da EXIF

Regra declarada antes: mediana abaixo de 5% e p95 abaixo de 15% para usar Depth
Pro nas duas rotas.

```
razao fx_depthpro / fx_exif  (n = 11.635)
  erro relativo: mediana = 0,171   p95 = 0,526
  dentro de 5%: 16,4%   dentro de 20%: 57,0%
```

**Não passou.** Decisão pela regra: EXIF na rota b, Depth Pro na rota c,
heterogeneidade declarada como limitação. Afeta só o canal de curvatura.

---

## 7. O resultado que temos, e é negativo

`E_bleed` medido em 7 modelos, LF-Bokeh reproduzido, 40 imagens, k-escala 3.0,
512px. A métrica usa o MESMO mapa de oclusão e o MESMO `tau` do condicionamento,
senão mediria uma região diferente da que a perda supervisiona.

| modelo | E_bleed | E_fora | razão |
|---|---|---|---|
| linha de identidade | 0,04903 | 0,02200 | 2,229 |
| **A** (fase2-real, step 15.000) | **0,03780** | 0,02648 | 1,427 |
| **A'** (15K, perda ponderada) | **0,04452** | 0,03463 | **1,286** |
| fase2-real, 60K | 0,03645 | 0,02439 | 1,494 |
| kfix, 60K | 0,04328 | 0,02633 | 1,644 |
| rota c apenas, 60K | 0,03529 | 0,02339 | 1,509 |
| oficial do paper | 0,03241 | 0,02226 | 1,456 |

O par que responde é **A contra A'**: mesmo dado, mesmo LoRA de partida, mesmos
hiperparâmetros, mesmos 15.000 steps, diferindo só na perda ponderada. O A' ficou
pior na borda (+0,0067), abaixo do limiar de 0,01 declarado antes de olhar.

**Mas o mecanismo funcionou em parte.** O A' tem a menor razão borda/fora de
todos os treinados (1,286 contra 1,427 do A): a reponderação deslocou erro para
fora da borda, que era o objetivo. O problema é que o modelo piorou em **toda
parte** (E_fora +31%), então o ganho relativo virou perda absoluta.

### 7.1 Por que piorou, medido

```
fracao de PIXELS com O > 0,3 : 0,0121
fracao de TOKENS com O > 0   : 1,0000
espalhamento do max-pool 16x : 82,7x

distribuicao de O_token: p50=0,019  p75=0,123  p90=0,639  p99=1,000
```

**A borda é 1,2% dos pixels e praticamente 100% dos tokens.** A 16x16 px por
token, quase todo token toca alguma descontinuidade. Com peso contínuo e
normalização pela média, os tokens de menor oclusão caíram para **0,816**: a
supervisão diminuiu ~18% em 74% dos tokens, para financiar o topo.

Erramos ao supor que a borda ficaria esparsa em espaço de token. A decisão de
usar max em vez de média continua certa pelo motivo original (a média diluiria a
amplitude por 1/16), mas ela não basta.

### 7.2 A correção em teste

Limiarizar o peso em token: `w = 1 + lambda [O_token > theta]`. O número que
importa não é a fração de tokens pesados, é o **piso**.

| theta | lambda | fração | piso | contraste |
|---|---|---|---|---|
| 0,0 | 2,0 | 0,992 | 0,806 | 3,00 (o A' treinado) |
| 0,3 | 2,0 | 0,167 | 0,750 | 3,00 |
| **0,9** | **1,0** | **0,075** | **0,930** | **2,00** |

Limiarizar sozinho **piora** o piso: o orçamento de peso é conservado pela
normalização, então concentrar em menos tokens só os torna mais pesados. O piso
sobe quando se reduz o `lambda` junto. Em treino com theta = 0,9 e lambda = 1,0.

---

## 8. Erros cometidos no caminho

Registrados porque cada um mudou uma decisão, e porque a maioria seria invisível.

| erro | como apareceu | correção |
|---|---|---|
| Recomendei `s1` como plano de foco | T1 mediu 178/180 a favor do `z_focus_m` | seguiu a medição |
| Reportei "PSNR 41,68" do BokehMe como medição | era arquivo de 1 de setembro; o harness lê saída existente em vez de exigir que ela tenha acabado de ser produzida | BokehMe não roda, falta `cupy` |
| Bancada de VRAM deu OOM na condição B | eu não chamava `transformer.train()`, e o checkpointing do Genfocus só age com `self.training=True` | 29,4 GB, cabe com folga |
| `_geo_init()` foi parar no dataset de deblur | o trecho que usei como âncora aparece nas duas classes | 22 testes de fiação |
| Peso contínuo rebaixou 74% dos tokens | E_fora +31% no A' | limiar em teste |
| Heurística "10-20% dos tokens" escolheu o pior theta | o piso é o que importa, não a fração | theta 0,9 + lambda 1,0 |
| Testes meus com premissa errada (2x) | rampa linear em Z não dá gradiente constante em `u`; limiar não eleva o piso sozinho | expectativas derivadas do array |
| **Disco cheio matou 3 treinos** | subi 3 simultâneos com a cota em 499G de 500G | `keep_last_n` de 5 para 2, 217G liberados |

O último custou ~18h de 3 GPUs. Só o último checkpoint de cada treino corrompeu
(o que estava sendo escrito), verificado um a um com `torch.load`, e os três
retomaram do anterior.

---

## 9. Perguntas em aberto

1. **A perda ponderada, no espaço certo.** O diagnóstico está fechado. As saídas
   que enxergamos: limiarizar (em teste), não normalizar e ajustar o lr, ou
   abandonar. Ponto conceitual: reponderar a perda de difusão de forma não
   uniforme enviesa o estimador do score. É prática padrão (min-SNR e afins), mas
   o texto tem de dizer "reponderação perceptual da supervisão", não "perda
   ponderada da verossimilhança".
2. **O teto do `k`.** 47% da rota c presa em 300. Vale instalar `cupy` para o
   BokehMe e reetiquetar, ou marcar e tirar só da avaliação de controlabilidade?
3. **A focal da rota c.** Erro de 17,1% mediano contra a EXIF, afetando só a
   curvatura. Aceitável, ou a curvatura sai da rota c?
4. **O LVCorr não sustenta limiar.** Troca de **sinal** entre modelos no mesmo
   benchmark (kfix +0,4599, oficial -0,9644 no RealDOF, amplitude 1,42). É a
   métrica de controlabilidade e é a que os canais deveriam melhorar.

---

## 10. O que vem

O B fecha primeiro. Quando terminar, as 4 GPUs dele vão para o A'', que é o
gargalo, e roda-se o `E_bleed` do B contra a tabela da seção 7.

O par **B/B'** é o que sustenta qualquer afirmação sobre os canais: os dois têm a
mesma arquitetura e o mesmo número de tokens, e diferem só em *o que* os canais
carregam. Sem ele, um ganho em B não é atribuível a informação geométrica, e isso
não sustenta publicação.

A ablação de composição de dados (Tabela 6 do paper) está 3/4 pronta de graça:
`a`, `a+c` e `a+b+c` já existem como modelos treinados; falta só `a+b`.
