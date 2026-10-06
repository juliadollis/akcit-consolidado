# Registro da frente de condicionamento geométrico

> Log corrido do que foi feito, medido e decidido. Append-only: entrada nova no
> fim, nada é reescrito. O plano vive em `PLANO_CONDICIONAMENTO_GEOMETRICO.md`,
> a auditoria dos dados em `AUDITORIA_DADOS_ROTAS_BC.md`, o código em `geo_cond/`.

---

## 2026-09-03 · auditoria dos dfs

Dois agentes independentes, métodos diferentes, mais uma terceira medição em
pixels para desempatar. Detalhes em `AUDITORIA_DADOS_ROTAS_BC.md`. O que ficou:

- rota b 11.635 linhas, rota c 2.932, schemas idênticos.
- `exif` 100% preenchida na b, 100% **NULA** na c. `calibration_ssim` o espelho.
- tabela kfix: bijeção exata com a rota b, **zero** interseção com a rota c.
- `depth` é profundidade **métrica** min-max, e `data.py:483` está correto.
  Erro de 0,011% contra 3,008% da hipótese de disparidade, medido em pixels.
- `focallength_px` é calculado pelo Depth Pro e **jogado fora** nas 3 cópias do
  pipeline. Uma passada na rota c resolve escala e focal de uma vez.
- Sentinelas: `z_max == 10000` em 25,7%; cena útil em menos de 256 níveis em
  24,7%; `k == 300` em 47,0% da rota c; mapa kfix satura em 17,3%.

## 2026-09-03 · correção de premissa na linha de base

A tabela curada, lida por inteiro, mostra que a `kfix` é o **pior** modelo
treinado nas três mesas e nas três métricas. Nove comparações concordantes.
A campanha não parte dela. Composição do dado: rotas b + c, a fase 2 do paper.

## 2026-09-04 · pacote `geo_cond`, 43 testes

`constants.py` (8 constantes, sem default), `signals.py` (6 canais),
`loss_weight.py` (peso em espaço de token), mais 43 testes.

Validado **no container do cluster** (`julia-genrefocus:1.0`), sem GPU, mount
`:ro`, sem escrever nada: `43 passed in 5.30s`. O `_pack_latents` do diffusers
0.37.1 do container é idêntico ao 0.40.0 do Mac, então a ordenação dos tokens é
a mesma nos dois.

Dois testes meus estavam errados e foram corrigidos: eu afirmava que rampa
linear em Z dá gradiente constante em `u = 1/Z`, o que é falso, e depois errei a
expectativa numérica. A correção virou um teste que afirma a propriedade física
que justifica `field="inverse"`, com a expectativa derivada do próprio array.

## 2026-09-04 · T1: o plano de foco é `z_focus_m`, não `s1`

**Eu tinha recomendado `s1`. A medição diz o contrário.**

Eq. 4 do paper medida em pixels, 180 amostras da rota b, limiar de máscara 0,5:

```
|Eq4 - s1                    |  mediana=0.00321  p90=0.06491
|Eq4 - z_focus_m normalizado |  mediana=0.00001  p90=0.00007

z_focus_m mais perto em 178/180 (98,9%)
so em z_max NAO saturado (n=138): s1 vence em 0 (0,0%)
```

O `z_focus_m` **é** a mediana da profundidade dentro da máscara do BiRefNet, a
1e-5. O `s1` não é.

**T1b, confirmação independente por nitidez** (120 amostras, janela 32 px,
variância do laplaciano sobre o bokeh REAL, sem usar máscara nenhuma):

```
|nitidez - s1      |  mediana=0.01815  p90=0.23065
|nitidez - z_focus |  mediana=0.00756  p90=0.18821
z_focus mais perto em 66/120 (55,0%)
```

Aponta na mesma direção, mas **fracamente**: 55 contra 45 é quase moeda, e os
erros são 100x maiores que os do T1 dos dois lados. A janela mais nítida é
dominada por textura, não por foco. Serve para dizer que não há contradição, não
para confirmar sozinha.

**DECISAO: `z_focus_m`.** É o que implementa a Eq. 4 literalmente, e o teste
físico independente não contradiz.

**Consequência que precisa entrar no plano:** `z_focus_m` só existe para a rota b
(tabela kfix). Para a rota c ele tem de ser calculado, e a Eq. 4 precisa de
profundidade métrica **e** da máscara. A boa notícia é que a coluna
`foreground_mask` da rota c já existe e está 100% preenchida (auditoria, V1),
então o job da rota c continua sendo só Depth Pro, e a Eq. 4 sai da máscara que
já está lá.

**Tensão que fica registrada:** o modelo `kfix`, que usa `z_focus_m`, é o pior
das três mesas. Como ele mudou três coisas de uma vez (K pela Eq. 3, plano de
foco, e `max_coc` global), isso não é atribuível ao plano de foco. Mas também não
some: se a campanha usar `z_focus_m` e for mal, este é o primeiro suspeito.

## 2026-09-04 · T2: DESVIO REGISTRADO, o renderizador não está no repositório

O T2 do plano manda refazer o sweep da Eq. 5 com `K_max = 1000`. Isso exige o
renderizador `R`, que é o `_render_bokeh_simple` do `bokehnet-data-pipeline`.

```
grep -rln "_render_bokeh_simple" .   ->  nenhum resultado
```

Reimplementar tornaria o teste inútil: mediria o NOSSO renderizador, não o que
produziu os rótulos, e a pergunta é sobre o comportamento daquele.

**Substituto, em `geo_cond/jobs/t2_teto_do_k.py`.** As duas hipóteses fazem
previsões diferentes sobre grandezas que já existem:

| hipótese | previsão testável |
|---|---|
| A, censura de faixa | o alvo das censuradas é mais borrado; `blur_ratio` acompanha o `k` |
| B, saturação do kernel de 51 px | há um **joelho** no `calibration_ssim` em torno de um CoC de 51 px, e as censuradas são as que o ultrapassam |

Mede `blur_ratio = var_lap(bokeh)/var_lap(aif)` (a mesma medida do
`audit_phase2_defocus.py`) e `coc_max_px = k · max|depth01 − s1|`.

**Se o time de dados devolver o `_render_bokeh_simple`, o teste literal da Eq. 5
volta a ser possível e este vira diagnóstico complementar.** Vale pedir.

**Segundo desvio, de acesso:** a rota c **não tem viewer** no datasets-server
(`viewer=false, filter=false, statistics=false`), então `/rows` falha em todos os
offsets. Medido: 0 de 300 amostras. O T2 foi adaptado para ler pela biblioteca
`datasets`, que é o mesmo caminho do dataloader do treino, e por isso **roda no
cluster, dentro do container, sem GPU**. Pendente da VPN voltar.

## 2026-09-04 · T2: é censura de faixa, não saturação do renderizador

Rodado no cluster, container sem GPU, 300 amostras da rota c (140 censuradas em
`k = 300`, 156 interiores).

| | censuradas (k=300) | interiores (0<k<300) |
|---|---|---|
| `blur_ratio` mediano (menor = alvo mais borrado) | **0,2255** | 0,4322 |
| `coc_max_px` implicado, mediana | 225,5 | 48,3 |
| `calibration_ssim` mediano | 0,8744 | 0,8999 |

**Hipótese A, censura de faixa: SUSTENTADA.**
O alvo das censuradas é **1,92x mais borrado** que o das interiores. E dentro do
interior, `corr(k, blur_ratio) = -0,483`: mais `k` significa alvo mais borrado,
ou seja o sweep está rastreando borrão real, não ruído. O `K` verdadeiro dessas
amostras é genuinamente maior que 300.

**Hipótese B, saturação do kernel de 51 px: NAO SUSTENTADA.**
O relatório automático do job dizia "joelho presente", mas o teste estava
confundido: 100% das censuradas têm `coc > 51 px`, então os dois grupos quase
coincidiam. Refazendo **só com as interiores**, que têm amostras dos dois lados:

```
SSIM mediano por faixa de coc_max_px, so interiores:
  [  0,  25)  n=59  SSIM=0.9110
  [ 25,  51)  n=25  SSIM=0.8877
  [ 51, 100)  n=32  SSIM=0.8537
  [100, 200)  n=24  SSIM=0.8999   <- SOBE de novo
  [200, 999)  n=16  SSIM=0.8709
```

Um teto rígido em 51 px daria degradação **monótona** acima de 51, sem
recuperação. Não é o que se vê. A queda em `[51,100)` acompanha o `blur_ratio`
(0,579 abaixo de 51 contra 0,340 acima), ou seja é "alvo mais borrado é mais
difícil de casar", não "o renderizador bateu no teto". Ressalva: `[100,200)` tem
n=24, então a recuperação é ruidosa; o que é sólido é a AUSENCIA de degradação
monótona.

**Consequência: a ação certa é alargar `K_max` e reetiquetar, o que exige o
renderizador que não temos.** Portanto:

1. **Pedir o `_render_bokeh_simple` ao time de dados** virou item de caminho
   crítico, não "vale a pena". Sem ele não dá para corrigir os rótulos de 47% da
   rota c.
2. **Enquanto isso, NAO filtrar.** Descartar custa 47% do dado que produziu o
   melhor modelo das três mesas, e o rótulo não é errado, é um limite inferior.
   As censuradas continuam ensinando aparência óptica real, que é o que a rota c
   contribui.
3. **Marcar as censuradas e excluí-las da supervisão e da avaliação de
   CONTROLABILIDADE.** Elas ensinam "300 = pelo menos isto", o que comprime o
   topo da faixa de controle. É a explicação mais direta para o LVCorr instável,
   que troca de sinal entre modelos no mesmo benchmark.

Isso vira uma flag nova no config: `k_censurado_fora_do_controle`, junto do
`min_calibration_ssim` que já existe.

## 2026-09-04 · O BokehMe ESTA instalado no cluster (desbloqueia o T2)

O `DECISOES_FASE2.md` §7 registra que o `_render_bokehme` levantava `ImportError`
incondicional porque "o BokehMe nunca foi instalado", e que por isso a rota a
usou o `_render_bokeh_simple` com kernel limitado a 51 px.

**Isso mudou e ninguém atualizou o documento.** Em `/raid/user_juliadollis/julia_docker`:

```
concorrentes/BokehMe/{demo.py, neural_renderer.py, classical_renderer/}
concorrentes/BokehMe/checkpoints/{iunet.pth, arnet.pth}     <- os pesos, presentes
valida_bokehme.py     <- harness que reproduz a Tab. 1 do paper deles no BLB
```

O `valida_bokehme.py` já existe e tem o critério certo escrito no docstring: se o
PSNR do nível 1 sair perto de 43,30, o harness está certo; se não, não se publica
nada dele.

**Consequência:** o renderizador `R` da Eq. 5 existe, e é o do paper, não o nosso
substituto. A ação que o T2 apontou (alargar `K_max` e reetiquetar as 1.379
amostras censuradas) passa a ser executável. Não depende de mais ninguém.

## 2026-09-04 · F0b rodando na GPU 7, e o percentil ja provou seu valor

`julia_f0b_rotac`, container detached, imagem `julia-genrefocus-eval:3.0` (a
`julia-genrefocus:1.0` NAO tem `depth_pro`). Smoke de 10 amostras antes de soltar
as 2.932: **2,46 s/img, zero puladas**.

Duas coisas que o smoke resolveu e valem registro:

**O `depth_pro` procura o checkpoint em `./checkpoints/depth_pro.pt` RELATIVO ao
cwd.** Nao ha parametro para apontar outro caminho em
`create_model_and_transforms()`. A convencao do projeto (ver
`run_deblur_depth.slurm:37`) e dar `cd` na raiz que contem `checkpoints/`. Aqui:
`-w /modelos` com `-v $B/models:/modelos:ro`.

**A sentinela do `z_max` apareceu logo na amostra `c_1000`:**

| | |
|---|---|
| `z_max` bruto | 10.000,0 m, o teto do Depth Pro |
| `z_max` p99,5 | 388,0 m |

O job grava os DOIS (`z_max_m_bruto` e `z_max_m`), entao a escolha fica no config
e nao embutida no dado. As 10 amostras do smoke vieram com `z_focus` valido e
dentro da faixa, e quantizacao mediana de 63.356 niveis, bem acima do gate de 256.

## 2026-09-04 · Ponte com o dataloader, e o bug do flip que ninguem veria

`geo_cond/dataloader.py`, 51 testes no total agora.

A parte dificil nao e o sinal, e a geometria. `prepare_aligned_bokeh` faz resize,
crop e hflip, e cada um mexe nos intrinsecos:

```
resize do lado menor  ->  fx' = fx * escala
crop em (x, y)        ->  cx' = cx*escala - x
hflip                 ->  cx'' = S-1-cx'  E  n_x troca de SINAL
```

**O terceiro e o que passa despercebido.** As normais sao um campo VETORIAL:
espelhar a imagem nao e espelhar o array. A componente x aponta para o outro
lado. Como o canal e gravado em [0,1] por `(n_x+1)/2`, espelhar certo e

    canal = 1 - flip_horizontal(canal)

e nao so `flip_horizontal(canal)`. Errar isso ensina a rede a associar inclinacao
para a direita com inclinacao para a esquerda em METADE das amostras, e nao
levanta excecao nenhuma. Ha um teste dedicado
(`test_flip_de_rampa_troca_o_lado_da_inclinacao`) que exige que `n_x` cruze o
neutro para o lado oposto, com a mesma magnitude.

**Decisao de projeto: calcular ANTES do crop e recortar depois.** Evita borda
falsa em cada lado do recorte (as derivadas usam padding replicado) e elimina a
contabilidade de `cx, cy` do crop. Custa calcular em ~1024x683 em vez de 512x512,
desprezivel perto do VAE. O teste `test_recorte_comuta_com_o_calculo_no_interior`
ja mostrava que as duas ordens batem no interior, entao e escolha por robustez de
borda, nao mudanca de resultado.

**A correcao do `fx` pelo resize esta na ponte e tem teste:** mesma cena a 256 e
a 128 px devolve a mesma curvatura normalizada, porque a ponte deduz a escala a
partir de `largura_px`/`altura_px` da `GeoAmostra` e aplica em `fx`.

## 2026-09-04 · F0b concluido (rota c) e DUAS correcoes de desenho

**F0b: 2.932/2.932, zero puladas, 51,5 min na GPU 7.** Cobertura 100% em
`z_min_m`, `z_max_m`, `z_focus_m` e `focallength_px`. Nenhum `z_focus` fora da
faixa. **Os 4 canais metricos estao destravados para a rota c.**

Focal estimada pelo Depth Pro: p25=2.149, p50=2.892, p99=6.521 px, todas as
imagens 2000x1500. `z_focus` (Eq. 4): p25=1,27 m, p50=2,28 m, p75=4,58 m.

### Correcao 1: o percentil NAO serve para reconstruir

Eu tinha desenhado o `z_max_m` percentilado (p99,5) para domar a sentinela de
10.000 m. **O job F0c mostrou que usa-lo na reconstrucao daria profundidade
errada.** Ajuste afim `z ~ a*depth01 + b` contra uma execucao nova do Depth Pro,
40 amostras:

```
R^2:              p10=1.00000  mediana=1.00000  min=0.99875
erro rel. de a:   mediana=0.0004  p90=0.0008
erro rel. de b:   mediana=0.0001  p90=0.0006
ambos < 5%:       38/40 (95,0%)
```

O `a` ajustado bate com `z_max_BRUTO - z_min`, nao com o percentil. Ou seja **a
coluna `depth` armazenada foi normalizada com o max bruto, ceu incluido**.
Reconstruir com o percentil produziria um `z` errado em toda amostra, e o erro
seria invisivel no mapa de defocus, que reescala tudo por `max_coc`.

**Decisao: a reconstrucao usa `z_max_m_bruto`.** O percentil continua gravado,
mas serve para diagnostico e para o gate, nao para reconstruir. Registrado na
docstring de `depth01_to_metric`.

Efeito colateral bom: isso tambem confirma, por um caminho independente, que a
coluna `depth` E profundidade metrica min-max (R^2 = 1 contra uma execucao nova
do Depth Pro), reforcando o achado da AUDITORIA secao 3.

### Correcao 2: o meu gate de quantizacao media a coisa errada

O `niveis_quant` do F0b conta niveis no mapa INTEIRO, ceu incluido, e por isso
deu numeros altos e tranquilizadores (p50 = 63.868, zero abaixo de 256). Isso
esconde o problema: numa cena com ceu no teto, os niveis estao quase todos no
fundo irrelevante enquanto a cena util vive nos primeiros milesimos.

Contando na REGIAO UTIL (`z <= 20 * z_focus`), na codificacao realmente
armazenada:

| subconjunto | n | p10 | p50 | abaixo de 256 |
|---|---|---|---|---|
| `z_max` nao saturado | 2.653 | 65.535 | 65.535 | 0 (0,0%) |
| `z_max` saturado em 10.000 m | 279 | 133 | 921 | **62 (22,2%)** |

**O problema e real mas concentrado: 62 de 2.932 amostras (2,1%).** Nelas a
derivada segunda e ruido de quantizacao, e os canais de 2a ordem saem neutros.

Nova funcao `signals.niveis_uteis(z_min, z_max_bruto, z_focus)`, com teste que
exige que ela distinga cena tipica (65.535) de cena com ceu (menos de 500),
enquanto a contagem no mapa inteiro nao distingue.

55 testes passando.

## 2026-09-04 · Integracao no genfocus_train, 74 testes passando

Feita. **19 testes originais do projeto + 55 do geo_cond, todos passando**, o que
mostra que a integracao nao quebrou o caminho existente.

| onde | o que mudou |
|---|---|
| `data.py` `DatasetRuntimeConfig` | 5 campos geo, todos com default que reproduz o comportamento anterior |
| `data.py` `prepare_aligned_bokeh` | `retornar_geometria=True` devolve o depth redimensionado, a caixa de crop e o flip |
| `data.py` `_GeoMixin` | logica compartilhada pelos dois datasets de bokeh |
| `data.py` os dois `__getitem__` | chave `geo_map` (6, S, S) so quando ligado |
| `models.py` `TrainBatchOutputs` | campo `weight` |
| `models.py` `BokehNet.make_train_batch` | 2 branches geometricos + peso da perda |
| `models.py` `flow_matching_loss` | argumento `weight` opcional |
| `trainer.py` | os DOIS call sites da perda e os DOIS sites de `DatasetRuntimeConfig` |
| `config.py` | `RuntimeConfig` (2 campos) e `StageConfig` (5) **mais** `_as_stage_config` |

**Decisao de projeto: `geo_condition=False` e o default em todo lugar.** Com ele
desligado o dataloader nao paga nada e nao acrescenta chave ao batch, e
`occlusion_lambda=0.0` reproduz a perda anterior BIT A BIT (ha teste). Isso e o
que torna a condicao A da ablacao gratuita: e o codigo novo rodando como o antigo.

**Agrupamento dos 6 canais em 2 branches, e por que nao e arbitrario.** O VAE
recebe imagens de 3 canais, entao 6 canais sao 2 branches. Como o `group_mask`
(`backbone.py:431-434`) faz cada condicao atender so a si mesma, ao texto e ao
principal, os dois branches sao MUTUAMENTE CEGOS. O que precisa ser lido junto no
mesmo pixel fica junto:

```
G1 = [ s, n_x, n_y ]   termo de PRIMEIRA ORDEM completo (magnitude E direcao)
G2 = [ u, O, K~ ]      escala, visibilidade, segunda ordem
```

A combinacao entre G1 e G2 acontece no branch principal, que enxerga tudo.

**A armadilha do `_as_stage_config` esta coberta**: `StageConfig` e montado campo
a campo, entao adicionar na dataclass sem adicionar a linha no construtor faz a
chave do YAML ser silenciosamente ignorada. E os DOIS sites de
`DatasetRuntimeConfig` no trainer (treino e smoke) foram atualizados juntos.

## 2026-09-04 · CORRECAO: o "PSNR 41,68" do BokehMe era artefato velho

Eu reportei que o harness do BokehMe dava PSNR 41,68 contra 43,30 publicado, e
tratei isso como medicao. **Estava errado, e a licao vale registro.**

O `valida_bokehme.py` faz `if not os.path.exists(p): FALHOU` e, se o arquivo
existir, le e mede. O `bokeh_pred.jpg` da cena 277 ja existia, de **1 de
setembro**. Ou seja o numero veio de um artefato de outra execucao, nao da minha.
As outras 9 cenas nao tinham artefato e falharam, o que foi o que expos o caso.

**O fato real: o BokehMe NAO roda em nenhuma das duas imagens.**

```
classical_renderer/scatter.py:7  ->  import cupy
julia-genrefocus-eval:3.0        ->  ModuleNotFoundError: No module named 'cupy'
julia-genrefocus:1.0             ->  ModuleNotFoundError: No module named 'cupy'
```

Ou seja a minha entrada anterior ("o BokehMe ESTA instalado, desbloqueia o T2")
estava certa sobre os arquivos e os pesos, e ERRADA sobre executabilidade. O
repositorio e os checkpoints estao la; a dependencia do renderizador classico
nao. Isso explica, por um caminho independente, por que o `_render_bokehme` do
pipeline de dados levantava `ImportError` incondicional: nunca foi so falta de
clonar o repo.

**Licao de metodo, que vale para o resto da campanha:** um harness que LE um
arquivo de saida em vez de exigir que ele tenha acabado de ser produzido mede o
passado. Qualquer job de reetiquetagem tem de escrever em diretorio novo por
execucao, ou apagar a saida antes, ou gravar e conferir um carimbo de tempo.

**Consequencia para o T2 e a reetiquetagem do K:**

1. O renderizador **neural** (`neural_renderer.py`) so importa torch, entao ele
   sozinho roda. Mas o `demo.py` importa o classico no topo, incondicionalmente,
   antes de qualquer flag. Usar so o neural exige um driver proprio.
2. Instalar `cupy` mexeria no ambiente que produziu os modelos treinados. Pelas
   regras do projeto, nao faco isso sem a usuaria decidir.
3. Enquanto nao houver renderizador, **a recomendacao do T2 continua valendo e e
   suficiente para treinar**: nao filtrar as 1.379 censuradas, marca-las, e
   exclui-las da supervisao e da avaliacao de CONTROLABILIDADE. Elas seguem
   ensinando aparencia optica real, que e o que a rota c contribui.

Reetiquetar melhora o topo da faixa de controle, mas **nao bloqueia o treino**.

## 2026-09-05 · F0b rota b concluida, T3 decidido, constantes calibradas

**F0b rota b: 11.635/11.635, zero puladas, 91,7 min (0,47 s/img).**

### Validacao forte, que eu nao esperava tao limpa

O `z_focus_m` que o meu job F0b calcula (Eq. 4, mediana na `foreground_mask`)
contra o `z_focus_m` gravado na tabela kfix, 11.624 amostras pareadas:

```
erro relativo: mediana = 0.0000   p90 = 0.0000
```

Zero. **O meu job reproduz o pipeline original exatamente**, o que quer dizer
tres coisas de uma vez: (1) o F0b esta correto de ponta a ponta, validado contra
uma referencia independente; (2) os numeros da rota c, onde nao ha referencia,
podem ser confiados pelo mesmo motivo; (3) e uma **terceira confirmacao** do T1,
por um caminho que nao usa nem pixels de `depth` nem o teste de nitidez.

### T3: a focal do Depth Pro NAO bate com a da EXIF

Regra declarada antes de olhar: mediana abaixo de 5% e p95 abaixo de 15% para
usar Depth Pro nas duas rotas.

```
razao fx_depthpro / fx_exif  (n = 11.635)
  p10=0.642  p25=0.769  MEDIANA=0.912  p75=1.064  p90=1.267
erro relativo: mediana = 0.171   p95 = 0.526
  dentro de  5%:  1.904/11.635 (16,4%)
  dentro de 10%:  3.633/11.635 (31,2%)
  dentro de 20%:  6.637/11.635 (57,0%)
```

**NAO PASSA.** O Depth Pro subestima a focal em ~9% na mediana, com dispersao
larga (p10 a p90 vai de 0,64 a 1,27).

**DECISAO, pela regra: EXIF na rota b, Depth Pro na rota c, heterogeneidade
declarada como limitacao.** Na rota b a EXIF e a camera real e vence. Na rota c
nao existe alternativa: a coluna `exif` e 100% NULA (auditoria V1), entao o erro
de ~17% mediano na focal e irredutivel ali e afeta so o canal de curvatura, que
e o de menor retorno esperado dos seis.

Note que isso e o OPOSTO da minha recomendacao anterior (usar Depth Pro nas duas
por proveniencia homogenea). A regra foi declarada antes e o numero decidiu.

### Constantes calibradas (rota c, 300 amostras, field=inverse, 512px)

```json
{
  "tau_occlusion": 88.17023931749958,
  "u_max": 2.7231549947753972,
  "s_max": 4.479333796246694,
  "k0_curvature": 14.599951909926324,
  "kt_max": 7.567014022271208,
  "z_percentile_max": 99.5,
  "smooth_sigma": 2.0,
  "min_quant_levels": 256
}
```

Criterio: menor percentil cuja fracao saturada fique abaixo de 1%, com a tabela
inteira impressa para ser auditavel. `k0` e a mediana de |K| nao nulo, para o log
operar onde o sinal vive. 293 das 300 amostras tinham 2a ordem valida pelo gate
de quantizacao, batendo com os 2,1% medidos na rota c inteira.

## 2026-09-05 · Smoke PASSOU com os 6 canais, e os numeros medidos

```
[data] geo: 14556 amostras com escalares (/saida/f0b_todas.jsonl)
[smoke] step=1/3 loss=0.825542
[smoke] step=2/3 loss=0.755746
[smoke] step=3/3 loss=0.635528
[smoke] PASSOU. Param diff apos 3 steps: 1.60e-04
```

### Bancada de custo, medida na GPU 3

| condicao | branches geo | s/micro-batch | s/step (accum 16) | VRAM pico |
|---|---|---|---|---|
| A' (so a perda) | nao | 1,242 | **19,9 s** | **28,2 GB** |
| B (6 canais) | sim | 2,112 | **33,8 s** | **29,4 GB** |

A estimativa do plano era 2,3x a 2,9x sobre os 14,8 s da fase 1, ou seja 34 a
43 s/step. **Medido: 33,8 s.** A estimativa estava boa, no limite inferior.

**A VRAM e a surpresa boa: 29,4 GB de 80, nao 66.** Com `gradient_checkpointing`
ligado sobra muita folga. O numero de 66 GB do `historico-ultimo.md` e com
checkpointing DESLIGADO.

### Um bug na minha bancada, que vale registro

A primeira medicao deu 71,1 GB para A' e **OOM para B**. Era erro meu: eu nao
chamava `transformer.train()`, e o `transformer_forward` do Genfocus so ativa o
checkpointing quando `self.training and self.gradient_checkpointing`
(`flux.py:426` e `445`). O trainer faz isso em `trainer.py:604`; a bancada nao
fazia. Sem o `.train()` a medida fica 2,4x acima do treino real e leva a
conclusao oposta (nao cabe, quando cabe com folga).

Licao: bancada de VRAM tem de reproduzir o MODO do treino, nao so as chamadas.

### Projecao com os numeros medidos

```
A'       15K = 3,5 dias    60K = 13,8 dias
B        15K = 5,9 dias    60K = 23,5 dias
B'       15K = 5,9 dias
```

Com 6 GPUs em 3 pares (2 GPUs x accum 16 = batch efetivo 32, igual ao paper), as
tres condicoes rodam em paralelo e o screen de 15K fecha em **~6 dias**.

### Bug de fiacao que os 74 testes nao pegaram

O `self._geo_init()` foi parar no `HuggingFaceDeblurDataset` em vez do de bokeh:
o trecho que usei como ancora para a insercao aparece nas DUAS classes e o
replace pegou a primeira. Passou nos 74 testes porque nenhum instancia o dataset
de bokeh com geo ligado (exigiria rede); so apareceu no smoke.

Corrigido, e coberto por `geo_cond/tests/test_integracao.py`, 22 testes novos que
olham a FIACAO e nao so o comportamento: qual classe chama `_geo_init`, se o
`_as_stage_config` le cada campo (a armadilha do ignorar em silencio), se os DOIS
sites do trainer repassam, e se o default reproduz o comportamento anterior.

**96 testes passando.**

## 2026-09-05 · A' TREINANDO, e o E_bleed pronto

**`julia_geo_Alinha` no ar**, GPUs 0-3.

```
[export] 688 tensores LoRA carregados de outputs/bokehnet_synth_2gpu/bokeh/checkpoints/step_40000.pt
[train] fase 2: LoRA inicializado da fase 1 (step 40000)
[train] stage=bokeh steps=15000 start=0 num_gpus=4 grad_accum=8 batch_efetivo=32
```

Batch efetivo 32, identico ao paper e identico a fase 2, entao os checkpoints sao
diretamente comparaveis aos modelos ja treinados. Snapshot a cada 1.000 steps no
repo `bokehnet-geo-Alinha`, que nao colide com nada.

**A' e a condicao certa para ir primeiro**: e o item que o documento de proposta
chama de melhor custo-beneficio (nao muda arquitetura, nao adiciona parametro,
nao muda inferencia), e o mais barato (19,9 contra 33,8 s/step em 2 GPUs), e ataca
direto o vazamento de cor. Se ele nao der ganho, a hipotese da perda morreu antes
de gastar 6 dias em B.

Duas armadilhas de lancamento, para nao repetir:
- o `docker_bokeh_bootstrap.sh` faz `cd /workspace/genrefocus_deblurnet_paper`,
  entao o mount tem de ser o `julia_docker` INTEIRO em `/workspace`, nao a pasta
  do projeto. Montar so o projeto da `No such file or directory` e sai(1).
- `--gpus` com varias GPUs exige as aspas EMBUTIDAS: `"\"device=0,1,2,3\""`.

### E_bleed implementado, `geo_cond/ebleed.py`

Usa o MESMO mapa de oclusao e o MESMO `tau` do condicionamento. Isso nao e
detalhe: com outro limiar a metrica mediria uma regiao diferente da que a perda
supervisiona, e a comparacao entre condicoes deixaria de ser sobre a mesma coisa.

Reporta o par (dentro, fora) de proposito: sem a regiao complementar, um modelo
que borra tudo poderia baixar o E_bleed por acidente.

Testes: erro colado na descontinuidade aparece; erro longe dela nao; predicao
perfeita zera os dois; cena plana devolve `nan` explicito em vez de um numero que
parece medida; e depth em resolucao diferente da imagem ABORTA, porque ali a
borda cairia no lugar errado e o numero sairia plausivel e sem sentido.

## 2026-09-05 · A' e B treinando

| condicao | GPUs | accum | batch efetivo | repo HF |
|---|---|---|---|---|
| A' (perda ponderada) | 0,1,2,3 | 8 | **32** | `bokehnet-geo-Alinha` |
| B (perda + 6 canais) | 5,6 | 16 | **32** | `bokehnet-geo-B` |

GPU 4 e de outra pessoa. GPU 7 fica para a avaliacao do E_bleed.

**Nao usei 3 GPUs no B, e a razao importa.** 3 nao divide 32: daria accum 11 e
batch efetivo 33. B passaria a diferir de A' em DUAS coisas (os canais E o batch),
e a ablacao deixaria de ser limpa. Com 2 GPUs o batch e 32 exato, igual a A',
igual a fase 2 e igual ao paper, e os checkpoints sao comparaveis sem ressalva.

### O guard fez o que devia, e derrubou o primeiro lancamento

```
KeyError: "sem escalares geometricos para 'b_010604'. Rode o F0b na rota dela,
ou use geo_sem_escalares='pula'. NUNCA invente escalares."
```

11 das 11.635 amostras da rota b (**0,076%**) nao tem `z_focus_m`, porque a
`foreground_mask` delas e vazia ou tem menos de 64 px. Sem mascara a Eq. 4 nao e
definida, entao elas nao tem plano de foco.

O codigo se recusou a inventar um, que era o comportamento pedido. Mas o lugar de
tratar isso nao e no worker: uma excecao dentro do DataLoader derruba o rank
inteiro. Implementado `_filtrar_sem_escalares`, no MESMO padrao do
`_filter_by_calibration_ssim` que ja existia: filtra na carga, conta e avisa.

```
[data] geo: 11/14486 amostras sem escalares (0.076%) descartadas
```

E aborta se o filtro zerar o dataset, senao um erro de configuracao (tabela
errada, rota nao coberta) viraria um treino silenciosamente vazio.

**102 testes passando.**

### Ordem da campanha, e por que A' primeiro

A' e o item que o documento de proposta chama de melhor custo-beneficio: nao muda
arquitetura, nao adiciona parametro, nao muda a inferencia. Se ele nao der ganho,
a hipotese da perda morreu barato. B esta rodando em paralelo porque ha GPU
livre, e porque a pergunta dele ("a geometria acrescenta SOBRE a perda?") so faz
sentido com os dois medidos.

B' (controle de ruido) entra depois, e so se B ganhar. Sem ele um ganho em B nao
e atribuivel a informacao geometrica, e isso nao sustenta publicacao.

## 2026-09-06 · 24h de treino, e o E_bleed validado na linha de base

| condicao | step | ritmo real | previsto na bancada | falta |
|---|---|---|---|---|
| A' (4 GPUs) | 8.230/15.000 | 10,6 s/step | 9,9 | ~20 h |
| B (2 GPUs) | 2.630/15.000 | 33,1 s/step | 33,8 | ~4,7 dias |

**A bancada acertou**: 10,6 medido contra 9,9 previsto, e 33,1 contra 33,8. A
diferenca do A' e o overhead de all_reduce em 4 GPUs, que a bancada de 1 GPU nao
capturava. Loss caindo nos dois (A' ema 0,185; B ema 0,193).

Snapshots subindo: 8 no `bokehnet-geo-Alinha`, 2 no `bokehnet-geo-B`.

### E_bleed medido na linha de identidade (LF-Bokeh, n=40)

```
E_bleed (borda) : media=0.04903  mediana=0.04966
E_fora          : media=0.02200  mediana=0.02012
razao borda/fora: 2.229
fracao de borda : mediana=0.0439
```

**A metrica mede o que devia.** O erro na regiao de borda e **2,23x** o de fora,
com a borda ocupando so 4,4% dos pixels. Isso responde a pergunta que o portao
existia para responder: o artefato de borda e real e concentrado, e ha espaco
para melhora. Se a razao fosse ~1, nao haveria artefato a atacar e a campanha
inteira perderia o sentido.

A fracao de 4,4% tambem valida o `theta = 0,3`: nem ~0 (regiao vazia, metrica sem
sentido) nem ~1 (regiao e a imagem toda, metrica igual a global).

**Sobre a profundidade nos benchmarks:** nenhum dos tres tem coluna `depth`, o que
parecia bloquear. Nao bloqueia: o job roda o Depth Pro na hora, como o proprio
pipeline de avaliacao ja faz (`bokeh_net.py:93-141`), e como e a MESMA
profundidade em todas as condicoes comparadas, o estimador cancela na comparacao
pareada.

## 2026-09-07 · A' terminou, e o resultado e NEGATIVO. Com um porem.

A' fechou 15.000/15.000 em ~1,8 dia (4 GPUs). Inferencia dos 6 modelos no
LF-repro (40 imagens, k-escala 3.0, 512px, o mesmo protocolo da identidade).

### A tabela

| modelo | E_bleed | E_fora | razao borda/fora |
|---|---|---|---|
| linha de identidade | 0,04903 | 0,02200 | 2,229 |
| **A** (fase2-real, step 15.000) | **0,03780** | 0,02648 | 1,427 |
| **A'** (15K, perda ponderada) | **0,04452** | 0,03463 | **1,286** |
| nosso (fase2-real, 60K) | 0,03645 | 0,02439 | 1,494 |
| kfix (60K) | 0,04328 | 0,02633 | 1,644 |
| rotac-only (60K) | 0,03529 | 0,02339 | 1,509 |
| oficial do paper | 0,03241 | 0,02226 | 1,456 |

O par que responde a pergunta e **A contra A'**: mesmo dado, mesmo LoRA de
partida, mesmos hiperparametros, mesmos 15.000 steps, diferindo SO na perda
ponderada.

```
dE_bleed = +0,00672   (pior)
dE_fora  = +0,00815   (pior)
drazao   = -0,142     (melhor)
```

**Leitura honesta: a perda ponderada nao entregou.** O A' e pior na borda em
termos absolutos, e o |dE_bleed| de 0,0067 fica ABAIXO do limiar de 0,01 que
declaramos antes de olhar, entao nem como "piora significativa" ele qualifica.

**Mas o mecanismo funcionou parcialmente, e isso importa para o proximo passo.**
O A' tem a MENOR razao borda/fora de todos os modelos treinados (1,286 contra
1,427 do A). Ou seja a reponderacao DE FATO deslocou erro para fora da borda,
que era exatamente o objetivo. O problema e que o modelo piorou em TODA parte
(E_fora subiu 31%), entao o ganho relativo virou perda absoluta.

### Por que piorou em toda parte: medido, nao especulado

`geo_cond/jobs/f2_diag_peso.py`, 40 amostras da rota c:

```
fracao de PIXELS com O > 0.3 : 0,0121
fracao de TOKENS com O > 0   : 1,0000
fator de espalhamento do max-pool 16x: 82,7x

 lambda  tokens w>1    w_min    w_p10    w_max
    0.5      0.2583   0.9467   0.9482   1.3936
    1.0      0.2583   0.8988   0.9016   1.7351
    2.0      0.2583   0.8163   0.8215   2.3228     <- o usado no A'
    3.0      0.2583   0.7479   0.7549   2.7915
```

**A borda e 1,2% dos PIXELS e 100% dos TOKENS.** O max-pool 16x espalha por um
fator de **82,7x**: praticamente todo token de 16x16 toca alguma descontinuidade.

Com a normalizacao pela media, isso significa que os tokens de menor oclusao
foram rebaixados a **0,816** enquanto os de maior subiram a 2,32. Ou seja, com
lambda = 2,0, a supervisao caiu ~18% em 74% dos tokens. O modelo treinou com
menos sinal na maior parte da imagem, e piorou ali, que e exatamente o que o
E_fora +31% mostra.

**A decisao de usar max em vez de media continua certa** pelo motivo original (a
media diluiria a amplitude por 1/16). O que estava errado era supor que a borda
ficaria esparsa em espaco de token. Nao fica: a 16x16 px por token, a 512, quase
tudo toca borda.

### O que fazer, e o que NAO fazer

Nao vale reciclar o A' com lambda menor sem mudar mais nada: com lambda 0,5 o
piso sobe para 0,947, ou seja o efeito quase some, e a condicao vira A com ruido.
O caminho que o diagnostico aponta:

1. **Peso em espaco de PIXEL, com a perda ainda em token.** Nao da: a perda vive
   em token e decodificar o latente por step poria o VAE no gradiente.
2. **Limiar no peso**: `w = 1 + lambda * (O_token > theta)`, com theta alto o
   suficiente para o peso ser esparso EM TOKEN. Com o espalhamento de 82,7x,
   theta teria de subir muito, e isso e mensuravel antes de treinar.
3. **Nao normalizar pela media, e reduzir o lr na proporcao.** Separa a
   reponderacao do rebaixamento, ao custo de mais um hiperparametro.

A opcao 2 e a mais direta e da para calibrar com o f2_diag_peso ANTES de gastar
GPU: escolher theta tal que a fracao de tokens com peso alto fique em torno de
10 a 20%, e nao 100%.

### Consequencia para B e B'

B (6 canais) e B' (controle) usam o MESMO lambda = 2,0 e portanto herdam o mesmo
rebaixamento. Isso NAO invalida o par B/B': os dois sofrem igual, e a comparacao
entre eles continua isolando informacao contra capacidade. Mas o valor absoluto
dos dois vai carregar a mesma penalidade do A', e a comparacao contra A tem de
levar isso em conta.

Deixei os dois rodando por essa razao: a pergunta "a geometria acrescenta" ainda
e respondida pelo par, e interromper agora perderia 2 dias de B por um defeito
que afeta os dois lados igualmente.

## 2026-09-07 · A'' -- a correcao que o diagnostico apontou

### A distribuicao de O_token, medida

```
p1=0.0003  p10=0.0014  p25=0.0042  p50=0.0191  p75=0.1225  p90=0.6387  p99=1.0000
```

Fortemente assimetrica. **99,2% dos tokens tem O_token > 0, mas a mediana e
0,019.** Com peso continuo `w = 1 + 2*O_token` e normalizacao pela media, isso
significa que o token mediano fica em 0,84 e o piso em 0,806: a reponderacao
financiou o topo cobrando de praticamente toda a imagem.

### O criterio de escolha, e o erro que ele corrigiu

Minha primeira heuristica foi "escolher theta que deixe 10 a 20% dos tokens
pesados". Ela recomendou theta=0,3, que e **o pior ponto da tabela**: piso 0,750,
ainda mais baixo que o do A'.

O numero que importa nao e a fracao, e o **piso**, porque ele mede quanto a
supervisao cai onde nao ha borda. Piso = `1 / (1 + lambda * fracao)`:

| theta | lambda | fracao | piso | topo | contraste |
|---|---|---|---|---|---|
| 0,0 | 2,0 | 0,992 | 0,806 | 2,42 | 3,00 |  <- o A' treinado |
| 0,3 | 2,0 | 0,167 | 0,750 | 2,25 | 3,00 |
| 0,9 | 2,0 | 0,075 | 0,870 | 2,61 | 3,00 |
| **0,9** | **1,0** | **0,075** | **0,930** | **1,86** | **2,00** |

**Escolhido: theta = 0,9, lambda = 1,0.** Piso 0,930 contra 0,806 do A', e o peso
alto cai so nos 7,5% de tokens que sao borda de verdade. O contraste cai de 3,0
para 2,0, o que e o preco: menos amplitude, mas paga por quase toda a imagem em
vez de por 7,5% dela.

### Um teste meu estava errado, de novo pelo mesmo motivo

Escrevi `test_theta_protege_os_tokens_sem_borda_da_normalizacao` afirmando que o
limiar ELEVA o piso. Nao eleva por si: o orcamento de peso e conservado pela
normalizacao. Com theta=0,6 e lambda fixo o piso ate CAI, porque o peso alto fica
concentrado em menos tokens e portanto e mais alto.

O que o limiar entrega e CONTRASTE dirigido: peso alto so onde ha borda. O piso
sobe quando se reduz o lambda JUNTO, que e o par (0,9 / 1,0) escolhido. Teste
reescrito para afirmar o que e verdade.

### A'' no ar

`julia_geo_A2linha`, GPU 7, 1 GPU com accum 32 = batch efetivo 32, o mesmo de
todas as outras condicoes. `occlusion_theta: 0.9`, `occlusion_lambda: 1.0`.

Ocupacao total: 0-3 = B', 5-6 = B, 7 = A''. So a GPU 4, de outra pessoa, fora.

**105 testes passando**, incluindo um novo que segue o `occlusion_theta` do YAML
ate a perda, elo por elo, porque cada elo foi editado a mao.
