# Auditoria da avaliação de bokeh

Feita em 2026-09-01/02, a pedido de uma exigência simples: que os resultados
estejam certos. Este documento lista **o que foi verificado, o que estava
errado, e o que continua sem resposta**. A tabela em si está em
`TABELA_FINAL.md` e nos repos do Hub listados no fim.

---

## 1. A prova de que o pipeline de métrica está certo

O teste que vale é pegar o peso **oficial** do paper, rodar no **nosso** código,
e ver se sai o número **publicado** por eles.

**DeblurNet, Tab. 2 do paper:**

| | medido aqui | publicado |
|---|---|---|
| RealDOF LPIPS | 0,2385 | 0,2408 |
| RealDOF DISTS | 0,1161 | 0,1126 |
| DPDD LPIPS | 0,1537 | 0,1440 |
| DPDD DISTS | 0,0779 | 0,0772 |

Um a sete por cento de diferença, reproduzindo número publicado. O empilhamento
de métricas (`lpips+`, `dists`, CLIP ViT-B/32) está correto.

**BokehMe, Tab. 1 do paper deles, no BLB nível 1:** medido PSNR 41,68 contra
43,30 publicado, numa cena e num plano de foco (eles reportam a média sobre 100
amostras). Perto o bastante para validar o harness do concorrente.

Não existe prova equivalente do lado do **bokeh do GenRefocus**, e a razão está
na seção 4.

---

## 2. Erros encontrados e corrigidos

### 2.1 O sinal do LVCorr estava invertido em relação ao paper

`eval_bokeh_synthesis.py` calcula `pearsonr(K, variância do Laplaciano)` cru.
Fisicamente, mais K pede mais desfoque, e mais desfoque **reduz** a variância do
Laplaciano: um modelo que obedece o K tem correlação **negativa**. O paper
reporta LVCorr com "quanto maior melhor" e o GenRefocus deles em **+0,9368**.

Consequência na tabela antiga: o **FLUX sem treino aparecia com +0,9769, o
melhor de todos**, e os nossos bons modelos com -0,82. Qualquer leitura daquela
coluna concluía que o modelo não treinado era o mais controlável.

Corrigido: a nova tabela traz `LVCorr_convencao_paper` (simétrico) ao lado do
Pearson cru.

### 2.2 Cinco repos de imagem tinham linhas duplicadas

As voltas do keeper reprocessaram itens e **anexaram** linhas repetidas:

| repo | linhas | únicas |
|---|---|---|
| `bokeh-eval-lfrepro-oficial` | 840 | 500 |
| `bokeh-eval-lfrepro-nofilter28k` | 980 | 500 |
| `bokeh-eval-lfrepro-rotac60k` | 920 | 500 |
| `bokeh-eval-rb-kfix-full` | 407 | 217 |
| `bokeh-eval-rb-nosso-full` | 377 | 217 |

Medir por cima disso pondera algumas cenas em dobro. A linha agregada antiga
tinha sido calculada **antes** da duplicação, então ela estava certa e a
recontagem é que vinha errada, o que só ficou visível porque o SSIM (código que
nunca mudou) divergia. Corrigido com deduplicação por `file_name_base`.

### 2.3 O `n` do RealDOF entrava como nulo

São **50** imagens, 50 cenas. Agora medido, não suposto.

### 2.4 A tabela `deblur-metrics` está obsoleta e contradiz a validada

`juliadollis/deblur-metrics` usa `lpips` simples e diz que **ganhamos** do
oficial na DPDD (0,1446 contra 0,1553). `juliadollis/tab2-metricas` usa `lpips+`,
que é a variante que reproduz o paper, e diz o contrário:

| | nosso 60k | oficial |
|---|---|---|
| RealDOF LPIPS | 0,2433 | 0,2385 |
| DPDD LPIPS | 0,1930 | 0,1537 |

**Não ganhamos do oficial no DeblurNet.** Usar `tab2-metricas`; a `deblur-metrics`
não deve ser citada.

### 2.5 O `k_ref` da nossa mesa BLB está no espaço errado

O `info.json` do BLB traz `blur_parameters` de 138 a 692, e foi isso que entrou
na coluna `k_ref`. Mas o K que reproduz os alvos do BLB, com a disparidade
normalizada em [0,1] que o próprio BLB distribui, é **10, 20, 30, 40, 50** por
nível. Verificado empiricamente: renderizando com K=138,47 o PSNR é 17,85;
com K=10 é 41,68 contra 43,30 publicado.

A razão entre os dois é a disparidade máxima da cena (`blur_parameter / K = 1/menor
profundidade`), e ela vale para 8 das 10 cenas com min(`focus_distances`); nas
cenas 279 e 293 o normalizador é a disparidade real da cena, não a lista de
focos.

Isso **não afeta** os números de LPIPS/DISTS já medidos, porque o pipeline faz
busca binária e nunca lê `k_ref`. Afeta quem usar a coluna, e afeta a afirmação
do cartão do dataset de que o K de referência é uma vantagem sobre o LF-Bokeh.


### 2.6 O plano de foco NAO era a Eq. 4 em nenhuma campanha (o mais grave)

`einops` e `kornia` faltavam na imagem Docker, o BiRefNet nunca importou, e o
plano de foco caiu no **pixel central** em toda a campanha, inclusive na curada.
O proprio codigo avisava em todo log:

```
[FASE 1b] ATENCAO: BiRefNet indisponivel (ImportError: einops, kornia)
[FASE 1b] Sem mascara o plano de foco cai no PIXEL CENTRAL, que NAO e a Eq. 4.
```

Enquanto isso a tabela curada declarava "plano de foco pela Eq. 4 (BiRefNet)".

**Por que importa.** A Eq. 4 do paper e `D_focus = mediana(D[M])`, a profundidade
mediana dentro da mascara do objeto em foco. Ela alimenta a Eq. 2,
`D_def = K |D - D_focus|`, que e o mapa que condiciona toda a geracao. `D_focus`
e o plano de referencia: errar isso nao borra "um pouco errado", desloca o campo
inteiro. Com o pixel central, se o assunto nao esta no meio do quadro o foco cai
no fundo e o modelo faz o oposto do pedido.

**Corrigido** instalando as duas bibliotecas num diretorio a parte e montando no
PYTHONPATH. Agora sao 46 de 50 imagens na RealDOF e 189 de 217 no RealBokeh
usando a Eq. 4 de verdade (o resto e mascara vazia, e ai o centro e o fallback
previsto no codigo).

**O efeito, medido.** LPIPS, mesmo modelo, mesma mesa, so o plano de foco muda:

| mesa | modelo | pixel central | Eq. 4 | diferenca |
|---|---|---|---|---|
| RealDOF | **oficial** | 0,2677 | **0,1900** | **-0,078** |
| RealDOF | rotac60k (nosso) | 0,1263 | 0,1178 | -0,009 |
| RealDOF | nosso fase 2 | 0,2148 | 0,2157 | +0,001 |
| RealBokeh | **oficial** | 0,3454 | **0,2867** | **-0,059** |
| RealBokeh | rotac60k (nosso) | 0,1134 | 0,1122 | -0,001 |

O plano de foco errado penalizava **o modelo oficial muito mais que os nossos**.
E coerente: o oficial obedece o condicionamento com muito mais fidelidade
(LVCorr +0,94), entao informacao errada estraga mais o resultado dele. Os nossos
ignoram parte do condicionamento e sofrem menos com a mesma informacao errada.

**Consequencia na afirmacao principal.** Na RealDOF, a vantagem do nosso
`so rota c` sobre o oficial era -0,141 de LPIPS. Com a Eq. 4 correta e **-0,072**.
Continuamos ganhando, com folga, mas por **metade** do que a tabela anterior
dizia. Qualquer numero da campanha do pixel central esta aposentado.

**Ressalva que vai junto.** No item (c) do paper, sobre LFDOF e RealBokeh, os
autores dizem que a mascara do BiRefNet "as vezes e pouco confiavel" em cenas
complexas e que por isso adicionaram um **passo manual** de correcao. Esse passo
nos nao temos. A nossa Eq. 4 e a parte automatica do procedimento deles.

### 2.7 Uma mesa nova, porque a antiga nao tinha poder

`juliadollis/bokeh-bench-ebb400`: 400 cenas do EBB! (Everything is Better with
Bokeh), pares fotografados com Canon 7D, a mesma cena em f/16 e f/1.8. **Uma
cena por linha**, contra as 10 cenas do LF-repro.

Construcao: medi a razao de desfoque dos **4.400 pares antes de escolher
qualquer um** (escolher primeiro e medir depois enviesaria a mesa). Mediana
0,313, ou seja, o alvo e cerca de tres vezes menos nitido que a entrada.
**57 pares tinham o alvo MAIS NITIDO que a entrada** (rotulo trocado ou
desalinhamento) e foram descartados. Dos 4.150 que passaram, sorteei 400 com
semente 42.

Ela nao substitui o LF-repro em tudo: cada cena tem um unico nivel de desfoque,
entao para controlabilidade o LF-repro continua sendo o unico lugar com alvo
conhecido por nivel de K.

---

## 3. O erro estatístico que mudava a leitura

**O `LF-Bokeh reproduzido` tem 500 imagens mas só 10 cenas.** Cada cena aparece
50 vezes, com 5 aberturas e 10 planos de foco. Tratar as 500 como independentes
encolhe o intervalo de confiança por um fator de cerca de sete e inventa
precisão que o dado não tem.

Com o bootstrap **agrupado por cena**, o tamanho efetivo de amostra ali é 10, e
o resultado é que **nada é significativo nessa mesa**: a vantagem do nosso melhor
modelo sobre o oficial dá p=0,22, e mesmo a linha de identidade só chega a
p=0,058. Aquela mesa não distingue modelo nenhum.

RealBokeh (217 imagens, 217 cenas) e RealDOF (50 e 50) são uma cena por linha, e
ali o agrupamento não muda nada.

---

## 4. O que não tem conserto por enquanto

**O LF-Bokeh do paper não foi liberado.** O roadmap do repositório oficial lista
"Release Benchmark data" como tarefa futura, e o código de avaliação também não
saiu. Só saíram pesos e scripts de inferência.

Consequências:

- Não é possível reproduzir a Tab. 3. Rodando os pesos **oficiais** na nossa
  reconstrução (BLB), o LPIPS dá 0,2047 contra 0,0833 publicado, e não há como
  saber se a diferença é a mesa ou o pipeline.
- Não existe, do lado do bokeh, a prova que existe do lado do DeblurNet.
- **Construir uma mesa própria deixou de ser opcional.** O BLB tem só 10 cenas
  (é todo o dataset, não uma amostra), e é justamente o que trava a estatística.
  Uma mesa nova, com muitas cenas, resolve as duas coisas de uma vez e é
  contribuição publicável.

Segue de pé, do registro anterior: a **procedência da rota a** não permite
certificar que um benchmark esteja fora do treino da fase 1.

---

## 5. O que a tabela mostra agora

Detalhe completo em `TABELA_FINAL.md`. O resumo:

**Fidelidade, RealDOF, fora do treino dos dois modelos, 50 cenas.** O nosso
`só rota c` bate o oficial por -0,141 de LPIPS, IC95 [-0,172, -0,110], p<0,001
no pareado cena a cena. É o resultado mais forte que temos.

**Controlabilidade, mesma mesa.** O oficial tem LVCorr +0,9644 e o nosso melhor
tem **-0,21**. Perdemos feio. E há um agravante: ao longo do treino do `só rota
c` a controlabilidade **degrada** (+0,25 no step 10.000, -0,21 no step 45.000)
enquanto a fidelidade melhora. É a assinatura esperada do defeito da rota b, em
que o K não chega a ser aprendido.

**Ablação de dados, no formato da Tab. 6 deles.** Eles medem que somar tudo é o
melhor: (a) 0,1289, (a+c) 0,0972, (a+b+c) 0,0833. Nós medimos o oposto: o
`só rota c` (a+c) bate o `fase 2` (a+b+c) nas três mesas. Com o defeito da rota b
documentado, isso é resultado, não anomalia.

---

## 6. Onde está tudo

| o que | onde |
|---|---|
| Métricas por imagem, 8.138 linhas, 69 repos | `juliadollis/bokeh-metricas-por-imagem` |
| Tabela final com IC95 agrupado por cena | `juliadollis/genrefocus-tabela-final` |
| Testes pareados cena a cena | `juliadollis/genrefocus-tabela-final-pareado` |
| DeblurNet validado contra o paper | `juliadollis/tab2-metricas` |
| Tabela curada anterior (LVCorr com sinal cru) | `juliadollis/genrefocus-resultados-curados` |
| Log bruto de append, não publicar direto | `juliadollis/bokeh-eval-metricas` |

Scripts novos: `vision-pipeline/metricas_por_imagem.py` (métrica por imagem, com
conferência automática contra a tabela antiga) e
`vision-pipeline/tabela_estilo_paper.py` (tabela final, IC agrupado, pareado).
