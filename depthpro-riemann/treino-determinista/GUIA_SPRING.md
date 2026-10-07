# Guia Spring — o dataset onde o DepthPro é mais fraco em bordas

Documento de implementação. O Spring é o alvo científico mais importante da segunda fase:
é onde a acurácia de borda reportada do DepthPro é a mais baixa de todas, e é sintético com
ground truth pixel-perfeito, o que torna a nossa métrica de borda confiável.

Este guia é mais longo que o do DIODE por um motivo: o Spring **não distribui
profundidade**, e sim disparidade estéreo. A conversão tem armadilhas que produzem
resultados plausíveis à vista mas errados nos números, então há uma seção inteira de
verificação.

---

## 1. Por que o Spring

Depois de descobrir que o Hypersim está no treino do DepthPro, precisávamos de datasets
fora daquele domínio. O Spring atende por dois critérios ao mesmo tempo:

| Critério | Situação |
|---|---|
| Está no treino do DepthPro? | Não. A tabela de datasets o marca apenas como teste. |
| O DepthPro vai bem nele? | Não. F1 de borda reportado: **0.079**, contra 0.409 no Sintel e 0.176 no iBims. |
| Ground truth confiável para borda? | Sim. Sintético, denso, pixel-perfeito, e o dataset foi construído para ter detalhe fino. |
| Quantidade de cenas | **37 sequências** com ground truth público, contra 29 cenas no nosso teste do Hypersim. |

Esse F1 de 0.079 é a razão central para o esforço: é a maior margem de melhoria disponível
justamente na métrica que a nossa hipótese ataca. Se a curvatura tiver mérito, é ali que
vai aparecer.

Um contraste que vale registrar: o Sintel tem o pior δ1 do DepthPro (40.0), o que o faria
parecer um alvo melhor. Descartamos porque o modo de falha lá é névoa, transparência e
espalhamento volumétrico — limitações que o próprio paper deles reconhece e que a nossa
perda geométrica não resolve. Seria um experimento que não responde à pergunta.

---

## 2. A conversão, em detalhe

A relação é

```
Z = fx · B / d
```

onde `Z` é a profundidade em metros, `fx` a distância focal em pixels, `B` a linha de base
estéreo e `d` a disparidade em pixels. No Spring, **B é sempre 0.065 m**, e `fx` vem do
`intrinsics.txt` de cada sequência.

### 2.1 A armadilha da resolução dobrada

O ground truth do Spring é distribuído em resolução maior que o RGB, para dar precisão
subpixel. Isso importa porque **disparidade é medida em pixels**: o mesmo deslocamento
físico corresponde ao dobro de pixels numa grade com o dobro da densidade.

Então ao reduzir o mapa de disparidade para a resolução do RGB, é preciso dividir os
**valores** pelo mesmo fator. Quem esquece isso obtém profundidade pela metade, de forma
uniforme — e como o nosso pipeline alinha predição e ground truth por transformação afim,
um erro uniforme de escala **passaria despercebido nas métricas** e só apareceria como
absurdo se alguém olhasse os metros absolutos.

O script detecta a razão entre as resoluções e aplica o fator automaticamente, reportando
no log qual valor usou. O parâmetro `--disp-escala` permite forçar, para conferência.

### 2.2 Céu e disparidade próxima de zero

O Spring inclui o céu no ground truth de propósito, diferente de outros datasets que o
deixam vazio. Onde a disparidade tende a zero, `Z` explode. Esses pixels são **removidos da
máscara**, e não limitados a um teto — limitar criaria uma parede falsa a uma distância
fixa, que a nossa perda de curvatura interpretaria como geometria real.

Dois parâmetros controlam isso: `--min-disp` (disparidade mínima em pixels, padrão 0.05) e
`--max-depth` (padrão 100 m).

### 2.3 Por que subamostrar quadros

Cada sequência tem cerca de 130 quadros, e quadros consecutivos de uma animação são quase
idênticos. Usar todos multiplica o custo sem acrescentar informação, e pior: infla
artificialmente o número de amostras. Para a nossa estatística, a unidade de agrupamento é
a **sequência**, e é o número de sequências que determina o poder do teste. O `--passo`
subamostra; o padrão de 10 dá cerca de 13 quadros por sequência.

---

## 3. O que muda no pipeline

Nada além do script novo. O `scripts/prepare_spring.py` escreve no mesmo layout
`rgb/`, `depth/`, `mask/` que o DIODE e o Hypersim usam, então o `dataset.py`, as perdas, o
treinador, as métricas, a estatística e as figuras funcionam sem alteração.

A pasta `mask/` já era suportada desde o suporte ao DIODE, e é essencial aqui por causa do
céu.

---

## 4. Sobre os splits

O ground truth do split de **teste** do Spring é fechado: fica no servidor do benchmark e
não é distribuído. Por isso usamos o split de **treino** do Spring, que tem GT público.

Como não estamos treinando no Spring nesta fase, isso não configura vazamento — estamos
apenas avaliando modelos treinados no Hypersim. Mas se um dia formos treinar no Spring,
será preciso dividir as 37 sequências em treino, validação e teste por conta própria, com
sequências disjuntas. Deixo o alerta registrado porque é fácil esquecer.

---

## 5. Passo a passo

### 5.1 Baixar

O Spring está em `spring-benchmark.org`. Para o nosso uso bastam três pacotes do split de
treino, o que reduz muito o download em relação ao dataset completo:

| Pacote | Para quê |
|---|---|
| `train_frame_left.zip` | imagens RGB da câmera esquerda |
| `train_disp1_left.zip` | disparidade de referência da esquerda |
| `train_cam_data.zip` | intrínsecos, de onde sai o `fx` |

Descompacte preservando a estrutura `<raiz>/<sequência>/frame_left/` e
`<raiz>/<sequência>/disp1_left/`.

Não precisamos dos pacotes de fluxo óptico, de `disp2`, nem da câmera direita.

### 5.2 Converter

```bash
python scripts/prepare_spring.py \
  --spring-root /data/spring/train \
  --out-root /data/spring_prep/eval \
  --camera left \
  --passo 10 \
  --max-depth 100
```

O script imprime, por sequência, o `fx` usado e a origem dele, e no fim um relatório de
sanidade.

### 5.3 Conferir a conversão — **não pule esta etapa**

É aqui que um erro de fator 2 seria pego. Confira três coisas no relatório final:

**Primeiro, o fator de escala da disparidade.** O log informa qual razão foi detectada
entre a resolução do ground truth e a do RGB. Se aparecer mais de um valor na lista, algo
está inconsistente entre sequências e vale investigar antes de prosseguir.

**Segundo, a faixa de profundidade.** O relatório mostra os percentis 5, 50 e 95. O Spring é
uma cena de filme animado ao ar livre e em interiores, então espere mediana na casa de
poucos metros a algumas dezenas de metros. Se a mediana vier em centímetros, a disparidade
está grande demais por um fator 2; se vier em centenas de metros, pequena demais. Nesse
caso rode de novo com `--disp-escala 1` e depois `--disp-escala 2`, e compare qual produz
valores fisicamente plausíveis.

**Terceiro, a cobertura da máscara.** Espere algo alto, mas não 100%, porque o céu é
removido.

Além do relatório, vale uma conferência visual:

```bash
python scripts/make_riemannian_figure.py \
  --checkpoint /models/checkpoints/depth_pro.pt \
  --data-root /data/spring_prep/eval \
  --out-dir /workspace/figuras_spring_sanidade \
  --variant heads --n-imagens 3
```

Sem `--weights`, isso usa o modelo base. O que interessa não é a qualidade do modelo, e sim
se o **mapa de profundidade do ground truth** faz sentido: objetos próximos claros, fundo
escuro, sem faixas estranhas e sem uma parede uniforme no lugar do céu.

### 5.4 A pergunta central: existe headroom em borda?

```bash
python scripts/evaluate_paired.py \
  --checkpoint /models/checkpoints/depth_pro.pt \
  --seeds-dir  /workspace/runs/champion_heads_final \
  --data-root  /data/spring_prep/eval \
  --out-dir    /workspace/runs/champion_heads_final/aval_spring \
  --variant heads_final --nivel cena
```

E a tabela consolidada, incluindo as duas campeãs se quiser compará-las:

```bash
python scripts/make_results_table.py \
  --entrada heads_final=/workspace/runs/champion_heads_final/aval_spring \
            heads=/workspace/runs/champion_heads/aval_spring \
  --out-dir /workspace/relatorio_spring \
  --titulo "Spring — fora do domínio de treino do DepthPro"
```

**Como ler.** Assim como no DIODE, o número que decide o rumo **não é** se as nossas
campeãs ganham: elas foram treinadas no Hypersim e devem transferir mal. O que interessa é
a linha do **modelo pré-treinado**. Especificamente o `bF-max` dele: se for baixo, confirma
que existe muito espaço de melhoria em borda, e aí o passo seguinte é **retreinar no
Spring**, que é quando a hipótese da curvatura finalmente será testada em condições justas.

### 5.5 Se o headroom se confirmar: treinar no Spring

Aí sim é preciso dividir as sequências, de forma disjunta:

```bash
# particoes disjuntas das 37 sequencias com GT publico
python scripts/prepare_spring.py --spring-root /data/spring/train \
  --out-root /data/spring_prep/train --passo 5 \
  --sequencias 0001 0002 0003 ... # primeiras ~28

python scripts/prepare_spring.py --spring-root /data/spring/train \
  --out-root /data/spring_prep/val --passo 10 \
  --sequencias ...                # ~9 sequencias

python scripts/prepare_spring.py --spring-root /data/spring/train \
  --out-root /data/spring_prep/test --passo 10 \
  --sequencias ...                # ~10 sequencias
```

E então a ablação, com um cuidado de calibração descrito abaixo.

---

## 6. Recalibração necessária antes de treinar

Os tetos da perda foram calibrados no Hypersim, que é interior com profundidade de 1 a 10
metros. O Spring tem faixa muito maior e cenas externas. Dois parâmetros precisam ser
revistos:

- `gauss_clamp` (padrão 50) e `metric_clamp` (padrão 100) — tetos por pixel dos termos de
  segunda ordem e do tensor métrico.

O procedimento é barato: rode um treino curto e olhe as colunas `raw_*` no log, que trazem
as magnitudes **brutas** de cada termo. Se um termo estiver colado no teto, o teto está
cortando sinal; se estiver ordens de magnitude abaixo, está frouxo.

```bash
python scripts/train_single.py \
  --train-root /data/spring_prep/train --val-root /data/spring_prep/val \
  --checkpoint /models/checkpoints/depth_pro.pt \
  --out-dir /workspace/runs/spring_calib \
  --variant heads_final --berhu 0.7 --gauss 0.45 --normal 0.9 \
  --epochs 2 --seeds 1 --lr 1e-5
```

Olhe `train_raw=` no log de época. Reporte os valores antes de disparar a ablação completa.

---

## 7. O que enviar de volta

| Item | Onde |
|---|---|
| Log completo da conversão, com o relatório de sanidade | terminal |
| Figuras de conferência do ground truth | `figuras_spring_sanidade/` |
| Resumo estatístico | `aval_spring/resumo.txt` |
| Tabela consolidada | `relatorio_spring/tabela_geral.*` |
| Magnitudes brutas dos termos, se for treinar | log do `train_single` |

---

## 8. Resumo do risco

O maior risco desta etapa não é técnico, é silencioso: um erro de fator na conversão de
disparidade **não apareceria nas métricas**, porque o alinhamento afim do nosso pipeline
absorve escala uniforme. Ele só apareceria como valores absurdos em metros, ou como
curvatura sistematicamente errada — e a curvatura é justamente a nossa contribuição
central.

Por isso a seção 5.3 não é opcional. Dois minutos de conferência ali evitam refazer
semanas de experimento.
