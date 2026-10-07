# Plano experimental — o que ainda falta rodar

Documento operacional. Diz **quais configurações rodar, em que ordem e com quais
comandos**, e por quê. Os detalhes de correções de código estão no
`CORRECOES_REVISAO.md`; aqui é só o plano de execução.

---

## 1. Onde estamos

| Item | Situação |
|---|---|
| Ablação das 39 configurações, escopo `heads`, 1 semente | concluída |
| Vencedora `B3_berhu+grad+normal+gauss`, 3 sementes | concluída (em validação) |
| Escopo `heads_final`, apenas `B0` e `B3` | concluído (1 semente) |
| Variante `heads_lora` | inválida, bug de gradiente; adiada |
| Conjunto de **teste** disjunto | **não existe ainda** |
| Avaliação emparelhada por cena | **não executada** |
| Mapas de profundidade e curvatura | **não gerados** |

Resultados atuais, todos em **validação** (4 cenas), que também foi usada para
selecionar as vencedoras. Há viés otimista e nenhum número é final.

---

## 2. Por que NÃO repetir as 39 configurações

As 39 esgotam as combinações dos seis termos de perda. Rodá-las de novo amplia a busca
no eixo que já foi varrido. Depois do que sabemos sobre o DepthPro original, o eixo
inexplorado é outro:

- O DepthPro já treina com perdas de **primeira e segunda ordem** na etapa 2 (MAGE, MALE,
  MSGE), sendo o MALE o erro absoluto de Laplaciano, que é segunda ordem. Ou seja, o
  modelo base já foi otimizado sob supervisão de segunda ordem com foco declarado em
  afiar bordas. Isso explica o headroom quase nulo em borda e obriga o trabalho a se
  posicionar como **curvatura intrínseca (Gaussiana) versus Laplaciano extrínseco**, e
  não como "introduzimos segunda ordem".
- A cabeça de FOV foi treinada **separadamente**, sobre features congeladas, e os autores
  afirmam que separar é melhor que treinar junto. Nosso escopo `heads` liberava o FOV
  junto, contrariando o desenho original. Daí o ganho de AbsRel ao fechar o escopo.

Portanto o que falta variar é o **escopo**, não a perda.

---

## 3. A grade que falta: escopo × perda

Sob `heads` já temos tudo. Sob `heads_final` temos apenas `B0` e `B3`. A pergunta em
aberto, e que um revisor fará, é: **o ganho de AbsRel do escopo fechado vem da curvatura
ou vem do escopo sozinho?** Só dá para responder rodando os termos isolados nesse escopo.

**10 configurações, escopo `heads_final`:**

| Bloco | Configurações |
|---|---|
| B0 (1) | `B0_berhu` |
| B1 (5) | `B1_berhu+grad`, `B1_berhu+normal`, `B1_berhu+gauss`, `B1_berhu+geod`, `B1_berhu+metric` |
| B7 (4) | `B7_gaussheavy`, `B7_gauss_dom`, `B7_normal_dom`, `B7_champion_prev` |

O `B3_berhu+grad+normal+gauss` já foi rodado nesse escopo e serve de referência.

---

## 4. Ordem de execução

### Passo 1 — Conjunto de teste (fazer primeiro, roda em paralelo)

O gargalo estatístico não são as 300 imagens da validação, são as **4 cenas**. Imagens
da mesma cena são fortemente correlacionadas, então o tamanho efetivo de amostra é o
número de cenas. O teste precisa de **15 a 20 cenas**, disjuntas de treino e validação.

```bash
python scripts/download_hypersim_stream.py \
  --out-root /data/hypersim/test --n-images 1500 \
  --start-index 300 --limit-scenes 20
```

Verificar: `ls /data/hypersim/test/rgb | wc -l` e conferir que os nomes de cena não
aparecem em treino nem em validação.

### Passo 2 — Ablação no escopo fechado (10 configurações)

```bash
python scripts/run_ablation.py \
  --train-root /data/hypersim/ablation/train \
  --val-root   /data/hypersim/ablation/val \
  --checkpoint /models/checkpoints/depth_pro.pt \
  --out-dir    /workspace/runs/ablation_heads_final \
  --variant heads_final \
  --filter B0 B1 B7 \
  --epochs 30 --lr 1e-5 --batch-size 2 --grad-accum 4 --size 512 \
  --monitor boundary_fscore
```

O zero-shot é medido automaticamente como primeira linha do CSV, no mesmo protocolo.
Para o monitor, use `--monitor abs_rel` se o interesse nesse escopo for precisão
métrica; vale rodar as duas seleções e reportar as duas, já que o escopo controla
exatamente esse compromisso.

### Passo 3 — Sementes das duas vencedoras

Uma vencedora por escopo. Para cada uma:

```bash
python scripts/train_single.py \
  --train-root /data/hypersim/ablation/train \
  --val-root   /data/hypersim/ablation/val \
  --test-root  /data/hypersim/test \
  --checkpoint /models/checkpoints/depth_pro.pt \
  --out-dir    /workspace/runs/champion_<escopo> \
  --variant <heads|heads_final> \
  --berhu 0.7 --grad 0.3 --normal 0.9 --gauss 0.45 \
  --epochs 60 --seeds 5 --lr 1e-5 --batch-size 2 --grad-accum 4 \
  --monitor boundary_fscore --export-curvature
```

**Use 5 sementes, não 3.** Com 3 o intervalo por t tem multiplicador 4.30 e não separa
efeitos pequenos; com 5 cai para 2.78. Foi exatamente isso que deixou o ganho de borda
inconclusivo.

### Passo 4 — Avaliação final, emparelhada, no teste

```bash
python scripts/evaluate_paired.py \
  --checkpoint /models/checkpoints/depth_pro.pt \
  --seeds-dir  /workspace/runs/champion_<escopo> \
  --data-root  /data/hypersim/test \
  --out-dir    /workspace/runs/champion_<escopo>/avaliacao_final \
  --variant <heads|heads_final> --nivel cena
```

Gera `metricas_por_imagem.csv`, `comparacao_pareada.json` e `resumo.txt`. Reporta os
**dois testes** (t pareado e Wilcoxon) com os dois intervalos (t e bootstrap), e só
declara significância quando ambos concordam.

### Passo 5 — Artefatos visuais

```bash
# painel comparativo zero-shot versus treinado
python scripts/make_figures.py \
  --checkpoint /models/checkpoints/depth_pro.pt \
  --weights /workspace/runs/champion_heads/seed_0/best.pt \
  --data-root /data/hypersim/test --out-dir /workspace/figuras \
  --variant heads --n-imagens 6 --zoom 180 220 140 140

# conjunto completo de sinais geométricos e sobreposições
python scripts/visual_signals.py \
  --checkpoint /models/checkpoints/depth_pro.pt \
  --weights /workspace/runs/champion_heads/seed_0/best.pt \
  --data-root /data/hypersim/test --out-dir /workspace/figuras_sinais \
  --variant heads --n-imagens 6
```

---

## 5. Custo estimado

Com 3000 imagens de treino, lote 2 e acumulação 4, cada época tem 1500 micro-lotes e 375
passos de otimizador.

| Etapa | Treinos | Épocas | Passos de otimizador |
|---|---|---|---|
| Ablação `heads_final` | 10 | 30 | ~112 mil |
| Sementes, 2 vencedoras | 10 | 60 | ~225 mil |
| Avaliação e figuras | — | — | inferência apenas |

A ablação no escopo fechado é mais barata que a anterior: `heads_final` treina 491 mil
parâmetros contra 342 milhões, então o passo de otimizador é muito mais leve, ainda que o
forward continue custando o mesmo (entrada fixa de 1536).

---

## 6. Como reportar

O resultado principal é a **comparação emparelhada por cena no conjunto de teste**, com
ganho médio, intervalo da diferença e os dois p-valores. O intervalo entre sementes mede
reprodutibilidade do treino, não o efeito contra o baseline, e deve aparecer como
informação secundária.

A narrativa que os dados sustentam hoje tem dois pontos de operação da mesma família de
perdas: escopo amplo favorece borda, escopo fechado favorece precisão métrica. Isso deve
ser apresentado como um compromisso controlável, e não como um único modelo que vence em
tudo, porque são modelos diferentes selecionados em métricas diferentes.
