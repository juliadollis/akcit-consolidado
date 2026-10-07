# Runbook — rodar tudo de uma vez

Documento único de execução. Roda na ordem, do começo ao fim, e produz **todos** os
artefatos que faltam para o report: a tabela geral com zero-shot, os sinais geométricos do
zero-shot, e os painéis comparativos.

**Nada aqui exige treinar de novo.** Tudo é inferência sobre os `best.pt` que já existem
no servidor. O tempo total é de dezenas de minutos, não dias.

---

## Por que estamos refazendo

Três problemas foram encontrados depois da última rodada, todos de medição ou de
visualização, nenhum de treino:

1. **A métrica de borda usava limiar relativo ao gradiente MÁXIMO da imagem**, o que a
   torna refém de um único pixel. Em teste controlado, uma predição idêntica ao ground
   truth com um único pixel outlier caiu de F=1.000 para F=0.596, com precisão 0.98 e
   recall 0.43. Essa é exatamente a assinatura que apareceu no `heads_final`, então a
   conclusão de "piora a borda" pode ser artefato. Corrigido para limiar por percentil,
   mais uma varredura que reporta **bF-max**, o melhor F de cada modelo, que é a
   comparação justa.

2. **Os mapas de curvatura estavam saturados** (52% dos pixels), por suavização
   insuficiente e por um teto que serve à loss mas destrói a figura. Corrigido.

3. **Faltavam o zero-shot nas figuras e uma tabela geral.** Sem ver os sinais do
   zero-shot, não dá para julgar se os do modelo treinado estão melhores. Corrigido com o
   modo comparativo e o script de tabela.

---

## Pré-requisitos

Já devem existir no servidor, dos passos anteriores:

| Item | Caminho esperado |
|---|---|
| Pesos base do DepthPro | `/models/checkpoints/depth_pro.pt` |
| Conjunto de teste (29 cenas) | `/data/hypersim/test` |
| Campeã escopo amplo, 5 sementes | `/workspace/runs/champion_heads/seed_*/best.pt` |
| Campeã escopo fechado, 5 sementes | `/workspace/runs/champion_heads_final/seed_*/best.pt` |

Ajuste os caminhos abaixo se os seus forem outros.

---

## Passo A — Avaliação pareada com a métrica corrigida

Recalcula tudo com o limiar robusto e com o bF-max. É o número que vai para a tabela.

```bash
# escopo amplo
python scripts/evaluate_paired.py \
  --checkpoint /models/checkpoints/depth_pro.pt \
  --seeds-dir  /workspace/runs/champion_heads \
  --data-root  /data/hypersim/test \
  --out-dir    /workspace/runs/champion_heads/aval_v2 \
  --variant heads --nivel cena

# escopo fechado
python scripts/evaluate_paired.py \
  --checkpoint /models/checkpoints/depth_pro.pt \
  --seeds-dir  /workspace/runs/champion_heads_final \
  --data-root  /data/hypersim/test \
  --out-dir    /workspace/runs/champion_heads_final/aval_v2 \
  --variant heads_final --nivel cena
```

Saída por variante: `resumo.txt`, `comparacao_pareada.json`, `metricas_por_imagem.csv`.
As métricas novas `boundary_fmax` e `boundary_f_auc` aparecem junto das antigas, então dá
para ver quanto da conclusão dependia do limiar fixo.

---

## Passo B — Tabela geral com o zero-shot

Consolida as duas variantes e o zero-shot numa tabela só.

```bash
python scripts/make_results_table.py \
  --entrada heads=/workspace/runs/champion_heads/aval_v2 \
            heads_final=/workspace/runs/champion_heads_final/aval_v2 \
  --out-dir /workspace/relatorio \
  --titulo "Resultados no teste held-out (29 cenas)"
```

Gera `tabela_geral.md`, `.csv` e `.txt`. A tabela tem duas partes: valores absolutos com o
zero-shot como coluna, e o ganho de cada campeã com intervalo, os dois p-valores e o
veredito.

---

## Passo C — Sinais geométricos do ZERO-SHOT

Sem esta referência não dá para julgar os sinais do modelo treinado. Basta **omitir**
`--weights`, que o script usa o modelo base.

```bash
python scripts/visual_signals.py \
  --checkpoint /models/checkpoints/depth_pro.pt \
  --data-root /data/hypersim/test \
  --out-dir /workspace/figuras_sinais/zero_shot \
  --variant heads --n-imagens 6
```

Produz profundidade, gradiente, tensor métrico, elemento de área, normais, curvatura
Gaussiana, curvatura média, índice de forma, curvedness e oclusão, mais nove
sobreposições, tudo na dimensão da imagem original.

E os mesmos sinais para as campeãs, para comparar:

```bash
python scripts/visual_signals.py \
  --checkpoint /models/checkpoints/depth_pro.pt \
  --weights /workspace/runs/champion_heads_final/seed_0/best.pt \
  --data-root /data/hypersim/test \
  --out-dir /workspace/figuras_sinais/heads_final \
  --variant heads_final --n-imagens 6

python scripts/visual_signals.py \
  --checkpoint /models/checkpoints/depth_pro.pt \
  --weights /workspace/runs/champion_heads/seed_0/best.pt \
  --data-root /data/hypersim/test \
  --out-dir /workspace/figuras_sinais/heads \
  --variant heads --n-imagens 6
```

> Use os **mesmos `--n-imagens`** nas três chamadas: o script escolhe índices espaçados de
> forma determinística, então as cenas coincidem e a comparação fica alinhada.

---

## Passo D — Painéis riemannianos comparativos

O painel que responde diretamente "o treinado melhorou a geometria?". A flag `--comparar`
coloca zero-shot em cima e treinado embaixo, com **escala de cor compartilhada** entre as
linhas. Sem escala compartilhada a comparação seria inválida, porque cada painel se
auto-normalizaria.

```bash
# escopo fechado
python scripts/make_riemannian_figure.py \
  --checkpoint /models/checkpoints/depth_pro.pt \
  --weights /workspace/runs/champion_heads_final/seed_0/best.pt \
  --data-root /data/hypersim/test \
  --out-dir /workspace/figuras_riemann/heads_final \
  --variant heads_final --n-imagens 6 --comparar \
  --titulo "Escopo fechado (heads_final)"

# escopo amplo
python scripts/make_riemannian_figure.py \
  --checkpoint /models/checkpoints/depth_pro.pt \
  --weights /workspace/runs/champion_heads/seed_0/best.pt \
  --data-root /data/hypersim/test \
  --out-dir /workspace/figuras_riemann/heads \
  --variant heads --n-imagens 6 --comparar \
  --titulo "Escopo amplo (heads)"
```

E, para a figura de destaque do report, a versão de painel único (2×3, sem `--comparar`)
da melhor campeã:

```bash
python scripts/make_riemannian_figure.py \
  --checkpoint /models/checkpoints/depth_pro.pt \
  --weights /workspace/runs/champion_heads_final/seed_0/best.pt \
  --data-root /data/hypersim/test \
  --out-dir /workspace/figuras_riemann/destaque \
  --variant heads_final --n-imagens 3 \
  --titulo "Hypersim — configuração campeã"
```

**Ajuste fino da figura de área:** se o fundo ficar avermelhado em vez de preto, suba
`--piso` (tente 80). Se as bordas sumirem, baixe. O `--gama` (padrão 1.8) controla o
contraste entre o piso de ruído e as bordas. Gere uma cena primeiro e confira antes de
rodar as seis.

---

## Passo E — Painéis de profundidade e zoom de borda

```bash
python scripts/make_figures.py \
  --checkpoint /models/checkpoints/depth_pro.pt \
  --weights /workspace/runs/champion_heads_final/seed_0/best.pt \
  --data-root /data/hypersim/test \
  --out-dir /workspace/figuras/heads_final \
  --variant heads_final --n-imagens 6 --zoom 180 220 140 140
```

Este já era comparativo (zero-shot e treinado lado a lado) e agora usa a curvatura com os
parâmetros corrigidos.

---

## O que enviar de volta

| Arquivo | Onde |
|---|---|
| Tabela geral | `/workspace/relatorio/tabela_geral.{md,csv,txt}` |
| Resumos estatísticos | `/workspace/runs/champion_*/aval_v2/resumo.txt` |
| JSONs da comparação | `/workspace/runs/champion_*/aval_v2/comparacao_pareada.json` |
| Painéis comparativos | `/workspace/figuras_riemann/*/` |
| Sinais geométricos | `/workspace/figuras_sinais/*/contato/` e `/paineis/` |

Se o pacote ficar grande, os `.npy` podem ficar no servidor; os PNG e o `contato/` já
bastam para a decisão.

---

## Verificação rápida antes de enviar

1. No `resumo.txt`, confira que `boundary_fmax` aparece. Se não aparecer, o código antigo
   ainda está ativo.
2. Na tabela geral, a coluna `Zero-shot` precisa estar preenchida em todas as linhas.
3. Nos painéis comparativos, o fundo do mapa de área deve estar **preto**, não vermelho.
4. Nos sinais do zero-shot, a pasta `contato/` deve ter as mesmas cenas das campeãs.
