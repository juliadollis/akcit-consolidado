# Fine-tuning do DepthPro com Perda Riemanniana — Guia de Execução

Este pacote faz **fine-tuning do DepthPro** usando uma **Perda Riemanniana** (com
curvatura Gaussiana e normais de superfície) para produzir mapas de profundidade com
**bordas mais nítidas e geometricamente consistentes**. Também **exporta mapas de
curvatura** reutilizáveis.

> **Para quem vai executar:** siga os passos na ordem. Cada comando está pronto para
> copiar e colar. Onde houver `/caminho/...`, substitua pelo caminho real na sua máquina.
> Você **não precisa entender a matemática** para rodar — o guia é operacional.

---

## 0. O que este pacote produz

- Um **DepthPro afinado** (checkpoint `best.pt`) que gera profundidade com bordas melhores.
- Uma **tabela de ablação** (CSV) comparando ~39 configurações de perda.
- Uma **fronteira de Pareto** (Optuna) com os melhores pesos.
- **Mapas de curvatura Gaussiana** (`.npy` + `.png`) para uso futuro.

---

## 1. Requisitos de hardware

| Item | Recomendado |
|---|---|
| GPU | NVIDIA H100 80GB (defaults ajustados) — roda também em A100 / RTX 4090 |
| RAM | 64 GB |
| Disco | ~250 GB (dataset Hypersim + modelos + saídas) |
| CUDA | 12.x |

> **Perfil H100 (80GB):** os defaults já usam batch 8, 512px, sem `--grad-accum` nem
> `--grad-checkpointing` (a VRAM é folgada). A ablação COMPLETA (39 configs) e a busca
> Optuna de 200 trials são viáveis. Para GPUs menores, reduza `--batch-size`/`--size` e
> ligue `--grad-checkpointing`. Detalhes no `README_CONTAINER.md`.

---

## 2. Instalação do ambiente

```bash
# 2.1 Criar ambiente
conda create -n riemann python=3.12 -y
conda activate riemann

# 2.2 PyTorch com CUDA (ajuste o índice à sua CUDA; exemplo p/ CUDA 12.4)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# 2.3 Instalar este pacote (traz as dependências de requirements.txt)
pip install -e .

# 2.4 Instalar o DepthPro (Apple ml-depth-pro)
git clone https://github.com/apple/ml-depth-pro.git
pip install -e ml-depth-pro
```

---

## 3. Baixar os pesos do DepthPro

```bash
mkdir -p models/checkpoints
cd models/checkpoints
# Pesos do DepthPro (mesmos usados no projeto Genfocus)
wget https://huggingface.co/nycu-cplab/Genfocus-Model/resolve/main/checkpoints/depth_pro.pt
cd ../..
```

Ao final você deve ter: `models/checkpoints/depth_pro.pt`

---

## 4. TESTE DE SANIDADE (faça isto ANTES de tudo)

Valida que a geometria (curvatura Gaussiana) está numericamente correta no seu ambiente.
**Leva segundos e roda em CPU.**

```bash
python scripts/test_geometry.py
```

Saída esperada:
```
Esfera R=2.0: K=0.2500  (teórico 0.2500)
Plano: |K|médio=4e-10  (esperado ~0)
✅ Geometria OK — a curvatura Gaussiana está numericamente correta.
```

Se este teste falhar, **pare** e reporte — não adianta treinar com a geometria errada.

---

## 5. Preparar o dataset (Hypersim)

O treino usa **Hypersim** (profundidade sintética de alta qualidade, ideal para curvatura).

### 5.1 Obter o Hypersim
Baixe o release oficial: https://github.com/apple/ml-hypersim
(São muitos GB. Baixe pelo menos um subconjunto de cenas.)

### 5.2 Converter para o layout do treino
```bash
# Gera pares rgb/ + depth/ a partir dos .hdf5 do Hypersim
python scripts/prepare_hypersim.py \
    --hypersim-root /caminho/hypersim_raw \
    --out-root /caminho/hypersim/train

# Repita para um conjunto de validação (cenas diferentes)
python scripts/prepare_hypersim.py \
    --hypersim-root /caminho/hypersim_raw_val \
    --out-root /caminho/hypersim/val
```

O layout final deve ser:
```
/caminho/hypersim/train/rgb/*.png
/caminho/hypersim/train/depth/*.npy
/caminho/hypersim/val/rgb/*.png
/caminho/hypersim/val/depth/*.npy
```

> **Alternativa:** qualquer dataset com o mesmo layout `rgb/` + `depth/` funciona
> (ETH3D, DIML, ou dados próprios de alta resolução). Só aponte `--train-root`/`--val-root`.

### 5.3 Teste rápido do pipeline (10 min, poucas amostras)
Antes de rodar tudo, valide a pipeline completa com poucas amostras:
```bash
python scripts/run_ablation.py \
    --train-root /caminho/hypersim/train \
    --val-root   /caminho/hypersim/val \
    --checkpoint models/checkpoints/depth_pro.pt \
    --out-dir    ./runs/smoke \
    --variant heads --epochs 2 --batch-size 2 \
    --filter B0 B1 --max-train 20
```
Se rodar sem erro e gerar `./runs/smoke/ablation_results.csv`, está tudo certo.

---

> **Batches por fase:** use `scripts/make_subset.py` para recortar subconjuntos
> (smoke/ablação/full). O `RUNBOOK.md` traz o passo a passo de execução; o
> `ROTEIRO_IMPLEMENTACAO.md` detalha os tamanhos recomendados de dataset por fase.

## 6. Rodar a ABLAÇÃO (experimento principal)

Compara ~39 configurações de perda para descobrir **qual geometria importa**.

```bash
python scripts/run_ablation.py \
    --train-root /caminho/hypersim/train \
    --val-root   /caminho/hypersim/val \
    --checkpoint models/checkpoints/depth_pro.pt \
    --out-dir    ./runs/ablation_heads \
    --variant heads \
    --epochs 40 --batch-size 4 --size 512
```

**Repita com a outra variante de modelo** (compara congelar só as cabeças vs. cabeças+LoRA):
```bash
python scripts/run_ablation.py \
    ... (mesmos args) ... \
    --out-dir ./runs/ablation_lora \
    --variant heads_lora
```

**Resultado:** `./runs/ablation_heads/ablation_results.csv` — abra e ordene por `abs_rel`
(precisão) e por `boundary_fscore` (qualidade de borda). O script também imprime os TOP-10
de cada no final.

### Opções úteis
| Flag | Efeito |
|---|---|
| `--filter B1 B7` | Roda só os blocos indicados (ex.: marginais e foco-em-borda) |
| `--epochs N` | Épocas por config (40 é um bom equilíbrio; aumente para o resultado final) |
| `--batch-size N` | Ajuste conforme a VRAM (A100 40GB: 4–6 em 512px) |
| `--size N` | Resolução (512 recomendado; 768 se tiver 80GB) |

---

## 7. Rodar a BUSCA DE HIPERPARÂMETROS (Optuna)

Encontra os **melhores pesos** da perda, otimizando **dois objetivos ao mesmo tempo**:
precisão (AbsRel ↓) e qualidade de borda (boundary F-score ↑).

```bash
python scripts/run_optuna.py \
    --train-root /caminho/hypersim/train \
    --val-root   /caminho/hypersim/val \
    --checkpoint models/checkpoints/depth_pro.pt \
    --out-dir    ./runs/optuna \
    --variant heads \
    --trials 200 --epochs-per-trial 20
```

**Resultado:**
- `./runs/optuna/pareto_front.json` — as melhores combinações (trade-off precisão × borda)
- `./runs/optuna/term_presence.json` — **em quantos % das melhores soluções cada termo
  aparece** (é assim que se descobre qual geometria domina — no trabalho anterior, a
  curvatura Gaussiana apareceu em 100%).

---

## 8. Treinar a configuração CAMPEÃ (resultado final)

Pegue os melhores pesos (da ablação ou do Optuna) e treine por mais épocas, com
múltiplas seeds, exportando os mapas de curvatura.

```bash
python scripts/train_single.py \
    --train-root /caminho/hypersim/train \
    --val-root   /caminho/hypersim/val \
    --checkpoint models/checkpoints/depth_pro.pt \
    --out-dir    ./runs/champion \
    --variant heads \
    --berhu 0.7 --normal 0.9 --gauss 0.45 \
    --epochs 100 --seeds 3 \
    --export-curvature
```

Troque `--berhu/--normal/--gauss/--grad/--geod/--metric` pelos pesos vencedores.

**Resultado:**
- `./runs/champion/seed_0/best.pt` (e seed_1, seed_2) — o modelo afinado
- `./runs/champion/curvature_maps/*.npy` e `*.png` — os mapas de curvatura

---

## 9. Ordem recomendada de execução (resumo)

1. `scripts/test_geometry.py` — sanidade (segundos)
2. `scripts/prepare_hypersim.py` — preparar dados (uma vez)
3. `scripts/run_ablation.py --filter B0 B1 ... --max-train 20 --epochs 2` — smoke test (10 min)
4. `scripts/run_ablation.py` completo, variante `heads` — experimento principal (horas)
5. `scripts/run_ablation.py` completo, variante `heads_lora` — comparação (horas)
6. `scripts/run_optuna.py` — refinar pesos (horas)
7. `scripts/train_single.py --export-curvature` — campeã + curvatura (horas)

> Avaliação fora do domínio de treino (DIODE, Spring): ver `README_DATASETS.md`
> e os guias `GUIA_DIODE.md` / `GUIA_SPRING.md`.

---

## 10. O que cada arquivo faz

```
riemann/
├── losses.py            Perda Riemanniana (berHu, grad, normal, gauss, geod, metric)
│                        + cálculo da curvatura Gaussiana (validado em esfera/plano)
├── model.py             Wrapper do DepthPro com 3 variantes (heads / heads_final / heads_lora)
├── dataset.py           Dataloader do layout rgb/ + depth/ (Hypersim e genéricos)
├── metrics.py           Métricas MDE padrão + boundary F-score (qualidade de borda)
├── trainer.py           Loop de treino/validação + exportação de curvatura
├── ablation_configs.py  Gera as ~39 configurações da ablação
├── geometry_maps.py     Sinais geométricos (det(g), normais, curvaturas, oclusão)
└── repro.py             Estatística: IC t de Student, teste pareado (t + Wilcoxon)

scripts/
├── test_geometry.py       Sanidade da curvatura (rode primeiro!)
├── prepare_hypersim.py    Converte o Hypersim para o layout de treino
├── make_subset.py         Recorta batches do Hypersim sem duplicar dados
├── download_hypersim_stream.py  Baixa o Hypersim cena a cena (sem disco p/ tudo)
├── download_weights.py    Baixa os pesos do DepthPro
├── run_ablation.py        Roda a ablação expandida
├── run_optuna.py          Busca bayesiana bi-objetivo dos pesos
├── train_single.py        Treina a config campeã + exporta curvatura
├── evaluate_paired.py     Avaliação final pareada (modelo vs zero-shot por cena)
├── make_figures.py        Painéis comparativos zero-shot vs treinado
└── visual_signals.py      Gera todos os sinais geométricos (PNG + npy)
```

---

## 11. Problemas comuns

**`ImportError: depth_pro`** → você pulou o passo 2.4. Rode `pip install -e ml-depth-pro`.

**`CUDA out of memory`** → reduza `--batch-size` (tente 2) ou `--size` (tente 384).

**`Nenhum par (rgb, depth) encontrado`** → o layout está errado. Confira que existem
`rgb/` e `depth/` com nomes de arquivo correspondentes (mesmo *stem*).

**Ablação muito lenta** → use `--filter` para rodar blocos por vez, ou reduza `--epochs`.
Comece por `--filter B1 B7` (os mais informativos para a hipótese central).

**O teste de geometria falha** → não treine. A curvatura estaria errada. Reporte.

---

## 12. O que reportar de volta

Ao terminar, envie:
1. `runs/ablation_heads/ablation_results.csv` e `runs/ablation_lora/ablation_results.csv`
2. `runs/optuna/pareto_front.json` e `runs/optuna/term_presence.json`
3. `runs/champion/seed_*/summary.json` (métricas finais das seeds)
4. Alguns exemplos de `runs/champion/curvature_maps/*.png`

Isso é suficiente para analisar qual configuração venceu e escrever os resultados.
