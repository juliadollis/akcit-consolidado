# Fine-tuning do DepthPro com Perda Riemanniana — Guia de Execução (Container · H100)

**Para quem vai executar:** este guia assume que você **não conhece o projeto**. Siga os
passos na ordem. Cada etapa tem um comando pronto e uma **verificação de sucesso (✓)** —
só avance quando a verificação passar. Onde houver `/seu/caminho/...`, troque pelo caminho
real na máquina.

---

## 0. O que este projeto faz (contexto mínimo)

Faz **fine-tuning de um modelo de estimação de profundidade** (o DepthPro, da Apple)
usando uma função de perda geométrica ("Perda Riemanniana") para gerar **mapas de
profundidade com bordas mais nítidas**. Roda três tipos de experimento e produz:

- uma **tabela comparando configurações de perda** (qual combinação é melhor),
- os **melhores pesos** encontrados por busca automática (Optuna),
- um **modelo treinado** (arquivo `best.pt`),
- **mapas de curvatura** (subproduto geométrico reutilizável).

Você **não precisa entender a matemática** para executar. É tudo por linha de comando
dentro de um container Docker.

**Hardware-alvo: NVIDIA H100 80GB.** Com 80GB de VRAM, roda a ablação COMPLETA (39
configurações), batch grande e resolução alta sem os artifícios de economia de memória
que uma GPU menor exigiria.

---

## 1. Requisitos

| Item | Necessário |
|---|---|
| GPU | NVIDIA H100 80GB (também roda em A100/menores com ajustes — ver seção 11) |
| Disco livre | **~250 GB** (Hypersim ~150GB + modelos + saídas) |
| RAM | 64 GB recomendável |
| Software | Docker + driver NVIDIA + NVIDIA Container Toolkit |

---

## 2. Passo 1 — Instalar o NVIDIA Container Toolkit (uma vez)

Sem isto, o Docker não enxerga a GPU.
```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
  sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```
**✓ Verificação:**
```bash
docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi
```
Deve imprimir uma tabela listando a **H100**. Se der erro de driver, pare e resolva antes.

---

## 3. Passo 2 — Construir a imagem

Na pasta que contém o `Dockerfile` (raiz deste projeto):
```bash
docker build -t riemann-depthpro:latest .
```
O build (~10–20 min) instala PyTorch+CUDA, clona/instala o DepthPro, instala dependências
e **roda um teste de geometria** — se a matemática da curvatura estiver quebrada, o build
falha de propósito antes de gastar GPU.

**✓ Verificação:** a última linha deve conter
`Geometria OK — a curvatura Gaussiana está numericamente correta.` e:
```bash
docker images | grep riemann-depthpro
```

---

## 4. Passo 3 — Criar as pastas do host

| Pasta no host (você cria) | No container | Para quê |
|---|---|---|
| `./runs` | `/workspace/runs` | Resultados (checkpoints, CSVs, curvatura) |
| `/seu/caminho/data` | `/data` | Datasets |
| `/seu/caminho/models` | `/models` | Pesos do modelo |

```bash
mkdir -p runs
mkdir -p /seu/caminho/data /seu/caminho/models
```

---

## 5. Passo 4 — Baixar os pesos do DepthPro (uma vez, ~1.5 GB)

```bash
docker run --rm --gpus all -v /seu/caminho/models:/models \
  riemann-depthpro:latest \
  python scripts/download_weights.py --out-dir /models
```
**✓ Verificação:**
```bash
ls -lh /seu/caminho/models/checkpoints/depth_pro.pt   # deve ter ~1.5 GB
```

---

## 6. Passo 5 — Obter e preparar o dataset (Hypersim)

O treino usa o **Hypersim** (sintético de alta qualidade), que é **grande (~150 GB)**.

### 6.1 Baixar (no host)
Siga as instruções em **https://github.com/apple/ml-hypersim** (há script de download no
repositório). Baixe para, ex.: `/seu/caminho/data/hypersim_raw`.
> Pode começar com um subconjunto de cenas. Use **cenas diferentes** para treino e validação.

### 6.2 Converter para o layout do treino
```bash
# TREINO
docker run --rm --gpus all -v /seu/caminho/data:/data \
  riemann-depthpro:latest \
  python scripts/prepare_hypersim.py \
    --hypersim-root /data/hypersim_raw --out-root /data/hypersim/train

# VALIDAÇÃO (cenas diferentes)
docker run --rm --gpus all -v /seu/caminho/data:/data \
  riemann-depthpro:latest \
  python scripts/prepare_hypersim.py \
    --hypersim-root /data/hypersim_raw_val --out-root /data/hypersim/val
```
**✓ Verificação:**
```bash
ls /seu/caminho/data/hypersim/train/rgb   | head
ls /seu/caminho/data/hypersim/train/depth | head
```
Cada arquivo em `rgb/` deve ter um par de **mesmo nome** em `depth/`. Se uma pasta estiver
vazia, o `prepare_hypersim` não achou os dados — confira o `--hypersim-root`.

### 6.0 (Recomendado se falta disco) Download em streaming
Se **não há espaço para os ~150 GB** do Hypersim, use o `download_hypersim_stream.py`: ele
baixa cena a cena, converte e apaga o zip na hora — o pico de disco é uma cena + o batch.
```bash
docker compose run --rm trainer python scripts/download_hypersim_stream.py \
  --out-root /data/hypersim/ablation/train --n-images 3000 --limit-scenes 60
```
Aceita `--n-images N` (por quantidade) ou `--scenes ai_001_001 ...` (lista exata). Detalhes
e a separação treino/val no **`ROTEIRO_IMPLEMENTACAO.md`**, seção 1.1-B.
> ⚠️ Precisa de internet aberta (acessa a Apple). Teste com uma cena antes de baixar em massa.

### 6.3 Criar subconjuntos menores (batches) para cada fase
O Hypersim é grande. Para testar e treinar por fases, use o `make_subset.py` para recortar
subconjuntos reprodutíveis (smoke ~60, ablação ~3000, full ~20000). **Veja o documento
`ROTEIRO_IMPLEMENTACAO.md`** para os tamanhos recomendados e o passo a passo completo.
```bash
docker compose run --rm trainer python scripts/make_subset.py \
  --src /data/hypersim_full/train --dst /data/hypersim/ablation/train --n 3000 --seed-stride
```

> **Alternativa (sem Hypersim):** qualquer dataset no layout `rgb/*.png` + `depth/*.npy|png`
> com nomes correspondentes funciona (ETH3D, DIML, próprio). Aponte `--train-root`/`--val-root`.

---

## 7. Passo 6 — Configurar o docker-compose (recomendado)

Abra `docker-compose.yml` e troque os caminhos do host nos volumes:
```yaml
    volumes:
      - ./runs:/workspace/runs
      - /seu/caminho/data:/data       # <-- troque
      - /seu/caminho/models:/models   # <-- troque
```

---

## 8. Passo 7 — Rodar os experimentos (na ordem)

> **Na H100 os defaults já são de alto desempenho** (batch 8, 512px, sem checkpointing).
> Não precisa de `--grad-accum` nem `--grad-checkpointing` no caso normal.

### 8.1 Sanidade (segundos)
```bash
docker compose run --rm trainer python scripts/test_geometry.py
```
**✓** termina com `Geometria OK`.

### 8.2 Smoke test (poucos minutos) — valida o pipeline com poucos dados
```bash
docker compose run --rm trainer python scripts/run_ablation.py \
  --train-root /data/hypersim/train --val-root /data/hypersim/val \
  --checkpoint /models/checkpoints/depth_pro.pt \
  --out-dir /workspace/runs/smoke \
  --variant heads --epochs 2 --filter B0 B1 --max-train 40
```
**✓** cria `runs/smoke/ablation_results.csv` sem erro. **Só prossiga se este passo funcionar.**

### 8.3 Ablação COMPLETA (39 configs) — o experimento principal
Na H100 a varredura completa é viável:
```bash
docker compose run --rm trainer python scripts/run_ablation.py \
  --train-root /data/hypersim/train --val-root /data/hypersim/val \
  --checkpoint /models/checkpoints/depth_pro.pt \
  --out-dir /workspace/runs/ablation_heads \
  --variant heads --epochs 40 --batch-size 8 --size 512
```
E a **segunda variante de modelo** (a H100 comporta a comparação heads vs heads+LoRA):
```bash
docker compose run --rm trainer python scripts/run_ablation.py \
  --train-root /data/hypersim/train --val-root /data/hypersim/val \
  --checkpoint /models/checkpoints/depth_pro.pt \
  --out-dir /workspace/runs/ablation_lora \
  --variant heads_lora --epochs 40 --batch-size 8 --size 512
```

### 8.4 Busca de hiperparâmetros (Optuna, 200 trials)
```bash
docker compose run --rm trainer python scripts/run_optuna.py \
  --train-root /data/hypersim/train --val-root /data/hypersim/val \
  --checkpoint /models/checkpoints/depth_pro.pt \
  --out-dir /workspace/runs/optuna \
  --variant heads --trials 200 --epochs-per-trial 20 \
  --batch-size 8 --size 512
```

### 8.5 Treinar a campeã + mapas de curvatura
Use os melhores pesos da ablação/Optuna (troque os `--berhu/--normal/--gauss` se preciso):
```bash
docker compose run --rm trainer python scripts/train_single.py \
  --train-root /data/hypersim/train --val-root /data/hypersim/val \
  --checkpoint /models/checkpoints/depth_pro.pt \
  --out-dir /workspace/runs/champion \
  --variant heads --berhu 0.7 --normal 0.9 --gauss 0.45 \
  --epochs 100 --seeds 3 --batch-size 8 --size 512 --export-curvature
```

> **Opcional (alta resolução):** a H100 comporta 768px ligando `--grad-checkpointing`
> (adicione `--size 768 --grad-checkpointing --batch-size 4`). Dá bordas ainda mais finas,
> ao custo de tempo.

---

## 9. Rodar treinos longos em segundo plano

Ablação completa e Optuna levam **horas a dias**. Não deixe preso ao terminal:
```bash
docker compose run -d --rm --name ablation trainer \
  python scripts/run_ablation.py ...(mesmos args)... \
  > runs/ablation.log 2>&1
docker logs -f ablation     # acompanhar
```
Ou use `tmux`/`screen` no host. O CSV da ablação é salvo **incrementalmente**, então o
progresso parcial fica visível mesmo se interromper.

---

## 10. Alternativa sem docker-compose (docker run puro)
```bash
docker run --rm --gpus all --shm-size 32g --ipc host \
  -v $(pwd)/runs:/workspace/runs \
  -v /seu/caminho/data:/data -v /seu/caminho/models:/models \
  riemann-depthpro:latest \
  python scripts/run_ablation.py \
    --train-root /data/hypersim/train --val-root /data/hypersim/val \
    --checkpoint /models/checkpoints/depth_pro.pt \
    --out-dir /workspace/runs/ablation_heads --variant heads \
    --epochs 40 --batch-size 8 --size 512
```

---

## 11. Notas de recurso (H100 80GB)

Na H100 o gargalo de memória some no caso normal. Os defaults já usam batch 8 / 512px sem
acumulação nem checkpointing. As alavancas existem se você quiser ir além:

| Flag | Quando usar na H100 |
|---|---|
| `--batch-size 8` (default) | Confortável em 512px; pode subir para 12–16 se sobrar VRAM |
| `--size 768` + `--grad-checkpointing` | Para bordas ainda mais finas (mais lento) |
| `--variant heads_lora` | A H100 comporta em 512px — inclua na comparação |
| `--grad-accum N` | Raramente necessário; só se subir muito a resolução |

Se (improvável) aparecer `CUDA out of memory` em 768px: ligue `--grad-checkpointing` e/ou
reduza `--batch-size`.

---

## 12. Onde ficam os resultados (host, em `runs/`)

| Arquivo | Conteúdo |
|---|---|
| `runs/ablation_heads/ablation_results.csv` | Comparação das configs (variante heads) |
| `runs/ablation_lora/ablation_results.csv` | Comparação das configs (variante heads_lora) |
| `runs/optuna/pareto_front.json` | Melhores combinações de pesos |
| `runs/optuna/term_presence.json` | Em % das melhores soluções, quais termos aparecem |
| `runs/champion/seed_*/best.pt` | **O modelo treinado** |
| `runs/champion/seed_*/summary.json` | Métricas finais por seed |
| `runs/champion/curvature_maps/*.png` `*.npy` | Mapas de curvatura |

---

## 13. Tempos estimados na H100

| Etapa | Tempo aproximado |
|---|---|
| Build da imagem | 10–20 min |
| Download dos pesos | poucos minutos |
| Download + preparação do Hypersim | horas (depende da rede) |
| Smoke test | poucos minutos |
| Ablação completa (39 configs, 512px) | ~1 dia (bem mais rápido que numa GPU menor) |
| Optuna (200 trials, 512px) | ~1–2 dias |
| Campeã (512px × 3 seeds × 100 épocas) | horas |

---

## 14. Solução de problemas

| Sintoma | Causa / o que fazer |
|---|---|
| `could not select device driver "nvidia"` | NVIDIA Container Toolkit ausente → refaça o Passo 1 |
| `CUDA out of memory` (só em 768px) | Ligue `--grad-checkpointing`; reduza `--batch-size` |
| DataLoader trava / shared memory | Confirme `--shm-size 32g` e `--ipc host` (o compose já inclui) |
| `Nenhum par (rgb, depth) encontrado` | Layout errado → `rgb/` e `depth/` com arquivos de **mesmo nome** |
| Build falha no teste de geometria | Problema numérico → **reporte, não prossiga** |
| `[Modelo] aviso: não foi possível ativar gradient checkpointing` | A versão do DepthPro não expõe a API. Só afeta o modo 768px; ignore em 512px |
| Erro sobre nomes de camada / poucos params treináveis | Ver seção 15 |

---

## 15. Ressalva técnica (para quem for depurar)

O código identifica as "cabeças de decodificação" e as camadas de atenção do DepthPro por
**substrings de nome** (`head`, `decoder`, `attn`, etc.), que podem variar entre versões do
pacote `depth_pro`. **No primeiro run, confira a linha impressa:**
```
[Modelo] variante=heads | params treináveis=XXX,XXX | grad_ckpt=False
```
Se `params treináveis` for **0** ou absurdamente pequeno, o código não encontrou as
cabeças — quem mantém o projeto ajusta as substrings em `riemann/model.py` (funções
`_build` e `_enable_grad_checkpointing`). **Reporte esse valor** se parecer errado.

---

## 16. O que enviar de volta ao final

1. `runs/ablation_heads/ablation_results.csv` e `runs/ablation_lora/ablation_results.csv`
2. `runs/optuna/pareto_front.json` e `runs/optuna/term_presence.json`
3. `runs/champion/seed_*/summary.json`
4. Alguns `runs/champion/curvature_maps/*.png`
5. A linha `[Modelo] ... params treináveis=...` do log.
