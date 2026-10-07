# Rodar o depth-riemannian (Francisco) na H100 via SLURM + Singularity

Wrappers SLURM/Singularity para rodar o codigo do Francisco
(`github.com/AKCIT-PIXEL/depth-riemannian`) no nosso cluster, SEM Docker e SEM
modificar o codigo dele. Traduz o fluxo Docker/docker-compose do `README_CONTAINER.md`
dele para o nosso padrao (Singularity `transformers-pytorch-gpu.sif`, bind
`/raid/user_juliadollis:/workspace`, deps num `PYTHONUSERBASE` proprio, seletor de
GPU vazia para jobs de 1 GPU).

> O objetivo desta rodada e SO rodar o codigo dele e coletar os resultados.
> A avaliacao das melhorias (e os fixes que discutimos) fica para depois.

## O que cada script faz

| Script | Papel | SBATCH |
|---|---|---|
| `run_riemann_setup.slurm` | 1x: instala deps + `depth_pro` + roda o teste de geometria + baixa `depth_pro.pt` | 1 GPU, 6h |
| `run_riemann_data.slurm` | Baixa o Hypersim em streaming (cena a cena) para `rgb/`+`depth/` | 1 GPU (ociosa), 24h |
| `run_riemann_train.slurm` | `STEP=smoke\|ablation\|optuna\|champion` (dispatcher) | 1 GPU, 7d |

Nada do codigo do Francisco foi alterado. Estes 3 `.slurm` sao os unicos arquivos novos.

## Decisoes desta rodada
- No: **h100n2** (raid propria `/raid/user_juliadollis`).
- Dados: **stream-download** do Hypersim (o downloader dele).
- Imagem: **reuso do `transformers-pytorch-gpu.sif`** (o mesmo do bokeh/eval).
- Deps isoladas em `/workspace/python-packages-riemann` (NAO mistura com o
  `python-packages` do bokeh).

---

## Passo a passo (na ordem)

### 0. Uma vez, no host da h100n2
```bash
cd /raid/user_juliadollis/projects
git clone https://github.com/AKCIT-PIXEL/depth-riemannian
# copie estes wrappers para dentro do repo (do Mac):
#   rsync -av /Users/juliadollis/Projects_Code/repositorio_ref/depth-riemannian-slurm/ \
#     user_juliadollis@dgx-H100-02:/raid/user_juliadollis/projects/depth-riemannian/slurm/
cd /raid/user_juliadollis/projects/depth-riemannian
```
Confira que o `.sif` existe em `/raid/user_juliadollis/images/transformers-pytorch-gpu.sif`
(senao aponte `IMAGE_PATH=...`).

### 1. Setup (deps + depth_pro + geometria + pesos)
```bash
sbatch slurm/run_riemann_setup.slurm
```
Confira no log:
- `Geometria OK` (o mesmo teste que o Dockerfile dele roda no build).
- `[ok] torch ... cuda? True`
- `depth_pro.pt` com ~1.5 GB em `/raid/user_juliadollis/models/checkpoints/`.

### 2. Dados (Hypersim streaming)
PRIMEIRO teste 1 cena (o proprio Francisco marcou que o download da Apple nao foi
testado no ambiente dele):
```bash
MODE=test sbatch slurm/run_riemann_data.slurm
```
Confira no log que `rgb` e `depth` batem e sao > 0. Se derem 0, a CDN da Apple
(`docs-assets.developer.apple.com`) esta bloqueada na h100n2 (ver Caveats).

Se o teste passar, baixe os subsets (cenas DISTINTAS entre train/val):
```bash
# smoke
DATA_OUT=hypersim/smoke/train N_IMAGES=60  LIMIT_SCENES=8               sbatch slurm/run_riemann_data.slurm
DATA_OUT=hypersim/smoke/val   N_IMAGES=20  LIMIT_SCENES=6 START_INDEX=200 sbatch slurm/run_riemann_data.slurm
# ablation
DATA_OUT=hypersim/ablation/train N_IMAGES=3000 LIMIT_SCENES=80               sbatch slurm/run_riemann_data.slurm
DATA_OUT=hypersim/ablation/val   N_IMAGES=500  LIMIT_SCENES=25 START_INDEX=200 sbatch slurm/run_riemann_data.slurm
# full (so quando for treinar a campea)
DATA_OUT=hypersim/full/train N_IMAGES=20000 LIMIT_SCENES=600               sbatch slurm/run_riemann_data.slurm
DATA_OUT=hypersim/full/val   N_IMAGES=2000  LIMIT_SCENES=120 START_INDEX=700 sbatch slurm/run_riemann_data.slurm
```

### 3. Smoke (valida pipeline; minutos)
```bash
STEP=smoke sbatch slurm/run_riemann_train.slurm
```
**PARE e confira** no log a linha:
```
[Modelo] variante=heads | params treinaveis=XXX,XXX | grad_ckpt=False
```
Se `params treinaveis` for 0 ou minusculo, o codigo dele nao achou as cabecas do
DepthPro nesta versao do pacote (ver secao 15 do `ROTEIRO_IMPLEMENTACAO.md` dele).
Nesse caso, PARE e me avise o numero. Se for grande (milhoes), siga.

### 4. Experimento principal (horas a dias; use segundo plano)
```bash
# ablacao completa (39 configs), variante heads:
STEP=ablation VARIANT=heads sbatch slurm/run_riemann_train.slurm
# (opcional) segunda variante:
STEP=ablation VARIANT=heads_lora sbatch slurm/run_riemann_train.slurm
# busca Optuna:
STEP=optuna VARIANT=heads TRIALS=200 EPOCHS_PER_TRIAL=20 sbatch slurm/run_riemann_train.slurm
```
O CSV da ablacao e escrito a cada config concluida (da pra acompanhar parcial).

### 5. Campea (multi-seed + curvatura)
Pegue os pesos vencedores da ablacao/Optuna e passe nos env:
```bash
STEP=champion VARIANT=heads BERHU=0.7 NORMAL=0.9 GAUSS=0.45 SEEDS=3 EPOCHS=100 \
  sbatch slurm/run_riemann_train.slurm
```

---

## Acompanhar os logs (completos, sem truncar)
```bash
tail -n +1 -f /raid/user_juliadollis/projects/depth-riemannian/logs/riemann-train-<JOBID>.out \
             /raid/user_juliadollis/projects/depth-riemannian/logs/riemann-train-<JOBID>.err
```

## Onde ficam as saidas
`/raid/user_juliadollis/projects/depth-riemannian/runs/`:
- `runs/ablation_heads/ablation_results.csv` (e `_heads_lora`)
- `runs/optuna_heads/pareto_front.json`, `term_presence.json`
- `runs/champion/seed_*/best.pt`, `summary.json`, `curvature_maps/*.png|*.npy`

## O que coletar no fim (o que o Francisco pediu)
1. `ablation_results.csv` (heads e heads_lora)
2. `pareto_front.json` + `term_presence.json`
3. `summary.json` de cada seed da campea
4. Alguns `curvature_maps/*.png`
5. A linha `[Modelo] ... params treinaveis=...` do primeiro run

---

## Caveats (importante)
- **Internet Apple:** o `download_hypersim_stream.py` baixa da CDN da Apple. Se a
  h100n2 nao tiver saida para `docs-assets.developer.apple.com`, o `MODE=test`
  volta com 0 pares. Nesse caso: rodar o download num host com internet aberta,
  ou pedir liberacao, ou apontar `TRAIN_ROOT/VAL_ROOT` para outro dataset ja no
  layout `rgb/`+`depth/` que a gente ja tenha.
- **Pesos do depth_pro:** o setup baixa `depth_pro.pt` do repo `nycu-cplab/Genfocus-Model`
  (o MESMO que ja usamos no Genfocus) e cria um symlink em `<repo>/checkpoints/`
  porque o `create_model_and_transforms` do depth_pro procura os pesos ali por padrao.
- **`params treinaveis`:** o unico ponto do codigo dele que PODE precisar de ajuste
  (substrings de nome das camadas). So mexer se o smoke acusar 0. Nao alteramos nada
  preventivamente.
- **`wget`/`curl` e `git` no container:** o setup clona o `ml-depth-pro` no HOST
  (fora do container) e usa `huggingface-cli` como fallback do download de pesos,
  para nao depender de `wget`/`git` dentro do `.sif`.
- **1 GPU:** todos os jobs usam o seletor de GPU vazia (aborta se a mais livre tiver
  >500 MiB), igual ao eval. O treino do bokeh (4 GPUs) e este job (1 GPU) coexistem
  na h100n2 desde que sobre GPU livre.
