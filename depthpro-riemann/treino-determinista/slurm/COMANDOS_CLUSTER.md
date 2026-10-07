# Runbook: rodar o depth-riemannian na h100n2 (copiar e colar, na ordem)

Todos os comandos ja com os caminhos reais. Blocos numerados; so avance quando
a verificacao (VERIFICAR) de cada bloco passar.

---

## 0. Do Mac: enviar TUDO pronto para o cluster (um comando)

A arvore local ja esta montada e conferida: codigo do Francisco + `slurm/` +
`.env` (com o token) + `logs/`. Nao ha nada para clonar nem criar no cluster.

```bash
# (no seu Mac)
rsync -av \
  /Users/juliadollis/Projects_Code/repositorio_ref/depth-riemannian/ \
  dgx-H100-02:/raid/user_juliadollis/projects/depth-riemannian/
```
O `.env` VAI junto de proposito (e o unico jeito do token chegar la) e o
`rsync -a` preserva o `chmod 600`. O `logs/` tambem vai junto, o que resolve o
bloqueador do `#SBATCH --output`: o SLURM abre esse arquivo ANTES de executar o
script, entao se `logs/` nao existisse o primeiro job iria a FAILED sem produzir
nenhuma saida (o `mkdir` de dentro do .slurm roda tarde demais).

## 1. No cluster: conferir que chegou inteiro

```bash
ssh dgx-H100-02
cd /raid/user_juliadollis/projects/depth-riemannian
ls riemann_depthpro_h100_container/scripts   # run_ablation.py, run_optuna.py, train_single.py...
ls slurm                                      # os 3 .slurm + dl_weights.py + READMEs
ls -ld logs && ls -l .env                     # logs/ existe; .env com permissao -rw-------
```
VERIFICAR: os tres `ls` respondem sem erro e o `.env` aparece como `-rw-------`.

## 2. Setup (deps + depth_pro + geometria + pesos)  [~15-30 min]

```bash
cd /raid/user_juliadollis/projects/depth-riemannian
sbatch slurm/run_riemann_setup.slurm
# acompanhar (log completo):
tail -n +1 -f logs/riemann-setup-*.out logs/riemann-setup-*.err
```
VERIFICAR no log:
- `Geometria OK`
- `[ok] torch ... cuda? True`
- `[ok] depth_pro importavel`
- ao final, `depth_pro.pt` com ~1.5 GB.

## 3. Dados: TESTE de 1 cena (valida a internet da Apple)  [minutos]

```bash
MODE=test sbatch slurm/run_riemann_data.slurm
tail -n +1 -f logs/riemann-data-*.out logs/riemann-data-*.err
```
VERIFICAR: `rgb` e `depth` com o MESMO numero e > 0.
- Se derem 0: a CDN da Apple esta bloqueada na h100n2. PARE e me avise (a gente
  aponta para outro dataset no layout rgb/+depth/).

## 4. Dados: subsets (so depois do teste passar)

```bash
# SMOKE
DATA_OUT=hypersim/smoke/train N_IMAGES=60  LIMIT_SCENES=8               sbatch slurm/run_riemann_data.slurm
DATA_OUT=hypersim/smoke/val   N_IMAGES=20  LIMIT_SCENES=6 START_INDEX=200 sbatch slurm/run_riemann_data.slurm

# ABLATION (cenas distintas entre train/val)
DATA_OUT=hypersim/ablation/train N_IMAGES=3000 LIMIT_SCENES=80               sbatch slurm/run_riemann_data.slurm
DATA_OUT=hypersim/ablation/val   N_IMAGES=500  LIMIT_SCENES=25 START_INDEX=200 sbatch slurm/run_riemann_data.slurm

# FULL (so quando for treinar a campea)
DATA_OUT=hypersim/full/train N_IMAGES=20000 LIMIT_SCENES=600               sbatch slurm/run_riemann_data.slurm
DATA_OUT=hypersim/full/val   N_IMAGES=2000  LIMIT_SCENES=120 START_INDEX=700 sbatch slurm/run_riemann_data.slurm
```
VERIFICAR (exemplo): `ls /raid/user_juliadollis/data/hypersim/ablation/train/rgb | wc -l` perto de 3000.

## 5. Smoke test (valida pipeline)  [minutos]

```bash
STEP=smoke sbatch slurm/run_riemann_train.slurm
tail -n +1 -f logs/riemann-train-*.out logs/riemann-train-*.err
```
VERIFICAR (CRITICO): a linha
```
[Modelo] variante=heads | params treinaveis=XXX,XXX | grad_ckpt=False
```
- Se `params treinaveis` = 0 ou minusculo: PARE e me avise o numero.
- Se for grande (milhoes): siga.
E: cria `runs/smoke/ablation_results.csv` sem erro.

## 6. Experimento principal (rodar em background com -d ja e o default do sbatch)

```bash
# ablacao completa (39 configs), heads
STEP=ablation VARIANT=heads sbatch slurm/run_riemann_train.slurm
# (opcional) segunda variante
STEP=ablation VARIANT=heads_lora sbatch slurm/run_riemann_train.slurm
# Optuna
STEP=optuna VARIANT=heads TRIALS=200 EPOCHS_PER_TRIAL=20 sbatch slurm/run_riemann_train.slurm

# acompanhar o CSV parcial a qualquer momento:
cat runs/ablation_heads/ablation_results.csv
```

## 7. Campea (pesos vencedores da ablacao/Optuna)  [horas]

```bash
STEP=champion VARIANT=heads BERHU=0.7 NORMAL=0.9 GAUSS=0.45 SEEDS=3 EPOCHS=100 \
  sbatch slurm/run_riemann_train.slurm
```

## 8. Coletar no fim
- `runs/ablation_heads/ablation_results.csv` (e `_heads_lora`)
- `runs/optuna_heads/pareto_front.json` + `term_presence.json`
- `runs/champion/seed_*/summary.json`
- alguns `runs/champion/curvature_maps/*.png`
- a linha `[Modelo] ... params treinaveis=...`

---

## Fila / status
```bash
squeue -u user_juliadollis          # ver seus jobs
scancel <JOBID>                     # cancelar um job
```
