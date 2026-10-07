#!/usr/bin/env python3
"""Monta as arvores de staging por HARDLINK (nao copia dados, nao apaga nada)
e gera os README.md com os numeros vindos do disco."""
import json, os, shutil
from collections import defaultdict
from pathlib import Path

H = Path("/host")
STG_M = H / "_hub_stage_modelos"
STG_A = H / "_hub_stage_avaliacoes"

man_m = json.load(open(H / "_hub_manifesto_modelos.json"))
man_a = json.load(open(H / "_hub_manifesto_avaliacoes.json"))

def liga(src, dest_root, dest_rel):
    d = dest_root / dest_rel
    d.parent.mkdir(parents=True, exist_ok=True)
    if d.exists():
        if d.stat().st_size == os.stat(src).st_size:
            return "ja"
        d.unlink()          # so remove link de staging, nunca a origem
    try:
        os.link(src, d); return "link"
    except OSError:
        shutil.copy2(src, d); return "copia"

cont = defaultdict(int)
for it in man_m:
    cont[liga(it["src"], STG_M, it["dest"])] += 1
for it in man_a:
    cont[liga(it["src"], STG_A, it["dest"])] += 1
shutil.copy2(H / "_hub_avaliacoes.csv", STG_A / "avaliacoes.csv")
print("staging:", dict(cont))

# ---------- inventario a partir do manifesto ----------
seeds = defaultdict(lambda: defaultdict(list))   # prefixo -> braco -> [seed]
origem_de = {}
for it in man_m:
    p = it["dest"].split("/")
    if p[0].startswith("pesos_") and p[-1] == "best.pt":
        seeds[p[0]][p[1]].append(int(p[2].split("_")[1]))
        origem_de[(p[0], p[1], p[2])] = Path(it["src"]).parts[2]  # /host/<raiz>/...
abl = sorted({it["dest"].split("/")[1] for it in man_m
              if it["dest"].startswith("ablacao_spring/") and it["dest"].count("/") == 2})

def tabela(prefixo):
    out, tot = [], 0
    for braco in sorted(seeds[prefixo]):
        ss = sorted(seeds[prefixo][braco])
        tot += len(ss)
        fontes = sorted({origem_de[(prefixo, braco, f"seed_{s}")] for s in ss})
        out.append(f"| `{braco}` | {len(ss)} | {', '.join(str(s) for s in ss)} "
                   f"| {', '.join('`'+f+'`' for f in fontes)} |")
    return "\n".join(out), tot

t_orig, n_orig = tabela("pesos_originais")
t_ret,  n_ret  = tabela("pesos_retreino")
n_bytes = sum(it["size"] for it in man_m)
n_bytes_a = sum(it["size"] for it in man_a) + (STG_A / "avaliacoes.csv").stat().st_size
n_linhas = sum(1 for _ in open(STG_A / "avaliacoes.csv")) - 1

PROTOCOLO = """
## Protocolo

| Item | Valor |
|---|---|
| Dataset | Spring |
| Split | por **sequencias disjuntas**, seed 42, fracoes 0,50 / 0,15 / 0,35 |
| Treino | 593 imagens, 18 sequencias |
| Validacao | 184 imagens, 6 sequencias |
| Teste | 485 imagens, 13 sequencias |
| Modelo | DepthPro, variante `heads` (encoder ViT congelado) |
| Learning rate | 1e-5 |
| Batch | 8 |
| Resolucao | 512 px |
| Epocas | ate 100, com early stop no **F-score de borda de validacao** |
| Teste | avaliado **uma unica vez no fim**, por seed |
| Curvatura metrica | fx = 689,6 px e fy = 1225,9 px na resolucao de 512 |

As contagens 593 / 184 / 485 e 18 / 6 / 13 foram conferidas em disco no split
usado pelos treinos (`/data/spring_split`).

> **Excecao ao protocolo:** o braco `B0_berhu_size768` foi treinado a **768 px**,
> com batch 4, acumulacao de gradiente 2 e gradient checkpointing ligado — nao a
> 512 px / batch 8 como os demais. A **avaliacao** dele, porem, foi feita a 512 px
> como a de todos os outros. Fonte: `cadeia_gpu6.sh` e o log `p4_B0_berhu_size768_s0-2.log`.

## Os bracos

| Braco | Perda |
|---|---|
| `B0` | berHu 0,7 puro — **controle** |
| `B1` | berHu 0,7 + curvatura 0,45 — **curvatura isolada** |
| `B3` | berHu 0,7 + normal 0,9 + curvatura 0,45 |
| `Bgrad` | berHu 0,7 + grad 0,3 |
| `Bmetric` | berHu 0,7 + metric 0,2 |
| `Bgeod` | berHu 0,7 + geod 0,1 |

O sufixo `teto5` / `teto50` / `teto1000` e o valor de `gauss_clamp`, o teto da
curvatura Gaussiana, em **1/m²**. Os pesos `--berhu 0.7`, `--gauss 0.45` e
`--normal 0.9` foram lidos de `fila_retreino.sh` no cluster.
"""

CARREGAR = """
## Como carregar um peso

`best.pt` guarda **apenas os parametros treinaveis** (o encoder ViT fica congelado,
so as cabecas sao treinadas), entao ele **nao e um modelo completo**. Ele precisa ser
aplicado por cima do `depth_pro.pt` oficial, com `strict=False`:

```python
import torch
from riemann.model import build_model          # de github.com/juliadollis/akcit-julia

modelo = build_model("/caminho/para/depth_pro.pt", variant="heads", device="cuda")
sd = torch.load("pesos_originais/B3_gauss_metrica_teto50/seed_0/best.pt",
                map_location="cuda")
faltando = modelo.load_state_dict(sd, strict=False)
assert not faltando.unexpected_keys, faltando.unexpected_keys
modelo.eval()
```

`missing_keys` vai listar o backbone congelado — isso e o esperado. O que **nao** pode
aparecer e `unexpected_keys`. Ver `riemann/trainer.py::_save_trainable` e
`scripts/make_figures.py` no repositorio de codigo.
"""

AVISO = """
## Aviso mais importante: `pesos_retreino/` nao sao os originais

Em **2026-09-10** uma liberacao de quota apagou 37 dos 47 `best.pt` da campanha.
As **metricas** sobreviveram (estao em `metricas_campanha_completa/`), os **pesos** nao.

Os arquivos em `pesos_retreino/` **NAO sao copias dos originais perdidos**. Sao
**execucoes novas**, da mesma config e da mesma seed. Nos bracos com curvatura elas
**nao reproduzem** o original: o caminho da curvatura e **nao deterministico na GPU**,
porque o backward do `F.pad(mode="replicate")` usa `atomicAdd`. O codigo nao chama
`torch.use_deterministic_algorithms`, e o early stop amplifica qualquer diferenca.
(`F.pad(..., mode="replicate")` aparece em `riemann/geometry.py` e `riemann/losses.py`,
no caminho de `surface_curvatures`.)

Consequencias praticas:

- **Somente `pesos_originais/` corresponde aos numeros do relatorio.**
- Para qualquer peso, **o `test_metrics.json` ao lado dele e a verdade daquele peso**.
  Nao use o numero do relatorio para descrever um peso de `pesos_retreino/`.
- `metricas_campanha_completa/` guarda as metricas da campanha original, inclusive das
  seeds cujos pesos foram perdidos. Sao numeros **sem peso correspondente** aqui.
"""

readme_m = f"""---
license: other
library_name: pytorch
tags:
- depth-estimation
- monocular-depth-estimation
- depthpro
- riemannian-geometry
- spring
---

# depthpro-riemann-modelos

Checkpoints do projeto **depth-riemannian**: testar se um termo de **curvatura
Gaussiana metrica** na funcao de perda melhora a **borda** do DepthPro em estimativa
monocular de profundidade.

Codigo: <https://github.com/juliadollis/akcit-julia>
Avaliacoes: `akcit-dephpro/depthpro-riemann-avaliacoes` (repo de dataset)

Conteudo: **{len([i for i in man_m if i['dest'].endswith('best.pt')])} checkpoints**,
{len(man_m)} arquivos, {n_bytes/2**30:.1f} GiB.
{PROTOCOLO}
## O que tem aqui

### `pesos_originais/` — {n_orig} seeds

Treinos originais cujos pesos sobreviveram. **Sao estes que correspondem aos numeros
do relatorio.**

| Braco | Seeds | Quais | Origem no cluster |
|---|---|---|---|
{t_orig}

### `pesos_retreino/` — {n_ret} seeds

Reexecucoes feitas depois da perda de 2026-09-10. **Leia o aviso abaixo antes de usar.**

| Braco | Seeds | Quais | Origem no cluster |
|---|---|---|---|
{t_ret}

### `pesos_confirma/` — vazio

A campanha de confirmacao dos termos `grad` / `metric` / `geod` estava **em andamento**
no momento desta publicacao. A unica seed existente (`Bgrad/seed_0`) tinha apenas
`best.pt`, **sem `test_metrics.json`**, ou seja, era um parcial de treino em andamento.
Por isso **nao foi publicada**: um peso sem o numero dele enganaria quem baixasse.

### `ablacao_spring/` — {len(abl)} configuracoes

{chr(10).join('- `' + c + '`' for c in abl)}

> **Atencao — a ablacao nao tem `test_metrics.json`.** Diferente das seeds, cada config
> da ablacao traz apenas `best.pt`, `history.json` e `summary.json`. Os numeros dela
> estao em `ablacao_spring/ablation_results.csv`. E esse CSV reporta a metrica de
> **validacao na melhor epoca**, nao um teste retido: para toda config o campo
> `boundary_fscore` do CSV e **identico** ao `best_metric` do `summary.json`, que e a
> metrica monitorada na validacao. **Nao compare esses numeros com os
> `test_metrics.json` das seeds** — nao sao a mesma medida.
>
> A configuracao `heads__B7_normal_dom` estava **treinando** no momento da publicacao
> (o diretorio existia vazio) e **nao foi publicada**.

### `metricas_campanha_completa/` — so metricas, sem pesos

Copia de `runs_riemann_metricas_preservadas/`: os `test_metrics.json`, `summary.json`,
`history.json` e `test_summary.json` da campanha original, incluindo as seeds cujos
pesos foram perdidos. Inclui tambem `ZERO_SHOT_spring/`, o DepthPro **sem fine-tune**
na mesma mesa de teste.
{CARREGAR}{AVISO}
## Refazer um teste

Cada `seed_N/` tem `best.pt` junto de `test_metrics.json` (o resultado no teste),
`summary.json` (melhor epoca e metrica monitorada) e `history.json` (curva por epoca).
Para reavaliar um peso numa mesa, carregue como acima e rode a mesa de teste a **512 px**,
que e a resolucao usada em todas as avaliacoes publicadas.

## Procedencia

Publicado a partir de `/raid/user_juliadollis/julia_docker/` no cluster DGX-H100-01.
Nada foi apagado na origem. Um espelho anterior, parcial e no namespace pessoal, existe
em `juliadollis/depthpro-spring-ft`.
"""

readme_a = f"""---
license: other
tags:
- depth-estimation
- evaluation
- depthpro
- spring
configs:
- config_name: default
  data_files:
  - split: train
    path: avaliacoes.csv
---

# depthpro-riemann-avaliacoes

Avaliacoes do projeto **depth-riemannian**: testar se um termo de **curvatura Gaussiana
metrica** na funcao de perda melhora a **borda** do DepthPro.

Pesos: `akcit-dephpro/depthpro-riemann-modelos`
Codigo: <https://github.com/juliadollis/akcit-julia>

Conteudo: **{n_linhas} avaliacoes** (modelo x mesa), {len(man_a)} arquivos brutos,
{n_bytes_a/2**20:.1f} MiB.
{PROTOCOLO}
## Estrutura

```
avaliacoes.csv                      tabela consolidada, 1 linha por (modelo, mesa)
<rotulo>/<mesa>/test_metrics.json   metricas agregadas daquela avaliacao
<rotulo>/<mesa>/por_imagem.csv      uma linha por imagem do teste
<rotulo>/<mesa>/meta.json           checkpoint avaliado, raiz, n_imagens, n_cenas, size
```

O `rotulo` tem o formato `<origem>__<braco>__seed_N`, e `origem` e o diretorio de
runs no cluster (`runs_riemann` = originais, `runs_retreino` / `runs_b0_retreino` =
retreino). A unica excecao e `zero_shot`, o DepthPro **sem fine-tune**, que nao tem
origem, braco nem seed — nessas tres colunas ele fica **vazio**.

Todas as {n_linhas} avaliacoes usaram a mesma mesa (`spring_test`): 485 imagens,
13 cenas, 512 px, batch 8.

## `avaliacoes.csv`

{n_linhas} linhas. Colunas: `rotulo`, `origem`, `braco`, `seed`, `mesa`, `n_imagens`,
`n_cenas`, seguidas de **todas** as metricas do `test_metrics.json`
(`abs_rel`, `sq_rel`, `rmse`, `rmse_log`, `log10`, `d1`, `d2`, `d3`,
`boundary_precision`, `boundary_recall`, `boundary_fscore`, `boundary_fmax`,
`boundary_fmax_thresh`, `boundary_precision_at_fmax`, `boundary_recall_at_fmax`,
`boundary_f_auc`) e, no fim, `checkpoint`, o caminho do peso avaliado no cluster.

```python
from datasets import load_dataset
ds = load_dataset("akcit-dephpro/depthpro-riemann-avaliacoes")["train"]
```

## Por que existe `por_imagem.csv`: o n efetivo e 13, nao 485

A mesa de teste tem **485 imagens**, mas elas vem de apenas **13 cenas**. Imagens da
mesma sequencia nao sao observacoes independentes. O **n efetivo e 13**.

Tratar n = 485 produz barras de erro pequenas demais e faz qualquer diferenca parecer
significativa. Agrupando por cena, o **erro padrao da media do `boundary_fscore` e
0,0595** para o `zero_shot` — **da mesma ordem do maior efeito medido**. Conferido a
partir dos proprios `por_imagem.csv`: nas {n_linhas} avaliacoes esse erro padrao vai de
**0,039 a 0,060** (mediana 0,051), sempre com n = 13 cenas.

**Toda estatistica honesta aqui e agrupada por cena.** O `por_imagem.csv` existe
exatamente para permitir isso: ele tem a coluna `cena`, entao da para agregar por cena
antes de comparar bracos.

```python
import pandas as pd
d = pd.read_csv("runs_riemann__B3_gauss_metrica_teto50__seed_0/spring_test/por_imagem.csv")
por_cena = d.groupby("cena")["boundary_fscore"].mean()   # n = 13
media, erro = por_cena.mean(), por_cena.std(ddof=1) / len(por_cena) ** 0.5
```

## Aviso: originais x retreino

As linhas com `origem = runs_riemann` avaliam os pesos **originais**. As com
`origem = runs_retreino` ou `runs_b0_retreino` avaliam **reexecucoes** feitas depois de
uma perda de checkpoints em 2026-09-10; nos bracos com curvatura elas **nao reproduzem**
o treino original, porque o caminho da curvatura e nao deterministico na GPU. Cada linha
descreve **o peso que ela avaliou**, e nada alem disso. Detalhes no README do repo de
modelos.

## Procedencia

Publicado a partir de `/raid/user_juliadollis/julia_docker/avaliacoes/` no cluster
DGX-H100-01. Nada foi apagado na origem.
"""

(STG_M / "README.md").write_text(readme_m)
(STG_A / "README.md").write_text(readme_a)
print(f"READMEs escritos. modelos: {n_orig} originais + {n_ret} retreino + {len(abl)} ablacao")
print(f"avaliacoes: {n_linhas} linhas")
