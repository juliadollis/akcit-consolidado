# depthpro-riemann — campanha riemanniana do DepthPro

Todo o código da campanha que testou se adicionar **termos geométricos
riemannianos** à perda de fine-tuning do **DepthPro** melhora a qualidade de
borda da profundidade monocular.

---

## 1. O que o estudo testou e o que achou

A hipótese era que regularizar a superfície de profundidade com quantidades da
geometria riemanniana — gradiente, normais, geodésica, tensor métrico e
**curvatura gaussiana** — daria bordas mais nítidas que a berHu pura. Fizemos
fine-tune do DepthPro no Spring (split 593/184/485, seed 42, sequências
disjuntas), **n = 6 seeds por braço**, e avaliamos nas três mesas: teste do
Spring (485 imagens / 13 cenas), DIODE val *indoors* (325 imagens / 10 scans) e
DIODE *outdoor*. O resultado é um **nulo, e para a curvatura um nulo negativo**:
o termo de curvatura gaussiana **piora a borda nos três conjuntos**, com apenas
**1 seed a favor em 36 comparações pareadas**, e o braço que a isola (B1) perde
mais que o que a mistura com o termo normal (B3) — isto é, o normal *mascara*
parte do estrago em vez de ajudar. Nenhum outro termo tem ganho que sobreviva
aos três conjuntos: o `grad` vence 6/6 no Spring mas empata no DIODE interno
(−0,0002) e o `geod` faz o inverso (+0,0134 no DIODE, 4/6 no Spring). A ablação
de termos isolados do Wallisson, rodada com n=1 e depois ampliada a n=6
(seeds 42–47), aponta na mesma direção para `gauss` e `normal`.

Os números completos estão em [`relatorios/`](relatorios/) — comece por
[`relatorios/RESULTADOS-FINAIS.md`](relatorios/RESULTADOS-FINAIS.md) e pelo
dossiê técnico [`relatorios/dossie-riemann.html`](relatorios/dossie-riemann.html).

---

## 2. Mapa da pasta

### Árvores de treino

| Item | Para que serve / quando se usa |
|---|---|
| [`treino/`](treino/) | O código da **campanha de seeds** — é o que produziu os números de `RESULTADOS-FINAIS.md`. `riemann/losses.py` tem os seis termos, `riemann/geometry.py` as curvaturas, `scripts/train_single.py` é o ponto de entrada de um braço × faixa de seeds. Use este quando quiser reproduzir a campanha **como ela foi**. |
| [`treino-determinista/`](treino-determinista/) | Cópia de `treino/` **com a correção do `F.pad`** (armadilha (a) abaixo), mais `corrige_determinismo.py` (aplica o patch) e `compara_determinismo.py` (compara duas rodadas idênticas e dá o veredito). Use este para **qualquer experimento novo** com termos de curvatura: aqui a mesma seed reproduz bit a bit. |
| [`ablacao/`](ablacao/) | Ablação dos cinco termos **isolados**, código de terceiro (Wallisson), **intocado de propósito** — só o dataset mudou para o Spring. Não aplique a correção de determinismo aqui: misturaria duas variáveis e tornaria a seed 42 existente incomparável com as novas. |

### Operação

| Item | Para que serve / quando se usa |
|---|---|
| [`avaliacao/avalia_modelos.py`](avaliacao/avalia_modelos.py) | Avalia N checkpoints × M mesas, gerando agregado (`test_metrics.json`) **e** por imagem (`por_imagem.csv`). Separa treino de avaliação: dado um `best.pt`, mede quantas vezes for preciso. Tem reivindicação atômica por `mkdir`, então dois fiscais podem varrer a mesma lista sem colidir. |
| `avaliacao/avalia_modelos.py.antes_do_claim` | A versão **sem** a reivindicação atômica, guardada como registro: era ela que fazia duas GPUs avaliarem o mesmo par e uma não render nada. |
| [`avaliacao/patch_claim.py`](avaliacao/patch_claim.py) | O patch que transformou uma na outra. Leia-o para entender por que `mkdir` resolve e um `if not exists` não. |
| [`avaliacao/fiscal_avaliacao.sh`](avaliacao/fiscal_avaliacao.sh) | Fiscal que fica varrendo checkpoints pendentes numa GPU. Rode um por placa. |
| [`avaliacao/roda_avaliacao.sh`](avaliacao/roda_avaliacao.sh) | Reavalia todos os checkpoints numa ou mais mesas (`nome=raiz`). Sem argumento de mesa, usa o teste do Spring. |
| [`avaliacao/roda_zs.sh`](avaliacao/roda_zs.sh) | Zero-shot do DepthPro no teste do Spring — o piso de cada mesa. |
| [`avaliacao/valid_out.py`](avaliacao/valid_out.py) | Sanidade do DIODE preparado: fração de pixels válidos e faixa de profundidade, interno vs. externo. |
| [`avaliacao/diag_cenas.py`](avaliacao/diag_cenas.py) | Por que `seq0020` e `seq0043` pontuam tão mal: o `fx` varia 4,7× entre sequências do Spring e o `--max-depth 100` recorta de forma não-linear. |
| [`avaliacao/teste_alinhamento.py`](avaliacao/teste_alinhamento.py) | Verifica se o alinhamento afim está no espaço certo (profundidade vs. disparidade). O MiDaS/DPT alinha em disparidade; mínimos quadrados em Z é dominado pelos pixels longe. |
| [`avaliacao/dp_sem.py`](avaliacao/dp_sem.py) | Erro padrão da média **por cena** do `boundary_fscore` — o n efetivo do Spring é 13, não 485. |
| [`avaliacao/dp_avaliacoes_csv.py`](avaliacao/dp_avaliacoes_csv.py) | Consolida `avaliacoes/` num CSV de uma linha por (modelo, mesa), para subir ao Hub. |

| Item | Para que serve / quando se usa |
|---|---|
| [`filas/passo4_seeds.sh`](filas/passo4_seeds.sh) · `passo4_seeds2.sh` | **O orquestrador da campanha de seeds** (n=3 → n=10). Recebe tarefas `<b0\|b1\|b3>:<teto>:<seed_inicio>:<quantas>`, faixas disjuntas por construção, e o `train_single.py` pula seed que já tem `test_metrics.json`. |
| [`filas/passo4.sh`](filas/passo4.sh) · `passo4_teto.sh` · `roda_passo4.sh` · `prep_passo4.sh` | O par B3+B0 original; a variante com teto de curvatura parametrizado; o lançador com guarda de memória e retentativa (um treino ocupa ~80 GB, a placa inteira); e o preparo de pré-requisitos (pesos + os três splits do Spring). |
| [`filas/fila_ablacao_seeds.sh`](filas/fila_ablacao_seeds.sh) · `fila_abl2.sh` | Levam a ablação de n=1 a n=6 (seeds 43–47). Têm reivindicação atômica por `mkdir`, guarda de quota (param em 490 GiB) e só entram em placa vazia. |
| [`filas/fila_retreino.sh`](filas/fila_retreino.sh) · [`filas/retreina_b0.sh`](filas/retreina_b0.sh) | Retreino do B0 berHu 512, o melhor braço, cujos pesos foram apagados. Deliberadamente **não** ligam `use_deterministic_algorithms`: o objetivo é ficar o mais igual possível ao original. |
| [`filas/fila_confirma.sh`](filas/fila_confirma.sh) · `fila_grad.sh` | Confirmação de braços específicos e a fila do `grad`. |
| [`filas/roda_ablacao.sh`](filas/roda_ablacao.sh) | Roda a ablação do Wallisson exatamente como entregue, só trocando o dataset. Todos os defaults dele preservados. |
| [`filas/valida_determinismo.sh`](filas/valida_determinismo.sh) | Roda a **mesma** seed do **mesmo** braço duas vezes com o código corrigido e compara. Usa o B1 teto 5, que é curvatura isolada — o caminho mais curto até `surface_curvatures`, logo o teste mais sensível. |
| [`filas/pipeline_noite.sh`](filas/pipeline_noite.sh) · [`filas/vigia_noite.sh`](filas/vigia_noite.sh) | Cadeia sem intervenção da inferência até a tabela, e o registro contínuo do estado a cada 5 min — para haver uma linha do tempo de manhã, mesmo que ninguém estivesse olhando. |

| Item | Para que serve / quando se usa |
|---|---|
| [`dados/baixa_spring.sh`](dados/baixa_spring.sh) | Baixa do DaRUS os 3 pacotes do Spring: só treino, câmera esquerda, ~24 GB em vez dos ~280 GB do dataset inteiro. |
| [`dados/prep_spring_full.sh`](dados/prep_spring_full.sh) | Prepara o Spring completo (37 seqs × 248 quadros, passo 10) e refaz a medida de \|K\| sobre esse conjunto maior. Escreve em `spring_prep/full`, não mexe no `test`. |
| [`dados/zip_remoto.py`](dados/zip_remoto.py) | Extrai membros **escolhidos** de um zip remoto por HTTP Range, sem baixar o zip. Ler 20 imagens de um pacote de 24 GB custaria ~2 h de download; aqui custa segundos. |
| [`dados/baixa_diode.sh`](dados/baixa_diode.sh) · `baixa_diode_par.sh` · `rejunta_diode.sh` | Download do DIODE, serial e paralelo, e a remontagem dos pedaços. |
| [`dados/acha_diode.py`](dados/acha_diode.py) · `confere_diode.py` · [`inspeciona_diode.py`](dados/inspeciona_diode.py) | Localizam o DIODE, conferem a contagem e validam que a hierarquia e a física (metros, máscara binária) são as esperadas. |

| Item | Para que serve / quando se usa |
|---|---|
| [`hub/dp_stage.py`](hub/dp_stage.py) | Monta as árvores de staging do Hub **por hardlink** (não copia dados, não apaga nada) e gera os `README.md` com os números lidos do disco. É o começo de qualquer publicação. |
| [`hub/dp_manifesto.py`](hub/dp_manifesto.py) · [`dp_verifica.py`](hub/dp_verifica.py) | Geram o manifesto do que deve subir e conferem, por tamanho, que cada arquivo chegou. |
| [`hub/dp_sobe_modelos.py`](hub/dp_sobe_modelos.py) · `dp_sobe_avaliacoes.py` | O upload em si, com `upload_large_folder` (resumível, com retry, aguenta os 70 GiB). |
| [`hub/sobe_seed_repo.py`](hub/sobe_seed_repo.py) · `sobe_um_seed.py` · `sobe_um_seed2.py` · `sobe_ablacao_seed.py` · `sobe_ablacao.py` · `sobe_ckpt.py` · `sobe_riemann.py` · `sobe_retreino.py` | Sobem **uma seed** ou um conjunto logo que o treino fecha, preservando o layout `<condição>/seed_<N>/best.pt` — condição e semente são a identidade da execução. Vários comparam o número novo contra o original: o retreino não é bit a bit igual, e essa diferença é a estimativa do piso de ruído do protocolo. |
| [`hub/salva_orfaos.py`](hub/salva_orfaos.py) · [`sobe_orfaos.sh`](hub/sobe_orfaos.sh) · [`orfaos.py`](hub/orfaos.py) | Encontram e salvam o que existe **só em disco**. Foram escritos quando o `akcit-dephpro` bateu o limite e as seeds 45/46 falharam calado: a fila anterior registrava "subiu" sem checar o código de saída. Aqui o resultado é verificado. |
| [`hub/apaga_seguro.py`](hub/apaga_seguro.py) | Apaga `best.pt` local **somente** se existir gêmeo byte a byte no Hub, provado por sha256 antes de tocar em qualquer arquivo. Nunca apaga `.json`. Sem `--executar`, só relata. |
| [`hub/hash_local.sh`](hub/hash_local.sh) | sha256 dos checkpoints em disco, para o cruzamento com os sha LFS do Hub. |
| [`hub/baixa_pesos_hub.py`](hub/baixa_pesos_hub.py) | Traz pesos de volta do Hub quando o disco já não tem. |
| [`hub/confere_hub_ablacao.py`](hub/confere_hub_ablacao.py) · `confere_tudo_hf.py` · `confere_hub.py` · `confere_37.py` · `confere_agente.py` · `confere_repo.py` · `confere_retreino_hub.py` · `balanco_hub.py` · `quais_faltam.py` · `falta_no_hub.py` · `ve_repo_abl.py` | Conferência: o que está no Hub, o que falta, quanto pesa, por seed e por braço. `confere_37.py` existe por causa da perda de 2026-09-10. |
| [`hub/sonda_orgs.py`](hub/sonda_orgs.py) · `acha_repo_hf.py` · `acha_repo_hf2.py` · `ver_org.py` · `ver_repo.py` · `quota_hf.py` · `testa_cota.py` · `testa_akcitpixel.py` · `cria_org_arquivo.py` | Sondagem de orgs e cota. Foram escritos para achar **onde** ainda cabia peso quando o `akcit-dephpro` começou a devolver 403 no LFS. |
| `hub/dp_probe.py` · `dp_probe2.py` · `dp_patch_readme.py` · `busca_perdidos.py` · `salva_h100n1.py` | Sondagens pontuais de repo, correção do README publicado, busca pelos checkpoints perdidos e o arquivamento geral na `akcit-h100n1`. |
| `hub/roda_*.sh` | Invólucros de uma linha que rodam o `.py` correspondente dentro do container com `HF_TOKEN` lido do `.env` — é como os scripts de Hub foram de fato executados. |

| Item | Para que serve / quando se usa |
|---|---|
| [`tabelas/tabela_out.py`](tabelas/tabela_out.py) | A tabela principal da campanha: agregado por braço e o contraste **pareado** seed a seed contra o controle. O pareamento remove a variação do sorteio, e é dele que sai o "1 seed a favor em 36". |
| [`tabelas/tabela_diode.py`](tabelas/tabela_diode.py) · `tabela_diode2.py` | As mesmas tabelas para o DIODE interno e externo. |
| [`tabelas/tabela_ablacao.py`](tabelas/tabela_ablacao.py) | Tabela da ablação de termos isolados. |
| [`tabelas/dur.py`](tabelas/dur.py) | Duração de cada treino, lida dos `summary.json` (com fallback por mtime dos artefatos). |

### Rota C do bokeh

Esta campanha emprestou o DepthPro para alinhar profundidade na rota C do
BokehNet. O resto daquela árvore já está publicado em
[`../bokehnet-regen/`](../bokehnet-regen/) — aqui ficam **só as peças que
escrevemos** e o registro do que mudou.

| Item | Para que serve / quando se usa |
|---|---|
| [`bokeh/nosso/realdof.py`](bokeh/nosso/realdof.py) · [`realdof_images.py`](bokeh/nosso/realdof_images.py) | A fonte de dados RealDOF da rota C. |
| [`bokeh/nosso/depth_align.py`](bokeh/nosso/depth_align.py) | O alinhamento afim de profundidade por amostra (mínimos quadrados), que é onde o DepthPro entra na rota C. |
| [`bokeh/patches/`](bokeh/patches/) | `diff -u` entre os backups `*.antes_caminho_a` e a versão atual de `scripts/run_route_c.py` e `src/routes/route_c.py` — o registro exato da mudança do "caminho A". |
| [`bokeh/caminho_a_scripts/`](bokeh/caminho_a_scripts/) | Os scripts nossos do caminho A: `patch_rotac.py` (a mudança em si), `smoke_caminho_a.py`, `sonda_dominio.py`, `roda_condicoes.sh`, `encadeia.sh`, `tabela.py`. |
| [`bokeh/caminho_b_analise/merge_ckpt.py`](bokeh/caminho_b_analise/merge_ckpt.py) | O **merge de checkpoint** do caminho B, com `diff_ckpt.py` / `quais_mudaram.py` / `inspect_ckpt.py` ao lado para ver quais tensores de fato mudaram antes de fundir. |

### Relatórios

| Item | Para que serve / quando se usa |
|---|---|
| [`relatorios/RESULTADOS-FINAIS.md`](relatorios/RESULTADOS-FINAIS.md) | **Comece aqui.** Tabela do teste do Spring, contraste pareado e a leitura. |
| [`relatorios/RESULTADOS-DIODE.md`](relatorios/RESULTADOS-DIODE.md) | Generalização fora do domínio: DIODE interno e externo. |
| [`relatorios/ABLACAO.md`](relatorios/ABLACAO.md) · [`ablacao_spring_results.csv`](relatorios/ablacao_spring_results.csv) | A ablação de termos isolados, e por que os números dela **não** se comparam com os da campanha (ela avalia validação, 184/6; a campanha avalia teste, 485/13). |
| [`relatorios/RESULTADOS-BOKEH-REALDOF.md`](relatorios/RESULTADOS-BOKEH-REALDOF.md) | O que saiu da rota C / RealDOF. |
| [`relatorios/dossie-riemann.html`](relatorios/dossie-riemann.html) | O dossiê técnico completo da campanha. |

---

## 3. Como reproduzir

Tudo roda em container. A imagem sai do `Dockerfile` de `treino/`:

```bash
docker build -t riemann-depthpro:latest treino/
```

### Treino — campanha de seeds

`FX=2585.859` é a mediana da focal das 37 sequências do Spring. O B3 é a config
"champion" (berHu 0,7 + normal 0,9 + curvatura 0,45); o B0 é o controle berHu
puro; o B1 isola a curvatura.

```bash
B=/raid/user_juliadollis/julia_docker
R=$B/depth-riemannian
FX=2585.859
NOME=B3_gauss_metrica_teto50

docker run --rm --name julia_seeds --gpus '"device=0"' \
  --user "$(id -u):$(id -g)" --shm-size=32g --ipc=host \
  -v "$R/scripts":/workspace/scripts:ro -v "$R/riemann":/workspace/riemann:ro \
  -v "$B/data":/data -v "$B/models":/models -v "$B/runs_riemann":/workspace/runs \
  -e HOME=/tmp -e PYTHONUNBUFFERED=1 -w /workspace riemann-depthpro:latest \
  python scripts/train_single.py \
    --train-root /data/spring_split/train \
    --val-root   /data/spring_split/val \
    --test-root  /data/spring_split/test \
    --checkpoint /models/checkpoints/depth_pro.pt \
    --out-dir "/workspace/runs/$NOME" \
    --seed-inicio 0 --seeds 6 \
    --berhu 0.7 --normal 0.9 --gauss 0.45 --grad 0 --geod 0 --metric 0 \
    --gauss-metrica --fx-orig "$FX" --gauss-clamp 50
```

Na prática use [`filas/passo4_seeds.sh`](filas/passo4_seeds.sh), que faz isso
com guarda de memória, retentativa e faixas de seed disjuntas entre GPUs:

```bash
bash filas/passo4_seeds.sh 0 b0:-:0:6 b3:50:0:6 b1:5:0:6
```

### Ablação de termos isolados

Código do Wallisson, defaults dele preservados. **`--focal 2585.859` é
obrigatório** — sem ele a conversão disparidade→profundidade do Spring usa o
`fx` errado.

```bash
docker run --rm --gpus '"device=6"' \
  --user "$(id -u):$(id -g)" --shm-size=32g --ipc=host \
  -v "$B/ablacao-wallisson":/workspace \
  -v "$B/data":/data -v "$B/models":/models -v "$B":/host \
  -e HOME=/tmp -e PYTHONUNBUFFERED=1 -w /workspace riemann-depthpro:latest \
  python3 /workspace/scripts/run_ablation.py \
    --train-root /data/spring_split/train \
    --val-root   /data/spring_split/val \
    --checkpoint /models/checkpoints/depth_pro.pt \
    --out-dir    /host/runs_ablacao_seeds/seed_43 \
    --focal 2585.859 \
    --filter B0 B1 \
    --seed 43 \
    --no-eval-zero-shot
```

### Avaliação

```bash
python3 avaliacao/avalia_modelos.py \
  --checkpoints "/workspace/runs/*/seed_*/best.pt" \
  --dataset spring=/data/spring_split/test \
  --dataset diode_in=/data/diode_prep/val \
  --dataset diode_out=/data/diode_out_prep/val \
  --checkpoint-base /models/checkpoints/depth_pro.pt \
  --saida /host/avaliacoes --incluir-zero-shot
```

> **O `--batch-size` tem de ser 8.** O agregado sai do mesmo
> `Trainer.validate()` que gerou o `test_metrics.json` de cada seed, e o
> `validate` tira a média das métricas **por batch** (o `mde_metrics` junta os
> pixels válidos do batch inteiro). Mudar o batch muda o agregado, e o número
> deixa de ser comparável com o histórico. O `por_imagem.csv` é outra
> agregação — mede cada imagem isolada — e as duas convivem de propósito.

---

## 4. Onde estão os pesos

| Repo | O que tem |
|---|---|
| `akcit-dephpro/depthpro-riemann-modelos` (privado) | A campanha de seeds inteira e a ablação das seeds **42, 43, 44 e 47**. |
| `akcit-h100n1/depthpro-ablacao-seeds` (privado) | As seeds **45 e 46** da ablação. |
| `akcit-dephpro/depthpro-riemann-avaliacoes` (dataset, privado) | Os `test_metrics.json` e `por_imagem.csv` de cada (modelo, mesa). |

**A org `akcit-dephpro` bateu 100 GB de armazenamento privado e devolve `403
Forbidden` no LFS.** Foi por isso que as seeds 45 e 46 foram para a
`akcit-h100n1`: não houve escolha de organização, houve falta de cota. Qualquer
upload novo para `akcit-dephpro` vai falhar — e, como a fila antiga não checava
o código de saída do upload, **falhava calado**. Use
[`hub/sobe_orfaos.sh`](hub/sobe_orfaos.sh), que verifica.

**Dois checkpoints da seed 42 foram perdidos:** `heads__B7_champion_prev` e
`heads__B7_normal_dom`. Existiam em disco, **nunca subiram**, e o disco foi
varrido. As métricas deles sobreviveram; os pesos não. Reavaliar esses dois
braços hoje significa retreinar. É o mesmo padrão de 2026-09-10, quando 37 de
47 `best.pt` foram embora numa liberação de quota. O `/raid` **não** é
armazenamento durável: **checkpoint que não está no Hub não existe.**

---

## 5. As duas armadilhas conhecidas

### (a) O backward do `F.pad(mode="replicate")` não é determinístico na CUDA

O backward de `F.pad` com `mode="replicate"` usa `atomicAdd` na CUDA. São **8
pontos** no caminho geométrico, e a soma em ponto flutuante não é associativa:
a ordem em que as threads acumulam muda entre execuções. Resultado: os braços
**com curvatura** não reproduzem. Duas execuções da **mesma seed** diferem por
**\|Δ\| médio de 0,0148** no F-score de borda (máximo observado 0,0389) — da
ordem do efeito que se quer medir. Os braços sem geometria (berHu puro) batem
exato, o que é exatamente o que aponta o dedo para o padding.

A correção está em [`treino-determinista/`](treino-determinista/);
[`treino-determinista/corrige_determinismo.py`](treino-determinista/corrige_determinismo.py)
é o patch e `compara_determinismo.py` dá o veredito. Valide com
[`filas/valida_determinismo.sh`](filas/valida_determinismo.sh).

Consequência prática: **qualquer número de braço com curvatura produzido pela
árvore `treino/` tem esse piso de ruído**, e um resultado só é afirmável se o
efeito for maior que ele. Foi por isso que a campanha foi de n=3 a n=6.

### (b) Bind mount do Docker cujo alvo não existe nasce como `root`

Se você passa `-v /caminho/que/nao/existe:/dentro` e o caminho do host não
existe, o Docker **cria** o diretório — como `root`. Com `--user $(id -u):$(id -g)`,
o processo dentro do container não consegue escrever nele, e o treino morre de
`Permission denied` numa pasta que "está ali". Pior: o diretório fica no host
com dono `root` e você não consegue removê-lo.

**Crie o diretório antes do `docker run`:**

```bash
mkdir -p "$B/runs_riemann/$NOME"
docker run ... -v "$B/runs_riemann":/workspace/runs ...
```

Os scripts de `filas/` já fazem isso. Reproduzir à mão, não.

---

## 6. O que ficou de fora

**Segredos.** Nenhum `.env` subiu. Os scripts leem `HF_TOKEN` do ambiente ou de
um `.env` do cluster em tempo de execução; não há token no código. O
[`treino/slurm/.env.example`](treino/slurm/.env.example) mostra as chaves
esperadas com valores vazios.

**Binários e dados.** Nenhum `*.pt`, `*.safetensors`, `*.npy`, imagem, `*.tar`,
`*.zip` ou `*.parquet`. Pesos estão no Hub (seção 4); dados vêm do DaRUS
(Spring) e do DIODE, com os scripts de `dados/`. Também ficaram fora os `.git`
internos das subpastas, `venv/`, `__pycache__/`, `wandb/`, `outputs/`,
`runs_*/`, `data/`.

**Saídas de execução.** `caminho_a_rotac/out/` e `caminho_b_analise/saida/` são
milhares de `.json` de resultado por imagem — os números consolidados estão em
`relatorios/`.

**A árvore `bokehnet-regen/` inteira.** Já está publicada em
[`../bokehnet-regen/`](../bokehnet-regen/), e um `find` ingênuo nela acha ~103
mil "arquivos de código" que são quase todos de venv ou cache. Em
[`bokeh/`](bokeh/) ficaram só as peças nossas e os patches.

**104 dos 196 scripts soltos na raiz do cluster.** Pertencem a outras linhas de
trabalho ou são rascunho de uma tentativa só:

| Grupo | Arquivos | Por quê |
|---|---|---|
| **Bokeh / BokehMe / RealBokeh** | `analise_par.py`, `cadeia_rb.sh`, `cmp_piloto.py`, `curar_resultados_bokeh.py`, `diag_210.py`, `diag_foco.py`, `diag_img.py`, `diag_meta.py`, `faixa_k.py`, `k_stats.py`, `mosaico.py`, `nitidez.py`, `rb_levels.py`, `rb_pairs.py`, `rb_test.py`, `rb_test2.py`, `relancar_Blinha.sh`, `rotac_src.py`, `smoke_scene.py`, `tabela_final.py`, `testa_variantes.py`, `valida.py`, `valida_bokehme.py`, `verif_av.py`, `metricas_sub.py`, `pipeline_avaliacao.sh`, `encadeia_tabela.sh`, `roda_porimg.sh`, `roda_busca.sh`, `roda.sh`, `rodar_antigo.sh`, `sobe_por_imagem.py`, `resume_riemann.py` | Linha do BokehNet, não desta campanha. O `resume_riemann.py` tem "riemann" no nome mas lê `ri_eq4_gpu*/riemann_por_imagem.parquet`, que são as métricas riemannianas do **campo de desfoque** do bokeh. |
| **DeblurNet / GenRefocus** | `ler_deblur.py`, `lanca_semtreino.sh`, `test_birefnet.py`, `confere_final.py`, `confere_publico.py`, `readme_v2.py`, `subir_akcit.py`, `subir_hf.py`, `subir_juliadollis.py`, `sobe_faltantes.py`, `sobe_docs.py`, `verifica_hf.py`, `lista_repos.py`, `lista_repos2.py`, `lista_todos.py`, `inv_hub.py`, `consolida_final.sh`, `cria_repos.py` | Publicação e conferência dos repos de bokeh/deblur. |
| **EBB (Everything-Bokeh-Blur)** | `amostra_ebb.py`, `busca_ebb.py`, `insp_ebb.py`, `build_bench_ebb.py`, `republica_ebb.py`, `depois_da_noite_ebb.sh` | Outro dataset, outra linha. |
| **Eq. 4 / tabela 2 do paper de bokeh** | `cols_eq4.py`, `esq_tab2.py`, `ler_tab2.py`, `tab2_estado.py`, `monta_listas_eq4.py`, `separa_eq4.py`, `sobe_eq4.py`, `valida_eq4.py`, `cobertura.py` | Campanha da Eq. 4, independente desta. |
| **LFDOF / BLB / k-escala** | `conta_kesc.py`, `conta_lfrepro.py`, `conta_tudo_kesc.py`, `le_kesc.py`, `le_kesc2.py`, `esquema_blb.py`, `inspeciona_lfdof.py`, `ler_curados.py`, `prim40.py`, `bench40.py`, `build_bench.py`, `build_bench_v2.py` | Datasets de outras rotas. |
| **Rascunhos de inspeção de uma tentativa** | `insp2.py`, `insp3.py`, `insp4.py`, `insp5.py`, `insp_dup.py`, `insp_ids.py`, `insp_meta.py`, `insp_rb.py`, `inspect_ds.py`, `busca_ds.py`, `busca_rb.py`, `keys.py`, `keys2.py`, `ler_bruto.py`, `ler_bruto2.py`, `checa_n.py`, `card.py`, `analisa_conf.py`, `ver_metricas.py`, `monta_listas.py`, `monta_listas2.py` | Um `print` de 10 linhas escrito para responder uma pergunta e nunca reusado. O que sobreviveu está em `avaliacao/` e `hub/`. |
| **Orgs de bokeh / destrutivos** | `cria_org.py`, `testa_org.py`, `limpa_teste.py` | `cria_org.py` e `testa_org.py` criam a org `akcit-pixel4` (bokeh); `limpa_teste.py` **apaga** um repo do Hub. Fora de propósito num repositório público. |
| **Encadeadores de outras cadeias** | `cadeia_gpu6.sh`, `depois.sh`, `roda_confere.sh`*, `roda_sonda.sh`*, `roda_ve.sh`*, `roda_salva.sh`*, `roda_orfaos.sh`*, `roda_confere_hub.sh`*, `sobe_orfaos.sh`* | `cadeia_gpu6.sh` e `depois.sh` emendam cadeias de bokeh. Os marcados com `*` **foram incluídos** em `hub/` — são os invólucros de container dos scripts de Hub desta campanha. |

Também ficaram fora da raiz os `.md` que não são desta campanha
(`AUDITORIA_AVALIACAO.md`, `TABELA_FINAL.md`, `TABELA_EQ4.md`,
`README_15k.md`, `README_60k.md`, `README_HF.md` — todos de bokeh/deblur).

**Dois patches saíram vazios.** `src/renderer/calibration.py` e
`src/qc/metrics.py` são hoje **byte a byte iguais** aos seus backups
`*.antes_caminho_a` — a modificação do caminho A foi revertida ou nunca chegou
a eles, e um `diff -u` não produziu nada. Não há backup de
`scripts/run_route_b.py`, então não existe diff a gerar; a versão atual já está
publicada em [`../bokehnet-regen/scripts/run_route_b.py`](../bokehnet-regen/scripts/run_route_b.py)
(conferido por md5 contra o cluster).
