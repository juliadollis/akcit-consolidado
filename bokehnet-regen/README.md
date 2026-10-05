# bokehnet-regen

Regeração dos dados de treino da BokehNet, com os nove agentes de revisão que
impedem a repetição dos defeitos catalogados.

O código do pipeline vive em `src/`. O `../bokehnet-preprocessing` fica intocado, como
referência histórica.

## Por que existe

O pré-processamento anterior construía o mapa de defocus em **profundidade linear
normalizada por imagem**; a inferência oficial constrói em **disparidade métrica
absoluta**. As duas fórmulas se parecem e não são a mesma. Toda a fase 2 foi treinada
assim, e a controlabilidade medida caiu de **+0,91** (fase 1 sintética) para **+0,44**
(fase 2 nas rotas reais), degradando monotonicamente com o tempo de treino.

Consertar só o `K` da rota B já recupera o LF-Bokeh de +0,4365 para +0,8288, colando
nos pesos oficiais (+0,8868). É a prova direta de que o rótulo era o gargalo — não a
arquitetura, não os steps, não a aparência.

O inventário completo dos 33 defeitos e o plano por rota estão em
`../genrefocus_deblurnet_paper/PLANO_REGERACAO_BOKEHNET.txt`.

## Os nove agentes

Cada um cobre uma classe de defeito que já passou por revisão humana neste projeto.
Não são revisores genéricos: carregam os números e as assinaturas específicas.

### `paper-fidelity`
Confere afirmação ou código contra o paper, citando seção, equação ou figura. Separa
**o paper diz** de **o paper cala** de **decisão nossa**, e recusa preencher silêncio
com suposição. Pega: `[80]` lido como DiffCamera quando é Generative Photography;
`pixel_ratio` com a largura quando a Fig. 16 diz *largest edge*; média onde a Eq. 4
pede mediana; achar que a extensão privada do renderer afeta a Eq. 5.

### `unit-contract`
Análise dimensional de tudo que toque em K, profundidade, disparidade, CoC ou
resolução. Escreve a unidade de cada termo e verifica o cancelamento. Pega: `k_eq3` em
milímetros alimentado com disparidade em 1/m (fator 1000); `max_coc` por fonte; o
fator `512/min(H,W)` do crop que ninguém aplica; raio contra diâmetro.

### `fallback-hunter`
Caça todo caminho em que o pipeline substitui um valor que falhou por um default sem
levantar erro e sem gravar o que aconteceu. Pega: `k = 50,0` em 11.635 de 11.635
amostras; o `except: pass` que troca DepthPro por Depth Anything e **inverte o sinal da
profundidade** sem deixar rastro; `main_adapter=None` que zera o LoRA principal em
silêncio; `mask_source="automatic"` mentindo depois do GrabCut.

### `data-contract`
Audita o artefato, não o código. Schema, escalares que reconstroem o controle,
round-trip sem perda de coluna, proveniência completa e verdadeira, censura marcada e
respeitada, split por cena materializado. Pega: `dm/dm.max()` num bloco de
visualização que virou o dado; `pack_files_to_hf` largando `focus_depth_m`.

### `evidence-auditor`
Mede em vez de acreditar, etiquetando `[M]` medido, `[I]` inferido, `[A]` assumido.
Lê **68 KB de um shard de 470 MB** com HTTP Range e projeção de coluna. Já refutou
"as 1.028 cenas sumiram por falha do parser" (o parser casa com 20.554/20.554) e
qualificou "`focal_length_35` em 100%" (verdade — mas 30,33% com crop factor
exatamente 1,0).

### `renderer-verifier`
O contrato do BokehMe: disco e não gaussiana, e raio igual a `K·Δdisp` a 2%. Risco alto
porque o pipeline antigo **nunca instalou o BokehMe** — caía sempre num gaussiano com
kernel travado em 51 px, que renderizou ~70K alvos da rota A e calibrou os K da rota C.

### `runtime-smoke`
Responde só "isto roda?". Classe própria porque `compileall` e parsing AST não pegam
`NameError` — foi assim que um `import json` removido matou um pipeline na primeira
amostra depois de passar por revisão.

### `cluster-safety`
A única classe de dano sem desfazer. Reprova `scancel`, `rm -rf`, `rsync --delete` sem
excludes, `delete_folder` no HF, token em log, `--time` baixo. Já houve um
`rsync --delete` que apagou checkpoints.

### `defect-regression`
Percorre os 33 defeitos e verifica se fecharam **e** se não abriram novos. Existe
porque correção parcial já se mostrou pior que nenhuma: o `kfix` consertou a rota B,
deixou a C na convenção antiga, e ficou com o pior LPIPS entre as variantes reais.

## Como usar

```
Use o agente unit-contract para revisar src/control/contract.py
Use o agente paper-fidelity para conferir se a Eq. 3 está implementada certa
Use o agente evidence-auditor para medir quantas cenas do LFDOF têm mais de 4 alvos
```

Antes de rodar qualquer coisa pesada, a ordem é: `fallback-hunter` no código novo →
`unit-contract` no que tocar no contrato → smoke de 2 amostras → `data-contract` na
saída do smoke.

## Layout

```
CLAUDE.md                regras do cluster, contrato, regras de código
reference/CONTRATO.md    a definição canônica do sinal de controle — uma página
reference/ACHADOS.md     tudo que foi medido, com [M]/[I]/[A] e origem
reference/paper.txt      texto extraído do paper, grepável (inclui o supplement)
.claude/agents/          os nove revisores
src/control/             implementação única de K, disparidade, defocus
src/routes/              route_a.py, route_b.py, route_c.py
src/renderer/            BokehMe in-process + busca ternária
src/qc/                  gates, fila de revisão, registro de par
src/dataio/              writers, manifestos, split por cena
tests/                   invariantes: raio do renderer, razão de f-stops, pós-crop
scripts/                 entrypoints
manifests/               manifestos versionados de fonte e split
```

## Estado

**Etapas 1 e 2 concluídas e verificadas.** 1.386 linhas em `src/`, 640 de teste,
**48 testes passando**.

| módulo | o que faz |
|---|---|
| `src/control/contract.py` | a fórmula canônica, implementação única |
| `src/qc/rejection.py` | o registro que substitui o fallback, com histograma |
| `src/qc/metrics.py` | SSIM da Eq. 5 e variância do Laplaciano, numpy puro |
| `src/renderer/verification.py` | o harness dos três testes do `renderer-verifier` |
| `src/renderer/calibration.py` | Eq. 5: bracket + seção áurea + censura |
| `src/renderer/bokehme.py` | adaptador in-process do BokehMe público |

Dos 33 defeitos: **13 fechados com teste**, 3 fechados no contrato aguardando as rotas,
17 abertos. Os testes já pegaram dois defeitos no próprio código novo — quantização
para uint8 que zerava a medição de raio, e um bug de broadcast no SSIM.

Um `[A]` fechou por medição: o `image_focus` do espelho `akcit-pixel/RealBokeh` é o
`train/in/<id>_f22.JPG` **byte a byte** (sha256), então a rota C pode consumi-lo direto
e D5 e D6 fecham juntos.

**Etapa 3: renderer verificado em GPU** (job 32212, h100n3, `COMPLETED`). Os três
testes passaram contra o BokehMe real — disco (`edge_width_ratio = 0,143`), linearidade
exata (`bokeh_classical` com resíduo **0,0000 px**) e escala (`slope = 0,9873`).
`is_final_label_renderer` deixou de ser declaração e virou medição, com proveniência
criptográfica: commit, sha256 do `pipeline` extraído, do `scatter.py` e dos dois
checkpoints.

**14 dos 33 defeitos fechados com teste**, 3 no contrato, 16 abertos.

Próximo: `src/routes/route_c.py`. Todos os bloqueadores caíram — renderer verificado,
espelho confirmado byte a byte, e o `metadata/` dá o f-number de cada `level`.
Detalhes e auditoria em `REGISTRO.md`.

## Como rodar os testes

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -p "test_*.py"
```
