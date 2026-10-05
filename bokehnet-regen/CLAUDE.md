# bokehnet-regen

Repositório de trabalho da regeração dos dados da BokehNet. **O código do pipeline
vive aqui**, em `src/`. Nada é editado em `../bokehnet-preprocessing` — aquele
repositório é referência histórica e fonte dos defeitos catalogados.

## Cluster — inegociável

1. **NUNCA excluir job.** Nem `scancel` em job seu, nem de terceiros.
2. **NUNCA apagar nada no cluster sem perguntar antes** — checkpoints, caches, logs,
   datasets, diretórios "temporários". Já houve um `rsync --delete` que apagou checkpoints.
3. **`--time` sempre bem alto** (ex. `7-00:00:00`).
4. Segredos vêm de `.env` git-ignored. Nunca hardcodar, nunca imprimir token.
5. Antes de escrever em qualquer lugar compartilhado (HF, pasta de colega): perguntar,
   e preferir **criar destino novo** a sobrescrever.
6. Avisar sempre quais arquivos mudaram localmente, para o rsync.

QOS `onejob` permite 2 jobs simultâneos. GPU se escolhe **por UUID, nunca por índice**
(a GPU4 da h100n3 tem defeito e contamina o stdout do `nvidia-smi`). `/raid` é local
por nó; `/home` é compartilhado. Detalhes em `../INSTRUCOES_H100.md`.

## O contrato

`reference/CONTRATO.md` é a única definição válida do sinal de controle. Leia antes de
tocar em qualquer coisa que envolva K, profundidade, disparidade, CoC ou resolução.

```python
z          = depth_pro(aif)              # METROS
disp       = 1.0 / z                     # 1/m
focus_disp = median(disp[mask])          # NA disparidade, não 1/median(z)
K          = k_eq3 / 1000.0              # Eq. 3 é em mm; a inferência é em 1/m
max_coc    = 100.0                       # congelado, global
defocus    = clip(abs(K*(disp - focus_disp)) / max_coc, 0.0, 1.0)
```

**Uma implementação só**, em `src/control/`, importada por geração, dataloader,
avaliação e inferência. Cópias divergem — foi assim que chegamos a quatro
interpretações de K.

## Regras de código

- **Sem fallback numérico.** Faltou EXIF, faltou sensor, falhou o modelo: levanta
  exceção, registra o motivo, rejeita a amostra. Nunca substitui por constante.
- **Todo run termina imprimindo histograma de motivos de rejeição.** Sem ele não dá
  para calibrar limiar nenhum, e é ele que denuncia fallback novo.
- **Proveniência por amostra**: commit, `control_version`, e hash de *todo* modelo que
  influenciou o rótulo — DeblurNet, DepthPro, **BiRefNet**, e commit do BokehMe mais
  hashes de `arnet.pth`/`iunet.pth`. Mais seed e resolução processada.
- **Toda quantidade em pixel carrega a resolução em que foi medida.**
- **Split por cena, materializado no dataset**, nunca deixado para o config.
- **Contagens em cenas E em amostras.** 20.554 amostras de 3.960 cenas não são 20.554
  unidades de diversidade.
- `pip install --target` em pasta do projeto. **Nunca** no ambiente do usuário nem no
  `~/.local`, que é compartilhado entre nós.

## Antes de rodar qualquer coisa

Smoke de 2 amostras primeiro. `compileall` e parsing AST não pegam `NameError` — foi
assim que um `import json` removido matou um pipeline na primeira amostra depois de
passar por revisão.

## Os agentes

Cinco revisores em `.claude/agents/`, cada um mapeado a uma classe de defeito real
deste projeto. Chame por nome com a Task tool.

| agente | quando |
|---|---|
| `paper-fidelity` | antes de implementar equação, escolher fonte de dado, fixar hiperparâmetro |
| `unit-contract` | toda mudança que encoste em K, depth, disparidade, CoC ou resolução |
| `fallback-hunter` | todo código novo antes de rodar; e quando uma taxa de sucesso parecer boa demais |
| `data-contract` | antes de publicar release, depois de mexer no writer, em todo repack |
| `evidence-auditor` | antes de decidir com base em "acho que o dataset tem X" |

`reference/ACHADOS.md` guarda tudo que já foi **medido**, com etiqueta `[M]` medido,
`[I]` inferido, `[A]` assumido. Cite de lá em vez de re-medir. Acrescente linha quando
medir coisa nova. Nunca promova `[A]` a `[M]` sem a medição.

## Layout

```
src/control/    o contrato — implementação única de K, disparidade, defocus
src/routes/     route_a.py, route_b.py, route_c.py
src/renderer/   BokehMe in-process (modelo carregado uma vez) + busca ternária
src/qc/         gates, fila de revisão, registro de par
src/dataio/     writers, manifestos, split por cena
tests/          testes de invariante — raio do renderer, razão de f-stops, pós-crop
scripts/        entrypoints
manifests/      manifestos versionados de fonte e split
reference/      paper.txt, CONTRATO.md, ACHADOS.md
```
