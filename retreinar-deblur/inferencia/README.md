# `inferencia/` — reproduzir a Tabela 2 do paper

Inferência e avaliação da DeblurNet. Consolida o código do time
(`vision-pipeline/`, `deblurnet-eval-pipeline/`) num caminho único, com o
protocolo registrado em cada execução.

**Leia o [`PROTOCOLO.md`](PROTOCOLO.md) antes de comparar qualquer número com o
publicado.** O paper não informa a resolução de avaliação nem a variante de cada
métrica, e as duas coisas mudam os valores absolutos.

## Os três comandos

No cluster, dentro do container. `$P` = raiz do projeto, `$W` = peso a avaliar.

```bash
# 0) a linha Input — identidade, sem modelo, sem GPU de inferência.
#    É o teste de protocolo mais barato: se a nossa Input não bate com a
#    publicada, o problema é o protocolo, não o modelo.
python3 inferencia/avaliar_deblur.py --identidade \
    --dataset akcit-pixel/DDPD --split test --out saidas/tab2/dpdd/input

# 1) a tabela inteira (as duas mesas, linha Input + a nossa)
python3 inferencia/rodar_tabela2.py --lora "$W" --out saidas/tab2

# 2) a análise de sensibilidade de variante (ver PROTOCOLO.md §3)
python3 inferencia/rodar_tabela2.py --lora "$W" --variantes base --out saidas/tab2_base
```

Saídas: `saidas/tab2/TABELA2.md`, `saidas/tab2/tabela2.json` e, por mesa e linha,
`protocolo_e_resultado.json` + `por_imagem.json`.

## Antes da rodada longa

```bash
# 5 imagens de uma mesa, para validar o caminho sem gastar ~1h45 de GPU
python3 inferencia/rodar_tabela2.py --lora "$W" --mesa dpdd -n 5 --out /tmp/smoke

# o desvio C6 do --text-adapter, sem GPU
python3 inferencia/test_c6_monkeypatch.py

# quais variantes de métrica estão em uso
python3 inferencia/metricas.py
```

## Peso main+cond

O peso de 60k do projeto (`juliadollis/genrefocus-deblurnet-paper-4gpu`) é a
variante **main+cond**, apesar do nome do repo. Ele exige `--main-adapter`, e o
branch de texto tem de ficar **fora** do adapter, porque o treino nunca o
treinou (ver `PROTOCOLO.md` §5):

```bash
python3 inferencia/rodar_tabela2.py --lora nosso.safetensors \
    --main-adapter deblurring --text-adapter none --out saidas/tab2_maincond
```

Com `--main-adapter none` (o default oficial) a saída desse peso sai **lavada**.

## Os arquivos

| Arquivo | O que é | GPU? |
|---|---|---|
| `rodar_tabela2.py` | orquestra as duas mesas e emite `TABELA2.md` | sim |
| `avaliar_deblur.py` | inferência + as 5 métricas + protocolo em JSON | sim (não com `--identidade`) |
| `infer_deblur.py` | só inferência, todos os eixos por CLI | sim |
| `metricas.py` | as 5 métricas, variante configurável | sim |
| `test_c6_monkeypatch.py` | testa o desvio do `--text-adapter` | não |
| `_bootstrap.py` | `sys.path` + helpers (reexporta `scripts/_comum.py`) | não |
| `PROTOCOLO.md` | o que torna a tabela interpretável | — |

## Dependência

`pyiqa` fornece as 5 métricas e **não** está no container. Instale no projeto,
nunca no `~/.local` (ver `INSTRUCOES_H100.md`):

```bash
pip install --target "$P/.pydeps" pyiqa
export PYTHONPATH="$P/.pydeps${PYTHONPATH:+:$PYTHONPATH}"
```

## O que este pipeline não faz

As Tabelas 3 (bokeh, LF-Bokeh) e 4 (refocusing, LF-Refocus) do paper usam
datasets que os autores não publicaram. Só a Tabela 2 é reproduzível.
