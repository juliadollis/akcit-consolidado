# `scripts/` — medições e inferência

> **Inferência e avaliação mudaram de pasta.** `infer_deblur.py`,
> `avaliar_deblur.py` e `test_c6_monkeypatch.py` agora vivem em
> [`../inferencia/`](../inferencia/), junto do `rodar_tabela2.py`, do
> `metricas.py` e do `PROTOCOLO.md`. O que ficou aqui é só diagnóstico de dados
> e de ambiente.


Utilitários da árvore `retreinar-deblur/`. Nenhum deles treina, e **nenhum
apaga nada**: só leem, medem e escrevem em `outputs/`.

Todos aceitam `--help`. O `HF_TOKEN` sai do ambiente ou do `.env` da raiz
(`.env.example` tem o modelo) e nunca é impresso.

## Tabela

| Script | O que responde | GPU? | Rede? | Tempo |
|---|---|---|---|---|
| `c0_1_resolucao_dfs.py` | Em que resolução os dfs estão **realmente** gravados? Recalcula a tabela de regimes do C4 com o valor medido. | não | sim (streaming) | ~1 min |
| `c0_2_gts_distintas.py` | Quantas GTs **distintas** o `top_k_sharpest` seleciona de fato? (hipótese: ~580 para 3000 linhas) | não | sim | minutos a ~1 h se recalcular |
| `c0_3_contagem_lora.py` | Quantos módulos LoRA são injetados — 343 (oficial) ou 344 (com o `proj_out` de topo)? | não no modo estrutura; sim com `--pesos-reais` | sim (config; pesos só com `--pesos-reais`) | segundos / minutos |
| `verificar_mu.py` | A fórmula do `mu` bate com o `calculate_shift` do diffusers? Qual o cronograma de sigma de cada regime? | não | não | instantâneo |
| `prefetch_datasets.py` | Pré-aquece o cache: baixa DDPD (9,6 GB) + RealBokeh (46,9 GB) e popula o cache do filtro de nitidez, para o job de GPU não ficar ocioso baixando. | não | **sim, ~56 GB** | ~40 min + ~7 min do filtro |
| `_comum.py` | Não é executável: helpers compartilhados (carregar `.env`, reta do `mu`, formatação). | — | — | — |

## Ordem sugerida

```bash
# 1. os fatos que o plano precisa medir antes de decidir qualquer coisa
python3 scripts/c0_1_resolucao_dfs.py -n 50
python3 scripts/verificar_mu.py                    # já lê o JSON do c0_1
python3 scripts/c0_2_gts_distintas.py

# 2. prova do C1 (falha com código 1 se a contagem divergir de 343)
python3 scripts/c0_3_contagem_lora.py

# 3. checagem do desvio C6, sem GPU
python3 inferencia/test_c6_monkeypatch.py
```

## Os defaults do `infer_deblur.py` reproduzem o oficial

(o arquivo está em `../inferencia/infer_deblur.py`)

`long_side=0`, `steps=28`, `guidance_scale=3.5`, `main_adapter=none`,
`text_adapter` igual ao main, `seed=42`, tiling ligado quando `min(w,h) >= 512`.
Rodar sem trocar nada dá o mesmo que `third_party/Genfocus/Inference_deblurNet.py`.

Os eixos existem porque os experimentos do plano precisam variá-los:

| Flag | Item | Para quê |
|---|---|---|
| `--guidance-scale` | C2 | o deblur oficial não passa o valor e cai no default 3.5; o treino usava 1.0 |
| `--long-side` | C4 | `0` = nativo com tiling; `512` = reduz o lado maior (muda escala **e** `mu`) |
| `--main-adapter` | — | `none` = cond-only (oficial); `deblurring` = main+cond (nosso 60k) |
| `--text-adapter` | C6 | `none` tira o LoRA do texto, que é o que o treino main+cond fez |

## Sobre o `--text-adapter` (C6)

O `generate` oficial monta `adapters = [main_adapter] * 2 + c_adapters`, e o
índice 0 é o branch de **texto**, não o principal. Passar
`--main-adapter deblurring` liga LoRA no texto em 38 blocos single — o treino
nunca fez isso.

`third_party/Genfocus/` é clone de referência e não foi tocado. O desvio é um
**monkeypatch** de `Genfocus.pipeline.flux.transformer_forward`, aplicado só
quando `--text-adapter` difere do main. Verificado por AST que as três chamadas
dentro do `generate` (ramo com tiling, ramo sem, e ramo
`image_guidance_scale != 1.0`) resolvem a função como global do módulo e passam
`adapters`/`text_features` por palavra-chave — então uma única substituição
cobre as três. `test_c6_monkeypatch.py` cobre os 6 cenários, inclusive a
restauração após exceção.

Copiar o `flux.py` (882 linhas) foi descartado: divergiria em silêncio do
upstream na primeira atualização, e é justamente a divergência silenciosa entre
treino e inferência que originou este plano.

## Sobre o JSON de protocolo do `avaliar_deblur.py`

(o arquivo está em `../inferencia/avaliar_deblur.py`)

Registra resolução em que a métrica foi calculada, se e como o GT foi
redimensionado, `long_side`, `guidance_scale`, variante de adapter e o sha256 do
peso. Sem isso nenhuma tabela montada depois é interpretável: LPIPS não é
invariante a escala, então comparar com o número **publicado** exige conferir os
dois protocolos — enquanto comparar modelos **dentro** do mesmo JSON é justo.