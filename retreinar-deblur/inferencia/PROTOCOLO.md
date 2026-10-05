# Protocolo de avaliação — Tabela 2 (defocus deblurring)

Este documento existe para que a tabela seja **interpretável**. Sem ele, um
número ao lado de um número publicado sugere uma comparação que o protocolo não
sustenta.

Regra que o documento impõe: cada decisão abaixo está marcada como **[PAPER]**
(o paper determina), **[OFICIAL]** (o código de inferência dos autores
determina, o paper é silencioso) ou **[NOSSA]** (escolha nossa, o paper e o
código são silenciosos).

---

## 1. O alvo

Tabela 2 do arXiv 2512.16923v3. Valores publicados:

| | LPIPS ↓ | DISTS ↓ | CLIP-IQA ↑ | MANIQA ↑ | MUSIQ ↑ |
|---|---|---|---|---|---|
| **DPDD** Input | 0.3485 | 0.1827 | 0.4337 | 0.3325 | 45.5376 |
| **DPDD** GenRefocus | **0.1440** | **0.0772** | **0.4755** | **0.3452** | **49.4122** |
| **RealDOF** Input | 0.5241 | 0.2865 | 0.3562 | 0.2213 | 28.7087 |
| **RealDOF** GenRefocus | **0.2408** | **0.1126** | **0.4595** | **0.2884** | **43.5222** |

O paper traz também DRBNet, Restormer, INIKNet, Bokehlicious e DiffCamera. Não
os reproduzimos, então eles **não** aparecem na nossa tabela — listar baseline
não medida ao lado de medida é o caminho mais curto para alguém ler como se
tivesse sido medida.

**Por que só a Tabela 2.** As Tabelas 3 e 4 usam LF-Bokeh e LF-Refocus, que a
org `nycu-cplab` não publicou. A Tabela 2 é o único número do paper que pode ir
lado a lado com o nosso.

---

## 2. As mesas

| Mesa | Repo:split | n | Observação |
|---|---|---|---|
| DPDD | `akcit-pixel/DDPD:test` | 75 | **[NOSSA]** split de teste, o benchmark canônico do DPDD |
| RealDOF | `akcit-pixel/RealDOF:validation` | 50 | é o **único** split do repo; corresponde às 50 imagens de teste do RealDOF |

Contagens conferidas em `HANDOFF_PROJECT_HISTORY.md` seção 4.

Duas armadilhas de nome:

* O repo chama-se **`DDPD`**, com D e P trocados em relação ao `DPDD` do paper.
  Não é erro de digitação nosso.
* O RealDOF tem as 50 imagens sob o nome `validation`. O paper usa o conjunto de
  teste do RealDOF, que tem 50 imagens — então o split é o certo apesar do nome.

**[NOSSA] e a verificar:** o paper diz apenas *"We conduct deblurring experiments
on RealDOF and DPDD datasets"*, sem nomear o split do DPDD. Escolhemos `test`
por ser o benchmark canônico. O `run_tabela2.py` antigo do time usava
`validation-*.parquet` para as DUAS mesas, ou seja, avaliava o DPDD no split de
**validação** (73 imagens) — divergência que vale reconciliar antes de comparar
qualquer número antigo com os novos.

---

## 3. Variantes de métrica — a ressalva mais importante

O paper cita LPIPS [83], DISTS [16], CLIP-IQA [67], MANIQA [76] e MUSIQ [29]
**sem dizer qual implementação nem qual checkpoint**. O `pyiqa` oferece várias
para cada nome, e elas não são intercambiáveis.

| Métrica | Usamos **[NOSSA]** | Alternativa que o nome sugere | Mesma rede? |
|---|---|---|---|
| LPIPS | `lpips+` | `lpips` | **não** |
| DISTS | `dists` | `dists` | sim |
| CLIP-IQA | `clipiqa+` | `clipiqa` | **não** |
| MANIQA | `maniqa-kadid` | `maniqa` (KonIQ) | **não** (treinos diferentes) |
| MUSIQ | `musiq` | `musiq` | sim |

O default reproduz o pipeline do time
(`vision-pipeline/evaluation/src/deblur_evaluator.py:38-54`), para não criar um
terceiro protocolo no projeto. Mas **três das cinco são variantes "+", não as
versões base**, e essa escolha não foi confirmada contra o paper.

**Consequência:** os valores absolutos podem não ser comparáveis aos publicados
por esse motivo, de forma **independente** da resolução. Antes de afirmar
"batemos" ou "ficamos abaixo", rode os dois conjuntos:

```bash
python3 inferencia/rodar_tabela2.py --lora P.safetensors --variantes time --out s/time
python3 inferencia/rodar_tabela2.py --lora P.safetensors --variantes base --out s/base
```

Se a distância entre os dois conjuntos for da ordem da distância até o
publicado, a comparação com o paper não decide nada.

---

## 4. Resolução de avaliação

**[OFICIAL]** `long_side = 0`, que é o default do `Inference_deblurNet.py`. Nesse
caminho o `resize_and_pad_image` **arredonda para cima** ao múltiplo de 16 e
**não reduz** a imagem. Com `long_side > 0` o comportamento é outro: reduz o
lado maior, trunca a múltiplo de 16 e corta no centro.

**[NOSSA]** o ground-truth passa pelo **mesmo** `resize_and_pad_image` da
entrada. Se não passasse, predição e GT teriam shapes diferentes e a métrica
compararia coisas de tamanhos distintos.

**[A VERIFICAR]** o paper **não informa** em que resolução avaliou. Como LPIPS e
DISTS não são invariantes a escala, isto é suficiente para explicar diferença nos
valores absolutos. O que **é** válido sem essa informação: comparar modelos
entre si dentro deste pipeline, porque todos passam pelo mesmo protocolo.

Cada execução registra em `protocolo_e_resultado.json` as resoluções exatas em
que a métrica foi calculada. Se a lista tiver mais de um par, a média mistura
resoluções — o que é aceitável desde que seja igual para todas as linhas
comparadas, e está registrado para se poder checar.

---

## 5. Geração

| Parâmetro | Valor | Origem |
|---|---|---|
| `steps` | 28 | **[PAPER]** §4.1: *"each stage employs 28 denoising steps"* |
| `prompt` | `"a sharp photo with everything in focus"` | **[OFICIAL]** `Inference_deblurNet.py` |
| `guidance_scale` | 3.5 | **[OFICIAL]** é o default de `generate` (`flux.py:467`); o deblur oficial **não passa** este valor |
| `main_adapter` | `None` (cond-only) | **[OFICIAL]** `Inference_deblurNet.py` não passa, então fica `None` |
| tiling | ligado quando `min(w,h) >= 512` | **[OFICIAL]** `force_no_tile = min(w, h) < 512` |
| `seed` | 42 | **[OFICIAL]** `seed_everything(42)` |
| `position_delta` | `[0, 0]`, `position_scale` 1.0 | **[OFICIAL]** |

### O caso do peso main+cond (C6)

O `generate` oficial monta, nas três chamadas a `transformer_forward`:

```python
adapters = [main_adapter] * 2 + c_adapters
```

e em `transformer_forward` o índice **0 é o branch de TEXTO**, não o principal.
Então `--main-adapter deblurring` — necessário para o peso main+cond do projeto —
liga LoRA no texto em 38 blocos single, coisa que o treino **nunca** fez.

Para avaliar esse peso como ele foi treinado:

```bash
--main-adapter deblurring --text-adapter none
```

O `--text-adapter` é um desvio **[NOSSO]** do upstream, implementado por
monkeypatch local (o `third_party/Genfocus/` é clone de referência e não é
modificado). Coberto por `test_c6_monkeypatch.py`, 6 cenários, roda sem GPU.

---

## 6. Defeitos encontrados no pipeline do time

Achados ao consolidar. Nenhum foi "corrigido no lugar" — o código do time não
foi alterado; estes scripts simplesmente não os repetem.

### 6.1 Redimensionamento silencioso do GT — falsifica a métrica

`vision-pipeline/evaluation/src/core/evaluator.py:60-78`: quando os shapes
divergem, o maior é reduzido por `F.interpolate(mode="bicubic")` **sem aviso**.
Uma métrica calculada entre uma predição nativa e um GT reamostrado mede também
o reamostramento. Aqui shape divergente **aborta** com mensagem.

### 6.2 Métrica que falha vira 0.0 — melhora a média em silêncio

Mesmo arquivo, linhas 47-52: exceção numa métrica é registrada como `0.0`. Para
LPIPS e DISTS, onde **menor é melhor**, um erro vira nota **perfeita**. Aqui um
erro devolve `None`, é contado, e o nº de falhas aparece no resumo e no JSON.

### 6.3 Média dividida pelo total, não pelos válidos

`core/evaluator.py:113` faz `total_score / num_pairs`, mas pares que lançaram
exceção foram descartados com `continue` (linha 105) sem decrementar
`num_pairs`. A média fica enviesada **para baixo** — o que, em LPIPS, parece
melhora. Aqui a média é sobre os pares válidos, com `n_validos` e `n_falhas`
registrados.

### 6.4 O split do DPDD

`genrefocus_deblurnet_paper/vision-pipeline/run_tabela2.py` usa
`validation_pattern="data/validation-*.parquet"` para as duas mesas, avaliando o
DPDD em `validation` (73) em vez de `test` (75). Ver seção 2.

### 6.5 Variante diferente entre avaliadores

O avaliador de **deblur** usa `lpips+`; o de **bokeh** usa `lpips` simples. São
redes diferentes. Isso já está anotado no `run_tabela2.py` do time, mas vale
repetir: números de deblur e de bokeh do projeto nunca foram comparáveis entre
si, e nenhum dos dois é automaticamente comparável ao publicado.

---

## 7. O que conferir antes de publicar qualquer comparação

1. Rodar `--variantes time` e `--variantes base` e ver se a distância entre eles
   é menor que a distância até o publicado. **Se não for, a comparação com o
   paper não decide nada.**
2. Registrar as resoluções de `protocolo_e_resultado.json`. Se variarem entre as
   linhas comparadas, a comparação está contaminada.
3. Conferir `n_falhas == 0` em todas as métricas de todas as linhas.
4. Conferir que a linha `Input` medida fica perto da `Input` publicada. **Esse é
   o melhor teste de protocolo que existe**: a linha Input não depende de modelo
   nenhum, só do dado e da métrica. Se a nossa `Input` não bater com a publicada,
   a divergência está no protocolo — não no nosso modelo — e comparar a linha do
   modelo é inútil até resolver isso.
5. Para peso main+cond, conferir que `text_adapter` é `none` e não
   `igual_ao_main`.
6. Conferir o `sha256` do peso no JSON contra o artefato que se pretende
   reportar.

O item 4 é o mais barato e o mais informativo. Ele não precisa de GPU para o
modelo — só roda as métricas — e diz se o resto da tabela significa algo.
