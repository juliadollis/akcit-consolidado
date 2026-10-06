# Onde cada coisa fica registrada

Duas linhas de trabalho correm em paralelo nesta máquina, com naturezas
diferentes de resultado. Este documento diz onde cada coisa mora e por quê, para
não depender de ninguém lembrar.

Princípio que vale para as duas: **nada é sobrescrito.** Repositório novo a cada
campanha, e o bruto nunca é editado — o que se publica é uma camada curada
POR CIMA dele.

---

## 1. GenRefocus (bokeh) — resultados no Hugging Face

### 1.1 A camada bruta, que não se lê a olho nu

`juliadollis/bokeh-eval-metricas` é um **log de append**. Cada rodada anexa uma
linha. Ele contém, misturados e sem distinção no próprio arquivo:

- campanhas medidas com o pipeline **ainda bugado** (profundidade sem
  redimensionar, plano de foco pelo pixel central, `k-escala 0.01` matando o
  mapa de condicionamento);
- campanhas com o pipeline corrigido;
- linhas repetidas das voltas do keeper.

**Não publique nada a partir dele diretamente.** Ele existe como histórico e
para auditoria.

### 1.2 A camada curada, que é a publicável

`juliadollis/genrefocus-resultados-curados` — gerada por
`scripts/curar_resultados_bokeh.py` (no `julia_docker` da máquina).

Uma linha por (modelo, benchmark), só da campanha final, com:

| coluna | para quê |
|---|---|
| `modelo`, `step_do_checkpoint` | identifica o peso exato |
| `benchmark`, `n` | qual mesa e com quantas cenas |
| `SSIM`, `LPIPS`, `DISTS`, `CLIP_I`, `LVCorr` | as métricas |
| `margem_LPIPS_sobre_identidade` | **a única comparação válida entre benchmarks** |
| `faixa_K`, `resolucao`, `pipeline` | as condições de medida |
| `observacao_benchmark` | o que aquela mesa mede e o que não mede |
| `repo_metricas_bruto`, `id_interno` | rastro de volta para o bruto |

A coluna da margem existe porque **os pisos dos três benchmarks são muito
diferentes**: a linha de identidade dá LPIPS 0,2371 no LF-repro e 0,3587 no
RealBokeh. Comparar LPIPS absoluto entre mesas não significa nada; o que se
compara é a distância até o piso de cada uma.

Para regerar depois de novas rodadas:

```bash
docker run --rm --user $(id -u):$(id -g) \
  -v $B/hf-cache-julia:/workspace/hf-cache -v $B:/host \
  -e HF_HOME=/workspace/hf-cache -e HOME=/workspace/hf-cache/home \
  -e HF_TOKEN=... -w /host julia-genrefocus-eval:2.0 \
  python3 curar_resultados_bokeh.py juliadollis/genrefocus-resultados-curados
```

### 1.3 Imagens geradas e pesos

- Imagens de cada rodada: um repo `juliadollis/bokeh-eval-*` por experimento.
- LoRAs: `juliadollis/genrefocus-bokehnet-*`, com checkpoint por step.
- Benchmarks: `juliadollis/bokeh-bench-realbokeh-test-v2`,
  `juliadollis/lf-bokeh-repro-blb`, `akcit-pixel/RealDOF`.

---

## 2. Reteste da curvatura (depth-riemannian) — resultados no repositório

Aqui o resultado é pequeno (JSONs de algumas dezenas de KB) e pertence ao código
que o gerou, então mora no repo, versionado, e **não** no Hugging Face.

```
depth-riemannian/
  resultados/reteste_curvatura_<data>/
    B3_gauss_metrica_teto1000/seed_{0,1,2}/{summary,test_metrics,history}.json
    B0_berhu/seed_{0,1,2}/...
    logs/passo4_b{0,3}.log, prep_passo4.log
    RESULTADOS.md          <- GERADO, nunca editado a mão
  MELHORIAS_RETESTE.md     <- o que discutir com o José Ricardo
  scripts/consolida_reteste.py
```

`RESULTADOS.md` sai de:

```bash
python3 scripts/consolida_reteste.py resultados/reteste_curvatura_<data> \
  | tee resultados/reteste_curvatura_<data>/RESULTADOS.md
```

O consolidador lê os JSONs e monta média, amplitude e a comparação com o
controle. **Ele se recusa a comparar quando os dois lados têm número diferente
de seeds** — imprime um aviso em vez de um número enganoso. É de propósito:
transcrever métrica à mão foi como uma tabela com faixas de K misturadas quase
foi publicada nesta reprodução.

Os **checkpoints (6,4 GB)** ficam em `/raid/user_juliadollis/julia_docker/runs_riemann`
na máquina. Não vão para o repo nem para o HF por enquanto — se algum virar
resultado a defender, aí sim sobe para um repo HF próprio.

---

## 3. O que ainda não está resolvido

- **Procedência da rota a.** O pipeline gravou só um UUID como nome, sem caminho
  de origem. Consequência permanente: **não conseguimos certificar que um
  benchmark qualquer está fora do treino da fase 1.** Qualquer paper nosso
  carrega essa ressalva.
- **Convenção de sinal do LVCorr.** Fixamos Pearson(K, variância do Laplaciano
  no fundo), sinal cru. O avaliador do projeto usa a imagem INTEIRA. Os dois
  números não são a mesma coisa e não podem ir na mesma coluna.
- **O `n` do RealDOF** ainda entra como nulo na tabela curada; falta ler o
  tamanho real do split.
