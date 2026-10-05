# Auditoria do pipeline e da ligação ao treino — 22/09/2026

Veredito: não é possível aprovar como 100% correto ou como reprodução integral do paper. Há lacunas concretas de configuração e validação. Esta é uma auditoria local; não certifica um job em execução no cluster.

## Escopo e evidência

A pasta encontrada foi `bokehnet-regen` (não `bokeh-regen`). Ela gera os dados; o consumidor localizado foi `retreinar-bokeh`. O texto extraído novamente de `paper.pdf` com `pdftotext -layout` foi idêntico a `reference/paper.txt`: SHA-1 25d9e1f91ee728f85e806db471948e6d848ef2d8. Conferidas as seções 3.2, 3.3, 4.1 e suplemento B.2; inspeção visual da página 7 para as equações 3/4 e o refinamento de foco.

## Achados

### 1. Alta prioridade — filtro obrigatório de SSIM pode ficar ausente de ponta a ponta

O paper §3.2(c), Eq. 5, exige filtrar a calibração por um limiar de SSIM. O valor não é publicado e deve ser calibrado, não inventado.

- `src/routes/route_c.py:147`: limiar default `None`.
- `slurm/route_c_shards.slurm:83-90`: geração em shards não passa `--min-calibration-ssim`.
- `src/qc/gates.py:81`: limiar ausente aprova o gate em modo medir.
- `src/dataio/sample.py:315`: `is_valid_for_control` depende do flag e da censura, não exige limiar de SSIM congelado.
- `../retreinar-bokeh/configs/train_bokeh_fase2_real.yaml:168`: filtro de SSIM também `null` no consumidor.

Reprodução local: `calibration_ssim_is_reliable(0.01, min_ssim=None).to_dict()` retornou `passed=True`, `measured_only=True`. Isso comprova o comportamento do gate; não significa que todas as amostras reais tenham SSIM baixo ou que nenhum filtro externo tenha sido aplicado.

Gerar um lote bruto para calibrar é legítimo. Utilizá-lo como supervisão final sem filtragem posterior não cumpre esse passo do paper. Correção necessária: definir limiar com dados reais, aplicá-lo antes do treino e exigir evidência dessa filtragem para a rota C.

### 2. Alta prioridade operacional — configuração local da fase 2 está incompleta

`../retreinar-bokeh/configs/train_bokeh_fase2_real.yaml:113-116` contém `PREENCHER/release-rota-b` e `PREENCHER/release-rota-c`. Esses valores não identificam os releases destinados ao treinamento. Há ainda destinos de upload por preencher. O arquivo é um template, não uma configuração pronta para certificar o treino. Um override externo pode resolver isso, mas não foi apresentado nem inspecionado.

### 3. Prioridade média — monitor de controlabilidade solicitado é desligado

O YAML da fase 2 pede probe a cada 5.000 passos, mas deixa `probe_set_dir: null`. Em `../retreinar-bokeh/genfocus_train/trainer.py:1597-1612`, a falta da pasta emite aviso e zera `probe_every`; falha ao carregar o conjunto também desliga o probe. O YAML de fase 1 para h100n1 já tem `probe_every_steps: 0`.

Isso não torna a equação da loss errada e não é requisito explícito do paper. Porém, permite treinar sem medir a resposta ao controle K. A loss de flow matching isolada não comprova controlabilidade. É necessário um conjunto fixo e medições reais, além de loss de validação.

### 4. Prioridade média — proveniência omite parâmetro que altera o foco

`src/routes/route_c.py:457` usa `config.focus_min_detail_percentile` para selecionar a região de foco. `_focus_provenance`, linhas 482-498, não o registra.

Reprodução local: configurações com percentis 20 e 80 produziram dicionários de proveniência de refinamento idênticos. O parâmetro pode alterar máscara, foco e K calibrado; um registro isolado da amostra não permite reconstruir essa decisão. Registrar esse parâmetro junto aos demais, preservando a identidade das versões existentes.

### 5. Limitação científica relevante — qualidade métrica do foco ainda não está demonstrada para o lote final

O refinamento automático substitui o refinamento manual do paper. É uma variante metodológica, não reprodução literal nem, por si só, um bug.

O registro local mais recente consultado contradiz a afirmação antiga de que a validação nunca rodou: `REGISTRO.md:1246-1287` relata um piloto de 302 amostras. Nos 88 casos refinados com baseline, a concordância dentro de ±25% melhora de 19,3% para 23,9%. O registro relata 32,1% no conjunto completo e 33,1% dos gabaritos fora da faixa de disparidade estimada da cena.

Esses números são evidência histórica relatada pelo projeto, não medidas refeitas nesta auditoria nem taxas do lote final. Eles impedem concluir que máscaras e profundidades estão corretas apenas porque os testes passam. Exigem validação no release final, com estratificação por fonte e por tipo de refinamento.

## Pontos consistentes na inspeção

- Eq. 3 com conversão metros/milímetros e `k_eq3 / 1000` em `src/control/contract.py`.
- Controle em disparidade métrica, mediana da disparidade na máscara e normalizador global 100, coerentes com a cópia local da inferência oficial. O paper chama D de profundidade; essa distinção já é documentada pelo projeto.
- Rota C calibra K por SSIM com renderer; o K analítico permanece diagnóstico.
- Consumidor contém reescala de K e transformação alinhada de imagens/disparidade, em `prepare_aligned_bokeh_metric`.
- Configurações principais declaram LoRA 64 e currículo 40K sintético/60K real, batch por GPU 1 e acumulação 8, conforme §4.1. Batch efetivo 32 depende de executar de fato em 4 GPUs; não foi medido um run.
- Loop inspecionado usa flow matching e sincronização manual de gradientes no caso multi-GPU. Inspeção estática não certifica execução distribuída.

## Testes e limites

Executado em `bokehnet-regen` com o Python do runtime Codex:

```
PYTHONPATH=src <python-do-runtime> -m unittest discover -s tests -p 'test_*.py'
Ran 871 tests in 16.982s
OK (skipped=16)
```

São 855 testes executados sem falha e 16 ignorados. Parte dos testes usa modelos falsos e renderer sintético; não equivale a smoke com FLUX, Depth Pro, BiRefNet e BokehMe reais.

O Python padrão falhou por ausência de NumPy; o runtime resolveu essa limitação para a geração. A suíte do consumidor não foi validada: faltam pytest e PyYAML no runtime utilizado. Não foram instaladas dependências nem iniciado treinamento pesado.

Não foram verificados nesta auditoria: releases finais (não presentes na pasta local de output), checkpoints de treino, curvas atuais, jobs do cluster, execução CUDA/distribuída ou métricas de inferência do modelo treinado. O output local contém apenas o relatório de verificação do renderer.

## Para aprovar uma execução concreta

1. Resolver o YAML efetivamente usado, releases e checkpoint de inicialização da fase 1.
2. Calibrar/aplicar SSIM na rota C e conferir metadados, splits por cena e reconstrução do controle no release final.
3. Completar proveniência do refinamento e validar foco/profundidade no lote final.
4. Rodar a suíte do consumidor e smoke com modelos reais: loss finita, gradientes LoRA, atualização dos pesos, checkpoint e retomada.
5. Validar controlabilidade e qualidade em cenas separadas, com K variável, no treino real.

Único arquivo criado nesta auditoria: este relatório. Código, configs e datasets não foram modificados.
