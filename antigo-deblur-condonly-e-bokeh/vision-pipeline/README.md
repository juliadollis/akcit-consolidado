# Vision Pipeline

Repo único para inferência e avaliação

## Estrutura

- `inference/deblur_net.py` gera datasets Hugging Face com `image_generated` e `image_focus` e dispara a avaliação em seguida.
- `evaluation/eval_deblur_net.py` rodar a etapa de avaliação, quando necessário.
- `inference/bokeh_net.py` é um pipeline local com saída em PNG.
- `evaluation/eval_bokeh_synthesis.py` espera o dataset Hugging Face já estruturado para bokeh.
- Os scripts em `inference/*.py` são entrypoints finos; implementação fica em `inference/src/`.

## Preparação

1. Crie `.env` na raiz, por exemplo com `cp .env.example .env`, e preencha:

```text
HF_TOKEN=...
DEBLUR_EVAL_OUTPUT_REPO=AkcitPixel2/ResultadosInferencia
DEBLUR_MODEL_NAME=DeblurNet
```

1. Instale dependências:

```bash
pip install -r requirements.txt
```

1. Baixe pesos e valide o ambiente:

```bash
python inference/prepare_environment.py
```

## Uso

### Inferência de deblur

```bash
python inference/deblur_net.py --experiment_name meu_experimento
```

Esse comando agora executa o fluxo completo de deblur: inferência nos datasets configurados em `inference/src/pipelines/deblur_net.py` e avaliação automática logo após cada upload gerado.

O argumento `--experiment_name` (opcional, padrão `Unknown`) nomeia a execução e é gravado na coluna `Experiment` de cada linha do `results.csv` enviado ao repositório em `DEBLUR_EVAL_OUTPUT_REPO`.

### Inferência de bokeh

```bash
python inference/bokeh_net.py
```

### Avaliação de bokeh

```bash
python evaluation/eval_bokeh_synthesis.py \
  --hf_dataset seu-usuario/dataset-bokeh \
  --model_name BokehNet \
  --output_hf_repo seu-usuario/historico-bokeh \
  --hf_split validation
```

## Docker

Build:

```bash
docker compose build
```

Container interativo:

```bash
docker compose run --rm vision-pipeline bash
```

Dentro do container, rode os mesmos comandos acima a partir de `/workspace`.
