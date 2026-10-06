# GenRefocus — DeblurNet (Stage 1)

Reprodução da **DeblurNet**, primeira etapa do paper *Generative Refocusing:
Flexible Defocus Control from a Single Image*. Backbone **FLUX.1-dev + LoRA
(rank 128)**, condicionamento OminiControl-style (`S_t = [X_t ; E(I_in)]`),
objetivo **rectified flow** (`v = ε - x_0`). Dados vêm do **Hugging Face**
(`akcit-pixel/*`).

Esta pasta junta:
- o **código de treino** fiel ao paper (FLUX real, rectified flow, forward dos
  autores via `Genfocus.pipeline.flux.transformer_forward`); e
- o **dataloader do colega** (Hugging Face `datasets`), adaptado para entregar
  o contrato que o backbone espera.

## Layout

```
genrefocus_deblurnet/
├── genfocus_train/
│   ├── backbone.py     # FluxBackbone + LoRA + sigma sampling (rectified flow)
│   ├── models.py       # DeblurNet (+ BokehNet scaffold p/ Stage 2) + loss
│   ├── data.py         # dataloader HF → {blurry_image, aif_image} em [-1,1]
│   ├── env.py          # HF_TOKEN via .env
│   ├── config.py       # schema (datasets HF + hiperparâmetros)
│   ├── math_utils.py   # helpers (usados só pela BokehNet/Stage 2)
│   ├── trainer.py      # loop, smoke test, checkpoint (LoRA only)
│   └── train.py        # CLI (smoke | deblur)
├── configs/
│   ├── train_base.yaml   # 60K steps, rank 128, 512², grad_accum 32 (paper)
│   └── train_smoke.yaml  # 3 steps, validação, 256²
├── scripts/
│   ├── setup_genfocus.sh    # clona o repo oficial (transformer_forward)
│   └── train_deblurnet.slurm
├── tests/test_data_pipeline.py   # contrato de dados (sem rede/GPU)
├── requirements.txt
└── .env.example
```

## Pré-requisitos

1. **GPU CUDA** (alvo: 1× H100). FLUX-1-dev exige bastante VRAM mesmo com LoRA + GC.
2. **Genfocus** (repo oficial dos autores) importável como `Genfocus.pipeline.flux`.
   Sem isso, `backbone.py` não carrega (mensagem de erro explica). Rode:
   ```bash
   bash scripts/setup_genfocus.sh
   # O pacote `Genfocus` fica DENTRO do clone (third_party/Genfocus/Genfocus/),
   # por isso o PYTHONPATH aponta para third_party/Genfocus, não third_party.
   export PYTHONPATH="$PWD:$PWD/third_party/Genfocus:$PYTHONPATH"
   ```
   > ⚠️ Confirme a URL do repo oficial em `scripts/setup_genfocus.sh`
   > (variável `GENFOCUS_REPO`) antes de rodar.
3. **Licença FLUX.1-dev** aceita no Hugging Face: `huggingface-cli login`.
4. **HF_TOKEN** para os datasets `akcit-pixel/*`: copie `.env.example` → `.env`.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
accelerate config default          # uma vez
bash scripts/setup_genfocus.sh
cp .env.example .env               # preencha HF_TOKEN
```

## Dados

A DeblurNet treina em pares (blurry, AIF). O dataloader lê dos repos HF:

| Coluna HF          | Uso                          |
|--------------------|------------------------------|
| `image_blur`       | entrada **blurry** (`blurry_image`) |
| `image_focus`      | alvo **all-in-focus** (`aif_image`) |
| `image_pre_deblur` | pré-foco gerado pela DRB-Net (variante opcional do paper) — **não usado** |
| `file_name_base`   | id                           |

Composição **fiel ao paper**: `akcit-pixel/DDPD` (DPDD) + `akcit-pixel/RealBokeh`
(subset). `akcit-pixel/LFDOF` está disponível mas **desligado** por padrão
(o paper não o usa na DeblurNet) — para incluí-lo, adicione em `train_base.yaml`.

**Pré-processamento** (`data.py`): resize do lado-menor para `image_size` +
crop `S×S` **alinhado** entre blurry/AIF (random no treino, central no smoke),
flip horizontal sincronizado no treino, normalização para **[-1, 1]**. Sem
distorção de aspect ratio.

## Smoke test (sempre antes do treino real)

```bash
python -m genfocus_train.train smoke --config configs/train_smoke.yaml
```
Valida: dataset carrega, forward via `transformer_forward` retorna shape certo,
loss finita, parâmetros LoRA mudam após 3 steps, export `.safetensors`.

Pontos mais prováveis de falha (sem fallback silencioso):
- import de `Genfocus.pipeline.flux` (PYTHONPATH);
- assinatura de `transformer_forward` (confronte com `flux.py` ~linha 360);
- `LORA_TARGET_MODULES` (`backbone.py`) que não bate com o FluxTransformer2DModel.

## Treino real (1× H100)

```bash
python -m genfocus_train.train deblur --config configs/train_base.yaml
# ou via slurm:
sbatch scripts/train_deblurnet.slurm
```

Default fiel ao paper: 60K steps, rank 128, 512², bf16, **batch efetivo 32**
(`gradient_accumulation_steps: 32` numa GPU), GC ligado, checkpoint a cada 1000
steps (LoRA only), mantém os 3 últimos. `runtime.resume: true` retoma do último
checkpoint automaticamente (validação de shape estrita).

## Output

- `outputs/<run>/deblur/checkpoints/step_*.pt` — checkpoints (LoRA, ~50–200 MB).
- `outputs/<run>/deblur/deblur.safetensors` — LoRA final (`pipe.load_lora_weights`).
- `outputs/<run>/deblur/metrics.jsonl` — métricas por step.
- `outputs/<run>/effective_config.yaml` — config materializada (auditoria).

## Decisões vs paper (fidelidade)

**Explícito no paper e reproduzido:** rank 128, 60K steps, batch efetivo 32,
DPDD + subset RealBokeh, FLUX-1-dev + LoRA, condicionamento token-concat,
28 denoising steps (inferência).

**[UNSPECIFIED] — decisões herdadas de FLUX LoRA (documentadas, configuráveis):**
optimizer AdamW lr=1e-4 wd=1e-4, cosine warmup 500, bf16, resolução 512²,
prompt fixo `"a sharp photo with everything in focus"`, objetivo velocity
(`v = ε - x_0`). O paper não fixa esses valores.

## A auditar antes dos 60K steps

1. Kwargs de `transformer_forward` em `Genfocus/pipeline/flux.py`.
2. `LORA_TARGET_MODULES` (`backbone.py`) vs o `.safetensors` oficial:
   ```python
   from safetensors.torch import load_file
   sd = load_file("deblurNet.safetensors")
   print(sorted({k.split(".lora_")[0].split(".")[-1] for k in sd}))
   ```
3. Se o subset RealBokeh no HF já está filtrado por variância laplaciana
   (top-3000, como o paper) ou se isso precisa ser feito no dataset.
4. Prompt fixo do treino vs o prompt da inferência oficial.

## Fora de escopo (Stage 2 / futuro)

BokehNet (`models.BokehNet`) existe como scaffold mas **não está integrado** ao
dataloader HF (o dataset HF não fornece depth/defocus/K). Pre-deblur module
(§3.4) e aperture-shape control (§3.3) não implementados.
