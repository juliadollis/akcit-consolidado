### 🧩 DeblurNet: LoRA sobre FLUX.1-dev

**Setup (fiel ao paper §4.1):** `FLUX.1-dev` congelado; LoRA r=128, α=128, init gaussiano, dropout 0. Módulos-alvo (10): `to_q, to_k, to_v, to_out.0, norm1.linear, ff.net.2, norm.linear, proj_mlp, proj_out, x_embedder`, idênticos às chaves do `deblurNet.safetensors` oficial.

**Condicionamento (OminiControl, token-concat):** o `transformer_forward` processa 3 branches concatenados, `[texto, main (xt), condição (E(I_blur))]`, com atenção cruzada via `group_mask`. Objetivo: rectified flow `v = eps - x0`, MSE (fp32) no latente.

**LoRA por-branch (`specify_lora`, scaling 1/0):** é o único ponto de divergência do oficial.

* **Oficial:** inferência com `main_adapter=None`; LoRA ativo **só na condição**.
* **Nosso:** treino com `adapters=[None, ADAPTER, ADAPTER]`, inferência com `main_adapter="deblurring"`; LoRA ativo no **main + condição**.

**Restrição treino/inferência:** o branch com LoRA no treino precisa ter LoRA na inferência.

* nosso peso com `main_adapter=None` (pipeline oficial): LoRA inativo no `main`, LPIPS ≈ 0.85 (saída perto da média);
* com `main_adapter="deblurring"`: deblur correto (mesmo peso, mesma seed).

**Treino:** 512², bf16, GC off, batch efetivo 32 (1 × gacc 8 × 4 GPU), AdamW lr=1e-4 wd=1e-4, cosine warmup 500, 60K steps. Custo: ~79 h, ~318 GPU-h (4× H100, 15 chunks).

**Variante `_condlora`:** `adapters=[None, None, ADAPTER]` (cond-only) = design oficial; roda no `Inference_deblurNet.py` sem `main_adapter`.
