# 🧩 O LoRA da DeblurNet — o que usamos e a diferença pro original do paper

## 1. O básico (igual ao paper)

O DeblurNet **não treina o FLUX inteiro** — treina um **LoRA** (adaptador de baixo
posto) sobre o `FLUX.1-dev` **congelado**.

| Parâmetro | Valor | Igual ao paper? |
|---|---|---|
| Backbone | `black-forest-labs/FLUX.1-dev` (congelado) | ✅ |
| **Rank do LoRA** | **128** | ✅ (§4.1) |
| `lora_alpha` | 128 | ✅ |
| Init | gaussiano | ✅ |
| Módulos-alvo | `to_q, to_k, to_v, to_out.0, norm1.linear, ff.net.2, norm.linear, proj_mlp, proj_out, x_embedder` | ✅ (batem 1:1 com o `deblurNet.safetensors` oficial) |

Pesos, rank e módulos **idênticos ao paper**. A diferença **não é no LoRA em si** —
é em **ONDE ele é aplicado** no forward.

---

## 2. Conceito-chave: os branches (OminiControl)

O condicionamento é estilo **OminiControl**: a imagem borrada entra como **condição**.
O `transformer_forward` processa **3 branches** juntos, com atenção cruzada:

```
[ texto ]   [ main (imagem sendo gerada) ]   [ condição (imagem borrada) ]
```

E o LoRA pode ser **ligado/desligado por branch** (via `specify_lora`). É isso que
diferencia a nossa versão da oficial.

---

## 3. A DIFERENÇA 🔴 — onde o LoRA age

| | LoRA ligado em... | Inferência |
|---|---|---|
| **Oficial (paper)** | **só na condição** | `Inference_deblurNet.py` usa `generate(main_adapter=None)` → LoRA só na cond; texto e main rodam FLUX puro |
| **A NOSSA (projeto principal)** | **main + condição** | precisa de `generate(main_adapter="deblurring")` |

No `genfocus_train/backbone.py` é literalmente **uma linha**:

```python
# NOSSA (main+cond):
adapters = [None, ADAPTER_NAME] + [ADAPTER_NAME] * n_cond   # texto=None, main=LoRA, cond=LoRA

# PAPER (cond-only):
adapters = [None, None]        + [ADAPTER_NAME] * n_cond    # texto=None, main=None, cond=LoRA
```

---

## 4. Por que importa (o bug que pegamos)

Treino e inferência **têm que casar**. Como treinamos o LoRA no branch **main**, a
inferência **também** precisa aplicá-lo lá:

- **Inferência oficial** (`main_adapter=None`) no **nosso** peso → o deblur aprendido
  **não é aplicado** na imagem gerada → **saída lavada** (foi o LPIPS ~0.85).
- **Inferência com `main_adapter="deblurring"`** no **nosso** peso → **deblur nítido
  e fiel** ✅.

> Confirmado empiricamente: mesmo peso, `main_adapter=deblurring` = nítido;
> `main_adapter=None` = lavado. O treino sempre esteve **certo** — o problema era só
> a inferência não ligar o LoRA no branch certo.

---

## 5. As duas versões

| Projeto | LoRA | Inferência | Repo HF |
|---|---|---|---|
| `genrefocus_deblurnet/` (treinado, 60K) | **main+cond** | `scripts/infer_mainlora.py` (`main_adapter="deblurring"`) | `genrefocus-deblurnet-paper-4gpu` |
| `genrefocus_deblurnet_paper/` (em treino) | **cond-only** (= paper) | `Inference_deblurNet.py` oficial (`main_adapter=None`) | `genrefocus-deblurnet-condlora-*` |

- **main+cond**: funciona ótimo, mas exige inferência custom (`main_adapter`).
- **cond-only**: bate exatamente com o modelo oficial → roda direto na pipeline do
  paper, sem inferência custom.

---

## 6. Resumo em uma frase

> Usamos o **mesmo LoRA do paper** (rank 128, mesmos módulos), mas aplicado em **um
> branch a mais** (o `main`, além da condição). Isso deixou o modelo bom no treino,
> porém **incompatível com a inferência oficial** — daí a variante
> `main_adapter="deblurring"`. A cópia `_paper` refaz o treino **só na condição** pra
> ficar 100% compatível com a pipeline oficial do paper.

---

## Custo de treino (referência)

- DeblurNet 60K (main+cond): **~79 h de compute** em 15 chunks de ~6h = **~318 GPU-horas** (4× H100).
