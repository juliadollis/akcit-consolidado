# third_party

## BokehMe (`JuewenPeng/BokehMe`)

O renderer `[43]` que o paper cita na Eq. 5 e na Fig. 3(a). Clonado, não vendorizado:

```bash
git clone --depth 1 https://github.com/JuewenPeng/BokehMe.git
```

Os checkpoints `arnet.pth` (11 MB) e `iunet.pth` (2,9 MB) vêm **dentro do repositório**
— não precisa Google Drive.

### Dependências que não estão no container

O `classical_renderer.scatter` usa CuPy para o kernel de scatter em CUDA:

```bash
pip install --no-deps --target <projeto>/.pydeps "cupy-cuda12x<13" fastrlock
export PYTHONPATH=<projeto>/.pydeps${PYTHONPATH:+:$PYTHONPATH}   # PREFIXA, nunca substitui
```

**A versão importa, e por dois motivos independentes:**

| versão | problema |
|---|---|
| `cupy 14.x` | compilado contra numpy 2.x; o container tem 1.26.4 → `numpy.core.multiarray failed to import` |
| `cupy 13.x` | removeu `cupy.cuda.compile_with_cache`, que o `scatter.py` usa |
| **`cupy 12.3`** | **funciona** — mas precisa do patch abaixo |

`--no-deps` é obrigatório: sem ele o cupy arrasta numpy 2.2.6 para o `.pydeps` e
sombreia o 1.26.4 do container.

### O patch

`bokehme_cupy12_int_alias.patch` — 3 ocorrências de `cupy.int(x)` → `int(x)`.

`cupy.int` era um alias do `int` builtin (igual a `numpy.int`), removido no cupy 12.
A substituição é semanticamente idêntica; o BokehMe é de 2022 e foi escrito para
cupy ~9/10.

```bash
cd BokehMe && git apply ../bokehme_cupy12_int_alias.patch
```

O sha256 de `scatter.py` **depois** do patch entra na proveniência de cada amostra,
junto com o commit do checkout, o sha256 do `pipeline` extraído do `demo.py` e o dos
dois checkpoints. Assim dá para provar depois exatamente qual renderer produziu cada K.

### Descoberta lateral

O BokehMe já traz `classical_renderer/scatter_ex.py` com `ModuleRenderScatterEX`, que
aceita `poly_sides` — **formato de abertura poligonal**. O paper diz que "public
implementations typically omit this functionality" (§3.3). Não é PSF raster arbitrária
como a Eq. 6 pede, mas é mais do que o paper dá a entender, e é ponto de partida para
o LoRA de formato de abertura.

---

## Genfocus (`nycu-cplab/Genfocus`)

A pipeline de conditioning in-context que a **DeblurNet** usa. Clonado, não vendorizado —
mesma disciplina do BokehMe:

```bash
git clone https://github.com/nycu-cplab/Genfocus.git third_party/Genfocus
cd third_party/Genfocus && git rev-parse HEAD    # congele o commit no seu registro
```

`src/model_runtime/deblurnet.py` espera exatamente este layout, e valida os dois antes de
importar torch:

```
third_party/Genfocus/Inference_deblurNet.py          # prova que é o checkout certo
third_party/Genfocus/Genfocus/pipeline/flux.py       # Condition, generate, seed_everything
```

**Não** importe `bokehnet-preprocessing/src/vendor/genfocus_flux.py`. Aquele arquivo é a
mesma pipeline vendorizada, mas o repositório é referência histórica — e é onde o defeito
B1 vive (`generate` chamado sem `main_adapter`).

### O que é hasheado, e o que não é

Vai na proveniência de cada amostra:

| campo | o que é |
|---|---|
| `genfocus_commit` | `git rev-parse HEAD` do checkout |
| `genfocus_pipeline_sha256` | sha256 do fonte de `Genfocus/pipeline/flux.py` |
| `deblur_lora_sha256` | sha256 do `.safetensors` da variante |
| `flux_backbone_fingerprint` | **não** é hash de conteúdo do FLUX.1-dev |

O FLUX.1-dev tem dezenas de GB; hashear o conteúdo por run é inviável, e um hash que
ninguém roda é pior que nenhum. O fingerprint cobre caminho relativo + tamanho de todo
arquivo, mais o conteúdo integral dos `.json`/`.txt` pequenos — detecta snapshot trocado,
incompleto ou com config alterada. A chave `flux_backbone_fingerprint_kind` diz isso no
próprio dado, para a proveniência não afirmar mais do que verificou.

### A checagem por AST, e por que ela existe

`_inspect_flux_generate` parseia `flux.py` e **exige** que `generate` ainda declare
`main_adapter`. Não é zelo: a assinatura de `generate` termina em `**params: dict`, então
um kwarg renomeado pelo upstream é **absorvido em silêncio** e a inferência volta a ser
cond-only — sem `TypeError`, sem log, só uma AIF lavada. Por AST e não por `import`
porque `flux.py` importa torch, diffusers e **cv2** no nível de módulo, e no cluster o
cv2 do `~/.local` quebra com `GLIBC_2.38 not found` (mesmo motivo da sentinela em
`renderer/bokehme.py`).

### Os pesos: uma tripla por variante, indivisível

| variante | repo HF | arquivo | `main_adapter` |
|---|---|---|---|
| `OURS_MAIN_COND` | `juliadollis/genrefocus-deblurnet-paper-4gpu` | `deblur.safetensors` | `"deblurring"` |
| `OFFICIAL_COND_ONLY` | `nycu-cplab/Genfocus-Model` | `deblurNet.safetensors` | `None` |

Baixe **sempre** por `model_runtime.deblurnet.resolve_weights(variante)`, que deriva repo
e nome do arquivo da variante. `hf_hub_download` na mão permite pedir o arquivo de uma
variante no repositório da outra, e o peso não sabe de qual convenção ele precisa: LoRA
main+cond e cond-only têm as mesmas chaves.
