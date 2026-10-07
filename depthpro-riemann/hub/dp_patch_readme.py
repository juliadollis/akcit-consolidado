#!/usr/bin/env python3
import os
from pathlib import Path
from huggingface_hub import HfApi

p = Path("/host/_hub_stage_modelos/README.md")
t = p.read_text()

velho = """Conteudo: **55 checkpoints**,
332 arquivos, 70.2 GiB."""
novo = """Conteudo: **47 checkpoints de seed** (10 originais + 37 retreino) mais
**8 checkpoints de ablacao**, 332 arquivos, 70.2 GiB."""
assert velho in t, "trecho 1 nao encontrado"
t = t.replace(velho, novo)

velho2 = """| `B3_gauss_metrica_teto50` | 6 | 0, 1, 2, 3, 4, 5 | `runs_riemann` |"""
novo2 = velho2 + """

> Por que `B1_gauss_metrica_teto5` tem so a seed 3 aqui: foi a unica desse braco cujo
> peso sobreviveu. O retreino desse braco cobriu exatamente as seeds 0, 1, 2, 4 e 5 —
> **pulou a 3** justamente porque ela ja existia. Os dois diretorios se complementam."""
assert velho2 in t, "trecho 2 nao encontrado"
t = t.replace(velho2, novo2)

p.write_text(t)
api = HfApi(token=os.environ["HF_TOKEN"])
api.upload_file(path_or_fileobj=str(p), path_in_repo="README.md",
                repo_id="akcit-dephpro/depthpro-riemann-modelos", repo_type="model",
                commit_message="README: desambigua contagem de checkpoints e explica a seed 3 do B1_teto5")
print("README atualizado")
