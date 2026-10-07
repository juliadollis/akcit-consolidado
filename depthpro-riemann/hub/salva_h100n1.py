#!/usr/bin/env python3
"""Arquiva na org akcit-h100n1 tudo que so existe na maquina.

REGRA: o /raid nao e duravel. Em 2026-09-10 uma liberacao de quota apagou 37 de
47 checkpoints. O que nao esta no Hub nao existe.

NAO sobe: caches do HF (reproduziveis), datasets publicos, libs de terceiros e o
depth_pro.pt oficial da Apple. Sobe o que foi PRODUZIDO aqui e nao tem copia.
"""
import glob
import os

from huggingface_hub import HfApi

api = HfApi(token=os.environ["HF_TOKEN"])
ORG = "akcit-h100n1"
B = "/host"

GRUPOS = {
    "bokehnet-checkpoints": {
        "desc": "Checkpoints completos de treino do BokehNet (com estado do otimizador)",
        "padroes": [
            f"{B}/genrefocus_deblurnet_paper/outputs/*/bokeh/checkpoints/*.pt",
            f"{B}/genrefocus_deblurnet_paper/outputs/*/bokeh/*.safetensors",
            f"{B}/genrefocus_deblurnet_paper/outputs/*/*.safetensors",
        ],
        "raiz": f"{B}/genrefocus_deblurnet_paper/outputs",
    },
    "deblurnet-checkpoints": {
        "desc": "Checkpoints completos de treino do DeblurNet (com estado do otimizador)",
        "padroes": [
            f"{B}/retreinar-deblur/outputs/*/deblur/checkpoints/*.pt",
            f"{B}/retreinar-deblur/outputs/*/deblur/*.safetensors",
        ],
        "raiz": f"{B}/retreinar-deblur/outputs",
    },
}

CABECALHO = """---
license: apache-2.0
---

# {titulo}

{desc}.

## Por que este repositorio existe

**Checkpoint que nao esta no Hub nao existe.**

Em 2026-09-10 uma liberacao de quota no `/raid` da dgx-H100-01 apagou 37 de 47
checkpoints de uma campanha. As metricas sobreviveram porque alguem as separou a
mao antes; os pesos nao. O `/raid` e area de trabalho, nao armazenamento duravel.

Este repositorio e o arquivo do que foi PRODUZIDO naquela maquina e nao tinha
copia em lugar nenhum.

## O que tem aqui

Os caminhos abaixo espelham a estrutura original em
`/raid/user_juliadollis/julia_docker/`.

{listagem}

## O que NAO esta aqui

| o que | por que | onde esta |
|---|---|---|
| caches do Hugging Face (357 GB) | sao downloads do proprio Hub | Hugging Face |
| datasets (86 GB) | publicos | DaRUS e as fontes originais |
| `depth_pro.pt` | peso oficial da Apple | `scripts/download_weights.py` |
| o codigo | versionado | https://github.com/juliadollis/akcit-julia |

## Codigo

Todo o codigo que produziu estes pesos esta em
<https://github.com/juliadollis/akcit-julia>, na pasta `h100n1-todo-o-codigo/`.
"""


def main():
    total_bytes = 0
    for nome, cfg in GRUPOS.items():
        repo = f"{ORG}/{nome}"
        arquivos = []
        for p in cfg["padroes"]:
            arquivos.extend(glob.glob(p))
        arquivos = sorted(set(arquivos))
        if not arquivos:
            print(f"[{nome}] nenhum arquivo, pulando", flush=True)
            continue

        api.create_repo(repo, repo_type="model", private=True, exist_ok=True)
        ja = set()
        try:
            ja = set(api.list_repo_files(repo, repo_type="model"))
        except Exception:
            pass

        print(f"\n[{nome}] {len(arquivos)} arquivos candidatos", flush=True)
        linhas = []
        for a in arquivos:
            rel = os.path.relpath(a, cfg["raiz"])
            tam = os.path.getsize(a)
            linhas.append(f"| `{rel}` | {tam/1e9:.2f} GB |")
            if rel in ja:
                print(f"  ja no Hub: {rel}", flush=True)
                continue
            print(f"  subindo {rel} ({tam/1e9:.2f} GB) ...", flush=True)
            api.upload_file(path_or_fileobj=a, path_in_repo=rel,
                            repo_id=repo, repo_type="model")
            total_bytes += tam

        listagem = ("| arquivo | tamanho |\n|---|---|\n" + "\n".join(linhas))
        readme = CABECALHO.format(titulo=nome.replace("-", " ").title(),
                                  desc=cfg["desc"], listagem=listagem)
        caminho = f"/tmp/README_{nome}.md"
        with open(caminho, "w") as f:
            f.write(readme)
        api.upload_file(path_or_fileobj=caminho, path_in_repo="README.md",
                        repo_id=repo, repo_type="model")
        print(f"[{nome}] README enviado", flush=True)

    print(f"\nTOTAL SUBIDO: {total_bytes/1e9:.1f} GB", flush=True)
    print("FIM", flush=True)


if __name__ == "__main__":
    main()
