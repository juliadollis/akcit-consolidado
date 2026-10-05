#!/usr/bin/env python3
"""Sobe um diretório `runs_*` para o HF como dataset privado, com README gerado.

POR QUE EXISTE
--------------
Os cinco `runs_*` somam 113 GB e 87 checkpoints em `/raid`, e a cota do cluster
é 500 GB (soft, já sem período de graça) / 600 GB (hard) por usuário. Eles
estavam impedindo a fase 2 da BokehNet de rodar.

**Não havia checkpoint intermediário para podar.** Cada `.pt` é o `best` de um
experimento DISTINTO — variante × seed —, então apagar qualquer um perde um
resultado. A saída é subir para o Hub e só então liberar o disco.

O README de cada repo é GERADO da árvore real: variantes, seeds, contagem e
tamanho. Um README escrito à mão envelhece; este descreve o que está lá.

Uso:  sobe_runs_para_hf.py <dir> <repo_id> [--publico]
"""
from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict
from pathlib import Path


def inventario(raiz: Path) -> tuple[dict[str, list[str]], int, int]:
    """(variante -> [seeds], n_pesos, bytes_totais) lidos da árvore real."""
    grupos: dict[str, list[str]] = defaultdict(list)
    n = 0
    total = 0
    for p in sorted(raiz.rglob("*")):
        if p.suffix not in (".pt", ".safetensors", ".bin") or not p.is_file():
            continue
        n += 1
        total += p.stat().st_size
        rel = p.relative_to(raiz).parts
        if len(rel) >= 3:
            grupos[rel[0]].append(rel[1])
        elif len(rel) == 2:
            grupos[rel[0]].append("-")
    return {k: sorted(set(v)) for k, v in grupos.items()}, n, total


def readme(raiz: Path, repo_id: str, grupos, n, total) -> str:
    linhas = [
        "---",
        "license: other",
        f"pretty_name: {raiz.name}",
        "tags: [depth-estimation, checkpoints, ablation]",
        "---",
        "",
        f"# {repo_id}",
        "",
        f"Backup dos checkpoints de treino de **`{raiz.name}`** — experimentos de "
        "estimativa de profundidade do projeto Riemann.",
        "",
        "Subido para liberar disco no cluster: os cinco diretórios `runs_*` somavam "
        "113 GB contra uma cota de 500 GB por usuário, e estavam impedindo outro "
        "treino de rodar. **Nada foi descartado na subida** — cada arquivo aqui é o "
        "`best.pt` de um experimento distinto (variante × seed), não um checkpoint "
        "intermediário.",
        "",
        "## Conteúdo",
        "",
        f"| | |",
        f"|---|---|",
        f"| pesos | **{n}** |",
        f"| tamanho | **{total / 1e9:.1f} GB** |",
        f"| variantes | {len(grupos)} |",
        "",
        "| variante | seeds / cabeças |",
        "|---|---|",
    ]
    for k in sorted(grupos):
        vs = grupos[k]
        mostra = ", ".join(vs[:8]) + (f" … (+{len(vs) - 8})" if len(vs) > 8 else "")
        linhas.append(f"| `{k}` | {mostra} |")
    linhas += [
        "",
        "## Layout",
        "",
        "```",
        f"{raiz.name}/<variante>/<seed>/best.pt",
        "```",
        "",
        "Junto de cada `best.pt` vão os `.json` de métrica que o treino gravou, "
        "quando existiam.",
        "",
        "## Nomenclatura",
        "",
        "Os nomes de variante codificam a composição da loss:",
        "",
        "- `B0`…`B7` — a família de configuração da cabeça;",
        "- `berhu` — a loss base;",
        "- `+normal`, `+gauss`, `+geod`, `+metric`, `+grad` — termos adicionais;",
        "- `teto<N>` — teto aplicado ao termo métrico;",
        "- `size768` — resolução de treino.",
        "",
        "> Gerado por `scripts/sobe_runs_para_hf.py` a partir da árvore real, "
        "não escrito à mão.",
    ]
    return "\n".join(linhas) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("repo_id")
    ap.add_argument("--publico", action="store_true")
    args = ap.parse_args()

    raiz = Path(args.dir)
    if not raiz.is_dir():
        sys.exit(f"não é diretório: {raiz}")

    from huggingface_hub import HfApi

    grupos, n, total = inventario(raiz)
    print(f"{raiz.name}: {n} pesos, {total/1e9:.1f} GB, {len(grupos)} variantes",
          flush=True)
    if n == 0:
        sys.exit("nenhum peso encontrado — nada a subir")

    texto = readme(raiz, args.repo_id, grupos, n, total)
    (raiz / "README.md").write_text(texto, encoding="utf-8")
    print("README gerado", flush=True)

    api = HfApi()
    api.create_repo(args.repo_id, repo_type="dataset",
                    private=not args.publico, exist_ok=True)
    print(f"repo {args.repo_id} (private={not args.publico}) pronto; subindo…",
          flush=True)
    api.upload_large_folder(
        repo_id=args.repo_id, repo_type="dataset", folder_path=str(raiz),
        print_report=True,
    )
    print(f"PRONTO: https://huggingface.co/datasets/{args.repo_id}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
