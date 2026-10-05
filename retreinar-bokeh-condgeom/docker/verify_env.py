#!/usr/bin/env python3
"""Confere se o ambiente DENTRO do container bate com o manifesto comprovado.

Rode logo depois do `docker build`, ANTES de gastar dias de GPU:

    docker run --rm --user $(id -u):$(id -g) \\
        -v /raid/user_juliadollis/julia_docker/genrefocus_deblurnet_paper:/workspace/genrefocus_deblurnet_paper \\
        julia-genrefocus:1.0 python3 docker/verify_env.py

O manifesto veio da H100-02 (2026-08-13), onde a fase 1 rodou 40K steps sem
falha de ambiente. A base `huggingface/transformers-pytorch-gpu:latest` e uma
tag MOVEL, entao o transformers/torch da imagem pode mudar sem aviso — este
script existe para pegar isso na hora, nao no meio do treino.

Sai com codigo != 0 se algo essencial divergir.
"""

from __future__ import annotations

import importlib.metadata as md
import sys

# Versoes exatas do ambiente que rodou a fase 1 sem falha.
ESPERADO = {
    "torch": "2.9.0",            # prefixo: o +cu126 varia com a base
    "transformers": "5.0.0.dev0",
    "diffusers": "0.37.1",
    "peft": "0.18.1",
    "accelerate": "1.11.0",
    "datasets": "4.3.0",
    "huggingface-hub": "1.0.0rc6",
    "safetensors": "0.6.2",
    "numpy": "1.26.4",
    "wandb": "0.28.1",
}

# Divergencia nestes quebra o treino (ver secao 4 do historico-ultimo.md).
CRITICOS = {"transformers", "diffusers", "peft", "torch"}


def main() -> int:
    print(f"Python: {sys.version.split()[0]}")
    problemas = []
    for pkg, alvo in ESPERADO.items():
        try:
            atual = md.version(pkg)
        except md.PackageNotFoundError:
            print(f"  {pkg:20s} AUSENTE            (esperado {alvo})")
            problemas.append((pkg, "ausente", alvo))
            continue
        ok = atual.startswith(alvo)
        marca = "ok " if ok else "DIVERGE"
        print(f"  {pkg:20s} {atual:22s} {marca} (esperado {alvo})")
        if not ok:
            problemas.append((pkg, atual, alvo))

    # A regra que custou 4 jobs: diffusers 0.37.x exige peft >= 0.17.
    try:
        peft_v = md.version("peft")
        maj, minor = (int(x) for x in peft_v.split(".")[:2])
        if (maj, minor) < (0, 17):
            print(f"\nERRO: peft {peft_v} < 0.17 — diffusers 0.37.1 exige peft>=0.17.")
            problemas.append(("peft", peft_v, ">=0.17"))
    except Exception:  # noqa: BLE001
        pass

    # Import real: versao certa mas import quebrado nao serve de nada.
    print("\nImports:")
    for mod in ("torch", "transformers", "diffusers", "peft", "accelerate", "datasets"):
        try:
            __import__(mod)
            print(f"  {mod:14s} ok")
        except Exception as exc:  # noqa: BLE001
            print(f"  {mod:14s} FALHOU: {type(exc).__name__}: {exc}")
            problemas.append((mod, "import falhou", "importavel"))

    try:
        import torch

        print(f"\nCUDA disponivel: {torch.cuda.is_available()} | GPUs visiveis: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  [{i}] {torch.cuda.get_device_name(i)}")
    except Exception as exc:  # noqa: BLE001
        print(f"\nERRO consultando CUDA: {exc}")
        problemas.append(("cuda", str(exc), "disponivel"))

    criticos = [p for p in problemas if p[0] in CRITICOS or p[0] == "cuda"]
    print("\n" + "=" * 60)
    if not problemas:
        print("AMBIENTE OK — bate com o manifesto da H100-02.")
        return 0
    if criticos:
        print(f"AMBIENTE COM {len(criticos)} DIVERGENCIA(S) CRITICA(S). NAO rode o treino:")
        for pkg, atual, alvo in criticos:
            print(f"  {pkg}: {atual} (esperado {alvo})")
        return 1
    print(f"{len(problemas)} divergencia(s) NAO-critica(s) — provavelmente ok, mas confira:")
    for pkg, atual, alvo in problemas:
        print(f"  {pkg}: {atual} (esperado {alvo})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
