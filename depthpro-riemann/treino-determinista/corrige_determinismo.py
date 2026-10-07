#!/usr/bin/env python3
"""Troca F.pad(mode="replicate") por uma versao deterministica, numa COPIA.

O backward do replication padding na CUDA usa atomicAdd e e nao deterministico.
E isso que faz todo braco geometrico nao reproduzir: medimos 0,0148 de |delta|
medio no F-borda entre duas execucoes da MESMA seed, contra 0,0000 no berHu puro.

A troca e por fatiamento mais torch.cat, que da o MESMO forward. O backward vira
slice mais soma de bordas, sem atomicAdd.

NAO altera o codigo em uso: escreve em depth-riemannian-det/.
"""
import os
import re
import shutil
import sys

ORIG = "/host/depth-riemannian"
DEST = "/host/depth-riemannian-det"

AJUDANTE = '''

def _pad_repl_det(t, pad):
    """Replicate padding DETERMINISTICO (fatiamento + cat, sem atomicAdd).

    pad = (esquerda, direita, cima, baixo), a mesma ordem do F.pad.
    Existe porque o backward do replication_pad2d na CUDA usa atomicAdd e nao e
    reproduzivel; isso contaminava todo termo geometrico da perda.
    """
    import torch as _t
    l, r, c, b = pad
    if l or r:
        partes = []
        if l:
            partes.append(t[..., :1].repeat(*([1] * (t.dim() - 1)), l))
        partes.append(t)
        if r:
            partes.append(t[..., -1:].repeat(*([1] * (t.dim() - 1)), r))
        t = _t.cat(partes, dim=-1)
    if c or b:
        partes = []
        if c:
            partes.append(t[..., :1, :].repeat(*([1] * (t.dim() - 2)), c, 1))
        partes.append(t)
        if b:
            partes.append(t[..., -1:, :].repeat(*([1] * (t.dim() - 2)), b, 1))
        t = _t.cat(partes, dim=-2)
    return t
'''

PADRAO = re.compile(r'F\.pad\(\s*([^,]+?),\s*(\([^)]*\)),\s*mode\s*=\s*"replicate"\s*\)')


def main():
    if os.path.exists(DEST):
        shutil.rmtree(DEST)
    shutil.copytree(ORIG, DEST)
    total = 0
    for rel in ("riemann/geometry.py", "riemann/losses.py"):
        caminho = os.path.join(DEST, rel)
        with open(caminho) as f:
            src = f.read()
        novo, n = PADRAO.subn(r"_pad_repl_det(\1, \2)", src)
        if n:
            # insere o ajudante logo apos os imports
            marca = novo.index("\n\n", novo.index("import "))
            novo = novo[:marca] + AJUDANTE + novo[marca:]
            with open(caminho, "w") as f:
                f.write(novo)
        print(f"  {rel}: {n} ocorrencias trocadas")
        total += n
    print(f"total: {total} (esperado 8)")
    if total != 8:
        print("ATENCAO: numero diferente do esperado, confira antes de usar")
        sys.exit(1)
    # sanidade: o modulo ainda importa?
    sys.path.insert(0, DEST)
    import importlib
    for m in ("riemann.geometry", "riemann.losses"):
        importlib.import_module(m)
    print("import OK nos dois modulos")


if __name__ == "__main__":
    main()
