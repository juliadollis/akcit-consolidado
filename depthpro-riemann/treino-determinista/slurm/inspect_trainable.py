#!/usr/bin/env python3
"""
Diagnostico: QUAIS modulos do DepthPro estao realmente sendo treinados.

Motivo: riemann/model.py:123 descongela por substring
    head_substr = ("head", "decoder", "fov", "upsample", "final")
e o DepthPro tem um modulo `fov` que contem um ViT inteiro. Isso infla a
contagem de "params treinaveis" (342M no nosso smoke) e faz parecer que o
encoder esta sendo treinado.

Mas params "treinaveis" nao e o mesmo que params TREINADOS: se o modulo nao
participa da loss, ele nao recebe gradiente e o AdamW o ignora (p.grad is None).
O forward de treino usa out[0] (a profundidade) e DESCARTA a saida de fov.

Este script separa as duas coisas:
  (A) o que esta marcado requires_grad=True
  (B) o que REALMENTE recebe gradiente apos um backward de verdade

Uso (dentro do container, a partir da raiz do repo do Francisco):
    python3 /workspace/projects/depth-riemannian/slurm/inspect_trainable.py \
        --checkpoint /workspace/models/checkpoints/depth_pro.pt --variant heads
"""

import argparse
import collections
import sys

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="/workspace/models/checkpoints/depth_pro.pt")
    ap.add_argument("--variant", default="heads", choices=["heads", "heads_lora"])
    ap.add_argument("--size", type=int, default=384, help="lado da imagem de teste")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    from riemann.model import build_model

    m = build_model(args.checkpoint, args.variant, device=args.device)

    # (A) marcados como treinaveis, agrupados por modulo de topo
    marked = collections.Counter()
    for n, p in m.core.named_parameters():
        if p.requires_grad:
            marked[n.split(".")[0]] += p.numel()

    # (B) quem realmente recebe gradiente
    m.core.zero_grad(set_to_none=True)
    x = torch.randn(1, 3, args.size, args.size, device=args.device)
    pred = m(x)
    loss = pred.mean()
    loss.backward()

    got_grad = collections.Counter()
    no_grad = collections.Counter()
    for n, p in m.core.named_parameters():
        if not p.requires_grad:
            continue
        top = n.split(".")[0]
        if p.grad is None:
            no_grad[top] += p.numel()
        else:
            got_grad[top] += p.numel()

    tot_marked = sum(marked.values())
    tot_grad = sum(got_grad.values())
    tot_nograd = sum(no_grad.values())

    print("\n" + "=" * 72)
    print("POR MODULO DE TOPO (params marcados requires_grad=True)")
    print("=" * 72)
    print(f"{'modulo':<24}{'marcados':>15}{'com grad':>15}{'SEM grad':>15}")
    for k, v in marked.most_common():
        print(f"{k:<24}{v:>15,}{got_grad.get(k, 0):>15,}{no_grad.get(k, 0):>15,}")

    print("-" * 72)
    print(f"{'TOTAL':<24}{tot_marked:>15,}{tot_grad:>15,}{tot_nograd:>15,}")
    print("=" * 72)

    pct = 100.0 * tot_nograd / tot_marked if tot_marked else 0.0
    print(f"\nParams marcados como treinaveis que NAO recebem gradiente: "
          f"{tot_nograd:,} ({pct:.1f}%)")
    if tot_nograd > 0:
        print("Esses params inflam a contagem do log, rodam no forward (custam tempo)")
        print("e NAO sao atualizados pelo otimizador. Modulos afetados:",
              ", ".join(sorted(no_grad)))
    print(f"\nParams EFETIVAMENTE treinados: {tot_grad:,}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
