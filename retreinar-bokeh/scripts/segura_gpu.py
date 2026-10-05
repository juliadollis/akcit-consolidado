#!/usr/bin/env python3
"""Reserva as GPUs enquanto os dados terminam de baixar.

POR QUE EXISTE
--------------
Esta máquina não tem fila: quem chega primeiro pega. As GPUs 0 e 1 ficaram
livres agora e o treino da fase 2 só pode começar quando ~44 GB de espelho e
~70 mil arquivos de release terminarem de chegar. Sem reserva, a janela entre
"dado pronto" e "treino no ar" é tempo em que outra pessoa legitimamente ocupa
a placa — e aí a espera recomeça.

COMO SOLTA
----------
Duas formas, as duas limpas:
  1. criar o arquivo `/workspace/releases/LIBERAR_GPUS` — o processo devolve a
     memória e sai sozinho. É o handoff preferido: o treino sobe logo depois,
     sem `docker stop` em nada.
  2. `docker stop` no container.

O QUE FAZ
---------
Aloca um bloco grande em cada GPU visível e dorme. Não computa nada: o objetivo
é ocupar memória, não queimar energia. Imprime o estado a cada 5 min para quem
olhar o log saber o que é isto e de quem é.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import torch


def main() -> int:
    alvo_gb = float(os.environ.get("SEGURA_GB", "70"))
    marca = Path(os.environ.get("MARCA_LIBERAR", "/workspace/releases/LIBERAR_GPUS"))
    n = torch.cuda.device_count()
    if n == 0:
        print("[segura] nenhuma GPU visível — nada a fazer.", flush=True)
        return 1

    blocos = []
    for d in range(n):
        torch.cuda.set_device(d)
        livre, total = torch.cuda.mem_get_info(d)
        # deixa uma folga: encostar no teto faz o próprio treino falhar depois
        gb = min(alvo_gb, livre / 1e9 - 2.0)
        if gb <= 0:
            print(f"[segura] GPU {d} já está cheia ({livre/1e9:.1f} GB livres); "
                  "não reservo.", flush=True)
            continue
        n_elem = int(gb * 1e9 / 2)          # float16 = 2 bytes
        blocos.append(torch.empty(n_elem, dtype=torch.float16, device=f"cuda:{d}"))
        print(f"[segura] GPU {d}: reservados {gb:.1f} GB", flush=True)

    if not blocos:
        return 1
    print(f"[segura] segurando {len(blocos)} GPU(s). Para soltar: "
          f"`touch {marca}` (preferido) ou `docker stop` neste container.",
          flush=True)

    t0 = time.time()
    while not marca.exists():
        time.sleep(30)
        if int(time.time() - t0) % 300 < 30:
            print(f"[segura] segurando há {(time.time()-t0)/60:.0f} min "
                  f"({len(blocos)} GPUs). Solte com `touch {marca}`.", flush=True)
    print("[segura] marca de liberação encontrada; devolvendo a memória.", flush=True)
    del blocos
    torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    sys.exit(main())
