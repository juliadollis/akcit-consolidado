#!/usr/bin/env python3
"""Baixa os shards `train` e `validation` da RealBokeh (~44 GB, 90 arquivos).

Por que é preciso: o snapshot local tinha SÓ o split `test` (1.257 linhas) e
94,7% da rota C sai de `train`/`validation`. Ver PLANO §11.7.

Três coisas que já falharam aqui e viraram guarda:

1. `snapshot_download` desta versão NÃO aceita `max_retries`. O laço retentava um
   `TypeError`, ou seja, um erro de programação — que nunca melhora sozinho.
2. Só erro de REDE é retentado; o resto sobe na hora.
3. **O retorno de `snapshot_download` NÃO prova que baixou.** Medido em
   2026-09-27: depois de dois `OSError: Disk quota exceeded`, a terceira
   tentativa voltou "OK em 0.0 min" com 53 de 96 shards em disco. Ele devolve o
   caminho do snapshot, não uma confirmação. Então aqui a gente CONTA os
   arquivos e só declara sucesso se estiverem todos.
"""
import sys
import time
from pathlib import Path

from huggingface_hub import snapshot_download

REPO = "akcit-pixel/RealBokeh"
ESPERADO = 96            # 85 train + 5 validation + 6 test
REDE = ("Protocol", "Timeout", "Connect", "HTTPError", "LocalEntryNotFound")


def quantos(destino: str) -> int:
    return len(list((Path(destino) / "data").glob("*.parquet")))


for tentativa in range(1, 9):
    try:
        t0 = time.time()
        destino = snapshot_download(
            REPO, repo_type="dataset", max_workers=4,
            allow_patterns=["data/train-*.parquet", "data/validation-*.parquet"],
        )
        n = quantos(destino)
        if n < ESPERADO:
            # NÃO é sucesso, mesmo tendo retornado sem exceção.
            print(f"tentativa {tentativa}: voltou sem erro mas há {n}/{ESPERADO} "
                  "shards em disco — provavelmente cota de disco. Retentando.",
                  flush=True)
            time.sleep(30)
            continue
        print(f"PRONTO: {n}/{ESPERADO} shards em {(time.time() - t0) / 60:.1f} min "
              f"-> {destino}", flush=True)
        break
    except Exception as exc:  # noqa: BLE001
        nome = type(exc).__name__
        if "quota" in str(exc).lower() or "No space" in str(exc):
            sys.exit(
                f"SEM ESPACO EM DISCO: {exc}\n"
                "A cota do /raid e por usuario (500 GB de aviso, 600 GB de teto). "
                "Libere espaco e rode de novo — nada e apagado automaticamente."
            )
        if not any(m in nome for m in REDE):
            raise
        print(f"tentativa {tentativa}: {nome}: {str(exc)[:200]}", flush=True)
        time.sleep(30)
else:
    raise SystemExit("desisti depois de 8 tentativas")
