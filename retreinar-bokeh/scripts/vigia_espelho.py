#!/usr/bin/env python3
"""Espera o espelho da RealBokeh desbloquear, baixa, reindexa e valida.

CONTEXTO
--------
O release da rota C está completo em disco (15.423 amostras, 0 ausentes), mas o
ESPELHO não pode ser lido: `akcit-pixel/RealBokeh` é privado e a org estourou a
cota de armazenamento do HF, então os 90 shards `train`/`validation` respondem

    403 Forbidden: Private repository storage limit reached for akcit-pixel

Liberar espaço ou subir o plano é ação humana, na conta do HF. Este processo
existe para que, no instante em que isso acontecer, ninguém precise estar olhando:
ele testa o acesso, e quando passar executa a sequência inteira sozinho.

O QUE FAZ, EM ORDEM
-------------------
1. Testa a leitura de UM shard a cada `--intervalo` segundos. Teste barato: um
   `HEAD` via `get_hf_file_metadata`, sem baixar 500 MB para descobrir.
2. Desbloqueou: baixa `data/train-*` e `data/validation-*` (~44 GB, 90 arquivos).
3. Reconstrói o índice do espelho com `forcar=True`. **Isto não é opcional:** o
   cache em disco tem as 1.257 linhas do split `test` e seria lido como válido,
   deixando 94,7% dos nomes ausentes — com o agravante de que o `KeyError` viria
   já dentro do treino.
4. Confere a cobertura: todos os 15.423 nomes da rota C contra o índice.
5. Escreve o veredito em `releases/VIGIA_ESPELHO.txt`. A última linha é
   `PRONTO_PARA_TREINAR` ou `FALHOU`, para dar para conferir com um `tail -1`.

Não sobe treino nenhum: quem decide subir é a usuária.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

SHARD_TESTE = "data/train-00000-of-00085.parquet"
REPO = "akcit-pixel/RealBokeh"


def anota(caminho: Path, msg: str) -> None:
    linha = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(linha, flush=True)
    with caminho.open("a", encoding="utf-8") as h:
        h.write(linha + "\n")


def liberou() -> tuple[bool, str]:
    """HEAD num shard. Barato: não baixa os 500 MB para descobrir."""
    from huggingface_hub import get_hf_file_metadata, hf_hub_url

    try:
        get_hf_file_metadata(hf_hub_url(REPO, SHARD_TESTE, repo_type="dataset"))
        return True, "acesso OK"
    except Exception as exc:  # noqa: BLE001
        texto = str(exc)
        if "403" in texto or "storage limit" in texto:
            return False, "403 / cota de armazenamento"
        return False, f"{type(exc).__name__}: {texto[:120]}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--intervalo", type=int, default=600)
    ap.add_argument("--release", default="/workspace/releases/rota_c")
    ap.add_argument("--espelho", required=True, help="snapshot dir do espelho")
    ap.add_argument("--saida", default="/workspace/releases/VIGIA_ESPELHO.txt")
    args = ap.parse_args()

    saida = Path(args.saida)
    anota(saida, f"vigia iniciado; testando {REPO} a cada {args.intervalo}s")

    espera = 0
    while True:
        ok, motivo = liberou()
        if ok:
            anota(saida, f"DESBLOQUEOU depois de {espera/3600:.1f} h ({motivo})")
            break
        if espera % 3600 == 0:
            anota(saida, f"ainda bloqueado ({motivo}); esperando há {espera/3600:.0f} h")
        time.sleep(args.intervalo)
        espera += args.intervalo

    # ── 2. baixa ───────────────────────────────────────────────────────────
    from huggingface_hub import snapshot_download

    anota(saida, "baixando data/train-* e data/validation-* (~44 GB)")
    for tentativa in range(1, 9):
        try:
            t0 = time.time()
            destino = snapshot_download(
                REPO, repo_type="dataset", max_workers=4,
                allow_patterns=["data/train-*.parquet", "data/validation-*.parquet"],
            )
            anota(saida, f"download OK em {(time.time()-t0)/60:.1f} min -> {destino}")
            break
        except Exception as exc:  # noqa: BLE001
            anota(saida, f"tentativa {tentativa} falhou: {type(exc).__name__}: "
                         f"{str(exc)[:150]}")
            time.sleep(60)
    else:
        anota(saida, "FALHOU")
        return 1

    # ── 3. reindexa, FORÇADO ───────────────────────────────────────────────
    sys.path.insert(0, "/workspace/retreinar-bokeh")
    from genfocus_train import release as rel

    anota(saida, "reconstruindo o índice do espelho (forcar=True)")
    idx = rel.IndiceEspelho.carregar(args.espelho, forcar=True)
    anota(saida, f"índice: {len(idx)} linhas em {len(idx.shards)} shards")

    # ── 4. cobertura ───────────────────────────────────────────────────────
    linhas = [json.loads(l) for l in
              (Path(args.release) / "manifest.jsonl").read_text(
                  encoding="utf-8").splitlines() if l.strip()]
    nomes = {str(x["source_sample_id"]) for x in linhas}
    falta = sorted(n for n in nomes if n not in idx)
    cobertura = 100.0 * (len(nomes) - len(falta)) / max(len(nomes), 1)
    anota(saida, f"cobertura da rota C: {len(nomes)-len(falta)}/{len(nomes)} "
                 f"= {cobertura:.1f}%")
    if falta:
        anota(saida, f"ainda faltam {len(falta)} nomes (ex.: {falta[:3]})")
        anota(saida, "FALHOU")
        return 1

    anota(saida, "PRONTO_PARA_TREINAR")
    return 0


if __name__ == "__main__":
    sys.exit(main())
