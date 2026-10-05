"""Pré-aquece o cache: baixa os datasets do treino e popula o cache do filtro.

Roda em job SLURM SEM GPU (`INSTRUCOES_H100.md`: trabalho pesado de CPU deve ir
para o SLURM, e nada pode ser baixado fora dele). Existe para que o job de 4
GPUs comece a treinar imediatamente, em vez de segurar as GPUs ociosas durante
o download (~56 GB) e a medição de nitidez do RealBokeh (~20 min, e pior com 4
ranks disputando disco).

Duas fases, deliberadamente independentes:

  1. DOWNLOAD — usa só `datasets.load_dataset`, sem nenhum código do projeto.
     É o caminho longo e não pode falhar por bug nosso.
  2. FILTRO — usa `genfocus_train.data._select_top_k_sharpest` para gravar o
     cache de índices. É best-effort: se falhar, o treino apenas refaz a
     medição, sem perder nada.

NÃO apaga nada. NÃO escreve fora do HF_HOME e do cache do filtro.
"""

from __future__ import annotations

import argparse
import os
import sys
import time


def _log(msg: str) -> None:
    print(f"[prefetch] {msg}", flush=True)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/train_deblur_paper.yaml")
    p.add_argument("--pular-filtro", action="store_true",
                   help="só baixa, não mede nitidez")
    args = p.parse_args()

    from datasets import load_dataset

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    if not token:
        _log("ERRO: HF_TOKEN ausente no ambiente.")
        return 1
    _log(f"HF_HOME={os.environ.get('HF_HOME', '(default)')}")

    # ── fase 1: download ────────────────────────────────────────────────────
    # Lista fixa e explícita. Não deriva do config de propósito: se o config
    # mudar, quero que alguém releia isto em vez de baixar 47 GB por engano.
    alvos = [
        ("akcit-pixel/DDPD", "train"),
        ("akcit-pixel/DDPD", "validation"),
        ("akcit-pixel/RealBokeh", "train"),
    ]
    baixados = {}
    for repo, split in alvos:
        t0 = time.time()
        _log(f"baixando {repo}:{split} ...")
        ds = load_dataset(repo, split=split, token=token)
        baixados[(repo, split)] = ds
        _log(f"  ok: {len(ds)} linhas | colunas={ds.column_names} | {time.time()-t0:.0f}s")

    # ── fase 2: cache do filtro de nitidez (best-effort) ────────────────────
    if args.pular_filtro:
        _log("filtro pulado por --pular-filtro.")
        return 0

    try:
        sys.path.insert(0, os.getcwd())
        from genfocus_train.config import load_config, stage_config
        from genfocus_train.data import _select_top_k_sharpest

        cfg = load_config(args.config)
        sc = stage_config(cfg, "deblur")
        for fonte in sc.datasets:
            if fonte.top_k_sharpest is None:
                continue
            ds = baixados.get((fonte.name, fonte.split))
            if ds is None:
                ds = load_dataset(fonte.name, split=fonte.split, token=token)
            t0 = time.time()
            _log(f"medindo nitidez de {fonte.name}:{fonte.split} "
                 f"(top {fonte.top_k_sharpest}, modo={sc.top_k_mode}) ...")
            sel = _select_top_k_sharpest(
                ds, k=int(fonte.top_k_sharpest), column=fonte.sharpness_column,
                cache_id=f"{fonte.name}_{fonte.split}",
                top_k_mode=sc.top_k_mode, scene_key=sc.scene_key,
            )
            _log(f"  ok: {len(sel)} selecionadas | {time.time()-t0:.0f}s")
    except TypeError as exc:
        # assinatura do seletor mudou — não é fatal, o treino refaz a medição
        _log(f"AVISO: assinatura de _select_top_k_sharpest diferente ({exc}). "
             "Cache do filtro não populado; o treino vai medir sozinho.")
    except Exception as exc:  # noqa: BLE001
        _log(f"AVISO: filtro falhou ({type(exc).__name__}: {exc}). "
             "Não é fatal — o treino refaz a medição.")

    _log("FIM.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
