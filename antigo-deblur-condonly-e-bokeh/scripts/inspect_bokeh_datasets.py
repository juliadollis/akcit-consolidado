"""Inspeciona os dfs de bokeh ANTES de escrever o dataloader.

O que precisamos descobrir (cada um muda o código de treino):
  1. Nomes/tipos EXATOS das colunas.
  2. `defocus_map` já vem PRÉ-COMPUTADO? Se sim, não precisamos de depth/K/focus_plane
     em runtime: basta usar o mapa direto como condição 2.
  3. Qual o RANGE do defocus_map (uint8 0..255? float 0..1?). A inferência oficial
     alimenta o VAE com o mapa em [0,1], então precisamos saber como desnormalizar.
  4. Existe coluna de K e de plano de foco? Com que nome?
  5. `depth` é profundidade ou DISPARIDADE (1/depth)? A Eq. 2 usa disparidade.

Não baixa o dataset inteiro (streaming=True). Roda em CPU.

Uso:
  python3 scripts/inspect_bokeh_datasets.py
"""

from __future__ import annotations

import os

import numpy as np
from datasets import load_dataset

DATASETS = [
    "AKCITPixel3/AfONERuvNmglv",   # rota "a"
    "AKCITPixel3/BKXcuVXCmeRvN",   # rota "b" (Flickr, k=50 fixo)
    "AKCITPixel3/CMiQdveBBzNii",   # rota "c" (RealBokeh_3MP, k calibrado)
]
N_ROWS = 8
TOKEN = os.environ.get("HF_TOKEN")


def describe_value(name, v):
    # PIL image?
    if hasattr(v, "size") and hasattr(v, "mode"):
        a = np.asarray(v)
        extra = ""
        if a.size:
            extra = (f" | dtype={a.dtype} shape={a.shape} "
                     f"min={a.min()} max={a.max()} mean={a.mean():.3f}")
        # canais iguais? (o defocus map oficial e' 1 canal repetido 3x)
        if a.ndim == 3 and a.shape[2] == 3:
            same = np.array_equal(a[..., 0], a[..., 1]) and np.array_equal(a[..., 1], a[..., 2])
            extra += f" | 3 canais identicos={same}"
        return f"IMAGE size={v.size} mode={v.mode}{extra}"
    if isinstance(v, (int, float, bool)):
        return f"{type(v).__name__} = {v!r}"
    if isinstance(v, str):
        return f"str = {v[:80]!r}"
    if isinstance(v, (list, tuple)):
        a = np.asarray(v)
        return f"list len={len(v)} dtype={a.dtype} shape={a.shape} min={a.min()} max={a.max()}"
    if v is None:
        return "None"
    return f"{type(v).__name__}"


def main() -> None:
    for name in DATASETS:
        print("=" * 78)
        print(name)
        print("=" * 78)
        try:
            ds = load_dataset(name, split="train", streaming=True, token=TOKEN)
        except Exception as exc:
            print(f"  ERRO ao abrir: {exc}")
            continue

        feats = getattr(ds, "features", None)
        print("--- FEATURES (schema declarado) ---")
        if feats:
            for col, ft in feats.items():
                print(f"  {col:<26} {ft}")
        else:
            print("  (features nao expostas em streaming)")

        print(f"\n--- PRIMEIRAS {N_ROWS} LINHAS ---")
        numeric_acc: dict[str, list[float]] = {}
        for i, row in enumerate(ds):
            if i >= N_ROWS:
                break
            print(f"\n  [linha {i}]")
            for col, v in row.items():
                print(f"    {col:<24} {describe_value(col, v)}")
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    numeric_acc.setdefault(col, []).append(float(v))

        if numeric_acc:
            print(f"\n--- COLUNAS NUMERICAS (range nas {N_ROWS} linhas) ---")
            for col, vals in numeric_acc.items():
                print(f"  {col:<24} min={min(vals):.4f} max={max(vals):.4f} vals={vals[:8]}")
        print()


if __name__ == "__main__":
    main()
