"""Prefetch RESUMÍVEL de um dataset de bokeh do HF para uma pasta local,
redimensionando para lado-menor 512 no caminho.

POR QUE ISSO EXISTE: a rota a (AKCITPixel3/AfONERuvNmglv) tem 341.6 GB no hub
(medido via datasets-server /size). O cache do `datasets` (download + arrow)
dobraria isso — não cabe na cota de 500 GB do cluster. Mas o treino roda em
512²: o dataloader reduz o lado menor de TODA imagem para image_size antes do
crop. Então este script faz um único passe em STREAMING (nada de cache local),
aplica AQUI o mesmo resize que o dataloader aplicaria (mesma fórmula, mesmos
filtros), e salva PNGs pequenos. 341 GB viram ~40-100 GB, e o dataloader local
(genfocus_train/data.py::LocalBokehFolderDataset) pula o resize (no-op).

FIDELIDADE: a fórmula de target é IDÊNTICA à de prepare_aligned_bokeh
(data.py) — scale = S/min(w,h) derivado da AIF; BICUBIC para aif/bokeh;
BILINEAR (em float) para depth. Como o pipeline original era
original --BICUBIC--> (new_w,new_h) --crop--> treino, e o novo é
original --BICUBIC--> (new_w,new_h) [aqui] --no-op--> --crop--> treino,
o tensor que chega no treino é o MESMO (única perda: re-quantização do depth
para uint16, erro <= 1/65535 — a mesma ordem do ruído já presente no df).

Salva: <out>/aif/NNNNNNN_<stem>.png (RGB), <out>/bokeh/... (RGB),
       <out>/depth/... (PNG 16-bit), <out>/metadata.jsonl (1 linha por amostra:
       stem, k, s1, arquivos). NÃO salva defocus_map (o treino recomputa de
       depth+k+s1 — ver DEFOCUS_SOURCES em data.py).

RESUME: conta as linhas do metadata.jsonl e faz ds.skip(n). Pode matar e
relançar à vontade. Roda no NÓ DE LOGIN (sem GPU, sem fila):

  cd /raid/user_juliadollis/projects/genrefocus_deblurnet_paper
  set -a; . .env; set +a
  nohup singularity exec --bind /raid/user_juliadollis:/workspace \
    --env HF_TOKEN=$HF_TOKEN \
    /raid/user_juliadollis/images/transformers-pytorch-gpu.sif \
    python3 scripts/prefetch_bokeh_local.py \
      --dataset AKCITPixel3/AfONERuvNmglv \
      --out /workspace/data-bokeh/rota_a \
      > logs/prefetch_rota_a.log 2>&1 &
  tail -f logs/prefetch_rota_a.log
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
from PIL import Image

DEPTH_U16 = 65535.0


def target_dims(w: int, h: int, min_side: int) -> tuple[int, int]:
    """MESMA fórmula de prepare_aligned_bokeh (data.py). Mudou lá? Mude aqui."""
    scale = min_side / min(w, h)
    new_w = max(min_side, round(w * scale))
    new_h = max(min_side, round(h * scale))
    return new_w, new_h


def to_rgb(img) -> Image.Image:
    if isinstance(img, Image.Image):
        return img.convert("RGB")
    return Image.fromarray(np.asarray(img)).convert("RGB")


def depth_to_float01(img) -> np.ndarray:
    """PIL I;16 (uint16) ou array -> float32 [0,1]. Mesma leitura do dataloader."""
    arr = np.asarray(img)
    if arr.ndim == 3:
        arr = arr[..., 0]
    arr = arr.astype(np.float32)
    if arr.max() > 1.5:  # veio em uint16/uint8 cru
        arr = arr / (DEPTH_U16 if arr.max() > 255.5 else 255.0)
    return arr


def process_row(row, idx: int, out: str, min_side: int) -> dict:
    aif = to_rgb(row["aif"])
    bokeh = to_rgb(row["bokeh"])
    depth01 = depth_to_float01(row["depth"])

    w, h = aif.size
    new_w, new_h = target_dims(w, h, min_side)

    aif_r = aif.resize((new_w, new_h), Image.BICUBIC)
    bokeh_r = bokeh.resize((new_w, new_h), Image.BICUBIC)
    dep_r = Image.fromarray(depth01, mode="F").resize((new_w, new_h), Image.BILINEAR)
    dep_u16 = (np.clip(np.asarray(dep_r, dtype=np.float32), 0.0, 1.0) * DEPTH_U16).astype(np.uint16)

    stem_raw = str(row.get("stem", "")) or "row"
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in stem_raw)[:48]
    # índice na frente: stems da rota a NÃO são únicos (mesma AIF, k/s1 diferentes)
    name = f"{idx:07d}_{safe}.png"

    aif_r.save(os.path.join(out, "aif", name))
    bokeh_r.save(os.path.join(out, "bokeh", name))
    Image.fromarray(dep_u16).save(os.path.join(out, "depth", name))  # PNG I;16

    return {
        "idx": idx,
        "stem": stem_raw,
        "file": name,
        "k": float(row["k"]),
        "s1": float(row["s1"]),
        "orig_w": w,
        "orig_h": h,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dataset", required=True, help="repo HF, ex. AKCITPixel3/AfONERuvNmglv")
    ap.add_argument("--out", required=True, help="pasta local de saída")
    ap.add_argument("--split", default="train")
    ap.add_argument("--min-side", type=int, default=512)
    ap.add_argument("--limit", type=int, default=0, help="parar após N amostras NOVAS (0 = tudo)")
    args = ap.parse_args()

    token = os.environ.get("HF_TOKEN")
    if not token:
        print("ERRO: exporte HF_TOKEN (ex.: set -a; . .env; set +a)", file=sys.stderr)
        return 2

    for sub in ("aif", "bokeh", "depth"):
        os.makedirs(os.path.join(args.out, sub), exist_ok=True)
    meta_path = os.path.join(args.out, "metadata.jsonl")

    done = 0
    if os.path.isfile(meta_path):
        with open(meta_path, "r", encoding="utf-8") as fh:
            done = sum(1 for _ in fh)
    print(f"[prefetch] {args.dataset} -> {args.out}  (ja prontas: {done}; resume automatico)",
          flush=True)

    from datasets import load_dataset  # import tardio: erro de dep fica claro no log

    ds = load_dataset(args.dataset, split=args.split, streaming=True, token=token)
    if done:
        ds = ds.skip(done)

    t0 = time.time()
    new = 0
    with open(meta_path, "a", encoding="utf-8") as meta:
        for i, row in enumerate(ds):
            idx = done + i
            try:
                rec = process_row(row, idx, args.out, args.min_side)
            except Exception as exc:  # noqa: BLE001 — 1 amostra ruim não mata 68K
                print(f"[prefetch] WARN idx={idx}: {type(exc).__name__}: {exc}", flush=True)
                rec = {"idx": idx, "stem": "SKIPPED", "file": None, "error": str(exc)[:200]}
            meta.write(json.dumps(rec, ensure_ascii=False) + "\n")
            meta.flush()
            new += 1
            if new % 100 == 0:
                rate = new / max(time.time() - t0, 1e-9)
                print(f"[prefetch] {idx + 1} amostras ({rate:.1f}/s)", flush=True)
            if args.limit and new >= args.limit:
                print(f"[prefetch] --limit {args.limit} atingido.", flush=True)
                break

    print(f"[prefetch] FIM: +{new} novas (total {done + new}) em {(time.time() - t0) / 60:.1f} min",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
