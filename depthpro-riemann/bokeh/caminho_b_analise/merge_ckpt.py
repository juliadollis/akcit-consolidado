import torch, hashlib, sys, collections
from pathlib import Path
B=Path("/raid/user_juliadollis/julia_docker")
base_p = B/"models/checkpoints/depth_pro.pt"
base = torch.load(base_p, map_location="cpu", weights_only=False)
alvos = {"Bgeod_s0": B/"runs_confirma/Bgeod/seed_0/best.pt",
         "Bgrad_s0": B/"runs_confirma/Bgrad/seed_0/best.pt"}
out_dir = B/"caminho_b_analise/ckpt_merged"
for nome, p in alvos.items():
    ft = torch.load(p, map_location="cpu", weights_only=False)
    strip = {k[5:] if k.startswith("core.") else k: v for k,v in ft.items()}
    unexpected = [k for k in strip if k not in base]
    if unexpected:
        raise SystemExit(f"ABORTA: chaves inesperadas em {nome}: {unexpected[:10]}")
    bad = [k for k in strip if tuple(base[k].shape)!=tuple(strip[k].shape)]
    if bad: raise SystemExit(f"ABORTA: shape mismatch em {nome}: {bad[:10]}")
    merged = dict(base)
    merged.update({k: v for k,v in strip.items()})
    assert set(merged)==set(base), "conjunto de chaves divergiu da base"
    dest = out_dir/f"depth_pro_{nome}.pt"
    torch.save(merged, dest)
    h=hashlib.sha256()
    with open(dest,"rb") as f:
        for b in iter(lambda: f.read(1<<20), b""): h.update(b)
    veio_ft = len(strip); veio_base = len(base)-len(strip)
    print(f"{nome}: {dest}")
    print(f"   chaves totais={len(merged)}  do finetune={veio_ft}  da base={veio_base}  unexpected=0  missing=0")
    print(f"   sha256={h.hexdigest()}")
