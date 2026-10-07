#!/usr/bin/env python3
"""Garante que TUDO que existe em disco tem gemeo no Hub, por sha256.

Nao apaga nada. So relata o que falta, para nao existir arquivo produzido aqui
que nao tenha copia publicada.
"""
import glob
import hashlib
import os

from huggingface_hub import HfApi

REPOS = {
    "akcit-dephpro/depthpro-riemann-modelos": "model",
    "akcit-h100n1/bokehnet-checkpoints": "model",
    "akcit-h100n1/deblurnet-checkpoints": "model",
    "juliadollis/depthpro-spring-ft": "model",
}
RAIZES = ["/host/runs_riemann", "/host/runs_retreino", "/host/runs_b0_retreino",
          "/host/runs_confirma", "/host/runs_ablacao_spring",
          "/host/genrefocus_deblurnet_paper/outputs",
          "/host/retreinar-deblur/outputs"]


def sha256(c, b=8 * 1024 * 1024):
    h = hashlib.sha256()
    with open(c, "rb") as f:
        for p in iter(lambda: f.read(b), b""):
            h.update(p)
    return h.hexdigest()


def main():
    api = HfApi(token=os.environ["HF_TOKEN"])
    no_hub = {}
    for rid, tipo in REPOS.items():
        try:
            info = api.repo_info(rid, repo_type=tipo, files_metadata=True)
            for s in info.siblings:
                sha = None
                if s.lfs is not None:
                    sha = getattr(s.lfs, "sha256", None) or (
                        s.lfs.get("sha256") if isinstance(s.lfs, dict) else None)
                if sha:
                    no_hub.setdefault(sha, []).append(f"{rid}::{s.rfilename}")
            print(f"  {rid}: {len(info.siblings)} arquivos", flush=True)
        except Exception as e:
            print(f"  {rid}: ERRO {str(e)[:80]}", flush=True)
    print(f"\nsha256 distintos no Hub: {len(no_hub)}\n", flush=True)

    locais = []
    for r in RAIZES:
        for ext in ("*.pt", "*.safetensors", "*.ckpt"):
            locais.extend(glob.glob(f"{r}/**/{ext}", recursive=True))
    locais = sorted(set(locais))
    print(f"pesos em disco: {len(locais)}\n", flush=True)

    faltam, ok = [], 0
    for c in locais:
        s = sha256(c)
        if s in no_hub:
            ok += 1
        else:
            faltam.append((c, os.path.getsize(c)))
    print(f"JA NO HUB : {ok}")
    print(f"FALTANDO  : {len(faltam)}")
    for c, t in sorted(faltam, key=lambda x: -x[1]):
        print(f"   {t/1e9:6.2f} GB  {c}")
    if faltam:
        print(f"\ntotal faltando: {sum(t for _, t in faltam)/1e9:.1f} GB")


if __name__ == "__main__":
    main()
