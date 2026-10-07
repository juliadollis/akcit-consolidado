#!/usr/bin/env python3
"""Apaga best.pt local SOMENTE se existir gemeo byte a byte identico no Hub.

Em 2026-09-10 alguem apertado de espaco apagou 37 checkpoints que NAO estavam no
Hub. A diferenca agora e que estes estao, e este script prova isso por sha256
antes de tocar em qualquer arquivo.

Regras:
  - so apaga arquivo cujo sha256 local bate com o sha256 LFS do Hub
  - NUNCA apaga .json: as metricas sao pequenas e sao o registro irremediavel
  - NUNCA toca em runs_confirma nem runs_ablacao_spring (treino em andamento)
  - sem --executar, so relata

Uso: python3 apaga_seguro.py [--executar]
"""
import hashlib
import os
import sys

from huggingface_hub import HfApi

REPO = "akcit-dephpro/depthpro-riemann-modelos"
MAPA = {
    "/host/runs_riemann": "pesos_originais",
    "/host/runs_retreino": "pesos_retreino",
    "/host/runs_b0_retreino": "pesos_retreino",
}
EXECUTAR = "--executar" in sys.argv


def sha256(caminho, bloco=8 * 1024 * 1024):
    h = hashlib.sha256()
    with open(caminho, "rb") as f:
        for p in iter(lambda: f.read(bloco), b""):
            h.update(p)
    return h.hexdigest()


def main():
    api = HfApi(token=os.environ["HF_TOKEN"])
    info = api.repo_info(REPO, repo_type="model", files_metadata=True)
    no_hub = {}
    for s in info.siblings:
        sha = None
        if s.lfs is not None:
            sha = getattr(s.lfs, "sha256", None) or (
                s.lfs.get("sha256") if isinstance(s.lfs, dict) else None)
        no_hub[s.rfilename] = (sha, s.size)
    print(f"Hub: {len(no_hub)} arquivos em {REPO}\n", flush=True)

    ok, divergem, ausentes, sem_sha = [], [], [], []
    for raiz, prefixo in MAPA.items():
        if not os.path.isdir(raiz):
            continue
        for dirpath, _, nomes in os.walk(raiz):
            for n in nomes:
                if n != "best.pt":
                    continue
                local = os.path.join(dirpath, n)
                rel = os.path.relpath(local, raiz)
                destino = f"{prefixo}/{rel}"
                if destino not in no_hub:
                    ausentes.append((local, destino))
                    continue
                sha_hub, tam_hub = no_hub[destino]
                if not sha_hub:
                    sem_sha.append((local, destino))
                    continue
                if os.path.getsize(local) != tam_hub:
                    divergem.append((local, destino, "tamanho"))
                    continue
                if sha256(local) != sha_hub:
                    divergem.append((local, destino, "sha256"))
                    continue
                ok.append((local, destino, os.path.getsize(local)))
                print(f"  OK {rel}", flush=True)

    gb = sum(t for _, _, t in ok) / 1e9
    print(f"\n{'='*70}")
    print(f"IDENTICOS no Hub (seguros para apagar): {len(ok)}  ->  {gb:.1f} GB")
    print(f"DIVERGEM (NAO apagar)                 : {len(divergem)}")
    for l, d, m in divergem:
        print(f"    {m}: {l}")
    print(f"AUSENTES do Hub (NAO apagar)          : {len(ausentes)}")
    for l, d in ausentes:
        print(f"    {l}  ->  esperado {d}")
    print(f"SEM sha no Hub (NAO apagar)           : {len(sem_sha)}")
    for l, d in sem_sha:
        print(f"    {l}")
    print("=" * 70)

    if not EXECUTAR:
        print("\nMODO SECO: nada foi apagado. Rode com --executar para apagar os IDENTICOS.")
        return

    apagados = 0
    for local, _, _ in ok:
        os.remove(local)
        apagados += 1
    print(f"\nAPAGADOS {apagados} best.pt, {gb:.1f} GB liberados.")
    print("Os .json (metricas) foram mantidos, assim como runs_confirma e a ablacao.")


if __name__ == "__main__":
    main()
