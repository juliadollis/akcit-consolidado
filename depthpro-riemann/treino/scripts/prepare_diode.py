#!/usr/bin/env python3
"""
scripts/prepare_diode.py
========================
Organiza o dataset DIODE no layout que o nosso pipeline espera, sem converter nada.

Por que o DIODE dá pouco trabalho: ele já distribui a profundidade em `.npy` float32 em
metros, na mesma resolução do RGB, mais uma máscara de validade do scanner. O nosso
`HighQualityDepthDataset` lê `.npy` nativamente e agora aceita uma pasta `mask/` opcional.
Então este script só percorre a hierarquia, renomeia e cria links.

Estrutura de origem do DIODE:

    <raiz>/<split>/<dominio>/scene_XXXXX/scan_YYYYY/
        00019_00183_indoors_110_000.png              RGB
        00019_00183_indoors_110_000_depth.npy        profundidade (m)
        00019_00183_indoors_110_000_depth_mask.npy   validade (1 = válido)
        00019_00183_indoors_110_000_normal.npy       normais (não usamos)

Estrutura de destino (a mesma do Hypersim, então tudo o mais funciona igual):

    <saida>/rgb/<agrupamento>__<crop>.png
    <saida>/depth/<agrupamento>__<crop>.npy
    <saida>/mask/<agrupamento>__<crop>.npy

O `<agrupamento>` vira a "cena" para toda a estatística pareada (ver `repro.scene_of`,
que separa no `__`). A escolha importa: recortes de um mesmo **scan** vêm da mesma posição
do scanner e são altamente correlacionados, então tratá-los como amostras independentes
inflaria a significância. Por padrão agrupamos por scan; `--agrupar scene` é ainda mais
conservador.

Uso típico (avaliação cruzada, subconjunto interno):

    python scripts/prepare_diode.py \\
        --diode-root /data/diode \\
        --split val --dominio indoors \\
        --out-root /data/diode_prep/val_indoor
"""

import argparse
import os
import shutil
import sys
from collections import Counter
from pathlib import Path

import numpy as np


def achar_amostras(raiz: Path, split: str, dominio: str):
    """
    Percorre <raiz>/<split>/<dominio>/scene_*/scan_*/ e devolve as amostras encontradas.
    Tolerante à ausência do nível <split> ou <dominio>, caso a pasta já venha recortada.
    """
    bases = []
    cand = raiz / split / dominio
    if cand.exists():
        bases.append(cand)
    else:
        alt = raiz / split
        if alt.exists():
            bases.append(alt)
        elif raiz.exists():
            bases.append(raiz)

    amostras = []
    for base in bases:
        for depth_path in sorted(base.rglob("*_depth.npy")):
            stem = depth_path.name[: -len("_depth.npy")]
            pasta = depth_path.parent
            rgb_path = pasta / f"{stem}.png"
            if not rgb_path.exists():
                achou = list(pasta.glob(f"{stem}.*"))
                achou = [p for p in achou if p.suffix.lower() in {".png", ".jpg", ".jpeg"}]
                if not achou:
                    continue
                rgb_path = achou[0]
            mask_path = pasta / f"{stem}_depth_mask.npy"

            # .../scene_XXXXX/scan_YYYYY/arquivo
            scan = pasta.name
            scene = pasta.parent.name
            amostras.append({"stem": stem, "rgb": rgb_path, "depth": depth_path,
                             "mask": mask_path if mask_path.exists() else None,
                             "scene": scene, "scan": scan})
    return amostras


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--diode-root", required=True,
                    help="raiz onde o DIODE foi descompactado")
    ap.add_argument("--out-root", required=True, help="pasta de saida do pipeline")
    ap.add_argument("--split", default="val", choices=["train", "val"])
    ap.add_argument("--dominio", default="indoors",
                    choices=["indoors", "outdoor", "ambos"],
                    help="'indoors' e o padrao: densidade de retorno ~99,6%% contra ~67%% "
                         "no externo, e faixa de profundidade compativel com a calibracao "
                         "atual da loss")
    ap.add_argument("--agrupar", default="scan", choices=["scan", "scene"],
                    help="unidade de agrupamento estatistico. 'scan' = mesma posicao do "
                         "scanner; 'scene' e mais conservador")
    ap.add_argument("--copiar", action="store_true",
                    help="copia os arquivos em vez de criar links simbolicos")
    ap.add_argument("--max-amostras", type=int, default=None)
    ap.add_argument("--max-depth", type=float, default=None,
                    help="descarta amostras cuja mediana valida exceda este valor (m). "
                         "Util para o dominio externo.")
    args = ap.parse_args()

    raiz = Path(args.diode_root)
    out = Path(args.out_root)
    for sub in ("rgb", "depth", "mask"):
        (out / sub).mkdir(parents=True, exist_ok=True)

    dominios = (["indoors", "outdoor"] if args.dominio == "ambos" else [args.dominio])
    amostras = []
    for d in dominios:
        achadas = achar_amostras(raiz, args.split, d)
        print(f"[busca] {args.split}/{d}: {len(achadas)} amostras")
        amostras.extend(achadas)

    if not amostras:
        print(f"ERRO: nada encontrado em {raiz}. Confira a estrutura "
              f"<raiz>/{args.split}/<dominio>/scene_*/scan_*/")
        sys.exit(1)

    if args.max_amostras:
        amostras = amostras[: args.max_amostras]

    ligar = shutil.copy2 if args.copiar else os.symlink
    n_ok = 0
    n_pulado = 0
    grupos = Counter()
    faixas = []

    for a in amostras:
        # O prefixo antes de "__" e exatamente o que `repro.scene_of` vai devolver, entao
        # contamos por ele para o numero reportado bater com o n da estatistica.
        if args.agrupar == "scan":
            grupo = f"{a['scene']}_{a['scan']}"
            chave = f"{grupo}__{a['stem']}"
        else:
            grupo = a["scene"]
            chave = f"{grupo}__{a['scan']}_{a['stem']}"
        chave = chave.replace("/", "_")

        if args.max_depth is not None:
            d = np.load(a["depth"])
            d = d[..., 0] if d.ndim == 3 else d
            val = d[np.isfinite(d) & (d > 0)]
            if val.size and float(np.median(val)) > args.max_depth:
                n_pulado += 1
                continue

        destinos = [(a["rgb"], out / "rgb" / f"{chave}{a['rgb'].suffix}"),
                    (a["depth"], out / "depth" / f"{chave}.npy")]
        if a["mask"] is not None:
            destinos.append((a["mask"], out / "mask" / f"{chave}.npy"))

        for src, dst in destinos:
            if dst.exists() or dst.is_symlink():
                dst.unlink()
            ligar(str(src.resolve()), str(dst))

        grupos[grupo] += 1
        n_ok += 1
        if len(faixas) < 40:
            d = np.load(a["depth"])
            d = d[..., 0] if d.ndim == 3 else d
            v = d[np.isfinite(d) & (d > 0)]
            if v.size:
                faixas.append((float(np.percentile(v, 5)), float(np.percentile(v, 95))))

    print(f"\n[saida] {out}")
    print(f"  amostras preparadas : {n_ok}")
    if n_pulado:
        print(f"  puladas por max-depth: {n_pulado}")
    print(f"  grupos ({args.agrupar}) : {len(grupos)}  <- este e o n efetivo da estatistica")
    if len(grupos) < 8:
        print(f"  [ATENCAO] apenas {len(grupos)} grupos. A comparacao pareada usa o numero")
        print( "            de grupos como tamanho de amostra, entao os intervalos ficarao")
        print( "            largos. Inclua mais scans (ex.: --split train) ou, se estiver")
        print( "            em --agrupar scene, troque para --agrupar scan.")
    if faixas:
        lo = np.median([f[0] for f in faixas])
        hi = np.median([f[1] for f in faixas])
        print(f"  faixa tipica de profundidade (p5-p95): {lo:.2f} - {hi:.2f} m")
        print( "  -> compare com o Hypersim (~1-10 m). Se for muito diferente, revise")
        print( "     gauss_clamp e metric_clamp antes de treinar (ver GUIA_DIODE.md).")
    print(f"  mascaras de validade  : {'sim' if (out/'mask').iterdir() else 'nao'}")
    print(f"\n  distribuicao por grupo (top 5): "
          f"{', '.join(f'{g}={c}' for g, c in grupos.most_common(5))}")


if __name__ == "__main__":
    main()
