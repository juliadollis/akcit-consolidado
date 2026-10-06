#!/usr/bin/env python3
"""
build_bench_ebb.py
==================
Monta a mesa de avaliacao a partir do EBB! (Everything is Better with Bokeh),
no mesmo formato das nossas outras mesas.

POR QUE ESTA MESA
-----------------
A mesa que usavamos para sintese de bokeh (LF-repro, sobre o BLB) tem 500
imagens mas apenas **10 cenas**: cada cena aparece 50 vezes, com 5 aberturas e
10 planos de foco. Com bootstrap agrupado por cena, o tamanho efetivo de amostra
e 10, e medimos que ela nao distingue modelo nenhum -- nem a linha de identidade
separa do resto (p=0,058).

O EBB! resolve isso pela raiz: sao pares FOTOGRAFADOS, um por cena, mesma cena
em f/16 (nitida) e f/1.8 (bokeh forte), Canon 7D com lente 50mm. Uma cena por
linha significa que o n estatistico e o n de imagens.

Ganhos sobre a mesa anterior:
  * optica REAL, nao ray tracing de Blender
  * centenas de cenas em vez de 10
  * fora do treino dos dois modelos: o paper do GenRefocus lista ITW, RealBokeh
    e LFDOF como dado real da fase 2, e o EBB! nao esta la
  * o BokehMe publica numeros no EBB400, construido do mesmo EBB!, entao existe
    linha de base publicada para conferir o pipeline

O que ela NAO da: um sweep de K por cena. Cada cena tem um unico nivel de
desfoque (f/1.8). Para fidelidade e a troca certa; para controlabilidade o
LF-repro continua sendo o unico lugar com alvo conhecido por nivel de K.

O FILTRO, E POR QUE ELE E OBRIGATORIO
-------------------------------------
Nem todo par presta. Medindo a variancia do Laplaciano dos dois lados, aparecem
pares em que o "bokeh" esta MAIS NITIDO que a entrada (razao > 1). Isso e
desalinhamento ou troca de rotulo, e um par assim envenena qualquer metrica de
pixel: pede ao modelo que borre uma imagem cujo alvo esta mais nitido. A mesa
do RealBokeh ja descartava isso, e aqui vale igual.

Uso:
    python3 build_bench_ebb.py --n 400 --saida juliadollis/bokeh-bench-ebb400
"""
import argparse
import io
import os

import cv2
import numpy as np
from datasets import Dataset, Features, Image as HFImage, Value, load_dataset
from PIL import Image

FONTE = "comHannah/bokeh-dataset"   # espelho do EBB!, 4.400 pares


def lap_var(pil, lado=512):
    """Variancia do Laplaciano numa versao reamostrada, para a medida nao
    depender da resolucao original de cada foto."""
    im = pil.convert("RGB")
    w, h = im.size
    s = lado / max(w, h)
    im = im.resize((max(1, int(w * s)), max(1, int(h * s))), Image.LANCZOS)
    g = cv2.cvtColor(np.array(im), cv2.COLOR_RGB2GRAY)
    return float(cv2.Laplacian(g, cv2.CV_64F).var())


def em_bytes(pil, qualidade=95, lado_max=1600):
    im = pil.convert("RGB")
    w, h = im.size
    if max(w, h) > lado_max:
        s = lado_max / max(w, h)
        im = im.resize((int(w * s), int(h * s)), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=qualidade)
    return buf.getvalue()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=400, help="quantas cenas manter")
    ap.add_argument("--razao-max", type=float, default=0.80,
                    help="mantem so pares cujo alvo e ao menos 20%% menos nitido")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--saida", default="juliadollis/bokeh-bench-ebb400")
    ap.add_argument("--so-medir", action="store_true", help="mede e nao publica")
    args = ap.parse_args()
    token = os.environ.get("HF_TOKEN")

    print(f"[ebb] lendo {FONTE}", flush=True)
    d = load_dataset(FONTE, split="train", token=token)
    print(f"[ebb] {len(d)} pares na origem", flush=True)

    # 1) mede a razao de desfoque de TODOS antes de escolher qualquer coisa.
    #    Escolher primeiro e medir depois enviesaria a mesa.
    medidas = []
    for i in range(len(d)):
        try:
            r = d[i]
            la = lap_var(r["original_image"])
            lb = lap_var(r["bokeh_image"])
            medidas.append((i, la, lb, (lb / la) if la > 0 else np.inf))
        except Exception as e:
            print(f"  [falha] par {i}: {str(e)[:80]}", flush=True)
        if (i + 1) % 500 == 0:
            print(f"  medidos {i+1}/{len(d)}", flush=True)

    razoes = np.array([m[3] for m in medidas])
    finitas = razoes[np.isfinite(razoes)]
    print(f"\n[ebb] razao alvo/entrada: mediana {np.median(finitas):.3f}  "
          f"p10 {np.percentile(finitas,10):.3f}  p90 {np.percentile(finitas,90):.3f}")
    print(f"[ebb] pares com alvo MAIS NITIDO que a entrada (razao>1): "
          f"{int((finitas>1).sum())} de {len(finitas)}")

    bons = [m for m in medidas if np.isfinite(m[3]) and m[3] <= args.razao_max]
    print(f"[ebb] passam no filtro (razao <= {args.razao_max}): {len(bons)}")
    if args.so_medir:
        return 0

    # 2) amostra com semente fixa, para a mesa ser reproduzivel
    rng = np.random.default_rng(args.seed)
    escolhidos = sorted(rng.permutation(len(bons))[:args.n].tolist())
    escolhidos = [bons[k] for k in escolhidos]
    print(f"[ebb] mesa final: {len(escolhidos)} cenas (seed {args.seed})")

    linhas = []
    for i, la, lb, razao in escolhidos:
        r = d[i]
        linhas.append({
            "file_name_base": f"ebb_{i:05d}",
            "image_focus": em_bytes(r["original_image"]),
            "image_blur": em_bytes(r["bokeh_image"]),
            "lv_aif": la, "lv_alvo": lb, "razao_desfoque": razao,
            "indice_na_origem": i,
        })
        if len(linhas) % 100 == 0:
            print(f"  montadas {len(linhas)}/{len(escolhidos)}", flush=True)

    feats = Features({
        "file_name_base": Value("string"),
        "image_focus": HFImage(), "image_blur": HFImage(),
        "lv_aif": Value("float32"), "lv_alvo": Value("float32"),
        "razao_desfoque": Value("float32"), "indice_na_origem": Value("int32"),
    })
    ds = Dataset.from_list(linhas, features=feats)
    # O SPLIT TEM DE SE CHAMAR "validation". O pipeline de inferencia procura
    # literalmente `data/validation-*.parquet` (bokeh_net.py, chave
    # "padrao_validacao"), entao um split chamado "test" gera arquivos que ele
    # nao acha, e o item morre com "all the data_files are invalid".
    ds.push_to_hub(args.saida, split="validation", token=token, private=True)
    print(f"[ebb] publicado em {args.saida} (split test, {len(ds)} cenas)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
