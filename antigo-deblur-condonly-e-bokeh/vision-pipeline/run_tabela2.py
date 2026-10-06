#!/usr/bin/env python3
"""Reproduz a TABELA 2 do paper (defocus deblurring) — RealDOF + DPDD.

POR QUE ESTA E A UNICA COMPARACAO DIRETA COM O PAPER HOJE
O paper tem 3 tabelas e cada uma usa um dataset diferente:
  Tab. 2  deblurring     -> RealDOF + DPDD   -> DeblurNet   <== reproduzivel
  Tab. 3  bokeh          -> LF-Bokeh         -> BokehNet    <== dataset NAO publico
  Tab. 4  refocusing     -> LF-Refocus       -> pipeline    <== dataset NAO publico
Verificado em 2026-08-20: a org `nycu-cplab` nao publicou nenhum dataset, e nao
existe LF-Bokeh/LF-Refocus no Hugging Face. Por isso a Tab. 2 e o unico numero
que podemos colocar lado a lado com o publicado.

METRICAS: o avaliador de deblur (evaluation/src/deblur_evaluator.py) ja usa as
variantes que o paper reporta — lpips+, dists, clipiqa+, maniqa-kadid, musiq.
(O avaliador de BOKEH usa `lpips` simples, variante diferente; foi por isso que
os numeros de bokeh nao eram comparaveis.)

VALORES PUBLICADOS (Tab. 2), para conferencia:
                LPIPS   DISTS   CLIP-IQA  MANIQA  MUSIQ
  DPDD          0.1440  0.0772  0.4755    0.3452  49.4122
  RealDOF       0.2408  0.1126  0.4595    0.2884  43.5222

Uso:
  python3 run_tabela2.py --modelo nosso     # nosso DeblurNet
  python3 run_tabela2.py --modelo oficial   # peso oficial do paper
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "inference"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from huggingface_hub import hf_hub_download

MODELOS = {
    "oficial": {
        "repo": "nycu-cplab/Genfocus-Model", "arquivo": "deblurNet.safetensors",
        "nome": "DeblurNet-oficial-paper",
        # cond-only, como a inferencia oficial (Inference_deblurNet.py)
        "main_adapter": None,
    },
    "nosso": {
        # ATENCAO: este peso e a variante main+cond (o nome do repo e enganoso,
        # herdado de antes). A inferencia dele exige main_adapter="deblurring";
        # com main_adapter=None a saida sai LAVADA. Ver secao 6 do handoff.
        "repo": "juliadollis/genrefocus-deblurnet-paper-4gpu", "arquivo": "deblur.safetensors",
        "nome": "DeblurNet-nosso-60k",
        # main+cond: SEM isto a saida sai lavada e o numero fica invalido
        "main_adapter": "deblurring",
    },
}
# Repos de saida NOSSOS (o pipeline do time escreve em AkcitPixel2/*, que nao e
# nosso). Nada e sobrescrito.
SAIDAS = {
    "akcit-pixel/RealDOF": "juliadollis/tab2-infer-realdof",
    "akcit-pixel/DDPD": "juliadollis/tab2-infer-ddpd",
}
METRICS_REPO = "juliadollis/tab2-metricas"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--modelo", required=True, choices=sorted(MODELOS))
    p.add_argument("--experimento", default="tab2")
    p.add_argument("--datasets", default="all",
                   help="'all', 'realdof' ou 'ddpd' — para nao re-inferir o que ja esta pronto")
    p.add_argument("--so-metricas", action="store_true",
                   help="pula a inferencia (ja no Hub) e so calcula as metricas")
    args = p.parse_args()

    token = os.environ.get("HF_TOKEN")
    cfg = MODELOS[args.modelo]
    lora = None
    if not args.so_metricas:
        print(f"[setup] baixando {cfg['repo']}/{cfg['arquivo']}", flush=True)
        lora = hf_hub_download(cfg["repo"], cfg["arquivo"], token=token)
    print(f"[setup] LoRA em {lora}", flush=True) if lora else None
    print(f"[setup] main_adapter = {cfg.get('main_adapter')!r}", flush=True)

    os.environ["DEBLUR_EVAL_OUTPUT_REPO"] = METRICS_REPO
    os.environ["DEBLUR_MODEL_NAME"] = cfg["nome"]

    from src.pipelines.deblur_net import (
        run_hf_batch_inference, run_hf_deblur_evaluation,
    )

    alvos = SAIDAS
    if args.datasets != "all":
        chave = {"realdof": "akcit-pixel/RealDOF", "ddpd": "akcit-pixel/DDPD"}[args.datasets]
        alvos = {chave: SAIDAS[chave]}
    for entrada, saida in alvos.items():
        saida = f"{saida}-{args.modelo}"
        print(f"\n=== {cfg['nome']} | {entrada} -> {saida} ===", flush=True)
        if args.so_metricas:
            ok = True
            print("[metricas] pulando a inferencia (--so-metricas)", flush=True)
        else:
            ok = run_hf_batch_inference(
                input_repo_id=entrada,
                validation_pattern="data/validation-*.parquet",
                output_repo_id=saida,
                experiment_name=args.experimento,
                batch_size=50,
                lora_path=lora,
                main_adapter=cfg.get("main_adapter"),
                hf_token=token,
            )
        if not ok:
            print(f"[aviso] inferencia falhou em {entrada}; pulando as metricas", flush=True)
            continue
        run_hf_deblur_evaluation(
            generated_repo_id=saida,
            evaluation_repo_id=METRICS_REPO,
            model_name=cfg["nome"],
            experiment_name=args.experimento,
            hf_token=token,
        )
    print(f"\n=== FIM {cfg['nome']} -> metricas em {METRICS_REPO} ===", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
