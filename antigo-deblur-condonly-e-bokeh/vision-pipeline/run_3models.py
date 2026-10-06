#!/usr/bin/env python3
"""Avaliacao quantitativa do BokehNet comparando 3 modelos.

Reusa o pipeline do time (inference/src/pipelines/bokeh_net.py, do Francisco),
que ja implementa o protocolo do paper:
  - Depth Pro METRICO -> disparidade -> plano de foco;
  - BUSCA BINARIA por K (secao 4.1: "per-image binary search over K ... that
    maximizes SSIM with the target");
  - sweep K em {1,5,10,15} para a LVCorr (Fig. 12);
  - metricas LPIPS / DISTS / CLIP-I / SSIM / LVCorr via CloudBokehEvaluator.

Este runner so orquestra os 3 modelos e manda cada um para um repo HF proprio:

  oficial     nycu-cplab/Genfocus-Model : bokehNet.safetensors  (peso do paper)
  sem-treino  nenhum LoRA                                        (FLUX.1-dev cru)
  nosso       juliadollis/genrefocus-bokehnet-fase2-real : bokeh.safetensors

Uso (1 GPU por processo; escolha com CUDA_VISIBLE_DEVICES):
  python3 run_3models.py --modelo oficial --limite 20
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
        "repo": "nycu-cplab/Genfocus-Model", "arquivo": "bokehNet.safetensors",
        "nome": "GenRefocus-oficial-paper",
        "saida": "juliadollis/bokeh-eval-infer-oficial",
    },
    "sem-treino": {
        "repo": None, "arquivo": None,
        "nome": "FLUX.1-dev-sem-treino",
        "saida": "juliadollis/bokeh-eval-infer-sem-treino",
    },
    "nosso": {
        "repo": "juliadollis/genrefocus-bokehnet-fase2-real", "arquivo": "bokeh.safetensors",
        "nome": "nosso-bokehnet-fase2",
        "saida": "juliadollis/bokeh-eval-infer-nosso",
    },
}

# Repo UNICO de metricas: o evaluator ANEXA uma linha por (modelo, dataset),
# entao os 3 ficam na mesma tabela, que e o que permite comparar.
# Sobrescritivel por env para que uma campanha nova (ex.: o benchmark do
# RealBokeh test) escreva numa tabela PROPRIA, sem misturar com os numeros
# antigos da DPDD, que foram medidos com o pipeline ainda bugado.
METRICS_REPO = os.environ.get("BOKEH_METRICS_REPO", "juliadollis/bokeh-eval-metricas")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--modelo", choices=sorted(MODELOS),
                   help="atalho para um dos 3 modelos da comparacao principal")
    # Modo generico: qualquer checkpoint. Usado pela fila de experimentos
    # (PLANO_EXPERIMENTOS.md) para avaliar a fase 1, steps intermediarios, etc.
    p.add_argument("--lora-repo", help="repo HF do LoRA ('none' = sem LoRA)")
    p.add_argument("--lora-file", default="bokeh.safetensors")
    p.add_argument("--nome", help="nome do modelo na tabela de metricas")
    p.add_argument("--saida", help="repo HF onde gravar as imagens geradas")
    p.add_argument("--dataset-entrada", default="akcit-pixel/DDPD")
    p.add_argument("--padrao", default="data/validation-*.parquet")
    p.add_argument("--split-saida", default="validation")
    p.add_argument("--limite", type=int, default=20, help="0 = todas as imagens")
    p.add_argument("--long-side", type=int, default=512)
    p.add_argument("--lote", type=int, default=10)
    p.add_argument("--k-escala", type=float, default=1.0,
                   help="multiplica a faixa de busca do K (0.01 = busca 0.01..1.0)")
    p.add_argument("--so-avaliar", action="store_true",
                   help="pula a inferencia e so roda as metricas do que ja esta no HF")
    args = p.parse_args()

    token = os.environ.get("HF_TOKEN")
    if args.modelo:
        cfg = MODELOS[args.modelo]
    else:
        if not (args.lora_repo and args.nome and args.saida):
            p.error("use --modelo OU (--lora-repo + --nome + --saida)")
        rp = None if args.lora_repo.lower() == "none" else args.lora_repo
        cfg = {"repo": rp, "arquivo": args.lora_file if rp else None,
               "nome": args.nome, "saida": args.saida}

    caminho_lora = None
    if cfg["repo"]:
        print(f"[setup] baixando LoRA {cfg['repo']}/{cfg['arquivo']}", flush=True)
        caminho_lora = hf_hub_download(cfg["repo"], cfg["arquivo"], token=token)
        print(f"[setup] LoRA em {caminho_lora}", flush=True)

    if not args.so_avaliar:
        from src.pipelines.bokeh_net import rodar_inferencia_em_lotes_hf
        print(f"\n=== INFERENCIA: {cfg['nome']} -> {cfg['saida']} ===", flush=True)
        rodar_inferencia_em_lotes_hf(
            repo_id_entrada=args.dataset_entrada,
            padrao_validacao=args.padrao,
            repo_id_saida=cfg["saida"],
            tamanho_lote=args.lote,
            caminho_completo_lora=caminho_lora,
            token_hf=token,
            long_side=args.long_side,
            limite=args.limite,
            k_escala=args.k_escala,
        )

    # NAO reanexar metricas ja calculadas: o keeper reexecuta a fila em loop
    # para a GPU nunca ficar ociosa, e sem esta guarda a tabela acumula linhas
    # identicas do mesmo (modelo, dataset) a cada volta.
    try:
        from datasets import load_dataset as _ld
        _ja = _ld(METRICS_REPO, split="train", token=token, download_mode="force_redownload")
        for _r in _ja:
            if _r.get("Model") == cfg["nome"] and _r.get("Dataset") == cfg["saida"]:
                print(f"[metricas] {cfg['nome']} ja esta na tabela — pulando.", flush=True)
                print(f"\n=== FIM {cfg['nome']} (nada a fazer) ===", flush=True)
                return 0
    except Exception as _e:
        print(f"[metricas] tabela ainda nao existe ou ilegivel ({type(_e).__name__}) — seguindo.", flush=True)

    from evaluation import CloudBokehEvaluator
    print(f"\n=== METRICAS: {cfg['nome']} ===", flush=True)
    CloudBokehEvaluator().run_pipeline(
        hf_datasets=cfg["saida"],
        output_repo_id=METRICS_REPO,
        model_name=cfg["nome"],
        hf_split=args.split_saida,
        hf_token=token,
    )
    print(f"\n=== FIM {cfg['nome']} -> metricas em {METRICS_REPO} ===", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
