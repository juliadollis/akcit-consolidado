#!/usr/bin/env python3
"""Por que o BiRefNet devolve máscara vazia em tantas cenas?

O piloto rejeitou 20,6% das amostras com `focus_mask_empty`, e a rejeição caiu em
**cenas inteiras** (10 de 29; zero cenas com aceite e rejeição misturados) — o que é
esperado, já que a máscara sai da AIF e todos os níveis de uma cena compartilham a AIF.

A pergunta que sobra é outra, e é ela que este script responde: o modelo **não acha
nada**, ou acha e o limiar de 0,5 corta? São diagnósticos opostos.

  - Probabilidade máxima baixa (~0,1) => a cena não tem objeto saliente. Baixar o
    limiar só produziria máscara de ruído, e o descarte está certo.
  - Probabilidade máxima alta (~0,9) com área minúscula => o modelo achou algo e o
    limiar está cortando. Aí o descarte é nosso, não da cena.

Não altera nada. Mede, imprime e grava JSON.
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--mirror-dir", required=True)
    p.add_argument("--raw-dir", required=True)
    p.add_argument("--models-dir", required=True)
    p.add_argument("--output-json", required=True)
    p.add_argument("--scenes", default="", help="cenas separadas por vírgula")
    p.add_argument("--limit-scenes", type=int, default=12)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    from model_runtime import BiRefNetRuntime
    from qc.rejection import RejectionLog
    from sources.mirror_images import MirrorIndex, MirrorImageLoader
    from sources.realbokeh import (MIRROR_DATASET, MIRROR_IMAGE_HW, enumerate_pairs,
                                   load_scene_metadata)

    index = MirrorIndex.build(args.mirror_dir)
    metadata = load_scene_metadata(args.raw_dir)
    pares = enumerate_pairs(index.names(), metadata, log=RejectionLog(),
                            source_dataset=MIRROR_DATASET)

    alvo = {s.strip() for s in args.scenes.split(",") if s.strip()}
    #: um par por cena — a AIF é a mesma em todos os níveis
    por_cena = {}
    for par in pares:
        if alvo and par.scene_id not in alvo:
            continue
        por_cena.setdefault(par.scene_id, par)
    if alvo and not por_cena:
        raise SystemExit(
            f"nenhuma das cenas pedidas existe: {sorted(alvo)}\n"
            "Lembre que `scene_id` é `<split>_<numero>` (ex.: `train_102`), enquanto o "
            "`sample_id` é `c_realbokeh_<scene_id>_l<nivel>`. Não passe o prefixo do "
            "sample_id.")
    escolhidas = list(por_cena.items())[:args.limit_scenes]

    loader = MirrorImageLoader(args.mirror_dir, index, expected_hw=MIRROR_IMAGE_HW)
    runtime = BiRefNetRuntime(Path(args.models_dir) / "BiRefNet", device=args.device)

    linhas = []
    for cena, par in escolhidas:
        aif_bgr, _ = loader(par)
        rgb = np.ascontiguousarray(aif_bgr[..., ::-1])
        prob = np.asarray(runtime.infer_probabilities(rgb), dtype=np.float32)
        areas = {f"area@{t:.2f}": float((prob >= t).mean()) for t in
                 (0.05, 0.10, 0.20, 0.30, 0.50)}
        linhas.append({"scene_id": cena, "sample_id": par.sample_id,
                       "prob_max": float(prob.max()), "prob_p999": float(np.percentile(prob, 99.9)),
                       "prob_mean": float(prob.mean()), **areas})

    largura = max(len(l["scene_id"]) for l in linhas)
    print(f"\n  {'cena':<{largura}}  {'max':>6} {'p99.9':>7} {'média':>7}  "
          + "  ".join(f"{k:>10}" for k in ("area@0.05", "area@0.10", "area@0.20",
                                           "area@0.30", "area@0.50")))
    for l in sorted(linhas, key=lambda x: x["prob_max"]):
        print(f"  {l['scene_id']:<{largura}}  {l['prob_max']:6.3f} {l['prob_p999']:7.3f} "
              f"{l['prob_mean']:7.4f}  " + "  ".join(
                  f"{l[f'area@{t:.2f}']:10.5f}" for t in (0.05, 0.10, 0.20, 0.30, 0.50)))

    vazias_em_050 = [l for l in linhas if l["area@0.50"] == 0.0]
    recuperaveis = [l for l in vazias_em_050 if l["area@0.20"] > 0.005]
    print(f"\n  vazias a 0,50            : {len(vazias_em_050)} de {len(linhas)}")
    print(f"  destas, com área >0,5% a 0,20: {len(recuperaveis)}")
    print("  Se este segundo número for ~0, o modelo não acha nada e o descarte está")
    print("  certo. Se for alto, o limiar é nosso problema, não da cena.\n")

    Path(args.output_json).write_text(
        json.dumps({"scenes": linhas, "empty_at_050": len(vazias_em_050),
                    "recoverable_at_020": len(recuperaveis)}, indent=2),
        encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
