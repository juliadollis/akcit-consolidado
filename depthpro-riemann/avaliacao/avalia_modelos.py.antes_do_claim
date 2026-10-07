#!/usr/bin/env python3
"""
avalia_modelos.py
=================
Avalia N checkpoints em M datasets, produzindo agregado E por imagem.

Por que existe
--------------
A campanha original avaliou cada seed UMA vez, no fim do treino, e guardou so o
agregado. Isso impediu duas coisas que depois fizeram falta: estatistica agrupada
por cena (o teste do Spring tem 485 imagens mas so 13 cenas, e o n efetivo e 13)
e reavaliacao num conjunto corrigido. Este script separa treino de avaliacao: dado
um `best.pt`, mede quantas vezes for preciso, em quantos datasets houver.

Comparabilidade
---------------
O agregado sai do MESMO `Trainer.validate()` que gerou o `test_metrics.json` de
cada seed treinada, com o mesmo batch, entao e diretamente comparavel com os
numeros ja reportados. O batch importa: `validate` tira a media das metricas POR
BATCH e o `mde_metrics` junta os pixels validos do batch inteiro, entao mudar o
batch muda o agregado. Mantenha 8.

O arquivo por imagem NAO e a mesma agregacao: ele mede cada imagem isolada. As
duas coisas convivem de proposito, e o numero a comparar com o historico e o do
`test_metrics.json`.

Uso:
    python3 avalia_modelos.py \
        --checkpoints "/workspace/runs/*/seed_*/best.pt" \
        --dataset spring=/data/spring_split/test \
        --checkpoint-base /models/checkpoints/depth_pro.pt \
        --saida /host/avaliacoes --incluir-zero-shot
"""

import argparse
import csv
import glob
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, "/workspace")
from riemann.dataset import HighQualityDepthDataset
from riemann.losses import RiemannWeights
from riemann.metrics import all_metrics
from riemann.model import build_model
from riemann.repro import scene_of, set_seed
from riemann.trainer import Trainer


def _rotulo(caminho):
    """.../<origem>/<braco>/seed_N/best.pt -> <origem>__<braco>__seed_N

    A origem entra no rotulo de proposito. Os pesos de runs_riemann sao os que
    sobreviveram e correspondem aos numeros do relatorio; os de runs_retreino sao
    execucoes NOVAS da mesma config e seed, que nos bracos com curvatura NAO
    reproduzem o original. Misturar os dois sem rotulo seria erro de leitura.
    """
    p = Path(caminho).parts
    return f"{p[-4]}__{p[-3]}__{p[-2]}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoints", required=True,
                    help="glob dos best.pt. Use aspas para o shell nao expandir.")
    ap.add_argument("--dataset", action="append", required=True, metavar="NOME=RAIZ",
                    help="pode repetir. Ex.: --dataset spring=/data/spring_split/test")
    ap.add_argument("--checkpoint-base", required=True, help="depth_pro.pt oficial")
    ap.add_argument("--saida", required=True)
    ap.add_argument("--variant", default="heads")
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=8,
                    help="TEM de ser o mesmo do treino: o agregado e media por batch")
    ap.add_argument("--num-workers", type=int, default=16)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--incluir-zero-shot", action="store_true",
                    help="avalia tambem o modelo SEM fine-tune, o piso de cada mesa")
    args = ap.parse_args()

    mesas = {}
    for d in args.dataset:
        nome, raiz = d.split("=", 1)
        mesas[nome] = raiz
    ckpts = sorted(glob.glob(args.checkpoints))
    extra = len(mesas) if args.incluir_zero_shot else 0
    print(f"[avalia] {len(ckpts)} checkpoints x {len(mesas)} mesas "
          f"= {len(ckpts) * len(mesas) + extra} avaliacoes", flush=True)

    saida = Path(args.saida)
    saida.mkdir(parents=True, exist_ok=True)
    set_seed(0)

    loaders = {}
    for nome, raiz in mesas.items():
        ds = HighQualityDepthDataset(raiz, size=(args.size, args.size))
        loaders[nome] = (ds, DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                                        num_workers=args.num_workers, pin_memory=True))
        print(f"[avalia] mesa {nome}: {len(ds)} imagens, "
              f"{len({scene_of(k) for k in ds.keys})} cenas", flush=True)

    w = RiemannWeights(berhu=0.7, grad=0.0, normal=0.0, gauss=0.0, geod=0.0, metric=0.0)

    tarefas = [(c, _rotulo(c)) for c in ckpts]
    if args.incluir_zero_shot:
        tarefas.insert(0, (None, "zero_shot"))

    for caminho, rotulo in tarefas:
        pendentes = [n for n in loaders
                     if not (saida / rotulo / n / "test_metrics.json").exists()]
        if not pendentes:
            print(f"[avalia] {rotulo}: todas as mesas ja feitas, pulando", flush=True)
            continue

        model = build_model(args.checkpoint_base, variant=args.variant, device=args.device)
        if caminho is not None:
            model.load_state_dict(torch.load(caminho, map_location=args.device),
                                  strict=False)
        trainer = Trainer(model, w, device=args.device)

        for nome in pendentes:
            ds, dl = loaders[nome]
            dest = saida / rotulo / nome
            dest.mkdir(parents=True, exist_ok=True)

            tmet = trainer.validate(dl)
            with open(dest / "test_metrics.json", "w") as f:
                json.dump(tmet, f, indent=2)

            linhas = []
            model.eval()
            with torch.no_grad():
                for batch in dl:
                    rgb = batch["rgb"].to(args.device)
                    gt = batch["depth"].to(args.device)
                    mask = batch["mask"].to(args.device)
                    pred = model(rgb)
                    if pred.shape[-2:] != gt.shape[-2:]:
                        pred = torch.nn.functional.interpolate(
                            pred, size=gt.shape[-2:], mode="bilinear",
                            align_corners=False)
                    for i, k in enumerate(batch["key"]):
                        m = all_metrics(pred[i:i + 1], gt[i:i + 1], mask[i:i + 1])
                        linhas.append({"key": k, "cena": scene_of(k), **m})
            campos = ["key", "cena"] + [c for c in linhas[0]
                                        if c not in ("key", "cena")]
            with open(dest / "por_imagem.csv", "w", newline="") as f:
                wr = csv.DictWriter(f, fieldnames=campos)
                wr.writeheader()
                wr.writerows(linhas)

            with open(dest / "meta.json", "w") as f:
                json.dump({"checkpoint": caminho or "(sem fine-tune)", "mesa": nome,
                           "raiz": mesas[nome], "n_imagens": len(ds),
                           "n_cenas": len({scene_of(k) for k in ds.keys}),
                           "batch_size": args.batch_size, "size": args.size}, f,
                          indent=2)
            print(f"[avalia] {rotulo} x {nome}: AbsRel={tmet['abs_rel']:.4f} "
                  f"bF={tmet['boundary_fscore']:.4f} ({len(linhas)} imagens)",
                  flush=True)

        del model, trainer
        if args.device == "cuda":
            torch.cuda.empty_cache()

    print("[avalia] FIM", flush=True)


if __name__ == "__main__":
    main()
