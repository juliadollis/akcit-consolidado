#!/usr/bin/env python3
"""
scripts/zero_shot_spring.py
===========================
Mede o DepthPro CRU (sem fine-tune) no split de TESTE, para dar a linha de base
que faltava no reteste da curvatura.

Por que ele existe
------------------
Todos os bracos do passo 4 (B0 berHu, B1 e B3 com teto 5 e 1000) sao TREINADOS.
Sem o zero-shot nao da para responder se o fine-tuning ajuda em alguma coisa no
Spring, nem se ate o controle B0 ja degrada em relacao ao modelo de prateleira.
O `evaluate_paired.py` tem uma passada zero-shot, mas ela nunca foi rodada aqui:
o `passo4.sh` e o `passo4_teto.sh` chamam so o `train_single.py`.

Comparabilidade
---------------
O numero agregado sai de `Trainer.validate()`, a MESMA funcao que gerou o
`test_metrics.json` de cada seed treinada. Mesmo split, mesmo tamanho (512),
mesmo batch (8), mesma ordem (shuffle=False), mesmo alinhamento afim. A unica
diferenca e que nao houve treino e nenhum `best.pt` e carregado.

Atencao ao agregado: `validate` tira a media das metricas POR BATCH, e o
`mde_metrics` junta os pixels validos do batch inteiro num tensor so. Entao o
agregado NAO e a media das metricas por imagem. As duas coisas sao gravadas em
arquivos separados de proposito, e o numero a comparar com os braços treinados
e o do `test_metrics.json`.

Determinismo: nao ha treino, o modelo fica em eval() e o loader nao embaralha,
entao uma passada basta. Nao existe variacao por seed a reportar.

Uso (dentro do container riemann-depthpro):
    python scripts/zero_shot_spring.py \
        --test-root /data/spring_split/test \
        --checkpoint /models/checkpoints/depth_pro.pt \
        --out-dir /workspace/runs/ZERO_SHOT_spring
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from riemann.losses import RiemannWeights
from riemann.model import build_model
from riemann.dataset import HighQualityDepthDataset
from riemann.metrics import all_metrics
from riemann.trainer import Trainer
from riemann.repro import set_seed, scene_of


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-root", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--variant", default="heads",
                    choices=["heads", "heads_lora", "heads_final"],
                    help="so define o escopo de congelamento; sem treino nenhum "
                         "peso muda, entao nao afeta o numero. Fica igual ao dos "
                         "bracos treinados por simetria.")
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=8,
                    help="TEM de ser o mesmo dos bracos treinados: o agregado e "
                         "media por batch, entao mudar o batch muda o numero.")
    ap.add_argument("--num-workers", type=int, default=16)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out = Path(args.out_dir)
    # Guarda contra sobrescrita, no mesmo espirito do passo4_teto.sh.
    if (out / "test_metrics.json").exists():
        print(f"[zero-shot] ABORTADO: {out}/test_metrics.json ja existe.")
        return 3
    out.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)

    test_ds = HighQualityDepthDataset(args.test_root, size=(args.size, args.size))
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True)
    print(f"[zero-shot] teste: {len(test_ds)} imagens, "
          f"{len(set(scene_of(k) for k in test_ds.keys))} cenas, "
          f"batch={args.batch_size}, size={args.size}")

    model = build_model(args.checkpoint, variant=args.variant, device=args.device)
    # SEM fit(), SEM load_state_dict de best.pt. Os pesos sao os do depth_pro.pt.

    # Pesos da loss irrelevantes aqui: validate() nao usa o criterio. Passamos os
    # do B0 (berHu puro) so para o Trainer construir.
    w = RiemannWeights(berhu=0.7, grad=0.0, normal=0.0, gauss=0.0,
                       geod=0.0, metric=0.0, scale=0.0)
    trainer = Trainer(model, w, device=args.device)

    # ---- 1. agregado, pelo MESMO caminho dos bracos treinados ----------------
    print("[zero-shot] 1/2 agregado via Trainer.validate() ...")
    tmet = trainer.validate(test_loader)
    with open(out / "test_metrics.json", "w") as f:
        json.dump(tmet, f, indent=2)
    print(f"  AbsRel={tmet['abs_rel']:.4f}  bF={tmet['boundary_fscore']:.4f}  "
          f"d1={tmet['d1']:.4f}  RMSE={tmet['rmse']:.4f}")

    # ---- 2. por imagem, para um teste pareado futuro -------------------------
    # Nao substitui o agregado acima; serve para comparar cena a cena quando os
    # bracos treinados forem reavaliados por imagem.
    print("[zero-shot] 2/2 metricas por imagem ...")
    linhas = []
    model.eval()
    with torch.no_grad():
        for batch in test_loader:
            rgb = batch["rgb"].to(args.device)
            gt = batch["depth"].to(args.device)
            mask = batch["mask"].to(args.device)
            pred = model(rgb)
            if pred.shape[-2:] != gt.shape[-2:]:
                pred = torch.nn.functional.interpolate(
                    pred, size=gt.shape[-2:], mode="bilinear", align_corners=False)
            for i, k in enumerate(batch["key"]):
                m = all_metrics(pred[i:i+1], gt[i:i+1], mask[i:i+1])
                linhas.append({"key": k, "cena": scene_of(k), **m})

    campos = ["key", "cena"] + [c for c in linhas[0] if c not in ("key", "cena")]
    with open(out / "por_imagem.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=campos)
        wr.writeheader()
        wr.writerows(linhas)
    print(f"  {len(linhas)} imagens gravadas em por_imagem.csv")

    with open(out / "meta.json", "w") as f:
        json.dump({
            "o_que_e": "DepthPro sem fine-tune (zero-shot) no split de teste",
            "checkpoint": args.checkpoint,
            "test_root": args.test_root,
            "variant": args.variant,
            "size": args.size,
            "batch_size": args.batch_size,
            "seed": args.seed,
            "n_imagens": len(test_ds),
            "n_cenas": len(set(scene_of(k) for k in test_ds.keys)),
            "agregado_de": "Trainer.validate(), mesma funcao dos bracos treinados",
            "treinou": False,
            "carregou_best_pt": False,
        }, f, indent=2)

    print(f"\nConcluido. Artefatos em {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
