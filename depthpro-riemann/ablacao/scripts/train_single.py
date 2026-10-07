#!/usr/bin/env python3
"""
scripts/train_single.py
=======================
Treina UMA configuração (ex.: a campeã da ablação/Optuna) por mais épocas, com
multi-seed opcional, e exporta os mapas de curvatura Gaussiana ao final.

Uso:
    python scripts/train_single.py \\
        --train-root /data/hypersim/train --val-root /data/hypersim/val \\
        --checkpoint /models/checkpoints/depth_pro.pt \\
        --out-dir ./runs/champion --variant heads \\
        --berhu 0.7 --normal 0.9 --gauss 0.45 \\
        --epochs 100 --seeds 3 --batch-size 8 --export-curvature
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from riemann.losses import RiemannWeights
from riemann.model import build_model
from riemann.dataset import HighQualityDepthDataset
from riemann.trainer import Trainer
from riemann.repro import set_seed, bootstrap_ci


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-root", required=True)
    ap.add_argument("--val-root", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out-dir", default="./runs/champion")
    ap.add_argument("--variant", default="heads", choices=["heads", "heads_lora", "heads_final"])
    # pesos da loss
    ap.add_argument("--berhu", type=float, default=0.7)
    ap.add_argument("--grad", type=float, default=0.0)
    ap.add_argument("--normal", type=float, default=0.9)
    ap.add_argument("--gauss", type=float, default=0.45)
    ap.add_argument("--geod", type=float, default=0.0)
    ap.add_argument("--metric", type=float, default=0.0)
    ap.add_argument("--scale", type=float, default=0.0,
                    help="[NÃO RECOMENDADO] peso do termo de escala absoluta. A avaliação "
                         "é afim-invariante, então otimizar escala absoluta NÃO pode "
                         "melhorar o AbsRel reportado — e na B200 piorou bordas feio "
                         "(F 0.645->0.363). Mantido só para diagnóstico. Deixe em 0.")
    ap.add_argument("--gauss-smooth-sigma", type=float, default=0.5)
    ap.add_argument("--align-mode", choices=["detach", "full", "scale"], default="full",
                    help="modo do alinhamento afim. Padrão 'full' (avaliação é afim-"
                         "invariante). 'detach'/'scale' são só para diagnóstico.")
    ap.add_argument("--monitor", default="boundary_fscore",
                    choices=["boundary_fscore", "abs_rel"],
                    help="métrica que define best.pt e early-stop. PADRÃO boundary_fscore: "
                         "é onde o método promete ganho e onde há headroom (o DepthPro já "
                         "faz AbsRel ~0.11 zero-shot, quase sem espaço). Use abs_rel só se "
                         "o objetivo for precisão métrica.")
    # treino
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=8,
                    help="H100 80GB: 8 em 512px; reduza só se usar 768px")
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--grad-checkpointing", action="store_true",
                    help="opcional na H100; use se treinar em 768px")
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-5,
                    help="1e-5: melhor da varredura na B200. 2e-4 (antigo default) degrada "
                         "o modelo já na 1a epoca — o DepthPro pre-treinado e forte demais "
                         "para LR alto.")
    ap.add_argument("--weight-decay", type=float, default=1e-5)
    ap.add_argument("--num-workers", type=int, default=16)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--export-curvature", action="store_true")
    ap.add_argument("--test-root", default=None,
                    help="conjunto de TESTE separado (held-out), avaliado só no fim, por "
                         "seed. É o número a reportar — evita vazamento da validação usada "
                         "para selecionar configs.")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    train_ds = HighQualityDepthDataset(args.train_root, size=(args.size, args.size))
    val_ds = HighQualityDepthDataset(args.val_root, size=(args.size, args.size))
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)
    test_loader = None
    if args.test_root:
        test_ds = HighQualityDepthDataset(args.test_root, size=(args.size, args.size))
        test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                                 num_workers=args.num_workers, pin_memory=True)

    w = RiemannWeights(berhu=args.berhu, grad=args.grad, normal=args.normal,
                       gauss=args.gauss, geod=args.geod, metric=args.metric,
                       scale=args.scale,
                       gauss_smooth_sigma=args.gauss_smooth_sigma,
                       align_mode=args.align_mode)

    test_absrel, test_bf = [], []  # métricas de TESTE por seed, para IC entre seeds
    for seed in range(args.seeds):
        set_seed(seed)  # semeia Python/NumPy/torch/cuDNN, não só torch
        print(f"\n{'='*60}\nSeed {seed}\n{'='*60}")
        model = build_model(args.checkpoint, variant=args.variant, device=args.device,
                            grad_checkpointing=args.grad_checkpointing)
        trainer = Trainer(model, w, device=args.device,
                          lr=args.lr, weight_decay=args.weight_decay,
                          grad_accum_steps=args.grad_accum)
        seed_dir = out / f"seed_{seed}"
        trainer.fit(train_loader, val_loader, epochs=args.epochs,
                    out_dir=str(seed_dir), monitor=args.monitor,
                    minimize=(args.monitor == "abs_rel"))

        # Carregar o MELHOR checkpoint (só params treináveis → strict=False) antes de
        # qualquer avaliação/exportação.
        best = seed_dir / "best.pt"
        if best.exists():
            model.load_state_dict(
                torch.load(best, map_location=args.device), strict=False)

        # Avaliação no TESTE held-out (número a reportar)
        if test_loader is not None:
            tmet = trainer.validate(test_loader)
            test_absrel.append(tmet.get("abs_rel", float("nan")))
            test_bf.append(tmet.get("boundary_fscore", float("nan")))
            with open(seed_dir / "test_metrics.json", "w") as f:
                json.dump(tmet, f, indent=2)
            print(f"  [TESTE seed {seed}] AbsRel={tmet.get('abs_rel'):.4f} "
                  f"bF={tmet.get('boundary_fscore'):.4f}")

        if args.export_curvature and seed == 0:
            trainer.export_curvature_maps(
                val_loader, out_dir=str(out / "curvature_maps"),
                smooth_sigma=args.gauss_smooth_sigma)

        del model, trainer
        if args.device == "cuda":
            torch.cuda.empty_cache()

    # Resumo estatístico entre seeds (IC por bootstrap) no conjunto de TESTE
    if test_absrel:
        m_ar, lo_ar, hi_ar = bootstrap_ci(test_absrel)
        m_bf, lo_bf, hi_bf = bootstrap_ci(test_bf)
        summary = {
            "n_seeds": args.seeds,
            "test_abs_rel": {"mean": m_ar, "ci95": [lo_ar, hi_ar], "per_seed": test_absrel},
            "test_boundary_fscore": {"mean": m_bf, "ci95": [lo_bf, hi_bf], "per_seed": test_bf},
        }
        with open(out / "test_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\n===== TESTE (held-out) — média ± IC95% entre {args.seeds} seeds =====")
        print(f"  AbsRel          : {m_ar:.4f}  [{lo_ar:.4f}, {hi_ar:.4f}]")
        print(f"  boundary_fscore : {m_bf:.4f}  [{lo_bf:.4f}, {hi_bf:.4f}]")
    else:
        print("\n[aviso] sem --test-root: reportando só val (sujeito a vazamento de seleção).")

    print(f"\nConcluído. Artefatos em {out}")


if __name__ == "__main__":
    main()
