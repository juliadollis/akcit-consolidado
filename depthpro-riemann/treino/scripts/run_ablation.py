#!/usr/bin/env python3
"""
scripts/run_ablation.py
=======================
Roda a ablação EXPANDIDA (39 configs × variantes de modelo) do fine-tuning do DepthPro
com a Riemannian-Aware Loss.

Cada config treina por `--epochs` épocas e registra as métricas de validação. Resultados
agregados vão para um CSV ordenável, permitindo identificar qual geometria domina.

Uso:
    python scripts/run_ablation.py \\
        --train-root /data/hypersim/train \\
        --val-root   /data/hypersim/val \\
        --checkpoint /models/checkpoints/depth_pro.pt \\
        --out-dir    ./runs/ablation \\
        --variant heads \\
        --epochs 40 --batch-size 4

    # rodar só um subconjunto de blocos (ex.: B1 e B7):
    python scripts/run_ablation.py ... --filter B1 B7
"""

import argparse
import csv
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from riemann.losses import RiemannWeights
from riemann.model import build_model
from riemann.dataset import HighQualityDepthDataset
from riemann.trainer import Trainer
from riemann.ablation_configs import build_ablation_configs
from riemann.repro import set_seed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-root", required=True)
    ap.add_argument("--val-root", required=True)
    ap.add_argument("--checkpoint", required=True, help="depth_pro.pt")
    ap.add_argument("--out-dir", default="./runs/ablation")
    ap.add_argument("--variant", default="heads", choices=["heads", "heads_lora", "heads_final"])
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=2,
                    help="input do DepthPro é fixo em 1536 (caro); batch 2 cabe na H100 "
                         "com grad-checkpointing. Suba com cautela.")
    ap.add_argument("--grad-accum", type=int, default=4,
                    help="batch efetivo = batch_size × grad_accum (compensa batch pequeno)")
    ap.add_argument("--grad-checkpointing", action=argparse.BooleanOptionalAction, default=True,
                    help="LIGADO por padrão: a 1536 a memória aperta mesmo na H100")
    ap.add_argument("--size", type=int, default=512,
                    help="resolução de TRABALHO da loss/GT (o input do modelo é sempre "
                         "1536). 512 é um bom equilíbrio custo/qualidade de borda.")
    ap.add_argument("--lr", type=float, default=1e-5,
                    help="1e-5: melhor da varredura na B200. 2e-4 (antigo default) degrada "
                         "o modelo já na 1a epoca — o DepthPro pre-treinado e forte demais "
                         "para LR alto.")
    ap.add_argument("--weight-decay", type=float, default=1e-5)
    ap.add_argument("--num-workers", type=int, default=16)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--filter", nargs="*", default=["B0", "B1", "B7"],
                    help="Prefixos de bloco a incluir. Default: B0 B1 B7 (os mais "
                         "informativos p/ a hipótese central — ~10 configs, viável na "
                         "H100). Use --filter '' para rodar TODAS as 39 (dias de GPU).")
    ap.add_argument("--max-train", type=int, default=None, help="limitar amostras (debug)")
    ap.add_argument("--seed", type=int, default=42,
                    help="semente global (reprodutibilidade — a ablação antes não semeava)")
    ap.add_argument("--eval-zero-shot", action=argparse.BooleanOptionalAction, default=True,
                    help="mede o modelo sem fine-tune como 1a linha do CSV, no MESMO "
                         "protocolo — referência honesta para todas as comparações.")
    ap.add_argument("--monitor", default="boundary_fscore",
                    choices=["boundary_fscore", "abs_rel"],
                    help="métrica de best.pt/early-stop. Padrão boundary_fscore "
                         "(onde o método promete e há headroom).")
    ap.add_argument("--align-mode", choices=["detach", "full", "scale"], default="detach",
                    help="modo do alinhamento afim (experimento hip. 4 do report)")
    ap.add_argument("--tf32", action="store_true",
                    help="B200: TF32 acelera matmul (inclui a loss fp32). Precisão ~1e-3 "
                         "menor — VALIDAR contra --no-tf32 antes de usar em produção.")
    args = ap.parse_args()

    set_seed(args.seed)

    if args.tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Datasets (carregados uma vez, reutilizados por todas as configs)
    train_ds = HighQualityDepthDataset(args.train_root, size=(args.size, args.size),
                                       max_samples=args.max_train)
    val_ds = HighQualityDepthDataset(args.val_root, size=(args.size, args.size))
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)

    configs = build_ablation_configs()
    # --filter '' (string vazia) significa "rodar TODAS as configs".
    active_filters = [f for f in (args.filter or []) if f]
    if active_filters:
        configs = [c for c in configs if any(c["name"].startswith(f) for f in active_filters)]
    print(f"[Ablação] {len(configs)} configs | variante={args.variant} | "
          f"train={len(train_ds)} val={len(val_ds)}")

    results = []

    # ---- Linha de referência ZERO-SHOT (sem fine-tune) -----------------------------
    # Mede o DepthPro pré-treinado no MESMO protocolo (mesmo val_loader, mesmo
    # alinhamento, mesmas métricas). Sem isto, toda comparação depende de um número
    # medido à parte, e fica sempre a dúvida se o baseline é comparável. Custa uma
    # passada de validação.
    if args.eval_zero_shot:
        print("\n[Referência] avaliando o modelo ZERO-SHOT (sem fine-tune)...")
        zs_model = build_model(args.checkpoint, variant=args.variant, device=args.device,
                               grad_checkpointing=args.grad_checkpointing)
        zs_trainer = Trainer(zs_model, RiemannWeights(berhu=1.0), device=args.device)
        zs_val = zs_trainer.validate(val_loader)
        results.append({"config": "ZERO_SHOT_sem_finetune", "variant": args.variant,
                        "best_epoch": 0, "status": "referencia", **zs_val})
        print(f"  zero-shot: AbsRel={zs_val.get('abs_rel', float('nan')):.4f} "
              f"bF={zs_val.get('boundary_fscore', float('nan')):.4f} "
              f"bPrec={zs_val.get('boundary_precision', float('nan')):.4f} "
              f"bRecall={zs_val.get('boundary_recall', float('nan')):.4f}")
        _write_csv(out / "ablation_results.csv", results)
        del zs_model, zs_trainer
        if args.device == "cuda":
            torch.cuda.empty_cache()
    for i, cfg in enumerate(configs, 1):
        name = cfg["name"]
        print(f"\n{'='*70}\n[{i}/{len(configs)}] {name}\n{'='*70}")
        w = RiemannWeights(**cfg["weights"], align_mode=args.align_mode)

        # Rebuild do modelo por config (estado limpo)
        model = build_model(args.checkpoint, variant=args.variant, device=args.device,
                            grad_checkpointing=args.grad_checkpointing)
        trainer = Trainer(model, w, device=args.device,
                          lr=args.lr, weight_decay=args.weight_decay,
                          grad_accum_steps=args.grad_accum)

        run_dir = out / f"{args.variant}__{name}"
        summary = trainer.fit(train_loader, val_loader, epochs=args.epochs,
                              out_dir=str(run_dir), monitor=args.monitor,
                              minimize=(args.monitor == "abs_rel"))

        # Reavaliar o melhor checkpoint para pegar todas as métricas
        best_path = run_dir / "best.pt"
        if best_path.exists():
            model.load_state_dict(torch.load(best_path, map_location=args.device), strict=False)
        final_val = trainer.validate(val_loader)

        row = {"config": name, "variant": args.variant,
               "best_epoch": summary["best_epoch"],
               "status": "DIVERGIU" if summary.get("diverged") else "ok",
               **final_val}
        results.append(row)

        # Salvar CSV incremental (resiliente a interrupções)
        _write_csv(out / "ablation_results.csv", results)
        del model, trainer
        if args.device == "cuda":
            torch.cuda.empty_cache()

    # Ordenar por AbsRel e por boundary_fscore
    print("\n\n===== TOP por AbsRel =====")
    for r in sorted(results, key=lambda x: x.get("abs_rel", 9))[:10]:
        print(f"  {r['config']:38s} AbsRel={r.get('abs_rel',float('nan')):.4f} "
              f"bF={r.get('boundary_fscore',float('nan')):.4f}")
    print("\n===== TOP por boundary F-score =====")
    print("  (bRecall é o sinal discriminante: o berHu puro sobe precisão e DERRUBA "
          "recall — achata bordas. Os termos geométricos devem recuperar recall.)")
    for r in sorted(results, key=lambda x: -x.get("boundary_fscore", 0))[:10]:
        print(f"  {r['config']:38s} bF={r.get('boundary_fscore',float('nan')):.4f} "
              f"bPrec={r.get('boundary_precision',float('nan')):.4f} "
              f"bRecall={r.get('boundary_recall',float('nan')):.4f} "
              f"AbsRel={r.get('abs_rel',float('nan')):.4f}")
    print(f"\nResultados completos: {out/'ablation_results.csv'}")


def _write_csv(path, rows):
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
