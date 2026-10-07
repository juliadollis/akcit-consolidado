#!/usr/bin/env python3
"""
scripts/run_optuna.py
=====================
Busca bayesiana multi-objetivo (Optuna, TPE) dos pesos da Riemannian-Aware Loss.

Dois objetivos SIMULTÂNEOS:
  (1) minimizar AbsRel        — precisão de profundidade
  (2) maximizar boundary F-score — qualidade de borda (liga ao refocusing e ao QC)

A fronteira de Pareto resultante mostra o trade-off. Espelha o protocolo do trabalho
anterior (onde λ_gauss apareceu em 100% das soluções Pareto-ótimas), agora com o
objetivo de borda adicionado.

Uso:
    python scripts/run_optuna.py \\
        --train-root /data/hypersim/train --val-root /data/hypersim/val \\
        --checkpoint /models/checkpoints/depth_pro.pt \\
        --out-dir ./runs/optuna --variant heads \\
        --trials 200 --epochs-per-trial 20
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
from riemann.repro import set_seed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-root", required=True)
    ap.add_argument("--val-root", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out-dir", default="./runs/optuna")
    ap.add_argument("--variant", default="heads", choices=["heads", "heads_final", "heads_lora"])
    ap.add_argument("--trials", type=int, default=40,
                    help="a 1536 cada trial é caro; 40 é realista na H100. Suba se tiver "
                         "tempo de GPU sobrando.")
    ap.add_argument("--epochs-per-trial", type=int, default=12)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--grad-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--num-workers", type=int, default=16)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--monitor", default="boundary_fscore",
                    choices=["boundary_fscore", "abs_rel"],
                    help="métrica de best.pt/early-stop. Padrão boundary_fscore "
                         "(onde o método promete e há headroom).")
    ap.add_argument("--align-mode", choices=["detach", "full", "scale"], default="detach",
                    help="modo do alinhamento afim (experimento hip. 4 do report)")
    ap.add_argument("--max-train", type=int, default=None)
    args = ap.parse_args()

    try:
        import optuna
    except ImportError:
        raise ImportError("Instale Optuna: pip install optuna")

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    train_ds = HighQualityDepthDataset(args.train_root, size=(args.size, args.size),
                                       max_samples=args.max_train)
    val_ds = HighQualityDepthDataset(args.val_root, size=(args.size, args.size))
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)

    def objective(trial):
        # Espaço de busca dos pesos
        w = RiemannWeights(
            berhu=trial.suggest_float("berhu", 0.3, 1.0),
            grad=trial.suggest_float("grad", 0.0, 0.6),
            normal=trial.suggest_float("normal", 0.0, 1.0),
            gauss=trial.suggest_float("gauss", 0.0, 1.0),
            geod=trial.suggest_float("geod", 0.0, 0.3),
            metric=trial.suggest_float("metric", 0.0, 0.5),
            gauss_smooth_sigma=trial.suggest_float("gauss_smooth_sigma", 0.0, 1.0),
            # Termo de escala NÃO entra na busca: a avaliação é afim-invariante, logo
            # otimizar escala absoluta não melhora o alvo e degrada bordas (B200).
            align_mode=args.align_mode,
        )
        # Faixa baseada na varredura da B200: 5e-5 e acima degradam já na 1ª época;
        # 1e-5 foi o melhor. Buscamos em torno disso, não acima.
        lr = trial.suggest_float("lr", 1e-6, 3e-5, log=True)
        wd = trial.suggest_float("wd", 1e-6, 1e-4, log=True)

        # Semear cada trial de forma reprodutível (derivada da seed global + nº do trial)
        set_seed(args.seed + trial.number)

        trial_dir = out / f"trial_{trial.number}"
        model = build_model(args.checkpoint, variant=args.variant, device=args.device,
                            grad_checkpointing=args.grad_checkpointing)
        trainer = Trainer(model, w, device=args.device, lr=lr, weight_decay=wd,
                          grad_accum_steps=args.grad_accum)
        trainer.fit(train_loader, val_loader, epochs=args.epochs_per_trial,
                    out_dir=str(trial_dir),
                    monitor=args.monitor, minimize=(args.monitor == "abs_rel"),
                    early_stop_patience=8)

        # AVALIAR O MELHOR CHECKPOINT, não o estado da última época (bug corrigido):
        # fit() salva best.pt no melhor val_abs_rel. Carregamos antes de medir.
        best_path = trial_dir / "best.pt"
        if best_path.exists():
            model.load_state_dict(torch.load(best_path, map_location=args.device), strict=False)
        val = trainer.validate(val_loader)
        del model, trainer
        if args.device == "cuda":
            torch.cuda.empty_cache()
        # Objetivos: minimizar AbsRel, maximizar boundary F-score
        return val.get("abs_rel", 9.0), val.get("boundary_fscore", 0.0)

    # Estudo multi-objetivo (sem pruner — limitação conhecida do Optuna MO)
    study = optuna.create_study(
        directions=["minimize", "maximize"],
        sampler=optuna.samplers.TPESampler(seed=args.seed),
    )
    study.optimize(objective, n_trials=args.trials)

    # Salvar fronteira de Pareto
    pareto = []
    for t in study.best_trials:
        pareto.append({"number": t.number, "abs_rel": t.values[0],
                       "boundary_fscore": t.values[1], "params": t.params})
    with open(out / "pareto_front.json", "w") as f:
        json.dump(pareto, f, indent=2)

    # Presença de termos nas soluções Pareto-ótimas — COM correção de taxa-base.
    # Ponto da revisão: "gauss em 100% do Pareto" é enganoso se a taxa-base (fração de
    # TODOS os trials com gauss>thr) já for ~95%. Só é evidência se a presença no Pareto
    # for MAIOR que na população geral. Reportamos ambos + o "lift" (razão).
    thr = 0.05
    terms = ["grad", "normal", "gauss", "geod", "metric"]
    all_trials = [t for t in study.trials if t.params]
    base_rate = {}
    for term in terms:
        c = sum(1 for t in all_trials if t.params.get(term, 0) > thr)
        base_rate[term] = c / max(len(all_trials), 1)

    n = max(len(pareto), 1)
    pareto_rate = {}
    for term in terms:
        c = sum(1 for p in pareto if p["params"].get(term, 0) > thr)
        pareto_rate[term] = c / n

    presence = {
        term: {
            "pareto_pct": round(100 * pareto_rate[term], 1),
            "base_pct": round(100 * base_rate[term], 1),
            "lift": round(pareto_rate[term] / base_rate[term], 2) if base_rate[term] > 0 else None,
        } for term in terms
    }
    with open(out / "term_presence.json", "w") as f:
        json.dump(presence, f, indent=2)

    print("\n===== Fronteira de Pareto (top) =====")
    for p in sorted(pareto, key=lambda x: x["abs_rel"])[:10]:
        print(f"  trial {p['number']:3d}: AbsRel={p['abs_rel']:.4f} "
              f"bF={p['boundary_fscore']:.4f}")
    print("\n===== Presença de termos (Pareto vs base — lift>1 = evidência real) =====")
    for term in sorted(terms, key=lambda t: -(presence[t]["lift"] or 0)):
        d = presence[term]
        lift = d["lift"] if d["lift"] is not None else float("nan")
        print(f"  {term:8s}: pareto={d['pareto_pct']:5.1f}%  base={d['base_pct']:5.1f}%  "
              f"lift={lift:.2f}")
    print("  (lift ≈ 1 significa 'igual ao acaso'; só lift claramente > 1 é evidência)")
    print(f"\nArtefatos: {out/'pareto_front.json'}, {out/'term_presence.json'}")


if __name__ == "__main__":
    main()
