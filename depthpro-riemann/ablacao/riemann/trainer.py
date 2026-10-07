"""
riemann/trainer.py
==================
Loop de treino/validação para o fine-tuning do DepthPro com a Riemannian-Aware Loss.
Inclui extração e salvamento dos mapas de curvatura Gaussiana como artefato reutilizável.
"""

from __future__ import annotations
import json
import time
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from torch.utils.data import DataLoader

from .losses import RiemannianAwareLoss, RiemannWeights, gaussian_curvature
from .metrics import all_metrics


def _avg(dicts):
    if not dicts:
        return {}
    keys = dicts[0].keys()
    return {k: float(np.mean([d[k] for d in dicts])) for k in keys}


class Trainer:
    def __init__(self, model, weights: RiemannWeights, device: str = "cuda",
                 lr: float = 2e-4, weight_decay: float = 1e-5,
                 amp: bool = True, grad_accum_steps: int = 1):
        """
        grad_accum_steps: nº de micro-batches acumulados antes de cada passo do
        otimizador. Em GPUs de menor VRAM (ex.: RTX 4090 24GB) usa-se batch pequeno
        (1-2) + grad_accum para simular um batch efetivo maior sem estourar memória.
        batch_efetivo = batch_size × grad_accum_steps.
        """
        self.model = model
        self.device = device
        self.criterion = RiemannianAwareLoss(weights)
        self.opt = torch.optim.AdamW(
            model.trainable_parameters(), lr=lr, weight_decay=weight_decay)
        self.amp = amp and device == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.amp)
        self.grad_accum_steps = max(1, grad_accum_steps)
        self._global_step = 0
        self._nonfinite_steps = 0
        self._total_steps_seen = 0

    def train_epoch(self, loader: DataLoader) -> Dict[str, float]:
        self.model.train()
        parts_log = []
        accum = self.grad_accum_steps
        self.opt.zero_grad(set_to_none=True)

        for i, batch in enumerate(loader):
            rgb = batch["rgb"].to(self.device)
            gt = batch["depth"].to(self.device)
            mask = batch["mask"].to(self.device)

            with torch.amp.autocast("cuda", enabled=self.amp):
                pred = self.model(rgb)
                if pred.shape[-2:] != gt.shape[-2:]:
                    pred = torch.nn.functional.interpolate(
                        pred, size=gt.shape[-2:], mode="bilinear", align_corners=False)
                # Focal do lote. E constante dentro do dataset, entao tomamos o
                # primeiro elemento; -1 sinaliza desconhecida e a loss cai no modo antigo.
                _f = None
                if "fx" in batch:
                    _fx = float(batch["fx"][0]); _fy = float(batch["fy"][0])
                    if _fx > 0 and _fy > 0:
                        _f = (_fx, _fy)
                loss, parts = self.criterion(pred, gt, mask, focal=_f)
                # Escalar a loss pelo nº de passos de acumulação (média dos micro-batches)
                loss_scaled = loss / accum

            # GUARDA DE NaN: se a loss não for finita, NÃO fazemos backward — propagar
            # envenenaria os pesos e todas as épocas seguintes virariam lixo. Pulamos o
            # micro-batch e contamos; se persistir, o fit() aborta a config (ver
            # nonfinite_frac no summary). Isso evita queimar horas de GPU num run morto.
            if not torch.isfinite(loss):
                self._nonfinite_steps += 1
                self.opt.zero_grad(set_to_none=True)
                continue
            self._total_steps_seen += 1

            self.scaler.scale(loss_scaled).backward()
            parts_log.append(parts)

            # Congela as escalas de normalização após N passos: a partir daí o total é um
            # sinal de treino legítimo (desce quando o modelo melhora), em vez de ficar
            # ancorado ~1.0 pela EMA perseguindo o próprio termo.
            self._global_step += 1
            fs = getattr(self.criterion.w, "norm_freeze_steps", 0)
            if fs and self._global_step >= fs:
                self.criterion.freeze_normalization()

            # Passo do otimizador a cada `accum` micro-batches
            if (i + 1) % accum == 0:
                self.scaler.step(self.opt)
                self.scaler.update()
                self.opt.zero_grad(set_to_none=True)

        # Flush: se sobraram micro-batches não múltiplos de accum
        if (len(loader) % accum) != 0:
            self.scaler.step(self.opt)
            self.scaler.update()
            self.opt.zero_grad(set_to_none=True)

        return _avg(parts_log)

    def _save_trainable(self, path):
        """
        Salva só os parâmetros treináveis (cabeças/LoRA). O backbone congelado vem
        do checkpoint base — salvá-lo repetiria ~GB por config/trial. Carregar com
        load_state_dict(..., strict=False) sobre um modelo recém-construído.
        """
        trainable = {n for n, p in self.model.named_parameters() if p.requires_grad}
        sd = {k: v for k, v in self.model.state_dict().items() if k in trainable}
        torch.save(sd, path)

    @torch.no_grad()
    def validate(self, loader: DataLoader) -> Dict[str, float]:
        self.model.eval()
        met_log = []
        for batch in loader:
            rgb = batch["rgb"].to(self.device)
            gt = batch["depth"].to(self.device)
            mask = batch["mask"].to(self.device)
            pred = self.model(rgb)
            if pred.shape[-2:] != gt.shape[-2:]:
                pred = torch.nn.functional.interpolate(
                    pred, size=gt.shape[-2:], mode="bilinear", align_corners=False)
            met_log.append(all_metrics(pred, gt, mask))
        return _avg(met_log)

    def fit(self, train_loader, val_loader, epochs: int,
            out_dir: str, early_stop_patience: int = 25,
            warmup_epochs: int = 5, monitor: str = "abs_rel",
            minimize: bool = True) -> Dict:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        history = []
        best = float("inf") if minimize else -float("inf")
        best_epoch = -1
        patience = 0

        for ep in range(1, epochs + 1):
            t0 = time.time()
            train_parts = self.train_epoch(train_loader)
            val_met = self.validate(val_loader)
            dt = time.time() - t0

            row = {"epoch": ep, "time_s": round(dt, 1),
                   **{f"train_{k}": v for k, v in train_parts.items()},
                   **{f"val_{k}": v for k, v in val_met.items()}}
            history.append(row)
            print(f"[{ep:03d}/{epochs}] "
                  f"train_raw={train_parts.get('raw_total', float('nan')):.4f} "
                  f"train_total={train_parts.get('total', float('nan')):.4f} "
                  f"val_absrel={val_met.get('abs_rel', float('nan')):.4f} "
                  f"val_bF={val_met.get('boundary_fscore', float('nan')):.4f} "
                  f"({dt:.0f}s)")

            cur = val_met.get(monitor, float("inf"))

            # ABORTO POR DIVERGÊNCIA: se a maior parte dos micro-batches da época foi
            # não-finita, a config está morta — abortamos em vez de queimar as épocas
            # restantes. O run é marcado como divergido no summary para o CSV.
            seen = self._nonfinite_steps + self._total_steps_seen
            frac_bad = self._nonfinite_steps / max(seen, 1)
            if frac_bad > 0.5:
                print(f"  [DIVERGIU] {100*frac_bad:.0f}% dos micro-batches não-finitos — "
                      f"abortando esta config na época {ep} (economiza GPU).")
                return {"best_metric": float("nan"), "best_epoch": -1,
                        "monitor": monitor, "history": history,
                        "diverged": True, "nonfinite_frac": frac_bad}

            improved = (cur < best) if minimize else (cur > best)
            if ep > warmup_epochs and improved:
                best = cur
                best_epoch = ep
                patience = 0
                self._save_trainable(out / "best.pt")
            elif ep > warmup_epochs:
                patience += 1
                if patience >= early_stop_patience:
                    print(f"  Early stop na época {ep} (best={best:.4f} @ {best_epoch})")
                    break

        with open(out / "history.json", "w") as f:
            json.dump(history, f, indent=2)
        summary = {"best_metric": best, "best_epoch": best_epoch,
                   "monitor": monitor, "history_len": len(history)}
        with open(out / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        return summary

    @torch.no_grad()
    def export_curvature_maps(self, loader: DataLoader, out_dir: str,
                              smooth_sigma: float = 0.5, clamp_val: float = 50.0):
        """
        Gera e salva os mapas de curvatura Gaussiana do depth predito.
        Artefato reutilizável: marca dobras/oclusões (bordas críticas p/ o bokeh).
        Salva PNG 16-bit (visualização) + .npy (valores).
        """
        self.model.eval()
        od = Path(out_dir)
        od.mkdir(parents=True, exist_ok=True)
        try:
            import cv2
            has_cv2 = True
        except ImportError:
            has_cv2 = False

        for batch in loader:
            rgb = batch["rgb"].to(self.device)
            keys = batch["key"]
            pred = self.model(rgb)
            K = gaussian_curvature(pred, smooth_sigma, clamp_val)  # (B,1,H,W)
            for i, key in enumerate(keys):
                k = K[i, 0].cpu().numpy()
                np.save(od / f"{key}_curvature.npy", k.astype(np.float32))
                # visualização normalizada [0,1] -> 16-bit
                kn = (k - k.min()) / (k.max() - k.min() + 1e-6)
                if has_cv2:
                    cv2.imwrite(str(od / f"{key}_curvature.png"),
                                (kn * 65535).astype(np.uint16))
        print(f"[Curvatura] mapas salvos em {od}")
