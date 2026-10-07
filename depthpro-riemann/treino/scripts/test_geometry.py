#!/usr/bin/env python3
"""
scripts/test_geometry.py
Teste de sanidade da geometria: valida a curvatura Gaussiana em superfícies
analíticas de curvatura conhecida. Rode ANTES de treinar para garantir que a
loss está numericamente correta no seu ambiente.
"""
import sys
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from riemann.losses import gaussian_curvature

def main():
    H = W = 96
    R = 2.0
    xs = torch.linspace(-1, 1, W); ys = torch.linspace(-1, 1, H)
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    h = 2.4 / (W - 1)  # espaçamento físico (coords vão de -1.2 a 1.2)

    # Esfera: K = 1/R²
    rr = (gx * 1.2) ** 2 + (gy * 1.2) ** 2
    sphere = torch.sqrt(torch.clamp(R**2 - rr, min=0.25)).view(1, 1, H, W)
    Ks = gaussian_curvature(sphere, smooth_sigma=0, h=h)[0, 0, H//2, W//2]
    print(f"Esfera R={R}: K={Ks:.4f}  (teórico {1/R**2:.4f})")
    assert abs(float(Ks) - 1/R**2) < 0.05, "FALHA: curvatura da esfera incorreta"

    # Plano: K = 0
    plane = (0.3 * gx * 1.2 + 0.2 * gy * 1.2).view(1, 1, H, W)
    Kp = gaussian_curvature(plane, smooth_sigma=0, h=h)[:, :, 4:-4, 4:-4].abs().mean()  # interior (evita moldura do padding)
    print(f"Plano: |K|médio={Kp:.2e}  (esperado ~0)")
    assert float(Kp) < 1e-2, "FALHA: plano deveria ter K=0"

    print("\n✅ Geometria OK — a curvatura Gaussiana está numericamente correta.")


def test_regression_fixes():
    """
    Testes de regressão dos bugs corrigidos na revisão (resolução, fp16, sqrt, afim).
    Rode junto do teste de geometria para garantir que as correções seguem válidas.
    """
    import torch
    from riemann.losses import (robust_affine_align, metric_tensor_loss,
                                gaussian_curvature)

    # 1) Alinhamento afim casa pred reescalada (corrige inversão + escala)
    gt = torch.linspace(1, 5, 64).view(1, 1, 1, 64).repeat(1, 1, 64, 1)
    pred_scaled = 3.0 * gt + 10.0
    aligned = robust_affine_align(pred_scaled, gt, torch.ones_like(gt))
    assert (aligned - gt).abs().mean() < 1e-3, "afim não casou pred reescalada"

    # 2) metric_tensor_loss tem gradiente finito em região plana (sqrt+eps)
    flat = (torch.ones(1, 1, 48, 48) * 3.0).requires_grad_(True)
    lm = metric_tensor_loss(flat, flat * 1.001)
    lm.backward()
    assert torch.isfinite(flat.grad).all(), "metric loss gerou grad não-finito em plano"

    # 3) curvatura clampeada é finita (não estoura em fp32)
    K = gaussian_curvature(torch.rand(1, 1, 128, 128), smooth_sigma=0.5, clamp_val=50.0)
    assert torch.isfinite(K).all(), "curvatura gerou não-finito"

    print("✅ Regressão OK — afim, metric-loss em plano e curvatura estão estáveis.")


def test_term_balance():
    """
    Regressão do problema reportado: com depth métrico, termos em unidades diferentes
    (metric ~10³, curvatura ~10¹, berHu ~10⁰) faziam a loss explodir. A normalização
    adaptativa deve trazer cada termo à proporção do seu peso.
    """
    import torch
    from riemann.losses import RiemannianAwareLoss, RiemannWeights

    gt = torch.linspace(1, 6, 128).view(1, 1, 1, 128).repeat(2, 1, 128, 1).clone()
    gt[:, :, 40:88, 40:88] += 2.0
    pred = 0.3 * gt + 5.0
    w = RiemannWeights(berhu=0.7, normal=0.9, gauss=0.45, metric=0.2)
    crit = RiemannianAwareLoss(w)
    for _ in range(6):  # deixar a EMA estabilizar
        _, parts = crit(pred, gt, None)
    vals = {k: v for k, v in parts.items()
            if k != "total" and not k.startswith("raw_")}
    ratio = max(vals.values()) / min(vals.values())
    assert ratio < 20, f"termos desbalanceados (razão {ratio:.0f}x) — normalização falhou"
    print(f"✅ Balanceamento OK — razão max/min entre termos = {ratio:.1f}x (era ~6000x).")


def test_scale_term():
    """
    Regressão da hipótese 4: o termo de escala (sem alinhamento) deve dar gradiente
    não-nulo para uma predição com escala errada — o sinal que os termos alinhados
    descartam. Sem ele, dLoss/da ≈ 0 (loss cega à escala).
    """
    import torch
    from riemann.losses import scale_loss, berhu_loss, robust_affine_align

    gt = torch.linspace(1, 5, 32).view(1, 1, 1, 32).repeat(1, 1, 32, 1).contiguous()

    # termo de escala: gradiente vivo para escala 2x errada
    a = torch.tensor(0.5, requires_grad=True)
    g_scale = torch.autograd.grad(scale_loss(a * gt, gt), a)[0]
    assert abs(float(g_scale)) > 1e-3, "scale_loss não recuperou o sinal de escala"

    # termo alinhado: gradiente ~0 para o mesmo caso (confirma a cegueira)
    a2 = torch.tensor(0.5, requires_grad=True)
    aligned = robust_affine_align(a2 * gt, gt, torch.ones_like(gt), mode="full")
    g_aligned = torch.autograd.grad(berhu_loss(aligned, gt, torch.ones_like(gt)), a2)[0]
    assert abs(float(g_aligned)) < 1e-3, "afim deveria ser cego à escala pura"

    print(f"✅ Termo de escala OK — grad escala={float(g_scale):.3f} (vivo), "
          f"grad alinhado={float(g_aligned):.1e} (~0, confirma hip. 4).")


def test_observability():
    """
    Regressão do problema central do report: com normalize_terms ligado, o total fica
    ancorado ~1.0 e mascara o progresso. Verifica (a) que os valores CRUS estão sempre
    disponíveis e (b) que, após congelar, o total acompanha o erro real.
    """
    import numpy as np
    import torch
    from riemann.losses import RiemannianAwareLoss, RiemannWeights

    gt = torch.linspace(1, 6, 32).view(1, 1, 1, 32).repeat(1, 1, 32, 1).contiguous()
    w = RiemannWeights(berhu=1.0, normalize_terms=True, norm_freeze_steps=0)
    crit = RiemannianAwareLoss(w)

    _, parts = crit(gt + 1.0 * torch.randn_like(gt), gt, None)
    assert "raw_total" in parts and "raw_berhu" in parts, "valores crus ausentes nos logs"

    # Após congelar, uma melhora real deve reduzir o total proporcionalmente
    errs = np.linspace(1.0, 0.3, 60)
    tot_first = tot_last = raw_first = raw_last = None
    for i, e in enumerate(errs):
        if i == 20:
            crit.freeze_normalization()
        _, p = crit(gt + float(e) * torch.randn_like(gt), gt, None)
        if i == 20:
            tot_first, raw_first = p["total"], p["raw_total"]
        if i == len(errs) - 1:
            tot_last, raw_last = p["total"], p["raw_total"]
    assert tot_last < tot_first, "total não desceu mesmo com o erro real caindo"
    print(f"✅ Observabilidade OK — crus expostos; após congelar, total "
          f"{tot_first:.3f}->{tot_last:.3f} acompanha raw {raw_first:.3f}->{raw_last:.3f}.")


if __name__ == "__main__":
    main()
    test_regression_fixes()
    test_term_balance()
    test_scale_term()
    test_observability()
