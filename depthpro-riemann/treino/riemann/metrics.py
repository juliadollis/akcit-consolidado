"""
riemann/metrics.py
==================
Métricas de avaliação:
  - Padrão MDE (8): AbsRel, SqRel, RMSE, RMSE_log, log10, δ<1.25, δ<1.25², δ<1.25³
  - Qualidade de borda (o diferencial que liga ao refocusing e ao QC cross-domain):
    boundary F-score entre bordas do depth predito e do GT.

A métrica de borda é a que conecta esta melhoria ao caso de uso: o defocus map
Ddef=|D−S₁|·K só gera bokeh limpo se as bordas do depth forem nítidas e bem localizadas.
"""

from __future__ import annotations
from typing import Dict, Optional
import torch
import torch.nn.functional as F


@torch.no_grad()
def mde_metrics(pred: torch.Tensor, target: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> Dict[str, float]:
    """
    Métricas MDE padrão. pred/target: (B,1,H,W).
    Alinha pred ao GT pela MESMA transformação afim (escala+shift) usada na loss, para
    consistência. Antes usava só mediana (escala), que não corrige deslocamento.
    """
    from .losses import robust_affine_align
    if mask is None:
        mask = torch.ones_like(pred)

    # Alinhamento afim robusto pred->GT (mesmo da loss)
    # Avaliação usa alinhamento afim COMPLETO (padrão MiDaS/DPT): sem backward aqui, então
    # queremos a comparação afim-invariante clássica, independente do modo usado no treino.
    pred = robust_affine_align(pred, target, mask, mode="full")

    m = mask > 0.5
    valid = m & (target > 1e-6)
    p = pred[valid].clamp(min=1e-6)
    t = target[valid].clamp(min=1e-6)
    if p.numel() == 0:
        return {k: float("nan") for k in
                ["abs_rel", "sq_rel", "rmse", "rmse_log", "log10", "d1", "d2", "d3"]}

    abs_rel = torch.mean(torch.abs(p - t) / t)
    sq_rel = torch.mean(((p - t) ** 2) / t)
    rmse = torch.sqrt(torch.mean((p - t) ** 2))
    rmse_log = torch.sqrt(torch.mean((torch.log(p) - torch.log(t)) ** 2))
    log10 = torch.mean(torch.abs(torch.log10(p) - torch.log10(t)))

    ratio = torch.max(p / t, t / p)
    d1 = torch.mean((ratio < 1.25).float())
    d2 = torch.mean((ratio < 1.25 ** 2).float())
    d3 = torch.mean((ratio < 1.25 ** 3).float())

    return {
        "abs_rel": float(abs_rel), "sq_rel": float(sq_rel),
        "rmse": float(rmse), "rmse_log": float(rmse_log), "log10": float(log10),
        "d1": float(d1), "d2": float(d2), "d3": float(d3),
    }


def _depth_edges(depth: torch.Tensor, rel_thresh: float = 0.08,
                 modo: str = "percentil", pct: float = 99.0) -> torch.Tensor:
    """
    Bordas do depth: magnitude do gradiente acima de rel_thresh * referencia.

    O parametro `modo` define a referencia de normalizacao:
      "percentil" (PADRAO)  referencia = percentil `pct` do gradiente. ROBUSTO.
      "max"                 referencia = maximo do gradiente. FRAGIL, mantido apenas
                            para reproduzir resultados antigos.

    Por que isto importa: normalizar pelo MAXIMO torna a metrica refem de um unico
    pixel. Um outlier de gradiente eleva a referencia, o limiar sobe, e quase nenhuma
    borda e detectada. O efeito medido e caracteristico: a precisao se mantem alta e o
    recall despenca. Em teste controlado, uma predicao IDENTICA ao ground truth com um
    unico pixel outlier caiu de F=1.000 para F=0.596 (P=0.98, R=0.43) no modo "max",
    enquanto no modo "percentil" ficou em F=0.996.
    """
    dp = F.pad(depth, (1, 1, 1, 1), mode="replicate")
    gx = (dp[:, :, 1:-1, 2:] - dp[:, :, 1:-1, :-2]) * 0.5
    gy = (dp[:, :, 2:, 1:-1] - dp[:, :, :-2, 1:-1]) * 0.5
    g = torch.sqrt(gx ** 2 + gy ** 2 + 1e-12)
    B = g.shape[0]
    flat = g.view(B, -1)
    if modo == "max":
        ref = flat.max(dim=1)[0]
    else:
        ref = torch.quantile(flat, pct / 100.0, dim=1)
    return g > (rel_thresh * ref.view(B, 1, 1, 1).clamp(min=1e-6))


@torch.no_grad()
def boundary_fscore(pred: torch.Tensor, target: torch.Tensor,
                    tolerance_px: int = 2, rel_thresh: float = 0.08,
                    modo: str = "percentil", pct: float = 99.0,
                    mask: Optional[torch.Tensor] = None) -> Dict[str, float]:
    """
    Boundary F-score entre bordas do depth predito e do GT, com tolerância.

    precision = fração de bordas-pred a ≤ tol de uma borda-GT
    recall    = fração de bordas-GT a ≤ tol de uma borda-pred
    F = 2PR/(P+R)

    A tolerância é implementada por dilatação (max-pool) das bordas.
    """
    ep = _depth_edges(pred, rel_thresh, modo, pct).float()
    et = _depth_edges(target, rel_thresh, modo, pct).float()

    # MASCARA DE VALIDADE: o dataset zera os pixels invalidos, e a fronteira entre a
    # regiao valida e os zeros vira uma BORDA FALSA no alvo, que a predicao (continua)
    # nao tem. O efeito e caracteristico e severo: precisao alta, recall derrubado.
    # Medido: uma predicao quase perfeita cai de bF-max 1.000 para 0.792 com apenas 1%
    # da imagem zerada. Aqui removemos as bordas que tocam regiao invalida.
    if mask is not None:
        val = (mask > 0.5).float()
        # erodir a validade pelo raio da tolerancia: uma borda so conta se toda a
        # vizinhanca considerada no casamento for valida
        k = 2 * tolerance_px + 1
        val_ero = -F.max_pool2d(-val, k, 1, tolerance_px)
        ep = ep * val_ero
        et = et * val_ero

    k = 2 * tolerance_px + 1
    et_dil = F.max_pool2d(et, k, stride=1, padding=tolerance_px)
    ep_dil = F.max_pool2d(ep, k, stride=1, padding=tolerance_px)

    tp_p = (ep * et_dil).sum()
    tp_r = (et * ep_dil).sum()
    n_pred = ep.sum().clamp(min=1.0)
    n_gt = et.sum().clamp(min=1.0)

    precision = float(tp_p / n_pred)
    recall = float(tp_r / n_gt)
    f = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {"boundary_precision": precision, "boundary_recall": recall, "boundary_fscore": f}


@torch.no_grad()
def all_metrics(pred: torch.Tensor, target: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> Dict[str, float]:
    out = mde_metrics(pred, target, mask)
    out.update(boundary_fscore(pred, target, mask=mask))
    out.update(boundary_fscore_sweep(pred, target, mask=mask))
    return out


@torch.no_grad()
def boundary_fscore_sweep(pred: torch.Tensor, target: torch.Tensor,
                          tolerance_px: int = 2,
                          limiares=(0.02, 0.04, 0.06, 0.08, 0.12, 0.16, 0.24, 0.32),
                          modo: str = "percentil", pct: float = 99.0,
                          mask: Optional[torch.Tensor] = None
                          ) -> Dict[str, float]:
    """
    Varre o limiar de deteccao de borda e devolve o MELHOR F alcancavel, alem da curva.

    Por que: comparar dois modelos num UNICO limiar confunde qualidade de borda com
    agressividade de deteccao. Um modelo mais conservador detecta menos bordas, o que
    sobe a precisao e derruba o recall sem que a geometria tenha melhorado ou piorado.
    O F-max e o ponto de operacao otimo de cada modelo, e por isso e a comparacao justa.

    Retorna boundary_fmax, o limiar que o atingiu, a precisao e o recall nesse ponto,
    e ainda a media do F ao longo da curva (uma aproximacao de area sob a curva).
    """
    melhor = {"boundary_fmax": -1.0}
    fs = []
    for t in limiares:
        m = boundary_fscore(pred, target, tolerance_px, t, modo, pct, mask)
        fs.append(m["boundary_fscore"])
        if m["boundary_fscore"] > melhor["boundary_fmax"]:
            melhor = {"boundary_fmax": m["boundary_fscore"],
                      "boundary_fmax_thresh": float(t),
                      "boundary_precision_at_fmax": m["boundary_precision"],
                      "boundary_recall_at_fmax": m["boundary_recall"]}
    melhor["boundary_f_auc"] = float(sum(fs) / max(len(fs), 1))
    return melhor
