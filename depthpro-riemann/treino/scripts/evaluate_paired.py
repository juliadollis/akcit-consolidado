#!/usr/bin/env python3
"""
scripts/evaluate_paired.py
==========================
Avaliação final com comparação EMPARELHADA contra o modelo zero-shot.

Por que este script existe: comparar dois intervalos de confiança independentes é a
análise mais fraca possível neste problema. A variância entre cenas (cena fácil vs cena
difícil) é muito maior que o efeito que se quer medir, e mascara ganhos reais. Ao
subtrair o baseline do modelo NA MESMA CENA, essa variância desaparece.

Além disso, o script separa corretamente as duas fontes de incerteza:
  - variação entre SEMENTES  -> intervalo por t de Student (n pequeno)
  - variação entre CENAS     -> comparação emparelhada

Uso típico (após treinar a config campeã com --seeds 3):

    python scripts/evaluate_paired.py \\
        --checkpoint /models/checkpoints/depth_pro.pt \\
        --seeds-dir /workspace/runs/champion \\
        --data-root /data/hypersim/test \\
        --out-dir /workspace/runs/champion/avaliacao_final \\
        --variant heads

Gera:
  metricas_por_imagem.csv   uma linha por imagem por modelo (com a coluna cena)
  comparacao_pareada.json   resultado estatístico completo
  resumo.txt                relatório legível
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from riemann.dataset import HighQualityDepthDataset          # noqa: E402
from riemann.metrics import all_metrics                       # noqa: E402
from riemann.model import build_model                         # noqa: E402
from riemann.repro import (compare_paired, formatar_comparacao,  # noqa: E402
                           mean_ci, resumo_sementes, scene_of)

METRICAS = ["abs_rel", "rmse", "d1", "boundary_fscore",
            "boundary_precision", "boundary_recall",
            # F-max: melhor F ao longo da varredura de limiar. E a comparacao JUSTA de
            # borda, porque nao penaliza um modelo por ser mais conservador no limiar
            # arbitrario de 0.08. Reporte ESTE como resultado de borda.
            "boundary_fmax", "boundary_f_auc"]
# True = maior é melhor
SENTIDO = {"abs_rel": False, "rmse": False, "d1": True, "boundary_fscore": True,
           "boundary_precision": True, "boundary_recall": True,
           "boundary_fmax": True, "boundary_f_auc": True}


def metricas_por_imagem(model, loader, device):
    """Roda o modelo e devolve {chave_da_imagem: {metrica: valor}}."""
    model.eval()
    saida = {}
    with torch.no_grad():
        for batch in loader:
            rgb = batch["rgb"].to(device)
            gt = batch["depth"].to(device).float()
            mask = batch["mask"].to(device)
            chaves = batch["key"]
            pred = model(rgb).float()
            if pred.shape[-2:] != gt.shape[-2:]:
                pred = torch.nn.functional.interpolate(
                    pred, size=gt.shape[-2:], mode="bilinear", align_corners=False)
            # métrica POR IMAGEM: fatiamos o batch, pois as métricas reduzem sobre
            # todos os pixels válidos do tensor recebido.
            for i, k in enumerate(chaves):
                m = all_metrics(pred[i:i + 1], gt[i:i + 1], mask[i:i + 1])
                saida[k] = {c: float(m.get(c, float("nan"))) for c in METRICAS}
    return saida


def carregar_checkpoints(seeds_dir):
    """Encontra os best.pt de cada semente em runs/<algo>/seed_*/best.pt."""
    d = Path(seeds_dir)
    achados = sorted(d.glob("seed_*/best.pt"))
    if not achados:
        solto = d / "best.pt"
        if solto.exists():
            achados = [solto]
    return achados


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, help="pesos base do DepthPro")
    ap.add_argument("--seeds-dir", required=True,
                    help="pasta com seed_*/best.pt da config a avaliar")
    ap.add_argument("--data-root", required=True,
                    help="conjunto de avaliação (idealmente o TESTE held-out)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--variant", default="heads", choices=["heads", "heads_final", "heads_lora"])
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--nivel", default="cena", choices=["cena", "item"],
                    help="agregar por cena antes de comparar (recomendado)")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    ds = HighQualityDepthDataset(args.data_root, size=(args.size, args.size))
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True)
    cenas = sorted({scene_of(k) for k in ds.keys})
    print(f"[dados] {len(ds)} imagens em {len(cenas)} cenas: {args.data_root}")
    if len(cenas) < 8:
        print(f"  [ATENCAO] apenas {len(cenas)} cenas. O tamanho efetivo de amostra e o "
              f"numero de CENAS, nao de imagens. Intervalos serao largos.")

    # ---- 1. zero-shot (determinístico, uma passada) -----------------------
    print("\n[1/2] avaliando o modelo ZERO-SHOT (sem fine-tune)...")
    m0 = build_model(args.checkpoint, variant=args.variant, device=args.device)
    zs = metricas_por_imagem(m0, loader, args.device)
    del m0
    if args.device == "cuda":
        torch.cuda.empty_cache()

    # ---- 2. cada semente da config treinada -------------------------------
    ckpts = carregar_checkpoints(args.seeds_dir)
    if not ckpts:
        print(f"ERRO: nenhum best.pt encontrado em {args.seeds_dir}")
        sys.exit(1)
    print(f"\n[2/2] avaliando {len(ckpts)} semente(s)...")

    por_semente = []
    for i, ck in enumerate(ckpts):
        modelo = build_model(args.checkpoint, variant=args.variant, device=args.device)
        # best.pt guarda só os parâmetros treináveis -> strict=False
        modelo.load_state_dict(torch.load(ck, map_location=args.device), strict=False)
        r = metricas_por_imagem(modelo, loader, args.device)
        por_semente.append(r)
        med = {c: float(np.nanmean([v[c] for v in r.values()])) for c in METRICAS}
        print(f"  semente {i} ({ck.parent.name}): "
              f"AbsRel={med['abs_rel']:.4f}  bF={med['boundary_fscore']:.4f}")
        del modelo
        if args.device == "cuda":
            torch.cuda.empty_cache()

    # média por imagem entre as sementes (o modelo "típico" da configuração)
    chaves = sorted(set(zs) & set(por_semente[0]))
    media_sementes = {
        k: {c: float(np.nanmean([s[k][c] for s in por_semente])) for c in METRICAS}
        for k in chaves}

    # ---- CSV por imagem ---------------------------------------------------
    csv_path = out / "metricas_por_imagem.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["modelo", "semente", "chave", "cena"] + METRICAS)
        for k in chaves:
            w.writerow(["zero_shot", "-", k, scene_of(k)] +
                       [f"{zs[k][c]:.6f}" for c in METRICAS])
        for i, s in enumerate(por_semente):
            for k in chaves:
                w.writerow(["treinado", i, k, scene_of(k)] +
                           [f"{s[k][c]:.6f}" for c in METRICAS])
    print(f"\n[saida] metricas por imagem: {csv_path}")

    # ---- Estatística ------------------------------------------------------
    relatorio = []
    resultados = {"n_imagens": len(chaves), "n_cenas": len(cenas),
                  "n_sementes": len(por_semente), "nivel": args.nivel,
                  "metricas": {}}

    relatorio.append("=" * 78)
    relatorio.append("AVALIACAO FINAL - comparacao emparelhada contra o zero-shot")
    relatorio.append("=" * 78)
    relatorio.append(f"dados     : {args.data_root}")
    relatorio.append(f"imagens   : {len(chaves)} em {len(cenas)} cenas")
    relatorio.append(f"sementes  : {len(por_semente)}")
    relatorio.append("")

    for c in METRICAS:
        maior_melhor = SENTIDO[c]

        # (a) variação entre SEMENTES, com t de Student
        med_por_semente = [float(np.nanmean([s[k][c] for k in chaves]))
                           for s in por_semente]
        base_media = float(np.nanmean([zs[k][c] for k in chaves]))
        relatorio.append(resumo_sementes(med_por_semente, baseline=base_media,
                                         maior_melhor=maior_melhor, nome=c))

        # (b) comparação EMPARELHADA por cena (a análise principal)
        res = compare_paired({k: media_sementes[k][c] for k in chaves},
                             {k: zs[k][c] for k in chaves},
                             maior_melhor=maior_melhor, nivel=args.nivel)
        relatorio.append(formatar_comparacao(res, c))
        relatorio.append("")

        res.pop("chaves", None)
        resultados["metricas"][c] = {
            "por_semente": med_por_semente,
            "media_semente_ic95_t": list(mean_ci(med_por_semente)),
            "zero_shot": base_media,
            "pareado": res,
        }

    relatorio.append("-" * 78)
    relatorio.append("Como reportar: use a comparacao EMPARELHADA como resultado")
    relatorio.append("principal (ganho medio + IC + p). O intervalo entre sementes")
    relatorio.append("mede reprodutibilidade do treino, nao o efeito contra o baseline.")
    texto = "\n".join(relatorio)
    print("\n" + texto)

    (out / "resumo.txt").write_text(texto, encoding="utf-8")
    with open(out / "comparacao_pareada.json", "w") as f:
        json.dump(resultados, f, indent=2)
    print(f"\n[saida] {out/'resumo.txt'}  e  {out/'comparacao_pareada.json'}")


if __name__ == "__main__":
    main()
