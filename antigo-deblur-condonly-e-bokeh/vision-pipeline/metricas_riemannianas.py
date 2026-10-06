#!/usr/bin/env python3
"""
metricas_riemannianas.py
========================
Avalia bokeh pela GEOMETRIA do campo de desfoque, e não só pela distância
perceptual média da imagem inteira.

POR QUE
-------
LPIPS e DISTS dão UM número para a imagem toda. Mas a parte difícil do bokeh não
está espalhada: está nas DESCONTINUIDADES DE PROFUNDIDADE, onde o desfoque tem
de mudar de regime e a oclusão tem de ser tratada. É exatamente o que a Fig. 5
do paper mostra em recorte ampliado, e é exatamente o que uma média sobre a
imagem inteira dilui. Um modelo pode acertar 95% de céu e parede lisos e errar
toda a silhueta do objeto, e o LPIPS mal registra.

Aqui as métricas medem isso diretamente. As duas primeiras são transplante do
`riemann/metrics.py` do projeto de profundidade, que já foi validado lá.

O QUE É MEDIDO
--------------
1. `bF_desfoque`: F-score de borda entre o CAMPO DE DESFOQUE predito e o do
   alvo. O campo de desfoque é a variância local do Laplaciano, calculada em
   janela deslizante: baixa onde está borrado, alta onde está nítido. Tratado
   como campo escalar, ele entra no mesmo `boundary_fscore` que o projeto de
   profundidade usa em mapas de profundidade. Responde: **o modelo põe a
   transição de desfoque no lugar certo?**

2. `ssim_borda` e `ssim_plano`: o mapa de SSIM (não o escalar) mediado
   separadamente numa faixa em torno das descontinuidades de profundidade e no
   resto da imagem. A profundidade vem do mapa do Depth Pro que o próprio
   pipeline usou para renderizar, então a faixa é a mesma geometria que gerou a
   imagem. Responde: **o erro está concentrado na silhueta ou espalhado?**

3. `corr_desfoque`: correlação de Pearson entre os campos de desfoque predito e
   alvo, pixel a pixel. É a versão espacialmente resolvida do LVCorr, que hoje
   compara só quatro números por imagem.

O QUE ISTO NÃO É
----------------
Não é curvatura de superfície. O campo de desfoque não é uma superfície imersa
em R³ com escala métrica, então K = 1/R² não tem significado físico aqui. Usar
`surface_curvatures` sobre ele daria número, não sentido. As métricas acima usam
só primeira ordem e casamento de bordas, que é o que se sustenta.

Uso:
    python3 metricas_riemannianas.py --repos juliadollis/bokeh-eq4-rd-rotac60k ...
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from datasets import Dataset, get_dataset_split_names, load_dataset
from skimage.metrics import structural_similarity as ssim_fn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/workspace/depth-riemannian")
from evaluation.eval_bokeh_synthesis import get_pil_image, img_to_gray_cv2  # noqa: E402
from riemann.metrics import boundary_fscore, _depth_edges  # noqa: E402

JANELA_DESFOQUE = 11      # lado da janela da variância local do Laplaciano
TOL_BORDA_PX = 2          # mesma tolerância usada no projeto de profundidade
FAIXA_BORDA_PX = 6        # meia-largura da faixa em torno da descontinuidade


def campo_de_desfoque(cinza: np.ndarray, janela: int = JANELA_DESFOQUE) -> torch.Tensor:
    """Variância local do Laplaciano, em janela deslizante.

    É a mesma grandeza que o LVCorr usa (`cv2.Laplacian(...).var()`), só que
    espacialmente resolvida em vez de reduzida a um escalar por imagem. Alta
    onde a imagem está nítida, baixa onde está borrada.
    """
    x = torch.from_numpy(cinza.astype(np.float32) / 255.0)[None, None]
    nucleo = torch.tensor([[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]])[None, None]
    lap = F.conv2d(F.pad(x, (1, 1, 1, 1), mode="replicate"), nucleo)
    p = janela // 2
    media = F.avg_pool2d(F.pad(lap, (p, p, p, p), mode="replicate"), janela, stride=1)
    media_sq = F.avg_pool2d(F.pad(lap ** 2, (p, p, p, p), mode="replicate"), janela, stride=1)
    return (media_sq - media ** 2).clamp(min=0.0)


def faixa_de_borda(depth: np.ndarray, meia_largura: int = FAIXA_BORDA_PX) -> np.ndarray:
    """Máscara booleana da vizinhança das descontinuidades de profundidade.

    Usa o MESMO detector de bordas do projeto de profundidade (percentil, que é
    o modo robusto), dilatado por `meia_largura`.
    """
    d = torch.from_numpy(depth.astype(np.float32))[None, None]
    bordas = _depth_edges(d, rel_thresh=0.08, modo="percentil", pct=99.0).float()
    k = 2 * meia_largura + 1
    return (F.max_pool2d(bordas, k, stride=1, padding=meia_largura)[0, 0].numpy() > 0.5)


def por_imagem(repo, dir_depth, split="test", token=None):
    try:
        ds = load_dataset(repo, split=split, token=token)
    except Exception:
        ds = load_dataset(repo, split=get_dataset_split_names(repo, token=token)[0], token=token)

    chave = "file_name_base" if "file_name_base" in ds.column_names else None
    vistos, indices = set(), []
    if chave:
        for i, k in enumerate(ds[chave]):
            if str(k) not in vistos:
                vistos.add(str(k)); indices.append(i)
    else:
        indices = list(range(len(ds)))

    linhas, sem_depth = [], 0
    for i in indices:
        linha = ds[i]
        try:
            nome = str(linha[chave]) if chave else str(i)
            alvo = get_pil_image(linha["image_real_bokeh"]).convert("RGB")
            pred = get_pil_image(linha["image_best_k"]).convert("RGB")
            ga, gp = img_to_gray_cv2(alvo), img_to_gray_cv2(pred)
            if gp.shape != ga.shape:
                import cv2
                gp = cv2.resize(gp, (ga.shape[1], ga.shape[0]))

            ca, cp = campo_de_desfoque(ga), campo_de_desfoque(gp)

            # 1) casamento das bordas do campo de desfoque
            bf = boundary_fscore(cp, ca, tolerance_px=TOL_BORDA_PX,
                                 rel_thresh=0.08, modo="percentil", pct=99.0)

            # 3) correlacao espacial dos campos de desfoque
            va, vp = ca.flatten().numpy(), cp.flatten().numpy()
            corr = float(np.corrcoef(va, vp)[0, 1]) if va.std() > 0 and vp.std() > 0 else 0.0

            reg = {"repo": repo, "image_id": nome,
                   "bF_desfoque": bf["boundary_fscore"],
                   "precisao_desfoque": bf["boundary_precision"],
                   "recall_desfoque": bf["boundary_recall"],
                   "corr_desfoque": corr}

            # 2) SSIM separado por regiao, so quando ha mapa de profundidade
            cam = os.path.join(dir_depth, f"{nome}_depth.npy")
            if os.path.exists(cam):
                dep = np.load(cam).astype(np.float32)
                if dep.shape[:2] != ga.shape:
                    import cv2
                    dep = cv2.resize(dep, (ga.shape[1], ga.shape[0]), interpolation=cv2.INTER_NEAREST)
                _, mapa = ssim_fn(ga, gp, full=True)
                faixa = faixa_de_borda(dep)
                if faixa.any() and (~faixa).any():
                    reg["ssim_borda"] = float(mapa[faixa].mean())
                    reg["ssim_plano"] = float(mapa[~faixa].mean())
                    # o que interessa e a QUEDA na borda: um modelo que trata bem
                    # oclusao tem as duas proximas; um que borra a silhueta cai so
                    # na borda e a media global mal registra
                    reg["queda_na_borda"] = reg["ssim_plano"] - reg["ssim_borda"]
                    reg["fracao_de_borda"] = float(faixa.mean())
            else:
                sem_depth += 1
            linhas.append(reg)
        except Exception as e:
            print(f"  [falha] {repo} item {i}: {str(e)[:120]}", flush=True)
    return linhas, sem_depth


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repos", nargs="*", default=[])
    ap.add_argument("--lista", default=None)
    ap.add_argument("--split", default="test")
    ap.add_argument("--dir-depth", default="/workspace/vision-pipeline/temp_depth_maps")
    ap.add_argument("--saida-local", default="/host/riemann_bokeh")
    args = ap.parse_args()

    repos = list(args.repos)
    if args.lista:
        repos += [l.strip() for l in open(args.lista) if l.strip() and not l.startswith("#")]
    if not repos:
        print("passe --repos ou --lista"); return 2

    token = os.environ.get("HF_TOKEN")
    os.makedirs(args.saida_local, exist_ok=True)
    todas = []
    for r in repos:
        print(f"\n[riemann] === {r} ===", flush=True)
        linhas, sem = por_imagem(r, args.dir_depth, args.split, token)
        if not linhas:
            continue
        todas += linhas
        bf = np.mean([l["bF_desfoque"] for l in linhas])
        co = np.mean([l["corr_desfoque"] for l in linhas])
        com_ssim = [l for l in linhas if "queda_na_borda" in l]
        extra = ""
        if com_ssim:
            extra = ("  ssim_borda=%.4f ssim_plano=%.4f queda=%.4f" % (
                np.mean([l["ssim_borda"] for l in com_ssim]),
                np.mean([l["ssim_plano"] for l in com_ssim]),
                np.mean([l["queda_na_borda"] for l in com_ssim])))
        print(f"  n={len(linhas)} (sem mapa de profundidade: {sem})  "
              f"bF_desfoque={bf:.4f}  corr_desfoque={co:+.4f}{extra}", flush=True)

    if todas:
        cam = os.path.join(args.saida_local, "riemann_por_imagem.parquet")
        Dataset.from_list(todas).to_parquet(cam)
        print(f"\n[riemann] {len(todas)} linhas em {cam}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
