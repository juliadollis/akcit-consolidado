#!/usr/bin/env python3
"""
metricas_por_imagem.py
======================
Recalcula as metricas de bokeh guardando o valor POR IMAGEM, em vez de so a
media.

POR QUE EXISTE
--------------
O `eval_bokeh_synthesis.py` acumula soma corrente e divide por n no fim: os
valores por imagem sao descartados. Consequencia: a tabela de resultados nao tem
desvio, nem intervalo de confianca, nem teste pareado. Nao da para dizer se a
diferenca entre dois modelos esta fora do ruido, que e exatamente a pergunta que
um revisor faz primeiro.

Isto NAO refaz inferencia. As imagens geradas ja estao nos repos `bokeh-eval-*`
do Hub; aqui so se repassa a metrica sobre elas.

A VERIFICACAO EMBUTIDA
----------------------
O script recomputa a MEDIA a partir dos valores por imagem e compara com a linha
que ja esta em `juliadollis/bokeh-eval-metricas`. Os dois caminhos sao
independentes (um acumula em streaming, o outro guarda tudo e faz mean no fim).

  - batem dentro da tolerancia -> a linha antiga fica CONFIRMADA
  - nao batem                  -> ha erro em um dos dois, e o script grita

E de proposito que a comparacao seja automatica: e a unica forma barata de
auditar 76 linhas de uma tabela que ninguem consegue reconferir a mao.

USO
    python3 metricas_por_imagem.py --repos juliadollis/bokeh-eval-rd-nosso-full ...
    python3 metricas_por_imagem.py --lista repos.txt --saida juliadollis/bokeh-metricas-por-imagem
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
from datasets import Dataset, load_dataset
from scipy.stats import pearsonr
from skimage.metrics import structural_similarity as ssim

# Reusa EXATAMENTE o mesmo codigo de metrica do avaliador oficial do projeto.
# Importar em vez de copiar e o que garante que o numero por imagem e o mesmo
# numero que a media antiga usou; uma copia divergiria em silencio.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evaluation.eval_bokeh_synthesis import (  # noqa: E402
    K_IMAGE_FIELDS,
    K_VALUES,
    CloudBokehEvaluator,
    calculate_laplacian_variance,
    get_pil_image,
    img_to_gray_cv2,
)

TOLERANCIA = 5e-4  # diferenca aceitavel entre a media nova e a linha antiga


def ic95_bootstrap(vals, n_boot=10000, seed=0):
    """IC95 por bootstrap percentil da MEDIA. Sem suposicao de normalidade:
    LPIPS e DISTS sao assimetricos e limitados por baixo."""
    v = np.asarray([x for x in vals if np.isfinite(x)], dtype=float)
    if len(v) < 2:
        return (float(v.mean()) if len(v) else float("nan"), float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    medias = v[rng.integers(0, len(v), size=(n_boot, len(v)))].mean(axis=1)
    return float(v.mean()), float(np.percentile(medias, 2.5)), float(np.percentile(medias, 97.5))


def abre(repo, split, token):
    """Carrega o split pedido; se o repo nao tiver esse nome, usa o unico que
    existir. Os repos de RealDOF vieram com 'validation' e os demais com 'test',
    e travar num nome so faria metade da tabela falhar por detalhe de nomenclatura."""
    try:
        return load_dataset(repo, split=split, token=token), split
    except Exception:
        from datasets import get_dataset_split_names
        nomes = get_dataset_split_names(repo, token=token)
        if not nomes:
            raise
        print(f"  [split] '{split}' nao existe em {repo}; usando '{nomes[0]}'", flush=True)
        return load_dataset(repo, split=nomes[0], token=token), nomes[0]


def por_imagem(ev, repo, split="test", token=None):
    """Devolve uma lista de dicts, um por imagem, com todas as metricas."""
    ds, split_usado = abre(repo, split, token)

    # DEDUPLICACAO (obrigatoria, nao cosmetica).
    # As voltas do keeper reprocessaram itens e ANEXARAM linhas repetidas em
    # alguns repos de imagem: o lfrepro-oficial tem 840 linhas para 500 ids
    # unicos, e dois repos do RealBokeh tem 377 e 407 para 217. Medir por cima
    # disso pondera algumas cenas em dobro e move a media. A chave e o
    # `file_name_base`; guardamos a PRIMEIRA ocorrencia, que e a da campanha
    # original, a mesma que a linha agregada antiga mediu.
    chave = ("file_name_base" if "file_name_base" in ds.column_names
             else ("image_id" if "image_id" in ds.column_names else None))
    indices, vistos = [], set()
    if chave is None:
        indices = list(range(len(ds)))
    else:
        for i, k in enumerate(ds[chave]):
            k = str(k)
            if k not in vistos:
                vistos.add(k); indices.append(i)
    duplicatas = len(ds) - len(indices)
    if duplicatas:
        print(f"  [dedup] {repo}: {len(ds)} linhas -> {len(indices)} unicas "
              f"({duplicatas} repetidas descartadas)", flush=True)

    linhas = []
    falhas = 0
    for i in indices:
        row = ds[i]
        try:
            real = get_pil_image(row["image_real_bokeh"]).convert("RGB")
            gerada = get_pil_image(row["image_best_k"]).convert("RGB")

            real_g, ger_g = img_to_gray_cv2(real), img_to_gray_cv2(gerada)
            if ger_g.shape != real_g.shape:
                import cv2
                ger_g = cv2.resize(ger_g, (real_g.shape[1], real_g.shape[0]))
            ssim_val = ssim(real_g, ger_g)

            rt = ev.to_tensor(real).unsqueeze(0).to(ev.device)
            gt = ev.to_tensor(gerada).unsqueeze(0).to(ev.device)
            if gt.shape != rt.shape:
                import torch.nn.functional as F
                gt = torch.clamp(F.interpolate(gt, size=rt.shape[2:], mode="bicubic",
                                               align_corners=False), 0, 1)
            with torch.no_grad():
                lpips_val = ev.lpips(gt, rt).item()
                dists_val = ev.dists(gt, rt).item()
                clip_val = torch.cosine_similarity(
                    ev.get_clip_embedding(gerada), ev.get_clip_embedding(real)).item()

            lvs = [calculate_laplacian_variance(get_pil_image(row[f])) for f in K_IMAGE_FIELDS]
            lv = 0.0 if (np.std(lvs) == 0 or np.std(K_VALUES) == 0) else pearsonr(K_VALUES, lvs)[0]

            linhas.append({
                "repo": repo,
                "split": split_usado,
                "indice": i,
                "image_id": str(row.get(chave, i)) if chave else str(i),
                "SSIM": ssim_val, "LPIPS": lpips_val, "DISTS": dists_val,
                "CLIP-I": clip_val, "LVCorr": lv,
                # LVCorr no sinal do PAPER: la a controlabilidade e "quanto maior
                # melhor" e o GenRefocus aparece com +0.9368. O nosso Pearson cru
                # e negativo quando o modelo OBEDECE o K (mais K -> mais borrado
                # -> menos variancia do Laplaciano). Sem esta coluna a tabela faz
                # o modelo sem treino parecer o mais controlavel de todos.
                "LVCorr_convencao_paper": -lv,
                "variancia_laplaciano_por_K": json.dumps(
                    {str(k): float(v) for k, v in zip(K_VALUES, lvs)}),
                # o nome real da coluna e best_k_value; `best_k` nao existe e
                # antes isso gravava None em todas as linhas
                "best_k": (float(row["best_k_value"])
                           if row.get("best_k_value") is not None else None),
                "ssim_do_pipeline": (float(row["ssim_score"])
                                     if row.get("ssim_score") is not None else None),
            })
        except Exception as e:
            falhas += 1
            print(f"  [falha] {repo} item {i}: {str(e)[:140]}", flush=True)
    return linhas, falhas, duplicatas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repos", nargs="*", default=[])
    ap.add_argument("--lista", default=None, help="arquivo com um repo por linha")
    ap.add_argument("--split", default="test")
    ap.add_argument("--saida", default=None, help="repo HF onde gravar as linhas por imagem")
    ap.add_argument("--saida-local", default="/workspace/hf-cache/por_imagem")
    ap.add_argument("--repo-agregado", default="juliadollis/bokeh-eval-metricas",
                    help="tabela antiga, usada so para CONFERIR a media")
    args = ap.parse_args()

    repos = list(args.repos)
    if args.lista:
        with open(args.lista) as f:
            repos += [l.strip() for l in f if l.strip() and not l.startswith("#")]
    if not repos:
        print("nada a fazer: passe --repos ou --lista"); return 2

    token = os.environ.get("HF_TOKEN")
    print(f"[metricas] {len(repos)} repos a processar", flush=True)

    # tabela antiga, indexada por Dataset (a coluna que guarda o repo de imagens)
    antigo = {}
    try:
        d = load_dataset(args.repo_agregado, split="train", token=token).to_pandas()
        for _, r in d.iterrows():
            antigo[str(r["Dataset"])] = r
        print(f"[metricas] tabela antiga carregada: {len(antigo)} linhas", flush=True)
    except Exception as e:
        print(f"[metricas] AVISO: nao consegui ler a tabela antiga ({str(e)[:100]}). Sem conferencia.", flush=True)

    ev = CloudBokehEvaluator(device="cuda")
    todas, relatorio = [], []
    os.makedirs(args.saida_local, exist_ok=True)

    for repo in repos:
        print(f"\n[metricas] === {repo} ===", flush=True)
        try:
            linhas, falhas, duplicatas = por_imagem(ev, repo, args.split, token)
        except Exception as e:
            print(f"  [ERRO] {repo}: {str(e)[:200]}", flush=True)
            relatorio.append({"repo": repo, "status": "ERRO", "detalhe": str(e)[:200]})
            continue
        if not linhas:
            relatorio.append({"repo": repo, "status": "VAZIO"}); continue
        todas += linhas

        reg = {"repo": repo, "n": len(linhas), "falhas": falhas,
               "duplicatas_descartadas": duplicatas}
        for m in ["SSIM", "LPIPS", "DISTS", "CLIP-I", "LVCorr"]:
            media, lo, hi = ic95_bootstrap([l[m] for l in linhas])
            reg[m] = media; reg[f"{m}_ic95_lo"] = lo; reg[f"{m}_ic95_hi"] = hi
            reg[f"{m}_desvio"] = float(np.std([l[m] for l in linhas], ddof=1))

        # ---- a conferencia contra a linha antiga ----
        ref = antigo.get(repo)
        if ref is None:
            reg["conferencia"] = "SEM LINHA ANTIGA"
        else:
            difs = {m: abs(reg[m] - float(ref[m])) for m in ["SSIM", "LPIPS", "DISTS", "CLIP-I", "LVCorr"]
                    if m in ref and ref[m] is not None}
            pior = max(difs.values()) if difs else float("nan")
            reg["conferencia"] = "CONFERE" if pior <= TOLERANCIA else "DIVERGE"
            reg["pior_diferenca"] = pior
            marca = "OK " if reg["conferencia"] == "CONFERE" else "!! "
            print(f"  {marca}conferencia contra a tabela antiga: pior diferenca {pior:.6f}", flush=True)
            if reg["conferencia"] == "DIVERGE":
                for m, v in sorted(difs.items(), key=lambda x: -x[1]):
                    print(f"     {m}: novo={reg[m]:.6f}  antigo={float(ref[m]):.6f}  dif={v:.6f}", flush=True)

        print(f"  n={len(linhas)}  LPIPS={reg['LPIPS']:.4f} [{reg['LPIPS_ic95_lo']:.4f}, {reg['LPIPS_ic95_hi']:.4f}]"
              f"  DISTS={reg['DISTS']:.4f}  CLIP-I={reg['CLIP-I']:.4f}  LVCorr={reg['LVCorr']:+.4f}", flush=True)
        relatorio.append(reg)

    if todas:
        ds = Dataset.from_list(todas)
        caminho = os.path.join(args.saida_local, "por_imagem.parquet")
        ds.to_parquet(caminho)
        print(f"\n[metricas] {len(todas)} linhas por imagem em {caminho}", flush=True)
        if args.saida:
            ds.push_to_hub(args.saida, token=token, private=True)
            print(f"[metricas] enviado para {args.saida}", flush=True)

    with open(os.path.join(args.saida_local, "resumo.json"), "w") as f:
        json.dump(relatorio, f, indent=2)

    conf = [r for r in relatorio if r.get("conferencia") == "CONFERE"]
    div = [r for r in relatorio if r.get("conferencia") == "DIVERGE"]
    print(f"\n[metricas] RESUMO: {len(conf)} repos CONFEREM com a tabela antiga, {len(div)} DIVERGEM", flush=True)
    for r in div:
        print(f"  DIVERGE: {r['repo']}  pior={r.get('pior_diferenca'):.6f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
