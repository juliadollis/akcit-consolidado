#!/usr/bin/env python3
"""Exercita o dataloader contra as amostras de um release que JÁ baixaram.

Por que agora e não depois: o download das rotas b e c leva ~2 h. Se o loader
quebrar contra o dado real, a hora de descobrir é enquanto ele baixa, não com as
GPUs paradas. As amostras completas já são suficientes para percorrer TODO o
caminho — inclusive a resolução de pixel no espelho, que é onde as duas rotas
diferem da rota A e onde nenhum teste sintético alcança.

Monta um sub-release por SYMLINK só com as amostras completas, passa pelo
dataloader e MEDE o sinal de controle. Não copia nem apaga nada.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np


def amostras_completas(raiz: Path, papeis: list[str]) -> list[dict]:
    linhas = [json.loads(l) for l in (raiz / "manifest.jsonl").read_text(
        encoding="utf-8").splitlines() if l.strip()]
    ok = []
    for reg in linhas:
        sid, cena = str(reg["sample_id"]), str(reg["scene_id"])
        caminhos = [raiz / "depth" / cena / f"{sid}.png",
                    raiz / "meta" / cena / f"{sid}.json"]
        caminhos += [raiz / "generated" / cena / f"{sid}_{p}.jpg" for p in papeis]
        if all(c.is_file() for c in caminhos):
            ok.append(reg)
    return ok


def monta(raiz: Path, destino: Path, linhas: list[dict]) -> Path:
    destino.mkdir(parents=True, exist_ok=True)
    cenas = {str(r["scene_id"]) for r in linhas}
    for sub in ("depth", "meta", "generated"):
        (destino / sub).mkdir(exist_ok=True)
        for cena in cenas:
            org, lnk = raiz / sub / cena, destino / sub / cena
            if org.is_dir() and not lnk.exists():
                lnk.symlink_to(org)
    (destino / "manifest.jsonl").write_text(
        "\n".join(json.dumps(r) for r in linhas) + "\n", encoding="utf-8")
    atribuicao = json.loads((raiz / "split.json").read_text(encoding="utf-8"))
    (destino / "split.json").write_text(json.dumps(atribuicao), encoding="utf-8")
    for extra in ("source_images.jsonl", "generated_images.jsonl", "run_config.json"):
        org, lnk = raiz / extra, destino / extra
        if org.is_file() and not lnk.exists():
            lnk.symlink_to(org)
    return destino


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--release", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--papeis", default="", help="ex.: aif (rota b) | vazio (rota c)")
    ap.add_argument("--n", type=int, default=48)
    ap.add_argument("--somente-cenas", default=None,
                    help="JSON com lista de scene_id: restringe a verificação a "
                         "elas. Serve para exercitar o caminho de pixel quando só "
                         "parte do espelho está em cache.")
    args = ap.parse_args()

    raiz, out = Path(args.release), Path(args.out)
    papeis = [p for p in args.papeis.split(",") if p]
    linhas = amostras_completas(raiz, papeis)
    print(f"amostras completas em disco: {len(linhas)}", flush=True)
    if args.somente_cenas:
        alvo = set(json.loads(Path(args.somente_cenas).read_text(encoding="utf-8")))
        linhas = [r for r in linhas if str(r["scene_id"]) in alvo]
        print(f"restrito a {len(alvo)} cenas -> {len(linhas)} amostras", flush=True)
    if len(linhas) < 8:
        print("ainda não há amostras suficientes; volte depois.", flush=True)
        return 0
    sub = monta(raiz, out / "parcial", linhas)
    print(f"sub-release por symlink: {sub}", flush=True)

    import yaml

    from genfocus_train.config import load_config
    from genfocus_train.data import DatasetRuntimeConfig, build_dataset

    payload = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    b = payload["data"]["bokeh"]
    b["datasets"] = [{"name": str(sub), "split": "train"}]
    b["val_datasets"] = []
    payload["runtime"]["output_dir"] = str(out / "run")
    cfgp = out / "cfg.yaml"
    cfgp.write_text(yaml.safe_dump(payload, allow_unicode=True), encoding="utf-8")
    cfg = load_config(cfgp)

    print("\n=== DATALOADER CONTRA O DADO REAL ===", flush=True)
    sc = cfg.data.bokeh
    ds = build_dataset(
        stage="bokeh", stage_config=sc,
        runtime=DatasetRuntimeConfig(
            image_size=sc.image_size, train=False,
            defocus_source="metric_disparity", release_format="arvore",
            scene_split_partition=sc.scene_split_partition,
            max_levels_per_scene=sc.max_levels_per_scene,
            mirror_roots=sc.mirror_roots, permitir_download_do_release=False,
            verificar_sha256_da_origem=sc.verificar_sha256_da_origem,
            min_calibration_ssim=sc.min_calibration_ssim,
            exclude_censored_k=sc.exclude_censored_k,
            require_valid_for_control=sc.require_valid_for_control,
            excluir_cenas_de_avaliacao=sc.excluir_cenas_de_avaliacao,
        ),
    )
    print(f"tipo={type(ds).__name__}  amostras={len(ds)}", flush=True)
    if len(ds) == 0:
        print("!!! dataset VAZIO — algum filtro está comendo tudo.", flush=True)
        return 1

    ks, mx, sat, seq, shapes = [], [], [], [], Counter()
    for i in range(min(args.n, len(ds))):
        it = ds[i]
        ks.append(float(it["k_efetivo"]))
        mx.append(float(it["defocus_max"]))
        sat.append(float(it["defocus_frac_saturado"]))
        seq.append(int(it["full_seq_len"]))
        shapes[tuple(it["aif_image"].shape)] += 1
    q = lambda v, p: float(np.percentile(v, p))  # noqa: E731
    print("\n=== SINAL DE CONTROLE, MEDIDO ===", flush=True)
    print(f"  k_efetivo p5/p50/p95   {q(ks,5):.3f} / {q(ks,50):.3f} / {q(ks,95):.3f}")
    print(f"  k_efetivo distintos    {len({round(x,4) for x in ks})} de {len(ks)}")
    print(f"  defocus_max p5/p50/p95 {q(mx,5):.4f} / {q(mx,50):.4f} / {q(mx,95):.4f}")
    print(f"  frac_saturado mediana  {q(sat,50):.5f}")
    print(f"  full_seq_len           {min(seq)}..{max(seq)}")
    print(f"  shapes                 {dict(shapes)}")

    alertas = []
    if len({round(x, 4) for x in ks}) <= 1:
        alertas.append("K CONSTANTE — o defeito D2 de volta")
    if q(mx, 50) >= 0.999:
        alertas.append("defocus_max ~1,0 na mediana — assinatura do D1")
    if q(sat, 50) > 0.05:
        alertas.append("mapa saturando — max_coc errado para a fonte")
    print(f"\n  ALERTAS: {alertas if alertas else 'nenhum'}", flush=True)
    return 1 if alertas else 0


if __name__ == "__main__":
    sys.exit(main())
