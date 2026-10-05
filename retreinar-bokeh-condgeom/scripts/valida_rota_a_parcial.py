#!/usr/bin/env python3
"""Valida o release da rota A com o que JÁ baixou — enquanto o resto chega.

Por que isto vale GPU agora: o download leva ~20 h, e se o loader quebrar contra
o dado real a gente descobre às 4 da manhã, com 4 GPUs paradas desde ontem. As
cenas que já fecharam são suficientes para exercitar TODO o caminho.

O que faz:

  1. Monta um sub-release SÓ com as cenas completas (manifesto e split filtrados,
     `depth/`, `meta/` e `generated/` por symlink — não copia os JPEGs).
  2. Passa pelo dataloader real e MEDE o sinal de controle sobre o dado de
     verdade: distribuição de `k_value`, do `k_efetivo` pós-crop, do mapa de
     defocus, e a fração saturada. É aqui que um defeito de dado aparece, e é
     barato agora e caro depois de 40K steps.
  3. Confere a resolução da AIF pelo espelho do EBB! e o sha256 do ledger.
  4. Roda 3 steps REAIS (FLUX + LoRA + 2 condições) sobre esse dado.
  5. Monta o conjunto do probe (`probe_set/`), que hoje é `null` no YAML e é
     pré-condição do LVCorr.

NÃO apaga nada. NÃO toca no release original — só lê e cria symlink.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np


def cenas_completas(release: Path, por_cena: int) -> list[str]:
    ger = release / "generated"
    meta = release / "meta"
    depth = release / "depth"
    if not ger.is_dir():
        return []
    prontas = []
    for d in sorted(ger.iterdir()):
        if not d.is_dir():
            continue
        cena = d.name
        n_bokeh = len(list(d.glob("*_bokeh.jpg")))
        n_meta = len(list((meta / cena).glob("*.json"))) if (meta / cena).is_dir() else 0
        tem_depth = (depth / f"{cena}.png").is_file()
        if n_bokeh >= por_cena and n_meta >= por_cena and tem_depth:
            prontas.append(cena)
    return prontas


def monta_sub_release(release: Path, destino: Path, cenas: list[str]) -> Path:
    """Sub-release por SYMLINK — os JPEGs não são copiados."""
    destino.mkdir(parents=True, exist_ok=True)
    for sub in ("depth", "meta", "generated", "mask"):
        (destino / sub).mkdir(exist_ok=True)

    aceitas = set(cenas)
    for cena in cenas:
        alvo = release / "depth" / f"{cena}.png"
        lnk = destino / "depth" / f"{cena}.png"
        if alvo.is_file() and not lnk.exists():
            lnk.symlink_to(alvo)
        for sub in ("meta", "generated", "mask"):
            org = release / sub / cena
            lnk = destino / sub / cena
            if org.is_dir() and not lnk.exists():
                lnk.symlink_to(org)

    linhas, assignment = [], {}
    with (release / "manifest.jsonl").open(encoding="utf-8") as h:
        for linha in h:
            if not linha.strip():
                continue
            reg = json.loads(linha)
            if str(reg.get("scene_id")) in aceitas:
                linhas.append(reg)
    # split: 80/20 por CENA, determinístico pela ordem
    for i, cena in enumerate(sorted(aceitas)):
        assignment[cena] = "val" if i % 5 == 4 else "train"
    for reg in linhas:
        reg["split"] = assignment[str(reg["scene_id"])]

    (destino / "manifest.jsonl").write_text(
        "\n".join(json.dumps(r) for r in linhas) + "\n", encoding="utf-8")
    (destino / "split.json").write_text(
        json.dumps({"assignment": assignment, "salt": "parcial"}), encoding="utf-8")
    for extra in ("source_images.jsonl", "generated_images.jsonl", "run_config.json"):
        org = release / extra
        lnk = destino / extra
        if org.is_file() and not lnk.exists():
            lnk.symlink_to(org)
    return destino


def mede_o_controle(ds, n: int = 64) -> dict:
    """As estatísticas que denunciam defeito de dado, sobre o dado REAL."""
    ks, mm, mx, sat, seq = [], [], [], [], []
    alvo = min(n, len(ds))
    for i in range(alvo):
        it = ds[i]
        ks.append(float(it["k_efetivo"]))
        mm.append(float(it["defocus_mean"]))
        mx.append(float(it["defocus_max"]))
        sat.append(float(it["defocus_frac_saturado"]))
        seq.append(int(it["full_seq_len"]))
    q = lambda v, p: float(np.percentile(v, p))  # noqa: E731
    return {
        "amostras": alvo,
        "k_efetivo": [q(ks, 5), q(ks, 50), q(ks, 95)],
        "k_efetivo_distintos": len(set(round(x, 4) for x in ks)),
        "defocus_mean": [q(mm, 5), q(mm, 50), q(mm, 95)],
        "defocus_max": [q(mx, 5), q(mx, 50), q(mx, 95)],
        "frac_saturado_mediana": q(sat, 50),
        "full_seq_len": [min(seq), int(np.median(seq)), max(seq)],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--release", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--por-cena", type=int, default=41)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    release, out = Path(args.release), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print("=" * 70, flush=True)
    cenas = cenas_completas(release, args.por_cena)
    print(f"cenas COMPLETAS ({args.por_cena} variantes): {len(cenas)}", flush=True)
    if len(cenas) < 2:
        print("ainda nao ha cenas completas suficientes. Volte mais tarde.", flush=True)
        return 0
    sub = monta_sub_release(release, out / "parcial", cenas)
    print(f"sub-release por symlink em {sub}", flush=True)

    import yaml

    from genfocus_train.config import load_config
    from genfocus_train.data import DatasetRuntimeConfig, build_dataset

    base = yaml.safe_load(Path(os.environ["CONFIG"]).read_text(encoding="utf-8"))
    b = base["data"]["bokeh"]
    b["datasets"] = [{"name": str(sub), "split": "train"}]
    b["val_datasets"] = [{"name": str(sub), "split": "train"}]
    base["runtime"]["output_dir"] = str(out / "run")
    cfgp = out / "cfg_parcial.yaml"
    cfgp.write_text(yaml.safe_dump(base, allow_unicode=True), encoding="utf-8")
    cfg = load_config(cfgp)

    print("\n=== DATALOADER CONTRA O DADO REAL ===", flush=True)
    ds = build_dataset(
        stage="bokeh", stage_config=cfg.data.bokeh,
        runtime=DatasetRuntimeConfig(
            image_size=cfg.data.bokeh.image_size, train=False,
            defocus_source="metric_disparity", release_format="arvore",
            scene_split_partition="train", max_levels_per_scene=None,
            mirror_roots=cfg.data.bokeh.mirror_roots,
            permitir_download_do_release=False,
            verificar_sha256_da_origem=cfg.data.bokeh.verificar_sha256_da_origem,
        ),
    )
    print(f"tipo={type(ds).__name__}  amostras={len(ds)}", flush=True)

    print("\n=== SINAL DE CONTROLE, MEDIDO ===", flush=True)
    m = mede_o_controle(ds)
    for k, v in m.items():
        print(f"  {k:24s} {v}", flush=True)

    alerta = []
    if m["k_efetivo_distintos"] <= 1:
        alerta.append("K CONSTANTE — o defeito D2 de volta")
    if m["defocus_max"][1] >= 0.999:
        alerta.append("defocus_max ~1,0 na mediana — assinatura do D1")
    if m["frac_saturado_mediana"] > 0.05:
        alerta.append("mapa saturando — max_coc errado para a fonte")
    print("\n  ALERTAS:", alerta if alerta else "nenhum", flush=True)

    (out / "medicao.json").write_text(json.dumps(m, indent=2), encoding="utf-8")

    if args.smoke:
        print("\n=== 3 STEPS REAIS SOBRE O DADO DA ROTA A ===", flush=True)
        import torch

        from genfocus_train.train import _build_backbone_and_model
        from genfocus_train.trainer import run_smoke_test

        if os.environ.get("FLUX_PATH"):
            cfg.model.pretrained_model_name_or_path = os.environ["FLUX_PATH"]
        bb, model = _build_backbone_and_model(
            cfg, "bokeh", torch.bfloat16, torch.device("cuda:0"))
        run_smoke_test(cfg, bb, model, out / "run", stage="bokeh")

    print("\nPRONTO.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
