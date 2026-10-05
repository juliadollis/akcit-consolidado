#!/usr/bin/env python3
"""Teste em GPU do treino da BokehNet — o que NÃO deu para verificar sem placa.

Roda na h100n1, em Docker, numa GPU só. Não treina de verdade: exercita, contra
o FLUX real e o PEFT real, exatamente os pontos que os testes sem GPU não
alcançam e que por isso atravessaram quatro revisões sem execução:

  T1  versões e importabilidade do `transformer_forward` dos autores
  T2  `GradientAccumulationPlugin(sync_with_dataloader=False)` — a API que eu
      afirmei existir sem poder conferir (correção da sobra de época)
  T3  injeção de LoRA: 343 módulos, e o `transformer.proj_out` de topo FORA
  T4  o `mu` vetorizado contra o `calculate_shift` do diffusers instalado
  T5  §3.3: `add_adapter` desativa o base? o `set_adapter([base, forma])`
      reativa? o base fica ATIVO e CONGELADO e só a forma treina?
  T6  release sintético em ÁRVORE -> dataloader -> 3 steps REAIS
      (VAE, transformer_forward, loss, backward, optimizer, checkpoint, export)

O release da rota A está incompleto (só `depth/`), então o T6 monta um release
de mentira **no formato real do `bokehnet-regen`** — manifest/meta/depth/
generated/split — com 4 amostras de 2 cenas. O que ele prova é o CAMINHO, não o
dado.

NÃO apaga nada. NÃO toca em container ou job de terceiro. Escreve só dentro do
`--out`.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path

import numpy as np

FALHAS: list[str] = []
OKS: list[str] = []


def secao(titulo: str) -> None:
    print("\n" + "=" * 72, flush=True)
    print(titulo, flush=True)
    print("=" * 72, flush=True)


def ok(nome: str, detalhe: str = "") -> None:
    OKS.append(nome)
    print(f"  [OK]    {nome}" + (f" — {detalhe}" if detalhe else ""), flush=True)


def falha(nome: str, detalhe: str) -> None:
    FALHAS.append(f"{nome}: {detalhe}")
    print(f"  [FALHA] {nome} — {detalhe}", flush=True)


# =============================================================================
# T1 — ambiente
# =============================================================================

def t1_ambiente() -> None:
    secao("T1 — ambiente")
    import torch

    mods = {}
    for nome in ("torch", "diffusers", "peft", "accelerate", "transformers", "datasets"):
        try:
            mods[nome] = __import__(nome).__version__
        except Exception as exc:  # noqa: BLE001
            falha(f"import {nome}", repr(exc))
    for k, v in mods.items():
        print(f"    {k:14s} {v}", flush=True)
    print(f"    cuda           {torch.version.cuda} | gpus visiveis {torch.cuda.device_count()}",
          flush=True)
    if torch.cuda.device_count() < 1:
        falha("GPU", "nenhuma GPU visivel")
        return
    livre, total = torch.cuda.mem_get_info(0)
    print(f"    GPU 0 do container: {(total-livre)/1e9:.1f} GB em uso de {total/1e9:.1f} GB",
          flush=True)
    ok("ambiente", f"torch {mods.get('torch')} / diffusers {mods.get('diffusers')}")

    try:
        from Genfocus.pipeline.flux import transformer_forward  # noqa: F401
        ok("Genfocus.transformer_forward importavel")
    except Exception as exc:  # noqa: BLE001
        falha("Genfocus", repr(exc))


# =============================================================================
# T2 — a API que eu afirmei sem conferir
# =============================================================================

def t2_plugin_de_acumulacao() -> None:
    secao("T2 — GradientAccumulationPlugin(sync_with_dataloader=False)")
    try:
        import inspect

        from accelerate.utils import GradientAccumulationPlugin

        params = set(inspect.signature(GradientAccumulationPlugin).parameters)
        print(f"    parametros aceitos: {sorted(params)}", flush=True)
        if "sync_with_dataloader" not in params:
            falha("sync_with_dataloader", "a versao instalada NAO aceita o parametro — "
                  "a sobra de epoca continua (um step/epoca com batch menor)")
            return
        p = GradientAccumulationPlugin(
            num_steps=8, sync_with_dataloader=False, adjust_scheduler=False
        )
        ok("plugin construido", f"num_steps=8 sync_with_dataloader=False -> {type(p).__name__}")

        from genfocus_train.config import load_config
        from genfocus_train.trainer import _plugin_de_acumulacao

        cfg = load_config(os.environ["CONFIG_SMOKE"])
        real = _plugin_de_acumulacao(cfg)
        if real is None:
            falha("_plugin_de_acumulacao", "devolveu None — cairia no caminho antigo")
        else:
            ok("_plugin_de_acumulacao", f"{type(real).__name__}, num_steps="
               f"{getattr(real, 'num_steps', '?')}")
    except Exception as exc:  # noqa: BLE001
        falha("T2", f"{type(exc).__name__}: {exc}")


# =============================================================================
# T3/T4/T5 — backbone real
# =============================================================================

def t3_t4_t5_backbone(stage: str = "bokeh") -> None:
    secao(f"T3/T4/T5 — backbone real (stage={stage})")
    import torch

    from genfocus_train.backbone import (
        ADAPTER_NAME, SHAPE_ADAPTER_NAME, create_backbone,
    )
    from genfocus_train.config import load_config

    chave = "CONFIG_SHAPE" if stage == "bokeh_shape" else "CONFIG_SMOKE"
    cfg = load_config(os.environ.get(chave) or os.environ["CONFIG_SMOKE"])
    # O cache do HF aqui e root-owned e o symlink nao resolve com HF_HUB_OFFLINE.
    # Apontar o SNAPSHOT direto pula a resolucao de cache inteira — e e o que a
    # inferencia oficial aceita tambem (`from_pretrained` com caminho local).
    if os.environ.get("FLUX_PATH"):
        cfg.model.pretrained_model_name_or_path = os.environ["FLUX_PATH"]
        print(f"    FLUX de: {cfg.model.pretrained_model_name_or_path}", flush=True)
    bb = create_backbone(cfg, stage=stage)
    print("    carregando FLUX (pode levar minutos na primeira vez)...", flush=True)
    bb.load(dtype=torch.bfloat16, device=torch.device("cuda:0"))

    # T3 — contagem de modulos LoRA
    n = int(getattr(bb, "n_lora_modules", -1))
    esperado = int(cfg.model.expected_lora_modules)
    if n == esperado:
        ok("T3 modulos LoRA", f"{n} (o checkpoint oficial tem {esperado})")
    else:
        falha("T3 modulos LoRA", f"{n} != {esperado}")

    from peft.tuners.lora import LoraLayer

    de_topo = [
        nome for nome, mod in bb.transformer.named_modules()
        if isinstance(mod, LoraLayer) and "." not in nome
    ]
    if de_topo == ["x_embedder"]:
        ok("T3 proj_out de topo FORA", f"modulos de topo com LoRA: {de_topo}")
    else:
        falha("T3 proj_out de topo", f"esperado ['x_embedder'], veio {de_topo}")

    # T4 — mu
    try:
        bb._verificar_mu_bate_com_diffusers()
        ok("T4 mu vetorizado == calculate_shift do diffusers")
    except Exception as exc:  # noqa: BLE001
        falha("T4 mu", repr(exc))

    # T5 — §3.3
    if stage == "bokeh_shape":
        camadas = [m for _, m in bb.transformer.named_modules() if isinstance(m, LoraLayer)]
        ativos = set(camadas[0].active_adapters) if camadas else set()
        print(f"    active_adapters numa LoraLayer: {sorted(ativos)}", flush=True)
        if {ADAPTER_NAME, SHAPE_ADAPTER_NAME} <= ativos:
            ok("T5 os DOIS adapters ATIVOS", "o base participa do forward")
        else:
            falha("T5 adapters ativos", f"{sorted(ativos)} — o base sairia do forward")

        treinaveis = [n for n, p in bb.transformer.named_parameters() if p.requires_grad]
        vazados = [n for n in treinaveis if f".{SHAPE_ADAPTER_NAME}." not in n]
        if treinaveis and not vazados:
            ok("T5 so a forma treina", f"{len(treinaveis)} tensores treinaveis")
        else:
            falha("T5 congelamento", f"{len(vazados)} treinaveis fora do adapter de forma")

    del bb
    torch.cuda.empty_cache()


# =============================================================================
# T6 — release sintetico em ARVORE + smoke real
# =============================================================================

def monta_release_falso(raiz: Path, n_cenas: int = 2, por_cena: int = 2) -> Path:
    """Release no formato REAL do `bokehnet-regen`, com dados de mentira."""
    from PIL import Image

    from genfocus_train import control

    raiz.mkdir(parents=True, exist_ok=True)
    for sub in ("depth", "meta", "generated"):
        (raiz / sub).mkdir(exist_ok=True)

    H, W = 320, 448           # nao-multiplo de 16 de proposito
    manifesto, assignment = [], {}
    rng = np.random.default_rng(0)

    for c in range(n_cenas):
        cena = f"falsa_{c:03d}"
        assignment[cena] = "train" if c == 0 else "val"
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        disp = 0.05 + 0.9 * (xx / (W - 1)) + 0.05 * np.sin(yy / 17.0)
        dmin, dmax = float(disp.min()), float(disp.max())
        u16 = np.round((disp - dmin) / (dmax - dmin) * 65535).astype(np.uint16)

        for v in range(por_cena):
            sid = f"{cena}_v{v:02d}"
            Image.fromarray(u16, mode="I;16").save(raiz / "depth" / f"{sid}.png")
            aif = (rng.random((H, W, 3)) * 255).astype(np.uint8)
            Image.fromarray(aif).save(raiz / "generated" / f"{sid}_aif.jpg", quality=95)
            bok = (aif.astype(np.float32) * 0.8).astype(np.uint8)
            Image.fromarray(bok).save(raiz / "generated" / f"{sid}_bokeh.jpg", quality=95)

            meta = {
                "sample_id": sid, "scene_id": cena, "route": "a",
                "control_version": control.CONTROL_VERSION,
                "k_value": float(8.0 + 4.0 * v), "focus_disparity": float(np.median(disp)),
                "disparity_min": dmin, "disparity_max": dmax,
                "max_coc": control.MAX_COC,
                "image_h": H, "image_w": W, "depth_h": H, "depth_w": W,
                "is_valid_for_control": True, "is_k_censored": False,
                "k_source": "sampled_from_bc", "depth_backend": "depth_pro",
                "mask_source": "depth_band",
                "generated_images": ["aif", "bokeh"],
                "source_dataset": "falso/ebb", "source_sample_id": sid,
                "split": assignment[cena],
            }
            (raiz / "meta" / f"{sid}.json").write_text(json.dumps(meta), encoding="utf-8")
            manifesto.append({k: meta[k] for k in (
                "sample_id", "scene_id", "route", "control_version", "k_value",
                "is_valid_for_control", "is_k_censored", "split")})

    (raiz / "manifest.jsonl").write_text(
        "\n".join(json.dumps(l) for l in manifesto) + "\n", encoding="utf-8")
    (raiz / "split.json").write_text(
        json.dumps({"assignment": assignment, "salt": "teste"}), encoding="utf-8")
    return raiz


def t6_smoke(release: Path, out: Path) -> None:
    secao("T6 — release sintetico em ARVORE + 3 steps REAIS")
    import torch
    import yaml

    from genfocus_train.config import load_config
    from genfocus_train.data import build_dataset, DatasetRuntimeConfig
    from genfocus_train.train import _build_backbone_and_model
    from genfocus_train.trainer import run_smoke_test

    base = yaml.safe_load(Path(os.environ["CONFIG_SMOKE"]).read_text(encoding="utf-8"))
    b = base["data"]["bokeh"]
    b["datasets"] = [{"name": str(release), "split": "train"}]
    b["val_datasets"] = [{"name": str(release), "split": "train"}]
    b["release_format"] = "arvore"
    b["scene_split_partition"] = "train"
    b["max_levels_per_scene"] = None
    base["runtime"]["output_dir"] = str(out)
    cfg_path = out / "cfg_teste.yaml"
    out.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text(yaml.safe_dump(base, allow_unicode=True), encoding="utf-8")

    cfg = load_config(cfg_path)
    if os.environ.get("FLUX_PATH"):
        cfg.model.pretrained_model_name_or_path = os.environ["FLUX_PATH"]

    # o dataloader, antes do FLUX
    try:
        ds = build_dataset(
            stage="bokeh", stage_config=cfg.data.bokeh,
            runtime=DatasetRuntimeConfig(
                image_size=cfg.data.bokeh.image_size, train=False,
                defocus_source="metric_disparity", release_format="arvore",
                scene_split_partition="train", max_levels_per_scene=None,
            ),
        )
        item = ds[0]
        print(f"    tipo: {type(ds).__name__} | amostras: {len(ds)}", flush=True)
        print(f"    chaves: {sorted(item)}", flush=True)
        print(f"    aif {tuple(item['aif_image'].shape)} [{item['aif_image'].min():.2f},"
              f"{item['aif_image'].max():.2f}]", flush=True)
        print(f"    defocus {tuple(item['defocus_map'].shape)} "
              f"[{item['defocus_map'].min():.4f},{item['defocus_map'].max():.4f}] "
              f"k_efetivo={float(item['k_efetivo']):.3f} "
              f"full_seq_len={int(item['full_seq_len'])}", flush=True)
        ok("T6 dataloader (arvore)", f"{len(ds)} amostras, chaves completas")
    except Exception as exc:  # noqa: BLE001
        falha("T6 dataloader", f"{type(exc).__name__}: {exc}")
        traceback.print_exc()
        return

    try:
        bb, model = _build_backbone_and_model(
            cfg, "bokeh", torch.bfloat16, torch.device("cuda:0")
        )
        run_smoke_test(cfg, bb, model, out, stage="bokeh")
        ok("T6 smoke", "3 steps reais: forward, backward, optimizer, export")
    except Exception as exc:  # noqa: BLE001
        falha("T6 smoke", f"{type(exc).__name__}: {exc}")
        traceback.print_exc()


def _tenta(fn, *a) -> None:
    """Uma falha isolada nao pode matar os testes seguintes — foi o que
    aconteceu na primeira rodada: o estagio de forma abortou e o T6, que era o
    mais valioso, nunca rodou."""
    try:
        fn(*a)
    except Exception as exc:  # noqa: BLE001
        falha(f"{getattr(fn,'__name__',fn)}{a}", f"{type(exc).__name__}: {exc}")
        traceback.print_exc()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--pular-flux", action="store_true")
    ap.add_argument("--stage-shape", action="store_true")
    args = ap.parse_args()
    out = Path(args.out)

    print(f"### TESTE EM GPU — {os.uname().nodename} ###", flush=True)
    t1_ambiente()
    t2_plugin_de_acumulacao()
    if not args.pular_flux:
        _tenta(t3_t4_t5_backbone, "bokeh")
        if args.stage_shape:
            _tenta(t3_t4_t5_backbone, "bokeh_shape")
        rel = monta_release_falso(out / "release_falso")
        t6_smoke(rel, out / "run")

    secao("RESUMO")
    print(f"  passaram: {len(OKS)}", flush=True)
    for o in OKS:
        print(f"    OK  {o}", flush=True)
    print(f"  falharam: {len(FALHAS)}", flush=True)
    for f in FALHAS:
        print(f"    XX  {f}", flush=True)
    return 1 if FALHAS else 0


if __name__ == "__main__":
    sys.exit(main())
