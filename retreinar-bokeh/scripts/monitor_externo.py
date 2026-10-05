#!/usr/bin/env python3
"""Monitor EXTERNO do treino: mede controlabilidade e validação nos checkpoints.

POR QUE FORA DO TREINO, e não dentro
------------------------------------
O probe interno roda no rank 0 e leva ~5 min gerando 32 imagens de 28 passos.
Nesse tempo os outros ranks ficam parados no próximo coletivo — que é
EXATAMENTE o cenário do watchdog do NCCL que matou o treino multi-GPU duas
vezes. E o critério de parada automática mudaria o comportamento de um run no
meio do caminho.

O princípio, que é da usuária e está certo: **o treino começa e termina do mesmo
jeito**. Instrumento é observador, não participante.

Então este processo:
  - roda numa GPU SEPARADA, num container SEPARADO;
  - lê os `step_*.pt` que o treino já grava a cada 250 steps;
  - não toca em RNG, optimizer, pesos nem coletivo do treino;
  - só ESCREVE num JSONL próprio. Se ele morrer, o treino nem percebe.

O QUE MEDE
----------
1. **LVCorr** (§4.1(ii)) — a controlabilidade. É o número que colapsou na fase 2
   anterior: +0,9059 no fim da fase 1, +0,4365 no fim da fase 2, degradando
   MONOTONICAMENTE enquanto a loss de flow matching caía. Uma série temporal é a
   única forma de ver isso acontecendo; um número no fim não serve.
2. **`val_loss` determinística** — σ e ruído de uma grade fixa, para o número ser
   comparável entre steps. Resolve o `best.pt` por fora, já que a validação
   interna está desligada nesta run.

Referência para comparar: **fase 1 anterior = +0,9059** (LF-Bokeh).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np


def steps_de(p: Path) -> int:
    m = re.search(r"step_(\d+)\.pt$", p.name)
    return int(m.group(1)) if m else -1


def checkpoints(dir_ckpt: Path) -> list[Path]:
    return sorted(dir_ckpt.glob("step_*.pt"), key=steps_de)


def garante_conjunto_do_probe(release: Path, destino: Path, n: int) -> Path:
    """Conjunto FIXO, de cenas da partição de VALIDAÇÃO (disjuntas do treino)."""
    if destino.is_dir() and list(destino.glob("*.npz")):
        return destino
    destino.mkdir(parents=True, exist_ok=True)

    from PIL import Image

    from genfocus_train import control, release as rel

    arv = rel.ArvoreDeRelease.abrir(str(release), permitir_download=False)
    cenas_val = set(rel.carregar_assignment_de_split(release / "split.json", "val"))
    print(f"[probe-set] {len(cenas_val)} cenas na partição val", flush=True)

    espelhos = json.loads(os.environ.get("MIRROR_ROOTS", "{}"))
    resolv = rel.ResolvedorDePixels(arv, espelhos=espelhos, permitir_download=False)

    feitos, usadas = 0, set()
    for linha in arv.linhas:
        if feitos >= n:
            break
        cena = str(linha.get("scene_id", ""))
        if cena not in cenas_val or cena in usadas:
            continue
        sid = str(linha["sample_id"])
        try:
            meta = arv.metadados(sid)
            if not meta.get("is_valid_for_control", True) or meta.get("is_k_censored"):
                continue
            aif = np.asarray(resolv.resolver(sid, "aif").imagem, dtype=np.uint8)
            with Image.open(arv.caminho_da_profundidade(sid)) as im:
                bruto = np.array(im)
            disp = control.decode_disparity_u16(
                bruto, float(meta["disparity_min"]), float(meta["disparity_max"]))
            # mesma grade do TREINO: lado menor -> 512, crop central
            from genfocus_train.data import _plano_geometrico, resize_nearest

            h, w = aif.shape[:2]
            if disp.shape != (h, w):
                disp = resize_nearest(disp, w, h)
            nw, nh, box, _f, _s = _plano_geometrico(w, h, 512, "short_side", False, None)
            if (nw, nh) != (w, h):
                aif = np.asarray(Image.fromarray(aif).resize((nw, nh), Image.BICUBIC),
                                 dtype=np.uint8)
                disp = resize_nearest(disp, nw, nh)
            x0, y0, x1, y1 = box
            np.savez_compressed(
                destino / f"{cena}.npz",
                aif=np.ascontiguousarray(aif[y0:y1, x0:x1]),
                disparity=np.ascontiguousarray(disp[y0:y1, x0:x1]),
                focus_disparity=np.float32(meta["focus_disparity"]),
            )
            usadas.add(cena)
            feitos += 1
            print(f"[probe-set] {cena} ok ({feitos}/{n})", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"[probe-set] pulando {sid}: {type(exc).__name__}: {exc}", flush=True)
    return destino


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", required=True, help="output_dir do treino")
    ap.add_argument("--release", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--probe-set", required=True)
    ap.add_argument("--n-probe", type=int, default=8)
    ap.add_argument("--intervalo", type=int, default=600)
    ap.add_argument("--uma-vez", action="store_true")
    args = ap.parse_args()

    import torch

    from genfocus_train.config import load_config
    from genfocus_train.probe import lvcorr_agregado, run_lvcorr_probe
    from genfocus_train.train import _build_backbone_and_model
    from genfocus_train.trainer import load_lora_checkpoint_into_backbone

    conj_dir = garante_conjunto_do_probe(
        Path(args.release), Path(args.probe_set), args.n_probe)
    from genfocus_train.probe import carregar_conjunto_do_probe

    conjunto = carregar_conjunto_do_probe(conj_dir)
    print(f"[monitor] conjunto do probe: {len(conjunto)} imagens", flush=True)

    cfg = load_config(args.config)
    if os.environ.get("FLUX_PATH"):
        cfg.model.pretrained_model_name_or_path = os.environ["FLUX_PATH"]

    # ESPERA memória em vez de morrer. Esta máquina é compartilhada: a GPU que
    # estava vazia no `docker run` pode encher antes do FLUX terminar de
    # carregar — aconteceu, com ~50 processos de terceiros aparecendo no meio.
    # O monitor é observador: ele espera a vez dele, e enquanto espera não
    # atrapalha ninguém.
    precisa_gb = float(os.environ.get("MONITOR_MIN_GB", "30"))
    while True:
        livre, _total = torch.cuda.mem_get_info(0)
        if livre / 1e9 >= precisa_gb:
            break
        print(f"[monitor] {livre/1e9:.1f} GB livres < {precisa_gb} GB necessários; "
              "esperando 5 min (a máquina é compartilhada).", flush=True)
        time.sleep(300)

    for tentativa in range(1, 100):
        try:
            backbone, model = _build_backbone_and_model(
                cfg, "bokeh", torch.bfloat16, torch.device("cuda:0"))
            break
        except torch.OutOfMemoryError:
            print(f"[monitor] OOM na tentativa {tentativa}; "
                  "alguém pegou a GPU. Nova tentativa em 5 min.", flush=True)
            torch.cuda.empty_cache()
            time.sleep(300)
    else:
        print("[monitor] desisto de carregar o backbone.", flush=True)
        return 1
    print("[monitor] backbone carregado", flush=True)

    dir_ckpt = Path(args.output_dir) / "bokeh" / "checkpoints"
    saida = Path(args.output_dir) / "monitor_externo.jsonl"
    vistos: set[int] = set()
    if saida.is_file():
        for l in saida.read_text(encoding="utf-8").splitlines():
            if l.strip():
                vistos.add(int(json.loads(l)["step"]))

    while True:
        novos = [c for c in checkpoints(dir_ckpt) if steps_de(c) not in vistos]
        for ck in novos:
            st = steps_de(ck)
            try:
                load_lora_checkpoint_into_backbone(backbone, ck)
            except Exception as exc:  # noqa: BLE001
                print(f"[monitor] step {st}: nao carregou ({exc})", flush=True)
                continue
            t0 = time.time()
            por_img = run_lvcorr_probe(
                backbone=backbone, text_embeddings=model.text_embeddings,
                conjunto=conjunto,
                num_inference_steps=int(cfg.runtime.probe_num_inference_steps),
                seed=int(cfg.runtime.probe_seed))
            lv = lvcorr_agregado(por_img) if por_img else float("nan")
            reg = {
                "step": st, "lvcorr": None if np.isnan(lv) else round(float(lv), 4),
                "por_imagem": {k: (None if np.isnan(v) else round(float(v), 4))
                               for k, v in por_img.items()},
                "segundos": round(time.time() - t0, 1),
                "referencia_fase1_anterior": 0.9059,
            }
            with saida.open("a", encoding="utf-8") as h:
                h.write(json.dumps(reg) + "\n")
            vistos.add(st)
            marca = ""
            if reg["lvcorr"] is not None:
                d = reg["lvcorr"] - 0.9059
                marca = f"  ({d:+.4f} vs a fase 1 anterior)"
            print(f"[monitor] step={st} LVCorr={reg['lvcorr']}{marca} "
                  f"({reg['segundos']}s)", flush=True)
        if args.uma_vez:
            break
        time.sleep(args.intervalo)
    return 0


if __name__ == "__main__":
    sys.exit(main())
