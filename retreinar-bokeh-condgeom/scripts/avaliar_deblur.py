#!/usr/bin/env python3
"""Avalia a DeblurNet: LPIPS e DISTS contra `image_focus`, com o PROTOCOLO registrado.

POR QUE O REGISTRO DO PROTOCOLO É O PONTO PRINCIPAL (C11-1)
------------------------------------------------------------
Comparabilidade de métrica é questão do PROTOCOLO DE AVALIAÇÃO, não da
resolução de treino. LPIPS não é invariante a escala: calcular a métrica em
1024×688 e em 1680×1120 dá números diferentes para o mesmo modelo, e a direção
do efeito não deve ser afirmada sem medir.

Disso decorrem duas coisas, e este script existe para tornar as duas explícitas:

* Comparar **nosso vs oficial DENTRO deste pipeline** é justo em qualquer
  resolução — os dois passam pelo mesmo protocolo. É o controle correto, e o
  projeto já o tem.
* Comparar **nosso número com o número PUBLICADO no paper** exige conferir os
  dois protocolos. Sem o JSON que este script emite, nenhuma tabela montada
  depois é interpretável.

Por isso a saída registra: resolução em que a métrica foi calculada, se o GT
passou por resize (e por qual caminho), `long_side`, `guidance_scale`, variante
de adapter, e identificação do peso (caminho, tamanho, sha256).

ATENÇÃO ao GT: por padrão o ground-truth passa pelo MESMO `resize_and_pad_image`
da entrada, que é o que a nossa pipeline de avaliação já fazia. Com
`long_side=0` essa função arredonda PARA CIMA ao múltiplo de 16 (não reduz).
`--gt-sem-resize` desliga isso — mas aí predição e GT podem ficar com shapes
diferentes e o script aborta em vez de redimensionar em silêncio.

USO
---
    python3 scripts/avaliar_deblur.py --lora pesos/deblurNet.safetensors \
        --dataset akcit-pixel/DDPD --split test --out outputs/eval_oficial

    # varredura de guidance do C2, no MESMO peso
    for g in 1.0 3.5; do
      python3 scripts/avaliar_deblur.py --lora nosso.safetensors --guidance-scale $g \
          --out outputs/eval_g$g
    done

GPU: sim. Tempo: ~50 s/imagem + métricas.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path

from _comum import adicionar_raiz_ao_path, imprimir_tabela, token_hf

adicionar_raiz_ao_path()

MESMO_QUE_MAIN = "__same_as_main__"


def sha256_arquivo(p: Path, limite_mb: int = 4096) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for bloco in iter(lambda: fh.read(1 << 20), b""):
            h.update(bloco)
    return h.hexdigest()


def carregar_metricas(device: str):
    """Instancia LPIPS e DISTS do pyiqa, com erro claro se faltar."""
    try:
        import pyiqa
    except ImportError as exc:
        raise SystemExit(
            "ERRO: `pyiqa` não instalado — é ele que fornece LPIPS e DISTS.\n"
            "  pip install pyiqa\n"
            "(traz torch/torchvision como dependência; no cluster, instale dentro do container)"
        ) from exc

    metricas = {}
    for nome in ("lpips", "dists"):
        try:
            metricas[nome] = pyiqa.create_metric(nome, device=device)
        except Exception as exc:  # noqa: BLE001
            print(f"  AVISO: não consegui criar a métrica {nome}: {exc}")
    if not metricas:
        raise SystemExit("Nenhuma métrica pôde ser criada; abortando.")
    # registra a variante EXATA de cada métrica: 'lpips' e 'lpips+' são
    # implementações diferentes e não são comparáveis entre si.
    detalhes = {}
    for nome, m in metricas.items():
        detalhes[nome] = {
            "metric_name": getattr(m, "metric_name", nome),
            "lower_better": bool(getattr(m, "lower_better", True)),
            "net": str(getattr(getattr(m, "net", None), "__class__", type(None)).__name__),
        }
    return metricas, detalhes


def para_tensor(img, device):
    import numpy as np
    import torch

    a = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(a).permute(2, 0, 1).unsqueeze(0).to(device)


def main() -> None:
    p = argparse.ArgumentParser(
        description="Avalia a DeblurNet (LPIPS/DISTS) registrando o protocolo completo.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--lora", required=True)
    p.add_argument("--dataset", default="akcit-pixel/DDPD")
    p.add_argument("--split", default="test")
    p.add_argument("-n", type=int, default=None, help="limita o nº de imagens")
    p.add_argument("--out", required=True, help="pasta de saída")
    p.add_argument("--coluna-entrada", default="image_blur")
    p.add_argument("--coluna-gt", default="image_focus")

    p.add_argument("--steps", type=int, default=28)
    p.add_argument("--long-side", type=int, default=0)
    p.add_argument("--guidance-scale", type=float, default=3.5)
    p.add_argument("--main-adapter", default="none")
    p.add_argument("--text-adapter", default=MESMO_QUE_MAIN)
    p.add_argument("--no-tiling", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--gt-sem-resize", action="store_true",
                   help="NÃO passa o GT pelo resize_and_pad_image (aborta se o shape divergir)")
    p.add_argument("--salvar-imagens", action="store_true", help="grava os PNGs gerados")
    args = p.parse_args()

    lora = Path(args.lora)
    if not lora.is_file():
        raise SystemExit(f"LoRA não encontrado: {lora}")

    saida = Path(args.out)
    saida.mkdir(parents=True, exist_ok=True)

    def _norm(v):
        return None if str(v).lower() in {"none", "null", ""} else v

    main_adapter = _norm(args.main_adapter)
    text_adapter = args.text_adapter if args.text_adapter == MESMO_QUE_MAIN else _norm(args.text_adapter)

    import torch
    from datasets import load_dataset
    from diffusers import FluxPipeline
    from Genfocus.pipeline.flux import Condition, generate, seed_everything

    from infer_deblur import (
        MODEL_ID, NOME_ADAPTER, PROMPT, adapter_de_texto, resize_and_pad_image,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    print(f"[eval] device={device}")

    metricas, detalhes_metricas = carregar_metricas(device)
    print(f"[eval] métricas: {list(metricas)}")

    token = token_hf()
    split = f"{args.split}[:{args.n}]" if args.n else args.split
    ds = load_dataset(args.dataset, split=split, token=token)
    print(f"[eval] {len(ds)} amostras de {args.dataset}:{args.split}")

    print("[eval] carregando FLUX...")
    pipe = FluxPipeline.from_pretrained(MODEL_ID, torch_dtype=dtype)
    if device == "cuda":
        pipe.to("cuda")
    pipe.load_lora_weights(str(lora.parent), weight_name=lora.name, adapter_name=NOME_ADAPTER)
    pipe.set_adapters([NOME_ADAPTER])

    protocolo = {
        # ── o que foi avaliado ──────────────────────────────────────────────
        "peso": {"caminho": str(lora), "nome": lora.name,
                 "bytes": lora.stat().st_size, "sha256": sha256_arquivo(lora)},
        "variante_adapter": ("cond-only" if main_adapter is None else "main+cond"),
        "main_adapter": main_adapter,
        "text_adapter": ("igual_ao_main" if text_adapter == MESMO_QUE_MAIN else text_adapter),
        # ── como foi gerado ─────────────────────────────────────────────────
        "model_id": MODEL_ID, "prompt": PROMPT, "steps": args.steps,
        "guidance_scale": args.guidance_scale, "long_side": args.long_side,
        "no_tiling_forcado": bool(args.no_tiling), "seed": args.seed,
        # ── como a métrica foi calculada (o ponto do C11-1) ─────────────────
        "dataset": args.dataset, "split": args.split, "n_amostras": len(ds),
        "gt_passou_por_resize": not args.gt_sem_resize,
        "gt_resize_fn": ("nenhum" if args.gt_sem_resize else
                         f"resize_and_pad_image(long_side={args.long_side})"),
        "gt_resize_obs": ("long_side=0 arredonda PARA CIMA ao múltiplo de 16, não reduz"
                          if args.long_side == 0 else
                          "long_side>0 reduz o lado maior, trunca a múltiplo de 16 e corta no centro"),
        "metricas": detalhes_metricas,
        "resolucoes_da_metrica": [],
        "ambiente": {"python": sys.version.split()[0], "plataforma": platform.platform(),
                     "torch": torch.__version__},
        "aviso": ("LPIPS/DISTS não são invariantes a escala. Comparar com números "
                  "PUBLICADOS exige conferir o protocolo dos dois lados; comparar "
                  "modelos DENTRO deste JSON é justo."),
    }

    por_imagem = []
    resolucoes = set()

    with adapter_de_texto(text_adapter) as aplicou:
        if aplicou:
            print(f"[eval] DESVIO C6 ativo: texto com adapter={text_adapter!r}")

        for i, linha in enumerate(ds, 1):
            nome = str(linha.get("file_name_base") or f"{i:05d}")
            entrada = linha[args.coluna_entrada].convert("RGB")
            gt = linha[args.coluna_gt].convert("RGB")

            proc = resize_and_pad_image(entrada, args.long_side)
            w, h = proc.size
            sem_tiling = bool(args.no_tiling) or min(w, h) < 512

            cond = Condition(proc, NOME_ADAPTER, [0, 0], 1.0)
            seed_everything(args.seed)
            with torch.no_grad():
                pred = generate(
                    pipe, height=h, width=w, prompt=PROMPT,
                    num_inference_steps=args.steps,
                    guidance_scale=args.guidance_scale,
                    conditions=[cond], main_adapter=main_adapter,
                    NO_TILED_DENOISE=sem_tiling,
                ).images[0]

            gt_proc = gt if args.gt_sem_resize else resize_and_pad_image(gt, args.long_side)
            if pred.size != gt_proc.size:
                raise SystemExit(
                    f"Shape divergente em {nome}: predição {pred.size} vs GT {gt_proc.size}.\n"
                    "Redimensionar em silêncio aqui falsificaria a métrica. "
                    "Rode sem --gt-sem-resize, ou investigue a amostra."
                )
            resolucoes.add(tuple(pred.size))

            t_pred = para_tensor(pred, device)
            t_gt = para_tensor(gt_proc, device)
            valores = {}
            for nm, m in metricas.items():
                with torch.no_grad():
                    valores[nm] = float(m(t_pred, t_gt).item())

            if args.salvar_imagens:
                (saida / "imagens").mkdir(exist_ok=True)
                pred.save(saida / "imagens" / f"{nome}.png")

            por_imagem.append({"nome": nome, "wh": list(pred.size), **valores})
            txt = "  ".join(f"{k.upper()} {v:.4f}" for k, v in valores.items())
            print(f"[eval] {i}/{len(ds)} {nome} ({w}×{h})  {txt}")

    # ── agregação ───────────────────────────────────────────────────────────
    resumo = {}
    for nm in metricas:
        vals = [r[nm] for r in por_imagem if nm in r]
        if vals:
            vals_ord = sorted(vals)
            resumo[nm] = {
                "media": sum(vals) / len(vals),
                "mediana": vals_ord[len(vals_ord) // 2],
                "min": vals_ord[0], "max": vals_ord[-1], "n": len(vals),
            }

    protocolo["resolucoes_da_metrica"] = [
        {"w": w, "h": h} for (w, h) in sorted(resolucoes)
    ]
    protocolo["resultado"] = resumo

    (saida / "protocolo_e_resultado.json").write_text(
        json.dumps(protocolo, ensure_ascii=False, indent=2), encoding="utf-8")
    (saida / "por_imagem.json").write_text(
        json.dumps(por_imagem, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n" + "=" * 72)
    print("RESULTADO")
    print("=" * 72)
    imprimir_tabela(
        [{"métrica": k.upper(), "média": f"{v['media']:.4f}",
          "mediana": f"{v['mediana']:.4f}", "n": v["n"]} for k, v in resumo.items()],
        ["métrica", "média", "mediana", "n"],
    )
    print(f"\n  protocolo: guidance={args.guidance_scale} long_side={args.long_side} "
          f"adapter={protocolo['variante_adapter']} "
          f"texto={protocolo['text_adapter']} steps={args.steps}")
    print(f"  resoluções em que a métrica foi calculada: "
          f"{[f'{w}×{h}' for w, h in sorted(resolucoes)]}")
    print(f"\n[eval] salvo em {saida}/protocolo_e_resultado.json")


if __name__ == "__main__":
    main()
