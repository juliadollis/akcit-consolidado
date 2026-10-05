#!/usr/bin/env python3
"""Inferência da DeblurNet com TODOS os eixos expostos por CLI.

POR QUE NÃO USAR O `Inference_deblurNet.py` OFICIAL DIRETO
----------------------------------------------------------
Ele fixa os eixos que os experimentos do plano precisam VARIAR:
`guidance_scale` (fica no default 3.5), `main_adapter` (fica None) e não tem
como separar o adapter do branch de texto do adapter do branch principal.

Este script preserva o comportamento oficial nos defaults e permite variar cada
eixo. Os defaults reproduzem `Inference_deblurNet.py` exatamente:
long_side=0, steps=28, guidance_scale=3.5, main_adapter=None,
text_adapter=igual ao main, seed=42, tiling ligado quando min(w,h) >= 512.

O DESVIO DO `--text-adapter` (C6)
---------------------------------
O `generate` oficial monta, nas TRÊS chamadas a `transformer_forward`
(ramo com tiling, ramo sem tiling e ramo `image_guidance_scale != 1.0`):

    adapters = [main_adapter] * 2 + c_adapters

e em `transformer_forward` o índice 0 é o branch de **TEXTO** (txt_n = 1), não o
principal. Em `single_block_forward` o texto passa por
`specify_lora((self.norm.linear, self.proj_mlp), adapters[0])` e
`specify_lora((self.proj_out,), adapters[0])`. Ou seja: passar
`main_adapter="deblurring"` — necessário para o nosso modelo main+cond — liga
LoRA no texto em 38 blocos single, coisa que o treino NUNCA fez (ele usa
`adapters = [None, ADAPTER, ADAPTER]`, com o texto sem adapter).

Nos blocos duplos não há efeito: os módulos do texto (`norm1_context.linear`,
`add_q_proj`, `to_add_out`, `ff_context.net.2`) não são alvos de LoRA nem passam
por `specify_lora`.

COMO O DESVIO FOI IMPLEMENTADO — e por que monkeypatch e não cópia do flux.py
-----------------------------------------------------------------------------
`third_party/Genfocus/` é clone de referência e não pode ser modificado.
Verificado por AST que, nas três chamadas, `transformer_forward` é resolvido
como GLOBAL do módulo (`ast.Name`), que `adapters` e `text_features` vão sempre
por palavra-chave, e que `generate` não tem parâmetro local com esse nome. Logo
substituir `Genfocus.pipeline.flux.transformer_forward` intercepta as três de
uma vez, sem duplicar as 882 linhas do arquivo — uma cópia divergiria em
silêncio na primeira atualização do upstream.

O patch só é aplicado quando o texto REALMENTE difere do main; no caminho
padrão nada é trocado e a execução é idêntica à oficial.

USO
---
    # reproduz a inferência oficial
    python3 scripts/infer_deblur.py --lora pesos/deblurNet.safetensors -i foto.jpg -o out.png

    # nosso modelo main+cond, com o texto FORA do adapter (o que o treino fez)
    python3 scripts/infer_deblur.py --lora nosso.safetensors -i foto.jpg -o out.png \
        --main-adapter deblurring --text-adapter none

    # varredura de guidance (experimento do C2)
    python3 scripts/infer_deblur.py --lora nosso.safetensors --dataset akcit-pixel/DDPD \
        --split test -n 20 -o saidas/g1.0 --guidance-scale 1.0

GPU: sim. Tempo: ~50 s/imagem numa A6000 sem tiling (paper §4.1).
"""

from __future__ import annotations

import argparse
import contextlib
import json
from pathlib import Path

from _comum import adicionar_raiz_ao_path, token_hf

adicionar_raiz_ao_path()

MODEL_ID = "black-forest-labs/FLUX.1-dev"
PROMPT = "a sharp photo with everything in focus"   # == Inference_deblurNet.py
NOME_ADAPTER = "deblurring"                          # == Inference_deblurNet.py
MESMO_QUE_MAIN = "__same_as_main__"


# =============================================================================
# Pré-processamento — cópia VERBATIM do resize_and_pad_image oficial
# =============================================================================

def resize_and_pad_image(img, target_long_side: int):
    """Idêntico ao `Inference_deblurNet.py` oficial. Não "melhorar".

    Dois caminhos, e a diferença entre eles importa para o C4:
      long_side > 0 → redimensiona o lado MAIOR, TRUNCA a múltiplo de 16 e corta no centro
      long_side = 0 → mantém a resolução e arredonda PARA CIMA a múltiplo de 16
    """
    from PIL import Image

    w, h = img.size

    if target_long_side and target_long_side > 0:
        target_max = int(target_long_side)
        if w >= h:
            new_w = target_max
            scale = target_max / w
            new_h = int(h * scale)
        else:
            new_h = target_max
            scale = target_max / h
            new_w = int(w * scale)

        img = img.resize((new_w, new_h), Image.LANCZOS)

        final_w = (new_w // 16) * 16
        final_h = (new_h // 16) * 16
        final_w = max(final_w, 16)
        final_h = max(final_h, 16)

        left = (new_w - final_w) // 2
        top = (new_h - final_h) // 2
        return img.crop((left, top, left + final_w, top + final_h))

    final_w = ((w + 15) // 16) * 16
    final_h = ((h + 15) // 16) * 16
    if final_w == w and final_h == h:
        return img
    return img.resize((final_w, final_h), Image.LANCZOS)


# =============================================================================
# C6 — monkeypatch do adapter do branch de texto
# =============================================================================

@contextlib.contextmanager
def adapter_de_texto(text_adapter):
    """Faz o branch de TEXTO usar `text_adapter` em vez do `main_adapter`.

    `text_adapter=MESMO_QUE_MAIN` → não faz nada (comportamento upstream).

    Intercepta `Genfocus.pipeline.flux.transformer_forward`, que as três
    chamadas dentro de `generate` resolvem como global no momento da chamada.
    Só reescreve as `txt_n` primeiras posições de `adapters`, calculando `txt_n`
    a partir de `text_features` em vez de assumir 1.
    """
    import Genfocus.pipeline.flux as flux

    if text_adapter == MESMO_QUE_MAIN:
        yield False
        return

    original = flux.transformer_forward

    def wrapper(transformer, *args, **kwargs):
        adapters = kwargs.get("adapters")
        text_features = kwargs.get("text_features")
        if adapters is not None and text_features is not None:
            txt_n = len(text_features)
            novos = list(adapters)
            for i in range(min(txt_n, len(novos))):
                novos[i] = text_adapter
            kwargs["adapters"] = novos
        return original(transformer, *args, **kwargs)

    wrapper.__wrapped__ = original
    flux.transformer_forward = wrapper
    try:
        yield True
    finally:
        flux.transformer_forward = original


# =============================================================================
# Entradas
# =============================================================================

def carregar_entradas(args) -> list[tuple[str, "object"]]:
    """Devolve [(identificador, PIL.Image), ...] a partir de --input ou --dataset."""
    from PIL import Image

    itens: list[tuple[str, object]] = []

    if args.input:
        caminhos: list[Path] = []
        for padrao in args.input:
            p = Path(padrao)
            if p.is_dir():
                for ext in ("*.png", "*.jpg", "*.jpeg", "*.webp", "*.bmp"):
                    caminhos.extend(sorted(p.glob(ext)))
            elif p.exists():
                caminhos.append(p)
            else:
                caminhos.extend(sorted(Path().glob(padrao)))
        if not caminhos:
            raise SystemExit(f"Nenhuma imagem encontrada em: {args.input}")
        for c in caminhos[: args.n] if args.n else caminhos:
            itens.append((c.stem, Image.open(c).convert("RGB")))
        return itens

    from datasets import load_dataset

    token = token_hf()
    split = f"{args.split}[:{args.n}]" if args.n else args.split
    ds = load_dataset(args.dataset, split=split, token=token)
    print(f"[entrada] {len(ds)} linhas de {args.dataset}:{args.split}")
    for i, linha in enumerate(ds):
        nome = str(linha.get("file_name_base") or f"{i:05d}")
        img = linha[args.coluna_entrada]
        itens.append((nome, img.convert("RGB")))
    return itens


# =============================================================================
# Principal
# =============================================================================

def main() -> None:
    p = argparse.ArgumentParser(
        description="Inferência da DeblurNet com guidance, adapters e escala configuráveis.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog="Os defaults reproduzem exatamente o Inference_deblurNet.py oficial.",
    )
    p.add_argument("--lora", required=True, help="caminho do .safetensors do LoRA")
    p.add_argument("-i", "--input", nargs="*", default=None,
                   help="imagem, pasta ou glob (alternativa a --dataset)")
    p.add_argument("--dataset", default=None, help="repo HF, ex. akcit-pixel/DDPD")
    p.add_argument("--split", default="test")
    p.add_argument("--coluna-entrada", default="image_blur")
    p.add_argument("-n", type=int, default=None, help="limita o nº de imagens")
    p.add_argument("-o", "--out", required=True, help="arquivo (1 imagem) ou pasta")

    p.add_argument("--steps", type=int, default=28, help="passos de denoise (paper §4.1)")
    p.add_argument("--long-side", type=int, default=0,
                   help="0 = resolução nativa (default oficial)")
    p.add_argument("--guidance-scale", type=float, default=3.5,
                   help="default do generate oficial; o deblur oficial NÃO passa este valor")
    p.add_argument("--main-adapter", default="none",
                   help="'none' (cond-only, oficial) ou 'deblurring' (main+cond)")
    p.add_argument("--text-adapter", default=MESMO_QUE_MAIN,
                   help="'none' tira o LoRA do texto (C6); default segue o main-adapter")
    p.add_argument("--no-tiling", action="store_true", help="NO_TILED_DENOISE=True")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--salvar-protocolo", default=None,
                   help="JSON com os parâmetros usados (default: <out>/protocolo.json)")
    args = p.parse_args()

    if not args.input and not args.dataset:
        raise SystemExit("Informe --input ou --dataset.")
    if not Path(args.lora).is_file():
        raise SystemExit(f"LoRA não encontrado: {args.lora}")

    def _norm(v):
        return None if str(v).lower() in {"none", "null", ""} else v

    main_adapter = _norm(args.main_adapter)
    text_adapter = args.text_adapter if args.text_adapter == MESMO_QUE_MAIN else _norm(args.text_adapter)

    import torch
    from diffusers import FluxPipeline
    from Genfocus.pipeline.flux import Condition, generate, seed_everything

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    print(f"[infer] device={device} dtype={dtype}")

    itens = carregar_entradas(args)
    print(f"[infer] {len(itens)} imagem(ns)")

    print("[infer] carregando FLUX...")
    pipe = FluxPipeline.from_pretrained(MODEL_ID, torch_dtype=dtype)
    if device == "cuda":
        pipe.to("cuda")

    lora = Path(args.lora)
    pipe.load_lora_weights(str(lora.parent), weight_name=lora.name, adapter_name=NOME_ADAPTER)
    pipe.set_adapters([NOME_ADAPTER])
    print(f"[infer] LoRA carregado: {lora.name}")

    destino = Path(args.out)
    varios = len(itens) > 1 or destino.suffix == ""
    if varios:
        destino.mkdir(parents=True, exist_ok=True)
    else:
        destino.parent.mkdir(parents=True, exist_ok=True)

    protocolo = {
        "lora": str(lora), "lora_bytes": lora.stat().st_size,
        "prompt": PROMPT, "steps": args.steps, "long_side": args.long_side,
        "guidance_scale": args.guidance_scale,
        "main_adapter": main_adapter,
        "text_adapter": ("igual_ao_main" if text_adapter == MESMO_QUE_MAIN else text_adapter),
        "no_tiling_forcado": bool(args.no_tiling),
        "seed": args.seed, "model_id": MODEL_ID,
        "variante": ("cond-only" if main_adapter is None else "main+cond"),
        "imagens": [],
    }

    with adapter_de_texto(text_adapter) as aplicou:
        if aplicou:
            print(f"[infer] DESVIO C6 ativo: branch de texto usa adapter={text_adapter!r}")
        else:
            print("[infer] sem desvio: branch de texto segue o main_adapter (upstream)")

        for k, (nome, img) in enumerate(itens, 1):
            proc = resize_and_pad_image(img, args.long_side)
            w, h = proc.size
            sem_tiling = bool(args.no_tiling) or min(w, h) < 512

            print(f"[infer] {k}/{len(itens)} {nome}: {img.size[0]}×{img.size[1]} "
                  f"→ {w}×{h} | tiling={'off' if sem_tiling else 'on'}")

            cond = Condition(proc, NOME_ADAPTER, [0, 0], 1.0)
            seed_everything(args.seed)

            with torch.no_grad():
                saida = generate(
                    pipe, height=h, width=w, prompt=PROMPT,
                    num_inference_steps=args.steps,
                    guidance_scale=args.guidance_scale,
                    conditions=[cond],
                    main_adapter=main_adapter,
                    NO_TILED_DENOISE=sem_tiling,
                ).images[0]

            caminho = (destino / f"{nome}.png") if varios else destino
            saida.save(caminho)
            protocolo["imagens"].append({
                "nome": nome, "entrada_wh": list(img.size),
                "processada_wh": [w, h], "tiling": not sem_tiling,
                "saida": str(caminho),
            })
            print(f"           salvo em {caminho}")

    jp = Path(args.salvar_protocolo) if args.salvar_protocolo else (
        (destino if varios else destino.parent) / "protocolo.json")
    jp.parent.mkdir(parents=True, exist_ok=True)
    jp.write_text(json.dumps(protocolo, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[infer] protocolo salvo em {jp}")
    print("[infer] Sem este JSON as métricas não são interpretáveis depois (C11-1).")


if __name__ == "__main__":
    main()
