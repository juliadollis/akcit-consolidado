#!/usr/bin/env python3
"""Avalia a DeblurNet nas 5 métricas da Tabela 2, registrando o PROTOCOLO inteiro.

POR QUE O REGISTRO DO PROTOCOLO É O PONTO PRINCIPAL
---------------------------------------------------
Comparabilidade de métrica é questão do PROTOCOLO DE AVALIAÇÃO. LPIPS e DISTS
não são invariantes a escala: a mesma predição avaliada em 1024×688 e em
1680×1120 dá números diferentes. E as variantes do `pyiqa` (`lpips` vs `lpips+`)
são redes diferentes, que também não são comparáveis entre si.

Disso decorrem duas coisas que este script torna explícitas:

* Comparar **modelos DENTRO deste JSON** é justo — todos passaram pelo mesmo
  protocolo. É o controle correto.
* Comparar **nosso número com o PUBLICADO no paper** exige conferir os dois
  protocolos, e o do paper é parcialmente desconhecido (ele não diz resolução
  de avaliação nem variante de métrica). Ver PROTOCOLO.md.

Por isso o JSON registra: resolução em que cada métrica foi calculada, se o GT
passou por resize e por qual caminho, `long_side`, `guidance_scale`, variante de
adapter (cond-only vs main+cond), adapter do texto, variante EXATA de cada uma
das 5 métricas, e a identificação do peso (caminho, bytes, sha256).

DOIS DEFEITOS DO PIPELINE DO TIME QUE ESTE SCRIPT NÃO REPETE
------------------------------------------------------------
1. `vision-pipeline/evaluation/src/core/evaluator.py:60-78` redimensiona o GT
   com bicubic EM SILÊNCIO quando os shapes divergem. Isso falsifica a métrica.
   Aqui shape divergente ABORTA.
2. O mesmo arquivo, linha 47-52, registra 0.0 quando uma métrica lança exceção.
   Para LPIPS/DISTS (menor é melhor) um erro vira nota perfeita. Aqui um erro
   devolve None, é contado e aparece no resumo.

USO
---
    # nosso peso, cond-only (default oficial)
    python3 inferencia/avaliar_deblur.py --lora pesos/deblur.safetensors \
        --dataset akcit-pixel/DDPD --split test --out saidas/ddpd_nosso

    # peso main+cond (exige main_adapter; e o texto FORA do adapter — C6)
    python3 inferencia/avaliar_deblur.py --lora nosso.safetensors \
        --main-adapter deblurring --text-adapter none --out saidas/ddpd_maincond

    # linha Input do paper (identidade: blurry vs focus, sem modelo)
    python3 inferencia/avaliar_deblur.py --identidade \
        --dataset akcit-pixel/DDPD --split test --out saidas/ddpd_input

GPU: sim (exceto --identidade, que não carrega o FLUX).
Tempo: ~50 s/imagem + métricas (paper §4.1).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path

import metricas as M
from _bootstrap import imprimir_tabela, token_hf

MESMO_QUE_MAIN = "__same_as_main__"


def sha256_arquivo(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for bloco in iter(lambda: fh.read(1 << 20), b""):
            h.update(bloco)
    return h.hexdigest()


def _norm_adapter(v):
    return None if str(v).lower() in {"none", "null", ""} else v


def avaliar(
    *,
    dataset: str,
    split: str,
    out: Path,
    lora: Path | None = None,
    identidade: bool = False,
    n: int | None = None,
    coluna_entrada: str = "image_blur",
    coluna_gt: str = "image_focus",
    steps: int = 28,
    long_side: int = 0,
    guidance_scale: float = 3.5,
    main_adapter=None,
    text_adapter=MESMO_QUE_MAIN,
    no_tiling: bool = False,
    seed: int = 42,
    gt_sem_resize: bool = False,
    salvar_imagens: bool = False,
    conjunto_variantes: str = "time",
    sobrescrever_variantes: dict | None = None,
    rotulo: str | None = None,
) -> dict:
    """Roda inferência (ou identidade) + as 5 métricas. Devolve o dict do protocolo.

    `identidade=True` reproduz a linha `Input` do paper: não carrega o FLUX e
    usa a própria imagem borrada como "predição". É a referência que diz se um
    número significa "o modelo melhorou" ou "o modelo não piorou".
    """
    import torch
    from datasets import load_dataset

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[eval] device={device} | modo={'IDENTIDADE' if identidade else 'modelo'}")

    metricas, detalhes_metricas = M.criar_metricas(
        conjunto=conjunto_variantes, device=device, sobrescrever=sobrescrever_variantes)
    print(f"[eval] métricas: " + ", ".join(
        f"{k}={v['variante_pyiqa']}" for k, v in detalhes_metricas.items()))

    tok = token_hf()
    split_q = f"{split}[:{n}]" if n else split
    ds = load_dataset(dataset, split=split_q, token=tok)
    print(f"[eval] {len(ds)} amostras de {dataset}:{split}")

    # `resize_and_pad_image` é importado do infer_deblur, que é cópia VERBATIM do
    # oficial — o GT tem de passar pelo MESMO caminho que a entrada.
    from infer_deblur import resize_and_pad_image

    pipe = None
    gerar = None
    if not identidade:
        if lora is None or not Path(lora).is_file():
            raise SystemExit(f"LoRA não encontrado: {lora}")
        from diffusers import FluxPipeline
        from Genfocus.pipeline.flux import Condition, generate, seed_everything
        from infer_deblur import MODEL_ID, NOME_ADAPTER, PROMPT, adapter_de_texto

        dtype = torch.bfloat16 if device == "cuda" else torch.float32
        print("[eval] carregando FLUX...")
        pipe = FluxPipeline.from_pretrained(MODEL_ID, torch_dtype=dtype)
        if device == "cuda":
            pipe.to("cuda")
        lora = Path(lora)
        pipe.load_lora_weights(str(lora.parent), weight_name=lora.name,
                               adapter_name=NOME_ADAPTER)
        pipe.set_adapters([NOME_ADAPTER])
        print(f"[eval] LoRA: {lora.name}")

        def gerar(proc, w, h, sem_tiling):  # noqa: F811
            cond = Condition(proc, NOME_ADAPTER, [0, 0], 1.0)
            seed_everything(seed)
            with torch.no_grad():
                return generate(
                    pipe, height=h, width=w, prompt=PROMPT,
                    num_inference_steps=steps, guidance_scale=guidance_scale,
                    conditions=[cond], main_adapter=main_adapter,
                    NO_TILED_DENOISE=sem_tiling,
                ).images[0]

    protocolo = {
        "rotulo": rotulo or ("Input (identidade)" if identidade else "modelo"),
        "modo": "identidade" if identidade else "modelo",
        # ── o que foi avaliado ──────────────────────────────────────────────
        "peso": None if identidade else {
            "caminho": str(lora), "nome": Path(lora).name,
            "bytes": Path(lora).stat().st_size, "sha256": sha256_arquivo(Path(lora)),
        },
        "variante_adapter": None if identidade else (
            "cond-only" if main_adapter is None else "main+cond"),
        "main_adapter": None if identidade else main_adapter,
        "text_adapter": None if identidade else (
            "igual_ao_main" if text_adapter == MESMO_QUE_MAIN else text_adapter),
        # ── como foi gerado ─────────────────────────────────────────────────
        "steps": None if identidade else steps,
        "guidance_scale": None if identidade else guidance_scale,
        "no_tiling_forcado": bool(no_tiling),
        "seed": seed,
        # ── como a métrica foi calculada (o ponto) ──────────────────────────
        "dataset": dataset, "split": split, "n_amostras": len(ds),
        "long_side": long_side,
        "gt_passou_por_resize": not gt_sem_resize,
        "gt_resize_fn": ("nenhum" if gt_sem_resize
                         else f"resize_and_pad_image(long_side={long_side})"),
        "gt_resize_obs": ("long_side=0 arredonda PARA CIMA ao múltiplo de 16, não reduz"
                          if long_side == 0 else
                          "long_side>0 reduz o lado maior, trunca a múltiplo de 16 e corta no centro"),
        "metricas": detalhes_metricas,
        "resolucoes_da_metrica": [],
        "ambiente": {"python": sys.version.split()[0],
                     "plataforma": platform.platform(),
                     "torch": torch.__version__},
        "aviso": ("LPIPS/DISTS não são invariantes a escala, e as variantes do pyiqa "
                  "não são intercambiáveis. Comparar com números PUBLICADOS exige "
                  "conferir o protocolo dos dois lados. Ver PROTOCOLO.md."),
    }

    por_imagem: list[dict] = []
    resolucoes: set[tuple[int, int]] = set()

    import contextlib
    ctx = contextlib.nullcontext(False)
    if not identidade:
        from infer_deblur import adapter_de_texto
        ctx = adapter_de_texto(text_adapter)

    with ctx as aplicou:
        if aplicou:
            print(f"[eval] DESVIO C6 ativo: texto com adapter={text_adapter!r}")

        for i, linha in enumerate(ds, 1):
            nome = str(linha.get("file_name_base") or f"{i:05d}")
            entrada = linha[coluna_entrada].convert("RGB")
            gt = linha[coluna_gt].convert("RGB")

            proc = resize_and_pad_image(entrada, long_side)
            w, h = proc.size
            sem_tiling = bool(no_tiling) or min(w, h) < 512

            pred = proc if identidade else gerar(proc, w, h, sem_tiling)

            gt_proc = gt if gt_sem_resize else resize_and_pad_image(gt, long_side)
            if pred.size != gt_proc.size:
                raise SystemExit(
                    f"Shape divergente em {nome}: predição {pred.size} vs GT {gt_proc.size}.\n"
                    "NÃO vou redimensionar em silêncio — isso falsificaria a métrica\n"
                    "(é o defeito de core/evaluator.py:60-78 do pipeline do time).\n"
                    "Rode sem --gt-sem-resize, ou investigue esta amostra."
                )
            resolucoes.add(tuple(pred.size))

            valores = M.calcular(metricas, M.para_tensor(pred, device),
                                 M.para_tensor(gt_proc, device))

            if salvar_imagens and not identidade:
                (out / "imagens").mkdir(exist_ok=True)
                pred.save(out / "imagens" / f"{nome}.png")

            por_imagem.append({"nome": nome, "wh": list(pred.size), **valores})
            txt = "  ".join(f"{k} {v:.4f}" if v is not None else f"{k} ERRO"
                            for k, v in valores.items())
            print(f"[eval] {i}/{len(ds)} {nome} ({w}×{h})  {txt}")

    resumo = M.agregar(por_imagem)
    protocolo["resolucoes_da_metrica"] = [{"w": w, "h": h} for (w, h) in sorted(resolucoes)]
    protocolo["resultado"] = resumo

    (out / "protocolo_e_resultado.json").write_text(
        json.dumps(protocolo, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "por_imagem.json").write_text(
        json.dumps(por_imagem, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n" + "=" * 74)
    print(f"RESULTADO — {protocolo['rotulo']} | {dataset}:{split}")
    print("=" * 74)
    imprimir_tabela(
        [{"métrica": k, "sentido": v.get("sentido", ""),
          "média": f"{v['media']:.4f}" if v.get("media") is not None else "—",
          "n": v["n_validos"], "falhas": v["n_falhas"]}
         for k, v in resumo.items()],
        ["métrica", "sentido", "média", "n", "falhas"],
    )
    print(f"\n  resoluções da métrica: {[f'{w}×{h}' for w, h in sorted(resolucoes)]}")
    print(f"[eval] salvo em {out}/protocolo_e_resultado.json")
    return protocolo


def main() -> None:
    p = argparse.ArgumentParser(
        description="Avalia a DeblurNet nas 5 métricas da Tabela 2, com protocolo registrado.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--lora", default=None, help="caminho do .safetensors (não usar com --identidade)")
    p.add_argument("--identidade", action="store_true",
                   help="linha Input do paper: usa a própria borrada como predição")
    p.add_argument("--dataset", default="akcit-pixel/DDPD")
    p.add_argument("--split", default="test")
    p.add_argument("-n", type=int, default=None)
    p.add_argument("--out", required=True)
    p.add_argument("--coluna-entrada", default="image_blur")
    p.add_argument("--coluna-gt", default="image_focus")
    p.add_argument("--steps", type=int, default=28)
    p.add_argument("--long-side", type=int, default=0)
    p.add_argument("--guidance-scale", type=float, default=3.5)
    p.add_argument("--main-adapter", default="none")
    p.add_argument("--text-adapter", default=MESMO_QUE_MAIN)
    p.add_argument("--no-tiling", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--gt-sem-resize", action="store_true")
    p.add_argument("--salvar-imagens", action="store_true")
    p.add_argument("--rotulo", default=None, help="nome desta linha na tabela")
    M.adicionar_args(p)
    args = p.parse_args()

    if args.identidade and args.lora:
        raise SystemExit("--identidade não usa --lora.")
    if not args.identidade and not args.lora:
        raise SystemExit("Informe --lora, ou use --identidade para a linha Input.")

    ta = args.text_adapter if args.text_adapter == MESMO_QUE_MAIN else _norm_adapter(args.text_adapter)
    avaliar(
        dataset=args.dataset, split=args.split, out=Path(args.out),
        lora=Path(args.lora) if args.lora else None, identidade=args.identidade,
        n=args.n, coluna_entrada=args.coluna_entrada, coluna_gt=args.coluna_gt,
        steps=args.steps, long_side=args.long_side, guidance_scale=args.guidance_scale,
        main_adapter=_norm_adapter(args.main_adapter), text_adapter=ta,
        no_tiling=args.no_tiling, seed=args.seed, gt_sem_resize=args.gt_sem_resize,
        salvar_imagens=args.salvar_imagens, conjunto_variantes=args.variantes,
        sobrescrever_variantes=M.variantes_dos_args(args), rotulo=args.rotulo,
    )


if __name__ == "__main__":
    main()
