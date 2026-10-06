"""Inferência de teste do BokehNet — valida o condicionamento ponta a ponta.

Segue EXATAMENTE o contrato do Inference_bokehNet.py oficial:
  - 2 condições, adapter "bokeh":
      cond_img = Condition(aif_pil, "bokeh")                       # No_preprocess=False -> VAE [-1,1]
      cond_dmf = Condition(defocus_[0,1], "bokeh", [0,0], 1.0, No_preprocess=True)  # VAE [0,1]
  - prompt = "an excellent photo with a large aperture", guidance_scale=1.0
  - main_adapter=None  (LoRA cond-only, igual ao oficial e ao NOSSO treino de bokeh)

DIFERENÇA proposital vs oficial: em vez de rodar depth_pro, usamos o `depth`
PRÉ-COMPUTADO do df e RECOMPOMOS o mapa com clip(k*|D-s1|/max_coc, 0, 1) — a
MESMA fórmula que o dataloader de treino usa (defocus_source="recompute") e a
mesma da inferência oficial (defocus_abs/MAX_COC). Assim o teste valida o
conditioning exatamente na representação que o modelo viu no treino.
ATENÇÃO: NÃO usamos a coluna `defocus_map` do df — ela está normalizada POR
IMAGEM (o k não influencia; ver DEFOCUS_SOURCES em genfocus_train/data.py).
`--defocus-source column` existe só para comparar com o comportamento antigo.

Uso:
  python3 scripts/infer_bokeh_test.py \
    --hf-model-repo juliadollis/genrefocus-bokehnet-condlora-4gpu \
    --hf-model-file bokeh.safetensors \
    --source-dataset AKCITPixel3/CMiQdveBBzNii --n 8 \
    [--examples-hf-repo juliadollis/bokeh-exemplos]
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import torch
from datasets import load_dataset
from PIL import Image

from Genfocus.pipeline.flux import Condition, generate, seed_everything

MODEL_ID = "black-forest-labs/FLUX.1-dev"
PROMPT = "an excellent photo with a large aperture"   # == Inference_bokehNet.py
DEFOCUS_U16 = 65535.0


def resize_and_pad(img: Image.Image, long_side: int) -> Image.Image:
    img = img.convert("RGB")
    w, h = img.size
    if long_side:
        if w >= h:
            nw, nh = long_side, int(h * (long_side / w))
        else:
            nh, nw = long_side, int(w * (long_side / h))
        img = img.resize((nw, nh), Image.LANCZOS)
        w, h = img.size
    fw, fh = max((w // 16) * 16, 16), max((h // 16) * 16, 16)
    left, top = (w - fw) // 2, (h - fh) // 2
    return img.crop((left, top, left + fw, top + fh))


def _u16_to_float01(img) -> Image.Image:
    """Imagem uint16 (I;16) -> PIL 'F' em [0,1]."""
    arr = np.asarray(img)
    if arr.ndim == 3:
        arr = arr[..., 0]
    return Image.fromarray(arr.astype(np.float32) / DEFOCUS_U16, mode="F")


def _to_3ch_tensor(m: Image.Image, size_wh, device, dtype) -> torch.Tensor:
    m = m.resize(size_wh, Image.BILINEAR)                   # (w, h) da AIF processada
    a = np.clip(np.asarray(m, dtype=np.float32), 0.0, 1.0)
    t = torch.from_numpy(a)[None, None].repeat(1, 3, 1, 1)  # (1,3,H,W)
    return t.to(device=device, dtype=dtype)


def defocus_recomputed(rec, size_wh, max_coc, device, dtype) -> torch.Tensor:
    """clip(k*|depth-s1|/max_coc, 0, 1) — mesma fórmula do treino e do oficial."""
    depth = np.asarray(_u16_to_float01(rec["depth"]), dtype=np.float32)
    dmap = np.clip(float(rec["k"]) * np.abs(depth - float(rec["s1"])) / float(max_coc), 0.0, 1.0)
    return _to_3ch_tensor(Image.fromarray(dmap, mode="F"), size_wh, device, dtype)


def defocus_from_column(rec, size_wh, device, dtype) -> torch.Tensor:
    """LEGADO: coluna defocus_map do df (normalizada POR IMAGEM — k não age)."""
    return _to_3ch_tensor(_u16_to_float01(rec["defocus_map"]), size_wh, device, dtype)


def build_pipe(args, dtype, token):
    from diffusers import FluxPipeline

    pipe = FluxPipeline.from_pretrained(MODEL_ID, torch_dtype=dtype)
    if torch.cuda.is_available():
        pipe.to("cuda")
    if args.hf_model_repo:
        print(f"[bokeh] LoRA do HF: {args.hf_model_repo}/{args.hf_model_file}")
        pipe.load_lora_weights(
            args.hf_model_repo, weight_name=args.hf_model_file,
            adapter_name="bokeh", token=token,
        )
    else:
        print(f"[bokeh] LoRA local: {args.weight_dir}/{args.weight_name}")
        pipe.load_lora_weights(args.weight_dir, weight_name=args.weight_name, adapter_name="bokeh")
    pipe.set_adapters(["bokeh"])
    return pipe


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--source-dataset", default="AKCITPixel3/CMiQdveBBzNii", help="df com aif/bokeh/defocus_map")
    p.add_argument("--split", default="train")
    p.add_argument("--hf-model-repo", default=None)
    p.add_argument("--hf-model-file", default="bokeh.safetensors")
    p.add_argument("--weight-dir", default="/tmp")
    p.add_argument("--weight-name", default="bokeh.safetensors")
    p.add_argument("--n", type=int, default=8)
    p.add_argument("--steps", type=int, default=28)
    p.add_argument("--long-side", type=int, default=512)
    p.add_argument("--main-adapter", default="none", help="'none' = oficial/cond-only (NOSSO bokeh); 'bokeh' = main+cond")
    p.add_argument("--defocus-source", default="recompute", choices=["recompute", "column"],
                   help="'recompute' (default) = clip(k*|depth-s1|/max_coc,0,1), igual ao treino; "
                        "'column' = coluna defocus_map do df (LEGADO, normalizada por imagem)")
    p.add_argument("--max-coc", type=float, default=100.0)
    p.add_argument("--out-dir", default="/tmp/bokeh_test")
    p.add_argument("--examples-hf-repo", default=None, help="df HF pra subir aif|defocus|gerado|GT")
    # PRIVADO por default: as tiras contêm o `aif` e o `bokeh` (GT) vindos dos
    # datasets AKCITPixel3/*, que são PRIVADOS. Subir aberto publicaria dado de
    # terceiro, e repo público é difícil de despublicar depois (cache/índice).
    # Use --examples-public para abrir conscientemente.
    p.add_argument("--examples-public", action="store_true",
                   help="publica o df de exemplos como PÚBLICO (default: privado)")
    p.add_argument("--hf-token", default=os.environ.get("HF_TOKEN"))
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    main_adapter = None if args.main_adapter.lower() in ("none", "null", "") else args.main_adapter

    pipe = build_pipe(args, dtype, args.hf_token)

    # streaming evita baixar os GB; pegamos só as N primeiras
    ds = load_dataset(args.source_dataset, split=args.split, streaming=True, token=args.hf_token)

    os.makedirs(args.out_dir, exist_ok=True)
    rows = []
    for i, rec in enumerate(ds):
        if i >= args.n:
            break
        aif = resize_and_pad(rec["aif"], args.long_side)
        w, h = aif.size
        if args.defocus_source == "recompute":
            defocus_t = defocus_recomputed(rec, (w, h), args.max_coc, device, dtype)
        else:
            defocus_t = defocus_from_column(rec, (w, h), device, dtype)

        seed_everything(42)
        out = generate(
            pipe,
            height=h, width=w,
            prompt=PROMPT,
            num_inference_steps=args.steps,
            conditions=[
                Condition(aif, "bokeh"),                                  # No_preprocess=False
                Condition(defocus_t, "bokeh", [0, 0], 1.0, No_preprocess=True),
            ],
            guidance_scale=1.0,
            main_adapter=main_adapter,
            NO_TILED_DENOISE=min(w, h) < 512,
        ).images[0]

        gt = rec["bokeh"].convert("RGB").resize((w, h), Image.LANCZOS)
        # grid horizontal: AIF | defocus | gerado | GT
        dv = (defocus_t[0].permute(1, 2, 0).float().cpu().numpy() * 255).astype("uint8")
        strip = Image.new("RGB", (w * 4, h))
        strip.paste(aif, (0, 0))
        strip.paste(Image.fromarray(dv), (w, 0))
        strip.paste(out, (w * 2, 0))
        strip.paste(gt, (w * 3, 0))
        path = os.path.join(args.out_dir, f"bokeh_{i:03d}_{rec.get('stem', i)}.png")
        strip.save(path)
        print(f"[bokeh] {i+1}/{args.n} -> {path}  (main_adapter={main_adapter})")
        rows.append({"aif": aif, "defocus": Image.fromarray(dv), "generated": out, "bokeh_gt": gt,
                     "stem": str(rec.get("stem", i))})

    if args.examples_hf_repo and rows:
        from datasets import Dataset, Features, Value
        from datasets import Image as HFImage
        feats = Features({"aif": HFImage(), "defocus": HFImage(), "generated": HFImage(),
                          "bokeh_gt": HFImage(), "stem": Value("string")})
        Dataset.from_dict(
            {k: [r[k] for r in rows] for k in ("aif", "defocus", "generated", "bokeh_gt", "stem")},
            features=feats,
        ).push_to_hub(
            args.examples_hf_repo,
            token=args.hf_token,
            private=not args.examples_public,
        )
        visib = "PUBLICO" if args.examples_public else "privado"
        print(f"[bokeh] exemplos ({visib}) -> https://huggingface.co/datasets/{args.examples_hf_repo}")


if __name__ == "__main__":
    main()
