"""Gera exemplos antes→depois da DeblurNet: [borrada | nosso deblur | ground-truth].

Puxa N amostras de um dataset HF (que tem image_blur e image_focus), roda o
deblur (com main_adapter="deblurring", a inferência CORRETA pro nosso peso) e
salva um PNG lado-a-lado por amostra + um grid com todas.
"""

import argparse
import os

import torch
from datasets import load_dataset
from PIL import Image, ImageDraw

from Genfocus.pipeline.flux import Condition, generate, seed_everything

MODEL_ID = "black-forest-labs/FLUX.1-dev"
PROMPT = "a sharp photo with everything in focus"


def resize_and_pad(img: Image.Image, long_side: int) -> Image.Image:
    img = img.convert("RGB")
    if not long_side:
        w, h = img.size
        fw, fh = ((w + 15) // 16) * 16, ((h + 15) // 16) * 16
        return img if (fw == w and fh == h) else img.resize((fw, fh), Image.LANCZOS)
    w, h = img.size
    if w >= h:
        nw, nh = long_side, int(h * (long_side / w))
    else:
        nh, nw = long_side, int(w * (long_side / h))
    img = img.resize((nw, nh), Image.LANCZOS)
    fw, fh = max((nw // 16) * 16, 16), max((nh // 16) * 16, 16)
    left, top = (nw - fw) // 2, (nh - fh) // 2
    return img.crop((left, top, left + fw, top + fh))


def _sq(img: Image.Image, size: int) -> Image.Image:
    return img.convert("RGB").resize((size, size), Image.LANCZOS)


def _label(img: Image.Image, text: str) -> Image.Image:
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, img.width, 22], fill=(0, 0, 0))
    d.text((6, 5), text, fill=(255, 255, 255))
    return img


def _hconcat(imgs, gap=8, bg=(255, 255, 255)) -> Image.Image:
    h = max(i.height for i in imgs)
    w = sum(i.width for i in imgs) + gap * (len(imgs) - 1)
    canvas = Image.new("RGB", (w, h), bg)
    x = 0
    for im in imgs:
        canvas.paste(im, (x, 0))
        x += im.width + gap
    return canvas


def _vconcat(imgs, gap=8, bg=(255, 255, 255)) -> Image.Image:
    w = max(i.width for i in imgs)
    h = sum(i.height for i in imgs) + gap * (len(imgs) - 1)
    canvas = Image.new("RGB", (w, h), bg)
    y = 0
    for im in imgs:
        canvas.paste(im, (0, y))
        y += im.height + gap
    return canvas


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="akcit-pixel/DDPD")
    p.add_argument("--split", default="validation")
    p.add_argument("--n", type=int, default=5)
    p.add_argument("--weight-dir", default="/tmp")
    p.add_argument("--weight-name", default="deblur_export.safetensors")
    p.add_argument("--out-dir", default="outputs/examples")
    p.add_argument("--steps", type=int, default=28)
    p.add_argument("--long_side", type=int, default=512)
    p.add_argument("--disp", type=int, default=512, help="tamanho de cada painel no PNG")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    from diffusers import FluxPipeline

    pipe = FluxPipeline.from_pretrained(MODEL_ID, torch_dtype=dtype)
    if torch.cuda.is_available():
        pipe.to("cuda")
    pipe.load_lora_weights(args.weight_dir, weight_name=args.weight_name, adapter_name="deblurring")
    pipe.set_adapters(["deblurring"])

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    ds = load_dataset(args.dataset, split=f"{args.split}[:{args.n}]", token=token)
    print(f"[examples] {len(ds)} amostras de {args.dataset}:{args.split}")

    rows = []
    for i, rec in enumerate(ds):
        blur = rec["image_blur"].convert("RGB")
        gt = rec["image_focus"].convert("RGB")
        proc = resize_and_pad(blur, args.long_side)
        w, h = proc.size
        seed_everything(42)
        cond = Condition(proc, "deblurring", [0, 0], 1.0)
        out = generate(
            pipe,
            height=h,
            width=w,
            prompt=PROMPT,
            num_inference_steps=args.steps,
            conditions=[cond],
            main_adapter=None,   # cond-only (variante paper): LoRA só na condição = inferência OFICIAL
            NO_TILED_DENOISE=min(w, h) < 512,
        ).images[0]

        d = args.disp
        row = _hconcat([
            _label(_sq(blur, d), "ENTRADA (borrada)"),
            _label(_sq(out, d), "NOSSO DEBLUR"),
            _label(_sq(gt, d), "GROUND-TRUTH"),
        ])
        path = os.path.join(args.out_dir, f"example_{i:02d}.png")
        row.save(path)
        rows.append(row)
        print(f"[examples] salvo {path}")

    grid = _vconcat(rows)
    grid_path = os.path.join(args.out_dir, "grid_all.png")
    grid.save(grid_path)
    print(f"[examples] grid completo -> {grid_path}")


if __name__ == "__main__":
    main()
