"""Teste de diagnóstico: inferência da DeblurNet controlando o `main_adapter`.

Igual ao Inference_deblurNet.py oficial, mas gera DUAS saídas do MESMO peso:
  - <prefix>_mainlora.png  → main_adapter="deblurring" (LoRA no main+cond = como
                             o nosso 21K foi TREINADO)
  - <prefix>_official.png  → main_adapter=None (LoRA só na cond = inferência
                             oficial, que deu a saída lavada)

Se a "mainlora" deblurar e a "official" vier lavada, o bug do descasamento de
branch está 100% confirmado.
"""

import argparse

import torch
from diffusers import FluxPipeline
from PIL import Image

from Genfocus.pipeline.flux import Condition, generate, seed_everything

MODEL_ID = "black-forest-labs/FLUX.1-dev"
PROMPT = "a sharp photo with everything in focus"


def resize_and_pad_image(img: Image.Image, target_long_side: int) -> Image.Image:
    w, h = img.size
    if target_long_side and target_long_side > 0:
        tm = int(target_long_side)
        if w >= h:
            new_w, new_h = tm, int(h * (tm / w))
        else:
            new_h, new_w = tm, int(w * (tm / h))
        img = img.resize((new_w, new_h), Image.LANCZOS)
        fw, fh = max((new_w // 16) * 16, 16), max((new_h // 16) * 16, 16)
        left, top = (new_w - fw) // 2, (new_h - fh) // 2
        return img.crop((left, top, left + fw, top + fh))
    fw, fh = ((w + 15) // 16) * 16, ((h + 15) // 16) * 16
    return img if (fw == w and fh == h) else img.resize((fw, fh), Image.LANCZOS)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", "-i", required=True)
    p.add_argument("--out-prefix", "-o", default="out")
    p.add_argument("--weight-dir", default="/tmp")
    p.add_argument("--weight-name", default="deblur_export.safetensors")
    p.add_argument("--steps", type=int, default=28)
    p.add_argument("--long_side", type=int, default=0)
    args = p.parse_args()

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    raw = Image.open(args.input).convert("RGB")
    img = resize_and_pad_image(raw, args.long_side)
    w, h = img.size
    print(f"[infer] entrada {w}x{h}  peso={args.weight_dir}/{args.weight_name}")

    pipe = FluxPipeline.from_pretrained(MODEL_ID, torch_dtype=dtype)
    if torch.cuda.is_available():
        pipe.to("cuda")
    pipe.load_lora_weights(args.weight_dir, weight_name=args.weight_name, adapter_name="deblurring")
    pipe.set_adapters(["deblurring"])

    no_tiled = min(w, h) < 512

    # Roda os dois modos com o MESMO pipe/peso (carrega o FLUX 1x só).
    for tag, main_adapter in [("mainlora", "deblurring"), ("official", None)]:
        seed_everything(42)
        cond0 = Condition(img, "deblurring", [0, 0], 1.0)
        out = generate(
            pipe,
            height=h,
            width=w,
            prompt=PROMPT,
            num_inference_steps=args.steps,
            conditions=[cond0],
            main_adapter=main_adapter,
            NO_TILED_DENOISE=no_tiled,
        ).images[0]
        path = f"{args.out_prefix}_{tag}.png"
        out.save(path)
        print(f"[infer] {tag}: main_adapter={main_adapter} -> {path}")


if __name__ == "__main__":
    main()
