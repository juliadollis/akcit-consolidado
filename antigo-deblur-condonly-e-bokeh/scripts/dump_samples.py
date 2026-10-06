"""Salva 1 amostra de cada coluna de imagem dos datasets HF como PNG.

Objetivo: olhar as imagens com o olho humano e decidir qual coluna é a
ENTRADA borrada e qual é o ALVO nítido (all-in-focus), já que os nomes
das colunas (`image_pre_deblur`, `image_blur`, `image_focus`) não deixam
isso 100% óbvio.

Para cada dataset, puxa N amostras via streaming (NÃO baixa o dataset
inteiro) e salva cada coluna de imagem como:

    <OUT_DIR>/<dataset>/sample<idx>__<coluna>.png

Também imprime, por amostra, o tamanho (WxH) e o modo de cada imagem e
uma estimativa de "nitidez" (variância do Laplaciano aproximada): quanto
MAIOR, mais nítida/contrastada a imagem tende a ser. Isso ajuda a confirmar
visualmente qual é a borrada (variância baixa) e qual é a nítida (alta).

Uso (dentro do container, com HF_TOKEN no ambiente):
    python3 scripts/dump_samples.py
"""

from __future__ import annotations

import os

import numpy as np
from datasets import load_dataset

# (repo HF, split preferido, nº de amostras a salvar)
DATASETS = [
    ("akcit-pixel/DDPD", "train", 2),
    ("akcit-pixel/RealBokeh", "validation", 2),
]

# Colunas que são imagens (PIL).
IMAGE_COLUMNS = ["image_pre_deblur", "image_blur", "image_focus"]

# Saída: dentro do projeto (host: /raid/user_juliadollis/projects/.../sample_dump).
OUT_DIR = os.path.join(
    "/workspace/projects/genrefocus_deblurnet",
    "sample_dump",
)


def _token() -> str:
    tok = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
    if not tok:
        raise SystemExit("ERRO: HF_TOKEN não definido no ambiente.")
    return tok


def _sharpness(img) -> float:
    """Estimativa barata de nitidez: variância de um Laplaciano 3x3 no canal de luma.

    Não depende de OpenCV — usa só numpy. Valor MAIOR = mais detalhe/borda
    (mais nítida); valor MENOR = mais borrada.
    """
    arr = np.asarray(img.convert("L"), dtype=np.float64)
    if arr.ndim != 2 or arr.size == 0:
        return float("nan")
    # Kernel Laplaciano 3x3 via diferenças finitas (sem scipy).
    lap = (
        -4.0 * arr
        + np.roll(arr, 1, axis=0)
        + np.roll(arr, -1, axis=0)
        + np.roll(arr, 1, axis=1)
        + np.roll(arr, -1, axis=1)
    )
    # Descarta bordas (wrap do np.roll) para não poluir a variância.
    inner = lap[1:-1, 1:-1]
    return float(inner.var())


def dump(name: str, split: str, n_samples: int, token: str) -> None:
    print("=" * 70)
    print(f"DATASET: {name}  (split='{split}', {n_samples} amostra(s))")
    print("=" * 70)

    safe_name = name.replace("/", "__")
    out_subdir = os.path.join(OUT_DIR, safe_name)
    os.makedirs(out_subdir, exist_ok=True)

    try:
        stream = load_dataset(name, split=split, streaming=True, token=token)
    except Exception as exc:  # noqa: BLE001
        print(f"[aviso] não consegui abrir o split '{split}': {exc}")
        return

    it = iter(stream)
    for idx in range(n_samples):
        try:
            record = next(it)
        except StopIteration:
            print(f"[aviso] split acabou após {idx} amostra(s).")
            break

        fid = record.get("file_name_base", f"idx{idx}")
        print(f"\n  amostra {idx}  (file_name_base={fid!r}):")
        for col in IMAGE_COLUMNS:
            if col not in record:
                print(f"    - {col}: [coluna ausente]")
                continue
            img = record[col]
            size = getattr(img, "size", None)
            mode = getattr(img, "mode", None)
            if size is None:
                print(f"    - {col}: [não é imagem PIL: {type(img)}]")
                continue
            sharp = _sharpness(img)
            out_path = os.path.join(out_subdir, f"sample{idx}__{col}.png")
            try:
                img.save(out_path)
                saved = out_path
            except Exception as exc:  # noqa: BLE001
                saved = f"[falha ao salvar: {exc}]"
            print(
                f"    - {col}: size={size} mode={mode} "
                f"sharpness(lap_var)={sharp:,.1f}  ->  {saved}"
            )


def main() -> None:
    token = _token()
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"Salvando PNGs em: {OUT_DIR}")
    print("(host: /raid/user_juliadollis/projects/genrefocus_deblurnet/sample_dump)\n")
    for name, split, n in DATASETS:
        try:
            dump(name, split, n, token)
        except Exception as exc:  # noqa: BLE001
            print(f"[ERRO] falha ao processar {name}: {exc}\n")
    print("\nLeitura sugerida:")
    print("  - MENOR sharpness(lap_var) => imagem mais BORRADA (provável ENTRADA).")
    print("  - MAIOR sharpness(lap_var) => imagem mais NÍTIDA  (provável ALVO AIF).")
    print("Pronto.")


if __name__ == "__main__":
    main()
