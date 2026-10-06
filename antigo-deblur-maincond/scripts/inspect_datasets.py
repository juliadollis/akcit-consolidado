"""Inspeciona os datasets HF da DeblurNet (akcit-pixel/*) sem treinar nada.

Imprime, para cada dataset/split:
  - nome dos splits e nº de exemplos
  - nome das colunas e o tipo (feature) de cada uma
  - de UMA amostra: para colunas de imagem, o tamanho (WxH) e modo (RGB/L/...);
    para o resto, o valor (truncado).

Objetivo: descobrir qual coluna é a imagem borrada (entrada) e qual é a nítida
(alvo do deblur), já que os nomes diferem entre DDPD e RealBokeh.

Uso (dentro do container, com HF_TOKEN no ambiente):
    python3 scripts/inspect_datasets.py
"""

from __future__ import annotations

import os

from datasets import load_dataset, load_dataset_builder

# (repo HF, split preferido para puxar 1 amostra)
DATASETS = [
    ("akcit-pixel/DDPD", "train"),
    ("akcit-pixel/RealBokeh", "validation"),
]


def _token() -> str:
    tok = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
    if not tok:
        raise SystemExit("ERRO: HF_TOKEN não definido no ambiente.")
    return tok


def _describe_value(value: object) -> str:
    """Descreve uma célula: se for imagem PIL, dá tamanho/modo; senão, repr curto."""
    # PIL.Image tem .size e .mode
    size = getattr(value, "size", None)
    mode = getattr(value, "mode", None)
    if size is not None and mode is not None:
        return f"<Image size={size} mode={mode}>"
    text = repr(value)
    return text if len(text) <= 120 else text[:117] + "..."


def inspect(name: str, sample_split: str, token: str) -> None:
    print("=" * 70)
    print(f"DATASET: {name}")
    print("=" * 70)

    # 1) Metadados (rápido, sem baixar imagens).
    try:
        builder = load_dataset_builder(name, token=token)
        info = builder.info
        splits = info.splits or {}
        print("Splits:")
        for split_name, split_info in splits.items():
            n = getattr(split_info, "num_examples", "?")
            print(f"  - {split_name}: {n} exemplos")
        print("\nColunas (feature types):")
        for col, feat in info.features.items():
            print(f"  - {col}: {feat}")
    except Exception as exc:  # noqa: BLE001
        print(f"[aviso] não consegui ler metadados via builder: {exc}")
        splits = {}

    # 2) Uma amostra real (streaming = não baixa o dataset todo).
    split_to_use = sample_split if (not splits or sample_split in splits) else next(iter(splits))
    print(f"\nAmostra (1 exemplo do split '{split_to_use}', via streaming):")
    try:
        stream = load_dataset(name, split=split_to_use, streaming=True, token=token)
        first = next(iter(stream))
        for key, value in first.items():
            print(f"  - {key}: {_describe_value(value)}")
    except Exception as exc:  # noqa: BLE001
        print(f"[aviso] não consegui puxar amostra: {exc}")
    print()


def main() -> None:
    token = _token()
    for name, sample_split in DATASETS:
        try:
            inspect(name, sample_split, token)
        except Exception as exc:  # noqa: BLE001
            print(f"[ERRO] falha ao inspecionar {name}: {exc}\n")
    print("Pronto.")


if __name__ == "__main__":
    main()
