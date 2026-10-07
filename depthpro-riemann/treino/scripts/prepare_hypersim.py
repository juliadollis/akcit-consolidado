#!/usr/bin/env python3
"""
scripts/prepare_hypersim.py
===========================
Prepara o Hypersim no layout esperado pelo dataloader (rgb/ + depth/).

O Hypersim distribui:
  - RGB em .../images/scene_camXX_final_preview/frame.NNNN.color.jpg
  - Profundidade (distância ao plano da câmera) em
    .../images/scene_camXX_geometry_hdf5/frame.NNNN.depth_meters.hdf5

Este script varre um diretório do release oficial, lê os .hdf5 de profundidade,
converte a distância radial para profundidade planar (z), e salva pares
rgb/<key>.png + depth/<key>.npy no diretório de saída.

Uso:
    python scripts/prepare_hypersim.py \\
        --hypersim-root /data/hypersim_raw \\
        --out-root /data/hypersim/train \\
        --split-list train_scenes.txt   # opcional: subconjunto de cenas

Requer: h5py, opencv-python. Veja o README para onde obter o Hypersim.
"""

import argparse
from pathlib import Path
import numpy as np

try:
    import h5py
    import cv2
except ImportError:
    h5py = None
    cv2 = None


def radial_to_planar(depth_radial: np.ndarray, focal_px: float) -> np.ndarray:
    """
    Hypersim fornece distância radial ao centro da câmera. Converte para profundidade
    planar (z) que é o que o depth map convencional representa.

        z = depth_radial / sqrt(1 + (x/f)² + (y/f)²)
    """
    H, W = depth_radial.shape
    cx, cy = W / 2.0, H / 2.0
    xs = np.arange(W) - cx
    ys = np.arange(H) - cy
    gx, gy = np.meshgrid(xs, ys)
    denom = np.sqrt(1.0 + (gx / focal_px) ** 2 + (gy / focal_px) ** 2)
    return depth_radial / denom


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hypersim-root", required=True)
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--split-list", default=None,
                    help="txt com nomes de cenas (uma por linha) a incluir")
    ap.add_argument("--focal-px", type=float, default=886.81,
                    help="focal em px do Hypersim (padrão do dataset a 1024px de largura)")
    ap.add_argument("--max", type=int, default=None)
    args = ap.parse_args()

    if h5py is None or cv2 is None:
        raise ImportError("Instale h5py e opencv-python: pip install h5py opencv-python")

    root = Path(args.hypersim_root)
    out = Path(args.out_root)
    (out / "rgb").mkdir(parents=True, exist_ok=True)
    (out / "depth").mkdir(parents=True, exist_ok=True)

    scenes = None
    if args.split_list:
        scenes = set(Path(args.split_list).read_text().split())

    color_files = sorted(root.rglob("*.color.jpg"))
    n = 0
    for color_path in color_files:
        # Nome da cena a partir do caminho
        scene_name = color_path.parts[len(root.parts)] if len(color_path.parts) > len(root.parts) else "scene"
        if scenes is not None and scene_name not in scenes:
            continue

        # Caminho do hdf5 de profundidade correspondente
        depth_path = Path(str(color_path)
                          .replace("_final_preview", "_geometry_hdf5")
                          .replace(".color.jpg", ".depth_meters.hdf5"))
        if not depth_path.exists():
            continue

        try:
            with h5py.File(depth_path, "r") as f:
                depth_radial = np.array(f["dataset"]).astype(np.float32)
            depth = radial_to_planar(depth_radial, args.focal_px)

            rgb = cv2.imread(str(color_path))
            key = f"{scene_name}__{color_path.stem.replace('.color','')}"
            cv2.imwrite(str(out / "rgb" / f"{key}.png"), rgb)
            np.save(out / "depth" / f"{key}.npy", depth.astype(np.float32))
            n += 1
            if n % 200 == 0:
                print(f"  {n} pares processados...")
            if args.max and n >= args.max:
                break
        except Exception as e:
            print(f"  [skip] {color_path.name}: {e}")

    print(f"Concluído: {n} pares em {out}")


if __name__ == "__main__":
    main()
