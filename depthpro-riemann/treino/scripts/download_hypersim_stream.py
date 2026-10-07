#!/usr/bin/env python3
"""
scripts/download_hypersim_stream.py
===================================
Baixa o Hypersim de forma SELETIVA e em STREAMING, sem nunca armazenar o dataset inteiro.

Problema que resolve: o Hypersim tem ~150 GB. Se não há disco para tudo, este script
baixa UMA cena por vez, extrai só o RGB + profundidade, converte para o layout
rgb/ + depth/, e APAGA o zip antes de ir para a próxima cena. O pico de disco é
"uma cena (~1-2 GB) + o batch convertido (pequeno)".

Duas formas de escolher o que baixar (o usuário controla):
  --n-images N     : baixa cenas até acumular ~N imagens (conveniência)
  --scenes LISTA   : baixa exatamente as cenas nomeadas (controle/reprodutibilidade)
                     ex.: --scenes ai_001_001 ai_001_002 ai_002_001
  --scenes-file F  : lê os nomes de cena de um arquivo (uma por linha)

Fonte oficial: cada cena é um zip em
  https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/<cena>.zip
Dentro do zip, interessam:
  - RGB:   .../images/scene_cam_XX_final_preview/frame.NNNN.color.jpg
  - Depth: .../images/scene_cam_XX_geometry_hdf5/frame.NNNN.depth_meters.hdf5  (distância radial)

O depth radial é convertido para profundidade planar (z), igual ao prepare_hypersim.py.

⚠️ IMPORTANTE (rede): a URL da Apple (docs-assets.developer.apple.com) precisa de acesso à
internet aberta. ESTE SCRIPT NÃO FOI TESTADO CONTRA A APPLE no ambiente de desenvolvimento
(o domínio não era acessível lá). A lógica de extração/conversão foi validada com zips
sintéticos. No primeiro uso real, rode com --scenes ai_001_001 --limit-scenes 1 e confira
o resultado antes de baixar em massa.
"""

import argparse
import os
import shutil
import sys
import tempfile
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

import numpy as np

BASE_URL = ("https://docs-assets.developer.apple.com/ml-research/datasets/"
            "hypersim/v1/scenes/{scene}.zip")

# Focal padrão do Hypersim (px) para conversão radial->planar.
DEFAULT_FOCAL_PX = 886.81


# ---------------------------------------------------------------------------
# Geração de nomes de cena (ai_VVV_NNN)
# ---------------------------------------------------------------------------
def enumerate_scene_names(max_scenes: int = 999):
    """
    Gera nomes de cena plausíveis ai_VVV_NNN em ordem. Nem todo par existe (há lacunas
    no dataset), então o downloader deve tolerar 404 e seguir para a próxima.
    Cobre volumes 001..055 (faixa do release público) com cenas 001..020.
    """
    names = []
    for vol in range(1, 56):
        for scene in range(1, 21):
            names.append(f"ai_{vol:03d}_{scene:03d}")
    return names[:max_scenes] if max_scenes else names


# ---------------------------------------------------------------------------
# Conversão radial -> planar (idêntica ao prepare_hypersim.py)
# ---------------------------------------------------------------------------
def radial_to_planar(depth_radial: np.ndarray, focal_px: float) -> np.ndarray:
    H, W = depth_radial.shape
    cx, cy = W / 2.0, H / 2.0
    xs = np.arange(W) - cx
    ys = np.arange(H) - cy
    gx, gy = np.meshgrid(xs, ys)
    denom = np.sqrt(1.0 + (gx / focal_px) ** 2 + (gy / focal_px) ** 2)
    return depth_radial / denom


# ---------------------------------------------------------------------------
# Download de uma cena (urllib, sem depender de curl/wget), tolerante a 404
# ---------------------------------------------------------------------------
def download_scene_zip(scene: str, dst_zip: Path) -> bool:
    url = BASE_URL.format(scene=scene)
    try:
        urllib.request.urlretrieve(url, dst_zip)
    except urllib.error.URLError as e:
        # 404 = cena inexistente (normal, o dataset tem lacunas); o resto é erro
        # real de rede e precisa aparecer no log, não ser mascarado como lacuna.
        if not (isinstance(e, urllib.error.HTTPError) and e.code == 404):
            print(f"[erro de rede: {e}]", end=" ")
        if dst_zip.exists():
            dst_zip.unlink()
        return False
    if not dst_zip.exists() or dst_zip.stat().st_size < 1024:
        if dst_zip.exists():
            dst_zip.unlink()
        return False
    return True


# ---------------------------------------------------------------------------
# Extrair + converter os pares de uma cena a partir do zip
# ---------------------------------------------------------------------------
def process_scene_zip(zip_path: Path, scene: str, out_rgb: Path, out_depth: Path,
                      focal_px: float, remaining: int) -> int:
    """
    Abre o zip da cena, encontra pares (color.jpg, depth_meters.hdf5) e salva convertidos.
    Retorna quantas imagens foram efetivamente salvas (limitado por `remaining`).
    """
    try:
        import h5py
        import cv2
    except ImportError:
        print("  [ERRO] h5py/opencv ausentes. pip install h5py opencv-python-headless")
        sys.exit(1)

    saved = 0
    with zipfile.ZipFile(zip_path, "r") as zf:
        names = zf.namelist()
        # Mapear frames de profundidade e cor por (cam, frame)
        color = {}
        depth = {}
        for n in names:
            if n.endswith(".color.jpg") and "final_preview" in n:
                key = _frame_key(n)
                if key:
                    color[key] = n
            elif n.endswith(".depth_meters.hdf5") and "geometry_hdf5" in n:
                key = _frame_key(n)
                if key:
                    depth[key] = n

        keys = sorted(set(color) & set(depth))
        for key in keys:
            if saved >= remaining:
                break
            try:
                # Ler color
                with zf.open(color[key]) as f:
                    img_bytes = np.frombuffer(f.read(), np.uint8)
                    rgb = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)
                # Ler depth (hdf5) — precisa extrair para arquivo temporário
                with tempfile.NamedTemporaryFile(suffix=".hdf5", delete=False) as tmp:
                    tmp.write(zf.read(depth[key]))
                    tmp_path = tmp.name
                with h5py.File(tmp_path, "r") as hf:
                    depth_radial = np.array(hf["dataset"]).astype(np.float32)
                os.unlink(tmp_path)

                depth_planar = radial_to_planar(depth_radial, focal_px)

                cam_frame = key  # ex.: cam_00__0000
                out_key = f"{scene}__{cam_frame}"
                cv2.imwrite(str(out_rgb / f"{out_key}.png"), rgb)
                np.save(out_depth / f"{out_key}.npy", depth_planar.astype(np.float32))
                saved += 1
            except Exception as e:
                print(f"    [skip] {scene} {key}: {e}")
    return saved


def _frame_key(path: str):
    """
    Extrai uma chave (cam + frame) do caminho interno do zip.
    Ex.: .../scene_cam_00_final_preview/frame.0000.color.jpg -> 'cam_00__0000'
    """
    import re
    m_cam = re.search(r"scene_(cam_\d+)_", path)
    m_frame = re.search(r"frame\.(\d+)\.", path)
    if m_cam and m_frame:
        return f"{m_cam.group(1)}__{m_frame.group(1)}"
    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", required=True,
                    help="destino no layout rgb/ + depth/ (ex.: /data/hypersim/ablation/train)")
    # Modo de seleção (um dos três)
    ap.add_argument("--n-images", type=int, default=None,
                    help="baixar cenas até acumular ~N imagens")
    ap.add_argument("--scenes", nargs="*", default=None,
                    help="lista explícita de cenas (ex.: ai_001_001 ai_001_002)")
    ap.add_argument("--scenes-file", default=None,
                    help="arquivo com nomes de cena (uma por linha)")
    # Controles
    ap.add_argument("--focal-px", type=float, default=DEFAULT_FOCAL_PX)
    ap.add_argument("--limit-scenes", type=int, default=None,
                    help="teto de cenas a TENTAR baixar (segurança)")
    ap.add_argument("--start-index", type=int, default=0,
                    help="pular as primeiras N cenas da enumeração (para dividir train/val)")
    ap.add_argument("--keep-zip", action="store_true",
                    help="NÃO apagar o zip após processar (debug; gasta disco)")
    args = ap.parse_args()

    if not args.n_images and not args.scenes and not args.scenes_file:
        ap.error("escolha um modo: --n-images N, ou --scenes ..., ou --scenes-file F")

    out = Path(args.out_root)
    out_rgb = out / "rgb"
    out_depth = out / "depth"
    out_rgb.mkdir(parents=True, exist_ok=True)
    out_depth.mkdir(parents=True, exist_ok=True)

    # Montar a lista de cenas a tentar
    if args.scenes:
        scene_list = args.scenes
    elif args.scenes_file:
        scene_list = Path(args.scenes_file).read_text().split()
    else:
        scene_list = enumerate_scene_names()
        if args.start_index:
            scene_list = scene_list[args.start_index:]
    if args.limit_scenes:
        scene_list = scene_list[:args.limit_scenes]

    target = args.n_images if args.n_images else float("inf")
    total_saved = 0
    scenes_ok = 0
    scenes_missing = 0

    tmp_dir = Path(tempfile.mkdtemp(prefix="hypersim_dl_"))
    print(f"[stream] destino: {out}")
    print(f"[stream] modo: "
          f"{'n-images='+str(args.n_images) if args.n_images else ('scenes='+str(len(scene_list)))}")

    try:
        for scene in scene_list:
            if total_saved >= target:
                break
            zip_path = tmp_dir / f"{scene}.zip"
            print(f"[stream] baixando {scene} ...", end=" ", flush=True)
            ok = download_scene_zip(scene, zip_path)
            if not ok:
                print("(inexistente, pulando)")
                scenes_missing += 1
                continue

            remaining = int(target - total_saved) if args.n_images else 10**9
            saved = process_scene_zip(zip_path, scene, out_rgb, out_depth,
                                      args.focal_px, remaining)
            total_saved += saved
            scenes_ok += 1
            print(f"OK (+{saved} imgs, total={total_saved})")

            if not args.keep_zip and zip_path.exists():
                zip_path.unlink()  # libera disco imediatamente
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    print(f"\n[stream] concluído: {total_saved} imagens de {scenes_ok} cenas "
          f"({scenes_missing} inexistentes puladas)")
    print(f"  rgb/   : {out_rgb}")
    print(f"  depth/ : {out_depth}")
    if args.n_images and total_saved < args.n_images:
        print(f"  [aviso] alvo de {args.n_images} não atingido — aumente --limit-scenes "
              f"ou a faixa de cenas.")


if __name__ == "__main__":
    main()
