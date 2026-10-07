"""
riemann/dataset.py
==================
Dataset de profundidade de ALTA QUALIDADE para o fine-tuning.

Primário: Hypersim (sintético fotorrealista, GT de profundidade perfeito e HR — ideal
para curvatura de 2ª ordem, que exige GT limpo).
Também suporta um layout genérico de pastas (rgb/ + depth/) para ETH3D/DIML/custom.

Hypersim distribui profundidade como distância ao plano da câmera (.hdf5). Este loader
assume uma versão pré-extraída em pastas rgb/ e depth/ (PNG 16-bit ou .npy), que é o
formato mais simples de consumir. O script scripts/prepare_hypersim.py descreve como
gerar esse layout a partir do release oficial.

Layout esperado:
    root/
      rgb/    scene_xxx.png
      depth/  scene_xxx.png   (16-bit, ou .npy)
"""

from __future__ import annotations
from pathlib import Path
from typing import Optional, Tuple, List
import numpy as np
import torch
from torch.utils.data import Dataset

try:
    import cv2
    _HAS_CV2 = True
except ImportError:
    _HAS_CV2 = False
    from PIL import Image

IMG_EXTS = {".png", ".jpg", ".jpeg"}
DEPTH_EXTS = {".png", ".npy"}


def _read_rgb(path: Path) -> np.ndarray:
    if _HAS_CV2:
        img = cv2.imread(str(path))
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return np.array(Image.open(path).convert("RGB"))


def _read_depth(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        return np.load(path).astype(np.float32)
    if _HAS_CV2:
        d = cv2.imread(str(path), cv2.IMREAD_UNCHANGED).astype(np.float32)
    else:
        d = np.array(Image.open(path)).astype(np.float32)
    return d


class HighQualityDepthDataset(Dataset):
    """
    Dataset genérico de profundidade HR.

    Args:
        root: pasta contendo rgb/ e depth/.
        size: (H, W) para redimensionar (múltiplo de 16 recomendado p/ ViT).
        rgb_mean/rgb_std: normalização do RGB (padrão ImageNet).
        max_samples: limitar (debug).

    Nota: entregamos profundidade métrica CRUA (sem min-max). O alinhamento pred<->GT é
    afim, feito na loss/métrica. O arg `normalize_depth` é aceito mas IGNORADO (mantido só
    para compatibilidade com chamadas antigas).
    """
    def __init__(self, root: str, size: Tuple[int, int] = (512, 512),
                 normalize_depth: bool = False,  # ignorado (compat.)
                 rgb_mean=(0.485, 0.456, 0.406), rgb_std=(0.229, 0.224, 0.225),
                 max_samples: Optional[int] = None,
                 focal_px: Optional[float] = None):
        self.root = Path(root)
        self.size = size
        self.rgb_mean = np.array(rgb_mean, np.float32)
        self.rgb_std = np.array(rgb_std, np.float32)
        # Focal em PIXELS na resolucao ORIGINAL das imagens em disco. Necessaria para
        # a curvatura em unidades metricas. Se nao vier por argumento, procuramos
        # intrinsics.json na raiz do dataset.
        self.focal_px = focal_px
        if self.focal_px is None:
            _fj = self.root / "intrinsics.json"
            if _fj.exists():
                import json as _json
                with open(_fj) as _f:
                    _d = _json.load(_f)
                self.focal_px = float(_d.get("fx", _d.get("focal_px", 0)) or 0) or None

        rgb_dir = self.root / "rgb"
        depth_dir = self.root / "depth"
        if not rgb_dir.exists() or not depth_dir.exists():
            raise FileNotFoundError(
                f"Esperado {rgb_dir} e {depth_dir}. Veja scripts/prepare_hypersim.py.")

        rgb_files = {p.stem: p for p in rgb_dir.iterdir() if p.suffix.lower() in IMG_EXTS}
        depth_files = {p.stem: p for p in depth_dir.iterdir() if p.suffix.lower() in DEPTH_EXTS}
        self.keys: List[str] = sorted(set(rgb_files) & set(depth_files))
        if max_samples:
            self.keys = self.keys[:max_samples]
        self.rgb_files = rgb_files
        self.depth_files = depth_files

        if not self.keys:
            raise RuntimeError(f"Nenhum par (rgb, depth) encontrado em {self.root}")

    def __len__(self):
        return len(self.keys)

    def _resize(self, arr: np.ndarray, is_depth: bool):
        H, W = self.size
        if _HAS_CV2:
            interp = cv2.INTER_NEAREST if is_depth else cv2.INTER_AREA
            return cv2.resize(arr, (W, H), interpolation=interp)
        from PIL import Image
        mode = "F" if is_depth else "RGB"
        im = Image.fromarray(arr) if not is_depth else Image.fromarray(arr, mode="F")
        return np.array(im.resize((W, H)))

    def __getitem__(self, idx):
        k = self.keys[idx]
        rgb = _read_rgb(self.rgb_files[k]).astype(np.float32)
        depth = _read_depth(self.depth_files[k])
        if depth.ndim == 3:
            depth = depth[..., 0]
        H0, W0 = depth.shape[:2]     # resolucao ORIGINAL, antes de redimensionar

        rgb = self._resize(rgb, is_depth=False)
        depth = self._resize(depth, is_depth=True)

        # Máscara de validade (depth > 0 e finito)
        mask = (np.isfinite(depth) & (depth > 0)).astype(np.float32)

        # NÃO normalizamos o GT por min-max (isso reescalava o eixo z e, como a curvatura
        # não é invariante a reescala, deformava a gauss_loss e a fazia explodir vs berHu).
        # Entregamos profundidade métrica CRUA; o alinhamento pred<->GT é afim, na loss.
        depth = np.where(mask > 0, depth, 0.0).astype(np.float32)

        # Normalizar RGB
        rgb = rgb / 255.0
        rgb = (rgb - self.rgb_mean) / self.rgb_std

        rgb_t = torch.from_numpy(rgb.transpose(2, 0, 1)).float()
        depth_t = torch.from_numpy(depth).unsqueeze(0).float()
        mask_t = torch.from_numpy(mask).unsqueeze(0).float()
        # A focal acompanha o redimensionamento, e as escalas horizontal e vertical
        # podem DIFERIR quando a imagem original nao e quadrada. O Spring e 1920x1080
        # e o pipeline trabalha em 512x512, entao fx e fy divergem por construcao.
        # Esquecer isto produz curvatura errada por um fator que nao aparece em teste
        # com resolucao unica.
        Hn, Wn = depth_t.shape[-2:]
        if self.focal_px:
            fx = float(self.focal_px) * (Wn / float(W0))
            fy = float(self.focal_px) * (Hn / float(H0))
        else:
            fx = fy = -1.0           # desconhecida: a loss cai no modo antigo
        return {"rgb": rgb_t, "depth": depth_t, "mask": mask_t, "key": k,
                "fx": torch.tensor(fx, dtype=torch.float32),
                "fy": torch.tensor(fy, dtype=torch.float32)}
