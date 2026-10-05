"""
Helpers matemáticos.

Stage 1 (DeblurNet) NÃO usa este módulo. Mantido aqui porque `models.py`
importa `compute_defocus_map` / `normalize_defocus_condition` no BokehNet,
e o pacote precisa importar limpo.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Tuple

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class CameraMetadata:
    """Metadados de câmera para estimar K via EXIF (paper Eq. 3)."""

    focal_length_mm: float
    f_number: float
    pixel_ratio: float


def estimate_focus_plane_from_mask(
    depth_map: torch.Tensor, focus_mask: torch.Tensor
) -> torch.Tensor:
    """D_focus = median(D[M]) por item do batch (paper Eq. 4)."""
    if depth_map.ndim != 3:
        raise ValueError("depth_map must have shape (B, H, W).")
    if focus_mask.ndim != 3:
        raise ValueError("focus_mask must have shape (B, H, W).")

    values = []
    for depth, mask in zip(depth_map, focus_mask.bool(), strict=True):
        selected = depth[mask]
        if selected.numel() == 0:
            raise ValueError("Focus mask is empty.")
        values.append(selected.median())
    return torch.stack(values, dim=0)


def select_focus_plane_from_point(
    depth_map: torch.Tensor, points_xy: Sequence[Tuple[int, int]]
) -> torch.Tensor:
    """Seleciona D_focus a partir de pontos clicados."""
    if depth_map.ndim != 3:
        raise ValueError("depth_map must have shape (B, H, W).")
    if len(points_xy) != depth_map.shape[0]:
        raise ValueError("Esperado um ponto por item do batch.")

    values = []
    for depth, (x, y) in zip(depth_map, points_xy, strict=True):
        h, w = depth.shape
        x = max(0, min(int(x), w - 1))
        y = max(0, min(int(y), h - 1))
        values.append(depth[y, x])
    return torch.stack(values, dim=0)


def compute_defocus_map(
    depth_map: torch.Tensor, focus_plane: torch.Tensor, bokeh_level: torch.Tensor
) -> torch.Tensor:
    """D_def = K · |D − D_focus| (paper Eq. 2)."""
    if depth_map.ndim != 3:
        raise ValueError("depth_map must have shape (B, H, W).")
    focus_plane = focus_plane.view(-1, 1, 1)
    bokeh_level = bokeh_level.view(-1, 1, 1)
    return bokeh_level * (depth_map - focus_plane).abs()


def approximate_bokeh_level_from_exif(
    focus_plane_mm: torch.Tensor, metadata: CameraMetadata
) -> torch.Tensor:
    """K ≈ f² · D_focus / (2 · F · (D_focus − f)) · pixel_ratio (paper Eq. 3)."""
    f = torch.as_tensor(
        metadata.focal_length_mm, dtype=focus_plane_mm.dtype, device=focus_plane_mm.device
    )
    f_number = torch.as_tensor(
        metadata.f_number, dtype=focus_plane_mm.dtype, device=focus_plane_mm.device
    )
    pixel_ratio = torch.as_tensor(
        metadata.pixel_ratio, dtype=focus_plane_mm.dtype, device=focus_plane_mm.device
    )
    denominator = 2.0 * f_number * (focus_plane_mm - f)
    if torch.any(denominator == 0):
        raise ValueError("Denominador zero ao estimar K via EXIF.")
    return (f.square() * focus_plane_mm / denominator) * pixel_ratio


def normalize_defocus_condition(
    defocus_map: torch.Tensor, max_coc: float = 100.0
) -> torch.Tensor:
    """Normaliza D_def para [0,1] e replica em 3 canais."""
    if defocus_map.ndim != 3:
        raise ValueError("defocus_map must have shape (B, H, W).")
    condition = (defocus_map / max_coc).clamp(0.0, 1.0)
    return condition.unsqueeze(1).repeat(1, 3, 1, 1)


def laplacian_variance(image: torch.Tensor) -> torch.Tensor:
    """Variância da resposta laplaciana (usada na métrica LVCorr)."""
    if image.ndim == 3:
        image = image.unsqueeze(1)
    if image.ndim != 4:
        raise ValueError("image must have shape (B, C, H, W).")
    if image.shape[1] > 1:
        image = image.mean(dim=1, keepdim=True)

    kernel = torch.tensor(
        [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]],
        dtype=image.dtype,
        device=image.device,
    ).view(1, 1, 3, 3)

    response = F.conv2d(image, kernel, padding=1)
    return response.flatten(start_dim=1).var(dim=1, unbiased=False)


def pearson_corrcoef(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    if x.ndim != 1 or y.ndim != 1 or x.numel() != y.numel():
        raise ValueError("x e y devem ser 1D do mesmo tamanho.")
    x_centered = x - x.mean()
    y_centered = y - y.mean()
    denominator = torch.sqrt(x_centered.square().sum() * y_centered.square().sum())
    if denominator == 0:
        return torch.tensor(0.0, dtype=x.dtype, device=x.device)
    return (x_centered * y_centered).sum() / denominator


def lvcorr(
    k_values: Sequence[float] | torch.Tensor,
    laplacian_variances: Sequence[float] | torch.Tensor,
) -> torch.Tensor:
    """Métrica LVCorr (controllability — paper Tab. 3)."""
    x = torch.as_tensor(k_values, dtype=torch.float32)
    y = torch.as_tensor(laplacian_variances, dtype=torch.float32)
    return pearson_corrcoef(x, y)
