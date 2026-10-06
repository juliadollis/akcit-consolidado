"""Testes do contrato de dados da DeblurNet (sem rede / sem GPU).

Cobrem as funções puras de pré-processamento e o coerce de config. Não baixam
nada do Hugging Face — validam a transformação que o dataloader aplica a cada par.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from genfocus_train.data import collate_strict, prepare_aligned_pair
from genfocus_train.config import _coerce_config_dict


def _fake_image(w: int, h: int, value: int) -> np.ndarray:
    return np.full((h, w, 3), value, dtype=np.uint8)


def test_output_shape_dtype_and_range():
    blur = _fake_image(800, 600, 40)
    aif = _fake_image(800, 600, 200)
    b, a = prepare_aligned_pair(blur, aif, image_size=512, train=False)

    assert b.shape == (3, 512, 512)
    assert a.shape == (3, 512, 512)
    assert b.dtype == torch.float32 and a.dtype == torch.float32
    # [-1, 1]
    assert float(b.min()) >= -1.0 - 1e-6 and float(b.max()) <= 1.0 + 1e-6
    # valor 40/127.5 - 1 ≈ -0.686 ; 200/127.5 - 1 ≈ 0.569
    assert abs(float(b.mean()) - (40 / 127.5 - 1.0)) < 1e-3
    assert abs(float(a.mean()) - (200 / 127.5 - 1.0)) < 1e-3


def test_alignment_blurry_and_aif_share_geometry():
    # Metade esquerda escura, metade direita clara — mesma geometria nas duas.
    img = np.zeros((600, 800, 3), dtype=np.uint8)
    img[:, 400:, :] = 255
    b, a = prepare_aligned_pair(img.copy(), img.copy(), image_size=512, train=False)
    # Como blurry e aif vêm da MESMA imagem, o resultado deve ser idêntico.
    assert torch.allclose(b, a)


def test_square_input_is_not_distorted():
    # Imagem quadrada → resize para 512×512 sem crop perde nada de geometria.
    img = _fake_image(1024, 1024, 128)
    b, _ = prepare_aligned_pair(img, img, image_size=512, train=False)
    assert b.shape == (3, 512, 512)


def test_rejects_non_multiple_of_16():
    img = _fake_image(300, 300, 100)
    with pytest.raises(ValueError):
        prepare_aligned_pair(img, img, image_size=500, train=False)


def test_center_crop_is_deterministic():
    blur = _fake_image(900, 600, 70)
    aif = _fake_image(900, 600, 70)
    b1, _ = prepare_aligned_pair(blur, aif, image_size=512, train=False)
    b2, _ = prepare_aligned_pair(blur, aif, image_size=512, train=False)
    assert torch.allclose(b1, b2)


def test_collate_stacks_tensors_and_lists_strings():
    sample = {
        "id": "x",
        "file_name_base": "x",
        "blurry_image": torch.zeros(3, 64, 64),
        "aif_image": torch.ones(3, 64, 64),
    }
    out = collate_strict([sample, sample])
    assert out["blurry_image"].shape == (2, 3, 64, 64)
    assert out["aif_image"].shape == (2, 3, 64, 64)
    assert out["id"] == ["x", "x"]


def test_config_coerce_uses_dataset_sources():
    payload = {
        "data": {
            "deblur": {
                "datasets": [
                    {"name": "akcit-pixel/DDPD", "split": "train"},
                    {"name": "akcit-pixel/RealBokeh", "split": "train"},
                ],
                "steps": 60000,
                "image_size": 512,
            }
        },
        "runtime": {"gradient_accumulation_steps": 32},
    }
    cfg = _coerce_config_dict(payload)
    assert [d.name for d in cfg.data.deblur.datasets] == [
        "akcit-pixel/DDPD",
        "akcit-pixel/RealBokeh",
    ]
    assert cfg.data.deblur.steps == 60000
    assert cfg.runtime.gradient_accumulation_steps == 32
    assert cfg.model.deblur_lora_rank == 128  # default do paper
