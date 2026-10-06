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


# =============================================================================
# BokehNet: mapa de defocus (regressão do bug da coluna normalizada por imagem)
# =============================================================================

def _fake_depth_u16(w: int, h: int) -> np.ndarray:
    """Rampa de profundidade em uint16, como o df entrega (I;16, [0,65535])."""
    ramp = np.linspace(0.0, 1.0, w, dtype=np.float32)
    return np.tile(ramp, (h, 1)) .astype(np.float32) * 65535.0


def test_defocus_from_depth_reproduz_formula_oficial():
    from genfocus_train.data import defocus_from_depth
    depth = np.array([[0.0, 0.5, 1.0]], dtype=np.float32)
    out = defocus_from_depth(depth, k=100.0, s1=0.5, max_coc=100.0)
    # |100*(D-0.5)|/100 = |D-0.5|
    assert np.allclose(out, [[0.5, 0.0, 0.5]], atol=1e-6)


def test_defocus_recomputado_satura_e_respeita_clamp():
    from genfocus_train.data import defocus_from_depth
    depth = np.array([[0.0, 1.0]], dtype=np.float32)
    out = defocus_from_depth(depth, k=1000.0, s1=0.5, max_coc=100.0)
    assert out.min() >= 0.0 and out.max() <= 1.0
    assert np.allclose(out, 1.0)   # 1000*0.5/100 = 5.0 -> clampa em 1


def test_k_MUDA_o_mapa_recomputado():
    """O bug: na coluna do df o k nao influenciava o mapa. Aqui ele DEVE influenciar."""
    from genfocus_train.data import prepare_aligned_bokeh
    aif = _fake_image(512, 512, 128)
    bok = _fake_image(512, 512, 100)
    depth = _fake_depth_u16(512, 512)
    _, _, d_baixo = prepare_aligned_bokeh(
        aif, bok, image_size=512, train=False,
        depth_like=depth, k=20.0, s1=0.5, defocus_source="recompute",
    )
    _, _, d_alto = prepare_aligned_bokeh(
        aif, bok, image_size=512, train=False,
        depth_like=depth, k=150.0, s1=0.5, defocus_source="recompute",
    )
    assert float(d_alto.mean()) > float(d_baixo.mean()) * 2.0, (
        "k maior tem que produzir um mapa mais claro (mais blur)"
    )
    assert 0.0 <= float(d_baixo.min()) and float(d_alto.max()) <= 1.0
    assert d_baixo.shape == (3, 512, 512)


def test_coluna_legacy_ignora_o_k_regressao_documentada():
    """Documenta o defeito da coluna: normalizada por imagem, o k some."""
    from genfocus_train.data import prepare_aligned_bokeh
    aif = _fake_image(512, 512, 128)
    bok = _fake_image(512, 512, 100)
    # coluna do df: |D-s1|/max|D-s1| (o k JA foi apagado na geracao)
    col = np.abs(np.tile(np.linspace(0, 1, 512, dtype=np.float32), (512, 1)) - 0.5)
    col = (col / col.max() * 65535.0)
    _, _, d = prepare_aligned_bokeh(
        aif, bok, col, image_size=512, train=False, defocus_source="column",
    )
    assert abs(float(d.max()) - 1.0) < 1e-3   # sempre chega em 1.0, independente do k


def test_defocus_source_invalido_falha_alto():
    from genfocus_train.data import prepare_aligned_bokeh
    with pytest.raises(ValueError, match="defocus_source"):
        prepare_aligned_bokeh(
            _fake_image(64, 64, 10), _fake_image(64, 64, 20),
            image_size=64, train=False, defocus_source="sei_la",
        )


def test_recompute_exige_depth_k_s1():
    from genfocus_train.data import prepare_aligned_bokeh
    with pytest.raises(ValueError, match="recompute"):
        prepare_aligned_bokeh(
            _fake_image(64, 64, 10), _fake_image(64, 64, 20),
            image_size=64, train=False, defocus_source="recompute",
        )


# =============================================================================
# LocalBokehFolderDataset (pasta do prefetch_bokeh_local.py)
# =============================================================================

def _make_local_bokeh_folder(tmp_path, n=3, size=(600, 512)):
    """Cria uma pasta no layout do prefetch: aif/bokeh RGB + depth PNG 16-bit."""
    import json as _json
    from PIL import Image as _Image
    w, h = size
    for sub in ("aif", "bokeh", "depth"):
        (tmp_path / sub).mkdir(parents=True, exist_ok=True)
    with open(tmp_path / "metadata.jsonl", "w", encoding="utf-8") as fh:
        for i in range(n):
            name = f"{i:07d}_teste.png"
            _Image.fromarray(_fake_image(w, h, 100 + i)).save(tmp_path / "aif" / name)
            _Image.fromarray(_fake_image(w, h, 50 + i)).save(tmp_path / "bokeh" / name)
            ramp = np.tile(np.linspace(0, 65535, w, dtype=np.float32), (h, 1)).astype(np.uint16)
            _Image.fromarray(ramp).save(tmp_path / "depth" / name)
            fh.write(_json.dumps({"idx": i, "stem": "teste", "file": name,
                                  "k": 50.0 + 50 * i, "s1": 0.5}) + "\n")
        # linha de erro do prefetch deve ser IGNORADA pelo loader
        fh.write(_json.dumps({"idx": n, "stem": "SKIPPED", "file": None,
                              "error": "boom"}) + "\n")


def test_local_folder_dataset_carrega_e_recomputa(tmp_path):
    from genfocus_train.data import LocalBokehFolderDataset, DatasetRuntimeConfig
    _make_local_bokeh_folder(tmp_path, n=3)
    ds = LocalBokehFolderDataset(str(tmp_path), DatasetRuntimeConfig(image_size=512, train=False))
    assert len(ds) == 3  # a linha de erro nao conta
    s0, s2 = ds[0], ds[2]
    for s in (s0, s2):
        assert s["aif_image"].shape == (3, 512, 512)
        assert s["defocus_map"].min() >= 0.0 and s["defocus_map"].max() <= 1.0
    # k=50 (idx0) vs k=150 (idx2), mesmo depth => mapa mais claro no idx2
    assert float(s2["defocus_map"].mean()) > float(s0["defocus_map"].mean()) * 2.0


def test_local_folder_exige_recompute(tmp_path):
    from genfocus_train.data import LocalBokehFolderDataset, DatasetRuntimeConfig
    _make_local_bokeh_folder(tmp_path, n=1)
    with pytest.raises(ValueError, match="recompute"):
        LocalBokehFolderDataset(
            str(tmp_path), DatasetRuntimeConfig(image_size=512, train=False,
                                                defocus_source="column"))


def test_local_folder_faltando_da_erro_claro(tmp_path):
    from genfocus_train.data import LocalBokehFolderDataset, DatasetRuntimeConfig
    with pytest.raises(FileNotFoundError, match="prefetch"):
        LocalBokehFolderDataset(str(tmp_path / "nao_existe"),
                                DatasetRuntimeConfig(image_size=512, train=False))


def test_build_dataset_roteia_fonte_local(tmp_path):
    from genfocus_train.data import build_dataset, DatasetRuntimeConfig
    from genfocus_train.config import StageConfig, DatasetSourceConfig
    _make_local_bokeh_folder(tmp_path, n=2)
    cfg = StageConfig(datasets=[DatasetSourceConfig(name=str(tmp_path), split="train")],
                      steps=1)
    ds = build_dataset("bokeh", cfg, DatasetRuntimeConfig(image_size=512, train=False))
    assert len(ds) == 2 and ds[0]["aif_image"].shape == (3, 512, 512)


def test_prefetch_resize_e_identico_ao_dataloader(tmp_path):
    """A fórmula de target do prefetch TEM que casar com prepare_aligned_bokeh."""
    import importlib.util, pathlib
    spec = importlib.util.spec_from_file_location(
        "prefetch", pathlib.Path(__file__).parent.parent / "scripts" / "prefetch_bokeh_local.py")
    prefetch = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(prefetch)
    for (w, h) in [(4000, 3000), (3000, 4000), (512, 512), (300, 700), (513, 511)]:
        nw, nh = prefetch.target_dims(w, h, 512)
        # réplica da fórmula do dataloader (prepare_aligned_bokeh)
        scale = 512 / min(w, h)
        assert (nw, nh) == (max(512, round(w * scale)), max(512, round(h * scale))), (w, h)
        assert min(nw, nh) >= 512


def test_local_folder_dedup_por_idx(tmp_path, capsys):
    """metadata.jsonl append-only: prefetch rodado 2x repete linhas (idx igual)."""
    import json as _json
    from genfocus_train.data import LocalBokehFolderDataset, DatasetRuntimeConfig
    _make_local_bokeh_folder(tmp_path, n=3)
    # simula a 2a rodada do prefetch: reescreve as MESMAS 3 linhas no fim
    linhas = (tmp_path / "metadata.jsonl").read_text().splitlines()
    validas = [l for l in linhas if _json.loads(l).get("file")]
    with open(tmp_path / "metadata.jsonl", "a", encoding="utf-8") as fh:
        for l in validas:
            fh.write(l + "\n")
    ds = LocalBokehFolderDataset(str(tmp_path), DatasetRuntimeConfig(image_size=512, train=False))
    assert len(ds) == 3, f"esperava 3 apos dedup, veio {len(ds)}"
    assert [r["idx"] for r in ds.records] == [0, 1, 2]   # ordenado por idx
    assert "dedup por idx" in capsys.readouterr().out
    assert ds[2]["aif_image"].shape == (3, 512, 512)     # ainda carrega
