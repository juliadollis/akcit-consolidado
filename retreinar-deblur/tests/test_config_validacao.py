"""Config inválido tem que EXPLODIR no load, não virar comportamento silencioso.

Um typo em `scale_mode` que caísse no default gastaria uma fila de cluster
inteira produzindo o braço errado do experimento, e o resultado pareceria
válido. Por isso os vocabulários são fechados e validados no `__post_init__`.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import yaml

from genfocus_train.config import (
    DEFOCUS_SOURCES,
    SCALE_MODES,
    SIGMA_MU_SOURCES,
    TOP_K_MODES,
    StageConfig,
    load_config,
)

BASE = {"datasets": [{"name": "r/d", "split": "train"}], "steps": 100}


def _carregar(**extra):
    bloco = dict(BASE, **extra)
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "cfg.yaml"
        p.write_text(yaml.safe_dump({"data": {"deblur": bloco}}), encoding="utf-8")
        return load_config(p)


class TestVocabularios(unittest.TestCase):
    def test_scale_mode_invalido(self):
        with self.assertRaises(ValueError) as ctx:
            _carregar(scale_mode="nativo")     # pt-BR por engano
        self.assertIn("scale_mode", str(ctx.exception))

    def test_sigma_mu_source_invalido(self):
        with self.assertRaises(ValueError):
            _carregar(sigma_mu_source="imagem_inteira")

    def test_top_k_mode_invalido(self):
        with self.assertRaises(ValueError):
            _carregar(top_k_mode="cena")

    def test_defocus_source_invalido(self):
        with self.assertRaises(ValueError):
            _carregar(defocus_source="coluna")

    def test_todos_os_valores_do_vocabulario_sao_aceitos(self):
        for m in SCALE_MODES:
            extra = {"scale_mode": m}
            if m == "long_side":
                extra["batch_size"] = 1
            _carregar(**extra)
        for m in SIGMA_MU_SOURCES:
            _carregar(sigma_mu_source=m)
        for m in TOP_K_MODES:
            _carregar(top_k_mode=m)
        for m in DEFOCUS_SOURCES:
            _carregar(defocus_source=m)


class TestRestricoesNumericas(unittest.TestCase):
    def test_image_size_precisa_ser_multiplo_de_16(self):
        # VAE 8x + _pack_latents 2x: 500 não fecha em token inteiro.
        with self.assertRaises(ValueError) as ctx:
            _carregar(image_size=500)
        self.assertIn("16", str(ctx.exception))

    def test_image_size_multiplo_de_16_passa(self):
        for s in (256, 512, 768, 1024):
            _carregar(image_size=s)

    def test_long_side_com_batch_maior_que_um_e_recusado(self):
        # aspecto variável por amostra -> o collate exigiria shapes iguais
        with self.assertRaises(ValueError) as ctx:
            _carregar(scale_mode="long_side", batch_size=4)
        self.assertIn("batch_size", str(ctx.exception))

    def test_long_side_com_batch_um_passa(self):
        cfg = _carregar(scale_mode="long_side", batch_size=1)
        self.assertEqual(cfg.data.deblur.scale_mode, "long_side")

    def test_datasets_vazio_e_recusado(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "cfg.yaml"
            p.write_text(
                yaml.safe_dump({"data": {"deblur": {"datasets": [], "steps": 10}}}),
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                load_config(p)

    def test_construcao_direta_tambem_valida(self):
        # __post_init__ protege quem monta o StageConfig em código, não só via YAML
        with self.assertRaises(ValueError):
            StageConfig(datasets=[object()], steps=1, scale_mode="xxx")


if __name__ == "__main__":
    unittest.main()
