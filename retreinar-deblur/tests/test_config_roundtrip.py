"""Toda chave de `StageConfig` sobrevive à ida e volta pelo YAML.

POR QUE ESTE TESTE EXISTE: `_as_stage_config` monta o `StageConfig` campo a
campo, à mão. Adicionar um campo na dataclass sem adicionar a linha
correspondente lá faz a chave do YAML ser SILENCIOSAMENTE IGNORADA — o treino
roda, não reclama, e usa o default. Foi assim que um eixo de experimento
poderia ficar hardcodado sem ninguém notar.

O teste enumera os campos com `dataclasses.fields`, não com uma lista fixa.
Campo novo sem valor de teste registrado aqui FALHA o teste de propósito: é o
lembrete de manter os dois lugares em dia.
"""

from __future__ import annotations

import dataclasses
import tempfile
import unittest
from pathlib import Path

import yaml

from genfocus_train.config import StageConfig, load_config

# valor_yaml, valor_esperado_em_python. Sempre DIFERENTE do default, senão o
# teste passaria mesmo com a chave sendo ignorada.
VALORES: dict[str, tuple[object, object]] = {
    "datasets": (
        [{"name": "repo/treino", "split": "train",
          "top_k_sharpest": 7, "sharpness_column": "coluna_x"}],
        None,   # conferido à parte (é lista de dataclass)
    ),
    "steps": (1234, 1234),
    "batch_size": (3, 3),
    "image_size": (256, 256),
    "shuffle": (False, False),
    "max_samples": (11, 11),
    "augment": (False, False),
    "val_datasets": (
        [{"name": "repo/val", "split": "validation"}],
        None,   # conferido à parte
    ),
    "scale_mode": ("native", "native"),
    "sigma_mu_source": ("full_image", "full_image"),
    "top_k_mode": ("scene", "scene"),
    "scene_key": ("image_hash", "image_hash"),
    "defocus_source": ("kfix", "kfix"),
    "min_calibration_ssim": (0.77, 0.77),
    "kfix_repo": ("repo/kfix", "repo/kfix"),
    "geo_condition": (True, True),
    "geo_escalares": ("/tmp/escalares.jsonl", "/tmp/escalares.jsonl"),
    "geo_constantes": ({"alpha": 1.5}, {"alpha": 1.5}),
    "geo_field": ("depth", "depth"),
    "geo_sem_escalares": ("pula", "pula"),
    "geo_ruido_controle": (True, True),
}


def _carregar(bloco: dict) -> StageConfig:
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "cfg.yaml"
        p.write_text(
            yaml.safe_dump({"data": {"deblur": bloco}}, allow_unicode=True),
            encoding="utf-8",
        )
        return load_config(p).data.deblur


class TestRoundTrip(unittest.TestCase):
    def test_todo_campo_da_dataclass_tem_valor_de_teste(self):
        campos = {f.name for f in dataclasses.fields(StageConfig)}
        faltando = campos - set(VALORES)
        self.assertEqual(
            faltando, set(),
            msg=(
                f"Campos novos em StageConfig sem valor de teste: {sorted(faltando)}. "
                "Adicione em VALORES aqui E confira que _as_stage_config lê a chave "
                "— senão o YAML é ignorado em silêncio."
            ),
        )
        sobrando = set(VALORES) - campos
        self.assertEqual(sobrando, set(),
                         msg=f"VALORES tem campos que não existem mais: {sorted(sobrando)}")

    def test_cada_campo_chega_do_yaml_ate_a_dataclass(self):
        bloco = {nome: yaml_val for nome, (yaml_val, _) in VALORES.items()}
        cfg = _carregar(bloco)

        for nome, (_, esperado) in VALORES.items():
            if esperado is None:
                continue
            with self.subTest(campo=nome):
                obtido = getattr(cfg, nome)
                self.assertEqual(
                    obtido, esperado,
                    msg=(
                        f"StageConfig.{nome} = {obtido!r}, esperado {esperado!r}. "
                        "Provável causa: falta a linha desse campo em "
                        "_as_stage_config (config.py)."
                    ),
                )

    def test_datasets_e_val_datasets_viram_dataclasses(self):
        bloco = {nome: yaml_val for nome, (yaml_val, _) in VALORES.items()}
        cfg = _carregar(bloco)

        self.assertEqual(len(cfg.datasets), 1)
        d = cfg.datasets[0]
        self.assertEqual(d.name, "repo/treino")
        self.assertEqual(d.split, "train")
        self.assertEqual(d.top_k_sharpest, 7)
        self.assertEqual(d.sharpness_column, "coluna_x")

        self.assertEqual(len(cfg.val_datasets), 1)
        v = cfg.val_datasets[0]
        self.assertEqual(v.name, "repo/val")
        self.assertEqual(v.split, "validation")
        # default preservado quando a chave não vem
        self.assertIsNone(v.top_k_sharpest)
        self.assertEqual(v.sharpness_column, "image_focus")

    def test_val_datasets_ausente_vira_lista_vazia_e_nao_none(self):
        cfg = _carregar({"datasets": VALORES["datasets"][0], "steps": 10})
        self.assertEqual(cfg.val_datasets, [])

    def test_defaults_quando_o_yaml_so_traz_o_obrigatorio(self):
        cfg = _carregar({"datasets": VALORES["datasets"][0], "steps": 10})
        self.assertEqual(cfg.scale_mode, "short_side")
        self.assertEqual(cfg.sigma_mu_source, "crop")
        self.assertEqual(cfg.top_k_mode, "row")
        self.assertEqual(cfg.image_size, 512)


if __name__ == "__main__":
    unittest.main()
