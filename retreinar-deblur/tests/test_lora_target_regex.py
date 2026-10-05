"""C1 — o alvo do LoRA não pode capturar a projeção final do transformer.

DEFEITO ORIGINAL: `LORA_TARGET_MODULES` era uma LISTA de strings soltas,
incluindo "proj_out". O PEFT casa lista por
`key == target or key.endswith("." + target)`, e a projeção final do
`FluxTransformer2DModel` chama-se literalmente `proj_out` no topo do módulo.
Resultado: além dos 38 `single_transformer_blocks.N.proj_out`, o LoRA era
injetado em `transformer.proj_out` — que `transformer_forward` executa FORA de
qualquer `specify_lora`, portanto ativo incondicionalmente, inclusive com
`main_adapter=None`. Ou seja, a variante dita "cond-only" não era cond-only.

Aritmética contra o checkpoint oficial (bokehNet.safetensors, 686 tensores =
343 módulos LoRA):
    19 duplos x 6 + 38 single x 6 + x_embedder = 343   <- oficial
    + transformer.proj_out                            = 344   <- nosso, errado

CORREÇÃO: `target_modules` vira um regex ancorado nos blocos. Quando
`target_modules` é `str`, o PEFT usa `re.fullmatch(target_modules, key)`.

Este teste lê a constante do backbone.py REAL (via AST, para não importar
torch) — não copia o padrão para dentro do teste.
"""

from __future__ import annotations

import re
import unittest

from _helpers import extrair_constante

N_DUPLOS = 19
N_SINGLE = 38

MODULOS_DUPLO = ["attn.to_q", "attn.to_k", "attn.to_v", "attn.to_out.0",
                 "norm1.linear", "ff.net.2"]
MODULOS_SINGLE = ["attn.to_q", "attn.to_k", "attn.to_v",
                  "norm.linear", "proj_mlp", "proj_out"]

# Módulos do branch de TEXTO e da cabeça: nenhum pode casar.
NAO_PODE_CASAR = [
    "proj_out",                 # <-- a projeção final. O bug do C1.
    "norm_out.linear",
    "x_embedder_nao_existe",
    "context_embedder",
    "time_text_embed.timestep_embedder.linear_1",
    "time_text_embed.timestep_embedder.linear_2",
    "time_text_embed.guidance_embedder.linear_1",
    "time_text_embed.text_embedder.linear_1",
    "transformer_blocks.0.norm1_context.linear",
    "transformer_blocks.0.ff_context.net.2",
    "transformer_blocks.0.attn.add_q_proj",
    "transformer_blocks.0.attn.add_k_proj",
    "transformer_blocks.0.attn.add_v_proj",
    "transformer_blocks.0.attn.to_add_out",
    "transformer_blocks.0.norm2",
    "single_transformer_blocks.0.attn.to_out.0",   # single não tem to_out
]


def arvore_de_modulos() -> list[str]:
    """Nomes de módulo do FLUX.1-dev, no formato que o PEFT vê."""
    nomes = ["x_embedder", "proj_out", "norm_out.linear", "context_embedder"]
    nomes += [f"time_text_embed.{s}.linear_{i}"
              for s in ("timestep_embedder", "guidance_embedder", "text_embedder")
              for i in (1, 2)]
    for b in range(N_DUPLOS):
        for m in MODULOS_DUPLO:
            nomes.append(f"transformer_blocks.{b}.{m}")
        for m in ("norm1_context.linear", "ff_context.net.2", "attn.add_q_proj",
                  "attn.add_k_proj", "attn.add_v_proj", "attn.to_add_out", "norm2"):
            nomes.append(f"transformer_blocks.{b}.{m}")
    for b in range(N_SINGLE):
        for m in MODULOS_SINGLE:
            nomes.append(f"single_transformer_blocks.{b}.{m}")
    return nomes


def casa_peft(padrao, chave: str) -> bool:
    """Reproduz `peft.tuners.tuners_utils.check_target_module_exists`."""
    if isinstance(padrao, str):
        return re.fullmatch(padrao, chave) is not None
    if chave in padrao:
        return True
    return any(chave.endswith(f".{alvo}") for alvo in padrao)


class TestAlvoLoRA(unittest.TestCase):
    def setUp(self):
        self.padrao = extrair_constante("backbone.py", "LORA_TARGET_MODULES")
        self.modulos = arvore_de_modulos()

    def test_o_alvo_e_um_regex_ancorado(self):
        self.assertIsInstance(
            self.padrao, str,
            msg=(
                "LORA_TARGET_MODULES ainda é uma lista de strings soltas. "
                "O C1 não foi aplicado: a string 'proj_out' captura também a "
                "projeção final do transformer."
            ),
        )
        self.assertTrue(self.padrao.startswith("^") or "|^" in self.padrao,
                        "regex sem âncora ^ casaria por sufixo")
        self.assertTrue(self.padrao.rstrip().endswith("$"),
                        "regex sem âncora $ casaria por prefixo")

    def test_exatamente_343_modulos_casam(self):
        casados = [m for m in self.modulos if casa_peft(self.padrao, m)]
        self.assertEqual(
            len(casados), 343,
            msg=(
                f"{len(casados)} módulos casaram; o checkpoint oficial tem 343. "
                f"Casados de topo (sem ponto): "
                f"{[m for m in casados if '.' not in m]}"
            ),
        )

    def test_a_projecao_final_do_transformer_nao_casa(self):
        self.assertFalse(
            casa_peft(self.padrao, "proj_out"),
            msg=(
                "`proj_out` de TOPO casou. É o bug do C1: essa camada roda fora "
                "do specify_lora, então o LoRA ficaria ativo mesmo com "
                "main_adapter=None."
            ),
        )

    def test_o_proj_out_dos_single_blocks_continua_casando(self):
        for b in (0, 17, 37):
            self.assertTrue(casa_peft(self.padrao, f"single_transformer_blocks.{b}.proj_out"))

    def test_x_embedder_de_topo_casa(self):
        self.assertTrue(casa_peft(self.padrao, "x_embedder"))

    def test_nada_do_branch_de_texto_nem_da_cabeca_casa(self):
        for m in NAO_PODE_CASAR:
            with self.subTest(modulo=m):
                self.assertFalse(casa_peft(self.padrao, m),
                                 msg=f"{m} não deveria receber LoRA")

    def test_contagem_por_grupo(self):
        casados = [m for m in self.modulos if casa_peft(self.padrao, m)]
        duplos = [m for m in casados if m.startswith("transformer_blocks.")]
        single = [m for m in casados if m.startswith("single_transformer_blocks.")]
        topo = [m for m in casados if "." not in m]
        self.assertEqual(len(duplos), N_DUPLOS * len(MODULOS_DUPLO))   # 114
        self.assertEqual(len(single), N_SINGLE * len(MODULOS_SINGLE))  # 228
        self.assertEqual(topo, ["x_embedder"])

    def test_a_lista_antiga_de_fato_produzia_344(self):
        """Prova de que o defeito era real, com a lista original."""
        lista_antiga = ["to_q", "to_k", "to_v", "to_out.0", "norm1.linear",
                        "ff.net.2", "norm.linear", "proj_mlp", "proj_out",
                        "x_embedder"]
        casados = [m for m in self.modulos if casa_peft(lista_antiga, m)]
        self.assertEqual(len(casados), 344)
        self.assertIn("proj_out", casados)


if __name__ == "__main__":
    unittest.main()
