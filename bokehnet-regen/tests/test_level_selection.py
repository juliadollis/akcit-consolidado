"""Testes do teto de níveis por cena.

O que precisa ser provado, e por quê:

* **Nenhuma cena é descartada.** O teto tira peso excessivo de poucas cenas; se ele
  eliminasse cenas, estaria reduzindo diversidade em vez de reequilibrá-la — o oposto
  do objetivo.
* **A aritmética do paper é reproduzida.** O histograma real do split `train` com teto
  de 4 tem que dar 13.799, o número que fecha com os "13K" publicados. Se isso mudar,
  a justificativa do teto caiu junto.
* **A seleção é determinística.** Duas execuções dão os mesmos `sample_id`, senão o
  release não é reproduzível.
* **Os extremos de abertura sobrevivem.** É a amplitude de K dentro da cena que carrega
  a supervisão; um seletor que ficasse com o miolo jogaria fora justamente o sinal.
"""

from __future__ import annotations

import sys
import unittest
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sources.level_selection import (                                 # noqa: E402
    PAPER_MAX_LEVELS_PER_SCENE, _uniform_indices, cap_levels_per_scene, cap_summary,
)


class _Par:
    def __init__(self, cena: str, nivel: int):
        self.scene_id = cena
        self.level = nivel
        self.sample_id = f"c_realbokeh_{cena}_l{nivel}"

    def __repr__(self):
        return self.sample_id


def _cena(cena: str, niveis: int) -> list[_Par]:
    return [_Par(cena, n) for n in range(1, niveis + 1)]


#: Histograma MEDIDO do split `train` do espelho: {níveis por cena: quantas cenas}.
#: 3.959 cenas, 20.495 amostras. Ver reference/ACHADOS.md.
HISTOGRAMA_TRAIN = {1: 2, 2: 705, 3: 621, 4: 1, 5: 2341, 6: 1, 7: 9, 9: 34, 12: 1, 21: 244}

#: O mesmo, nos TRÊS splits — que é o que o enumerador percorre. 4.399 cenas,
#: 22.990 amostras. Impresso pelo `enumeration_summary` do job 32224.
HISTOGRAMA_TODOS = {1: 2, 2: 754, 3: 655, 4: 1, 5: 2663, 6: 1, 7: 10, 9: 37, 12: 1,
                    21: 275}


class TestIndicesUniformes(unittest.TestCase):

    def test_inclui_os_extremos(self):
        for total in (2, 5, 9, 21):
            with self.subTest(total=total):
                idx = _uniform_indices(total, 4)
                self.assertEqual(idx[0], 0)
                self.assertEqual(idx[-1], total - 1)

    def test_vinte_e_um_para_quatro(self):
        self.assertEqual(_uniform_indices(21, 4), [0, 7, 13, 20])

    def test_menos_que_o_teto_devolve_tudo(self):
        self.assertEqual(_uniform_indices(3, 4), [0, 1, 2])
        self.assertEqual(_uniform_indices(1, 4), [0])

    def test_espacamento_e_regular(self):
        idx = _uniform_indices(21, 5)
        passos = [b - a for a, b in zip(idx, idx[1:])]
        self.assertLessEqual(max(passos) - min(passos), 1)


class TestTeto(unittest.TestCase):

    def test_corta_para_o_teto(self):
        self.assertEqual(len(cap_levels_per_scene(_cena("a", 21), max_levels=4)), 4)

    def test_cena_menor_que_o_teto_fica_inteira(self):
        self.assertEqual(len(cap_levels_per_scene(_cena("a", 2), max_levels=4)), 2)

    def test_NENHUMA_cena_e_descartada(self):
        pares = _cena("a", 21) + _cena("b", 2) + _cena("c", 5) + _cena("d", 1)
        depois = cap_levels_per_scene(pares, max_levels=4)
        self.assertEqual({p.scene_id for p in depois}, {"a", "b", "c", "d"})

    def test_mantem_os_extremos_de_abertura(self):
        """A amplitude de K dentro da cena é o que carrega a supervisão."""
        depois = cap_levels_per_scene(_cena("a", 21), max_levels=4)
        niveis = sorted(p.level for p in depois)
        self.assertEqual(niveis[0], 1)
        self.assertEqual(niveis[-1], 21)

    def test_preserva_a_ordem_de_entrada(self):
        """Quem chamou já pode ter ordenado para leitura sequencial de parquet."""
        pares = [_Par("a", 3), _Par("b", 1), _Par("a", 1), _Par("b", 2), _Par("a", 2)]
        depois = cap_levels_per_scene(pares, max_levels=2)
        self.assertEqual(depois, [p for p in pares if p in depois])

    def test_e_deterministico(self):
        pares = _cena("a", 21) + _cena("b", 9)
        a = [p.sample_id for p in cap_levels_per_scene(pares, max_levels=4)]
        b = [p.sample_id for p in cap_levels_per_scene(pares, max_levels=4)]
        self.assertEqual(a, b)

    def test_teto_de_um_e_valido(self):
        self.assertEqual(len(cap_levels_per_scene(_cena("a", 21), max_levels=1)), 1)

    def test_teto_zero_e_erro(self):
        with self.assertRaises(ValueError):
            cap_levels_per_scene(_cena("a", 5), max_levels=0)

    def test_lista_vazia_devolve_vazio(self):
        self.assertEqual(cap_levels_per_scene([], max_levels=4), [])

    def test_niveis_fora_de_ordem_ainda_ordenam_por_abertura(self):
        """O espaçamento é pela ABERTURA, não pela ordem no shard."""
        pares = [_Par("a", n) for n in (5, 1, 3, 2, 4)]
        niveis = sorted(p.level for p in cap_levels_per_scene(pares, max_levels=3))
        self.assertEqual(niveis, [1, 3, 5])


class TestAritmeticaDoPaper(unittest.TestCase):
    """Se estes números mudarem, a justificativa do teto caiu junto."""

    def _universo_train(self) -> list[_Par]:
        pares, i = [], 0
        for niveis, quantas in HISTOGRAMA_TRAIN.items():
            for _ in range(quantas):
                i += 1
                pares.extend(_cena(f"train_{i}", niveis))
        return pares

    def test_o_histograma_medido_soma_o_que_medimos(self):
        pares = self._universo_train()
        self.assertEqual(len({p.scene_id for p in pares}), 3959)
        self.assertEqual(len(pares), 20495)

    def test_teto_de_4_reproduz_os_13K_do_paper(self):
        """13.799 ≈ "13K", com folga para o limiar de SSIM que eles aplicam depois."""
        depois = cap_levels_per_scene(self._universo_train(),
                                      max_levels=PAPER_MAX_LEVELS_PER_SCENE)
        self.assertEqual(len(depois), 13799)

    def test_os_13K_sao_de_TREINO_e_a_comparacao_tem_que_respeitar_isso(self):
        """A imprecisão que uma auditoria pegou, e a conta que a resolve.

        O enumerador percorre os três splits, e o total com teto 4 é **15.427** — que
        comparado a "13K" pareceria 18% acima e sugeriria trocar o teto para 3. Mas
        `test` e `validation` são RETIDOS, não treinados, e os 13K do paper descrevem
        dado de treino. Contra o split `train`, o teto 4 dá 13.799: **+6,1%**.

        Com o teto 3 a conclusão se inverteria para −14,1% — e é por isso que a
        comparação precisa dizer contra o quê está comparando.
        """
        def total(hist, cap):
            return sum(min(k, cap) * n for k, n in hist.items())

        self.assertEqual(total(HISTOGRAMA_TRAIN, 4), 13799)
        self.assertEqual(total(HISTOGRAMA_TODOS, 4), 15427)
        self.assertEqual(total(HISTOGRAMA_TODOS, 4) - total(HISTOGRAMA_TRAIN, 4), 1628)

        # e o teto 3, para que a alternativa fique medida e não suposta
        self.assertEqual(total(HISTOGRAMA_TRAIN, 3), 11168)
        self.assertEqual(total(HISTOGRAMA_TODOS, 3), 12439)

        de_treino = abs(total(HISTOGRAMA_TRAIN, 4) / 13000 - 1)
        self.assertLess(de_treino, 0.10, "13.799 fica a +6,1% de 13K")

    def test_o_histograma_dos_tres_splits_soma_o_medido(self):
        self.assertEqual(sum(HISTOGRAMA_TODOS.values()), 4399)
        self.assertEqual(sum(k * n for k, n in HISTOGRAMA_TODOS.items()), 22990)

    def test_teto_de_3_ficaria_longe_demais(self):
        self.assertEqual(len(cap_levels_per_scene(self._universo_train(), max_levels=3)),
                         11168)

    def test_sem_teto_fica_58_porcento_acima_do_publicado(self):
        pares = self._universo_train()
        self.assertAlmostEqual(len(pares) / 13799, 1.485, places=2)

    def test_o_desequilibrio_que_o_teto_corrige(self):
        """As 244 cenas de 21 níveis são 6,2% das cenas e geram 25% das amostras.

        É este número que torna "20.495 amostras" uma afirmação enganosa: um quarto do
        sinal de treino vinha de 6,2% do conteúdo, repetido 21 vezes.
        """
        pares = self._universo_train()
        tamanho = Counter(p.scene_id for p in pares)
        cenas_de_21 = {c for c, n in tamanho.items() if n == 21}
        amostras_de_21 = sum(tamanho[c] for c in cenas_de_21)

        self.assertEqual(len(cenas_de_21), 244)
        self.assertAlmostEqual(len(cenas_de_21) / 3959, 0.062, places=3)
        self.assertEqual(amostras_de_21, 5124)
        self.assertAlmostEqual(amostras_de_21 / 20495, 0.250, places=3)

        # e depois do teto elas passam a pesar 976 de 13.799 — de 25% para 7,1%
        depois = cap_levels_per_scene(pares, max_levels=4)
        de_21_depois = sum(1 for p in depois if p.scene_id in cenas_de_21)
        self.assertEqual(de_21_depois, 976)
        self.assertAlmostEqual(de_21_depois / 13799, 0.071, places=3)

    def test_o_teto_nao_perde_cena_nenhuma_no_universo_real(self):
        pares = self._universo_train()
        depois = cap_levels_per_scene(pares, max_levels=4)
        self.assertEqual(len({p.scene_id for p in depois}), 3959)


class TestResumo(unittest.TestCase):

    def test_conta_em_cenas_E_em_amostras(self):
        antes = _cena("a", 21) + _cena("b", 2)
        depois = cap_levels_per_scene(antes, max_levels=4)
        texto = cap_summary(antes, depois, max_levels=4)
        self.assertIn("amostras", texto)
        self.assertIn("cenas", texto)
        self.assertIn("23", texto)          # antes
        self.assertIn("6", texto)           # depois: 4 + 2

    def test_denuncia_o_desequilibrio(self):
        antes = _cena("a", 21) + _cena("b", 2)
        texto = cap_summary(antes, cap_levels_per_scene(antes, max_levels=4),
                            max_levels=4)
        self.assertIn("acima do teto", texto)

    def test_cita_o_paper(self):
        antes = _cena("a", 5)
        self.assertIn("paper.txt",
                      cap_summary(antes, cap_levels_per_scene(antes, max_levels=4),
                                  max_levels=4))


if __name__ == "__main__":
    unittest.main()
