"""Testes do script que valida o refinamento da região em foco contra gabarito.

O que precisa ser provado aqui, e por quê:

* **A separação por `focus_source` é real** — sem ela não dá para treinar com e sem as
  amostras refinadas e medir a diferença, que é o requisito que originou a marcação.
* **O veredito não maquia.** Um script de validação que só sabe dizer "melhorou" não
  valida nada. Há um teste para o caso em que o refinamento **piora**, e ele exige a
  palavra PIOROU no relatório.
* **Amostra sem gabarito é contada, não ignorada.** Uma fração de acerto calculada sobre
  um subconjunto silencioso é o jeito mais fácil de publicar número bonito.
* **A separação de causas funciona**: gabarito fora da faixa de disparidade da cena é
  divergência de PROFUNDIDADE, e o relatório tem que dizer isso em vez de culpar a
  máscara.
* **Contagem em cenas E em amostras**, porque os níveis de uma cena compartilham a AIF.
"""

from __future__ import annotations

import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

_RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_RAIZ / "src"))
sys.path.insert(0, str(_RAIZ / "scripts"))

from validate_focus_refinement import (                                # noqa: E402
    BASELINE_WITHIN_25_PCT, avalia, carrega_gabarito, carrega_metadados,
    comparacao_pareada, estatisticas, imprime_relatorio, maiores_divergencias,
    por_cena, separacao_de_causas,
)


# --------------------------------------------------------------------------------
# Piloto sintético
# --------------------------------------------------------------------------------

def _meta(sample_id, scene_id, *, focus_disparity, focus_source="birefnet",
          base=None, disparity_min=0.01, disparity_max=10.0, **over):
    meta = {
        "sample_id": sample_id, "scene_id": scene_id, "route": "c",
        "focus_disparity": focus_disparity,
        "focus_source": focus_source,
        "focus_was_refined": focus_source != "birefnet",
        "focus_agreement": 0.0 if focus_source == "retention_only" else 0.7,
        "focus_retention_in_region": 0.8,
        "focus_region_area_ratio": 0.05,
        "focus_disparity_from_initial_mask": base,
        "disparity_min": disparity_min, "disparity_max": disparity_max,
    }
    meta.update(over)
    return meta


def _escreve_piloto(root: Path, metas):
    (root / "meta").mkdir(parents=True, exist_ok=True)
    for meta in metas:
        (root / "meta" / f"{meta['sample_id']}.json").write_text(
            json.dumps(meta), encoding="utf-8")


def _gabarito(root: Path, mapa):
    caminho = root / "gabarito.json"
    caminho.write_text(json.dumps(mapa), encoding="utf-8")
    return caminho


class Avaliacao(unittest.TestCase):

    def test_razao_e_obtida_sobre_medida_e_bate_com_a_medicao_registrada(self):
        """A convenção tem que ser a mesma de `reference/MEDICAO_PLANO_FOCO.md`.

        Lá, `train_91`: foco medido a 10,230 m, implicado pela máscara a 1,649 m, razão
        **6,20**. Ou seja razão = medida ÷ implicada = obtida ÷ gabarito, nas
        disparidades. Se esta convenção virar do avesso, todo número deste script fica
        incomparável com o baseline de 35,2%.
        """
        metas = [_meta("s1", "train_91", focus_disparity=1.0 / 1.649)]
        linhas, fora = avalia(metas, {"train_91": {
            "focus_plane_distance_m": 10.230, "focus_plane_uncertainty_m": 2.06}})
        self.assertEqual(fora, {})
        self.assertAlmostEqual(linhas[0]["razao"], 6.20, delta=0.01)
        self.assertAlmostEqual(linhas[0]["implicada_m"], 1.649, places=3)

    def test_amostra_sem_gabarito_e_CONTADA_e_nao_silenciada(self):
        metas = [_meta("s1", "cena_a", focus_disparity=0.5),
                 _meta("s2", "sem_gabarito", focus_disparity=0.5)]
        linhas, fora = avalia(metas, {"cena_a": {"focus_plane_distance_m": 2.0,
                                                 "focus_plane_uncertainty_m": 0.01}})
        self.assertEqual(len(linhas), 1)
        self.assertEqual(fora["sem_gabarito_para_a_cena"], 1)

    def test_metadado_de_release_ANTIGO_nao_e_tratado_como_birefnet(self):
        """Fingir `birefnet` num release sem `focus_source` inventaria proveniência."""
        meta = _meta("s1", "cena_a", focus_disparity=0.5)
        del meta["focus_source"]
        linhas, fora = avalia([meta], {"cena_a": {"focus_plane_distance_m": 2.0,
                                                  "focus_plane_uncertainty_m": 0.01}})
        self.assertEqual(linhas, [])
        self.assertEqual(fora["metadado_sem_focus_source"], 1)

    def test_gabarito_de_distancia_invalida_e_descartado(self):
        with tempfile.TemporaryDirectory() as tmp:
            caminho = _gabarito(Path(tmp), {"c": {"focus_plane_distance_m": 0.0}})
            self.assertEqual(carrega_gabarito(raw_dir=None,
                                              gabarito_json=str(caminho)), {})

    def test_gabarito_aceita_numero_solto(self):
        with tempfile.TemporaryDirectory() as tmp:
            caminho = _gabarito(Path(tmp), {"c": 2.5})
            g = carrega_gabarito(raw_dir=None, gabarito_json=str(caminho))
            self.assertEqual(g["c"]["focus_plane_distance_m"], 2.5)


class SeparacaoPorFonte(unittest.TestCase):

    def _linhas(self):
        metas = [
            _meta("b1", "c1", focus_disparity=0.50, focus_source="birefnet", base=0.50),
            _meta("b2", "c2", focus_disparity=0.48, focus_source="birefnet", base=0.48),
            _meta("r1", "c3", focus_disparity=0.50, focus_source="birefnet_refined",
                  base=0.10),
            _meta("r2", "c4", focus_disparity=2.00, focus_source="birefnet_refined",
                  base=0.50),
            _meta("t1", "c5", focus_disparity=0.55, focus_source="retention_only"),
        ]
        gab = {f"c{i}": {"focus_plane_distance_m": 2.0,
                         "focus_plane_uncertainty_m": 0.01} for i in range(1, 6)}
        return avalia(metas, gab)[0]

    def test_cada_fonte_tem_estatistica_propria(self):
        linhas = self._linhas()
        por_fonte = {f: [l for l in linhas if l["focus_source"] == f]
                     for f in ("birefnet", "birefnet_refined", "retention_only")}
        self.assertEqual([len(v) for v in por_fonte.values()], [2, 2, 1])
        self.assertEqual(estatisticas(por_fonte["birefnet"])["dentro"]["±10%"], 1.0)
        # `r2` está 4x fora; `r1` acerta. Metade dentro de ±10%.
        self.assertEqual(estatisticas(por_fonte["birefnet_refined"])["dentro"]["±10%"],
                         0.5)

    def test_conta_em_cenas_E_em_amostras(self):
        """Níveis da mesma cena compartilham AIF, máscara e profundidade: contar cada
        nível como unidade independente infla qualquer fração de acerto."""
        metas = [_meta(f"n{i}", "mesma_cena", focus_disparity=0.5) for i in range(5)]
        metas.append(_meta("outra", "outra_cena", focus_disparity=0.1))
        gab = {"mesma_cena": {"focus_plane_distance_m": 2.0,
                              "focus_plane_uncertainty_m": 0.0},
               "outra_cena": {"focus_plane_distance_m": 2.0,
                              "focus_plane_uncertainty_m": 0.0}}
        linhas = avalia(metas, gab)[0]
        st = estatisticas(linhas)
        self.assertEqual(st["n_amostras"], 6)
        self.assertEqual(st["n_cenas"], 2)
        # 5 de 6 amostras acertam (83%), mas 1 de 2 cenas (50%).
        self.assertAlmostEqual(st["dentro"]["±10%"], 5 / 6, places=6)
        self.assertAlmostEqual(st["dentro_por_cena"]["±10%"], 0.5, places=6)

    def test_por_cena_usa_a_mediana_dos_niveis(self):
        """A `focus_source` pode variar entre níveis da mesma cena — a retenção depende
        da bokeh, que muda com o nível. Então a agregação por cena não pode assumir um
        valor único."""
        metas = [_meta("a", "c", focus_disparity=0.5, focus_source="birefnet"),
                 _meta("b", "c", focus_disparity=0.5, focus_source="retention_only"),
                 _meta("d", "c", focus_disparity=1.0, focus_source="retention_only")]
        gab = {"c": {"focus_plane_distance_m": 2.0, "focus_plane_uncertainty_m": 0.0}}
        linhas = avalia(metas, gab)[0]
        self.assertEqual(len(por_cena(linhas)), 1)
        self.assertAlmostEqual(por_cena(linhas)[0], 1.0, places=6)


class ComparacaoPareada(unittest.TestCase):

    def _roda(self, metas, gab):
        return comparacao_pareada(avalia(metas, gab)[0])

    def test_amostra_de_mascara_vazia_conta_como_RECUPERADA(self):
        """Sem linha de base, a alternativa não é um rótulo pior: é amostra nenhuma."""
        metas = [_meta("t1", "c1", focus_disparity=0.5,
                       focus_source="retention_only", base=None)]
        gab = {"c1": {"focus_plane_distance_m": 2.0, "focus_plane_uncertainty_m": 0.0}}
        pareada = self._roda(metas, gab)
        self.assertEqual(pareada["recuperadas_sem_linha_de_base"]["n_amostras"], 1)
        self.assertEqual(pareada["so_as_refinadas"]["n_amostras"], 0)

    def test_amostra_nao_refinada_tem_antes_igual_a_depois(self):
        """Sanidade: no caminho `birefnet` a máscara não mudou, então o pareado tem que
        dar exatamente o mesmo número. Se der diferente, algo mexeu no rótulo sem dizer.
        """
        metas = [_meta("b1", "c1", focus_disparity=0.4, base=0.4)]
        gab = {"c1": {"focus_plane_distance_m": 2.0, "focus_plane_uncertainty_m": 0.0}}
        bloco = self._roda(metas, gab)["todas_com_linha_de_base"]
        self.assertEqual(bloco["antes"]["razao_mediana"], bloco["depois"]["razao_mediana"])

    def test_conta_melhorou_piorou_empatou_por_amostra(self):
        gab = {f"c{i}": {"focus_plane_distance_m": 2.0,
                         "focus_plane_uncertainty_m": 0.0} for i in range(1, 4)}
        metas = [
            _meta("m", "c1", focus_disparity=0.50, focus_source="birefnet_refined",
                  base=0.10),                                   # melhorou
            _meta("p", "c2", focus_disparity=0.10, focus_source="birefnet_refined",
                  base=0.50),                                   # piorou
            _meta("e", "c3", focus_disparity=0.30, focus_source="birefnet_refined",
                  base=0.30),                                   # empatou
        ]
        contagem = self._roda(metas, gab)["por_amostra_refinada"]
        self.assertEqual(contagem, {"melhorou": 1, "piorou": 1, "empatou": 1})


class SeparacaoDeCausas(unittest.TestCase):

    def test_gabarito_fora_da_faixa_e_culpa_da_PROFUNDIDADE(self):
        """A cena vai de 1/10 a 1/2 em disparidade e o gabarito pede 1/0,4 = 2,5.

        Nenhuma máscara produziria isso: a mediana de um subconjunto está sempre dentro
        do intervalo do conjunto. Atribuir esta divergência à máscara seria erro de causa.
        """
        metas = [_meta("s1", "c1", focus_disparity=0.5,
                       disparity_min=0.1, disparity_max=0.5)]
        gab = {"c1": {"focus_plane_distance_m": 0.4, "focus_plane_uncertainty_m": 0.0}}
        linhas = avalia(metas, gab)[0]
        self.assertFalse(linhas[0]["gabarito_alcancavel"])
        causas = separacao_de_causas(linhas)
        self.assertEqual(
            causas["gabarito_fora_da_faixa_de_profundidade"]["n_amostras"], 1)

    def test_gabarito_dentro_da_faixa_e_atribuivel_a_mascara(self):
        metas = [_meta("s1", "c1", focus_disparity=0.5,
                       disparity_min=0.1, disparity_max=3.0)]
        gab = {"c1": {"focus_plane_distance_m": 0.4, "focus_plane_uncertainty_m": 0.0}}
        linhas = avalia(metas, gab)[0]
        self.assertTrue(linhas[0]["gabarito_alcancavel"])
        self.assertEqual(separacao_de_causas(linhas)
                         ["gabarito_fora_da_faixa_de_profundidade"]["n_amostras"], 0)

    def test_faixa_ausente_no_metadado_e_None_e_nao_False(self):
        """"Não dá para saber" é diferente de "não alcançável"."""
        meta = _meta("s1", "c1", focus_disparity=0.5)
        del meta["disparity_min"]
        gab = {"c1": {"focus_plane_distance_m": 2.0, "focus_plane_uncertainty_m": 0.0}}
        linhas = avalia([meta], gab)[0]
        self.assertIsNone(linhas[0]["gabarito_alcancavel"])
        self.assertEqual(separacao_de_causas(linhas)["sem_faixa_no_metadado"], 1)

    def test_escala_global_e_a_mediana_das_razoes(self):
        metas = [_meta(f"s{i}", f"c{i}", focus_disparity=0.25) for i in range(3)]
        gab = {f"c{i}": {"focus_plane_distance_m": 2.0,
                         "focus_plane_uncertainty_m": 0.0} for i in range(3)}
        causas = separacao_de_causas(avalia(metas, gab)[0])
        self.assertAlmostEqual(causas["escala_global_estimada"], 0.5, places=6)
        # Corrigida a escala, o mesmo lote passa a concordar — e o relatório avisa que
        # isso mede DISPERSÃO, não acerto.
        corr = causas["com_escala_global_corrigida"]
        self.assertIn("DISPERSÃO", corr["nota"])

    def test_o_relatorio_declara_o_que_NAO_da_para_separar(self):
        metas = [_meta("s1", "c1", focus_disparity=0.5)]
        gab = {"c1": {"focus_plane_distance_m": 2.0, "focus_plane_uncertainty_m": 0.0}}
        texto = separacao_de_causas(avalia(metas, gab)[0])["o_que_NAO_da_para_separar"]
        self.assertIn("Depth Pro", texto)

    def test_gabarito_frouxo_e_isolado(self):
        """`1,92 ± 0,12 m` e `1,92 ± 0,00 m` não autorizam a mesma conclusão."""
        metas = [_meta("s1", "c1", focus_disparity=0.5),
                 _meta("s2", "c2", focus_disparity=0.5)]
        gab = {"c1": {"focus_plane_distance_m": 2.0, "focus_plane_uncertainty_m": 0.0},
               "c2": {"focus_plane_distance_m": 2.0, "focus_plane_uncertainty_m": 1.0}}
        causas = separacao_de_causas(avalia(metas, gab)[0])
        self.assertEqual(causas["gabarito_frouxo"]["n_amostras"], 1)
        self.assertEqual(
            causas["gabarito_frouxo"]["concordancia_sem_elas"]["n_amostras"], 1)


class Veredito(unittest.TestCase):
    """O bloco que decide. Tem que saber dizer que piorou."""

    def _relatorio(self, metas, gab):
        linhas, fora = avalia(metas, gab)
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            relatorio = imprime_relatorio(linhas, fora, quantas=5)
        return relatorio, buffer.getvalue()

    def _gab(self, n):
        return {f"c{i}": {"focus_plane_distance_m": 2.0,
                          "focus_plane_uncertainty_m": 0.0} for i in range(n)}

    def test_diz_MELHOROU_quando_melhorou(self):
        metas = [_meta(f"r{i}", f"c{i}", focus_disparity=0.50,
                       focus_source="birefnet_refined", base=0.05) for i in range(6)]
        relatorio, saida = self._relatorio(metas, self._gab(6))
        self.assertEqual(relatorio["veredito"]["decisao"], "melhorou")
        self.assertGreater(relatorio["veredito"]["delta_pp"], 0)
        self.assertIn("MELHOROU", saida)

    def test_diz_PIOROU_quando_piorou_e_nao_maquia(self):
        """O teste que impede um validador que só sabe elogiar.

        Aqui a máscara inicial acertava o gabarito e o refinamento a estragou. O
        relatório tem que dizer PIOROU, em maiúsculas, e recomendar não congelar nada.
        """
        metas = [_meta(f"r{i}", f"c{i}", focus_disparity=0.05,
                       focus_source="birefnet_refined", base=0.50) for i in range(6)]
        relatorio, saida = self._relatorio(metas, self._gab(6))
        self.assertEqual(relatorio["veredito"]["decisao"], "PIOROU")
        self.assertLess(relatorio["veredito"]["delta_pp"], 0)
        self.assertIn("PIOROU", saida)
        self.assertIn("Não congele nada", saida)

    def test_avisa_quando_mais_amostras_pioram_do_que_melhoram(self):
        """A fração agregada pode subir com poucas correções grandes enquanto a maioria
        das amostras piora um pouco. As duas coisas aparecem."""
        gab = self._gab(5)
        metas = [_meta("bom", "c0", focus_disparity=0.5,
                       focus_source="birefnet_refined", base=0.05)]
        metas += [_meta(f"ruim{i}", f"c{i}", focus_disparity=0.44,
                        focus_source="birefnet_refined", base=0.46)
                  for i in range(1, 5)]
        relatorio, saida = self._relatorio(metas, gab)
        self.assertGreater(relatorio["veredito"]["por_amostra"]["piorou"],
                           relatorio["veredito"]["por_amostra"]["melhorou"])
        self.assertIn("ATENÇÃO", saida)

    def test_sem_refinadas_com_base_nao_conclui_nada(self):
        metas = [_meta("b", "c0", focus_disparity=0.5, base=0.5)]
        relatorio, saida = self._relatorio(metas, self._gab(1))
        self.assertEqual(relatorio["veredito"]["decisao"],
                         "sem_amostras_refinadas_com_linha_de_base")
        self.assertIn("nada a concluir", saida)

    def test_o_baseline_medido_aparece_no_relatorio(self):
        metas = [_meta("r", "c0", focus_disparity=0.5,
                       focus_source="birefnet_refined", base=0.05)]
        relatorio, saida = self._relatorio(metas, self._gab(1))
        self.assertEqual(relatorio["baseline"]["dentro_25pct"], BASELINE_WITHIN_25_PCT)
        self.assertIn("35.2%", saida)

    def test_sem_gabarito_nenhum_o_relatorio_nao_inventa_numero(self):
        relatorio, saida = self._relatorio([_meta("s", "c", focus_disparity=0.5)], {})
        self.assertEqual(relatorio["n_amostras"], 0)
        self.assertIn("Nenhuma amostra tem gabarito", saida)

    def test_maiores_divergencias_sao_simetricas_em_log(self):
        """Razão 4 e razão 1/4 erram o mesmo tanto, em direções opostas. Ordenar por
        `|razão - 1|` poria a primeira na frente e esconderia a segunda."""
        gab = self._gab(3)
        metas = [_meta("longe", "c0", focus_disparity=0.125),      # razão 0,25
                 _meta("perto", "c1", focus_disparity=2.0),        # razão 4,0
                 _meta("certo", "c2", focus_disparity=0.5)]        # razão 1,0
        linhas = avalia(metas, gab)[0]
        piores = maiores_divergencias(linhas, 2)
        self.assertEqual({p["sample_id"] for p in piores}, {"longe", "perto"})

    def test_grava_json_com_a_separacao_por_fonte(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _escreve_piloto(root, [
                _meta("b", "c0", focus_disparity=0.5, base=0.5),
                _meta("t", "c1", focus_disparity=0.5,
                      focus_source="retention_only")])
            metas = carrega_metadados(root)
            self.assertEqual(len(metas), 2)
            linhas, fora = avalia(metas, self._gab(2))
            with redirect_stdout(io.StringIO()):
                relatorio = imprime_relatorio(linhas, fora, quantas=3)
        self.assertEqual(set(relatorio["por_focus_source"]),
                         {"birefnet", "birefnet_refined", "retention_only"})
        self.assertIsNone(relatorio["por_focus_source"]["birefnet_refined"])
        self.assertEqual(relatorio["por_focus_source"]["retention_only"]["n_amostras"], 1)
        json.dumps(relatorio, default=str)         # tem que serializar


class FonteForaDoEnum(unittest.TestCase):
    def test_fonte_desconhecida_aparece_no_relatorio(self):
        """Vocabulário aberto é defeito; um relatório que só imprime os três valores
        conhecidos esconderia um quarto."""
        metas = [_meta("x", "c0", focus_disparity=0.5, focus_source="grabcut")]
        gab = {"c0": {"focus_plane_distance_m": 2.0, "focus_plane_uncertainty_m": 0.0}}
        linhas, fora = avalia(metas, gab)
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            imprime_relatorio(linhas, fora, quantas=3)
        self.assertIn("FORA DO ENUM", buffer.getvalue())


if __name__ == "__main__":
    unittest.main(verbosity=2)
