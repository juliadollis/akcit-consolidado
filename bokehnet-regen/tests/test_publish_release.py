"""Testes do validador de release.

Um validador que nunca reprova é pior que validador nenhum — foi exatamente o caso do
`check_no_leak` antigo, cujo teste chamado "detecta vazamento" afirmava `clean is True`.
Então aqui **cada** checagem tem um teste que a faz REPROVAR, e só depois um que a faz
passar.
"""

from __future__ import annotations

import hashlib
import io
import json
import shutil
import sys
import tempfile
import types
import unittest
from collections import Counter
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

_RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_RAIZ / "src"))
sys.path.insert(0, str(_RAIZ / "scripts"))

import numpy as np                                                    # noqa: E402
from PIL import Image                                                 # noqa: E402

from dataio.split import SceneSplit                                   # noqa: E402
import publish_release                                                # noqa: E402
from publish_release import Problema, escreve_card, publica, valida   # noqa: E402
from publish_release import main                                   # noqa: E402


# --------------------------------------------------------------------------------
# O que cada rota grava — e é POR ISSO que um card só da rota C está errado
# --------------------------------------------------------------------------------
#
# As três diferem em fonte, equações, modelos e artefatos. As fixtures abaixo replicam
# essas diferenças no disco, porque um teste que só monta a rota C não pode detectar um
# card que só descreve a rota C.

#: `(fonte de origem, k_source, focus_source, mask_source, papel do pixel gerado,
#:   campo de sha256 do ledger, unidade do ledger de origem)`.
_ROTAS_NA_FIXTURE = {
    "a": {"fonte": "GenerativePhotography", "k_source": "sampled_from_bc",
          "focus_source": "sampled_plane", "mask_source": "sampled_plane",
          "papel": "bokeh", "campo_sha": "bokeh_jpeg_sha256",
          "unidade_origem": "scene_id"},
    "b": {"fonte": "atfortes/BokehDiffusion", "k_source": "eq3_exif",
          "focus_source": "birefnet", "mask_source": "birefnet",
          "papel": "aif", "campo_sha": "aif_jpeg_sha256",
          "unidade_origem": "sample_id"},
    "c": {"fonte": "akcit-pixel/RealBokeh", "k_source": "eq5_ssim_sweep",
          "focus_source": "birefnet", "mask_source": "birefnet",
          "papel": None, "campo_sha": None, "unidade_origem": "sample_id"},
}


def _meta_da_rota(rota: str, sample_id: str, cena: str) -> dict:
    """O `meta/<id>.json` como a rota o grava — e não como a rota C o grava.

    A rota A é o caso que importa: `focus_was_refined` sai **True** porque
    `FocusRegionRecord.was_refined` é `focus_source is not birefnet`
    (`dataio/sample.py:228`) e `validate_metadata` EXIGE essa coerência
    (`dataio/sample.py:472-476`). O plano de foco foi sorteado; nada foi refinado; e o
    metadado, internamente coerente, afirma o contrário. É esse par que o relatório tem
    que desmontar em vez de repetir.
    """
    perfil = _ROTAS_NA_FIXTURE[rota]
    prov = {"pipeline_commit": "abc123", "depth_model_sha256": "d" * 64,
            "mask_model_sha256": "m" * 64, "image_hw": [1500, 2000],
            "image_h": 1500, "image_w": 2000}
    extra = {}
    if rota == "a":
        # Sem segmentador: a proveniência da máscara é a REGRA, não um checkpoint.
        prov["mask_model_sha256"] = ""
        extra = {"mask_rule": {"banda": "derivada do plano sorteado"}}
        prov["extra"] = extra
    return _meta(
        sample_id, route=rota, scene_id=cena,
        source_dataset=perfil["fonte"], source_sample_id=f"src_{sample_id}",
        k_source=perfil["k_source"],
        focus_source=perfil["focus_source"], mask_source=perfil["mask_source"],
        focus_was_refined=perfil["focus_source"] != "birefnet",
        calibration_ssim=0.91 if rota == "c" else None,
        k_analytic=15.2 if rota == "c" else None,
        k_effective_factor=0.9873 if rota in ("a", "c") else None,
        provenance=prov)


def _sha256_do_arquivo(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _meta(sample_id: str, **over) -> dict:
    base = {
        "sample_id": sample_id, "route": "c", "scene_id": "0001",
        "k_value": 16.6, "k_source": "eq5_ssim_sweep", "focus_disparity": 0.5,
        "max_coc": 100.0, "is_k_censored": False, "is_valid_for_control": True,
        "calibration_ssim": 0.91, "k_analytic": 15.2, "k_effective_factor": 0.9873,
        "mask_source": "birefnet", "depth_backend": "depth_pro",
        # Marcação da região em foco (§3.2(c)). `birefnet` = M confiável, não refinada.
        "focus_source": "birefnet", "focus_was_refined": False,
        "focus_agreement": 0.78, "focus_retention_in_region": 0.88,
        "focus_region_area_ratio": 0.21,
        "focus_retention_h": 384, "focus_retention_w": 512,
        "focus_retention_window_px": 33,
        "focus_initial_mask_was_empty": False,
        "focus_initial_mask_area_ratio": 0.21,
        "focus_disparity_from_initial_mask": 0.5,
        "control_version": "metric_disparity_official_v1",
        "image_h": 1500, "image_w": 2000, "depth_h": 576, "depth_w": 768,
        "depth_encoding": "disparity_uint16",
        "disparity_min": 0.05, "disparity_max": 2.5,
        "source_dataset": "akcit-pixel/RealBokeh", "source_sample_id": f"src_{sample_id}",
        "provenance": {"pipeline_commit": "abc123", "depth_model_sha256": "d" * 64,
                       "mask_model_sha256": "m" * 64, "image_hw": [1500, 2000],
                       "image_h": 1500, "image_w": 2000},
    }
    base.update(over)
    return base


class _Release:
    """Monta um release mínimo e válido no disco, para então quebrá-lo de um jeito."""

    def __init__(self, root: Path, *, n: int = 4, val_scenes: int = 1,
                 rota: str = "c", dedup: bool = False):
        self.root = root
        self.rota = rota
        #: `dedup=True` monta o release no layout `per_scene`: um PNG de profundidade
        #: por CENA, `depth_ref` em cada `meta/` e em cada linha do manifesto, e
        #: `depth_layout` no `run_config.json`. `dedup=False` reproduz o layout dos
        #: releases JÁ PUBLICADOS — sem campo nenhum —, que é o que garante que a
        #: compatibilidade de leitura continua sendo testada e não só afirmada.
        self.dedup = dedup
        perfil = _ROTAS_NA_FIXTURE[rota]
        self.papel = perfil["papel"]
        self.campo_sha = perfil["campo_sha"]
        subdirs = ["depth", "mask", "meta"] + (["generated"] if self.papel else [])
        for sub in subdirs:
            (root / sub).mkdir(parents=True, exist_ok=True)

        self.ids, linhas, ledger, gerado, assignment = [], [], [], [], {}
        self.cenas = []
        for i in range(n):
            cena = f"{i // 2:04d}"
            lado = "val" if i // 2 < val_scenes else "train"
            assignment[cena] = lado
            if cena not in self.cenas:
                self.cenas.append(cena)
            sid = f"{rota}_fonte_{cena}_l{i % 2 + 1}"
            self.ids.append(sid)

            depth_ref = (f"depth/{cena}.png" if dedup else f"depth/{sid}.png")
            Image.fromarray(np.zeros((8, 8), dtype=np.uint16), mode="I;16").save(
                root / depth_ref)
            Image.fromarray(np.zeros((8, 8), dtype=np.uint8), mode="L").save(
                root / "mask" / f"{sid}.png")
            meta = _meta_da_rota(rota, sid, cena)
            if dedup:
                meta["depth_ref"] = depth_ref
                meta["depth_layout"] = "per_scene"
            (root / "meta" / f"{sid}.json").write_text(json.dumps(meta), encoding="utf-8")

            if self.papel:
                # O pixel que ESTA rota gera, e o sha256 dos BYTES EM DISCO — que é o
                # que o ledger promete e o que a validação recalcula.
                caminho = root / "generated" / f"{sid}_{self.papel}.jpg"
                buffer = io.BytesIO()
                Image.fromarray(
                    np.full((8, 8, 3), i * 7 % 256, dtype=np.uint8)).save(
                        buffer, format="JPEG", quality=95)
                caminho.write_bytes(buffer.getvalue())
                gerado.append({"sample_id": sid, "route": rota, "scene_id": cena,
                               self.campo_sha: _sha256_do_arquivo(caminho),
                               f"{self.papel}_jpeg_bytes": caminho.stat().st_size})

            linhas.append({"sample_id": sid, "route": rota, "scene_id": cena,
                           "split": lado, "source_dataset": meta["source_dataset"],
                           **({"depth_ref": depth_ref} if dedup else {}),
                           "source_sample_id": meta["source_sample_id"],
                           "k_value": meta["k_value"], "k_source": meta["k_source"],
                           "focus_disparity": 0.5, "max_coc": 100.0,
                           "is_k_censored": False, "is_valid_for_control": True,
                           "calibration_ssim": meta["calibration_ssim"],
                           "mask_source": meta["mask_source"],
                           "focus_source": meta["focus_source"],
                           "focus_was_refined": meta["focus_was_refined"],
                           "focus_agreement": 0.78,
                           "focus_retention_in_region": 0.88,
                           "focus_region_area_ratio": 0.21,
                           "focus_retention_h": 384, "focus_retention_w": 512,
                           "depth_backend": "depth_pro",
                           "control_version": meta["control_version"]})
            if perfil["unidade_origem"] == "sample_id":
                ledger.append({"sample_id": sid,
                               "source_sample_id": meta["source_sample_id"],
                               "shard": "s00.parquet", "row": i,
                               "aif_sha256": "a" * 64, "bokeh_sha256": "b" * 64})

        if perfil["unidade_origem"] == "scene_id":
            # Ledger POR CENA: as variantes de uma cena saem dos MESMOS bytes de AIF,
            # então gravá-lo por amostra provaria 41 vezes a mesma coisa
            # (`sources/genphoto_ebb.py:565-569`). Não há `sample_id` em linha nenhuma —
            # e era exatamente isso que derrubava a validação com `KeyError`.
            ledger = [{"scene_id": cena, "role": "aif_reference",
                       "source_dataset": perfil["fonte"],
                       "source_sample_id": f"img_{cena}",
                       "aif_ref": f"{perfil['fonte']}#{cena}",
                       "aif_sha256": "a" * 64,
                       "aif_image_h": 1500, "aif_image_w": 2000}
                      for cena in self.cenas]

        self.escreve_manifesto(linhas)
        self.escreve_jsonl("source_images.jsonl", ledger)
        if self.papel:
            self.escreve_jsonl("generated_images.jsonl", gerado)
        SceneSplit(assignment, "teste", val_scenes / max(len(assignment), 1)).save(
            root / "split.json")
        prov = {"pipeline_commit": "abc123", "depth_backend": "depth_pro",
                "split_origin": "sorteado_por_cena"}
        if rota == "a":
            prov.update({"mask_backend": None, "mask_model_sha256": "",
                         "renderer": {"renderer_commit": "beefcafe1234"},
                         "k_effective_factor": 0.9873,
                         "k_distribution_sha256": "k" * 64})
        elif rota == "b":
            prov.update({"mask_backend": "birefnet", "mask_model_sha256": "m" * 64,
                         "renderer": None,
                         "deblurnet": {"deblur_lora_sha256": "l" * 64}})
        else:
            prov.update({"mask_backend": "birefnet", "mask_model_sha256": "m" * 64,
                         "renderer": {"renderer_commit": "beefcafe1234"},
                         "k_effective_factor": 0.9873,
                         "source_dataset": perfil["fonte"]})
        cfg = {"provenance_base": prov}
        if dedup:
            cfg["depth_layout"] = "per_scene"
        (root / "run_config.json").write_text(json.dumps(cfg), encoding="utf-8")

    def escreve_jsonl(self, nome: str, linhas: list[dict]) -> None:
        (self.root / nome).write_text(
            "\n".join(json.dumps(x) for x in linhas) + "\n", encoding="utf-8")

    def gerado(self, sid: str) -> Path:
        return self.root / "generated" / f"{sid}_{self.papel}.jpg"

    def ledger_gerado(self) -> list[dict]:
        return [json.loads(l) for l in
                (self.root / "generated_images.jsonl").read_text(
                    encoding="utf-8").splitlines() if l]

    def escreve_manifesto(self, linhas: list[dict]) -> None:
        (self.root / "manifest.jsonl").write_text(
            "\n".join(json.dumps(x) for x in linhas) + "\n", encoding="utf-8")

    def manifesto(self) -> list[dict]:
        return [json.loads(l) for l in
                (self.root / "manifest.jsonl").read_text(encoding="utf-8").splitlines() if l]


class _Base(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.mkdtemp()
        self.root = Path(self._tmp)
        self.rel = _Release(self.root)

    def tearDown(self):
        shutil.rmtree(self._tmp, ignore_errors=True)

    def assertReprova(self, trecho: str):
        with self.assertRaises(Problema) as ctx:
            valida(self.root)
        self.assertIn(trecho, str(ctx.exception))


class TestPassaQuandoEstaCerto(_Base):

    def test_release_valido_passa(self):
        resumo = valida(self.root)
        self.assertEqual(resumo["amostras"], 4)
        self.assertEqual(resumo["cenas"], 2)
        self.assertEqual(resumo["por_split"], {"val": 2, "train": 2})
        self.assertEqual(resumo["control_version"], "metric_disparity_official_v1")
        self.assertFalse(resumo["autocontido"])

    def test_avisa_que_nao_e_autocontido(self):
        self.assertTrue(any("NÃO autocontido" in a for a in valida(self.root)["avisos"]))

    def test_com_source_vira_autocontido_e_some_o_aviso(self):
        (self.root / "source").mkdir()
        (self.root / "source" / "x_aif.jpg").write_bytes(b"\xff\xd8fake")
        resumo = valida(self.root)
        self.assertTrue(resumo["autocontido"])
        self.assertFalse(any("autocontido" in a for a in resumo["avisos"]))


class TestReprovaQuandoEstaErrado(_Base):

    def test_sem_manifesto(self):
        (self.root / "manifest.jsonl").unlink()
        self.assertReprova("não há release")

    def test_sem_split_materializado(self):
        (self.root / "split.json").unlink()
        self.assertReprova("Split materializado é requisito")

    def test_sample_id_repetido(self):
        linhas = self.rel.manifesto()
        linhas.append(dict(linhas[0]))
        self.rel.escreve_manifesto(linhas)
        self.assertReprova("repetidos no manifesto")

    def test_arquivo_de_depth_faltando(self):
        (self.root / "depth" / f"{self.rel.ids[0]}.png").unlink()
        self.assertReprova("sem arquivo de profundidade alcançável")

    def test_metadado_invalido(self):
        """`max_coc` fora de 100 é o normalizador escondido que quebrou o treino."""
        sid = self.rel.ids[0]
        meta = _meta(sid, scene_id="0000", max_coc=10.510746)
        (self.root / "meta" / f"{sid}.json").write_text(json.dumps(meta), encoding="utf-8")
        self.assertReprova("metadados inválidos")

    def test_control_version_divergente(self):
        sid = self.rel.ids[0]
        meta = _meta(sid, scene_id="0000", control_version="kfix_v0")
        (self.root / "meta" / f"{sid}.json").write_text(json.dumps(meta), encoding="utf-8")
        self.assertReprova("control_version divergente")

    def test_depth_backend_divergente(self):
        sid = self.rel.ids[0]
        meta = _meta(sid, scene_id="0000", depth_backend="depth_anything")
        (self.root / "meta" / f"{sid}.json").write_text(json.dumps(meta), encoding="utf-8")
        self.assertReprova("depth_backend divergente")

    def test_amostra_sem_sha256_de_origem(self):
        (self.root / "source_images.jsonl").unlink()
        self.assertReprova("sem linha em source_images.jsonl")

    def test_cena_fora_do_split(self):
        linhas = self.rel.manifesto()
        linhas[0]["scene_id"] = "9999"
        self.rel.escreve_manifesto(linhas)
        self.assertReprova("fora do split.json")

    def test_vazamento_de_split(self):
        """A mesma cena marcada dos dois lados no manifesto."""
        linhas = self.rel.manifesto()
        linhas[0]["split"] = "train" if linhas[0]["split"] == "val" else "val"
        self.rel.escreve_manifesto(linhas)
        self.assertReprova("vazamento de split")

    def test_censura_acima_de_20_por_cento_reprova(self):
        """47% era o número do release antigo. Acima de 20% o teto está errado."""
        for sid in self.rel.ids[:2]:
            cena = sid.split("_")[2]
            meta = _meta(sid, scene_id=cena, is_k_censored=True)
            (self.root / "meta" / f"{sid}.json").write_text(json.dumps(meta),
                                                            encoding="utf-8")
        self.assertReprova("K censurado no teto")

    def test_censura_baixa_e_so_aviso(self):
        """1 em 20 = 5%: passa, mas o número aparece. Reprovar 5% descartaria dado bom;
        não dizer nada é como 47% passou batido no release anterior."""
        rel = _Release(Path(tempfile.mkdtemp(dir=self._tmp)), n=20, val_scenes=3)
        sid = rel.ids[0]
        meta = _meta(sid, scene_id=sid.split("_")[2], is_k_censored=True)
        (rel.root / "meta" / f"{sid}.json").write_text(json.dumps(meta), encoding="utf-8")
        self.assertTrue(any("censurado" in a for a in valida(rel.root)["avisos"]))

    def test_validacao_vazia_e_aviso(self):
        rel = _Release(Path(tempfile.mkdtemp(dir=self._tmp)), n=4, val_scenes=0)
        self.assertTrue(any("Validação vazia" in a for a in valida(rel.root)["avisos"]))


class TestCard(_Base):

    def test_card_diz_o_essencial(self):
        resumo = valida(self.root)
        card = escreve_card(self.root, resumo, "akcit-pixel/teste").read_text(
            encoding="utf-8")
        for exigido in ("metric_disparity_official_v1", "max_coc", "cenas",
                        "rejections.jsonl", "split.json", "licença da origem",
                        "abc123"):
            with self.subTest(exigido=exigido):
                self.assertIn(exigido, card)

    def test_card_diz_que_os_pixels_estao_fora_quando_estao(self):
        card = escreve_card(self.root, valida(self.root), "x/y").read_text(
            encoding="utf-8")
        self.assertIn("não** estão aqui", card)

    def test_card_muda_quando_o_release_e_autocontido(self):
        (self.root / "source").mkdir()
        (self.root / "source" / "x_aif.jpg").write_bytes(b"\xff\xd8fake")
        card = escreve_card(self.root, valida(self.root), "x/y").read_text(
            encoding="utf-8")
        self.assertIn("bytes originais", card)

    def test_card_diz_de_onde_a_regiao_em_foco_veio(self):
        """Quem baixa o release precisa poder montar o treino COM e SEM as amostras
        refinadas — e saber quantas são antes de baixar 31 GB."""
        resumo = valida(self.root)
        self.assertEqual(resumo["focus_sources"], {"birefnet": len(self.rel.ids)})
        card = escreve_card(self.root, resumo, "x/y").read_text(encoding="utf-8")
        self.assertIn("focus_source", card)
        self.assertIn("35,2%", card)
        self.assertIn("REGIÃO FINAL de foco", card)


class TestMarcacaoDoRefinamento(_Base):

    def _com_fonte(self, fonte: str, mask_source: str):
        for sid in self.rel.ids:
            path = self.root / "meta" / f"{sid}.json"
            meta = json.loads(path.read_text(encoding="utf-8"))
            meta.update({"focus_source": fonte, "focus_was_refined": True,
                         "mask_source": mask_source})
            path.write_text(json.dumps(meta), encoding="utf-8")

    def test_lote_dominado_pelo_refinamento_AVISA(self):
        """Não reprova — o refinamento é o comportamento certo pelo §3.2(c) —, mas um
        release em que ele domina não pode virar treino sem o laudo de validação."""
        self._com_fonte("retention_only", "retention_only")
        resumo = valida(self.root)
        self.assertTrue(any("REFINADA" in a for a in resumo["avisos"]),
                        resumo["avisos"])

    def test_mascara_gravada_que_MENTE_reprova_o_release(self):
        """`focus_source=retention_only` com `mask_source=birefnet` afirma que o arquivo
        em `mask/<id>.png` é a máscara do segmentador. Quem auditasse o rótulo olharia a
        máscara errada."""
        self._com_fonte("retention_only", "birefnet")
        with self.assertRaises(Problema) as ctx:
            valida(self.root)
        self.assertIn("metadados inválidos", str(ctx.exception))

    def test_release_sem_a_marcacao_reprova(self):
        for sid in self.rel.ids:
            path = self.root / "meta" / f"{sid}.json"
            meta = json.loads(path.read_text(encoding="utf-8"))
            del meta["focus_source"]
            path.write_text(json.dumps(meta), encoding="utf-8")
        with self.assertRaises(Problema):
            valida(self.root)


class _FakeApi:
    """Registra as chamadas em vez de falar com o HF, e **guarda o que subiu**.

    Guardar importa: `publica` confere, depois do upload, se a contagem por pasta no
    repositório bate com a do disco — a checagem que faltava quando
    `juliadollis/bokehnet-regen-rota-b` foi publicado com 13.615 `mask/` e 4.688
    `meta/`. Um fake que aceitasse o upload e continuasse listando vazio testaria o
    contrário do que acontece.
    """

    def __init__(self, arquivos: list[str] | None = None, *, engole: bool = False):
        self.arquivos = arquivos
        self.chamadas: list[tuple[str, dict]] = []
        #: `True` simula o Hub aceitando o commit e não guardando os arquivos — a
        #: assinatura do release pela metade.
        self.engole = engole

    def _registra(self, caminhos) -> None:
        if self.engole:
            return
        if self.arquivos is None:
            self.arquivos = []
        self.arquivos.extend(caminhos)

    def list_repo_files(self, repo_id, repo_type):
        if self.arquivos is None:
            raise RuntimeError("RepositoryNotFoundError")
        return list(self.arquivos)

    def create_repo(self, repo_id, repo_type, private):
        self.chamadas.append(("create_repo", {"private": private}))
        if self.arquivos is None:
            self.arquivos = [".gitattributes"]

    def upload_folder(self, **kw):
        self.chamadas.append(("upload_folder", kw))
        raiz = Path(kw["folder_path"])
        self._registra(p.relative_to(raiz).as_posix()
                       for p in publish_release._arquivos(raiz))

    def create_commit(self, **kw):
        self.chamadas.append(("create_commit", kw))
        # O delete também vale: sem aplicá-lo, o fake guardaria para sempre o
        # `depth/antigo.png` do layout plano que o commit acabou de apagar.
        for op in kw["operations"]:
            if hasattr(op, "is_folder") and self.arquivos:
                prefixo = op.path_in_repo + ("/" if op.is_folder else "")
                self.arquivos = [f for f in self.arquivos
                                 if not (f == op.path_in_repo or f.startswith(prefixo))]
        self._registra(op.path_in_repo for op in kw["operations"]
                       if hasattr(op, "path_or_fileobj"))

    @property
    def ops(self) -> list[str]:
        return [nome for nome, _ in self.chamadas]


class TestPublica(_Base):
    """O upload propriamente dito: `huggingface_hub` é falsificado, então nada sai daqui.

    Existe porque as duas promessas do `publica` — **privado por default** e **não
    sobrescreve release existente** — não tinham teste nenhum, e porque o commit único
    foi recusado com **413** num release de 46 mil arquivos: a escolha de estratégia
    passou a ser comportamento, e comportamento sem teste é intenção.
    """

    def setUp(self):
        super().setUp()
        self._hub_salvo = sys.modules.get("huggingface_hub")
        # Os limites de verdade são 8.000 e 10.000; recriá-los em disco em cada teste
        # custaria 30 mil arquivos por teste para exercer a MESMA aritmética.
        self._constantes_salvas = {
            n: getattr(publish_release, n)
            for n in ("LIMITE_COMMIT_UNICO_ARQUIVOS", "MAX_ARQUIVOS_POR_PASTA",
                      "LOTE_POR_COMMIT")}
        publish_release.LIMITE_COMMIT_UNICO_ARQUIVOS = 8
        publish_release.MAX_ARQUIVOS_POR_PASTA = 6
        publish_release.LOTE_POR_COMMIT = 5

    def tearDown(self):
        for nome, valor in self._constantes_salvas.items():
            setattr(publish_release, nome, valor)
        if self._hub_salvo is None:
            sys.modules.pop("huggingface_hub", None)
        else:
            sys.modules["huggingface_hub"] = self._hub_salvo
        super().tearDown()

    def _instala(self, api: _FakeApi) -> _FakeApi:
        modulo = types.ModuleType("huggingface_hub")
        modulo.HfApi = lambda: api                      # type: ignore[attr-defined]

        class CommitOperationAdd:                       # o de verdade é um dataclass
            def __init__(self, path_in_repo, path_or_fileobj):
                self.path_in_repo = path_in_repo
                self.path_or_fileobj = path_or_fileobj

        class CommitOperationDelete:
            def __init__(self, path_in_repo, is_folder=False):
                self.path_in_repo = path_in_repo
                self.is_folder = is_folder

        modulo.CommitOperationAdd = CommitOperationAdd   # type: ignore[attr-defined]
        modulo.CommitOperationDelete = CommitOperationDelete  # type: ignore

        constantes = types.ModuleType("huggingface_hub.constants")
        constantes.DEFAULT_REQUEST_TIMEOUT = 10          # o default de verdade
        modulo.constants = constantes                    # type: ignore[attr-defined]
        self.constantes = constantes
        sys.modules["huggingface_hub"] = modulo
        return api

    def test_eleva_o_timeout_de_10s_antes_do_primeiro_request(self):
        """O default de 10 s do `huggingface_hub` não cabe um commit de milhares de
        arquivos: ele estoura em ReadTimeout e deixa o release pela metade."""
        self._instala(_FakeApi(arquivos=None))
        publica(self.root, "x/y", private=True, allow_existing=False)
        self.assertEqual(self.constantes.DEFAULT_REQUEST_TIMEOUT,
                         publish_release.TIMEOUT_REQUEST_S)

    def test_retenta_no_429_e_no_timeout_e_so_neles(self):
        self.assertTrue(publish_release._retentavel(
            RuntimeError("Client error '429 Too Many Requests' for url …")))
        self.assertTrue(publish_release._retentavel(
            TimeoutError("The read operation timed out")))
        self.assertFalse(publish_release._retentavel(
            RuntimeError("Client error '413 Payload Too Large' for url …")))
        self.assertFalse(publish_release._retentavel(RuntimeError("401 Unauthorized")))

    def test_cria_repo_privado_e_release_pequeno_vai_em_um_commit(self):
        """Privado é o default, e release que cabe num commit vai num commit — atômico:
        ou aparece inteiro, ou não aparece."""
        publish_release.LIMITE_COMMIT_UNICO_ARQUIVOS = 10_000
        publish_release.MAX_ARQUIVOS_POR_PASTA = 10_000
        api = self._instala(_FakeApi(arquivos=None))
        publica(self.root, "x/y", private=True, allow_existing=False)
        self.assertEqual(api.ops, ["create_repo", "upload_folder"])
        self.assertIs(api.chamadas[0][1]["private"], True)

    def test_recusa_sobrescrever_repo_que_ja_tem_release(self):
        api = self._instala(_FakeApi(arquivos=[".gitattributes", "manifest.jsonl"]))
        with self.assertRaises(Problema) as ctx:
            publica(self.root, "x/y", private=True, allow_existing=False)
        self.assertIn("já tem 1 arquivos de release", str(ctx.exception))
        self.assertEqual(api.ops, [])

    def test_gitattributes_sozinho_nao_tranca_a_retentativa(self):
        """O `.gitattributes` é do `create_repo`, não do release. Contá-lo impedia
        retomar exatamente depois de um upload que caiu no meio."""
        api = self._instala(_FakeApi(arquivos=[".gitattributes"]))
        publica(self.root, "x/y", private=True, allow_existing=False)
        self.assertNotIn("create_repo", api.ops, "o repo já existia")
        self.assertTrue(api.ops, "não subiu nada")

    def test_release_grande_vai_em_lotes_grossos_e_cobre_tudo(self):
        """Os dois limites do Hub batem em sentidos opostos: um commit com tudo devolve
        **413**, e um commit por punhado de arquivos devolve **429** (256 commits/hora).
        Então o lote tem que ser grosso e o número de commits, pequeno — e nenhum
        arquivo pode ficar de fora do fatiamento."""
        api = self._instala(_FakeApi(arquivos=None))
        alvo = self.root / "depth"
        for i in range(publish_release.LIMITE_COMMIT_UNICO_ARQUIVOS + 1):
            (alvo / f"enche_{i}.png").write_bytes(b"x")

        publica(self.root, "x/y", private=False, allow_existing=False)
        self.assertIs(api.chamadas[0][1]["private"], False)

        commits = [kw for nome, kw in api.chamadas if nome == "create_commit"]
        esperados = publish_release._tamanho(self.root)[0]
        lote = publish_release.LOTE_POR_COMMIT
        self.assertEqual(len(commits), -(-esperados // lote),
                         "lote grosso: um commit por lote cheio, não centenas")
        for kw in commits:
            self.assertLessEqual(len(kw["operations"]),
                                 publish_release.LOTE_POR_COMMIT)
        enviados = [op.path_in_repo for kw in commits for op in kw["operations"]]
        self.assertEqual(len(enviados), esperados)
        self.assertEqual(len(set(enviados)), esperados, "arquivo em dois commits")
        self.assertIn("manifest.jsonl", enviados)

    def _estoura_a_pasta(self) -> None:
        """Passa `depth/` do limite de 10.000 arquivos por diretório do Hub, com
        arquivos que o manifesto conhece — é o manifesto que dá a cena de cada um."""
        linhas = self.rel.manifesto()
        modelo = linhas[0]
        for i in range(publish_release.MAX_ARQUIVOS_POR_PASTA + 1):
            sid, cena = f"enchendo_{i}", f"c{i % 7:02d}"
            for sub, ext in (("depth", ".png"), ("mask", ".png"), ("meta", ".json")):
                (self.root / sub / f"{sid}{ext}").write_bytes(b"x")
            linhas.append({**modelo, "sample_id": sid, "scene_id": cena})
        self.rel.escreve_manifesto(linhas)

    def test_pasta_estourada_vai_agrupada_por_cena(self):
        """O Hub recusa o push (400) de qualquer diretório com mais de 10.000 arquivos,
        e `depth/`, `mask/` e `meta/` têm 15.423 cada neste release. O agrupamento é por
        cena porque `scene_id` já está no manifesto: o caminho se remonta lendo o campo,
        sem hash nenhum."""
        api = self._instala(_FakeApi(arquivos=None))
        self._estoura_a_pasta()
        self.assertTrue(publish_release._agrupa_por_cena(self.root))

        publica(self.root, "x/y", private=True, allow_existing=False)
        enviados = [op.path_in_repo
                    for nome, kw in api.chamadas if nome == "create_commit"
                    for op in kw["operations"]]

        por_pasta = Counter(d.rsplit("/", 1)[0] for d in enviados)
        self.assertLessEqual(max(por_pasta.values()),
                             publish_release.MAX_ARQUIVOS_POR_PASTA)
        sid, cena = self.rel.ids[0], "0000"
        self.assertIn(f"depth/{cena}/{sid}.png", enviados)
        self.assertIn(f"meta/{cena}/{sid}.json", enviados)
        self.assertIn("manifest.jsonl", enviados, "arquivo de raiz não se move")

    def test_depth_deduplicado_fica_PLANO_enquanto_as_outras_sao_reagrupadas(self):
        """O reagrupamento passou a ser POR PASTA, e é a dedup que obriga.

        Antes era um booleano para o release inteiro: com `mask/` e `meta/` acima do
        limite, `depth/` era reagrupado junto e `depth/<scene_id>.png` virava
        `depth/<scene_id>/<scene_id>.png` — quebrando o `depth_ref` que cada `meta/`
        gravou, num release em que `depth/` cabe folgado (1.700 arquivos contra os
        10.000 do limite).
        """
        api = self._instala(_FakeApi(arquivos=None))
        linhas = self.rel.manifesto()
        modelo = linhas[0]
        # Só `mask/` e `meta/` estouram; `depth/` continua com um arquivo por cena.
        for i in range(publish_release.MAX_ARQUIVOS_POR_PASTA + 1):
            # duas cenas só: `depth/` fica em 6 arquivos, dentro do limite do teste
            sid, cena = f"enchendo_{i}", f"c{i % 2:02d}"
            (self.root / "depth" / f"{cena}.png").write_bytes(b"x")
            for sub, ext in (("mask", ".png"), ("meta", ".json")):
                (self.root / sub / f"{sid}{ext}").write_bytes(b"x")
            linhas.append({**modelo, "sample_id": sid, "scene_id": cena,
                           "depth_ref": f"depth/{cena}.png"})
        self.rel.escreve_manifesto(linhas)

        self.assertEqual(publish_release._agrupa_por_cena(self.root),
                         frozenset({"mask", "meta"}))
        publica(self.root, "x/y", private=True, allow_existing=False)
        enviados = [op.path_in_repo
                    for nome, kw in api.chamadas if nome == "create_commit"
                    for op in kw["operations"]]
        self.assertIn("depth/c00.png", enviados, "depth/ não devia ganhar nível")
        self.assertIn(f"mask/0000/{self.rel.ids[0]}.png", enviados,
                      "mask/ estourou e tinha que ser reagrupada")

    def test_o_card_descreve_o_layout_que_sera_publicado(self):
        """Card dizendo `depth/<id>.png` num repositório agrupado por cena manda quem
        baixa procurar num caminho que não existe."""
        resumo = valida(self.root)
        plano = escreve_card(self.root, resumo, "x/y").read_text(encoding="utf-8")
        self.assertIn("depth/<id>.png", plano)

        self._estoura_a_pasta()
        agrupado = escreve_card(self.root, resumo, "x/y").read_text(
            encoding="utf-8")
        self.assertIn("depth/<scene_id>/<id>.png", agrupado)
        self.assertIn("10.000 arquivos", agrupado)

    def test_apaga_o_layout_plano_de_uma_tentativa_anterior(self):
        """Um upload plano interrompido deixa `depth/<id>.png` para trás; publicar por
        cima sem apagar deixaria cada amostra duas vezes no repositório."""
        api = self._instala(_FakeApi(arquivos=[".gitattributes", "depth/antigo.png"]))
        self._estoura_a_pasta()
        publica(self.root, "x/y", private=True, allow_existing=True)
        primeiro = api.chamadas[0]
        self.assertEqual(primeiro[0], "create_commit")
        self.assertEqual([op.path_in_repo for op in primeiro[1]["operations"]],
                         ["depth"])

    def test_release_grande_nao_usa_upload_folder(self):
        api = self._instala(_FakeApi(arquivos=None))
        for i in range(publish_release.LIMITE_COMMIT_UNICO_ARQUIVOS + 1):
            (self.root / "depth" / f"enche_{i}.png").write_bytes(b"x")
        publica(self.root, "x/y", private=True, allow_existing=False)
        self.assertNotIn("upload_folder", api.ops)

    def test_o_cache_do_retomar_nao_conta_como_release(self):
        cache = self.root / ".cache" / "huggingface"
        cache.mkdir(parents=True)
        (cache / "diario").write_bytes(b"y" * 10)
        n_com_cache, _ = publish_release._tamanho(self.root)
        (cache / "diario").unlink()
        n_sem_cache, _ = publish_release._tamanho(self.root)
        self.assertEqual(n_com_cache, n_sem_cache)


# --------------------------------------------------------------------------------
# Defeito 1 — o card era FIXO na rota C, e já saiu errado no Hub
# --------------------------------------------------------------------------------

class _BaseDeRota(unittest.TestCase):
    """Monta um release de uma rota qualquer, não só da C."""

    ROTA = "c"

    def setUp(self):
        self._tmp = tempfile.mkdtemp()
        self.root = Path(self._tmp)
        self.rel = _Release(self.root, rota=self.ROTA)

    def tearDown(self):
        shutil.rmtree(self._tmp, ignore_errors=True)

    def card(self) -> str:
        return escreve_card(self.root, valida(self.root), "x/y").read_text(
            encoding="utf-8")


class TestCardPorRota(unittest.TestCase):
    """O card não pode citar artefato nem modelo que a rota não usa.

    Este é o teste do defeito que **já foi publicado**: `bokehnet-regen-rota-b` e
    `bokehnet-regen-rota-b-oficial` estão no Hub com `pretty_name: BokehNet regen —
    rota C` e com o corpo descrevendo RealBokeh, BiRefNet e o sweep da Eq. 5 — nenhum
    dos três existe na rota B. O card vinha de uma f-string com o texto da rota C
    escrito à mão; só os números saíam do release.

    Para cada rota: o que o card **tem** que dizer, e o que ele **não pode** dizer.
    """

    #: Termos que, aparecendo no card daquela rota, provam que ele descreve outra.
    PROIBIDOS = {
        # A rota A não tem segmentador (o plano de foco é SORTEADO), não tem DeblurNet
        # (não há AIF a produzir: a AIF é a foto de origem), não aplica Eq. 3/4/5, e
        # não vê RealBokeh, LFDOF nem o ITW.
        "a": ("BiRefNet", "DeblurNet", "Eq. 5", "Eq. 3", "Eq. 4", "RealBokeh",
              "LFDOF", "ITW", "calibration_ssim", "sweep", "rota C", "rota B"),
        # A rota B não tem renderizador nenhum (paper.txt:271,283,292,321), logo não
        # tem BokehMe, não tem Eq. 5 e não tem K censurado no teto de busca nenhum.
        "b": ("BokehMe", "Eq. 5", "RealBokeh", "LFDOF", "GenerativePhotography",
              "EBB!", "refinada", "rota A", "rota C"),
        # A rota C não gera pixel: ela rotula pares que já existem. E não usa DeblurNet.
        # `sorteado` sozinho não serve como termo proibido: `split_origin` da rota C é
        # literalmente `sorteado_por_cena`, e esse é um valor MEDIDO e correto. Os
        # termos abaixo são os do sorteio de K e do plano de foco, que só a rota A faz.
        "c": ("DeblurNet", "generated/", "generated_images.jsonl",
              "sampled_from_bc", "hipercubo latino", "plano de foco é sorteado",
              "rota A", "rota B"),
    }

    #: O que o card daquela rota tem que dizer — o outro lado do mesmo defeito: o card
    #: da rota C, reusado na rota A, OMITIA `generated/`, que é onde vive o alvo.
    EXIGIDOS = {
        "a": ("rota A", "§3.2(a)", "BokehMe", "sorteado", "generated/",
              "generated_images.jsonl", "GenerativePhotography"),
        "b": ("rota B", "§3.2(b)", "DeblurNet", "BiRefNet", "Eq. 3",
              "generated/", "atfortes/BokehDiffusion"),
        "c": ("rota C", "§3.2(c)", "BiRefNet", "Eq. 5", "akcit-pixel/RealBokeh"),
    }

    def _card(self, rota: str) -> str:
        tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmp, True)
        raiz = Path(tmp)
        _Release(raiz, rota=rota)
        return escreve_card(raiz, valida(raiz), f"juliadollis/bokehnet-regen-{rota}"
                            ).read_text(encoding="utf-8")

    def test_rota_a(self):
        self._confere("a")

    def test_rota_b(self):
        self._confere("b")

    def test_rota_c(self):
        self._confere("c")

    def _confere(self, rota: str) -> None:
        card = self._card(rota)
        self.assertIn(f"pretty_name: BokehNet regen — rota {rota.upper()}", card)
        for termo in self.PROIBIDOS[rota]:
            with self.subTest(rota=rota, proibido=termo):
                self.assertNotIn(
                    termo, card,
                    f"o card da rota {rota.upper()} cita {termo!r}, que essa rota não "
                    "usa — é exatamente o defeito publicado em bokehnet-regen-rota-b")
        for termo in self.EXIGIDOS[rota]:
            with self.subTest(rota=rota, exigido=termo):
                self.assertIn(termo, card)

    def test_a_licenca_sai_das_fontes_do_manifesto(self):
        """`provenance_base` da rota A não tem `source_dataset`, e o card imprimia
        "Os pixels seguem a licença da origem (`(não registrada)`)"."""
        card = self._card("a")
        self.assertNotIn("(não registrada)", card)
        self.assertIn("GenerativePhotography", card.split("## Licença")[-1])

    def test_nenhuma_rota_publica_placeholder(self):
        for rota in ("a", "b", "c"):
            with self.subTest(rota=rota):
                card = self._card(rota)
                self.assertNotIn("(não registrada)", card)
                self.assertNotIn("| `?`", card)
                self.assertNotIn("BokehMe `?`", card)

    def test_release_com_duas_rotas_nao_vira_card(self):
        """Um card descreve UMA rota. Misturar duas afirmaria de cada amostra o que só
        vale para metade — que é a forma geral do defeito publicado."""
        tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmp, True)
        raiz = Path(tmp)
        rel = _Release(raiz, rota="c")
        linhas = rel.manifesto()
        linhas[0]["route"] = "a"
        rel.escreve_manifesto(linhas)
        with self.assertRaises(Problema) as ctx:
            valida(raiz)
        self.assertIn("rotas", str(ctx.exception))


# --------------------------------------------------------------------------------
# Defeito 2 — `KeyError: 'sample_id'` derrubava a validação da rota A
# --------------------------------------------------------------------------------

class TestLedgerDeOrigemPorRota(unittest.TestCase):
    """Cada ledger é lido pela unidade em que a rota o escreve.

    O `source_images.jsonl` da rota A é por **cena** — as variantes saem dos mesmos
    bytes de AIF (`sources/genphoto_ebb.py:565-569`) — e não tem `sample_id` em linha
    nenhuma. `publish_release.py:138` indexava os dois ledgers por `sample_id`, e a
    validação da rota A morria com `KeyError: 'sample_id'` antes de conferir coisa
    alguma. A rota B nunca viu o defeito só porque não escrevia o arquivo: sem arquivo,
    `_linhas` devolve `[]` e o laço nunca itera.
    """

    def setUp(self):
        self._tmp = tempfile.mkdtemp()
        self.root = Path(self._tmp)
        self.rel = _Release(self.root, rota="a")

    def tearDown(self):
        shutil.rmtree(self._tmp, ignore_errors=True)

    def test_a_rota_a_valida_com_ledger_por_cena(self):
        """O teste do `KeyError`: antes do conserto isto explodia com uma exceção NÃO
        tratada — nem o relatório parcial saía."""
        ledger = json.loads(
            (self.root / "source_images.jsonl").read_text(
                encoding="utf-8").splitlines()[0])
        self.assertNotIn("sample_id", ledger, "a fixture tem que ser o ledger por cena")
        resumo = valida(self.root)
        self.assertEqual(resumo["rota"], "a")
        self.assertEqual(resumo["amostras"], 4)

    def test_cena_fora_do_ledger_de_origem_reprova(self):
        """Ler sem quebrar não basta: a validação tem que MEDIR alguma coisa."""
        linhas = [json.loads(l) for l in
                  (self.root / "source_images.jsonl").read_text(
                      encoding="utf-8").splitlines() if l]
        self.rel.escreve_jsonl("source_images.jsonl", linhas[:-1])
        with self.assertRaises(Problema) as ctx:
            valida(self.root)
        self.assertIn("scene_id da rota A sem linha em source_images.jsonl",
                      str(ctx.exception))

    def test_ledger_da_outra_rota_e_denunciado_nao_ignorado(self):
        """Um `source_images.jsonl` por amostra num release da rota A é o ledger de
        outra rota. Pular a linha com `if "scene_id" in l` silenciaria o problema; o
        conserto conta e reporta."""
        self.rel.escreve_jsonl("source_images.jsonl",
                               [{"sample_id": sid, "aif_sha256": "a" * 64}
                                for sid in self.rel.ids])
        with self.assertRaises(Problema) as ctx:
            valida(self.root)
        self.assertIn("sem 'scene_id'", str(ctx.exception))

    def test_rota_c_continua_sendo_lida_por_amostra(self):
        tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmp, True)
        raiz = Path(tmp)
        rel = _Release(raiz, rota="c")
        self.assertEqual(valida(raiz)["rota"], "c")
        rel.escreve_jsonl("source_images.jsonl", [])
        with self.assertRaises(Problema) as ctx:
            valida(raiz)
        self.assertIn("sample_id da rota C sem linha em source_images.jsonl",
                      str(ctx.exception))


# --------------------------------------------------------------------------------
# Defeito 3 — a rota A passava sem checagem nenhuma de pixel
# --------------------------------------------------------------------------------

class TestPixelGerado(unittest.TestCase):
    """O pixel que a rota gera existe, não é vazio, e é o que o ledger diz.

    Na rota A o `generated/<id>_bokeh.jpg` é **o que a rede aprende a produzir**, e a
    validação percorria só `depth/`, `mask/` e `meta/`: um release com 10.000 bokehs
    faltando — ou de 0 byte — passava com zero problemas.
    """

    def setUp(self):
        self._tmp = tempfile.mkdtemp()
        self.root = Path(self._tmp)
        self.rel = _Release(self.root, rota="a")

    def tearDown(self):
        shutil.rmtree(self._tmp, ignore_errors=True)

    def test_release_completo_passa(self):
        resumo = valida(self.root)
        self.assertEqual(resumo["pixel_gerado_sha256_conferidos"], 4)

    def test_arquivo_ausente_em_generated_reprova(self):
        self.rel.gerado(self.rel.ids[0]).unlink()
        with self.assertRaises(Problema) as ctx:
            valida(self.root)
        self.assertIn("sem arquivo em generated/", str(ctx.exception))
        self.assertIn("o ALVO do treino", str(ctx.exception))

    def test_jpeg_de_zero_byte_reprova(self):
        """Escrita interrompida no meio. A contagem de arquivos não denuncia."""
        self.rel.gerado(self.rel.ids[1]).write_bytes(b"")
        with self.assertRaises(Problema) as ctx:
            valida(self.root)
        self.assertIn("0 byte em generated/", str(ctx.exception))

    def test_sha256_que_nao_bate_com_o_disco_reprova(self):
        """O ledger e o release descrevendo imagens diferentes."""
        alvo = self.rel.gerado(self.rel.ids[2])
        alvo.write_bytes(alvo.read_bytes() + b"\x00outro")
        with self.assertRaises(Problema) as ctx:
            valida(self.root)
        self.assertIn("DIFERENTE dos bytes em disco", str(ctx.exception))

    def test_amostra_sem_linha_no_ledger_de_gerados_reprova(self):
        linhas = self.rel.ledger_gerado()
        self.rel.escreve_jsonl("generated_images.jsonl", linhas[:-1])
        with self.assertRaises(Problema) as ctx:
            valida(self.root)
        self.assertIn("sem linha em generated_images.jsonl", str(ctx.exception))

    def test_sha256_nulo_no_ledger_reprova(self):
        """O ledger existe e não prova nada — o desfecho de um `generated/` escrito
        depois da linha do ledger."""
        linhas = self.rel.ledger_gerado()
        linhas[0]["bokeh_jpeg_sha256"] = None
        self.rel.escreve_jsonl("generated_images.jsonl", linhas)
        with self.assertRaises(Problema) as ctx:
            valida(self.root)
        self.assertIn("nulo", str(ctx.exception))

    def test_o_tamanho_da_amostragem_vai_no_relatorio(self):
        """Amostragem cujo `n` não aparece é indistinguível de nenhuma amostragem."""
        publish_release.AMOSTRAGEM_SHA256_PIXEL = 2
        self.addCleanup(setattr, publish_release, "AMOSTRAGEM_SHA256_PIXEL", 256)
        resumo = valida(self.root)
        self.assertEqual(resumo["pixel_gerado_sha256_conferidos"], 2)
        card = escreve_card(self.root, resumo, "x/y").read_text(encoding="utf-8")
        self.assertIn("reconferidos byte a byte", card)

    def test_a_amostragem_e_deterministica_e_nao_e_o_prefixo_do_manifesto(self):
        publish_release.AMOSTRAGEM_SHA256_PIXEL = 2
        self.addCleanup(setattr, publish_release, "AMOSTRAGEM_SHA256_PIXEL", 256)
        ids = [f"a_fonte_{i:04d}_l1" for i in range(50)]
        uma = publish_release._amostragem(ids, 2)
        outra = publish_release._amostragem(list(reversed(ids)), 2)
        self.assertEqual(uma, outra, "a escolha não pode depender da ordem")
        self.assertNotEqual(uma, ids[:2], "fatiar o começo é amostrar seis cenas")

    def test_rota_b_tambem_e_conferida_porque_tambem_gera_pixel(self):
        """A rota B gera a AIF — a ENTRADA da rede, não o alvo. O pixel é nosso do mesmo
        jeito, e o sha256 dele prova o que o release contém."""
        tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmp, True)
        raiz = Path(tmp)
        rel = _Release(raiz, rota="b")
        self.assertEqual(valida(raiz)["pixel_gerado_sha256_conferidos"], 4)
        rel.gerado(rel.ids[0]).unlink()
        with self.assertRaises(Problema) as ctx:
            valida(raiz)
        self.assertIn("a ENTRADA da rede", str(ctx.exception))

    def test_rota_c_nao_exige_generated_porque_nao_gera_pixel(self):
        """Exigir `generated/` da rota C reprovaria um release correto: ela rotula
        pares que já existem."""
        tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmp, True)
        raiz = Path(tmp)
        _Release(raiz, rota="c")
        resumo = valida(raiz)
        self.assertEqual(resumo["pixel_gerado_sha256_conferidos"], 0)
        self.assertFalse((raiz / "generated").exists())


class TestContagemPorDiretorio(unittest.TestCase):
    """As pastas por amostra têm que ter a MESMA contagem.

    `juliadollis/bokehnet-regen-rota-b` está publicado com **13.615 arquivos em `mask/`
    e 4.688 em `meta/`** — faltam ~8.900 `meta/`, que é onde vivem K, profundidade,
    plano de foco e toda a proveniência. Sem `meta/` o dataset não treina. Nada
    comparava as pastas entre si, e foi por essa fresta que o upload passou.
    """

    def setUp(self):
        self._tmp = tempfile.mkdtemp()
        self.root = Path(self._tmp)
        self.rel = _Release(self.root, rota="c", n=6, val_scenes=1)

    def tearDown(self):
        shutil.rmtree(self._tmp, ignore_errors=True)

    def test_meta_curto_reprova_dizendo_qual_pasta_e_quantos(self):
        for sid in self.rel.ids[:2]:
            (self.root / "meta" / f"{sid}.json").unlink()
        with self.assertRaises(Problema) as ctx:
            valida(self.root)
        recado = str(ctx.exception)
        self.assertIn("meta/ tem 4 arquivos para 6 amostras", recado)
        self.assertIn("faltam 2", recado)
        self.assertIn("'mask': 6", recado, "a mensagem tem que mostrar as outras pastas")

    def test_pasta_com_arquivo_a_mais_tambem_reprova(self):
        """Resto de outro run sobe junto e vira amostra fantasma no repositório."""
        (self.root / "mask" / "sobra_de_outro_run.png").write_bytes(b"x")
        with self.assertRaises(Problema) as ctx:
            valida(self.root)
        self.assertIn("1 a mais", str(ctx.exception))

    def test_generated_curto_reprova_na_rota_que_gera_pixel(self):
        tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmp, True)
        raiz = Path(tmp)
        rel = _Release(raiz, rota="a", n=6, val_scenes=1)
        rel.gerado(rel.ids[0]).unlink()
        with self.assertRaises(Problema) as ctx:
            valida(raiz)
        self.assertIn("generated/ tem 5 arquivos para 6 amostras", str(ctx.exception))


class TestRefinamentoNaoInventado(unittest.TestCase):
    """`refined N de N` era falso positivo na rota A.

    `FocusRegionRecord.was_refined` é `focus_source is not birefnet`
    (`dataio/sample.py:228`), então o plano de foco **sorteado** sai marcado como
    refinado — e `validate_metadata` exige essa coerência, de modo que o metadado está
    internamente correto e mesmo assim afirma o que não houve. O relatório passou a
    contar pela FONTE e a denunciar a divergência.
    """

    def setUp(self):
        self._tmp = tempfile.mkdtemp()
        self.root = Path(self._tmp)
        self.rel = _Release(self.root, rota="a")

    def tearDown(self):
        shutil.rmtree(self._tmp, ignore_errors=True)

    def test_sorteio_nao_conta_como_refinamento(self):
        resumo = valida(self.root)
        self.assertEqual(resumo["focus_sources"], {"sampled_plane": 4})
        self.assertEqual(resumo["refined"], 0, "nada foi refinado: o plano foi SORTEADO")
        self.assertEqual(resumo["marcadas_como_refinadas"], 4)

    def test_a_divergencia_do_booleano_vira_aviso(self):
        avisos = valida(self.root)["avisos"]
        self.assertTrue(any("focus_was_refined" in a and "focus_source" in a
                            for a in avisos), avisos)

    def test_nao_pede_o_laudo_de_refinamento_onde_nao_houve_refinamento(self):
        avisos = valida(self.root)["avisos"]
        self.assertFalse(any("REFINADA" in a for a in avisos), avisos)

    def test_o_card_da_rota_a_nao_tem_secao_de_refinamento(self):
        card = escreve_card(self.root, valida(self.root), "x/y").read_text(
            encoding="utf-8")
        self.assertNotIn("refinamento da região em foco", card)
        self.assertNotIn("região em foco refinada", card)

    def test_a_rota_c_continua_contando_o_refinamento_de_verdade(self):
        tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmp, True)
        raiz = Path(tmp)
        rel = _Release(raiz, rota="c")
        for sid in rel.ids:
            path = raiz / "meta" / f"{sid}.json"
            meta = json.loads(path.read_text(encoding="utf-8"))
            meta.update({"focus_source": "retention_only", "focus_was_refined": True,
                         "mask_source": "retention_only"})
            path.write_text(json.dumps(meta), encoding="utf-8")
        resumo = valida(raiz)
        self.assertEqual(resumo["refined"], 4)
        self.assertEqual(resumo["marcadas_como_refinadas"], 4)
        self.assertFalse(any("focus_was_refined" in a for a in resumo["avisos"]))


# --------------------------------------------------------------------------------
# A propriedade de segurança: nada sobe sem `--yes`
# --------------------------------------------------------------------------------

class TestMainNaoPublicaSozinho(unittest.TestCase):
    """É a única coisa entre um bug e uma publicação errada no Hub.

    `publica()` é a única função que fala com o Hub. Aqui ela é substituída por um
    espião, então **nenhum request sai** nem quando o teste passa `--yes`: o que se
    mede é se `main()` a chamaria.
    """

    def setUp(self):
        self._tmp = tempfile.mkdtemp()
        self.root = Path(self._tmp)
        self.rel = _Release(self.root, rota="c")
        self.chamadas = []
        patch = mock.patch.object(
            publish_release, "publica",
            lambda *a, **kw: self.chamadas.append((a, kw)))
        patch.start()
        self.addCleanup(patch.stop)

    def tearDown(self):
        shutil.rmtree(self._tmp, ignore_errors=True)

    def _main(self, *args: str) -> int:
        with mock.patch.object(sys, "argv",
                               ["publish_release.py", "--release-dir", str(self.root),
                                *args]), redirect_stdout(io.StringIO()):
            return main()

    def test_sem_yes_nao_chama_publica(self):
        self.assertEqual(self._main(), 0)
        self.assertEqual(self.chamadas, [])

    def test_sem_yes_com_repo_id_nao_chama_publica(self):
        self.assertEqual(self._main("--repo-id", "juliadollis/qualquer"), 0)
        self.assertEqual(self.chamadas, [])

    def test_sem_repo_id_nao_escreve_card(self):
        """Sem destino não há card: o card do release anterior ficaria no disco e
        seguiria para o próximo upload."""
        self._main()
        self.assertFalse((self.root / "README.md").exists())

    def test_yes_sem_repo_id_nao_publica_e_sai_com_erro(self):
        self.assertEqual(self._main("--yes"), 1)
        self.assertEqual(self.chamadas, [])

    def test_release_reprovado_nao_publica_nem_com_yes(self):
        (self.root / "split.json").unlink()
        self.assertEqual(self._main("--yes", "--repo-id", "x/y"), 1)
        self.assertEqual(self.chamadas, [])

    def test_so_com_yes_E_repo_id_e_que_publica(self):
        self.assertEqual(self._main("--yes", "--repo-id", "x/y"), 0)
        self.assertEqual(len(self.chamadas), 1)
        self.assertIs(self.chamadas[0][1]["private"], True, "privado por default")

    def test_privado_por_default_e_publico_so_com_a_flag(self):
        self._main("--yes", "--repo-id", "x/y", "--public")
        self.assertIs(self.chamadas[0][1]["private"], False)


class TestConfereOQueSubiu(_Base):
    """Depois do upload: o repositório tem o que saiu daqui?

    É o mesmo `list_repo_files` que `publica` já chamava ANTES de subir. O que faltava
    era perguntar depois — e é exatamente a assinatura de
    `juliadollis/bokehnet-regen-rota-b`, com `mask/` completo e `meta/` pela metade.
    """

    def setUp(self):
        super().setUp()
        self._hub_salvo = sys.modules.get("huggingface_hub")
        self.addCleanup(self._restaura)

    def _restaura(self):
        if self._hub_salvo is None:
            sys.modules.pop("huggingface_hub", None)
        else:
            sys.modules["huggingface_hub"] = self._hub_salvo

    def _instala(self, api: _FakeApi) -> _FakeApi:
        modulo = types.ModuleType("huggingface_hub")
        modulo.HfApi = lambda: api                       # type: ignore[attr-defined]
        constantes = types.ModuleType("huggingface_hub.constants")
        constantes.DEFAULT_REQUEST_TIMEOUT = 10
        modulo.constants = constantes                    # type: ignore[attr-defined]
        sys.modules["huggingface_hub"] = modulo
        return api

    def test_upload_que_chega_inteiro_passa(self):
        api = self._instala(_FakeApi(arquivos=None))
        with redirect_stdout(io.StringIO()) as saida:
            publica(self.root, "x/y", private=True, allow_existing=False)
        self.assertIn("cada pasta chegou com a contagem do disco", saida.getvalue())
        self.assertTrue(api.ops)

    def test_upload_pela_metade_reprova_dizendo_a_pasta_e_o_quanto(self):
        self._instala(_FakeApi(arquivos=None, engole=True))
        with self.assertRaises(Problema) as ctx, redirect_stdout(io.StringIO()):
            publica(self.root, "x/y", private=True, allow_existing=False)
        recado = str(ctx.exception)
        self.assertIn("meta/: 0 no Hub, 4 no disco", recado)
        self.assertIn("faltam 4", recado)
        self.assertIn("pela metade", recado)


# --------------------------------------------------------------------------------
# A trava de contagem depois da dedup — comparar o que DEVE ser comparado
# --------------------------------------------------------------------------------
#
# `_confere_contagem_por_diretorio` comparava `depth/` com `meta/` contando arquivos. Com
# a profundidade deduplicada por cena, `depth/` tem 1.700 arquivos para 69.700 amostras,
# e essa comparação reprova um release CORRETO. O que `depth/` tem que satisfazer é
# outra coisa, e `_confere_profundidade` é quem confere:
#
#   1. toda amostra tem profundidade ALCANÇÁVEL (pela referência, não pela convenção);
#   2. nenhum arquivo de profundidade sem dono — a metade da trava antiga que não pode
#      se perder na mudança;
#   3. a referência bate com o layout que o `run_config.json` DECLARA.

class TestProfundidadeDeduplicada(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.mkdtemp()
        self.root = Path(self._tmp)
        # 6 amostras em 3 cenas, duas por cena: `depth/` fica com 3 arquivos e `meta/`
        # com 6. É exatamente a assimetria que a trava antiga chamaria de "faltam 3".
        self.rel = _Release(self.root, rota="c", n=6, val_scenes=1, dedup=True)

    def tearDown(self):
        shutil.rmtree(self._tmp, ignore_errors=True)

    def assertReprova(self, trecho: str):
        with self.assertRaises(Problema) as ctx:
            valida(self.root)
        self.assertIn(trecho, str(ctx.exception))

    def test_release_deduplicado_passa_com_depth_menor_que_meta(self):
        """A invariante que a mudança da trava existe para permitir: um release em que
        `depth/` tem MENOS arquivos que `meta/` é correto, e tem que passar."""
        self.assertEqual(len(list((self.root / "depth").glob("*.png"))), 3)
        self.assertEqual(len(list((self.root / "meta").glob("*.json"))), 6)
        resumo = valida(self.root)
        self.assertEqual(resumo["amostras"], 6)
        self.assertEqual(resumo["depth_layout"], "per_scene")
        self.assertEqual(resumo["depth_arquivos"], 3)

    def test_profundidade_inalcancavel_reprova(self):
        """A trava não afrouxou: a pergunta mudou de "existe um arquivo com o nome
        dela?" para "o `depth_ref` dela resolve?", e continua reprovando."""
        (self.root / "depth" / "0000.png").unlink()
        self.assertReprova("sem arquivo de profundidade alcançável")

    def test_arquivo_de_profundidade_sem_dono_reprova(self):
        """A metade da trava antiga que não pode se perder: com 1.700 arquivos
        reivindicados e 69.700 no disco, é isto que denuncia uma migração pela metade —
        e é isto que impede o resto de outro run de subir junto."""
        (self.root / "depth" / "sobra_de_outro_run.png").write_bytes(b"x")
        self.assertReprova("linha nenhuma do manifesto reivindica")

    def test_referencia_fora_do_layout_declarado_reprova(self):
        """`run_config.json` diz `per_scene` e o manifesto grava `depth/<sample_id>.png`:
        a dedup não aconteceu, e o card mandaria quem baixa procurar no caminho errado."""
        linhas = self.rel.manifesto()
        linhas[0]["depth_ref"] = f"depth/{linhas[0]['sample_id']}.png"
        self.rel.escreve_manifesto(linhas)
        self.assertReprova("fora do layout declarado")

    def test_layout_divergente_entre_meta_e_run_config_reprova(self):
        """Duas afirmações sobre a mesma coisa não podem divergir em silêncio: um
        release assim foi remontado de dois runs, e metade dele aponta para arquivos que
        o card não descreve."""
        cfg = json.loads((self.root / "run_config.json").read_text(encoding="utf-8"))
        cfg["depth_layout"] = "per_sample"
        (self.root / "run_config.json").write_text(json.dumps(cfg), encoding="utf-8")
        self.assertReprova("layout de profundidade divergente")

    def test_layout_desconhecido_reprova_em_vez_de_adivinhar(self):
        cfg = json.loads((self.root / "run_config.json").read_text(encoding="utf-8"))
        cfg["depth_layout"] = "por_cena"
        (self.root / "run_config.json").write_text(json.dumps(cfg), encoding="utf-8")
        with self.assertRaises(Problema) as ctx:
            valida(self.root)
        self.assertIn("desconhecido", str(ctx.exception))

    def test_as_outras_pastas_continuam_sendo_comparadas_entre_si(self):
        """A dedup não pode afrouxar a trava para `mask/` e `meta/`: é ela que pega os
        13.615 `mask/` contra 4.688 `meta/` de `bokehnet-regen-rota-b`."""
        (self.root / "meta" / f"{self.rel.ids[0]}.json").unlink()
        self.assertReprova("meta/ tem 5 arquivos para 6 amostras")

    def test_o_card_manda_ler_a_referencia_em_vez_de_remontar_o_caminho(self):
        card = escreve_card(self.root, valida(self.root), "x/y").read_text(
            encoding="utf-8")
        self.assertIn("depth/<scene_id>.png", card)
        self.assertIn("depth_ref", card)
        self.assertIn("Não remonte o caminho", card)


class TestReleasePublicadoContinuaValidando(unittest.TestCase):
    """Compatibilidade de leitura, e não só de escrita.

    `juliadollis/bokehnet-regen-rota-c` está publicado no layout por amostra, sem
    `depth_ref` e sem `depth_layout`, e o `-rota-a` está subindo agora no mesmo formato.
    Se a validação passasse a exigir os campos, os dois virariam releases reprovados —
    e a tarefa era o caminho de gravação daqui para a frente, não reescrever o passado.
    """

    def setUp(self):
        self._tmp = tempfile.mkdtemp()
        self.root = Path(self._tmp)
        self.rel = _Release(self.root, rota="c", n=4, val_scenes=1)   # sem dedup

    def tearDown(self):
        shutil.rmtree(self._tmp, ignore_errors=True)

    def test_sem_os_campos_o_release_passa_e_e_lido_como_per_sample(self):
        for linha in self.rel.manifesto():
            self.assertNotIn("depth_ref", linha)
        resumo = valida(self.root)
        self.assertEqual(resumo["depth_layout"], "per_sample")
        self.assertEqual(resumo["depth_arquivos"], 4)

    def test_e_a_trava_continua_pegando_profundidade_faltando(self):
        (self.root / "depth" / f"{self.rel.ids[0]}.png").unlink()
        with self.assertRaises(Problema) as ctx:
            valida(self.root)
        self.assertIn("sem arquivo de profundidade alcançável", str(ctx.exception))

    def test_e_continua_pegando_orfao_em_depth(self):
        (self.root / "depth" / "resto.png").write_bytes(b"x")
        with self.assertRaises(Problema) as ctx:
            valida(self.root)
        self.assertIn("linha nenhuma do manifesto reivindica", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
