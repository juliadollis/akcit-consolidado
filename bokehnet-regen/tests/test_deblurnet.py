"""Testes do runtime da DeblurNet — o que dá para verificar sem GPU e sem baixar peso.

A inferência não roda aqui (precisa de FLUX.1-dev, do LoRA e de uma H100). O que roda, e
o que importa testar ANTES, é o contrato:

1. **o cruzamento de variantes é inexpressável** — não há assinatura que o aceite, e o
   acidente realista (apontar o peso do outro repositório) levanta exceção;
2. `main_adapter` é passado sempre, explicitamente, e o call site é conferido por AST —
   é o defeito B1, e um teste que só olhasse a proveniência não pegaria a regressão;
3. o recorte por `long_side` está medido e registrado, nunca silencioso;
4. a proveniência não mente.

Padrão de `tests/test_model_runtime.py` e `tests/test_renderer.py`: caminhos e config
validados antes de importar torch, sentinelas de arquivo, e AST em vez de execução.
"""

from __future__ import annotations

import ast
import dataclasses
import hashlib
import itertools
import inspect
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from control.contract import SampleRejected                          # noqa: E402
from model_runtime import DEBLUR_BACKEND                             # noqa: E402
from model_runtime import deblurnet as mod                           # noqa: E402
from model_runtime.deblurnet import (                                # noqa: E402
    ADAPTER_NAME,
    PROMPT,
    DeblurNetRuntime,
    DeblurVariant,
    DeblurVariantMismatch,
    DeblurVariantSpec,
    ProcessingPlan,
    ResizePolicy,
    _inspect_flux_generate,
    filename_is_ambiguous,
    plan_processing,
    resolve_weights,
    variant_of_weight_filename,
    variants_of_weight_filename,
)

_RAIZ = Path(__file__).resolve().parents[1]

#: O `flux.py` oficial deste checkout, quando existe. Serve para provar que a checagem
#: por AST aceita o upstream de verdade, e não só o dublê do teste.
_FLUX_OFICIAL = (_RAIZ.parents[0] / "genrefocus_deblurnet_paper" / "third_party"
                 / "Genfocus" / "Genfocus" / "pipeline" / "flux.py")


# ==============================================================================
# Ambiente de teste — um checkout falso, sem torch e sem peso de verdade
# ==============================================================================

_FLUX_PY_FALSO = '''
"""Dublê de Genfocus/pipeline/flux.py. Nunca é importado: só parseado por AST."""
import torch                       # o real importa torch, diffusers e cv2 aqui


def seed_everything(seed: int = 42):
    ...


class Condition(object):
    def __init__(self, condition, adapter_setting, position_delta=None,
                 position_scale=1.0):
        ...


def generate(pipeline, prompt=None, height=512, width=512,
             num_inference_steps=28, main_adapter=None, conditions=[],
             transformer_kwargs={}, NO_TILED_DENOISE=False, batch_tiles=False,
             **params):
    ...
'''


class _Ambiente:
    """Cria pesos, backbone e checkout num tmpdir. Nada aqui é um modelo de verdade."""

    def __init__(self, caso: unittest.TestCase, *, flux_py: str = _FLUX_PY_FALSO):
        tmp = tempfile.TemporaryDirectory()
        caso.addCleanup(tmp.cleanup)
        self.raiz = Path(tmp.name)

        self.pesos_dir = self.raiz / "pesos"
        self.pesos_dir.mkdir()
        # UM arquivo por NOME, não por variante: `deblur.safetensors` é o nome das DUAS
        # variantes nossas, e escrever "um por variante" fazia a segunda sobrescrever a
        # primeira — o mesmo silêncio que o `_FILENAME_OWNERS` do módulo passou a expor.
        # Um único arquivo com esse nome é também o que existe na máquina de verdade.
        for nome in sorted({v.spec.weight_filename for v in DeblurVariant}):
            (self.pesos_dir / nome).write_bytes(f"lora-falso-{nome}".encode("utf-8"))

        # `OURS_COND_ONLY` é fixada no sha256 do peso REAL, que não está nesta máquina e
        # que um teste não pode baixar. O que está sob teste é a REGRA — "se a spec fixa
        # um sha, ele é obrigatório" —, não o dígito; então aqui a spec é repontada para
        # o sha do peso falso e a regra continua valendo inteira. Que o dígito de
        # verdade é o medido, quem prova é `SpecsDeclaradas.test_pinned_sha_e_o_medido`,
        # que lê o módulo sem passar por este ambiente.
        self._repontar_pinned(caso, self.sha(self.pesos(DeblurVariant.OURS_COND_ONLY)))

        self.flux_dir = self.raiz / "FLUX.1-dev"
        (self.flux_dir / "transformer").mkdir(parents=True)
        (self.flux_dir / "model_index.json").write_text('{"_class_name":"FluxPipeline"}')
        (self.flux_dir / "transformer" / "diffusion_pytorch_model.safetensors").write_bytes(
            b"\x00" * 4096
        )

        self.genfocus_dir = self.raiz / "Genfocus"
        (self.genfocus_dir / "Genfocus" / "pipeline").mkdir(parents=True)
        (self.genfocus_dir / "Inference_deblurNet.py").write_text("# dublê\n")
        (self.genfocus_dir / "Genfocus" / "pipeline" / "flux.py").write_text(flux_py)

    @staticmethod
    def sha(caminho: Path) -> str:
        return hashlib.sha256(Path(caminho).read_bytes()).hexdigest()

    @staticmethod
    def _repontar_pinned(caso: unittest.TestCase, sha: str) -> None:
        original = mod._VARIANT_SPECS[DeblurVariant.OURS_COND_ONLY]
        caso.addCleanup(mod._VARIANT_SPECS.__setitem__,
                        DeblurVariant.OURS_COND_ONLY, original)
        mod._VARIANT_SPECS[DeblurVariant.OURS_COND_ONLY] = dataclasses.replace(
            original, pinned_sha256=sha)

    def pesos(self, variante: DeblurVariant) -> Path:
        return self.pesos_dir / variante.spec.weight_filename

    def outro_peso(self, variante: DeblurVariant, conteudo: bytes) -> Path:
        """Um arquivo com o NOME da variante, em outra pasta, com outro conteúdo.

        É o acidente realista depois do `b_release`: dois snapshots do HF no mesmo
        cache, os dois com `deblur.safetensors`, e só o sha256 separando os dois.
        """
        pasta = self.raiz / f"outro_{variante.value}"
        pasta.mkdir(exist_ok=True)
        alvo = pasta / variante.spec.weight_filename
        alvo.write_bytes(conteudo)
        return alvo

    def runtime(self, variante: DeblurVariant, **kwargs) -> DeblurNetRuntime:
        kwargs.setdefault("weights_path", self.pesos(variante))
        caminho = Path(kwargs["weights_path"])
        # Nome ambíguo + spec sem sha fixo = o módulo EXIGE `expected_lora_sha256`. Os
        # testes que são sobre outra coisa não deveriam ter que saber disso; os que são
        # sobre isso passam o argumento (nem que seja `None`) e este default sai do
        # caminho, porque a checagem é `in kwargs`, não truthiness.
        if ("expected_lora_sha256" not in kwargs
                and variante.spec.pinned_sha256 is None
                and filename_is_ambiguous(caminho.name)
                and caminho.is_file()):
            kwargs["expected_lora_sha256"] = self.sha(caminho)
        return DeblurNetRuntime(
            variant=variante,
            flux_dir=self.flux_dir,
            genfocus_dir=self.genfocus_dir,
            device="cpu",
            **kwargs,
        )


# ==============================================================================
# 1. A tripla é indivisível — o cruzamento não tem como ser expresso
# ==============================================================================

class TriplaIndivisivel(unittest.TestCase):
    def test_enum_rejeita_string_livre(self):
        """O pipeline de avaliação tem `--main-adapter` de texto livre
        (`infer_and_eval.py:98-101`). Aqui a variante é enum fechado."""
        for lixo in ("deblurring", "none", "ours", "", "OURS_MAIN_COND"):
            with self.assertRaises(ValueError):
                DeblurVariant(lixo)

    def test_toda_variante_tem_spec(self):
        """Cobertura do enum. Uma variante sem spec só apareceria em produção, como
        `KeyError` sem contexto — o módulo confere isso na importação, e aqui também."""
        variantes = list(DeblurVariant)
        self.assertEqual(len(variantes), 3)
        for variante in variantes:
            with self.subTest(variante=variante.value):
                spec = variante.spec              # KeyError aqui é o teste falhando
                self.assertIs(spec.variant, variante)
                self.assertTrue(spec.repo_id and spec.weight_filename and spec.evidence)
        self.assertEqual({v.spec.repo_id for v in variantes}, {
            "juliadollis/genrefocus-deblurnet-paper-4gpu",
            "juliadollis/genrefocus-deblurnet",
            "nycu-cplab/Genfocus-Model",
        }, "duas variantes no mesmo repositório")

    def test_modo_e_adapter_sao_um_para_um_em_toda_variante(self):
        for variante in DeblurVariant:
            with self.subTest(variante=variante.value):
                spec = variante.spec
                self.assertIn(spec.lora_mode, ("main_cond", "cond_only"))
                self.assertEqual(
                    spec.main_adapter,
                    ADAPTER_NAME if spec.lora_mode == "main_cond" else None)

    def test_o_nome_do_arquivo_NAO_e_disjunto_e_o_modulo_sabe_disso(self):
        """O fato que produziu o `b_release` errado, agora como invariante.

        `deblur.safetensors` é o nome de DUAS variantes nossas. Antes, `_FILENAME_OWNER`
        era um dict de um dono só e a segunda sobrescrevia a primeira em silêncio; a
        única trava do pipeline era esse nome, e ele não distingue nada.
        """
        arquivos = [v.spec.weight_filename for v in DeblurVariant]
        self.assertEqual(len(set(arquivos)), 2, "os nomes deixaram de se repetir?")
        donos = variants_of_weight_filename("deblur.safetensors")
        self.assertEqual(set(donos),
                         {DeblurVariant.OURS_MAIN_COND, DeblurVariant.OURS_COND_ONLY})
        self.assertTrue(filename_is_ambiguous("deblur.safetensors"))
        self.assertFalse(filename_is_ambiguous("deblurNet.safetensors"))

    def test_a_tripla_bate_com_a_evidencia(self):
        nossa = DeblurVariant.OURS_MAIN_COND.spec
        self.assertEqual(nossa.repo_id, "juliadollis/genrefocus-deblurnet-paper-4gpu")
        self.assertEqual(nossa.weight_filename, "deblur.safetensors")
        self.assertEqual(nossa.main_adapter, "deblurring")

        nova = DeblurVariant.OURS_COND_ONLY.spec
        self.assertEqual(nova.repo_id, "juliadollis/genrefocus-deblurnet")
        self.assertEqual(nova.weight_filename, "deblur.safetensors")
        self.assertIsNone(nova.main_adapter, "o treino tem lora_on_main: false")
        self.assertEqual(nova.lora_mode, "cond_only")

        oficial = DeblurVariant.OFFICIAL_COND_ONLY.spec
        self.assertEqual(oficial.repo_id, "nycu-cplab/Genfocus-Model")
        self.assertEqual(oficial.weight_filename, "deblurNet.safetensors")
        self.assertIsNone(oficial.main_adapter)

    def test_pinned_sha_e_o_medido(self):
        """O dígito exato, lido do módulo sem passar pelo `_Ambiente`.

        Medido no peso de `outputs/deblur_docker_4gpu/deblur/deblur.safetensors`. O
        literal está quebrado em duas linhas no fonte; este teste é o que pega uma
        concatenação truncada, que passaria despercebida e faria TODA seleção da
        variante certa falhar no cluster com "peso errado".
        """
        fixado = DeblurVariant.OURS_COND_ONLY.spec.pinned_sha256
        self.assertEqual(
            fixado,
            "a1ed05e48f7a3469cf825eda6a8ceca4cca1e6f6dbdb665d78a6f9e207d8cecc")
        self.assertEqual(len(fixado), 64)
        self.assertIsNone(DeblurVariant.OURS_MAIN_COND.spec.pinned_sha256,
                          "o peso main+cond não está em disco: não há o que fixar")
        self.assertIsNone(DeblurVariant.OFFICIAL_COND_ONLY.spec.pinned_sha256)

    def test_spec_com_adapter_trocado_nao_pode_ser_construida(self):
        """Mesmo montada à mão, a tripla incoerente morre no `__post_init__`."""
        with self.assertRaises(DeblurVariantMismatch) as ctx:
            DeblurVariantSpec(
                variant=DeblurVariant.OURS_MAIN_COND,
                repo_id="x/y", weight_filename="deblur.safetensors",
                main_adapter=None, lora_mode="main_cond", evidence="teste",
            )
        self.assertIn("LAVADO", str(ctx.exception))

        with self.assertRaises(DeblurVariantMismatch):
            DeblurVariantSpec(
                variant=DeblurVariant.OFFICIAL_COND_ONLY,
                repo_id="x/y", weight_filename="deblurNet.safetensors",
                main_adapter=ADAPTER_NAME, lora_mode="cond_only", evidence="teste",
            )

        # A variante nova, que é o peso que existe em disco: cond-only com o adapter no
        # main é literalmente o que rodou em `b_release`.
        with self.assertRaises(DeblurVariantMismatch) as ctx:
            DeblurVariantSpec(
                variant=DeblurVariant.OURS_COND_ONLY,
                repo_id="juliadollis/genrefocus-deblurnet",
                weight_filename="deblur.safetensors",
                main_adapter=ADAPTER_NAME, lora_mode="cond_only", evidence="teste",
                pinned_sha256="0" * 64,
            )
        self.assertIn("cond_only", str(ctx.exception))

    def test_spec_com_lora_mode_desconhecido_nao_pode_ser_construida(self):
        """O par (modo, adapter) é 1-para-1 e fechado: um modo novo teria que declarar
        qual adapter ele exige, e falha na IMPORTAÇÃO do módulo, não no meio do run."""
        with self.assertRaises(ValueError) as ctx:
            DeblurVariantSpec(
                variant=DeblurVariant.OURS_COND_ONLY,
                repo_id="x/y", weight_filename="deblur.safetensors",
                main_adapter=None, lora_mode="cond-only", evidence="teste",
            )
        self.assertIn("lora_mode", str(ctx.exception))

    def test_spec_e_congelada(self):
        spec = DeblurVariant.OURS_MAIN_COND.spec
        with self.assertRaises(Exception):
            spec.main_adapter = None            # type: ignore[misc]

    def test_nenhuma_api_publica_aceita_main_adapter(self):
        """A defesa principal: não existe assinatura em que o cruzamento caiba.

        Cobre também os outros eixos que, livres, permitiriam a mesma mistura:
        repositório, nome do arquivo de pesos e prompt.
        """
        proibidos = ("main_adapter", "adapter", "repo", "repo_id", "filename",
                     "weight_filename", "weight_name", "prompt", "spec")
        alvos = [DeblurNetRuntime.__init__, DeblurNetRuntime.infer,
                 DeblurNetRuntime.plan_for, resolve_weights, plan_processing]
        for alvo in alvos:
            nomes = set(inspect.signature(alvo).parameters)
            for proibido in proibidos:
                self.assertNotIn(
                    proibido, nomes,
                    f"{alvo.__qualname__} aceita {proibido!r}: isso reabre o cruzamento",
                )

    def test_peso_da_outra_variante_e_rejeitado_nos_dois_sentidos(self):
        """**O teste que prova que não dá para cruzar.**

        Peso main+cond com a convenção cond-only sai LAVADO; peso cond-only com o
        adapter no main aplica o LoRA num branch em que ele não foi treinado. Nenhum
        dos dois levanta exceção na inferência — por isso levanta aqui.
        """
        amb = _Ambiente(self)
        # Variantes SEM `pinned_sha256`: o nome do arquivo é a única evidência, e é ele
        # que acusa o cruzamento.
        cruzamentos = [
            (DeblurVariant.OURS_MAIN_COND, DeblurVariant.OFFICIAL_COND_ONLY),
            (DeblurVariant.OFFICIAL_COND_ONLY, DeblurVariant.OURS_MAIN_COND),
        ]
        for selecionada, peso_de in cruzamentos:
            with self.subTest(variante=selecionada.value, peso=peso_de.value):
                self.assertIsNone(selecionada.spec.pinned_sha256)
                with self.assertRaises(DeblurVariantMismatch) as ctx:
                    amb.runtime(selecionada, weights_path=amb.pesos(peso_de))
                msg = str(ctx.exception)
                self.assertIn("cruzamento de variante", msg)
                self.assertIn(peso_de.spec.repo_id, msg)
                self.assertIn(selecionada.spec.repo_id, msg)
                self.assertIn("INDIVISÍVEL", msg)

        # Variante COM pin: quem acusa é o HASH, e tem que continuar acusando. O nome
        # `deblurNet.safetensors` não pode mais decidir nada aqui — é também o nome sob
        # o qual o NOSSO peso está publicado no Hub.
        with self.assertRaises(DeblurVariantMismatch) as ctx:
            amb.runtime(DeblurVariant.OURS_COND_ONLY,
                        weights_path=amb.pesos(DeblurVariant.OFFICIAL_COND_ONLY))
        msg = str(ctx.exception)
        self.assertIn("sha256", msg)
        self.assertIn(DeblurVariant.OFFICIAL_COND_ONLY.spec.repo_id, msg)
        self.assertIn(DeblurVariant.OURS_COND_ONLY.spec.repo_id, msg)
        self.assertIn("INDIVISÍVEL", msg)

    def test_o_par_correto_passa(self):
        """O contraponto obrigatório: a checagem não pode reprovar o caso legítimo."""
        amb = _Ambiente(self)
        for variante in DeblurVariant:
            rt = amb.runtime(variante)
            self.assertIs(rt.variant, variante)
            self.assertEqual(rt.weights_path.name, variante.spec.weight_filename)

    def test_peso_com_nome_desconhecido_e_rejeitado(self):
        amb = _Ambiente(self)
        estranho = amb.raiz / "pesos" / "checkpoint.safetensors"
        estranho.write_bytes(b"qualquer")
        with self.assertRaises(DeblurVariantMismatch) as ctx:
            amb.runtime(DeblurVariant.OURS_MAIN_COND, weights_path=estranho)
        self.assertIn("expected_lora_sha256", str(ctx.exception))

    def test_peso_renomeado_e_pego_pelo_sha_esperado(self):
        """Renomear contorna a checagem de nome; o hash não se contorna.

        E é a única verificação possível sobre o arquivo: main+cond e cond-only têm as
        mesmas chaves de LoRA — o peso NÃO sabe de qual convenção precisa.
        """
        amb = _Ambiente(self)
        certo = amb.runtime(DeblurVariant.OURS_MAIN_COND)
        sha_certo = certo.provenance()["deblur_lora_sha256"]

        disfarcado = amb.pesos_dir / "deblur.safetensors"
        original = disfarcado.read_bytes()
        self.addCleanup(disfarcado.write_bytes, original)
        disfarcado.write_bytes(b"lora-falso-official_cond_only")   # outro conteúdo

        with self.assertRaises(DeblurVariantMismatch) as ctx:
            amb.runtime(DeblurVariant.OURS_MAIN_COND, expected_lora_sha256=sha_certo)
        self.assertIn("sha256", str(ctx.exception))

    def test_variant_of_weight_filename_nao_chuta_em_nome_ambiguo(self):
        """`None` para nome AMBÍGUO, igual a nome desconhecido.

        Nos dois casos o nome não identifica variante nenhuma, e devolver uma das duas
        seria exatamente o palpite que gerou o `b_release`. Antes esta função devolvia
        `OURS_MAIN_COND` para `deblur.safetensors` — porque o dict tinha um dono só e a
        ordem de inserção decidia qual.
        """
        self.assertIsNone(variant_of_weight_filename("deblur.safetensors"))
        self.assertIsNone(variant_of_weight_filename("/tmp/a/deblur.safetensors"))
        self.assertIs(variant_of_weight_filename("/tmp/a/deblurNet.safetensors"),
                      DeblurVariant.OFFICIAL_COND_ONLY)
        self.assertIsNone(variant_of_weight_filename("outro.safetensors"))

    def test_variants_of_weight_filename_devolve_os_dois_donos(self):
        self.assertEqual(
            set(variants_of_weight_filename("deblur.safetensors")),
            {DeblurVariant.OURS_MAIN_COND, DeblurVariant.OURS_COND_ONLY})
        self.assertEqual(variants_of_weight_filename("/tmp/a/deblurNet.safetensors"),
                         (DeblurVariant.OFFICIAL_COND_ONLY,))
        self.assertEqual(variants_of_weight_filename("outro.safetensors"), ())

    def test_resolve_weights_deriva_repo_e_arquivo_da_variante(self):
        amb = _Ambiente(self)
        for variante in DeblurVariant:
            caminho = resolve_weights(variante, local_dir=amb.pesos_dir)
            self.assertEqual(caminho.name, variante.spec.weight_filename)

    def test_resolve_weights_sem_o_arquivo_da_variante_falha(self):
        amb = _Ambiente(self)
        vazio = amb.raiz / "vazio"
        vazio.mkdir()
        with self.assertRaises(FileNotFoundError) as ctx:
            resolve_weights(DeblurVariant.OURS_MAIN_COND, local_dir=vazio)
        self.assertIn("deblur.safetensors", str(ctx.exception))

    def test_resolve_weights_rejeita_variante_invalida(self):
        with self.assertRaises(ValueError):
            resolve_weights("deblurring")                      # type: ignore[arg-type]


# ==============================================================================
# 1b. O nome do arquivo NÃO identifica a variante — o sha256 identifica
#
# O lote `output/b_release` inteiro (13.615 amostras) saiu com o peso cond-only
# (`juliadollis/genrefocus-deblurnet`, `lora_on_main: false`) invocado como main+cond,
# porque `deblur.safetensors` é o nome das DUAS variantes nossas e a única trava era o
# nome. Zero exceções; só AIF macia. Cada teste abaixo é um lado dessa fresta.
# ==============================================================================

class OSha256IdentificaAVariante(unittest.TestCase):
    def test_variante_fixada_rejeita_peso_com_outro_sha_e_cita_a_outra(self):
        """Regra 1: se a spec fixa `pinned_sha256`, ele é OBRIGATÓRIO.

        E a mensagem tem que citar a outra dona do nome — porque o acidente realista é
        justamente estar com o peso de lá, e "sha divergente" sozinho não diz onde olhar.
        """
        amb = _Ambiente(self)
        impostor = amb.outro_peso(DeblurVariant.OURS_COND_ONLY,
                                  b"conteudo-do-treino-main+cond-antigo")
        with self.assertRaises(DeblurVariantMismatch) as ctx:
            amb.runtime(DeblurVariant.OURS_COND_ONLY, weights_path=impostor)
        msg = str(ctx.exception)
        self.assertIn("ours_cond_only", msg)
        self.assertIn("ours_main_cond", msg, "não cita a outra dona do mesmo nome")
        self.assertIn(amb.sha(impostor), msg, "não diz o hash que encontrou")
        self.assertIn(DeblurVariant.OURS_COND_ONLY.spec.pinned_sha256, msg)

    def test_variante_fixada_com_o_peso_certo_constroi(self):
        """O contraponto obrigatório: o par legítimo passa sem `expected_lora_sha256`,
        porque a spec já fixa o hash."""
        amb = _Ambiente(self)
        rt = amb.runtime(DeblurVariant.OURS_COND_ONLY)
        self.assertIs(rt.variant, DeblurVariant.OURS_COND_ONLY)
        self.assertIsNone(rt.expected_lora_sha256)
        self.assertIsNone(rt.provenance()["main_adapter"])

    def test_ours_main_cond_com_nome_ambiguo_exige_sha_esperado(self):
        """Regra 2: nome ambíguo + spec sem hash fixo = recusa, não palpite.

        `OURS_MAIN_COND` não tem peso em disco para fixar. Escolher a variante pelo nome
        do arquivo é exatamente o que produziu `b_release`; aqui o runtime se recusa a
        carregar e diz qual hash o arquivo tem, para quem chama poder conferir de onde
        ele veio antes de afirmar qualquer coisa.
        """
        amb = _Ambiente(self)
        with self.assertRaises(DeblurVariantMismatch) as ctx:
            amb.runtime(DeblurVariant.OURS_MAIN_COND, expected_lora_sha256=None)
        msg = str(ctx.exception)
        self.assertIn("expected_lora_sha256", msg)
        self.assertIn("ours_main_cond", msg)
        self.assertIn("ours_cond_only", msg, "não diz com quem o nome é compartilhado")
        self.assertIn(amb.sha(amb.pesos(DeblurVariant.OURS_MAIN_COND)), msg)

    def test_ours_main_cond_com_sha_esperado_correto_passa(self):
        """Dito o hash em voz alta, a variante constrói — e a proveniência registra
        main+cond, que é o que ela é."""
        amb = _Ambiente(self)
        peso = amb.pesos(DeblurVariant.OURS_MAIN_COND)
        rt = amb.runtime(DeblurVariant.OURS_MAIN_COND,
                         expected_lora_sha256=amb.sha(peso))
        prov = rt.provenance()
        self.assertEqual(prov["deblur_variant"], "ours_main_cond")
        self.assertEqual(prov["main_adapter"], ADAPTER_NAME)
        self.assertEqual(prov["deblur_lora_sha256"], amb.sha(peso))

    def test_a_variante_oficial_nao_regride(self):
        """`deblurNet.safetensors` não é ambíguo e a spec oficial não fixa hash: ela tem
        que continuar construindo SEM `expected_lora_sha256`.

        Não é detalhe: é a variante que está gerando dados agora. A regra nova é sobre
        nome compartilhado, e não pode virar "todo mundo precisa de hash".
        """
        amb = _Ambiente(self)
        rt = amb.runtime(DeblurVariant.OFFICIAL_COND_ONLY, expected_lora_sha256=None)
        prov = rt.provenance()
        self.assertEqual(prov["deblur_variant"], "official_cond_only")
        self.assertIsNone(prov["main_adapter"])
        self.assertEqual(prov["deblur_repo_id"], "nycu-cplab/Genfocus-Model")

    def test_o_nosso_peso_com_o_NOME_da_oficial_constroi_como_nossa(self):
        """**O caso do download do Hub, e o defeito que ele expôs.**

        `juliadollis/genrefocus-deblurnet` publica o nosso peso como
        `deblurNet.safetensors` — o nome declarado pela variante OFICIAL —, com o mesmo
        sha256 `a1ed05e4…` e os mesmos 1.854.533.784 B da cópia do cluster (medido com
        `HfApi.repo_info(files_metadata=True)`, sem baixar). Com a checagem de nome antes
        do hash, apontar para esse arquivo dava "cruzamento de variante: este é o peso de
        official_cond_only" — sobre o NOSSO peso, com o sha certo, mandando fazer a troca
        errada. O hash casa: é o nosso, e o nome não tem voto.
        """
        amb = _Ambiente(self)
        conteudo = amb.pesos(DeblurVariant.OURS_COND_ONLY).read_bytes()
        baixado = amb.raiz / "hub" / "deblurNet.safetensors"
        baixado.parent.mkdir(exist_ok=True)
        baixado.write_bytes(conteudo)
        self.assertEqual(baixado.name,
                         DeblurVariant.OFFICIAL_COND_ONLY.spec.weight_filename)
        self.assertEqual(amb.sha(baixado),
                         DeblurVariant.OURS_COND_ONLY.spec.pinned_sha256)

        rt = amb.runtime(DeblurVariant.OURS_COND_ONLY, weights_path=baixado)
        self.assertIs(rt.variant, DeblurVariant.OURS_COND_ONLY)

        prov = rt.provenance()
        self.assertEqual(prov["deblur_variant"], "ours_cond_only")
        self.assertEqual(prov["deblur_weight_filename"], "deblurNet.safetensors",
                         "a proveniência tem que dizer o arquivo que foi CARREGADO")
        self.assertEqual(prov["deblur_weight_filename_declared"], "deblur.safetensors")
        self.assertIsNone(prov["main_adapter"], "cond-only, como o treino")

    def test_o_peso_OFICIAL_com_esse_mesmo_nome_continua_recusado(self):
        """O contraponto que impede o conserto de virar afrouxamento.

        Mesmo nome de arquivo, variante `ours_cond_only` selecionada, conteúdo da
        oficial: o nome agora não decide — e o sha recusa. Sem este teste, "o hash manda"
        poderia ter sido implementado como "o nome não importa mais".
        """
        amb = _Ambiente(self)
        oficial = amb.pesos(DeblurVariant.OFFICIAL_COND_ONLY)
        self.assertEqual(oficial.name, "deblurNet.safetensors")
        with self.assertRaises(DeblurVariantMismatch) as ctx:
            amb.runtime(DeblurVariant.OURS_COND_ONLY, weights_path=oficial)
        msg = str(ctx.exception)
        self.assertIn(amb.sha(oficial), msg)
        self.assertIn(DeblurVariant.OURS_COND_ONLY.spec.pinned_sha256, msg)
        self.assertIn("official_cond_only", msg, "não aponta de quem é este peso")

    def test_variante_SEM_pin_nao_e_afrouxada_pelo_conserto(self):
        """`ours_main_cond` e `official_cond_only` não fixam hash: para elas o nome
        continua sendo a única evidência, e a regra de sempre vale inteira."""
        amb = _Ambiente(self)
        for variante in (DeblurVariant.OURS_MAIN_COND,
                         DeblurVariant.OFFICIAL_COND_ONLY):
            with self.subTest(variante=variante.value):
                self.assertIsNone(variante.spec.pinned_sha256)
                estranho = amb.raiz / f"estranho_{variante.value}.safetensors"
                estranho.write_bytes(b"peso-com-nome-de-ninguem")
                with self.assertRaises(DeblurVariantMismatch) as ctx:
                    amb.runtime(variante, weights_path=estranho)
                self.assertIn("expected_lora_sha256", str(ctx.exception))

    def test_o_mesmo_arquivo_com_a_outra_variante_nossa_e_recusado(self):
        """O cruzamento que o nome NÃO pega, nos dois sentidos, sobre UM único arquivo.

        É o `deblur.safetensors` que existe em disco. Como `OURS_COND_ONLY` (correto)
        ele passa; como `OURS_MAIN_COND` sem hash declarado, não passa. Nenhuma das duas
        levantaria exceção na inferência.
        """
        amb = _Ambiente(self)
        peso = amb.pesos(DeblurVariant.OURS_COND_ONLY)
        self.assertEqual(peso, amb.pesos(DeblurVariant.OURS_MAIN_COND),
                         "as duas variantes nossas compartilham o arquivo")
        amb.runtime(DeblurVariant.OURS_COND_ONLY, weights_path=peso)
        with self.assertRaises(DeblurVariantMismatch):
            amb.runtime(DeblurVariant.OURS_MAIN_COND, weights_path=peso,
                        expected_lora_sha256=None)


# ==============================================================================
# 2. A chamada a `generate` — o defeito B1, travado por AST
# ==============================================================================

def _chamadas_a(nome_atributo: str) -> list[ast.Call]:
    fonte = Path(mod.__file__).read_text(encoding="utf-8")
    arvore = ast.parse(fonte)
    return [no for no in ast.walk(arvore)
            if isinstance(no, ast.Call)
            and isinstance(no.func, ast.Attribute)
            and no.func.attr == nome_atributo]


class ChamadaAGenerate(unittest.TestCase):
    def test_o_call_site_passa_main_adapter_explicitamente(self):
        """A regressão do B1 em uma linha: alguém apaga o kwarg e a AIF sai lavada.

        Conferido no FONTE, por AST, porque `generate` termina em `**params` — omitir o
        kwarg não levanta `TypeError`, e um teste de comportamento sem GPU não veria.
        """
        chamadas = _chamadas_a("_generate")
        self.assertEqual(len(chamadas), 1, "esperava exatamente uma chamada a generate")
        kwargs = {kw.arg: kw.value for kw in chamadas[0].keywords}
        self.assertIn("main_adapter", kwargs,
                      "`generate` chamado sem `main_adapter`: é o defeito B1")

        valor = kwargs["main_adapter"]
        self.assertNotIsInstance(
            valor, ast.Constant,
            "`main_adapter` literal no call site: tem que vir da spec da variante")
        self.assertIsInstance(valor, ast.Attribute)
        self.assertEqual(valor.attr, "main_adapter")
        self.assertIsInstance(valor.value, ast.Attribute)
        self.assertEqual(valor.value.attr, "_spec")

    def test_o_call_site_usa_o_prompt_constante(self):
        chamadas = _chamadas_a("_generate")
        kwargs = {kw.arg: kw.value for kw in chamadas[0].keywords}
        self.assertIsInstance(kwargs["prompt"], ast.Name)
        self.assertEqual(kwargs["prompt"].id, "PROMPT")
        self.assertEqual(PROMPT, "a sharp photo with everything in focus")

    def test_o_token_do_hf_nunca_e_impresso(self):
        fonte = Path(mod.__file__).read_text(encoding="utf-8")
        for no in ast.walk(ast.parse(fonte)):
            if isinstance(no, ast.Call) and isinstance(no.func, ast.Name) \
                    and no.func.id == "print":
                nomes = {n.id for n in ast.walk(no) if isinstance(n, ast.Name)}
                self.assertNotIn("token", nomes, "token não vai para stdout")

    def test_guarda_aceita_o_flux_do_dube(self):
        amb = _Ambiente(self)
        info = _inspect_flux_generate(
            amb.genfocus_dir / "Genfocus" / "pipeline" / "flux.py")
        self.assertTrue(info["accepts_main_adapter"])
        self.assertTrue(info["main_adapter_default_is_none"])
        self.assertTrue(info["swallows_unknown_kwargs"])
        self.assertEqual(len(info["flux_py_sha256"]), 64)

    @unittest.skipUnless(_FLUX_OFICIAL.is_file(),
                         f"flux.py oficial ausente neste checkout: {_FLUX_OFICIAL}")
    def test_guarda_aceita_o_flux_oficial_e_confirma_o_default_none(self):
        """Prova, contra o upstream de verdade, a premissa que motivou o módulo:
        `main_adapter` existe, o default é `None` (= cond-only), e há `**params` para
        engolir em silêncio um kwarg renomeado."""
        info = _inspect_flux_generate(_FLUX_OFICIAL)
        self.assertTrue(info["accepts_main_adapter"])
        self.assertTrue(info["main_adapter_default_is_none"],
                        "o default de main_adapter deixou de ser None: revise o módulo")
        self.assertTrue(info["swallows_unknown_kwargs"],
                        "`**params` sumiu — a guarda por AST pode ser relaxada")

    def test_guarda_reprova_flux_sem_main_adapter(self):
        """O modo de falha que nenhuma exceção de runtime pegaria."""
        sem_kwarg = _FLUX_PY_FALSO.replace("main_adapter=None,", "main_lora=None,")
        amb = _Ambiente(self, flux_py=sem_kwarg)
        with self.assertRaises(RuntimeError) as ctx:
            amb.runtime(DeblurVariant.OURS_MAIN_COND)
        msg = str(ctx.exception)
        self.assertIn("main_adapter", msg)
        self.assertIn("engolido em silêncio", msg)

    def test_guarda_reprova_flux_sem_condition(self):
        sem_condition = _FLUX_PY_FALSO.replace("class Condition(object):",
                                               "class Conditions(object):")
        amb = _Ambiente(self, flux_py=sem_condition)
        with self.assertRaises(RuntimeError) as ctx:
            amb.runtime(DeblurVariant.OURS_MAIN_COND)
        self.assertIn("Condition", str(ctx.exception))

    def test_guarda_reprova_flux_sem_seed_everything(self):
        sem_seed = _FLUX_PY_FALSO.replace("def seed_everything", "def seed_all")
        amb = _Ambiente(self, flux_py=sem_seed)
        with self.assertRaises(RuntimeError) as ctx:
            amb.runtime(DeblurVariant.OURS_MAIN_COND)
        self.assertIn("seed_everything", str(ctx.exception))


# ==============================================================================
# 3. Falha alto e cedo — sem fallback, antes de importar torch
# ==============================================================================

class FalhaAltoECedo(unittest.TestCase):
    def test_sem_o_peso_falha(self):
        amb = _Ambiente(self)
        ausente = amb.raiz / "nao_existe" / "deblur.safetensors"
        with self.assertRaises(FileNotFoundError) as ctx:
            amb.runtime(DeblurVariant.OURS_MAIN_COND, weights_path=ausente)
        self.assertIn("NÃO existe fallback", str(ctx.exception))

    def test_sem_o_backbone_falha(self):
        amb = _Ambiente(self)
        (amb.flux_dir / "model_index.json").unlink()
        with self.assertRaises(FileNotFoundError) as ctx:
            amb.runtime(DeblurVariant.OURS_MAIN_COND)
        self.assertIn("model_index.json", str(ctx.exception))

    def test_sem_o_checkout_do_genfocus_falha(self):
        amb = _Ambiente(self)
        (amb.genfocus_dir / "Inference_deblurNet.py").unlink()
        with self.assertRaises(FileNotFoundError) as ctx:
            amb.runtime(DeblurVariant.OURS_MAIN_COND)
        msg = str(ctx.exception)
        self.assertIn("Genfocus", msg)
        self.assertIn("vendor/genfocus_flux.py", msg)

    def test_sem_o_flux_py_falha(self):
        amb = _Ambiente(self)
        (amb.genfocus_dir / "Genfocus" / "pipeline" / "flux.py").unlink()
        with self.assertRaises(FileNotFoundError):
            amb.runtime(DeblurVariant.OURS_MAIN_COND)

    def test_num_steps_invalido(self):
        amb = _Ambiente(self)
        for passos in (0, -1):
            with self.assertRaises(ValueError):
                amb.runtime(DeblurVariant.OURS_MAIN_COND, num_steps=passos)

    def test_long_side_negativo(self):
        amb = _Ambiente(self)
        with self.assertRaises(ValueError):
            amb.runtime(DeblurVariant.OURS_MAIN_COND, long_side=-16)

    def test_long_side_menor_que_o_multiplo_do_vae(self):
        amb = _Ambiente(self)
        with self.assertRaises(ValueError) as ctx:
            amb.runtime(DeblurVariant.OURS_MAIN_COND, long_side=8)
        self.assertIn("16", str(ctx.exception))

    def test_resize_policy_rejeita_string_livre(self):
        amb = _Ambiente(self)
        with self.assertRaises(ValueError):
            amb.runtime(DeblurVariant.OURS_MAIN_COND, resize_policy="crop")

    def test_importa_sem_torch(self):
        """Validação de caminho e de config tem que aparecer como erro de config, não
        como `No module named 'torch'` — e o módulo tem que carregar sem GPU."""
        codigo = (
            "import sys; import model_runtime.deblurnet as m; "
            "assert 'torch' not in sys.modules, 'torch importado no nível de módulo'; "
            "assert 'diffusers' not in sys.modules; "
            "print(m.BACKEND_NAME)"
        )
        proc = subprocess.run(
            [sys.executable, "-c", codigo], capture_output=True, text=True,
            cwd=str(_RAIZ), env={"PYTHONPATH": "src", "PATH": "/usr/bin:/bin"},
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(proc.stdout.strip(), DEBLUR_BACKEND)


# ==============================================================================
# 4. Proveniência que não mente
# ==============================================================================

class Proveniencia(unittest.TestCase):
    def test_hasheia_o_lora_antes_de_carregar(self):
        amb = _Ambiente(self)
        rt = amb.runtime(DeblurVariant.OURS_MAIN_COND)
        prov = rt.provenance()
        self.assertEqual(len(prov["deblur_lora_sha256"]), 64)
        self.assertIsNone(rt._pipe, "carregou o FLUX só para hashear")

    def test_grava_a_tripla_e_o_adapter_efetivo(self):
        amb = _Ambiente(self)
        esperado = {
            DeblurVariant.OURS_MAIN_COND: ("deblurring", "main_cond"),
            DeblurVariant.OURS_COND_ONLY: (None, "cond_only"),
            DeblurVariant.OFFICIAL_COND_ONLY: (None, "cond_only"),
        }
        for variante, (adapter, modo) in esperado.items():
            with self.subTest(variante=variante.value):
                prov = amb.runtime(variante).provenance()
                self.assertEqual(prov["deblur_variant"], variante.value)
                self.assertEqual(prov["deblur_repo_id"], variante.spec.repo_id)
                self.assertEqual(prov["deblur_weight_filename"],
                                 variante.spec.weight_filename)
                self.assertEqual(prov["main_adapter"], adapter)
                self.assertEqual(prov["deblur_lora_mode"], modo)
                self.assertEqual(prov["adapter_name"], ADAPTER_NAME)
                self.assertEqual(prov["prompt"], PROMPT)

    def test_lora_mode_nunca_e_vazio(self):
        """`_REQUIRED_PROVENANCE` valida por truthiness (`dataio/sample.py:249`), e
        `main_adapter=None` é o valor CORRETO da variante oficial. Quem quiser exigir
        campo obrigatório exige `deblur_lora_mode`, que nunca é falsy."""
        amb = _Ambiente(self)
        for variante in DeblurVariant:
            prov = amb.runtime(variante).provenance()
            self.assertTrue(prov["deblur_lora_mode"])
            self.assertTrue(prov["deblur_variant"])
            self.assertTrue(prov["deblur_lora_sha256"])

    def test_nenhum_par_de_variantes_produz_a_mesma_proveniencia(self):
        """`deblur_variant` e `deblur_repo_id` diferem em TODO par — é o que permite a
        `_confere_variante_uniforme` (`run_route_b.py`) recusar um lote misto.

        E o teste registra o que NÃO distingue, de propósito: as duas variantes nossas
        têm o mesmo `deblur_weight_filename` e o mesmo `deblur_lora_sha256` (é um único
        arquivo em disco), e as duas cond-only têm o mesmo `main_adapter` e o mesmo
        `deblur_lora_mode`. Um checador de uniformidade que olhasse para qualquer um
        desses campos em vez de `deblur_variant` aprovaria um release misto.
        """
        amb = _Ambiente(self)
        provs = {v: amb.runtime(v).provenance() for v in DeblurVariant}
        for a, b in itertools.combinations(DeblurVariant, 2):
            for chave in ("deblur_variant", "deblur_repo_id"):
                with self.subTest(par=(a.value, b.value), chave=chave):
                    self.assertNotEqual(provs[a][chave], provs[b][chave],
                                        f"{chave} não distingue as variantes")

        nossas = (provs[DeblurVariant.OURS_MAIN_COND],
                  provs[DeblurVariant.OURS_COND_ONLY])
        self.assertEqual(nossas[0]["deblur_weight_filename"],
                         nossas[1]["deblur_weight_filename"])
        self.assertEqual(nossas[0]["deblur_lora_sha256"],
                         nossas[1]["deblur_lora_sha256"])
        self.assertNotEqual(nossas[0]["main_adapter"], nossas[1]["main_adapter"],
                            "o que separa as duas é o roteamento, e só ele")

    def test_fingerprint_do_backbone_diz_o_que_e(self):
        """Não é hash de conteúdo dos pesos (dezenas de GB). A chave diz isso, para a
        proveniência não afirmar mais do que verificou."""
        amb = _Ambiente(self)
        prov = amb.runtime(DeblurVariant.OURS_MAIN_COND).provenance()
        self.assertEqual(prov["flux_backbone_fingerprint_kind"],
                         "paths_sizes_and_small_configs")
        self.assertEqual(len(prov["flux_backbone_fingerprint"]), 64)

    def test_fingerprint_muda_se_o_snapshot_mudar(self):
        amb = _Ambiente(self)
        antes = amb.runtime(DeblurVariant.OURS_MAIN_COND).provenance()
        (amb.flux_dir / "model_index.json").write_text('{"_class_name":"Outra"}')
        depois = amb.runtime(DeblurVariant.OURS_MAIN_COND).provenance()
        self.assertNotEqual(antes["flux_backbone_fingerprint"],
                            depois["flux_backbone_fingerprint"])

    def test_fingerprint_ignora_o_cache_do_hf(self):
        """Mesmo motivo de `segmentation._model_files`: `.cache/huggingface/...`
        carrega etag e horário e varia entre máquinas para o MESMO snapshot."""
        amb = _Ambiente(self)
        antes = amb.runtime(DeblurVariant.OURS_MAIN_COND).provenance()
        sujeira = amb.flux_dir / ".cache" / "huggingface" / "download"
        sujeira.mkdir(parents=True)
        (sujeira / "model_index.json.metadata").write_text("etag-2026-09-10")
        depois = amb.runtime(DeblurVariant.OURS_MAIN_COND).provenance()
        self.assertEqual(antes["flux_backbone_fingerprint"],
                         depois["flux_backbone_fingerprint"])

    def test_registra_a_assinatura_do_generate_que_rodou(self):
        amb = _Ambiente(self)
        prov = amb.runtime(DeblurVariant.OURS_MAIN_COND).provenance()
        self.assertTrue(prov["genfocus_generate_accepts_main_adapter"])
        self.assertTrue(prov["genfocus_generate_main_adapter_default_is_none"])
        self.assertTrue(prov["genfocus_generate_swallows_unknown_kwargs"])
        self.assertEqual(len(prov["genfocus_pipeline_sha256"]), 64)

    def test_registra_passos_semente_e_resolucao_processada(self):
        amb = _Ambiente(self)
        prov = amb.runtime(DeblurVariant.OURS_MAIN_COND).provenance()
        self.assertEqual(prov["num_inference_steps"], 28)
        self.assertEqual(prov["seed"], 42)
        self.assertEqual(prov["long_side"], 0)
        self.assertEqual(prov["resize_policy"], "no_crop_multiple_of_16")


# ==============================================================================
# 5. Geometria — o recorte por `long_side`, medido e nunca silencioso
# ==============================================================================

def _oficial(w: int, h: int, long_side: int) -> tuple[int, int]:
    """Reimplementação literal de `Inference_deblurNet.py:13-49`, só para comparar."""
    if long_side and long_side > 0:
        if w >= h:
            nw, nh = long_side, int(h * (long_side / w))
        else:
            nh, nw = long_side, int(w * (long_side / h))
        return max((nw // 16) * 16, 16), max((nh // 16) * 16, 16)
    return ((w + 15) // 16) * 16, ((h + 15) // 16) * 16


_RESOLUCOES = [(4032, 3024), (3024, 4032), (3000, 2000), (2000, 3000), (1600, 1067),
               (2048, 1365), (5184, 3456), (1500, 1000), (1920, 1080), (1023, 682)]


class Geometria(unittest.TestCase):
    def test_politica_default_nunca_recorta(self):
        for w, h in _RESOLUCOES:
            for ls in (0, 512, 768, 1024):
                with self.subTest(res=f"{w}x{h}", long_side=ls):
                    plano = plan_processing((h, w), long_side=ls)
                    self.assertFalse(plano.crop_applied)
                    self.assertIsNone(plano.crop_box)
                    self.assertEqual(plano.fov_retained_hw, (1.0, 1.0))

    def test_sem_recorte_o_round_trip_e_identidade_exata(self):
        """`y_out = y_orig`, zero exato — não "aproximadamente zero"."""
        for w, h in _RESOLUCOES:
            for ls in (0, 512, 1024):
                plano = plan_processing((h, w), long_side=ls)
                self.assertEqual(plano.max_registration_shift_px, 0.0,
                                 f"{w}x{h} long_side={ls}")

    def test_processada_e_sempre_multiplo_de_16(self):
        for w, h in _RESOLUCOES:
            for ls in (0, 512, 768):
                for politica in ResizePolicy:
                    plano = plan_processing((h, w), long_side=ls, policy=politica)
                    ph, pw = plano.processed_hw
                    self.assertEqual((ph % 16, pw % 16), (0, 0), f"{ph}x{pw}")
                    self.assertGreaterEqual(min(ph, pw), 16)

    def test_long_side_zero_coincide_com_o_oficial_nas_duas_politicas(self):
        """`long_side=0` é o que a rota B usa (`route_b.py:104-109`) e o default oficial
        (`Inference_deblurNet.py:57`). Aí não há divergência nenhuma a declarar."""
        for w, h in _RESOLUCOES:
            ofic_w, ofic_h = _oficial(w, h, 0)
            for politica in ResizePolicy:
                plano = plan_processing((h, w), long_side=0, policy=politica)
                self.assertEqual(plano.processed_hw, (ofic_h, ofic_w), f"{w}x{h}")
                self.assertFalse(plano.crop_applied)

    def test_a_politica_oficial_reproduz_a_aritmetica_oficial(self):
        for w, h in _RESOLUCOES:
            for ls in (512, 768, 1024):
                ofic_w, ofic_h = _oficial(w, h, ls)
                plano = plan_processing((h, w), long_side=ls,
                                        policy=ResizePolicy.OFFICIAL_CENTER_CROP_16)
                self.assertEqual(plano.processed_hw, (ofic_h, ofic_w),
                                 f"{w}x{h} long_side={ls}")

    def test_recorte_oficial_perde_fov_e_desloca_o_par(self):
        """Os números medidos que estão no docstring do módulo.

        O gate do pipeline antigo era `--max-pair-shift-px 6.0` (`route_b.py:411-422`);
        30,86 px num 5184x3456 é 5,1x isso — e o código novo não tem esse gate ainda.
        """
        casos = [
            # (w, h, long_side, processada (h, w), shift esperado em px)
            (4032, 3024, 512, (384, 512), 0.00),
            (3000, 2000, 512, (336, 512), 17.86),
            (2000, 3000, 512, (512, 336), 17.86),
            (5184, 3456, 512, (336, 512), 30.86),
            (2048, 1365, 768, (496, 768), 22.02),
            (3000, 2000, 1024, (672, 1024), 14.88),
        ]
        for w, h, ls, processada, shift in casos:
            with self.subTest(res=f"{w}x{h}", long_side=ls):
                plano = plan_processing((h, w), long_side=ls,
                                        policy=ResizePolicy.OFFICIAL_CENTER_CROP_16)
                self.assertEqual(plano.processed_hw, processada)
                self.assertAlmostEqual(plano.max_registration_shift_px, shift, places=1)
                self.assertEqual(plano.crop_applied, shift > 0.0)

    def test_recorte_registra_caixa_e_fov(self):
        plano = plan_processing((2000, 3000), long_side=512,
                                policy=ResizePolicy.OFFICIAL_CENTER_CROP_16)
        d = plano.to_dict()
        self.assertTrue(d["crop_applied"])
        self.assertEqual(d["crop_box"], [0, 2, 512, 338])
        self.assertAlmostEqual(d["fov_retained_h"], 336 / 341, places=6)
        self.assertEqual(d["fov_retained_w"], 1.0)
        self.assertAlmostEqual(d["max_registration_shift_px"], 17.86, places=1)
        self.assertEqual((d["processed_h"], d["processed_w"]), (336, 512))
        self.assertEqual(d["resize_policy"], "official_center_crop_16")

    def test_resolucao_processada_esta_sempre_na_proveniencia(self):
        """Mesmo quando não houve recorte: "não recortou" é uma afirmação, e afirmação
        por omissão é o que `route_b.py:239-278` fazia (defeito B11)."""
        for politica in ResizePolicy:
            plano = plan_processing((3000, 2000), long_side=0, policy=politica)
            d = plano.to_dict()
            for chave in ("processed_h", "processed_w", "crop_applied", "crop_box",
                          "fov_retained_h", "fov_retained_w",
                          "max_registration_shift_px", "resize_policy", "long_side"):
                self.assertIn(chave, d)
            self.assertFalse(d["crop_applied"])

    def test_recorte_exige_reconhecimento_explicito(self):
        """Nunca recorta em silêncio: a configuração que recorta não constrói sem que
        alguém diga isso em voz alta."""
        amb = _Ambiente(self)
        with self.assertRaises(ValueError) as ctx:
            amb.runtime(DeblurVariant.OURS_MAIN_COND, long_side=512,
                        resize_policy=ResizePolicy.OFFICIAL_CENTER_CROP_16)
        msg = str(ctx.exception)
        self.assertIn("RECORTA", msg)
        self.assertIn("acknowledge_fov_crop", msg)
        self.assertIn("30,86", msg)

    def test_recorte_reconhecido_constroi_e_registra(self):
        amb = _Ambiente(self)
        rt = amb.runtime(DeblurVariant.OURS_MAIN_COND, long_side=512,
                         resize_policy=ResizePolicy.OFFICIAL_CENTER_CROP_16,
                         acknowledge_fov_crop=True)
        self.assertEqual(rt.provenance()["resize_policy"], "official_center_crop_16")
        plano = rt.plan_for((2000, 3000))
        self.assertTrue(plano.crop_applied)
        self.assertGreater(plano.max_registration_shift_px, 6.0)

    def test_politica_sem_recorte_nao_exige_reconhecimento(self):
        amb = _Ambiente(self)
        rt = amb.runtime(DeblurVariant.OURS_MAIN_COND, long_side=512)
        self.assertFalse(rt.plan_for((2000, 3000)).crop_applied)

    def test_politica_oficial_com_long_side_zero_nao_exige_reconhecimento(self):
        """Sem `long_side` o caminho oficial também não recorta (:43-49)."""
        amb = _Ambiente(self)
        rt = amb.runtime(DeblurVariant.OURS_MAIN_COND, long_side=0,
                         resize_policy=ResizePolicy.OFFICIAL_CENTER_CROP_16)
        self.assertFalse(rt.plan_for((2000, 3000)).crop_applied)

    def test_no_tiled_denoise_bate_com_o_oficial(self):
        """`min(w, h) < 512` — `Inference_deblurNet.py:95`."""
        self.assertTrue(plan_processing((400, 800)).no_tiled_denoise)
        self.assertFalse(plan_processing((3000, 2000)).no_tiled_denoise)
        self.assertTrue(plan_processing((3000, 2000), long_side=512).no_tiled_denoise)

    def test_plano_rejeita_resolucao_invalida(self):
        for hw in ((0, 100), (100, 0), (-1, 10)):
            with self.assertRaises(SampleRejected) as ctx:
                plan_processing(hw)
            self.assertEqual(ctx.exception.reason, "resolution_invalid")

    def test_plano_rejeita_long_side_negativo(self):
        with self.assertRaises(ValueError):
            plan_processing((100, 100), long_side=-1)

    def test_plano_rejeita_long_side_menor_que_16(self):
        """O `max(..., 16)` do oficial elevaria o lado final ACIMA do redimensionado e a
        caixa de recorte sairia negativa. Pedido sem sentido, erro alto."""
        for ls in (1, 8, 15):
            with self.assertRaises(ValueError):
                plan_processing((100, 100), long_side=ls)
            with self.assertRaises(ValueError):
                plan_processing((100, 100), long_side=ls,
                                policy=ResizePolicy.OFFICIAL_CENTER_CROP_16)

    def test_plano_e_congelado(self):
        plano = plan_processing((100, 100))
        self.assertIsInstance(plano, ProcessingPlan)
        with self.assertRaises(Exception):
            plano.long_side = 512                          # type: ignore[misc]


# ==============================================================================
# 6. `infer` de verdade, com um `generate` dublê
#
# Os testes acima leem o fonte; este EXECUTA `infer`. Precisa dos dois: AST prova que o
# kwarg está escrito, execução prova que ele chega. E `compileall`/AST não pegam
# `NameError` — foi assim que um `import json` removido matou um pipeline na primeira
# amostra depois de passar por revisão (CLAUDE.md).
# ==============================================================================

try:
    import numpy as _np
    import torch as _torch                                      # noqa: F401
    from PIL import Image as _PILImage
    _TEM_RUNTIME = True
except Exception:                                              # pragma: no cover
    _TEM_RUNTIME = False


class _GenerateEspiao:
    """Dublê de `generate`. Registra os kwargs e devolve uma imagem do tamanho pedido."""

    def __init__(self):
        self.chamadas: list[dict] = []

    def __call__(self, pipe, **kwargs):
        self.chamadas.append(dict(kwargs))
        largura, altura = int(kwargs["width"]), int(kwargs["height"])
        imagem = _PILImage.new("RGB", (largura, altura), (7, 11, 13))

        class _Saida:
            images = [imagem]

        return _Saida()


class _CondicaoEspia:
    def __init__(self, condition, adapter_setting, position_delta=None,
                 position_scale=1.0):
        self.condition = condition
        self.adapter = adapter_setting


@unittest.skipUnless(_TEM_RUNTIME, "PIL/torch ausentes — `infer` usa torch.no_grad")
class InferenciaComDuble(unittest.TestCase):
    def _preparar(self, variante: DeblurVariant, amb: "_Ambiente" = None,
                  **kwargs) -> tuple:
        amb = amb or _Ambiente(self)
        rt = amb.runtime(variante, **kwargs)
        espiao = _GenerateEspiao()
        sementes: list[int] = []
        # Encurta o `load()`: aqui não há FLUX nem GPU, e o que se testa é o corpo de
        # `infer` — inclusive os nomes que ele referencia.
        rt._pipe = object()
        rt._generate = espiao
        rt._condition_cls = _CondicaoEspia
        rt._seed_everything = sementes.append
        return rt, espiao, sementes

    def test_main_adapter_chega_a_generate_nas_tres_variantes(self):
        """A prova comportamental do conserto do B1: o valor CHEGA, e é o da variante."""
        esperado = {DeblurVariant.OURS_MAIN_COND: "deblurring",
                    DeblurVariant.OURS_COND_ONLY: None,
                    DeblurVariant.OFFICIAL_COND_ONLY: None}
        for variante, adapter in esperado.items():
            with self.subTest(variante=variante.value):
                rt, espiao, _ = self._preparar(variante)
                rt.infer(_np.zeros((512, 512, 3), dtype=_np.uint8))
                self.assertEqual(len(espiao.chamadas), 1)
                kwargs = espiao.chamadas[0]
                self.assertIn("main_adapter", kwargs,
                              "generate chamado sem main_adapter: defeito B1")
                self.assertEqual(kwargs["main_adapter"], adapter)

    def test_o_mesmo_arquivo_de_peso_roteia_diferente_por_variante(self):
        """**A asserção que teria evitado o `b_release`.**

        Um único `deblur.safetensors`, as duas variantes nossas: `ours_cond_only` tem que
        chegar em `generate` com `main_adapter=None` e `ours_main_cond` com
        `"deblurring"`. Sem GPU e sem peso de verdade, interceptando a chamada — é o
        único nível em que isso é observável, porque `generate` termina em `**params` e
        um `main_adapter` errado (ou ausente) não levanta nada: produz AIF macia.

        Nenhum teste de AST veria isto: lá o kwarg é `self._spec.main_adapter`, que está
        certo nas duas. O que estava errado era a VARIANTE selecionada pela rota B.
        """
        amb = _Ambiente(self)
        peso = amb.pesos(DeblurVariant.OURS_COND_ONLY)
        self.assertEqual(peso, amb.pesos(DeblurVariant.OURS_MAIN_COND))

        recebidos = {}
        for variante in (DeblurVariant.OURS_COND_ONLY, DeblurVariant.OURS_MAIN_COND):
            rt, espiao, _ = self._preparar(
                variante, weights_path=peso,
                expected_lora_sha256=(None if variante.spec.pinned_sha256
                                      else amb.sha(peso)),
                amb=amb)
            rt.infer(_np.zeros((64, 64, 3), dtype=_np.uint8))
            self.assertEqual(len(espiao.chamadas), 1)
            recebidos[variante] = espiao.chamadas[0]["main_adapter"]

        self.assertIsNone(recebidos[DeblurVariant.OURS_COND_ONLY],
                          "cond-only rodado como main+cond: é o defeito do b_release")
        self.assertEqual(recebidos[DeblurVariant.OURS_MAIN_COND], ADAPTER_NAME)
        self.assertNotEqual(recebidos[DeblurVariant.OURS_COND_ONLY],
                            recebidos[DeblurVariant.OURS_MAIN_COND],
                            "o MESMO arquivo tem que rotear diferente por variante")

    def test_passa_prompt_passos_e_tiling(self):
        rt, espiao, sementes = self._preparar(DeblurVariant.OURS_MAIN_COND)
        rt.infer(_np.zeros((300, 400, 3), dtype=_np.uint8))
        kwargs = espiao.chamadas[0]
        self.assertEqual(kwargs["prompt"], PROMPT)
        self.assertEqual(kwargs["num_inference_steps"], 28)
        self.assertTrue(kwargs["NO_TILED_DENOISE"], "min(w,h) < 512")
        self.assertEqual(sementes, [42], "seed_everything(42) — igual ao oficial")
        self.assertEqual(kwargs["height"], 304)      # 300 -> ceil(16)
        self.assertEqual(kwargs["width"], 400)
        self.assertEqual(kwargs["conditions"][0].adapter, ADAPTER_NAME)

    def test_aif_volta_na_resolucao_da_bokeh(self):
        """K vive na escala de pixel da imagem fonte: as duas têm que coincidir."""
        for hw in ((300, 400), (512, 512), (1000, 1500)):
            rt, _, _ = self._preparar(DeblurVariant.OURS_MAIN_COND)
            resultado = rt.infer(_np.zeros((*hw, 3), dtype=_np.uint8))
            self.assertEqual(resultado.aif_rgb.shape, (*hw, 3))
            self.assertEqual(resultado.aif_rgb.dtype, _np.uint8)

    def test_a_geometria_vai_na_proveniencia_de_cada_amostra(self):
        rt, _, _ = self._preparar(DeblurVariant.OURS_MAIN_COND)
        resultado = rt.infer(_np.zeros((2000, 3000, 3), dtype=_np.uint8))
        prov = resultado.provenance
        self.assertEqual(prov["deblur_variant"], "ours_main_cond")
        self.assertEqual(prov["main_adapter"], "deblurring")
        geo = prov["geometry"]
        self.assertEqual((geo["image_h"], geo["image_w"]), (2000, 3000))
        self.assertEqual((geo["processed_h"], geo["processed_w"]), (2000, 3008))
        self.assertFalse(geo["crop_applied"])
        self.assertEqual(geo["max_registration_shift_px"], 0.0)

    def test_com_recorte_a_aif_volta_no_tamanho_e_o_recorte_fica_gravado(self):
        rt, espiao, _ = self._preparar(
            DeblurVariant.OURS_MAIN_COND, long_side=512,
            resize_policy=ResizePolicy.OFFICIAL_CENTER_CROP_16,
            acknowledge_fov_crop=True,
        )
        resultado = rt.infer(_np.zeros((2000, 3000, 3), dtype=_np.uint8))
        self.assertEqual(resultado.aif_rgb.shape, (2000, 3000, 3))
        self.assertEqual((espiao.chamadas[0]["height"],
                          espiao.chamadas[0]["width"]), (336, 512))
        geo = resultado.provenance["geometry"]
        self.assertTrue(geo["crop_applied"])
        self.assertEqual(geo["crop_box"], [0, 2, 512, 338])
        self.assertGreater(geo["max_registration_shift_px"], 6.0)

    def test_conta_as_chamadas(self):
        rt, _, _ = self._preparar(DeblurVariant.OURS_MAIN_COND)
        for _ in range(3):
            rt.infer(_np.zeros((64, 64, 3), dtype=_np.uint8))
        self.assertEqual(rt.calls, 3)

    def test_rejeita_entrada_que_nao_e_hxwx3(self):
        rt, _, _ = self._preparar(DeblurVariant.OURS_MAIN_COND)
        for forma in ((64, 64), (64, 64, 1), (64, 64, 4), (3, 64, 64, 3)):
            with self.assertRaises(SampleRejected) as ctx:
                rt.infer(_np.zeros(forma, dtype=_np.uint8))
            self.assertEqual(ctx.exception.reason, "resolution_invalid")

    def test_generate_devolvendo_tamanho_errado_e_erro(self):
        """Se `generate` devolver outra resolução, a geometria gravada descreveria
        outra imagem. Erro alto, não `resize` silencioso."""
        rt, _, _ = self._preparar(DeblurVariant.OURS_MAIN_COND)

        def _errado(pipe, **kwargs):
            class _Saida:
                images = [_PILImage.new("RGB", (32, 32))]
            return _Saida()

        rt._generate = _errado
        with self.assertRaises(RuntimeError) as ctx:
            rt.infer(_np.zeros((64, 64, 3), dtype=_np.uint8))
        self.assertIn("geometria", str(ctx.exception))


if __name__ == "__main__":
    unittest.main(verbosity=2)
