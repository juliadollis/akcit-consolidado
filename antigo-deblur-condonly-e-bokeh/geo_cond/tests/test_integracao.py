"""Testes da fiação no genfocus_train. Sem rede, sem GPU.

Existem por causa de um bug real: o `self._geo_init()` foi parar no
`HuggingFaceDeblurDataset` em vez do de bokeh, porque o trecho de código que eu
casei para inserir aparece nas DUAS classes e o replace pegou a primeira. Os 74
testes passaram assim mesmo, porque nenhum instancia o dataset de bokeh com geo
ligado (isso exigiria rede). O erro só apareceu no smoke, no cluster.

A lição: quando a fiação é por edição de texto em vários pontos, o teste tem de
olhar a fiação, não só o comportamento das funções puras.
"""

from __future__ import annotations

import inspect

import pytest

from genfocus_train import data as D
from genfocus_train.config import RuntimeConfig, StageConfig, _as_stage_config


# ------------------------------------------------ mixin nas classes certas

@pytest.mark.parametrize("cls", [D.HuggingFaceBokehDataset, D.LocalBokehFolderDataset])
def test_datasets_de_bokeh_inicializam_o_geo(cls):
    fonte = inspect.getsource(cls.__init__)
    assert "_geo_init()" in fonte, (
        f"{cls.__name__}.__init__ não chama _geo_init(); o `geo_map` vai falhar "
        "em runtime com AttributeError, e só no primeiro batch."
    )
    assert D._GeoMixin in cls.__mro__


def test_dataset_de_deblur_NAO_inicializa_o_geo():
    """O deblur não recebe condicionamento geométrico (plano §5.5: a geometria
    viria da imagem borrada que se quer corrigir). Se aparecer aqui, é o bug."""
    assert "_geo_init()" not in inspect.getsource(D.HuggingFaceDeblurDataset.__init__)
    assert D._GeoMixin not in D.HuggingFaceDeblurDataset.__mro__


# --------------------------------------------- config chega no dataloader

CAMPOS_GEO = ("geo_condition", "geo_escalares", "geo_constantes", "geo_field",
              "geo_sem_escalares", "geo_ruido_controle")


@pytest.mark.parametrize("campo", CAMPOS_GEO)
def test_campo_existe_nas_duas_pontas(campo):
    assert hasattr(StageConfig, "__dataclass_fields__")
    assert campo in StageConfig.__dataclass_fields__, f"falta em StageConfig: {campo}"
    assert campo in D.DatasetRuntimeConfig.__dataclass_fields__, \
        f"falta em DatasetRuntimeConfig: {campo}"


@pytest.mark.parametrize("campo", CAMPOS_GEO)
def test_as_stage_config_nao_ignora_o_campo(campo):
    """A armadilha documentada: StageConfig é montado campo a campo, então uma
    chave do YAML sem a linha correspondente é SILENCIOSAMENTE ignorada."""
    assert campo in inspect.getsource(_as_stage_config), (
        f"_as_stage_config não lê {campo!r}: a chave do YAML seria ignorada em "
        "silêncio, sem erro nenhum."
    )


@pytest.mark.parametrize("campo", ("occlusion_lambda", "occlusion_pool", "geo_branches"))
def test_runtime_config_tem_os_campos_da_perda(campo):
    assert campo in RuntimeConfig.__dataclass_fields__


def test_yaml_com_campos_geo_e_lido_de_ponta_a_ponta():
    cfg = _as_stage_config({
        "datasets": [{"name": "x/y", "split": "train"}],
        "steps": 10,
        "geo_condition": True,
        "geo_escalares": "/tmp/t.jsonl",
        "geo_constantes": {"tau_occlusion": 1.0},
        "geo_field": "depth",
        "geo_sem_escalares": "pula",
        "geo_ruido_controle": True,
    })
    assert cfg.geo_condition is True
    assert cfg.geo_escalares == "/tmp/t.jsonl"
    assert cfg.geo_constantes == {"tau_occlusion": 1.0}
    assert cfg.geo_field == "depth"
    assert cfg.geo_sem_escalares == "pula"
    assert cfg.geo_ruido_controle is True


def test_default_reproduz_o_comportamento_anterior():
    """A condição A da ablação é gratuita: o código novo desligado é o antigo."""
    cfg = _as_stage_config({"datasets": [{"name": "x/y", "split": "train"}], "steps": 10})
    assert cfg.geo_condition is False and cfg.geo_ruido_controle is False
    assert RuntimeConfig().occlusion_lambda == 0.0
    assert RuntimeConfig().geo_branches is False


# ------------------------------------------------- trainer repassa tudo

def _fonte(nome: str) -> str:
    """Lê o módulo como TEXTO, sem importar.

    `trainer` e `models` importam `backbone`, que exige `peft` e o repo Genfocus.
    Importar aqui tornaria estes testes dependentes do ambiente de treino, e o
    ponto deles é justamente rodar em qualquer lugar."""
    import pathlib
    raiz = pathlib.Path(__file__).resolve().parents[2]
    return (raiz / "genfocus_train" / f"{nome}.py").read_text()


def test_trainer_repassa_geo_nos_dois_sites_de_runtime():
    fonte = _fonte("trainer")
    assert fonte.count("geo_sem_escalares=stage_cfg.geo_sem_escalares") == 2, (
        "os dois sites de DatasetRuntimeConfig (treino e smoke) têm de receber "
        "os campos geo; esquecer um faz o smoke passar e o treino divergir."
    )
    assert fonte.count("outputs.weight") == 2, \
        "os dois call sites da perda têm de passar o peso"


def test_models_aceita_geo_e_o_peso():
    fonte = _fonte("models")
    for campo in ("geo_map", "geo_branches", "occlusion_lambda", "occlusion_pool"):
        assert campo in fonte, f"BokehNet.make_train_batch sem {campo}"
    assert "weight: torch.Tensor | None = None" in fonte, \
        "flow_matching_loss tem de aceitar o peso"
    # os 6 canais viram 2 branches de 3, na ordem de geo_cond.signals.CANAIS
    assert "geo_map[:, [2, 3, 4]]" in fonte and "geo_map[:, [0, 1, 5]]" in fonte


# ------------------------------------------ filtro de amostras sem escalar

def test_existe_filtro_de_amostras_sem_escalares():
    """As 11 amostras da rota b sem `foreground_mask` util (0,076%) tem de ser
    descartadas NA CARGA, nao pulando no worker: uma excecao dentro do DataLoader
    derruba o rank inteiro, que foi como isto apareceu ao subir o A'."""
    fonte = inspect.getsource(D._GeoMixin)
    assert "_filtrar_sem_escalares" in fonte
    assert "self.dataset.select(" in fonte, "o filtro tem de reduzir o dataset"
    assert "zerou o dataset" in fonte, (
        "tem de abortar se o filtro zerar tudo, como o filtro de SSIM ja faz; "
        "senao um erro de configuracao viraria um treino silenciosamente vazio"
    )


def test_theta_chega_do_yaml_ate_a_perda():
    """O caminho completo do `occlusion_theta`: RuntimeConfig -> trainer ->
    models -> loss_weight. Cada elo foi editado a mao, entao cada elo e testado."""
    assert "occlusion_theta" in RuntimeConfig.__dataclass_fields__
    assert "occlusion_theta=config.runtime.occlusion_theta" in _fonte("trainer")
    assert "occlusion_theta=occlusion_theta" in _fonte("trainer")
    assert "theta=occlusion_theta" in _fonte("models")
    assert RuntimeConfig().occlusion_theta == 0.0, \
        "o default tem de reproduzir o A' ja treinado"
