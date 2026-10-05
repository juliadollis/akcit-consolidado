"""As rotas B e C, exercitadas contra releases sintéticos no formato REAL.

A rota A já rodou em produção (40K steps). B e C nunca foram exercitadas, e elas
diferem da A em pontos que importam:

| | rota A | rota B | rota C |
|---|---|---|---|
| AIF | referência (EBB!/GenPhoto) | **GERADA** pela DeblurNet | referência |
| bokeh | **GERADO** pelo BokehMe | referência (`repo#split[i].col`) | referência |
| `K` | sorteado | **Eq. 3** da EXIF | **Eq. 5**, sweep de SSIM |
| `calibration_ssim` | ausente | presente e **NULA** | preenchida |
| ledger indexado por | `scene_id` | — | `sample_id` |

Cada uma dessas diferenças já causou defeito neste projeto. O que estes testes
travam é que o dataloader lida com as TRÊS formas sem adivinhar nada.

NÃO substituem rodar contra o release real — substituem não ter nada.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
from PIL import Image

from genfocus_train import control, data
from genfocus_train import release as rel


# =============================================================================
# Fixtures: releases sintéticos no formato do `bokehnet-regen`
# =============================================================================

def _depth_u16(h: int, w: int):
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    disp = 0.05 + 0.9 * (xx / max(w - 1, 1)) + 0.03 * np.sin(yy / 11.0)
    dmin, dmax = float(disp.min()), float(disp.max())
    u16 = np.round((disp - dmin) / (dmax - dmin) * 65535).astype(np.uint16)
    return u16, dmin, dmax


def _monta(raiz, rota: str, n_cenas=2, por_cena=2, h=96, w=128,
           ssim_preenchido=True, aif_gerada=False, bokeh_gerado=False):
    """Escreve um release mínimo, mas com o schema REAL de `writer.py`."""
    for sub in ("depth", "meta", "generated"):
        (raiz / sub).mkdir(parents=True, exist_ok=True)
    linhas, assignment, ledger_origem = [], {}, []

    for c in range(n_cenas):
        cena = f"{rota}_cena{c:02d}"
        assignment[cena] = "train"
        u16, dmin, dmax = _depth_u16(h, w)
        Image.fromarray(u16, mode="I;16").save(raiz / "depth" / f"{cena}.png")
        (raiz / "meta" / cena).mkdir(exist_ok=True)
        (raiz / "generated" / cena).mkdir(exist_ok=True)
        ledger_origem.append({
            "scene_id": cena, "role": "aif_reference",
            "source_dataset": "FonteFalsa", "aif_ref": f"imgs/{cena}.png",
            "aif_sha256": "0" * 64,
        })
        for v in range(por_cena):
            sid = f"{cena}_v{v:02d}"
            gerados = []
            if aif_gerada:
                Image.fromarray((np.random.rand(h, w, 3) * 255).astype(np.uint8)).save(
                    raiz / "generated" / cena / f"{sid}_aif.jpg", quality=95)
                gerados.append("aif")
            if bokeh_gerado:
                Image.fromarray((np.random.rand(h, w, 3) * 255).astype(np.uint8)).save(
                    raiz / "generated" / cena / f"{sid}_bokeh.jpg", quality=95)
                gerados.append("bokeh")
            meta = {
                "sample_id": sid, "scene_id": cena, "route": rota,
                "control_version": control.CONTROL_VERSION,
                "k_value": 6.0 + 3.0 * v, "focus_disparity": 0.30,
                "disparity_min": dmin, "disparity_max": dmax,
                "image_h": h, "image_w": w, "depth_h": h, "depth_w": w,
                "max_coc": 100.0, "is_valid_for_control": True,
                "is_k_censored": False,
                "k_source": "eq3_exif" if rota == "b" else "eq5_ssim_sweep",
                "calibration_ssim": (0.4 + 0.3 * v) if ssim_preenchido else None,
                "depth_backend": "depth_pro", "mask_source": "birefnet",
                "depth_ref": f"depth/{cena}.png", "depth_layout": "per_scene",
                "generated_images": gerados, "split": "train",
                "source_dataset": "FonteFalsa", "source_sample_id": sid,
                "aif_ref": f"imgs/{cena}.png",
                "bokeh_ref": None if bokeh_gerado else f"imgs/{sid}_bokeh.png",
            }
            (raiz / "meta" / cena / f"{sid}.json").write_text(
                json.dumps(meta), encoding="utf-8")
            linhas.append({k: meta[k] for k in (
                "sample_id", "scene_id", "route", "split", "depth_ref",
                "k_value", "calibration_ssim", "is_valid_for_control",
                "is_k_censored", "control_version")})

    (raiz / "manifest.jsonl").write_text(
        "\n".join(json.dumps(x) for x in linhas) + "\n", encoding="utf-8")
    (raiz / "split.json").write_text(
        json.dumps({"assignment": assignment}), encoding="utf-8")
    (raiz / "source_images.jsonl").write_text(
        "\n".join(json.dumps(x) for x in ledger_origem) + "\n", encoding="utf-8")

    # espelho: as imagens que o release REFERENCIA
    espelho = raiz.parent / "espelho"
    (espelho / "imgs").mkdir(parents=True, exist_ok=True)
    for c in range(n_cenas):
        cena = f"{rota}_cena{c:02d}"
        Image.fromarray((np.random.rand(h, w, 3) * 255).astype(np.uint8)).save(
            espelho / "imgs" / f"{cena}.png")
        for v in range(por_cena):
            Image.fromarray((np.random.rand(h, w, 3) * 255).astype(np.uint8)).save(
                espelho / "imgs" / f"{rota}_cena{c:02d}_v{v:02d}_bokeh.png")
    return raiz, {"FonteFalsa": str(espelho)}


def _runtime(**kw):
    base = dict(
        image_size=64, train=False, defocus_source="metric_disparity",
        release_format="arvore", scene_split_partition="train",
        max_levels_per_scene=None, permitir_download_do_release=False,
        verificar_sha256_da_origem=False,
    )
    base.update(kw)
    return data.DatasetRuntimeConfig(**base)


# =============================================================================
# ROTA B — AIF gerada pela DeblurNet, bokeh referenciado, SSIM nulo
# =============================================================================

def test_rota_b_aif_vem_de_generated_e_bokeh_do_espelho(tmp_path):
    """Na rota B a AIF é PRODUZIDA por nós e o bokeh é a foto real."""
    raiz, espelhos = _monta(tmp_path / "relB", "b",
                            aif_gerada=True, bokeh_gerado=False)
    ds = data.build_dataset(
        stage="bokeh",
        stage_config=_stage(raiz),
        runtime=_runtime(mirror_roots=espelhos),
    )
    assert len(ds) == 4
    it = ds[0]
    assert it["aif_image"].shape == (3, 64, 64)
    assert it["bokeh_image"].shape == (3, 64, 64)
    assert float(it["defocus_map"].max()) <= 1.0


def test_rota_b_ssim_nulo_nao_descarta_a_rota_inteira(tmp_path):
    """A rota B TEM a coluna `calibration_ssim`, só que NULA em todas as linhas.

    Checar apenas a EXISTÊNCIA da coluna descartaria as 11.635 amostras dela —
    foi o que o smoke pegou uma vez. O limiar do §3.2(c) é do item (c), e a
    rota B deriva o K da Eq. 3, não de sweep.
    """
    raiz, espelhos = _monta(tmp_path / "relB", "b", aif_gerada=True,
                            ssim_preenchido=False)
    ds = data.build_dataset(
        stage="bokeh",
        stage_config=_stage(raiz, min_calibration_ssim=0.6),
        runtime=_runtime(mirror_roots=espelhos, min_calibration_ssim=0.6),
    )
    assert len(ds) == 4, "SSIM nulo não pode zerar a rota B"


# =============================================================================
# ROTA C — os dois lados referenciados, SSIM preenchido, limiar da Eq. 5
# =============================================================================

def test_rota_c_limiar_de_ssim_corta_por_amostra(tmp_path):
    """§3.2(c): o K só vale se o SSIM do sweep passar do limiar.

    As amostras têm ssim 0,4 (v00) e 0,7 (v01). Com limiar 0,6 só as v01 ficam.
    """
    raiz, espelhos = _monta(tmp_path / "relC", "c", ssim_preenchido=True)
    ds = data.build_dataset(
        stage="bokeh",
        stage_config=_stage(raiz, min_calibration_ssim=0.6),
        runtime=_runtime(mirror_roots=espelhos, min_calibration_ssim=0.6),
    )
    assert len(ds) == 2, f"esperado 2 (só as v01), veio {len(ds)}"


def test_rota_c_sem_limiar_mantem_tudo(tmp_path):
    raiz, espelhos = _monta(tmp_path / "relC", "c", ssim_preenchido=True)
    ds = data.build_dataset(
        stage="bokeh", stage_config=_stage(raiz),
        runtime=_runtime(mirror_roots=espelhos),
    )
    assert len(ds) == 4


def test_rota_c_ledger_por_sample_id(tmp_path):
    """A rota C indexa o ledger por `sample_id`; a A, por `scene_id`.

    O leitor indexa pelas DUAS chaves — sem isso, uma das rotas fica sem
    conferência de sha256 e o `verificar_sha256_da_origem: true` mente.
    """
    raiz, _esp = _monta(tmp_path / "relC", "c")
    # reescreve o ledger na forma da rota C
    (raiz / "source_images.jsonl").write_text(
        json.dumps({"sample_id": "c_cena00_v00", "aif_sha256": "a" * 64}) + "\n",
        encoding="utf-8")
    arv = rel.ArvoreDeRelease.abrir(str(raiz), permitir_download=False)
    led = arv.ledger_de_origem()
    assert "c_cena00_v00" in led, "a chave por sample_id (rota C) tem que entrar"


# =============================================================================
# MISTURA B + C — a fase 2
# =============================================================================

def test_fase2_mistura_b_e_c_com_pesos(tmp_path):
    """A proporção entre rotas é DECISÃO, não tamanho relativo de shard (T10)."""
    pesos = data._pesos_por_rota(["b"] * 30 + ["c"] * 10, {"b": 0.5, "c": 0.5})
    assert np.isclose(sum(pesos[:30]), sum(pesos[30:])), (
        "com peso 50/50 as duas rotas têm que ter a mesma massa, "
        "mesmo com 30 contra 10 amostras"
    )


def test_fase2_peso_ausente_e_erro(tmp_path):
    with pytest.raises(ValueError, match="route_weights"):
        data._pesos_por_rota(["b", "c"], {"b": 1.0})


# =============================================================================
# helper
# =============================================================================

def _stage(raiz, **kw):
    from genfocus_train.config import DatasetSourceConfig, StageConfig

    base = dict(
        datasets=[DatasetSourceConfig(name=str(raiz), split="train")],
        steps=10, image_size=64, release_format="arvore",
        scene_split_partition="train", max_levels_per_scene=None,
        permitir_download_do_release=False, verificar_sha256_da_origem=False,
    )
    base.update(kw)
    return StageConfig(**base)


# =============================================================================
# VAZAMENTO — a rota C sai do split `test` da RealBokeh, de onde vêm os benches
# =============================================================================

def test_cenas_de_avaliacao_saem_do_treino(tmp_path):
    """Medido em 2026-09-25: 220 de 220 cenas dos benches estão na rota C.

    `source_split: "test"` no meta de toda amostra da rota C, e os benches
    `bokeh-bench-realbokeh-test` e `-v2` saem do MESMO split. Treinar nelas e
    avaliar nelas mede memorização — e mede para CIMA justo quando o modelo
    piora. Custa 5,3% da rota C tirar.
    """
    raiz, espelhos = _monta(tmp_path / "relC", "c", n_cenas=3, por_cena=2)
    bloqueio = tmp_path / "bloqueadas.json"
    bloqueio.write_text(json.dumps(["c_cena00", "c_cena02"]), encoding="utf-8")

    ds = data.build_dataset(
        stage="bokeh",
        stage_config=_stage(raiz, excluir_cenas_de_avaliacao=str(bloqueio)),
        runtime=_runtime(mirror_roots=espelhos,
                         excluir_cenas_de_avaliacao=str(bloqueio)),
    )
    assert len(ds) == 2, f"só a cena01 devia sobrar; vieram {len(ds)} amostras"


def test_lista_de_bloqueio_ausente_e_erro(tmp_path):
    """Bloqueio que some em silêncio é pior que não ter: o log mente."""
    with pytest.raises(ValueError, match="não existe"):
        data.carregar_cenas_bloqueadas(str(tmp_path / "nao_existe.json"))


def test_lista_de_bloqueio_vazia_e_erro(tmp_path):
    p = tmp_path / "vazia.json"
    p.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="vazia"):
        data.carregar_cenas_bloqueadas(str(p))


def test_lista_aceita_uma_cena_por_linha(tmp_path):
    p = tmp_path / "cenas.txt"
    p.write_text("test_1\ntest_2\n\ntest_3\n", encoding="utf-8")
    assert data.carregar_cenas_bloqueadas(str(p)) == frozenset({"test_1", "test_2", "test_3"})


def test_cobertura_do_espelho_incompleta_falha_no_arranque(tmp_path):
    """Espelho que cobre PARTE do release tem de falhar no minuto 0.

    O `preflight_de_pixels` resolve 8 amostras espalhadas — pega um espelho que
    não foi montado, e NÃO pega um espelho incompleto. Medido em 2026-09-25: o
    snapshot local da RealBokeh tinha 1.257 de 15.423 nomes (5,3%), e a falta só
    apareceu depois de uma tentativa de baixar 44 GB. Um buraco de 5% passaria
    pelas oito amostras e explodiria no step 8.000.
    """
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq

    from genfocus_train import release as rel

    raiz = tmp_path / "relC"
    _monta(raiz, "c", n_cenas=3, por_cena=2)

    # espelho parquet com SÓ UM dos seis nomes
    esp = tmp_path / "espelho_parquet"
    esp.mkdir()
    pq.write_table(
        pa.table({rel.COLUNA_NOME_NO_ESPELHO: ["c_cena00_v00"],
                  "image_focus": [b"x"], "image_blur": [b"y"]}),
        esp / "train-00000.parquet",
    )
    # as refs precisam ser da forma "coluna" para passar pelo índice
    for cena in ("c_cena00", "c_cena01", "c_cena02"):
        for v in ("v00", "v01"):
            f = raiz / "meta" / cena / f"{cena}_{v}.json"
            d = json.loads(f.read_text(encoding="utf-8"))
            d["aif_ref"] = "image_focus"
            d["bokeh_ref"] = "image_blur"
            d["source_sample_id"] = f"{cena}_{v}"
            f.write_text(json.dumps(d), encoding="utf-8")

    arv = rel.ArvoreDeRelease.abrir(str(raiz), permitir_download=False)
    resolv = rel.ResolvedorDePixels(
        arv, espelhos={"FonteFalsa": str(esp)}, permitir_download=False)
    ids = [str(l["sample_id"]) for l in arv.linhas]

    with pytest.raises(rel.ReleaseError, match="NÃO cobre"):
        rel.conferir_cobertura_do_espelho(arv, resolv, ids)


def test_rota_b_aceita_a_linha_zero(tmp_path):
    """`0 or -1` é `-1` em Python: a guarda rejeitava a LINHA 0, que é válida.

    Só UMA amostra em 13.765 endereça a linha 0 do espelho, então isto passou
    por todos os testes sintéticos e apareceu depois de 1.764 steps de smoke em
    GPU. A mensagem de erro saía se contradizendo: "pede a linha 0 de
    'atfortes/BokehDiffusion#train', que tem 15305 linhas".
    """
    from genfocus_train import release as rel

    assert (0 or -1) == -1, "premissa do bug: zero é falsy"

    # a guarda, isolada: a expressão antiga reprovava a linha 0
    def antiga(linha, n):
        return 0 <= int(linha or -1) < n

    def atual(linha, n):
        return linha is not None and 0 <= int(linha) < n

    assert not antiga(0, 15305), "o bug era este"
    assert atual(0, 15305), "a linha 0 tem de passar"
    assert atual(15304, 15305)
    assert not atual(15305, 15305), "fora do fim continua erro"
    assert not atual(None, 15305), "referência sem linha continua erro"
    assert not atual(-1, 15305)

    # e a gramática de referência realmente produz linha=0
    ref = rel.parse_referencia("atfortes/BokehDiffusion#train[0].image")
    assert ref.tipo == "linha_hf" and ref.linha == 0


def test_etiqueta_de_rota_vem_do_manifesto_nao_do_caminho(tmp_path):
    """`route_weights: {b: 0.5, c: 0.5}` tem de casar com o dado real.

    O rótulo vinha de `source.name`, ou seja do CAMINHO do release, então em
    produção ele era '/workspace/releases/rota_c' e a forma óbvia de escrever o
    YAML falhava. Pior: dois releases da mesma rota receberiam rótulos distintos
    e o peso pedido para a rota valeria para cada um por separado.
    """
    assert data._etiqueta_de_rota({"route": "c"}, "/workspace/releases/rota_c") == "c"
    assert data._etiqueta_de_rota({"route": "b"}, "qualquer/coisa") == "b"
    # sem a coluna (release antigo) cai no nome da fonte, sem inventar
    assert data._etiqueta_de_rota({}, "fonte_x") == "fonte_x"
    assert data._etiqueta_de_rota({"route": None}, "fonte_x") == "fonte_x"


def test_erro_de_route_weights_mostra_os_rotulos_reais(tmp_path):
    """Erro que não diz qual rótulo existe obriga a adivinhar o YAML."""
    with pytest.raises(ValueError) as exc:
        data._pesos_por_rota(["c"] * 3 + ["b"] * 2, {"rota_b": 1.0})
    msg = str(exc.value)
    assert "'b': 2" in msg and "'c': 3" in msg, f"faltam os rótulos reais: {msg}"
    assert "rota_b" in msg, "tem de mostrar o que o YAML declarou"


def test_pesos_por_rota_com_as_etiquetas_do_manifesto():
    """b/c com 50/50: massa igual, apesar de contagens diferentes."""
    rotas = ["b"] * 13106 + ["c"] * 12925
    pesos = data._pesos_por_rota(rotas, {"b": 0.5, "c": 0.5})
    assert np.isclose(sum(pesos[:13106]), 0.5)
    assert np.isclose(sum(pesos[13106:]), 0.5)
