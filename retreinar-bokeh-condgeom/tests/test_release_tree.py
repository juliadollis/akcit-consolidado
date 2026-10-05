"""L1 — ler o release na forma que o `bokehnet-regen` de fato publica.

Sem GPU e sem rede. Todas as fixtures montam a ÁRVORE DE ARQUIVOS de mentira em
`tmp_path`, com os mesmos nomes e as mesmas chaves que
`bokehnet-regen/src/dataio/writer.py:FileSampleWriter` escreve — o valor de um
teste destes é exatamente o quanto a fixture se parece com o release real, então
as chaves foram copiadas de lá e não inventadas.

Cada teste trava um item nomeado da lacuna L1 da `AUDITORIA_TREINO_BOKEHNET.md`.
"""

from __future__ import annotations

import hashlib
import io
import json
import os

import numpy as np
import pytest
from PIL import Image

from genfocus_train import control, data, release


# =============================================================================
# Fixtura: uma árvore de release de mentira
# =============================================================================

#: Resolução da imagem. Quadrada e múltipla do `image_size` do teste de
#: propósito: com `short_side` a escala fica exatamente 0,5 e o crop central é a
#: imagem inteira, então dá para conferir o mapa de defocus contra uma conta
#: feita à mão, sem reimplementar `_plano_geometrico` dentro do teste.
IMG_HW = (64, 64)
#: A profundidade é gravada numa grade MENOR que a imagem, como no release real
#: (lado longo 768 contra fotos de 1500×2000). 16 -> 32 é uma ampliação exata por
#: 2, e nearest exato por 2 é `np.repeat`, o que dá o oráculo independente.
DEPTH_HW = (16, 16)
IMAGE_SIZE = 32
DISP_MIN, DISP_MAX = 0.01, 1.0


def _png_u16(path, array: np.ndarray) -> None:
    """PNG uint16, o mesmo `I;16` que `writer.py` grava.

    Sem passar `mode=` de propósito: o Pillow deduz `I;16` do dtype, e o
    parâmetro `mode` está deprecado (sai no Pillow 13). O que importa para o
    teste é o dtype que volta da leitura, não como ele foi escrito — e é por
    dtype que `control.decode_disparity_u16` decide dividir por 65535.
    """
    imagem = Image.fromarray(array.astype(np.uint16))
    assert imagem.mode == "I;16", imagem.mode
    imagem.save(path)


def _jpeg_bytes(seed: int, hw=IMG_HW) -> bytes:
    rng = np.random.default_rng(seed)
    array = rng.integers(0, 256, size=(hw[0], hw[1], 3), dtype=np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format="JPEG", quality=95, subsampling=0)
    return buffer.getvalue()


def _disparidade_u16(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 65536, size=DEPTH_HW, dtype=np.uint16)


def _linha_de_manifesto(
    sample_id: str, scene_id: str, route: str, meta: dict, split: str
) -> dict:
    """As chaves do `writer.py:FileSampleWriter.write`, na mesma ordem.

    `split` vem do split MATERIALIZADO, como no writer real
    (`split_side = self.split.of(sample.refs.scene_id)`). Fixar `"train"` aqui
    produziria um release que o próprio `check_no_leak` reprovaria — e um teste
    montado sobre um release inválido não prova nada sobre um válido.
    """
    return {
        "sample_id": sample_id, "route": route, "scene_id": scene_id,
        "split": split,
        "source_dataset": meta["source_dataset"],
        "source_sample_id": meta["source_sample_id"],
        "k_value": meta["k_value"], "k_source": meta["k_source"],
        "focus_disparity": meta["focus_disparity"],
        "image_h": meta["image_h"], "image_w": meta["image_w"],
        "depth_h": meta["depth_h"], "depth_w": meta["depth_w"],
        "max_coc": meta["max_coc"],
        "is_k_censored": meta["is_k_censored"],
        "is_valid_for_control": meta["is_valid_for_control"],
        "calibration_ssim": meta["calibration_ssim"],
        "mask_source": meta["mask_source"],
        "depth_backend": meta["depth_backend"],
        "control_version": meta["control_version"],
        "focus_source": meta["focus_source"],
        "focus_was_refined": meta["focus_was_refined"],
        "focus_agreement": meta["focus_agreement"],
        "focus_retention_in_region": meta["focus_retention_in_region"],
        "focus_region_area_ratio": meta["focus_region_area_ratio"],
        "focus_retention_h": meta["focus_retention_h"],
        "focus_retention_w": meta["focus_retention_w"],
    }


def _meta(sample_id: str, scene_id: str, *, route: str, k_value: float,
          gerados: list[str], aif_ref, bokeh_ref,
          source_dataset: str, source_sample_id: str) -> dict:
    return {
        "sample_id": sample_id, "route": route,
        "control_version": control.CONTROL_VERSION,
        "source_dataset": source_dataset, "source_sample_id": source_sample_id,
        "scene_id": scene_id, "aif_ref": aif_ref, "bokeh_ref": bokeh_ref,
        "source_split": "train",
        "k_value": float(k_value), "k_source": "eq5_ssim_sweep",
        "focus_disparity": 0.25, "focus_depth_m": 4.0,
        "max_coc": control.MAX_COC,
        "is_k_censored": False, "is_valid_for_control": True,
        "depth_backend": "depth_pro", "calibration_ssim": 0.82,
        "k_analytic": None, "k_effective_factor": 0.9873,
        "mask_source": "birefnet", "mask_area_ratio": 0.3,
        "disparity_min": DISP_MIN, "disparity_max": DISP_MAX,
        "z_min_m": 1.0 / DISP_MAX, "z_max_m": 1.0 / DISP_MIN,
        "image_h": IMG_HW[0], "image_w": IMG_HW[1],
        "depth_h": DEPTH_HW[0], "depth_w": DEPTH_HW[1],
        "depth_encoding": "uint16_linear_in_disparity",
        "focus_source": "birefnet", "focus_was_refined": False,
        "focus_agreement": 0.9, "focus_retention_in_region": 0.8,
        "focus_region_area_ratio": 0.3,
        "focus_retention_h": 64, "focus_retention_w": 64,
        "focus_retention_window_px": 33,
        "provenance": {"pipeline_commit": "abc123", "depth_model_sha256": "de" * 32,
                       "mask_model_sha256": "ma" * 32, "image_h": IMG_HW[0],
                       "image_w": IMG_HW[1]},
        "quality": {}, "generated_images": sorted(gerados), "channel_order": "rgb",
    }


def montar_release(
    raiz,
    *,
    n: int = 2,
    route: str = "c",
    autocontido: bool = True,
    agrupado_por_cena: bool = False,
    com_ledger: bool = True,
    source_dataset: str = "akcit-pixel/RealBokeh",
    cenas_val: tuple[str, ...] = (),
    k_values: tuple[float, ...] | None = None,
) -> dict:
    """Escreve uma árvore de release e devolve o que os testes precisam saber.

    `route` decide quem GERA o quê, seguindo a tabela do
    `bokehnet-regen/src/dataio/sample.py`: C referencia os dois, B gera a AIF, A
    gera a bokeh.
    """
    raiz = str(raiz)
    for sub in ("depth", "mask", "meta", "generated"):
        os.makedirs(os.path.join(raiz, sub), exist_ok=True)

    gerados_por_rota = {"a": ["bokeh"], "b": ["aif"], "c": []}
    gerados = gerados_por_rota[route]

    manifesto, metas, bytes_por_papel, disparidades = [], {}, {}, {}
    cenas = {}
    ledger = []
    for i in range(n):
        scene_id = f"cena_{i}"
        sample_id = f"{route}_{i}"
        cenas[sample_id] = scene_id
        k = (k_values[i] if k_values else 12.0 + i)
        meta = _meta(
            sample_id, scene_id, route=route, k_value=k, gerados=gerados,
            aif_ref=("image_focus" if "aif" not in gerados else None),
            bokeh_ref=("image_blur" if "bokeh" not in gerados else None),
            source_dataset=source_dataset,
            source_sample_id=f"origem_{i}",
        )
        metas[sample_id] = meta
        manifesto.append(_linha_de_manifesto(
            sample_id, scene_id, route, meta,
            "val" if scene_id in cenas_val else "train"))

        def caminho(sub, nome):
            if agrupado_por_cena:
                os.makedirs(os.path.join(raiz, sub, scene_id), exist_ok=True)
                return os.path.join(raiz, sub, scene_id, nome)
            return os.path.join(raiz, sub, nome)

        disp = _disparidade_u16(seed=100 + i)
        disparidades[sample_id] = disp
        _png_u16(caminho("depth", f"{sample_id}.png"), disp)
        Image.fromarray(
            (np.ones(IMG_HW, dtype=np.uint8) * 255)
        ).save(caminho("mask", f"{sample_id}.png"))
        with open(caminho("meta", f"{sample_id}.json"), "w", encoding="utf-8") as h:
            json.dump(meta, h)

        linha_ledger = {"sample_id": sample_id, "source_sample_id": f"origem_{i}",
                        "shard": "data/train-00000.parquet", "row": i}
        for papel in ("aif", "bokeh"):
            dados = _jpeg_bytes(seed=1000 * (i + 1) + (0 if papel == "aif" else 1))
            bytes_por_papel[(sample_id, papel)] = dados
            if papel in gerados:
                with open(caminho("generated", f"{sample_id}_{papel}.jpg"), "wb") as h:
                    h.write(dados)
                continue
            linha_ledger[f"{papel}_sha256"] = hashlib.sha256(dados).hexdigest()
            linha_ledger[f"{papel}_bytes"] = len(dados)
            linha_ledger[f"{papel}_column"] = (
                "image_focus" if papel == "aif" else "image_blur")
            if autocontido:
                pasta = os.path.join(raiz, "source")
                os.makedirs(pasta, exist_ok=True)
                nome = f"{sample_id}_{papel}.jpg"
                with open(os.path.join(pasta, nome), "wb") as h:
                    h.write(dados)
                linha_ledger[f"{papel}_file"] = nome
        if len(linha_ledger) > 4:
            ledger.append(linha_ledger)

    with open(os.path.join(raiz, "manifest.jsonl"), "w", encoding="utf-8") as h:
        for linha in manifesto:
            h.write(json.dumps(linha) + "\n")
    if com_ledger and ledger:
        with open(os.path.join(raiz, "source_images.jsonl"), "w", encoding="utf-8") as h:
            for linha in ledger:
                h.write(json.dumps(linha) + "\n")

    atribuicao = {
        c: ("val" if c in cenas_val else "train") for c in sorted(set(cenas.values()))
    }
    with open(os.path.join(raiz, "split.json"), "w", encoding="utf-8") as h:
        json.dump({"salt": "teste", "val_fraction": 0.5,
                   "counts": {}, "assignment": atribuicao}, h)

    return {"raiz": raiz, "manifesto": manifesto, "metas": metas,
            "bytes": bytes_por_papel, "disparidades": disparidades,
            "cenas": cenas, "atribuicao": atribuicao}


def _runtime(**kwargs) -> data.DatasetRuntimeConfig:
    base = dict(image_size=IMAGE_SIZE, train=False, max_levels_per_scene=None)
    base.update(kwargs)
    return data.DatasetRuntimeConfig(**base)


def _fonte(nome: str):
    from genfocus_train.config import DatasetSourceConfig
    return DatasetSourceConfig(name=str(nome), split="train")


# =============================================================================
# L1 — o leitor da árvore de arquivos
# =============================================================================

def test_le_a_arvore_de_arquivos_que_o_regen_publica(tmp_path):
    """O release é `manifest.jsonl` + `depth/` + `meta/`, não uma tabela.

    Antes desta implementação o dataloader chamava `load_dataset(repo)` e lia
    `registro["aif"]` — os pixels dentro de uma coluna. O release real não tem
    coluna nenhuma.
    """
    montar_release(tmp_path, n=2)
    ds = data.BokehReleaseTreeDataset([_fonte(tmp_path)], _runtime())
    assert len(ds) == 2
    amostra = ds[0]
    assert amostra["aif_image"].shape == (3, IMAGE_SIZE, IMAGE_SIZE)
    assert amostra["bokeh_image"].shape == (3, IMAGE_SIZE, IMAGE_SIZE)
    assert amostra["defocus_map"].shape == (3, IMAGE_SIZE, IMAGE_SIZE)
    assert amostra["id"] == "c_0"


def test_chaves_do_batch_sao_as_mesmas_do_leitor_de_tabela(tmp_path):
    """`collate_strict` recusa batch com chaves heterogêneas.

    Se os dois leitores divergirem numa chave, misturar releases das duas formas
    (ou o replay sintético do T16) morre no collate, e o erro apontaria para o
    lugar errado.
    """
    import ast
    import pathlib

    montar_release(tmp_path, n=1)
    ds = data.BokehReleaseTreeDataset([_fonte(tmp_path)], _runtime())
    obtidas = set(ds[0])

    # As chaves do leitor de tabela são lidas do FONTE, não copiadas para cá:
    # copiá-las faria o teste passar mesmo que os dois divergissem.
    fonte = pathlib.Path(data.__file__).read_text(encoding="utf-8")
    arvore = ast.parse(fonte)
    esperadas = None
    for no in ast.walk(arvore):
        if isinstance(no, ast.ClassDef) and no.name == "BokehMetricDataset":
            for filho in ast.walk(no):
                if isinstance(filho, ast.Return) and isinstance(filho.value, ast.Dict):
                    esperadas = {
                        k.value for k in filho.value.keys
                        if isinstance(k, ast.Constant)
                    }
    assert esperadas, "não achei o dict de retorno do BokehMetricDataset"
    assert obtidas == esperadas


def test_o_K_e_reescalado_tambem_no_leitor_de_arvore(tmp_path):
    """T2 no caminho novo: `CoC_px` escala com a resolução, disparidade não.

    O oráculo é independente do código sob teste: a ampliação nearest de 16 para
    32 é exatamente `np.repeat`, então o mapa esperado sai de uma conta escrita
    aqui, não de uma segunda chamada a `resize_nearest`.
    """
    info = montar_release(tmp_path, n=1, k_values=(20.0,))
    ds = data.BokehReleaseTreeDataset([_fonte(tmp_path)], _runtime())
    amostra = ds[0]

    escala = IMAGE_SIZE / min(IMG_HW)          # 32/64 = 0,5
    k_esperado = 20.0 * escala
    assert float(amostra["k_efetivo"]) == pytest.approx(k_esperado)

    u16 = info["disparidades"]["c_0"].astype(np.float64)
    disp = DISP_MIN + u16 / 65535.0 * (DISP_MAX - DISP_MIN)
    fator = IMAGE_SIZE // DEPTH_HW[0]
    disp_ampliada = np.repeat(np.repeat(disp, fator, axis=0), fator, axis=1)
    esperado = np.clip(
        np.abs(k_esperado * (disp_ampliada - 0.25)) / control.MAX_COC, 0.0, 1.0
    )
    obtido = amostra["defocus_map"][0].numpy()
    assert np.allclose(obtido, esperado, atol=1e-5)

    # E o K CRU daria outro mapa — senão o teste passaria sem a correção.
    cru = np.clip(np.abs(20.0 * (disp_ampliada - 0.25)) / control.MAX_COC, 0.0, 1.0)
    assert not np.allclose(obtido, cru, atol=1e-3)


def test_layout_agrupado_por_cena_e_lido_igual_ao_plano(tmp_path):
    """D-R2: `publish_release.py` reagrupa em `depth/<scene_id>/<id>.png`.

    Ele faz isso porque o Hub recusa mais de 10.000 arquivos por diretório. O
    release em disco fica plano e o publicado fica agrupado — o mesmo release,
    dois caminhos. Ler só um dos dois deixaria metade dos releases ilegíveis.
    """
    plano = tmp_path / "plano"
    agrupado = tmp_path / "agrupado"
    montar_release(plano, n=2)
    montar_release(agrupado, n=2, agrupado_por_cena=True)

    a = data.BokehReleaseTreeDataset([_fonte(plano)], _runtime())[0]
    b = data.BokehReleaseTreeDataset([_fonte(agrupado)], _runtime())[0]
    assert np.allclose(a["defocus_map"].numpy(), b["defocus_map"].numpy())
    assert np.allclose(a["aif_image"].numpy(), b["aif_image"].numpy())


def test_meta_vence_o_manifesto_quando_os_dois_trazem_a_chave(tmp_path):
    """D-R1: o `meta/` é a FONTE; o manifesto é a cópia para varredura."""
    montar_release(tmp_path, n=1)
    arvore = release.ArvoreDeRelease(tmp_path)
    caminho = tmp_path / "meta" / "c_0.json"
    meta = json.loads(caminho.read_text())
    meta["k_value"] = 99.0
    caminho.write_text(json.dumps(meta))
    arvore._meta_cache.clear()
    assert arvore.registro("c_0")["k_value"] == 99.0
    # E o manifesto, que é a cópia, continua com o valor antigo.
    assert arvore.por_id["c_0"]["k_value"] == 12.0


def test_disparity_min_max_so_existem_no_meta(tmp_path):
    """Por que ler o `meta/` por amostra não é opcional.

    `writer.py` não põe `disparity_min`/`disparity_max` na linha do manifesto, e
    sem eles a profundidade uint16 não decodifica. Um leitor que só varresse o
    manifesto não conseguiria montar o mapa.
    """
    montar_release(tmp_path, n=1)
    linha = json.loads(
        (tmp_path / "manifest.jsonl").read_text().splitlines()[0])
    assert "disparity_min" not in linha and "disparity_max" not in linha
    registro = release.ArvoreDeRelease(tmp_path).registro("c_0")
    assert registro["disparity_min"] == DISP_MIN


def test_sample_id_repetido_no_manifesto_e_recusado(tmp_path):
    montar_release(tmp_path, n=1)
    caminho = tmp_path / "manifest.jsonl"
    linha = caminho.read_text().splitlines()[0]
    caminho.write_text(linha + "\n" + linha + "\n")
    with pytest.raises(release.ReleaseError, match="repetidos"):
        release.ArvoreDeRelease(tmp_path)


def test_manifesto_heterogeneo_e_recusado(tmp_path):
    """Chave presente em umas linhas e ausente em outras = lote misto.

    Tratar a ausente como `None` faria um filtro decidir por um valor que
    ninguém mediu — a mesma família do `k = 50,0`.
    """
    montar_release(tmp_path, n=2)
    caminho = tmp_path / "manifest.jsonl"
    linhas = [json.loads(l) for l in caminho.read_text().splitlines()]
    linhas[1].pop("is_valid_for_control")
    caminho.write_text("\n".join(json.dumps(l) for l in linhas) + "\n")
    with pytest.raises(release.ReleaseError, match="lote misto"):
        data.BokehReleaseTreeDataset([_fonte(tmp_path)], _runtime())


# =============================================================================
# L1 — o resolvedor de referências
# =============================================================================

def test_modo_autocontido_le_de_source(tmp_path):
    """Release com `source/`: os pixels estão dentro, nos bytes ORIGINAIS."""
    info = montar_release(tmp_path, n=1, autocontido=True)
    arvore = release.ArvoreDeRelease(tmp_path)
    assert arvore.autocontido
    resolvedor = release.ResolvedorDePixels(arvore)
    for papel in ("aif", "bokeh"):
        resolvido = resolvedor.resolver("c_0", papel)
        assert resolvido.origem == "source"
        assert resolvido.sha256_conferido is True
    assert resolvedor.contagem_por_origem["source"] == 2


def test_rota_b_le_aif_de_generated_e_bokeh_de_source(tmp_path):
    """A tabela do `sample.py`: B GERA a AIF (DeblurNet) e referencia a bokeh."""
    montar_release(tmp_path, n=1, route="b", autocontido=True)
    arvore = release.ArvoreDeRelease(tmp_path)
    resolvedor = release.ResolvedorDePixels(arvore)
    assert resolvedor.resolver("b_0", "aif").origem == "generated"
    assert resolvedor.resolver("b_0", "bokeh").origem == "source"


def test_rota_a_le_bokeh_de_generated_e_aif_de_source(tmp_path):
    """E A é o espelho de B: gera a bokeh (BokehMe) e referencia a AIF."""
    montar_release(tmp_path, n=1, route="a", autocontido=True)
    arvore = release.ArvoreDeRelease(tmp_path)
    resolvedor = release.ResolvedorDePixels(arvore)
    assert resolvedor.resolver("a_0", "bokeh").origem == "generated"
    assert resolvedor.resolver("a_0", "aif").origem == "source"


def test_sha256_divergente_e_recusado(tmp_path):
    """O ledger existe para que o join seja VERIFICÁVEL.

    Sem conferir, "os pixels vieram daquele espelho" é afirmação, não evidência
    — e o K desta amostra foi calibrado contra bytes específicos.
    """
    montar_release(tmp_path, n=1, autocontido=True)
    (tmp_path / "source" / "c_0_aif.jpg").write_bytes(_jpeg_bytes(seed=7777))
    resolvedor = release.ResolvedorDePixels(release.ArvoreDeRelease(tmp_path))
    with pytest.raises(release.ReleaseError, match="sha256"):
        resolvedor.resolver("c_0", "aif")


def test_sem_pixels_e_sem_espelho_o_erro_diz_as_duas_saidas(tmp_path):
    """Falta de dado levanta com MOTIVO NOMEADO, nunca vira imagem substituta."""
    montar_release(tmp_path, n=1, autocontido=False)
    resolvedor = release.ResolvedorDePixels(release.ArvoreDeRelease(tmp_path))
    with pytest.raises(release.ReleaseError) as exc:
        resolvedor.resolver("c_0", "aif")
    texto = str(exc.value)
    assert "mirror_roots" in texto
    assert "--store-source-images" in texto
    assert "akcit-pixel/RealBokeh" in texto


def test_pixel_declarado_em_generated_e_ausente_no_disco_e_erro(tmp_path):
    """"a rota não gera isso" e "o arquivo sumiu" são coisas diferentes.

    Quem decide a primeira é `meta["generated_images"]`; a segunda tem que ser
    erro, senão o leitor cairia no espelho para buscar um pixel que o release
    afirma ter gerado — e aí o par seria (AIF gerada, bokeh de outra origem).
    """
    montar_release(tmp_path, n=1, route="b", autocontido=True)
    os.remove(tmp_path / "generated" / "b_0_aif.jpg")
    resolvedor = release.ResolvedorDePixels(release.ArvoreDeRelease(tmp_path))
    with pytest.raises(release.ReleaseError, match="Release incompleto"):
        resolvedor.resolver("b_0", "aif")


def test_preflight_falha_no_minuto_zero_e_nao_no_step_8000(tmp_path):
    montar_release(tmp_path, n=4, autocontido=False)
    ds = data.BokehReleaseTreeDataset([_fonte(tmp_path)], _runtime())
    with pytest.raises(release.ReleaseError, match="mirror_roots"):
        ds.preflight(n=2)


# =============================================================================
# L1 — o índice do espelho (reuso de `mirror_images.py`)
# =============================================================================

def _montar_espelho(raiz, nomes, *, bytes_por_nome) -> None:
    """Um snapshot parquet com as colunas do espelho real.

    `file_name_base` é a coluna de nome (`mirror_images.NAME_COLUMN`), e
    `image_focus`/`image_blur` são as colunas de pixel que `aif_ref`/`bokeh_ref`
    nomeiam (`sources/realbokeh.py:138-139`).
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    os.makedirs(os.path.join(str(raiz), "data"), exist_ok=True)
    tabela = pa.table({
        "file_name_base": pa.array(nomes),
        "image_focus": pa.array([{"bytes": bytes_por_nome[(n, "aif")], "path": None}
                                 for n in nomes]),
        "image_blur": pa.array([{"bytes": bytes_por_nome[(n, "bokeh")], "path": None}
                                for n in nomes]),
    })
    pq.write_table(tabela, os.path.join(str(raiz), "data", "train-00000.parquet"))


def test_resolve_no_espelho_por_source_sample_id(tmp_path):
    """O caminho completo: `source_sample_id` -> (shard, linha) -> coluna.

    É a lógica do `mirror_images.MirrorIndex.locate`, do lado de leitura.
    """
    info = montar_release(tmp_path / "rel", n=2, autocontido=False)
    espelho = tmp_path / "espelho"
    _montar_espelho(
        espelho, ["origem_0", "origem_1"],
        bytes_por_nome={
            ("origem_0", "aif"): info["bytes"][("c_0", "aif")],
            ("origem_0", "bokeh"): info["bytes"][("c_0", "bokeh")],
            ("origem_1", "aif"): info["bytes"][("c_1", "aif")],
            ("origem_1", "bokeh"): info["bytes"][("c_1", "bokeh")],
        },
    )
    arvore = release.ArvoreDeRelease(tmp_path / "rel")
    resolvedor = release.ResolvedorDePixels(
        arvore, espelhos={"akcit-pixel/RealBokeh": str(espelho)})
    resolvido = resolvedor.resolver("c_1", "bokeh")
    assert resolvido.origem == "espelho"
    # O sha256 do ledger bate: os bytes são os mesmos que rotularam a amostra.
    assert resolvido.sha256_conferido is True
    assert resolvido.imagem.size == (IMG_HW[1], IMG_HW[0])


def test_indice_do_espelho_le_o_cache_gravado_pela_geracao(tmp_path):
    """O cache é COMPARTILHADO com `bokehnet-regen`, e o formato é o de lá.

    `{"shards": [...], "locations": {nome: [shard, linha]}}`. Se este leitor
    inventasse um formato próprio, cada lado reconstruiria o índice do outro — e
    o custo de reconstruir é ler 85 shards.
    """
    raiz = tmp_path / "espelho"
    os.makedirs(raiz)
    (raiz / release.NOME_CACHE_INDICE).write_text(json.dumps({
        "shards": ["data/a.parquet", "data/b.parquet"],
        "locations": {"x": ["data/b.parquet", 7], "y": ["data/a.parquet", 3]},
    }))
    indice = release.IndiceEspelho.carregar(raiz)
    assert len(indice) == 2
    assert indice.locate("x") == release.LocalizacaoNoEspelho("data/b.parquet", 7)
    # A ordem de leitura é (shard, linha) — ver o cabeçalho do módulo.
    assert indice.sort_key("y") < indice.sort_key("x")
    with pytest.raises(KeyError, match="não está no índice"):
        indice.locate("z")


def test_indice_construido_grava_o_formato_que_a_geracao_le(tmp_path):
    info = montar_release(tmp_path / "rel", n=1, autocontido=False)
    espelho = tmp_path / "espelho"
    _montar_espelho(espelho, ["origem_0"], bytes_por_nome={
        ("origem_0", "aif"): info["bytes"][("c_0", "aif")],
        ("origem_0", "bokeh"): info["bytes"][("c_0", "bokeh")],
    })
    release.IndiceEspelho.carregar(espelho)
    payload = json.loads((espelho / release.NOME_CACHE_INDICE).read_text())
    assert set(payload) == {"shards", "locations"}
    assert payload["locations"]["origem_0"] == ["data/train-00000.parquet", 0]


def test_indice_de_outro_snapshot_e_pego_pela_conferencia_do_nome(tmp_path):
    """Índice cacheado de um snapshot remontado troca TODOS os pares.

    E o histograma de rejeição fica limpo, porque nada falha. A defesa é uma
    comparação de string contra a coluna de nome da linha apontada.
    """
    info = montar_release(tmp_path / "rel", n=2, autocontido=False)
    espelho = tmp_path / "espelho"
    _montar_espelho(espelho, ["origem_0", "origem_1"], bytes_por_nome={
        (n, p): info["bytes"][(f"c_{n[-1]}", p)]
        for n in ("origem_0", "origem_1") for p in ("aif", "bokeh")
    })
    (espelho / release.NOME_CACHE_INDICE).write_text(json.dumps({
        "shards": ["data/train-00000.parquet"],
        "locations": {"origem_0": ["data/train-00000.parquet", 1],   # trocado
                      "origem_1": ["data/train-00000.parquet", 0]},
    }))
    resolvedor = release.ResolvedorDePixels(
        release.ArvoreDeRelease(tmp_path / "rel"),
        espelhos={"akcit-pixel/RealBokeh": str(espelho)})
    with pytest.raises(release.ReleaseError, match="outro snapshot"):
        resolvedor.resolver("c_0", "aif")


# =============================================================================
# L1 — a gramática das referências
# =============================================================================

def test_parse_referencia_reconhece_as_tres_gramaticas():
    coluna = release.parse_referencia("image_focus")
    assert coluna.tipo == "coluna" and coluna.coluna == "image_focus"

    linha = release.parse_referencia("atfortes/BokehDiffusion#train[41].image")
    assert linha.tipo == "linha_hf"
    assert linha.dataset == "atfortes/BokehDiffusion"
    assert (linha.split, linha.linha, linha.coluna) == ("train", 41, "image")

    caminho = release.parse_referencia("EBB/train/1234.jpg")
    assert caminho.tipo == "caminho" and caminho.caminho == "EBB/train/1234.jpg"


def test_parse_referencia_malformada_e_erro_e_nao_palpite():
    with pytest.raises(release.ReleaseError, match="malformada"):
        release.parse_referencia("repo/nome#train[abc].image")
    with pytest.raises(release.ReleaseError):
        release.parse_referencia("   ")


def _release_rota_a_com_caminho(tmp_path, *, com_ledger: bool):
    info = montar_release(tmp_path / "rel", n=1, route="a", autocontido=False,
                          com_ledger=com_ledger, source_dataset="genphoto/EBB")
    caminho = tmp_path / "rel" / "meta" / "a_0.json"
    meta = json.loads(caminho.read_text())
    meta["aif_ref"] = "train/0001.jpg"
    caminho.write_text(json.dumps(meta))

    espelho = tmp_path / "espelho"
    os.makedirs(espelho / "train")
    (espelho / "train" / "0001.jpg").write_bytes(info["bytes"][("a_0", "aif")])
    return espelho


def test_referencia_por_caminho_le_do_espelho(tmp_path):
    """Rota A: `aif_ref` é um caminho RELATIVO (`genphoto_ebb.py:374`)."""
    espelho = _release_rota_a_com_caminho(tmp_path, com_ledger=True)
    resolvedor = release.ResolvedorDePixels(
        release.ArvoreDeRelease(tmp_path / "rel"),
        espelhos={"genphoto/EBB": str(espelho)})
    resolvido = resolvedor.resolver("a_0", "aif")
    assert resolvido.origem == "espelho"
    assert resolvido.sha256_conferido is True


def test_sem_sha_no_ledger_a_proveniencia_diz_None_e_nao_False(tmp_path):
    """`None` é "não havia sha para conferir"; `False` seria "não bateu".

    Os dois não podem virar o mesmo valor: o segundo levanta, e o primeiro é uma
    lacuna de evidência que precisa aparecer no resumo — é ela que diz quantos
    pixels entraram sem prova de origem.
    """
    espelho = _release_rota_a_com_caminho(tmp_path, com_ledger=False)
    resolvedor = release.ResolvedorDePixels(
        release.ArvoreDeRelease(tmp_path / "rel"),
        espelhos={"genphoto/EBB": str(espelho)})
    assert resolvedor.resolver("a_0", "aif").sha256_conferido is None
    assert resolvedor.sem_sha_para_conferir == 1


# =============================================================================
# L1 — split por cena, materializado no release (T6)
# =============================================================================

def test_split_json_do_release_e_lido(tmp_path):
    """O formato do `SceneSplit.to_json` — `{"assignment": {cena: lado}}`.

    Era o único dos três formatos que `carregar_split_por_cena` não lia, e é
    justamente o que o release grava.
    """
    montar_release(tmp_path, n=3, cenas_val=("cena_2",))
    caminho = str(tmp_path / "split.json")
    assert data.carregar_split_por_cena(caminho, "train") == {"cena_0", "cena_1"}
    assert data.carregar_split_por_cena(caminho, "val") == {"cena_2"}


def test_particao_vazia_no_split_e_erro(tmp_path):
    montar_release(tmp_path, n=2)
    with pytest.raises(release.ReleaseError, match="nenhuma cena na partição"):
        data.carregar_split_por_cena(str(tmp_path / "split.json"), "val")


def test_split_do_release_e_usado_sem_scene_split_manifest(tmp_path):
    """T6: o split é MATERIALIZADO no release; o treino não precisa copiá-lo.

    Config é editável, some no rsync e diverge entre runs; um arquivo dentro do
    release não.
    """
    montar_release(tmp_path, n=4, cenas_val=("cena_2", "cena_3"))
    ds = data.BokehReleaseTreeDataset([_fonte(tmp_path)], _runtime())
    assert sorted(sid for _, sid in ds.itens) == ["c_0", "c_1"]
    reg = ds.registros_descarte[str(tmp_path)].resumo()
    assert reg["scene_split_excluded"] == 2

    val = data.BokehReleaseTreeDataset(
        [_fonte(tmp_path)], _runtime(scene_split_partition="val"))
    assert sorted(sid for _, sid in val.itens) == ["c_2", "c_3"]


def test_split_gravado_divergente_do_split_json_e_recusado(tmp_path):
    """A versão de leitura do `check_no_leak`.

    Pega o caso em que o release foi gerado com um split e alguém trocou o
    arquivo depois. Nada disso aparece na loss: a validação simplesmente passa a
    medir memorização.
    """
    montar_release(tmp_path, n=2)
    caminho = tmp_path / "split.json"
    payload = json.loads(caminho.read_text())
    payload["assignment"]["cena_1"] = "val"     # o manifesto diz "train"
    caminho.write_text(json.dumps(payload))
    with pytest.raises(release.ReleaseError, match="diverge"):
        release.ArvoreDeRelease(tmp_path).verificar_split()


# =============================================================================
# L1 — escolha do leitor, e o caminho de tabela que tem que continuar existindo
# =============================================================================

def test_formato_auto_detecta_a_arvore_por_evidencia_local(tmp_path):
    montar_release(tmp_path / "rel", n=1)
    os.makedirs(tmp_path / "prefetch")
    runtime = _runtime()
    assert data.formato_do_release(str(tmp_path / "rel"), runtime) == "arvore"
    # Pasta sem manifesto NÃO vira árvore: o diagnóstico certo para ela é o do
    # caminho antigo, não "faltou manifest.jsonl".
    assert data.formato_do_release(str(tmp_path / "prefetch"), runtime) == "tabela"


def test_formato_forcado_vence_a_deteccao(tmp_path):
    montar_release(tmp_path, n=1)
    assert data.formato_do_release(
        str(tmp_path), _runtime(release_format="tabela")) == "tabela"


def test_misturar_as_duas_formas_no_mesmo_estagio_e_erro(tmp_path):
    montar_release(tmp_path / "rel", n=1)
    os.makedirs(tmp_path / "prefetch")
    with pytest.raises(ValueError, match="formatos DIFERENTES"):
        data.construir_dataset_metric(
            [_fonte(tmp_path / "rel"), _fonte(tmp_path / "prefetch")], _runtime())


def test_build_dataset_roteia_o_release_em_arvore(tmp_path):
    """A porta de entrada do treino, ponta a ponta, sem rede."""
    from genfocus_train.config import StageConfig

    montar_release(tmp_path, n=2)
    cfg = StageConfig(datasets=[_fonte(tmp_path)], steps=1, image_size=IMAGE_SIZE)
    ds = data.build_dataset("bokeh", cfg, _runtime())
    assert isinstance(ds, data.BokehReleaseTreeDataset)
    assert ds[1]["bokeh_image"].shape == (3, IMAGE_SIZE, IMAGE_SIZE)


def test_build_dataset_faz_o_preflight_e_falha_na_partida(tmp_path):
    """A falha por espelho ausente tem que ser no minuto 0, não no step 8.000."""
    from genfocus_train.config import StageConfig

    montar_release(tmp_path, n=4, autocontido=False)
    cfg = StageConfig(datasets=[_fonte(tmp_path)], steps=1, image_size=IMAGE_SIZE)
    with pytest.raises(release.ReleaseError, match="mirror_roots"):
        data.build_dataset("bokeh", cfg, _runtime())


def test_pasta_local_sem_manifesto_continua_recusada_no_contrato_novo(tmp_path):
    from genfocus_train.config import StageConfig

    os.makedirs(tmp_path / "prefetch")
    cfg = StageConfig(datasets=[_fonte(tmp_path / "prefetch")], steps=1,
                      image_size=IMAGE_SIZE)
    with pytest.raises(ValueError, match="manifest.jsonl"):
        data.build_dataset("bokeh", cfg, _runtime())


# =============================================================================
# L1 — o conjunto do probe também sai da árvore (T11 dependia do release)
# =============================================================================

def test_build_probe_set_le_a_arvore_e_exclui_o_split_de_treino(tmp_path, monkeypatch):
    """`scripts/build_probe_set.py` tinha o MESMO defeito: lia `registro["aif"]`.

    Sem isto, o probe de controlabilidade — que é a correção de maior retorno da
    auditoria — não teria como ser montado a partir do release publicado, e o
    `train check` reprova o run sem conjunto de probe.
    """
    import importlib.util
    import pathlib
    import sys

    montar_release(tmp_path / "rel", n=4, cenas_val=("cena_2", "cena_3"))
    saida = tmp_path / "probe_set"

    caminho = pathlib.Path(__file__).parent.parent / "scripts" / "build_probe_set.py"
    spec = importlib.util.spec_from_file_location("build_probe_set", caminho)
    modulo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modulo)

    monkeypatch.setattr(sys, "argv", [
        "build_probe_set.py",
        "--release", str(tmp_path / "rel"),
        "--out", str(saida),
        "--n", "2",
        # O probe tem que ser DISJUNTO do treino: excluímos a partição `train`.
        "--particao-excluida", "train",
        # A3 — a grade de saída. O default é 512 (a do treino e o TILE_SIZE da
        # inferência); aqui usamos 32 só para a fixture caber, e o que o teste
        # prova é que o conjunto sai na grade PEDIDA e não na nativa.
        "--image-size", "32",
    ])
    modulo.main()

    gerados = sorted(p.name for p in saida.glob("*.npz"))
    assert gerados == ["cena_2.npz", "cena_3.npz"]
    dados = np.load(saida / "cena_2.npz")

    # ── A3 — o conjunto sai na grade do TREINO, não na nativa do release ─────
    # Antes este teste afirmava `IMG_HW` (a resolução nativa). Era o
    # comportamento, e o comportamento estava errado: o `probe.py` afirma 512² e
    # justifica `NO_TILED_DENOISE=True` com isso; rodar em nativo desalinha
    # latente/AIF/mapa quando H,W não são múltiplos de 16, custa ~40 GB de VRAM
    # numa foto 3MP, e mede o modelo fora da distribuição em que ele treinou.
    assert dados["aif"].shape == (32, 32, 3)
    assert dados["disparity"].shape == (32, 32)
    # `focus_disparity` NÃO é reescalado: disparidade é 1/m e não depende da
    # resolução. Se um dia alguém "corrigir" isto, o plano de foco anda.
    assert float(dados["focus_disparity"]) == pytest.approx(0.25)


def test_A3_probe_set_usa_o_mesmo_plano_geometrico_do_dataloader(tmp_path):
    """A grade do probe e a do treino não podem divergir por construção.

    Não basta o `build_probe_set.py` produzir 512² por conta própria: ele tem que
    usar a MESMA função (`_plano_geometrico`), senão as duas podem divergir numa
    refatoração futura sem nada avisar — que é o modo de falha que este projeto
    já viu no resize e no K.
    """
    import pathlib as _pl

    fonte = (_pl.Path(__file__).resolve().parent.parent
             / "scripts" / "build_probe_set.py").read_text(encoding="utf-8")
    assert "_plano_geometrico" in fonte, (
        "o probe-set tem que derivar a grade da MESMA função do dataloader"
    )
    assert "short_side" in fonte and "train=False" in fonte, (
        "crop CENTRAL e determinístico: o conjunto do probe é FIXO"
    )


# =============================================================================
# D-R3 — baixar não acontece por default
# =============================================================================

def test_download_desligado_por_default(tmp_path):
    """552 GB de egress inexplicado é o motivo, e ele está no módulo.

    Um download que acontece por default é um download que ninguém decidiu.
    """
    chamadas = []

    def baixador_falso(**kwargs):
        chamadas.append(kwargs)
        return str(tmp_path)

    montar_release(tmp_path, n=1)
    release.resolver_raiz_do_release("org/repo-que-nao-existe-em-disco",
                                     baixador=baixador_falso)
    assert chamadas[0]["local_files_only"] is True

    release.resolver_raiz_do_release("org/repo-que-nao-existe-em-disco",
                                     permitir_download=True,
                                     baixador=baixador_falso)
    assert chamadas[1]["local_files_only"] is False


def test_falha_de_snapshot_explica_a_porta(tmp_path):
    def baixador_quebrado(**kwargs):
        raise OSError("não está no cache")

    with pytest.raises(release.ReleaseError) as exc:
        release.resolver_raiz_do_release("org/repo", baixador=baixador_quebrado)
    assert "permitir_download_do_release" in str(exc.value)


def test_diretorio_local_nao_chama_o_baixador(tmp_path):
    montar_release(tmp_path, n=1)

    def baixador_proibido(**kwargs):
        raise AssertionError("não deveria baixar: a origem é um diretório local")

    raiz = release.resolver_raiz_do_release(str(tmp_path), baixador=baixador_proibido)
    assert raiz == tmp_path


# =============================================================================
# Release real da rota A (medido em 2026-09-19): profundidade POR CENA
# =============================================================================

def test_depth_ref_do_metadado_ganha_da_convencao(tmp_path):
    """O release passou a gravar a profundidade por CENA, e a DECLARA.

    Medido no release real: `depth/ebb_002b23c56bb8.png` (sem `_vNN`), com
    `depth_ref: "depth/<cena>.png"` e `depth_layout: "per_scene"` em cada
    `meta/<cena>/<id>.json`. Antes eram 41 cópias idênticas por cena.

    A regra: o que o metadado DIZ ganha. Adivinhar o layout quando o dado o
    declara é a classe de defeito que este projeto persegue.
    """
    import json

    import numpy as np
    from PIL import Image

    from genfocus_train import release as _rel

    raiz = tmp_path / "rel"
    (raiz / "depth").mkdir(parents=True)
    (raiz / "meta" / "cena1").mkdir(parents=True)
    u16 = np.zeros((32, 48), dtype=np.uint16)
    # UMA profundidade, nomeada pela CENA
    Image.fromarray(u16, mode="I;16").save(raiz / "depth" / "cena1.png")

    linhas = []
    for v in range(2):
        sid = f"cena1_v{v:02d}"
        meta = {
            "sample_id": sid, "scene_id": "cena1", "route": "a",
            "control_version": _rel.control.CONTROL_VERSION,
            "k_value": 5.0, "focus_disparity": 0.2,
            "disparity_min": 0.05, "disparity_max": 0.3,
            "image_h": 64, "image_w": 96, "depth_h": 32, "depth_w": 48,
            "max_coc": 100.0, "is_valid_for_control": True, "is_k_censored": False,
            "depth_ref": "depth/cena1.png", "depth_layout": "per_scene",
            "generated_images": ["bokeh"], "split": "train",
        }
        (raiz / "meta" / "cena1" / f"{sid}.json").write_text(
            json.dumps(meta), encoding="utf-8")
        linhas.append({k: meta[k] for k in
                       ("sample_id", "scene_id", "split", "depth_ref")})
    (raiz / "manifest.jsonl").write_text(
        "\n".join(json.dumps(l) for l in linhas) + "\n", encoding="utf-8")
    (raiz / "split.json").write_text(
        json.dumps({"assignment": {"cena1": "train"}}), encoding="utf-8")

    arv = _rel.ArvoreDeRelease.abrir(str(raiz), permitir_download=False)
    # as DUAS variantes apontam para o MESMO arquivo, o da cena
    p0 = arv.caminho_da_profundidade("cena1_v00")
    p1 = arv.caminho_da_profundidade("cena1_v01")
    assert p0 == p1 == raiz / "depth" / "cena1.png"


def test_depth_ref_que_nao_existe_e_erro_e_nao_palpite(tmp_path):
    """Metadado e release discordando é ERRO — não é hora de adivinhar caminho."""
    import json

    from genfocus_train import release as _rel

    raiz = tmp_path / "rel"
    (raiz / "depth").mkdir(parents=True)
    (raiz / "meta").mkdir(parents=True)
    meta = {
        "sample_id": "c_v00", "scene_id": "c", "route": "a",
        "control_version": _rel.control.CONTROL_VERSION,
        "k_value": 5.0, "focus_disparity": 0.2,
        "disparity_min": 0.05, "disparity_max": 0.3,
        "image_h": 64, "image_w": 96,
        "depth_ref": "depth/NAO_EXISTE.png",
        "generated_images": ["bokeh"], "split": "train",
    }
    (raiz / "meta" / "c_v00.json").write_text(json.dumps(meta), encoding="utf-8")
    (raiz / "manifest.jsonl").write_text(
        json.dumps({"sample_id": "c_v00", "scene_id": "c", "split": "train",
                    "depth_ref": "depth/NAO_EXISTE.png"}) + "\n", encoding="utf-8")
    (raiz / "split.json").write_text(
        json.dumps({"assignment": {"c": "train"}}), encoding="utf-8")

    arv = _rel.ArvoreDeRelease.abrir(str(raiz), permitir_download=False)
    with pytest.raises(Exception) as ctx:
        arv.caminho_da_profundidade("c_v00")
    assert "depth_ref" in str(ctx.value)
