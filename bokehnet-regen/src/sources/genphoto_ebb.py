"""Fonte de AIF da rota A — Generative Photography `[80]` + EBB! `[27]`.

**Estas são as duas fontes que o paper nomeia, e são as únicas.** O supplement B.2 diz
*"We draw candidate images from **[80]** and the **EBB [27]** collections"*
(`paper.txt:996`), a §4.1 diz *"∼70K synthetic pairs derived from **[27, 80]**"*
(`paper.txt:527-528`), e a bibliografia fecha os dois números: `[80]` é *Generative
Photography* (`paper.txt:941-943`) e `[27]` é *Rendering natural camera bokeh effect with
deep learning*, que é o **EBB!** (`paper.txt:814-815`).

## DiffCamera é `[69]`, e não é fonte de dado

O pipeline antigo aceitava `{"DiffCamera", "EBB!"}` e **recusava qualquer outra coisa**
(`bokehnet-preprocessing/src/pipelines/route_a.py:32`, `build_route_a_manifest.py:13`),
com as docstrings afirmando fidelidade ao paper. É a divergência nº 1 da auditoria (A1).

DiffCamera é `[69]`: aparece em `paper.txt:115` (linha de baseline da Tab. 1), `225`,
`385`, `498`, `587`, `1019`, `971` e `1050` (*"Additional Comparison with DiffCamera"*),
bibliografia em `paper.txt:914`. **Zero ocorrências em §3.2, na Fig. 3 ou no B.2.** Metade
do pool vinha de um dataset que o paper nunca usou como dado — não muda a matemática do
rótulo, muda o que o modelo vê no pré-treino inteiro, e torna a coluna (a) da ablação da
Tab. 6 incomparável.

Cuidado de leitura, registrado: o `paper.txt` tem **duas numerações de citação**. As
legendas das Figs. 4 e 6 usam a antiga (`DiffCamera [67]`); o corpo e a bibliografia usam
a atual. Nas linhas que importam — `527-528` e `996` — a numeração é a atual.

Por isso `PAPER_SOURCES` é fechado e uma fonte fora dele é **erro de configuração**
(`SystemExit` no entrypoint), não rejeição de amostra: um dataset errado não é um caso a
calibrar no histograma, é um run a não fazer.

## O filtro de nitidez, e por que ele é POR FONTE

`paper.txt:997`: *"similar to the DeblurNet stage, we use Laplacian variance to filter out
blurry examples"*, resultando em *"a refined pool of approximately 1.7K sharp images"*
(`paper.txt:998`). O paper **não diz** se o ranking é por fonte ou global.

O pipeline antigo ranqueava **global**: as duas fontes iam para uma lista só, ordenada por
`laplacian_variance`, e o corte era `candidates[:top_n]`
(`build_route_a_manifest.py:45-46`). Isso é demonstravelmente enviesado, e o nosso próprio
gate já diz por quê (`qc/gates.py:279-284`): a variância do Laplaciano **escala com
resolução e com compressão**, então um ranking único entre EBB! e Generative Photography
seleciona pela fonte de maior resolução, não pela mais nítida.

Aqui o ranking e o corte acontecem **dentro de cada fonte**, com quota declarada por
fonte. A divisão do pool de 1,7K entre as duas fontes o paper não publica (`[A]` A4 da
auditoria) — então ela é um argumento obrigatório, nunca um default: quem roda declara, e
a quota efetiva de cada fonte vai para a proveniência de cada amostra.

E a variância vai gravada **com a resolução em que foi medida** (`sharpness_hw`), pela
regra 3 do contrato: variância de Laplaciano é quantidade que depende da grade, e sem a
grade ao lado ela não é comparável nem com ela mesma.

## O que este módulo NÃO faz

Não estima profundidade, não sorteia K, não renderiza e não decide split. Ele enumera
arquivos, mede nitidez, ordena dentro da fonte, corta, e afirma o sha256 de cada arquivo.
O rótulo inteiro da rota A vive em `routes/route_a.py`.

## O que fica `[A]`, e o que resolveria

* **Qual artefato de `[80]`** — o release público dos autores da Generative Photography,
  ou imagens que nós geramos com o modelo deles. `paper.txt:996` diz *"candidate images
  from [80] ... collections"* e não dá revisão nem contagem. `source_revision` existe
  para carregar a resposta quando ela existir; é argumento e não tem default.
* **Qual lado da EBB!** — ela é pareada (bokeh / nítida) e o paper não diz qual entra no
  pool. Resolve medindo a variância de Laplaciano dos dois lados e vendo qual sobrevive
  ao corte que produz ~1,7K. `source_note` grava a escolha por release.
* **A divisão do pool entre as duas fontes.** Nenhuma medição resolve — o paper não
  publica. Declarar a quota e gravar por amostra qual fonte a gerou é o que dá.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Mapping, Optional, Sequence

import numpy as np

from control.contract import SampleRejected, reject
from qc.metrics import laplacian_variance
from qc.rejection import RejectionLog

#: `[80]` — Generative Photography (`paper.txt:941-943`).
SOURCE_GENERATIVE_PHOTOGRAPHY = "GenerativePhotography"
#: `[27]` — EBB! (`paper.txt:814-815`).
SOURCE_EBB = "EBB!"

#: As duas fontes do §3.2(a), e **só** elas (`paper.txt:996`, `527-528`).
PAPER_SOURCES = frozenset({SOURCE_GENERATIVE_PHOTOGRAPHY, SOURCE_EBB})

#: Nome -> prefixo do `scene_id`. Existe para que dois arquivos com o mesmo nome em
#: fontes diferentes **não colidam** num `sample_id` só — o defeito A9, em que o
#: identificador era o índice na lista e reordenar o manifesto renomeava tudo.
SOURCE_SLUGS = {SOURCE_GENERATIVE_PHOTOGRAPHY: "gp", SOURCE_EBB: "ebb"}

#: O que o paper nomeia como BASELINE e nunca como dado. Aqui para que a mensagem de
#: erro possa dizer o porquê em vez de só recusar.
NOT_A_SOURCE = {
    "DiffCamera": "DiffCamera é [69], linha de baseline (paper.txt:115, 385, 1050). "
                  "Zero ocorrências em §3.2, na Fig. 3 ou no supplement B.2. O pipeline "
                  "antigo a aceitava como fonte — divergência nº 1 da auditoria (A1).",
}

#: Extensões consideradas. `.jpeg` e `.jpg` separados de propósito: um `set` de sufixos
#: é mais auditável que um `glob` com maiúsculas e minúsculas espalhadas.
IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"})

#: Tamanho do pool que o paper publica depois do filtro (`paper.txt:998`). Não é um
#: default de quota: é a âncora contra a qual a soma das quotas declaradas é conferida.
PAPER_POOL_SIZE = 1700


# --------------------------------------------------------------------------------
# Um candidato
# --------------------------------------------------------------------------------

@dataclass(frozen=True)
class AifImage:
    """Uma imagem AIF candidata, com tudo que a rota A precisa saber dela.

    Satisfaz o `AifSource` de `routes/route_a.py` por estrutura, não por herança — a rota
    recebe a fonte por `Protocol`, como a rota C faz com `PairSource`, e não conhece este
    módulo.

    `scene_id` é o id da **imagem**, compartilhado pelas N variantes que a rota A gera
    dela. É a unidade do split: 41 variantes de uma imagem são **uma** cena, e um split
    por amostra as espalharia dos dois lados — vazamento garantido, e a validação mediria
    memorização (defeito A9).
    """

    scene_id: str
    source_dataset: str
    source_sample_id: str
    #: Caminho relativo à raiz da fonte. É REFERÊNCIA: a rota A não regrava a AIF.
    aif_ref: str
    #: Caminho absoluto, para o loader. Não vai para o metadado — caminho de máquina não
    #: é proveniência reprodutível; o sha256 é.
    path: Path
    sha256: str
    image_hw: tuple[int, int]
    laplacian_variance: float
    #: Resolução em que a variância foi medida. Obrigatória: variância de Laplaciano
    #: escala com a grade, e sem ela o número não é comparável nem consigo mesmo.
    sharpness_hw: tuple[int, int]
    #: Revisão/commit/tag da fonte. `None` é legítimo e fica visível como `None` — o
    #: histórico não gravou revisão nenhuma, e é um dos itens que ficou impossível
    #: certificar no dataset publicado.
    source_revision: Optional[str] = None
    source_note: Optional[str] = None
    #: A rota A não herda split de origem: as duas fontes não publicam um.
    source_split: Optional[str] = None

    def to_provenance(self) -> dict:
        return {
            "source_dataset": self.source_dataset,
            "source_sample_id": self.source_sample_id,
            "source_revision": self.source_revision,
            "source_note": self.source_note,
            "aif_ref": self.aif_ref,
            "aif_sha256": self.sha256,
            "aif_image_h": int(self.image_hw[0]),
            "aif_image_w": int(self.image_hw[1]),
            "aif_laplacian_variance": float(self.laplacian_variance),
            "aif_sharpness_h": int(self.sharpness_hw[0]),
            "aif_sharpness_w": int(self.sharpness_hw[1]),
        }


# --------------------------------------------------------------------------------
# Medição de nitidez — injetável, para o teste não precisar de arquivo de imagem
# --------------------------------------------------------------------------------

@dataclass(frozen=True)
class SharpnessMeasurement:
    image_hw: tuple[int, int]
    sharpness_hw: tuple[int, int]
    laplacian_variance: float
    sha256: str


#: Mede um arquivo. Injetável de propósito: o adaptador fica testável sem escrever
#: 1,7 mil JPEGs, e quem quiser medir de outra forma (por exemplo já tendo o array em
#: memória) não precisa tocar na enumeração.
MeasureImage = Callable[[Path], SharpnessMeasurement]


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for bloco in iter(lambda: handle.read(1 << 20), b""):
            digest.update(bloco)
    return digest.hexdigest()


def _require_pillow():
    """PIL sem cv2 — o opencv completo do `~/.local` quebra o container (GLIBC 2.38
    contra 2.35). Mesma razão de `dataio/writer.py:30-39`."""
    try:
        from PIL import Image
    except ImportError as exc:                       # sem fallback
        raise ImportError(
            "Pillow é necessário para ler as imagens da fonte. Instale com "
            "`pip install --no-deps --target <projeto>/.pydeps pillow`."
        ) from exc
    return Image


def load_bgr(path: Path, *, expected_hw: Optional[tuple[int, int]] = None) -> np.ndarray:
    """Lê a imagem em BGR uint8. Rejeita, nunca conserta.

    Dois casos que **não** são conversão silenciosa:

    * **RGBA com alpha < 255** → `source_image_alpha_not_opaque`. `convert("RGB")`
      comporia sobre preto e INVENTARIA pixel na entrada do Depth Pro e do renderer. É o
      mesmo slug e a mesma razão do adaptador do LFDOF (`contract.py:147-150`).
    * **resolução diferente da declarada** → `source_image_unreadable`. Resolução
      heterogênea muda K em pixel sem mudar nada visível no JSON, e na rota A o K é
      justamente `k_per_long_side · max(H, W)`.
    """
    Image = _require_pillow()
    try:
        with Image.open(path) as imagem:
            imagem.load()
            modo = imagem.mode
            if modo in ("RGBA", "LA", "PA"):
                alpha = np.asarray(imagem.convert("RGBA"))[..., 3]
                if int(alpha.min()) < 255:
                    reject("source_image_alpha_not_opaque",
                           f"{path.name}: alpha mínimo {int(alpha.min())} < 255")
            rgb = np.asarray(imagem.convert("RGB"), dtype=np.uint8)
    except SampleRejected:
        raise
    except Exception as exc:                          # arquivo truncado, formato exótico
        reject("source_image_unreadable", f"{path.name}: {type(exc).__name__}: {exc}")
    if rgb.ndim != 3 or rgb.shape[2] != 3 or min(rgb.shape[:2]) < 1:
        reject("source_image_unreadable", f"{path.name}: shape {rgb.shape}")
    if expected_hw is not None and tuple(rgb.shape[:2]) != tuple(expected_hw):
        reject("source_image_unreadable",
               f"{path.name}: {rgb.shape[:2]} != {tuple(expected_hw)} declarado na "
               "enumeração. K vive na escala de pixel desta imagem.")
    return np.ascontiguousarray(rgb[..., ::-1])


def measure_with_pillow(path: Path, *, long_side: Optional[int] = None) -> SharpnessMeasurement:
    """Variância do Laplaciano com Pillow, mais o sha256 do arquivo.

    `long_side` reduz a grade **antes** de medir, e o default é `None` = resolução
    cheia. Reduzir tem um efeito real e declarado: torna a variância comparável ENTRE
    fontes de resoluções diferentes, ao custo de não ser mais a nitidez da imagem que vai
    para o renderer. Como o corte é por fonte de qualquer forma (ver o cabeçalho), o
    default não reduz — e a grade usada vai gravada em `sharpness_hw`.
    """
    Image = _require_pillow()
    bgr = load_bgr(Path(path))
    image_hw = (int(bgr.shape[0]), int(bgr.shape[1]))
    if long_side and max(image_hw) > int(long_side):
        escala = float(long_side) / float(max(image_hw))
        destino = (max(1, round(image_hw[1] * escala)), max(1, round(image_hw[0] * escala)))
        reduzida = np.asarray(
            Image.fromarray(bgr[..., ::-1]).resize(destino, Image.BOX), dtype=np.uint8)
        medida = np.ascontiguousarray(reduzida[..., ::-1])
    else:
        medida = bgr
    return SharpnessMeasurement(
        image_hw=image_hw,
        sharpness_hw=(int(medida.shape[0]), int(medida.shape[1])),
        laplacian_variance=float(laplacian_variance(medida)),
        sha256=sha256_of(Path(path)),
    )


# --------------------------------------------------------------------------------
# Enumeração
# --------------------------------------------------------------------------------

def validate_source_name(source_dataset: str) -> str:
    """Erro de CONFIGURAÇÃO, não rejeição de amostra. Ver o cabeçalho do módulo."""
    if source_dataset in PAPER_SOURCES:
        return source_dataset
    porque = NOT_A_SOURCE.get(source_dataset, "")
    raise ValueError(
        f"fonte {source_dataset!r} não é fonte da rota A. O §3.2(a) usa "
        f"{sorted(PAPER_SOURCES)} (paper.txt:996, 527-528)."
        + (f"\n  {porque}" if porque else ""))


def enumerate_candidates(
    root: str | Path,
    source_dataset: str,
    *,
    log: RejectionLog,
    revision: Optional[str] = None,
    note: Optional[str] = None,
    measure: Optional[MeasureImage] = None,
    sharpness_long_side: Optional[int] = None,
) -> list[AifImage]:
    """Enumera as imagens de UMA fonte, medindo nitidez e sha256 de cada uma.

    `log` é **obrigatório e sem default**, pela mesma razão de `realbokeh.enumerate_pairs`:
    um default `None` transformaria a chamada curta — que é a que todo mundo escreve — em
    descarte silencioso.

    Dedup é por **sha256 do conteúdo**, não por nome: duas cópias do mesmo arquivo com
    nomes diferentes gerariam 2 × 41 variantes da mesma imagem em cenas diferentes, e o
    split as separaria como se fossem cenas distintas — vazamento por duplicata, que é o
    `[A]` A10 da auditoria em miniatura, dentro do próprio pool.
    """
    validate_source_name(source_dataset)
    raiz = Path(root).expanduser().resolve()
    if not raiz.is_dir():
        raise ValueError(f"raiz da fonte {source_dataset!r} não é diretório: {raiz}")

    medidor: MeasureImage = measure or (
        lambda p: measure_with_pillow(p, long_side=sharpness_long_side))
    arquivos = sorted(p for p in raiz.rglob("*")
                      if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)

    encontrados: list[AifImage] = []
    por_hash: dict[str, str] = {}
    slug = SOURCE_SLUGS[source_dataset]
    for caminho in arquivos:
        relativo = caminho.relative_to(raiz).as_posix()
        sample_id_origem = relativo
        try:
            medida = medidor(caminho)
        except SampleRejected as exc:
            log.reject_from(f"{slug}:{relativo}", exc,
                            {"source_dataset": source_dataset})
            continue
        except Exception as exc:
            # Qualquer falha de leitura vira slug registrado. Sem isto, um arquivo
            # truncado mataria a enumeração inteira na imagem 900 de 1.700.
            try:
                reject("source_image_unreadable",
                       f"{relativo}: {type(exc).__name__}: {exc}")
            except SampleRejected as convertida:
                log.reject_from(f"{slug}:{relativo}", convertida,
                                {"source_dataset": source_dataset})
            continue

        if min(medida.image_hw) <= 0 or not np.isfinite(medida.laplacian_variance):
            try:
                reject("source_metadata_field_invalid",
                       f"{relativo}: hw={medida.image_hw}, "
                       f"lapvar={medida.laplacian_variance!r}")
            except SampleRejected as exc:
                log.reject_from(f"{slug}:{relativo}", exc,
                                {"source_dataset": source_dataset})
            continue

        if medida.sha256 in por_hash:
            try:
                reject("source_duplicate_sample",
                       f"{relativo} tem o mesmo sha256 de {por_hash[medida.sha256]!r}: "
                       "duas cenas para a mesma imagem espalhariam as variantes dos dois "
                       "lados do split")
            except SampleRejected as exc:
                log.reject_from(f"{slug}:{relativo}", exc,
                                {"source_dataset": source_dataset})
            continue
        por_hash[medida.sha256] = relativo

        imagem = AifImage(
            # O `scene_id` carrega a fonte e os 12 primeiros dígitos do sha256: estável
            # sob reordenação do diretório (o defeito A9 era o índice na lista) e sem
            # colidir entre fontes.
            scene_id=f"{slug}_{medida.sha256[:12]}",
            source_dataset=source_dataset,
            source_sample_id=sample_id_origem,
            aif_ref=relativo,
            path=caminho,
            sha256=medida.sha256,
            image_hw=medida.image_hw,
            laplacian_variance=medida.laplacian_variance,
            sharpness_hw=medida.sharpness_hw,
            source_revision=revision,
            source_note=note,
        )
        encontrados.append(imagem)
        log.accept(imagem.scene_id, {"source_dataset": source_dataset,
                                     "laplacian_variance": imagem.laplacian_variance,
                                     "image_h": imagem.image_hw[0],
                                     "image_w": imagem.image_hw[1]})
    return encontrados


# --------------------------------------------------------------------------------
# O corte de nitidez — POR FONTE, com quota declarada
# --------------------------------------------------------------------------------

@dataclass(frozen=True)
class SourceCut:
    source_dataset: str
    n_candidates: int
    n_kept: int
    quota: Optional[int]
    min_variance: Optional[float]
    #: Variância do candidato mais fraco que ficou — o corte EFETIVO, que é o número que
    #: o histórico não gravou e que ninguém conseguiu certificar depois.
    cut_variance: Optional[float]
    variance_p05: Optional[float]
    variance_p50: Optional[float]
    variance_p95: Optional[float]
    sharpness_hw_values: tuple[tuple[int, int], ...]

    def to_dict(self) -> dict:
        return {
            "source_dataset": self.source_dataset,
            "n_candidates": self.n_candidates,
            "n_kept": self.n_kept,
            "quota": self.quota,
            "min_variance": self.min_variance,
            "cut_variance_effective": self.cut_variance,
            "variance_p05": self.variance_p05,
            "variance_p50": self.variance_p50,
            "variance_p95": self.variance_p95,
            "sharpness_hw_values": [list(hw) for hw in self.sharpness_hw_values],
            "ranking_scope": "per_source",
            "ranking_scope_evidence": (
                "paper.txt:997 não diz se o ranking é por fonte; global é "
                "demonstravelmente enviesado porque a variância do Laplaciano escala com "
                "resolução e compressão (qc/gates.py:279-284). Defeito A8/D13."),
        }


def rank_and_cut_per_source(
    candidates: Iterable[AifImage],
    *,
    quota: Optional[Mapping[str, int]] = None,
    min_variance: Optional[Mapping[str, float]] = None,
) -> tuple[list[AifImage], list[SourceCut]]:
    """Ordena por nitidez e corta **dentro de cada fonte**. Nunca entre fontes.

    `quota` é por fonte e sem default. A divisão do pool de 1,7K entre `[80]` e a EBB! não
    é publicada (`[A]` A4): um default aqui seria um número inventado governando metade do
    pré-treino. `None` = sem corte por quantidade, só o piso de variância se houver.

    `min_variance` também é por fonte e também `None` por default — é a mesma regra dos
    gates: **limiar não medido não bloqueia**. O corte certo sai do histograma do piloto,
    e o corte EFETIVO de cada fonte fica gravado em `SourceCut.cut_variance`, que é
    exatamente o que o histórico não registrou.

    Empate é desfeito pelo sha256, não pela ordem do filesystem: dois arquivos com a
    mesma variância têm que sair na mesma ordem em qualquer máquina.
    """
    por_fonte: dict[str, list[AifImage]] = {}
    for imagem in candidates:
        validate_source_name(imagem.source_dataset)
        por_fonte.setdefault(imagem.source_dataset, []).append(imagem)

    if quota:
        desconhecidas = sorted(set(quota) - set(PAPER_SOURCES))
        if desconhecidas:
            raise ValueError(f"quota para fonte que não é do paper: {desconhecidas}")
    if min_variance:
        desconhecidas = sorted(set(min_variance) - set(PAPER_SOURCES))
        if desconhecidas:
            raise ValueError(f"piso de variância para fonte que não é do paper: "
                             f"{desconhecidas}")

    mantidas: list[AifImage] = []
    cortes: list[SourceCut] = []
    for fonte in sorted(por_fonte):
        grupo = sorted(por_fonte[fonte],
                       key=lambda im: (-float(im.laplacian_variance), im.sha256))
        piso = (min_variance or {}).get(fonte)
        elegiveis = ([im for im in grupo if float(im.laplacian_variance) >= float(piso)]
                     if piso is not None else list(grupo))
        limite = (quota or {}).get(fonte)
        selecionadas = elegiveis[:int(limite)] if limite is not None else elegiveis

        variancias = np.asarray([im.laplacian_variance for im in grupo], dtype=np.float64)
        p05, p50, p95 = ((float(x) for x in np.percentile(variancias, [5, 50, 95]))
                         if variancias.size else (None, None, None))
        cortes.append(SourceCut(
            source_dataset=fonte,
            n_candidates=len(grupo),
            n_kept=len(selecionadas),
            quota=None if limite is None else int(limite),
            min_variance=None if piso is None else float(piso),
            cut_variance=(float(selecionadas[-1].laplacian_variance)
                          if selecionadas else None),
            variance_p05=p05, variance_p50=p50, variance_p95=p95,
            sharpness_hw_values=tuple(sorted({im.sharpness_hw for im in grupo})),
        ))
        mantidas.extend(selecionadas)
    return mantidas, cortes


def enumeration_summary(
    kept: Sequence[AifImage], log: RejectionLog, cuts: Sequence[SourceCut],
) -> str:
    """O relatório que fecha a enumeração — em CENAS e em fontes.

    Imprime a quota efetiva de cada fonte ao lado da âncora de 1,7K (`paper.txt:998`),
    porque é a soma das quotas que decide se a nossa leitura do paper se sustenta: se o
    pool sobrevivente não chega perto de 1,7K, a leitura está errada (`[A]` A1).
    """
    por_fonte = Counter(im.source_dataset for im in kept)
    linhas = ["", "=" * 62,
              f"  fonte da rota A : {sorted(PAPER_SOURCES)}  "
              "(paper.txt:996, 527-528)",
              f"  imagens (cenas) : {len(kept)}   âncora do paper: ~{PAPER_POOL_SIZE} "
              "(paper.txt:998)"]
    for fonte, n in sorted(por_fonte.items()):
        linhas.append(f"    {fonte:<26} {n:>6}")
    resolucoes = Counter(im.image_hw for im in kept)
    linhas.append("  resoluções (as 5 mais comuns) — decidem o orçamento de disco (F8):")
    for hw, n in resolucoes.most_common(5):
        linhas.append(f"    {hw[0]:>5} x {hw[1]:<5} {n:>6}")
    linhas.append("-" * 62)
    for corte in cuts:
        linhas.append(
            f"  {corte.source_dataset}: {corte.n_kept}/{corte.n_candidates} mantidas · "
            f"quota={corte.quota} · piso={corte.min_variance}")
        linhas.append(
            f"    variância do Laplaciano  p05 {corte.variance_p05}  "
            f"mediana {corte.variance_p50}  p95 {corte.variance_p95}")
        linhas.append(f"    corte EFETIVO desta fonte: {corte.cut_variance}")
        if len(corte.sharpness_hw_values) > 1:
            linhas.append(
                f"    ATENÇÃO: {len(corte.sharpness_hw_values)} resoluções de medição "
                "DENTRO da mesma fonte — a variância não é comparável entre elas. "
                "Considere --sharpness-long-side.")
    linhas.append("  ranking e corte são POR FONTE. Global enviesaria para a fonte de "
                  "maior resolução (A8).")
    linhas.append("=" * 62)
    return "\n".join(linhas) + log.summary()


# --------------------------------------------------------------------------------
# Carregar os pixels
# --------------------------------------------------------------------------------

class AifImageLoader:
    """Carrega a AIF e mantém o ledger de bytes de origem.

    Separado da enumeração de propósito — é a mesma fronteira de
    `realbokeh.py` × `mirror_images.py` (`sources/__init__.py:6-10`): enumerar/validar é
    uma coisa, ler pixel é outra. Aqui as duas vivem no mesmo arquivo porque a origem é
    um diretório local e não há índice de shard a construir; a fronteira continua sendo
    de função, e a rota A recebe só `load_aif`.

    O ledger (`source_images.jsonl`) grava o sha256 do arquivo de origem por amostra. É o
    que `scripts/publish_release.py:129-137` exige para provar contra quais bytes o
    rótulo foi produzido — na rota A a AIF é **referência** e a bokeh é **gerada**, o
    inverso da rota B.
    """

    def __init__(self, *, ledger_path: Optional[str | Path] = None):
        self.ledger_path = None if ledger_path is None else Path(ledger_path)
        self._handle = None
        if self.ledger_path is not None:
            self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
            self._handle = self.ledger_path.open("a", encoding="utf-8")
        self._logged: set[str] = set()

    def __call__(self, image: AifImage) -> np.ndarray:
        bgr = load_bgr(Path(image.path), expected_hw=image.image_hw)
        if self._handle is not None and image.scene_id not in self._logged:
            self._handle.write(json.dumps({
                "scene_id": image.scene_id,
                "role": "aif_reference",
                **image.to_provenance(),
            }, ensure_ascii=False) + "\n")
            self._handle.flush()
            self._logged.add(image.scene_id)
        return bgr

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None
