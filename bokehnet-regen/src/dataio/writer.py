"""Gravação de amostras em disco.

Duas garantias que o writer antigo não dava:

1. **Nada é normalizado na gravação.** O defeito D1 nasceu de um `dm / dm.max()` que
   vivia num bloco de visualização e virou o dado, apagando o K algebricamente. Aqui
   o writer não faz aritmética sobre o sinal — só codifica o que recebe.

2. **Campo ausente é erro, não omissão.** O writer antigo filtrava valores `None` do
   JSON, então a diferença entre "não medido" e "medido como zero" desaparecia.
   `validate_metadata` roda antes de cada gravação.

O mapa de defocus **não é gravado**. Ele é derivado no dataloader pela mesma função
da geração — ver `dataio.sample` para o porquê.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from dataio.layout import (
    DEFAULT_DEPTH_LAYOUT, DepthLayout, depth_ref, resolve_depth_path,
)
from dataio.sample import Sample, metadata_to_json, validate_metadata
from dataio.split import SceneSplit


def _require_png_writer():
    """PNG uint16 sem cv2, que quebra o container (GLIBC 2.38 contra 2.35)."""
    try:
        from PIL import Image
    except ImportError as exc:                       # sem fallback
        raise ImportError(
            "Pillow é necessário para gravar PNG. Instale com "
            "`pip install --no-deps --target <projeto>/.pydeps pillow`."
        ) from exc
    return Image


@dataclass
class WriteStats:
    samples: int = 0
    bytes_written: int = 0
    #: Quantos PNG de profundidade foram de fato gravados. No layout `per_scene` é o
    #: número de CENAS, não de amostras — e é essa diferença que o run tem que imprimir.
    #: Sem ela, "69.700 amostras gravadas" não distingue 69.700 PNG de 1.700.
    depth_files: int = 0
    #: Quantas vezes uma amostra reusou o PNG já gravado da sua cena.
    depth_reused: int = 0

    def summary(self) -> str:
        if self.samples == 0:
            return "[writer] nada gravado."
        mb = self.bytes_written / 1e6
        linha = (f"[writer] {self.samples} amostras · {mb:.1f} MB · "
                 f"{mb / self.samples:.2f} MB por amostra")
        if self.depth_reused:
            linha += (f"\n[writer] profundidade: {self.depth_files} arquivos para "
                      f"{self.samples} amostras ({self.depth_reused} reusos por cena)")
        return linha


class FileSampleWriter:
    """Um diretório por release, arquivos planos por amostra.

    Layout, com `<id>` = `sample_id`:

        <out>/depth/<cena>.png      disparidade uint16 (ver `dataio.encoding`), UMA vez
                                    por cena no layout `per_scene`; `depth/<id>.png` no
                                    `per_sample`. Quem lê não deduz: cada `meta/` traz
                                    `depth_ref` com o caminho, e `dataio.layout` o
                                    resolve nos dois layouts
        <out>/mask/<id>.png         máscara final de foco, uint8 — na rota C é a
                                    REGIÃO REFINADA, a mesma que produziu
                                    `focus_disparity`; `mask_source` diz qual é
        <out>/meta/<id>.json        TUDO que não é pixel
        <out>/generated/<id>_*.jpg  só o que ESTA rota gera (AIF em B, bokeh em A)
        <out>/manifest.jsonl        uma linha por amostra, para varredura rápida

    **A profundidade é por CENA, e a máscara não.** A profundidade sai da AIF, que as
    variantes de uma cena compartilham; a máscara sai da região de foco, que é o que
    varia entre elas. Deduplicar a máscara junto seria colapsar exatamente o sinal que
    distingue uma variante da outra.
    """

    def __init__(self, output_dir: str | Path, *, split: SceneSplit,
                 jpeg_quality: int = 95,
                 depth_layout: DepthLayout | str = DEFAULT_DEPTH_LAYOUT):
        self.root = Path(output_dir)
        self.jpeg_quality = int(jpeg_quality)
        self.depth_layout = DepthLayout(depth_layout)
        for sub in ("depth", "mask", "meta", "generated"):
            (self.root / sub).mkdir(parents=True, exist_ok=True)

        # O split é MATERIALIZADO no release, não deixado para o config do treino.
        # Config é editável, some no rsync e diverge entre runs; um arquivo no release
        # não. Antes `SceneSplit.save()` existia e ninguém o chamava daqui.
        self.split = split
        self.split_path = split.save(self.root / "split.json")

        self.manifest_path = self.root / "manifest.jsonl"
        self._manifest = self.manifest_path.open("a", encoding="utf-8")
        self._seen: set[str] = self.completed_ids()
        self.stats = WriteStats()

        # A identidade da profundidade de cada cena já gravada, por sha256 do array
        # uint16 — não dos bytes do PNG, que carregam data e nível de compressão.
        # 1.700 entradas de 64 hex para o maior release da rota A: cabe em memória, e é
        # o que permite a checagem de baixo custar um hash de 884 KB em vez de reler o
        # arquivo. `_fonte_por_cena` guarda de que dataset a cena veio, que é o que
        # nomeia a colisão `train_1275` da RealBokeh contra a `train_1275` do LFDOF.
        self._depth_por_cena: dict[str, str] = {}
        self._fonte_por_cena: dict[str, tuple[str, str]] = {}
        self._cena_por_casefold: dict[str, str] = {}

    # -- gravação --------------------------------------------------------------

    def write(self, sample: Sample) -> dict:
        Image = _require_png_writer()
        meta = sample.metadata()
        validate_metadata(meta)                      # antes de tocar no disco

        # Colisão de id gravava 2 linhas de manifesto e 1 arquivo `meta/`, com a
        # segunda sobrescrevendo a primeira em silêncio — a contagem publicada
        # inflava sem que nada denunciasse.
        if sample.sample_id in self._seen:
            raise ValueError(
                f"sample_id repetido: {sample.sample_id!r}. Já gravado neste release."
            )

        # O split da amostra vem do split MATERIALIZADO, por cena. Se a cena não está
        # nele, é erro de manifesto — não caso a resolver em runtime.
        split_side = self.split.of(sample.refs.scene_id)

        # A profundidade é função da CENA. No layout por cena o PNG é gravado uma vez e
        # referenciado pelas N amostras; `_grava_depth` é quem decide, e quem recusa
        # quando duas amostras da mesma cena trazem profundidades diferentes.
        ref, written = self._grava_depth(sample, Image)
        # A referência entra no metadado ANTES da segunda validação: é ela que torna a
        # profundidade alcançável sem convenção, e um `meta/` gravado sem ela num
        # release por cena é uma amostra sem profundidade encontrável.
        meta["depth_ref"] = ref
        meta["depth_layout"] = self.depth_layout.value
        validate_metadata(meta)

        mask_path = self.root / "mask" / f"{sample.sample_id}.png"
        mask_u8 = (np.asarray(sample.mask) > 0.5).astype(np.uint8) * 255
        Image.fromarray(mask_u8, mode="L").save(mask_path, optimize=True)
        written += mask_path.stat().st_size

        # Ordem de canal é DECLARADA, não assumida. A inversão cega `[..., ::-1]`
        # transformava um RGB [200,0,0] em [0,0,200] sem que nada registrasse a
        # convenção — e na rota B a AIF gerada pela DeblurNet é exatamente uma imagem
        # de 3 canais.
        for name, image in sample.generated_images.items():
            path = self.root / "generated" / f"{sample.sample_id}_{name}.jpg"
            array = np.asarray(image)
            if array.ndim == 3 and array.shape[2] == 3:
                if sample.channel_order not in ("bgr", "rgb"):
                    raise ValueError(f"channel_order desconhecido: {sample.channel_order!r}")
                if sample.channel_order == "bgr":
                    array = array[..., ::-1]
            Image.fromarray(np.clip(array, 0, 255).astype(np.uint8)).save(
                path, quality=self.jpeg_quality, subsampling=0)
            written += path.stat().st_size

        meta_path = self.root / "meta" / f"{sample.sample_id}.json"
        meta_path.write_text(metadata_to_json(meta), encoding="utf-8")
        written += meta_path.stat().st_size

        # A linha do manifesto é a superfície de varredura rápida — então ela carrega
        # justamente os campos que denunciariam um fallback: `split`, `mask_source`,
        # `depth_backend`, `control_version` e `max_coc`. Antes ela omitia os cinco.
        self._manifest.write(json.dumps({
            "sample_id": sample.sample_id, "route": sample.route,
            "scene_id": sample.refs.scene_id, "split": split_side,
            # A referência da profundidade viaja no manifesto pelo mesmo motivo que a
            # marcação do refinamento viaja: o manifesto é a superfície de varredura
            # rápida, e conferir que 69.700 amostras têm profundidade alcançável não
            # pode exigir abrir 69.700 JSONs. É por esta chave que a validação do
            # release compara o que existe em `depth/` com o que é reivindicado.
            "depth_ref": ref,
            "source_dataset": sample.refs.source_dataset,
            "source_sample_id": sample.refs.source_sample_id,
            "k_value": meta["k_value"], "k_source": meta["k_source"],
            "focus_disparity": meta["focus_disparity"],
            # K é um número EM PIXEL. Sem a resolução ao lado dele, a linha do
            # manifesto não diz o que o K significa, e quem for reamostrar ou
            # comparar entre rotas não tem como normalizar. É a regra do CLAUDE.md:
            # toda quantidade em pixel carrega a resolução em que foi medida.
            "image_h": meta["image_h"], "image_w": meta["image_w"],
            "depth_h": meta["depth_h"], "depth_w": meta["depth_w"],
            "max_coc": meta["max_coc"],
            "is_k_censored": meta["is_k_censored"],
            "is_valid_for_control": meta["is_valid_for_control"],
            "calibration_ssim": meta["calibration_ssim"],
            "mask_source": meta["mask_source"],
            "depth_backend": meta["depth_backend"],
            "control_version": meta["control_version"],
            # A marcação do refinamento da região em foco viaja no manifesto porque é
            # por ela que se monta o treino COM e SEM as amostras refinadas — e
            # filtrar 22.990 amostras não pode exigir abrir 22.990 JSONs. Se o número
            # de `retention_only` for alto, ele tem que estar visível numa varredura
            # de uma linha, não escondido no metadado.
            "focus_source": meta["focus_source"],
            "focus_was_refined": meta["focus_was_refined"],
            "focus_agreement": meta["focus_agreement"],
            "focus_retention_in_region": meta["focus_retention_in_region"],
            "focus_region_area_ratio": meta["focus_region_area_ratio"],
            # `focus_region_area_ratio` é fração, mas foi medida numa grade específica,
            # e `focus_retention_window_px` é pixel puro. Mesma regra do `image_h`
            # acima: quantidade em pixel viaja com a resolução em que foi medida.
            "focus_retention_h": meta["focus_retention_h"],
            "focus_retention_w": meta["focus_retention_w"],
        }, ensure_ascii=False) + "\n")
        self._manifest.flush()

        self._seen.add(sample.sample_id)
        self.stats.samples += 1
        self.stats.bytes_written += written
        return meta

    # -- profundidade por cena -------------------------------------------------

    def _grava_depth(self, sample: Sample, Image) -> tuple[str, int]:
        """Grava — ou reusa — o PNG de profundidade. Devolve `(depth_ref, bytes)`.

        No layout `per_sample` isto é a gravação de sempre. No `per_scene` a primeira
        amostra de uma cena grava e as seguintes reusam, **depois de três checagens que
        são a razão de este método existir**:

        1. **Colisão entre fontes.** `sources/realbokeh.scene_key` e
           `sources/lfdof.scene_key` são a MESMA função — `f"{split}_{numero}"` —, então
           a cena `train_1275` da RealBokeh e a `train_1275` do LFDOF têm o mesmo
           `scene_id`. Por amostra isso nunca doeu, porque o `sample_id` carrega a fonte
           (`c_realbokeh_…` contra `c_lfdof_…`); por cena as duas escreveriam o mesmo
           `depth/train_1275.png`, e a segunda sobrescreveria a primeira **em silêncio**,
           deixando metade do release com a profundidade da cena errada. Recusa alto.

        2. **Colisão por caixa.** `Train_1` e `train_1` são dois arquivos no Hub e no
           ext4, e **um só** no APFS e no NTFS. Um release gerado no cluster e lido num
           laptop perderia uma das duas cenas na cópia, sem erro nenhum.

        3. **A profundidade não é função da cena.** A premissa inteira da dedup é que as
           N amostras de uma cena têm a MESMA profundidade — medido, 120/120 cenas byte
           a byte na rota A. Se um dia não for (Depth Pro não determinístico, AIF
           trocada, `image_hw` diferente entre variantes), reusar o primeiro PNG rotula
           N−1 amostras com a profundidade de outra imagem. Aqui isso vira erro, não
           rótulo errado.

        O hash é do **array uint16**, não dos bytes do PNG: o PNG carrega nível de
        compressão e pode mudar de bytes sem mudar de conteúdo, e é o conteúdo que o
        rótulo usa.
        """
        cena = sample.refs.scene_id
        ref = depth_ref(sample_id=sample.sample_id, scene_id=cena,
                        layout=self.depth_layout)
        caminho = self.root / ref

        if self.depth_layout is not DepthLayout.PER_SCENE:
            Image.fromarray(sample.depth.disparity_u16, mode="I;16").save(
                caminho, optimize=True)
            self.stats.depth_files += 1
            return ref, caminho.stat().st_size

        digest = _hash_u16(sample.depth.disparity_u16)
        fonte = (sample.refs.source_dataset, sample.sample_id)

        anterior_fonte = self._fonte_por_cena.get(cena)
        if anterior_fonte is not None and anterior_fonte[0] != fonte[0]:
            raise ValueError(
                f"scene_id {cena!r} aparece em duas fontes: {anterior_fonte[0]!r} "
                f"(em {anterior_fonte[1]!r}) e {fonte[0]!r} (em {fonte[1]!r}). No "
                f"layout `per_scene` as duas gravariam {ref!r}, e a segunda apagaria a "
                "profundidade da primeira. `realbokeh.scene_key` e `lfdof.scene_key` "
                "são a mesma função, então `train_1275` existe nas duas fontes e são "
                "cenas físicas diferentes: qualifique o `scene_id` pela fonte antes de "
                "misturá-las num release.")

        anterior_caixa = self._cena_por_casefold.get(cena.casefold())
        if anterior_caixa is not None and anterior_caixa != cena:
            raise ValueError(
                f"scene_id {cena!r} e {anterior_caixa!r} diferem só na caixa. São dois "
                "arquivos no Hub e um só em APFS/NTFS: o release perderia uma das duas "
                "cenas na primeira cópia para um laptop, sem erro nenhum.")

        conhecido = self._depth_por_cena.get(cena)
        if conhecido is None and caminho.is_file():
            # Retomada: o PNG desta cena veio de um run anterior deste mesmo release.
            # Reler e hashear uma vez por cena é o preço de não confiar em memória que
            # o processo anterior levou embora.
            conhecido = _hash_u16(np.asarray(Image.open(caminho)))

        if conhecido is not None:
            if conhecido != digest:
                raise ValueError(
                    f"a profundidade de {sample.sample_id!r} difere da já gravada para "
                    f"a cena {cena!r} em {ref!r}. A dedup por cena se apoia em a "
                    "profundidade ser função da CENA — medido, 120/120 cenas da rota A "
                    "byte a byte idênticas entre as 41 variantes. Aqui não é: reusar o "
                    "PNG rotularia esta amostra com a profundidade de outra imagem. "
                    "Grave este release com `depth_layout='per_sample'` ou descubra "
                    "por que a profundidade mudou dentro da cena.")
            # Memoriza também quando o hash veio do disco, para que a retomada releia
            # o PNG uma vez por cena e não uma vez por amostra.
            self._depth_por_cena[cena] = conhecido
            self._fonte_por_cena.setdefault(cena, fonte)
            self._cena_por_casefold.setdefault(cena.casefold(), cena)
            self.stats.depth_reused += 1
            return ref, 0

        Image.fromarray(sample.depth.disparity_u16, mode="I;16").save(
            caminho, optimize=True)
        self._depth_por_cena[cena] = digest
        self._fonte_por_cena[cena] = fonte
        self._cena_por_casefold[cena.casefold()] = cena
        self.stats.depth_files += 1
        return ref, caminho.stat().st_size

    # -- retomada --------------------------------------------------------------

    def completed_ids(self) -> set[str]:
        """IDs já gravados, lidos do manifesto. Base da retomada."""
        done: set[str] = set()
        if not self.manifest_path.exists():
            return done
        with self.manifest_path.open(encoding="utf-8") as handle:
            for line in handle:
                try:
                    done.add(json.loads(line)["sample_id"])
                except (json.JSONDecodeError, KeyError):
                    continue
        return done

    def close(self) -> None:
        self._manifest.close()

    def __enter__(self) -> "FileSampleWriter":
        return self

    def __exit__(self, *exc_info) -> None:
        print(self.stats.summary())
        self.close()


def _hash_u16(array: np.ndarray) -> str:
    """sha256 do CONTEÚDO da profundidade, não dos bytes do arquivo.

    O shape entra no hash junto com os bytes: dois arrays com os mesmos valores em
    grades diferentes (576x768 e 768x576) têm o mesmo buffer e não são a mesma
    profundidade.
    """
    array = np.ascontiguousarray(np.asarray(array, dtype=np.uint16))
    h = hashlib.sha256(repr(array.shape).encode("ascii"))
    h.update(array.tobytes())
    return h.hexdigest()


def read_metadata(output_dir: str | Path, sample_id: str) -> dict:
    return json.loads((Path(output_dir) / "meta" / f"{sample_id}.json").read_text(encoding="utf-8"))


def read_depth_u16(output_dir: str | Path, registro: dict) -> np.ndarray:
    """A profundidade uint16 de uma amostra, **nos dois layouts**.

    Este é o leitor do repositório. Ele existe porque não há dataloader aqui — o treino
    da BokehNet vive noutro repositório — e porque a alternativa era cada leitor
    remontar `depth/` + `sample_id` + `.png` por conta própria, que é precisamente a
    convenção que a dedup quebra. Quem consumir um release chama isto, ou copia esta
    resolução; não inventa a terceira.

    Aceita um `meta/<id>.json` ou uma linha do `manifest.jsonl` — os dois carregam
    `depth_ref`, `sample_id` e `scene_id`. Resolve o release plano do disco e o
    reagrupado por cena do Hub (`dataio.layout.resolve_depth_path`).

    O valor devolvido é a disparidade quantizada; para voltar a metros faça
    `EncodedDepth(u16, meta["disparity_min"], meta["disparity_max"], …)` e
    `decode_depth_m`. Os dois limites são **por amostra** no `meta/`, e continuam sendo
    no layout por cena — são eles que reconstroem a escala, e o PNG compartilhado só
    carrega a forma.
    """
    Image = _require_png_writer()
    return np.asarray(Image.open(resolve_depth_path(output_dir, registro)))


def iter_manifest(output_dir: str | Path):
    """Itera o manifesto sem carregar tudo em memória."""
    path = Path(output_dir) / "manifest.jsonl"
    if not path.exists():
        return
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def estimate_disk_budget(
    n_samples: int, depth_long_side: int, *,
    generates_image: bool, image_megapixels: float,
    n_scenes: Optional[int] = None,
) -> dict:
    """Orçamento de disco ANTES de rodar. A cota é 500 GB soft, 600 GB hard.

    Constantes medidas: PNG uint16 de profundidade suave ~1,2 B/px; JPEG q95 ~0,63 B/px.
    Existe porque guardar as imagens de origem de novo custaria 365 GB contra 115 GB
    de folga — e isso precisa aparecer antes do run, não na hora 9.

    **A profundidade é orçada por CENA e a máscara por AMOSTRA**, porque é assim que
    elas são gravadas no layout `per_scene`. `n_scenes=None` significa uma cena por
    amostra — o layout `per_sample`, que é o da rota B e o dos releases já publicados.

    Sem esta separação o orçamento da rota A saía **41x maior** que o real (31,7 GB de
    profundidade contra 0,77 GB), e um número impresso antes do run que erra por uma
    ordem de grandeza é pior que número nenhum: foi ele que sustentou a decisão de não
    guardar as imagens de origem.
    """
    depth_px = depth_long_side * depth_long_side * 0.75
    cenas = n_samples if n_scenes is None else int(n_scenes)
    depth_gb = cenas * depth_px * 1.2 / 1e9
    mask_gb = n_samples * depth_px * 0.04 / 1e9
    control_gb = depth_gb + mask_gb
    image_gb = n_samples * image_megapixels * 1e6 * 0.63 / 1e9 if generates_image else 0.0
    return {
        "n_samples": n_samples,
        "n_scenes": cenas,
        "depth_gb": round(depth_gb, 2),
        "control_gb": round(control_gb, 1),
        "generated_image_gb": round(image_gb, 1),
        "total_gb": round(control_gb + image_gb, 1),
    }
