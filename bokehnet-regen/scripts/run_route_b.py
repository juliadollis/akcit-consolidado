#!/usr/bin/env python3
"""Entrypoint da rota B — §3.2(b) do paper: foto com bokeh real, K pela Eq. 3.

Piloto primeiro, sempre:

    python3 scripts/run_route_b.py --source itw --pilot 200 \\
        --output-dir output/b_pilot_official \\
        --deblur-variant official_cond_only \\
        --models-dir  /raid/.../models \\
        --flux-dir    /raid/.../models/FLUX.1-dev \\
        --genfocus-dir third_party/Genfocus

O piloto **não congela limiar nenhum**: todos os gates saem em modo medir. O que ele
produz é o histograma de motivos de rejeição e a distribuição de cada métrica — e, nesta
rota, três números que ainda não existem em lugar nenhum:

* a distribuição de `k_value` pela Eq. 3 sobre o ITW, contra as âncoras 16,6 / 20,1 / 15,0;
* o deslocamento AIF↔bokeh que a DeblurNet introduz — o `[A]` A13, **nunca medido**;
* a taxa de `focus_mask_empty`, que é o que decide se a Decisão 2 (não refinar a região
  em foco) se sustenta no domínio do ITW.

## Três versões, e por que elas não podem se misturar

`--deblur-variant` escolhe **qual DeblurNet**, e a escolha é uma convenção de inferência,
não só um arquivo de peso:

| variante | repositório | arquivo | `main_adapter` | LoRA |
|---|---|---|---|---|
| `official_cond_only` (**default**) | `nycu-cplab/Genfocus-Model` | `deblurNet.safetensors` | `None` | cond-only |
| `ours_cond_only` | `juliadollis/genrefocus-deblurnet` | `deblur.safetensors` | `None` | cond-only |
| `ours_main_cond` | `juliadollis/genrefocus-deblurnet-paper-4gpu` | `deblur.safetensors` | `"deblurring"` | main+cond |

O default é a **oficial**, por decisão declarada: é o peso publicado pelos autores, é
cond-only, e rodá-la com `main_adapter=None` é o correto
(`Inference_deblurNet.py:103-111`).

**A NOSSA é `ours_cond_only`.** É o retreino de 2026-09: `lora_on_main: false`
(`retreinar-deblur/configs/train_deblur_docker_4gpu.yaml:100`) e
`adapter_variant: "cond-only"` no sidecar irmão do peso
(`outputs/deblur_docker_4gpu/deblur/deblur.json`) — logo `main_adapter=None`, como a
oficial. É o único peso nosso que existe em disco no cluster.
`ours_main_cond` é o treino ANTIGO, de OUTRO repositório, que não está mais em disco;
ela permanece no enum porque lotes já publicados gravaram
`deblur_variant: "ours_main_cond"` na proveniência e remover quebraria a leitura desses
metadados.

**`ours_main_cond` exige `--deblur-lora-sha256`; as outras duas, não.** As duas variantes
nossas têm o MESMO nome de arquivo, `deblur.safetensors`, e esse nome era a única trava
do pipeline. Foi por essa fresta que `output/b_release` inteiro (13.615 amostras) saiu com
o peso **cond-only invocado como main+cond**: a LoRA entrou no ramo principal e no de
texto, que o treino nunca viu — AIF macia, e **nenhuma exceção** em 13.615 amostras.
Agora `ours_cond_only` é fixada por `pinned_sha256` na spec (`model_runtime/deblurnet.py`)
e o runtime confere o hash sozinho; `ours_main_cond` não tem peso para fixar, então o
runtime **se recusa a carregar** sem `--deblur-lora-sha256` explícito, em vez de escolher
a variante pelo nome do arquivo. `official_cond_only` tem nome próprio, não ambíguo, e
não precisa de nada.

**`ours_cond_only` na prática: use `--deblur-weights-dir`.** O peso publicado em
`juliadollis/genrefocus-deblurnet` é **bit a bit** o que gerou os lotes (medido com
`HfApi.repo_info(files_metadata=True)`, sem baixar: sha256 `a1ed05e4…`, 1.854.533.784 B),
**mas lá ele se chama `deblurNet.safetensors`** e em disco se chama `deblur.safetensors`.
Duas consequências práticas:

* `resolve_weights` pede à variante o nome `deblur.safetensors`, que não existe naquele
  repositório — sem `--deblur-weights-dir` o download não acha o arquivo. Aponte a flag
  para a cópia do cluster (`retreinar-deblur/outputs/deblur_docker_4gpu/deblur`), que é o
  que `docker/run_route_b_h100n1.sh` faz;
* se você baixar o arquivo à mão, o nome dele **não importa**: o `pinned_sha256` da spec
  identifica o peso, inclusive sob o nome `deblurNet.safetensors`, que é o declarado pela
  variante oficial. A proveniência grava o nome real do arquivo carregado.

A rota B é **regerada** por variante, sempre num `--output-dir` NOVO.

**Um `--output-dir` por variante.** Nunca a mesma pasta com sufixo no `sample_id`: o
`FileSampleWriter` grava `split.json` e `manifest.jsonl` no diretório, e duas variantes
ali dentro compartilhariam o manifesto. Este script **recusa** rodar numa pasta cuja
metade já gravada é de outra variante — ver `_confere_variante_uniforme`.

**O `sample_id` é o MESMO nas duas versões**, e isso é de propósito: mesma foto, mesma
EXIF, mesmo `scene_id`. Só assim a diferença de K entre as versões vira uma medida direta
de quanto a AIF influencia o rótulo — um experimento barato que só existe se os ids
baterem.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from control.contract import SampleRejected, reject                   # noqa: E402
from dataio import (                                                  # noqa: E402
    DEPTH_LONG_SIDE, build_scene_split, estimate_disk_budget,
)
from model_runtime.deblurnet import DeblurVariant, resolve_weights     # noqa: E402
from qc.rejection import RejectionLog                                  # noqa: E402
from routes.route_b import RouteBConfig, run_route_b                   # noqa: E402

# A proveniência do pipeline tem UMA implementação, e ela já existe em `run_route_c.py`.
# Copiar as três funções para cá criaria a segunda definição de "qual código produziu
# este rótulo" — que é a forma exata do defeito que este projeto persegue. São privadas
# porque não são API pública; o import explícito é preferível à cópia.
from run_route_c import _git_commit, _source_sha256                    # noqa: E402


#: O dataset. `[I]` A1: o paper cita [19] só pela publicação (paper.txt:791-793), sem URL.
#: A identificação se sustenta em três coisas — o autor de [19] é *Fortes, A.*, o handle
#: é `atfortes`, e as 13.800 linhas que passam o filtro (`ACHADOS.md:151-158`) casam com
#: os "13K previously filtered and verified images" de paper.txt:999-1001. É inferência,
#: não leitura.
ITW_DATASET = "atfortes/BokehDiffusion"
ITW_SPLIT = "train"

#: Colunas de que a rota depende. Ausência é erro de checkout do dataset, não caso a
#: tratar em runtime: sem elas não há Eq. 3.
ITW_REQUIRED_COLUMNS = ("image", "focal_length", "f_number", "focal_length_35",
                        "pseudo_aif", "flickr_photo_id")

#: Chaves em que a distância de foco pode aparecer dentro do EXIF bruto. Ela **nunca** é
#: rótulo (paper.txt:344-346); serve de validador independente. Lista `[A]`: o nome do
#: campo varia por fabricante, e não medimos quais aparecem neste dataset.
_CHAVES_DISTANCIA_FOCO = ("subject_distance", "SubjectDistance", "focus_distance",
                          "FocusDistance", "subject_distance_range")


# --------------------------------------------------------------------------------
# O adaptador do ITW — a única parte que conhece o dataset concreto
# --------------------------------------------------------------------------------

class ItwRow:
    """Uma linha do ITW, na forma do `BokehSource` de `routes/route_b.py`.

    Construção que **rejeita** em vez de completar: cada campo ausente ou inválido
    levanta `SampleRejected` com slug do conjunto fechado, e o histograma da enumeração
    mostra quantas linhas caíram por qual motivo. Não existe linha meio construída.

    Vive aqui, e não em `src/sources/`, por uma razão de escopo desta tarefa — o lugar
    definitivo é `src/sources/bokehdiffusion.py`, no molde de `sources/realbokeh.py`,
    com `MirrorIndex` para leitura sequencial. Ver o relatório. A rota não muda quando
    isso acontecer: ela recebe o protocolo, não a classe.
    """

    __slots__ = ("scene_id", "sample_id", "source_dataset", "source_sample_id",
                 "source_split", "bokeh_ref", "focal_length_mm", "f_number",
                 "focal_length_35mm", "exif_focus_distance_m", "camera_make",
                 "camera_model", "row_index", "pseudo_aif")

    def __init__(self, row: dict, index: int, *, source_dataset: str = ITW_DATASET):
        photo_id = row.get("flickr_photo_id")
        if photo_id in (None, "", 0):
            # A posição no dataset FILTRADO era a chave do pipeline antigo
            # (`route_b.py:144`): mudar o filtro renumerava tudo e a retomada passava a
            # pular as amostras erradas. Sem `flickr_photo_id` não há chave estável, e
            # inventar uma a partir do índice é reintroduzir o defeito B12.
            reject("source_metadata_missing",
                   f"linha {index} sem flickr_photo_id: não há chave estável de cena")

        self.scene_id = str(photo_id)
        self.sample_id = f"b_{self.scene_id}"
        self.source_dataset = source_dataset
        self.source_sample_id = str(photo_id)
        self.source_split = ITW_SPLIT
        self.bokeh_ref = f"{source_dataset}#{ITW_SPLIT}[{index}].image"
        self.row_index = int(index)

        self.pseudo_aif = bool(row.get("pseudo_aif", True))
        self.focal_length_mm = _float_positivo(row, "focal_length", self.scene_id)
        self.f_number = _float_positivo(row, "f_number", self.scene_id)
        # `focal_length_35` está presente em 13.800/13.800 das linhas que passam o filtro
        # (`ACHADOS.md:155-157`) [M]. Ainda assim não recebe default: `sensor_width_mm`
        # rejeita com `sensor_width_unresolvable`, e é assim que o defeito B6 fica morto.
        self.focal_length_35mm = _float_positivo_ou_none(row, "focal_length_35")

        exif = row.get("flickr_exif")
        self.camera_make = _texto(_do_exif(exif, ("make", "Make")))
        self.camera_model = _texto(_do_exif(exif, ("model", "Model")))
        self.exif_focus_distance_m = _distancia_de_foco(exif)


def _float_positivo(row: dict, campo: str, scene_id: str) -> float:
    """Número FINITO e positivo, ou rejeição com slug. Sem meio-termo.

    `math.isfinite` e não só `> 0`: a string `"inf"` passa por `float()`, é maior que
    zero, e é igual a si mesma. Um `inf` em `focal_length` produziria um `pixel_ratio`
    infinito e um K não-finito — que o contrato pegaria depois, mas com o slug errado
    (`k_non_finite` em vez de metadado inválido), e o histograma apontaria para o lugar
    errado.
    """
    valor = row.get(campo)
    try:
        numero = float(valor)
    except (TypeError, ValueError):
        reject("source_metadata_field_invalid", f"cena {scene_id}: {campo}={valor!r}")
    if not math.isfinite(numero) or numero <= 0:
        reject("source_metadata_field_invalid", f"cena {scene_id}: {campo}={valor!r}")
    return numero


def _float_positivo_ou_none(row: dict, campo: str) -> Optional[float]:
    """`None` quando a coluna não traz número. NÃO é fallback: quem rejeita é o contrato.

    A diferença importa. Um `36.0` aqui seria o defeito B6 (sensor full-frame assumido,
    K errado por até 5,6x). Um `None` chega a `sensor_width_mm`, que rejeita com slug e
    entra no histograma.
    """
    try:
        numero = float(row.get(campo))
    except (TypeError, ValueError):
        return None
    return numero if math.isfinite(numero) and numero > 0 else None


def _do_exif(exif: Any, chaves: tuple[str, ...]) -> Any:
    """Lê uma chave do EXIF bruto, seja ele dict ou JSON em texto.

    A estrutura interna de `flickr_exif` é `[A]`: `ACHADOS.md` registra que *"o
    `flickr_exif` tem make e model"*, sem publicar o schema. Aceitar as duas formas é
    tolerância de LEITURA de um campo de auditoria — não de um campo do rótulo. Nenhum
    valor daqui entra em K: `make`/`model` servem para auditar os 30,33% de crop factor
    1,0, e a distância de foco é validador.
    """
    if isinstance(exif, str):
        try:
            exif = json.loads(exif)
        except json.JSONDecodeError:
            return None
    if isinstance(exif, dict):
        for chave in chaves:
            if chave in exif:
                return exif[chave]
    return None


def _texto(valor: Any) -> Optional[str]:
    if valor is None:
        return None
    texto = str(valor).strip()
    return texto or None


def _distancia_de_foco(exif: Any) -> Optional[float]:
    """Distância de foco da EXIF, em METROS, quando a câmera publica.

    **Nunca rótulo**, por decisão explícita do paper: *"it is frequently missing or noisy;
    therefore, we do not rely on EXIF for D_focus"* (paper.txt:344-346). É validador
    independente do rótulo — não passa pelo Depth Pro nem pelo BiRefNet. Âncora: a
    mediana de `k_value` calculada assim em 318 amostras é 20,1 (`CONTRATO.md:77`).

    A unidade da EXIF é metro no campo `SubjectDistance`. `[A]`: não conferido neste
    dataset. Valor ausente, não numérico ou <= 0 devolve `None`, e o resumo do run diz em
    quantas amostras o validador ficou ausente — em vez de assumir um número.
    """
    bruto = _do_exif(exif, _CHAVES_DISTANCIA_FOCO)
    if bruto is None:
        return None
    try:
        metros = float(str(bruto).split()[0])
    except (TypeError, ValueError, IndexError):
        return None
    return metros if math.isfinite(metros) and metros > 0 else None


def _enumera_itw(dataset, *, log: RejectionLog, incluir_pseudo_aif: bool) -> list[ItwRow]:
    """Constrói as linhas, rejeitando com slug. Devolve as que passam.

    O filtro `not pseudo_aif` é `[A]` A3 — coluna do dataset, não do paper; herdado do
    pipeline antigo (`route_b.py:83-89`). Mantido para que a contagem seja comparável com
    as 13.800 medidas [M], e exposto em `--include-pseudo-aif` para que a decisão fique
    visível em vez de embutida.
    """
    linhas: list[ItwRow] = []
    vistos: set[str] = set()
    for index, row in enumerate(dataset):
        try:
            linha = ItwRow(row, index)
            if linha.pseudo_aif and not incluir_pseudo_aif:
                # SLUG PROVISÓRIO: o vocabulário fechado não tem um valor para "a linha é
                # de outra natureza". `source_pseudo_aif_excluded` seria o certo, e exige
                # mudança em `control/contract.py`. Ver o relatório.
                reject("source_metadata_field_invalid",
                       "pseudo_aif=True: filtro [A] A3, herdado do pipeline antigo")
            if linha.sample_id in vistos:
                reject("source_duplicate_sample",
                       f"flickr_photo_id repetido: {linha.scene_id}")
        except SampleRejected as exc:
            log.reject_from(f"b_row_{index}", exc, {"route": "b", "row_index": index})
            continue
        vistos.add(linha.sample_id)
        linhas.append(linha)
    return linhas


class ItwImageLoader:
    """Carrega a foto com bokeh do dataset, em BGR uint8, e mantém o ledger da fonte.

    O ledger existe porque a bokeh é **referência**, não cópia: o release aponta para
    `atfortes/BokehDiffusion`, e sem o sha256 dos bytes o join deixa de ser verificável.
    É o análogo do `source_images.jsonl` da rota C, e o par dele é o
    `generated_images.jsonl` que a rota escreve para a AIF.
    """

    def __init__(self, dataset, *, ledger_path: Optional[Path] = None):
        self.dataset = dataset
        self.ledger = None
        if ledger_path is not None:
            ledger_path.parent.mkdir(parents=True, exist_ok=True)
            self.ledger = ledger_path.open("a", encoding="utf-8")

    def __call__(self, source) -> "Any":
        import hashlib
        import io

        import numpy as np
        from PIL import Image

        valor = self.dataset[int(source.row_index)]["image"]
        if isinstance(valor, Image.Image):
            imagem, bytes_brutos = valor, None
        elif isinstance(valor, dict) and valor.get("bytes"):
            bytes_brutos = valor["bytes"]
            imagem = Image.open(io.BytesIO(bytes_brutos))
        elif isinstance(valor, dict) and valor.get("path"):
            bytes_brutos = Path(valor["path"]).read_bytes()
            imagem = Image.open(io.BytesIO(bytes_brutos))
        else:
            reject("source_image_unreadable",
                   f"cena {source.scene_id}: tipo de célula {type(valor).__name__}")

        if imagem.mode == "RGBA":
            # `convert("RGB")` comporia sobre preto e INVENTARIA pixel na entrada da
            # DeblurNet, do Depth Pro e do BiRefNet. Mesma política da rota C.
            alpha = np.asarray(imagem.getchannel("A"))
            if int(alpha.min()) < 255:
                reject("source_image_alpha_not_opaque",
                       f"cena {source.scene_id}: alpha mínimo {int(alpha.min())}")
        rgb = np.asarray(imagem.convert("RGB"), dtype=np.uint8)
        if rgb.ndim != 3 or min(rgb.shape[:2]) < 1:
            reject("source_image_unreadable", f"cena {source.scene_id}: shape {rgb.shape}")

        if self.ledger is not None:
            self.ledger.write(json.dumps({
                "sample_id": source.sample_id,
                "route": "b",
                "bokeh_role": "reference",
                "source_dataset": source.source_dataset,
                "source_sample_id": source.source_sample_id,
                "bokeh_ref": source.bokeh_ref,
                "bokeh_sha256": (hashlib.sha256(bytes_brutos).hexdigest()
                                 if bytes_brutos else None),
                "bokeh_bytes": len(bytes_brutos) if bytes_brutos else None,
                "image_h": int(rgb.shape[0]), "image_w": int(rgb.shape[1]),
            }, ensure_ascii=False) + "\n")
            self.ledger.flush()

        return np.ascontiguousarray(rgb[..., ::-1])          # BGR, como a rota espera

    def close(self) -> None:
        if self.ledger is not None:
            self.ledger.close()
            self.ledger = None


def _carrega_itw(args):
    """Devolve `(linhas, load_bokeh, extra_de_proveniencia)`."""
    from datasets import load_dataset

    print(f"[fonte] {ITW_DATASET} split {ITW_SPLIT}…")
    dataset = load_dataset(ITW_DATASET, split=ITW_SPLIT, cache_dir=args.cache_dir or None)
    faltando = [c for c in ITW_REQUIRED_COLUMNS if c not in dataset.column_names]
    if faltando:
        raise SystemExit(
            f"o dataset não tem as colunas {faltando}. Sem elas não há Eq. 3 — e não há "
            "default a assumir. Colunas presentes: {}".format(dataset.column_names))
    print(f"[fonte] {len(dataset)} linhas.")

    log = RejectionLog(Path(args.output_dir) / "source_rejections.jsonl")
    linhas = _enumera_itw(dataset, log=log,
                          incluir_pseudo_aif=args.include_pseudo_aif)
    print(log.summary())
    print(f"[fonte] {len(linhas)} linhas passam. Medido antes: 13.800 "
          f"(ACHADOS.md:151-158).")
    log.close()

    if args.num_shards > 1:
        # Fatia por `flickr_photo_id`, não por posição: a ordem de leitura pode mudar
        # e a fatia não pode mudar junto, senão relançar reprocessa outro conjunto.
        # Na rota B cada linha é uma foto independente — não há cena a manter inteira.
        import hashlib

        if not 0 <= args.shard < args.num_shards:
            raise SystemExit(f"--shard {args.shard} fora de [0, {args.num_shards})")
        antes = len(linhas)
        # `scene_id` é o `flickr_photo_id` (ver `ItwRow.__init__`), e é a chave estável
        # que a própria fonte escolheu. Usar o índice da linha faria a fatia mudar se a
        # ordem de leitura mudasse, e relançar reprocessaria outro conjunto.
        linhas = [l for l in linhas
                  if int.from_bytes(
                      hashlib.sha256(str(l.scene_id).encode()).digest()[:8],
                      "big") % args.num_shards == args.shard]
        print(f"[fatia] {args.shard}/{args.num_shards}: {len(linhas)} de {antes} linhas")

    if args.pilot:
        import random
        # Sorteia LINHAS e não um prefixo: na rota B cada linha é uma foto independente,
        # então não há cena a manter inteira — mas um prefixo da ordem de disco viria
        # todo do mesmo shard, e portanto da mesma faixa de upload do Flickr.
        rng = random.Random(args.seed)
        linhas = rng.sample(linhas, min(args.pilot, len(linhas)))
        # Reordena por posição no dataset: leitura sequencial em vez de seek aleatório.
        linhas.sort(key=lambda linha: linha.row_index)
        print(f"[fonte] piloto: {len(linhas)} linhas (seed {args.seed}).")

    loader = ItwImageLoader(
        dataset, ledger_path=Path(args.output_dir) / "source_images.jsonl")
    extra = {
        "source_dataset": ITW_DATASET,
        "source_split": ITW_SPLIT,
        "source_dataset_evidence": "[I] A1 — o paper cita [19] sem URL "
                                   "(paper.txt:791-793); autor + volume sustentam",
        "source_filter_pseudo_aif": not args.include_pseudo_aif,
        "source_filter_evidence": "[A] A3 — coluna do dataset, não do paper",
        "source_license": args.source_license or None,
    }
    return linhas, loader, extra


# --------------------------------------------------------------------------------
# A variante, e a recusa de misturar
# --------------------------------------------------------------------------------

def _confere_variante_uniforme(output_dir: Path, variante: DeblurVariant) -> None:
    """Recusa continuar um `--output-dir` cuja metade gravada é de OUTRA variante.

    Regra 1 e 3 do §5.3 da auditoria: um `--release-dir` por variante, e checagem de
    uniformidade. Sem isto, um `rsync` de duas metades produz um release misto e nada
    denuncia — a mesma família do `max_coc` por rota.

    A checagem lê o `generated_images.jsonl` que a rota escreve, porque a linha do
    manifesto ainda **não** carrega `deblur_variant` — é o item 4 do §5.3, e é mudança em
    `dataio/writer.py`, descrita no relatório desta tarefa e não aplicada aqui.
    """
    ledger = output_dir / "generated_images.jsonl"
    if not ledger.is_file():
        return
    encontradas = set()
    with ledger.open(encoding="utf-8") as handle:
        for linha in handle:
            linha = linha.strip()
            if not linha:
                continue
            try:
                encontradas.add(json.loads(linha).get("deblur_variant"))
            except json.JSONDecodeError:
                continue
    encontradas.discard(None)
    divergentes = encontradas - {variante.value}
    if divergentes:
        raise SystemExit(
            f"{output_dir} já tem amostras da variante {sorted(divergentes)} e você pediu "
            f"{variante.value}.\nUm --output-dir POR VARIANTE: as duas metades "
            "compartilhariam manifest.jsonl e split.json, e o release sairia misto sem "
            "que nada denunciasse. Use um diretório novo.")


# --------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Rota B — ITW dataset, K pela Eq. 3. O alvo é a própria foto.",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", default="itw", choices=["itw"])
    p.add_argument("--output-dir", required=True,
                   help="UM POR VARIANTE da DeblurNet. Ver o docstring do módulo.")
    p.add_argument("--models-dir", required=True,
                   help="contém checkpoints/depth_pro.pt e BiRefNet/")
    p.add_argument("--flux-dir", required=True,
                   help="snapshot do FLUX.1-dev (tem que ter model_index.json)")
    p.add_argument("--genfocus-dir", required=True,
                   help="checkout de nycu-cplab/Genfocus, com Inference_deblurNet.py e "
                        "Genfocus/pipeline/flux.py. NÃO usar o vendorizado de "
                        "bokehnet-preprocessing: é lá que o defeito B1 vive.")

    deblur = p.add_argument_group(
        "DeblurNet",
        "A variante é a tripla (repositório, arquivo, main_adapter), e ela é INDIVISÍVEL. "
        "Não existe --main-adapter: um default do lado errado produz o defeito B1 (AIF "
        "lavada) e um default do lado certo o esconde.")
    # `choices` DERIVA do enum: variante nova aparece aqui sem ninguém lembrar de
    # atualizar a lista, e um valor fora do enum morre no argparse.
    deblur.add_argument("--deblur-variant",
                        default=DeblurVariant.OFFICIAL_COND_ONLY.value,
                        choices=[v.value for v in DeblurVariant],
                        help="official_cond_only (DEFAULT) = nycu-cplab/Genfocus-Model, "
                             "deblurNet.safetensors, main_adapter=None — o peso "
                             "publicado pelos autores. "
                             "ours_cond_only = juliadollis/genrefocus-deblurnet, "
                             "deblur.safetensors, main_adapter=None — É A NOSSA, o "
                             "retreino de 2026-09 com lora_on_main: false. "
                             "ours_main_cond = "
                             "juliadollis/genrefocus-deblurnet-paper-4gpu, "
                             "deblur.safetensors, main_adapter='deblurring' — o treino "
                             "ANTIGO, que não está mais em disco; EXIGE "
                             "--deblur-lora-sha256, porque compartilha o nome do "
                             "arquivo com ours_cond_only e o nome não distingue as duas "
                             "(foi assim que o lote b_release inteiro saiu errado). "
                             "Um --output-dir NOVO por variante.")
    deblur.add_argument("--deblur-weights-dir", default="",
                        help="diretório local que contém o .safetensors DA VARIANTE. "
                             "Sem isto, baixa do repositório da variante.")
    deblur.add_argument("--deblur-lora-sha256", default="",
                        help="o sha256 esperado do LoRA. Opcional para "
                             "official_cond_only e para ours_cond_only (esta já é "
                             "fixada por sha na spec); OBRIGATÓRIO para "
                             "ours_main_cond, que divide o nome deblur.safetensors com "
                             "ours_cond_only — sem ele o runtime recusa em vez de "
                             "escolher a variante pelo nome. Um peso renomeado passa "
                             "pela checagem de nome; o hash não passa.")
    deblur.add_argument("--deblur-steps", type=int, default=None,
                        help="default 28 — oficial (Inference_deblurNet.py:56) e o que o "
                             "paper reporta (paper.txt:519-520).")
    deblur.add_argument("--deblur-long-side", type=int, default=0,
                        help="0 = resolução original, e é o default do oficial e o da "
                             "rota B. Valor > 0 com a política que recorta PERDE campo "
                             "de visão (medido: 30,86 px de deslocamento num 5184x3456) "
                             "e quebra a relação entre focal_length_35 e a geometria.")

    p.add_argument("--cache-dir", default="",
                   help="cache do `datasets`. Em pasta do projeto, nunca no ~/.local.")
    p.add_argument("--source-license", default="",
                   help="licença da fonte, gravada na proveniência de cada amostra.")
    p.add_argument("--include-pseudo-aif", action="store_true",
                   help="mantém as linhas com pseudo_aif=True. O filtro é [A] A3 — do "
                        "pipeline antigo, não do paper.")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--val-fraction", type=float, default=0.05)
    p.add_argument("--image-megapixels", type=float, default=6.0,
                   help="[A] megapixels médios da fonte, só para o orçamento de disco. "
                        "Não medimos a resolução do ITW (é o [A] A6).")

    escopo = p.add_argument_group("escopo")
    escopo.add_argument("--shard", type=int, default=0,
                        help="índice desta fatia, por hash do flickr_photo_id")
    escopo.add_argument("--num-shards", type=int, default=1,
                        help="quantas fatias. Cada uma escreve no próprio --output-dir.")
    escopo.add_argument("--pilot", type=int, default=None,
                        help="sorteia N linhas com seed antes de processar. É o certo "
                             "para calibrar limiar.")
    escopo.add_argument("--limit", type=int, default=None,
                        help="teto de amostras ACEITAS no laço.")

    lim = p.add_argument_group(
        "limiares",
        "TODOS default None = mede e não bloqueia. Congele a partir do piloto. Nenhum "
        "destes valores vem do paper.")
    for flag in ("min-mask-area-ratio", "max-mask-area-ratio",
                 "max-mask-border-coverage", "min-mask-iou",
                 "min-focus-mask-sharpness-ratio", "min-aif-laplacian-variance",
                 "max-bokeh-over-aif-sharpness", "min-deblur-structural-ssim",
                 "max-deblur-structural-ssim", "max-pair-registration-shift-px",
                 "min-pair-registration-response", "min-k-value", "max-k-value"):
        lim.add_argument(f"--{flag}", type=float, default=None)
    lim.add_argument("--min-depth-useful-levels", type=int, default=None)
    lim.add_argument("--retention-window-px", type=int, default=None,
                     help="janela do filtro de média da retenção, EM PIXEL da resolução "
                          "da imagem. A retenção aqui é DIAGNÓSTICO, não rótulo.")
    return p


def main() -> int:
    args = build_parser().parse_args()
    raiz = Path(__file__).resolve().parents[1]
    saida = Path(args.output_dir)
    saida.mkdir(parents=True, exist_ok=True)

    variante = DeblurVariant(args.deblur_variant)
    _confere_variante_uniforme(saida, variante)
    print(f"[rota-b] variante da DeblurNet: {variante.value} "
          f"({variante.spec.repo_id} / {variante.spec.weight_filename}, "
          f"main_adapter={variante.spec.main_adapter!r}, {variante.spec.lora_mode})")
    print(f"[rota-b] evidência: {variante.spec.evidence}")

    # -- os modelos, carregados uma vez ----------------------------------------
    from model_runtime import BiRefNetRuntime, DeblurNetRuntime, DepthProRuntime

    # `cache_dir=None` de propósito: `--cache-dir` é o cache do `datasets`, e enfiar
    # pesos de modelo dentro dele misturaria dois caches com políticas de limpeza
    # diferentes. O destino dos pesos vem de `HF_HOME`, que o job SLURM aponta para uma
    # pasta do PROJETO — nunca o `~/.cache`, que é compartilhado entre nós.
    pesos = resolve_weights(variante,
                            local_dir=args.deblur_weights_dir or None)
    print(f"[rota-b] LoRA: {pesos}")

    opcoes_deblur = {}
    if args.deblur_steps is not None:
        opcoes_deblur["num_steps"] = args.deblur_steps
    deblur = DeblurNetRuntime(
        variant=variante, weights_path=pesos, flux_dir=args.flux_dir,
        genfocus_dir=args.genfocus_dir, device=args.device,
        long_side=args.deblur_long_side,
        expected_lora_sha256=args.deblur_lora_sha256 or None,
        **opcoes_deblur)

    modelos = Path(args.models_dir)
    depth = DepthProRuntime(modelos / "checkpoints" / "depth_pro.pt", device=args.device)
    mask = BiRefNetRuntime(modelos / "BiRefNet", device=args.device)

    # -- fonte -----------------------------------------------------------------
    if args.source != "itw":
        raise SystemExit(
            f"fonte {args.source!r} não tem adaptador. A rota B recebe as linhas por "
            "protocolo — escrever o adaptador não exige tocar em routes/route_b.py.")
    linhas, load_bokeh, extra_fonte = _carrega_itw(args)
    if not linhas:
        raise SystemExit("nenhuma linha sobreviveu à enumeração — veja o histograma acima.")

    # O split é por CENA e materializado. No ITW cada linha é uma foto independente, então
    # `scene_id == flickr_photo_id` e o vazamento treino/val é menos agudo que na rota C —
    # mas o split materializado é regra do projeto, e a chave estável é a mesma que dá
    # `sample_id`, o que mantém as duas versões comparáveis par a par.
    split = build_scene_split({linha.scene_id for linha in linhas},
                              val_fraction=args.val_fraction)
    print(f"[split] sorteado por cena: {split.counts()}")

    # -- orçamento de disco ANTES de rodar -------------------------------------
    # A rota B grava a AIF em JPEG, ao contrário da C — e são DUAS versões.
    orcamento = estimate_disk_budget(len(linhas), DEPTH_LONG_SIDE,
                                     generates_image=True,
                                     image_megapixels=args.image_megapixels)
    print(f"[disco] {orcamento} — e isto é UMA variante; a cota é 500 GB soft.")

    # -- config ----------------------------------------------------------------
    # `None` significa "usa o default do módulo", e o default do módulo é a única
    # definição do valor. Repetir o número aqui criaria uma segunda definição.
    opcionais = {k: v for k, v in (
        ("retention_window_px", args.retention_window_px),) if v is not None}
    config = RouteBConfig(
        output_dir=saida, seed=args.seed, limit=args.limit,
        min_mask_area_ratio=args.min_mask_area_ratio,
        max_mask_area_ratio=args.max_mask_area_ratio,
        max_mask_border_coverage=args.max_mask_border_coverage,
        min_mask_iou=args.min_mask_iou,
        min_focus_mask_sharpness_ratio=args.min_focus_mask_sharpness_ratio,
        min_aif_laplacian_variance=args.min_aif_laplacian_variance,
        min_depth_useful_levels=args.min_depth_useful_levels,
        max_bokeh_over_aif_sharpness=args.max_bokeh_over_aif_sharpness,
        min_deblur_structural_ssim=args.min_deblur_structural_ssim,
        max_deblur_structural_ssim=args.max_deblur_structural_ssim,
        max_pair_registration_shift_px=args.max_pair_registration_shift_px,
        min_pair_registration_response=args.min_pair_registration_response,
        min_k_value=args.min_k_value, max_k_value=args.max_k_value,
        **opcionais)

    congelados = {k: v for k, v in vars(args).items()
                  if k.startswith(("min_", "max_")) and v is not None}
    print(f"[rota-b] limiares congelados: {congelados or 'nenhum — modo medir'}")
    if not congelados and not args.pilot:
        print("[rota-b] AVISO: run completo com todos os gates em modo medir. "
              "Isso é intencional só se você quer o dataset bruto para calibrar depois.")

    provenance_base = {
        "pipeline_commit": _git_commit(raiz) or f"sha256tree:{_source_sha256(raiz)}",
        "pipeline_source_sha256": _source_sha256(raiz),
        **depth.provenance(),
        **mask.provenance(),
        # NENHUM renderizador na rota B — paper.txt:271,283,292,321. A rota grava
        # `renderer=None`, e não um dicionário que afirme `is_final_label_renderer`.
        "renderer": None,
        "deblurnet": deblur.provenance(),
        "split_origin": "sorteado_por_cena",
        **extra_fonte,
    }
    (saida / "run_config.json").write_text(json.dumps(
        {"args": {k: (str(v) if isinstance(v, Path) else v)
                  for k, v in vars(args).items()},
         "provenance_base": provenance_base,
         # O layout de `depth/` DECLARADO, para que a validação e a leitura não
         # precisem inferi-lo contando arquivos. Um release por cena tem menos PNG que
         # amostras de propósito, e contar arquivos confundiria isso com um release
         # incompleto — que é a diferença entre publicar e não publicar.
         "depth_layout": config.depth_layout.value,
         "disk_budget": orcamento,
         "rows_enumerated": len(linhas),
         "deblur_weights_sha256": deblur.provenance()["deblur_lora_sha256"]},
        indent=2, ensure_ascii=False), encoding="utf-8")

    try:
        stats = run_route_b(linhas, load_bokeh=load_bokeh, deblur_runtime=deblur,
                            depth_runtime=depth, mask_runtime=mask, split=split,
                            config=config, provenance_base=provenance_base)
    finally:
        # O ledger da FONTE fica aberto durante o laço e é fechado aqui. `flush()` por
        # linha já garante que um crash não perde o histórico; o `close` é higiene.
        load_bokeh.close()
    return 0 if stats.written else 1


if __name__ == "__main__":
    raise SystemExit(main())
