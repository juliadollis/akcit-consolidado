"""DeblurNet — três variantes que NÃO podem se misturar.

A rota B começa aqui: `bokeh real ──DeblurNet──▶ I_aif`, e a AIF é a entrada do Depth
Pro, do BiRefNet, do plano de foco e do K da Eq. 3 (paper.txt:293-296, 314). Errar aqui
contamina o rótulo inteiro, e nenhum gate a jusante vê a causa.

## O defeito que este módulo existe para tornar inexpressável

O wrapper antigo chamava `generate(...)` **sem passar `main_adapter`**
(`bokehnet-preprocessing/src/model_runtime/deblurnet.py:201-211`), cujo default é `None`
(`src/vendor/genfocus_flux.py:485`). `generate` monta
`adapters = [main_adapter] * 2 + c_adapters` (:783), então `None` significa
**cond-only**: o LoRA age só nas condições. Mas a rota B exigia o NOSSO checkpoint
(`route_b.py:331-335`), que é a variante **main+cond** — e main+cond rodado como
cond-only sai **LAVADO** (`HANDOFF_PROJECT_HISTORY.md:102`; já custou LPIPS ~0,85,
:171). O gate que veria isso, `--min-aif-laplacian-variance`, tinha default 0,0
(`route_b.py:405-410`).

Duas propriedades desse defeito importam para o desenho daqui:

1. **O arquivo de pesos não sabe de qual convenção ele precisa.** Um LoRA main+cond e um
   cond-only têm o mesmo formato e as mesmas chaves; a diferença vive no *roteamento* em
   tempo de inferência, não no `.safetensors`. Nenhuma inspeção do peso separa os dois.
   Logo a única defesa possível é amarrar a convenção ao peso **no código**.
2. **`generate` termina em `**params: dict`** (verificado em
   `genrefocus_deblurnet_paper/third_party/Genfocus/Genfocus/pipeline/flux.py`, na
   assinatura que começa em :464 e no `**params` do fim). Se o upstream renomear
   `main_adapter`, o nosso `main_adapter=...` explícito é **engolido em silêncio** pelo
   `**params` e a inferência volta a ser cond-only sem erro nenhum. Por isso o
   `__post_init__` confere por AST, antes de importar torch, que `generate` ainda
   declara `main_adapter` — ver `_inspect_flux_generate`.

## As três variantes, e a tripla indivisível

| variante | repositório | arquivo | `main_adapter` | LoRA |
|---|---|---|---|---|
| `OURS_MAIN_COND` | `juliadollis/genrefocus-deblurnet-paper-4gpu` | `deblur.safetensors` | `"deblurring"` | main+cond |
| `OURS_COND_ONLY` | `juliadollis/genrefocus-deblurnet` | `deblur.safetensors` | `None` | cond-only |
| `OFFICIAL_COND_ONLY` | `nycu-cplab/Genfocus-Model` | `deblurNet.safetensors` | `None` | cond-only |

**O nome do arquivo não identifica a variante, nos DOIS sentidos.**

*Um nome, duas variantes*: as duas primeiras linhas têm o mesmo `deblur.safetensors`, e a
única trava que existia era o nome — foi por essa fresta que o lote `b_release` inteiro
(13.615 amostras) saiu com o peso cond-only rodado como main+cond, sem uma exceção.

*Uma variante, dois nomes*: o peso da `OURS_COND_ONLY` está publicado em
`juliadollis/genrefocus-deblurnet` sob o nome **`deblurNet.safetensors`** (medido com
`HfApi.repo_info(files_metadata=True)`: sha256 `a1ed05e4…`, 1.854.533.784 B, idêntico à
cópia do cluster, que se chama `deblur.safetensors`). Esse é o nome que a variante
OFICIAL declara. Deixar o nome decidir acusava o nosso próprio peso de ser o da oficial.

Daí a regra, em `_identifica_o_peso`: **o sha256 é a identidade do peso; o nome é uma
pista.** Se a spec fixa `pinned_sha256` e ele casa, o nome não tem voto; se não casa, é
erro. Só quando não há pin o nome volta a ser a evidência — e aí selecionar
`OURS_MAIN_COND` com `deblur.safetensors` EXIGE `expected_lora_sha256` explícito de quem
chama.

Enum fechado, no molde de `MaskSource`/`KSource` (`dataio/sample.py:52-68`). A tripla
`(repositório, arquivo, main_adapter)` é **indivisível**: `main_adapter` não é parâmetro
de nada aqui — não do `__init__`, não de `infer`, não de `resolve_weights`. Só
`_VARIANT_SPECS` conhece o valor, e só a partir da variante. Cruzar os pesos de uma
variante com o adapter da outra não é "desaconselhado": não há assinatura que o
expresse, e a checagem de nome de arquivo em `__post_init__` pega o acidente realista
(apontar para o snapshot do outro repositório).

O que este módulo deliberadamente NÃO oferece, porque foi assim que o defeito nasceu:

- nenhum parâmetro `main_adapter` (o `deblurnet-eval-pipeline/infer_and_eval.py:98-101`
  tem uma flag `--main-adapter` de texto livre com default `"deblurring"` — é o
  antipadrão exato: um default silencioso do lado errado produz o defeito, e um default
  do lado certo o esconde);
- nenhum parâmetro para `repo_id` nem para o nome do arquivo de pesos;
- nenhum `prompt` configurável (é constante, `PROMPT`), porque prompt divergente é um
  segundo eixo de mistura entre versões do release.

## Recorte por `long_side` — medido, e nunca em silêncio

`long_side > 0` no código antigo e no oficial **RECORTA** depois de redimensionar
(`bokehnet-preprocessing/src/model_runtime/deblurnet.py:55-60` ==
`Inference_deblurNet.py:30-41`), e depois o antigo volta ao tamanho original por
`cv2.resize` (:215-218) — o que é um zoom, não uma identidade. Medido aqui (números em
`reference/ACHADOS.md`, e reproduzíveis por `plan_processing`):

| resolução | `long_side` | processada (oficial) | FOV perdida | deslocamento máx. |
|---|---|---|---|---|
| 4032×3024 | 512 | 512×384 | 0 | 0,00 px |
| 3000×2000 | 512 | 512×336 | 1,47% em y | **17,86 px** |
| 5184×3456 | 512 | 512×336 | 1,47% em y | **30,86 px** |
| 2048×1365 | 768 | 768×496 | 2,94% em y | **22,02 px** |

O gate do pipeline antigo era `--max-pair-shift-px 6.0` (`route_b.py:411-422`); 30,86 px
é 5,1× isso. E recorte muda o campo de visão do lado curto, o que quebra a relação entre
`focal_length_35` e a geometria da imagem — a mesma classe de erro do `[A]` A6.

Tratamento explícito, com duas políticas fechadas (`ResizePolicy`):

- **`NO_CROP_MULTIPLE_OF_16`** (default): arredonda **para cima** até múltiplo de 16 e
  **nunca recorta**. O round-trip é uma identidade geométrica exata: `y_out = y_orig`,
  deslocamento 0,00 px por construção. Com `long_side = 0` — o valor da rota B
  (`route_b.py:104-109`) e o default oficial (`Inference_deblurNet.py:57`) — esta
  política é **byte a byte o caminho oficial**, porque o ramo sem `long_side` do oficial
  também é `ceil` para múltiplo de 16 e também não recorta (:43-49).
- **`OFFICIAL_CENTER_CROP_16`**: reproduz o oficial inclusive o recorte. Exige
  `acknowledge_fov_crop=True` quando `long_side > 0`, e cada amostra grava
  `crop_box`, `fov_retained`, `processed_hw` e `max_registration_shift_px`.

Em nenhum dos dois caminhos há recorte sem registro: `processed_hw` e `crop_applied` vão
para a proveniência de **toda** amostra, mesmo quando não houve recorte.

## Convenção de canal

`infer(bokeh_rgb) -> DeblurredAIF` em **RGB**, igual a `DepthProRuntime.infer` e a
`BiRefNetRuntime.infer`. A AIF sai pronta para alimentar os dois sem conversão. Quem
grava em disco converte para BGR e declara `Sample.channel_order` (`dataio/sample.py:142`).
"""

from __future__ import annotations

import ast
import hashlib
import subprocess
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional

import numpy as np

from control.contract import reject

#: Identificador do backend. Um só, como `depth_pro` e `birefnet`: não existe DeblurNet
#: alternativa, e a *variante* do checkpoint é um eixo separado (`DeblurVariant`).
BACKEND_NAME = "genfocus_deblurnet"

#: Prompt da inferência oficial (`Inference_deblurNet.py:107`). Constante, não parâmetro:
#: prompt divergente entre metades de um release é a mesma classe de erro do `max_coc`
#: por rota.
PROMPT = "a sharp photo with everything in focus"

#: Nome do adapter no `load_lora_weights`/`set_adapters`, comum às DUAS variantes
#: (`Inference_deblurNet.py:88-89`). NÃO confundir com `main_adapter`, que é o que
#: distingue main+cond de cond-only.
ADAPTER_NAME = "deblurring"

#: 28 passos: default oficial (`Inference_deblurNet.py:56`) e o que o paper reporta
#: (paper.txt:519-520).
NUM_INFERENCE_STEPS = 28

#: `seed_everything(42)` — oficial (`Inference_deblurNet.py:100`) e antigo (:198).
SEED = 42

#: `NO_TILED_DENOISE = min(w, h) < 512` (`Inference_deblurNet.py:95`).
TILING_MIN_SHORT_SIDE = 512

#: O VAE do FLUX exige múltiplo de 16 em cada lado.
SIZE_MULTIPLE = 16


# --------------------------------------------------------------------------------
# A variante — enum fechado, tripla indivisível
# --------------------------------------------------------------------------------

class DeblurVariantMismatch(ValueError):
    """Pesos de uma variante com a convenção da outra.

    Erro próprio, e não `ValueError` genérico, porque este é o defeito B1: quem
    escrever `except ValueError` num script não deve poder engolir exatamente este.
    """


class DeblurVariant(str, Enum):
    """Qual DeblurNet rodou. Enum fechado, não string livre.

    Cada valor carrega uma tripla `(repo_id, weight_filename, main_adapter)` que é
    **indivisível** — ver `_VARIANT_SPECS`. Não existe valor `AUTOMATIC` nem `DEFAULT`,
    pela mesma razão que `MaskSource` não tem `AUTOMATIC` (`dataio/sample.py:52-58`):
    seria vago demais para ser proveniência.
    """

    OURS_MAIN_COND = "ours_main_cond"
    #: O retreino de 2026-09, `lora_on_main: false`. É ESTE que existe em disco e o que
    #: a rota B deve usar. Não confundir com OURS_MAIN_COND: o nome do ARQUIVO é o mesmo
    #: (`deblur.safetensors`) nos dois, e foi exatamente nisso que a única trava se
    #: apoiava — por isso esta variante é fixada por sha256.
    OURS_COND_ONLY = "ours_cond_only"
    OFFICIAL_COND_ONLY = "official_cond_only"

    @property
    def spec(self) -> "DeblurVariantSpec":
        """A tripla desta variante. Única porta de acesso ao `main_adapter`."""
        return _VARIANT_SPECS[self]


@dataclass(frozen=True)
class DeblurVariantSpec:
    """A tripla indivisível. Congelada, e construída só dentro deste módulo.

    `main_adapter` mora **aqui e em lugar nenhum mais**. `DeblurNetRuntime` recebe uma
    `DeblurVariant` e vai buscar a spec em `_VARIANT_SPECS`; não recebe uma spec, então
    uma spec montada à mão com a combinação errada não tem por onde entrar.
    """

    variant: DeblurVariant
    repo_id: str
    weight_filename: str
    #: O que vai em `generate(main_adapter=...)`. `None` é um VALOR, não uma ausência:
    #: para a variante oficial, cond-only é o correto (`Inference_deblurNet.py:103-111`
    #: não passa o argumento, e está certo, porque o peso oficial é cond-only).
    main_adapter: Optional[str]
    #: Sempre uma string não vazia, mesmo quando `main_adapter is None`. Existe porque
    #: `_REQUIRED_PROVENANCE` valida por truthiness (`dataio/sample.py:249`), e
    #: `main_adapter=None` reprovaria uma proveniência que está correta.
    lora_mode: str
    evidence: str
    #: sha256 do peso desta variante, quando ele é conhecido e imutável. Não é adorno:
    #: é o que distingue duas variantes que compartilham o NOME do arquivo. Quando
    #: presente, divergência é erro; quando o nome é ambíguo e ele está ausente, o
    #: runtime se recusa a carregar em vez de adivinhar.
    pinned_sha256: Optional[str] = None

    def __post_init__(self) -> None:
        # Invariante do módulo: os dois modos possíveis, e o mapeamento entre modo e
        # adapter é 1-para-1. Se alguém acrescentar uma variante e errar o par, falha
        # na importação do módulo, não no meio de um run de 13.800 amostras.
        if self.lora_mode not in ("main_cond", "cond_only"):
            raise ValueError(f"lora_mode desconhecido: {self.lora_mode!r}")
        esperado = ADAPTER_NAME if self.lora_mode == "main_cond" else None
        if self.main_adapter != esperado:
            raise DeblurVariantMismatch(
                f"spec incoerente para {self.variant.value}: lora_mode={self.lora_mode!r} "
                f"exige main_adapter={esperado!r}, recebido {self.main_adapter!r}. "
                "main+cond rodado como cond-only sai LAVADO "
                "(HANDOFF_PROJECT_HISTORY.md:102)."
            )


_VARIANT_SPECS: dict[DeblurVariant, DeblurVariantSpec] = {
    DeblurVariant.OURS_MAIN_COND: DeblurVariantSpec(
        variant=DeblurVariant.OURS_MAIN_COND,
        repo_id="juliadollis/genrefocus-deblurnet-paper-4gpu",
        weight_filename="deblur.safetensors",
        main_adapter=ADAPTER_NAME,
        lora_mode="main_cond",
        # Treino ANTIGO, de outro repositório. Não está em disco nesta máquina. No Hub
        # ele existe e é `deblur.safetensors`, sha256 `deea4988…`, 1.856.139.632 B, 688
        # chaves (medido com `HfApi.repo_info(files_metadata=True)`) — mas só se conhece
        # o PREFIXO do digest, e pin pela metade é pior que pin nenhum: daria a impressão
        # de trava sem travar. Por isso esta variante não fixa hash, e selecioná-la com
        # um `deblur.safetensors` qualquer EXIGE `expected_lora_sha256` explícito — o
        # nome do arquivo não a distingue da OURS_COND_ONLY. Mantida porque lotes já
        # publicados gravaram `deblur_variant: "ours_main_cond"` na proveniência;
        # remover quebraria a leitura desses metadados.
        evidence="HANDOFF_PROJECT_HISTORY.md:92,109,161 (treino main+cond, step_60000); "
                 "RESULTADOS.md:156 registra este repo como 'antigo main+cond'; "
                 "no Hub: deblur.safetensors sha256 deea4988… (688 chaves)",
    ),
    DeblurVariant.OURS_COND_ONLY: DeblurVariantSpec(
        variant=DeblurVariant.OURS_COND_ONLY,
        repo_id="juliadollis/genrefocus-deblurnet",
        weight_filename="deblur.safetensors",
        main_adapter=None,
        lora_mode="cond_only",
        # MEDIDO: o sidecar irmão do peso (`outputs/deblur_docker_4gpu/deblur/deblur.json`)
        # diz `adapter_variant: "cond-only"`, `text_adapter: null`; o config que o treinou
        # (`train_deblur_docker_4gpu.yaml:100`) diz `lora_on_main: false`; e o header do
        # .safetensors traz rank 128 nos 686 tensores, que é o `paper.txt:513`.
        # Rodar ESTE com main_adapter="deblurring" foi o defeito do lote `b_release`
        # (ACHADOS.md, 2026-09-16).
        #
        # O peso publicado no Hub é BIT A BIT este: medido com
        # `HfApi.repo_info(files_metadata=True)`, sem baixar, o repositório
        # `juliadollis/genrefocus-deblurnet` traz sha256 `a1ed05e4…` e 1.854.533.784 B,
        # idênticos aos da cópia do cluster que gerou os lotes. (`deea4988…`, de 688
        # chaves, é do `…-paper-4gpu` — é o peso da OURS_MAIN_COND, não deste repo. A
        # atribuição em `ROTA_B_AUDITORIA_2.md:541` / achado A17 está trocada.)
        #
        # MAS o arquivo LÁ se chama `deblurNet.safetensors`, e aqui em disco se chama
        # `deblur.safetensors` — o mesmo peso com dois nomes, e um deles é o nome
        # declarado pela variante OFICIAL. É por isso que, quando há `pinned_sha256` e
        # ele CASA, a identificação do peso não passa pelo nome: o hash é evidência
        # estritamente mais forte, e deixar o nome decidir acusaria o nosso próprio peso
        # de ser o da oficial. Ver `_identifica_o_peso`.
        evidence="deblur.json adapter_variant=cond-only; "
                 "train_deblur_docker_4gpu.yaml:100 lora_on_main=false; "
                 "RESULTADOS.md:153 (step 60.000); rank 128 medido no header; "
                 "sha256 medido em AUDITORIA_TREINO_DEBLURNET.md:35 (1.854.533.784 B) e "
                 "no Hub por HfApi.repo_info, onde o arquivo é deblurNet.safetensors",
        pinned_sha256="a1ed05e48f7a3469cf825eda"
                      "6a8ceca4cca1e6f6dbdb665d78a6f9e207d8cecc",
    ),
    DeblurVariant.OFFICIAL_COND_ONLY: DeblurVariantSpec(
        variant=DeblurVariant.OFFICIAL_COND_ONLY,
        repo_id="nycu-cplab/Genfocus-Model",
        weight_filename="deblurNet.safetensors",
        main_adapter=None,
        lora_mode="cond_only",
        evidence="Inference_deblurNet.py:11,88-89,103-111; download_models.py:11",
    ),
}

#: Nome de arquivo -> variante a que ele pertence. Usado só para dar mensagem de erro
#: ESPECÍFICA quando alguém aponta o peso de uma variante com a outra selecionada, que é
#: o acidente realista (dois snapshots do HF no mesmo cache).
#: Um nome pode pertencer a MAIS DE UMA variante: `deblur.safetensors` é o nome tanto do
#: nosso treino main+cond antigo quanto do cond-only novo. Era um `dict` de um dono só, e
#: a segunda variante simplesmente sobrescrevia a primeira em silêncio.
_FILENAME_OWNERS: dict[str, tuple[DeblurVariant, ...]] = {}
for _variante, _spec in _VARIANT_SPECS.items():
    _FILENAME_OWNERS.setdefault(_spec.weight_filename, ())
    _FILENAME_OWNERS[_spec.weight_filename] += (_variante,)
del _variante, _spec

#: Nomes compartilhados por mais de uma variante. Para estes o nome de arquivo NÃO é
#: prova de nada, e o runtime exige sha256.
_AMBIGUOUS_FILENAMES: frozenset[str] = frozenset(
    nome for nome, donos in _FILENAME_OWNERS.items() if len(donos) > 1
)

# Cobertura total do enum, conferida na importação: uma variante sem spec só apareceria
# em produção, e apareceria como KeyError sem contexto.
_faltando = [v for v in DeblurVariant if v not in _VARIANT_SPECS]
if _faltando:
    raise RuntimeError(f"DeblurVariant sem spec: {[v.value for v in _faltando]}")
del _faltando


def variants_of_weight_filename(filename: str) -> tuple[DeblurVariant, ...]:
    """TODAS as variantes a que um nome de arquivo pertence. Vazio se desconhecido."""
    return _FILENAME_OWNERS.get(Path(filename).name, ())


def variant_of_weight_filename(filename: str) -> Optional[DeblurVariant]:
    """A ÚNICA variante dona deste nome, ou `None`.

    Devolve `None` tanto para nome desconhecido quanto para nome AMBÍGUO — porque nos
    dois casos o nome não identifica variante nenhuma, e devolver uma das duas seria
    exatamente o palpite que produziu o defeito. Quem precisa saber quem compartilha o
    nome usa `variants_of_weight_filename`.
    """
    donos = variants_of_weight_filename(filename)
    return donos[0] if len(donos) == 1 else None


def filename_is_ambiguous(filename: str) -> bool:
    """O nome pertence a mais de uma variante?"""
    return Path(filename).name in _AMBIGUOUS_FILENAMES


def resolve_weights(
    variant: DeblurVariant,
    *,
    local_dir: str | Path | None = None,
    cache_dir: str | Path | None = None,
    token: Optional[str] = None,
) -> Path:
    """Caminho local do `.safetensors` DA VARIANTE. Sem parâmetro de repo nem de arquivo.

    O antigo aceitava `reference` e `filename` livres
    (`bokehnet-preprocessing/src/model_runtime/deblurnet.py:69-95`), o que permite pedir
    `deblurNet.safetensors` no repositório do nosso LoRA e vice-versa. Aqui os dois vêm
    da variante e não há como contradizê-los.

    `local_dir` é o único ponto de flexibilidade: um diretório já baixado. O nome do
    arquivo dentro dele continua sendo o da variante.

    `token` nunca é impresso nem entra na proveniência.
    """
    variant = DeblurVariant(variant)
    spec = variant.spec

    if local_dir is not None:
        candidato = Path(local_dir).expanduser() / spec.weight_filename
        if not candidato.is_file():
            raise FileNotFoundError(
                f"{candidato} não existe. A variante {variant.value} exige o arquivo "
                f"{spec.weight_filename!r} (de {spec.repo_id}). NÃO troque pelo arquivo "
                "da outra variante: o peso não sabe de qual convenção de adapter ele "
                "precisa, e rodar main+cond como cond-only sai LAVADO."
            )
        return candidato.resolve()

    from huggingface_hub import hf_hub_download      # import tardio: dep opcional

    baixado = hf_hub_download(
        repo_id=spec.repo_id,
        filename=spec.weight_filename,
        cache_dir=str(cache_dir) if cache_dir is not None else None,
        token=token,
    )
    return Path(baixado).resolve()


# --------------------------------------------------------------------------------
# Geometria — o recorte por `long_side`, medido e registrado
# --------------------------------------------------------------------------------

class ResizePolicy(str, Enum):
    """Como a imagem chega ao múltiplo de 16 que o VAE exige. Enum fechado.

    `NO_CROP_MULTIPLE_OF_16` é o default e é o que a rota B usa. Coincide byte a byte
    com o oficial quando `long_side == 0`, que é o valor da rota B.
    """

    NO_CROP_MULTIPLE_OF_16 = "no_crop_multiple_of_16"
    OFFICIAL_CENTER_CROP_16 = "official_center_crop_16"


def _ceil_to(valor: int, multiplo: int) -> int:
    return max(multiplo, ((int(valor) + multiplo - 1) // multiplo) * multiplo)


def _floor_to(valor: int, multiplo: int) -> int:
    return max(multiplo, (int(valor) // multiplo) * multiplo)


def _max_shift_px(comprimento: int, redimensionado: int, final: int, offset: int) -> float:
    """Deslocamento máximo, em px da resolução ORIGINAL, do round-trip por eixo.

    Um ponto em `y` da original vai para `y * (redimensionado / comprimento) - offset` na
    imagem processada, e volta multiplicado por `comprimento / final`:

        y_out = y * (redimensionado / final) - offset * (comprimento / final)

    Sem recorte, `redimensionado == final` e `offset == 0`, logo `y_out == y` — zero
    exato, não "aproximadamente zero". Com recorte, a inclinação sai de 1 e aparece um
    deslocamento que cresce com o comprimento: 30,86 px medidos num 5184×3456 com
    `long_side=512`, contra o gate de 6,0 px do pipeline antigo (`route_b.py:411-422`).
    """
    if final <= 0:
        raise ValueError(f"lado final inválido: {final}")
    inclinacao = redimensionado / final
    deslocamento = offset * (comprimento / final)
    extremos = (0.0 * (inclinacao - 1.0) - deslocamento,
                float(comprimento) * (inclinacao - 1.0) - deslocamento)
    return max(abs(v) for v in extremos)


@dataclass(frozen=True)
class ProcessingPlan:
    """O que será feito com a imagem, decidido ANTES de tocar em pixel.

    Função pura de `(image_hw, long_side, policy)`: testável sem torch, sem GPU e sem
    checkpoint, que é como a tabela de recorte do docstring do módulo foi medida.
    """

    image_hw: tuple[int, int]
    #: Resolução intermediária da aritmética (lado longo em `long_side`, proporção
    #: preservada). MATERIALIZADA só quando há recorte — sem recorte a imagem vai direto
    #: para `processed_hw` num Lanczos só, e este campo fica como registro do cálculo.
    resized_hw: tuple[int, int]
    processed_hw: tuple[int, int]
    policy: ResizePolicy
    long_side: int
    #: `(left, top, right, bottom)` no referencial de `resized_hw`, ou `None`.
    crop_box: Optional[tuple[int, int, int, int]]
    #: Fração do campo de visão retida, `(y, x)`. `(1.0, 1.0)` quando não há recorte.
    fov_retained_hw: tuple[float, float]
    max_registration_shift_px: float

    @property
    def crop_applied(self) -> bool:
        return self.crop_box is not None

    @property
    def no_tiled_denoise(self) -> bool:
        """`min(w, h) < 512` — igual ao oficial (`Inference_deblurNet.py:95`)."""
        return min(self.processed_hw) < TILING_MIN_SHORT_SIDE

    def to_dict(self) -> dict:
        h, w = self.image_hw
        ph, pw = self.processed_hw
        return {
            "image_h": h, "image_w": w,
            "processed_h": ph, "processed_w": pw,
            "resized_h": self.resized_hw[0], "resized_w": self.resized_hw[1],
            "resize_policy": self.policy.value,
            "long_side": int(self.long_side),
            "crop_applied": self.crop_applied,
            "crop_box": list(self.crop_box) if self.crop_box else None,
            "fov_retained_h": self.fov_retained_hw[0],
            "fov_retained_w": self.fov_retained_hw[1],
            "max_registration_shift_px": self.max_registration_shift_px,
            "no_tiled_denoise": self.no_tiled_denoise,
        }


def plan_processing(
    image_hw: tuple[int, int],
    *,
    long_side: int = 0,
    policy: ResizePolicy = ResizePolicy.NO_CROP_MULTIPLE_OF_16,
) -> ProcessingPlan:
    """Decide a resolução de processamento, e mede o que ela custa em geometria.

    Reproduz a aritmética de `Inference_deblurNet.py:13-49` para
    `OFFICIAL_CENTER_CROP_16`, inclusive `int()` truncando o lado curto e
    `(new // 16) * 16` recortando — é para poder medir o recorte, não para usá-lo.
    """
    policy = ResizePolicy(policy)
    height, width = int(image_hw[0]), int(image_hw[1])
    if height <= 0 or width <= 0:
        reject("resolution_invalid", f"image_hw={image_hw!r}")
    long_side = int(long_side)
    if long_side < 0:
        raise ValueError(f"long_side não pode ser negativo: {long_side}")
    if 0 < long_side < SIZE_MULTIPLE:
        # `max(..., 16)` do oficial (`Inference_deblurNet.py:33-34`) elevaria o lado
        # final ACIMA do redimensionado, e a caixa de recorte sairia com coordenada
        # negativa. Não é um caso a tratar, é um pedido sem sentido.
        raise ValueError(
            f"long_side={long_side} é menor que o múltiplo exigido pelo VAE "
            f"({SIZE_MULTIPLE}); use 0 para manter a resolução original."
        )

    if long_side > 0:
        if width >= height:
            novo_w = long_side
            novo_h = int(height * (long_side / width))
        else:
            novo_h = long_side
            novo_w = int(width * (long_side / height))
        novo_h, novo_w = max(novo_h, 1), max(novo_w, 1)
    else:
        novo_h, novo_w = height, width

    if policy is ResizePolicy.OFFICIAL_CENTER_CROP_16 and long_side > 0:
        final_w = _floor_to(novo_w, SIZE_MULTIPLE)
        final_h = _floor_to(novo_h, SIZE_MULTIPLE)
        left = (novo_w - final_w) // 2
        top = (novo_h - final_h) // 2
        if (final_h, final_w) == (novo_h, novo_w):
            # `crop_applied` significa "campo de visão PERDIDO", não "a política é a que
            # recorta". Um 4:3 com long_side=512 cai em 512x384, múltiplo de 16 nos dois
            # lados, e o recorte é uma no-op — gravar `crop_applied: True` aí faria a
            # proveniência afirmar uma perda que não houve.
            crop_box = None
            fov = (1.0, 1.0)
            shift = 0.0
        else:
            crop_box = (left, top, left + final_w, top + final_h)
            fov = (final_h / novo_h, final_w / novo_w)
            shift = max(_max_shift_px(height, novo_h, final_h, top),
                        _max_shift_px(width, novo_w, final_w, left))
    else:
        # Único ramo que a rota B usa, e o mesmo que o oficial toma com `long_side=0`
        # (`Inference_deblurNet.py:43-49`): arredonda para CIMA, não recorta.
        final_w = _ceil_to(novo_w, SIZE_MULTIPLE)
        final_h = _ceil_to(novo_h, SIZE_MULTIPLE)
        crop_box = None
        fov = (1.0, 1.0)
        shift = 0.0

    return ProcessingPlan(
        image_hw=(height, width),
        resized_hw=(novo_h, novo_w),
        processed_hw=(final_h, final_w),
        policy=policy,
        long_side=long_side,
        crop_box=crop_box,
        fov_retained_hw=fov,
        max_registration_shift_px=float(shift),
    )


# --------------------------------------------------------------------------------
# Hashes e inspeção do terceiro — tudo antes de importar torch
# --------------------------------------------------------------------------------

def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


#: Extensões cujo conteúdo entra inteiro no fingerprint do backbone, quando pequenas.
_TEXT_SUFFIXES = frozenset({".json", ".txt", ".md", ".py"})
_TEXT_MAX_BYTES = 1 << 20


def _snapshot_fingerprint(root: Path) -> str:
    """Impressão digital de um snapshot GRANDE. NÃO é hash de conteúdo dos pesos.

    O FLUX.1-dev tem dezenas de GB; hashear o conteúdo inteiro por run é inviável, e um
    hash que ninguém roda é pior que nenhum. Aqui entram: caminho relativo e tamanho de
    todo arquivo não oculto, mais o conteúdo integral dos arquivos de configuração
    pequenos. Isso detecta snapshot trocado, incompleto ou com config alterada, e é
    honesto sobre o que não detecta — a chave da proveniência diz o tipo
    (`flux_backbone_fingerprint_kind`), justamente para não parecer o que não é.

    Ignora componente oculto pela mesma razão de `segmentation._model_files`: o
    `.cache/huggingface/download/*.metadata` carrega etag e horário e varia entre
    máquinas para o MESMO modelo.
    """
    root = Path(root)
    digest = hashlib.sha256()
    arquivos = sorted(
        p for p in root.rglob("*")
        if p.is_file() and not any(parte.startswith(".")
                                  for parte in p.relative_to(root).parts)
    )
    for arquivo in arquivos:
        rel = str(arquivo.relative_to(root))
        tamanho = arquivo.stat().st_size
        digest.update(f"{rel}:{tamanho}\n".encode("utf-8"))
        if arquivo.suffix.lower() in _TEXT_SUFFIXES and tamanho <= _TEXT_MAX_BYTES:
            digest.update(arquivo.read_bytes())
    return digest.hexdigest()


def _git_commit(repo: Path) -> Optional[str]:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            text=True, stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _inspect_flux_generate(flux_py: Path) -> dict:
    """Confere por AST, SEM importar o módulo, que `generate` ainda aceita `main_adapter`.

    Este é o guarda-corpo do defeito B1, e existe por um motivo concreto: a assinatura
    de `generate` termina em `**params: dict`. Um kwarg que o upstream renomeie ou remova
    **não** levanta `TypeError` — é absorvido pelo `**params` e ignorado, e a inferência
    volta a ser cond-only sem uma linha de log. Um teste de assinatura em tempo de
    inferência não pega isso; a checagem antes de carregar 50 GB de backbone, pega.

    Também confere `Condition` e `seed_everything`, que `load()` importa.

    Por AST e não por `import`: `Genfocus/pipeline/flux.py` importa torch, diffusers e
    cv2 no nível de módulo — e no cluster o cv2 do `~/.local` quebra com
    `GLIBC_2.38 not found` (mesmo motivo da sentinela em `renderer/bokehme.py:181`).
    Erro de checkout tem que aparecer como erro de checkout.
    """
    fonte = flux_py.read_text(encoding="utf-8")
    arvore = ast.parse(fonte)

    funcoes = {no.name: no for no in arvore.body
               if isinstance(no, (ast.FunctionDef, ast.AsyncFunctionDef))}
    classes = {no.name for no in arvore.body if isinstance(no, ast.ClassDef)}

    faltando = [nome for nome in ("generate", "seed_everything") if nome not in funcoes]
    if "Condition" not in classes:
        faltando.append("Condition")
    if faltando:
        raise RuntimeError(
            f"{flux_py} não define {sorted(faltando)}. O upstream do Genfocus mudou: "
            "revise este adaptador em vez de contornar."
        )

    generate = funcoes["generate"]
    argumentos = generate.args
    nomes = ([a.arg for a in getattr(argumentos, "posonlyargs", [])]
             + [a.arg for a in argumentos.args]
             + [a.arg for a in argumentos.kwonlyargs])
    tem_kwargs = argumentos.kwarg is not None

    if "main_adapter" not in nomes:
        raise RuntimeError(
            f"`generate` em {flux_py} NÃO declara `main_adapter`"
            + (" — e tem `**{}`, então o nosso `main_adapter=` seria engolido em "
               "silêncio e a inferência voltaria a ser cond-only (defeito B1, "
               "AIF LAVADA). ".format(argumentos.kwarg.arg) if tem_kwargs else ". ")
            + "Pare e reconcilie o adaptador com o upstream."
        )

    # O default de `main_adapter` é `None` = cond-only. Não é erro: é a razão de
    # passarmos o argumento SEMPRE, explicitamente. Fica registrado na proveniência
    # para que uma mudança de default no upstream seja visível no dado.
    default_e_none = False
    n_defaults = len(argumentos.defaults)
    posicionais = [a.arg for a in getattr(argumentos, "posonlyargs", [])] + \
                  [a.arg for a in argumentos.args]
    if n_defaults and "main_adapter" in posicionais:
        indice = posicionais.index("main_adapter") - (len(posicionais) - n_defaults)
        if 0 <= indice < n_defaults:
            default_e_none = isinstance(argumentos.defaults[indice], ast.Constant) and \
                             argumentos.defaults[indice].value is None
    else:
        for arg, default in zip(argumentos.kwonlyargs, argumentos.kw_defaults):
            if arg.arg == "main_adapter":
                default_e_none = isinstance(default, ast.Constant) and default.value is None

    return {
        "accepts_main_adapter": True,
        "main_adapter_default_is_none": bool(default_e_none),
        "swallows_unknown_kwargs": bool(tem_kwargs),
        "flux_py_sha256": hashlib.sha256(fonte.encode("utf-8")).hexdigest(),
    }


# --------------------------------------------------------------------------------
# O runtime
# --------------------------------------------------------------------------------

@dataclass
class DeblurredAIF:
    """A AIF e a geometria em que ela foi produzida.

    A geometria não é opcional nem sai por log: viaja com a imagem e vai para a
    proveniência da amostra. `route_b` grava `provenance` inteiro em
    `SampleProvenance.deblurnet`.
    """

    aif_rgb: np.ndarray
    plan: ProcessingPlan
    provenance: dict


@dataclass
class DeblurNetRuntime:
    """Carrega o FLUX + o LoRA da variante uma vez, e reusa por todo o run.

    Molde de `DepthProRuntime` (`model_runtime/depth.py:68-143`): `__post_init__` valida
    caminhos e calcula o sha256 **antes** de importar torch; `load()` é separado, para o
    hash ser barato; `provenance()` devolve o que foi usado de verdade.

    Note o que NÃO está na assinatura: `main_adapter`, `repo_id`, nome do arquivo de
    pesos, `prompt`. Todos derivam de `variant`, e é isso que torna o cruzamento
    inexpressável.
    """

    variant: DeblurVariant
    weights_path: Path
    flux_dir: Path
    genfocus_dir: Path
    device: str = "cuda"
    num_steps: int = NUM_INFERENCE_STEPS
    long_side: int = 0
    resize_policy: ResizePolicy = ResizePolicy.NO_CROP_MULTIPLE_OF_16
    seed: int = SEED
    #: Trava opcional de peso: o sha256 que se espera. Quando dado, divergência é erro.
    #: É a defesa contra o único cruzamento que a checagem de nome não pega — renomear
    #: o arquivo da outra variante.
    expected_lora_sha256: Optional[str] = None
    #: Recortar o campo de visão exige dizer isso em voz alta. Ver o docstring do módulo.
    acknowledge_fov_crop: bool = False
    compile_transformer: bool = False
    batch_tiles: bool = True

    _spec: Optional[DeblurVariantSpec] = None
    _lora_sha: Optional[str] = None
    _flux_fingerprint: Optional[str] = None
    _flux_generate: dict = field(default_factory=dict)
    _genfocus_commit: Optional[str] = None
    _pipe: Any = None
    _generate: Any = None
    _condition_cls: Any = None
    _seed_everything: Any = None
    _transformer_kwargs: dict = field(default_factory=dict)
    calls: int = 0

    # -- validação, tudo antes de importar torch ------------------------------

    def __post_init__(self) -> None:
        # 1. A variante. String livre morre aqui, com o ValueError do enum.
        self.variant = DeblurVariant(self.variant)
        self._spec = self.variant.spec

        self.resize_policy = ResizePolicy(self.resize_policy)

        if int(self.num_steps) <= 0:
            raise ValueError(f"num_steps tem que ser positivo: {self.num_steps}")
        if int(self.long_side) < 0:
            raise ValueError(f"long_side não pode ser negativo: {self.long_side}")
        if 0 < int(self.long_side) < SIZE_MULTIPLE:
            raise ValueError(
                f"long_side={self.long_side} é menor que {SIZE_MULTIPLE}, o múltiplo que "
                "o VAE do FLUX exige; use 0 para manter a resolução original."
            )

        # 2. Os pesos pertencem À VARIANTE selecionada. Esta é a checagem que torna o
        #    cruzamento um erro em vez de uma AIF ruim. O sha256 vem ANTES da checagem
        #    de nome: o hash é a identidade do peso, o nome é só uma pista — ver
        #    `_identifica_o_peso`.
        self.weights_path = Path(self.weights_path).expanduser()
        if not self.weights_path.is_file():
            raise FileNotFoundError(
                f"LoRA da DeblurNet não encontrado: {self.weights_path}. NÃO existe "
                f"fallback: a variante {self.variant.value} exige "
                f"{self._spec.weight_filename!r} de {self._spec.repo_id}, e trocar pelo "
                "arquivo da outra variante produz AIF lavada sem levantar exceção."
            )
        self.weights_path = self.weights_path.resolve()
        self._lora_sha = _sha256(self.weights_path)
        if self.expected_lora_sha256 and self._lora_sha != self.expected_lora_sha256:
            raise DeblurVariantMismatch(
                f"sha256 do LoRA divergente: esperado {self.expected_lora_sha256}, "
                f"encontrado {self._lora_sha} em {self.weights_path}. Um peso renomeado "
                "passa pela checagem de nome de arquivo; o hash não passa."
            )
        self._identifica_o_peso()

        # 3. O backbone. `model_index.json` é o que prova que é um snapshot do diffusers.
        self.flux_dir = Path(self.flux_dir).expanduser()
        if not (self.flux_dir / "model_index.json").is_file():
            raise FileNotFoundError(
                f"{self.flux_dir} não parece um snapshot do FLUX.1-dev "
                "(falta model_index.json). Baixe com `snapshot_download` antes; "
                "não existe backbone alternativo."
            )
        self.flux_dir = self.flux_dir.resolve()
        self._flux_fingerprint = _snapshot_fingerprint(self.flux_dir)

        # 4. O checkout do Genfocus, e a assinatura de `generate` por AST.
        self.genfocus_dir = Path(self.genfocus_dir).expanduser()
        flux_py = self.genfocus_dir / "Genfocus" / "pipeline" / "flux.py"
        if not (self.genfocus_dir / "Inference_deblurNet.py").is_file() or not flux_py.is_file():
            raise FileNotFoundError(
                f"{self.genfocus_dir} não parece um checkout do nycu-cplab/Genfocus "
                f"(esperado Inference_deblurNet.py e {flux_py.relative_to(self.genfocus_dir)}). "
                "Clone e congele o commit, como o BokehMe em third_party/README.md — o "
                "`bokehnet-preprocessing/src/vendor/genfocus_flux.py` é referência "
                "histórica e NÃO deve ser importado daqui."
            )
        self.genfocus_dir = self.genfocus_dir.resolve()
        self._flux_generate = _inspect_flux_generate(flux_py)
        self._genfocus_commit = _git_commit(self.genfocus_dir)

        # 5. Recorte de campo de visão: só com reconhecimento explícito.
        #    A condição é a POLÍTICA poder recortar, não uma imagem de exemplo recortar:
        #    um quadrado nunca recorta, e sondar com um quadrado deixaria passar a
        #    configuração que recorta todo retrato 3:2 do dataset.
        if self._pode_recortar() and not self.acknowledge_fov_crop:
            raise ValueError(
                f"resize_policy={self.resize_policy.value} com long_side={self.long_side} "
                "RECORTA o campo de visão. Medido: até 1,47% do lado curto e 30,86 px de "
                "deslocamento AIF↔bokeh num 5184x3456 com long_side=512, contra o gate de "
                "6,0 px do pipeline antigo (route_b.py:411-422). Recorte muda o campo de "
                "visão e portanto a relação entre focal_length_35 e a geometria da imagem. "
                "Se é isso que você quer, passe acknowledge_fov_crop=True — e a proveniência "
                f"de cada amostra vai gravar crop_box e o deslocamento. "
                f"Para não recortar, use ResizePolicy.{ResizePolicy.NO_CROP_MULTIPLE_OF_16.name}."
            )

    def _pode_recortar(self) -> bool:
        """A configuração PODE recortar campo de visão? Sobre a política, não sobre uma
        imagem: `OFFICIAL_CENTER_CROP_16` com `long_side > 0` recorta todo 3:2 do
        dataset e não recorta nenhum quadrado — sondar com uma imagem daria a resposta
        errada para o dataset inteiro."""
        return (self.resize_policy is ResizePolicy.OFFICIAL_CENTER_CROP_16
                and int(self.long_side) > 0)

    def _identifica_o_peso(self) -> None:
        """Decide se ESTE arquivo é o peso da variante selecionada. O hash manda.

        A ordem importa e ela é o conserto de um defeito concreto. O peso da
        `OURS_COND_ONLY` está publicado em `juliadollis/genrefocus-deblurnet` sob o nome
        **`deblurNet.safetensors`** — que é o nome declarado pela variante OFICIAL —, e
        em disco no cluster sob `deblur.safetensors`. Com a checagem de nome primeiro,
        baixar o NOSSO peso do NOSSO repositório e apontar para ele dava
        "cruzamento de variante: este é o peso de official_cond_only", sobre o nosso
        próprio arquivo, com o sha certo — e a mensagem mandava fazer justamente a troca
        errada.

        Regra: `pinned_sha256` que CASA encerra o assunto. Dois arquivos com o mesmo
        sha256 são o mesmo arquivo; o nome é metadado do sistema de arquivos, e um
        `mv` o muda. Não há afrouxamento aqui — a variante sem pin continua com o nome
        como única evidência, e por isso continua sujeita às regras de sempre.
        """
        if self._spec.pinned_sha256 is not None:
            # Casa → aceita, qualquer que seja o nome. Não casa → `_check_sha...`
            # levanta, e com a mensagem certa (quem mais é dono do nome).
            self._check_sha_identifies_variant()
            return

        # Sem pin: o nome é tudo o que há, e as duas regras de sempre valem.
        self._check_weights_belong_to_variant()
        self._check_sha_identifies_variant()

    def _check_sha_identifies_variant(self) -> None:
        """Quando o NOME não distingue variantes, o hash tem que distinguir.

        `deblur.safetensors` é o nome do nosso treino main+cond antigo E do cond-only
        novo. A checagem de nome aprova os dois para qualquer das duas variantes, e foi
        por essa fresta que o lote `b_release` inteiro saiu: peso cond-only rodado com
        `main_adapter="deblurring"`, LoRA aplicada num ramo que o treino nunca viu, AIF
        macia, Depth Pro achatando a cena, e nenhuma exceção em 13.615 amostras.

        Duas regras, nesta ordem:

        1. Se a spec fixa um `pinned_sha256`, ele é OBRIGATÓRIO. Não é aviso. E, como
           ele é evidência mais forte que o nome, quando ele CASA o `_identifica_o_peso`
           nem chega a consultar o nome do arquivo.
        2. Se o nome é ambíguo e a spec não fixa hash, exige-se `expected_lora_sha256`
           explícito de quem chamou. Recusar é o certo: a alternativa é escolher uma das
           duas variantes por sorteio de dicionário.
        """
        fixado = self._spec.pinned_sha256
        if fixado is not None:
            if self._lora_sha != fixado:
                outras = [v for v in variants_of_weight_filename(
                    self.weights_path.name) if v is not self.variant]
                descricao = "; ".join(
                    f"{v.value} ({v.spec.repo_id}, main_adapter={v.spec.main_adapter!r}, "
                    f"{v.spec.lora_mode})" for v in outras
                )
                raise DeblurVariantMismatch(
                    f"peso errado para a variante {self.variant.value} "
                    f"({self._spec.repo_id}, main_adapter={self._spec.main_adapter!r}, "
                    f"{self._spec.lora_mode}): ela é identificada pelo sha256 {fixado}, e "
                    f"{self.weights_path} tem {self._lora_sha}. "
                    + (f"O nome {self.weights_path.name!r} é o de {descricao} — é "
                       "provável que este seja o peso de lá. " if outras else "")
                    + "A identidade do peso é o HASH, não o nome do arquivo: a tripla "
                      "(repositório, arquivo, main_adapter) é INDIVISÍVEL e o nome não a "
                      "identifica."
                )
            return

        if filename_is_ambiguous(self.weights_path.name):
            donos = [v.value for v in
                     variants_of_weight_filename(self.weights_path.name)]
            if not self.expected_lora_sha256:
                raise DeblurVariantMismatch(
                    f"{self.weights_path.name!r} é o nome de MAIS DE UMA variante "
                    f"({donos}), e {self.variant.value} não fixa sha256. Passe "
                    f"`expected_lora_sha256` explicitamente — o hash deste arquivo é "
                    f"{self._lora_sha}. Sem isso eu estaria escolhendo a variante pelo "
                    "nome do arquivo, que é precisamente o que produziu a AIF errada em "
                    "`b_release` (ACHADOS.md, 2026-09-16)."
                )

    def _check_weights_belong_to_variant(self) -> None:
        """O nome do arquivo tem que ser o da variante. Sem exceção, sem heurística.

        O peso em si não distingue main+cond de cond-only — as chaves do LoRA são as
        mesmas, a diferença vive no roteamento em tempo de inferência. Então esta
        checagem, mais `expected_lora_sha256`, é tudo o que dá para verificar sobre o
        arquivo; o resto da defesa é a assinatura deste módulo não ter por onde
        expressar a combinação errada.
        """
        nome = self.weights_path.name
        if nome == self._spec.weight_filename:
            # O nome bate — mas se ele é compartilhado, bater não prova nada. Quem
            # decide nesse caso é o sha256, em `_check_sha_identifies_variant`.
            return
        donos = variants_of_weight_filename(nome)
        if donos:
            # TODOS os donos do nome, não um só. `variant_of_weight_filename` devolve
            # `None` para nome ambíguo, e usá-la aqui fazia o cruzamento mais realista
            # (o NOSSO `deblur.safetensors` com a variante oficial selecionada) cair no
            # ramo de baixo e sair como "não é o arquivo de nenhuma variante conhecida" —
            # o que é falso e manda a pessoa procurar a coisa errada.
            descricao = "; ".join(
                f"{d.value} ({d.spec.repo_id}, main_adapter={d.spec.main_adapter!r}, "
                f"{d.spec.lora_mode})" for d in donos
            )
            raise DeblurVariantMismatch(
                f"cruzamento de variante: {nome!r} é o peso de {descricao}, mas a "
                f"variante selecionada é "
                f"{self.variant.value} ({self._spec.repo_id}, "
                f"main_adapter={self._spec.main_adapter!r}, {self._spec.lora_mode}). "
                "A tripla (repositório, arquivo, main_adapter) é INDIVISÍVEL: rodar "
                "main+cond como cond-only sai LAVADO e rodar cond-only como main+cond "
                "aplica o LoRA num branch em que ele não foi treinado. Nenhuma das duas "
                "levanta exceção na inferência — só produz uma AIF ruim, e a AIF é a "
                "entrada do Depth Pro, do BiRefNet, do plano de foco e do K."
            )
        raise DeblurVariantMismatch(
            f"{nome!r} não é o arquivo de nenhuma variante conhecida. A variante "
            f"{self.variant.value} exige {self._spec.weight_filename!r}. Renomear o peso "
            "para contornar esta checagem é reintroduzir o defeito B1 — use "
            "`expected_lora_sha256` se o arquivo legitimamente tem outro nome."
        )

    # -- carga ----------------------------------------------------------------

    def plan_for(self, image_hw: tuple[int, int]) -> ProcessingPlan:
        """O plano de processamento desta imagem, sem tocar em pixel."""
        return plan_processing(image_hw, long_side=self.long_side,
                               policy=self.resize_policy)

    def load(self) -> "DeblurNetRuntime":
        """Carrega FLUX + LoRA. Separado do `__init__` para o hash ser barato de obter."""
        if self._pipe is not None:
            return self
        import sys

        import torch
        from diffusers import FluxPipeline

        if str(self.genfocus_dir) not in sys.path:
            sys.path.insert(0, str(self.genfocus_dir))
        try:
            from Genfocus.pipeline.flux import (              # type: ignore
                Condition, generate, seed_everything,
            )
        except ImportError as exc:                            # sem fallback, de propósito
            raise ImportError(
                f"não consegui importar Genfocus.pipeline.flux de {self.genfocus_dir}: "
                f"{type(exc).__name__}: {exc}. NÃO caia para "
                "`bokehnet-preprocessing/src/vendor/genfocus_flux.py`: aquele repositório "
                "é referência histórica e é onde o defeito B1 vive."
            ) from exc

        self._generate, self._condition_cls, self._seed_everything = (
            generate, Condition, seed_everything,
        )

        dtype = torch.bfloat16 if str(self.device).startswith("cuda") else torch.float32
        pipe = FluxPipeline.from_pretrained(str(self.flux_dir), torch_dtype=dtype)
        pipe.to(self.device)
        # `adapter_name="deblurring"` é comum às duas variantes
        # (Inference_deblurNet.py:88-89). O que distingue as variantes é `main_adapter`
        # em `generate`, não este nome.
        pipe.load_lora_weights(
            str(self.weights_path.parent),
            weight_name=self.weights_path.name,
            adapter_name=ADAPTER_NAME,
        )
        pipe.set_adapters([ADAPTER_NAME])
        self._pipe = pipe

        if self.compile_transformer:
            from Genfocus.pipeline.flux import (              # type: ignore
                block_forward, single_block_forward,
            )
            self._transformer_kwargs = {
                "block_forward": torch.compile(block_forward),
                "single_block_forward": torch.compile(single_block_forward),
            }
        return self

    # -- proveniência ---------------------------------------------------------

    def provenance(self) -> dict:
        """Vai inteiro em `SampleProvenance.deblurnet`, em CADA amostra.

        Grava a variante, o sha256 do LoRA, o repositório, o arquivo e o `main_adapter`
        **efetivamente passado** a `generate` — não o default de ninguém. Sem esses
        cinco campos não dá para provar depois se a AIF de uma amostra saiu lavada.
        """
        spec = self._spec
        return {
            "deblur_backend": BACKEND_NAME,
            "deblur_variant": self.variant.value,
            "deblur_lora_sha256": self._lora_sha,
            "deblur_repo_id": spec.repo_id,
            # O nome REAL do arquivo em disco, não o da spec: o mesmo peso está
            # publicado como `deblurNet.safetensors` e local como `deblur.safetensors`,
            # e gravar o nome declarado faria a proveniência mentir sobre o que foi
            # carregado. Quem quer o nome CANÔNICO da variante lê a spec pelo
            # `deblur_variant`, que também está aqui.
            "deblur_weight_filename": self.weights_path.name,
            "deblur_weight_filename_declared": spec.weight_filename,
            "deblur_weights_path": str(self.weights_path),
            # O valor literal passado a `generate`. `None` para a variante oficial, que
            # é o CORRETO para um LoRA cond-only — e é por isso que existe
            # `deblur_lora_mode`, que nunca é vazio e pode ser validado por truthiness.
            "main_adapter": spec.main_adapter,
            "deblur_lora_mode": spec.lora_mode,
            "adapter_name": ADAPTER_NAME,
            "prompt": PROMPT,
            "num_inference_steps": int(self.num_steps),
            "long_side": int(self.long_side),
            "resize_policy": self.resize_policy.value,
            "seed": int(self.seed),
            "device": self.device,
            "flux_dir": str(self.flux_dir),
            "flux_backbone_fingerprint": self._flux_fingerprint,
            "flux_backbone_fingerprint_kind": "paths_sizes_and_small_configs",
            "genfocus_dir": str(self.genfocus_dir),
            "genfocus_commit": self._genfocus_commit,
            "genfocus_pipeline_sha256": self._flux_generate.get("flux_py_sha256"),
            "genfocus_generate_accepts_main_adapter":
                self._flux_generate.get("accepts_main_adapter"),
            "genfocus_generate_main_adapter_default_is_none":
                self._flux_generate.get("main_adapter_default_is_none"),
            "genfocus_generate_swallows_unknown_kwargs":
                self._flux_generate.get("swallows_unknown_kwargs"),
            "compile_transformer": bool(self.compile_transformer),
            "batch_tiles": bool(self.batch_tiles),
        }

    # -- inferência -----------------------------------------------------------

    def infer(self, bokeh_rgb: np.ndarray) -> DeblurredAIF:
        """AIF em RGB uint8, na resolução da imagem recebida.

        RGB porque `DepthProRuntime.infer` e `BiRefNetRuntime.infer` também são RGB: a
        AIF que sai daqui entra nos dois sem conversão. Quem grava em disco converte e
        declara `Sample.channel_order`.

        O `main_adapter` é passado SEMPRE, explicitamente, a partir da spec da variante.
        Nunca omitido — o default de `generate` é `None`, e foi ele que produziu o
        defeito B1.
        """
        # Forma e plano ANTES de carregar: entrada errada tem que aparecer como
        # `resolution_invalid`, não como falha ao subir 50 GB de backbone.
        bokeh_rgb = np.asarray(bokeh_rgb)
        if bokeh_rgb.ndim != 3 or bokeh_rgb.shape[2] != 3:
            reject("resolution_invalid", f"esperado HxWx3, recebido {bokeh_rgb.shape}")
        image_hw = (int(bokeh_rgb.shape[0]), int(bokeh_rgb.shape[1]))
        plano = self.plan_for(image_hw)

        from PIL import Image as PILImage

        if self._pipe is None:
            self.load()

        entrada = PILImage.fromarray(bokeh_rgb.astype(np.uint8), mode="RGB")
        proc_h, proc_w = plano.processed_hw
        if plano.crop_box is None:
            # Um Lanczos só, direto para a resolução de processamento. Passar pela
            # resolução intermediária seria uma segunda reamostragem sem nenhum efeito
            # geométrico — só perda de detalhe.
            if entrada.size != (proc_w, proc_h):
                entrada = entrada.resize((proc_w, proc_h), PILImage.LANCZOS)
        else:
            redim_h, redim_w = plano.resized_hw
            if (redim_h, redim_w) != image_hw:
                entrada = entrada.resize((redim_w, redim_h), PILImage.LANCZOS)
            entrada = entrada.crop(plano.crop_box)
            if entrada.size != (proc_w, proc_h):
                raise RuntimeError(
                    f"recorte devolveu {entrada.size}, plano diz {(proc_w, proc_h)}"
                )

        self._seed_everything(int(self.seed))
        condicao = self._condition_cls(entrada, ADAPTER_NAME, [0, 0], 1.0)

        import torch
        with torch.no_grad():
            saida = self._generate(
                self._pipe,
                height=proc_h,
                width=proc_w,
                prompt=PROMPT,
                num_inference_steps=int(self.num_steps),
                conditions=[condicao],
                # ---------------------------------------------------------------
                # A LINHA QUE FALTAVA. Sem ela o default é None = cond-only, e o
                # nosso checkpoint main+cond sai LAVADO (defeito B1). Vem da spec da
                # variante, nunca de um parâmetro nem de um default nosso.
                main_adapter=self._spec.main_adapter,
                # ---------------------------------------------------------------
                NO_TILED_DENOISE=plano.no_tiled_denoise,
                transformer_kwargs=self._transformer_kwargs,
                batch_tiles=self.batch_tiles,
            ).images[0]

        if saida.size != (proc_w, proc_h):
            raise RuntimeError(
                f"`generate` devolveu {saida.size}, esperado {(proc_w, proc_h)}: a "
                "geometria gravada na proveniência descreveria outra imagem."
            )
        if saida.size != (image_hw[1], image_hw[0]):
            saida = saida.resize((image_hw[1], image_hw[0]), PILImage.LANCZOS)

        aif = np.asarray(saida.convert("RGB"), dtype=np.uint8)
        if aif.shape[:2] != image_hw:
            raise RuntimeError(
                f"AIF saiu em {aif.shape[:2]}, bokeh está em {image_hw}: K vive na "
                "escala de pixel da imagem fonte e as duas têm que coincidir."
            )

        self.calls += 1
        proveniencia = dict(self.provenance())
        proveniencia["geometry"] = plano.to_dict()
        return DeblurredAIF(aif_rgb=aif, plan=plano, provenance=proveniencia)
