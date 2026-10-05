#!/usr/bin/env python3
"""Traz para o disco da h100n3 **só o pedaço do release que a bancada de
condicionamento geométrico precisa** — nunca o release inteiro.

POR QUE ISTO EXISTE, E NÃO UM `snapshot_download` DO REPO TODO
--------------------------------------------------------------
O `juliadollis/bokehnet-regen-rota-a` completo é ~30 GB, dominados por
`generated/` (69.700 JPEGs, ~28 GB estimados). A cota do raid da n3 está
**ESTOURADA** (520G usados de 500G soft, limite duro 600G, com grace correndo),
e neste projeto já houve disco cheio matando 3 treinos — ~18 h de 3 GPUs.

A bancada de condicionamento geométrico, porém, não precisa dos 28 GB:

| o que a bancada faz            | o que ela lê                              |
|--------------------------------|-------------------------------------------|
| canais geométricos (u,O,s,n,K~)| `depth/<cena>.png` + `meta/<cena>/<id>.json` |
| calibrar as constantes         | idem, sobre o lado TRAIN                  |
| E_bleed                        | idem **+** `generated/` das cenas medidas |
| smoke do treino                | idem **+** `generated/`/`mask/` de poucas cenas |

`depth/` inteiro são 1.700 arquivos, ~0,8 GB — uma profundidade POR CENA,
compartilhada pelas 41 variantes (`genfocus_train/release.py`,
`caminho_da_profundidade`). `meta/` são 69.700 JSONs pequenos e é onde vive a
**escala métrica** (`disparity_min`/`disparity_max`), sem a qual os canais
geométricos não significam nada. Esses dois vêm INTEIROS. O que é caro —
`generated/` e `mask/` — vem por AMOSTRA DE CENAS, com `--cenas N`.

AS SETE TRAVAS, E O QUE CADA UMA IMPEDE
---------------------------------------
1. **Download é OPT-IN.** Sem `--baixar` o script só PLANEJA: lista o repo pela
   API (metadado, não conteúdo), estima o tamanho e imprime. Há registro de
   552 GB de egress inexplicado neste projeto vindo de download automático;
   aqui, baixar é um ato declarado, com log e número.
2. **Teto de volume.** Acima de `--teto-gb` (default 20 GB) o download exige
   `--confirmar`. `--cenas` conta CENA, não arquivo: cada cena traz ~41
   amostras em `generated/` e ~41 em `mask/`. A confusão "o limite conta por
   arquivo ou no total?" já custou 20 h de GPU neste cluster
   (`INSTRUCOES_H100.md`, item 3 do fluxo).
3. **Espaço e cota conferidos ANTES de escrever.** `df` do destino e `quota -s`.
   Estourar o limite duro aborta; o soft já estourado vira aviso alto.
4. **Split respeitado.** As cenas de validação saem do lado `val` do
   `split.json`, e as de treino do lado `train`. Validar em cena vista no treino
   mede memorização — a mesma cena aparece com 41 aberturas, então split por
   linha vaza a cena inteira.
5. **Amostragem DETERMINÍSTICA e gravada.** A ordem é a do `sha256(semente:cena)`
   — não `random.shuffle`, que muda com a versão do Python, nem a ordem de
   listagem do Hub, que muda quando o upload muda. A semente vai para o
   `staging_manifest.json`; outra pessoa reproduz a mesma bancada.
6. **sha256 conferido contra os ledgers.** `generated_images.jsonl` traz o
   `bokeh_jpeg_sha256` de cada amostra. Conferir é o que transforma "aponta para
   o espelho" em evidência; sem isso um JPEG truncado no meio do download treina
   em silêncio.
7. **IDEMPOTENTE e NÃO-DESTRUTIVO.** Arquivo já em disco com o tamanho do
   remoto não é rebaixado. O script não tem `rm`, não tem `--delete` e não
   sobrescreve o `staging_manifest.json`: ele acumula as execuções em
   `execucoes[]`.

O TOKEN
-------
Vem de `HF_TOKEN` (ou `HUGGINGFACE_HUB_TOKEN`) no ambiente, e o `.slurm` carrega
o `.env` com `set -a; . .env; set +a`. Nunca é hardcoded e nunca é impresso.

USO
---
    # 1. planejar (não baixa nada, não escreve nada; ~1 min)
    python3 scripts/stage_release_n3.py --destino /raid/.../releases/bancada \\
        --cenas 24

    # 2. baixar de verdade
    python3 scripts/stage_release_n3.py --destino /raid/.../releases/bancada \\
        --cenas 24 --baixar --confirmar

    # só o insumo geométrico, sem pixel nenhum (cabe em 1 GB):
    python3 scripts/stage_release_n3.py --destino ... --cenas 0 --baixar

CÓDIGOS DE SAÍDA
----------------
    0  ok (plano impresso, ou download concluído e conferido)
    3  espaço/cota insuficiente — nada foi baixado
    4  sha256 divergente: o que está em disco não é o que o ledger diz
    5  estimativa acima do teto e sem `--confirmar` — nada foi baixado
    6  release incompleto para o que foi pedido (ex.: `--cenas 8` e `generated/`
       ausente no repo)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

RAIZ_DO_PROJETO = Path(__file__).resolve().parent.parent

VERSAO = "1.0"
REPO_PADRAO = "juliadollis/bokehnet-regen-rota-a"
#: Semente default FIXA e escrita aqui, não `time()`: duas bancadas montadas no
#: mesmo dia têm que ser a mesma bancada.
SEMENTE_PADRAO = 20260922
TETO_GB_PADRAO = 20.0
#: Espaço livre que tem que SOBRAR depois do download. O treino escreve
#: checkpoint; deixar o disco no talo é como se perdeu 3 treinos antes.
MARGEM_GB_PADRAO = 20.0

NOME_MANIFESTO_STAGING = "staging_manifest.json"

#: Vêm INTEIROS: são o insumo geométrico e são baratos.
GRUPOS_INTEIROS = ("depth", "meta")
#: Vêm por cena amostrada: são o caro.
GRUPOS_POR_CENA = ("generated", "mask")

#: Campos de sha256 que os ledgers do `bokehnet-regen` usam, em ordem de
#: preferência (`scripts/publish_release.py`: rota A grava `bokeh_jpeg_sha256`).
CAMPOS_SHA = (
    "bokeh_jpeg_sha256", "aif_jpeg_sha256",
    "bokeh_sha256", "aif_sha256", "sha256",
)

GB = 1024.0 ** 3


# =============================================================================
# Utilidades
# =============================================================================

def humano(n: float) -> str:
    """Bytes em unidade legível. Base 1024, como `df` e `quota -s`."""
    v = float(n)
    for u in ("B", "KB", "MB", "GB", "TB"):
        if abs(v) < 1024.0 or u == "TB":
            return f"{v:,.2f} {u}".replace(",", ".")
        v /= 1024.0
    return f"{v:.2f} TB"


def carregar_env_local(raiz: Path = RAIZ_DO_PROJETO) -> None:
    """Lê `<raiz>/.env` SEM sobrescrever o que já está no ambiente.

    O `.slurm` já carrega o `.env` com `set -a; . .env; set +a`; isto aqui é para
    quem rodar o script na mão. Precedência: ambiente > `$ENV_FILE` > `.env`,
    a mesma que o `.env.example` declara. Nenhum valor é impresso.
    """
    candidatos = [os.environ.get("ENV_FILE", ""), str(raiz / ".env")]
    for caminho in candidatos:
        if not caminho:
            continue
        arq = Path(caminho)
        if not arq.is_file():
            continue
        for linha in arq.read_text(encoding="utf-8", errors="replace").splitlines():
            linha = linha.strip()
            if not linha or linha.startswith("#") or "=" not in linha:
                continue
            chave, _, valor = linha.partition("=")
            chave = chave.strip()
            valor = valor.strip().strip('"').strip("'")
            if chave and chave not in os.environ:
                os.environ[chave] = valor
        return


def token_hf() -> str:
    tok = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    if not tok:
        raise SystemExit(
            "ERRO: HF_TOKEN ausente. O release é PRIVADO. Defina no ambiente ou "
            f"em {RAIZ_DO_PROJETO / '.env'} (modelo em .env.example) e carregue "
            "com `set -a; . .env; set +a`. O valor nunca é impresso."
        )
    return tok


def _partes(caminho: str) -> list[str]:
    return [p for p in caminho.split("/") if p]


def grupo_de(caminho: str) -> str:
    """`generated/cena/x.jpg` -> `generated`; `manifest.jsonl` -> `(topo)`."""
    p = _partes(caminho)
    return p[0] if len(p) > 1 else "(topo)"


def cena_de(caminho: str) -> str | None:
    """A cena de um caminho por amostra, nos DOIS layouts do release.

    `generated/<cena>/<id>_bokeh.jpg` (agrupado — o que o Hub obriga acima de
    10.000 arquivos por pasta) e `depth/<cena>.png` (o layout por cena do
    release da rota A). Layout plano por AMOSTRA (`generated/<id>_bokeh.jpg`)
    não tem cena no caminho e devolve `None`: quem decide aí é o manifesto.
    """
    p = _partes(caminho)
    if len(p) >= 3:
        return p[1]
    if len(p) == 2 and p[0] == "depth":
        return p[1].rsplit(".", 1)[0]
    return None


# =============================================================================
# Listagem do repositório — metadado, não conteúdo
# =============================================================================

def listar_arvore(repo: str, revisao: str, token: str) -> tuple[list[tuple[str, int]], str]:
    """`[(caminho, bytes)]` de todo o repo, e o commit resolvido.

    Tenta `huggingface_hub` primeiro; se o SIF não o tiver, cai na API HTTP.
    Esta chamada é de METADADO: ela não baixa conteúdo nenhum, e é o que permite
    estimar o volume antes de gastar rede.
    """
    try:
        from huggingface_hub import HfApi  # type: ignore

        api = HfApi(token=token)
        info = api.dataset_info(repo, revision=revisao)
        commit = getattr(info, "sha", "") or revisao
        arquivos: list[tuple[str, int]] = []
        for entrada in api.list_repo_tree(
            repo, repo_type="dataset", revision=revisao, recursive=True, expand=True
        ):
            caminho = getattr(entrada, "path", None)
            if caminho is None or getattr(entrada, "tree_id", None) is not None:
                continue  # é diretório
            tamanho = getattr(entrada, "size", None)
            lfs = getattr(entrada, "lfs", None)
            if lfs is not None and getattr(lfs, "size", None):
                tamanho = lfs.size
            arquivos.append((str(caminho), int(tamanho or 0)))
        if arquivos:
            return arquivos, str(commit)
        print("[stage] AVISO: list_repo_tree veio vazio; tentando a API HTTP.")
    except ImportError:
        print("[stage] huggingface_hub indisponível no container; usando a API HTTP.")
    except Exception as exc:  # noqa: BLE001 — qualquer falha do hub cai no HTTP
        print(f"[stage] AVISO: list_repo_tree falhou ({exc!r}); usando a API HTTP.")

    return _listar_arvore_http(repo, revisao, token)


def _http(url: str, token: str, *, metodo: str = "GET"):
    req = urllib.request.Request(url, method=metodo)
    req.add_header("Authorization", f"Bearer {token}")
    req.add_header("User-Agent", f"stage_release_n3/{VERSAO}")
    return urllib.request.urlopen(req, timeout=120)


def _listar_arvore_http(repo: str, revisao: str, token: str) -> tuple[list[tuple[str, int]], str]:
    """Mesma listagem pela API HTTP do Hub, paginada pelo header `Link`."""
    commit = revisao
    try:
        with _http(f"https://huggingface.co/api/datasets/{repo}", token) as r:
            commit = json.loads(r.read().decode("utf-8")).get("sha", revisao)
    except Exception as exc:  # noqa: BLE001
        print(f"[stage] AVISO: não consegui resolver o commit ({exc!r}).")

    url = (
        f"https://huggingface.co/api/datasets/{repo}/tree/"
        f"{urllib.parse.quote(revisao, safe='')}?recursive=1&expand=1"
    )
    arquivos: list[tuple[str, int]] = []
    while url:
        with _http(url, token) as r:
            pagina = json.loads(r.read().decode("utf-8"))
            link = r.headers.get("Link", "") or ""
        for item in pagina:
            if item.get("type") != "file":
                continue
            tamanho = item.get("size") or 0
            lfs = item.get("lfs") or {}
            if lfs.get("size"):
                tamanho = lfs["size"]
            arquivos.append((str(item["path"]), int(tamanho)))
        casamento = re.search(r'<([^>]+)>;\s*rel="next"', link)
        url = casamento.group(1) if casamento else ""
    return arquivos, str(commit)


# =============================================================================
# Split e amostragem de cenas
# =============================================================================

def baixar_arquivo_unico(
    repo: str, revisao: str, caminho: str, destino: Path, token: str
) -> Path | None:
    """Um arquivo só, direto para `destino/<caminho>`. Usado para o `split.json`.

    O plano precisa do split (é ele que separa train de val), e ele tem dezenas
    de KB — baixá-lo no modo plano é honesto e está declarado no relatório.
    """
    alvo = destino / caminho
    if alvo.is_file() and alvo.stat().st_size > 0:
        return alvo
    alvo.parent.mkdir(parents=True, exist_ok=True)
    url = (
        f"https://huggingface.co/datasets/{repo}/resolve/"
        f"{urllib.parse.quote(revisao, safe='')}/{urllib.parse.quote(caminho)}"
    )
    parcial = alvo.with_suffix(alvo.suffix + ".part")
    try:
        with _http(url, token) as r, parcial.open("wb") as saida:
            shutil.copyfileobj(r, saida, 1024 * 1024)
        parcial.replace(alvo)  # atômico: ninguém vê arquivo pela metade
        return alvo
    except Exception as exc:  # noqa: BLE001
        if parcial.exists():
            # Só o .part que ESTE processo acabou de criar. Nada mais é removido.
            parcial.unlink()
        print(f"[stage] AVISO: não baixei {caminho} ({exc!r}).")
        return None


def ler_split(caminho: Path) -> dict[str, str]:
    """`{cena: 'train'|'val'}`, nas duas formas que o projeto conhece.

    A do release é `{"assignment": {cena: lado}}` (`dataio/split.py:SceneSplit`);
    a outra é `{"train": [...], "val": [...]}`. Mesmas formas que
    `genfocus_train/release.py:carregar_assignment_de_split` aceita — ler
    diferente daqui seria a divergência silenciosa que este projeto persegue.
    """
    payload = json.loads(caminho.read_text(encoding="utf-8"))
    atribuicao = payload.get("assignment")
    if isinstance(atribuicao, dict):
        return {str(c): str(l) for c, l in atribuicao.items()}
    mapa: dict[str, str] = {}
    for lado in ("train", "val", "validation", "test"):
        for cena in payload.get(lado, []) or []:
            mapa[str(cena)] = "val" if lado in ("val", "validation") else lado
    if not mapa:
        raise SystemExit(
            f"ERRO: não reconheci o formato de {caminho}. Esperado "
            '`{"assignment": {cena: lado}}` ou `{"train": [...], "val": [...]}`.'
        )
    return mapa


def ordem_deterministica(cenas, semente: int) -> list[str]:
    """Ordem estável por `sha256(semente:cena)`.

    NÃO é `random.shuffle` com seed: a sequência do `random` é garantida entre
    execuções, não entre versões do Python, e esta bancada tem que ser
    remontável daqui a meses num container diferente. Também não é a ordem do
    Hub, que muda quando o upload muda.
    """
    return sorted(
        cenas, key=lambda c: hashlib.sha256(f"{semente}:{c}".encode("utf-8")).digest()
    )


def escolher_cenas(
    disponiveis_por_lado: dict[str, list[str]],
    n_total: int,
    n_val: int | None,
    semente: int,
) -> dict[str, list[str]]:
    """Amostra `n_total` cenas RESPEITANDO o split.

    `n_val` default: a proporção do próprio split (81/1.700 ≈ 4,8%), com piso de
    1 cena sempre que couber. Uma bancada sem cena de val não tem validação — é
    o jeito mais silencioso de não validar.
    """
    train = disponiveis_por_lado.get("train", [])
    val = disponiveis_por_lado.get("val", [])
    total_split = len(train) + len(val)
    if n_val is None:
        fracao = (len(val) / total_split) if total_split else 0.0
        n_val = int(round(n_total * fracao))
        if n_total >= 2:
            n_val = max(1, n_val)
    n_val = min(n_val, n_total, len(val))
    n_train = min(n_total - n_val, len(train))

    escolhidas = {
        "train": ordem_deterministica(train, semente)[:n_train],
        "val": ordem_deterministica(val, semente)[:n_val],
    }
    sobreposicao = set(escolhidas["train"]) & set(escolhidas["val"])
    if sobreposicao:  # defeito do split, não desta amostragem — mas tem que gritar
        raise SystemExit(
            f"ERRO: {len(sobreposicao)} cenas em train E em val no split.json "
            f"(ex.: {sorted(sobreposicao)[:3]}). O release está inconsistente."
        )
    return escolhidas


# =============================================================================
# Espaço em disco e cota
# =============================================================================

_UNIDADES = {"": 1, "K": 1024, "M": 1024 ** 2, "G": 1024 ** 3, "T": 1024 ** 4}


def _numero_de_cota(texto: str) -> float | None:
    casamento = re.fullmatch(r"([0-9]+(?:[.,][0-9]+)?)([KMGTkmgt]?)\*?", texto.strip())
    if not casamento:
        return None
    return (float(casamento.group(1).replace(",", "."))
            * _UNIDADES[casamento.group(2).upper()])


def ler_cota() -> list[dict]:
    """`quota -s` em linhas estruturadas. Silencioso se o comando não existir."""
    try:
        saida = subprocess.run(
            ["quota", "-s"], capture_output=True, text=True, timeout=30
        ).stdout
    except Exception:  # noqa: BLE001 — sem `quota` no container é normal
        return []
    linhas = []
    for linha in saida.splitlines():
        campos = linha.split()
        # Linha de dados = 1º campo parece um filesystem (`master:/raid`,
        # `/dev/sdb1`). O cabeçalho ("Filesystem space quota ...") não passa.
        if len(campos) < 4 or ("/" not in campos[0] and ":" not in campos[0]):
            continue
        usado = _numero_de_cota(campos[1])
        soft = _numero_de_cota(campos[2])
        hard = _numero_de_cota(campos[3])
        if usado is None or soft is None or hard is None:
            continue
        linhas.append(
            {"fs": campos[0], "usado": usado, "soft": soft, "hard": hard,
             "linha_crua": linha.strip()}
        )
    return linhas


def checar_espaco(destino: Path, bytes_necessarios: int, margem_gb: float) -> list[str]:
    """Devolve a lista de MOTIVOS para abortar. Vazia = pode escrever.

    Confere as duas coisas que enchem disco de jeitos diferentes: o `df` (o
    filesystem acabou) e a `quota` (o filesystem tem espaço e o SEU limite não).
    Neste cluster é a segunda que está no talo: 520G de 500G soft, duro 600G.
    """
    motivos: list[str] = []
    margem = margem_gb * GB

    alvo_existente = destino
    while not alvo_existente.exists() and alvo_existente != alvo_existente.parent:
        alvo_existente = alvo_existente.parent
    uso = shutil.disk_usage(alvo_existente)
    print(f"[disco] {alvo_existente}: livre {humano(uso.free)} de {humano(uso.total)}")
    if uso.free < bytes_necessarios + margem:
        motivos.append(
            f"df: livre {humano(uso.free)}, preciso de {humano(bytes_necessarios)} "
            f"+ {margem_gb:.0f} GB de margem. Não cabe."
        )

    for linha in ler_cota():
        print(f"[cota] {linha['linha_crua']}")
        folga_soft = linha["soft"] - linha["usado"]
        folga_hard = linha["hard"] - linha["usado"]
        if folga_soft < 0:
            print(
                f"[cota] AVISO ALTO: a cota de {linha['fs']} JÁ ESTÁ ESTOURADA "
                f"({humano(linha['usado'])} de {humano(linha['soft'])} soft). O "
                "grace está correndo: quando ele expirar, escrita vira erro "
                "mesmo abaixo do limite duro."
            )
        if bytes_necessarios + margem > folga_hard:
            motivos.append(
                f"quota: {humano(linha['usado'])} usados de {humano(linha['hard'])} "
                f"duros em {linha['fs']}; sobram {humano(max(folga_hard, 0))} e eu "
                f"preciso de {humano(bytes_necessarios)} + margem. "
                "Não vou encher o disco."
            )
    return motivos


# =============================================================================
# Download
# =============================================================================

def baixar(
    repo: str, revisao: str, padroes: list[str], destino: Path, token: str,
    workers: int, caminhos_exatos: list[str],
) -> None:
    """`snapshot_download` com `allow_patterns`; HTTP sequencial como reserva.

    `allow_patterns` é o que impede de puxar o repositório inteiro. Os padrões
    são construídos a partir da listagem, então o que entra aqui já foi contado
    e impresso — nada é baixado sem ter aparecido na estimativa.
    """
    try:
        from huggingface_hub import snapshot_download  # type: ignore
    except ImportError:
        print("[stage] huggingface_hub indisponível; baixando pela API HTTP.")
        _baixar_http(repo, revisao, caminhos_exatos, destino, token)
        return

    destino.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=repo,
        repo_type="dataset",
        revision=revisao,
        token=token,
        local_dir=str(destino),
        allow_patterns=padroes,
        max_workers=workers,
    )


def _baixar_http(
    repo: str, revisao: str, caminhos: list[str], destino: Path, token: str
) -> None:
    total = len(caminhos)
    for i, caminho in enumerate(caminhos, 1):
        if i % 500 == 0 or i == total:
            print(f"[stage] http {i}/{total}", flush=True)
        baixar_arquivo_unico(repo, revisao, caminho, destino, token)


# =============================================================================
# sha256 contra os ledgers
# =============================================================================

def sha256_do_arquivo(caminho: Path) -> str:
    h = hashlib.sha256()
    with caminho.open("rb") as arq:
        for bloco in iter(lambda: arq.read(1024 * 1024), b""):
            h.update(bloco)
    return h.hexdigest()


def indexar_ledger(caminho: Path) -> dict[str, dict]:
    """Ledger indexado por `sample_id` E por `scene_id`.

    As duas chaves porque a **rota A grava o ledger de origem por `scene_id`** —
    as 41 variantes saem dos MESMOS bytes de AIF (`publish_release.py`:
    `unidade_do_ledger_de_origem`). Indexar só por `sample_id` foi o defeito que
    deixava o ledger da rota A vazio e a conferência de sha virar no-op com
    `verificar_sha256_da_origem: true` no YAML.
    """
    indice: dict[str, dict] = {}
    if not caminho.is_file():
        return indice
    with caminho.open(encoding="utf-8") as arq:
        for linha in arq:
            linha = linha.strip()
            if not linha:
                continue
            try:
                registro = json.loads(linha)
            except json.JSONDecodeError:
                continue
            for chave in ("sample_id", "scene_id"):
                valor = registro.get(chave)
                if valor:
                    indice.setdefault(str(valor), registro)
    return indice


def conferir_sha256(destino: Path, staged: list[str], limite: int) -> dict:
    """Confere os JPEGs staged contra `generated_images.jsonl`, quando há linha.

    "Quando a linha existir" é literal: amostra sem linha é CONTADA e reportada,
    não silenciada — um ledger que não cobre o que está em disco é um release
    que não prova o pixel que ele mesmo produziu.
    """
    ledger = indexar_ledger(destino / "generated_images.jsonl")
    resultado = {
        "ledger_linhas": len(ledger),
        "conferidos": 0, "divergentes": [], "sem_linha": 0, "sem_campo_sha": 0,
    }
    if not ledger:
        resultado["observacao"] = (
            "generated_images.jsonl ausente ou vazio — nada a conferir. O "
            "release não prova o pixel que gerou."
        )
        return resultado

    alvos = [c for c in staged if c.startswith("generated/")]
    if limite > 0:
        # Amostragem determinística pelo sha do próprio caminho — a mesma regra
        # do `publish_release.py:_amostragem`. Fatiar a lista testaria sempre as
        # mesmas cenas do começo do alfabeto.
        alvos = sorted(alvos, key=lambda c: hashlib.sha256(c.encode()).digest())[:limite]

    for caminho in alvos:
        arquivo = destino / caminho
        if not arquivo.is_file():
            continue
        nome = arquivo.name
        sample_id = re.sub(r"_(bokeh|aif)\.jpg$", "", nome)
        registro = ledger.get(sample_id)
        if registro is None:
            resultado["sem_linha"] += 1
            continue
        esperado = next((registro[c] for c in CAMPOS_SHA if registro.get(c)), None)
        if not esperado:
            resultado["sem_campo_sha"] += 1
            continue
        resultado["conferidos"] += 1
        if sha256_do_arquivo(arquivo) != esperado:
            resultado["divergentes"].append(caminho)
    return resultado


# =============================================================================
# Relatório
# =============================================================================

def tabela(titulo: str, linhas: list[tuple[str, int, int]]) -> None:
    print(f"\n{titulo}")
    print(f"  {'grupo':<12} {'arquivos':>10} {'bytes':>14}")
    print(f"  {'-' * 12} {'-' * 10} {'-' * 14}")
    tot_n = tot_b = 0
    for nome, n, b in linhas:
        print(f"  {nome:<12} {n:>10,} {humano(b):>14}".replace(",", "."))
        tot_n += n
        tot_b += b
    print(f"  {'-' * 12} {'-' * 10} {'-' * 14}")
    print(f"  {'TOTAL':<12} {tot_n:>10,} {humano(tot_b):>14}".replace(",", "."))


# =============================================================================
# main
# =============================================================================

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Traz para a n3 só o pedaço do release que a bancada precisa.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--repo", default=REPO_PADRAO, help=f"default: {REPO_PADRAO}")
    p.add_argument("--revisao", default="main")
    p.add_argument("--destino", required=True,
                   help="pasta de staging. NÃO aponte para dentro do projeto: o "
                        "rsync do código não deve carregar 20 GB de JPEG.")
    p.add_argument(
        "--cenas", type=int, required=True,
        help="quantas CENAS trazer em generated/ e mask/. CONTA CENA, NÃO "
             "ARQUIVO: cada cena tem ~41 amostras. `0` = nenhuma (traz só "
             "depth/, meta/ e o topo, que é o que a calibração precisa).",
    )
    p.add_argument("--cenas-val", type=int, default=None,
                   help="quantas das --cenas vêm do lado val do split. Default: "
                        "a proporção do próprio split, com piso de 1.")
    p.add_argument("--semente", type=int, default=SEMENTE_PADRAO)
    p.add_argument("--teto-gb", type=float, default=TETO_GB_PADRAO,
                   help=f"acima disto exige --confirmar (default {TETO_GB_PADRAO})")
    p.add_argument("--margem-gb", type=float, default=MARGEM_GB_PADRAO,
                   help="espaço que tem que SOBRAR depois do download")
    p.add_argument("--baixar", action="store_true",
                   help="OPT-IN: sem isto o script só planeja e imprime.")
    p.add_argument("--confirmar", action="store_true",
                   help="autoriza passar do --teto-gb")
    p.add_argument("--sem-depth", action="store_true", help="não traz depth/")
    p.add_argument("--sem-meta", action="store_true", help="não traz meta/")
    p.add_argument("--sem-mask", action="store_true", help="não traz mask/")
    p.add_argument("--min-amostras-por-cena", type=int, default=1,
                   help="cena com menos arquivos em generated/ que isto não é "
                        "elegível (upload parcial)")
    p.add_argument("--conferir-sha", type=int, default=0,
                   help="0 = confere TODOS os JPEGs trazidos; N>0 = amostra "
                        "determinística de N")
    p.add_argument("--sem-conferir-sha", action="store_true")
    p.add_argument("--workers", type=int, default=8)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    carregar_env_local()
    token = token_hf()
    destino = Path(os.path.expanduser(args.destino)).resolve()
    inicio = time.time()

    print("=" * 74)
    print(f"stage_release_n3 v{VERSAO}")
    print(f"  repo     : {args.repo} @ {args.revisao}")
    print(f"  destino  : {destino}")
    print(f"  cenas    : {args.cenas} (CENAS, ~41 amostras cada)")
    print(f"  semente  : {args.semente}")
    print(f"  modo     : {'BAIXAR' if args.baixar else 'SÓ PLANO (nada é escrito)'}")
    print("=" * 74)

    # -- 1. listagem: metadado, sem egress de conteúdo -------------------------
    arquivos, commit = listar_arvore(args.repo, args.revisao, token)
    if not arquivos:
        print("ERRO: listagem vazia. Repo errado, token sem acesso, ou repo vazio.")
        return 6
    tamanho_de = dict(arquivos)
    print(f"[stage] {len(arquivos):,} arquivos no repo, commit {commit[:12]}"
          .replace(",", "."))

    por_grupo: dict[str, list[str]] = {}
    for caminho, _ in arquivos:
        por_grupo.setdefault(grupo_de(caminho), []).append(caminho)
    for nome in sorted(por_grupo):
        n = len(por_grupo[nome])
        b = sum(tamanho_de[c] for c in por_grupo[nome])
        print(f"[stage]   {nome:<12} {n:>7,} arquivos  {humano(b):>12}".replace(",", "."))

    topo = sorted(por_grupo.get("(topo)", []))

    # -- 2. split: precisa vir antes da escolha das cenas ----------------------
    escolhidas = {"train": [], "val": []}
    disponiveis_por_lado: dict[str, list[str]] = {}
    fonte_do_split = "não consultado (--cenas 0)"
    if args.cenas > 0:
        if "split.json" not in [c for c in topo]:
            print("ERRO: split.json não está no repo. Sem ele não dá para "
                  "garantir que as cenas de val venham do lado val — e uma "
                  "validação em cena vista no treino mede memorização.")
            return 6
        caminho_split = destino / "split.json"
        if not caminho_split.is_file():
            if not args.baixar:
                print("[stage] baixando SÓ o split.json (dezenas de KB) para poder "
                      "planejar respeitando a fronteira de cena.")
            destino.mkdir(parents=True, exist_ok=True)
            if baixar_arquivo_unico(args.repo, args.revisao, "split.json",
                                    destino, token) is None:
                print("ERRO: não consegui obter o split.json.")
                return 6
        fonte_do_split = str(caminho_split)
        mapa_split = ler_split(caminho_split)

        # Cena elegível = está no split E tem arquivo em generated/ no repo.
        # Um upload interrompido deixa cena no split sem um único JPEG; escolher
        # uma dessas daria uma bancada vazia que só falharia lá na frente.
        contagem_generated: dict[str, int] = {}
        for caminho in por_grupo.get("generated", []):
            cena = cena_de(caminho)
            if cena:
                contagem_generated[cena] = contagem_generated.get(cena, 0) + 1
        if not contagem_generated:
            print("ERRO: `generated/` não tem arquivo nenhum no repo — não há "
                  "bokeh ALVO para trazer. Peça --cenas 0 se você só quer o "
                  "insumo geométrico (depth/ + meta/).")
            return 6
        for cena, lado in mapa_split.items():
            if contagem_generated.get(cena, 0) >= args.min_amostras_por_cena:
                disponiveis_por_lado.setdefault(lado, []).append(cena)
        print(f"[stage] split: {sum(1 for l in mapa_split.values() if l == 'train')} "
              f"train / {sum(1 for l in mapa_split.values() if l == 'val')} val; "
              f"elegíveis (com generated/ no repo): "
              f"{ {k: len(v) for k, v in disponiveis_por_lado.items()} }")

        escolhidas = escolher_cenas(
            disponiveis_por_lado, args.cenas, args.cenas_val, args.semente
        )
        pedidas, obtidas = args.cenas, len(escolhidas["train"]) + len(escolhidas["val"])
        if obtidas < pedidas:
            print(f"[stage] AVISO: pedi {pedidas} cenas e só {obtidas} são "
                  "elegíveis. O release provavelmente está incompleto.")
        if args.cenas >= 2 and not escolhidas["val"]:
            print("ERRO: nenhuma cena de VALIDAÇÃO elegível. Uma bancada sem val "
                  "não valida nada — e essa é a falha mais silenciosa possível.")
            return 6
        n_por_cena = [contagem_generated[c] for c in escolhidas["train"] + escolhidas["val"]]
        if n_por_cena:
            print(f"[stage] cenas escolhidas: {len(escolhidas['train'])} train + "
                  f"{len(escolhidas['val'])} val; amostras/cena "
                  f"min={min(n_por_cena)} max={max(n_por_cena)}")

    # -- 3. o conjunto desejado, arquivo a arquivo -----------------------------
    cenas_alvo = set(escolhidas["train"]) | set(escolhidas["val"])
    grupos_por_cena = tuple(g for g in GRUPOS_POR_CENA
                            if not (g == "mask" and args.sem_mask))
    desejados: list[str] = list(topo)
    for grupo in GRUPOS_INTEIROS:
        if grupo == "depth" and args.sem_depth:
            continue
        if grupo == "meta" and args.sem_meta:
            continue
        desejados.extend(por_grupo.get(grupo, []))
    for grupo in grupos_por_cena:
        for caminho in por_grupo.get(grupo, []):
            if cena_de(caminho) in cenas_alvo:
                desejados.append(caminho)
    desejados = sorted(set(desejados))

    # -- 4. idempotência: o que já está em disco com o tamanho certo não volta --
    faltando, ja_em_disco_bytes, ja_em_disco_n = [], 0, 0
    for caminho in desejados:
        local = destino / caminho
        if local.is_file() and local.stat().st_size == tamanho_de.get(caminho, -1):
            ja_em_disco_n += 1
            ja_em_disco_bytes += tamanho_de[caminho]
        else:
            faltando.append(caminho)
    bytes_faltando = sum(tamanho_de.get(c, 0) for c in faltando)

    resumo_grupos: dict[str, list[int]] = {}
    for caminho in desejados:
        g = grupo_de(caminho)
        slot = resumo_grupos.setdefault(g, [0, 0])
        slot[0] += 1
        slot[1] += tamanho_de.get(caminho, 0)
    tabela("O QUE A BANCADA PEDE (desejado, já em disco + a baixar):",
           [(g, v[0], v[1]) for g, v in sorted(resumo_grupos.items())])

    resumo_faltando: dict[str, list[int]] = {}
    for caminho in faltando:
        g = grupo_de(caminho)
        slot = resumo_faltando.setdefault(g, [0, 0])
        slot[0] += 1
        slot[1] += tamanho_de.get(caminho, 0)
    tabela("O QUE FALTA BAIXAR (isto é o egress desta execução):",
           [(g, v[0], v[1]) for g, v in sorted(resumo_faltando.items())] or [("(nada)", 0, 0)])
    print(f"\n[stage] já em disco: {ja_em_disco_n:,} arquivos, "
          f"{humano(ja_em_disco_bytes)} (não serão rebaixados)".replace(",", "."))
    print(f"[stage] a baixar   : {len(faltando):,} arquivos, {humano(bytes_faltando)}"
          .replace(",", "."))
    if len(faltando) > 20000:
        print(f"[stage] AVISO: {len(faltando):,} arquivos pequenos (meta/ tem 69.700). "
              "O gargalo vai ser número de requisições, não banda — conte dezenas "
              "de minutos.".replace(",", "."))

    # -- 5. o teto ------------------------------------------------------------
    acima_do_teto = bytes_faltando > args.teto_gb * GB
    if acima_do_teto:
        print(f"\n[stage] {humano(bytes_faltando)} passa do teto de "
              f"{args.teto_gb:.1f} GB.")
        if not args.confirmar:
            print("[stage] ABORTADO sem baixar nada. Se é isto mesmo, repita com "
                  "--confirmar (e confira antes se `--cenas` é o número que você "
                  "quis: ele conta CENA, e cada cena traz ~41 amostras).")
            return 5
        print("[stage] --confirmar presente: seguindo.")

    # -- 6. espaço e cota -----------------------------------------------------
    motivos = checar_espaco(destino, bytes_faltando, args.margem_gb)
    if motivos:
        print("\n[stage] ABORTADO — não há onde escrever isto:")
        for m in motivos:
            print(f"  - {m}")
        print("  Nada foi baixado e nada foi apagado. Libere espaço (com quem "
              "cuida da pasta) ou reduza --cenas.")
        return 3

    if not args.baixar:
        print("\n[stage] MODO PLANO: nada foi baixado além do split.json.")
        print("[stage] Para baixar de verdade, repita o comando com --baixar"
              + (" --confirmar" if acima_do_teto else "") + ".")
        return 0

    # -- 7. download ----------------------------------------------------------
    padroes: list[str] = list(topo)
    for grupo in GRUPOS_INTEIROS:
        if grupo == "depth" and args.sem_depth:
            continue
        if grupo == "meta" and args.sem_meta:
            continue
        if por_grupo.get(grupo):
            padroes.append(f"{grupo}/*")   # fnmatch: `*` atravessa `/`
    for grupo in grupos_por_cena:
        for cena in sorted(cenas_alvo):
            padroes.append(f"{grupo}/{cena}/*")
            padroes.append(f"{grupo}/{cena}.*")  # layout por cena (depth-like)

    print(f"\n[stage] baixando com {len(padroes)} allow_patterns "
          f"({args.workers} workers)...")
    t0 = time.time()
    baixar(args.repo, args.revisao, padroes, destino, token, args.workers, faltando)
    segundos = time.time() - t0

    # -- 8. o que de fato chegou ---------------------------------------------
    presentes, ausentes, bytes_em_disco = [], [], 0
    for caminho in desejados:
        local = destino / caminho
        if local.is_file():
            presentes.append(caminho)
            bytes_em_disco += local.stat().st_size
        else:
            ausentes.append(caminho)
    baixados_agora = [c for c in faltando if (destino / c).is_file()]
    bytes_baixados = sum((destino / c).stat().st_size for c in baixados_agora)
    print(f"[stage] {len(presentes):,} de {len(desejados):,} arquivos em disco "
          f"({humano(bytes_em_disco)}); {len(baixados_agora):,} novos "
          f"({humano(bytes_baixados)}) em {segundos/60:.1f} min".replace(",", "."))
    if ausentes:
        print(f"[stage] AVISO: {len(ausentes)} arquivos pedidos não chegaram "
              f"(ex.: {ausentes[:3]}).")

    # -- 9. sha256 ------------------------------------------------------------
    sha = {"executado": False}
    if not args.sem_conferir_sha:
        print("\n[stage] conferindo sha256 contra generated_images.jsonl...")
        sha = conferir_sha256(destino, presentes, args.conferir_sha)
        sha["executado"] = True
        print(f"[stage]   conferidos={sha['conferidos']} "
              f"divergentes={len(sha['divergentes'])} "
              f"sem_linha_no_ledger={sha['sem_linha']} "
              f"sem_campo_sha={sha['sem_campo_sha']}")
        if sha.get("observacao"):
            print(f"[stage]   {sha['observacao']}")

    # -- 10. o registro do que foi feito --------------------------------------
    execucao = {
        "versao_do_script": VERSAO,
        "quando": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "host": os.uname().nodename,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "repo": args.repo,
        "revisao_pedida": args.revisao,
        "commit": commit,
        "semente": args.semente,
        "cenas_pedidas": args.cenas,
        "cenas_val_pedidas": args.cenas_val,
        "fonte_do_split": fonte_do_split,
        "cenas": {"train": escolhidas["train"], "val": escolhidas["val"]},
        "grupos_inteiros": [
            g for g in GRUPOS_INTEIROS
            if not (g == "depth" and args.sem_depth) and not (g == "meta" and args.sem_meta)
        ],
        "grupos_por_cena": list(grupos_por_cena),
        "arquivos_de_topo": topo,
        "contagens": {
            g: {"arquivos": v[0], "bytes": v[1]} for g, v in sorted(resumo_grupos.items())
        },
        "bytes_desejados": sum(tamanho_de.get(c, 0) for c in desejados),
        "bytes_baixados_nesta_execucao": bytes_baixados,
        "arquivos_baixados_nesta_execucao": len(baixados_agora),
        "arquivos_ausentes": len(ausentes),
        "sha256": {k: v for k, v in sha.items() if k != "divergentes"}
                  | {"divergentes": sha.get("divergentes", [])[:20]},
        "teto_gb": args.teto_gb,
        "confirmar": args.confirmar,
        "duracao_s": round(time.time() - inicio, 1),
    }
    caminho_manifesto = destino / NOME_MANIFESTO_STAGING
    anterior = {}
    if caminho_manifesto.is_file():
        try:
            anterior = json.loads(caminho_manifesto.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            anterior = {}
    historico = list(anterior.get("execucoes", []))
    historico.append(execucao)
    caminho_manifesto.write_text(
        json.dumps(
            {
                "o_que_e": "staging PARCIAL do release para a bancada de "
                           "condicionamento geométrico. NÃO é o release completo: "
                           "generated/ e mask/ só têm as cenas listadas em "
                           "execucoes[].cenas.",
                "destino": str(destino),
                "atual": execucao,
                "execucoes": historico,
            },
            ensure_ascii=False, indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[stage] registro em {caminho_manifesto} "
          f"({len(historico)} execução(ões) acumuladas)")

    if sha.get("divergentes"):
        print(f"\n[stage] FALHOU: {len(sha['divergentes'])} arquivos com sha256 "
              f"DIFERENTE do ledger (ex.: {sha['divergentes'][:3]}). O que está em "
              "disco não é o que o release diz que é. NÃO treine com isto; "
              "rebaixe os arquivos em questão.")
        return 4

    print("\n[stage] OK. Aponte o YAML/`--release` para:")
    print(f"        {destino}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
