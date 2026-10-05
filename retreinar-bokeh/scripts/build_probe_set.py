#!/usr/bin/env python3
"""Monta o conjunto FIXO do probe de controlabilidade (T11).

O probe (`genfocus_train/probe.py`) gera a MESMA cena com K crescente e confere
que a nitidez cai monotonicamente. Para isso ele precisa, por imagem:

    aif             uint8   (H, W, 3)   — a entrada all-in-focus
    disparity       float32 (H, W)      — 1/z em 1/m, do Depth Pro
    focus_disparity float32 escalar     — mediana da disparidade na máscara

Três exigências que fazem o probe significar alguma coisa, e sem as quais ele
mede a diferença entre conjuntos em vez de entre modelos:

  1. **É FIXO.** O mesmo conjunto em todos os runs e em todos os braços. Por isso
     ele mora em disco como `.npz`, e não é um split sorteado do treino.
  2. **`scene_id` DISJUNTO do treino.** Um probe que aparece no treino mede
     memorização. O script recusa `sample_id` que já esteja no manifesto de
     split de treino, quando esse manifesto é passado.
  3. **É pequeno.** 8 imagens a 512² custam ~5 min de probe. Aumentar o conjunto
     não melhora o sinal proporcionalmente — a variância do LVCorr entre imagens
     é menor que a variância entre steps — e come orçamento de treino.

Este script NÃO roda GPU por padrão: ele consome um release já gerado pelo
`bokehnet-regen`, que já tem a profundidade e a região de foco medidas. Gerar
profundidade aqui significaria uma SEGUNDA passagem de Depth Pro, com outra
versão de pesos, e o número do probe deixaria de ser comparável com o rótulo do
treino.

As DUAS formas de release são aceitas (ver `genfocus_train/release.py`): a
ÁRVORE DE ARQUIVOS que o `bokehnet-regen` publica (`manifest.jsonl`, `depth/`,
`meta/`) e a tabela achatada com os pixels nas colunas. Quando é árvore, o
`split.json` do próprio release é usado como manifesto de exclusão por default —
que é justamente para isso que ele é materializado.

USO
    # release em árvore, autocontido (ou com o espelho apontado)
    python3 scripts/build_probe_set.py \\
        --release /dados/releases/rota_c \\
        --out probe_set/ \\
        --n 8 \\
        --particao-excluida train \\
        --mirror-root akcit-pixel/RealBokeh=/dados/espelho_realbokeh

    # release em tabela achatada
    python3 scripts/build_probe_set.py \\
        --release PREENCHER/release-rota-c \\
        --split train \\
        --out probe_set/ --n 8 \\
        --excluir-manifesto manifests/split_<release>.jsonl \\
        --particao-excluida train
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np


def _carregar_cenas(manifesto: str | None, particao: str) -> set[str]:
    if not manifesto:
        return set()
    caminho = Path(os.path.expanduser(manifesto))
    if not caminho.exists():
        raise FileNotFoundError(f"manifesto não encontrado: {caminho}")
    cenas: set[str] = set()
    texto = caminho.read_text(encoding="utf-8").strip()
    if caminho.suffix == ".json" and '"assignment"' in texto:
        # O `split.json` MATERIALIZADO do release. Delegado a `release.py` para
        # que exista uma leitura só desse formato.
        from genfocus_train import release as _release

        return set(_release.carregar_assignment_de_split(caminho, particao))
    if caminho.suffix == ".jsonl":
        for linha in texto.splitlines():
            if not linha.strip():
                continue
            reg = json.loads(linha)
            if str(reg.get("partition", reg.get("split"))) == particao:
                cenas.add(str(reg["scene_id"]))
    else:
        payload = json.loads(texto)
        if isinstance(payload, dict):
            cenas = {str(c) for c in payload.get(particao, [])}
        else:
            cenas = {
                str(r["scene_id"]) for r in payload
                if str(r.get("partition", r.get("split"))) == particao
            }
    return cenas


class _FonteTabela:
    """Release na forma de TABELA achatada, com os pixels nas colunas."""

    def __init__(self, repo: str, split: str) -> None:
        from datasets import load_dataset

        from genfocus_train.env import get_required_env

        self.ds = load_dataset(repo, split=split, token=get_required_env("HF_TOKEN"))

    def __len__(self) -> int:
        return len(self.ds)

    def registro(self, i: int) -> dict:
        return self.ds[int(i)]

    def aif(self, i: int):
        return np.asarray(self.ds[int(i)]["aif"], dtype=np.uint8)

    def depth(self, i: int):
        return self.ds[int(i)]["depth"]


class _FonteArvore:
    """Release na forma de ÁRVORE DE ARQUIVOS — a que o `bokehnet-regen` publica.

    Os pixels da AIF podem estar em `generated/` (rota B), em `source/` (release
    autocontido) ou só no espelho de origem; quem resolve é o
    `release.ResolvedorDePixels`, com o sha256 conferido contra o
    `source_images.jsonl`.
    """

    def __init__(self, origem: str, *, espelhos: dict, permitir_download: bool) -> None:
        from PIL import Image  # noqa: F401  (usado via o resolvedor)

        from genfocus_train import release as _release

        self.arvore = _release.ArvoreDeRelease.abrir(
            origem, permitir_download=permitir_download
        )
        self.arvore.verificar_split()
        self.resolvedor = _release.ResolvedorDePixels(
            self.arvore, espelhos=espelhos, permitir_download=permitir_download
        )
        self.ids = [str(l["sample_id"]) for l in self.arvore.linhas]

    def __len__(self) -> int:
        return len(self.ids)

    def registro(self, i: int) -> dict:
        return self.arvore.registro(self.ids[int(i)])

    def aif(self, i: int):
        return np.asarray(
            self.resolvedor.resolver(self.ids[int(i)], "aif").imagem, dtype=np.uint8
        )

    def depth(self, i: int):
        from PIL import Image

        caminho = self.arvore.caminho_da_profundidade(self.ids[int(i)])
        with Image.open(caminho) as im:
            # `np.array`, não `asarray`: o array tem que sobreviver ao `with`.
            return np.array(im)


def _pares_nomeados(valores) -> dict:
    """`NOME=CAMINHO` repetido -> dict. Forma sem `=` é erro, não palpite."""
    saida: dict[str, str] = {}
    for item in valores or []:
        if "=" not in item:
            raise SystemExit(
                f"--mirror-root {item!r} não está no formato NOME=CAMINHO. O NOME é "
                "o `source_dataset` exatamente como o release o grava."
            )
        nome, caminho = item.split("=", 1)
        saida[nome.strip()] = caminho.strip()
    return saida


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", required=True,
                        help="repo HF do release, ou o diretório da árvore em disco")
    parser.add_argument("--split", default="train",
                        help="só para release em TABELA achatada")
    parser.add_argument(
        "--release-format", default="auto", choices=("auto", "arvore", "tabela"),
        help="'auto' decide pela evidência local (pasta com manifest.jsonl)",
    )
    parser.add_argument(
        "--mirror-root", action="append", default=[], metavar="NOME=CAMINHO",
        help="espelho de origem para os pixels que o release referencia",
    )
    parser.add_argument(
        "--permitir-download", action="store_true",
        help="baixar release/espelho do Hub. Default: ler só do que já está em disco",
    )
    parser.add_argument("--out", required=True, help="diretório de saída (.npz por imagem)")
    parser.add_argument(
        "--image-size", type=int, default=512,
        help="lado do recorte gravado. 512 = a grade do treino e o TILE_SIZE da "
             "inferência. Mudar aqui descasa o probe do treino (A3).",
    )
    parser.add_argument("--n", type=int, default=8, help="quantas imagens (default 8)")
    parser.add_argument(
        "--excluir-manifesto", default=None,
        help="manifesto de split; cenas desta partição são EXCLUÍDAS do probe",
    )
    parser.add_argument("--particao-excluida", default="train")
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    from genfocus_train import control
    from genfocus_train import release as _release

    formato = args.release_format
    if formato == "auto":
        formato = ("arvore" if _release.ArvoreDeRelease.parece_arvore(args.release)
                   else "tabela")
    if formato == "arvore":
        fonte = _FonteArvore(
            args.release, espelhos=_pares_nomeados(args.mirror_root),
            permitir_download=bool(args.permitir_download),
        )
    else:
        fonte = _FonteTabela(args.release, args.split)
    print(f"[probe-set] release lido como {formato!r}: {len(fonte)} amostras.")

    manifesto_de_exclusao = args.excluir_manifesto
    if manifesto_de_exclusao is None and formato == "arvore" and fonte.arvore.tem_split:
        # O split é MATERIALIZADO no release; usar outra fonte de verdade aqui
        # seria a mesma família de defeito que o split sorteado no dataloader.
        manifesto_de_exclusao = str(fonte.arvore.caminho_do_split)
        print(f"[probe-set] exclusão vinda do próprio release: {manifesto_de_exclusao}")
    excluidas = _carregar_cenas(manifesto_de_exclusao, args.particao_excluida)
    if excluidas:
        print(f"[probe-set] excluindo {len(excluidas)} cenas da partição "
              f"{args.particao_excluida!r}.")

    saida = Path(os.path.expanduser(args.out))
    saida.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    ordem = rng.permutation(len(fonte))
    escolhidas, cenas_usadas = 0, set()

    for idx in ordem:
        if escolhidas >= args.n:
            break
        registro = fonte.registro(int(idx))
        cena = str(registro.get("scene_id", registro.get("sample_id", idx)))
        if cena in excluidas or cena in cenas_usadas:
            continue
        # Recusa amostra que a própria geração marcou como rótulo não confiável:
        # um probe montado sobre K censurado mediria a censura.
        if registro.get("is_valid_for_control") is False:
            continue
        if registro.get("is_k_censored") is True:
            continue
        try:
            control.validar_registro_controle(registro)
        except control.ControlContractError as exc:
            print(f"[probe-set] pulando {cena}: {exc}")
            continue

        disparidade = control.decode_disparity_u16(
            fonte.depth(int(idx)),
            float(registro["disparity_min"]),
            float(registro["disparity_max"]),
        )
        aif = fonte.aif(int(idx))
        if aif.ndim != 3 or aif.shape[2] != 3:
            print(f"[probe-set] pulando {cena}: AIF com shape {aif.shape}")
            continue

        # ── A3 — o conjunto é gravado na MESMA grade em que o treino vê ──────
        # Antes isto gravava a resolução NATIVA do release, enquanto o docstring
        # do `probe.py` afirmava 512² e justificava `NO_TILED_DENOISE=True` com
        # "o conjunto é 512², um tile só". Três problemas de uma vez:
        #
        #  1. ALINHAMENTO. Com H,W não múltiplos de 16, `generate` ajusta o
        #     latente (`height = 2*(h//16)`) e a AIF passa por
        #     `image_processor.preprocess`, mas o mapa de defocus NÃO
        #     (`No_preprocess=True`). Latente, AIF e mapa acabam em três grades
        #     diferentes e o mapa deixa de corresponder pixel a pixel. A
        #     inferência oficial evita por construção: `resize_and_pad_image`
        #     força múltiplo de 16 e `cv2.resize` põe o mapa na mesma grade.
        #  2. CUSTO. Sem tiling numa foto 3MP o próprio paper (§4.1, Runtime) dá
        #     ~40 GB de VRAM e ~50 s por imagem — ao lado do estado do treino,
        #     é OOM provável, e o custo do probe deixaria de ser ~1%.
        #  3. DISTRIBUIÇÃO. O modelo é treinado em crop 512²; probar noutra
        #     grade mede o modelo fora da distribuição em que ele aprendeu.
        #
        # `_plano_geometrico` é a MESMA função do dataloader, então a grade do
        # probe e a do treino não podem divergir por construção.
        from genfocus_train.data import _plano_geometrico, resize_nearest

        h_orig, w_orig = int(aif.shape[0]), int(aif.shape[1])
        if disparidade.shape != (h_orig, w_orig):
            # A profundidade é gravada com o lado longo limitado; primeiro ela
            # sobe para a grade da imagem, depois as duas descem juntas.
            disparidade = resize_nearest(disparidade, w_orig, h_orig)

        novo_w, novo_h, box, _flip, _seq = _plano_geometrico(
            w_orig, h_orig, int(args.image_size), "short_side",
            train=False, rng=None,      # crop CENTRAL: o conjunto é fixo
        )
        x0, y0, x1, y1 = box
        if (novo_w, novo_h) != (w_orig, h_orig):
            from PIL import Image as _Image

            aif = np.asarray(
                _Image.fromarray(aif).resize((novo_w, novo_h), _Image.BICUBIC),
                dtype=np.uint8,
            )
            disparidade = resize_nearest(disparidade, novo_w, novo_h)
        aif = np.ascontiguousarray(aif[y0:y1, x0:x1])
        disparidade = np.ascontiguousarray(disparidade[y0:y1, x0:x1])

        # `focus_disparity` NÃO é reescalado: disparidade é 1/m e não depende da
        # resolução. Quem depende é o K, e ele é do probe, não do conjunto.
        destino = saida / f"{cena}.npz"
        np.savez_compressed(
            destino,
            aif=aif,
            disparity=disparidade.astype(np.float32),
            focus_disparity=np.float32(registro["focus_disparity"]),
        )
        cenas_usadas.add(cena)
        escolhidas += 1
        print(f"[probe-set] {destino.name}  {aif.shape[1]}x{aif.shape[0]} "
              f"(de {w_orig}x{h_orig})  "
              f"focus_disp={float(registro['focus_disparity']):.4f}")

    if escolhidas < args.n:
        raise SystemExit(
            f"só consegui {escolhidas} de {args.n} imagens. Não completo o conjunto "
            "com amostras filtradas — um probe com amostra ruim mede a amostra ruim."
        )
    print(f"[probe-set] pronto: {escolhidas} imagens em {saida}")
    print("[probe-set] aponte `runtime.probe_set_dir` do YAML para este diretório.")


if __name__ == "__main__":
    main()
