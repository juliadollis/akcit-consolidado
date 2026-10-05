#!/usr/bin/env python3
"""Entrypoint da rota A — §3.2(a): AIF real, (D_focus, K) sorteados, alvo RENDERIZADO.

Piloto primeiro, sempre:

    python3 scripts/run_route_a.py \\
        --output-dir output/a_pilot --pilot 20 --samples-per-image 41 \\
        --k-distribution manifests/k_distribution_bc.json \\
        --source GenerativePhotography=/raid/.../data/GenerativePhotography \\
        --source 'EBB!=/raid/.../data/EBB_sharp' \\
        --top-n-per-source GenerativePhotography=850 --top-n-per-source 'EBB!=850' \\
        --models-dir /raid/.../models \\
        --bokehme-dir third_party/BokehMe \\
        --renderer-report output/renderer_verification.json

A ordem de verificação é a mesma da rota C, com um passo a mais no começo:

    vocabulário -> laudo do renderer -> distribuição de K -> modelos -> fonte ->
    split -> orçamento de disco -> run

O **vocabulário** vem primeiro porque é o único que falha sem custo: a rota A precisa de
`FocusSource.SAMPLED_PLANE` e `MaskSource.SAMPLED_PLANE`, e sem eles nenhuma amostra pode
ser gravada sem mentir na proveniência. Descobrir isso depois de carregar Depth Pro e
BokehMe seria desperdício; descobrir na amostra 40.000 seria pior.

O **BiRefNet não é carregado**, e isso é o paper: a rota A não tem máscara e o plano de
foco é sorteado (`paper.txt:329-330`). Se alguém acrescentar um segmentador aqui, é
invenção nossa e tem que ser declarada.

## O piloto não congela limiar nenhum

Os sete gates saem em modo medir. O que o piloto produz é o histograma de rejeição e a
distribuição de `coc_p99_px`, `defocus_saturation_ratio` e área da banda — que
`scripts/calibrate_thresholds.py` transforma em número para o run final. E produz a
medida que falta no projeto inteiro: **o custo de GPU por render do BokehMe**. O job 32212
mediu o *contrato* do renderer (`ACHADOS.md:245-290`), não o throughput, e 70 mil renders
é a única etapa sem estimativa. Cronometre o piloto antes de dimensionar `--time`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dataio import build_scene_split, estimate_disk_budget                 # noqa: E402
from qc.rejection import RejectionLog                                      # noqa: E402
from routes.route_a import (                                              # noqa: E402
    DEFAULT_SAMPLES_PER_IMAGE, RouteAConfig, VocabularyExtensionRequired,
    resolve_sampled_vocabulary, run_route_a,
)
from sources.genphoto_ebb import (                                        # noqa: E402
    PAPER_SOURCES, AifImageLoader, enumerate_candidates, enumeration_summary,
    rank_and_cut_per_source, validate_source_name,
)
from sources.k_distribution import DEFAULT_WIDEN_FRACTION, KDistribution   # noqa: E402


# --------------------------------------------------------------------------------
# Proveniência — idêntica à da rota C, de propósito
# --------------------------------------------------------------------------------

def _source_sha256(repo: Path) -> str:
    """Hash do CÓDIGO que produziu o rótulo: todo `.py` de `src/` e `scripts/`.

    Proveniência primária, e não o commit: um commit identifica o que estava versionado,
    este hash identifica o que **rodou**. A cópia que roda no cluster chega por rsync e
    não tem `.git`.
    """
    digest = hashlib.sha256()
    arquivos = sorted(
        p for base in ("src", "scripts") for p in (repo / base).rglob("*.py")
        if "__pycache__" not in p.parts)
    if not arquivos:
        raise RuntimeError(f"nenhum .py em {repo}/src ou {repo}/scripts")
    for arquivo in arquivos:
        digest.update(str(arquivo.relative_to(repo)).encode("utf-8"))
        digest.update(arquivo.read_bytes())
    return digest.hexdigest()


def _git_commit(repo: Path) -> Optional[str]:
    try:
        head = subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            text=True, stderr=subprocess.DEVNULL).strip()
        sujo = subprocess.check_output(
            ["git", "-C", str(repo), "status", "--porcelain"],
            text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None
    return f"{head}-dirty" if sujo else head


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for bloco in iter(lambda: fh.read(1 << 20), b""):
            h.update(bloco)
    return h.hexdigest()


# --------------------------------------------------------------------------------
# `NOME=valor` na linha de comando
# --------------------------------------------------------------------------------

def _pares_nomeados(valores, *, converte, rotulo: str) -> dict:
    """Parseia `NOME=valor`, recusando fonte que não é do paper.

    Fonte fora de `PAPER_SOURCES` é **erro de configuração** — `SystemExit` — e não
    rejeição de amostra: um dataset errado não polui o histograma, ele cancela o run. O
    pipeline antigo aceitava `DiffCamera` aqui (divergência nº 1 da auditoria) e as
    docstrings afirmavam fidelidade ao paper.
    """
    saida: dict = {}
    for item in valores or ():
        if "=" not in item:
            raise SystemExit(f"--{rotulo} espera NOME=valor; veio {item!r}")
        nome, bruto = item.split("=", 1)
        nome = nome.strip()
        try:
            validate_source_name(nome)
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
        if nome in saida:
            raise SystemExit(f"--{rotulo} repetido para {nome!r}")
        saida[nome] = converte(bruto.strip())
    return saida


# --------------------------------------------------------------------------------
# Fonte
# --------------------------------------------------------------------------------

def _carrega_fonte(args):
    """Enumera as duas fontes, ranqueia e corta **dentro de cada uma**.

    Ranking global entre fontes é o defeito A8/D13: a variância do Laplaciano escala com
    resolução e compressão, então um ranking único seleciona pela fonte de maior
    resolução, não pela mais nítida (`qc/gates.py:279-284`).
    """
    raizes = _pares_nomeados(args.source, converte=str, rotulo="source")
    if not raizes:
        raise SystemExit(
            "--source é obrigatório, no formato NOME=diretório. As fontes do §3.2(a) são "
            f"{sorted(PAPER_SOURCES)} (paper.txt:996, 527-528).")
    quotas = _pares_nomeados(args.top_n_per_source, converte=int,
                             rotulo="top-n-per-source")
    pisos = _pares_nomeados(args.min_laplacian_per_source, converte=float,
                            rotulo="min-laplacian-per-source")
    revisoes = _pares_nomeados(args.source_revision, converte=str,
                               rotulo="source-revision")
    notas = _pares_nomeados(args.source_note, converte=str, rotulo="source-note")

    log = RejectionLog(Path(args.output_dir) / "source_rejections.jsonl")
    candidatas = []
    for nome in sorted(raizes):
        print(f"[fonte] {nome}: medindo variância do Laplaciano em {raizes[nome]} …")
        candidatas.extend(enumerate_candidates(
            raizes[nome], nome, log=log, revision=revisoes.get(nome),
            note=notas.get(nome), sharpness_long_side=args.sharpness_long_side))

    mantidas, cortes = rank_and_cut_per_source(
        candidatas, quota=quotas or None, min_variance=pisos or None)
    print(enumeration_summary(mantidas, log, cortes))

    if args.num_shards > 1:
        # Fatia por IMAGEM, e por hash do `scene_id` — não por posição na lista. As 41
        # variantes de uma imagem compartilham a profundidade, então separá-las entre
        # processos recalcularia o Depth Pro 41 vezes em vez de uma; e o split tem a
        # imagem como unidade, então rachá-la seria vazamento.
        import hashlib

        if not 0 <= args.shard < args.num_shards:
            raise SystemExit(f"--shard {args.shard} fora de [0, {args.num_shards})")
        antes = len(mantidas)
        mantidas = [im for im in mantidas
                    if int.from_bytes(
                        hashlib.sha256(str(im.scene_id).encode()).digest()[:8],
                        "big") % args.num_shards == args.shard]
        print(f"[fatia] {args.shard}/{args.num_shards}: {len(mantidas)} de {antes} "
              f"imagens -> {len(mantidas) * args.samples_per_image} amostras")

    if args.pilot:
        # Sorteia IMAGENS, não amostras: 41 variantes de uma imagem são UMA cena, e é a
        # mesma unidade do split. Amostrar variantes espalharia meia cena no piloto.
        escolhidas = sorted(mantidas, key=lambda im: im.scene_id)
        random.Random(args.seed).shuffle(escolhidas)
        mantidas = escolhidas[:int(args.pilot)]
        print(f"[fonte] piloto: {len(mantidas)} imagens (seed {args.seed}) -> "
              f"{len(mantidas) * args.samples_per_image} amostras previstas.")

    extra = {
        "source_cuts": [c.to_dict() for c in cortes],
        "source_roots": raizes,
        "source_quota_evidence":
            "[A] a divisão do pool de ~1,7K entre [80] e EBB! não é publicada "
            "(paper.txt:998 dá só o total). A quota é declarada por fonte.",
        "sharpness_long_side": args.sharpness_long_side,
    }
    return mantidas, AifImageLoader(
        ledger_path=Path(args.output_dir) / "source_images.jsonl"), extra


# --------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Rota A — pré-treino sintético: (D_focus, K) sorteados, alvo "
                    "renderizado pelo BokehMe [43].",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--models-dir", required=True,
                   help="contém checkpoints/depth_pro.pt. O BiRefNet NÃO é usado: a "
                        "rota A não tem máscara (paper.txt:329-330).")
    p.add_argument("--bokehme-dir", required=True)
    p.add_argument("--renderer-report", required=True,
                   help="JSON de scripts/verify_renderer.py. Nesta rota o renderer "
                        "PRODUZ o rótulo, então o laudo não é formalidade: sem ele, "
                        "`is_final_label_renderer` é declaração e não medição.")
    p.add_argument("--k-distribution", required=True,
                   help="JSON de scripts/build_k_distribution.py. A rota A não tem "
                        "equação para K (paper.txt:329-330); ela sorteia da distribuição "
                        "medida em B e C.")
    p.add_argument("--source", action="append", default=None, metavar="NOME=DIR",
                   help=f"raiz de cada fonte. Só {sorted(PAPER_SOURCES)}. "
                        "DiffCamera é [69], baseline, e NÃO é fonte de dado.")
    p.add_argument("--source-revision", action="append", default=None,
                   metavar="NOME=REV", help="revisão/commit/tag da fonte, gravada por "
                                            "amostra. `[A]` quando ausente.")
    p.add_argument("--source-note", action="append", default=None, metavar="NOME=TEXTO",
                   help="nota livre por fonte, gravada por amostra. É onde declarar "
                        "QUAL LADO da EBB! entrou no pool — o paper não diz ([A] A2).")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--val-fraction", type=float, default=0.05)
    p.add_argument("--allow-degenerate-k-distribution", action="store_true",
                   help="roda mesmo com a distribuição reprovada pelos guardas. Medido: "
                        "a rota B publicada tem k=50,0 em 11.635/11.635 (ACHADOS.md:14) "
                        "e a rota C 47,0%% no teto (ACHADOS.md:19) — sortear daí é "
                        "sortear de constantes.")

    escopo = p.add_argument_group("escopo")
    escopo.add_argument("--samples-per-image", type=int,
                        default=DEFAULT_SAMPLES_PER_IMAGE,
                        help=f"variantes por imagem. Default {DEFAULT_SAMPLES_PER_IMAGE} "
                             "= [A]: é [I] de 70K/1,7K (paper.txt:527-528, 998), o paper "
                             "NÃO publica o número. Com 1, K fica confundido com "
                             "conteúdo — o defeito A2.")
    escopo.add_argument("--top-n-per-source", action="append", default=None,
                        metavar="NOME=N",
                        help="quota de imagens POR FONTE depois do ranking de nitidez. "
                             "Ranking e corte são por fonte (A8). A soma deve ficar perto "
                             "de 1.700 (paper.txt:998).")
    escopo.add_argument("--min-laplacian-per-source", action="append", default=None,
                        metavar="NOME=V",
                        help="piso de variância do Laplaciano POR FONTE. Default: "
                             "nenhum — limiar não medido não bloqueia.")
    escopo.add_argument("--sharpness-long-side", type=int, default=None,
                        help="mede a nitidez numa grade reduzida. Default: resolução "
                             "cheia. A grade vai gravada em sharpness_hw de qualquer "
                             "forma — variância de Laplaciano depende da grade.")
    escopo.add_argument("--shard", type=int, default=0,
                        help="índice desta fatia. Fatia por IMAGEM: as 41 variantes de "
                             "uma imagem ficam juntas, senão a profundidade seria "
                             "recalculada em cada processo e o split rachado.")
    escopo.add_argument("--num-shards", type=int, default=1,
                        help="quantas fatias. Cada uma escreve no próprio --output-dir.")
    escopo.add_argument("--pilot", type=int, default=None,
                        help="sorteia N IMAGENS inteiras (não variantes) antes de "
                             "processar. 41 variantes de uma imagem são UMA cena.")
    escopo.add_argument("--limit", type=int, default=None,
                        help="teto de amostras ACEITAS no laço. Cru: corta no meio de "
                             "uma imagem. Para piloto use --pilot.")

    sorteio = p.add_argument_group(
        "sorteio (§3.2(a))",
        "O paper diz só 'randomly sample' (paper.txt:329-330) e cala sobre a "
        "distribuição. Tudo aqui é [A] declarado.")
    sorteio.add_argument("--k-widen-fraction", type=float, default=None,
                         help=f"alargamento da faixa de K para cada lado (default "
                              f"{DEFAULT_WIDEN_FRACTION}). Existe porque a rota B vai ser "
                              "REGERADA com outra DeblurNet (PLANO_EXECUCAO.md): se o K "
                              "se mover, a cobertura precisa já existir.")
    sorteio.add_argument("--focus-quantile-low", type=float, default=None,
                         help="quantil inferior da disparidade em que o plano de foco é "
                              "sorteado. Sortear em disparidade, e não em z, é o "
                              "conserto do defeito A6.")
    sorteio.add_argument("--focus-quantile-high", type=float, default=None,
                         help="quantil superior. Perto de 1 o plano cai na sentinela de "
                              "10.000 m do Depth Pro (25,7%% das amostras medidas, "
                              "ACHADOS.md:54) e a amostra é rejeitada.")
    sorteio.add_argument("--focus-band-coc-px", type=float, default=None,
                         help="tolerância da banda gravada em mask/<id>.png, em px de "
                              "CoC. NÃO é limiar de rejeição: é a definição da banda.")

    lim = p.add_argument_group(
        "limiares",
        "TODOS default None = mede e não bloqueia. Congele a partir do piloto com "
        "scripts/calibrate_thresholds.py.")
    for flag in ("min-aif-laplacian-variance", "max-bokeh-over-aif-sharpness",
                 "min-rendered-coc-p99-px", "max-defocus-saturation-ratio",
                 "min-mask-area-ratio", "max-mask-area-ratio"):
        lim.add_argument(f"--{flag}", type=float, default=None)
    lim.add_argument("--min-depth-useful-levels", type=int, default=None)
    return p


def main() -> int:
    args = build_parser().parse_args()
    raiz = Path(__file__).resolve().parents[1]
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # -- 1. o vocabulário, antes de tudo: é o que falha sem custo ---------------
    try:
        resolve_sampled_vocabulary()
    except VocabularyExtensionRequired as exc:
        raise SystemExit(f"vocabulário incompleto para a rota A.\n\n{exc}")

    # -- 2. o laudo do renderer — aqui ele PRODUZ o rótulo ---------------------
    laudo_path = Path(args.renderer_report)
    if not laudo_path.is_file():
        raise SystemExit(
            f"laudo do renderer não encontrado: {laudo_path}\nRode antes:\n"
            "  python3 scripts/verify_renderer.py --bokehme-dir ... --output-json ...")
    laudo = json.loads(laudo_path.read_text(encoding="utf-8"))
    if not laudo.get("is_final_label_renderer"):
        raise SystemExit(
            "o renderer NÃO passou nos três testes de verificação. Na rota A o alvo É a "
            "saída dele (paper.txt:331), então gerar assim reproduziria o defeito A3: os "
            "~70K alvos publicados são gaussiana de 16 camadas, não bokeh.")

    # -- 3. a distribuição de K ------------------------------------------------
    distribuicao = KDistribution.load(args.k_distribution)
    print(distribuicao.summary(
        args.k_widen_fraction if args.k_widen_fraction is not None
        else DEFAULT_WIDEN_FRACTION))
    motivo = distribuicao.degenerate_reason()
    if motivo and not args.allow_degenerate_k_distribution:
        raise SystemExit(
            f"distribuição de K degenerada: {motivo}\n"
            "A rota A depende de B e C (PLANO_EXECUCAO.md). Regere as duas antes, ou "
            "rode com --allow-degenerate-k-distribution se souber o que está fazendo.")
    if motivo:
        print(f"[rota-a] AVISO: rodando com distribuição degenerada — {motivo}")

    # -- 4. modelos. Depth Pro e o BokehMe. **Sem BiRefNet.** ------------------
    from model_runtime import DepthProRuntime
    from renderer.bokehme import BokehMeConfig, BokehMeRenderer

    modelos = Path(args.models_dir)
    depth = DepthProRuntime(modelos / "checkpoints" / "depth_pro.pt", device=args.device)
    renderer = BokehMeRenderer(args.bokehme_dir, config=BokehMeConfig(),
                               device=args.device)

    proc_renderer = renderer.provenance()
    for campo in ("arnet_sha256", "iunet_sha256", "demo_pipeline_sha256"):
        if laudo["provenance"].get(campo) != proc_renderer.get(campo):
            raise SystemExit(
                f"o laudo é de OUTRO renderer: {campo} diverge.\n"
                f"  laudo   : {laudo['provenance'].get(campo)}\n"
                f"  checkout: {proc_renderer.get(campo)}\n"
                "Rode scripts/verify_renderer.py contra ESTE checkout antes de gerar "
                "dado — nesta rota o renderer é o produtor do rótulo.")

    # -- 5. fonte ---------------------------------------------------------------
    imagens, load_aif, extra_fonte = _carrega_fonte(args)
    if not imagens:
        raise SystemExit("nenhuma imagem sobreviveu à enumeração — veja o histograma.")

    # -- 6. split POR CENA, e a cena é a IMAGEM --------------------------------
    # As N variantes de uma imagem compartilham `scene_id`, então elas caem sempre do
    # mesmo lado. Sem isso, 41 variantes da mesma imagem ficariam dos dois lados e a
    # validação mediria memorização (defeito A9).
    split = build_scene_split({im.scene_id for im in imagens},
                              val_fraction=args.val_fraction)
    print(f"[split] por cena (= imagem): {split.counts()}")

    # -- 7. orçamento de disco, ANTES de gerar ---------------------------------
    n_previsto = len(imagens) * int(args.samples_per_image)
    megapixels = sum(im.image_hw[0] * im.image_hw[1] for im in imagens) / (
        1e6 * max(len(imagens), 1))
    # `n_scenes` é o que faz a conta bater com o layout: a profundidade é gravada UMA
    # vez por imagem (`depth/<scene_id>.png`), então orçá-la por amostra inflaria a
    # coluna de controle em 41x — 31,7 GB onde são 0,77 GB.
    orcamento = estimate_disk_budget(n_previsto, 768, generates_image=True,
                                     image_megapixels=megapixels,
                                     n_scenes=len(imagens))
    print(f"[disco] {n_previsto} amostras · {len(imagens)} cenas · "
          f"{megapixels:.2f} MP médios · {json.dumps(orcamento)}")
    print("[disco] folga medida: ~84 GB (115 GB menos os 31 GB de B+C, "
          "REGISTRO.md:585). A rota A cabe a <=1 MP e NÃO cabe a >=2 MP.")

    opcionais = {k: v for k, v in (
        ("k_widen_fraction", args.k_widen_fraction),
        ("focus_quantile_low", args.focus_quantile_low),
        ("focus_quantile_high", args.focus_quantile_high),
        ("focus_band_coc_px", args.focus_band_coc_px)) if v is not None}
    config = RouteAConfig(
        output_dir=Path(args.output_dir), k_distribution=distribuicao,
        samples_per_image=int(args.samples_per_image), seed=args.seed,
        limit=args.limit,
        min_aif_laplacian_variance=args.min_aif_laplacian_variance,
        max_bokeh_over_aif_sharpness=args.max_bokeh_over_aif_sharpness,
        min_rendered_coc_p99_px=args.min_rendered_coc_p99_px,
        max_defocus_saturation_ratio=args.max_defocus_saturation_ratio,
        min_mask_area_ratio=args.min_mask_area_ratio,
        max_mask_area_ratio=args.max_mask_area_ratio,
        min_depth_useful_levels=args.min_depth_useful_levels,
        **opcionais)

    congelados = {k: v for k, v in vars(args).items()
                  if k.startswith(("min_", "max_")) and v is not None}
    print(f"[rota-a] limiares congelados: {congelados or 'nenhum — modo medir'}")
    if not congelados and not args.pilot:
        print("[rota-a] AVISO: run completo com todos os gates em modo medir. "
              "Intencional só se você quer o lote bruto para calibrar depois.")

    provenance_base = {
        "pipeline_commit": _git_commit(raiz) or f"sha256tree:{_source_sha256(raiz)}",
        "pipeline_source_sha256": _source_sha256(raiz),
        **depth.provenance(),
        # A rota A não roda segmentador. String vazia e a regra da banda em `mask_rule`,
        # por amostra — item F6 da auditoria.
        "mask_model_sha256": "",
        "mask_backend": None,
        "renderer": proc_renderer,
        "k_effective_factor": (laudo.get("measurements") or {}).get("k_effective_factor"),
        "renderer_report_sha256": _sha256(laudo_path),
        "k_distribution_path": str(args.k_distribution),
        "k_distribution_sha256": _sha256(Path(args.k_distribution)),
        "split_origin": "sorteado_por_cena",
        **extra_fonte,
    }
    (Path(args.output_dir) / "run_config.json").write_text(json.dumps(
        {"args": {k: (str(v) if isinstance(v, Path) else v)
                  for k, v in vars(args).items()},
         "provenance_base": provenance_base,
         # O layout de `depth/` DECLARADO, para que a validação e a leitura não
         # precisem inferi-lo contando arquivos. Um release por cena tem menos PNG que
         # amostras de propósito, e contar arquivos confundiria isso com um release
         # incompleto — que é a diferença entre publicar e não publicar.
         "depth_layout": config.depth_layout.value,
         "images_enumerated": len(imagens),
         "samples_expected": n_previsto,
         "disk_budget": orcamento},
        indent=2, ensure_ascii=False), encoding="utf-8")

    stats = run_route_a(imagens, load_aif=load_aif, depth_runtime=depth,
                        render_fn=renderer, split=split, config=config,
                        provenance_base=provenance_base)
    return 0 if stats.written else 1


if __name__ == "__main__":
    raise SystemExit(main())
