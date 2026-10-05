#!/usr/bin/env python3
"""Entrypoint da rota C — §3.2(c) do paper: pares reais, K pelo sweep da Eq. 5.

Piloto primeiro, sempre:

    python3 scripts/run_route_c.py --source realbokeh --pilot 200 \\
        --output-dir output/c_pilot \\
        --mirror-dir  /raid/.../data/RealBokeh_mirror \\
        --raw-dir      /raid/.../data/RealBokeh_3MP \\
        --models-dir  /raid/.../models \\
        --bokehme-dir third_party/BokehMe \\
        --renderer-report output/renderer_verification.json

O piloto **não congela limiar nenhum**: todos os onze gates saem em modo medir. O que
ele produz é o histograma de motivos de rejeição e a distribuição de cada métrica, que
`scripts/calibrate_thresholds.py` transforma em números para o run final.

Depois do piloto, e ANTES do run completo, rode também:

    python3 scripts/validate_focus_refinement.py --pilot-dir output/c_pilot \\
        --raw-dir /raid/.../data/RealBokeh_3MP \\
        --output-json output/focus_validation.json

Ele compara o `focus_disparity` obtido contra a distância de foco **medida** que a
RealBokeh publica (incerteza mediana ±0,010 m) e separa o resultado por `focus_source`.
A régua é 35,2% dentro de ±25% — a concordância da máscara do BiRefNet crua, medida no
piloto anterior. Se o refinamento não bater isso, não vale gerar 22.990 amostras com ele.

Rodar o run completo antes do piloto é escolher limiar por intuição — foi assim que o
`--k-max 300` do pipeline antigo virou 47% de amostras com K censurado no valor exato
do teto, sem que nada denunciasse.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dataio import build_scene_split, split_from_source          # noqa: E402
from qc.rejection import RejectionLog                            # noqa: E402
from routes.route_c import RouteCConfig, run_route_c             # noqa: E402


# --------------------------------------------------------------------------------
# Proveniência
# --------------------------------------------------------------------------------

def _source_sha256(repo: Path) -> str:
    """Hash do CÓDIGO que produziu o rótulo: todo `.py` de `src/` e `scripts/`.

    Isto é a proveniência primária, e não o commit. Um commit identifica o que estava
    versionado; este hash identifica o que **rodou** — que é a pergunta que importa
    quando alguém, daqui a seis meses, quiser saber por que uma amostra tem o K que tem.
    Um checkout sujo tem commit e não tem correspondência com o código executado.

    Caminho relativo entra no hash junto do conteúdo, para mover arquivo contar como
    mudança. Ordem estável, independente do filesystem.
    """
    digest = hashlib.sha256()
    arquivos = sorted(
        p for base in ("src", "scripts") for p in (repo / base).rglob("*.py")
        if "__pycache__" not in p.parts)
    if not arquivos:
        raise RuntimeError(
            f"nenhum .py em {repo}/src ou {repo}/scripts — não há pipeline a identificar.")
    for arquivo in arquivos:
        digest.update(str(arquivo.relative_to(repo)).encode("utf-8"))
        digest.update(arquivo.read_bytes())
    return digest.hexdigest()


def _git_commit(repo: Path) -> Optional[str]:
    """Commit, **quando existe**. Complementa `_source_sha256`, não o substitui.

    A cópia que roda no cluster chega por rsync e não tem `.git`. Exigir commit ali
    impediria de gerar dado por uma razão que não é sobre a qualidade do dado — e a
    pergunta que a proveniência precisa responder ("qual código produziu este rótulo?")
    já está respondida pelo hash do fonte.
    """
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


class _ComSensor:
    """Anexa `sensor_width_mm` a um par, sem tocar na dataclass da fonte.

    A largura do sensor da RealBokeh_3MP **não** está publicada. Sem ela, a Eq. 3 não
    fecha e `route_c._analytic_k` devolve `None` — o rótulo (Eq. 5) não muda, mas a
    amostra fica sem validador independente.

    Passar `--sensor-width-mm` é assumir um valor: ele entra na proveniência marcado
    como `[A]`, e o validador passa a existir. Não passar é rodar sem validador e ver
    no resumo quantas amostras ficaram assim. As duas escolhas são defensáveis; o que
    não é defensável é cravar 36 mm em silêncio dentro do laço.
    """

    __slots__ = ("_par", "sensor_width_mm")

    def __init__(self, par, sensor_width_mm: float):
        object.__setattr__(self, "_par", par)
        object.__setattr__(self, "sensor_width_mm", float(sensor_width_mm))

    def __getattr__(self, nome):
        return getattr(self._par, nome)


# --------------------------------------------------------------------------------
# Fontes
# --------------------------------------------------------------------------------

def _carrega_realbokeh(args):
    """Devolve `(pares, load_pair, split_de_origem, extra_de_proveniencia)`."""
    from sources.mirror_images import (
        MirrorIndex, MirrorImageLoader, order_pairs_for_sequential_read,
        sample_pairs_for_pilot,
    )
    from sources.realbokeh import (
        MIRROR_DATASET, MIRROR_IMAGE_HW, enumerate_pairs, enumeration_summary,
        load_scene_metadata, scene_source_splits,
    )

    if not args.mirror_dir or not args.raw_dir:
        raise SystemExit("--source realbokeh exige --mirror-dir e --raw-dir")

    print("[fonte] indexando o espelho (só a coluna de nome)…")
    index = MirrorIndex.build(args.mirror_dir, force=args.rebuild_index)
    print(f"[fonte] índice: {len(index)} linhas.")

    # Chaveado por `scene_key(split, numero)`: a numeração de cena REINICIA em cada
    # split do espelho, então ler um `metadata/` só e chavear pelo número cru fundiria
    # a cena 1 de `train`, `test` e `validation` numa só.
    metadata = load_scene_metadata(args.raw_dir)
    print(f"[fonte] metadata: {len(metadata)} cenas do repositório bruto.")

    log = RejectionLog(Path(args.output_dir) / "source_rejections.jsonl")
    pares = enumerate_pairs(index.names(), metadata,
                            log=log, source_dataset=MIRROR_DATASET)
    print(enumeration_summary(pares, log))

    pares = _aplica_teto(pares, args)
    pares = _aplica_fatia(pares, args)

    if args.pilot:
        pares = sample_pairs_for_pilot(pares, limit=args.pilot, seed=args.seed)
        cenas = len({p.scene_id for p in pares})
        print(f"[fonte] piloto: {len(pares)} pares de {cenas} cenas (seed {args.seed}).")

    pares = order_pairs_for_sequential_read(pares, index)
    if args.sensor_width_mm:
        pares = [_ComSensor(p, args.sensor_width_mm) for p in pares]

    saida = Path(args.output_dir)
    loader = MirrorImageLoader(
        args.mirror_dir, index, expected_hw=MIRROR_IMAGE_HW,
        ledger_path=saida / "source_images.jsonl",
        store_dir=(saida / "source") if args.store_source_images else None)
    extra = {"source_dataset": MIRROR_DATASET,
             "source_image_hw": list(MIRROR_IMAGE_HW),
             "sensor_width_mm": args.sensor_width_mm,
             "sensor_width_mm_evidence": "[A] assumido via --sensor-width-mm"
                                         if args.sensor_width_mm else None}
    return pares, loader, scene_source_splits(pares), extra


def _aplica_fatia(pares, args):
    """Fatia o trabalho entre processos, **por CENA**.

    Por cena e não por par: todos os níveis de uma cena compartilham a mesma AIF, a
    mesma profundidade e a mesma máscara. Mantê-los no mesmo processo é o que permite
    reusar essas três coisas, e é pré-requisito para o `D_focus` único por série.

    A fatia sai de um hash estável do `scene_id`, não de `enumerate`: assim ela não
    muda se a ordem de leitura mudar, e relançar uma fatia reprocessa exatamente o
    mesmo conjunto.
    """
    if args.num_shards <= 1:
        return pares
    if not 0 <= args.shard < args.num_shards:
        raise SystemExit(f"--shard {args.shard} fora de [0, {args.num_shards})")

    import hashlib

    def da_fatia(cena: str) -> bool:
        h = hashlib.sha256(cena.encode("utf-8")).digest()
        return int.from_bytes(h[:8], "big") % args.num_shards == args.shard

    fatiados = [p for p in pares if da_fatia(p.scene_id)]
    print(f"[fatia] {args.shard}/{args.num_shards}: {len(fatiados)} de {len(pares)} pares, "
          f"{len({p.scene_id for p in fatiados})} cenas")
    return fatiados


def _aplica_teto(pares, args):
    """Teto de níveis por cena. **Antes** da amostragem do piloto, de propósito.

    Se o teto viesse depois, o piloto mediria a distribuição do dataset cheio e os
    limiares sairiam calibrados para um dataset que não é o que vamos gerar.
    """
    if not args.max_levels_per_scene:
        print("[fonte] SEM teto de níveis por cena. O paper descreve 2 a 4 por conjunto "
              "(paper.txt:1001-1003); sem teto, 25% das amostras saem de 6,2% das cenas.")
        return pares
    from sources.level_selection import cap_levels_per_scene, cap_summary

    depois = cap_levels_per_scene(pares, max_levels=args.max_levels_per_scene)
    print(cap_summary(pares, depois, max_levels=args.max_levels_per_scene))
    return depois


def _carrega_lfdof(args):
    """LFDOF: mesmo formato de espelho, sem metadata e sem EXIF.

    Não há `--raw-dir` nem `--sensor-width-mm`: o LFDOF não publica f-number, focal
    length nem distância de foco. Isso é o comportamento CORRETO — `PairSource` declara
    esses campos como `Optional`, o gate `aif_aperture_is_narrow` sai `applicable=False`
    e `_analytic_k` devolve `None`. Consequência declarada: as amostras do LFDOF não têm
    validador analítico, e não há gabarito de plano de foco para elas.
    """
    from sources.lfdof import (LFDOF_DATASET, LFDOF_IMAGE_HW, enumerate_pairs,
                               enumeration_summary, scene_source_splits)
    from sources.lfdof_images import LFDOFImageLoader
    from sources.mirror_images import (MirrorIndex, order_pairs_for_sequential_read,
                                       sample_pairs_for_pilot)

    if not args.mirror_dir:
        raise SystemExit("--source lfdof exige --mirror-dir (o espelho akcit-pixel/LFDOF)")

    print("[fonte] indexando o espelho do LFDOF (só a coluna de nome)…")
    index = MirrorIndex.build(args.mirror_dir, force=args.rebuild_index)
    print(f"[fonte] índice: {len(index)} linhas.")

    log = RejectionLog(Path(args.output_dir) / "source_rejections.jsonl")
    pares = enumerate_pairs(index.names(), log=log, source_dataset=LFDOF_DATASET)
    print(enumeration_summary(pares, log))

    pares = _aplica_teto(pares, args)
    pares = _aplica_fatia(pares, args)

    if args.pilot:
        pares = sample_pairs_for_pilot(pares, limit=args.pilot, seed=args.seed)
        print(f"[fonte] piloto: {len(pares)} pares de "
              f"{len({p.scene_id for p in pares})} cenas (seed {args.seed}).")

    pares = order_pairs_for_sequential_read(pares, index)

    saida = Path(args.output_dir)
    loader = LFDOFImageLoader(
        args.mirror_dir, index, expected_hw=LFDOF_IMAGE_HW,
        ledger_path=saida / "source_images.jsonl",
        store_dir=(saida / "source") if args.store_source_images else None)
    extra = {"source_dataset": LFDOF_DATASET,
             "source_image_hw": list(LFDOF_IMAGE_HW),
             "sensor_width_mm": None,
             "sensor_width_mm_evidence": "o LFDOF não publica EXIF; sem Eq. 3 a fechar"}
    return pares, loader, scene_source_splits(pares), extra


def _carrega_fonte(args):
    if args.source == "realbokeh":
        return _carrega_realbokeh(args)
    if args.source == "lfdof":
        return _carrega_lfdof(args)
    raise SystemExit(
        f"fonte {args.source!r} ainda não tem adaptador. A rota C recebe os pares por "
        "protocolo — escrever o adaptador não exige tocar em routes/route_c.py.")


# --------------------------------------------------------------------------------
# Split
# --------------------------------------------------------------------------------

def _monta_split(pares, splits_de_origem, args):
    """Origem manda quando ela publica os dois lados; senão, sorteia por cena.

    Medido: o espelho `akcit-pixel/RealBokeh` publica **só** `train` (20.495 de
    20.495 linhas). Alimentar `split_from_source` só com ele produz um split com
    `val_fraction == 0` — tecnicamente válido, inútil. As cenas de `test` e
    `validation` da RealBokeh_3MP existem no repositório bruto, mas não no espelho.

    Então: se a origem publica pelo menos uma cena de validação, ela manda, e a
    intenção de quem montou o dataset é preservada. Se não publica, sorteia por CENA,
    determinístico, e diz em voz alta que sorteou. O que não acontece é val vazio
    passar batido — validação vazia é o jeito mais silencioso de não ter validação.
    """
    if splits_de_origem:
        candidato = split_from_source(splits_de_origem)
        contagens = candidato.counts()
        if contagens.get("val", 0) > 0:
            print(f"[split] herdado da origem: {contagens}")
            return candidato, "origem"
        print(f"[split] a origem publica só um lado ({contagens}). Sorteando validação "
              f"por CENA com val_fraction={args.val_fraction}.")

    split = build_scene_split({p.scene_id for p in pares},
                              val_fraction=args.val_fraction)
    print(f"[split] sorteado por cena: {split.counts()}")
    return split, "sorteado_por_cena"


# --------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Rota C — pares reais, K pelo sweep da Eq. 5.",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", required=True, choices=["realbokeh", "lfdof"])
    p.add_argument("--output-dir", required=True)
    p.add_argument("--models-dir", required=True,
                   help="contém checkpoints/depth_pro.pt e BiRefNet/")
    p.add_argument("--bokehme-dir", required=True)
    p.add_argument("--renderer-report", required=True,
                   help="JSON de scripts/verify_renderer.py. Sem o laudo, "
                        "`is_final_label_renderer` é declaração, não medição.")
    p.add_argument("--mirror-dir", default="",
                   help="snapshot local de akcit-pixel/RealBokeh (85 shards parquet)")
    p.add_argument("--raw-dir", default="",
                   help="raiz do timseizinger/RealBokeh_3MP, com "
                        "<split>/metadata/<cena>.json nos três splits")
    p.add_argument("--rebuild-index", action="store_true")
    p.add_argument("--store-source-images", action="store_true",
                   help="copia os bytes ORIGINAIS da AIF e da bokeh para <out>/source/, "
                        "tornando o release autocontido (~49 GB na RealBokeh inteira). "
                        "Sem a flag, o release grava rótulo + sha256 de cada imagem em "
                        "source_images.jsonl e aponta para akcit-pixel/RealBokeh — "
                        "31 GB, e o join continua verificável byte a byte.")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--val-fraction", type=float, default=0.05)
    p.add_argument("--sensor-width-mm", type=float, default=None,
                   help="[A] largura do sensor da origem. Sem ela a Eq. 3 não fecha e "
                        "o validador analítico do sweep fica ausente — o rótulo não "
                        "muda, a auditoria fica mais pobre. Ver _ComSensor.")

    grupo = p.add_argument_group("escopo")
    grupo.add_argument("--shard", type=int, default=0,
                       help="índice desta fatia. Fatiamento é POR CENA.")
    grupo.add_argument("--num-shards", type=int, default=1,
                       help="quantas fatias no total. Cada uma escreve no próprio "
                            "--output-dir; juntar depois é concatenar manifestos.")
    grupo.add_argument("--max-levels-per-scene", type=int, default=4,
                       help="teto de aberturas por cena, espaçadas uniformemente. "
                            "Default 4, que é o que reproduz os 13K do paper "
                            "(paper.txt:1001-1003): 13.799 no split train, contra "
                            "20.495 sem teto. Use 0 para desligar.")
    grupo.add_argument("--pilot", type=int, default=None,
                       help="amostra ~N pares sorteando CENAS inteiras com seed, antes "
                            "de processar. É o certo para calibrar limiar: o prefixo "
                            "da ordem de disco sai de uma dúzia de cenas.")
    grupo.add_argument("--limit", type=int, default=None,
                       help="teto de amostras ACEITAS no laço. Cru: não reequilibra "
                            "por cena. Para piloto use --pilot.")

    lim = p.add_argument_group(
        "limiares",
        "TODOS default None = mede e não bloqueia. Congele a partir do piloto com "
        "scripts/calibrate_thresholds.py.")
    for flag in ("min-calibration-ssim", "min-focus-mask-sharpness-ratio",
                 "min-mask-area-ratio", "max-mask-area-ratio",
                 "max-mask-border-coverage", "min-aif-laplacian-variance",
                 "max-bokeh-over-aif-sharpness", "min-aif-f-number", "min-mask-iou",
                 "min-focus-region-retention"):
        lim.add_argument(f"--{flag}", type=float, default=None)
    lim.add_argument("--min-depth-useful-levels", type=int, default=None)

    foco = p.add_argument_group(
        "região em foco (§3.2(c))",
        "O substituto AUTOMÁTICO do refinamento manual da máscara. Nenhum destes "
        "valores vem do paper, que não publica nada sobre o passo de refinamento: são "
        "todos [A], e os defaults estão em src/qc/focus_region.py. Medido no piloto: a "
        "máscara do BiRefNet acerta o plano de foco em 35,2% das amostras e declina "
        "(probabilidade 0) em 20,6% delas. Valide com "
        "scripts/validate_focus_refinement.py ANTES de gerar o lote completo.")
    foco.add_argument("--focus-retention-long-side", type=int, default=0,
                      help="lado longo da grade em que a retenção é medida. **0 = "
                           "resolução cheia, e é o default por medição**: a 1500x2000 "
                           "o caminho reduzido custa 433 ms contra 391 ms do cheio, "
                           "porque reduzir duas fotos de 3 MP (185 ms cada) custa mais "
                           "que a retenção que a redução economiza (291 -> 22 ms). "
                           "A grade usada vai para o metadado de qualquer forma — "
                           "janela em pixel sem resolução ao lado não diz nada.")
    foco.add_argument("--focus-retention-window-px", type=int, default=None,
                      help="janela do filtro de média, EM PIXEL DA GRADE DE TRABALHO")
    foco.add_argument("--focus-top-fraction", type=float, default=None,
                      help="fração do quadro que forma a 'small yet reliable region'")
    foco.add_argument("--focus-agreement-floor", type=float, default=None,
                      help="acima disto a máscara do BiRefNet é MANTIDA como veio")

    busca = p.add_argument_group("busca de K (Eq. 5)")
    busca.add_argument("--k-min", type=float, default=None)
    busca.add_argument("--k-max", type=float, default=None)
    busca.add_argument("--k-absolute-max", type=float, default=None)
    return p


def main() -> int:
    args = build_parser().parse_args()
    raiz = Path(__file__).resolve().parents[1]
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # -- o laudo do renderer, antes de qualquer modelo -------------------------
    laudo_path = Path(args.renderer_report)
    if not laudo_path.is_file():
        raise SystemExit(
            f"laudo do renderer não encontrado: {laudo_path}\nRode antes:\n"
            "  python3 scripts/verify_renderer.py --bokehme-dir ... --output-json ...")
    laudo = json.loads(laudo_path.read_text(encoding="utf-8"))
    if not laudo.get("is_final_label_renderer"):
        raise SystemExit(
            "o renderer NÃO passou nos três testes de verificação. Gerar rótulo com ele "
            "reproduziria o defeito D4 — o gaussiano de fallback que ninguém viu porque "
            "nada media o borrão que saía.")

    # -- modelos, carregados uma vez -------------------------------------------
    from model_runtime import BiRefNetRuntime, DepthProRuntime
    from renderer.bokehme import BokehMeConfig, BokehMeRenderer

    modelos = Path(args.models_dir)
    depth = DepthProRuntime(modelos / "checkpoints" / "depth_pro.pt", device=args.device)
    mask = BiRefNetRuntime(modelos / "BiRefNet", device=args.device)
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
                "dado — o laudo é o que autoriza o renderer a virar rótulo.")

    # -- fonte ------------------------------------------------------------------
    pares, load_pair, splits_de_origem, extra_fonte = _carrega_fonte(args)
    if not pares:
        raise SystemExit("nenhum par sobreviveu à enumeração — veja o histograma acima.")

    split, origem_do_split = _monta_split(pares, splits_de_origem, args)

    # -- config -----------------------------------------------------------------
    # `None` significa "usa o default do módulo", e o default do módulo é a única
    # definição do valor. Repetir o número aqui criaria uma segunda definição, que é
    # como o projeto chegou a quatro interpretações de K.
    opcionais = {k: v for k, v in (
        ("k_min", args.k_min), ("k_max", args.k_max),
        ("k_absolute_max", args.k_absolute_max),
        ("focus_retention_window_px", args.focus_retention_window_px),
        ("focus_top_fraction", args.focus_top_fraction),
        ("focus_agreement_floor", args.focus_agreement_floor)) if v is not None}
    # 0 é a forma de pedir resolução cheia na CLI: `--focus-retention-long-side 0`.
    # `None` na dataclass já significa isso, e um inteiro não tem como ser "ausente".
    opcionais["focus_retention_long_side"] = (args.focus_retention_long_side or None)
    config = RouteCConfig(
        output_dir=Path(args.output_dir), seed=args.seed, limit=args.limit,
        min_calibration_ssim=args.min_calibration_ssim,
        min_focus_mask_sharpness_ratio=args.min_focus_mask_sharpness_ratio,
        min_mask_area_ratio=args.min_mask_area_ratio,
        max_mask_area_ratio=args.max_mask_area_ratio,
        max_mask_border_coverage=args.max_mask_border_coverage,
        min_aif_laplacian_variance=args.min_aif_laplacian_variance,
        max_bokeh_over_aif_sharpness=args.max_bokeh_over_aif_sharpness,
        min_depth_useful_levels=args.min_depth_useful_levels,
        min_aif_f_number=args.min_aif_f_number,
        min_mask_iou=args.min_mask_iou,
        min_focus_region_retention=args.min_focus_region_retention,
        **opcionais)

    congelados = {k: v for k, v in vars(args).items()
                  if k.startswith(("min_", "max_")) and v is not None}
    print(f"[rota-c] limiares congelados: {congelados or 'nenhum — modo medir'}")
    if not congelados and not args.pilot:
        print("[rota-c] AVISO: run completo com todos os gates em modo medir. "
              "Isso é intencional só se você quer o dataset bruto para calibrar depois.")

    provenance_base = {
        "pipeline_commit": _git_commit(raiz) or f"sha256tree:{_source_sha256(raiz)}",
        "pipeline_source_sha256": _source_sha256(raiz),
        **depth.provenance(),
        **mask.provenance(),
        "renderer": proc_renderer,
        "k_effective_factor": (laudo.get("measurements") or {}).get("k_effective_factor"),
        "renderer_report_sha256": _sha256(laudo_path),
        "split_origin": origem_do_split,
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
         "pairs_enumerated": len(pares)},
        indent=2, ensure_ascii=False), encoding="utf-8")

    stats = run_route_c(pares, load_pair=load_pair, depth_runtime=depth,
                        mask_runtime=mask, render_fn=renderer, split=split,
                        config=config, provenance_base=provenance_base)
    return 0 if stats.written else 1


if __name__ == "__main__":
    raise SystemExit(main())
