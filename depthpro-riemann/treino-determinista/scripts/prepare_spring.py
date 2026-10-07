#!/usr/bin/env python3
"""
scripts/prepare_spring.py
=========================
Converte o dataset Spring para o layout do nosso pipeline, transformando disparidade
estéreo em profundidade métrica.

A conversão é
        Z = fx · B / d
com fx a distância focal em pixels (arquivo `intrinsics.txt` de cada sequência) e B a
linha de base estéreo, que no Spring é sempre 0.065 m.

Três armadilhas que este script trata explicitamente, porque errar qualquer uma delas
produz profundidade plausível à vista mas errada nos números:

1. **Resolução dobrada do ground truth.** A disparidade do Spring é distribuída em
   resolução maior que o RGB, para precisão subpixel. Disparidade é medida em PIXELS,
   então ao reduzir o mapa para a resolução do RGB é preciso dividir os VALORES pelo
   mesmo fator. Sem isso a profundidade sai pela metade. O script detecta a razão entre
   as resoluções e aplica o fator, avisando no log.

2. **Céu e disparidade nula.** Onde d → 0 a profundidade explode. O Spring inclui o céu
   no ground truth de propósito, então esses pixels existem em quantidade. São removidos
   pela máscara, não clampeados, para não criarem uma parede falsa a uma distância fixa.

3. **Quadros consecutivos são quase idênticos.** Usar os ~130 quadros de uma sequência
   não acrescenta informação e multiplica o custo. O `--passo` subamostra. Para a
   estatística o que conta é o número de SEQUÊNCIAS, que é a unidade de agrupamento.

Observação sobre splits: o ground truth do split de teste do Spring é fechado (fica no
servidor do benchmark). Usamos o split de **treino** do Spring, que tem GT público. Como
não estamos treinando nele, isso não é vazamento — mas se um dia formos treinar, será
preciso dividir as 47 sequências em treino/validação/teste por conta própria.

Uso:
    python scripts/prepare_spring.py \\
        --spring-root /data/spring/train \\
        --out-root /data/spring_prep/eval \\
        --passo 10 --max-depth 100
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np

try:
    import h5py
except ImportError:
    print("ERRO: h5py e necessario para ler .dsp5 (pip install h5py)")
    sys.exit(1)

try:
    import cv2
    _CV2 = True
except ImportError:
    _CV2 = False

BASELINE_SPRING = 0.065  # metros, constante no dataset


def ler_dsp5(caminho: Path) -> np.ndarray:
    """
    Lê um arquivo .dsp5 (HDF5). O dataset interno costuma se chamar 'disparity'; se não
    existir, pegamos o primeiro dataset 2D do arquivo.
    """
    with h5py.File(caminho, "r") as f:
        if "disparity" in f:
            return np.array(f["disparity"]).astype(np.float32)
        for k in f.keys():
            arr = np.array(f[k])
            if arr.ndim >= 2:
                return arr.astype(np.float32)
    raise RuntimeError(f"nenhum dataset 2D encontrado em {caminho}")


def ler_fx(seq_dir: Path, override=None):
    """
    Extrai fx (pixels) do intrinsics.txt da sequência.

    O arquivo traz uma linha por quadro com os parâmetros intrínsecos. Assumimos que o
    primeiro número de cada linha é fx e usamos a mediana entre os quadros, o que é
    robusto a variação de foco ao longo da sequência.
    """
    if override:
        return float(override), "override"
    for nome in ("intrinsics.txt", "cam_data/intrinsics.txt"):
        p = seq_dir / nome
        if p.exists():
            vals = []
            for linha in p.read_text().strip().splitlines():
                nums = [float(x) for x in re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", linha)]
                if nums:
                    vals.append(nums[0])
            if vals:
                return float(np.median(vals)), str(p.name)
    return None, None


def achar_sequencias(raiz: Path, camera: str):
    """Devolve [(nome_seq, pasta_frames, pasta_disp, pasta_seq)] das sequências válidas."""
    saida = []
    for seq in sorted(p for p in raiz.iterdir() if p.is_dir()):
        fr = seq / f"frame_{camera}"
        dp = seq / f"disp1_{camera}"
        if fr.exists() and dp.exists():
            saida.append((seq.name, fr, dp, seq))
    return saida


def redimensionar(arr, hw):
    """Reduz um mapa para (H, W). Usa área para não criar aliasing na disparidade."""
    if arr.shape[:2] == tuple(hw):
        return arr
    if _CV2:
        return cv2.resize(arr, (hw[1], hw[0]), interpolation=cv2.INTER_AREA)
    fy = arr.shape[0] // hw[0]
    fx_ = arr.shape[1] // hw[1]
    if fy >= 1 and fx_ >= 1:
        return arr[::fy, ::fx_][: hw[0], : hw[1]]
    raise RuntimeError("instale opencv-python para redimensionar o ground truth")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spring-root", required=True,
                    help="pasta com as sequencias, ex.: /data/spring/train")
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--camera", default="left", choices=["left", "right"])
    ap.add_argument("--passo", type=int, default=10,
                    help="usa 1 quadro a cada N. Quadros consecutivos sao quase identicos; "
                         "o que conta para a estatistica e o numero de SEQUENCIAS.")
    ap.add_argument("--sequencias", nargs="*", default=None,
                    help="restringe a estas sequencias (ex.: 0001 0002)")
    ap.add_argument("--max-depth", type=float, default=100.0,
                    help="pixels acima disto saem da mascara (ceu e fundo distante). "
                         "0 desliga.")
    ap.add_argument("--min-disp", type=float, default=0.05,
                    help="disparidade minima em pixels; abaixo disto Z explode e o pixel "
                         "e invalidado")
    ap.add_argument("--fx", type=float, default=None,
                    help="forca a distancia focal em pixels (ignora intrinsics.txt)")
    ap.add_argument("--baseline", type=float, default=BASELINE_SPRING,
                    help="linha de base estereo em metros (Spring: 0.065)")
    ap.add_argument("--disp-escala", default="auto",
                    help="'auto' detecta a razao entre a resolucao do GT e a do RGB e "
                         "divide os valores de disparidade por ela; ou um numero fixo")
    ap.add_argument("--particao", default=None,
                    choices=["train", "val", "test"],
                    help="divide as sequencias automaticamente em 60/20/20 de forma "
                         "DETERMINISTICA e prepara apenas a particao pedida. Garante que "
                         "as sequencias sao disjuntas entre as tres chamadas, o que e o "
                         "erro mais facil de cometer na pressa.")
    ap.add_argument("--particao-seed", type=int, default=42,
                    help="semente do embaralhamento das sequencias na particao")
    ap.add_argument("--particao-fracoes", nargs=3, type=float,
                    default=[0.50, 0.15, 0.35], metavar=("TRAIN", "VAL", "TEST"),
                    help="fracoes de treino/val/teste. O padrao NAO e 60/20/20: o split "
                         "publico do Spring tem 37 sequencias (as 10 de teste tem GT "
                         "fechado), e 20%% dariam apenas 8 sequencias de teste, pouco "
                         "para a estatistica pareada. Com 50/15/35 sobram ~13. O volume "
                         "de treino se recupera reduzindo o --passo, ja que o que importa "
                         "para treinar e o numero de IMAGENS, nao de sequencias.")
    ap.add_argument("--max-por-seq", type=int, default=None,
                    help="teto de quadros por sequencia, depois do --passo")
    args = ap.parse_args()

    raiz = Path(args.spring_root)
    out = Path(args.out_root)
    for sub in ("rgb", "depth", "mask"):
        (out / sub).mkdir(parents=True, exist_ok=True)

    seqs = achar_sequencias(raiz, args.camera)
    if args.sequencias:
        alvo = set(args.sequencias)
        seqs = [s for s in seqs if s[0] in alvo]

    if args.particao:
        # Embaralhamento deterministico: as tres chamadas (train/val/test) com a mesma
        # semente produzem conjuntos DISJUNTOS e reproduziveis.
        nomes = sorted(x[0] for x in seqs)
        rng = np.random.default_rng(args.particao_seed)
        ordem = list(rng.permutation(nomes))
        n = len(ordem)
        f_tr, f_va, _ = args.particao_fracoes
        corte1, corte2 = int(f_tr * n), int((f_tr + f_va) * n)
        fatias = {"train": ordem[:corte1], "val": ordem[corte1:corte2],
                  "test": ordem[corte2:]}
        escolhidas = set(fatias[args.particao])
        seqs = [x for x in seqs if x[0] in escolhidas]
        print(f"[particao] {args.particao}: {len(escolhidas)} de {n} sequencias "
              f"(seed {args.particao_seed}, fracoes {args.particao_fracoes})")
        print(f"           {sorted(escolhidas)}")
        if args.particao == "test" and len(escolhidas) < 10:
            print(f"           [ATENCAO] apenas {len(escolhidas)} sequencias de teste. "
                  f"Esse e o n da estatistica pareada final e o IC ficara largo. "
                  f"Considere --particao-fracoes 0.45 0.15 0.40.")
    if not seqs:
        print(f"ERRO: nenhuma sequencia com frame_{args.camera}/ e disp1_{args.camera}/ "
              f"em {raiz}")
        sys.exit(1)
    print(f"[busca] {len(seqs)} sequencias em {raiz}")

    n_total = 0
    escalas_vistas = set()
    fx_vistos = []
    faixas = []
    seq_ok = 0

    for nome_seq, dir_fr, dir_dp, dir_seq in seqs:
        fx, origem = ler_fx(dir_seq, args.fx)
        if fx is None:
            print(f"  [pula] {nome_seq}: intrinsics.txt nao encontrado "
                  f"(use --fx para forcar)")
            continue
        fx_vistos.append(fx)

        frames = sorted(dir_fr.glob("*.png")) or sorted(dir_fr.glob("*.jpg"))
        frames = frames[:: max(1, args.passo)]
        if args.max_por_seq:
            frames = frames[: args.max_por_seq]

        n_seq = 0
        for fpath in frames:
            m = re.search(r"(\d+)(?=\.\w+$)", fpath.name)
            if not m:
                continue
            idx = m.group(1)
            dpath = dir_dp / f"disp1_{args.camera}_{idx}.dsp5"
            if not dpath.exists():
                cand = list(dir_dp.glob(f"*{idx}.dsp5"))
                if not cand:
                    continue
                dpath = cand[0]

            rgb = cv2.imread(str(fpath)) if _CV2 else None
            if rgb is None:
                from PIL import Image
                rgb = np.array(Image.open(fpath).convert("RGB"))[:, :, ::-1]
            H, W = rgb.shape[:2]

            disp = ler_dsp5(dpath)
            if disp.ndim == 3:
                disp = disp[..., 0]

            # --- fator de escala da disparidade ---------------------------
            if args.disp_escala == "auto":
                fator = disp.shape[1] / float(W)
            else:
                fator = float(args.disp_escala)
            escalas_vistas.add(round(fator, 3))

            if disp.shape[:2] != (H, W):
                disp = redimensionar(disp, (H, W))
            # disparidade e medida em PIXELS: ao reduzir a grade, os valores encolhem
            # na mesma proporcao
            if abs(fator - 1.0) > 1e-6:
                disp = disp / fator

            # --- disparidade -> profundidade -------------------------------
            d = np.abs(disp)
            valido = np.isfinite(d) & (d > args.min_disp)
            depth = np.zeros_like(d, dtype=np.float32)
            np.divide(fx * args.baseline, d, out=depth, where=valido)
            if args.max_depth and args.max_depth > 0:
                valido &= (depth <= args.max_depth)
            depth = np.where(valido, depth, 0.0).astype(np.float32)

            chave = f"seq{nome_seq}__{args.camera}_{idx}"
            if _CV2:
                cv2.imwrite(str(out / "rgb" / f"{chave}.png"), rgb)
            else:
                from PIL import Image
                Image.fromarray(rgb[:, :, ::-1]).save(out / "rgb" / f"{chave}.png")
            np.save(out / "depth" / f"{chave}.npy", depth)
            np.save(out / "mask" / f"{chave}.npy", valido.astype(np.float32))

            if len(faixas) < 60 and valido.any():
                v = depth[valido]
                faixas.append((float(np.percentile(v, 5)), float(np.percentile(v, 50)),
                               float(np.percentile(v, 95)), float(valido.mean())))
            n_seq += 1
            n_total += 1

        if n_seq:
            seq_ok += 1
            print(f"  [ok] {nome_seq}: {n_seq} quadros  (fx={fx:.1f} de {origem})")

    # ------------------------- relatório de sanidade -------------------------
    print(f"\n[saida] {out}")
    print(f"  quadros preparados : {n_total}")
    print(f"  sequencias         : {seq_ok}   <- este e o n efetivo da estatistica")
    if fx_vistos:
        print(f"  fx (px)            : mediana {np.median(fx_vistos):.1f}, "
              f"faixa {min(fx_vistos):.1f}-{max(fx_vistos):.1f}")
    print(f"  baseline (m)       : {args.baseline}")
    print(f"  fator de escala da disparidade detectado: {sorted(escalas_vistas)}")
    if escalas_vistas and max(escalas_vistas) > 1.01:
        print("    -> o ground truth vem em resolucao maior que o RGB; os VALORES de")
        print("       disparidade foram divididos pelo mesmo fator, que e o correto,")
        print("       porque disparidade e medida em pixels.")

    if faixas:
        p5 = np.median([f[0] for f in faixas])
        p50 = np.median([f[1] for f in faixas])
        p95 = np.median([f[2] for f in faixas])
        cob = np.median([f[3] for f in faixas])
        print(f"\n  CONFERENCIA DE SANIDADE DA CONVERSAO")
        print(f"    profundidade p5 / mediana / p95 : "
              f"{p5:.2f} / {p50:.2f} / {p95:.2f} m")
        print(f"    cobertura valida da mascara     : {100*cob:.1f}%")
        print( "    Espera-se uma cena de filme: mediana na casa de metros a algumas")
        print( "    dezenas de metros. Se a mediana vier em centimetros ou em centenas")
        print( "    de metros, ha erro de fator 2 na disparidade ou de fx. Nesse caso")
        print( "    rode com --disp-escala 1 (ou 2) e compare.")
    print("\n  Proximo passo: a conferencia visual descrita no GUIA_SPRING.md, secao 5.3.")


if __name__ == "__main__":
    main()
