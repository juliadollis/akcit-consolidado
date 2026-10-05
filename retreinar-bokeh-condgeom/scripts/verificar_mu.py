#!/usr/bin/env python3
"""Confere a aritmética do `mu` do C4 — sem GPU, sem rede.

FAZ DUAS COISAS
---------------
1. **Cruza** a fórmula vetorizada que o `backbone.sample_sigma` usa com o
   `calculate_shift` do diffusers, em vários `seq_len`. Elas TÊM que coincidir:
   o `sample_sigma` reimplementa a reta para aceitar um tensor `(B,)` de
   `seq_len` (um `mu` por amostra, exigido pelo `sigma_mu_source="full_image"`),
   e uma divergência aqui seria um treino silenciosamente fora do cronograma da
   inferência. Se `diffusers` não estiver instalado, o cruzamento é pulado e o
   script avisa.

2. **Imprime a tabela de regimes do C4**, de preferência a partir da resolução
   REAL medida pelo `c0_1_resolucao_dfs.py` (lê o JSON dele se existir).

CONTEXTO (C4)
-------------
FATO: os dois eixos descasam. O treino sorteia sigma com `mu` derivado do
`seq_len` do CROP (1024 para 512²), enquanto a inferência com tiling deriva o
`mu` do `seq_len` da imagem INTEIRA (`flux.py:624`, antes do tiling) e todo tile
herda esse cronograma.

HIPÓTESE, NÃO DEMONSTRADA: que casar os dois melhora o resultado. O modelo é
condicionado em sigma e o treino cobre (0,1) inteiro — é desbalanceamento de
DENSIDADE, não fora-de-domínio. Por isso `sigma_mu_source` é EIXO DE
EXPERIMENTO no `StageConfig`, não uma correção aplicada de ofício.

USO
---
    python3 scripts/verificar_mu.py
    python3 scripts/verificar_mu.py --res 1024x688 --image-size 512
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from _comum import (
    BASE_IMAGE_SEQ_LEN, BASE_SHIFT, MAX_IMAGE_SEQ_LEN, MAX_SHIFT,
    alinhar16, calculate_shift_local, imprimir_tabela, linha_regime, seq_len_de,
)

SEQ_DE_TESTE = [64, 256, 512, 672, 1024, 2048, 2752, 4096, 7350, 16384]


def cruzar_com_diffusers() -> bool:
    """True se conferiu contra o diffusers; False se o pacote não existe."""
    try:
        from diffusers.pipelines.flux.pipeline_flux import calculate_shift
    except Exception as exc:  # noqa: BLE001
        print(f"  diffusers indisponível ({type(exc).__name__}) — cruzamento PULADO.")
        print("  A tabela abaixo usa a reimplementação local de `_comum.py`.")
        return False

    print("  Cruzando `calculate_shift_local` (nossa) × `calculate_shift` (diffusers):\n")
    linhas, pior = [], 0.0
    for seq in SEQ_DE_TESTE:
        nosso = calculate_shift_local(seq)
        deles = calculate_shift(seq, BASE_IMAGE_SEQ_LEN, MAX_IMAGE_SEQ_LEN, BASE_SHIFT, MAX_SHIFT)
        d = abs(nosso - deles)
        pior = max(pior, d)
        linhas.append({"seq_len": seq, "nosso_mu": f"{nosso:.6f}",
                       "diffusers_mu": f"{deles:.6f}", "|dif|": f"{d:.2e}"})
    imprimir_tabela(linhas, ["seq_len", "nosso_mu", "diffusers_mu", "|dif|"])
    print(f"\n  maior divergência: {pior:.2e}", "OK" if pior < 1e-9 else "  ← DIVERGE!")
    return True


def conferir_backbone_vetorizado() -> None:
    """Se torch existir, confere a versão tensorizada contra a escalar."""
    try:
        import torch
    except Exception:
        print("  torch indisponível — checagem da versão vetorizada PULADA.")
        return

    m = (MAX_SHIFT - BASE_SHIFT) / (MAX_IMAGE_SEQ_LEN - BASE_IMAGE_SEQ_LEN)
    b = BASE_SHIFT - BASE_IMAGE_SEQ_LEN * m
    seq = torch.tensor(SEQ_DE_TESTE, dtype=torch.float32)
    mu_vet = seq * m + b
    mu_esc = torch.tensor([calculate_shift_local(s) for s in SEQ_DE_TESTE], dtype=torch.float32)
    d = (mu_vet - mu_esc).abs().max().item()
    print(f"\n  versão vetorizada (a de `sample_sigma`) × escalar: maior |dif| = {d:.2e}",
          "OK" if d < 1e-5 else "  ← DIVERGE!")


def sigma_de(u: float, mu: float) -> float:
    """A transformação de shift aplicada ao logit-normal, como no treino."""
    e = math.exp(mu)
    return (e * u) / (1.0 + (e - 1.0) * u)


def main() -> None:
    p = argparse.ArgumentParser(
        description="Confere a aritmética do mu (C4) e imprime a tabela de regimes.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--res", default=None, help="resolução armazenada, ex. 1024x688")
    p.add_argument("--image-size", type=int, default=512, help="crop de treino")
    p.add_argument("--json-c0-1", default="outputs/c0_1_resolucao.json",
                   help="saída do c0_1 para pegar a resolução medida")
    args = p.parse_args()

    print("=" * 72)
    print("Verificação da aritmética do mu (C4)")
    print("=" * 72)
    print("\n[1] Fórmula\n")
    cruzou = cruzar_com_diffusers()
    conferir_backbone_vetorizado()

    # ── resolução ───────────────────────────────────────────────────────────
    w = h = None
    origem = None
    if args.res:
        try:
            ws, hs = args.res.lower().split("x")
            w, h, origem = int(ws), int(hs), "--res"
        except Exception:
            raise SystemExit(f"--res inválido: {args.res!r} (esperado WxH, ex. 1024x688)")
    else:
        jp = Path(args.json_c0_1)
        if jp.is_file():
            try:
                dados = json.loads(jp.read_text(encoding="utf-8"))
                for f in dados.get("fontes", []):
                    moda = (f.get("colunas", {}).get("image_blur") or {}).get("moda")
                    if moda:
                        w, h, origem = moda["w"], moda["h"], f"{jp} ({f['repo']})"
                        break
            except Exception as exc:  # noqa: BLE001
                print(f"\n  (não consegui ler {jp}: {exc})")
        if w is None:
            w, h, origem = 1024, 688, "SUPOSIÇÃO do plano (rode o c0_1 para medir)"

    print(f"\n[2] Tabela de regimes — resolução {w}×{h}")
    print(f"    origem: {origem}\n")

    linhas = [linha_regime(f"treino {args.image_size}²", args.image_size, args.image_size)]
    if w >= h:
        nw, nh = 512, int(round(h * (512 / w)))
    else:
        nh, nw = 512, int(round(w * (512 / h)))
    nw, nh = alinhar16(nw, nh, para_cima=False)
    linhas.append(linha_regime("eval long_side=512", nw, nh))
    fw, fh = alinhar16(w, h, para_cima=True)
    linhas.append(linha_regime("eval long_side=0", fw, fh))

    fator_treino = args.image_size / min(w, h)
    for l in linhas:
        if l["regime"].startswith("treino"):
            l["escala_vs_treino"] = "1,000×"
        elif "512" in l["regime"]:
            l["escala_vs_treino"] = f"{(512 / max(w, h)) / fator_treino:.3f}×"
        else:
            l["escala_vs_treino"] = f"{1.0 / fator_treino:.3f}×"
    imprimir_tabela(linhas, ["regime", "w", "h", "seq", "mu", "exp_mu", "escala_vs_treino"])

    print("\n  Obs.: no regime long_side=0 o FORWARD roda em tiles de "
          f"{args.image_size}² (seq {seq_len_de(args.image_size, args.image_size)}),")
    print("  mas o `mu` do cronograma sai do seq da imagem INTEIRA (flux.py:624).")
    print("  É esse descasamento que o eixo `sigma_mu_source` do StageConfig permite testar.")

    # ── efeito prático do shift sobre a mediana de sigma ────────────────────
    print("\n[3] Efeito do shift sobre sigma (u = mediana do logit-normal = 0,5)\n")
    efeito = [
        {"regime": l["regime"], "exp_mu": l["exp_mu"],
         "sigma(u=0,5)": f"{sigma_de(0.5, l['mu']):.4f}",
         "sigma(u=0,25)": f"{sigma_de(0.25, l['mu']):.4f}",
         "sigma(u=0,75)": f"{sigma_de(0.75, l['mu']):.4f}"}
        for l in linhas
    ]
    imprimir_tabela(efeito, ["regime", "exp_mu", "sigma(u=0,5)", "sigma(u=0,25)", "sigma(u=0,75)"])
    print("\n  Quanto maior o exp(mu), mais densidade em sigma alto (mais ruído).")
    print("  FATO: os regimes diferem. HIPÓTESE: que isso importa. Só o fatorial do C4 decide.")

    if not cruzou:
        print("\n  AVISO: sem diffusers, a fórmula não foi cruzada com a fonte oficial.")


if __name__ == "__main__":
    main()
