#!/usr/bin/env python3
"""C0-3 — conta os módulos LoRA injetados. É o teste que prova o C1.

POR QUE ISTO EXISTE
-------------------
PEFT casa `target_modules` por `key == target or key.endswith("." + target)`.
A projeção final do `FluxTransformer2DModel` chama-se literalmente `proj_out`
no TOPO do módulo, então a string solta `"proj_out"` na `LORA_TARGET_MODULES`
capturava, além dos 38 `single_transformer_blocks.N.proj_out`, também
`transformer.proj_out` — que `transformer_forward` executa FORA de qualquer
`specify_lora`:

    image_hidden_states = self.norm_out(all_hidden_states[txt_n], tembs[txt_n])
    output = self.proj_out(image_hidden_states)      # sem specify_lora

Sem `specify_lora`, o `scaling` do adapter fica no valor natural (alpha/r = 1) e
a LoRA age incondicionalmente, inclusive com `main_adapter=None`. Ou seja: a
variante "cond-only" não era cond-only.

Contagem esperada (bate com o header do bokehNet.safetensors oficial, 686
tensores / 343 módulos):

    19 duplos × (to_q, to_k, to_v, to_out.0, norm1.linear, ff.net.2) = 114
    38 single × (to_q, to_k, to_v, norm.linear, proj_mlp, proj_out)  = 228
    x_embedder                                                       =   1
                                                              total  = 343

Antes da correção: 344 (o extra é `proj_out` de topo).

CUSTO
-----
Precisa dos PESOS do FLUX.1-dev. Por padrão este script usa `--so-estrutura`,
que instancia o transformer a partir da CONFIG (sem baixar os ~24 GB de pesos)
usando `init_empty_weights` do accelerate — suficiente para contar módulos, que
é uma propriedade da topologia. Com `--pesos-reais` ele carrega o modelo de
verdade (aí sim precisa do download e, de preferência, de GPU).

USO
---
    python3 scripts/c0_3_contagem_lora.py                 # só estrutura, sem pesos
    python3 scripts/c0_3_contagem_lora.py --pesos-reais   # caminho completo
    python3 scripts/c0_3_contagem_lora.py --stage bokeh   # rank 64

GPU: não (modo estrutura) / sim, recomendado (modo --pesos-reais).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from _comum import adicionar_raiz_ao_path, imprimir_tabela

adicionar_raiz_ao_path()

ESPERADO_OFICIAL = 343


def contar(transformer) -> tuple[list[str], list[str]]:
    """Devolve (todos os módulos LoRA, os que estão no TOPO do transformer)."""
    from peft.tuners.lora import LoraLayer

    todos = [n for n, m in transformer.named_modules() if isinstance(m, LoraLayer)]
    topo = [n for n in todos if "." not in n]
    return todos, topo


def resumo_por_categoria(modulos: list[str]) -> list[dict]:
    """Agrupa por sufixo, para ver de onde vem cada bloco da contagem."""
    from collections import Counter

    cont = Counter()
    for n in modulos:
        if n.startswith("transformer_blocks."):
            grupo = "duplo"
        elif n.startswith("single_transformer_blocks."):
            grupo = "single"
        else:
            grupo = "TOPO"
        # sufixo = tudo depois do índice do bloco
        partes = n.split(".")
        sufixo = ".".join(partes[2:]) if grupo != "TOPO" else n
        cont[(grupo, sufixo)] += 1
    return [
        {"grupo": g, "modulo": s, "n": c}
        for (g, s), c in sorted(cont.items(), key=lambda kv: (kv[0][0], kv[0][1]))
    ]


def construir_estrutura(model_id: str, subfolder: str):
    """Instancia o FluxTransformer2DModel a partir da config, sem baixar pesos."""
    from accelerate import init_empty_weights
    from diffusers import FluxTransformer2DModel

    cfg = FluxTransformer2DModel.load_config(model_id, subfolder=subfolder)
    with init_empty_weights():
        modelo = FluxTransformer2DModel.from_config(cfg)
    return modelo


def main() -> None:
    p = argparse.ArgumentParser(
        description="C0-3: conta os módulos LoRA injetados e prova o C1.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--stage", default="deblur", choices=["deblur", "bokeh"])
    p.add_argument("--pesos-reais", action="store_true",
                   help="carrega o FLUX inteiro via create_backbone (lento, precisa dos pesos)")
    p.add_argument("--model-id", default="black-forest-labs/FLUX.1-dev")
    p.add_argument("--esperado", type=int, default=ESPERADO_OFICIAL,
                   help="contagem do checkpoint oficial")
    p.add_argument("--out", default="outputs/c0_3_contagem_lora.json")
    args = p.parse_args()

    print("=" * 72)
    print(f"C0-3 — contagem de módulos LoRA (stage={args.stage})")
    print("=" * 72)

    from genfocus_train.backbone import LORA_TARGET_MODULES

    tipo_alvo = "regex (str)" if isinstance(LORA_TARGET_MODULES, str) else "lista de sufixos"
    print(f"\nLORA_TARGET_MODULES: {tipo_alvo}")
    print(f"  {LORA_TARGET_MODULES!r}")

    if args.pesos_reais:
        print("\nCarregando FLUX com pesos reais via create_backbone()...")
        import torch
        from genfocus_train.backbone import create_backbone
        from genfocus_train.config import TrainConfig

        cfg = TrainConfig()
        cfg.model.pretrained_model_name_or_path = args.model_id
        # a trava do C1 pode abortar aqui — é exatamente o comportamento desejado;
        # capturamos para conseguir REPORTAR a contagem em vez de só estourar.
        try:
            backbone = create_backbone(cfg.model, stage=args.stage)
        except TypeError:
            # assinatura nova aceita stage_config
            from genfocus_train.config import stage_config
            backbone = create_backbone(cfg.model, stage=args.stage,
                                       stage_config=stage_config(cfg, args.stage))
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
        try:
            backbone.load(dtype=dtype, device=device)
        except RuntimeError as exc:
            print(f"\n  A trava do C1 disparou (isto é o comportamento correto):\n  {exc}")
            raise SystemExit(2) from exc
        transformer = backbone.transformer
    else:
        print("\nInstanciando SÓ a estrutura (init_empty_weights, sem baixar pesos)...")
        try:
            transformer = construir_estrutura(args.model_id, "transformer")
        except Exception as exc:  # noqa: BLE001
            raise SystemExit(
                f"ERRO ao montar a estrutura do transformer: {exc}\n"
                "Isto precisa de `diffusers` + `accelerate` instalados e de acesso à\n"
                "config do modelo (só o JSON, não os pesos). Se o modelo for gated,\n"
                "faça login: `huggingface-cli login`. Alternativa: --pesos-reais."
            ) from exc

        from peft import LoraConfig
        from genfocus_train.backbone import ADAPTER_NAME
        from genfocus_train.config import ModelConfig, lora_rank_for, TrainConfig

        cfg = TrainConfig()
        rank = lora_rank_for(cfg, args.stage)
        transformer.requires_grad_(False)
        transformer.add_adapter(
            LoraConfig(r=rank, lora_alpha=rank, init_lora_weights="gaussian",
                       target_modules=LORA_TARGET_MODULES, lora_dropout=0.0, bias="none"),
            adapter_name=ADAPTER_NAME,
        )

    # ── contagem ────────────────────────────────────────────────────────────
    n_duplos = len([n for n, _ in transformer.named_modules()
                    if n.startswith("transformer_blocks.") and n.count(".") == 1])
    n_single = len([n for n, _ in transformer.named_modules()
                    if n.startswith("single_transformer_blocks.") and n.count(".") == 1])

    todos, topo = contar(transformer)

    print(f"\nBlocos do transformer: {n_duplos} duplos, {n_single} single")
    print(f"Módulos LoRA injetados: {len(todos)}")
    print(f"Módulos LoRA de TOPO:   {topo}")

    print("\nDe onde vem cada um:")
    imprimir_tabela(resumo_por_categoria(todos), ["grupo", "modulo", "n"])

    ok = len(todos) == args.esperado
    print("\n" + "=" * 72)
    if ok:
        print(f"OK — {len(todos)} módulos, igual ao checkpoint oficial ({args.esperado}).")
        if topo != ["x_embedder"]:
            print(f"  ATENÇÃO: os módulos de topo deveriam ser só ['x_embedder'], são {topo}.")
    else:
        print(f"DIVERGE — {len(todos)} módulos contra {args.esperado} do oficial.")
        extras = [n for n in topo if n != "x_embedder"]
        if extras:
            print(f"  Módulos de topo indevidos: {extras}")
            print("  É o C1: restrinja LORA_TARGET_MODULES a um regex ancorado nos blocos.")
    print("=" * 72)

    destino = Path(args.out)
    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_text(json.dumps({
        "stage": args.stage, "modo": "pesos_reais" if args.pesos_reais else "estrutura",
        "blocos_duplos": n_duplos, "blocos_single": n_single,
        "n_modulos_lora": len(todos), "modulos_topo": topo,
        "esperado_oficial": args.esperado, "bate": ok,
        "por_categoria": resumo_por_categoria(todos),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[c0-3] JSON salvo em {destino}")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
