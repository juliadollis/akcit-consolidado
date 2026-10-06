#!/usr/bin/env python3
"""Linha `Input` da Tabela 2 do paper: a entrada borrada, sem modelo nenhum.

POR QUE ELA IMPORTA
O paper coloca essa linha em primeiro lugar na Tab. 2, e com razao: as cinco
metricas comparam a saida com o alvo NITIDO, e a entrada borrada ja e a mesma
cena. Sem esse piso nao da para saber se um numero significa "o modelo
desborrou" ou "o modelo mexeu pouco". E o mesmo papel da nossa linha de
identidade no bokeh.

COMO E FEITO
Montamos um dataset no MESMO formato que a inferencia grava
(`file_name_base`, `image_generated`, `image_focus`), com `image_generated`
sendo a propria `image_blur` da origem. Assim a linha passa exatamente pelo
mesmo avaliador dos modelos, com as mesmas variantes de metrica (lpips+,
dists, clipiqa+, maniqa-kadid, musiq).
"""
import io, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "inference"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from datasets import Dataset, Features, Image as HFImage, Value, load_dataset

FONTES = {
    "akcit-pixel/RealDOF": "juliadollis/tab2-infer-realdof-input",
    "akcit-pixel/DDPD":    "juliadollis/tab2-infer-ddpd-input",
}
METRICS_REPO = "juliadollis/tab2-metricas"


def main():
    token = os.environ["HF_TOKEN"]
    from src.pipelines.deblur_net import run_hf_deblur_evaluation

    for entrada, saida in FONTES.items():
        print(f"\n=== Input | {entrada} -> {saida} ===", flush=True)
        d = load_dataset(entrada, split="validation", token=token)
        print(f"  {len(d)} cenas, colunas {d.column_names}", flush=True)

        linhas = []
        for r in d:
            linhas.append({
                "file_name_base": str(r["file_name_base"]),
                # a "predicao" e a propria entrada borrada: e isso que a linha mede
                "image_generated": r["image_blur"],
                "image_focus": r["image_focus"],
            })
        feats = Features({"file_name_base": Value("string"),
                          "image_generated": HFImage(), "image_focus": HFImage()})
        ds = Dataset.from_list(linhas, features=feats)
        ds.push_to_hub(saida, split="validation", token=token, private=True)
        print(f"  publicado em {saida}", flush=True)

        run_hf_deblur_evaluation(
            generated_repo_id=saida,
            evaluation_repo_id=METRICS_REPO,
            model_name="Input (entrada borrada, sem modelo)",
            experiment_name=None,
            hf_token=token,
        )
    print("\n=== FIM linha Input ===", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
