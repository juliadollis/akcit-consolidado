#!/usr/bin/env python3
"""LINHA DE BASE DE IDENTIDADE: devolver a entrada como se fosse a resposta.

POR QUE ESTA LINHA E OBRIGATORIA
Todas as metricas de sintese de bokeh (SSIM, LPIPS, DISTS, CLIP-I) comparam a
saida com o alvo. Como a entrada AIF ja e a MESMA CENA do alvo, um modelo que
nao faz nada ja pontua bem. Sem esta referencia nao da para saber se um numero
significa "o modelo aprendeu bokeh" ou "o modelo aprendeu a nao mexer".
Foi assim que descobrimos que a configuracao da DPDD estava errada: o peso
OFICIAL do paper pontuava LPIPS 0,6138 contra 0,2270 da identidade.

COMO E FEITO
Em vez de reimplementar as metricas, montamos um dataset no formato que o
CloudBokehEvaluator ja consome, com `image_best_k` = ENTRADA (identidade) e
`image_real_bokeh` = ALVO. Assim a linha de base passa exatamente pelo mesmo
codigo de metrica dos modelos — incluindo o `lpips+`, a mesma variante.

O pre-processamento (resize_and_pad_image no mesmo long_side) e identico ao da
inferencia, senao a comparacao seria entre resolucoes diferentes.
"""
from __future__ import annotations

import argparse
import io
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "inference"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pyarrow as pa
import pyarrow.parquet as pq
from datasets import load_dataset
from huggingface_hub import HfApi
from PIL import Image

from src.pipelines.bokeh_net import resize_and_pad_image

METADADOS_HF = {
    b"huggingface": b'{"info": {"features": {'
                    b'"file_name_base": {"dtype": "string", "_type": "Value"},'
                    b'"image_real_bokeh": {"_type": "Image"},'
                    b'"image_best_k": {"_type": "Image"},'
                    b'"best_k_value": {"dtype": "float32", "_type": "Value"},'
                    b'"ssim_score": {"dtype": "float32", "_type": "Value"},'
                    b'"image_k01": {"_type": "Image"},'
                    b'"image_k05": {"_type": "Image"},'
                    b'"image_k10": {"_type": "Image"},'
                    b'"image_k15": {"_type": "Image"}'
                    b'}}}'
}
ESQUEMA = pa.schema([
    pa.field("file_name_base", pa.string()),
    pa.field("image_real_bokeh", pa.binary()),
    pa.field("image_best_k", pa.binary()),
    pa.field("best_k_value", pa.float32()),
    pa.field("ssim_score", pa.float32()),
    pa.field("image_k01", pa.binary()),
    pa.field("image_k05", pa.binary()),
    pa.field("image_k10", pa.binary()),
    pa.field("image_k15", pa.binary()),
]).with_metadata(METADADOS_HF)


def para_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def abrir(dado) -> Image.Image:
    if hasattr(dado, "convert"):
        return dado
    if isinstance(dado, bytes):
        return Image.open(io.BytesIO(dado))
    return Image.open(io.BytesIO(dado["bytes"]))


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-entrada", required=True)
    p.add_argument("--padrao", default="data/validation-*.parquet")
    p.add_argument("--saida", required=True, help="repo HF onde gravar (sera PRIVADO)")
    p.add_argument("--nome", default="LINHA-DE-IDENTIDADE")
    p.add_argument("--limite", type=int, default=0)
    p.add_argument("--long-side", type=int, default=512)
    p.add_argument("--lote", type=int, default=20)
    args = p.parse_args()

    token = os.environ.get("HF_TOKEN")
    api = HfApi(token=token)
    api.create_repo(args.saida, repo_type="dataset", private=True, exist_ok=True)

    dados = load_dataset(
        "parquet",
        data_files={"validation": f"hf://datasets/{args.dataset_entrada}/{args.padrao}"},
        token=token,
    )["validation"]
    if args.limite > 0:
        dados = dados.select(range(min(args.limite, len(dados))))
    print(f"[identidade] {len(dados)} imagens de {args.dataset_entrada}", flush=True)

    linhas, nlote = [], 0
    for idx, linha in enumerate(dados):
        nome = linha.get("file_name_base") or f"val_{idx:05d}.png"
        entrada = resize_and_pad_image(abrir(linha["image_focus"]).convert("RGB"), args.long_side)
        alvo = resize_and_pad_image(abrir(linha["image_blur"]).convert("RGB"), args.long_side)
        b_ent, b_alvo = para_bytes(entrada), para_bytes(alvo)
        linhas.append({
            "file_name_base": str(nome),
            "image_real_bokeh": b_alvo,
            "image_best_k": b_ent,       # <- a identidade: a saida E a entrada
            "best_k_value": 0.0,
            "ssim_score": 0.0,
            # As 4 imagens do sweep sao iguais: a identidade nao responde ao K.
            # Com variancia zero o avaliador ja devolve LVCorr = 0, que e a
            # leitura correta ("nenhuma controlabilidade").
            "image_k01": b_ent, "image_k05": b_ent, "image_k10": b_ent, "image_k15": b_ent,
        })
        if len(linhas) == args.lote or (idx + 1) == len(dados):
            nlote += 1
            caminho = f"data/validation_part_{nlote:03d}.parquet"
            local = "temp_identidade.parquet"
            pq.write_table(pa.Table.from_pylist(linhas, schema=ESQUEMA), local)
            api.upload_file(path_or_fileobj=local, path_in_repo=caminho,
                            repo_id=args.saida, repo_type="dataset")
            print(f"[identidade] lote {nlote} ({len(linhas)} linhas) enviado", flush=True)
            os.remove(local)
            linhas = []

    from evaluation import CloudBokehEvaluator
    repo_metricas = os.environ.get("BOKEH_METRICS_REPO", "juliadollis/bokeh-eval-metricas")
    print(f"\n=== METRICAS DA IDENTIDADE -> {repo_metricas} ===", flush=True)
    CloudBokehEvaluator().run_pipeline(
        hf_datasets=args.saida, output_repo_id=repo_metricas,
        model_name=args.nome, hf_split="validation", hf_token=token,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
