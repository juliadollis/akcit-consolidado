#!/usr/bin/env python3
"""Constrói o JSON de distribuição de K que a rota A consome.

    python3 scripts/build_k_distribution.py \\
        --release-dir output/b_full \\
        --release-dir output/c_full \\
        --output-json manifests/k_distribution_bc.json

Lê `manifest.jsonl` de um ou mais diretórios de release das rotas B e C, converte cada
`k_value` para a grandeza **livre de resolução** `k_per_long_side = k_value/max(H,W)`, e
emite a função quantil empírica com a proveniência de cada manifesto lido.

## Por que este script existe separado do run da rota A

Porque a distribuição é uma **entrada auditável** do run, não um efeito colateral dele.
Se a rota A lesse os manifestos direto, a distribuição usada num release ficaria
implícita: ninguém conseguiria dizer depois de quais linhas ela saiu, e regerar a rota A
com a rota B regerada produziria outra distribuição sem que nada denunciasse. Com o JSON
no meio, o arquivo tem sha256, entra na proveniência de cada amostra e pode ser
comparado entre releases.

## O que ele recusa a fazer

**Não inventa resolução.** Uma linha de manifesto sem `image_h`/`image_w` é excluída e
contada, nunca dividida por um lado longo assumido — K sem a resolução em que foi medido
é o defeito A5 (`CONTRATO.md:47-50`).

**Não escreve distribuição degenerada em silêncio.** Ele escreve o arquivo com o campo
`degenerate_reason` preenchido e **termina com código 2**, imprimindo o motivo. Medido no
dado publicado hoje: a rota B tem `k = 50,0` em 11.635/11.635 (`ACHADOS.md:14`) e a rota
C tem 47,0% no teto exato de 300 (`ACHADOS.md:19`) — *"amostrar dessa distribuição hoje
é amostrar de duas constantes"*. A rota A não pode rodar antes de B e C regeradas, e isso
é dependência de ordem, não preferência.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from control.contract import CONTROL_VERSION                       # noqa: E402
from dataio import iter_manifest                                   # noqa: E402
from sources.k_distribution import (                               # noqa: E402
    DEFAULT_QUANTILE_KNOTS, DEFAULT_WIDEN_FRACTION, build_distribution,
    scan_manifest_rows,
)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for bloco in iter(lambda: fh.read(1 << 20), b""):
            h.update(bloco)
    return h.hexdigest()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Distribuição de K das rotas B e C, para o sorteio da rota A.",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--release-dir", action="append", required=True, default=None,
                   help="diretório de release com manifest.jsonl. Repetível: passe o da "
                        "rota B e o da rota C.")
    p.add_argument("--output-json", required=True)
    p.add_argument("--quantile-knots", type=int, default=DEFAULT_QUANTILE_KNOTS,
                   help=f"nós da função quantil (default {DEFAULT_QUANTILE_KNOTS} = "
                        "passo de 1%%, que resolve p01 e p99)")
    p.add_argument("--control-version", default=CONTROL_VERSION,
                   help="contrato esperado nas linhas lidas. Manifesto com outro "
                        "control_version é ERRO: duas convenções de normalização na "
                        "mesma distribuição foi o modo de falha do kfix.")
    return p


def main() -> int:
    args = build_parser().parse_args()

    scans = []
    versoes: set[str] = set()
    for diretorio in args.release_dir:
        raiz = Path(diretorio)
        manifesto = raiz / "manifest.jsonl"
        if not manifesto.is_file():
            raise SystemExit(f"manifesto não encontrado: {manifesto}")
        linhas = list(iter_manifest(raiz))
        for linha in linhas:
            if linha.get("control_version"):
                versoes.add(str(linha["control_version"]))
        scan = scan_manifest_rows(linhas, release_dir=str(raiz))
        scans.append(scan)
        print(f"[k-dist] {raiz}: {scan.lines_used}/{scan.lines_total} linhas usadas")
        for rota, info in sorted(scan.routes.items()):
            print(f"          rota {rota}: vistas={info['seen']} "
                  f"usadas={info['total']} censuradas={info['censored']}")
        print(f"          excluídas: {json.dumps(scan.to_dict()['excluded'])}")

    if len(versoes) > 1:
        raise SystemExit(
            f"control_version divergente entre os manifestos: {sorted(versoes)}. "
            "Misturar convenções numa distribuição só é o modo de falha do kfix "
            "(ACHADOS.md:192) — regere o lote antigo antes.")
    if versoes and args.control_version not in versoes:
        raise SystemExit(
            f"os manifestos afirmam {sorted(versoes)} e o esperado é "
            f"{args.control_version!r}. Não há conversão automática entre contratos.")

    distribuicao = build_distribution(
        scans, control_version=(next(iter(versoes)) if versoes else args.control_version),
        knots=args.quantile_knots,
        created_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"))

    caminho = distribuicao.save(args.output_json)
    print(distribuicao.summary(DEFAULT_WIDEN_FRACTION))
    print(f"[k-dist] escrito: {caminho}  sha256 {_sha256(caminho)}")

    motivo = distribuicao.degenerate_reason()
    if motivo:
        print("\n[k-dist] ARQUIVO ESCRITO, MAS A DISTRIBUIÇÃO É DEGENERADA:")
        print(f"         {motivo}")
        print("         A rota A não deve rodar com ela. Ver PLANO_EXECUCAO.md: B e C "
              "primeiro,\n         depois A.")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
