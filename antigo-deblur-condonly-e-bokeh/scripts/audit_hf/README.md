# Ferramentas de auditoria dos dfs no Hugging Face

Escritas em 2026-09-03 para a auditoria das rotas b e c
(ver `../../AUDITORIA_DADOS_ROTAS_BC.md`). Todas rodam **do Mac**, em stdlib pura:
esta máquina não tem `datasets`, `pyarrow`, `pandas`, `numpy` nem `PIL`.

Elas existem porque o `datasets-server` está **quebrado** para esses repos:

```
is-valid  rota b : viewer=true  filter=true  statistics=false
is-valid  rota c : viewer=false filter=false statistics=false
/statistics rota b -> HTTP 500 "Permission denied (os error 13)"
/filter     rota b -> "Unexpected error."
/info,/parquet rota c -> vazio, config-info/config-parquet failed
kfix (privado) -> "Private datasets are only supported for PRO users"
```

| arquivo | o que faz | limite conhecido |
|---|---|---|
| `pq.py` | footer Parquet (thrift compact) por HTTP Range; `analyze(url)` dá `num_rows`, schema, `null_count` e min/max por row-group | leitura remota via `curl`; exige `HF_TOKEN` no ambiente |
| `cols.py` | decodifica um column chunk (snappy + RLE/bitpack + dicionário) | **só colunas escalares.** Quebra em coluna de imagem (`depth.bytes` etc.): `struct.error` em `plain()`. Para imagem use `/rows` e baixe o asset |
| `png16.py` | decodificador PNG em stdlib: cinza 8/16 bits e RGB 8 bits, filtros 0 a 4, sem interlace | sem interlace, sem paleta, sem alpha |
| `test_png16.py` | round-trip dos 5 filtros em 3 formatos | roda offline, sem rede |
| `teste_convencao_depth.py` | o teste que decide se `depth` é profundidade métrica ou disparidade, comparando `coc_p99_px` gravado na kfix contra o recalculado dos pixels sob cada hipótese | precisa de `/tmp/rowsb*.json` (saída de `/rows`) e de `kfix.parquet` local |

## Como reproduzir a medição

```bash
set -a; . .env; set +a
curl -s -H "Authorization: Bearer $HF_TOKEN" \
  "https://datasets-server.huggingface.co/rows?dataset=AKCITPixel3%2FBKXcuVXCmeRvN&config=default&split=train&offset=0&length=12" \
  -o /tmp/rowsb.json
python3 scripts/audit_hf/teste_convencao_depth.py /tmp/rowsb.json
```

`/rows` devolve URL de asset para as colunas de imagem, e o asset do `depth` chega
como PNG de 16 bits intacto (conferido: `bitdepth=16`, `min=0`, `max=65535`).
O `foreground_mask` chega convertido para JPEG pelo servidor, então não serve para
medida numérica.

## Autoteste com resposta conhecida

Antes de confiar no decodificador, rode-o sobre `defocus_map`: o time já mediu que
essa coluna foi gravada normalizada por imagem, logo `max == 65535` em toda amostra.
Se o decodificador não reproduzir isso, ele está errado.
