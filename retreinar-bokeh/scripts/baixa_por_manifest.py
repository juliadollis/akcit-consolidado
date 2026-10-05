#!/usr/bin/env python3
"""Baixa um release derivando os caminhos do MANIFEST, sem listar a arvore.

`snapshot_download` primeiro LISTA o repo inteiro e so depois aplica
`ignore_patterns`. Com 55 mil arquivos essa listagem estoura em read timeout e
o download nunca comeca — foi o que travou a rota A (210.809 arquivos) e a
rota B. O manifest ja diz quais amostras existem, e o layout e deterministico,
entao da para pedir arquivo por arquivo e pular `mask/` de verdade.

Uso: baixa_por_manifest.py <repo> <dest> <papeis...>
  papeis: aif (generated/<cena>/<sid>_aif.jpg), bokeh, depth, meta
"""
import json, os, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import EntryNotFoundError

repo, dest = sys.argv[1], sys.argv[2]
papeis = sys.argv[3:] or ["depth", "meta"]

linhas = [json.loads(l) for l in open(f"{dest}/manifest.jsonl", encoding="utf-8") if l.strip()]
print(f"{len(linhas)} amostras no manifest; papeis={papeis}", flush=True)

alvos = []
for reg in linhas:
    sid, cena = str(reg["sample_id"]), str(reg["scene_id"])
    for p in papeis:
        if p == "depth":  alvos.append(f"depth/{cena}/{sid}.png")
        elif p == "meta": alvos.append(f"meta/{cena}/{sid}.json")
        else:             alvos.append(f"generated/{cena}/{sid}_{p}.jpg")

faltam = [a for a in alvos if not os.path.isfile(os.path.join(dest, a))]
print(f"{len(alvos)} arquivos, {len(faltam)} faltando", flush=True)

t0, feitos, erros = time.time(), 0, []

def puxa(rel):
    """Retenta ERRO DE REDE; nao retenta arquivo ausente nem erro de codigo.

    A rota C morreu com 3 de 30.852 arquivos faltando, num
    `RemoteProtocolError: Server disconnected` — uma hora de download perdida
    por um solucco de rede no fim. Retentar e barato; nao retentar custa o run.
    A distincao importa: `EntryNotFoundError` significa que o arquivo NAO EXISTE
    no repo, e insistir nele e so gastar cota.
    """
    for tentativa in range(6):
        try:
            hf_hub_download(repo, rel, repo_type="dataset", local_dir=dest)
            return None
        except EntryNotFoundError:
            return rel
        except Exception as exc:               # httpx/httpcore nao herdam de OSError
            nome = type(exc).__name__
            transitorio = isinstance(exc, OSError) or any(
                m in nome for m in ("Protocol", "Timeout", "Connect", "HTTPError")
            )
            if not transitorio:
                raise                          # erro de codigo: falha alto
            if tentativa == 5:
                print(f"  DESISTI de {rel}: {nome}: {exc}", flush=True)
                return rel
            time.sleep(2 ** tentativa)
    return rel

# 6 threads: a cota do Hub e 2500 requisicoes por 5 min (8,3/s). Sete
# downloaders paralelos derrubaram seis deles na rota A.
with ThreadPoolExecutor(max_workers=6) as ex:
    futs = {ex.submit(puxa, a): a for a in faltam}
    for f in as_completed(futs):
        r = f.result()
        if r: erros.append(r)
        feitos += 1
        if feitos % 2000 == 0:
            dt = time.time() - t0
            print(f"  {feitos}/{len(faltam)}  {feitos/max(dt,1):.1f} arq/s  "
                  f"faltam ~{(len(faltam)-feitos)/max(feitos/max(dt,1),0.1)/60:.0f} min",
                  flush=True)
print(f"PRONTO: {feitos} baixados em {(time.time()-t0)/60:.1f} min; "
      f"{len(erros)} ausentes no repo", flush=True)
for e in erros[:10]: print("  ausente:", e)
