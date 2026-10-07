#!/usr/bin/env python3
"""Compara as duas rodadas identicas e diz se a correcao funcionou."""
import json, os
A = "/host/runs_determinismo/rodada_A/B1_det/seed_0"
B = "/host/runs_determinismo/rodada_B/B1_det/seed_0"
if not (os.path.exists(f"{A}/test_metrics.json") and os.path.exists(f"{B}/test_metrics.json")):
    print("[det] faltou alguma rodada"); raise SystemExit(1)
a, b = json.load(open(f"{A}/test_metrics.json")), json.load(open(f"{B}/test_metrics.json"))
iguais = [k for k in a if abs(a[k] - b[k]) < 1e-12]
difs = {k: (a[k], b[k], a[k] - b[k]) for k in a if abs(a[k] - b[k]) >= 1e-12}
print(f"\n[det] metricas identicas: {len(iguais)}/{len(a)}")
for k, (x, y, d) in difs.items():
    print(f"[det]   DIFERE {k}: {x:.6f} vs {y:.6f}  ({d:+.2e})")
sa, sb = json.load(open(f"{A}/summary.json")), json.load(open(f"{B}/summary.json"))
print(f"[det] melhor epoca: {sa['best_epoch']} vs {sb['best_epoch']}")
print(f"[det] best_metric : {sa['best_metric']:.10f} vs {sb['best_metric']:.10f}")
print("[det] VEREDITO: CORRIGIDO, o caminho geometrico agora e reproduzivel"
      if not difs and sa["best_epoch"] == sb["best_epoch"]
      else "[det] VEREDITO: AINDA NAO REPRODUZ, sobra fonte de nao determinismo")
