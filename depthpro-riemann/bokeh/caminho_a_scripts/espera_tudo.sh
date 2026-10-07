#!/usr/bin/env bash
B=/raid/user_juliadollis/julia_docker
O=$B/caminho_a_rotac/out/full50
CONDS="base_busca Bgrad_al_busca Bgeod_al_busca Bgrad_cru_busca Bgeod_cru_busca base_kfixo Bgrad_al_kfixo Bgeod_al_kfixo"
while :; do
  pronto=1
  for c in $CONDS; do
    n=$(wc -l < "$O/$c/manifest.jsonl" 2>/dev/null || echo 0)
    [ "$n" -lt 50 ] && pronto=0
  done
  if [ "$pronto" = 1 ]; then
    # espera o container fechar de verdade antes de declarar fim
    [ "$(docker inspect -f '{{.State.Status}}' julia_ca_full_kfixo 2>/dev/null)" != "running" ] && break
  fi
  # aborta a espera se ambos os containers morreram sem completar
  s1=$(docker inspect -f '{{.State.Status}}' julia_ca_full_busca 2>/dev/null)
  s2=$(docker inspect -f '{{.State.Status}}' julia_ca_full_kfixo 2>/dev/null)
  if [ "$s1" != "running" ] && [ -n "$s2" ] && [ "$s2" != "running" ]; then
    break
  fi
  sleep 60
done
echo "=== ESTADO FINAL $(date -Is) ==="
for c in $CONDS; do echo "  $c: $(wc -l < "$O/$c/manifest.jsonl" 2>/dev/null || echo 0)/50"; done
