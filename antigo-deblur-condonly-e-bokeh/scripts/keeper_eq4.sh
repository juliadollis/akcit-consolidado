#!/usr/bin/env bash
# KEEPER da campanha Eq. 4: a GPU nunca fica ociosa enquanto houver fila.
#
# Igual em espirito ao keeper_gpu.sh, com uma diferenca que importa: chama o
# fila_gpu_eq4.sh, que injeta o PYTHONPATH do BiRefNet. Sem isso o plano de foco
# cai no pixel central e a rodada nao e da Eq. 4, e o keeper antigo chama o
# fila_gpu.sh, que nao injeta.
#
# Como injetar trabalho: EDITE o arquivo de fila. O keeper le o arquivo a cada
# volta, entao item novo entra sem reiniciar nada.
#
# Como parar:  touch /raid/user_juliadollis/julia_docker/PARAR_KEEPER_<gpu>
# A flag e vista ENTRE voltas e ENTRE itens da fila; o item que ja estiver
# rodando termina antes. Para devolver a placa na hora, alem da flag e preciso
# parar o container (`docker stop julia_eq4_<gpu>_<nome>`), o que so faz sentido
# se alguem estiver esperando a GPU.
#
# Uso: setsid nohup bash scripts/keeper_eq4.sh 2 /caminho/fila.txt &
set -uo pipefail
GPU="${1:?uso: keeper_eq4.sh <gpu> <arquivo_de_fila>}"
FILA="${2:?uso: keeper_eq4.sh <gpu> <arquivo_de_fila>}"
B=/raid/user_juliadollis/julia_docker
PARAR="$B/PARAR_KEEPER_${GPU}"
volta=0

echo "[keeper-eq4 gpu$GPU] iniciado. Para parar: touch $PARAR"
while [ ! -f "$PARAR" ]; do
  volta=$((volta + 1))
  echo "[keeper-eq4 gpu$GPU] ===== volta $volta - $(date -u +%FT%H:%M:%SZ) ====="
  bash "$(dirname "$0")/fila_gpu_eq4.sh" "$GPU" "$FILA"
  [ -f "$PARAR" ] && break
  # Volta numa fila ja concluida custa segundos: o run_3models.py pula o que ja
  # esta na tabela de metricas. Por isso repetir e barato, e a espera e curta.
  echo "[keeper-eq4 gpu$GPU] fila esgotada; nova volta em 60s"
  sleep 60
done
echo "[keeper-eq4 gpu$GPU] PARADO por $PARAR - $(date -u +%FT%H:%M:%SZ)"
