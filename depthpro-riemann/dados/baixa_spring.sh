#!/usr/bin/env bash
# Baixa do DaRUS os 3 pacotes do Spring que o GUIA_SPRING.md pede (secao 5.1).
# So o split de TREINO, camera esquerda: ~24 GB em vez do dataset inteiro (~280 GB).
# Nao precisamos de fluxo optico, disp2 nem camera direita.
set -uo pipefail
DEST=/raid/user_juliadollis/julia_docker/data/spring_zips
RAIZ=/raid/user_juliadollis/julia_docker/data/spring/train
mkdir -p "$DEST" "$RAIZ"
API=https://darus.uni-stuttgart.de/api/access/datafile

baixa () {
  local id="$1" nome="$2"
  if [ -s "$DEST/$nome" ]; then echo "[spring] $nome ja existe, pulando"; return 0; fi
  echo "[spring] baixando $nome (id=$id) $(date -u +%H:%M:%SZ)"
  curl -fL --retry 5 --retry-delay 10 -C - -o "$DEST/$nome" "$API/$id"
  echo "[spring] $nome rc=$? $(du -h "$DEST/$nome" | cut -f1) $(date -u +%H:%M:%SZ)"
}

baixa 198954 train_cam_data.zip
baixa 199097 train_frame_left.zip
baixa 198961 train_disp1_left.zip

echo "[spring] descompactando em $RAIZ"
for z in train_cam_data train_frame_left train_disp1_left; do
  echo "[spring] unzip $z"
  unzip -q -o "$DEST/$z.zip" -d "$RAIZ" && echo "[spring] $z OK" || echo "[spring] $z FALHOU"
done
echo "[spring] sequencias encontradas: $(ls "$RAIZ" | wc -l)"
ls "$RAIZ" | head -5
echo "[spring] FIM $(date -u +%H:%M:%SZ)"
