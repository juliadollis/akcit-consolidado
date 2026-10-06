#!/usr/bin/env bash
# Clona o repositório oficial dos autores (Genfocus), que provê
# `Genfocus.pipeline.flux.transformer_forward` — usado por genfocus_train/backbone.py.
#
# Coloca o repo em third_party/Genfocus e imprime o PYTHONPATH a exportar.
set -euo pipefail

# Raiz do projeto (um nível acima de scripts/)
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="${ROOT_DIR}/third_party/Genfocus"

# Repo oficial do paper (confirmado). O pacote python `Genfocus` fica DENTRO
# deste repo (DEST/Genfocus/pipeline/flux.py), por isso o PYTHONPATH abaixo
# aponta para DEST (a raiz do clone), e não para third_party/.
GENFOCUS_REPO="${GENFOCUS_REPO:-https://github.com/rayray9999/Genfocus.git}"

mkdir -p "${ROOT_DIR}/third_party"

if [ -d "${DEST}/.git" ]; then
  echo "[setup_genfocus] Repo já existe em ${DEST}; fazendo git pull."
  git -C "${DEST}" pull --ff-only
else
  echo "[setup_genfocus] Clonando ${GENFOCUS_REPO} -> ${DEST}"
  git clone "${GENFOCUS_REPO}" "${DEST}"
fi

# Verifica que o pacote interno está onde esperamos.
if [ ! -f "${DEST}/Genfocus/pipeline/flux.py" ]; then
  echo "[setup_genfocus] ERRO: não encontrei ${DEST}/Genfocus/pipeline/flux.py"
  echo "                 A estrutura do repo mudou? Confira manualmente."
  exit 1
fi

echo ""
echo "[setup_genfocus] Pronto. Adicione ao PYTHONPATH antes de treinar:"
echo "    export PYTHONPATH=\"${DEST}:\$PYTHONPATH\""
echo ""
echo "Teste:  PYTHONPATH=\"${DEST}\" python -c 'from Genfocus.pipeline.flux import transformer_forward; print(\"ok\")'"
