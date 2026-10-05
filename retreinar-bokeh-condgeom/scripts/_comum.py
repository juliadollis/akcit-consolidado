"""Helpers compartilhados pelos scripts de `retreinar-deblur/scripts/`.

Fica aqui (e não em `genfocus_train/`) porque é utilitário de linha de comando,
não parte do pacote de treino.
"""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path

# =============================================================================
# sys.path
# =============================================================================

RAIZ = Path(__file__).resolve().parent.parent          # retreinar-deblur/
GENFOCUS_OFICIAL = RAIZ / "third_party" / "Genfocus"   # clone de referência


def adicionar_raiz_ao_path() -> None:
    """Põe `retreinar-deblur/` e o clone oficial do Genfocus no `sys.path`.

    O `Genfocus/pipeline/flux.py` do repo oficial é importado como
    `Genfocus.pipeline.flux`; ele NÃO é instalável, então o diretório que o
    contém precisa estar no path.
    """
    for p in (RAIZ, GENFOCUS_OFICIAL):
        s = str(p)
        if s not in sys.path:
            sys.path.insert(0, s)


# =============================================================================
# Token do Hugging Face
# =============================================================================

def token_hf(obrigatorio: bool = True) -> str | None:
    """Devolve o HF_TOKEN do ambiente ou do `.env`. NUNCA imprime o valor."""
    adicionar_raiz_ao_path()
    try:
        from genfocus_train.env import load_local_env
        load_local_env()
    except Exception:
        # dotenv ausente não é fatal: a variável pode já estar no ambiente.
        pass
    tok = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
    if tok:
        return tok
    if obrigatorio:
        raise SystemExit(
            "ERRO: HF_TOKEN ausente. Defina no ambiente ou em "
            f"{RAIZ / '.env'} (veja .env.example). O valor nunca é impresso."
        )
    return None


# =============================================================================
# Aritmética do FLUX — a mesma conta que o plano usa (C4)
# =============================================================================

# Config do scheduler do FLUX.1-dev. Estes quatro números saem de
# `scheduler/scheduler_config.json` do black-forest-labs/FLUX.1-dev e são os
# mesmos que a inferência oficial usa via `calculate_shift`.
BASE_IMAGE_SEQ_LEN = 256
MAX_IMAGE_SEQ_LEN = 4096
BASE_SHIFT = 0.5
MAX_SHIFT = 1.15


def calculate_shift_local(
    seq_len: float,
    base_seq_len: int = BASE_IMAGE_SEQ_LEN,
    max_seq_len: int = MAX_IMAGE_SEQ_LEN,
    base_shift: float = BASE_SHIFT,
    max_shift: float = MAX_SHIFT,
) -> float:
    """Reimplementação da reta do `calculate_shift` do diffusers.

    mu = m * seq_len + b, com m = (max_shift - base_shift)/(max_seq - base_seq)
    e b = base_shift - base_seq * m. Reimplementada para os scripts rodarem sem
    diffusers instalado; `verificar_mu.py` cruza esta versão com a oficial.
    """
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - base_seq_len * m
    return seq_len * m + b


def alinhar16(w: int, h: int, para_cima: bool = False) -> tuple[int, int]:
    """Alinha (w, h) a múltiplos de 16 — a restrição do VAE 8× + _pack_latents 2×.

    `para_cima=True` reproduz o caminho `long_side=0` do `resize_and_pad_image`
    oficial, que arredonda PARA CIMA; `False` reproduz o caminho com
    `long_side>0`, que trunca e corta no centro.
    """
    if para_cima:
        return ((w + 15) // 16) * 16, ((h + 15) // 16) * 16
    return max((w // 16) * 16, 16), max((h // 16) * 16, 16)


def seq_len_de(w: int, h: int) -> int:
    """Nº de tokens FLUX de uma imagem w×h (já alinhada a 16)."""
    return (w // 16) * (h // 16)


def linha_regime(nome: str, w: int, h: int) -> dict:
    """Uma linha da tabela de regimes do plano: seq, mu, exp(mu)."""
    seq = seq_len_de(w, h)
    mu = calculate_shift_local(seq)
    return {
        "regime": nome, "w": w, "h": h,
        "seq": seq, "mu": round(mu, 4), "exp_mu": round(math.exp(mu), 3),
    }


def imprimir_tabela(linhas: list[dict], colunas: list[str] | None = None) -> None:
    """Tabela de texto simples, alinhada. Sem dependência externa."""
    if not linhas:
        print("  (vazio)")
        return
    colunas = colunas or list(linhas[0].keys())
    larg = {c: max(len(str(c)), *(len(str(l.get(c, ""))) for l in linhas)) for c in colunas}
    print("  " + "  ".join(str(c).ljust(larg[c]) for c in colunas))
    print("  " + "  ".join("-" * larg[c] for c in colunas))
    for l in linhas:
        print("  " + "  ".join(str(l.get(c, "")).ljust(larg[c]) for c in colunas))
