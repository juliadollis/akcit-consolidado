"""Helpers de ambiente para acesso aos datasets (Hugging Face).

`HF_TOKEN` é lido de uma variável de ambiente ou de um arquivo `.env` na raiz
do projeto (veja `.env.example`). Necessário para `datasets.load_dataset` nos
repositórios privados `akcit-pixel/*`.
"""

from __future__ import annotations

import os

from dotenv import load_dotenv


def load_local_env() -> None:
    load_dotenv(override=False)


def get_required_env(name: str) -> str:
    load_local_env()
    value = os.getenv(name)
    if value:
        return value
    raise RuntimeError(f"Variável de ambiente obrigatória ausente: {name}")
