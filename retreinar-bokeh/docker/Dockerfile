# =============================================================================
# Imagem do treino do GenRefocus para a dgx-H100-01 (DOCKER, SEM SLURM).
#
# POR QUE ESTA IMAGEM EXISTE
# O `scripts/docker_bootstrap.sh` antigo fazia:
#     pip install transformers diffusers peft accelerate ...
# SEM FIXAR VERSAO. Foi exatamente uma combinacao errada de versoes que custou 4
# jobs falhos na H100-02 (ver secao 4 do historico-ultimo.md):
#   - peft 0.13.2 quebra: diffusers 0.37.1 exige peft>=0.17;
#   - peft novo demais + transformers velho quebra no EncoderDecoderCache.
# Aqui as versoes ficam GRAVADAS na imagem, entao o problema nao volta.
#
# A BASE
# `huggingface/transformers-pytorch-gpu` e a MESMA origem do .sif que roda hoje
# na H100-02 (o .sif veio de `singularity pull docker://huggingface/...`). Ela ja
# traz Python 3.10, torch 2.9.0+cu126 e transformers 5.0.0.dev0 — este ultimo e
# um build de DESENVOLVIMENTO que NAO existe no PyPI, por isso ele vem da base e
# NAO e reinstalado aqui.
#
# O que instalamos por cima e exatamente o que na H100-02 vivia no ~/.local.
#
# ATENCAO: `:latest` e uma tag movel. Depois do build, RODE a verificacao de
# versoes (docker/verify_env.py) e compare com o manifesto abaixo. Se a base
# tiver mudado o transformers, e melhor descobrir agora do que 10 dias depois.
#
# MANIFESTO ALVO (extraido da H100-02 em 2026-08-13, ambiente comprovado):
#   Python 3.10.12 | torch 2.9.0+cu126 | torchvision 0.24.0+cu126
#   transformers 5.0.0.dev0 | diffusers 0.37.1 | peft 0.18.1
#   accelerate 1.11.0 | datasets 4.3.0 | huggingface-hub 1.0.0rc6
#   safetensors 0.6.2 | numpy 1.26.4 | Pillow 12.0.0
#   wandb 0.28.1 | python-dotenv 1.2.2 | pyyaml 6.0.3
#
# BUILD (na H100-01, a partir da raiz do projeto):
#   docker build -f docker/Dockerfile -t julia-genrefocus:1.0 docker/
# =============================================================================
FROM huggingface/transformers-pytorch-gpu:latest

# git: a base nao traz, e o setup_genfocus.sh precisa clonar o repo dos autores.
RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

# NAO reinstalar torch/torchvision (a base ja traz o par compativel com a CUDA
# da imagem) nem transformers (o 5.0.0.dev0 nao existe no PyPI).
RUN python3 -m pip install --no-cache-dir \
        diffusers==0.37.1 \
        peft==0.18.1 \
        accelerate==1.11.0 \
        datasets==4.3.0 \
        "huggingface_hub==1.0.0rc6" \
        safetensors==0.6.2 \
        numpy==1.26.4 \
        Pillow==12.0.0 \
        sentencepiece==0.2.1 \
        protobuf==6.33.0 \
        wandb==0.28.1 \
        python-dotenv==1.2.2 \
        pyyaml==6.0.3

# HOME gravavel para qualquer UID: rodamos com --user $(id -u):$(id -g) para NAO
# criar arquivo root no /raid (a rodada anterior nesta maquina deixou 54 GB de
# hf-cache pertencendo ao root, que a usuaria nao consegue apagar). Sem um HOME
# gravavel, bibliotecas que escrevem em ~/.cache ou ~/.config quebram.
ENV HOME=/tmp/juliahome
RUN mkdir -p /tmp/juliahome && chmod 777 /tmp/juliahome

ENV TOKENIZERS_PARALLELISM=false
ENV PYTHONUNBUFFERED=1

WORKDIR /workspace/genrefocus_deblurnet_paper
