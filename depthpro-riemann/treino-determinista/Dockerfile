# =============================================================================
# Fine-tuning do DepthPro com Perda Riemanniana — imagem de container
# Base: PyTorch 2.4.1 + CUDA 12.4 + cuDNN 9
# Compatível com NVIDIA H100 (Hopper, sm_90), A100 (Ampere, sm_80) e RTX 4090 (Ada, sm_89).
# Perfil de execução default ajustado para 80GB de VRAM (H100) — ver README_CONTAINER.
# =============================================================================
FROM pytorch/pytorch:2.4.1-cuda12.4-cudnn9-runtime

# --- Metadados ---
LABEL org.opencontainers.image.title="riemann-depthpro"
LABEL org.opencontainers.image.description="Fine-tuning do DepthPro com Perda Riemanniana (curvatura Gaussiana)"

# Evita prompts interativos no apt e melhora logs do Python
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TOKENIZERS_PARALLELISM=false \
    PIP_NO_CACHE_DIR=1

# --- Dependências de sistema (git p/ pip git+, wget p/ scripts de download) ---
# OpenCV é o headless (requirements.docker.txt) — não precisa das libs X11/libGL.
RUN apt-get update && apt-get install -y --no-install-recommends \
        git wget ca-certificates \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# --- 1. Dependências Python do pacote ---
# Usamos requirements.docker.txt, que NÃO reinstala torch/torchvision (a imagem base
# já traz o PyTorch com CUDA). Copiar só o requirements primeiro aproveita o cache.
COPY requirements.docker.txt /workspace/requirements.docker.txt
RUN pip install --upgrade pip && \
    pip install -r /workspace/requirements.docker.txt

# --- 2. Instalar o DepthPro (Apple ml-depth-pro) DENTRO da imagem ---
# Assim quem usa o container não precisa clonar/instalar nada manualmente.
# Pinado por SHA: builds reprodutíveis e cache com invalidação explícita
# (bump do ARG). Install não-editável — sem clone residual na layer.
ARG DEPTH_PRO_REF=9efe5c1def37a26c5367a71df664b18e1306c708
RUN pip install "git+https://github.com/apple/ml-depth-pro.git@${DEPTH_PRO_REF}"

# --- 3. Copiar e instalar o código do projeto ---
COPY riemann/  /workspace/riemann/
COPY scripts/  /workspace/scripts/
COPY README.md /workspace/README.md
COPY pyproject.toml requirements.txt /workspace/
# Instala o pacote riemann em modo editável (deps já vieram do requirements.docker.txt;
# requirements.txt precisa estar presente pois o pyproject o lê como metadado dinâmico).
RUN pip install --no-deps -e /workspace

# --- 4. Diretórios de montagem (dados, modelos, saídas) ---
# Estes serão montados como volumes em tempo de execução (ver docker-compose / run).
RUN mkdir -p /data /models /workspace/runs

# --- 5. Sanidade em build-time: valida a geometria da curvatura (roda em CPU) ---
# Se a matemática estiver quebrada, o build falha aqui — antes de gastar GPU.
RUN python scripts/test_geometry.py

# Sem ENTRYPOINT fixo: o usuário escolhe qual script rodar.
# Exemplos no README_CONTAINER.md.
CMD ["python", "scripts/test_geometry.py"]
