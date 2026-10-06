FROM pytorch/pytorch:2.9.0-cuda13.0-cudnn9-devel

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

COPY --from=ghcr.io/astral-sh/uv:0.11.20 /uv /usr/local/bin/uv

COPY requirements.txt .

RUN uv pip install --system -r requirements.txt

COPY . .

CMD ["sleep", "infinity"]
