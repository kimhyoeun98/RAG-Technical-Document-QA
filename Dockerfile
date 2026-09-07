# ST17 Service Manual QA — GPU 개발/학습 컨테이너
# 베이스: CUDA 12.6 + cuDNN (RTX A4000 호환), Python 3.10
FROM nvidia/cuda:12.6.0-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/root/.cache/huggingface

# ── 시스템 패키지 + Python 3.10 ──────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 python3.10-dev python3-pip \
        git curl ca-certificates build-essential \
    && ln -sf /usr/bin/python3.10 /usr/bin/python \
    && ln -sf /usr/bin/python3.10 /usr/bin/python3 \
    && python -m pip install --upgrade pip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# ── PyTorch (CUDA 12.6) ──────────────────────────────────────────────────────
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

# ── 프로젝트 의존성 (transformers / peft / langchain / faiss / anthropic 등) ──
COPY requirements.txt .
RUN pip install -r requirements.txt

EXPOSE 8888 8000

CMD ["bash"]
