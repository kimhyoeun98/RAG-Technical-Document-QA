# finalexam-gpu — GPU-Enabled Docker Dev Environment

NVIDIA GPU를 활용한 딥러닝 기반 개발 환경입니다.  
Gemma QLoRA 파인튜닝, LangChain/FAISS 기반 RAG 시스템 구축, PDF 데이터셋 구성을 위한 모든 패키지를 포함합니다.

---

## 사전 요구사항

| 항목 | 확인 명령 |
|------|-----------|
| NVIDIA 드라이버 (≥ 525) | `nvidia-smi` |
| nvidia-container-toolkit | `docker info \| grep -i runtime` → `nvidia` 포함 확인 |
| Docker Engine (≥ 20.10) | `docker --version` |

### nvidia-container-toolkit 미설치 시

```bash
# 설치
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
    | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
    | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit

# Docker 재시작
sudo systemctl restart docker
```

---

## 빌드

```bash
docker build -t finalexam-gpu .
```

---

## 실행

```bash
docker run --gpus all -it \
  --name finalexam-gpu \
  -v $(pwd)/workspace:/workspace \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -p 8888:8888 -p 8000:8000 \
  finalexam-gpu
```

- `workspace/` — 호스트와 공유되는 작업 디렉토리
- `~/.cache/huggingface` — 모델 가중치 캐시 (컨테이너 재시작 후에도 재다운로드 불필요)
- 포트 `8888` — JupyterLab, `8000` — API 서버용

### (참고) Gemma 등 게이트 모델 사용 시

Hugging Face 계정에서 발급한 토큰을 환경변수로 주입합니다.

```bash
docker run --gpus all -it \
  --name finalexam-gpu \
  -e HF_TOKEN=<your_huggingface_token> \
  -v $(pwd)/workspace:/workspace \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -p 8888:8888 -p 8000:8000 \
  finalexam-gpu
```

---

## 컨테이너 재진입

```bash
docker exec -it finalexam-gpu bash
```

---

## GPU 동작 확인

```bash
# GPU 인식 확인
docker exec -it finalexam-gpu nvidia-smi

# PyTorch CUDA 확인 (True 출력 시 정상)
docker exec -it finalexam-gpu python -c "import torch; print(torch.cuda.is_available())"
```

기대 출력:
- `nvidia-smi` → NVIDIA RTX A4000, CUDA Version 표시
- `torch.cuda.is_available()` → `True`

---

## JupyterLab 실행 (컨테이너 내부에서)

```bash
jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root
```

브라우저에서 `http://localhost:8888` 접속

---

## 컨테이너 종료 / 삭제

```bash
# 종료
docker stop finalexam-gpu

# 삭제
docker rm finalexam-gpu

# 이미지 삭제 (필요 시)
docker rmi finalexam-gpu
```

---

## 포함 패키지

| 카테고리 | 패키지 |
|----------|--------|
| 학습/파인튜닝 | transformers, accelerate, peft, bitsandbytes, datasets |
| RAG | langchain, langchain-community, langchain-huggingface, faiss-cpu, sentence-transformers |
| 토크나이저 | sentencepiece, protobuf, huggingface_hub |
| PDF 처리 | pypdf, pdfplumber, pymupdf |
| 개발 도구 | jupyterlab, ipywidgets |
| PyTorch | torch, torchvision, torchaudio (CUDA 12.6) |
