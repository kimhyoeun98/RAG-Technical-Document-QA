# RAG Technical Document QA

영문 기술문서를 기반으로 한국어 질문에 답변하는 RAG 기반 질의응답 프로젝트입니다.

Python, Linux, Docker 환경에서 문서 전처리, 임베딩, 벡터 검색, 답변 생성까지 이어지는 RAG 파이프라인을 구성하고, QLoRA 기반 모델 파인튜닝과 검색·생성 결과 평가를 수행했습니다.

본 프로젝트는 생성형 AI를 활용하여 코드 작성과 오류 해결을 진행했으며, AI가 생성한 코드를 그대로 사용하는 것이 아니라 각 코드의 역할과 실행 결과를 확인하며 수정하고 프로젝트를 완성했습니다.

---

## Project Overview

영문 서비스 매뉴얼을 기반으로 한국어 사용자의 질문에 적절한 근거를 검색하고 답변을 생성하는 질의응답 시스템을 구성했습니다.

주요 목표는 다음과 같습니다.

- 영문 기술문서를 한국어 질문으로 검색할 수 있는 교차언어 검색 구조 구성
- 기술문서를 RAG에서 활용할 수 있도록 청크 단위로 전처리
- 임베딩과 FAISS를 활용한 관련 문서 검색
- 검색된 문서를 기반으로 LLM 답변 생성
- QLoRA 기반 도메인 특화 모델 파인튜닝 실험
- 검색 및 답변 결과 평가

---

## System Flow

```text
Technical Document
        |
        v
Document Preprocessing
        |
        v
Chunking
        |
        v
Embedding (BGE-M3)
        |
        v
FAISS Vector Index
        |
        v
User Question
        |
        v
Relevant Chunk Retrieval
        |
        v
LLM Answer Generation
        |
        v
Evaluation
````

---

## Tech Stack

| Category      | Technology                |
| ------------- | ------------------------- |
| Language      | Python                    |
| Environment   | Linux, Docker             |
| LLM           | Gemma                     |
| Fine-tuning   | QLoRA, PEFT, TRL          |
| Embedding     | BGE-M3                    |
| Vector Search | FAISS                     |
| RAG           | LangChain                 |
| Deep Learning | PyTorch                   |
| Model Library | Hugging Face Transformers |

---

## Key Features

### 1. Technical Document Preprocessing

영문 기술문서를 RAG 검색에 사용할 수 있도록 전처리하고 문서 구조에 따라 청크 단위로 분리했습니다.

섹션, 절차, 표 등의 구조를 고려해 문맥이 지나치게 분리되지 않도록 청킹하는 것을 목표로 했습니다.

---

### 2. Korean QA Dataset

기술문서를 기반으로 한국어 질문과 답변 형태의 QA 데이터셋을 구성했습니다.

학습 데이터와 평가 데이터를 분리하여 모델 학습과 RAG 평가에 활용했습니다.

---

### 3. Cross-Lingual Retrieval

한국어 질문으로 영문 기술문서를 검색하기 위해 다국어 임베딩 모델인 `BAAI/bge-m3`를 사용했습니다.

문서 청크를 임베딩한 뒤 FAISS 벡터 인덱스를 구성하고, 사용자 질문과 의미적으로 가까운 문서를 검색하도록 구현했습니다.

---

### 4. RAG Pipeline

검색된 문서 내용을 LLM에 함께 전달하여 질문에 대한 답변을 생성하는 RAG 파이프라인을 구성했습니다.

전체 흐름은 다음과 같습니다.

```text
Question
→ Embedding
→ FAISS Search
→ Relevant Documents
→ Prompt
→ LLM
→ Answer
```

---

### 5. QLoRA Fine-Tuning

Gemma 기반 모델에 QLoRA 방식을 적용해 기술문서 질의응답 데이터로 파인튜닝하는 실험을 진행했습니다.

GPU 메모리 사용량을 줄이면서 도메인 데이터를 활용한 모델 학습 과정을 경험하는 것을 목표로 했습니다.

---

### 6. Evaluation

검색 결과와 생성 답변을 각각 확인할 수 있도록 평가 과정을 구성했습니다.

검색 단계에서는 관련 문서가 상위 검색 결과에 포함되는지 확인하고, 생성 단계에서는 질의와 답변 결과를 비교해 RAG 파이프라인의 동작을 확인했습니다.

---

## Project Structure

```text
RAG-Technical-Document-QA/
│
├── src/
│   ├── ask.py
│   ├── build_korean_dataset.py
│   ├── finetune.py
│   ├── rag_eval.py
│   └── rag_pipeline.py
│
├── data/
│   ├── chunks_for_index.jsonl
│   ├── eval_dataset.jsonl
│   └── train_dataset.jsonl
│
├── docs/
│   ├── chunking_rationale.md
│   └── training_report.md
│
├── results/
│   ├── loss_curve.png
│   └── rag_results.json
│
├── Dockerfile
├── Docker_README.md
├── requirements.txt
├── README.md
└── .gitignore
```

학습된 모델 가중치와 FAISS 인덱스는 저장소 용량 문제로 GitHub에 포함하지 않았습니다.

---

## How to Run

### 1. Clone Repository

```bash
git clone https://github.com/kimhyoeun98/RAG-Technical-Document-QA.git
cd RAG-Technical-Document-QA
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

또는 Docker 환경을 사용할 수 있습니다.

```bash
docker build -t rag-technical-document-qa .
```

Docker 실행 방법은 `Docker_README.md`에서 확인할 수 있습니다.

---

### 3. Dataset Generation

```bash
python src/build_korean_dataset.py
```

데이터셋 생성 과정에서 외부 API를 사용하는 경우 환경변수 설정이 필요합니다.

```bash
export ANTHROPIC_API_KEY="your_api_key"
```

API Key는 코드나 GitHub 저장소에 직접 저장하지 않습니다.

---

### 4. Fine-Tuning

```bash
python src/finetune.py
```

학습 결과는 모델 어댑터와 학습 결과 파일로 생성됩니다.

---

### 5. RAG Evaluation

```bash
python src/rag_eval.py
```

FAISS 인덱스가 존재하지 않는 경우 문서 청크 데이터를 기반으로 인덱스를 생성한 후 평가를 진행합니다.

---

## AI Usage

본 프로젝트는 생성형 AI를 활용하여 진행했습니다.

처음 접하는 RAG, 임베딩, 벡터 검색 및 모델 파인튜닝 기술을 학습하는 과정에서 AI를 코드 작성과 오류 해결을 위한 도구로 활용했습니다.

AI가 제시한 코드를 그대로 사용하는 것이 아니라 다음 과정을 반복하며 프로젝트를 진행했습니다.

* 코드의 역할과 실행 흐름 확인
* 실행 과정에서 발생하는 오류 확인
* 예상 결과와 실제 결과 비교
* 필요한 부분 수정
* 수정 이후 재실행 및 결과 확인

이를 통해 생성형 AI를 활용해 새로운 기술을 학습하고 실제 프로젝트 결과물로 연결하는 경험을 쌓았습니다.

---

## What I Learned

이 프로젝트를 통해 다음 내용을 경험했습니다.

* RAG의 전체적인 데이터 흐름
* 문서 전처리와 청킹의 중요성
* 임베딩을 활용한 의미 기반 검색
* FAISS 기반 벡터 검색
* 한국어 질문과 영문 문서 간 교차언어 검색
* Docker 기반 AI 실행 환경 구성
* QLoRA 기반 모델 파인튜닝 과정
* 검색 및 생성 결과 평가
* 생성형 AI를 활용한 코드 작성 및 문제 해결

---

## Limitations

본 프로젝트는 학습 목적의 개인 프로젝트로 진행되었습니다.

RAG 및 LLM 관련 기술을 처음 학습하며 구현했기 때문에 모든 구성 요소를 직접 설계하거나 최적화한 프로젝트는 아니며, 생성형 AI의 도움을 받아 구현과 오류 해결 과정을 진행했습니다.

향후에는 검색 성능 비교, 프롬프트 개선, 다양한 임베딩 모델 비교 및 RAG 평가 방식 개선 등을 추가로 진행할 수 있습니다.

---

## Author

**김효은**

* GitHub: [https://github.com/kimhyoeun98](https://github.com/kimhyoeun98)

```

이 버전의 장점은 명확합니다.

지금 README는 **“학교 과제를 재제출하면서 무엇을 고쳤는지”**가 중심인데, 위 버전은 **“이 프로젝트가 무엇이고, 내가 무엇을 경험했는지”**가 중심입니다.

특히 `AI Usage`와 `Limitations`를 넣은 이유가 중요합니다. 도치님은 이 프로젝트를 AI 도움을 받아 진행했다고 했기 때문에, `RAG 전문가처럼 보이게 포장`하는 것보다 **AI를 활용해서 배우고 구현한 프로젝트라는 사실을 명확히 쓰는 게 면접에서도 훨씬 안전합니다.**

그리고 하나 더 권한다면, `README` 맨 위에 `loss_curve.png`나 실제 실행 화면 캡처 하나를 넣는 것도 좋습니다. 그러면 GitHub 들어갔을 때 훨씬 포트폴리오처럼 보입니다.
```
