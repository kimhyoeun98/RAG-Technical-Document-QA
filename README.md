# T17 Service Manual QA — 기말평가 프로젝트

ST17 **영문** 서비스 매뉴얼(385페이지)을 기반으로 한
**한국어 QA 데이터셋 구축 → QLoRA 파인튜닝 → 교차언어 RAG 파이프라인** 통합 프로젝트.

> ### 재제출 변경사항 (이전 제출 대비)
> 1. **데이터셋을 영어 → 한국어로 재생성.** 자연스러운 한국어 질문 + 근거에 충실한
>    완결된 한국어 답변. raw PDF 텍스트를 그대로 붙이거나 답변 스팬이 잘리던 문제 해결.
>    (`build_korean_dataset.py`)
> 2. **임베딩을 다국어(`BAAI/bge-m3`)로 교체.** 한국어 질문이 영어 청크를 검색하는
>    교차언어(ko↔en) 검색을 위해 필수. 기존 `all-MiniLM`(영어 전용)은 한국어 질문에서
>    Recall이 붕괴함.
> 3. **`rag_eval.py` judge 버그 수정.** 이전 코드는 API 키를 `os.environ.get()`의
>    *변수 이름*으로 잘못 넘겨 Claude judge가 한 번도 동작하지 않았음. 정상화 +
>    폴백 judge도 다국어 모델로 교체.
> 4. **Recall@5 판정을 출처 청크 ID 기반으로 보강.** 한국어 답변은 영어 청크와 텍스트가
>    매칭되지 않으므로, "정답이 생성된 청크가 top-5에 포함되는지"로 측정.
> 5. **학습/추론 프롬프트 정합.** `finetune.py`와 `rag_pipeline.py`의 시스템 프롬프트·
>    한국어 지시 prefix를 일치시켜 도메인 적응 효과 극대화.

---

## 파일 구조

```
.
├── README.md                  # 이 파일
├── Docker_README.md           # 컨테이너 빌드/실행
├── build_korean_dataset.py    # ★ 한국어 QA 데이터셋 생성 (Claude API)
├── train_dataset.jsonl        # 학습 데이터 (build 스크립트가 생성, 300+쌍)
├── eval_dataset.jsonl         # 평가 데이터 (build 스크립트가 생성, 60+쌍, RAG 전용)
├── finetune.py                # QLoRA 파인튜닝
├── adapter/                   # 학습된 LoRA 어댑터
├── loss_curve.png             # 학습/검증 손실 곡선
├── training_report.md         # 파인튜닝 상세 리포트
├── rag_pipeline.py            # RAG 파이프라인 (bge-m3 검색 + 생성)
├── rag_eval.py                # RAG 평가 (Claude judge + 다국어 폴백)
├── rag_results.json           # RAG 평가 결과
├── chunks_for_index.jsonl     # FAISS 인덱스용 영어 청크 (351개, 변경 없음)
├── chunking_rationale.md      # 청킹 설계 근거
└── faiss_index_bge_m3/        # FAISS 벡터 인덱스 (자동 빌드)
```

---

## 실행 순서 (컨테이너 내부)

Claude judge / 데이터 생성을 위해 키를 환경변수로 주입(권장):

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
```

### 1) 데이터셋 생성 (2번 문제)

```bash
python build_korean_dataset.py            # train_dataset.jsonl, eval_dataset.jsonl 생성
# 구조만 점검(무과금): python build_korean_dataset.py --dry-run
```

- 영어 청크를 disjoint하게 분리(eval용/train용) → **train·eval 동일 QA 원천 차단**
- 약 340여 회 API 호출 (1청크당 1회). 비용 절감 시 스크립트 상단 `MODEL`을 Haiku로 변경.

### 2) 파인튜닝 (3번 문제)

```bash
python finetune.py
# 출력: adapter/, loss_curve.png, training_report.md
```

### 3) RAG 평가 (4번 문제)

```bash
python rag_eval.py
# faiss_index_bge_m3/ 없으면 chunks_for_index.jsonl에서 자동 빌드
# 출력: rag_results.json
```

---

## 주요 기술 스택

- **Base model**: `google/gemma-4-E4B-it`
- **Fine-tuning**: QLoRA (4-bit NF4, r=16, alpha=16, all-linear)
- **Embedding**: `BAAI/bge-m3` (다국어, 1024dim, ko↔en 교차검색)
- **Vector store**: FAISS Flat Index (351 chunks, 영어 원문)
- **Chunking**: 구조적 단위 청킹 (섹션·절차·표 경계 기준) — `chunking_rationale.md`
- **Judge**: Claude Haiku (LLM-as-a-Judge) / 다국어 의미유사도 폴백

---

## 환경

- Python 3.10+
- PyTorch + BitsAndBytes + PEFT + TRL + langchain + FAISS + sentence-transformers
- GPU: NVIDIA RTX A4000 (16GB)
