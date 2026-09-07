# Chunking Rationale — T17 Service Manual RAG Index

## 1. 고정 길이 대신 구조적 단위(조문) 기준 청킹을 선택한 이유

서비스 매뉴얼은 일반 산문(prose)과 달리 **논리적 경계가 명확한 구조화 문서**다.

| 문서 구조 단위 | 고정 길이 청킹 시 문제 | 구조적 청킹의 이점 |
|---|---|---|
| 번호 절차 (1. 2. 3. …) | 절차 중간에서 잘려 단계 앞뒤 맥락 소실 | 절차 경계에서만 분할 → 완전한 단계 보존 |
| Fault Code 표 | 코드·설명·해결책이 한 행인데 행이 분리됨 | 표 블록 단위로 묶어 코드+해결 동시 검색 |
| WARNING/CAUTION 박스 | 안전 경고가 본문과 분리될 위험 | WARNING 시작~빈 줄 끝까지 하나의 청크 |
| 스펙 테이블 (Nm, V, LPM 등) | 수치·단위가 절반씩 다른 청크에 배치 | 동일 블록 내 수치 보존 → factual_spec 검색 정확도 향상 |

**핵심 원칙**: 검색 결과 청크 하나만으로 질문에 답할 수 있어야 한다 — 분할된 절차 조각만 가져오면 "다음 단계는?"에 답할 수 없다.

---

## 2. 청킹 파라미터 설계

| 파라미터 | 값 | 선택 근거 |
|---|---|---|
| `MAX_CHUNK_CHARS` | 800 | `all-MiniLM-L6-v2` 최대 입력 256 토큰 ≈ 1000자; 800자 이하로 유지해 임베딩 품질 보장 |
| `MERGE_LIMIT_CHARS` | 600 | 인접 소단위(예: 경고 박스 + 짧은 설명)를 합쳐 의미 완결성 확보; 600자 초과 시 별도 청크 유지 |
| `MIN_CHUNK_CHARS` | 60 | 페이지 번호·헤더 잔재 등 노이즈 청크 제거 |
| 분할 우선순위 | ① 번호 단계 경계 → ② 단락(double newline) → ③ 단문 경계 | 구조 단위 보존 우선, 구조 없으면 단락 경계 |
| 페이지 창 크기 | 3페이지 배치 후 flush | 청크당 page_range를 ≤3페이지로 유지 → page_hit Recall 정확도 향상 |

---

## 3. 메타데이터 설계 근거

```json
{
  "chunk_id":      "rag_chunk_0042",
  "section_title": "Fault Codes Table (Pro-Panel and Current Standard Panel)",
  "page_range":    [133, 135],
  "content_type":  "troubleshooting",
  "text":          "…"
}
```

| 필드 | 목적 |
|---|---|
| `chunk_id` | 중복 청크 추적, 디버깅, 로그 |
| `section_title` | 어느 챕터/섹션에서 왔는지 추적성 확보; UI에서 출처 표시 가능 |
| `page_range` [start, end] | **Recall@5 page_hit** 계산에 직접 사용; RAG 결과에 "참조 페이지" 표기 가능 |
| `content_type` | 향후 `search_type="mmr"` + `filter={"content_type": "procedure"}` 형태의 필터링 검색 확장 가능; 현재는 분류 품질 분석용 |

**`page_range` 정확성**: 페이지를 3장씩 배치(flush)해 청크를 생성하므로, 각 청크의 page_range는 최대 3페이지 창으로 제한된다 → eval의 `source_page`가 그 창 안에 있으면 page_hit 성립.

---

## 4. 실제 통계

| 항목 | 값 |
|---|---|
| 전체 청크 수 | 351 |
| 평균 청크 길이 | 1,162 chars |
| 최대 청크 길이 | 4,999 chars (TOC 페이지 — 구조적 분할 어려움) |
| 총 색인 문자 수 | ~407,000 chars |
| 추출 가능 페이지 | 368 / 385 (OCR 불필요) |

### content_type 분포

| content_type | 청크 수 | 비율 |
|---|---:|---:|
| `troubleshooting` | 191 | 54.4% |
| `general` | 136 | 38.7% |
| `spec_table` | 19 | 5.4% |
| `warning` | 2 | 0.6% |
| `definition` | 3 | 0.9% |

`troubleshooting`이 전체의 54%를 차지하는 이유: T17 매뉴얼의 Troubleshooting 챕터(125~385쪽)가 전체 페이지의 68%를 차지하며, fault 코드·diagnostic 표가 대부분이기 때문이다.

---

## 5. 임베딩 모델 선택

**`sentence-transformers/all-MiniLM-L6-v2`** (384차원):
- `google/embeddinggemma-300m`은 Gated 모델(HF 수동 동의 필요)로 인증 없이 접근 불가
- all-MiniLM-L6-v2는 이미 환경에 캐시되어 있고, deduplication 단계에서 semantic 유사도 측정에 쓰인 모델로 성능 검증됨
- 384차원으로 FAISS flat index와 결합 시 351개 청크 규모에서 밀리초 수준 검색 가능

---

## 6. 개선 여지

- **Hybrid retrieval**: BM25(lexical) + dense(semantic) 결합으로 fault code 등 exact-match 검색 보완
- **Chunk overlap**: 절차 단계 경계를 넘는 질문(예: "4단계 직후 확인 사항")에 대해 ±1 단계 overlap 추가
- **Metadata 필터링**: RAG 질의에 "fault code" 키워드가 있으면 `content_type=troubleshooting` 청크 우선 검색
