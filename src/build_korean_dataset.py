"""
한국어 QA 데이터셋 생성 — ST17 영문 서비스 매뉴얼 기반

영문 매뉴얼(영어 청크)을 근거로, 자연스러운 '한국어 질문 + 한국어 답변' QA 쌍을
Claude API로 생성한다. 영어 데이터셋 감점 및 RAG 답변 품질 문제를 해결하기 위한
재생성 스크립트.

설계 요지
  - 입력: chunks_for_index.jsonl (영어 청크, page_range/content_type/section_title 포함)
  - 청크를 disjoint 분리 → eval용 / train용. 서로 다른 청크에서 QA를 뽑으므로
    train·eval 간 동일 QA가 원천적으로 발생하지 않음(중복 방지 자동 보장).
  - 답변은 청크 텍스트 안의 근거만 사용(grounding). 부품번호·에러코드·단위·제품명 등은
    원문 영어/영숫자 그대로 유지.
  - 질문 유형 6종, 생성 방향(forward/backward) 다양화로 다양성 확보.

실행
  export ANTHROPIC_API_KEY="sk-ant-..."        # 또는 아래 HARDCODED_KEY 에 직접 입력
  python build_korean_dataset.py               # 실제 생성
  python build_korean_dataset.py --dry-run     # API 없이 파이프라인/스키마만 검증

출력
  train_dataset.jsonl, eval_dataset.jsonl
"""
import sys
if not hasattr(sys, "get_int_max_str_digits"):
    _lim = 4300
    sys.get_int_max_str_digits = lambda: _lim
    sys.set_int_max_str_digits = lambda v: None

import argparse
import json
import os
import random
import re
import time
from pathlib import Path

# ── 설정 ────────────────────────────────────────────────────────────────────────
CHUNKS_PATH   = "chunks_for_index.jsonl"
TRAIN_PATH    = "train_dataset.jsonl"
EVAL_PATH     = "eval_dataset.jsonl"

MODEL         = "claude-sonnet-4-6"   # 데이터 품질 우선. 비용 절감 시 claude-haiku-4-5-20251001
HARDCODED_KEY = ""                    # 환경변수 대신 직접 키를 넣고 싶으면 여기에 (권장X)

SEED            = 42
N_EVAL_CHUNKS   = 75    # eval 후보 청크 수 (목표 60+ 쌍)
QA_PER_TRAIN    = 2     # train 청크당 생성 시도 수 (목표 300+ 쌍)
MIN_CHUNK_LEN   = 150   # 표지/페이지번호 잔재 등 노이즈 청크 제외
MAX_RETRIES     = 3

ALLOWED_TYPES = [
    "factual_spec", "procedural", "conceptual",
    "troubleshooting", "diagram_based", "comparison_list",
]

# content_type → 선호 question_type (모델에 힌트로만 전달, 강제 아님)
TYPE_HINT = {
    "troubleshooting": "troubleshooting",
    "spec_table":      "factual_spec",
    "definition":      "conceptual",
    "warning":         "procedural",
    "general":         "procedural",
}

SYSTEM = (
    "당신은 산업용 바닥 청소기 'ST17' 영문 서비스 매뉴얼을 한국어 QA 데이터셋으로 "
    "변환하는 전문 어노테이터입니다. 주어진 영어 매뉴얼 발췌(context)만을 근거로 "
    "현장 기술자가 실제로 물어볼 법한 자연스러운 한국어 질문과, 그 근거에 충실한 "
    "완결된 한국어 답변을 만듭니다."
)

PROMPT_TMPL = """다음은 ST17 서비스 매뉴얼의 한 단락(영어)입니다. 이 단락만을 근거로
한국어 QA 쌍을 {n}개 생성하세요.

[규칙]
1. 질문과 답변은 반드시 자연스러운 한국어로 작성. 원문 텍스트를 그대로 붙여넣지 말 것.
2. 답변은 이 단락 안에 실제로 있는 정보만 사용(추측·외부지식 금지). 완결된 문장으로.
3. 부품번호·에러코드(예: F3, 0x0D0A)·수치·단위(V, A, Nm, LPM)·제품명은 원문 영어/영숫자 그대로 유지.
4. 단락에 답이 없으면 그 질문은 만들지 말 것. 충분한 정보가 없으면 {n}개보다 적게 만들어도 됨.
5. question_type 은 다음 중 가장 알맞은 것: {types}
6. generation_direction: 질문→답(설명 요구)이면 "forward", 값/코드를 주고 그 의미·대상을 묻는 형태면 "backward".

[section_title] {section}
[content_type] {ctype}

[context]
\"\"\"
{context}
\"\"\"

[출력 형식] 아래 JSON 배열만 출력. 다른 텍스트 금지.
[
  {{"question": "...", "answer": "...", "question_type": "...", "generation_direction": "forward|backward"}}
]
"""


# ── Claude API ──────────────────────────────────────────────────────────────────
_client = None

def get_client():
    global _client
    if _client is None:
        import anthropic
        key = os.environ.get("ANTHROPIC_API_KEY", "").strip() or HARDCODED_KEY.strip()
        if not key:
            sys.exit("ANTHROPIC_API_KEY 가 없습니다. 환경변수로 주입하거나 HARDCODED_KEY 에 입력하세요.")
        _client = anthropic.Anthropic(api_key=key)
    return _client


def call_llm(prompt: str) -> str:
    client = get_client()
    for attempt in range(MAX_RETRIES):
        try:
            resp = client.messages.create(
                model=MODEL, max_tokens=1500, system=SYSTEM,
                messages=[{"role": "user", "content": prompt}],
            )
            return resp.content[0].text
        except Exception as e:
            if attempt == MAX_RETRIES - 1:
                print(f"  [경고] API 실패: {e}")
                return "[]"
            time.sleep(2 * (attempt + 1))


def parse_json_array(text: str):
    """모델 출력에서 JSON 배열만 안전하게 추출."""
    text = text.strip()
    m = re.search(r"\[.*\]", text, re.DOTALL)
    if not m:
        return []
    try:
        data = json.loads(m.group(0))
        return data if isinstance(data, list) else []
    except json.JSONDecodeError:
        return []


# ── 생성 ───────────────────────────────────────────────────────────────────────
def gen_for_chunk(chunk: dict, n: int, dry_run: bool):
    ctype = chunk.get("content_type", "general")
    if dry_run:
        # API 없이 스키마/파이프라인만 검증하기 위한 가짜 생성기 (실제 경로와 동일한 튜플 반환)
        return [(
            f"[DRY] {chunk['chunk_id']} 의 {i+1}번 질문 (page {chunk['page_range'][0]})",
            f"[DRY] 근거: {chunk['text'][:40].strip()} ...",
            TYPE_HINT.get(ctype, "procedural"),
            "forward" if i % 2 == 0 else "backward",
        ) for i in range(n)]

    prompt = PROMPT_TMPL.format(
        n=n, types=", ".join(ALLOWED_TYPES),
        section=chunk.get("section_title", ""), ctype=ctype,
        context=chunk["text"],
    )
    out = []
    for qa in parse_json_array(call_llm(prompt)):
        if not isinstance(qa, dict):
            continue
        q = str(qa.get("question", "")).strip()
        a = str(qa.get("answer", "")).strip()
        qt = qa.get("question_type", "")
        if qt not in ALLOWED_TYPES:
            qt = TYPE_HINT.get(ctype, "procedural")
        gd = qa.get("generation_direction", "forward")
        gd = gd if gd in ("forward", "backward") else "forward"
        if len(q) < 8 or len(a) < 4:
            continue
        # 한국어가 한 글자도 없으면(영어만) 버림 — 한국어 데이터셋 보장
        if not any("가" <= c <= "힣" for c in q + a):
            continue
        out.append((q, a, qt, gd))
    return out


def make_entry(idx_prefix, n, chunk, q, a, qt, gd):
    return {
        "id": f"{idx_prefix}_{n:04d}",
        "question": q,
        "answer": a,
        "context": chunk["text"],
        "source_page": chunk["page_range"][0],
        "source_chunk_id": chunk["chunk_id"],   # recall@5 정답 청크 판정용
        "question_type": qt,
        "generation_direction": gd,
        "source_modality": "text",
    }


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.lower()).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="API 호출 없이 구조만 검증")
    ap.add_argument("--limit", type=int, default=0, help="처리할 청크 수 제한(테스트용)")
    args = ap.parse_args()

    chunks = [json.loads(l) for l in open(CHUNKS_PATH, encoding="utf-8")]
    chunks = [c for c in chunks if len(c.get("text", "")) >= MIN_CHUNK_LEN]
    random.Random(SEED).shuffle(chunks)
    if args.limit:
        chunks = chunks[: args.limit]

    eval_chunks  = chunks[:N_EVAL_CHUNKS]
    train_chunks = chunks[N_EVAL_CHUNKS:]
    print(f"유효 청크 {len(chunks)}개 → eval용 {len(eval_chunks)} / train용 {len(train_chunks)}")

    # ── eval 생성 (청크당 1쌍) ──
    eval_rows, seen = [], set()
    for c in eval_chunks:
        for q, a, qt, gd in gen_for_chunk(c, 1, args.dry_run):
            key = (norm(q), norm(a))
            if key in seen:
                continue
            seen.add(key)
            eval_rows.append(make_entry("qa_eval", len(eval_rows) + 1, c, q, a, qt, gd))
            break  # eval은 청크당 1개만
        if len(eval_rows) % 10 == 0:
            print(f"  eval {len(eval_rows):3d} …", flush=True)
    print(f"\neval 생성 완료: {len(eval_rows)}쌍")

    # ── train 생성 (청크당 최대 QA_PER_TRAIN쌍) ──
    train_rows = []
    for c in train_chunks:
        for q, a, qt, gd in gen_for_chunk(c, QA_PER_TRAIN, args.dry_run):
            key = (norm(q), norm(a))
            if key in seen:
                continue
            seen.add(key)
            train_rows.append(make_entry("qa_train", len(train_rows) + 1, c, q, a, qt, gd))
        if len(train_rows) % 25 == 0:
            print(f"  train {len(train_rows):4d} …", flush=True)
    print(f"\ntrain 생성 완료: {len(train_rows)}쌍")

    # ── 저장 ──
    with open(EVAL_PATH, "w", encoding="utf-8") as f:
        for r in eval_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(TRAIN_PATH, "w", encoding="utf-8") as f:
        for r in train_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # ── 검증 리포트 ──
    from collections import Counter
    tr_keys = {(norm(r["question"]), norm(r["answer"])) for r in train_rows}
    ev_keys = {(norm(r["question"]), norm(r["answer"])) for r in eval_rows}
    print("\n=== 검증 ===")
    print(f"train {len(train_rows)}쌍 (>=300 {'OK' if len(train_rows)>=300 else 'FAIL'})")
    print(f"eval  {len(eval_rows)}쌍 (>=60 {'OK' if len(eval_rows)>=60 else 'FAIL'})")
    print(f"train 유형: {Counter(r['question_type'] for r in train_rows)}")
    print(f"eval  유형: {Counter(r['question_type'] for r in eval_rows)}")
    print(f"train↔eval 정확 중복: {len(tr_keys & ev_keys)} (0 이어야 함)")
    types_ok = len(set(r['question_type'] for r in train_rows + eval_rows)) >= 4
    print(f"질문 유형 4종 이상: {'OK' if types_ok else 'FAIL'}")


if __name__ == "__main__":
    main()
