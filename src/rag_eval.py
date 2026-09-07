"""
RAG evaluation on eval_dataset.jsonl.

Metrics:
  - Recall@5  : text_hit OR page_hit  (target >= 0.70)
  - Answer accuracy : LLM-as-Judge     (target >= 0.65)

Judge priority:
  1. Claude Haiku (claude-haiku-4-5-20251001) via ANTHROPIC_API_KEY
  2. Semantic similarity fallback (sentence-transformers cosine >= 0.65)

Output: rag_results.json
"""
import sys
if not hasattr(sys, 'get_int_max_str_digits'):
    _int_digit_limit = 4300
    def get_int_max_str_digits() -> int: return _int_digit_limit
    def set_int_max_str_digits(maxdigits: int) -> None:
        global _int_digit_limit; _int_digit_limit = maxdigits
    sys.get_int_max_str_digits = get_int_max_str_digits
    sys.set_int_max_str_digits = set_int_max_str_digits

import json
import os
import re
import string
import time
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path

# ── Import pipeline ────────────────────────────────────────────────────────────
from rag_pipeline import retrieve_chunks, generate_answer

EVAL_PATH   = Path("eval_dataset.jsonl")
RESULT_PATH = Path("rag_results.json")

# Claude judge 키: 환경변수 우선, 없으면 아래 상수.
#   권장: export ANTHROPIC_API_KEY=... 로 주입하고 이 상수는 비워 둔다(파일에 키를 남기지 않음).
HARDCODED_KEY = ""


def _resolve_key() -> str:
    return os.environ.get("ANTHROPIC_API_KEY", "").strip() or HARDCODED_KEY.strip()

# ── Recall@5 helpers ──────────────────────────────────────────────────────────

def normalize(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(f"[{re.escape(string.punctuation)}]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def page_in_range(source_page: int, doc_meta: dict) -> bool:
    p_start = doc_meta.get("page_start")
    p_end   = doc_meta.get("page_end")
    if p_start is None or p_end is None:
        return False
    return p_start <= source_page <= p_end


def recall_at_5_hit(gold_answer: str, gold_source_page: int, retrieved_docs,
                    gold_chunk_id: str | None = None) -> bool:
    # 1순위: 정답이 생성된 '출처 청크'가 top-5 안에 있는가 (가장 엄밀한 판정).
    #   답변이 한국어이고 청크가 영어라 텍스트 매칭은 신뢰할 수 없으므로 chunk_id 기준을 우선한다.
    if gold_chunk_id:
        if any(d.metadata.get("chunk_id") == gold_chunk_id for d in retrieved_docs):
            return True

    # 2순위(보강): 동일 페이지 청크 회수 — 출처 청크와 같은 페이지의 다른 청크도 정답 근거일 수 있음.
    page_hit = any(page_in_range(gold_source_page, d.metadata) for d in retrieved_docs)
    if page_hit:
        return True

    # 3순위(영문 답변 호환): 텍스트 직접 매칭 — 한국어 답변에서는 거의 발화하지 않음.
    norm_ans = normalize(gold_answer)
    if len(norm_ans) >= 10:
        return any(norm_ans in normalize(d.page_content) for d in retrieved_docs)
    tokens = set(norm_ans.split())
    return any(tokens.issubset(set(normalize(d.page_content).split())) for d in retrieved_docs)


# ── Judge: Claude Haiku (primary) ─────────────────────────────────────────────

JUDGE_PROMPT = """You are grading whether a candidate answer is semantically correct for a question about the ST17 floor scrubber service manual.
The reference and candidate answers are usually written in Korean; judge by MEANING, not language, ordering or surface wording.
Technical terms, part numbers, error codes and units may appear in English/alphanumeric form in either answer.

Question: {question}

Reference answer: {gold}

Candidate answer: {generated}

Grade CORRECT if the candidate correctly answers the question and is consistent with the key facts of the reference.
Apply these rules:
- Extra correct detail beyond the reference is fine — do NOT penalize a more complete answer.
- Paraphrasing, reordering and different phrasing are fine.
- Mark INCORRECT only if the candidate: contradicts a key fact of the reference (e.g. a wrong or merged number/value/unit),
  OR omits the core fact that the question specifically asks for, OR answers a different question/aspect,
  OR says it cannot find the answer when the reference has one.

Respond with exactly one line: CORRECT or INCORRECT, followed by a one-sentence reason."""


def _haiku_judge(gold: str, generated: str, question: str = "") -> tuple[bool, str] | None:
    """Try Claude judge. Returns None if API unavailable."""
    api_key = _resolve_key()
    if not api_key:
        return None
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
        prompt = JUDGE_PROMPT.format(question=question, gold=gold, generated=generated)
        for attempt in range(3):
            try:
                resp = client.messages.create(
                    model="claude-sonnet-4-6",
                    max_tokens=120,
                    messages=[{"role": "user", "content": prompt}],
                )
                text    = resp.content[0].text.strip()
                correct = text.upper().startswith("CORRECT")
                reason  = text.split("\n")[0]
                return correct, reason
            except Exception as e:
                if attempt == 2:
                    return None
                time.sleep(2)
    except ImportError:
        return None


# ── Judge: semantic similarity fallback ───────────────────────────────────────

_sem_model = None

def _sem_judge(gold: str, generated: str) -> tuple[bool, str]:
    """Semantic similarity judge (cosine >= 0.65).
    한국어 답변을 채점하므로 다국어 임베딩 모델을 사용한다(영어 전용 MiniLM은 한국어 의미를 못 잡음)."""
    global _sem_model
    if _sem_model is None:
        from sentence_transformers import SentenceTransformer
        _sem_model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    import numpy as np
    embs = _sem_model.encode([gold, generated], normalize_embeddings=True)
    sim  = float(np.dot(embs[0], embs[1]))
    correct = sim >= 0.65
    reason  = f"{'CORRECT' if correct else 'INCORRECT'}: semantic similarity={sim:.4f} (threshold=0.65)"
    return correct, reason


def llm_judge(gold: str, generated: str, question: str = "") -> tuple[bool, str]:
    result = _haiku_judge(gold, generated, question)
    if result is not None:
        return result
    return _sem_judge(gold, generated)


# ── Main evaluation loop ──────────────────────────────────────────────────────

def main():
    eval_data = [json.loads(l) for l in open(EVAL_PATH)]
    print(f"Loaded {len(eval_data)} eval pairs from {EVAL_PATH}")

    api_key = (os.environ.get("ANTHROPIC_API_KEY", "").strip() or HARDCODED_KEY.strip())
    judge_mode = "Claude Sonnet (claude-sonnet-4-6) LLM-as-Judge" if api_key else \
                 "Multilingual semantic similarity fallback (paraphrase-multilingual-MiniLM-L12-v2, cosine>=0.65)"
    print(f"Judge: {judge_mode}")
    print()

    details        = []
    recall_hits    = 0
    judge_corrects = 0

    for i, qa in enumerate(eval_data):
        qid         = qa.get("id", f"qa_{i:04d}")
        question    = qa["question"]
        gold_answer = qa["answer"]
        source_page = qa.get("source_page", 0)

        print(f"[{i+1:02d}/{len(eval_data)}] {qid[:20]} …", end=" ", flush=True)

        # Retrieve
        retrieved = retrieve_chunks(question)

        # Recall@5
        hit = recall_at_5_hit(gold_answer, source_page, retrieved,
                              gold_chunk_id=qa.get("source_chunk_id"))
        if hit:
            recall_hits += 1

        # Generate
        generated = generate_answer(question, retrieved)

        # Judge
        correct, reasoning = llm_judge(gold_answer, generated, question)
        if correct:
            judge_corrects += 1

        print(f"recall={'HIT' if hit else 'MISS'} judge={'✓' if correct else '✗'}")

        details.append({
            "id":               qid,
            "question":         question,
            "gold_answer":      gold_answer,
            "generated_answer": generated,
            "source_page":      source_page,
            "recall_hit":       hit,
            "judge_correct":    correct,
            "judge_reasoning":  reasoning,
        })

    n = len(eval_data)
    recall_at_5     = recall_hits    / n
    answer_accuracy = judge_corrects / n

    summary = {
        "recall_at_5":      round(recall_at_5,    4),
        "answer_accuracy":  round(answer_accuracy, 4),
        "recall_pass":      recall_at_5    >= 0.70,
        "accuracy_pass":    answer_accuracy >= 0.65,
        "n_eval":           n,
        "recall_hits":      recall_hits,
        "judge_corrects":   judge_corrects,
        "judge_mode":       judge_mode,
    }

    result = {"summary": summary, "details": details}
    RESULT_PATH.write_text(json.dumps(result, indent=2, ensure_ascii=False))

    print("\n" + "=" * 50)
    print(f"Recall@5       : {recall_at_5:.4f}  "
          f"({'PASS ✅' if summary['recall_pass']    else 'FAIL ❌'} >=0.70)")
    print(f"Answer accuracy: {answer_accuracy:.4f}  "
          f"({'PASS ✅' if summary['accuracy_pass'] else 'FAIL ❌'} >=0.65)")
    print(f"Judge mode     : {judge_mode}")
    print(f"Saved → {RESULT_PATH}")


if __name__ == "__main__":
    main()
