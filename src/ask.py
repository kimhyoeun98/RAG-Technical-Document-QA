"""
T17 서비스 매뉴얼 대화형 QA
실행: python ask.py
"""
import sys
if not hasattr(sys, 'get_int_max_str_digits'):
    _int_digit_limit = 4300
    def get_int_max_str_digits() -> int: return _int_digit_limit
    def set_int_max_str_digits(maxdigits: int) -> None:
        global _int_digit_limit; _int_digit_limit = maxdigits
    sys.get_int_max_str_digits = get_int_max_str_digits
    sys.set_int_max_str_digits = set_int_max_str_digits

import warnings
warnings.filterwarnings("ignore")

from rag_pipeline import RagPipeline

print("=" * 60)
print("T17 서비스 매뉴얼 QA (종료: q 또는 빈 줄 Enter)")
print("=" * 60)
print()

pipeline = RagPipeline()

while True:
    try:
        question = input("Q: ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\n종료합니다.")
        break

    if not question or question.lower() == "q":
        print("종료합니다.")
        break

    answer, docs = pipeline.answer(question)

    print(f"\nA: {answer}")
    print()
    print("  [참조 페이지]")
    for i, d in enumerate(docs, 1):
        m = d.metadata
        print(f"  {i}. p{m.get('page_start')}-{m.get('page_end')}  {m.get('section_title', '')[:50]}")
    print()
