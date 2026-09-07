"""
S T17 영문 서비스 매뉴얼 RAG 파이프라인 — 제출용 통합 스크립트
실행: python rag_pipeline.py "How often should the fuser unit be replaced?"
입력: chunks_for_index.jsonl (없으면 faiss_index/도 같이 없다는 뜻이므로 먼저 4번 문제의
      청킹 단계를 다시 실행해서 만들어야 함)
출력: faiss_index/ (최초 1회 빌드 후 재사용), 콘솔에 검색 결과 + 생성된 답변 출력
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
import warnings
warnings.filterwarnings("ignore")
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForImageTextToText, BitsAndBytesConfig
from peft import PeftModel
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document          # langchain.docstore 구버전 대체

MODEL_ID       = "google/gemma-4-E4B-it"
ADAPTER_DIR    = "adapter"
# 질문은 한국어, 매뉴얼 청크는 영어 → 교차언어(ko↔en) 검색이 가능한 다국어 임베딩 사용.
# all-MiniLM(영어 전용)으로는 한국어 질문이 영어 청크를 찾지 못해 Recall이 붕괴함.
EMBED_MODEL_ID = "BAAI/bge-m3"
CHUNKS_PATH    = "chunks_for_index.jsonl"
INDEX_DIR      = "faiss_index_bge_m3"   # 모델별 분리 → 구버전(384dim) 인덱스와 충돌 방지
TOP_K          = 5

SYSTEM_PROMPT = (
    "You are a technical support assistant specialized in the ST17 service manual. "
    "Answer the user's question using only the provided context. "
    "Answer the EXACT question asked, directly and completely: if it asks for a procedure, "
    "list every relevant step in order; if it asks for a value/spec, state the precise number(s), "
    "unit(s) and each distinct reference point exactly as given (do not merge two separate values "
    "into a range). Do not add steps, items or details that the question did not ask about. "
    "Be precise; do not paraphrase numbers, part numbers or error codes. "
    "Always respond in the same language the user used to ask the question, even though "
    "the provided context is in English. You may keep model-specific terms such as part "
    "numbers, error codes, or product names in their original English/alphanumeric form."
)


def _is_korean(text: str) -> bool:
    return any('가' <= c <= '힣' for c in text)


def load_chunks(path):
    chunks = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    return chunks


def get_vectorstore():
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL_ID,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    if Path(INDEX_DIR).exists():
        print(f"[인덱스] {INDEX_DIR} 에서 기존 FAISS 인덱스를 불러옵니다.")
        return FAISS.load_local(INDEX_DIR, embeddings, allow_dangerous_deserialization=True)

    if not Path(CHUNKS_PATH).exists():
        raise FileNotFoundError(
            f"{CHUNKS_PATH}가 없습니다. 4번 문제의 Phase 1(조문 단위 청킹)을 먼저 실행해서 "
            f"전체 문서 청크 파일을 만들어야 합니다."
        )

    print(f"[인덱스] {CHUNKS_PATH}로부터 새 FAISS 인덱스를 빌드합니다.")
    chunks = load_chunks(CHUNKS_PATH)
    docs = [
        Document(
            page_content=c["text"],
            metadata={
                "chunk_id":      c.get("chunk_id"),
                "section_title": c.get("section_title"),
                "page_range":    c.get("page_range"),
                "content_type":  c.get("content_type"),
                "page_start":    c["page_range"][0],
                "page_end":      c["page_range"][1],
            },
        )
        for c in chunks
    ]
    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local(INDEX_DIR)
    return vectorstore


def get_generator():
    torch_dtype = torch.bfloat16
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch_dtype,
        bnb_4bit_quant_storage=torch_dtype,
    )
    base_model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID, dtype=torch_dtype, device_map="auto", quantization_config=bnb_config
    )
    model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    return model, tokenizer


class RagPipeline:
    def __init__(self):
        self.vectorstore = get_vectorstore()
        self.retriever   = self.vectorstore.as_retriever(search_kwargs={"k": TOP_K})
        self.model, self.tokenizer = get_generator()

    def retrieve(self, question):
        return self.retriever.invoke(question)

    def generate(self, question, retrieved_docs, debug=False):
        context = "\n\n".join(d.page_content for d in retrieved_docs)

        # 한국어 질문이면 user 메시지 앞에 언어 지시를 명시적으로 추가
        # 시스템 프롬프트만으로는 영어 파인튜닝 편향을 이기지 못하는 경우가 있음
        lang_prefix = "[중요: 반드시 한국어로 답변하세요]\n\n" if _is_korean(question) else ""

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": f"{lang_prefix}Context:\n{context}\n\nQuestion:\n{question}"},
        ]
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        if debug:
            print("\n[DEBUG] ===== 모델에 실제로 들어간 최종 프롬프트 =====")
            print(prompt)
            print("[DEBUG] ===== 프롬프트 끝 =====\n")

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=640,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        raw = self.tokenizer.decode(
            output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=False
        )
        if debug:
            print(f"[DEBUG] ===== 디코딩 raw 출력(special token 포함) =====\n{raw}\n[DEBUG] ===== raw 출력 끝 =====\n")

        return self.tokenizer.decode(
            output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        ).strip()

    def answer(self, question, debug=False):
        docs = self.retrieve(question)
        return self.generate(question, docs, debug=debug), docs


# 하위 호환: rag_eval.py가 사용하는 함수형 API
_pipeline = None

def _get_pipeline():
    global _pipeline
    if _pipeline is None:
        _pipeline = RagPipeline()
    return _pipeline

def retrieve_chunks(question: str):
    return _get_pipeline().retrieve(question)

def generate_answer(question: str, retrieved_docs) -> str:
    return _get_pipeline().generate(question, retrieved_docs)

def rag_answer(question: str):
    return _get_pipeline().answer(question)


if __name__ == "__main__":
    question = sys.argv[1] if len(sys.argv) > 1 else "How often should the fuser unit be replaced?"
    pipeline = RagPipeline()
    answer, docs = pipeline.answer(question, debug=True)

    print("\n[질문]", question)
    print("\n[검색된 상위 5개 청크 — 전체 텍스트]")
    for i, d in enumerate(docs, 1):
        print(f"  --- {i}. page_range={d.metadata.get('page_range')} "
              f"content_type={d.metadata.get('content_type')} chunk_id={d.metadata.get('chunk_id')} ---")
        print(f"  {d.page_content}\n")
    print("\n[생성된 답변]\n", answer)
