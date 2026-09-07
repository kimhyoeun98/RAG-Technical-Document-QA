"""
QLoRA Fine-tuning: google/gemma-4-E4B-it on ST17 Service Manual QA
RTX A4000 16GB | target: train_loss<=0.8, val_loss<=1.0, time<=40min

Outputs (all at project root):
  adapter/           — LoRA adapter weights
  loss_curve.png     — training & validation loss curve
  training_report.md — full training report
"""

# ── Python 3.11.0rc1 compatibility fix ───────────────────────────────────────
import sys
if not hasattr(sys, 'get_int_max_str_digits'):
    _int_digit_limit = 4300
    def get_int_max_str_digits() -> int:
        return _int_digit_limit
    def set_int_max_str_digits(maxdigits: int) -> None:
        global _int_digit_limit
        _int_digit_limit = maxdigits
    sys.get_int_max_str_digits = get_int_max_str_digits
    sys.set_int_max_str_digits = set_int_max_str_digits

import json
import time
import random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForImageTextToText,
    BitsAndBytesConfig,
    EarlyStoppingCallback,
)
from peft import LoraConfig
from trl import SFTConfig, SFTTrainer

# ── Phase 0: env check ────────────────────────────────────────────────────────
print("=" * 60)
print("Phase 0: Environment Check")
print("=" * 60)
print(f"torch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    cap = torch.cuda.get_device_capability()
    name = torch.cuda.get_device_name(0)
    total_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"GPU: {name} (compute cap {cap[0]}.{cap[1]})")
    print(f"VRAM: {total_mem:.1f} GB")
    BF16_SUPPORTED = cap[0] >= 8
    print(f"bf16 supported: {BF16_SUPPORTED}")

# ── Phase 1: Dataset ──────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Phase 1: Dataset Conversion & Split")
print("=" * 60)

# rag_pipeline.py 와 동일한 시스템 프롬프트·user 포맷을 사용한다.
# 학습 프롬프트와 추론(RAG) 프롬프트가 일치해야 도메인 적응 효과가 최대화된다.
SYSTEM_PROMPT = (
    "You are a technical support assistant specialized in the ST17 service manual. "
    "Answer the user's question using only the provided context. Be concise and precise. "
    "Always respond in the same language the user used to ask the question, even though "
    "the provided context is in English. You may keep model-specific terms such as part "
    "numbers, error codes, or product names in their original English/alphanumeric form."
)

def _is_korean(text: str) -> bool:
    return any('가' <= c <= '힣' for c in text)

def to_chat(row: dict) -> dict:
    # 한국어 질문이면 rag_pipeline 과 똑같이 언어 지시 prefix 를 붙여 프롬프트 정합성 유지
    lang_prefix = "[중요: 반드시 한국어로 답변하세요]\n\n" if _is_korean(row["question"]) else ""
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"{lang_prefix}Context:\n{row['context']}\n\nQuestion:\n{row['question']}",
            },
            {"role": "assistant", "content": row["answer"]},
        ]
    }

random.seed(42)
# Read from root-level train_dataset.jsonl (or fallback to dataset/train.jsonl)
import os
train_path = "train_dataset.jsonl" if os.path.exists("train_dataset.jsonl") else "dataset/train.jsonl"
with open(train_path) as f:
    all_rows = [json.loads(line) for line in f]

random.shuffle(all_rows)
val_size = max(1, int(len(all_rows) * 0.1))
val_rows = all_rows[:val_size]
train_rows = all_rows[val_size:]

print(f"Total samples: {len(all_rows)}  (from {train_path})")
print(f"Train split: {len(train_rows)} | Validation split: {len(val_rows)}")
print("(eval_dataset.jsonl reserved for RAG evaluation — not used here)")

train_dataset = Dataset.from_list([to_chat(r) for r in train_rows])
val_dataset = Dataset.from_list([to_chat(r) for r in val_rows])

# ── Phase 2: Model & QLoRA ────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Phase 2: Model Load & QLoRA Config")
print("=" * 60)

model_id = "google/gemma-4-E4B-it"
torch_dtype = torch.bfloat16

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch_dtype,
    bnb_4bit_quant_storage=torch_dtype,
)

print(f"Loading model: {model_id}")
model = AutoModelForImageTextToText.from_pretrained(
    model_id,
    dtype=torch_dtype,
    device_map="auto",
    quantization_config=bnb_config,
    attn_implementation="sdpa",
)
tokenizer = AutoTokenizer.from_pretrained(model_id)

total = sum(p.numel() for p in model.parameters())
print(f"Model params: {total/1e9:.2f}B total")

peft_config = LoraConfig(
    r=16,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    target_modules="all-linear",
    task_type="CAUSAL_LM",
)
print("LoRA: r=16, alpha=16, dropout=0.05, target=all-linear")

# ── Phase 3: Training Config ──────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Phase 3: SFTConfig")
print("=" * 60)

OUTPUT_DIR = "adapter"   # ← root-level output

args = SFTConfig(
    output_dir=OUTPUT_DIR,
    max_length=384,
    num_train_epochs=4,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    gradient_checkpointing=True,
    optim="adamw_torch_fused",
    learning_rate=3e-4,
    lr_scheduler_type="constant",
    max_grad_norm=0.3,
    bf16=True,
    loss_type="chunked_nll",
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    logging_steps=5,
    report_to="tensorboard",
    dataset_kwargs={"add_special_tokens": False, "append_concat_token": True},
    dataloader_num_workers=2,
    save_total_limit=2,
)

early_stopping = EarlyStoppingCallback(early_stopping_patience=2)

# ── Phase 4: Train ────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Phase 4: Training")
print("=" * 60)

trainer = SFTTrainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    peft_config=peft_config,
    processing_class=tokenizer,
    callbacks=[early_stopping],
)

eff_batch = args.per_device_train_batch_size * args.gradient_accumulation_steps
print(f"Effective batch size: {eff_batch}")
trainable = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
print(f"Trainable params after LoRA: {trainable/1e6:.1f}M")

start = time.time()
train_result = trainer.train()
elapsed_minutes = (time.time() - start) / 60

trainer.save_model()
print(f"\nAdapter saved to: {OUTPUT_DIR}")

# ── Metrics ────────────────────────────────────────────────────────────────────
train_loss = train_result.metrics.get("train_loss", float("nan"))
eval_losses = [e["eval_loss"] for e in trainer.state.log_history if "eval_loss" in e]
final_eval_loss = eval_losses[-1] if eval_losses else float("nan")

print("\n" + "=" * 60)
print("Results")
print("=" * 60)
print(f"Training Loss:    {train_loss:.4f}  (target: <=0.8 excellent)")
print(f"Validation Loss:  {final_eval_loss:.4f}  (target: <=1.0 excellent)")
print(f"Elapsed time:     {elapsed_minutes:.1f} min  (target: <=40 excellent)")

# ── Phase 5: Loss Curve ────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Phase 5: Generating loss curve")
print("=" * 60)

log_history = trainer.state.log_history
train_steps, train_losses_list = [], []
eval_epochs_list, eval_losses_list = [], []

for entry in log_history:
    if "loss" in entry and "eval_loss" not in entry:
        train_steps.append(entry["step"])
        train_losses_list.append(entry["loss"])
    if "eval_loss" in entry:
        eval_epochs_list.append(entry.get("epoch", len(eval_losses_list) + 1))
        eval_losses_list.append(entry["eval_loss"])

fig, ax = plt.subplots(figsize=(10, 5))
if train_steps:
    ax.plot(train_steps, train_losses_list, label="Train Loss", alpha=0.7, color="steelblue")
if eval_epochs_list and eval_losses_list:
    max_step = max(train_steps) if train_steps else 1
    eval_steps = [e / max(eval_epochs_list) * max_step for e in eval_epochs_list]
    ax.plot(eval_steps, eval_losses_list, "o-", label="Validation Loss",
            color="tomato", linewidth=2)
ax.axhline(0.8, color="green", linestyle="--", alpha=0.5, label="Train target (0.8)")
ax.axhline(1.0, color="orange", linestyle="--", alpha=0.5, label="Val target (1.0)")
ax.set_xlabel("Step")
ax.set_ylabel("Loss")
ax.set_title("Training & Validation Loss — Gemma4-E4B QLoRA (ST17 Manual)")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("loss_curve.png", dpi=150)   # ← root-level
plt.close()
print("Loss curve saved to: loss_curve.png")

# ── Phase 6: Training Report ──────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Phase 6: Writing training_report.md")
print("=" * 60)

def grade(val, good, min_ok, lower_is_better=True):
    if lower_is_better:
        if val <= good:   return "우수 ✅"
        elif val <= min_ok: return "최소 달성 ⚠️"
        else:             return "미달 ❌"
    else:
        if val >= good:   return "우수 ✅"
        elif val >= min_ok: return "최소 달성 ⚠️"
        else:             return "미달 ❌"

train_grade = grade(train_loss, 0.8, 1.2)
eval_grade  = grade(final_eval_loss, 1.0, 1.5)
time_grade  = grade(elapsed_minutes, 40, 60)

epoch_rows = [e for e in log_history if "eval_loss" in e]
epoch_table = "\n".join(
    f"| {int(e.get('epoch', i+1))} | {e.get('eval_loss', float('nan')):.4f} |"
    for i, e in enumerate(epoch_rows)
)

report = f"""# QLoRA Fine-tuning Training Report
## ST17 Service Manual QA — google/gemma-4-E4B-it

---

## 1. 최종 결과 요약

| 항목 | 결과 | 최소 기준 | 우수 기준 | 평가 |
|------|------|----------|----------|------|
| Training Loss | {train_loss:.4f} | ≤ 1.2 | ≤ 0.8 | {train_grade} |
| Validation Loss | {final_eval_loss:.4f} | ≤ 1.5 | ≤ 1.0 | {eval_grade} |
| 학습 소요 시간 | {elapsed_minutes:.1f}분 | ≤ 60분 | ≤ 40분 | {time_grade} |

---

## 2. 하이퍼파라미터

| 파라미터 | 값 | 선택 근거 |
|----------|-----|----------|
| `model_id` | `{model_id}` | E4B가 16GB VRAM·60분 제한 내 최적 크기 |
| `load_in_4bit` | True | NF4 4-bit 양자화로 VRAM 절감 |
| `lora_r` | 16 | 도메인 특화 QA에서 capacity·속도 균형점 |
| `lora_alpha` | 16 | scaling=1.0으로 안정적 학습 |
| `lora_dropout` | 0.05 | 소규모 데이터 과적합 방지 |
| `target_modules` | all-linear | attention+FFN 전체 LoRA 적용 |
| `max_length` | 384 | vocab=262K OOM 방지 (512→384) |
| `num_train_epochs` | 4 | 우수 기준(≤40분) 달성 목표 |
| `per_device_train_batch_size` | 1 | batch=2 OOM 확인 후 1로 안정화 |
| `gradient_accumulation_steps` | 8 | effective batch=8로 gradient noise 보상 |
| `loss_type` | chunked_nll | 262K vocab logit OOM 방지 |
| `optim` | adamw_torch_fused | AdamW fused 구현으로 속도 향상 |
| `learning_rate` | 3e-4 | QLoRA 권장 범위 중간값 |
| `lr_scheduler_type` | constant | 에폭 수 적을 때 예측 가능한 수렴 |
| `max_grad_norm` | 0.3 | QLoRA 논문 권장값 |

---

## 3. 학습 곡선

![Loss Curve](loss_curve.png)

### Epoch별 검증 손실

| Epoch | Eval Loss |
|-------|-----------|
{epoch_table}

---

## 4. 데이터셋 분리

- 원본: `train_dataset.jsonl` ({len(all_rows)}개)
- Train: {len(train_rows)}개 (90%), Validation: {len(val_rows)}개 (10%), seed=42
- `eval_dataset.jsonl`: RAG 평가용으로 별도 청크에서 생성 (train과 출처 청크가 겹치지 않아 leakage 없음)

---

## 5. 대화 형식

```
system: ST17 전문 기술 지원 어시스턴트
user:   Context:\\n{{context}}\\n\\nQuestion:\\n{{question}}
assistant: {{answer}}
```

Open-book 형식으로 context를 포함 → RAG와 동일한 프롬프트 구조, hallucination 억제.

---

## 6. 최종 체크리스트

- [{'x' if train_loss <= 0.8 else ' '}] Training Loss ≤ 0.8 (우수) — 실제: {train_loss:.4f}
- [{'x' if train_loss <= 1.2 else ' '}] Training Loss ≤ 1.2 (최소) — 실제: {train_loss:.4f}
- [{'x' if final_eval_loss <= 1.0 else ' '}] Validation Loss ≤ 1.0 (우수) — 실제: {final_eval_loss:.4f}
- [{'x' if final_eval_loss <= 1.5 else ' '}] Validation Loss ≤ 1.5 (최소) — 실제: {final_eval_loss:.4f}
- [{'x' if elapsed_minutes <= 40 else ' '}] 학습 시간 ≤ 40분 (우수) — 실제: {elapsed_minutes:.1f}분
- [{'x' if elapsed_minutes <= 60 else ' '}] 학습 시간 ≤ 60분 (최소) — 실제: {elapsed_minutes:.1f}분
- [x] optimizer=AdamW (adamw_torch_fused), lr=3e-4, early stopping 적용
- [x] adapter/ 에 LoRA 어댑터 저장 완료

*Generated by finetune.py | Model: {model_id}*
"""

with open("training_report.md", "w") as f:   # ← root-level
    f.write(report)
print("training_report.md written")

print("\n" + "=" * 60)
print("All done.")
print(f"  adapter/          → LoRA adapter")
print(f"  loss_curve.png    → training curve")
print(f"  training_report.md → full report")
print("=" * 60)
