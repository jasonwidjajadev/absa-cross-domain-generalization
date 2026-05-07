"""
Evaluate the base model only (no LoRA) on a CSV test set.

Reports per-class and macro precision, recall, and F1 for polarity ∈ {positive, negative, neutral}.

Example:
  python eval_baseline.py --model-path models/Qwen2.5-1.5B-Instruct --test-csv data/test.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import re

import torch
from datasets import load_dataset
from sklearn.metrics import classification_report, precision_recall_fscore_support
from transformers import AutoModelForCausalLM, AutoTokenizer

MAX_NEW_TOKENS = 32

# Keep in sync with train_sft.py
SYSTEM_PROMPT = (
    "You classify sentiment toward one target aspect in English restaurant reviews. "
    "Reply with exactly one word: positive, negative, or neutral."
)


def build_user_message(review: str, target: str) -> str:
    # Build the user prompt from review text and target aspect.
    review = (review or "").strip()
    target = (target or "").strip()
    return (
        f"Review: {review}\n"
        f"Target aspect: {target}\n"
        "What is the sentiment toward this target aspect? Answer with one word only."
    )


LABEL_ORDER = ("negative", "neutral", "positive")
_LABEL_RE = re.compile(r"\b(positive|negative|neutral)\b", re.IGNORECASE)


def parse_polarity(text: str) -> str | None:
    # Extract one valid polarity label from model output text.
    if not text:
        return None
    m = _LABEL_RE.search(text)
    return m.group(1).lower() if m else None


def parse_args() -> argparse.Namespace:
    # Parse baseline evaluation CLI arguments.
    p = argparse.ArgumentParser(description="Evaluate base model (no LoRA) on a test CSV")
    p.add_argument(
        "--model-path",
        type=str,
        default="models/Qwen2.5-1.5B-Instruct",
        help="Base model checkpoint",
    )
    p.add_argument("--test-csv", type=str, default="data/test.csv")
    p.add_argument(
        "--predictions-csv",
        type=str,
        default=None,
        help="Optional path to save per-sample predictions CSV",
    )
    return p.parse_args()


@torch.inference_mode()
def main() -> None:
    # Run generation-based evaluation and print/report metrics.
    # Falls back to neutral when output cannot be parsed.
    args = parse_args()
    model_id = args.model_path
    test_csv = os.path.abspath(args.test_csv)

    if not os.path.isfile(test_csv):
        raise FileNotFoundError(f"Test CSV not found: {test_csv}")

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    use_cuda = torch.cuda.is_available()
    use_bf16 = use_cuda and torch.cuda.is_bf16_supported()
    use_fp16 = use_cuda and not use_bf16
    if use_bf16:
        dtype = torch.bfloat16
    elif use_fp16:
        dtype = torch.float16
    else:
        dtype = torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        torch_dtype=dtype,
        device_map="auto" if use_cuda else None,
    )
    print("Running baseline (base model only, no LoRA).")
    model.eval()

    ds = load_dataset("csv", data_files={"test": test_csv})["test"]
    required = ("text", "target", "polarity")
    missing = [c for c in required if c not in ds.column_names]
    if missing:
        raise ValueError(f"test CSV missing columns {missing}; need {list(required)}")
    ds = ds.select_columns(list(required))

    y_true: list[str] = []
    y_pred: list[str] = []
    parse_ok: list[bool] = []
    prediction_rows: list[dict[str, str]] = []

    device = next(model.parameters()).device

    for row in ds:
        gold = (row["polarity"] or "").strip().lower()
        if gold not in LABEL_ORDER:
            raise ValueError(f"Unexpected gold polarity: {row['polarity']!r}")
        messages: list[dict[str, str]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_message(row["text"], row["target"])},
        ]
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        prompt_len = inputs["input_ids"].shape[1]

        out = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
        gen_ids = out[0, prompt_len:]
        raw = tokenizer.decode(gen_ids, skip_special_tokens=True)
        pred = parse_polarity(raw)
        ok = pred is not None
        parse_ok.append(ok)
        if pred is None:
            pred = "neutral"
        y_true.append(gold)
        y_pred.append(pred)
        prediction_rows.append(
            {
                "text": row["text"] or "",
                "target": row["target"] or "",
                "ground_truth": gold,
                "prediction": pred,
                "raw_output": (raw or "").strip(),
                "parsed_ok": "1" if ok else "0",
            }
        )

    n_fail = sum(1 for x in parse_ok if not x)
    print(f"Samples: {len(y_true)}  (parse failures defaulted to neutral: {n_fail})\n")

    print(classification_report(y_true, y_pred, labels=list(LABEL_ORDER), digits=4, zero_division=0))

    p_macro, r_macro, f_macro, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=list(LABEL_ORDER),
        average="macro",
        zero_division=0,
    )
    print(f"macro precision: {p_macro:.4f}")
    print(f"macro recall:    {r_macro:.4f}")
    print(f"macro F1:        {f_macro:.4f}")

    if n_fail:
        mask = [ok for ok in parse_ok]
        y_t = [y_true[i] for i in range(len(y_true)) if mask[i]]
        y_p = [y_pred[i] for i in range(len(y_pred)) if mask[i]]
        if y_t:
            print("\n--- Metrics on successfully parsed predictions only ---")
            print(classification_report(y_t, y_p, labels=list(LABEL_ORDER), digits=4, zero_division=0))
            p2, r2, f2, _ = precision_recall_fscore_support(
                y_t,
                y_p,
                labels=list(LABEL_ORDER),
                average="macro",
                zero_division=0,
            )
            print(f"macro precision: {p2:.4f}")
            print(f"macro recall:    {r2:.4f}")
            print(f"macro F1:        {f2:.4f}")

    if args.predictions_csv:
        pred_path = os.path.abspath(args.predictions_csv)
        pred_dir = os.path.dirname(pred_path)
        if pred_dir:
            os.makedirs(pred_dir, exist_ok=True)
        with open(pred_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["text", "target", "ground_truth", "prediction", "raw_output", "parsed_ok"],
            )
            writer.writeheader()
            writer.writerows(prediction_rows)
        print(f"Saved predictions CSV: {pred_path}")


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
